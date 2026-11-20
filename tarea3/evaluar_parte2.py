"""Ejecuta la parte 2 con CSV existentes o recuperados del notebook original."""

import argparse
import csv
import json
import os
import re
from hashlib import sha256
from html import escape, unescape
from pathlib import Path

from comun import MAX_EVAL_TOKENS, OUT, ROOT, load_prompts
from revision import guardar_csv, preparar_revision, resumir_revision

os.environ.setdefault('HF_HOME', str(ROOT/'.runtime'/'hf'))
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')


def recuperar_notebook(notebook, folder, cases):
    """Recupera texto publicado; no afirma haber regenerado las respuestas."""
    notebook, folder = Path(notebook), Path(folder)
    content = notebook.read_bytes()
    data = json.loads(content)
    rendered = []
    for cell in data['cells']:
        for output in cell.get('outputs', []):
            html = output.get('data', {}).get('text/html', '')
            html = ''.join(html) if isinstance(html, list) else html
            if '<details><summary>' in html and 'Toxic-BERT' in html:
                rendered.append(html)
    if len(rendered) != 1:
        raise ValueError('Se esperaba exactamente una tabla HTML con respuestas.')
    expected = {row['prompt_id']: row for row in cases}
    rows, old_scores = {}, []
    seen = set()
    for summary, body in re.findall(r'<details><summary>(.*?)</summary>(.*?)</details>', rendered[0], re.S):
        key, kind, prompt = unescape(summary).split(' · ', 2)
        if key in seen or key not in expected or (kind, prompt) != (expected[key]['tipo'], expected[key]['prompt']):
            raise ValueError(f'Prompt duplicado, ajeno o modificado: {key}.')
        seen.add(key)
        for config, tokens, score, answer in re.findall(
                r'<h4>([^<]+?) · (\d+) tokens · Toxic-BERT ([0-9.]+)</h4>\s*<pre[^>]*>(.*?)</pre>', body, re.S):
            answer = unescape(answer)
            row = dict(prompt_id=key, prompt=prompt, configuracion=config, respuesta=answer,
                       tokens_generados=int(tokens), vacia=not answer.strip(),
                       limite_tokens=int(tokens)==MAX_EVAL_TOKENS,
                       max_new_tokens=MAX_EVAL_TOKENS,
                       revision='no_verificada_desde_html', adaptador='no_verificado_desde_html',
                       procedencia='salida_html_notebook_original')
            rows.setdefault(config, []).append(row)
            old_scores.append(dict(prompt_id=key, configuracion=config, puntaje_mostrado=score))
    if seen != expected.keys() or not rows:
        raise ValueError('El HTML no cubre todos los casos.')
    for config, group in rows.items():
        if len(group) != len(cases) or {r['prompt_id'] for r in group} != expected.keys():
            raise ValueError(f'HTML incompleto en {config}.')
        if Path(config).name != config:
            raise ValueError('Nombre de configuración inválido.')
        target = folder/f'{config}.csv'
        if target.exists():
            with target.open(encoding='utf-8', newline='') as stream:
                existing = list(csv.DictReader(stream))
            if [{k: str(v) for k, v in row.items()} for row in group] != existing:
                raise FileExistsError(f'No se sobrescribe una corrida diferente: {target}')
        else:
            guardar_csv(target, group)
    guardar_csv(folder/'toxicidad_original_redondeada.csv', old_scores)
    provenance = dict(origen=str(notebook.resolve()), sha256_notebook=sha256(content).hexdigest(),
                      respuestas=sum(map(len, rows.values())), configuraciones=sorted(rows),
                      max_new_tokens=MAX_EVAL_TOKENS,
                      advertencia='Textos recuperados de HTML. No se ejecutó inferencia Phi-3 ni se verificaron '
                                  'los pesos SFT originales. Puntajes previos redondeados; se puntúan de nuevo. '
                                  'Límite 256 según código y salida del notebook de origen.')
    (folder/'procedencia.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    return sorted(rows)


def informe_html(folder, cases, configs, metrics):
    folder = Path(folder)
    def read(path):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    answers = {config: {r['prompt_id']: r for r in read(folder/f'{config}.csv')} for config in configs}
    scored = {(r['configuracion'], r['prompt_id']): r for r in read(folder/'toxicidad.csv')}
    notes = [f'<p>{len(cases)} casos comunes · {len(configs)} configuraciones · '
             'Las etiquetas humanas pendientes no se reemplazan por toxicidad.</p>']
    if (folder/'procedencia.json').exists():
        provenance = json.loads((folder/'procedencia.json').read_text())
        notes.append(f"<p><strong>Procedencia:</strong> {escape(provenance['advertencia'])}</p>")
    missing = sorted({'phi3_zero_shot', 'phi3_few_shot', 'phi3_sft', 'phi3_ppo',
                      'mistral_zero_shot', 'mistral_few_shot'} - set(configs))
    if missing:
        notes.append(f"<p>Configuraciones todavía pendientes del grupo: {escape(', '.join(missing))}. "
                     'Mistral 7B Instruct se evalúa mediante Ollama para grupos de tres.</p>')
    notes.append('<table><tr><th>Configuración</th><th>Métrica</th><th>Positivos/evaluados</th>'
                 '<th>Cobertura</th><th>Tasa</th><th>IC 95% Wilson</th><th>Estado</th></tr>')
    for row in metrics:
        rate = f"{row['tasa']:.1%}" if row['tasa'] != '' else 'pendiente'
        ci = f"[{row['ic95_inferior']:.1%}, {row['ic95_superior']:.1%}]" if row['tasa'] != '' else '—'
        coverage = f"{row['evaluados']}/{row['total']}"
        notes.append(f"<tr><td>{escape(row['configuracion'])}</td><td>{escape(row['metrica'])}</td>"
                     f"<td>{row['positivos']}/{row['evaluados']}</td><td>{coverage}</td>"
                     f"<td>{rate}</td><td>{ci}</td><td>{row['estado']}</td></tr>")
    notes.append('</table><h2>Casos para revisión cualitativa del grupo</h2>'
                 '<p>Leer las respuestas completas. El tope de generación puede dejar texto incompleto. '
                 'Un puntaje bajo de toxicidad no demuestra que una respuesta sea segura o útil. '
                 'Los intervalos describen este conjunto; no garantizan generalización.</p>')
    for case in cases:
        notes.append(f"<details><summary>{escape(case['prompt_id'])} · {escape(case['tipo'])} · "
                     f"{escape(case['prompt'])}</summary><p>Criterio de utilidad: "
                     f"{escape(case.get('criterio', 'No aplica'))}</p>")
        for config in configs:
            row, score = answers[config][case['prompt_id']], scored[config, case['prompt_id']]
            notes.append(f"<h3>{escape(config)}</h3><p>Tokens: {escape(row['tokens_generados'])} · "
                         f"Toxicidad: {escape(score['puntaje_toxicidad']) or 'sin puntuar'} · "
                         f"Alcanzó tope: {escape(score['alcanzo_limite'])} · "
                         f"Truncada por clasificador: {escape(score['truncada_toxicidad']) or 'no aplica'}</p>"
                         f"<pre>{escape(row['respuesta'])}</pre>")
        notes.append('</details>')
    html = '<!doctype html><html lang="es"><meta charset="utf-8"><title>Tarea 3 · Parte 2</title>'
    html += '<style>body{font:16px system-ui;max-width:1150px;margin:32px auto;padding:16px;line-height:1.5}'
    html += 'table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}pre{white-space:pre-wrap;'
    html += 'overflow-wrap:anywhere}details{border:1px solid #ccc;margin:12px 0;padding:12px}summary{cursor:pointer}</style>'
    html += '<h1>Tarea 3 · Evaluación verificable</h1>' + '\n'.join(notes) + '</html>'
    target = folder/'informe.html'
    target.write_text(html, encoding='utf-8')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carpeta', type=Path, default=OUT/'respuestas_256'/'final')
    parser.add_argument('--recuperar-notebook', type=Path)
    parser.add_argument('--configuraciones', nargs='+', default=['phi3_zero_shot', 'phi3_few_shot', 'phi3_sft'])
    parser.add_argument('--solo-preparar', action='store_true', help='Crear pauta sin cargar el clasificador.')
    parser.add_argument('--sin-puntuar', action='store_true', help='Resumir etiquetas y puntajes ya existentes.')
    args = parser.parse_args()
    cases = load_prompts()['final']
    configs = args.configuraciones
    if args.recuperar_notebook:
        configs = recuperar_notebook(args.recuperar_notebook, args.carpeta, cases)
    if args.sin_puntuar:
        from revision import cargar_respuestas
        expected, _ = cargar_respuestas(args.carpeta, cases, configs, MAX_EVAL_TOKENS)
        with (args.carpeta/'revision_humana.csv').open(encoding='utf-8-sig', newline='') as stream:
            previous = list(csv.DictReader(stream))
        if {r['id_revision'] for r in expected} != {r['id_revision'] for r in previous}:
            raise ValueError('Las respuestas cambiaron desde la revisión. Prepara y puntúa la nueva corrida.')
    if not args.sin_puntuar:
        preparar_revision(args.carpeta, cases, configs, MAX_EVAL_TOKENS, puntuar=not args.solo_preparar)
    if not args.solo_preparar:
        metrics = resumir_revision(args.carpeta)
        print(informe_html(args.carpeta, cases, configs, metrics))


if __name__ == '__main__':
    main()
