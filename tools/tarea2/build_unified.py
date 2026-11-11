"""Reúne las tres partes corregidas sin duplicar la biblioteca ni reiniciar el estudio.

Solo se usa al editar el repositorio. El notebook resultante es autocontenido.
Antes de reemplazar el unificado conserva una copia íntegra en Material/.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import textwrap
from cell_status import add_completion_messages

ROOT = Path(__file__).resolve().parents[2]


def read(part):
    return json.loads((ROOT / f'Tarea_2_Parte_{part}.ipynb').read_text(encoding='utf-8'))


def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=(textwrap.dedent(source).strip() + '\n').splitlines(keepends=True))
    if kind == 'code':
        ast.parse(''.join(item['source']))
        item.update(execution_count=None, outputs=[])
    return item


def definitions(notebook):
    return {node.name: ast.dump(node, include_attributes=False)
            for c in notebook['cells'] if c['cell_type'] == 'code'
            for node in ast.parse(''.join(c['source'])).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def build():
    first, second, third = (read(part) for part in (1, 2, 3))
    target = ROOT / 'Tarea_2.ipynb'
    previous = json.loads(target.read_text(encoding='utf-8')) if target.exists() else None
    common = definitions(first)
    for notebook in (second, third):
        for name, source in definitions(notebook).items():
            if name in common and source != common[name]:
                raise ValueError(f'Las partes difieren en {name}; resolver antes de unificar.')
    # Conservar configuración, datos y biblioteca de Parte 1. Las acciones se añaden una sola vez debajo.
    boundary = next(i for i, c in enumerate(first['cells'])
                    if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## 4. Entrenamiento'))
    cells = copy.deepcopy(first['cells'][:boundary])
    cells[0] = cell('markdown', '''
        # Tarea 2 · Detección de organismos y robustez — versión unificada

        **Integrantes:** Diego Muñoz, Benjamín Arguelles e Ignacio Pastén.

        Este archivo reúne las tres partes corregidas y contiene todo el código necesario.
        **En local o Colab usa Ejecutar todo:** elige CUDA si está disponible y CPU en caso contrario. En CPU el entrenamiento puede tardar bastante más y AMP se desactiva. Prepara los datos, entrena o retoma
        CNN y YOLO desde cero, entrena o retoma ambos COCO, compara los cuatro en test y evalúa robustez.
        Entrenamiento, evaluación y robustez están activados; no hay límite horario.

        La configuración y el contrato de datos se inicializan una sola vez. Se usan la misma carpeta,
        etiqueta de estudio y versión del motor que en las partes separadas, por lo que sus checkpoints
        compatibles se pueden retomar. No ejecutes dos sesiones sobre el mismo modelo/estudio simultáneamente.
        No necesitas tener los notebooks separados para ejecutar este archivo.

        **Recorrido:** entorno → datos → arquitecturas y auxiliares comunes → Parte 1 → Parte 2 → Parte 3 → rúbrica.
        La selección y los umbrales se fijan con validación antes de consultar test.
    ''')
    setup = ''.join(cells[2]['source'])
    setup = setup.replace('PARTE = 1', 'PARTE = 0  # Flujo unificado: las tres partes se ejecutan en orden.')
    setup = setup.replace('# Misma carpeta y etiqueta en los tres notebooks. Cambiar etiqueta crea otro estudio.',
                          '# Compartido con las partes separadas. Cambiar etiqueta crea otro estudio.')
    tree = ast.parse(setup)
    replacements = {'EJECUTAR_ENTRENAMIENTO': 'True', 'EJECUTAR_EVALUACION': 'True',
                    'EJECUTAR_ROBUSTEZ': 'True',
                    'MODELOS_DE_ESTA_PARTE': '["cnn_ssd", "yolo_scratch", "yolo_coco", "fasterrcnn_coco"]'}
    lines = setup.splitlines(keepends=True)
    for node in reversed(tree.body):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                lines[node.lineno - 1:node.end_lineno] = [f'{name} = {replacements[name]}\n']
    cells[2] = cell('code', ''.join(lines))
    robust_cell = next(c for c in third['cells'] if c['cell_type'] == 'code'
                       and any(isinstance(n, ast.FunctionDef) and n.name == 'ejecutar_robustez'
                               for n in ast.parse(''.join(c['source'])).body))
    cells.extend([cell('markdown', '### Auxiliares comunes de robustez'), copy.deepcopy(robust_cell)])
    cells.append(cell('markdown', ''.join(first['cells'][boundary]['source']).replace(
        '## 4. Entrenamiento y diagnóstico previo', '## 4. Protocolo común y diagnóstico previo')))
    cells.extend([
        cell('markdown', '### Ejecución de un grupo de modelos'),
        cell('code', '''
            def ejecutar_grupo(nombres, usa_coco=False):
                configuraciones = {n: configuracion_modelo(n) for n in nombres}
                display(Markdown(tabla_markdown([
                    dict(modelo=n, epocas=c['epocas'], congeladas=c['epocas_congeladas'],
                         tasa=c['tasa'], tasa_fina=c['tasa_fina'], batch_efectivo=c['batch_efectivo'])
                    for n, c in configuraciones.items()])))
                if EJECUTAR_ENTRENAMIENTO or EJECUTAR_DIAGNOSTICO or EJECUTAR_EVALUACION:
                    exigir_dispositivo(DISPOSITIVO_EJECUCION)
                    if usa_coco:
                        descargar_pesos(PESOS_DIR)
                if EJECUTAR_DIAGNOSTICO:
                    for nombre, config in configuraciones.items():
                        diagnostico = diagnostico_real(config, auditoria, manifiesto, estadisticas, dispositivo=DISPOSITIVO_EJECUCION)
                        guardar_json(RAIZ_ESTUDIO / 'modelos' / nombre / 'diagnostico_train.json', diagnostico)
                        print(nombre, diagnostico)
                if EJECUTAR_ENTRENAMIENTO:
                    for nombre, config in configuraciones.items():
                        entrenar_modelo(config, auditoria, manifiesto, estadisticas)
                        graficar_historial(nombre)
                else:
                    print('Entrenamiento desactivado; se revisarán resultados guardados si se solicita evaluar.')
                if EJECUTAR_EVALUACION:
                    for nombre in nombres:
                        evaluar_validacion(nombre, auditoria, manifiesto, estadisticas, dispositivo=DISPOSITIVO_EJECUCION)
                    return comparar_disponibles(nombres, 'val')
                return None
        '''),
        cell('markdown', '''
            ## 5. Parte 1 — Modelos entrenados desde cero

            CNN propia + SSD y YOLO11n parten de pesos aleatorios. Comparten los mismos IDs,
            evaluador y protocolo de optimización. Esta etapa compara validación y reserva test.
        '''),
        cell('code', 'comparacion_parte_1 = ejecutar_grupo(["cnn_ssd", "yolo_scratch"])'),
        cell('markdown', '''
            ## 6. Parte 2 — Transfer learning

            YOLO11n COCO y Faster R-CNN COCO usan cinco épocas congeladas y luego fine-tuning,
            según las capas descritas en la sección de arquitecturas. Se comprueban los hashes de pesos oficiales.
            Al terminar los cuatro modelos, se cierra la selección de validación y se compara test.
        '''),
        cell('code', 'comparacion_parte_2 = ejecutar_grupo(["yolo_coco", "fasterrcnn_coco"], usa_coco=True)'),
    ])
    evaluation_text = next(c for c in first['cells'] if c['cell_type'] == 'markdown'
                           and ''.join(c['source']).startswith('## 5. Evaluación'))
    cells.append(cell('markdown', ''.join(evaluation_text['source']).replace('## 5. Evaluación y comparación',
                                                                          '### Evaluación común de los cuatro modelos')))
    close_cell = next(c for c in second['cells'] if c['cell_type'] == 'code'
                      and 'cierre = cerrar_seleccion(' in ''.join(c['source']))
    cells.append(copy.deepcopy(close_cell))
    robust_text = next(c for c in third['cells'] if c['cell_type'] == 'markdown'
                       and ''.join(c['source']).startswith('## 4. Evaluación de robustez'))
    cells.append(cell('markdown', ''.join(robust_text['source']).replace('## 4. Evaluación de robustez',
        '## 7. Parte 3 — Evaluación de robustez')))
    robust_action = next(c for c in third['cells'] if c['cell_type'] == 'code'
                         and ''.join(c['source']).startswith('if EJECUTAR_ROBUSTEZ:'))
    cells.append(copy.deepcopy(robust_action))
    discussion = next(c for c in first['cells'] if c['cell_type'] == 'markdown'
                      and ''.join(c['source']).startswith('## 6. Resultados'))
    cells.append(cell('markdown', ''.join(discussion['source']).replace('## 6. Resultados', '## 8. Resultados')))
    cells.extend([cell('code', 'evidencias_rubrica = actualizar_rubrica()'), copy.deepcopy(first['cells'][-1])])
    old_outputs = {}
    if previous:
        for old in previous['cells']:
            if old['cell_type'] == 'code' and old.get('outputs'):
                key = ''.join(old['source'])
                old_outputs[key] = (old.get('execution_count'), old['outputs'])
    for i, c in enumerate(cells):
        c['id'] = f'unificado-{i:03d}'
        if c['cell_type'] == 'code':
            c.update(outputs=[], execution_count=None)
            prior = old_outputs.get(''.join(c['source']))
            if prior:
                c['execution_count'], c['outputs'] = copy.deepcopy(prior)
            ast.parse(''.join(c['source']))
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=copy.deepcopy(first['metadata']), cells=cells)
    final = definitions(notebook)
    for source in (first, second, third):
        for name, body in definitions(source).items():
            if final.get(name) != body:
                raise ValueError(f'La unificación alteró o perdió {name}')
    names = [n.name for c in cells if c['cell_type'] == 'code' for n in ast.parse(''.join(c['source'])).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    if len(names) != len(set(names)):
        raise ValueError('Definiciones duplicadas en el unificado')
    add_completion_messages(notebook)
    if target.exists():
        raw = target.read_bytes()
        backup = ROOT / 'Material' / f'Tarea_2_antes_unificar_{hashlib.sha256(raw).hexdigest()[:12]}.ipynb'
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            backup.write_bytes(raw)
        print('Respaldo:', backup.relative_to(ROOT))
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print('Unificado:', len(cells), 'celdas;', len(names), 'definiciones sin duplicados')


if __name__ == '__main__':
    build()
