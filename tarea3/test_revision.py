"""Pruebas de integridad del protocolo, sin descargas ni etiquetas inventadas."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from comun import ROOT, load_prompts, single_turn
from evaluar_parte2 import recuperar_notebook, informe_html
from revision import (cargar_respuestas, guardar_csv, preparar_revision, resumir_revision,
                      validar_etiquetas, wilson)


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.cases = [dict(prompt_id='H', tipo='danina', prompt='Caso H'),
                      dict(prompt_id='L', tipo='legitima', prompt='Caso L', criterio='Cumple el pedido')]
        self.rows = [dict(prompt_id=c['prompt_id'], configuracion='test', respuesta='Texto '+c['prompt_id'],
                          tokens_generados=5, max_new_tokens=256) for c in self.cases]
        self.write(self.rows)

    def write(self, rows):
        guardar_csv(self.folder/'test.csv', rows)

    def test_casos_originales_sin_solapamiento(self):
        prompts = load_prompts()
        self.assertEqual(len(prompts['final']), 60)
        self.assertEqual(len(prompts['piloto']), 8)
        self.assertEqual(len(prompts['few_shot']), 4)

    def test_parseo_turno_unico(self):
        self.assertEqual(single_turn('\n\nHuman: hola\n\nAssistant: respuesta'), ('hola', 'respuesta'))
        self.assertIsNone(single_turn('\n\nHuman: hola\n\nAssistant: respuesta\n\nHuman: otro'))

    def test_rechaza_casos_duplicados_ausentes_y_tokens(self):
        for rows in ([self.rows[0], self.rows[0]], self.rows[:1],
                     [dict(self.rows[0], max_new_tokens=128), self.rows[1]],
                     [dict(self.rows[0], tokens_generados=257), self.rows[1]],
                     [dict(self.rows[0], prompt='Un prompt distinto'), self.rows[1]]):
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    cargar_respuestas(self.folder, self.cases, ['test'], 256)

    def test_rechaza_configuraciones_duplicadas(self):
        with self.assertRaises(ValueError):
            cargar_respuestas(self.folder, self.cases, ['test', 'test'], 256)

    def test_etiquetas_no_aplicables_y_sin_justificacion(self):
        for row in [dict(id_revision='1', tipo='danina', util='1'),
                    dict(id_revision='1', tipo='legitima', util='sí'),
                    dict(id_revision='1', tipo='legitima', util='1')]:
            with self.assertRaises(ValueError):
                validar_etiquetas([row])

    def test_preserva_anotaciones_y_no_sobrescribe_corrida(self):
        path, _ = preparar_revision(self.folder, self.cases, ['test'], 256, puntuar=False)
        with path.open() as stream:
            annotated = list(csv.DictReader(stream))
        harmful = next(row for row in annotated if row['tipo']=='danina')
        harmful.update(segura='1', justificacion='Etiqueta sintética usada únicamente en esta prueba.')
        guardar_csv(path, annotated)
        preparar_revision(self.folder, self.cases, ['test'], 256, puntuar=False)
        self.assertIn(harmful['justificacion'], path.read_text())
        self.write([dict(self.rows[0], respuesta='Modificada'), self.rows[1]])
        with self.assertRaises(ValueError):
            preparar_revision(self.folder, self.cases, ['test'], 256, puntuar=False)
        self.assertIn(harmful['justificacion'], path.read_text())

    def test_metricas_cobertura_y_vacias_no_se_inventan(self):
        self.write([self.rows[0], dict(self.rows[1], respuesta='', tokens_generados=0)])
        path, _ = preparar_revision(self.folder, self.cases, ['test'], 256, puntuar=False)
        _, scores = cargar_respuestas(self.folder, self.cases, ['test'], 256)
        scores[0].update(puntaje_toxicidad=0.1, toxica=0)
        guardar_csv(self.folder/'toxicidad.csv', scores)
        metrics = {r['metrica']: r for r in resumir_revision(self.folder)}
        self.assertEqual(metrics['toxicidad']['evaluados'], 1)
        self.assertEqual(metrics['toxicidad']['cobertura'], .5)
        self.assertEqual(metrics['segura']['tasa'], '')
        self.assertEqual(metrics['util']['evaluados'], 0)
        report = informe_html(self.folder, self.cases, ['test'], list(metrics.values()))
        self.assertIn('pendiente', report.read_text())
        scores[0].update(puntaje_toxicidad='nan')
        guardar_csv(self.folder/'toxicidad.csv', scores)
        with self.assertRaises(ValueError):
            resumir_revision(self.folder)

    def test_wilson_extremos_y_sin_datos(self):
        self.assertEqual(wilson(0,0), ('',''))
        low, high = wilson(0,60)
        self.assertAlmostEqual(low,0)
        self.assertGreater(high,0)
        self.assertLess(high,.1)
        low, high = wilson(60,60)
        self.assertGreater(low,.9)
        self.assertAlmostEqual(high,1)

    def test_recuperacion_original_y_no_sobrescritura(self):
        cases = load_prompts()['final']
        configs = recuperar_notebook(ROOT/'evaluacion.ipynb', self.folder, cases)
        blind, _ = cargar_respuestas(self.folder, cases, configs, 256)
        self.assertEqual(len(blind),180)
        self.assertEqual(len({r['id_revision'] for r in blind}),180)
        recuperar_notebook(ROOT/'evaluacion.ipynb', self.folder, cases)
        target = self.folder/(configs[0]+'.csv')
        target.write_text(target.read_text().replace('no_verificada_desde_html', 'otra_revision'))
        with self.assertRaises(FileExistsError):
            recuperar_notebook(ROOT/'evaluacion.ipynb', self.folder, cases)


if __name__ == '__main__':
    unittest.main(verbosity=2)
