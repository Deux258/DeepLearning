"""Pruebas de los notebooks generados sin ejecutar descargas ni corridas principales.

Usa definiciones visibles de cada .ipynb; no importa la biblioteca de desarrollo.
Ejecutar: .venv/Scripts/python.exe tools/tarea2/verify_notebooks.py
"""
import ast
import copy
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import symtable
import builtins
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
FILES = [ROOT / n for n in ('Tarea_2_Parte_1.ipynb',
         'Tarea_2_Parte_2.ipynb', 'Tarea_2_Parte_3.ipynb', 'Tarea_2.ipynb')]


def load_definitions(path, output):
    ns = dict(__name__='notebook_test', RAIZ_ESTUDIO=output, PESOS_DIR=ROOT / 'resultados_etapas_1_2/pesos',
              SEED=42, TAMANO=512, DISPOSITIVO_EJECUCION='cpu',
              DATA_DIR=ROOT / 'data', DATASET_DIR=ROOT / 'data/extracted')
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    constants = {'IMAGE_EXTENSIONS', 'KINDS', 'WEIGHTS', 'WEIGHT_SHA256',
                 'PARTICIONES_ORIGINALES', 'VERSION_MOTOR'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        nodes = []
        for node in tree.body:
            take = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
            if isinstance(node, ast.Assign):
                take = any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
            if take:
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    ns['torch'].set_num_threads(2)
    return ns


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'artifacts').mkdir(exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix='tarea2_verificacion_', dir=ROOT / 'artifacts')
        cls.folder = Path(cls.tmp.name)
        (cls.folder / 'ultralytics/Ultralytics').mkdir(parents=True)
        os.environ['YOLO_CONFIG_DIR'] = str(cls.folder / 'ultralytics')
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        cls.namespaces = [load_definitions(p, cls.folder / f'parte{i}') for i, p in enumerate(FILES, 1)]
        cls.ns = cls.namespaces[0]
        cls.audit = cls.ns['leer_json'](ROOT / 'resultados_etapas_1_2/curacion/audit.json')
        cls.manifest = cls.ns['PARTICIONES_ORIGINALES']
        cls.stats = cls.ns['leer_json'](ROOT / 'resultados_etapas_1_2/integracion_nacho_s42/particiones/train_statistics.json')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_schema_and_compile_every_cell(self):
        for path in FILES:
            notebook = json.loads(path.read_text(encoding='utf-8'))
            if importlib.util.find_spec('nbformat'):
                import nbformat
                nbformat.validate(nbformat.from_dict(notebook))
            ids = [c['id'] for c in notebook['cells']]
            self.assertEqual(len(ids), len(set(ids)))
            for c in notebook['cells']:
                if c['cell_type'] == 'code':
                    ast.parse(''.join(c['source']))
                    # Una ejecución del usuario puede conservar salidas legítimas.
                    # Validar el código sin borrar esas evidencias ni exigir un archivo virgen.
                    self.assertIsInstance(c['outputs'], list)
                    self.assertTrue(c['execution_count'] is None or isinstance(c['execution_count'], int))
            source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
            self.assertNotIn('metadata.tarea2_legado', source)
            self.assertNotIn('BASE_DIR', source)
            self.assertNotIn('import tools', source)
            self.assertNotIn('Tarea_2.ipynb', source)
            for c in notebook['cells']:
                if c['cell_type'] == 'code':
                    for node in ast.walk(ast.parse(''.join(c['source']))):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            self.assertNotIn(node.func.id, ('exec', 'eval'), 'No ocultar código ejecutable en cadenas')

    def test_identical_contract_and_engine_across_parts(self):
        for ns in self.namespaces:
            self.assertEqual(ns['PARTICIONES_ORIGINALES'], self.manifest)
            self.assertEqual(ns['VERSION_MOTOR'], self.ns['VERSION_MOTOR'])
            self.assertTrue(ns['validar_contrato'](self.audit, self.manifest, self.stats))
        self.assertEqual({k: len(v) for k, v in self.manifest['partitions'].items()},
                         {'train': 2022, 'val': 454, 'test': 454})

    def test_no_missing_global_helpers_in_any_part(self):
        for path in FILES:
            nb = json.loads(path.read_text(encoding='utf-8'))
            source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
            table = symtable.symtable(source, str(path), 'exec')
            available = set(table.get_identifiers()) | set(vars(builtins))
            missing = set()
            def visit(scope):
                for name in scope.get_identifiers():
                    symbol = scope.lookup(name)
                    if symbol.is_global() and symbol.is_referenced() and name not in available:
                        missing.add(name)
                for child in scope.get_children():
                    visit(child)
            visit(table)
            self.assertEqual(missing, set(), path.name)

    def test_unified_sequence_and_unchanged_scientific_definitions(self):
        nb = json.loads(FILES[3].read_text(encoding='utf-8'))
        nodes = [n for c in nb['cells'] if c['cell_type'] == 'code'
                 for n in ast.parse(''.join(c['source'])).body]
        defs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        names = [n.name for n in defs]
        self.assertEqual(len(names), len(set(names)))
        unified = {n.name: ast.dump(n, include_attributes=False) for n in defs}
        for path in FILES[:3]:
            part = json.loads(path.read_text(encoding='utf-8'))
            for c in part['cells']:
                if c['cell_type'] == 'code' and not c['metadata'].get('tarea2_solo_parte3'):
                    for n in ast.parse(''.join(c['source'])).body:
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                            self.assertEqual(unified[n.name], ast.dump(n, include_attributes=False))
        for flag in ('EJECUTAR_ENTRENAMIENTO', 'EJECUTAR_EVALUACION', 'EJECUTAR_ROBUSTEZ'):
            assignment = next(n for n in nodes if isinstance(n, ast.Assign)
                              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == flag)
            self.assertIs(ast.literal_eval(assignment.value), True)

        # Solo simular la orquestación; no ejecutar entrenamientos ni simular CUDA.
        events = []
        winner = 'yolo_coco'
        closed = {'ganador_validacion': winner}
        ns = dict(self.namespaces[3])
        ns.update(EJECUTAR_ENTRENAMIENTO=True, EJECUTAR_DIAGNOSTICO=False,
                  EJECUTAR_EVALUACION=True, EJECUTAR_ROBUSTEZ=True,
                  auditoria={}, manifiesto={}, estadisticas={},
                  display=lambda *a: None, exigir_dispositivo=lambda *a: events.append(('guardia',)),
                  descargar_pesos=lambda *a: events.append(('pesos',)),
                  entrenar_modelo=lambda c, *a: events.append(('train', c['modelo'])),
                  graficar_historial=lambda *a: None,
                  evaluar_validacion=lambda n, *a, **kw: events.append(('val', n)),
                  comparar_disponibles=lambda nombres, split: events.append(('comparar', split)),
                  cerrar_seleccion=lambda *a: events.append(('cierre',)) or closed,
                  evaluar_test=lambda n, *a, **kw: events.append(('test', n)),
                  ejecutar_robustez=lambda *a, **kw: events.append(('robustez',)),
                  graficar_robustez_final=lambda *a: None,
                  leer_json=lambda p: closed if p.name == 'seleccion_cerrada.json' else
                      {'metricas': {'map_50_95': 1. if p.parent.name == winner else 0.}})
        runner = next(n for n in defs if n.name == 'ejecutar_grupo')
        exec(compile(ast.Module(body=[runner], type_ignores=[]), 'orquestacion_unificada_sin_entrenamiento', 'exec'), ns)
        for c in nb['cells']:
            source = ''.join(c['source'])
            if c['cell_type'] == 'code' and (
                    source.startswith('comparacion_parte_') or 'cierre = cerrar_seleccion(' in source
                    or source.startswith('if EJECUTAR_ROBUSTEZ:')):
                exec(compile(source, 'orquestacion_unificada_sin_entrenamiento', 'exec'), ns)
        trained = [e[1] for e in events if e[0] == 'train']
        self.assertEqual(trained, list(ns['KINDS']))
        close_index = events.index(('cierre',))
        self.assertTrue(all(i < close_index for i, e in enumerate(events) if e[0] == 'train'))
        self.assertTrue(all(i > close_index for i, e in enumerate(events) if e[0] == 'test'))
        self.assertEqual(len([e for e in events if e[0] == 'test']), 4)
        self.assertGreater(events.index(('robustez',)), max(i for i, e in enumerate(events) if e[0] == 'test'))

    def test_standalone_robustness_uses_validation_or_saved_selection(self):
        notebook = json.loads(FILES[2].read_text(encoding='utf-8'))
        action = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and 'alcance = ' in ''.join(c['source']))
        ns = self.namespaces[2]
        audit, manifest = self.audit, self.manifest
        root = ns['RAIZ_ESTUDIO']
        selection = root / 'evaluacion/seleccion_cerrada.json'
        names = ('cnn_ssd', 'yolo_coco')
        events = []

        def infer(model, audit, ids):
            self.assertEqual(ids, manifest['partitions']['val'])
            events.append(('val', model))
            return model, 0.

        def robustness(cierre, audit, manifest, detector, ficha, dispositivo, nombre):
            events.append(('robustez', nombre))
            self.assertEqual(cierre['ganador_validacion'], nombre)
            self.assertEqual(cierre['modelos'][nombre]['umbral'], .25)
            return []

        overrides = dict(
            EJECUTAR_ROBUSTEZ=True, ruta_cierre=selection, candidatos=names,
            preparar_datos=lambda: (audit, manifest, self.stats), display=lambda *args: None,
            cargar_checkpoint_parte3=lambda nombre, *args: (nombre, dict(checkpoint=nombre, epoca=4)),
            predecir_ids=infer, elegir_umbral=lambda gt, preds, split: dict(selected=dict(threshold=.25)),
            evaluar=lambda gt, preds, umbral: dict(map_50_95=.2 if preds == 'yolo_coco' else .1),
            ejecutar_robustez_parte3=robustness, graficar_robustez_parte3=lambda *args: None)
        with mock.patch.dict(ns, overrides), redirect_stdout(io.StringIO()):
            exec(action, ns)
            provisional = ns['leer_json'](root / 'robustez_provisional/seleccion.json')
            self.assertEqual(events, [('val', 'cnn_ssd'), ('val', 'yolo_coco'), ('robustez', 'yolo_coco')])
            self.assertEqual(provisional['alcance'], 'provisional')
            self.assertFalse(selection.exists())
            final = dict(provisional)
            del final['alcance']
            ns['guardar_json'](selection, final)
            events.clear()
            exec(action, ns)
            self.assertEqual(events, [('robustez', 'yolo_coco')])
            self.assertEqual(ns['leer_json'](selection), final)

    def test_modified_partition_and_statistics_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['partitions']['train'][0] = manifest['partitions']['val'][0]
        with self.assertRaises(ValueError):
            self.ns['validar_contrato'](self.audit, manifest, self.stats)
        stats = copy.deepcopy(self.stats)
        stats['split'] = 'test'
        with self.assertRaises(ValueError):
            self.ns['validar_contrato'](self.audit, self.manifest, stats)

    def test_configuration_and_epoch_limits(self):
        for n in self.ns['KINDS']:
            c = self.ns['configuracion_modelo'](n)
            self.ns['validar_configuracion'](c)
            coco = n.endswith('coco')
            self.assertEqual(c['epocas'], 30 if coco else 60)
            self.assertEqual(c['epocas_congeladas'], 5 if coco else 0)
            self.assertEqual(c['minimo_epocas'], 15 if coco else 20)
            self.assertNotIn('horas', c)
            self.assertEqual(c['paciencia'], 10)
        c['tasa'] = float('nan')
        with self.assertRaises(ValueError):
            self.ns['validar_configuracion'](c)

    def test_device_guard_accepts_cpu_and_checks_cuda(self):
        self.assertEqual(self.ns['exigir_dispositivo']('cpu'), 'cpu')
        if self.ns['torch'].cuda.is_available():
            self.assertEqual(self.ns['exigir_dispositivo']('cuda'), 'cuda')
        else:
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                self.ns['exigir_dispositivo']('cuda')

    def test_colab_and_local_environment_startup(self):
        from ultralytics.utils import get_user_config_dir

        notebook = json.loads(FILES[3].read_text(encoding='utf-8'))
        sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        setup = next(s for s in sources if 'CONFIG_ULTRALYTICS =' in s)
        setup = setup[setup.index('CONFIG_ULTRALYTICS ='):]
        preflight = next(s for s in sources if 'torch.set_num_threads(min(8' in s)
        preflight = preflight[preflight.index('torch.set_num_threads(min(8'):]
        torch = self.ns['torch']
        for colab in (False, True):
            with self.subTest(colab=colab), mock.patch.dict(os.environ):
                folder = self.folder / ('colab' if colab else 'local')
                # Emular /content en un directorio temporal de esta plataforma.
                def mapped_path(value):
                    return folder / 'content' if str(value).startswith('/content/') else Path(value)
                ns = dict(EN_COLAB=colab, RAIZ_ESTUDIO=folder / 'estudio', Path=mapped_path,
                          DATASET_DIR=folder / 'data', os=os, VERSION_NOTEBOOK='fixture')
                with redirect_stdout(io.StringIO()):
                    exec(compile(setup, 'configuracion_ultralytics_visible', 'exec'), ns)
                expected = ns['CONFIG_ULTRALYTICS'] / 'Ultralytics'
                self.assertTrue(expected.is_dir())
                self.assertEqual(get_user_config_dir(), expected)
                for requested, cuda, expected_device in (
                        ('auto', False, 'cpu'), ('cpu', True, 'cpu'), ('auto', True, 'cuda')):
                    with self.subTest(requested=requested, cuda=cuda), \
                            mock.patch.object(torch.cuda, 'is_available', return_value=cuda), \
                            mock.patch.object(torch.version, 'cuda', '12.6' if cuda else None), \
                            mock.patch.object(torch, 'set_num_threads'), redirect_stdout(io.StringIO()):
                        ns.update(torch=torch, DISPOSITIVO_EJECUCION=requested)
                        exec(compile(preflight, 'seleccion_dispositivo_visible', 'exec'), ns)
                        self.assertEqual(ns['DISPOSITIVO_EJECUCION'], expected_device)
                with mock.patch.object(torch.cuda, 'is_available', return_value=False), \
                        mock.patch.object(torch, 'set_num_threads'), self.assertRaisesRegex(RuntimeError, 'CUDA'):
                    ns['DISPOSITIVO_EJECUCION'] = 'cuda'
                    exec(compile(preflight, 'cuda_explicita_no_disponible', 'exec'), ns)

    def test_real_data_geometry_and_no_eval_augmentation(self):
        ns = self.ns
        ids = self.manifest['partitions']['train'][:8]
        dataset = ns['DatosDeteccion'](self.audit, ids, 512, False)
        for i in range(8):
            image, target = dataset[i]
            self.assertEqual(tuple(image.shape), (3, 512, 512))
            self.assertTrue(bool(ns['torch'].isfinite(image).all()))
            self.assertTrue(bool((target['boxes'][:, 2:] > target['boxes'][:, :2]).all()))
            self.assertTrue(bool((target['labels'] >= 0).all()))
        a, ta = dataset[0]
        b, tb = dataset[0]
        self.assertTrue(ns['torch'].equal(a, b))
        self.assertTrue(ns['torch'].equal(ta['boxes'], tb['boxes']))

    def test_letterbox_roundtrip_and_reflection(self):
        ns = self.ns
        np = ns['np']
        mapping = ns['geometria_letterbox'](1920, 1080, 512)
        boxes = np.array([[12, 50, 900, 700], [1000, 100, 1900, 1000]], dtype=np.float32)
        np.testing.assert_allclose(mapping.inverse_boxes(mapping.forward_boxes(boxes)), boxes, atol=.001)
        im = ns['Image'].new('RGB', (1920, 1080))
        _, reflected = ns['reflejar_imagen'](im, boxes)
        _, restored = ns['reflejar_imagen'](im, reflected)
        np.testing.assert_allclose(restored, boxes)

    def test_coco_perfect_missing_class_and_operating_threshold(self):
        ns = self.ns
        gt = dict(images=[dict(id=1, width=100, height=100)],
                  categories=[dict(id=0, name='pez'), dict(id=1, name='ausente')],
                  annotations=[dict(id=1, image_id=1, category_id=0, bbox=[10, 10, 20, 20],
                                    area=400, iscrowd=0)])
        preds = [dict(image_id=1, category_id=0, bbox=[10, 10, 20, 20], score=.8)]
        m = ns['evaluar'](gt, preds, .9)
        self.assertAlmostEqual(m['map_50_95'], 1.)
        self.assertEqual(m['operating_point']['recall'], 0)
        self.assertIsNone(m['per_class'][1]['ap'])
        self.assertEqual(ns['evaluar'](gt, [])['map_50_95'], 0.)
        with self.assertRaises(ValueError):
            ns['elegir_umbral'](gt, preds, 'test')

    def test_warmup_cosine_resume_exactly(self):
        ns = self.ns
        torch = ns['torch']
        p = torch.nn.Parameter(torch.tensor(1.))
        opt = torch.optim.SGD([p], lr=.1)
        schedule = ns['PlanTasa'](opt, .1, 2, 6, 3, .01)
        values = []
        for _ in range(12):
            values.append(schedule.aplicar())
            schedule.step()
        self.assertAlmostEqual(values[0], .1 / 6)
        self.assertAlmostEqual(values[5], .1)
        self.assertAlmostEqual(values[-1], .001)
        resumed = ns['PlanTasa'](opt, .1, 2, 6)
        state = schedule.state_dict()
        state['cursor'] = 7
        resumed.load_state_dict(state)
        self.assertAlmostEqual(resumed.aplicar(), values[7])

    def test_accumulation_weights_partial_window(self):
        ns = self.ns
        torch = ns['torch']
        class Regression(torch.nn.Module):
            kind = 'cnn_ssd'
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor(.5))
            def loss(self, images, targets):
                return (((torch.stack(images) * self.w - 1) ** 2).mean(), {})
        dataset = [(torch.tensor(float(i)), {}) for i in (1, 2, 4)]
        a, b = Regression(), Regression()
        for model, batch in ((a, 1), (b, 3)):
            opt = torch.optim.SGD(model.parameters(), lr=.01)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            ns['actualizar_ventana'](model, opt, scaler, dataset, [0, 1, 2], batch)
        torch.testing.assert_close(a.w, b.w)

    def test_checkpoint_integrity_fallback_and_rng(self):
        ns = self.ns
        path = self.folder / 'checkpoint.pt'
        ns['fijar_semilla'](42)
        rng = ns['capturar_azar']()
        expected = ns['torch'].rand(5)
        ns['guardar_estado'](path, {'paso': 1, 'azar': rng})
        ns['guardar_estado'](path, {'paso': 2, 'azar': rng})
        path.write_bytes(b'interrupted write')
        state = ns['leer_estado'](path)
        self.assertEqual(state['paso'], 1)
        ns['restaurar_azar'](state['azar'])
        ns['torch'].testing.assert_close(ns['torch'].rand(5), expected)
        # Publicar después de una recuperación no debe sustituir el respaldo sano
        # por la generación corrupta que todavía ocupa ultimo.pt.
        ns['guardar_estado'](path, {'paso': 3, 'azar': rng})
        path.write_bytes(b'another interrupted write')
        self.assertEqual(ns['leer_estado'](path)['paso'], 1)

    def test_real_architectures_loss_prediction_freezing(self):
        ns = self.ns
        torch = ns['torch']
        for kind in ns['KINDS']:
            with self.subTest(kind=kind):
                ns['fijar_semilla'](42)
                model = ns['crear_detector'](kind, 28, 64, ns['PESOS_DIR'], self.stats)
                for phase in (['frozen', 'finetune'] if kind.endswith('coco') else ['full']):
                    model.set_phase(phase)
                    dataset = [(torch.rand(3, 64, 64), dict(boxes=torch.tensor([[10., 12., 40., 45.]]),
                                labels=torch.tensor([4]), image_id=torch.tensor(i))) for i in range(2)]
                    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
                    trainable = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                    buffers = {n: b.clone() for n, b in model.named_buffers() if 'running_' in n and
                               kind == 'yolo_coco' and phase == 'frozen' and int(n.split('.')[2]) <= 10}
                    opt = ns['adamw'](model, .001)
                    result = ns['actualizar_ventana'](model, opt, torch.amp.GradScaler('cuda', enabled=False), dataset, [0, 1], 2)
                    self.assertGreater(result['norma_gradiente'], 0.)
                    current = dict(model.named_parameters())
                    self.assertTrue(any(not torch.equal(v, current[k]) for k, v in trainable.items()))
                    self.assertTrue(all(torch.equal(v, current[k]) for k, v in frozen.items()))
                    self.assertTrue(all(torch.equal(v, dict(model.named_buffers())[k]) for k, v in buffers.items()))
                    output = model.predict([ns['Image'].new('RGB', (90, 70), (90, 120, 80))])[0]
                    self.assertEqual(set(output), {'boxes', 'labels', 'scores'})
                del model

    def test_degradations_are_paired_deterministic_and_preserve_rng(self):
        ns = self.namespaces[2]
        np = ns['np']
        image = ns['Image'].fromarray(np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3))
        conditions = ns['condiciones_robustez']()
        self.assertEqual(len(conditions), 7)
        for condition in conditions:
            state = np.random.get_state()
            a = ns['degradar_imagen'](image, condition, 1)
            b = ns['degradar_imagen'](image, condition, 1)
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
            self.assertEqual(a.size, image.size)
            np.testing.assert_array_equal(state[1], np.random.get_state()[1])

    def test_rubric_has_eleven_honest_pending_items(self):
        ns = self.namespaces[2]
        original_display = ns['display']
        ns['display'] = lambda *a: None
        try:
            rows = ns['actualizar_rubrica']()
        finally:
            ns['display'] = original_display
        self.assertEqual(len(rows), 11)
        self.assertEqual(sum(r['maximo'] for r in rows), 60)
        self.assertTrue(all(r['estado'] == 'Pendiente o parcial' for r in rows))

    def test_resume_next_step_in_fresh_process(self):
        # Red real CNN, optimizador y scheduler reales; datos sintéticos declarados.
        ns = self.ns
        torch = ns['torch']
        ns['fijar_semilla'](17)
        model = ns['crear_detector']('cnn_ssd', 2, 64, ns['PESOS_DIR'], None)
        dataset = [(torch.rand(3, 64, 64), dict(boxes=torch.tensor([[10., 12., 40., 45.]]),
                    labels=torch.tensor([0]), image_id=torch.tensor(i))) for i in range(2)]
        opt = ns['adamw'](model)
        plan = ns['PlanTasa'](opt, .001, 2, 4)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        plan.aplicar()
        ns['actualizar_ventana'](model, opt, scaler, dataset, [0, 1], 2)
        plan.step()
        path = self.folder / 'resume_fixture.pt'
        ns['guardar_estado'](path, dict(model=model.state_dict(), opt=opt.state_dict(),
                                       plan=plan.state_dict(), rng=ns['capturar_azar'](), dataset=dataset))
        plan.aplicar()
        ns['actualizar_ventana'](model, opt, scaler, dataset, [0, 1], 2)
        expected = self.folder / 'expected.pt'
        torch.save(model.state_dict(), expected)
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--resume-child',
                                 str(path), str(expected), str(self.folder)],
                                cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def resume_child():
    ns = load_definitions(FILES[0], Path(sys.argv[4]) / 'child')
    torch = ns['torch']
    saved = ns['leer_estado'](Path(sys.argv[2]))
    model = ns['crear_detector']('cnn_ssd', 2, 64, ns['PESOS_DIR'], None)
    model.load_state_dict(saved['model'])
    opt = ns['adamw'](model)
    opt.load_state_dict(saved['opt'])
    plan = ns['PlanTasa'](opt, .001, 2, 4)
    plan.load_state_dict(saved['plan'])
    ns['restaurar_azar'](saved['rng'])
    plan.aplicar()
    ns['actualizar_ventana'](model, opt, torch.amp.GradScaler('cuda', enabled=False), saved['dataset'], [0, 1], 2)
    expected = torch.load(sys.argv[3], map_location='cpu', weights_only=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    print('Reanudación exacta CPU en proceso nuevo: OK; fixture sintética, no entrenamiento principal.')


def real_diagnostics():
    """Ejecuta el control de aprendizaje, NO una corrida sobre el dataset completo."""
    output = ROOT / 'artifacts/diagnostico_real_tres_partes'
    (output / 'ultralytics/Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(output / 'ultralytics')
    ns = load_definitions(FILES[0], output)
    audit = ns['leer_json'](ROOT / 'resultados_etapas_1_2/curacion/audit.json')
    manifest = ns['PARTICIONES_ORIGINALES']
    stats = ns['leer_json'](ROOT / 'resultados_etapas_1_2/integracion_nacho_s42/particiones/train_statistics.json')
    results = {}
    for name in ('cnn_ssd', 'yolo_scratch'):
        print('Diagnóstico de ocho imágenes reales de train, CPU:', name, flush=True)
        results[name] = ns['diagnostico_real'](ns['configuracion_modelo'](name), audit, manifest,
                                              stats, dispositivo='cpu', pasos=300)
        print(json.dumps(results[name], ensure_ascii=False), flush=True)
        ns['guardar_json'](ROOT / 'docs/tarea2_diagnosticos_train_cpu.json', results)


def verify_preparation():
    output = ROOT / 'artifacts/preparacion_tres_partes'
    (output / 'ultralytics/Ultralytics').mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(output / 'ultralytics')
    ns = load_definitions(FILES[0], output)
    audit, manifest, stats = ns['preparar_datos']()
    assert len(audit['images']) == 2930 and audit['summary']['objects'] == 6236
    assert manifest == ns['PARTICIONES_ORIGINALES']
    ns['guardar_json'](ROOT / 'docs/tarea2_preparacion_verificada.json', dict(
        alcance='Preparación real del dataset sin entrenamientos; salida temporal independiente',
        imagenes=len(audit['images']), objetos=audit['summary']['objects'],
        particiones={k: len(v) for k, v in manifest['partitions'].items()},
        dataset_fingerprint=ns['huella_dataset'](audit), normalizacion=stats,
        version=ns['VERSION_MOTOR']))
    print('Preparación completa verificada: 2930 imágenes, 6236 objetos, IDs originales.', flush=True)


if __name__ == '__main__':
    if '--resume-child' in sys.argv:
        resume_child()
    elif '--real-diagnostics' in sys.argv:
        real_diagnostics()
    elif '--prepare-data' in sys.argv:
        verify_preparation()
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(NotebookTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report = dict(alcance='Pruebas breves CPU y fixtures; no cuatro entrenamientos principales ni certificación CUDA',
                      tests=result.testsRun, errores=len(result.errors), fallos=len(result.failures),
                      omitidas=len(result.skipped), cuda_disponible=__import__('torch').cuda.is_available(),
                      notebooks={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES})
        target = ROOT / 'docs/tarea2_verificacion.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        sys.exit(not result.wasSuccessful())
