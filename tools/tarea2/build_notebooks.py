"""Construye notebooks autocontenidos. Solo se usa durante desarrollo, nunca en Colab.

Ejecutar desde el repositorio: .venv/Scripts/python.exe tools/tarea2/build_notebooks.py
Las arquitecturas y la política de datos se extraen de Parte 1 sin modificarla.
"""
from pathlib import Path
import ast
import hashlib
import json
import pprint
import textwrap
from cell_status import add_completion_messages

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
source_path = ROOT / 'Tarea_2_Parte_1.ipynb'
modern_source = source_path.is_file()
original = json.loads((source_path if modern_source else ROOT / 'Tarea_2.ipynb').read_text(encoding='utf-8'))
definitions = {}
imports = []
constants = []
for index, cell in enumerate(original['cells']):
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            definitions[node.name] = ''.join(source.splitlines(keepends=True)[start - 1:node.end_lineno]).rstrip()
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PARTICIONES_ORIGINALES'
                                              for t in node.targets):
            original['metadata']['tarea2_particiones'] = ast.literal_eval(node.value)
        if index == (4 if modern_source else 5):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(source, node))
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                    {'IMAGE_EXTENSIONS', 'KINDS', 'WEIGHTS', 'WEIGHT_SHA256'} for t in node.targets):
                constants.append(ast.get_source_segment(source, node))


def extracted(names):
    return '\n\n\n'.join(definitions[n] for n in names.split())


GEOMETRY = extracted('leer_json reemplazar_atomico_con_reintento guardar_json calcular_sha256 yolo_a_cajas cajas_a_yolo '
                     'iou_cajas geometria_letterbox AjusteImagen ajustar_imagen reflejar_imagen filtrar_detecciones')
DOWNLOAD = extracted('abrir_url descargar_partes descargar_dataset')
AUDIT_GROUPS = [
    ('Etiquetas y catálogo', 'firma_archivos leer_clases ruta_etiqueta leer_etiquetas inspeccionar_imagenes'),
    ('Grupos y distribución', 'GruposConectados agrupar_similares resumir_auditoria'),
    ('Inspección visual', 'mostrar_muestra graficar_datos guardar_auditoria'),
    ('Auditoría de la fuente', 'auditar recuperar_etiquetas'),
    ('Depuración sin modificar originales', 'excluir_contradictorios actualizar_resumen depurar'),
]
CONTRACT = extracted('huella_dataset revisar_manifiesto formato_coco huella_particiones identidad_train revisar_particiones')
AUGMENT = extracted('politica_aumento validar_politica_aumento aumentar_entrenamiento DatosDeteccion')
METRICS = extracted('validar_predicciones calcular_metricas elegir_umbral evaluar metricas_por_tamano predicciones_coco')
MODELS = [
    ('Extractor propio y fases', extracted('ExtractorPropio ControlDetector')),
    ('Adaptador de pérdidas y predicciones', extracted('Detector')),
    ('Construcción de los cuatro detectores', extracted('crear_cnn_ssd crear_yolo crear_fasterrcnn crear_detector')),
]
# El nuevo código no usa BASE_DIR, ni el nombre o metadatos de ningún notebook.
MODELS = [(label, code.replace('(BASE_DIR / "resultados_etapas_1_2/ultralytics").resolve()',
                              '(RAIZ_ESTUDIO / "ultralytics").resolve()')) for label, code in MODELS]
OPTIM = extracted('adamw fijar_semilla capturar_azar restaurar_azar')
WEIGHTS = extracted('descargar_pesos')
ROBUST = extracted('condiciones_robustez degradar_imagen nombre_condicion dibujar_ejemplo')
RUNTIME = (HERE / 'runtime.py').read_text(encoding='utf-8')
REPORTING = (HERE / 'reporting.py').read_text(encoding='utf-8')
PREPARATION = (HERE / 'data_preparation.py').read_text(encoding='utf-8')
STANDALONE_ROBUSTNESS = (HERE / 'robustness_standalone.py').read_text(encoding='utf-8')
VERSION = hashlib.sha256(('\n'.join([GEOMETRY, DOWNLOAD, CONTRACT, AUGMENT, METRICS, OPTIM, WEIGHTS,
    ROBUST, RUNTIME, REPORTING, PREPARATION] + [c for _, c in MODELS]
    + [extracted(names) for _, names in AUDIT_GROUPS])).encode()).hexdigest()


def md(text):
    return dict(cell_type='markdown', metadata={}, source=textwrap.dedent(text).strip().splitlines(keepends=True))


def code(text):
    source = textwrap.dedent(text).strip() + '\n'
    ast.parse(source)
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[], source=source.splitlines(keepends=True))


def block(cells, title, source):
    cells.append(md('### ' + title))
    cells.append(code(source))


def sections(source):
    result = []
    for chunk in source.split('# %% ')[1:]:
        title, body = chunk.split('\n', 1)
        result.append((title, body))
    return result


def select_definitions(source, names):
    return '\n\n'.join(ast.get_source_segment(source, n) for n in ast.parse(source).body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)


def robustez_parte3(source):
    """Reutiliza la evaluación pareada con el checkpoint ya cargado y su carpeta de salida."""
    reemplazos = [
        ('ejecutar_robustez(cierre, auditoria, manifiesto, estadisticas, dispositivo=None, nombre=None)',
         'ejecutar_robustez_parte3(cierre, auditoria, manifiesto, detector, ficha, dispositivo, nombre)'),
        ('    dispositivo = dispositivo or DISPOSITIVO_EJECUCION\n', ''),
        ("    nombre = nombre or cierre['ganador_validacion']\n", ''),
        ('    detector, _, _ = cargar_mejor(nombre, auditoria, manifiesto, estadisticas, dispositivo)\n', ''),
        ("calcular_sha256(RAIZ_ESTUDIO / 'modelos' / nombre / 'mejor.pt')", "ficha['checkpoint']"),
        ("RAIZ_ESTUDIO / 'robustez' / nombre", "carpeta_robustez_parte3(cierre, nombre)"),
        ('graficar_robustez_final(nombre)', 'graficar_robustez_parte3(cierre, nombre)'),
        ('        etiqueta = nombre_condicion(condicion)\n',
         '        etiqueta = nombre_condicion(condicion)\n'
         "        print(f'Test de {nombre}, condición {etiqueta}: {len(ids)} imágenes.', flush=True)\n"),
    ]
    for antes, despues in reemplazos:
        if antes not in source:
            raise ValueError(f'Revisar la adaptación del protocolo para Parte 3: {antes!r}')
        source = source.replace(antes, despues)
    return source


def build(part):
    title = {1: 'Modelos desde cero', 2: 'Transfer learning y comparación final', 3: 'Robustez del mejor detector'}[part]
    names = {1: ['cnn_ssd', 'yolo_scratch'], 2: ['yolo_coco', 'fasterrcnn_coco'], 3: []}[part]
    introduccion = md('''
        Usa **Ejecutar todo** en un kernel nuevo, local o Colab. El notebook prepara o
        recupera el contrato de datos y reutiliza los checkpoints guardados. Configura `RUTA_PROYECTO`
        con la carpeta que contiene `resultados_tres_partes/` si abriste el notebook desde otra ubicación.

        Conserva la carpeta del estudio original. Para cada detector que quieras evaluar se necesitan
        `modelos/<modelo>/mejor.pt` y `mejor.sha256`. Si aún falta el cierre de las Partes 1 y 2,
        se compara en validación el mejor checkpoint de cada modelo disponible y se evalúa en test
        el de mayor mAP50–95, con umbral fijado en validación. El resultado queda en `robustez_provisional/`.
        Si existe una selección cerrada, se reutiliza y se guarda en `robustez/`.
        Esta parte ejecuta inferencia, sin entrenar ni retomar épocas.

        El adjunto registró para YOLO11n COCO un mAP50–95 de validación de **0,09572 en la época 26**.
        Para probar ese detector en otro equipo, copia su checkpoint y el hash desde el estudio que
        produjo ese resultado. La selección de este notebook usa los archivos que encuentre al ejecutarse.
        La inferencia usa CUDA cuando está disponible y CPU en caso contrario.
    ''' if part == 3 else '''
        Este notebook contiene su código y no necesita módulos Python del repositorio ni el notebook original.
        Los tres archivos comparten un contrato de datos y resultados en Drive. Ejecuta primero la preparación
        de la Parte 1; después cada parte funciona desde un kernel nuevo.

        **Uso local o en Colab:** `DISPOSITIVO_EJECUCION="auto"` elige CUDA si está disponible y CPU en caso contrario. En CPU el entrenamiento es más lento y AMP se desactiva.
        El entrenamiento y la evaluación están activados; se retoman los checkpoints compatibles.
        Los estados pendientes se muestran como pendientes, nunca como resultados cero.

        **Recorrido:** configuración → datos → arquitecturas → entrenamiento → evaluación → discusión.
        Los auxiliares se agrupan por responsabilidad. Las métricas usan escala 0–1.
    ''')
    portada = md(f'''
        # Tarea 2 · Parte {part}: {title}

        **Integrantes:** Diego Muñoz, Benjamín Arguelles e Ignacio Pastén.
    ''')
    portada['source'] = (''.join(portada['source']) + '\n\n' +
                         ''.join(introduccion['source'])).splitlines(keepends=True)

    cells = [portada, md('## 1. Configuración y entorno'), code(f'''
        from pathlib import Path
        import sys, os, importlib.util, importlib.metadata, subprocess

        PARTE = {part}
        # Misma carpeta y etiqueta en los tres notebooks. Cambiar etiqueta crea otro estudio.
        RUTA_PROYECTO = None  # None: MyDrive/Tarea2_DL en Colab; carpeta actual en local.
        VERSION_NOTEBOOK = '2026-11-09-colab-cpu-v3'
        DISPOSITIVO_EJECUCION = 'auto'  # 'auto', 'cpu' o 'cuda'.
        ETIQUETA_ESTUDIO = 'tres_partes_auto_v2_s42'
        EJECUTAR_ENTRENAMIENTO = {part in (1, 2)}
        EJECUTAR_DIAGNOSTICO = False  # Diagnóstico de 300 pasos sobre ocho imágenes de train.
        EJECUTAR_EVALUACION = {part in (1, 2)}
        EJECUTAR_ROBUSTEZ = {part == 3}
        INSTALAR_FALTANTES = True
        MODELOS_DE_ESTA_PARTE = {names!r}
        SEED, TAMANO = 42, 512
        VERSION_MOTOR = {VERSION!r}

        EN_KAGGLE = Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
        EN_COLAB = not EN_KAGGLE and (importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False)
        if EN_COLAB:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
        PROYECTO = Path(RUTA_PROYECTO) if RUTA_PROYECTO else (Path('/content/drive/MyDrive/Tarea2_DL') if EN_COLAB else Path.cwd())
        if not ETIQUETA_ESTUDIO or ETIQUETA_ESTUDIO in {{'.', '..'}} or any(c in ETIQUETA_ESTUDIO for c in '/\\\\:'):
            raise ValueError('ETIQUETA_ESTUDIO debe ser un nombre simple de carpeta')
        RAIZ_ESTUDIO = PROYECTO / 'resultados_tres_partes' / ETIQUETA_ESTUDIO
        DATA_DIR = Path('/content/fjordfish_datos') if EN_COLAB else PROYECTO / 'data'
        DATASET_DIR = DATA_DIR / 'extracted'
        PESOS_DIR = RAIZ_ESTUDIO / 'pesos_coco'
        RAIZ_ESTUDIO.mkdir(parents=True, exist_ok=True)
        CONFIG_ULTRALYTICS = Path('/content/tarea2_config/ultralytics') if EN_COLAB else RAIZ_ESTUDIO / 'ultralytics'
        (CONFIG_ULTRALYTICS / 'Ultralytics').mkdir(parents=True, exist_ok=True)
        os.environ['YOLO_CONFIG_DIR'] = str(CONFIG_ULTRALYTICS)
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        print('Notebook:', VERSION_NOTEBOOK)
        print('Estudio:', RAIZ_ESTUDIO)
        print('Datos:', DATASET_DIR)
    '''), code('''
        dependencias = {'numpy': 'numpy>=1.26', 'PIL': 'Pillow>=10', 'yaml': 'PyYAML>=6',
                        'matplotlib': 'matplotlib>=3.8', 'pycocotools': 'pycocotools>=2.0.8',
                        'torch': 'torch>=2.4', 'torchvision': 'torchvision>=0.19',
                        'ultralytics': 'ultralytics==8.4.174'}
        faltantes = [p for m, p in dependencias.items() if importlib.util.find_spec(m) is None]
        if importlib.util.find_spec('ultralytics') and importlib.metadata.version('ultralytics') != '8.4.174':
            raise RuntimeError('Instala ultralytics==8.4.174 y reinicia el kernel; no cambiar versiones durante una corrida.')
        if faltantes:
            if not INSTALAR_FALTANTES:
                raise RuntimeError(f'Faltan dependencias: {faltantes}')
            subprocess.run([sys.executable, '-m', 'pip', 'install', *faltantes], check=True)
    '''), code('\n'.join(imports) + '\n\n' + '\n\n'.join(constants) + '''

torch.set_num_threads(min(8, os.cpu_count() or 1))
if DISPOSITIVO_EJECUCION not in {'auto', 'cpu', 'cuda'}:
    raise ValueError('DISPOSITIVO_EJECUCION debe ser auto, cpu o cuda')
if DISPOSITIVO_EJECUCION == 'auto':
    DISPOSITIVO_EJECUCION = 'cuda' if torch.cuda.is_available() and torch.version.cuda is not None else 'cpu'
if DISPOSITIVO_EJECUCION == 'cuda' and (not torch.cuda.is_available() or torch.version.cuda is None):
    raise RuntimeError('Se solicitó CUDA pero PyTorch no la ofrece. Usa DISPOSITIVO_EJECUCION="cpu" o instala PyTorch CUDA.')
print('PyTorch:', torch.__version__, '| dispositivo:', DISPOSITIVO_EJECUCION, '| CUDA:', torch.cuda.is_available(), '| compilación CUDA:', torch.version.cuda)
if DISPOSITIVO_EJECUCION == 'cpu':
    print('Modo CPU activado. AMP desactivado; el entrenamiento puede tardar bastante más que en GPU.')
    if EN_COLAB:
        print('Colab: para usar GPU, selecciona un acelerador en Entorno de ejecución > Cambiar tipo de entorno de ejecución. Si sigue apareciendo +cpu, restablece el entorno administrado de Colab.')
''')]
    block(cells, 'Funciones de archivos y geometría', GEOMETRY)
    block(cells, 'Descarga verificada del dataset', DOWNLOAD)
    cells.append(md('''
        ## 2. Datos y particiones compartidas

        Se conservan los mismos IDs **2.022 train / 454 val / 454 test** y las 28 categorías declaradas
        (26 observadas). Se recupera un sidecar inequívoco y se apartan 72 imágenes contradictorias,
        sin alterar los originales: **2.930 imágenes y 6.236 objetos**.
        La firma comprueba que la depuración reproduce exactamente el conjunto utilizado anteriormente.
        No se redistribuyen imágenes para mejorar resultados. Val y test contienen 19 clases observadas cada uno.
        Los grupos dHash reducen similitudes entre particiones, pero no prueban independencia entre videos.
        La normalización de la CNN se calcula únicamente con train; los modelos COCO conservan sus transformaciones.
    '''))
    block(cells, 'Contrato explícito de IDs originales',
          'PARTICIONES_ORIGINALES = ' + pprint.pformat(original['metadata']['tarea2_particiones'], width=110, compact=True))
    block(cells, 'Identidad de datos y validación de particiones', CONTRACT)
    for title_, source in sections(RUNTIME)[:1]:
        block(cells, title_, source)
    if part in (1, 3):
        for label, identifiers in AUDIT_GROUPS:
            block(cells, label, extracted(identifiers))
        if part == 1:
            block(cells, 'Estadísticas y cobertura de anclas, solo train',
                  extracted('calcular_normalizacion revisar_anclas'))
        else:
            block(cells, 'Normalización, solo train', extracted('calcular_normalizacion'))
        for label, source in sections(PREPARATION):
            block(cells, label, source)
    if part == 1:
        cells.append(code('''
            auditoria, manifiesto, estadisticas = preparar_datos()
            print(json.dumps(auditoria['summary'], ensure_ascii=False, indent=2))
            print('Media:', estadisticas['mean'], '| desviación:', estadisticas['std'])
            cobertura_anclas = revisar_anclas(auditoria, manifiesto, TAMANO)
            guardar_json(RAIZ_ESTUDIO / 'datos/cobertura_anclas.json', cobertura_anclas)
            print('Cobertura de anclas IoU ≥ 0,5:', cobertura_anclas['recall_iou_50'])
            display(Image.open(RAIZ_ESTUDIO / 'datos/curacion/annotations.jpg'))
            display(Image.open(RAIZ_ESTUDIO / 'datos/curacion/class_distribution.png'))
        '''))
    elif part == 2:
        cells.append(code('''
            DATASET_DIR = asegurar_dataset()
            auditoria, manifiesto, estadisticas = cargar_contrato(DATASET_DIR)
            print('Contrato verificado:', {s: len(ids) for s, ids in manifiesto['partitions'].items()})
        '''))
    else:
        cells.append(code('''
            candidatos = [nombre for nombre in KINDS
                          if (RAIZ_ESTUDIO / 'modelos' / nombre / 'mejor.pt').is_file()]
            ruta_cierre = RAIZ_ESTUDIO / 'evaluacion/seleccion_cerrada.json'
            if EJECUTAR_ROBUSTEZ and not candidatos:
                raise FileNotFoundError(
                    f'Faltan los checkpoints entrenados en {RAIZ_ESTUDIO / "modelos"}. '
                    'Copia al menos un mejor.pt con su mejor.sha256 desde el estudio original.')
        '''))
    block(cells, 'Dataset y aumentos moderados, solo train', AUGMENT)
    cells.append(md('''
        ## 3. Arquitecturas y decisiones

        | Modelo | Inicialización | Diseño y fases |
        |---|---|---|
        | CNN propia + SSD | Aleatoria, sin ImageNet/COCO | Seis bloques propios (32/64/128/192/256/256); mapas stride 8/16/32/64. Torchvision aporta cabezales, matching, pérdidas y NMS SSD. |
        | YOLO11n desde cero | Aleatoria, configuración YAML | Misma arquitectura YOLO11n que el modelo COCO; todas las capas entrenables excepto DFL fijo. |
        | YOLO11n COCO | Checkpoint oficial verificado por SHA-256 | Cinco épocas congelando módulos 0–10 y sus BatchNorm; después se habilita el backbone, salvo DFL fijo. |
        | Faster R-CNN ResNet50-FPN | Checkpoint COCO oficial verificado | Predictor nuevo para 28 clases más fondo; cinco épocas con cuerpo ResNet congelado; luego layer4 entrenable. FPN/cabezales entrenables y FrozenBatchNorm. |

        Entrada 512 con letterbox para conservar proporciones; cajas evaluadas en coordenadas originales.
        Se mantienen las arquitecturas para poder contrastar el protocolo anterior. Las anclas y la resolución
        son hipótesis de diseño; los objetos pequeños y el fuerte desbalance requieren AP y soporte por clase.
        Los modelos tienen distinta capacidad y regímenes: la comparación será descriptiva, sin atribución causal
        exclusiva al preentrenamiento. Una semilla no permite afirmar significancia estadística.
    '''))
    for label, source in MODELS:
        block(cells, label, source)
    block(cells, 'Pesos oficiales COCO', WEIGHTS)
    block(cells, 'Optimizador, semilla y estado aleatorio', OPTIM)
    block(cells, 'Evaluador COCO y umbral de operación', METRICS)
    block(cells, 'Condiciones de degradación y dibujo de cajas', ROBUST)
    if part != 3:
        for label, source in sections(RUNTIME)[1:]:
            block(cells, label, source)
        # All helpers are defined before any training/evaluation action below.
        for label, source in sections(REPORTING):
            if label != 'Robustez pareada, sin recalibración':
                block(cells, label, source)
        cells.append(md('''
            ## 4. Entrenamiento y diagnóstico previo

            **Receta inicial, no un óptimo demostrado:** AdamW, tasa 1e-3 y 1e-4 en fine-tuning,
            weight decay 1e-4 (sin decay en bias/normalización), batch efectivo 8, máximo 60 épocas desde cero
            o 30 COCO (5 congeladas + 25 fine-tuning). Calentamiento lineal de tres épocas por fase,
            coseno hasta 1% de la tasa y clipping L2 10. AMP solo después de comprobar gradientes reales.
            El batch físico se mide entre 8/4/2/1 en ambas fases, dejando margen de VRAM.

            Train recibe reflexión p=0,5 y fotometría p=0,5 (brillo/contraste/saturación 0,85–1,15).
            Evitamos recortes agresivos inicialmente porque pueden eliminar organismos pequeños.
            La [documentación de SSD](https://pytorch.org/blog/torchvision-ssd-implementation/) fundamenta
            estudiar aumentos y duración; no demuestra que esta receta sea óptima para FjordFish.
            Warmup y clipping tienen precedentes en el [entrenador Ultralytics 8.4.174](https://github.com/ultralytics/ultralytics/blob/v8.4.174/ultralytics/engine/trainer.py).

            **Sin límite horario.** Parada tras diez épocas sin mejora de mAP de validación, desde época 20
            para scratch o 15 para COCO. Al cambiar a fine-tuning se reinician paciencia, optimizador y plan de tasa,
            conservando pesos; dentro de cada fase se restaura todo el estado al reanudar.
            No entrenar dos sesiones simultáneas sobre el mismo modelo/estudio.

            Primero se exige memorizar ocho imágenes de train sin aumentos (AP50 ≥ 0,5 y descenso de pérdida).
            Esta comprobación detecta problemas; su resultado **no es una evaluación de generalización**.
            Si falla, inspeccionar etiquetas, cajas, pérdidas y predicciones antes de aumentar el presupuesto.
        '''))
        cells.append(code('''
            configuraciones = {nombre: configuracion_modelo(nombre) for nombre in MODELOS_DE_ESTA_PARTE}
            display(Markdown(tabla_markdown([
                dict(modelo=n, epocas=c['epocas'], congeladas=c['epocas_congeladas'],
                     tasa=c['tasa'], tasa_fina=c['tasa_fina'], batch_efectivo=c['batch_efectivo'],
                     minimo_epocas=c['minimo_epocas'], paciencia=c['paciencia'])
                for n, c in configuraciones.items()])))
            if EJECUTAR_ENTRENAMIENTO or EJECUTAR_DIAGNOSTICO or EJECUTAR_EVALUACION:
                exigir_dispositivo(DISPOSITIVO_EJECUCION)
                if PARTE == 2:
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
                print('Entrenamiento desactivado. Cambia EJECUTAR_ENTRENAMIENTO al estar listo.')
        '''))
        cells.append(md('''
            ## 5. Evaluación y comparación

            Validación selecciona el mejor checkpoint por mAP50–95 y el umbral por F1.
            AP utiliza predicciones desde score 0,001, sin recortarlas al umbral de P/R/F1.
            Clases sin GT se informan como AP no definida. Las pérdidas de modelos distintos no comparten escala.
            Los tiempos separan entrenamiento, validación y escritura; la inferencia usa batch 1 e incluye
            apertura de imagen, preprocesamiento y posprocesamiento, sin tiempo de métricas COCO.
            Cada ejemplo conserva el GT amarillo y muestra las predicciones verdes sobre el umbral elegido.
        '''))
        cells.append(code('''
            if EJECUTAR_EVALUACION:
                for nombre in MODELOS_DE_ESTA_PARTE:
                    evaluar_validacion(nombre, auditoria, manifiesto, estadisticas, dispositivo=DISPOSITIVO_EJECUCION)
                comparacion_validacion = comparar_disponibles(MODELOS_DE_ESTA_PARTE, 'val')
            else:
                print('Evaluación pendiente; se necesitan entrenamientos completos.')
        '''))
        if part == 1:
            cells.append(md('''
                **Test se reserva para la Parte 2**, después de cerrar los cuatro modelos y umbrales.
                La comparación en validación permite revisar el entrenamiento sin consultar test.
                Las corridas antiguas CPU son antecedentes; no se importan como resultados de este protocolo.
            '''))
        else:
            cells.append(code('''
                if EJECUTAR_EVALUACION:
                    # Reevaluar también las bases con el mismo dispositivo/software y batch 1.
                    for nombre in ('cnn_ssd', 'yolo_scratch'):
                        evaluar_validacion(nombre, auditoria, manifiesto, estadisticas, dispositivo=DISPOSITIVO_EJECUCION)
                    cierre = cerrar_seleccion(auditoria, manifiesto, estadisticas)
                    print('Ganador fijado con validación:', cierre['ganador_validacion'])
                    for nombre in KINDS:
                        evaluar_test(nombre, cierre, auditoria, manifiesto, estadisticas, dispositivo=DISPOSITIVO_EJECUCION)
                    comparacion_test = comparar_disponibles(KINDS, 'test')
            '''))
    else:
        for label, source in sections(RUNTIME)[1:]:
            selected = select_definitions(source, {'configuracion_modelo', 'validar_configuracion',
                'identidad_corrida', 'leer_estado', 'predecir_ids'})
            if selected:
                block(cells, label, selected)
        for label, source in sections(REPORTING):
            if label == 'Figuras y comparación':
                source = select_definitions(source, {'guardar_ejemplos'})
            elif label == 'Robustez pareada, sin recalibración':
                standalone_defs = STANDALONE_ROBUSTNESS.split('# Acción de Parte 3\n', 1)[0]
                for standalone_label, standalone_source in sections(standalone_defs):
                    block(cells, standalone_label, standalone_source)
                    cells[-1]['metadata']['tarea2_solo_parte3'] = True
                source = robustez_parte3(source)
            block(cells, label, source)
            if label == 'Robustez pareada, sin recalibración':
                cells[-1]['metadata']['tarea2_solo_parte3'] = True
        cells.append(md('''
            ## 4. Evaluación de robustez

            El detector se selecciona por mAP50–95 de **validación**, antes de consultar test.
            Sobre sus mismas 454 imágenes de test se evalúan siete condiciones: limpio, iluminación ×0,7/×0,4,
            desenfoque gaussiano radio 1/2 píxeles y ruido gaussiano σ=0,02/0,05 sobre intensidades 0–1.
            El ruido usa una semilla determinista por imagen/condición. No cambian cajas, pesos ni umbral.
            Se informa caída absoluta y relativa respecto al limpio; si AP limpia es cero, la caída relativa es N/D.
            No se exige monotonía: la clasificación de peores condiciones debe salir de las mediciones.
        '''))
        cells.append(code(STANDALONE_ROBUSTNESS.split('# Acción de Parte 3\n', 1)[1]))
    cells.append(md('''
        ## 5. Resultados y discusión

        Revisa la caída de mAP50–95 respecto a limpio y los ejemplos pareados de las condiciones
        con mayor caída. Usa el resumen CSV y la figura de la carpeta del modelo. Cuando la selección
        es provisional, las métricas corresponden únicamente a los checkpoints disponibles en esa ejecución.
    ''' if part == 3 else '''
        ## 6. Resultados, discusión y evidencia de la rúbrica

        **Estado de entrega de este notebook:** código preparado; no se han ejecutado aquí los cuatro
        entrenamientos completos. Las salidas históricas y las pruebas sintéticas no se presentan como resultados nuevos.

        Completar la discusión después de ejecutar:

        - Contrastar curvas de train/validación, mejor época y motivo de término; diferenciar observaciones de hipótesis.
        - Comparar AP, precisión, recall y costo con valores reales; examinar ejemplos de errores y soportes por clase.
        - Explicar el efecto del desbalance, las especies ausentes de val/test y la falta de imágenes totalmente vacías.
        - En robustez, identificar las mayores caídas medidas y comprobarlas con ejemplos pareados; considerar los aumentos de train.

        La matriz siguiente marca evidencia disponible y pendiente; **no asigna una nota ni certifica la calidad de la discusión**.
        Los once criterios técnicos suman 60 puntos para este grupo de tres integrantes.
        Video de máximo 20 minutos y diapositivas son entregables adicionales que estos notebooks no generan.
    '''))
    cells.append(code('''
        if EJECUTAR_ROBUSTEZ and cierre.get('alcance') != 'provisional':
            evidencias_rubrica = actualizar_rubrica()
    ''' if part == 3 else 'evidencias_rubrica = actualizar_rubrica()'))
    cells.append(md('''
        **Referencias:** [FjordFish](https://zenodo.org/records/17950781),
        [SSD Torchvision](https://pytorch.org/blog/torchvision-ssd-implementation/),
        [Faster R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html),
        [YOLO11](https://docs.ultralytics.com/models/yolo11/),
        [COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

        El contrato se guarda en `datos/`; cada modelo tiene `ultimo.pt`, `mejor.pt`, hashes y estados en `modelos/`.
        Las evaluaciones y umbrales se guardan en `evaluacion/`; comparación en `comparacion/` y robustez en `robustez/`.
        Conserva la carpeta completa al retomar en otro runtime. Para cambiar código o hiperparámetros utiliza otro estudio.
    '''))
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'colab': {'provenance': [], 'toc_visible': True}}, cells=cells)
    for i, cell in enumerate(cells):
        cell['id'] = f'parte{part}-{i:03d}'
    add_completion_messages(notebook)
    filenames = {1: 'Tarea_2_Parte_1.ipynb', 2: 'Tarea_2_Parte_2.ipynb',
                 3: 'Tarea_2_Parte_3.ipynb'}
    path = ROOT / filenames[part]
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(path.name, len(cells), 'celdas;', sum(len(''.join(c['source']).splitlines()) for c in cells), 'líneas')


if __name__ == '__main__':
    for part in (1, 2, 3):
        build(part)
