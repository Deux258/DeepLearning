"""Create a Kaggle-ready copy of the unified notebook without changing it."""
import copy
import ast
import json
from pathlib import Path
from cell_status import add_completion_messages

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Tarea_2.ipynb'
TARGET = ROOT / 'Tarea_2_Kaggle.ipynb'


def main():
    notebook = json.loads(SOURCE.read_text(encoding='utf-8'))
    notebook = copy.deepcopy(notebook)

    intro = notebook['cells'][0]
    intro['source'] = '''# Tarea 2 · Versión Kaggle

Notebook autocontenido con las tres partes: entrenamiento desde cero, transferencia COCO,
comparación y robustez. Usa **Add Input** para adjuntar el dataset FjordFish extraído o un ZIP
que contenga `FjordFish/data.yaml`; si no lo adjuntas, el notebook intenta descargarlo de Zenodo.
Selecciona un acelerador GPU, habilita Internet en Session options y ejecuta las celdas desde
arriba. Los ZIP adjuntos se extraen en `/kaggle/temp`; la descarga oficial también usa ese espacio.

El dataset se lee desde `/kaggle/input` (solo lectura) y los contratos, checkpoints, métricas y
figuras se escriben en `/kaggle/working/resultados_tres_partes/kaggle_tres_partes_v1_s42`.
La estructura del dataset debe conservar la carpeta `FjordFish/` con `data.yaml`, `images/` y
`labels/`. Si hay más de un `data.yaml` adjunto, especifica su carpeta contenedora en
`RUTA_DATASET_KAGGLE` en la configuración. Habilita Internet en Session options para instalar
dependencias faltantes y descargar los pesos COCO oficiales.

Los checkpoints se guardan cada 50 actualizaciones. Kaggle limita cada sesión y el espacio de
salida; al pausar o terminar una sesión, conserva el contenido de `/kaggle/working` como output.
Para reanudar en una sesión limpia, adjunta el output anterior como input y define
`RUTA_ESTUDIO_PREVIO_KAGGLE` con la ruta al directorio del estudio anterior. Kaggle documenta
los ajustes de sesión y el guardado de outputs [aquí](https://www.kaggle.com/docs/notebooks).

**Importante:** activar GPU no amplía el límite de duración de la sesión. El protocolo completo
puede requerir varias sesiones; si se reanuda, el motor verifica la identidad del checkpoint,
datos, software y dispositivo antes de continuar.
'''.splitlines(keepends=True)

    config = next(c for c in notebook['cells'] if c['cell_type'] == 'code'
                  and 'RUTA_PROYECTO = None' in ''.join(c['source']))
    source = ''.join(config['source'])
    source = source.replace("ETIQUETA_ESTUDIO = 'tres_partes_auto_v2_s42'",
                            "ETIQUETA_ESTUDIO = 'kaggle_tres_partes_v1_s42'")
    source = source.replace('import sys, os, importlib.util, importlib.metadata, subprocess',
                            'import sys, os, shutil, zipfile, importlib.util, importlib.metadata, subprocess')
    source = source.replace("RUTA_PROYECTO = None  # None: MyDrive/Tarea2_DL en Colab; carpeta actual en local.",
                            "RUTA_PROYECTO = None  # En Kaggle usa /kaggle/working por defecto.\nRUTA_DATASET_KAGGLE = None  # Opcional: carpeta bajo /kaggle/input que contiene FjordFish/.\nRUTA_ESTUDIO_PREVIO_KAGGLE = None  # Opcional: ruta al estudio anterior adjuntado como input.")
    start = source.index('EN_KAGGLE =')
    end_marker = "print('Datos:', DATASET_DIR)"
    end = source.index(end_marker, start) + len(end_marker)
    environment = '''EN_KAGGLE = Path('/kaggle/input').is_dir() and Path('/kaggle/working').is_dir()
EN_COLAB = not EN_KAGGLE and (importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False)
if EN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
PROYECTO = Path(RUTA_PROYECTO) if RUTA_PROYECTO else (
    Path('/kaggle/working') if EN_KAGGLE else
    Path('/content/drive/MyDrive/Tarea2_DL') if EN_COLAB else Path.cwd())
if not ETIQUETA_ESTUDIO or ETIQUETA_ESTUDIO in {'.', '..'} or any(c in ETIQUETA_ESTUDIO for c in '/\\\\:'):
    raise ValueError('ETIQUETA_ESTUDIO debe ser un nombre simple de carpeta')
RAIZ_ESTUDIO = PROYECTO / 'resultados_tres_partes' / ETIQUETA_ESTUDIO
DATA_DIR = Path('/content/fjordfish_datos') if EN_COLAB else PROYECTO / 'data'
if EN_KAGGLE:
    if RUTA_DATASET_KAGGLE:
        DATASET_DIR = Path(RUTA_DATASET_KAGGLE).resolve()
        if len(list(DATASET_DIR.rglob('data.yaml'))) != 1:
            raise RuntimeError('RUTA_DATASET_KAGGLE debe contener exactamente un data.yaml del dataset FjordFish.')
    else:
        kaggle_input = Path('/kaggle/input')
        yaml_files = sorted(kaggle_input.rglob('data.yaml'))
        if yaml_files:
            roots = sorted({p.parent.parent.resolve() for p in yaml_files})
            if len(roots) != 1:
                raise RuntimeError(
                    f'Hay {len(yaml_files)} data.yaml en varios datasets. Indica en RUTA_DATASET_KAGGLE '
                    'la carpeta que contiene FjordFish/.')
            DATASET_DIR = roots[0]
            if len(list(DATASET_DIR.rglob('data.yaml'))) != 1:
                raise RuntimeError('El dataset de Kaggle debe tener exactamente un data.yaml bajo su carpeta raíz.')
        else:
            zip_candidates = []
            for archive_path in sorted(p for p in kaggle_input.rglob('*') if p.is_file() and p.suffix.lower() == '.zip'):
                try:
                    with zipfile.ZipFile(archive_path) as archive:
                        if any(name.replace('\\\\', '/').endswith('FjordFish/data.yaml')
                               for name in archive.namelist()):
                            zip_candidates.append(archive_path)
                except (OSError, zipfile.BadZipFile):
                    continue
            if len(zip_candidates) > 1:
                raise RuntimeError(
                    f'Se encontraron {len(zip_candidates)} ZIP compatibles con FjordFish. '
                    'Adjunta solo uno o indica la carpeta extraída en RUTA_DATASET_KAGGLE.')
            if not zip_candidates:
                DATA_DIR = Path('/kaggle/temp/tarea2_fjordfish_fuente')
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                DATASET_DIR = DATA_DIR / 'extracted'
                print('No hay dataset adjunto; se descargará FjordFish de Zenodo al preparar los datos.')
            else:
                archive_path = zip_candidates[0]
                target = Path('/kaggle/temp/tarea2_fjordfish')
                marker = target / '.kaggle_extraction.txt'
                stat = archive_path.stat()
                archive_identity = f'{archive_path.name}|{stat.st_size}|{stat.st_mtime_ns}'
                if not marker.is_file() or marker.read_text(encoding='utf-8') != archive_identity:
                    if target.exists():
                        shutil.rmtree(target)
                    target.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(archive_path) as archive:
                        files = [item for item in archive.infolist() if not item.is_dir()]
                        total = sum(item.file_size for item in files)
                        if shutil.disk_usage(target).free < total + 512 * 1024 * 1024:
                            raise OSError(
                                f'No hay espacio temporal suficiente para extraer {archive_path.name} '
                                f'({total / 1024**3:.1f} GiB). Adjunta el dataset ya extraído como Kaggle Input.')
                        root_resolved = target.resolve()
                        for item in files:
                            relative = Path(item.filename.replace('\\\\', '/'))
                            if relative.is_absolute() or '..' in relative.parts:
                                raise ValueError(f'Ruta insegura dentro del ZIP: {item.filename}')
                            destination = (target / relative).resolve()
                            if root_resolved not in destination.parents:
                                raise ValueError(f'Ruta insegura dentro del ZIP: {item.filename}')
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with archive.open(item) as src, destination.open('wb') as dst:
                                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                    if not (target / 'FjordFish' / 'data.yaml').is_file():
                        raise RuntimeError('El ZIP no conserva la ruta esperada FjordFish/data.yaml.')
                if len(list(target.rglob('data.yaml'))) != 1:
                    raise RuntimeError('El ZIP debe contener exactamente un data.yaml del dataset FjordFish.')
                    marker.write_text(archive_identity, encoding='utf-8')
                DATASET_DIR = target
else:
    DATASET_DIR = DATA_DIR / 'extracted'
PESOS_DIR = RAIZ_ESTUDIO / 'pesos_coco'
RAIZ_ESTUDIO.mkdir(parents=True, exist_ok=True)
if EN_KAGGLE and RUTA_ESTUDIO_PREVIO_KAGGLE:
    previo = Path(RUTA_ESTUDIO_PREVIO_KAGGLE).resolve()
    if not previo.is_dir():
        raise FileNotFoundError(f'No existe el estudio previo adjuntado: {previo}')
    if any(RAIZ_ESTUDIO.iterdir()):
        raise RuntimeError('El directorio de estudio destino no está vacío; no se copiará encima de checkpoints existentes.')
    shutil.copytree(previo, RAIZ_ESTUDIO, dirs_exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(RAIZ_ESTUDIO / 'ultralytics')
os.environ['YOLO_AUTOINSTALL'] = 'false'
print('Entorno:', 'Kaggle' if EN_KAGGLE else 'Colab' if EN_COLAB else 'local')
print('Estudio:', RAIZ_ESTUDIO)
print('Datos:', DATASET_DIR)
'''
    source = source[:start] + environment + source[end:]
    config['source'] = source.splitlines(keepends=True)

    dependencies = next(c for c in notebook['cells'] if c['cell_type'] == 'code'
                        and 'dependencias = {' in ''.join(c['source']))
    dep_source = ''.join(dependencies['source'])
    missing_line = "faltantes = [p for m, p in dependencias.items() if importlib.util.find_spec(m) is None]"
    preflight = """faltantes = [p for m, p in dependencias.items() if importlib.util.find_spec(m) is None]
necesita_pypi = bool(faltantes) or (
    importlib.util.find_spec('ultralytics') is not None
    and importlib.metadata.version('ultralytics') != '8.4.174')
if EN_KAGGLE and necesita_pypi:
    from urllib.request import urlopen
    try:
        with urlopen('https://pypi.org/simple/', timeout=5) as respuesta:
            respuesta.read(1)
    except Exception as error:
        raise RuntimeError(
            'Kaggle no puede conectar con PyPI y faltan paquetes o la versión fijada de Ultralytics. '
            'Abre Session options > Internet, actívalo, reinicia la sesión y vuelve a ejecutar desde arriba.') from error"""
    if missing_line not in dep_source:
        raise ValueError('No se encontró el inventario de dependencias esperado.')
    dep_source = dep_source.replace(missing_line, preflight)
    outdated = """if importlib.util.find_spec('ultralytics') and importlib.metadata.version('ultralytics') != '8.4.174':
    raise RuntimeError('Instala ultralytics==8.4.174 y reinicia el kernel; no cambiar versiones durante una corrida.')"""
    replace = """if importlib.util.find_spec('ultralytics') and importlib.metadata.version('ultralytics') != '8.4.174':
    if not INSTALAR_FALTANTES:
        raise RuntimeError('Se requiere ultralytics==8.4.174; habilita INSTALAR_FALTANTES e Internet para instalarlo.')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'ultralytics==8.4.174'], check=True)
    importlib.invalidate_caches()"""
    if outdated not in dep_source:
        raise ValueError('No se encontró la comprobación de versión de Ultralytics esperada.')
    dependencies['source'] = dep_source.replace(outdated, replace).splitlines(keepends=True)

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
            ast.parse(''.join(cell['source']))

    add_completion_messages(notebook)
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Creado: {TARGET.name} ({len(notebook["cells"])} celdas)')


if __name__ == '__main__':
    main()
