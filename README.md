# Tarea 2 — notebooks para ejecución local o Colab

**Versión unificada:** [Tarea_2.ipynb](Tarea_2.ipynb) reúne las tres partes corregidas.
En local o Colab, usar **Ejecutar todo** para entrenar o retomar los cuatro modelos,
compararlos y evaluar robustez. Comparte carpeta, etiqueta, protocolo y checkpoints compatibles
con las versiones separadas; no necesita abrirlas ni importarlas.
**Versión Kaggle:** [Tarea_2_Kaggle.ipynb](Tarea_2_Kaggle.ipynb) detecta Kaggle,
lee el dataset adjunto desde `/kaggle/input` y escribe el estudio en `/kaggle/working`.
Adjunta el dataset extraído conservando la carpeta `FjordFish/`; activa GPU e Internet en
las opciones de sesión. Para reanudar tras una sesión nueva, adjunta el output anterior y
define `RUTA_ESTUDIO_PREVIO_KAGGLE` en la configuración.
El unificado anterior se conserva en `Material/Tarea_2_antes_unificar_1aadf1d6485a.ipynb`.
No ejecutar el unificado y una parte simultáneamente sobre el mismo modelo/estudio.

Cada celda de código termina con una confirmación como `[OK] Celda de código 03/37 finalizada sin errores.`
Si la celda falla o se interrumpe, no alcanza esa confirmación y Jupyter/Colab muestra el error original.
Los mensajes están incluidos en los notebooks y sus generadores; no requieren instalar módulos adicionales.

La versión reorganizada se ejecuta en este orden:

1. [Parte 1: modelos desde cero](Tarea_2_Parte_1.ipynb): preparar datos y entrenar CNN propia + SSD y YOLO11n.
2. [Parte 2: transferencia](Tarea_2_Parte_2.ipynb): entrenar YOLO11n COCO y Faster R-CNN; comparar los cuatro en test.
3. [Parte 3: robustez](Tarea_2_Parte_3.ipynb): evaluar el ganador con iluminación, desenfoque y ruido.

Cada archivo incluye su código y abre en un kernel nuevo. **No necesita importar módulos del repositorio ni leer `Tarea_2.ipynb`.**
Los tres usan la misma `RUTA_PROYECTO` y `ETIQUETA_ESTUDIO`; comparten datos y resultados en la carpeta del proyecto.

Por defecto `DISPOSITIVO_EJECUCION="auto"` selecciona CUDA si está disponible; si no, usa CPU.
La ruta CPU permite ejecutar localmente con PyTorch `+cpu`; AMP queda desactivado y el entrenamiento
puede tardar mucho más. Se puede fijar `"cpu"` o `"cuda"` en la celda de configuración.
El arranque actualizado imprime `Notebook: 2026-11-09-colab-cpu-v3`.
Si una copia abierta en Colab sigue diciendo que el entrenamiento requiere CUDA, vuelve a subir el
archivo actualizado desde **Archivo > Subir notebook**, reinicia la sesión y ejecuta desde el principio.
Reiniciar la sesión por sí solo conserva el código de la copia anterior.
La configuración de Ultralytics se crea antes de importarlo; en Colab utiliza
`/content/tarea2_config/ultralytics/Ultralytics`, mientras los checkpoints siguen guardándose en Drive.
En Colab se monta Drive en `MyDrive/Tarea2_DL`,
las imágenes se extraen en `/content/fjordfish_datos` y el nuevo estudio se guarda en
`resultados_tres_partes/tres_partes_auto_v2_s42/`. Los estudios antiguos permanecen intactos.

**Entrenamiento y evaluación activados en Partes 1 y 2.** Abrir Parte 1 localmente o en Colab y usar
**Ejecutar todo**; al terminar, hacer lo mismo con Parte 2. La Parte 1 prepara primero el contrato de datos.
`EJECUTAR_ENTRENAMIENTO=True` inicia o retoma los modelos de cada parte.
Antes de cada corrida nueva, el código exige superar el diagnóstico de ocho imágenes de train
y calibra el batch y habilita AMP solo con CUDA. En CPU usa batch físico 1 y acumulación de ocho
microbatches para mantener el batch efectivo. La evaluación posterior ya está habilitada con `EJECUTAR_EVALUACION=True`.
En [Parte 3](Tarea_2_Parte_3.ipynb), usa **Ejecutar todo** con al menos un `mejor.pt` y su `mejor.sha256`
en la carpeta del modelo. El notebook prepara los datos si faltan y reutiliza la selección cerrada si existe.
Con una corrida parcial, elige entre los checkpoints disponibles por mAP50–95 de validación,
calibra el umbral con validación y guarda los resultados en `robustez_provisional/`.
El entrenamiento termina al alcanzar el máximo de épocas o agotar la paciencia.
No ejecutar simultáneamente dos sesiones sobre el mismo modelo y estudio.

El nuevo protocolo utiliza 60 épocas máximas desde cero y 30 COCO (cinco congeladas),
batch efectivo 8, warmup, coseno, clipping y aumentos moderados. No se garantiza una mejora
antes de ejecutar los experimentos. Test se reserva hasta cerrar modelos y umbrales con validación.

Ver [diagnóstico y revisión de la rúbrica](docs/Diagnostico_Tarea_2.md).
La verificación reproducible está en `tools/tarea2/verify_notebooks.py`; el generador de desarrollo
`tools/tarea2/build_notebooks.py` permite reconstruir los tres archivos. Esos scripts **no se suben a Colab**
ni son dependencias de ejecución de los notebooks.
`tools/tarea2/build_unified.py` reúne las partes actuales y comprueba que sus funciones compartidas coincidan.

## Instrucciones conservadas del notebook histórico

Este bloque describe el respaldo anterior a la unificación, no el flujo actual de arriba.

# Tarea 2 — flujo completo desde el notebook, CPU o CUDA

Archivo histórico: **Material/Tarea_2_antes_unificar_1aadf1d6485a.ipynb**, 95 celdas. Para iniciar:

1. Reiniciar el kernel/runtime.
2. Dejar en la primera celda `ACCION_LOCAL="completo"`, `MODO_EJECUCION="auto"`,
   `DISPOSITIVO_PARTE1="auto"` y `ETIQUETA_ESTUDIO=None`.
3. Usar **Ejecutar todo**. No activar individualmente todos los interruptores.

Al no detectar CUDA, se usa CPU float32 para los cuatro modelos, la calibración,
comparación, robustez y reproducción en un proceso nuevo. AMP se desactiva en CPU.
CUDA permanece como alternativa cuando está disponible; nunca se declara aprobada
una prueba CUDA omitida. La CPU local no se hace pasar por una GPU.

## Estudio nuevo y recuperación

La ampliación cambia el protocolo científico. El estudio automático se llama, por
ejemplo, `local_cpu_integral_v2_s42` o `colab_cuda_integral_v2_s42`. Las ejecuciones
posteriores con la misma etiqueta retoman sus checkpoints. Para comenzar otro
experimento desde cero, indicar otra etiqueta; no se borran resultados anteriores.
Las corridas antiguas siguen en `local_cpu_s42`, junto con sus respaldos del notebook.

CNN/YOLO base mantienen diez épocas, batch 2 y el motor original adjunto. Los dos
modelos COCO usan ambas fases; en CPU se calibra batch físico 1 con acumulación 8.
La calibración mide una época real por fase antes de fijar las épocas. El tope por
modelo es 24 horas CPU o 6 CUDA, configurable en `HORAS_MAXIMAS_POR_MODELO`; no obliga
a consumir todo ese tiempo. No hay un corte adicional de cinco horas por sesión.
Faster R-CNN en CPU puede necesitar muchas horas. Los cortes por presupuesto se
registran y el cierre no consulta test mientras falten modelos o fases.

## Acciones

`revisar`, `entrenar_base`, `ajustar`, `coco`, `comparar`, `robustez`, `entrega` y
`completo` activan solamente sus etapas. El ajuste de veinte épocas adicionales es
una opción CPU separada; no altera automáticamente las bases de la comparación.
Los interruptores antiguos PREPARAR/ENTRENAR_PARTE1_NUEVA están apagados para evitar
lanzar una segunda corrida CUDA después de una base CPU.

## Local y Colab

Localmente seleccionar `.venv/Scripts/python.exe`. Si el kernel abre fuera del
proyecto, indicar `RUTA_PROYECTO`. En Colab se monta Drive y se localiza esta copia
Tarea_2.ipynb; las imágenes se extraen en /content/fjordfish_datos y los checkpoints
se guardan en Drive. No es necesario ejecutar scripts externos ni clonar repositorios.
Los mismos IDs 2022/454/454 se usan en todos los modelos.

El cierre exige las pruebas del dispositivo utilizado y reproducción aislada de los
cuatro modelos. En CPU solo se admiten omisiones explícitas de pruebas CUDA. La memoria
CUDA se registra como no aplicable; el perfil no inventa un pico de memoria RAM.
La procedencia del dataset se toma de su carpeta real, también al empaquetar en Colab.

Las comprobaciones de desarrollo usan fixtures sintéticas temporales. No son resultados
ni entrenamientos principales de FjordFish. Los reportes de esta corrección están en
`resultados_etapas_1_2/etapas_3_5/diagnosticos/cpu_integral_v2/`. La edición no lanzó las
corridas largas. Se creó un respaldo anterior `Tarea_2_respaldo_antes_cpu_integral_*.ipynb`.
Diapositivas y video siguen fuera de esta intervención.

Validación final: 79 casos, 76 aprobados y 3 omisiones CUDA explícitas; cero errores. El archivo `suite_completa.json` registra la revisión compatible con el código actual.
