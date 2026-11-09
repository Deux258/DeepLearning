# Tarea 2 — tres notebooks para Colab con GPU

La versión reorganizada se ejecuta en este orden:

1. [Parte 1: modelos desde cero](Tarea_2_Parte_1.ipynb): preparar datos y entrenar CNN propia + SSD y YOLO11n.
2. [Parte 2: transferencia](Tarea_2_Parte_2.ipynb): entrenar YOLO11n COCO y Faster R-CNN; comparar los cuatro en test.
3. [Parte 3: robustez](Tarea_2_Parte_3.ipynb): evaluar el ganador con iluminación, desenfoque y ruido.

Cada archivo incluye su código y abre en un kernel nuevo. **No necesita importar módulos del repositorio ni leer `Tarea_2.ipynb`.**
Los tres usan la misma `RUTA_PROYECTO` y `ETIQUETA_ESTUDIO`; comparten datos y resultados en Drive.

En Colab, seleccionar un runtime **GPU**. Por defecto se monta Drive en `MyDrive/Tarea2_DL`,
las imágenes se extraen en `/content/fjordfish_datos` y el nuevo estudio se guarda en
`resultados_tres_partes/colab_gpu_tres_partes_v1_s42/`. Los estudios antiguos permanecen intactos.

**Entrenamiento y evaluación activados en Partes 1 y 2.** Abrir Parte 1 en Colab con GPU y usar
**Ejecutar todo**; al terminar, hacer lo mismo con Parte 2. La Parte 1 prepara primero el contrato de datos.
`EJECUTAR_ENTRENAMIENTO=True` inicia o retoma los modelos de cada parte.
Antes de cada corrida nueva, el código exige superar el diagnóstico de ocho imágenes de train
y calibra batch/AMP en GPU. La evaluación posterior ya está habilitada con `EJECUTAR_EVALUACION=True`.
En Parte 3 activar `EJECUTAR_ROBUSTEZ=True`. No hay límite horario; se termina por épocas o paciencia.
No ejecutar simultáneamente dos sesiones sobre el mismo modelo y estudio.

El nuevo protocolo utiliza 60 épocas máximas desde cero y 30 COCO (cinco congeladas),
batch efectivo 8, warmup, coseno, clipping y aumentos moderados. No se garantiza una mejora
antes de ejecutar los experimentos. Test se reserva hasta cerrar modelos y umbrales con validación.
Una instalación de PyTorch solo CPU impide iniciar los entrenamientos principales.

Ver [diagnóstico y revisión de la rúbrica](docs/Diagnostico_Tarea_2.md).
La verificación reproducible está en `tools/tarea2/verify_notebooks.py`; el generador de desarrollo
`tools/tarea2/build_notebooks.py` permite reconstruir los tres archivos. Esos scripts **no se suben a Colab**
ni son dependencias de ejecución de los notebooks.

## Instrucciones conservadas del notebook original

# Tarea 2 — flujo completo desde el notebook, CPU o CUDA

Archivo: **Tarea_2.ipynb**, 95 celdas. Para iniciar:

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
