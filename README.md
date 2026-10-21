# Tarea 2 — notebook completo

El entregable principal es `Tarea_2.ipynb`. Todo el código ejecutable, sus pruebas,
explicaciones, gráficos y análisis están dentro del notebook. Los archivos Python
auxiliares de desarrollo no son dependencias del entregable.

## Ejecución

1. Abrir el notebook en la carpeta del proyecto y seleccionar el entorno Python.
   En este equipo el intérprete local es `.venv/Scripts/python.exe`.
2. Revisar las rutas y dependencias de la sección 1. Los originales están en
   `data/extracted/` y los pesos en `resultados_etapas_1_2/pesos/`.
3. Ejecutar las secciones 1–8 para auditoría, particiones e integración.
4. Las secciones 9–12 definen y prueban el entrenamiento recuperable, la
   comparación y la robustez. Sus pruebas breves se pueden ejecutar en CPU.
5. Para entrenar en GPU, definir `RUTA_PERSISTENTE` en la sección 9 con una carpeta
   persistente ya montada y activar `EJECUTAR_ENTRENAMIENTOS`. La calibración fija
   el batch físico común antes de iniciar los cuatro modelos. Recorre una época
   por fase: completa en los modelos desde cero y congelada/fine-tuning en los
   dos preentrenados. Comprueba una actualización de pesos antes congelados,
   usa AMP y registra los máximos de memoria asignada y reservada. Un OOM en
   cualquiera de las fases obliga a repetir todos los modelos con batch 1.
6. Al terminar las corridas, activar `EJECUTAR_COMPARACION`. Se cierran decisiones
   con validación antes de evaluar test y medir latencia.
7. Activar `EJECUTAR_ROBUSTEZ` para evaluar el ganador de validación en seis
   degradaciones y la condición limpia, conservando pesos y umbral.

Los tres interruptores de ejecuciones prolongadas se entregan desactivados.
`EJECUTAR_PRUEBAS_NUEVAS` permite repetir las pruebas de aceptación. Los resultados
sintéticos son diagnósticos y no se admiten como evaluación final de la tarea.

## Recuperación

Mantener ruta, etiqueta de estudio y configuración al retomar. Los checkpoints
v2 incluyen pesos, optimizador, planificador, escalador AMP, orden y cursor de
muestreo, fase, historial y generadores aleatorios. Los checksums verifican su
integridad y se conserva una generación anterior. Una ventana interrumpida se
repite desde el último guardado completo.

Si una sesión murió y dejó `EN_USO.lock`, confirmar que ya no esté activa antes
de eliminar exclusivamente ese archivo. No ejecutar dos kernels sobre una misma
corrida. La reanudación CUDA requiere una cantidad compatible de GPUs.

Para cambiar código experimental o parámetros después del cierre, usar otra
`ETIQUETA_ESTUDIO`. Los botones, rutas, comentarios y salidas no alteran la huella
semántica del código de funciones y modelos.

## Resultados y alcance

Los artefactos de las etapas nuevas se agrupan en `etapas_3_5/`, o en la ruta
persistente configurada: entrenamientos, comparación y robustez. Los diagnósticos
locales se guardan en `resultados_etapas_1_2/etapas_3_5/diagnosticos/`.

La máquina local tiene PyTorch para CPU. Las pruebas CUDA omitidas siguen
pendientes; no acreditan entrenamiento ni precisión mixta en GPU. El estado
actual se registra en `resultados_etapas_1_2/etapas_3_5/estado_implementacion.json`.
La tarjeta detectada es una AMD Radeon RX 6600; este entorno no dispone de CUDA.

## Revisión de particiones y cachés

La división vigente se guarda en
`resultados_etapas_1_2/revision_particiones_v2/particiones/`: **2.051 imágenes
en train, 439 en validación y 440 en test**, con **26, 24 y 23 clases presentes**,
respectivamente. Se reasignaron 14 grupos completos desde la división inicial.
Con tres o más grupos por clase se exige presencia en los tres conjuntos; con
dos se priorizan train y validación; con uno, train. El manifiesto registra los
movimientos y las limitaciones de cobertura. Esta política no asegura soportes
suficientes para estimaciones estables en clases escasas.

Normalización, anclas y pilotos de la nueva división están dentro de esa misma
carpeta de revisión. La identidad incluye dataset, particiones, IDs de train,
tamaño y preprocesamiento. Los pilotos incluyen además la normalización en su
identidad, guardada tanto en el informe como en el checkpoint. Las cachés
anteriores sin esos campos se recalculan.

Los resultados de la división inicial permanecen en `particiones/`, `pilotos/`
y `memorizacion_cnn/` bajo `resultados_etapas_1_2/`. No se usan para describir
la división vigente. El notebook anterior a esta intervención está en
`Material/Tarea_2_antes_reequilibrio.ipynb`. El estudio principal nuevo se llama
`principal_particiones_v2_s42`.

La calibración solo reutiliza mediciones cuya identidad coincide en datos,
configuraciones, código y hardware. Los pesos empleados para medir recursos no
se usan para inicializar el entrenamiento principal. Los análisis junto a
particiones, estadísticas, anclas y pilotos se calculan desde sus informes
actuales al ejecutar el notebook.

El notebook conserva los datos auditados y particiones comunes. Las métricas de
pilotos, entrenamientos principales y pruebas sintéticas se distinguen de forma
explícita. Los resultados de test no se usan para reajustar los modelos.

Hay un respaldo anterior a las nuevas etapas en
`Material/Tarea_2_antes_etapas_3_5.ipynb`.
