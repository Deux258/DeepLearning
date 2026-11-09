# Diagnóstico de la Tarea 2 y reorganización

Revisión del 9 de noviembre de 2026. Se examinó el notebook adjunto desde el Escritorio,
los resultados persistentes del repositorio, el enunciado y la rúbrica. Se excluye la evaluación
del uso de IA, como solicitó el usuario. No se asigna una nota: hay componentes implementados
cuya ejecución y discusión todavía deben completarse.

## 1. Qué resultados son realmente evaluables

El notebook adjunto tiene 98 celdas y unas 12.600 líneas. La ejecución guardada es de Colab,
pero usa `torch 2.14.1+cpu`: CUDA no estaba disponible. Se interrumpe en
`guardar_justificaciones` con `ValueError: Falta el bloque de justificación del notebook`.
El texto existe; falta la etiqueta `tarea2_evidencia=justificaciones` en los metadatos de la celda.
Las celdas de entrenamiento, comparación final y robustez posteriores no tienen salidas.

Los mAP que aparecen en la suite anterior al error corresponden a fixtures sintéticas.
Una prueba con mAP alto sobre esas imágenes no demuestra aprendizaje del dataset real.
La copia local del notebook ya tenía restaurada la etiqueta; la copia adjunta no conserva esa corrección.
Los nuevos notebooks exportan configuraciones y evidencias directamente y eliminan esa dependencia.

Sí existen entrenamientos locales anteriores, en `resultados_etapas_1_2/etapas_3_5/local_cpu_s42/`.
Los resúmenes duplicados en `parte1_nueva/experimentos/` y `legado/trabajo/` tienen las mismas firmas:
son las mismas corridas, no repeticiones independientes.

| Mejor checkpoint por mAP50–95 de validación | CNN propia + SSD | YOLO11n desde cero |
|---|---:|---:|
| Épocas completadas | 10 | 10 |
| Mejor época | 8 | 10 |
| mAP50–95 | 0,000359 | 0,008426 |
| AP50 | 0,000906 | 0,029257 |
| Precisión al umbral elegido en val | 0,684 % | 5,343 % |
| Recall al umbral elegido en val | 5,655 % | 11,724 % |
| F1 | 1,221 % | 7,340 % |
| Umbral | 0,20 | 0,05 |
| Entrenamiento activo, segundos | 3452,21 | 3409,95 |
| Validación, segundos | 521,17 | 395,27 |

Los AP están en escala 0–1: mAP 0,008426 equivale a **0,8426 %**, no a 84,26 %.
Estos resultados son bajos en términos de detección. No hay un mAP mínimo exigido por la rúbrica;
eso no convierte una detección deficiente en un resultado satisfactorio sin explicación.
AP50 y P/R/F1 de la tabla son los del checkpoint elegido por mAP50–95, no máximos independientes.

La rama de ajuste CNN guarda seis épocas adicionales de las veinte previstas, por lo que está incompleta.
Su mejor mAP50–95 observado es 0,001572 (quinta época adicional, AP50 0,004599).
Es una mejora pequeña sobre la base y no un entrenamiento final del nuevo protocolo.
No se encontraron checkpoints principales COCO ni comparación final en test en ese estudio.

Fuentes de estas cifras: `legado/trabajo/*/training_summary.json`, `history.json`,
`evidencias/ajustes_locales/comparacion_val.csv` y
`ajustes_locales/coseno_hflip_20ep/cnn_ssd/history.json`, dentro del estudio indicado.

## 2. Qué se puede y qué no se puede concluir

- **YOLO aprende algo y sigue mejorando:** la pérdida baja de 23,11 a 14,89; la mejor validación es la última época.
  Diez épocas pueden resultar insuficientes. Esa trayectoria no demuestra que aumentar épocas resuelva todo.
- **La CNN baja su pérdida sin detectar bien:** pasa de 8,01 a 6,09 y alcanza AP casi nula.
  En el punto operativo de la base hay 41 verdaderos positivos, 5949 falsos positivos y 684 falsos negativos.
  Revisar capacidad de memorización, cajas, matching/anclas y componentes de pérdida antes de una corrida larga.
- **No está demostrado el sobreajuste:** falta comparar AP de train y val bajo el mismo protocolo.
  Que baje la pérdida de train y la AP de val sea baja no basta para diagnosticar una causa específica.
- **Datos difíciles y desbalanceados:** se conservan 2930 imágenes, 6236 objetos y 26 clases observadas de 28 declaradas.
  Goldsinny y corkwing_male concentran aproximadamente 42 % de los objetos. Algunas clases tienen uno o pocos ejemplos;
  val/test solo contienen 19 clases observadas cada uno. El promedio no evalúa especies ausentes.
- **Objetos pequeños:** la cobertura de anclas anterior es aproximadamente 76,0 % a IoU 0,5 y 10,9 % a 0,75.
  Las anclas pueden refinarse mediante regresión; esa cobertura no es una cota del AP final ni prueba una causa única.
- **CPU limita el tiempo disponible, no explica por sí sola el AP:** una GPU facilita más experimentos,
  pero no corrige errores de etiquetas, pérdida, preprocesamiento o evaluación.
- **Comparaciones descriptivas:** diferencias de aumentos, tasas, fases y duración impiden atribuir
  una diferencia entre YOLO scratch y COCO exclusivamente al preentrenamiento. Una semilla no permite significancia estadística.

## 3. Cumplimiento técnico de los once criterios

Estado de los resultados originales, antes de ejecutar las nuevas corridas:

| Criterio | Máximo | Evidencia y pendiente |
|---|---:|---|
| P1.1: CNN propia y YOLO desde cero | 4 | Implementados y con diez épocas reales. Declarar que Torchvision aporta componentes SSD. |
| P1.2: mismas particiones para entrenar, validar y probar | 3 | IDs 2022/454/454 compartidos; falta la evaluación final de test. |
| P1.3: justificar arquitectura y entrenamiento | 5 | Hay fundamentos escritos. Contrastar duración, aumentos y decisiones con resultados reales. |
| P1.4: comparar métricas y tiempo desde cero | 8 | Hay validación y tiempos locales; completar comparación final y discusión con ejemplos. |
| P2.1: YOLO y Faster R-CNN con COCO | 2 | Construcción y carga de pesos implementadas; entrenamientos y evaluación final pendientes. |
| P2.2: congelamiento y fine-tuning | 5 | Política programada; faltan evidencias de las fases de las corridas principales. |
| P2.3: justificar hiperparámetros y estrategias | 5 | Receta explícita; justificarla y discutir lo observado tras entrenar. |
| P2.4: comparar los cuatro modelos | 8 | Pendiente por falta de resultados finales COCO y test común. |
| P3.1: al menos tres degradaciones | 4 | Se implementan iluminación, desenfoque y ruido, con dos intensidades cada uno. Ejecución final pendiente. |
| P3.2: robustez del mejor modelo con mismas métricas | 6 | Flujo preparado; falta seleccionar un ganador y evaluar las siete condiciones. |
| P3.3: analizar sensibilidad y peores condiciones | 10 | Pendiente de métricas y ejemplos de robustez reales. |

Son 60 puntos técnicos para este grupo de tres integrantes. No se puede acreditar el cumplimiento
total por tener funciones implementadas o pruebas sintéticas exitosas. Los nuevos notebooks generan
una matriz actualizada de evidencias, sin asignar nota ni aprobar automáticamente el análisis escrito.
Video de hasta 20 minutos y diapositivas son requisitos adicionales; no fueron aportados para esta revisión
y no se crean en esta intervención.

## 4. Qué cambió y cómo ejecutar

Los originales permanecen intactos. Los tres notebooks nuevos son autocontenidos y conservan las cuatro
arquitecturas, el catálogo, la política de depuración y los IDs originales. El código activo está en celdas
visibles; no está oculto en metadatos ni necesita leer `Tarea_2.ipynb`.

1. Abrir Parte 1 en Colab con GPU y ejecutar las celdas de preparación. Usar la misma carpeta y etiqueta
   en los tres notebooks. La primera preparación audita el dataset y calcula estadísticas de train.
2. Usar **Ejecutar todo** en Parte 1: `EJECUTAR_ENTRENAMIENTO=True` ya está activado. Cada modelo nuevo exige primero un diagnóstico
   de 300 pasos sobre ocho imágenes de train, sin aumentos: AP50 ≥ 0,5 y descenso de pérdida.
   Si no pasa, inspeccionar el diagnóstico; no desactivar el control para fabricar un resultado completo.
3. Se calibra batch físico 8/4/2/1 en las fases necesarias, con batch efectivo 8 y margen de VRAM.
   La acumulación no hace equivalentes las estadísticas BatchNorm de batches físicos diferentes.
4. La evaluación posterior ya está activada (`EJECUTAR_EVALUACION=True`) en Parte 1 para fijar checkpoint y umbral de validación.
5. Usar **Ejecutar todo** en Parte 2, con entrenamiento y evaluación activados, para entrenar ambos COCO. Al evaluar, se revisan los cuatro modelos con batch 1,
   se fija selección/umbrales y solo entonces se consulta test. Se exportan tablas, AP por clase y ejemplos.
6. Ejecutar Parte 3 con `EJECUTAR_ROBUSTEZ=True`. Se conserva el ganador de validación y se usan
   las mismas 454 imágenes, pesos, umbral y cajas. Si test favorece otro modelo se añade un análisis secundario separado.

La receta inicial usa AdamW (1e-3; 1e-4 en fine-tuning), weight decay 1e-4, 512 píxeles,
semilla 42 y aumentos moderados solo en train. Máximos 60 épocas scratch y 30 COCO (5 + 25).
Warmup lineal de tres épocas por fase, coseno hasta 1 % de la tasa, clipping L2 10 y AMP verificado.
Parada tras diez épocas sin mejora, habilitada desde época 20 scratch o 15 COCO.
No hay límite horario. Son elecciones iniciales que requieren validación, no valores óptimos demostrados.

Se guardan último/mejor checkpoint, hashes, generación anterior recuperable, optimizador,
planificador de tasa, escalador AMP, RNG y cursor dentro de la época. Reiniciar el notebook con los
mismos archivos retoma la corrida. No usar dos escritores para el mismo modelo/estudio.
Si cambian código, configuración, software o GPU, crear otro estudio para mantener trazabilidad.

Los datos viven en `/content/fjordfish_datos`; resultados y pesos en Drive bajo
`Tarea2_DL/resultados_tres_partes/colab_gpu_tres_partes_v1_s42/`. Test se evalúa con el mismo hardware
y software para comparar tiempos. Las mediciones de inferencia incluyen lectura/preprocesamiento/red/NMS,
usan tres pasadas de calentamiento y no incluyen el cálculo de métricas COCO.

Referencias técnicas consultadas: [SSD de Torchvision](https://pytorch.org/blog/torchvision-ssd-implementation/),
[entrenador Ultralytics 8.4.174](https://github.com/ultralytics/ultralytics/blob/v8.4.174/ultralytics/engine/trainer.py),
[tutorial oficial de detección](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
Fundamentan mecanismos de implementación; no proporcionan garantías de AP para este dataset.

## 5. Verificación y límites

`tools/tarea2/verify_notebooks.py` carga las definiciones de los tres notebooks generados y comprueba
esquema/compilación, contrato de IDs, geometría, AP COCO, clases sin soporte, separación val/test,
acumulación, calentamiento y coseno, pesos oficiales, gradientes, actualización/congelamiento real de capas,
degradaciones deterministas e integridad de checkpoints. Una prueba verifica la reanudación exacta de
una CNN y su optimizador/planificador en un proceso Python nuevo con una fixture sintética explícita.

Los reportes de verificación se guardan junto a este diagnóstico. Este equipo tiene PyTorch solo CPU:
no se certifican CUDA, AMP ni el entrenamiento completo en Colab. El notebook comprobará esas capacidades
con operaciones y gradientes reales antes de entrenar; ante una instalación CPU detendrá la corrida principal.
Las pruebas breves y diagnósticos de ocho imágenes no sustituyen los cuatro experimentos finales.

**Resultado de la verificación:** 16 pruebas aprobadas, cero errores y cero fallos sobre los archivos finales.
Se reprodujo la preparación completa: 2930 imágenes, 6236 objetos, los IDs originales y exactamente las mismas
medias/desviaciones de train. No se alteraron los originales del repositorio ni el notebook adjunto;
se contrastaron sus hashes SHA-256.

Además se ejecutó el control de aprendizaje sobre ocho imágenes reales de train, en CPU, sin aumentos:

| Diagnóstico de 300 pasos; mismas ocho imágenes al entrenar y evaluar | CNN propia + SSD | YOLO11n desde cero |
|---|---:|---:|
| AP50 inicial | 0,0000 | 0,0000 |
| AP50 final | 0,9280 | 0,6840 |
| Pérdida media de los primeros diez pasos | 16,2667 | 13,8454 |
| Pérdida media de los últimos diez pasos | 0,0433 | 1,4085 |
| Control de memorización superado | Sí | Sí |

Ambos modelos pueden aprender este subconjunto; por tanto, los bajos resultados anteriores no demuestran
una incapacidad general de las arquitecturas. Sigue pendiente determinar cuánto contribuyen duración,
optimización, desbalance y dificultad del dataset a la generalización. **Estos AP no son validación ni test**
y no deben añadirse a la comparación final como si fueran resultados nuevos del conjunto completo.

Reportes: `tarea2_verificacion.json`, `tarea2_preparacion_verificada.json` y
`tarea2_diagnosticos_train_cpu.json`. Cada uno conserva las huellas de su versión. Los diagnósticos reales
se ejecutaron antes del último ajuste de publicación de checkpoints y de la matriz de evidencias;
las funciones de aprendizaje y preparación utilizadas no cambiaron después de esos diagnósticos.

No se han ejecutado las corridas completas, ni se fabricaron métricas finales de transferencia o robustez.
No se promete una mejora de AP hasta observar las nuevas curvas y evaluaciones.
