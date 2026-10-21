# Decisiones implementadas y límites de la evidencia

## Datos

Se mantiene el catálogo completo del archivo real. El texto descriptivo del
repositorio no reemplaza la inspección de las anotaciones. Prefijos como
`0: ballan` se limpian solo en los nombres, sin cambiar el significado de IDs.
IDs no contiguos, clases fuera del catálogo o cajas inválidas bloquean el flujo.

La separación por grupos tiene prioridad sobre conseguir exactamente 70/15/15.
Se agrupan duplicados de píxeles y candidatos de similitud dHash. Este último
es un filtro conservador: puede sobreagrupar y no sustituye conocer el video de
origen. Los pares y advertencias deben conservarse al presentar resultados.

Las estadísticas se calculan con los píxeles de entrenamiento que verá la CNN,
incluyendo el relleno del letterbox. Se divide por desviación estándar, no por
varianza. El fingerprint vincula catálogo, contenido, anotaciones y grupos;
evita combinar una auditoría con las particiones de otra versión.

La revisión v2 mantiene el algoritmo inicial como punto de partida reproducible
y cubre primero las clases con menos grupos. Luego aproxima los tamaños a
70/15/15 mediante movimientos que conservan la cobertura. Los empates se
resuelven por identificador de grupo. La nueva división se exporta en una
carpeta distinta para conservar las exportaciones anteriores y evitar archivos
YOLO residuales en conjuntos equivocados. Si otra división ya ocupa la carpeta
de salida, se exige usar una nueva.

La huella del dataset no identifica por sí sola el train: ahora se agregan una
huella de particiones y una identidad con dataset, IDs de train, tamaño y versión
del letterbox. Normalización, anclas y pilotos comprueban esa procedencia antes
de reutilizar resultados. Los checkpoints principales incorporan también código
y configuración. Las pruebas cambian el train manteniendo la misma auditoría
y verifican que las estadísticas y resultados anteriores sean rechazados.

## CNN propia y componentes reutilizados

El extractor propio utiliza convoluciones 3x3 y reducción progresiva con stride
2. Sus cuatro salidas mantienen resoluciones distintas y evitan que un único
vector global tenga que representar todos los objetos. No se usa `Flatten` ni
una salida de clasificación por imagen.

BatchNorm y ReLU son una elección inicial para facilitar la optimización. No
se afirma que sean superiores en este dataset sin medirlo. Las default boxes
se diagnostican contra TRAIN antes de considerar ajustes. La cobertura medida
no es recall del detector: solo informa si las cajas previas pueden representar
razonablemente las anotaciones antes de aprender desplazamientos.

Se reutilizan SSDHead, asignación, codificación, pérdidas y posprocesamiento de
Torchvision. Todo el modelo CNN+SSD parte de valores aleatorios. Esto debe
declararse como una arquitectura con extractor propio y componentes SSD, no
como una implementación original completa de SSD.

En la implementación estándar de SSD, un batch completamente vacío puede
tener pérdida cero por la minería de negativos. La prueba comprueba estabilidad
y ausencia de NaN, no demuestra aprendizaje de fondo en ese caso.

## Transferencia

Los dos YOLO usan la misma arquitectura YOLO11n para que la diferencia de
inicialización sea interpretable. En el modelo COCO se transfieren únicamente
tensores compatibles con la arquitectura adaptada al catálogo. Se registra
cuántos se transfieren y cuáles se reinicializan; cambiar el número de clases
puede afectar más tensores que la última salida.

La primera fase congela el backbone y también sus estadísticas BatchNorm. El
fine-tuning habilita el backbone de YOLO y solo layer4 del cuerpo ResNet en
Faster R-CNN. En Faster R-CNN se conserva FrozenBatchNorm, como en el checkpoint
COCO. Las pruebas comprueban una actualización real de parámetros recién
descongelados, no solo el valor de requires_grad.

## Métricas y pruebas

La interfaz de todos los modelos entrega cajas originales, clases desde cero y
scores. SSD/Faster R-CNN usan fondo únicamente dentro de sus adaptadores. El
evaluador COCO no vuelve a aplicar NMS; los adaptadores ya lo hicieron. Esto
permite probar también predicciones duplicadas y comprobar que generan falsos
positivos en P/R/F1.

Se informa AP no definida en categorías sin GT y se conserva el comportamiento
de COCOeval al promediar. No se interpreta una clase ausente como un desempeño
perfecto ni se fuerza su AP a cero. El umbral de F1 no debe sustituir el umbral
bajo usado para construir las curvas AP.

La prueba sintética de memorización verifica la capacidad de aprender un
problema controlado. Los pilotos de ocho imágenes reales son diagnósticos sobre
TRAIN: no estiman generalización ni permiten comparar definitivamente modelos.
Las métricas de VALIDACIÓN y PRUEBA se reservan para la etapa experimental.

Un informe `cpu_only` no equivale a GPU comprobada. El notebook portable exige
un paso backward y una operación NMS en CUDA y después repite el smoke de los
cuatro detectores. El diagnóstico local y el de nube se guardan por separado.

La calibración CUDA recorre una época completa por fase y detector, con AMP y
la misma acumulación que el entrenamiento. En los preentrenados verifica tanto
la fase congelada como fine-tuning, incluyendo un cambio real en un parámetro
antes congelado. Se registra memoria máxima asignada y reservada, y se prueba
también la inferencia de validación. El batch común solo se acepta si todas las
fases caben; un OOM obliga a repetirlas con batch 1. Las pruebas CPU del protocolo
y el OOM simulado verifican el flujo, pero no acreditan memoria disponible en GPU.

## Referencias técnicas

- https://docs.pytorch.org/vision/stable/_modules/torchvision/models/detection/ssd.html
- https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
- https://docs.ultralytics.com/models/yolo11/
- https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py
- https://zenodo.org/records/17950781
