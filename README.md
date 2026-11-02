# Tarea 2 — notebook y cierre técnico

La versión oficial está en esta carpeta. **Todo el código del flujo permanece en
`Tarea_2.ipynb`**: auditoría, particiones, pruebas, entrenamiento recuperable,
comparación, robustez, reproducción y empaquetado. `docs/` no es una dependencia.
No se prepararon diapositivas ni video en esta intervención.

## Ejecución local

Abrir el notebook desde esta carpeta y seleccionar un Python con las dependencias
indicadas en su configuración. `RUTA_PROYECTO` permite indicar otra raíz si el
kernel parte desde una carpeta distinta. El notebook no depende de una `.venv`
específica ni de módulos Python del proyecto.

Los datos están en `data/extracted/`. Los pesos y resultados ya están consolidados
en `resultados_etapas_1_2/`. Las auditorías se reubican sin cambiar su identidad;
las referencias COCO/YOLO se regeneran desde los mismos IDs.

Ejecutar desde el inicio para comprobar CPU, auditoría y pilotos. Las pruebas
sintéticas son diagnósticos: no se admiten como evaluación final de FjordFish.
El equipo local tiene AMD Radeon RX 6600 y este PyTorch no dispone de CUDA.

## Sesiones con GPU gratuita

1. Disponer de datos, notebook y pesos, y de una carpeta persistente ya montada.
2. Ejecutar desde el inicio y comprobar integración, AMP y pruebas CUDA.
3. En la configuración de etapas finales, definir `RUTA_PERSISTENTE` y conservar
   `ETIQUETA_ESTUDIO = "principal_particiones_v2_s42"`.
4. Elegir uno o varios `MODELOS_A_EJECUTAR`. Los valores posibles son `cnn_ssd`,
   `yolo_scratch`, `yolo_coco` y `fasterrcnn_coco`.
5. Activar `EJECUTAR_ENTRENAMIENTOS`. `LIMITE_SESION_HORAS = 2.0` pausa el bucle
   de entrenamiento con checkpoint; calibración, pruebas y evaluación se suman
   por separado. No exige una sesión continua de seis horas.
6. Retomar con la misma ruta, etiqueta, código y configuraciones. La pausa por
   sesión se distingue del presupuesto acumulado de seis horas por modelo.

La primera preparación calibra los cuatro modelos, incluyendo fase congelada y
fine-tuning. Fija el batch común y el software en `protocolo_estudio.json`. Una GPU
de otra familia debe admitir ese batch; no se cambia automáticamente al reanudar.
Las sesiones registran versiones, hardware y fecha. No ejecutar dos kernels sobre
la misma corrida. Si quedó `EN_USO.lock`, comprobar que la sesión haya muerto antes
de retirar solo ese archivo.

## Evaluación y cierre

- `EJECUTAR_COMPARACION`: exige los cuatro modelos, cierra checkpoint/umbral con
  validación y evalúa test. Genera tablas globales y por clase, errores, ocho
  imágenes comunes, curvas, mapas de AP y calidad/costo en una sesión de hardware.
- `EJECUTAR_ROBUSTEZ`: evalúa el ganador de validación con limpio y seis
  degradaciones, sin reajustes. Si otro modelo lidera test, genera su análisis
  secundario en una carpeta distinta, conservando el ganador principal.
- `EJECUTAR_REPRODUCCION`: inicia otro proceso desde el notebook guardado y los
  checkpoints; recalcula val/test y contrasta mAP, AP50, P/R/F1 y AP por clase.
- La matriz `evidencias/estado_rubrica.json` indica qué evidencia falta. No asigna
  nota ni certifica automáticamente la calidad de la interpretación.
- `EJECUTAR_EMPAQUETADO`: completar `INTEGRANTES` y `DECLARACION_IA`. Solo habilita
  el ZIP técnico cuando hay cuatro resultados reales, evaluación, robustez,
  pruebas CUDA y reproducción. Incluye mejores checkpoints e integridad SHA-256.
  El dataset se reconstruye desde su fuente y checksum. Diapositivas y video
  quedan expresamente pendientes, fuera de este paquete técnico.

## Datos y respaldos

División vigente: **2.051 / 439 / 440 imágenes**, con **26 / 24 / 23 clases
presentes**. Se mantienen 2.930 imágenes, 6.236 objetos y 14 grupos reasignados.
Normalización, anclas y pilotos vigentes están bajo `revision_particiones_v2/`.
La identidad contempla dataset, particiones, train, tamaño y preprocesamiento.

Los originales de la ubicación anterior permanecen intactos. La migración verificó
por SHA-256 las 2.930 imágenes. El respaldo del notebook previo al cierre está en
`Material/Tarea_2_antes_cierre_tecnico.ipynb`. No son dependencias de ejecución.

Para cambiar código experimental después de entrenar o cerrar decisiones, usar
otra etiqueta de estudio. Las rutas, interruptores, duración de sesión, integrantes
y textos de justificación no modifican la huella semántica de las funciones.
