# Tarea 2 — notebook preparado para ejecutar localmente

Archivo principal: **Tarea_2.ipynb**, con **94 celdas en total, 76 de código**.
Esta preparación no ejecutó el notebook ni inició entrenamientos. El proceso anterior
quedó detenido, con checkpoints en disco.

## Ejecutar cuando corresponda

1. En local: abrir el notebook en VS Code o Jupyter, desde esta carpeta y usar el kernel
   `.venv/Scripts/python.exe` del proyecto.
2. En Colab: abrir la notebook desde el repositorio o desde Drive. La primera celda automática
   detecta el entorno, monta Drive si existe y, si no encuentra el proyecto, clona el repositorio
   desde GitHub en `/content/DeepLearning`.
3. En la primera celda de configuración elegir `ACCION_LOCAL`:
   - `"revisar"` (predeterminado): revisar datos y resultados; no entrena ni descarga pesos COCO.
   - `"entrenar_base"`: iniciar o retomar las diez épocas base de CNN y YOLO.
   - `"ajustar"`: validar las bases y retomar las ramas de ajuste que correspondan.
4. Ejecutar de arriba hacia abajo. Las funciones auxiliares están agrupadas y plegadas;
   cada celda imprime qué hace, su número de avance y el tiempo utilizado.

`RUTA_PROYECTO=None` usa la carpeta del kernel. En Colab, la detección automática reemplaza la
edición manual de la ruta en la mayoría de los casos. Se mantienen ocho hilos CPU. La instalación
automática de dependencias queda habilitada cuando faltan paquetes.

## Avance conservado al preparar esta versión

- CNN y YOLO base: diez épocas terminadas por modelo.
- Ajuste CNN: seis épocas adicionales completas; retoma desde la adicional siete.
- Ajuste YOLO: todavía sin iniciar.
- Test sigue reservado. No hay un entrenamiento ni un publicador externo activo.

Estudio: `resultados_etapas_1_2/etapas_3_5/local_cpu_s42/`.
Las bases están en `parte1_nueva/experimentos/` y las variantes en
`ajustes_locales/coseno_hflip_20ep/`. Se guardan `best.pt`, `last.pt`, historial,
configuración, fuentes, procedencia, tablas y gráficos. Nunca ejecutar dos sesiones
sobre la misma carpeta. Se recupera la última época completa.

## Qué se conserva

Los IDs compartidos son 2022/454/454, huella `fc946375…`. La biblioteca científica,
los modelos, optimizadores, evaluador y fuentes originales no cambiaron al compactar.
La huella del protocolo continúa siendo `5b55ab5a…`, compatible con los checkpoints.
Las salidas guardadas están rotuladas como evidencia de ejecuciones anteriores.

El ajuste usa AdamW 3e-4→3e-5 por coseno, clipping L2 a 10 y reflexión horizontal,
con hasta veinte épocas adicionales. Las fuentes oficiales y justificaciones están
en la sección 8.1. Las comprobaciones previas del motor están registradas; durante
esta preparación solo se revisaron sintaxis, estructura, controles e identidad.

Los controles COCO, GPU, comparación, robustez y empaquetado siguen disponibles y
separados, pero desactivados. CPU no acredita CUDA. Diapositivas y video quedan fuera.

Respaldo: `Tarea_2_respaldo_antes_preparar_local_20261104_105841.ipynb`.
