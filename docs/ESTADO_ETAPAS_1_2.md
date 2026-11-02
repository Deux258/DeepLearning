# Estado comprobado de las etapas 1 y 2 — revisión de particiones v2

Fuente vigente: `Tarea_2.ipynb` y
`resultados_etapas_1_2/revision_particiones_v2/estado.json`.

- Datos depurados: **2.930 imágenes y 6.236 objetos**; 28 clases declaradas y 26 observadas.
- Particiones: **2.051 / 439 / 440 imágenes**, con **26 / 24 / 23 clases presentes**.
- Reequilibrio: **14 grupos completos**, sin solapamiento de imágenes ni grupos.
- Catálogo, contenido y anotaciones conservan su huella; la huella nueva de particiones es
  `a21a6236f26bda12d9634434afc596cf7a817591ebb1ad0d270f90185bd29c6a`.
- Normalización renovada con train: media `[0.321592, 0.502617, 0.379079]`,
  desviación `[0.194104, 0.185894, 0.186589]`.
- Anclas sobre **4852 objetos**: cobertura IoU 0,5 **75.74%**,
  e IoU 0,75 **10.86%**. No equivale al recall del detector.
- Integración de los cuatro detectores comprobada en CPU; pilotos reales renovados.
- Memorización CNN, cien pasos sobre ocho imágenes: **AP50 0.7733**,
  **mAP50–95 0.4696**. No mide generalización.
- Pruebas: **40 aprobadas y 2 CUDA omitidas**.
- GPU local: AMD Radeon RX 6600; este entorno no dispone de CUDA.
- Calibración por fase implementada; actualización de pesos antes congelados comprobada
  en CPU y cambio a batch 1 ante OOM validado con un caso simulado.
- Siguen pendientes calibración CUDA real, cuatro entrenamientos principales,
  comparación, robustez y materiales finales.

Los resultados anteriores se conservan en sus carpetas originales bajo
`resultados_etapas_1_2/`. La revisión activa usa `revision_particiones_v2/`;
el estudio principal se llama `principal_particiones_v2_s42`.
La carpeta `docs` no es una dependencia del notebook.

El cierre técnico y sus controles están implementados; ver `docs/CIERRE_TECNICO.md`.
