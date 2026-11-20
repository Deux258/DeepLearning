# Revisión técnica de la parte 1 y ejecución de la parte 2

Trabajo realizado sobre `Deux258/DeepLearning`, rama `Tarea3_entrenamiento`, commit de partida
`72a5b3e`. La parte 2 corresponde al segundo punto del enunciado: metodología y evaluación,
no al entrenamiento con PPO. Se consultaron el enunciado y la rúbrica proporcionados.

## Archivos principales

- [Notebook de evaluación](tarea3/parte2_evaluacion.ipynb).
- [Informe de tablas y respuestas](tarea3/salidas/recuperado_notebook/informe.html).
- [Pauta humana para completar](tarea3/salidas/recuperado_notebook/revision_humana.csv).
- [Guía de ejecución y criterios](tarea3/PARTE2.md).
- [Métricas con cobertura](tarea3/salidas/recuperado_notebook/metricas.csv).

## Parte 1: comprobaciones realizadas

| Componente | Evidencia o estado |
|---|---|
| Casos finales | 60: 30 dañinos y 30 legítimos; piloto y few-shot separados. |
| Particiones HH-RLHF | SFT: 512; recompensa train: 1024; recompensa validación: 128; PPO: 128. |
| Fuentes | Mitad helpful-base, mitad harmless-base en cada partición. |
| Separación | Cero solapamientos exactos normalizados entre particiones y casos de evaluación. No se verificó similitud semántica. |
| Reproducibilidad | Se reconstruyeron las particiones dos veces; mismos archivos byte por byte. |
| SFT | Enmascaramiento del prompt y padding comprobado; un paso con una Phi-3 diminuta aleatoria actualizó LoRA. Adaptador guardado y recargado; generación técnica comprobada. |
| Compatibilidad | Se usaron `gradient_checkpointing` y `torch_dtype`, compatibles con el entorno Transformers 4.48.3 probado. |
| Phi-3-mini completo | No ejecutado en este equipo: no hay GPU CUDA/ROCm. No se descargaron sus pesos completos. |
| Recompensa y PPO | Esta rama solo prepara las particiones; no contiene la implementación ni las salidas de esos entrenamientos. |

La prueba diminuta verifica el funcionamiento del entrenamiento, no la calidad ni las métricas
del modelo real. Tampoco permite certificar la compatibilidad del entrenamiento cuantizado
en otra GPU. No venían incluidos los adaptadores SFT que menciona el README.

Detalle reproducible: `tarea3/verificar_parte1.py`, `tarea3/salidas/verificacion_parte1.log`
y `tarea3/salidas/verificacion_tecnica/comprobaciones.json`.

## Parte 2: evaluación ejecutada

Se recuperaron las 180 respuestas que ya estaban publicadas como HTML en el notebook original.
Se preservó una copia de ese notebook en `tarea3/salidas/origen/evaluacion_original.ipynb`.
Los textos se validaron contra los 60 casos y se registró el hash del origen en `procedencia.json`.
No se regeneraron respuestas Phi-3 ni se verificó qué adaptador produjo el texto SFT original.

Se descargó y ejecutó en CPU `unitary/toxic-bert` con la revisión fija del proyecto,
independiente del modelo de recompensa. Se puntuó solo el texto de las respuestas.

| Configuración | Respuestas puntuadas | Toxicidad ≥ 0,5 | Vacías | Alcanzaron 256 tokens | Truncadas por Toxic-BERT |
|---|---:|---:|---:|---:|---:|
| Phi-3 zero-shot | 60 | 0 | 0 | 20 | 0 |
| Phi-3 few-shot | 60 | 0 | 0 | 2 | 0 |
| Phi-3 SFT | 60 | 0 | 0 | 8 | 0 |

Esta tabla registra salidas del programa, sin concluir sobre seguridad o utilidad.
Las métricas de seguridad, utilidad y rechazo innecesario están **pendientes**, porque no hay
etiquetas humanas. La toxicidad no reemplaza esas etiquetas. Los CSV originales de los puntajes
redondeados se conservaron aparte de los nuevos puntajes completos.

La implementación genera una planilla ciega, conserva las anotaciones existentes, rechaza
etiquetas inválidas, exige una justificación y exporta tasas con denominadores, cobertura e
intervalos de Wilson. El informe permite inspeccionar los textos completos por caso.
No se sobrescriben anotaciones de una corrida distinta.

## Comprobaciones automáticas

Pasaron nueve pruebas de integridad sobre casos, duplicados, límites de tokens, protección de
anotaciones, recuperación del notebook y métricas con cobertura incompleta. Se guardó el log
en `tarea3/salidas/pruebas_integridad.log`. Las celdas de los tres notebooks fueron compiladas
y sus estructuras validadas. El entorno usado está en `tarea3/salidas/entorno_cpu.txt`.
El notebook `parte2_evaluacion.ipynb` también se ejecutó de principio a fin: cuatro celdas de
código, sin errores, con las salidas guardadas. Reutilizó los puntajes de la ejecución real
de Toxic-BERT y comprobó la exportación de métricas e informe.

## Pendiente para una entrega completa

1. Aportar y ejecutar el modelo de recompensa y PPO requeridos por la parte 1, y generar sus
   respuestas con los mismos casos. La parte 2 acepta su CSV como `phi3_ppo`.
2. Para los tres integrantes indicados en el README, aportar Mistral 7B Instruct mediante
   Ollama en zero-shot y few-shot. Evaluarlo con los mismos casos y clasificador.
3. El grupo debe completar las etiquetas humanas, justificar y revisar casos cualitativos,
   resolver desacuerdos y volver a calcular las métricas.
4. El grupo debe realizar su análisis de resultados, conclusiones, diapositivas y video.

Se dejó documentada la asistencia de Codex/ChatGPT. La rúbrica exige declarar el uso real de IA
y prohíbe su uso en el diseño y análisis. Este informe registra implementación y comprobaciones;
no contiene etiquetas humanas ni conclusiones elaboradas para la entrega.
