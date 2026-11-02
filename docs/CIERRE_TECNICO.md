# Cierre técnico implementado

La ejecución guardada tiene **40 pruebas aprobadas y 2
omitidas por falta de CUDA**. Incluye una prueba integrada con los cuatro modelos
sobre datos sintéticos y reproducción de sus evaluaciones en un proceso nuevo.
Es evidencia de funcionamiento del flujo, no rendimiento final en FjordFish.

## Interfaces y artefactos

- Sesiones: `MODELOS_A_EJECUTAR` y `LIMITE_SESION_HORAS`; pausa recuperable sin
  modificar el presupuesto o configuración científica del modelo.
- Preparación: `protocolo_estudio.json`, calibración de todas las fases y
  comprobación del batch fijado cuando cambia la GPU.
- Comparación: tabla global, `por_clase.csv`, `errores.json` por modelo, muestra
  común de ocho imágenes, mapas de AP y gráficos de calidad/costo.
- Robustez: ganador de validación y, si corresponde, líder de test como análisis
  secundario. Ambos conservan checkpoints, umbrales y condiciones cerrados.
- Reproducción: proceso nuevo, notebook guardado, evaluación de val/test,
  tolerancia 1e-5 sobre métricas y verificación de soporte por clase.
- Rúbrica: 11 criterios y 60 puntos técnicos posibles; disponibilidad de evidencia
  separada de calidad y calificación. Sin resultados reales el cierre queda pendiente.
- Entrega: ZIP técnico, mejores checkpoints, datos reconstruibles, versiones,
  integrantes, declaración de herramientas y manifiesto de integridad.

## Orden operativo

| Fecha objetivo | Acción | Criterio de término |
|---|---|---|
| 20–21 octubre | Continuidad y comprobación CUDA | Rutas válidas, mismos datos y pruebas CUDA completas |
| 21–27 octubre | Cuatro entrenamientos | Checkpoints recuperables, curvas y causas de término |
| 28–30 octubre | Reserva y comparación | Cierre con val, métricas test y costo común |
| 31 octubre–1 noviembre | Robustez | Siete condiciones y análisis sustentado |
| 2–4 noviembre | Reproducción y revisión técnica | Matriz completa y prueba en proceso nuevo |
| 5 noviembre | Paquete técnico revisado | Integridad y contenidos verificados |

Esta implementación omite preparar diapositivas y video por solicitud del usuario.
Para la entrega académica completa seguirán siendo necesarios cuando se retome esa etapa.

## Revisión del análisis

El notebook incorpora una tabla de decisiones con fundamentos, alternativas,
evidencias y límites. Los textos numéricos se generan desde resultados calculados.
El equipo debe revisar ejemplos y conclusiones: las reglas de errores no prueban
causalidad y una sola semilla no permite afirmar superioridad estadística.

Persisten limitaciones de clases raras, especies sin ejemplos, agrupación dHash
sin metadatos de video y ausencia de imágenes negativas completas. Comparar
siempre AP junto con soporte; no tratar AP no definida como cero o perfección.
