# Tarea 2 — notebook integrado

El archivo ejecutable y de entrega es **`Tarea_2.ipynb`**. Contiene el flujo previo,
el aporte original de Ignacio Pastén y los dos entrenamientos COCO. Los datos y
checkpoints se guardan aparte; no depende de `prueba.py`, `docs/` ni de los otros notebooks.
La versión de fase 1 tiene 106 celdas en total, con la biblioteca técnica plegada
y las acciones de cada sesión separadas en el cuerpo del notebook.

## Qué se conserva

- División original **2.022/454/454**, huella `fc946375…`, sobre 2.930 imágenes
  y 6.236 objetos. El reequilibrio posterior se conserva como estudio histórico.
- CNN y YOLO desde cero de Ignacio: **diez épocas, batch 2 y entrenador original**.
  No se les agregan aumentos ni se cambia la configuración al reanudar.
- Fuentes históricas completas en `metadata.tarea2_legado`, con hashes SHA-256.
  Se reconstruyen temporalmente; no requieren el notebook de Ignacio para ejecutar.
- Auditoría, normalización y anclas en `resultados_etapas_1_2/integracion_nacho_s42/`.
  El manifiesto local sirve para inspección; entrenar exige verificar la exportación original.

El estudio nuevo se llama **`final_correcciones_s42`**. Las carpetas anteriores y los
notebooks de respaldo no se sobrescriben. El respaldo de esta intervención es
`Tarea_2_respaldo_antes_fase1_20261103.ipynb`.

## Fase 1: preparar y cerrar el protocolo

Abrir el notebook desde esta carpeta o definir `RUTA_PROYECTO`. Ejecutar desde el
principio para cargar las funciones y comprobar datos, entorno y pruebas.
Las acciones largas parten desactivadas.

1. **Exportar primero los archivos pequeños de Ignacio** desde su carpeta RUN_ID:

   ```text
   preparacion/curacion/audit.json
   preparacion/particiones/manifest.json
   preparacion/particiones/train_statistics.json
   codigo/                                  # cuatro fuentes originales
   experimentos/cnn_ssd/experiment.json
   experimentos/yolo_scratch/experiment.json
   ```

   Definir `RUTA_APORTE_IGNACIO`, `MANIFIESTO_COMPARTIDO` y `RUTA_PERSISTENTE`.
   Activar `IMPORTAR_CONTRATO_DATOS`. Se verifican imágenes, IDs, grupos, etiquetas,
   normalización, configuraciones y fuentes sin exigir los checkpoints grandes.

2. **Usar CUDA y software compatible con el aporte original.** Ejecutar las pruebas
   iniciales y avanzadas, incluidas recuperación CUDA y AMP en frozen/finetune.
   CPU permite diagnosticar, pero no acredita la preparación GPU ni cierra el protocolo.

3. **Activar `EJECUTAR_PERFIL_GPU`** en la sección 9. Se miden cuatro épocas completas:
   dos modelos COCO × dos fases. Incluye train, validación, métricas y guardado real.
   Fija batch físico común 2 o 1 y acumulación hasta batch efectivo 8, sin alterar a Ignacio.

4. **Revisar la propuesta antes de cerrar.** Los valores iniciales nativos son:
   YOLO COCO máximo 20 épocas; Faster R-CNN máximo 15; ambos cinco congeladas,
   seis horas, paciencia cinco desde la época diez. Se reserva 10 % de margen.
   Las épocas definitivas se calculan con los segundos medidos por fase. Si no
   alcanzan diez épocas, el cierre se bloquea para revisar el presupuesto.
   Cambiar solo épocas, horas o criterios de parada reutiliza el perfil;
   cambiar código, software, datos, aumentos o recursos obliga a medir nuevamente.

5. **Activar `CERRAR_PROTOCOLO`** tras revisar la tabla. Guarda las configuraciones
   definitivas y sus identidades. Desde ese momento conservar código y entorno.
   Los textos en Markdown y los interruptores pueden cambiar.

6. **Crear workers separados**, mediante `CREAR_PLANTILLA_WORKER` y nombres distintos
   en `NOMBRE_WORKER`. El coordinador conserva el estudio central. Cada integrante
   recibe el mismo notebook y plantilla cerrada. En su sesión configura `ROL_SESION = "worker"`
   y el `NOMBRE_WORKER` asignado: sus informes y registros también quedan separados del coordinador.

## Fase 2: entrenar y consolidar

- Diego: `MODELOS_A_EJECUTAR = ["yolo_coco"]`.
- Tercer integrante: `MODELOS_A_EJECUTAR = ["fasterrcnn_coco"]`.
- Ignacio: importar los checkpoints originales con `IMPORTAR_IGNACIO` y continuar
  lo pendiente con `REANUDAR_IGNACIO`. Recupera épocas completas; una época
  histórica interrumpida se repite, conservando su formato, software y hardware.

Los entrenamientos nativos escriben en `RAIZ_WORKER`, nunca simultáneamente en la
misma carpeta. `LIMITE_SESION_HORAS = 2` guarda y pausa; **se debe ejecutar nuevamente
la celda para retomar**. Los tiempos por época separan train/validación/guardado
y no suman la desconexión. Los tiempos históricos mantienen su definición original.

El coordinador indica `RUTAS_WORKERS_A_IMPORTAR` y activa `IMPORTAR_CORRIDAS_NATIVAS`.
El importador comprueba protocolo, software, configuración, datos, código, informes
y checkpoints antes de publicar una copia. Una copia idéntica se reutiliza;
un destino diferente se rechaza sin sobrescribir. Consolidar preferentemente las
corridas terminadas: una importación pausada no se actualiza encima de la anterior.
La prueba de transferencia previa usa fixtures temporales, sin ocupar las carpetas finales.

## Comparación y entrega técnica

1. Reunir los cuatro modelos completos. Los COCO deben acreditar épocas completas
   de frozen y finetune antes del test.
2. Activar `EJECUTAR_COMPARACION`: reproducir validación histórica de Ignacio con
   batch 2; seleccionar umbrales y ganador con validación común; evaluar el mismo
   test y medir latencia/memoria en una misma GPU.
3. Activar `EJECUTAR_ROBUSTEZ`: limpio y seis degradaciones, con los mismos pesos,
   umbral e IDs. Incluye métricas por clase y tamaño, caídas y ejemplos pareados.
4. Ejecutar `EJECUTAR_REPRODUCCION` en un proceso nuevo. Revisar los análisis con
   cifras reales y la matriz de evidencias; no tratar borradores automáticos como conclusiones.
5. Completar el tercer integrante y la declaración de herramientas antes de
   `EJECUTAR_EMPAQUETADO`. El paquete incorpora manifiesto, listas de IDs, tablas,
   gráficos, código, checkpoints y comprobación de integridad.

Solo train nativo recibe reflexión p=0,5 y fotometría p=0,5, con brillo, contraste
y saturación entre 0,85 y 1,15. Val/test son deterministas. El oscurecimiento de robustez
es más fuerte; considerar la exposición al brillo al interpretar los resultados.
Ignacio mantiene otro régimen de entrenamiento: la comparación con COCO **no aísla
causalmente el preentrenamiento**. Una semilla no acredita superioridad estadística.

AP por tamaño sigue los rangos COCO en píxeles originales: small [0,32²], medium
[32²,96²], large [96²,1e10]. Los bordes son inclusivos y se solapan; no sumar los
soportes como grupos disjuntos. Sin GT, AP es no definida. No equivale a área relativa
ni al tamaño de la caja después de redimensionar a 512.

Las pruebas sintéticas verifican implementación, no rendimiento real en FjordFish.
Este equipo Windows no tiene CUDA: perfil, pruebas GPU y entrenamientos siguen
pendientes hasta ejecutarlos en un entorno compatible. **Diapositivas y video
permanecen fuera de esta intervención.**
