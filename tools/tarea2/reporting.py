"""Celdas de figuras, robustez y evidencias insertadas en los notebooks."""

# %% Figuras y comparación
def guardar_ejemplos(detector, auditoria, ids, carpeta, umbral, condicion=None):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = {f['id']: f for f in auditoria['images']}
    for identificador in ids:
        fila = filas[identificador]
        with Image.open(Path(auditoria['root']) / fila['file_name']) as original:
            imagen = original.convert('RGB')
        if condicion is not None:
            imagen = degradar_imagen(imagen, condicion, identificador)
        prediccion = detector.predict([imagen])[0]
        dibujar_ejemplo(imagen, fila, prediccion, umbral).save(carpeta / f'{identificador}.jpg')


def graficar_historial(nombre):
    import matplotlib.pyplot as plt
    carpeta = RAIZ_ESTUDIO / 'modelos' / nombre
    historial = leer_json(carpeta / 'historial.json')
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
    epocas = [f['epoca'] for f in historial]
    for eje, clave, titulo in zip(ejes, ('perdida', 'map_50_95', 'tasa'),
                                 ('Pérdida de train', 'mAP50–95 de validación', 'Tasa de aprendizaje')):
        eje.plot(epocas, [f[clave] for f in historial], marker='.')
        eje.set(xlabel='Época', title=titulo)
        eje.grid(alpha=.25)
    fig.suptitle(nombre)
    fig.tight_layout()
    fig.savefig(carpeta / 'curvas.png', dpi=150)
    plt.show()
    plt.close(fig)


def comparar_disponibles(nombres, split='val'):
    import matplotlib.pyplot as plt
    filas = []
    gpu = None
    for nombre in nombres:
        ruta = RAIZ_ESTUDIO / 'evaluacion' / nombre / ('validacion.json' if split == 'val' else 'test.json')
        if not ruta.is_file():
            print('Pendiente:', nombre, split)
            continue
        r = leer_json(ruta)
        if split == 'test':
            actual = (r['dispositivo'], r.get('gpu'), r['software'])
            if gpu is not None and actual != gpu:
                raise ValueError('Reevaluar test en el mismo dispositivo/software para comparar tiempos')
            gpu = actual
        estado = leer_json(RAIZ_ESTUDIO / 'modelos' / nombre / 'estado.json')
        filas.append(dict(modelo=nombre, split=split, **resumen_metricas(r['metricas']),
                          epocas=estado['epoca'], mejor_epoca=estado['mejor_epoca'],
                          motivo=estado['motivo'], train_segundos=estado['segundos_train'],
                          validacion_segundos=estado['segundos_validacion'],
                          guardado_segundos=estado['segundos_guardado'],
                          inferencia_segundos=r['segundos_inferencia'],
                          milisegundos_por_imagen=1000 * r['segundos_inferencia'] / r['metricas']['images']))
    carpeta = RAIZ_ESTUDIO / 'comparacion'
    guardar_csv(carpeta / f'{split}.csv', filas)
    guardar_json(carpeta / f'{split}.json', filas)
    display(Markdown(tabla_markdown(filas)))
    if filas:
        fig, eje = plt.subplots(figsize=(8, 4))
        eje.bar([f['modelo'] for f in filas], [f['mAP50_95'] for f in filas])
        eje.set(ylabel='mAP50–95 (0–1)', title=f'Comparación en {split}')
        eje.tick_params(axis='x', rotation=15)
        fig.tight_layout()
        fig.savefig(carpeta / f'{split}.png', dpi=150)
        plt.show()
        plt.close(fig)
    return filas


# %% Robustez pareada, sin recalibración
def ejecutar_robustez(cierre, auditoria, manifiesto, estadisticas, dispositivo='cuda', nombre=None):
    nombre = nombre or cierre['ganador_validacion']
    if cierre['datos'] != huella_dataset(auditoria) or cierre['particiones'] != huella_particiones(manifiesto['partitions']):
        raise ValueError('El cierre no corresponde a estos datos')
    detector, _, _ = cargar_mejor(nombre, auditoria, manifiesto, estadisticas, dispositivo)
    digest = calcular_sha256(RAIZ_ESTUDIO / 'modelos' / nombre / 'mejor.pt')
    if digest != cierre['modelos'][nombre]['checkpoint']:
        raise ValueError('Los pesos cambiaron después de seleccionar el ganador')
    umbral = cierre['modelos'][nombre]['umbral']
    ids = manifiesto['partitions']['test']
    gt = formato_coco(auditoria, ids)
    carpeta = RAIZ_ESTUDIO / 'robustez' / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    for condicion in condiciones_robustez():
        etiqueta = nombre_condicion(condicion)
        preds, segundos = predecir_ids(detector, auditoria, ids, condicion)
        metricas = evaluar(gt, preds, umbral)
        identidad = dict(modelo=nombre, checkpoint=digest, cierre=huella(cierre),
                         condicion=condicion, umbral=umbral, split='test', ids=ids,
                         version=VERSION_MOTOR, dispositivo=dispositivo, software=entorno_actual())
        guardar_json(carpeta / f'{etiqueta}.json', dict(identidad=identidad, metricas=metricas,
                                                      segundos_inferencia=segundos))
        guardar_json(carpeta / f'{etiqueta}_predicciones.json', preds)
        guardar_csv(carpeta / f'{etiqueta}_clases.csv', metricas['per_class'])
        guardar_ejemplos(detector, auditoria, ids[:8], carpeta / 'ejemplos' / etiqueta, umbral, condicion)
        filas.append(dict(condicion=etiqueta, **resumen_metricas(metricas), segundos_inferencia=segundos))
    base = filas[0]['mAP50_95']
    for fila in filas:
        fila['caida_mAP_absoluta'] = base - fila['mAP50_95']
        fila['caida_mAP_relativa'] = (base - fila['mAP50_95']) / base if base else None
    guardar_json(carpeta / 'resumen.json', filas)
    guardar_csv(carpeta / 'resumen.csv', filas)
    display(Markdown(tabla_markdown(filas)))
    return filas


def graficar_robustez_final(nombre):
    import matplotlib.pyplot as plt
    carpeta = RAIZ_ESTUDIO / 'robustez' / nombre
    filas = leer_json(carpeta / 'resumen.json')
    fig, eje = plt.subplots(figsize=(11, 4))
    eje.bar([f['condicion'] for f in filas], [f['mAP50_95'] for f in filas])
    eje.set(ylabel='mAP50–95 (0–1)', title=f'Robustez: {nombre}; pesos y umbral fijos')
    eje.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(carpeta / 'robustez.png', dpi=150)
    plt.show()
    plt.close(fig)


# %% Matriz de los once criterios técnicos
def actualizar_rubrica():
    base = RAIZ_ESTUDIO
    def existe(ruta):
        return (base / ruta).is_file()
    def completo(n):
        p = base / 'modelos' / n / 'estado.json'
        return p.is_file() and leer_json(p).get('completo', False)
    desde_cero = all(completo(n) for n in ('cnn_ssd', 'yolo_scratch'))
    coco = all(completo(n) for n in ('yolo_coco', 'fasterrcnn_coco'))
    congelados = coco and all(existe(f'modelos/{n}/fase_{fase}.json')
                     for n in ('yolo_coco', 'fasterrcnn_coco') for fase in ('frozen', 'finetune'))
    test = all(existe(f'evaluacion/{n}/test.json') for n in KINDS)
    comparacion = test and existe('comparacion/test.csv')
    hay_robustez = False
    cierre = base / 'evaluacion/seleccion_cerrada.json'
    if cierre.is_file():
        ganador = leer_json(cierre)['ganador_validacion']
        p = base / f'robustez/{ganador}/resumen.json'
        hay_robustez = p.is_file() and len(leer_json(p)) == 7
    criterios = [
        ('P1.1', 4, 'CNN propia y YOLO desde cero', desde_cero, 'Parte 1; modelos/*/estado.json'),
        ('P1.2', 3, 'Mismas particiones en train/val/test', desde_cero and test, 'datos/manifiesto.json; evaluacion/*/test.json'),
        ('P1.3', 5, 'Justificar diseño y entrenamiento', False, 'Parte 1: fundamentos implementados; contrastar con resultados y discusión del equipo'),
        ('P1.4', 8, 'Comparar calidad y tiempo desde cero', desde_cero and comparacion, 'comparacion/test.csv; discusión del equipo pendiente'),
        ('P2.1', 2, 'YOLO y Faster R-CNN COCO evaluados', coco and test, 'Parte 2; procedencia en checkpoints; evaluacion/*/test.json'),
        ('P2.2', 5, 'Congelamiento y fine-tuning explícitos', congelados, 'modelos/*/fase_frozen.json; fase_finetune.json'),
        ('P2.3', 5, 'Justificar hiperparámetros y estrategias', False, 'Parte 2: fundamentos implementados; contrastar con resultados y discusión del equipo'),
        ('P2.4', 8, 'Comparar los cuatro modelos', coco and desde_cero and comparacion, 'comparacion/test.csv; discusión del equipo pendiente'),
        ('P3.1', 4, 'Tres tipos o niveles de degradación', hay_robustez, 'Parte 3; robustez/*/*.json'),
        ('P3.2', 6, 'Mejor modelo con las mismas métricas', hay_robustez, 'evaluacion/seleccion_cerrada.json; robustez/*/resumen.csv'),
        ('P3.3', 10, 'Discutir sensibilidad y peores condiciones', False, 'Parte 3: completar discusión con métricas y ejemplos reales'),
    ]
    filas = [dict(criterio=k, maximo=p, requisito=texto,
                  estado='Evidencia experimental disponible; revisar discusión' if logrado else 'Pendiente o parcial',
                  evidencia=evidencia) for k, p, texto, logrado, evidencia in criterios]
    guardar_csv(base / 'evidencias/rubrica.csv', filas)
    guardar_json(base / 'evidencias/rubrica.json', filas)
    display(Markdown(tabla_markdown(filas)))
    return filas
