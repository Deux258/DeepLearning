"""Celdas de la Parte 3 para evaluar checkpoints de corridas completas o parciales."""

# %% Selección del checkpoint y umbral
def cargar_checkpoint_parte3(nombre, auditoria, manifiesto, estadisticas, dispositivo):
    ruta = RAIZ_ESTUDIO / 'modelos' / nombre / 'mejor.pt'
    sello = ruta.with_suffix('.sha256').read_text(encoding='ascii').strip()
    guardado = leer_estado(ruta, recuperar=False)
    if ruta.with_suffix('.sha256').read_text(encoding='ascii').strip() != sello:
        raise ValueError(f'El checkpoint de {nombre} cambió mientras se leía; vuelve a ejecutar la evaluación.')
    config = validar_configuracion(guardado['configuracion'])
    identidad = identidad_corrida(config, auditoria, manifiesto, estadisticas)
    # El checkpoint completo permite inferencia en otra máquina o dispositivo.
    campos = ('version', 'configuracion', 'datos', 'particiones', 'normalizacion')
    if config['modelo'] != nombre or any(guardado['identidad'].get(k) != identidad[k] for k in campos):
        raise ValueError(f'Checkpoint de {nombre} incompatible con el modelo, los datos o el protocolo.')
    if nombre == 'fasterrcnn_coco' and not (PESOS_DIR / WEIGHTS[nombre][0]).is_file():
        descargar_pesos(PESOS_DIR)
    # Ambos YOLO usan la misma arquitectura. El checkpoint ya incluye todos sus pesos.
    arquitectura = 'yolo_scratch' if nombre == 'yolo_coco' else nombre
    detector = crear_detector(arquitectura, len(auditoria['categories']), TAMANO, PESOS_DIR, estadisticas)
    try:
        detector.load_state_dict(guardado['modelo'])
    except RuntimeError as error:
        raise ValueError(f'Los tensores de {nombre} no corresponden a la arquitectura instalada.') from error
    detector.eval().to(dispositivo)
    return detector, dict(checkpoint=sello, epoca=guardado['estado']['epoca'])


def seleccionar_provisional(candidatos, auditoria, manifiesto, estadisticas, dispositivo):
    ids = manifiesto['partitions']['val']
    gt = formato_coco(auditoria, ids)
    modelos = {}
    for nombre in candidatos:
        try:
            detector, ficha = cargar_checkpoint_parte3(nombre, auditoria, manifiesto, estadisticas, dispositivo)
        except (OSError, KeyError, ValueError) as error:
            print(f'{nombre} omitido: {error}', flush=True)
            continue
        print(f"Validación de {nombre}, checkpoint de la época {ficha['epoca']}.", flush=True)
        predicciones, _ = predecir_ids(detector, auditoria, ids)
        del detector
        umbral = elegir_umbral(gt, predicciones, 'val')['selected']['threshold']
        metricas = evaluar(gt, predicciones, umbral)
        modelos[nombre] = dict(ficha, umbral=umbral, mAP_validacion=metricas['map_50_95'])
    if not modelos:
        raise ValueError('No se encontró un checkpoint compatible. Revisa los archivos de la corrida original.')
    ganador = max(modelos, key=lambda nombre: modelos[nombre]['mAP_validacion'])
    return dict(version=VERSION_MOTOR, datos=huella_dataset(auditoria),
                particiones=huella_particiones(manifiesto['partitions']),
                ganador_validacion=ganador, modelos=modelos, alcance='provisional')


def carpeta_robustez_parte3(cierre, nombre):
    subcarpeta = 'robustez_provisional' if cierre.get('alcance') == 'provisional' else 'robustez'
    return RAIZ_ESTUDIO / subcarpeta / nombre


def lider_secundario_parte3(cierre):
    pruebas = {}
    for nombre in cierre['modelos']:
        ruta = RAIZ_ESTUDIO / 'evaluacion' / nombre / 'test.json'
        if not ruta.is_file():
            return None
        prueba = leer_json(ruta)
        if prueba.get('cierre') != huella(cierre) or prueba.get('checkpoint') != cierre['modelos'][nombre]['checkpoint']:
            return None
        pruebas[nombre] = prueba['metricas']['map_50_95']
    return max(pruebas, key=pruebas.get)


# Acción de Parte 3
if EJECUTAR_ROBUSTEZ:
    dispositivo = exigir_dispositivo(DISPOSITIVO_EJECUCION)
    auditoria, manifiesto, estadisticas = preparar_datos()
    if ruta_cierre.is_file():
        cierre = leer_json(ruta_cierre)
        if cierre['version'] != VERSION_MOTOR:
            raise ValueError('La selección cerrada pertenece a otra versión del protocolo.')
        alcance = 'Selección cerrada'
    else:
        cierre = seleccionar_provisional(candidatos, auditoria, manifiesto, estadisticas, dispositivo)
        alcance = 'Selección provisional entre checkpoints disponibles'
        guardar_json(RAIZ_ESTUDIO / 'robustez_provisional/seleccion.json', cierre)
    ganador = cierre['ganador_validacion']
    print(f'{alcance}: {ganador}.', flush=True)
    display(Markdown(tabla_markdown([dict(modelo=n, **fila) for n, fila in cierre['modelos'].items()])))

    detector, ficha = cargar_checkpoint_parte3(ganador, auditoria, manifiesto, estadisticas, dispositivo)
    resultados_robustez = ejecutar_robustez_parte3(cierre, auditoria, manifiesto, detector, ficha, dispositivo, ganador)
    graficar_robustez_parte3(cierre, ganador)
    del detector

    if cierre.get('alcance') != 'provisional':
        secundario = lider_secundario_parte3(cierre)
        if secundario in candidatos and secundario != ganador:
            print(f'Análisis secundario del líder de test: {secundario}.', flush=True)
            detector, ficha = cargar_checkpoint_parte3(secundario, auditoria, manifiesto, estadisticas, dispositivo)
            ejecutar_robustez_parte3(cierre, auditoria, manifiesto, detector, ficha, dispositivo, secundario)
            graficar_robustez_parte3(cierre, secundario)
else:
    print('Robustez desactivada en la configuración.')
