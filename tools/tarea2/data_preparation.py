"""Preparación exclusiva de Parte 1; conserva la auditoría y curación originales."""

# %% Auditoría, contrato y exportación compartida
def preparar_datos():
    raiz = asegurar_dataset()
    carpeta = RAIZ_ESTUDIO / 'datos'
    carpeta.mkdir(parents=True, exist_ok=True)
    if (carpeta / 'manifiesto.json').is_file():
        return cargar_contrato(raiz)
    print('Auditoría del dataset: no modifica imágenes ni etiquetas originales.')
    cruda = auditar(raiz, carpeta / 'auditoria_fuente')
    auditoria = depurar(cruda, carpeta / 'curacion')
    manifiesto = copy.deepcopy(PARTICIONES_ORIGINALES)
    if huella_dataset(auditoria) != manifiesto['dataset_fingerprint']:
        raise ValueError('La depuración no reproduce el dataset original; no se redistribuyen IDs automáticamente.')
    revisar_manifiesto(auditoria, manifiesto)
    estadisticas = calcular_normalizacion(auditoria, manifiesto, TAMANO)
    validar_contrato(auditoria, manifiesto, estadisticas)
    guardar_json(carpeta / 'auditoria.json', auditoria)
    guardar_json(carpeta / 'estadisticas_train.json', estadisticas)
    for split, ids in manifiesto['partitions'].items():
        guardar_json(carpeta / f'{split}_ids.json', ids)
        guardar_json(carpeta / f'{split}.coco.json', formato_coco(auditoria, ids))
    # Publicar manifiesto al final: su presencia indica contrato completo.
    guardar_json(carpeta / 'manifiesto.json', manifiesto)
    print({s: len(ids) for s, ids in manifiesto['partitions'].items()})
    print('Clases:', len(auditoria['categories']), '| objetos:', auditoria['summary']['objects'])
    return auditoria, manifiesto, estadisticas
