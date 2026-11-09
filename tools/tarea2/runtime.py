"""Código insertado como celdas visibles en cada notebook; no se importa al ejecutarlos."""

# %% Contrato persistente y entorno
def huella(valor):
    return hashlib.sha256(json.dumps(valor, sort_keys=True, allow_nan=False).encode()).hexdigest()


def entorno_actual():
    import importlib.metadata
    return {nombre: importlib.metadata.version(nombre) for nombre in
            ('torch', 'torchvision', 'ultralytics', 'numpy', 'Pillow', 'pycocotools')}


def exigir_gpu():
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError(
            'Entrenamiento principal detenido: PyTorch no tiene CUDA disponible. '
            'En Colab selecciona Entorno de ejecución > Cambiar tipo > GPU. '
            'Si torch termina en +cpu, instala la pareja CUDA de torch/torchvision '
            'compatible con ese runtime y reinicia la sesión antes de continuar.')
    # Comprobar operaciones de detección, no solamente la presencia del dispositivo.
    from torchvision.ops import nms
    nms(torch.tensor([[0., 0., 2., 2.]], device='cuda'),
        torch.tensor([1.], device='cuda'), .5)
    return 'cuda'


def validar_contrato(auditoria, manifiesto, estadisticas):
    revisar_manifiesto(auditoria, manifiesto)
    if manifiesto['partitions'] != PARTICIONES_ORIGINALES['partitions']:
        raise ValueError('Los IDs difieren de las particiones originales 2022/454/454')
    identidad = identidad_train(auditoria, manifiesto, TAMANO)
    if any(estadisticas.get(k) != v for k, v in identidad.items()):
        raise ValueError('Las estadísticas no corresponden a train y su preprocesamiento')
    if estadisticas['split'] != 'train' or estadisticas['images'] != 2022:
        raise ValueError('La normalización debe calcularse solo con las 2022 imágenes de train')
    if not np.isfinite(estadisticas['mean'] + estadisticas['std']).all() or min(estadisticas['std']) <= 0:
        raise ValueError('Normalización inválida')
    return True


def cargar_contrato(raiz_datos):
    carpeta = RAIZ_ESTUDIO / 'datos'
    if not (carpeta / 'manifiesto.json').is_file():
        raise FileNotFoundError('Falta el contrato de datos. Ejecuta primero la preparación de la Parte 1.')
    auditoria = leer_json(carpeta / 'auditoria.json')
    auditoria['root'] = str(Path(raiz_datos).resolve())
    manifiesto = leer_json(carpeta / 'manifiesto.json')
    estadisticas = leer_json(carpeta / 'estadisticas_train.json')
    validar_contrato(auditoria, manifiesto, estadisticas)
    # Verificar también los bytes actuales. La ruta puede cambiar de local a Colab.
    for fila in auditoria['images']:
        archivo = Path(auditoria['root']) / fila['file_name']
        if not archivo.is_file() or calcular_sha256(archivo) != fila['sha256']:
            raise ValueError(f'Imagen ausente o modificada: {archivo}')
    return auditoria, manifiesto, estadisticas


def asegurar_dataset():
    return DATASET_DIR if DATASET_DIR.is_dir() else descargar_dataset(DATA_DIR)


def guardar_csv(ruta, filas):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        return
    with ruta.open('w', encoding='utf-8-sig', newline='') as f:
        escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)


def resumen_metricas(m):
    punto = m['operating_point']
    return dict(mAP50_95=m['map_50_95'], AP50=m['ap_50'],
                precision=punto['precision'], recall=punto['recall'], F1=punto['f1'])


def tabla_markdown(filas):
    if not filas:
        return 'Sin resultados disponibles; no se sustituyen por ceros.'
    columnas = list(filas[0])
    def formato(v):
        return f'{v:.5f}' if isinstance(v, float) else ('N/D' if v is None else str(v))
    return ('| ' + ' | '.join(columnas) + ' |\n| ' + ' | '.join(['---'] * len(columnas)) + ' |\n'
            + '\n'.join('| ' + ' | '.join(formato(f[c]) for c in columnas) + ' |' for f in filas))


# %% Configuración y persistencia
def configuracion_modelo(nombre):
    if nombre not in KINDS:
        raise ValueError(nombre)
    coco = nombre.endswith('coco')
    return dict(modelo=nombre, semilla=42, tamano=512, epocas=30 if coco else 60,
                epocas_congeladas=5 if coco else 0, minimo_epocas=15 if coco else 20,
                paciencia=10, tasa=1e-3, tasa_fina=1e-4, weight_decay=1e-4,
                batch_efectivo=8, warmup_epocas=3, lr_final_factor=.01,
                clip_norm=10., amp=True, aumento=politica_aumento(),
                guardar_cada_pasos=50)


def validar_configuracion(c):
    esperadas = set(configuracion_modelo(c['modelo']))
    if set(c) != esperadas:
        raise ValueError('Claves de configuración desconocidas o faltantes')
    for k in ('semilla', 'tamano', 'epocas', 'minimo_epocas', 'paciencia',
              'batch_efectivo', 'guardar_cada_pasos'):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f'{k} debe ser entero positivo')
    if c['tamano'] != TAMANO or c['batch_efectivo'] != 8:
        raise ValueError('Este protocolo usa tamaño 512 y batch efectivo 8')
    if not 0 <= c['epocas_congeladas'] < c['epocas'] or c['minimo_epocas'] > c['epocas']:
        raise ValueError('Revisar las épocas y fases')
    for k in ('tasa', 'tasa_fina', 'weight_decay', 'clip_norm', 'lr_final_factor'):
        if not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError(f'{k} debe ser finito y positivo')
    if c['warmup_epocas'] < 0 or not 0 < c['lr_final_factor'] <= 1:
        raise ValueError('Planificación de tasa inválida')
    validar_politica_aumento(c['aumento'])
    return c


def identidad_corrida(c, auditoria, manifiesto, estadisticas):
    return dict(version=VERSION_MOTOR, configuracion=c,
                datos=huella_dataset(auditoria), particiones=huella_particiones(manifiesto['partitions']),
                normalizacion=huella(estadisticas), software=entorno_actual(), dispositivo='cuda')


def guardar_estado(ruta, estado):
    """Publica el checkpoint y su hash; mantiene una generación anterior recuperable."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + '.tmp')
    torch.save(estado, temporal)
    digest = calcular_sha256(temporal)
    sello_actual = ruta.with_suffix('.sha256')
    actual_integro = (ruta.is_file() and sello_actual.is_file()
                     and calcular_sha256(ruta) == sello_actual.read_text().strip())
    if actual_integro:
        anterior = ruta.with_name(ruta.stem + '_anterior.pt')
        shutil.copy2(ruta, anterior)
        shutil.copy2(sello_actual, anterior.with_suffix('.sha256'))
    os.replace(temporal, ruta)
    sello = ruta.with_suffix('.sha256')
    sello_temporal = sello.with_name(sello.name + '.tmp')
    sello_temporal.write_text(digest, encoding='ascii')
    os.replace(sello_temporal, sello)


def leer_estado(ruta, recuperar=True):
    ruta = Path(ruta)
    candidatos = [ruta, ruta.with_name(ruta.stem + '_anterior.pt')] if recuperar else [ruta]
    errores = []
    for candidato in candidatos:
        if not candidato.is_file():
            continue
        try:
            if calcular_sha256(candidato) != candidato.with_suffix('.sha256').read_text().strip():
                raise ValueError('Hash de checkpoint incorrecto')
            # Solo cargar checkpoints propios del estudio; torch.load permite objetos Python.
            estado = torch.load(candidato, map_location='cpu', weights_only=False)
            if candidato != ruta:
                print('Recuperando generación anterior:', candidato.name)
            return estado
        except (OSError, ValueError, RuntimeError, EOFError, KeyError) as e:
            errores.append(f'{candidato.name}: {e}')
    raise ValueError('No hay un checkpoint íntegro: ' + '; '.join(errores))


class PlanTasa:
    """Warmup lineal y coseno por actualización; su cursor se guarda al reanudar."""
    def __init__(self, optimizador, tasa, pasos_epoca, epocas, warmup=3, factor_final=.01):
        self.optimizador = optimizador
        self.tasa = tasa
        self.total = pasos_epoca * epocas
        self.warmup = min(pasos_epoca * warmup, max(0, self.total - 1))
        self.factor_final = factor_final
        self.cursor = 0

    def aplicar(self):
        if self.cursor < self.warmup:
            factor = (self.cursor + 1) / self.warmup
        else:
            progreso = (self.cursor - self.warmup) / max(1, self.total - self.warmup - 1)
            factor = self.factor_final + (1 - self.factor_final) * (1 + math.cos(math.pi * min(1., progreso))) / 2
        for grupo in self.optimizador.param_groups:
            grupo['lr'] = self.tasa * factor
        return self.tasa * factor

    def step(self):
        self.cursor += 1

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'optimizador'}

    def load_state_dict(self, estado):
        self.__dict__.update(estado)


def fase_actual(c, epoca):
    return ('frozen' if epoca < c['epocas_congeladas'] else 'finetune') if c['epocas_congeladas'] else 'full'


def componentes_entrenamiento(detector, c, fase, longitud):
    detector.set_phase(fase)
    tasa = c['tasa_fina'] if fase == 'finetune' else c['tasa']
    opt = adamw(detector, tasa)
    for g in opt.param_groups:
        if g['weight_decay']:
            g['weight_decay'] = c['weight_decay']
    epocas = (c['epocas_congeladas'] if fase == 'frozen' else
              c['epocas'] - c['epocas_congeladas'] if fase == 'finetune' else c['epocas'])
    plan = PlanTasa(opt, tasa, math.ceil(longitud / c['batch_efectivo']), epocas,
                   c['warmup_epocas'], c['lr_final_factor'])
    return opt, plan


# %% Optimización, memoria y comprobaciones reales
def perdida_media(detector, imagenes, objetivos):
    perdida, componentes = detector.loss(imagenes, objetivos)
    # DetectionModel devuelve pérdida multiplicada por el tamaño del microbatch.
    if detector.kind.startswith('yolo'):
        perdida = perdida / len(imagenes)
    return perdida, componentes


def actualizar_ventana(detector, optimizador, escalador, dataset, indices, batch, clip=10.):
    """Promedia por imágenes, incluida la última ventana incompleta."""
    dispositivo = next(detector.parameters()).device
    azar = capturar_azar()
    buffers = {n: b.detach().clone() for n, b in detector.named_buffers()}
    for intento in range(8):
        optimizador.zero_grad(set_to_none=True)
        total = 0.
        partes = defaultdict(float)
        for inicio in range(0, len(indices), batch):
            muestras = [dataset[i] for i in indices[inicio:inicio + batch]]
            imagenes = [x.to(dispositivo) for x, _ in muestras]
            objetivos = [{k: v.to(dispositivo) for k, v in y.items()} for _, y in muestras]
            with torch.autocast(device_type=dispositivo.type, enabled=escalador.is_enabled()):
                perdida, componentes = perdida_media(detector, imagenes, objetivos)
                if perdida.ndim != 0 or not bool(torch.isfinite(perdida)):
                    raise RuntimeError('Pérdida no finita o no escalar')
                ponderada = perdida * len(muestras) / len(indices)
            escalador.scale(ponderada).backward()
            total += float(perdida.detach()) * len(muestras) / len(indices)
            if isinstance(componentes, dict):
                for k, v in componentes.items():
                    partes[k] += float(v.detach()) * len(muestras) / len(indices)
            else:
                for k, v in enumerate(componentes.detach().flatten()):
                    partes[f'componente_{k}'] += float(v) * len(muestras) / len(indices)
        escalador.unscale_(optimizador)
        gradientes = [p.grad for p in detector.parameters() if p.requires_grad and p.grad is not None]
        if gradientes and all(bool(torch.isfinite(g).all()) for g in gradientes):
            norma = torch.nn.utils.clip_grad_norm_(detector.parameters(), clip)
            escalador.step(optimizador)
            escalador.update()
            optimizador.zero_grad(set_to_none=True)
            return dict(perdida=total, norma_gradiente=float(norma), componentes=dict(partes))
        if not escalador.is_enabled() or not gradientes:
            raise RuntimeError('Gradientes ausentes o no finitos')
        escalador.step(optimizador)  # Omite el paso con inf y reduce la escala.
        escalador.update()
        with torch.no_grad():
            for nombre, buffer in detector.named_buffers():
                buffer.copy_(buffers[nombre])
        restaurar_azar(azar)
    raise RuntimeError('AMP desborda tras ocho reintentos; conserva el último checkpoint')


def calibrar_batch(c, auditoria, manifiesto, estadisticas):
    exigir_gpu()
    datos = DatosDeteccion(auditoria, manifiesto['partitions']['train'], TAMANO, False)
    # Incluir imágenes densas en el sondeo de memoria de ambas fases.
    orden = sorted(range(len(datos)), key=lambda i: len(datos.rows[i]['annotations']), reverse=True)[:8]
    fases = ['frozen', 'finetune'] if c['epocas_congeladas'] else ['full']
    informe = []
    for batch in (8, 4, 2, 1):
        exito = True
        for fase in fases:
            detector = opt = scaler = None
            try:
                fijar_semilla(c['semilla'])
                detector = crear_detector(c['modelo'], len(auditoria['categories']), TAMANO, PESOS_DIR, estadisticas).cuda()
                detector.set_phase(fase)
                opt = adamw(detector, c['tasa'])
                scaler = torch.amp.GradScaler('cuda', enabled=c['amp'])
                torch.cuda.reset_peak_memory_stats()
                # También prueba la forma y el posprocesamiento en inferencia.
                fila = datos.rows[orden[0]]
                with Image.open(Path(auditoria['root']) / fila['file_name']) as imagen:
                    detector.predict([imagen.convert('RGB')])
                paso = actualizar_ventana(detector, opt, scaler, datos, orden, batch, c['clip_norm'])
                torch.cuda.synchronize()
                memoria = torch.cuda.max_memory_reserved()
                limite = torch.cuda.get_device_properties(0).total_memory
                if memoria > .9 * limite:
                    raise torch.cuda.OutOfMemoryError('Sondeo supera 90% de VRAM; se reserva margen')
                informe.append(dict(batch=batch, fase=fase, amp=scaler.is_enabled(),
                                    memoria_reservada=memoria, perdida=paso['perdida'], valido=True))
            except torch.cuda.OutOfMemoryError:
                informe.append(dict(batch=batch, fase=fase, valido=False, motivo='memoria'))
                exito = False
            finally:
                del detector, opt, scaler
                gc.collect()
                torch.cuda.empty_cache()
            if not exito:
                break
        if exito:
            return dict(batch_fisico=batch, acumulacion=8 // batch, fases=informe,
                        gpu=torch.cuda.get_device_name(0), software=entorno_actual())
    raise RuntimeError('Ni batch físico 1 cabe con margen; revisar resolución/modelo en una corrida nueva')


def predecir_ids(detector, auditoria, ids, condicion=None):
    filas = {f['id']: f for f in auditoria['images']}
    resultado = []
    if next(detector.parameters()).device.type == 'cuda':
        torch.cuda.synchronize()
    inicio = time.perf_counter()
    for identificador in ids:
        fila = filas[identificador]
        with Image.open(Path(auditoria['root']) / fila['file_name']) as original:
            imagen = original.convert('RGB')
        if condicion is not None:
            imagen = degradar_imagen(imagen, condicion, identificador)
        resultado.extend(predicciones_coco([identificador], detector.predict([imagen])))
    if next(detector.parameters()).device.type == 'cuda':
        torch.cuda.synchronize()
    return resultado, time.perf_counter() - inicio


def diagnostico_real(c, auditoria, manifiesto, estadisticas, dispositivo='cuda', pasos=300):
    """Ocho imágenes de TRAIN, sin aumentos: diagnóstico, nunca métrica final."""
    if dispositivo == 'cuda':
        exigir_gpu()
    azar = capturar_azar()
    try:
        fijar_semilla(c['semilla'])
        ids = manifiesto['partitions']['train'][:8]
        datos = DatosDeteccion(auditoria, ids, TAMANO, False)
        detector = crear_detector(c['modelo'], len(auditoria['categories']), TAMANO, PESOS_DIR, estadisticas).to(dispositivo)
        detector.set_phase('finetune' if c['epocas_congeladas'] else 'full')
        opt = adamw(detector, 1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=dispositivo == 'cuda' and c['amp'])
        gt = formato_coco(auditoria, ids)
        antes, _ = predecir_ids(detector, auditoria, ids)
        inicial = evaluar(gt, antes)
        perdidas = []
        for paso in range(pasos):
            indices = [(paso * 2) % 8, (paso * 2 + 1) % 8]
            for indice in indices:
                imagen, objetivo = datos[indice]
                assert imagen.shape == (3, TAMANO, TAMANO) and torch.isfinite(imagen).all()
                assert imagen.min() >= 0 and imagen.max() <= 1
                assert bool((objetivo['boxes'][:, 2:] > objetivo['boxes'][:, :2]).all())
                assert bool(((objetivo['labels'] >= 0) & (objetivo['labels'] < len(auditoria['categories']))).all())
            perdidas.append(actualizar_ventana(detector, opt, scaler, datos, indices, 2)['perdida'])
        despues, _ = predecir_ids(detector, auditoria, ids)
        final = evaluar(gt, despues)
        # Mejorar sobre cero no demuestra capacidad de memorizar: se exige AP50 apreciable.
        aprobado = final['ap_50'] >= .5 and np.mean(perdidas[-10:]) < np.mean(perdidas[:10])
        return dict(alcance='diagnostico_ocho_imagenes_train_no_validacion_ni_test',
                    modelo=c['modelo'], imagenes=ids, pasos=pasos, dispositivo=dispositivo,
                    ap50_inicial=inicial['ap_50'], ap50_final=final['ap_50'],
                    perdida_inicial=float(np.mean(perdidas[:10])), perdida_final=float(np.mean(perdidas[-10:])),
                    aprobado=bool(aprobado), version=VERSION_MOTOR, configuracion=huella(c),
                    datos=huella_dataset(auditoria), software=entorno_actual())
    finally:
        restaurar_azar(azar)


# %% Entrenamiento reanudable sin límite horario
def entrenar_modelo(c, auditoria, manifiesto, estadisticas, pausar_tras_pasos=None):
    exigir_gpu()
    validar_configuracion(c)
    validar_contrato(auditoria, manifiesto, estadisticas)
    carpeta = RAIZ_ESTUDIO / 'modelos' / c['modelo']
    carpeta.mkdir(parents=True, exist_ok=True)
    identidad = identidad_corrida(c, auditoria, manifiesto, estadisticas)
    ultimo = carpeta / 'ultimo.pt'
    guardado = leer_estado(ultimo) if ultimo.exists() else None
    if guardado and guardado['identidad'] != identidad:
        raise ValueError('Checkpoint incompatible. Usa otra ETIQUETA_ESTUDIO para cambiar el protocolo.')
    if guardado and guardado['estado']['completo']:
        print(c['modelo'], 'ya completado:', guardado['estado']['motivo'])
        return guardado['estado']
    if guardado is None:
        archivo_diagnostico = carpeta / 'diagnostico_train.json'
        diagnostico = leer_json(archivo_diagnostico) if archivo_diagnostico.is_file() else None
        vigente = (diagnostico and diagnostico.get('version') == VERSION_MOTOR
                   and diagnostico.get('configuracion') == huella(c)
                   and diagnostico.get('datos') == huella_dataset(auditoria)
                   and diagnostico.get('software') == entorno_actual()
                   and diagnostico.get('dispositivo') == 'cuda')
        if not vigente:
            diagnostico = diagnostico_real(c, auditoria, manifiesto, estadisticas)
            guardar_json(archivo_diagnostico, diagnostico)
        if not diagnostico['aprobado']:
            raise RuntimeError('No se logró memorizar ocho imágenes de train. Revisar el diagnóstico antes de entrenar.')
        calibracion = calibrar_batch(c, auditoria, manifiesto, estadisticas)
        guardar_json(carpeta / 'calibracion.json', calibracion)
    else:
        calibracion = guardado['calibracion']
        if calibracion['gpu'] != torch.cuda.get_device_name(0):
            raise ValueError('Cambió la GPU. Recalibra en un estudio nuevo para conservar la trazabilidad.')
    fijar_semilla(c['semilla'])
    detector = crear_detector(c['modelo'], len(auditoria['categories']), TAMANO, PESOS_DIR, estadisticas).cuda()
    datos = DatosDeteccion(auditoria, manifiesto['partitions']['train'], TAMANO, c['aumento'])
    estado = guardado['estado'] if guardado else dict(
        epoca=0, cursor=0, orden=[], pasos=0, fase=fase_actual(c, 0), mejor=-1.,
        mejor_epoca=None, sin_mejora=0, suma_perdida=0., ejemplos=0,
        componentes={}, segundos_train=0., segundos_validacion=0., segundos_guardado=0.,
        historial=[], completo=False, motivo=None)
    opt, plan = componentes_entrenamiento(detector, c, estado['fase'], len(datos))
    scaler = torch.amp.GradScaler('cuda', enabled=c['amp'])
    if guardado:
        detector.load_state_dict(guardado['modelo'])
        opt.load_state_dict(guardado['optimizador'])
        plan.load_state_dict(guardado['planificador'])
        scaler.load_state_dict(guardado['escalador'])
        restaurar_azar(guardado['azar'])
        # El sidecar incluye el tiempo de la última escritura, que el propio snapshot
        # todavía no podía medir. Solo aceptar el sidecar de ese mismo paso/época.
        ruta_estado = carpeta / 'estado.json'
        if ruta_estado.is_file():
            lateral = leer_json(ruta_estado)
            if lateral.get('pasos') == estado['pasos'] and lateral.get('epoca') == estado['epoca']:
                estado['segundos_guardado'] = max(estado['segundos_guardado'], lateral['segundos_guardado'])
    def publicar(mejor=False):
        inicio = time.perf_counter()
        snapshot = dict(identidad=identidad, configuracion=c, calibracion=calibracion,
                        modelo=detector.state_dict(), optimizador=opt.state_dict(),
                        planificador=plan.state_dict(), escalador=scaler.state_dict(),
                        estado=copy.deepcopy(estado), azar=capturar_azar(),
                        procedencia=detector.provenance)
        if mejor:
            guardar_estado(carpeta / 'mejor.pt', snapshot)
        guardar_estado(ultimo, snapshot)
        estado['segundos_guardado'] += time.perf_counter() - inicio
        guardar_json(carpeta / 'estado.json', estado)
    publicar()
    pasos_sesion = 0
    try:
        while estado['epoca'] < c['epocas']:
            fase = fase_actual(c, estado['epoca'])
            if fase != estado['fase']:
                opt, plan = componentes_entrenamiento(detector, c, fase, len(datos))
                estado.update(fase=fase, sin_mejora=0)
            guardar_json(carpeta / f'fase_{fase}.json', detector.parameter_report())
            if not estado['orden']:
                estado['orden'] = random.Random(c['semilla'] + estado['epoca']).sample(range(len(datos)), len(datos))
            while estado['cursor'] < len(datos):
                indices = estado['orden'][estado['cursor']:estado['cursor'] + 8]
                tasa = plan.aplicar()
                torch.cuda.synchronize()
                inicio = time.perf_counter()
                paso = actualizar_ventana(detector, opt, scaler, datos, indices,
                                          calibracion['batch_fisico'], c['clip_norm'])
                torch.cuda.synchronize()
                estado['segundos_train'] += time.perf_counter() - inicio
                plan.step()
                estado['cursor'] += len(indices)
                estado['pasos'] += 1
                pasos_sesion += 1
                estado['suma_perdida'] += paso['perdida'] * len(indices)
                estado['ejemplos'] += len(indices)
                for nombre, valor in paso['componentes'].items():
                    estado['componentes'][nombre] = estado['componentes'].get(nombre, 0.) + valor * len(indices)
                if estado['pasos'] % c['guardar_cada_pasos'] == 0:
                    publicar()
                    print(c['modelo'], 'época', estado['epoca'] + 1, 'imágenes', estado['cursor'], '/', len(datos), flush=True)
                if pausar_tras_pasos and pasos_sesion >= pausar_tras_pasos:
                    estado['motivo'] = 'pausa_solicitada'
                    publicar()
                    return estado
            preds, segundos = predecir_ids(detector, auditoria, manifiesto['partitions']['val'])
            inicio = time.perf_counter()
            metricas = evaluar(formato_coco(auditoria, manifiesto['partitions']['val']), preds)
            estado['segundos_validacion'] += segundos + time.perf_counter() - inicio
            puntuacion = metricas['map_50_95']
            if puntuacion is None:
                raise ValueError('Validación no tiene AP definida')
            mejora = puntuacion > estado['mejor']
            estado['epoca'] += 1
            fila = dict(epoca=estado['epoca'], fase=fase, perdida=estado['suma_perdida'] / estado['ejemplos'],
                        map_50_95=puntuacion, ap_50=metricas['ap_50'], tasa=tasa,
                        componentes={k: v / estado['ejemplos'] for k, v in estado['componentes'].items()},
                        train_segundos_acumulados=estado['segundos_train'],
                        validacion_segundos_acumulados=estado['segundos_validacion'])
            estado['historial'].append(fila)
            estado['sin_mejora'] = 0 if mejora else estado['sin_mejora'] + 1
            if mejora:
                estado.update(mejor=puntuacion, mejor_epoca=estado['epoca'])
            estado.update(cursor=0, orden=[], suma_perdida=0., ejemplos=0, componentes={})
            guardar_json(carpeta / 'historial.json', estado['historial'])
            print(f"{c['modelo']} | época {estado['epoca']}/{c['epocas']} | mAP {puntuacion:.5f} | AP50 {metricas['ap_50']:.5f}", flush=True)
            publicar(mejor=mejora)
            if estado['epoca'] >= c['minimo_epocas'] and estado['sin_mejora'] >= c['paciencia']:
                estado.update(completo=True, motivo='paciencia_validacion')
                break
        if not estado['completo']:
            estado.update(completo=True, motivo='maximo_epocas')
        publicar()
        guardar_json(carpeta / 'configuracion.json', c)
        return estado
    except KeyboardInterrupt:
        # El checkpoint publicado termina en una ventana completa. No guardar gradientes parciales.
        print('Interrumpido: retoma desde el último checkpoint publicado (hasta 50 pasos de trabajo).')
        raise


# %% Evaluación común y cierre antes de test
def cargar_mejor(nombre, auditoria, manifiesto, estadisticas, dispositivo='cuda'):
    carpeta = RAIZ_ESTUDIO / 'modelos' / nombre
    estado_final = leer_estado(carpeta / 'ultimo.pt')
    if not estado_final['estado']['completo']:
        raise ValueError(f'{nombre} está pausado; no habilita evaluación final')
    saved = leer_estado(carpeta / 'mejor.pt', recuperar=False)
    c = saved['configuracion']
    if saved['identidad'] != identidad_corrida(c, auditoria, manifiesto, estadisticas):
        raise ValueError('Datos, código, configuración o software incompatibles con el checkpoint')
    detector = crear_detector(nombre, len(auditoria['categories']), TAMANO, PESOS_DIR, estadisticas).to(dispositivo)
    detector.load_state_dict(saved['modelo'])
    detector.set_phase(saved['estado']['fase'])
    detector.eval()
    return detector, saved, estado_final['estado']


def evaluar_validacion(nombre, auditoria, manifiesto, estadisticas, dispositivo='cuda'):
    detector, saved, estado = cargar_mejor(nombre, auditoria, manifiesto, estadisticas, dispositivo)
    filas = {f['id']: f for f in auditoria['images']}
    with Image.open(Path(auditoria['root']) / filas[manifiesto['partitions']['val'][0]]['file_name']) as imagen:
        for _ in range(3):
            detector.predict([imagen.convert('RGB')])
    predicciones, segundos = predecir_ids(detector, auditoria, manifiesto['partitions']['val'])
    gt = formato_coco(auditoria, manifiesto['partitions']['val'])
    umbral = elegir_umbral(gt, predicciones, 'val')
    metricas = evaluar(gt, predicciones, umbral['selected']['threshold'])
    carpeta = RAIZ_ESTUDIO / 'evaluacion' / nombre
    carpeta.mkdir(parents=True, exist_ok=True)
    resultado = dict(modelo=nombre, split='val', metricas=metricas, umbral=umbral,
                     version=VERSION_MOTOR, checkpoint=calcular_sha256(RAIZ_ESTUDIO / 'modelos' / nombre / 'mejor.pt'),
                     particiones=huella_particiones(manifiesto['partitions']),
                     segundos_inferencia=segundos, estado_entrenamiento=estado,
                     alcance_tiempo='batch 1: apertura, preprocesamiento, red y posprocesamiento; sin métricas COCO',
                     dispositivo=dispositivo, gpu=torch.cuda.get_device_name(0) if dispositivo == 'cuda' else None,
                     software=entorno_actual())
    guardar_json(carpeta / 'validacion.json', resultado)
    guardar_json(carpeta / 'predicciones_val.json', predicciones)
    guardar_csv(carpeta / 'ap_por_clase_val.csv', metricas['per_class'])
    return resultado


def cerrar_seleccion(auditoria, manifiesto, estadisticas):
    resultados = {n: leer_json(RAIZ_ESTUDIO / 'evaluacion' / n / 'validacion.json') for n in KINDS}
    for n, r in resultados.items():
        if r['version'] != VERSION_MOTOR or r['checkpoint'] != calcular_sha256(RAIZ_ESTUDIO / 'modelos' / n / 'mejor.pt'):
            raise ValueError('La validación guardada no corresponde al checkpoint actual')
        if r['particiones'] != huella_particiones(manifiesto['partitions']):
            raise ValueError('Validación de otras particiones')
    ganador = max(KINDS, key=lambda n: resultados[n]['metricas']['map_50_95'])
    cierre = dict(version=VERSION_MOTOR, datos=huella_dataset(auditoria),
                  particiones=huella_particiones(manifiesto['partitions']), ganador_validacion=ganador,
                  modelos={n: dict(checkpoint=r['checkpoint'], umbral=r['umbral']['selected']['threshold'],
                                   mAP_validacion=r['metricas']['map_50_95']) for n, r in resultados.items()})
    ruta = RAIZ_ESTUDIO / 'evaluacion' / 'seleccion_cerrada.json'
    if ruta.exists() and leer_json(ruta) != cierre:
        raise ValueError('La selección ya estaba cerrada. Usa un estudio nuevo para cambiarla.')
    guardar_json(ruta, cierre)
    return cierre


def evaluar_test(nombre, cierre, auditoria, manifiesto, estadisticas, dispositivo='cuda'):
    if cierre['particiones'] != huella_particiones(manifiesto['partitions']) or cierre['datos'] != huella_dataset(auditoria):
        raise ValueError('El cierre pertenece a otros datos')
    detector, _, _ = cargar_mejor(nombre, auditoria, manifiesto, estadisticas, dispositivo)
    if cierre['modelos'][nombre]['checkpoint'] != calcular_sha256(RAIZ_ESTUDIO / 'modelos' / nombre / 'mejor.pt'):
        raise ValueError('Se modificó el checkpoint después del cierre')
    filas = {f['id']: f for f in auditoria['images']}
    with Image.open(Path(auditoria['root']) / filas[manifiesto['partitions']['test'][0]]['file_name']) as imagen:
        for _ in range(3):
            detector.predict([imagen.convert('RGB')])
    preds, segundos = predecir_ids(detector, auditoria, manifiesto['partitions']['test'])
    metricas = evaluar(formato_coco(auditoria, manifiesto['partitions']['test']), preds, cierre['modelos'][nombre]['umbral'])
    carpeta = RAIZ_ESTUDIO / 'evaluacion' / nombre
    resultado = dict(modelo=nombre, split='test', metricas=metricas, segundos_inferencia=segundos,
                     checkpoint=cierre['modelos'][nombre]['checkpoint'], cierre=huella(cierre),
                     dispositivo=dispositivo, gpu=torch.cuda.get_device_name(0) if dispositivo == 'cuda' else None,
                     software=entorno_actual(),
                     alcance_tiempo='batch 1: apertura, preprocesamiento, red y posprocesamiento; sin métricas COCO')
    guardar_json(carpeta / 'test.json', resultado)
    guardar_json(carpeta / 'predicciones_test.json', preds)
    guardar_csv(carpeta / 'ap_por_clase_test.csv', metricas['per_class'])
    guardar_ejemplos(detector, auditoria, manifiesto['partitions']['test'][:8], carpeta / 'ejemplos_test',
                     cierre['modelos'][nombre]['umbral'])
    return resultado
