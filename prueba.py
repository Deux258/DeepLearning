def abrir_url(url, **headers):
    return urllib.request.urlopen(
        urllib.request.Request(
            url, headers={"User-Agent": "FjordFish-course-audit/0.1", **headers}
        ),
        timeout=120,
    )


def descargar_partes(entry, partial, workers):
    """Descarga por partes; cada parte se puede retomar si se corta internet."""
    total, block = (entry["size"], 32 * 1024 * 1024)
    pieces = partial.parent / "download-pieces"
    pieces.mkdir(exist_ok=True)
    jobs = [
        (i, start, min(total - 1, start + block - 1))
        for i, start in enumerate(range(0, total, block))
    ]
    url = (
        entry["links"]["self"]
        .replace("/api/records/", "/records/")
        .removesuffix("/content")
        + "?download=1"
    )

    def fetch(job):
        index, start, end = job
        piece = pieces / f"{index:06d}.bin"
        temporary = piece.with_suffix(".part")
        size = end - start + 1
        if piece.exists() and piece.stat().st_size == size:
            return size
        for attempt in range(3):
            try:
                offset = temporary.stat().st_size if temporary.exists() else 0
                if not 0 <= offset <= size:
                    offset = 0
                if offset < size:
                    with abrir_url(
                        url, Range=f"bytes={start + offset}-{end}"
                    ) as response:
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range")
                            != f"bytes {start + offset}-{end}/{total}"
                        ):
                            raise ValueError("El servidor devolvió un rango incorrecto")
                        with temporary.open("ab" if offset else "wb") as stream:
                            while offset < size:
                                data = response.read(min(1024 * 1024, size - offset))
                                if not data:
                                    raise ValueError("Parte incompleta")
                                stream.write(data)
                                offset += len(data)
                temporary.replace(piece)
                return size
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 16))) as pool:
        complete = 0
        for future in as_completed([pool.submit(fetch, job) for job in jobs]):
            complete += future.result()
            print(f"Descarga: {complete / total:.0%}", flush=True)
    with partial.open("wb") as destination:
        for index, _, _ in jobs:
            with (pieces / f"{index:06d}.bin").open("rb") as stream:
                shutil.copyfileobj(stream, destination, 4 * 1024 * 1024)


def descargar_dataset(output, record="17950781", workers=8):
    """Verifica el archivo oficial y extrae sin alterar los datos originales."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with abrir_url(f"https://zenodo.org/api/records/{record}") as response:
        metadata = json.load(response)
    entry = next((item for item in metadata["files"] if item["key"] == "FjordFish.zip"))
    archive, partial = (output / "FjordFish.zip", output / "FjordFish.zip.part")
    if not archive.exists():
        if shutil.disk_usage(output).free < entry["size"] * 3:
            raise OSError("Falta espacio para las partes, el ZIP y su contenido")
        descargar_partes(entry, partial, workers)
        partial.replace(archive)
    algorithm, expected = entry["checksum"].split(":", 1)
    digest = hashlib.new(algorithm)
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if archive.stat().st_size != entry["size"] or digest.hexdigest() != expected:
        raise ValueError("El ZIP no coincide con el tamaño o checksum oficial")
    destination = output / "extracted"
    marker = destination / ".extraction.json"
    identity = {"checksum": entry["checksum"], "record": record}
    if marker.exists() and leer_json(marker) != identity:
        raise ValueError("La extracción existente corresponde a otra versión")
    if not marker.exists():
        destination.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as archive_file:
            if shutil.disk_usage(output).free < sum(
                (item.file_size for item in archive_file.infolist())
            ):
                raise OSError("Falta espacio para extraer el ZIP")
            for item in archive_file.infolist():
                safe = (
                    (destination / item.filename).resolve().is_relative_to(destination)
                )
                symbolic = item.external_attr >> 16 & 61440 == 40960
                if not safe or symbolic:
                    raise ValueError(f"Ruta insegura en el ZIP: {item.filename}")
            archive_file.extractall(destination)
        guardar_json(marker, identity)
    for piece in (output / "download-pieces").glob("*.bin"):
        piece.unlink()
    guardar_json(
        output / "source.json",
        {
            "record": record,
            "url": f"https://zenodo.org/records/{record}",
            "version": metadata["metadata"].get("version"),
            "file": entry,
            "verified": True,
            "extraction": str(destination),
        },
    )
    return destination


def graficar_datos(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["images"]
    if not rows:
        return
    classes = report["summary"]["class_counts"]
    fig, ax = plt.subplots(figsize=(12, max(5, len(classes) * 0.3)))
    y = np.arange(len(classes))
    ax.barh(y, [r["objects"] for r in classes], color="#247a80")
    ax.set_yticks(y, [f"{r['id']}: {r['name']}" for r in classes])
    ax.invert_yaxis()
    ax.set_xlabel("Objetos anotados")
    ax.set_title("Catálogo declarado y soporte real por clase")
    for i, row in enumerate(classes):
        ax.text(row["objects"], i, f"  {row['objects']}", va="center", fontsize=8)
    ax.margins(x=0.1)
    fig.tight_layout()
    fig.savefig(output / "class_distribution.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(
        [r["width"] for r in rows], [r["height"] for r in rows], alpha=0.25, s=12
    )
    axes[0].set(
        xlabel="Ancho (px)", ylabel="Alto (px)", title="Resoluciones originales"
    )
    counts = [len(r["annotations"]) for r in rows]
    axes[1].hist(counts, bins=np.arange(max(counts) + 2) - 0.5, color="#247a80")
    axes[1].set(
        xlabel="Objetos por imagen", ylabel="Imágenes", title="Multiplicidad de objetos"
    )
    areas = [
        (a["bbox_xyxy"][2] - a["bbox_xyxy"][0])
        * (a["bbox_xyxy"][3] - a["bbox_xyxy"][1])
        / (r["width"] * r["height"])
        for r in rows
        for a in r["annotations"]
    ]
    if areas:
        axes[2].hist(np.log10(np.maximum(areas, 1e-12)), bins=35, color="#247a80")
    axes[2].set(
        xlabel="log10(área caja / área imagen)",
        ylabel="Objetos",
        title="Escala de los objetos",
    )
    fig.tight_layout()
    fig.savefig(output / "data_distributions.png", dpi=140)
    plt.close(fig)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def firma_archivos(root):
    import json

    root = Path(root)
    entries = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | {
            ".txt",
            ".yaml",
            ".yml",
        }:
            stat = p.stat()
            entries.append(
                (p.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns)
            )
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def leer_clases(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("El YAML debe declarar names como lista o diccionario")
    names = {
        int(k): re.sub("^\\d+\\s*:\\s*", "", str(v)).strip() for k, v in names.items()
    }
    if sorted(names) != list(range(len(names))):
        raise ValueError("Los IDs del catálogo deben ser contiguos desde cero")
    if len(set(names.values())) != len(names):
        raise ValueError("Nombres de clases duplicados")
    return [names[i] for i in range(len(names))]


def ruta_etiqueta(image_path):
    parts = list(image_path.parts)
    matches = [i for i, part in enumerate(parts) if part.lower() == "images"]
    if not matches:
        raise ValueError(f"La imagen no está bajo un directorio images: {image_path}")
    parts[matches[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def leer_etiquetas(path, width, height, num_classes, allow_missing=False):
    if not path.exists():
        if allow_missing:
            return []
        raise ValueError(f"Etiqueta ausente: {path}")
    result = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 5:
            raise ValueError(f"{path}:{line_number}: se esperan class cx cy w h")
        values = np.asarray([float(x) for x in tokens], dtype=float)
        category = int(values[0]) if np.isfinite(values[0]) else -1
        if (
            not np.isfinite(values).all()
            or category != values[0]
            or (not 0 <= category < num_classes)
        ):
            raise ValueError(f"{path}:{line_number}: clase o valores inválidos")
        cx, cy, w, h = values[1:]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and (0 < w <= 1) and (0 < h <= 1)):
            raise ValueError(f"{path}:{line_number}: coordenadas YOLO inválidas")
        box = yolo_a_cajas(values[1:], width, height)
        tolerance = 0.0001 * max(width, height)
        if (
            min(box) < -tolerance
            or box[2] > width + tolerance
            or box[3] > height + tolerance
        ):
            raise ValueError(f"{path}:{line_number}: caja fuera de la imagen")
        box = [
            max(0.0, box[0]),
            max(0.0, box[1]),
            min(float(width), box[2]),
            min(float(height), box[3]),
        ]
        result.append({"category_id": category, "bbox_xyxy": box})
    return result


class GruposConectados:

    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def auditar(
    root, output, yaml_path=None, groups_csv=None, allow_missing=False, near_distance=4
):
    root, output = (Path(root).resolve(), Path(output))
    if not root.is_dir():
        raise ValueError(f"No existe el dataset: {root}")
    yaml_files = (
        sorted(root.rglob("data.y*ml")) if yaml_path is None else [Path(yaml_path)]
    )
    if len(yaml_files) != 1:
        raise ValueError(
            f"Se requiere un único data.yaml o --yaml explícito; encontrados: {yaml_files}"
        )
    yaml_path = yaml_files[0]
    names = leer_clases(yaml_path)
    paths = sorted(
        (
            p
            for p in root.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
            and "images" in [s.lower() for s in p.parts]
        )
    )
    if not paths:
        raise ValueError("No se encontraron imágenes bajo images/")
    rows, errors, warnings = ([], [], [])
    used_labels = set()
    for path in paths:
        try:
            with Image.open(path) as raw:
                if raw.getexif().get(274, 1) != 1:
                    raise ValueError(
                        f"{path}: orientación EXIF distinta de 1; revisar coordenadas antes de corregir"
                    )
                image = raw.convert("RGB")
                width, height = image.size
                small = np.asarray(image.convert("L").resize((9, 8)))
                bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
                dhash = int("".join(("1" if b else "0" for b in bits)), 2)
                pixel_hash = hashlib.sha256(
                    f"{width}x{height}".encode() + image.tobytes()
                ).hexdigest()
            labels_file = ruta_etiqueta(path)
            annotations = leer_etiquetas(
                labels_file, width, height, len(names), allow_missing
            )
            used_labels.add(labels_file.resolve())
            rows.append(
                {
                    "id": len(rows) + 1,
                    "file_name": path.relative_to(root).as_posix(),
                    "width": width,
                    "height": height,
                    "sha256": calcular_sha256(path),
                    "pixel_sha256": pixel_hash,
                    "dhash": f"{dhash:016x}",
                    "annotations": annotations,
                }
            )
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    orphan = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*.txt")
        if "labels" in [s.lower() for s in p.parts] and p.resolve() not in used_labels
    ]
    if orphan:
        errors.append(
            f"{len(orphan)} etiquetas huérfanas o asociadas a imágenes inválidas"
        )
    uf = GruposConectados(len(rows))
    exact, near = ([], [])
    seen_hash, seen_dhash = ({}, {})
    for i, row in enumerate(rows):
        sha = row["pixel_sha256"]
        if sha in seen_hash:
            other = seen_hash[sha]
            exact.append([rows[other]["id"], row["id"]])
            uf.union(i, other)
            a = sorted(
                (
                    (x["category_id"], tuple(np.round(x["bbox_xyxy"], 3)))
                    for x in row["annotations"]
                )
            )
            b = sorted(
                (
                    (x["category_id"], tuple(np.round(x["bbox_xyxy"], 3)))
                    for x in rows[other]["annotations"]
                )
            )
            if a != b:
                errors.append(
                    f"Duplicados con etiquetas diferentes: {row['file_name']} y {rows[other]['file_name']}"
                )
        seen_hash[sha] = i
        dh = int(row["dhash"], 16)
        if near_distance >= 0:
            for previous, indices in seen_dhash.items():
                if (dh ^ previous).bit_count() <= near_distance:
                    other = indices[0]
                    uf.union(i, other)
                    if rows[other]["pixel_sha256"] != sha:
                        near.append([rows[other]["id"], row["id"]])
            seen_dhash.setdefault(dh, []).append(i)
    if groups_csv:
        declared = {}
        covered_paths = set()
        by_path = {r["file_name"]: i for i, r in enumerate(rows)}
        with Path(groups_csv).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["file_name"].replace("\\", "/")
                if key not in by_path or not row.get("group_id"):
                    raise ValueError(f"Grupo inválido: {row}")
                i = by_path[key]
                covered_paths.add(key)
                if row["group_id"] in declared:
                    uf.union(i, declared[row["group_id"]])
                declared[row["group_id"]] = i
        if len(covered_paths) < len(rows):
            warnings.append(
                f"Metadatos de secuencia parciales: {len(covered_paths)}/{len(rows)} imágenes"
            )
    else:
        warnings.append(
            "Sin metadatos de video/secuencia: dHash no garantiza independencia entre escenas"
        )
    components = {}
    for i, row in enumerate(rows):
        component = uf.find(i)
        components[component] = min(components.get(component, row["id"]), row["id"])
    for i, row in enumerate(rows):
        row["group_id"] = f"g{components[uf.find(i)]:06d}"
    counts = Counter((a["category_id"] for r in rows for a in r["annotations"]))
    image_counts = Counter(
        (c for r in rows for c in {a["category_id"] for a in r["annotations"]})
    )
    group_sizes = Counter((r["group_id"] for r in rows))
    if group_sizes and max(group_sizes.values()) > len(rows) * 0.2:
        warnings.append(
            "Un grupo supera el 20% de las imágenes; revisar pares similares antes de fijar particiones"
        )
    relative_areas = [
        (a["bbox_xyxy"][2] - a["bbox_xyxy"][0])
        * (a["bbox_xyxy"][3] - a["bbox_xyxy"][1])
        / (r["width"] * r["height"])
        for r in rows
        for a in r["annotations"]
    ]
    report = {
        "schema_version": 1,
        "root": str(root),
        "yaml": str(yaml_path.resolve()),
        "yaml_sha256": calcular_sha256(yaml_path),
        "categories": [{"id": i, "name": name} for i, name in enumerate(names)],
        "images": rows,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "images": len(rows),
            "objects": sum(counts.values()),
            "classes_declared": len(names),
            "classes_observed": len(counts),
            "empty_images": sum((not r["annotations"] for r in rows)),
            "groups": len(group_sizes),
            "exact_duplicate_pairs": len(exact),
            "near_duplicate_pairs": len(near),
            "box_relative_area_quantiles": (
                np.quantile(relative_areas, [0, 0.25, 0.5, 0.75, 1]).tolist()
                if relative_areas
                else []
            ),
            "class_counts": [
                {"id": i, "name": n, "objects": counts[i], "images": image_counts[i]}
                for i, n in enumerate(names)
            ],
        },
        "exact_pairs": exact,
        "near_pairs": near,
        "orphan_labels": orphan,
        "valid": not errors,
        "source_files_signature": firma_archivos(root),
    }
    output.mkdir(parents=True, exist_ok=True)
    guardar_json(output / "audit.json", report)
    guardar_json(output / "catalog.json", report["categories"])
    with (output / "classes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "name", "objects", "images"])
        writer.writeheader()
        writer.writerows(report["summary"]["class_counts"])
    mostrar_muestra(report, output / "annotations.jpg")
    if near:
        by_id = {r["id"]: r for r in rows}
        pair_rows = [by_id[i] for pair in near[:12] for i in pair]
        mostrar_muestra(
            {**report, "images": pair_rows},
            output / "similarities.jpg",
            count=len(pair_rows),
        )
    with (output / "similarities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_a", "image_b", "hamming_distance"])
        by_id = {r["id"]: r for r in rows}
        for a, b in near:
            writer.writerow(
                [
                    by_id[a]["file_name"],
                    by_id[b]["file_name"],
                    (
                        int(by_id[a]["dhash"], 16) ^ int(by_id[b]["dhash"], 16)
                    ).bit_count(),
                ]
            )
    graficar_datos(report, output)
    lines = [
        "# Auditoría de datos",
        "",
        f"Válida: {report['valid']}",
        "",
        "```json",
        __import__("json").dumps(report["summary"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## Advertencias",
    ]
    lines += (
        [f"- {w}" for w in warnings] + ["", "## Errores"] + [f"- {e}" for e in errors]
    )
    (output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def depurar(source, output):
    """Reasociar sidecars inequívocos y excluir TODOS los duplicados contradictorios.

    Nunca se elige arbitrariamente una anotación ni se alteran archivos fuente.
    Otros tipos de errores siguen bloqueando el flujo.
    """
    import copy

    report = copy.deepcopy(source)
    root, output = (Path(source["root"]), Path(output))
    known = {r["file_name"] for r in report["images"]}
    recovered = []
    orphan_paths = {str((root / name).resolve()) for name in source["orphan_labels"]}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or "images" not in [
            p.lower() for p in path.parts
        ]:
            continue
        if path.relative_to(root).as_posix() in known:
            continue
        expected = ruta_etiqueta(path)
        if expected.exists():
            raise ValueError(
                f"Imagen inválida por una causa no reparable automáticamente: {path}"
            )
        candidates = [
            p
            for p in root.rglob(path.stem + ".txt")
            if str(p.resolve()) in orphan_paths
        ]
        if len(candidates) != 1:
            raise ValueError(f"No hay sidecar huérfano único para {path}")
        with Image.open(path) as raw:
            if raw.getexif().get(274, 1) != 1:
                raise ValueError("No se corrige automáticamente orientación EXIF")
            image = raw.convert("RGB")
            width, height = image.size
            small = np.asarray(image.convert("L").resize((9, 8)))
            dhash = int(
                "".join(
                    (
                        "1" if b else "0"
                        for b in (small[:, 1:] > small[:, :-1]).reshape(-1)
                    )
                ),
                2,
            )
            pixel_hash = hashlib.sha256(
                f"{width}x{height}".encode() + image.tobytes()
            ).hexdigest()
        identifier = max((r["id"] for r in report["images"])) + 1
        row = {
            "id": identifier,
            "file_name": path.relative_to(root).as_posix(),
            "width": width,
            "height": height,
            "sha256": calcular_sha256(path),
            "pixel_sha256": pixel_hash,
            "dhash": f"{dhash:016x}",
            "group_id": f"g{identifier:06d}",
            "annotations": leer_etiquetas(
                candidates[0], width, height, len(report["categories"])
            ),
        }
        related = {
            r["group_id"]
            for r in report["images"]
            if r["pixel_sha256"] == pixel_hash
            or (int(r["dhash"], 16) ^ dhash).bit_count() <= 4
        }
        if related:
            group = min(related | {row["group_id"]})
            for previous in report["images"]:
                if previous["group_id"] in related:
                    previous["group_id"] = group
            row["group_id"] = group
        report["images"].append(row)
        recovered.append(
            {
                "image": row["file_name"],
                "original_expected_label": expected.relative_to(root).as_posix(),
                "resolved_label": candidates[0].relative_to(root).as_posix(),
                "reason": "unique_orphan_sidecar_same_stem",
            }
        )
        orphan_paths.remove(str(candidates[0].resolve()))
    if orphan_paths:
        raise ValueError(f"Persisten etiquetas huérfanas no resueltas: {orphan_paths}")
    allowed_prefixes = ("Etiqueta ausente:", "Duplicados con etiquetas diferentes:")
    unexplained = [
        e
        for e in source["errors"]
        if not e.startswith(allowed_prefixes) and "etiquetas huérfanas" not in e
    ]
    if unexplained:
        raise ValueError(f"Errores que requieren revisión: {unexplained}")
    by_pixels = {}
    for row in report["images"]:
        by_pixels.setdefault(row["pixel_sha256"], []).append(row)
    excluded = []
    for same_image in by_pixels.values():
        signatures = {
            tuple(
                sorted(
                    (
                        (a["category_id"], tuple(np.round(a["bbox_xyxy"], 3)))
                        for a in r["annotations"]
                    )
                )
            )
            for r in same_image
        }
        if len(signatures) > 1:
            excluded.extend(
                (
                    {
                        "id": r["id"],
                        "file_name": r["file_name"],
                        "pixel_sha256": r["pixel_sha256"],
                        "annotations": r["annotations"],
                        "reason": "identical_pixels_conflicting_annotations",
                    }
                    for r in same_image
                )
            )
    excluded_ids = {r["id"] for r in excluded}
    report["images"] = [r for r in report["images"] if r["id"] not in excluded_ids]
    report["source_errors"] = source["errors"]
    report["errors"], report["orphan_labels"], report["valid"] = ([], [], True)
    report["curation"] = {
        "policy": "unique-sidecar reassociation; quarantine every conflicting-pixel duplicate",
        "recovered_labels": recovered,
        "excluded_images": excluded,
        "originals_modified": False,
    }
    report["warnings"] += [
        f"Curación explícita: {len(recovered)} sidecars reasociados y {len(excluded)} imágenes en cuarentena; originales intactos"
    ]
    report["exact_pairs"] = [
        p for p in source["exact_pairs"] if not set(p) & excluded_ids
    ]
    report["near_pairs"] = [
        p for p in source["near_pairs"] if not set(p) & excluded_ids
    ]
    counts = Counter(
        (a["category_id"] for r in report["images"] for a in r["annotations"])
    )
    image_counts = Counter(
        (
            c
            for r in report["images"]
            for c in {a["category_id"] for a in r["annotations"]}
        )
    )
    areas = [
        (a["bbox_xyxy"][2] - a["bbox_xyxy"][0])
        * (a["bbox_xyxy"][3] - a["bbox_xyxy"][1])
        / (r["width"] * r["height"])
        for r in report["images"]
        for a in r["annotations"]
    ]
    report["summary"] = {
        "images": len(report["images"]),
        "objects": sum(counts.values()),
        "classes_declared": len(report["categories"]),
        "classes_observed": len(counts),
        "empty_images": sum((not r["annotations"] for r in report["images"])),
        "groups": len({r["group_id"] for r in report["images"]}),
        "exact_duplicate_pairs": len(report["exact_pairs"]),
        "near_duplicate_pairs": len(report["near_pairs"]),
        "box_relative_area_quantiles": np.quantile(
            areas, [0, 0.25, 0.5, 0.75, 1]
        ).tolist(),
        "class_counts": [
            {**cat, "objects": counts[cat["id"]], "images": image_counts[cat["id"]]}
            for cat in report["categories"]
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    guardar_json(output / "audit.json", report)
    guardar_json(output / "curation.json", report["curation"])
    guardar_json(output / "catalog.json", report["categories"])
    mostrar_muestra(report, output / "annotations.jpg")
    graficar_datos(report, output)
    return report


def mostrar_muestra(report, destination, count=12):
    rows = report["images"]
    if not rows:
        return
    indices = np.linspace(0, len(rows) - 1, min(count, len(rows)), dtype=int)
    canvas = Image.new("RGB", (4 * 320, (len(indices) + 3) // 4 * 260), "#202020")
    for pos, index in enumerate(indices):
        row = rows[index]
        with Image.open(Path(report["root"]) / row["file_name"]) as raw:
            image = raw.convert("RGB")
        thumb = ImageOps.contain(image, (320, 230))
        draw = ImageDraw.Draw(thumb)
        sx, sy = (thumb.width / image.width, thumb.height / image.height)
        for ann in row["annotations"]:
            scaled = [v * scale for v, scale in zip(ann["bbox_xyxy"], [sx, sy, sx, sy])]
            draw.rectangle(scaled, outline="yellow", width=2)
            cat = report["categories"][ann["category_id"]]
            draw.text(
                tuple(scaled[:2]),
                f"{cat['id']}: {cat['name']}",
                fill="yellow",
                stroke_width=1,
                stroke_fill="black",
            )
        x, y = (pos % 4 * 320, pos // 4 * 260)
        canvas.paste(thumb, (x, y))
        ImageDraw.Draw(canvas).text(
            (x + 4, y + 235), row["file_name"][-45:], fill="white"
        )
    canvas.save(destination)
