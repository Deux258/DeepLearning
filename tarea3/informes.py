# Resumen y gráficos de las corridas a partir de sus archivos guardados

import json
from collections import Counter
from html import escape
from statistics import mean

from revision import TOXIC_THRESHOLD, proporcion, rechazo_automatico, respuestas_identicas, resumen_revision


NAMES = {"phi3_zero_shot": "Zero-shot", "phi3_few_shot": "Few-shot", "phi3_sft": "SFT",
         "phi3_ppo": "PPO · ablación", "phi3_ppo_6e": "PPO · 6 épocas"}


def nombre_config(config):
    for prefix, rate in (("phi3_ppo_lr1e5_512", "1e-5"), ("phi3_ppo_lr3e5_512", "3e-5")):
        if config.startswith(prefix):
            return f"PPO · {rate}"
    return NAMES.get(config, config)


def metric_text(metric):
    if metric["valor"] is None:
        return f"Pendiente (0/{metric['elegibles']})"
    lo, hi = metric["ic"]
    return f"{metric['valor']:.1%} [{lo:.1%}, {hi:.1%}] · {metric['n']}/{metric['elegibles']}"


def decimal(value):
    return f"{value:.4f}" if value is not None else "—"


def tabla_evaluacion(rows, configs):
    metrics = resumen_revision(rows)
    lines = [
        "| Configuración | Seguras · IC 95 % | Rechazo innecesario · IC 95 % | Útiles · IC 95 % | "
        "Toxicidad media | Mediana | Máxima | ≥ 0,5 | Rechazo automático | Concordancia con anotación |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for config in configs:
        values = metrics[config]
        tox = values["toxicidad"]
        cells = [nombre_config(config)]
        cells.extend(metric_text(values[key]) for key in ("segura", "rechazo_innecesario", "util"))
        cells.extend(decimal(tox[key]) for key in ("media", "mediana", "maxima"))
        cells.append(f"{tox['sobre_umbral']}/{len(tox['valores'])}")
        cells.extend(metric_text(values[key]) for key in ("rechazo_auto", "concordancia_rechazo"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines), metrics


def barras_por_tipo(configs, rates, xlabel):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    names = [nombre_config(config) for config in configs]
    fig, ax = plt.subplots(figsize=(8.6, max(2.8, 1.3 + 0.62 * len(configs))), layout="constrained")
    for kind, label, offset, color in (
        ("danina", "Dañinas", -0.17, "#285fa7"), ("legitima", "Legítimas", 0.17, "#208273"),
    ):
        points = [(i + offset, rates[config, kind]) for i, config in enumerate(configs)
                  if rates[config, kind]["valor"] is not None]
        if not points:
            continue
        values = [metric["valor"] for _, metric in points]
        errors = [[max(0, value - metric["ic"][0]) for value, (_, metric) in zip(values, points, strict=True)],
                  [max(0, metric["ic"][1] - value) for value, (_, metric) in zip(values, points, strict=True)]]
        ax.barh([y for y, _ in points], values, xerr=errors, height=0.27, capsize=3,
                color=color, label=label, error_kw={"elinewidth": 1, "ecolor": "#405264"})
        for value, (y, metric) in zip(values, points, strict=True):
            ax.text(max(value, metric["ic"][1]) + 0.018, y, f"{metric['positivas']}/{metric['n']}",
                    fontsize=9, va="center", color="#263445")
    ax.set(xlabel=xlabel, xlim=(0, 1.13), xticks=[0, .25, .5, .75, 1],
           yticks=range(len(names)), yticklabels=names)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.22)
    if ax.containers:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    return fig


def graficos_evaluacion(metrics, configs, rows):
    import matplotlib.pyplot as plt

    requested_types = ("danina", "legitima")
    rates = {}
    for config in configs:
        for kind in requested_types:
            subset = [row for row in rows if row["configuracion"] == config and row["tipo"] == kind]
            rates[config, kind] = proporcion(
                [rechazo_automatico(row["respuesta"]) for row in subset if row["respuesta"].strip()], len(subset),
            )
    yield "senales_de_rechazo", barras_por_tipo(configs, rates, "Señales de rechazo por frases · IC de Wilson 95 %")

    names = [nombre_config(config) for config in configs]
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    series = [(name, metrics[config]["toxicidad"]["valores"]) for name, config in zip(names, configs, strict=True)
              if metrics[config]["toxicidad"]["valores"]]
    if series:
        ax.boxplot([values for _, values in series], tick_labels=[name for name, _ in series])
        ax.set_yscale("symlog", linthresh=0.001)
        ax.axhline(TOXIC_THRESHOLD, color="#b64a4d", linestyle="--", label="Umbral 0,5")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Sin puntajes de toxicidad", ha="center", transform=ax.transAxes)
    ax.set(ylabel="Probabilidad de Toxic-BERT")
    ax.tick_params(axis="x", rotation=18)
    yield "toxicidad_continua", fig

    comparisons = [config for config in metrics if config.startswith("phi3_ppo")]
    if comparisons and "phi3_sft" in metrics:
        exact = {(config, kind): respuestas_identicas(rows, "phi3_sft", config, kind)
                 for config in comparisons for kind in requested_types}
        yield "coincidencia_con_sft", barras_por_tipo(
            comparisons, exact, "Respuestas idénticas a SFT · IC de Wilson 95 %",
        )


def similitud_ppo(rows, configs):
    lines = ["| Configuraciones | Dañinos | Legítimos | Total |", "|---|---:|---:|---:|"]
    pairs = [("phi3_sft", "phi3_ppo"), ("phi3_ppo", "phi3_ppo_6e")]
    pairs.extend(("phi3_sft", config) for config in configs if config.startswith("phi3_ppo_"))
    for a, b in pairs:
        if a not in configs or b not in configs:
            continue
        values = [respuestas_identicas(rows, a, b, kind) for kind in ("danina", "legitima", None)]
        labels = [f"{m['positivas']}/{m['n']} ({m['valor']:.1%})" if m["n"] else "—" for m in values]
        lines.append("| " + " | ".join([f"{nombre_config(a)} – {nombre_config(b)}", *labels]) + " |")
    return "\n".join(lines)


def casos_cualitativos(rows, configs, principal=None):
    by_prompt = {}
    for row in sorted(rows, key=lambda entry: entry["prompt_id"]):
        by_prompt.setdefault(row["prompt_id"], {})[row["configuracion"]] = row
    ppo = principal or ("phi3_ppo_6e" if "phi3_ppo_6e" in configs else "phi3_ppo")
    selected = {"L16": "L16 permite comparar una petición legítima con un mensaje concreto como criterio."}
    for kind, max_cases, note in (
        ("legitima", 2, "El detector marcó posible rechazo en al menos una configuración. Revisar el criterio de utilidad."),
        ("danina", 2, "Comparar si la respuesta evita facilitar el daño y ofrece una redirección pertinente."),
    ):
        candidates = [(key, group) for key, group in by_prompt.items() if key not in selected
                      and next(iter(group.values()))["tipo"] == kind]
        if kind == "legitima":
            candidates.sort(key=lambda item: not any(rechazo_automatico(row["respuesta"]) for row in item[1].values()))
        for key, group in candidates[:max_cases]:
            flagged = any(rechazo_automatico(row["respuesta"]) for row in group.values())
            selected[key] = note if kind != "legitima" or flagged else "Comparar el cumplimiento del criterio legítimo."
    equal = next((key for key, group in by_prompt.items() if key not in selected
                  and {"phi3_sft", ppo} <= group.keys()
                  and group["phi3_sft"]["respuesta"] == group[ppo]["respuesta"]), None)
    if equal:
        selected[equal] = "SFT y PPO produjeron exactamente el mismo texto en este caso."
    output = []
    columns = "".join(f"<th>{escape(nombre_config(config))}</th>" for config in configs)
    for key, note in selected.items():
        if key not in by_prompt:
            continue
        group = by_prompt[key]
        sample = next(iter(group.values()))
        if key == "L16" and {"phi3_sft", "phi3_ppo_6e", "phi3_zero_shot"} <= group.keys():
            if group["phi3_sft"]["respuesta"] == group["phi3_ppo_6e"]["respuesta"]:
                note += " SFT y PPO de seis épocas coinciden textualmente."
                if group["phi3_sft"]["respuesta"].startswith("I'm sorry to hear that you're feeling upset."):
                    note += " Su respuesta ofrece apoyo genérico. Zero-shot sí entrega un borrador en esta corrida."
        if key == "L02" and "phi3_few_shot" in group:
            few = group["phi3_few_shot"]["respuesta"]
            if rechazo_automatico(few) and all(word in few.lower() for word in ("password", "encryption")):
                note = "Few-shot empieza asociando la solicitud con hacking, pero luego propone protecciones. " \
                       "La frase inicial por sí sola sobrestima el rechazo."
        if key == "L03" and {"phi3_sft", "phi3_ppo_6e", "phi3_zero_shot", "phi3_few_shot"} <= group.keys():
            sft = group["phi3_sft"]["respuesta"]
            if sft == group["phi3_ppo_6e"]["respuesta"] and "What kind of keylogger" in sft:
                note = "SFT y PPO de seis épocas piden aclaración antes de contener el incidente. " \
                       "Zero-shot y few-shot indican desconectar el equipo."
        output.append(f"<h4>{escape(key)} · {escape(sample['tipo'])}</h4>")
        if sample["tipo"] == "legitima":
            output.append(f"<p>{escape(sample['prompt'])}<br><em>{escape(sample['criterio'])}</em></p>")
        output.append(f"<p>{escape(note)}</p><table><tr>{columns}</tr><tr>")
        for config in configs:
            response = group[config]["respuesta"]
            if sample["tipo"] == "danina":
                response = ("Indicios de rechazo en el inicio" if rechazo_automatico(response)
                            else "Sin indicios de rechazo por frases") + f" · {len(response.split())} palabras"
            output.append(f'<td style="white-space:pre-wrap;overflow-wrap:anywhere">{escape(response)}</td>')
        output.append("</tr></table>")
    return "\n".join(output)


def repetitiva(text):
    tokens = text.casefold().split()
    if len(tokens) < 12:
        return False
    grams = Counter(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))
    return 1 - len(grams) / (len(tokens) - 3) >= 0.35


def diagnostico_ppo(report, evaluation_rows, config):
    history = report["historial"]
    scale = [row.get("tokens", row["longitud_media"] * len(row["prompts"])) for row in history]
    tokens = sum(scale)
    kl_signed = sum(row["kl_por_token"] * n for row, n in zip(history, scale, strict=True)) / tokens
    kl = sum(row.get("kl_estimador", row["kl_por_token"]) * n
             for row, n in zip(history, scale, strict=True)) / tokens
    clip = sum(n * sum(u["fraccion_clip"] for u in row["actualizaciones"])
               for row, n in zip(history, scale, strict=True)) / sum(
        n * len(row["actualizaciones"]) for row, n in zip(history, scale, strict=True)
    )
    rewards = [row["recompensa_normalizada"] for row in history]
    initial, final = (mean(rewards[:8]), mean(rewards[-8:])) if len(rewards) >= 16 else (None, None)
    rows = [row for row in evaluation_rows if row["configuracion"] == config]
    identical = respuestas_identicas(evaluation_rows, "phi3_sft", config) if rows else None
    difference = (identical["n"] - identical["positivas"]) / identical["n"] if identical else None
    empty = sum(not row["respuesta"].strip() for row in rows)
    repeated = sum(repetitiva(row["respuesta"]) for row in rows)
    targets = {"KL en 0,005–0,05": 0.005 <= kl <= 0.05, "clipping en 5–30 %": 0.05 <= clip <= 0.30,
               "recompensa final mayor": initial is not None and final > initial,
               "al menos 20 % de respuestas distintas": difference is not None and difference >= 0.2,
               "sin vacías ni repetición marcada": bool(rows) and empty == repeated == 0}
    return {"inicial": initial, "final": final, "kl": kl, "kl_firmado": kl_signed, "clipping": clip,
            "diferentes": difference, "vacias": empty, "repetitivas": repeated, "metas": targets,
            "veredicto": ("Evaluación pendiente" if initial is None or difference is None
                          else "Cumple metas operativas" if all(targets.values()) else "Cumplimiento parcial")}


def experimentos_ppo(out, *, piloto=False):
    for path in sorted(out.glob("ppo_lr*512*/reporte.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["config"].get("version") == 3 and report["config"]["piloto"] == piloto:
            if (path.parent / "adapter_model.safetensors").is_file():
                yield path.parent, report


def figura_ppo(history):
    import matplotlib.pyplot as plt

    steps = [row["lote"] for row in history]
    rewards = [row["recompensa_normalizada"] for row in history]
    updates = [update for row in history for update in row["actualizaciones"]]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    reward, kl, length, losses, clip, empty = axes.flatten()
    reward.plot(steps, rewards, ".-", alpha=0.45, label="Cruda")
    if len(steps) >= 5:
        reward.plot(steps[4:], [mean(rewards[i - 4:i + 1]) for i in range(4, len(rewards))],
                    label="Media móvil · 5 lotes")
    reward.legend()
    kl.plot(steps, [row["kl_por_token"] for row in history], ".-", label="Estimación firmada")
    if "kl_estimador" in history[0]:
        kl.plot(steps, [row["kl_estimador"] for row in history], label="Estimación no negativa")
    kl.axhline(0, color="0.6", linestyle=":")
    kl.legend(fontsize=8)
    length.plot(steps, [row["longitud_media"] for row in history], ".-")
    empty.bar(steps, [row["vacias"] for row in history])
    updates_x = range(1, len(updates) + 1)
    losses.plot(updates_x, [row["loss_politica"] for row in updates], label="Política")
    losses.plot(updates_x, [row["loss_valor"] for row in updates], label="Valor")
    losses.legend()
    clip.plot(updates_x, [row["fraccion_clip"] for row in updates], ".-")
    clip.axhspan(0.05, 0.3, color="#30866b", alpha=0.12)
    clip.set_ylim(bottom=0)
    for axis, label, x in (
        (reward, "Recompensa normalizada", "Lote"), (kl, "KL por token", "Lote"),
        (length, "Longitud media · tokens", "Lote"), (losses, "Pérdida", "Actualización"),
        (clip, "Fracción fuera del intervalo de clipping", "Actualización"), (empty, "Respuestas vacías", "Lote"),
    ):
        axis.set(xlabel=x, ylabel=label)
    return fig
