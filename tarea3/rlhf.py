"""Entrenamiento por preferencias humanas de Phi-3."""

import gc
import json
import math
import random
from collections import Counter
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic

import torch
from peft import PeftModel, get_peft_model_state_dict, prepare_model_for_kbit_training
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig, set_seed

from comun import (
    DATASET_REV, MAX_PPO_PROMPT_TOKENS, MODEL_REV, OUT, SEED, SFT_DIR, SOURCES,
    chat_tokens, load_model, load_prompts, load_tokenizer, prepare_partitions, prompt_key,
)


REWARD_ID = "distilbert/distilbert-base-uncased"
REWARD_REV = "12040accade4e8a0f71eabdb258fecc2e7e948be"
REWARD_DIR = "recompensa_distilbert"
PPO_DIR = "ppo_lr3e6"
PPO_BATCH = 4
PPO_LR = 3e-6
VALUE_LR = 1e-4
KL_BETA = 0.05
CLIP = 0.2
GAE_LAMBDA = 0.95


def output_name(name, pilot):
    return f"{name}_piloto" if pilot else name


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def signature(obj):
    return sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stage_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix=f".{path.name}-", dir=path.parent))


def read_partitions(pilot):
    folder = OUT / "particiones"
    quotas = {"sft": 256, "recompensa_train": 512, "recompensa_val": 64, "ppo": 64}
    paths = {key: folder / f"{key}.jsonl" for key in quotas}
    if not all(path.is_file() for path in paths.values()):
        print("Preparando particiones de HH-RLHF…", flush=True)
        prepare_partitions(load_tokenizer())

    used = {prompt_key(row["prompt"]) for group in load_prompts().values() for row in group}
    ids, hashes, selections = set(), {}, {}
    for name in quotas:
        hashes[name] = digest(paths[name])
        rows = [json.loads(line) for line in paths[name].read_text(encoding="utf-8").splitlines()]
        if Counter(row["source"] for row in rows) != dict.fromkeys(SOURCES, quotas[name]):
            raise ValueError(f"El tamaño o balance de {name} difiere de la partición esperada.")
        split = "test" if name == "recompensa_val" else "train"
        for row in rows:
            key = prompt_key(row["prompt"])
            if not key or key in used or row["id"] in ids or row["split"] != split:
                raise ValueError(f"Partición inválida o con solapamiento: {name}, {row['id']}")
            if name.startswith("recompensa_"):
                chosen, rejected = row["chosen"].strip(), row["rejected"].strip()
                if not chosen or not rejected or chosen == rejected:
                    raise ValueError(f"Par de preferencias inválido: {row['id']}")
            used.add(key)
            ids.add(row["id"])
        if pilot and name != "sft":
            quota = 16 if name == "recompensa_train" else (8 if name == "recompensa_val" else 2)
            by_source = [[row for row in rows if row["source"] == source][:quota] for source in SOURCES]
            rows = [row for group in by_source for row in group]
        selections[name] = rows
    return selections, hashes


def reward_inputs(tokenizer, prompts, responses):
    """DistilBERT recibe [CLS], prompt, [SEP], respuesta, [SEP]."""
    p_ids = tokenizer(prompts, add_special_tokens=False, verbose=False)["input_ids"]
    r_ids = tokenizer(responses, add_special_tokens=False, verbose=False)["input_ids"]
    encoded, shortened = [], []
    for prompt, answer in zip(p_ids, r_ids, strict=True):
        prefix = prompt[:128]
        budget = 512 - len(prefix) - 3
        ids = [tokenizer.cls_token_id, *prefix, tokenizer.sep_token_id,
               *answer[:budget], tokenizer.sep_token_id]
        encoded.append({"input_ids": ids})
        shortened.append((len(prompt) > 128, len(answer) > budget))
    return encoded, shortened


def reward_pairs(tokenizer, rows):
    encoded, shortened = reward_inputs(
        tokenizer,
        [row["prompt"] for row in rows for _ in range(2)],
        [row[key] for row in rows for key in ("chosen", "rejected")],
    )
    return list(zip(encoded[::2], encoded[1::2], strict=True)), {
        "pares_prompt_truncado": sum(flag[0] for flag in shortened[::2]),
        "preferidas_truncadas": sum(flag[1] for flag in shortened[::2]),
        "rechazadas_truncadas": sum(flag[1] for flag in shortened[1::2]),
    }


def score_pairs(model, tokenizer, pairs):
    batch = tokenizer.pad([entry for pair in pairs for entry in pair], return_tensors="pt")
    return model(**batch.to(model.device)).logits.float().reshape(-1, 2)


def preference_metrics(margins):
    return {
        "pares": len(margins),
        "accuracy": (margins > 0).float().mean().item(),
        "empates": (margins == 0).sum().item(),
        "loss": -F.logsigmoid(margins).mean().item(),
    }


@torch.no_grad()
def validate_reward(model, tokenizer, rows, pairs):
    model.eval()
    scores = torch.cat([score_pairs(model, tokenizer, pairs[i:i + 8]).cpu()
                        for i in range(0, len(pairs), 8)])
    margins = scores[:, 0] - scores[:, 1]
    by_source = {
        source: preference_metrics(margins[[r["source"] == source for r in rows]])
        for source in SOURCES
    }
    raw = [
        {"id": row["id"], "source": row["source"], "chosen": chosen,
         "rejected": rejected, "margen": chosen - rejected}
        for row, (chosen, rejected) in zip(rows, scores.tolist(), strict=True)
    ]
    return {**preference_metrics(margins), "por_fuente": by_source}, raw


def train_reward(parts, hashes, pilot):
    path = OUT / output_name(REWARD_DIR, pilot)
    config = {
        "version": 1, "modelo": REWARD_ID, "revision": REWARD_REV, "dataset": DATASET_REV,
        "piloto": pilot, "seed": SEED, "particiones": hashes, "lr": 2e-5, "weight_decay": 0.01,
        "epochs": 1 if pilot else 3, "pares_por_lote": 8, "acumulacion": 2,
        "max_prompt": 128, "max_total": 512,
    }
    if path.exists():
        meta = read_json(path / "reporte.json")
        if meta["config"] != config or meta["pesos_sha256"] != digest(path / "model.safetensors"):
            raise ValueError(f"La recompensa existente no corresponde a esta corrida: {path}")
        print(f"Reutilizando recompensa en {path}", flush=True)
        return path, meta

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DistilBERT en {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(REWARD_ID, revision=REWARD_REV)
    train_pairs, train_trunc = reward_pairs(tokenizer, parts["recompensa_train"])
    val_rows = parts["recompensa_val"]
    val_pairs, val_trunc = reward_pairs(tokenizer, val_rows)
    model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_ID, revision=REWARD_REV, num_labels=1, use_safetensors=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    rng = random.Random(SEED)
    history, best_state, best_key = [], None, (-math.inf, -math.inf)
    batch_size, accumulation = config["pares_por_lote"], config["acumulacion"]
    for epoch in range(1, config["epochs"] + 1):
        start = monotonic()
        model.train()
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(batches):
            with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
                scores = score_pairs(model, tokenizer, [train_pairs[j] for j in batch])
                loss = -F.logsigmoid(scores[:, 0] - scores[:, 1]).mean()
            total_loss += loss.detach().item() * len(batch)
            group = batches[i // accumulation * accumulation:(i // accumulation + 1) * accumulation]
            scaler.scale(loss * len(batch) / sum(map(len, group))).backward()
            if (i + 1) % accumulation == 0 or i + 1 == len(batches):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 16 == 0 or i + 1 == len(batches):
                print(f"Recompensa {epoch}/{config['epochs']} · lote {i + 1}/{len(batches)}", flush=True)
        metrics, _ = validate_reward(model, tokenizer, val_rows, val_pairs)
        train_loss = total_loss / len(order)
        if not all(math.isfinite(value) for value in (train_loss, metrics["loss"], metrics["accuracy"])):
            raise FloatingPointError("Pérdida o precisión no finita al entrenar la recompensa.")
        history.append({"epoca": epoch, "loss_train": train_loss, **metrics,
                        "segundos": round(monotonic() - start, 2)})
        key = (metrics["accuracy"], -metrics["loss"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {name: weight.detach().cpu().clone() for name, weight in model.state_dict().items()}
        print(f"Época {epoch} · loss {train_loss:.4f} · accuracy {metrics['accuracy']:.3f}", flush=True)

    model.load_state_dict(best_state)
    metrics, raw = validate_reward(model, tokenizer, val_rows, val_pairs)
    with torch.no_grad():
        scores = torch.cat([score_pairs(model, tokenizer, train_pairs[i:i + 8]).cpu().flatten()
                            for i in range(0, len(train_pairs), 8)])
    meta = {
        "config": config, "historial": history, "mejor_epoca": best_epoch,
        "validacion": metrics, "recortes": {"train": train_trunc, "val": val_trunc},
        "normalizacion": {"media": scores.mean().item(), "desv": scores.std(correction=0).item(),
                          "respuestas": scores.numel()},
    }
    stage = stage_output(path)
    model.save_pretrained(stage)
    tokenizer.save_pretrained(stage)
    meta["pesos_sha256"] = digest(stage / "model.safetensors")
    write_json(stage / "validacion.json", raw)
    write_json(stage / "reporte.json", meta)
    stage.replace(path)
    return path, meta


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gae(rewards, values):
    """Gamma=1 con bootstrap cero en el último token emitido."""
    advantage = 0.0
    next_value = 0.0
    advantages = torch.empty_like(values)
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + next_value - values[i]
        advantage = delta + GAE_LAMBDA * advantage
        advantages[i] = advantage
        next_value = values[i]
    return advantages, advantages + values


def ppo_losses(new_logp, old_logp, values, old_values, advantages, returns):
    ratio = (new_logp - old_logp).exp()
    policy = -torch.minimum(ratio * advantages, ratio.clamp(1 - CLIP, 1 + CLIP) * advantages)
    clipped_values = old_values + (values - old_values).clamp(-CLIP, CLIP)
    value = 0.5 * torch.maximum((values - returns).square(), (clipped_values - returns).square())
    return policy, value, (ratio - 1).abs() > CLIP


def action_states(hidden, prompt_length, action_count):
    # La posición anterior a cada token es la que predice esa acción.
    return hidden[:, prompt_length - 1:prompt_length - 1 + action_count]


def action_stats(base, critic, sequence, prompt_length):
    targets = sequence[:, prompt_length:]
    hidden = base.model(input_ids=sequence, use_cache=False).last_hidden_state
    head = base.get_output_embeddings()
    states = action_states(hidden, prompt_length, targets.shape[-1])
    logits = head(states.to(head.weight.dtype)).float()
    logp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).reshape(-1)
    return logp, critic(states.float()).reshape(-1)


def check_ppo_math():
    rewards = torch.tensor([0.0, 1.0])
    advantages, returns = gae(rewards, torch.tensor([0.2, 0.3]))
    torch.testing.assert_close(advantages, torch.tensor([0.765, 0.7]))
    torch.testing.assert_close(returns, torch.tensor([0.965, 1.0]))
    zero = torch.zeros(2)
    policy, value, clipped = ppo_losses(
        torch.tensor([1.5, 0.5]).log(), zero, torch.tensor([0.5, -0.5]), zero,
        torch.tensor([1.0, -1.0]), torch.tensor([0.5, -0.5]),
    )
    torch.testing.assert_close(policy, torch.tensor([-1.2, 0.8]))
    torch.testing.assert_close(value, torch.tensor([0.045, 0.045]))
    assert clipped.all()
    assert torch.equal(action_states(torch.arange(5).view(1, -1), 3, 2), torch.tensor([[2, 3]]))


def balanced_batches(rows):
    rng = random.Random(SEED)
    helpful, harmless = [[row for row in rows if row["source"] == source] for source in SOURCES]
    rng.shuffle(helpful)
    rng.shuffle(harmless)
    half = PPO_BATCH // 2
    for offset in range(0, len(helpful), half):
        batch = helpful[offset:offset + half] + harmless[offset:offset + half]
        rng.shuffle(batch)
        yield batch


def save_ppo_checkpoint(path, config_hash, model, critic, optimizer, history):
    step = len(history)
    target = path / f"{step:03d}"
    stage = stage_output(target)
    model.save_pretrained(stage, selected_adapters=["default"])
    torch.save({
        "critic": critic.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
        "gpu_rng": torch.cuda.get_rng_state(), "python_rng": random.getstate(),
    }, stage / "estado.pt")
    write_json(stage / "progreso.json", {"firma": config_hash, "siguiente": step, "historial": history})
    stage.replace(target)


def load_ppo_checkpoint(path, config_hash):
    candidates = sorted(path.glob("[0-9][0-9][0-9]"), reverse=True)
    if not candidates:
        return None, []
    for candidate in candidates:
        if all((candidate / filename).is_file()
               for filename in ("progreso.json", "estado.pt", "adapter_model.safetensors", "adapter_config.json")):
            info = read_json(candidate / "progreso.json")
            if info["firma"] == config_hash and info["siguiente"] == len(info["historial"]) == int(candidate.name):
                return candidate, info["historial"]
    raise ValueError(f"No hay un checkpoint compatible en {path}. Usa otra carpeta para una corrida distinta.")


@torch.no_grad()
def collect_rollouts(batch, model, critic, tokenizer, reward_model, reward_tokenizer, generation, stats):
    model.eval()
    base = model.get_base_model()
    prompts = [chat_tokens(tokenizer, [{"role": "user", "content": row["prompt"]}], generation=True)
               for row in batch]
    width = max(map(len, prompts))
    if width > MAX_PPO_PROMPT_TOKENS:
        raise ValueError("El lote contiene un prompt de PPO fuera del límite.")
    pads = [width - len(prompt) for prompt in prompts]
    inputs = torch.tensor([[generation.pad_token_id] * pad + prompt
                           for pad, prompt in zip(pads, prompts, strict=True)], device=model.device)
    mask = torch.tensor([[0] * pad + [1] * len(prompt)
                         for pad, prompt in zip(pads, prompts, strict=True)], device=model.device)
    generated = model.generate(input_ids=inputs, attention_mask=mask, generation_config=generation)
    eos = generation.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or [])
    rollouts = []
    for row, prompt, tokens in zip(batch, prompts, generated[:, width:].tolist(), strict=True):
        end = next((i + 1 for i, token in enumerate(tokens) if token in stop_ids), len(tokens))
        if not end:
            raise RuntimeError("La política no emitió ninguna acción.")
        actions = tokens[:end]
        sequence = torch.tensor([prompt + actions], device=model.device)
        answer = tokenizer.decode(actions, skip_special_tokens=True).strip()
        encoded, shortened = reward_inputs(reward_tokenizer, [row["prompt"]], [answer])
        reward = reward_model(**reward_tokenizer.pad(encoded, return_tensors="pt")).logits.item()
        if not math.isfinite(reward):
            raise FloatingPointError("El modelo de recompensa produjo un puntaje no finito.")
        normalized = max(-5.0, min(5.0, (reward - stats["media"]) / max(stats["desv"], 1e-6)))
        old_logp, old_values = action_stats(base, critic, sequence, len(prompt))
        model.set_adapter("referencia", inference_mode=True)
        try:
            ref_logp, _ = action_stats(base, critic, sequence, len(prompt))
        finally:
            model.set_adapter("default")
        kl = old_logp - ref_logp
        rewards = -KL_BETA * kl
        rewards[-1] += normalized
        advantages, returns = gae(rewards, old_values)
        rollouts.append({
            "sequence": sequence, "prompt_length": len(prompt), "old_logp": old_logp,
            "old_values": old_values, "advantages": advantages, "returns": returns, "kl": kl,
            "reward": reward, "normalized": normalized, "blank": not answer, "truncated": shortened[0],
        })

    advantages_all = torch.cat([row["advantages"] for row in rollouts])
    mean, std = advantages_all.mean(), advantages_all.std(correction=0).clamp_min(1e-8)
    for row in rollouts:
        row["advantages"] = (row["advantages"] - mean) / std
    return rollouts


def update_policy(model, critic, optimizer, rollouts, parameters, epochs):
    model.train()
    base = model.get_base_model()
    actions = sum(row["old_logp"].numel() for row in rollouts)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=model.device)
        ratio_error = 0.0
        for row in rollouts:
            new_logp, values = action_stats(base, critic, row["sequence"], row["prompt_length"])
            policy, value, clipped = ppo_losses(
                new_logp, row["old_logp"], values, row["old_values"], row["advantages"], row["returns"],
            )
            if epoch == 0:
                ratio_error = max(ratio_error, ((new_logp.detach() - row["old_logp"]).exp() - 1).abs().max().item())
            policy_sum, value_sum = policy.sum(), value.sum()
            ((policy_sum + 0.5 * value_sum) / actions).backward()
            totals += torch.stack((policy_sum.detach(), value_sum.detach(), clipped.sum()))

        if epoch == 0 and ratio_error > 0.02:
            raise RuntimeError(f"El ratio anterior a la primera actualización se aleja de uno: {ratio_error:.4g}")
        norm = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        policy_loss, value_loss, clip_fraction = (totals / actions).tolist()
        history.append({"epoca_ppo": epoch + 1, "loss_politica": policy_loss, "loss_valor": value_loss,
                        "fraccion_clip": clip_fraction, "norma_grad": norm.item(),
                        "error_ratio_inicial": ratio_error if epoch == 0 else None})
    optimizer.zero_grad(set_to_none=True)
    return history


def train_ppo(parts, hashes, reward_path, reward_meta, pilot):
    check_ppo_math()
    stats = reward_meta["normalizacion"]
    if reward_meta["validacion"]["accuracy"] <= 0.5 or stats["desv"] <= 1e-6:
        raise ValueError("La recompensa debe superar 0,5 de accuracy y discriminar antes de entrenar PPO.")

    sft = OUT / SFT_DIR
    sft_hash = {name: digest(sft / name) for name in ("adapter_model.safetensors", "adapter_config.json")}
    path = OUT / output_name(PPO_DIR, pilot)
    config = {
        "version": 2, "modelo_revision": MODEL_REV, "particiones": hashes, "seed": SEED,
        "piloto": pilot, "sft": sft.name, "sft_sha256": sft_hash,
        "recompensa": reward_path.name, "recompensa_sha256": reward_meta["pesos_sha256"],
        "normalizacion": stats, "prompts": len(parts["ppo"]), "max_prompt": MAX_PPO_PROMPT_TOKENS,
        "max_response": 32 if pilot else 64, "gamma": 1.0, "lambda": GAE_LAMBDA,
        "precision": "NF4/FP16", "lr_politica": PPO_LR, "lr_valor": VALUE_LR, "clip": CLIP,
        "beta_kl": KL_BETA, "coef_valor": 0.5, "max_grad_norm": 1.0, "batch": PPO_BATCH, "epochs": 2,
        "checkpoint_cada": 1 if pilot else 8,
    }
    config_hash = signature(config)
    if path.exists():
        meta = read_json(path / "reporte.json")
        if meta["firma"] != config_hash or meta["pesos_sha256"] != digest(path / "adapter_model.safetensors"):
            raise ValueError(f"El adaptador existente no corresponde a esta corrida: {path}")
        print(f"Reutilizando adaptador PPO completo en {path}", flush=True)
        return path, meta

    checkpoints = OUT / f"{path.name}_checkpoints" / config_hash
    checkpoint, history = load_ppo_checkpoint(checkpoints, config_hash)
    set_seed(SEED)
    tokenizer = load_tokenizer()
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_path).eval().requires_grad_(False)
    base = prepare_model_for_kbit_training(
        load_model(), use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(base, str(checkpoint or sft), is_trainable=True)
    model.load_adapter(str(sft), adapter_name="referencia", is_trainable=False)
    model.set_adapter("default")
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    generation = GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=config["max_response"],
        pad_token_id=tokenizer.pad_token_id, eos_token_id=model.generation_config.eos_token_id,
        use_cache=True,
    )

    critic = nn.Linear(model.config.hidden_size, 1, device=model.device)
    nn.init.zeros_(critic.weight)
    nn.init.zeros_(critic.bias)
    policy_params = [param for param in model.parameters() if param.requires_grad]
    critic_params = list(critic.parameters())
    if not policy_params or any(param.requires_grad and ".default." not in name
                                for name, param in model.named_parameters()):
        raise RuntimeError("Solo el adaptador de política debe tener gradientes en Phi-3.")
    optimizer = torch.optim.AdamW([
        {"params": policy_params, "lr": PPO_LR}, {"params": critic_params, "lr": VALUE_LR},
    ], weight_decay=0.0)
    if checkpoint:
        state = torch.load(checkpoint / "estado.pt", weights_only=True, map_location="cpu")
        critic.load_state_dict(state["critic"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["gpu_rng"])
        random.setstate(state["python_rng"])
        del state
        print(f"PPO retoma desde el lote {len(history)}.", flush=True)

    batches = list(balanced_batches(parts["ppo"]))
    for step, batch in enumerate(batches[len(history):], start=len(history) + 1):
        start = monotonic()
        rollouts = collect_rollouts(batch, model, critic, tokenizer, reward_model, reward_tokenizer, generation, stats)
        kl = torch.cat([row["kl"] for row in rollouts])
        if step == 1 and not torch.allclose(kl, torch.zeros_like(kl), atol=2e-3, rtol=0):
            raise RuntimeError("La política inicial y la referencia SFT discrepan.")
        updates = update_policy(model, critic, optimizer, rollouts, policy_params + critic_params, config["epochs"])
        report = {
            "lote": step, "prompts": [row["id"] for row in batch], "kl_por_token": kl.mean().item(),
            "kl_inicial_max": kl.abs().max().item() if step == 1 else None,
            "recompensa": sum(row["reward"] for row in rollouts) / len(rollouts),
            "recompensa_normalizada": sum(row["normalized"] for row in rollouts) / len(rollouts),
            "longitud_media": len(kl) / len(rollouts), "vacias": sum(row["blank"] for row in rollouts),
            "recorte_recompensa_prompt": sum(row["truncated"][0] for row in rollouts),
            "recorte_recompensa_respuesta": sum(row["truncated"][1] for row in rollouts),
            "segundos": round(monotonic() - start, 2), "actualizaciones": updates,
        }
        history.append(report)
        print(f"PPO {step}/{len(batches)} · recompensa {report['recompensa']:.3f} · "
              f"KL {report['kl_por_token']:.4f} · {report['segundos']:.0f} s", flush=True)
        if step % config["checkpoint_cada"] == 0 or step == len(batches):
            save_ppo_checkpoint(checkpoints, config_hash, model, critic, optimizer, history)

    original = load_file(sft / "adapter_model.safetensors")
    reference = get_peft_model_state_dict(model, adapter_name="referencia")
    policy = get_peft_model_state_dict(model, adapter_name="default")
    if original.keys() != reference.keys() or not all(
        torch.equal(reference[name].cpu(), original[name]) for name in original
    ):
        raise RuntimeError("La referencia SFT cambió durante PPO.")
    if not any(not torch.equal(policy[name].cpu(), original[name]) for name in original) or not any(
        torch.count_nonzero(param).item() for param in critic_params
    ):
        raise RuntimeError("PPO no actualizó la política y la cabeza de valor.")
    assert len(history) == len(batches)
    meta = {"config": config, "firma": config_hash, "historial": history, "referencia_intacta": True,
            "politica_actualizada": True, "valor_actualizado": True}
    stage = stage_output(path)
    model.save_pretrained(stage, selected_adapters=["default"])
    meta["pesos_sha256"] = digest(stage / "adapter_model.safetensors")
    write_json(stage / "reporte.json", meta)
    stage.replace(path)
    return path, meta
