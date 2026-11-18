"""Datos y configuración compartidos por los cuadernos de la Tarea 3."""

import gzip
import json
import os
import random
import re
import sys
from collections import Counter
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "salidas"
SEED = 42
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MODEL_REV = "f39ac1d28e925b323eae81227eaba4464caced4e"
DATASET_ID = "Anthropic/hh-rlhf"
DATASET_REV = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
SOURCES = ("helpful-base", "harmless-base")
MAX_SFT_TOKENS = 256
MAX_PPO_PROMPT_TOKENS = 128
MAX_EVAL_TOKENS = 256
SFT_LR = 5e-5
SFT_DIR = "sft_lr5e5"
TURN = re.compile(r"\n\n(Human|Assistant):")


def preparar_notebook():
    missing = [name for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes")
               if find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"El kernel {sys.executable} no tiene: {', '.join(missing)}. "
            "En VS Code, selecciona el kernel 'Tarea 3 (GPU)' o, en Python Environments, "
            f"el intérprete {ROOT / '.runtime' / 'rocm' / 'Scripts' / 'python.exe'}. "
            "Para preparar este equipo por primera vez, sigue la sección Entorno del README."
        )
    os.environ.setdefault("HF_HOME", str(ROOT / ".runtime" / "hf"))
    print(f"Python del cuaderno: {sys.executable}")


def prompt_key(text):
    return " ".join(text.casefold().split())


def load_prompts():
    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))
    rows = []
    for name, count in (("final", 30), ("piloto", 4), ("few_shot", 2)):
        group = prompts[name]
        assert Counter(row["tipo"] for row in group) == {"danina": count, "legitima": count}
        for row in group:
            assert row["prompt_id"] and row["prompt"].strip()
            if name == "few_shot":
                assert row["respuesta"].strip()
            elif row["tipo"] == "legitima":
                assert row["criterio"].strip()
        rows.extend(group)
    assert len({row["prompt_id"] for row in rows}) == len(rows)
    assert len({prompt_key(row["prompt"]) for row in rows}) == len(rows)
    return prompts


def single_turn(text):
    prefix, *turns = TURN.split(text)
    if len(turns) == 4 and not prefix.strip() and turns[0::2] == ["Human", "Assistant"]:
        user, assistant = turns[1].strip(), turns[3].strip()
        if user and assistant:
            return user, assistant
    return None


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def chat_tokens(tokenizer, messages, *, generation=False):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=generation,
    )


def sft_example(tokenizer, prompt, chosen):
    user = {"role": "user", "content": prompt}
    prefix = chat_tokens(tokenizer, [user], generation=True)
    full = chat_tokens(tokenizer, [user, {"role": "assistant", "content": chosen}])
    if len(full) > MAX_SFT_TOKENS or len(full) <= len(prefix) or full[:len(prefix)] != prefix:
        return None
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * len(prefix) + full[len(prefix):],
    }


def prepare_partitions(tokenizer):
    from huggingface_hub import hf_hub_download

    prompts = load_prompts()
    seen = {prompt_key(row["prompt"]) for group in prompts.values() for row in group}
    rng = random.Random(SEED)
    parts = {name: [] for name in ("sft", "recompensa_train", "recompensa_val", "ppo")}

    # Reservar la validación primero evita que un prompt repetido entre en entrenamiento.
    for part, split, quota in (
        ("recompensa_val", "test", 64), ("sft", "train", 256),
        ("recompensa_train", "train", 512), ("ppo", "train", 64),
    ):
        for source in SOURCES:
            filename = hf_hub_download(
                DATASET_ID, f"{source}/{split}.jsonl.gz", repo_type="dataset", revision=DATASET_REV,
            )
            with gzip.open(filename, "rt", encoding="utf-8") as stream:
                lines = list(enumerate(stream, start=1))
            rng.shuffle(lines)
            selected = 0
            for number, line in lines:
                pair = json.loads(line)
                chosen = single_turn(pair["chosen"])
                rejected = single_turn(pair["rejected"])
                if not chosen or not rejected or chosen[0] != rejected[0] or chosen[1] == rejected[1]:
                    continue
                prompt, response = chosen
                key = prompt_key(prompt)
                if key in seen or (part == "sft" and sft_example(tokenizer, prompt, response) is None):
                    continue
                if part == "ppo":
                    messages = [{"role": "user", "content": prompt}]
                    if len(chat_tokens(tokenizer, messages, generation=True)) > MAX_PPO_PROMPT_TOKENS:
                        continue
                row = {
                    "id": f"{source}/{split}/{number}", "source": source, "split": split, "prompt": prompt,
                }
                if part != "ppo":
                    row["chosen"] = response
                if part.startswith("recompensa_"):
                    row["rejected"] = rejected[1]
                parts[part].append(row)
                seen.add(key)
                selected += 1
                if selected == quota:
                    break
            if selected != quota:
                raise ValueError(f"{part}, {source}: se encontraron {selected} de {quota} filas válidas")

    folder = OUT / "particiones"
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in parts.items():
        rng.shuffle(rows)
        target = folder / f"{name}.jsonl"
        temporary = target.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        temporary.replace(target)
    return folder


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} no detecta una GPU en {sys.executable}. "
            "Selecciona el kernel 'Tarea 3 (GPU)' con PyTorch ROCm/CUDA o activa una GPU en Colab."
        )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, quantization_config=quantization,
        dtype=torch.float16, device_map={"": 0}, attn_implementation="eager",
    )
