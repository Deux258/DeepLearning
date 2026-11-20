"""Comprueba datos reales y un paso SFT con Phi-3 diminuto aleatorio en CPU.

Esta prueba técnica no entrena ni evalúa Phi-3-mini y no mide su rendimiento.
"""

import json
import math
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parent/'.runtime'/'hf'))

from comun import (OUT, MAX_PPO_PROMPT_TOKENS, chat_tokens, load_prompts, load_tokenizer,
                   prepare_partitions, prompt_key, sft_example)


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (DataCollatorForSeq2Seq, Phi3Config, Phi3ForCausalLM,
                              Trainer, TrainingArguments, set_seed)

    set_seed(42)
    torch.set_num_threads(2)
    tokenizer = load_tokenizer()
    folder = prepare_partitions(tokenizer)
    seen = {prompt_key(row['prompt']) for group in load_prompts().values() for row in group}
    expected = dict(sft=512, recompensa_train=1024, recompensa_val=128, ppo=128)
    counts = {}
    for part, count in expected.items():
        rows = [json.loads(line) for line in (folder/f'{part}.jsonl').read_text().splitlines()]
        assert len(rows) == count
        assert Counter(r['source'] for r in rows) == {'helpful-base': count//2, 'harmless-base': count//2}
        assert all(r['split'] == ('test' if part=='recompensa_val' else 'train') for r in rows)
        for row in rows:
            key = prompt_key(row['prompt'])
            assert key not in seen, 'Solapamiento entre entrenamiento, validación o evaluación.'
            seen.add(key)
            if part == 'sft':
                assert sft_example(tokenizer, row['prompt'], row['chosen']) is not None
            elif part == 'ppo':
                assert len(chat_tokens(tokenizer,[dict(role='user', content=row['prompt'])],generation=True)) <= MAX_PPO_PROMPT_TOKENS
                assert 'chosen' not in row and 'rejected' not in row
            else:
                assert row['chosen'] != row['rejected']
        counts[part] = count
    # Reproducibilidad de las particiones originales con la misma semilla y revisión.
    before = {p.name: p.read_bytes() for p in folder.glob('*.jsonl')}
    prepare_partitions(tokenizer)
    assert before == {p.name: p.read_bytes() for p in folder.glob('*.jsonl')}
    print('Particiones reales verificadas y reproducibles:', counts, flush=True)

    examples = [sft_example(tokenizer, 'Say hello.', 'Hello!'),
                sft_example(tokenizer, 'What is two plus two?', 'Four.')]
    assert all(examples)
    for example in examples:
        start = next(i for i, value in enumerate(example['labels']) if value != -100)
        assert start > 0
        assert example['labels'] == [-100]*start + example['input_ids'][start:]
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    batch = collator(examples)
    assert torch.all(batch['labels'][batch['attention_mask']==0] == -100)
    config = Phi3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                        max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
    model = get_peft_model(Phi3ForCausalLM(config), LoraConfig(task_type='CAUSAL_LM', r=8,
                           lora_alpha=16, lora_dropout=.05, target_modules=['qkv_proj','o_proj']))
    model.config.use_cache = False
    weight = next(p for name,p in model.named_parameters() if 'lora_B.' in name)
    previous = weight.detach().clone()
    target = OUT/'verificacion_tecnica'
    args = TrainingArguments(output_dir=str(target/'tiny'), use_cpu=True, max_steps=1,
                             per_device_train_batch_size=1, gradient_accumulation_steps=2,
                             learning_rate=5e-5, gradient_checkpointing=True,
                             gradient_checkpointing_kwargs={'use_reentrant':False},
                             save_strategy='no', report_to='none', disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=examples,
                      data_collator=collator, processing_class=tokenizer)
    result = trainer.train()
    changed = not torch.equal(previous, weight.detach())
    assert math.isfinite(result.training_loss) and changed
    trainer.save_model(str(target/'tiny'))
    # Verifica el adaptador guardado y una generación; tokens aleatorios sin interpretación.
    from peft import PeftModel
    reloaded = PeftModel.from_pretrained(Phi3ForCausalLM(config), target/'tiny').eval()
    assert torch.equal(weight.detach(), next(p for n,p in reloaded.named_parameters() if 'lora_B.' in n))
    prefix = torch.tensor([chat_tokens(tokenizer,[dict(role='user',content='Say hello.')],generation=True)])
    with torch.inference_mode():
        generated = reloaded.generate(prefix, attention_mask=torch.ones_like(prefix), max_new_tokens=2,
                                      do_sample=False, pad_token_id=tokenizer.pad_token_id)
    assert generated.shape[1] > prefix.shape[1]
    evidence = dict(particiones=counts, particiones_reproducibles=True, solapamientos_exactos=0,
                    sft_enmascaramiento=True, sft_padding_enmascarado=True,
                    prueba='Phi3Config de 1 capa con pesos aleatorios; no Phi-3-mini preentrenado',
                    perdida_prueba=result.training_loss, lora_actualizada=changed,
                    adaptador_recargado=True, generacion_tecnica=True,
                    gpu_disponible=torch.cuda.is_available(), entrenamiento_real_phi3='NO EJECUTADO')
    (target/'comprobaciones.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
