"""Fine-tune and evaluate an extractive QA model on SQuAD 2.0.

Example: python train.py --train-limit 12000 --eval-limit 3000 --epochs 2
Use --train-limit 0 and --eval-limit 0 for the full official splits.
"""
import argparse
import csv
import json
import math
import random
import re
import string
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForQuestionAnswering, AutoTokenizer,
                          Trainer, TrainingArguments, default_data_collator)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='distilbert/distilbert-base-uncased')
    p.add_argument('--output-dir', default='weights/squad2-distilbert')
    p.add_argument('--log-dir', default='logs')
    p.add_argument('--train-limit', type=int, default=12000)
    p.add_argument('--eval-limit', type=int, default=3000)
    p.add_argument('--epochs', type=float, default=2)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def limited(ds, n, seed):
    if not n or n >= len(ds):
        return ds
    # Keep unanswerable examples close to their natural proportion while shuffling.
    rng = random.Random(seed)
    no_answer = [i for i, x in enumerate(ds) if not x['answers']['text']]
    answerable = [i for i, x in enumerate(ds) if x['answers']['text']]
    n_no = min(len(no_answer), round(n * len(no_answer) / len(ds)))
    chosen = rng.sample(no_answer, n_no) + rng.sample(answerable, n - n_no)
    rng.shuffle(chosen)
    return ds.select(chosen)


def normalize(s):
    s = s.lower()
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(w for w in s.split() if w not in {'a', 'an', 'the'})


def f1_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for w in p:
        common[w] = min(p.count(w), g.count(w))
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def main():
    a = args_parser()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    log_dir = Path(a.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(a.output_dir)
    raw = load_dataset('squad_v2')
    train_raw = limited(raw['train'], a.train_limit, a.seed)
    eval_raw = limited(raw['validation'], a.eval_limit, a.seed)
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)

    def preprocess(batch):
        questions = [q.lstrip() for q in batch['question']]
        enc = tokenizer(questions, batch['context'], truncation='only_second',
                        max_length=a.max_length, stride=a.stride,
                        return_overflowing_tokens=True, return_offsets_mapping=True,
                        padding='max_length')
        offsets = enc.pop('offset_mapping')
        samples = enc.pop('overflow_to_sample_mapping')
        starts, ends = [], []
        for i, offset in enumerate(offsets):
            sample = samples[i]
            sequence_ids = enc.sequence_ids(i)
            answer = batch['answers'][sample]
            if not answer['text']:
                starts.append(0); ends.append(0); continue
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])
            token_start = 0
            while sequence_ids[token_start] != 1: token_start += 1
            token_end = len(offset) - 1
            while sequence_ids[token_end] != 1: token_end -= 1
            if offset[token_start][0] > start_char or offset[token_end][1] < end_char:
                starts.append(0); ends.append(0); continue
            while token_start < len(offset) and offset[token_start][0] <= start_char:
                token_start += 1
            starts.append(token_start - 1)
            while offset[token_end][1] >= end_char:
                token_end -= 1
            ends.append(token_end + 1)
        enc['start_positions'], enc['end_positions'] = starts, ends
        return enc

    train_ds = train_raw.map(preprocess, batched=True, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(preprocess, batched=True, remove_columns=eval_raw.column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(a.model)
    training = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=a.epochs,
        learning_rate=a.learning_rate, per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size, weight_decay=0.01,
        warmup_ratio=0.1, eval_strategy='epoch', save_strategy='epoch',
        logging_strategy='steps', logging_steps=100, report_to='none',
        save_total_limit=1, seed=a.seed, data_seed=a.seed,
        dataloader_num_workers=0, fp16=False)
    trainer = Trainer(model=model, args=training, train_dataset=train_ds,
                      eval_dataset=eval_ds, processing_class=tokenizer,
                      data_collator=default_data_collator)
    start = time.time()
    result = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    elapsed = time.time() - start

    # Validation uses the best span among token start/end candidates and an explicit null.
    predictions = trainer.predict(eval_ds).predictions
    start_logits, end_logits = predictions
    examples = {x['id']: x for x in eval_raw}
    # eval_ds windows preserve example order via overflow mapping; rebuild feature map.
    features = eval_raw.map(lambda b: tokenizer(
        [q.lstrip() for q in b['question']], b['context'], truncation='only_second',
        max_length=a.max_length, stride=a.stride, return_overflowing_tokens=True,
        return_offsets_mapping=True, padding='max_length'), batched=True)
    feature_to_example = features['overflow_to_sample_mapping']
    by_example = {i: [] for i in range(len(eval_raw))}
    for fi, exi in enumerate(feature_to_example):
        offsets = features[fi]['offset_mapping']
        valid = [(s, e) for s, e in enumerate(offsets) if e[0] is not None]
        best = ('', -1e9)
        for s, _ in valid:
            for e, _ in valid:
                if e < s or e - s > 30: continue
                score = float(start_logits[fi][s] + end_logits[fi][e])
                if score > best[1]:
                    char_s, char_e = offsets[s][0], offsets[e][1]
                    best = (eval_raw[exi]['context'][char_s:char_e], score)
        null_score = float(start_logits[fi][0] + end_logits[fi][0])
        by_example[exi].append((best[0], best[1] - null_score))
    rows, groups = [], {'answerable': [], 'unanswerable': []}
    for i, ex in enumerate(eval_raw):
        candidate = max(by_example[i], key=lambda x: x[1])[0] if by_example[i] else ''
        refs = ex['answers']['text'] or ['']
        em = max(float(normalize(candidate) == normalize(ref)) for ref in refs)
        f1 = max(f1_score(candidate, ref) for ref in refs)
        group = 'answerable' if ex['answers']['text'] else 'unanswerable'
        groups[group].append((em, f1))
        rows.append({'id': ex['id'], 'prediction': candidate, 'references': json.dumps(refs),
                     'answerable': group == 'answerable', 'em': em, 'f1': f1})
    with (log_dir / 'validation_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    metrics = {'train_examples': len(train_raw), 'validation_examples': len(eval_raw),
               'training_seconds': elapsed, 'train_loss': result.training_loss}
    for group, vals in groups.items():
        metrics[group] = {'count': len(vals), 'EM': 100 * sum(x[0] for x in vals) / max(1, len(vals)),
                          'F1': 100 * sum(x[1] for x in vals) / max(1, len(vals))}
    (log_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (log_dir / 'train_history.json').write_text(json.dumps(trainer.state.log_history, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
