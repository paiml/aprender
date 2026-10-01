"""Python SetFit reference for emotion (apr setfit only ingests the attested stance dataset).

Same encoder (sentence-transformers/all-MiniLM-L6-v2) and the knobs of our committed train-config.json
(oversampling pairs, 1 epoch, batch 16, lr 2e-5, logistic head C=1) -- phases 1-5 proved the Rust trainer
matches this library on that config, so this row stands in for `apr setfit` on a task apr cannot ingest.
Shots use the same sample_shots(seed) rows as the Kev emotion cells.
"""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score
from tasks import load, sample_shots

SEEDS, SHOTS = [13, 17, 23, 29, 31], [8, 16, 32, 64]
train, test = load("emotion", "train", 3000), load("emotion", "test", 1000)
out = HERE.parent / "results" / "setfit-emotion.json"; rows = json.load(open(out)) if out.exists() else []
done = {(r["shots"], r["seed"]) for r in rows}
for k in SHOTS:
    for seed in SEEDS:
        if (k, seed) in done: continue
        ids = sample_shots(train, 6, k, seed)
        ds = Dataset.from_dict({"text": [train[i][0] for i in ids], "label": [train[i][1] for i in ids]})
        model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", head_params={"C": 1.0, "max_iter": 2000})
        args = TrainingArguments(batch_size=16, num_epochs=1, body_learning_rate=2e-5, sampling_strategy="oversampling", seed=seed)
        t0 = time.perf_counter(); Trainer(model=model, args=args, train_dataset=ds).train(); wall = time.perf_counter() - t0
        pred = np.asarray(model.predict([t for t, _ in test])); y = np.array([c for _, c in test])
        rows.append({"variant": "setfit", "shots": k, "seed": seed, "acc": float((pred == y).mean()),
                     "macro_f1": float(f1_score(y, pred, average="macro")), "train_s": wall})
        json.dump(rows, open(out, "w"), indent=1); print(rows[-1], flush=True)
