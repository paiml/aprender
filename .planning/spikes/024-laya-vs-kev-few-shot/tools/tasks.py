"""The two few-shot tasks, built identically for Kev and SetFit.

stance-abortion: cardiffnlp/tweet_eval stance_abortion (0 none, 1 against, 2 favor) -- the deployed SetFit
pilot task, absent from every Kev training and eval source (checked 2026-09-23 at kev@7405b72).
emotion: dair-ai/emotion split (6 classes) -- absent from Kev training, but a Kev transfer-v4 DEV source, so
Kev's authors selected checkpoints partly on it. Reported with that flag.
"""
from datasets import load_dataset

TASKS = {
    "stance-abortion": {
        "repo": ("cardiffnlp/tweet_eval", "stance_abortion"), "text": "text", "train": "train", "test": "test",
        "labels": ["none", "against", "favor"],
        "instructions": "What stance does the author of this tweet take on abortion?",
        # descriptions = the business's zero-shot steering; names_only = what a caller writes with no effort
        "criteria": {"none": "No stance on abortion, or the tweet is not about abortion",
                     "against": "Opposes abortion (pro-life)",
                     "favor": "Supports abortion rights (pro-choice)"},
    },
    "emotion": {
        "repo": ("dair-ai/emotion", "split"), "text": "text", "train": "train", "test": "test",
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "instructions": "Which emotion does the writer express?",  # Kev's own transfer-v4 wording
        "criteria": {k: None for k in ["sadness", "joy", "love", "anger", "fear", "surprise"]},
    },
}


def load(task, split, limit=None, seed=0):
    t = TASKS[task]
    ds = load_dataset(*t["repo"], split=t[split])
    if limit and len(ds) > limit:
        ds = ds.shuffle(seed=seed).select(range(limit))
    return [(r[t["text"]], int(r["label"])) for r in ds]


def request(task, text, criteria=None):
    t = TASKS[task]
    return {"state": text, "questions": {"label": {"type": "choice", "instructions": t["instructions"],
                                                   "criteria": criteria if criteria is not None else t["criteria"]}}}


def sample_shots(rows, n_labels, k, seed):
    """k per class, deterministic; SetFit's sample_dataset convention (shuffle then take the first k of each label)."""
    import random
    rng = random.Random(seed)
    idx = list(range(len(rows))); rng.shuffle(idx)
    out, seen = [], {c: 0 for c in range(n_labels)}
    for i in idx:
        c = rows[i][1]
        if seen[c] < k:
            out.append(i); seen[c] += 1
    return out
