"""Write Kev --data files for the stance SetFit selections (same rows, same labels as the committed SetFit cells)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from tasks import TASKS, load
ROOT = HERE.parent; BENCH = ROOT.parents[2] / "benchmarks" / "tweeteval-stance"
t = TASKS["stance-abortion"]; train = load("stance-abortion", "train")
out = ROOT / "runs" / "shots"; out.mkdir(parents=True, exist_ok=True)
for sel in sorted((BENCH / "selections").iterdir()):
    ex = json.load(open(sel / "selection-manifest.json"))["payload"]["ordered_examples"]
    with open(out / f"{sel.name}.jsonl", "w") as f:
        for e in ex:
            text, y = train[int(e["id"].split(":")[1])]; assert y == e["label"]
            f.write(json.dumps({"state": text, "questions": {"label": {"type": "choice", "instructions": t["instructions"],
                    "criteria": t["criteria"], "label": t["labels"][y]}}}) + "\n")
print(len(list(out.iterdir())), "files")
