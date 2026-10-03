#!/usr/bin/env python3
"""EXT-24 CRUX bind census (look-ahead, la-0.71).

Reads the competitor registry from validator.rs and the contracts/crux-*.yaml
files, then checks the two preconditions EXT-001 §10 assumes but main does not
yet meet:

  B1  every competitor arm named in EXT-001 §10 C1-C7 is in CRUX_COMPETITORS
  B2  each category letter maps to exactly one category name

Exit 0 means bound, exit 2 means RED with one line per violation.
--self-test runs the planted fixtures under fixtures/crux_bind/.
"""
import json
import re
import sys
from pathlib import Path

# Competitor arms in EXT-001 §10, as CRUX_COMPETITORS-style slugs.
EXT_ARMS = ["llama_cpp", "ollama", "huggingface", "unsloth", "bartowski",
            "mistral_rs", "mlflow", "dvc", "wandb", "qwen"]
REG_RE = re.compile(r"CRUX_COMPETITORS: \[&str; \d+\] = \[(.*?)\];", re.S)
CAT_RE = re.compile(r"^\s*category:\s*['\"]?([A-Z])\s*[—-]\s*([^'\"#\n]+)", re.M)


def registry(validator_src):
    m = REG_RE.search(validator_src)
    if not m:
        return None
    body = "\n".join(l.split("//")[0] for l in m.group(1).splitlines())
    return set(re.findall(r'"([^"]+)"', body))


def categories(yaml_texts):
    names = {}
    for text in yaml_texts:
        m = CAT_RE.search(text)
        if m:
            names.setdefault(m.group(1), set()).add(m.group(2).strip())
    return names


def evaluate(validator_src, yaml_texts, arms=EXT_ARMS):
    reg = registry(validator_src)
    if reg is None:
        return ["B0 CRUX_COMPETITORS not found"]
    errs = [f"B1 competitor not registered: {a}" for a in arms if a not in reg]
    cats = categories(yaml_texts)
    if not cats:
        errs.append("B2 no crux category found (vacuous)")
    for letter, names in sorted(cats.items()):
        if len(names) > 1:
            errs.append(f"B2 letter {letter} has {len(names)} names: {sorted(names)}")
    return errs


def from_tree(root):
    root = Path(root)
    src = (root / "crates/aprender-contracts/src/schema/validator.rs").read_text()
    texts = [p.read_text() for p in sorted(root.glob("contracts/crux-*.yaml"))]
    return src, texts


def self_test(fix):
    cases = json.loads((fix / "cases.json").read_text())
    bad = 0
    for c in cases:
        src = (fix / c["validator"]).read_text()
        texts = [(fix / y).read_text() for y in c["yaml"]]
        got = 2 if evaluate(src, texts, c.get("arms", EXT_ARMS)) else 0
        if got != c["expect"]:
            bad += 1
            print(f"FAIL {c['id']}: expect {c['expect']} got {got}")
    print(f"self-test: {len(cases) - bad}/{len(cases)} ok")
    return 0 if bad == 0 else 1


def main(argv):
    if argv[1:2] == ["--self-test"]:
        return self_test(Path(__file__).resolve().parents[2] / "fixtures/crux_bind")
    errs = evaluate(*from_tree(argv[1] if len(argv) > 1 else "."))
    for e in errs:
        print(e)
    print("crux-bind:", "RED" if errs else "BOUND")
    return 2 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
