#!/usr/bin/env python3
"""crux_prompt_certify: offline certification of the CRUX prompt set (#3962, quorum revision Q1).

A prompt is USED for a (model, quant) only once this receipt admits it there. Admission needs every
LEG correct, under every thinking mode the model has:

  ggml@bf16    llama.cpp / ollama / llamafile on the model's BF16 GGUF
  hf@bf16      transformers on the pinned source weights
  vllm@bf16    vLLM on the same pinned source weights
  ggml@quant   the ggml family on the IDENTICAL quantized GGUF apr is gated on

"Correct" is decided by scripts/lib/crux_oracles.py, the same code the gate uses. A leg with no row, a
refused row, or any wrong row rejects the prompt for that (model, quant), and the reason names the first
such cell. Missing is never agreement (#3957 F3), and one right row does not outvote a wrong one.

Rows are CRUX row contract v1 (#3739 comment 5765991210); each engine's output is read through the
judge's own parsers (crux_inference_judge.engine_entry), so certification and the gate cannot disagree
about what an engine said.

inventory.json - one entry per model family:
  [{"model": "Qwen3.5-4B", "source": {"repo": "Qwen/Qwen3.5-4B", "revision": "<sha>"},
    "bf16_gguf": "<sha256>", "quants": {"UD-Q4_K_XL": "<sha256>"}, "thinking": ["on", "off"]}]

CLI:
  crux_prompt_certify.py certify --prompts P --inventory I --apr-commit SHA -o OUT MANIFEST...
  crux_prompt_certify.py check --prompts P --receipt R      the judge's gate (J2): exit 0 only when the
                                                            receipt certifies exactly these prompt bytes
  exit 0 ok · 1 refused (check) · 2 usage/ENV
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "crux-prompt-certification/v1"
GGML = ("llama.cpp", "ollama", "llamafile")
LEGS = ("ggml@bf16", "hf@bf16", "vllm@bf16", "ggml@quant")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oracles = _load("crux_oracles")
judge = _load("crux_inference_judge")


def sha256_path(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_rows(paths: list) -> list:
    rows = []
    for p in paths:
        for n, line in enumerate(Path(p).read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                row["_at"] = f"{p}#L{n}"
                rows.append(row)
    return [r for r in rows if r.get("kind") == "gen"]


def leg_of(row: dict, model: dict, quant_sha: str):
    eng = row.get("engine")
    if eng in GGML:
        if row.get("model_sha256") == model["bf16_gguf"]:
            return "ggml@bf16"
        if row.get("model_sha256") == quant_sha:
            return "ggml@quant"
        return None
    if eng in ("hf", "vllm") and (row.get("source") or {}).get("repo") == model["source"]["repo"]:
        if (row.get("source") or {}).get("revision") != model["source"]["revision"]:
            return None  # a moving or different revision is not the pinned source
        return f"{eng}@bf16"
    return None


def reply_of(row: dict, prompt: dict) -> tuple:
    """((text, turns) for the oracles, why-not). vLLM writes the same JSON as hf (row contract v1)."""
    r = dict(row, engine="hf") if row.get("engine") == "vllm" else row
    e = judge.engine_entry(r, prompt)
    if e.get("answer") is None:
        return None, e.get("why") or "no answer"
    if row.get("rc") != 0:
        return None, f"exit {row.get('rc')}"
    return (e["answer"], e.get("turns")), None


def certify_one(prompt: dict, model: dict, quant: str, quant_sha: str, rows: list) -> tuple:
    cells, first_bad = [], None
    for thinking in model.get("thinking") or ["off"]:
        for leg in LEGS:
            mine = [r for r in rows if r.get("prompt_id") == prompt["id"] and r.get("thinking") == thinking
                    and r.get("verb") in prompt["verb"] and leg_of(r, model, quant_sha) == leg]
            if not mine:
                first_bad = first_bad or f"{leg} thinking={thinking}: no row"
                cells.append({"leg": leg, "thinking": thinking, "correct": False, "why": "no row", "row": None})
                continue
            for r in mine:
                reply, why = reply_of(r, prompt)
                v = oracles.evaluate(prompt, *reply) if reply else {"correct": False, "why": why, "extracted": None}
                cells.append({"leg": leg, "thinking": thinking, "engine": r["engine"], "verb": r["verb"],
                              "host": r.get("host"), "correct": v["correct"], "why": v["why"],
                              "extracted": v["extracted"], "row": r["_at"]})
                if not v["correct"]:
                    first_bad = first_bad or f"{leg} {r['engine']} {r['verb']} thinking={thinking} on {r.get('host')}: {v['why']}"
    return first_bad is None, first_bad, cells


def certify(a) -> int:
    prompts_path = Path(a.prompts)
    doc = json.loads(prompts_path.read_text(encoding="utf-8"))
    errs = oracles.validate_set(doc)
    if errs:
        print("crux_prompt_certify: the prompt set is invalid:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2
    inventory = json.loads(Path(a.inventory).read_text(encoding="utf-8"))
    rows = read_rows(a.manifests)
    admitted, rejected, cells = {}, {}, []
    for model in inventory:
        for quant, qsha in sorted(model["quants"].items()):
            key = f"{model['model']}/{quant}"
            admitted[key], rejected[key] = [], {}
            for p in doc["prompts"]:
                ok, why, cs = certify_one(p, model, quant, qsha, rows)
                cells += [dict(c, prompt_id=p["id"], model=key) for c in cs]
                if ok:
                    admitted[key].append(p["id"])
                else:
                    rejected[key][p["id"]] = why
    # A (model, quant) whose positive controls did not certify cannot be gated at all (#3957: no control, no run).
    controls = [p["id"] for p in doc["prompts"] if p.get("control")]
    uncontrolled = sorted(k for k, ids in admitted.items() if not set(controls) <= set(ids))
    receipt = {
        "schema": SCHEMA,
        "prompts": str(prompts_path),
        "prompts_sha256": sha256_path(prompts_path),
        "apr_commit": a.apr_commit,
        "inventory_sha256": sha256_path(Path(a.inventory)),
        "manifests": {m: sha256_path(Path(m)) for m in a.manifests},
        "admitted": admitted,
        "rejected": rejected,
        "uncontrolled": uncontrolled,
        "cells": cells,
    }
    Path(a.out).write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for k in admitted:
        print(f"{k}: {len(admitted[k])} admitted, {len(rejected[k])} rejected"
              + ("  [CONTROL NOT CERTIFIED]" if k in uncontrolled else ""))
    return 0


def check(a) -> int:
    receipt = json.loads(Path(a.receipt).read_text(encoding="utf-8"))
    if receipt.get("schema") != SCHEMA:
        print(f"refused: receipt schema {receipt.get('schema')!r}, want {SCHEMA!r}")
        return 1
    have = sha256_path(Path(a.prompts))
    if receipt.get("prompts_sha256") != have:
        print(f"refused: {a.prompts} is sha256 {have[:12]}, the certification covers {str(receipt.get('prompts_sha256'))[:12]}"
              " - an edited prompt set is uncertified until it is certified again")
        return 1
    if receipt.get("uncontrolled"):
        print("refused: positive control not certified for " + ", ".join(receipt["uncontrolled"]))
        return 1
    print(f"certified: {sum(len(v) for v in receipt['admitted'].values())} (model, prompt) admissions")
    return 0


def main(argv: list) -> int:
    p = argparse.ArgumentParser(prog="crux_prompt_certify.py")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("certify")
    c.add_argument("--prompts", required=True)
    c.add_argument("--inventory", required=True)
    c.add_argument("--apr-commit", required=True)
    c.add_argument("-o", "--out", required=True)
    c.add_argument("manifests", nargs="+")
    k = sub.add_parser("check")
    k.add_argument("--prompts", required=True)
    k.add_argument("--receipt", required=True)
    a = p.parse_args(argv)
    return certify(a) if a.cmd == "certify" else check(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
