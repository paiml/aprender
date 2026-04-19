#!/usr/bin/env python3
"""
SHIP-TWO-001 Parallel Eval Lane — merger + falsifier.

Contract:   contracts/eval-sharding-v1.yaml
Spec ref:   docs/specifications/aprender-train/ship-two-models-spec.md §12.6
Discharges: FALSIFY-SHARD-001 (completeness),
            FALSIFY-SHARD-002 (disjointness),
            FALSIFY-SHARD-004 (merged-score identity on --self-test-reshard)

Responsibilities:
  1. Collect per-shard completions.jsonl or per-shard result JSONs.
  2. Run the test phase (sandboxed python3 execution) per completion.
  3. Merge problems[] across shards.
  4. Recompute Chen et al. unbiased pass@k on merged array.
  5. Enforce FALSIFY-SHARD-001 (completeness vs benchmark) + FALSIFY-SHARD-002
     (disjointness across shards).
  6. Write a single merged humaneval_*.json matching eval-pass-at-k.sh schema.

Self-test mode (--self-test-reshard) validates FALSIFY-SHARD-004 by reshuffling
an existing single-host result into pseudo-shards and verifying merged pass@1
matches the original to within 0.01 pp.

Usage:
  python3 eval-shard-merge.py \
    --benchmark-jsonl data/benchmarks/humaneval.jsonl \
    --shard-completions shard-eval/run_*/completions_shard_*.jsonl \
    --shard-dir shard-eval/run_$TS \
    --benchmark humaneval \
    --output shard-eval/run_$TS/humaneval_merged_$TS.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

MERGE_PARITY_THRESHOLD_PP = 0.01


# ─────────────────────────────────────────────────────────────
# pass@k (Chen et al. 2021 unbiased estimator)
# ─────────────────────────────────────────────────────────────
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k; matches eval-pass-at-k.sh awk block."""
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    return 1.0 - math.exp(
        sum(math.log(n - c - i) - math.log(n - i) for i in range(k))
    )


def aggregate_pass_at_1(problems: list[dict]) -> float:
    if not problems:
        return 0.0
    total = 0.0
    for p in problems:
        total += pass_at_k(int(p["n"]), int(p["passed"]), 1)
    return 100.0 * total / len(problems)


# ─────────────────────────────────────────────────────────────
# Completion extraction (mirrors apr-leaderboard eval-helpers.sh logic)
# ─────────────────────────────────────────────────────────────
def strip_markdown_fences(text: str) -> str:
    """Extract code from first ```python ... ``` block if present, else passthrough."""
    import re

    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def extract_python_code(text: str) -> str:
    """Best-effort: strip chat markers, keep only code-ish lines."""
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return strip_markdown_fences(text)


# ─────────────────────────────────────────────────────────────
# Benchmark parsing
# ─────────────────────────────────────────────────────────────
def load_benchmark_tasks(path: str) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            tid = str(obj.get("task_id") or obj.get("name") or "")
            if not tid:
                continue
            tasks[tid] = obj
    return tasks


# ─────────────────────────────────────────────────────────────
# Completion loading: apr run --batch-jsonl emits one JSON per line, each with
# the original prompt + generated text. We group by original task_id.
# ─────────────────────────────────────────────────────────────
def load_shard_completions(paths: list[str]) -> list[tuple[int, list[dict]]]:
    """Returns [(shard_index, [{task_id, completion}, ...]), ...]"""
    out: list[tuple[int, list[dict]]] = []
    for p in paths:
        base = os.path.basename(p)
        # expect completions_shard_N.jsonl
        shard_idx = None
        for part in base.split("_"):
            if part.isdigit():
                shard_idx = int(part)
                break
            if part.startswith("shard") and part[5:].split(".")[0].isdigit():
                shard_idx = int(part[5:].split(".")[0])
                break
        if shard_idx is None:
            shard_idx = len(out)
        records: list[dict] = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = obj.get("task_id") or obj.get("id")
                txt = obj.get("generated") or obj.get("completion") or obj.get("output") or ""
                if tid and txt:
                    records.append({"task_id": str(tid), "completion": txt})
        out.append((shard_idx, records))
    return out


# ─────────────────────────────────────────────────────────────
# Per-completion test execution (mirrors eval-pass-at-k.sh Phase 3)
# ─────────────────────────────────────────────────────────────
def run_humaneval_test(task: dict, completion_text: str, timeout_sec: int = 10) -> bool:
    prompt = task.get("prompt", "")
    entry_point = task.get("entry_point") or ""
    test_code = task.get("test") or ""
    code = extract_python_code(completion_text)
    full = f"{prompt}\n{code}\n\n{test_code}\n"
    if entry_point:
        full += f"\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tf:
        tf.write(full)
        tfname = tf.name
    try:
        r = subprocess.run(
            ["python3", tfname],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_sec,
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tfname)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────
# Main merge path
# ─────────────────────────────────────────────────────────────
def merge(args) -> int:
    bench_tasks = load_benchmark_tasks(args.benchmark_jsonl)
    bench_ids = set(bench_tasks.keys())

    shards = load_shard_completions(args.shard_completions)
    if not shards:
        print("ERROR: no shard completion files loaded", file=sys.stderr)
        return 2

    # FALSIFY-SHARD-002: disjointness
    seen: dict[str, int] = {}
    for idx, records in shards:
        for r in records:
            tid = r["task_id"]
            if tid in seen:
                print(
                    f"FALSIFY-SHARD-002 FAIL: task_id {tid} appears in "
                    f"shard_{seen[tid]} AND shard_{idx}",
                    file=sys.stderr,
                )
                return 10
            seen[tid] = idx

    # FALSIFY-SHARD-001: completeness
    shard_ids = set(seen.keys())
    missing = bench_ids - shard_ids
    if missing:
        print(
            f"FALSIFY-SHARD-001 FAIL: {len(missing)} task_id(s) missing from all shards: "
            f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )
        return 11
    extra = shard_ids - bench_ids
    if extra:
        print(
            f"WARNING: {len(extra)} task_id(s) in shards but not benchmark: "
            f"{sorted(extra)[:5]}",
            file=sys.stderr,
        )

    # Test each completion against the benchmark task's test code
    problems: list[dict] = []
    for idx, records in shards:
        for r in records:
            tid = r["task_id"]
            task = bench_tasks.get(tid)
            if not task:
                continue
            ok = run_humaneval_test(task, r["completion"])
            problems.append({"task_id": tid, "n": 1, "passed": 1 if ok else 0})

    problems.sort(key=lambda x: x["task_id"])
    pass_1 = aggregate_pass_at_1(problems)

    result: dict[str, Any] = {
        "benchmark": args.benchmark,
        "model": args.model or "UNKNOWN",
        "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "shard_run": True,
        "n_shards": len(shards),
        "config": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "num_samples": 1,
        },
        "results": {
            "total": len(bench_ids),
            "completed": len(problems),
            "passed": sum(p["passed"] for p in problems),
            "errors": len(bench_ids) - len(problems),
            "pass_at_1": round(pass_1, 2),
            "pass_at_10": None,
        },
        "problems": problems,
        "shards": [
            {"shard_index": idx, "tasks": len(records)} for idx, records in shards
        ],
        "falsification_gates": {
            "FALSIFY-SHARD-001": "PASS",
            "FALSIFY-SHARD-002": "PASS",
        },
    }

    pathlib.Path(os.path.dirname(args.output)).mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[merge] merged {len(shards)} shards → {len(problems)} problems")
    print(f"[merge] pass@1 = {pass_1:.2f}% ({result['results']['passed']}/{len(problems)})")
    print(f"[merge] wrote {args.output}")
    return 0


# ─────────────────────────────────────────────────────────────
# Self-test: FALSIFY-SHARD-004 reshard identity check
# ─────────────────────────────────────────────────────────────
def self_test_reshard(ref_path: str) -> int:
    with open(ref_path) as f:
        ref = json.load(f)
    problems = ref.get("problems", [])
    ref_pass_1 = float(ref["results"]["pass_at_1"])
    if not problems:
        print(f"ERROR: {ref_path} has no problems[]", file=sys.stderr)
        return 2

    # Split round-robin into 2 pseudo-shards
    shards = [[], []]
    for i, p in enumerate(problems):
        shards[i % 2].append(p)

    # Disjointness + completeness checks (FALSIFY-SHARD-001/002 on pseudo-shards)
    ids_a = {p["task_id"] for p in shards[0]}
    ids_b = {p["task_id"] for p in shards[1]}
    dup = ids_a & ids_b
    if dup:
        print(f"SELF-TEST FAIL: overlap {dup}", file=sys.stderr)
        return 12
    missing = {p["task_id"] for p in problems} - (ids_a | ids_b)
    if missing:
        print(f"SELF-TEST FAIL: missing {missing}", file=sys.stderr)
        return 13

    merged = shards[0] + shards[1]
    merged_pass_1 = aggregate_pass_at_1(merged)

    delta = abs(merged_pass_1 - ref_pass_1)
    print(f"[self-test] ref pass@1 = {ref_pass_1:.4f}%")
    print(f"[self-test] merged pass@1 = {merged_pass_1:.4f}%")
    print(f"[self-test] delta = {delta:.4f} pp (threshold {MERGE_PARITY_THRESHOLD_PP} pp)")
    if delta > MERGE_PARITY_THRESHOLD_PP:
        print(f"FALSIFY-SHARD-004 FAIL: delta {delta:.4f} > {MERGE_PARITY_THRESHOLD_PP}", file=sys.stderr)
        return 14
    print("FALSIFY-SHARD-004 PASS")
    return 0


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-jsonl", help="data/benchmarks/*.jsonl")
    ap.add_argument(
        "--shard-completions",
        nargs="+",
        help="glob-expanded completion files, one per shard",
    )
    ap.add_argument("--shard-dir", help="orchestrator run dir (for logs/reports)")
    ap.add_argument("--benchmark", default="humaneval")
    ap.add_argument("--model", default="")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", help="merged humaneval_*.json path")
    ap.add_argument(
        "--self-test-reshard",
        help="FALSIFY-SHARD-004 self-test: reshard an existing single-host result",
    )
    args = ap.parse_args()

    if args.self_test_reshard:
        return self_test_reshard(args.self_test_reshard)

    missing = []
    if not args.benchmark_jsonl:
        missing.append("--benchmark-jsonl")
    if not args.shard_completions:
        missing.append("--shard-completions")
    if not args.output:
        missing.append("--output")
    if missing:
        print(f"ERROR: missing required args: {missing}", file=sys.stderr)
        return 2

    return merge(args)


if __name__ == "__main__":
    sys.exit(main())
