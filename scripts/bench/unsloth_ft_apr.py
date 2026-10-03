#!/usr/bin/env python3
"""apr side of beat-unsloth-finetune-throughput-v1 (0.72 T2, row R5): an adapter.

Called by unsloth_ft_apr_side.sh, which unsloth_finetune_throughput.sh runs
under gpu-q. It runs `apr finetune` once on the canonical task and copies the
fields of apr's own run receipt into the receipt unsloth_ft_verdict.py reads.

  APR_FT_BASE=<local base model> APR_FT_DATA=<data.jsonl> \
    unsloth_ft_apr.py --side apr --model M --gpu N --run K --receipt PATH [--planted-half-targets]

--model is the name both receipts carry (the incumbent loads it as an HF id);
`apr finetune` takes a local path as its positional FILE, so that comes from
APR_FT_BASE.

It copies, it never computes or translates: every verdict field comes from a
named key of apr's receipt (APR_KEYS), so the number the verdict judges is the
number the binary measured. A key apr does not write is a GAP. The run then
writes no receipt and exits 4, naming every missing key (the orchestrator skips
the run and the verdict says NOT_MEASURED). The flags in apr_argv() and the keys
in APR_KEYS are what R15 must add to `apr finetune`. On 316dee2cd4 most of them
do not exist yet; the test prints the gap list for a TRR-only receipt.

  - `--json` tokens_per_sec (apr-finetune-metrics-v1) is samples/s x avg_seq_len
    over the whole run. It is not label tokens over a timed window after warmup,
    so it is never read here.
  - The apr receipt base is train-run-receipt-v1 (TRR): apr_version, apr_git_sha,
    data_sha256[] and status. The rest (device, recipe, timed_window,
    trainable_params) is the T2 extension R15 owes.

The pure parts (argv, key mapping, gaps) are tested on a CPU box by
scripts/tests/unsloth_ft_apr_test.py, which also runs this file end to end
against a stub `apr`.
"""
import argparse
import json
import os
import subprocess
import sys

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
HALF_TARGETS = TARGETS[:4]  # attention only: the planted SAME-WORK falsifier
TASK = dict(rank=16, alpha=32, seq_len=512, batch=4, grad_accum=1,
            warmup_steps=50, timed_steps=200)
# verdict field <- dotted key in apr's receipt. Copied verbatim, no translation.
APR_KEYS = dict(
    apr_version="apr_version", apr_git_sha="apr_git_sha",
    gpu_name="device.name", gpu_uuid="device.uuid", device_trace_line="device.trace_line",
    rank="recipe.rank", alpha="recipe.alpha", targets="recipe.targets",
    optimizer="recipe.optimizer", grad_checkpointing="recipe.grad_checkpointing",
    packing="recipe.packing", seq_len="recipe.seq_len", batch="recipe.batch",
    grad_accum="recipe.grad_accum", warmup_steps="recipe.warmup_steps",
    timed_steps="recipe.timed_steps", trainable_params="trainable_params",
    label_tokens_timed="timed_window.label_tokens", timed_seconds="timed_window.seconds",
    timed_after_compile="timed_window.after_compile",
)
GAP_EXIT = 4


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--side", required=True, choices=["apr"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--run", required=True, type=int)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--planted-half-targets", action="store_true")
    return ap.parse_args(argv)


def apr_argv(apr, args, base, data, trr):
    """The canonical task as `apr finetune` flags. [U] R15: most do not exist yet."""
    targets = HALF_TARGETS if args.planted_half_targets else TARGETS
    return [apr, "finetune", base, "--method", "lora", "--precision", "bf16",
            "--gpu-backend", "cuda", "--gpus", str(args.gpu),
            "--rank", str(TASK["rank"]), "--alpha", str(TASK["alpha"]),
            "--targets", ",".join(targets), "--data", data,
            "--max-seq-len", str(TASK["seq_len"]), "--batch-size", str(TASK["batch"]),
            "--grad-accum", str(TASK["grad_accum"]), "--warmup-steps", str(TASK["warmup_steps"]),
            "--timed-steps", str(TASK["timed_steps"]), "--optimizer", "adamw_fp32",
            "--no-grad-checkpointing", "--no-packing", "--receipt", trr]


def lookup(doc, dotted):
    """(True, value) if every part of the dotted key exists and the value is not None."""
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return cur is not None, cur


def single_data_sha(trr):
    shas = trr.get("data_sha256")
    if isinstance(shas, list) and len(shas) == 1 and shas[0]:
        return shas[0]
    return None


def to_verdict(trr, model):
    """(receipt, gaps). gaps lists every key apr did not write; receipt is None then."""
    if trr.get("status") != "success":
        return None, ["status=%r (want 'success')" % trr.get("status")]
    out, gaps = dict(side="apr", model=model), []
    for field, key in APR_KEYS.items():
        ok, value = lookup(trr, key)
        if ok:
            out[field] = value
        else:
            gaps.append(key)
    sha = single_data_sha(trr)
    if sha is None:
        gaps.append("data_sha256 (exactly one file)")
    out["data_sha256"] = sha
    return (None, gaps) if gaps else (out, [])


def env_file(name):
    """The path in env var `name` if it exists (a file, or a model directory), else None."""
    path = os.environ.get(name, "")
    if path and os.path.exists(path):
        return path
    print("%s does not exist: %r" % (name, path), file=sys.stderr)
    return None


def main(argv):
    args = parse(argv)
    data, base = env_file("APR_FT_DATA"), env_file("APR_FT_BASE")
    if data is None or base is None or not os.path.isfile(data):
        return 3
    trr = args.receipt + ".apr.json"
    cmd = apr_argv(os.environ.get("APR_BIN", "apr"), args, base, data, trr)
    print("+ " + " ".join(cmd), file=sys.stderr)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print("apr finetune exited %d" % rc, file=sys.stderr)
        return rc
    try:
        with open(trr, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        print("GAP: no readable apr receipt at %s (%s)" % (trr, e), file=sys.stderr)
        return GAP_EXIT
    receipt, gaps = to_verdict(doc, args.model)
    if gaps:
        print("GAP: apr receipt lacks " + ", ".join(gaps), file=sys.stderr)
        return GAP_EXIT
    with open(args.receipt, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
