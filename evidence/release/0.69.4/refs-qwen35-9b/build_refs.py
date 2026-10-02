"""#4261: build the Qwen3.5-9B temp-0 refs (refs.json) and the receipt table from one harness run.

Usage: python3 build_refs.py <harness out dir>. Reads rc-9b-<p>.json / base-9b-<p>.json (apr run --json)
and llama-9b-<p>.json (llama-server on the GGUF's own template). Every cell carries its own ids, so a
cell that did not run is written as missing, never as a match.
"""
import hashlib, json, os, re, sys

OUT = sys.argv[1]
PROMPTS = ("p850", "p4k", "p32k")
TRACE = re.compile(r"\[qwen35\] batched prefill: (\d+) tokens in (\d+) ms \((\d+) tok/s")


def load(key):
    path = f"{OUT}/{key}.json"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    d = json.load(open(path))
    toks = d["tokens"]
    d["tokens"] = json.loads(toks) if isinstance(toks, str) else toks
    err = f"{OUT}/{key}.err"
    d["_err"] = open(err).read() if os.path.exists(err) else ""
    return d


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))


refs, rows, bad = {}, [], 0
for p in PROMPTS:
    rc, base, ll = load(f"rc-9b-{p}"), load(f"base-9b-{p}"), load(f"llama-9b-{p}")
    cell = {"prompt_sha256": hashlib.sha256(open(f"{OUT}/../{p}.txt", "rb").read()).hexdigest()}
    if rc is None:
        bad += 1
        rows.append(f"| {p} | - | **MISSING** | | | |")
        refs[p] = cell
        continue
    assert rc["backend"]["ran"] == "gpu" and not rc["backend"]["fell_back"], f"{p}: rc did not run on GPU"
    t = TRACE.search(rc["_err"])
    cell.update(prompt_tokens=rc["prompt_tokens"], apr_rc_ids=rc["tokens"], apr_rc_text=rc["text"],
                apr_rc_finish=rc["finish_reason"])
    if base is None:
        vb = "not run"
    else:
        d = first_diff(rc["tokens"], base["tokens"])
        vb = f"IDENTICAL ({len(rc['tokens'])} ids)" if d is None else f"**DIFFER** at {d}"
        bad += d is not None
        cell["apr_base_ids_equal"] = d is None
    if ll is None:
        vl = "not run"
    else:
        d = first_diff(rc["tokens"], ll["tokens"])
        same_prompt = ll["prompt_tokens"] == rc["prompt_tokens"]
        vl = (f"IDENTICAL ({len(rc['tokens'])} ids)" if d is None else f"agree on first {d} ids") + \
             ("" if same_prompt else f"; prompt tokens {ll['prompt_tokens']} vs apr {rc['prompt_tokens']}")
        cell.update(llama_ids=ll["tokens"], llama_prompt_tokens=ll["prompt_tokens"], llama_first_diff=d)
    refs[p] = cell
    tr = f"{t.group(1)} tok / {t.group(2)} ms / {t.group(3)} tok/s" if t else "**NO TRACE LINE**"
    rows.append(f"| {p} | {rc['prompt_tokens']} | {len(rc['tokens'])} ({rc['finish_reason']}) | {vb} | {vl} | {tr} |")

json.dump(refs, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "refs.json"), "w"), indent=1)
print("| prompt | prompt tokens | apr rc ids | rc vs base 0.69.1 | rc vs llama.cpp | rc prefill trace |")
print("|---|---|---|---|---|---|")
print("\n".join(rows))
print("\nrc/base regressions:", bad)
sys.exit(1 if bad else 0)
