"""#3596 receipt tables, generated from the run files — never transcribed by hand.

usage: mk_tables.py <label>=<dir> [<label>=<dir> ...]
Every apr run is a `<tag>.done` with .err/.json/.version/.start/.gpu_at_start beside it;
every llama.cpp run is `llama-<rung>.json`.
"""
import json, os, re, sys

RUNGS = ["brief", "p4k", "p8k", "p20k", "p60k", "p148k", "p262k"]
NEEDLE_FILE = "forward_qwen35_cuda_prefill.rs"
NEEDLE_LINE = "fn chunk_rows_for(d: Qwen35CudaDims, total_positions: usize) -> usize"
SKIP = ("llama-", "parity", "ladder", "swap", "build", "fill", "flash_ladder", "tail", "prof", "done")


def rd(d, tag, ext):
    p = os.path.join(d, tag + ext)
    return open(p, errors="replace").read().strip() if os.path.exists(p) else ""


def runs(d):
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".done") or f.startswith(SKIP):
            continue
        tag = f[:-5]
        err, done = rd(d, tag, ".err"), rd(d, tag, ".done")
        m = re.search(r"batched prefill: (\d+) tokens in (\d+) ms \((\d+) tok/s, chunk (\d+) rows(?:, attention ([^)]*\)?))?\)", err)
        try:
            js = json.loads(rd(d, tag, ".json") or "{}")
        except ValueError:
            js = {}
        model = next((k for k in ("27B", "9B", "4B", "2B", "0.8B") if k.lower() in (js.get("model", "") + tag).lower()), "?")
        rung = next((r for r in RUNGS if re.search(rf"(^|-){r}($|-)", tag)), None)
        ver = rd(d, tag, ".version")
        sha = re.search(r"\(([0-9a-f]{7,})\)", ver)
        gpu = (rd(d, tag, ".gpu_at_start").splitlines() or [""])[0]
        util = re.search(r"(\d+) ?%", gpu)
        text = js.get("text")
        refused = re.search(r"GPU capacity refused: (.*)", err) or re.search(r"(Context limit exceeded.*)", err)
        att = (m.group(5) or "") if m else ""
        mode = "flash" if att.startswith("flash") else ("f32" if m else None)
        out.append({
            "tag": tag, "model": model, "rung": rung, "mode": mode,
            "sha": sha.group(1)[:9] if sha else "pre-record",
            "tokens": int(m.group(1)) if m else None, "ms": int(m.group(2)) if m else None,
            "rate": int(m.group(3)) if m else None, "chunk": int(m.group(4)) if m else None,
            "wall": float(re.search(r"wall=([\d.]+)s", done).group(1)) if "wall=" in done else None,
            "rc": (re.search(r"rc=(\d+)", done) or [None, None])[1],
            "n": js.get("max_tokens"), "text": text, "start": rd(d, tag, ".start"),
            "busy": int(util.group(1)) if util else None,
            "refused": refused.group(1)[:400] if refused else None,
            "answer": grade(text),
        })
    return out


NEEDLE_PATH = "crates/aprender-serve/src/gguf/cuda/forward_qwen35_cuda_prefill.rs"
SECTION = [l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "needle_section.txt")).read().split("\n") if l.strip()]


def grade(text):
    """The question names a file and asks for "its first line of code"; the model reads
    "its" as the file's or the function's, and both are lines of the needle file. So:
    the EXACT path, and a quoted line that is verbatim (a prefix, when -n cuts it) of a
    line of that file. With -n 1 only the first token exists and nothing is graded."""
    if text is None:
        return "—"
    if len(text) < 16:
        return f"first token {json.dumps(text)} (n=1, not graded)"
    path_ok = NEEDLE_PATH in text
    quotes = re.findall(r"`([^`]+)`", text) + re.findall(r"`([^`]+)$", text)
    quotes = [q.strip() for q in quotes if q.strip() and NEEDLE_PATH not in q and q.strip() not in ("chunk_rows_for", "chunk_rows_for()")]
    line_ok = any(len(q) >= 12 and any(l.startswith(q) or (q.startswith(l) and len(l) >= 12) for l in SECTION) for q in quotes) \
        or any(len(l) >= 20 and l in text for l in SECTION)
    garbled = re.findall(r"`?(crates/[A-Za-z0-9_./-]+\.rs)`?", text)
    garbled = [g for g in garbled if g != NEEDLE_PATH]
    parts = ["path ✓" if path_ok else (f"path ✗ ({garbled[0]})" if garbled else "path ✗")]
    parts.append("verbatim line ✓" if line_ok else "line ✗")
    return ", ".join(parts)


def llama(d):
    out = {}
    for r in RUNGS:
        p = os.path.join(d, f"llama-{r}.json")
        if r == "brief":
            p = os.path.join(d, "llama-brief2.json") if os.path.exists(os.path.join(d, "llama-brief2.json")) else os.path.join(d, "llama-brief.json")
        if os.path.exists(p):
            try:
                out[r] = json.load(open(p))
            except ValueError:
                pass
    return out


def cell(rs):
    """Every clean sample of one (model, rung, mode): ms and rate, sha, n."""
    if not rs:
        return "—"
    parts = []
    for r in rs:
        if r["refused"]:
            parts.append(f"REFUSED ({r['sha']})")
        elif r["rc"] == "124":
            parts.append(f"TIMEOUT after {r['wall']:.0f} s ({r['sha']})")
        elif r["ms"] is not None:
            flag = f" ⚠busy {r['busy']}%" if (r["busy"] or 0) > 5 else ""
            parts.append(f"{r['ms']:,} ms · {r['rate']:,} tok/s · {r['sha']} n={r['n']}{flag}")
    return "<br>".join(parts) or "—"


CURRENT = {"242d7e1a4"}


def main():
    for arg in sys.argv[1:]:
        label, d = arg.split("=", 1)
        rs_all, ll = runs(d), llama(d)
        for era, keep in (("#3726 binary (merge 242d7e1a4)", lambda r: r["sha"] in CURRENT),
                          ("SUPERSEDED — pre-#3726 tokenizer binaries", lambda r: r["sha"] not in CURRENT)):
          rs = [r for r in rs_all if keep(r) and r["rung"]]
          sup = era.startswith("SUPERSEDED")
          if sup and rs:
            print(f"\n<details><summary>{label}: SUPERSEDED rows — pre-#3726 tokenizer binaries (kept, not mixed in)</summary>")
          for model in ("9B", "27B"):
            mr = [r for r in rs if r["model"] == model]
            if not mr:
                continue
            print(f"\n#### {label} — Qwen3.5-{model}-Q4_K_M — {era}\n")
            print("| rung | positions (apr) | cuBLAS f32 attention | flash (f16 in, f32 acc) | llama.cpp d1d3c3396 prompt | apr ÷ llama (best clean rate) | answer |")
            print("|---|---|---|---|---|---|---|")
            for rung in RUNGS:
                rr = [r for r in mr if r["rung"] == rung]
                if not rr:
                    continue
                f32 = [r for r in rr if r["mode"] == "f32" or (r["refused"] and "flash" not in r["tag"])]
                fl = [r for r in rr if r["mode"] == "flash"]
                toks = sorted({r["tokens"] for r in rr if r["tokens"]})
                lj = ll.get(rung) if model == "9B" else None
                lcell = f"{lj['prompt_ms']:,.0f} ms · {lj['prompt_per_second']:,.0f} tok/s ({lj['prompt_n']:,} tok)" if lj else "—"
                clean = [r["rate"] for r in rr if r["rate"] and (r["busy"] or 0) <= 5]
                ratio = f"{max(clean) / lj['prompt_per_second']:.3f}" if lj and clean else "—"
                ans = "; ".join(sorted({r["answer"] for r in rr if r["answer"] != "—"})) or "—"
                print(f"| {rung} | {', '.join(f'{t:,}' for t in toks) or '—'} | {cell(f32)} | {cell(fl)} | {lcell} | {ratio} | {ans} |")
          if sup and rs:
            print("\n</details>")
        rs = rs_all
        print(f"\n<details><summary>{label}: every run ({len(rs)})</summary>\n")
        print("| tag | model | mode | sha | positions | prefill ms | tok/s | chunk | n | wall s | GPU busy at start | answer / refusal |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            what = r["refused"] if r["refused"] else r["answer"]
            print(f"| {r['tag']} | {r['model']} | {r['mode'] or '—'} | {r['sha']} | {r['tokens'] or '—'} | {r['ms'] or '—'} | {r['rate'] or '—'} | {r['chunk'] or '—'} | {r['n'] or '—'} | {r['wall'] or '—'} | {r['busy'] if r['busy'] is not None else '?'}% | {what} |")
        print("\n</details>")


main()
