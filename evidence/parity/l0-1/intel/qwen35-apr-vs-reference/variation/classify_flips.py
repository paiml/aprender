#!/usr/bin/env python3
"""Classify argmax flips per prompt and pool the near-tie distribution (PMAT-3091 variation).

Inputs per prompt: the comparator JSONs A (ref=per-token llama, sub=batched llama),
B (ref=per-token llama, sub=apr), C (ref=batched llama, sub=apr), plus the three APRRAWLG bins
(for apr's own top-2 gap and top-5 lists). Classes, per position:
  llama-self    : per-token argmax != batched argmax
  persistent    : per-token == batched, apr differs from both
  mode-specific : apr equals exactly one llama mode
  three-way     : all three differ
Gap context is the PER-TOKEN llama top1-top2 gap (the mode that matches apr's one-token forward).
No threshold on anything except the brief's named >=1.0 defect-candidate label. Numbers only.

Usage: classify_flips.py <spec.json> <out.json>
  spec = {"model":..,"gguf_py":..,"prompts":[{"k":0,"kind":"...","A":..,"B":..,"C":..,"pt":..,"bat":..,"apr":..}]}
"""
import json
import sys

import numpy as np

CANDIDATE_GAP = 1.0


def load_bin(path):
    buf = open(path, "rb").read()
    assert buf[:8] == b"APRRAWLG", path
    _, n_pos, n_vocab = (int(v) for v in np.frombuffer(buf, "<i4", 3, 8))
    ids = np.frombuffer(buf, "<i4", n_pos, 20)
    lg = np.frombuffer(buf, "<f4", n_pos * n_vocab, 20 + 4 * n_pos).reshape(n_pos, n_vocab)
    return ids, lg


def top_k(row, k):
    idx = np.argpartition(row, -k)[-k:]
    idx = idx[np.argsort(-row[idx].astype(np.float64), kind="stable")]
    return [(int(i), float(row[i])) for i in idx]


def own_gap(row):
    t = np.partition(row.astype(np.float64), -2)[-2:]
    return float(t[1] - t[0])


def load_vocab(model, gguf_py):
    try:
        sys.path.insert(0, gguf_py)
        from gguf import GGUFReader  # noqa: PLC0415
        f = GGUFReader(model).fields["tokenizer.ggml.tokens"]
        return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
    except Exception as e:  # token strings are context only
        print(f"vocab unavailable: {e}", file=sys.stderr)
        return None


def piece(vocab, i):
    return repr(vocab[i]) if vocab else f"<{i}>"


def flip_class(am_pt, am_bat, am_apr):
    """None (no apr-vs-llama flip), persistent, mode_specific or three_way."""
    if am_pt == am_bat:
        return "persistent" if am_apr != am_pt else None
    return "mode_specific" if am_apr in (am_pt, am_bat) else "three_way"


def base_record(a, b, c, token, apr_row, vocab):
    assert b["argmax_ref"] == a["argmax_ref"]
    assert c["argmax_ref"] == a["argmax_apr"]
    assert c["argmax_apr"] == b["argmax_apr"]
    return dict(pos=a["pos"], token_in=token, tok_in=piece(vocab, token),
                argmax_pt=a["argmax_ref"], argmax_bat=a["argmax_apr"], argmax_apr=b["argmax_apr"],
                llama_pt_gap=round(b["ref_top1_top2_gap"], 6),
                llama_bat_gap=round(c["ref_top1_top2_gap"], 6),
                apr_rank_of_pt_argmax=b["apr_rank_of_ref_argmax"],
                apr_rank_of_bat_argmax=c["apr_rank_of_ref_argmax"],
                apr_gap=round(own_gap(apr_row), 6),
                cos_B=round(b["cosine"], 6))


def top5_pieces(row, vocab):
    return [(i, round(v, 4), piece(vocab, i)) for i, v in top_k(row, 5)]


def decorate_persistent(rec, rows3, vocab):
    rec.update(tok_llama=piece(vocab, rec["argmax_pt"]), tok_apr=piece(vocab, rec["argmax_apr"]))
    if rec["llama_pt_gap"] >= CANDIDATE_GAP:
        pt_row, bat_row, apr_row = rows3
        rec["top5_llama_pt"] = top5_pieces(pt_row, vocab)
        rec["top5_llama_bat"] = top5_pieces(bat_row, vocab)
        rec["top5_apr"] = top5_pieces(apr_row, vocab)


def decorate_mode_specific(rec):
    rec["apr_matches"] = "per-token" if rec["argmax_apr"] == rec["argmax_pt"] else "batched"


def classify_prompt(p, vocab):
    jA, jB, jC = (json.load(open(p[x])) for x in "ABC")
    _, pt = load_bin(p["pt"])
    _, bat = load_bin(p["bat"])
    ids, apr = load_bin(p["apr"])
    out = dict(k=p["k"], kind=p["kind"], n_tokens=len(jA["rows"]), llama_self=[], persistent=[],
               mode_specific=[], three_way=[], pt_gaps=[],
               min_cosine=dict(A=jA["summary"]["min_cosine"], B=jB["summary"]["min_cosine"],
                               C=jC["summary"]["min_cosine"]))
    for a, b, c in zip(jA["rows"], jB["rows"], jC["rows"]):
        pos = a["pos"]
        rec = base_record(a, b, c, int(ids[pos]), apr[pos], vocab)
        out["pt_gaps"].append(b["ref_top1_top2_gap"])
        if rec["argmax_pt"] != rec["argmax_bat"]:
            out["llama_self"].append(pos)
        cls = flip_class(rec["argmax_pt"], rec["argmax_bat"], rec["argmax_apr"])
        if cls == "persistent":
            decorate_persistent(rec, (pt[pos], bat[pos], apr[pos]), vocab)
        elif cls == "mode_specific":
            decorate_mode_specific(rec)
        if cls:
            out[cls].append(rec)
    return out


def pool(prompts):
    gaps = np.array([g for p in prompts for g in p["pt_gaps"]])
    pers = [dict(prompt=p["k"], **r) for p in prompts for r in p["persistent"]]
    pers_sorted = sorted(pers, key=lambda r: r["llama_pt_gap"])
    return dict(
        positions=int(gaps.size), persistent_flips=len(pers),
        llama_self_flips=sum(len(p["llama_self"]) for p in prompts),
        mode_specific_flips=sum(len(p["mode_specific"]) for p in prompts),
        three_way_flips=sum(len(p["three_way"]) for p in prompts),
        persistent_gaps_sorted=[r["llama_pt_gap"] for r in pers_sorted],
        all_pt_gap_p5_p25_p50=[round(float(np.percentile(gaps, q)), 6) for q in (5, 25, 50)],
        fraction_of_all_positions_with_pt_gap_below_each_persistent_gap=[
            round(float((gaps < r["llama_pt_gap"]).mean()), 4) for r in pers_sorted],
        largest_gap_persistent=pers_sorted[-1] if pers_sorted else None,
        defect_candidates=[r for r in pers if r["llama_pt_gap"] >= CANDIDATE_GAP],
    )


def main(argv):
    spec = json.load(open(argv[1]))
    vocab = load_vocab(spec["model"], spec["gguf_py"])
    prompts = [classify_prompt(p, vocab) for p in spec["prompts"]]
    pooled = pool(prompts)
    for p in prompts:
        del p["pt_gaps"]
    doc = dict(prompts=prompts, pooled=pooled)
    json.dump(doc, open(argv[2], "w"), indent=1)
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
