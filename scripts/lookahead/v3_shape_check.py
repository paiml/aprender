#!/usr/bin/env python3
"""Is a perf041 batch-invariance witness admissible as a 0.71 V3 serving-shape parity receipt?

The perf041 probe (PP-26 v3.1) gates intra-batch agreement and frozen slots.
V3 adds what makes it a *serving-shape* receipt: the blessed model, every
declared band formed at its full width (m_formed == c), an identity-bearing
witness, and agreement with the m=1 reference to declared_min unless the
divergence is explained by a recorded near-tie (top-2 margin below epsilon).

Exit codes: 0 ADMISSIBLE, 1 NOT ADMISSIBLE (reasons printed), 2 RED (unreadable/empty).
"""
import argparse
import json
import sys
from pathlib import Path

BLESSED_PREFIX = "qwen3.5-4b"
BANDS = (1, 4, 8, 16)
IDENTITY = ("binary_sha256", "commit", "host", "prompt_sha256")
NEAR_TIE_EPS = 0.05
HERE = Path(__file__).resolve().parent


def model_reasons(w):
    path = str((w.get("model") or {}).get("path", "")).lower()
    sha = (w.get("model") or {}).get("sha256")
    out = [] if path.startswith(BLESSED_PREFIX) else [f"model {path!r} is not the blessed Qwen3.5-4B"]
    return out + ([] if sha else ["model sha256 missing"])


def identity_reasons(w):
    return [f"identity field {k} missing" for k in IDENTITY if not w.get(k)]


def explained(band):
    margin = band.get("top2_margin_at_divergence")
    return isinstance(margin, (int, float)) and margin < NEAR_TIE_EPS


def band_reasons(band):
    c, out = band.get("c"), []
    if band.get("result") != "PASS":
        out.append(f"c={c}: perf041 result {band.get('result')}")
    if band.get("m_formed") != c:
        out.append(f"c={c}: m_formed {band.get('m_formed')} != c (shape not served)")
    if (band.get("divergence_at") or 0) < band.get("declared_min", 0) and not explained(band):
        out.append(f"c={c}: diverges from m=1 at {band.get('divergence_at')} < "
                   f"{band.get('declared_min')} with no near-tie margin recorded")
    return out


def bands_reasons(w):
    bands = {b.get("c"): b for b in w.get("bands") or []}
    out = [f"band c={c} not measured" for c in BANDS if c not in bands]
    for c in BANDS:
        out += band_reasons(bands[c]) if c in bands else []
    return out


def evaluate(w):
    if not isinstance(w, dict) or not w.get("bands"):
        return None
    return model_reasons(w) + identity_reasons(w) + bands_reasons(w)


def load(path):
    text = Path(path).read_text()
    return json.loads(text) if text.strip() else None


def verdict(path):
    reasons = evaluate(load(path))
    return (2, ["RED: empty or unreadable witness"]) if reasons is None else (1 if reasons else 0, reasons)


EXPECT = {"admissible.json": 0, "near-tie.json": 0, "short-band.json": 1,
          "wrong-model.json": 1, "unexplained-divergence.json": 1, "empty.json": 2}


def self_test():
    fx = HERE / "fixtures" / "v3"
    bad = [f"{n}: got {verdict(fx / n)[0]}, want {w}" for n, w in EXPECT.items() if verdict(fx / n)[0] != w]
    for b in bad:
        print("FALSIFIER SURVIVED:", b)
    print("self-test:", "RED" if bad else f"ok ({len(EXPECT)} fixtures)")
    return 2 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("witness", nargs="?")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test or not a.witness:
        return self_test()
    code, reasons = verdict(a.witness)
    for r in reasons:
        print("-", r)
    print(["ADMISSIBLE", "NOT ADMISSIBLE", "RED"][code])
    return code


if __name__ == "__main__":
    sys.exit(main())
