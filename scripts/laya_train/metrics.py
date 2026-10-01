"""Gate metrics for the Laya back office -- numpy only, shared by fixtures.py and the 08-08 gate.

No torch import (at module level or anywhere): the self-test must run without the ML stack.

    macro_f1(P, y)          sklearn f1_score(average="macro") semantics: mean F1 over the labels
                            present in y UNION pred -- the set aprender-core
                            metrics::classification::f1_score(.., Average::Macro) averages over.
    f_avg(P, y, labels)     mean F1 over the GIVEN label indices (TweetEval stance: against=1,
                            favor=2); a label absent from both y and pred scores 0 (zero_division=0).
    ece_top_label(P, y, bins=15)
                            THE HOUSE ECE (contracts/laya-finetune-gate-v1.yaml `ece_top_label`,
                            aprender-core calibration::expected_calibration_error_top_label):
                            conf_i = max_k p_ik, pred_i = argmax_k p_ik,
                            bin(i) = min(floor(conf_i * bins), bins - 1),
                            ECE = sum_b (n_b / N) * |acc_b - conf_b|.
                            Spike 024 binned right-closed (lo, hi]; the two differ only for a
                            confidence exactly on a bin edge -- a recorded deviation, and the Rust
                            verifier is the authority (one Rust ECE, OPS-03).
    nll(P, y)               mean -log(clip(p_true, 1e-12, 1)), spike 024's definition.

`P` is an [N, K] array of probabilities (rows sum to 1), `y` an [N] array of true label indices.

    python metrics.py --selftest    hand-computed cases + a replay of every frozen house case in
                                    scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json;
                                    exits non-zero on any mismatch, prints METRICS SELFTEST OK.
"""
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ECE_CASES = REPO / "scripts" / "setfit_fixtures" / "claims_stats" / "ece_top_label_cases.json"
FROZEN_TOL = 1e-6


def _as_arrays(P, y):
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if P.ndim != 2 or P.shape[0] == 0 or P.shape[1] < 2:
        raise ValueError("P must be a non-empty [N, K] array with K >= 2, got shape %s" % (P.shape,))
    if y.shape != (P.shape[0],):
        raise ValueError("y must hold one label per row: %d rows, %s labels" % (P.shape[0], y.shape))
    if (y < 0).any() or (y >= P.shape[1]).any():
        raise ValueError("every label must index a column of P (K = %d)" % P.shape[1])
    if not np.isfinite(P).all():
        raise ValueError("P holds a non-finite probability")
    return P, y


def _f1(pred, y, label):
    tp = int(((pred == label) & (y == label)).sum())
    fp = int(((pred == label) & (y != label)).sum())
    fn = int(((pred != label) & (y == label)).sum())
    den = 2 * tp + fp + fn
    return 0.0 if den == 0 else 2.0 * tp / den


def macro_f1(P, y):
    P, y = _as_arrays(P, y)
    pred = P.argmax(1)
    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    f1s = [_f1(pred, y, c) for c in labels]
    return math.fsum(f1s) / len(f1s)


def f_avg(P, y, labels):
    P, y = _as_arrays(P, y)
    if not labels:
        raise ValueError("f_avg needs at least one label index")
    pred = P.argmax(1)
    f1s = [_f1(pred, y, int(c)) for c in labels]
    return math.fsum(f1s) / len(f1s)


def ece_top_label(P, y, bins=15):
    P, y = _as_arrays(P, y)
    if bins < 1:
        raise ValueError("bins must be >= 1")
    conf = P.max(1)
    pred = P.argmax(1)
    correct = (pred == y).astype(np.float64)
    idx = np.minimum(np.floor(conf * bins).astype(np.int64), bins - 1)
    n = len(y)
    terms = []
    for b in range(bins):
        sel = idx == b
        nb = int(sel.sum())
        if nb:
            acc = math.fsum(correct[sel].tolist()) / nb
            mean_conf = math.fsum(conf[sel].tolist()) / nb
            terms.append((nb / n) * abs(acc - mean_conf))
    return math.fsum(terms)


def rank_key(ece_post, rank_scale):
    """laya-finetune-gate-v1 seed_policy.rank_rule: floor(ece_post x rank_scale) as an int, the
    INTEGER scale multiplied in (gate.rank_key refuses a non-finite ECE, then calls this)."""
    return int(math.floor(float(ece_post) * int(rank_scale)))


def nll(P, y):
    P, y = _as_arrays(P, y)
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean())


# ------------------------------------------------------------------------------ numeric agreement
#
# contracts/laya-finetune-gate-v1.yaml `numeric_agreement`: every gate quantity is f64 with
# exactly-rounded sums (math.fsum here, aprender metrics::fsum in Rust), in the same order, so the
# two languages return the SAME BITS. numeric_cases.json freezes constructed boundary cases and
# seeded random cases with their exact results; `--selftest` here and the Rust test
# aprender_decide verify::tests::gate_numeric_cases_agree_bit_for_bit replay every one.

NUMERIC_CASES = REPO / "scripts" / "laya_train" / "numeric_cases.json"
NUMERIC_SCHEMA = "laya-numeric-cases-v1"
# laya-finetune-gate-v1 constants.gate_min_macro_f1_margin (the Rust replay asserts they agree).
CASE_MIN_MARGIN = 0.05


def f64_hex(x):
    """IEEE-754 binary64 bits, 16 lowercase hex digits, big-endian (Rust `format!("{:016x}", x.to_bits())`)."""
    return struct.pack(">d", float(x)).hex()


def f64_from_hex(h):
    return struct.unpack(">d", bytes.fromhex(h))[0]


def f32_hex(x):
    """IEEE-754 binary32 bits, 8 lowercase hex digits; x must already be a float32 value."""
    v = np.float32(x)
    if float(v) != float(x):
        raise ValueError("%r is not a float32 value" % (x,))
    return struct.pack(">f", float(v)).hex()


def f32_from_hex(h):
    return struct.unpack(">f", bytes.fromhex(h))[0]


def probs_from_hex(rows):
    """float32 bits -> the float64 array the trainer scores (train.py `as64`: an exact widening)."""
    return np.array([[f32_from_hex(h) for h in r] for r in rows], dtype=np.float64)


def _pred_row(k, pred, conf):
    """A float32 probability row whose argmax is `pred` with confidence `conf` (> 1/k)."""
    other = np.float32((1.0 - float(np.float32(conf))) / (k - 1))
    row = [other] * k
    row[pred] = np.float32(conf)
    return [float(v) for v in row]


def _splits(n, k):
    """Every way to split n rows over k predicted classes, in lexicographic order."""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _splits(n - i, k - 1):
            yield (i,) + rest


def _cm_counts(cm):
    k = len(cm)
    tp = [cm[c][c] for c in range(k)]
    fp = [sum(cm[r][c] for r in range(k) if r != c) for c in range(k)]
    fn = [sum(cm[c][j] for j in range(k) if j != c) for c in range(k)]
    present = [c for c in range(k) if sum(cm[c]) > 0 or fp[c] > 0]
    return tp, fp, fn, present


def _cm_f64(cm):
    """metrics.py macro_f1 of a confusion matrix (the same 2.0 * tp / den and fsum mean)."""
    tp, fp, fn, present = _cm_counts(cm)
    f1s = [0.0 if 2 * tp[c] + fp[c] + fn[c] == 0 else 2.0 * tp[c] / (2 * tp[c] + fp[c] + fn[c])
           for c in present]
    return math.fsum(f1s) / len(f1s)


def _cm_exact(cm):
    from fractions import Fraction
    tp, fp, fn, present = _cm_counts(cm)
    den = [2 * tp[c] + fp[c] + fn[c] for c in present]
    return sum((Fraction(0) if d == 0 else Fraction(2 * tp[c], d)) for c, d in zip(present, den)) / len(present)


def _cm_head_f32(cm):
    """aprender-core f1_score(.., Average::Macro) as HEAD computes it in f32 (2PR/(P+R), a
    sequential f32 sum, / count) -- used ONLY to pick constructed cases that path gets wrong."""
    f = np.float32
    tp, fp, fn, present = _cm_counts(cm)
    s = f(0.0)
    for c in present:
        pr = f(0.0) if tp[c] + fp[c] == 0 else f(tp[c]) / f(tp[c] + fp[c])
        rc = f(0.0) if tp[c] + fn[c] == 0 else f(tp[c]) / f(tp[c] + fn[c])
        s = s + (f(0.0) if pr + rc == f(0.0) else (f(2.0) * pr * rc) / (pr + rc))
    return float(s / f(len(present)))


def _cm_rows(cm):
    k = len(cm)
    y = [r for r in range(k) for c in range(k) for _ in range(cm[r][c])]
    pred = [c for r in range(k) for c in range(k) for _ in range(cm[r][c])]
    return y, pred


def _exact_margin_case(counts, name, py_pass):
    """The first (zero-shot, fine-tuned) pair of confusion matrices over true-class counts
    `counts` (lexicographic, then f64 order) whose EXACT rational macro-F1 margin is 1/20 -- the
    gate boundary -- and on which metrics.py's f64 verdict is `py_pass` while HEAD's f32 path
    decides the other way: the reviewers' V7-a / A5-2 disagreement, reconstructed."""
    import bisect
    import itertools
    from fractions import Fraction
    k = len(counts)
    seen = {}
    for cm in itertools.product(*[list(_splits(n, k)) for n in counts]):
        seen.setdefault((_cm_f64(cm), _cm_head_f32(cm)), cm)
    table = sorted(seen.items())
    keys = [v64 for (v64, _), _ in table]
    for (zs64, zs32), zs_cm in table:
        lo = bisect.bisect_left(keys, zs64 + CASE_MIN_MARGIN - 1e-12)
        hi = bisect.bisect_right(keys, zs64 + CASE_MIN_MARGIN + 1e-12)
        for (ft64, ft32), ft_cm in table[lo:hi]:
            if (ft64 - zs64 >= CASE_MIN_MARGIN) != py_pass or (ft32 - zs32 >= CASE_MIN_MARGIN) == py_pass:
                continue
            if _cm_exact(ft_cm) - _cm_exact(zs_cm) != Fraction(1, 20):
                continue
            y, ft_pred = _cm_rows(ft_cm)
            y_zs, zs_pred = _cm_rows(zs_cm)
            if y != y_zs:
                raise RuntimeError("row order drifted between the two matrices")
            confs = [0.55 + 0.05 * (i % 8) for i in range(len(y))]
            return {
                "name": name, "k": k, "bins": 15, "labels": y, "f_avg_labels": None,
                "probabilities_f32_hex": [[f32_hex(v) for v in _pred_row(k, p, c)]
                                          for p, c in zip(ft_pred, confs)],
                "zero_shot_probabilities_f32_hex": [[f32_hex(v) for v in _pred_row(k, p, c)]
                                                    for p, c in zip(zs_pred, reversed(confs))],
            }
    raise RuntimeError("no exact-margin case over counts %s" % (counts,))


def _ece_head_f32(rows, y, bins):
    """aprender-core expected_calibration_error_top_label as HEAD computes it in f32 (f32 bin
    index, sequential f32 bin sums, f32 accumulation) -- used ONLY to pick the grid-line case."""
    f = np.float32
    sums = [f(0.0)] * bins
    hits = [f(0.0)] * bins
    counts = [0] * bins
    for row, label in zip(rows, y):
        r = [f(v) for v in row]
        pred = max(range(len(r)), key=lambda j: (r[j], -j))
        conf = r[pred]
        b = min(int(conf * f(bins)), bins - 1)
        sums[b] = sums[b] + conf
        hits[b] = hits[b] + (f(1.0) if pred == label else f(0.0))
        counts[b] += 1
    n = f(len(y))
    ece = f(0.0)
    for b in range(bins):
        if counts[b]:
            c = f(counts[b])
            ece = ece + (c / n) * abs(sums[b] / c - hits[b] / c)
    return float(ece)


def _random_rows(rng, n, k, sharp):
    """n float32 rows of k probabilities (argmax sharpened by `sharp`) and labels drawn from them."""
    rows, y = [], []
    for _ in range(n):
        w = rng.random_sample(k) + 1e-3
        w = w ** sharp
        p = (w / w.sum()).astype(np.float32)
        rows.append([float(v) for v in p])
        cum = np.cumsum(p.astype(np.float64))
        y.append(int(min(np.searchsorted(cum, rng.random_sample() * cum[-1], side="right"), k - 1)))
    return rows, y


def _rank_grid_case(name, rank_scale, n=459, k=3, bins=15):
    """The first seeded 459 x 3 set whose rank key floor(ece x rank_scale) differs between
    metrics.py's f64 ECE and HEAD's f32 ECE: an ECE on a 1e-4 grid line, decided by the last bit
    (the reviewers' 0.0679999937 vs 0.0680000111 pair, reconstructed)."""
    for seed in range(1, 200000):
        rng = np.random.RandomState(0x0827_0000 + seed)
        rows, y = _random_rows(rng, n, k, 2)
        e64 = ece_top_label(np.array(rows), np.array(y), bins)
        e32 = _ece_head_f32(rows, y, bins)
        if rank_key(e64, rank_scale) != rank_key(e32, rank_scale):
            return {"name": name, "k": k, "bins": bins, "labels": y, "f_avg_labels": [1, 2],
                    "probabilities_f32_hex": [[f32_hex(v) for v in r] for r in rows],
                    "search_seed": seed}
    raise RuntimeError("no grid-line case found")


def _random_case(i):
    """Seeded random case i: N 3..600 rows (case 1 at the floor, N = 3), K = 2 + i mod 5 (every
    K in 2..6 four times), 15 bins; a zero-shot set on even i."""
    rng = np.random.RandomState(0x0827_1000 + i)
    n = 3 + int(rng.randint(598))
    n = 3 if i == 1 else n
    k = 2 + i % 5
    sharp = [1, 2, 4][int(rng.randint(3))]
    rows, y = _random_rows(rng, n, k, sharp)
    case = {"name": "random_%02d" % i, "k": k, "bins": 15, "labels": y,
            "f_avg_labels": [1, 2] if k >= 3 else None,
            "probabilities_f32_hex": [[f32_hex(v) for v in r] for r in rows]}
    if i % 2 == 0:
        zs_rows, _ = _random_rows(rng, n, k, 1)
        case["zero_shot_probabilities_f32_hex"] = [[f32_hex(v) for v in r] for r in zs_rows]
    return case


CASE_RANK_SCALE = 10000
N_RANDOM_CASES = 20


def case_expected(case):
    """The exact f64 results of one case, as metrics.py computes them (hex; rank_key an int)."""
    y = np.array(case["labels"], dtype=np.int64)
    bins = int(case["bins"])
    fav = case.get("f_avg_labels")

    def block(P, prefix):
        out = {prefix + "macro_f1_f64_hex": f64_hex(macro_f1(P, y)),
               prefix + "ece_f64_hex": f64_hex(ece_top_label(P, y, bins)),
               prefix + "f_avg_f64_hex": None if fav is None else f64_hex(f_avg(P, y, list(fav)))}
        return out

    P = probs_from_hex(case["probabilities_f32_hex"])
    out = block(P, "")
    out["rank_key"] = rank_key(ece_top_label(P, y, bins), CASE_RANK_SCALE)
    if "zero_shot_probabilities_f32_hex" in case:
        Z = probs_from_hex(case["zero_shot_probabilities_f32_hex"])
        out.update(block(Z, "zero_shot_"))
        margin = macro_f1(P, y) - macro_f1(Z, y)
        out["margin_f64_hex"] = f64_hex(margin)
        out["margin_pass"] = bool(math.isfinite(margin) and margin >= CASE_MIN_MARGIN)
    return out


def build_numeric_cases():
    cases = [
        _exact_margin_case((3, 3, 3), "margin_exact_1_20_9row", py_pass=True),
        _exact_margin_case((10, 10, 10), "margin_exact_1_20_30row", py_pass=False),
        _rank_grid_case("rank_grid_line_459x3", CASE_RANK_SCALE),
    ] + [_random_case(i) for i in range(N_RANDOM_CASES)]
    for c in cases:
        c["expected"] = case_expected(c)
    return {
        "schema": NUMERIC_SCHEMA,
        "generator": "uv run --project scripts/laya_train --frozen python scripts/laya_train/metrics.py "
                     "--write-numeric-cases",
        "hex": "f32 = 8 lowercase hex digits, f64 = 16, big-endian IEEE-754 bits (Rust to_bits)",
        "min_macro_f1_margin": CASE_MIN_MARGIN,
        "rank_scale": CASE_RANK_SCALE,
        "cases": cases,
    }


def numeric_cases_text():
    """The file's exact bytes: the header keys pretty-printed, then ONE compact case per line."""
    doc = build_numeric_cases()
    head = {k: v for k, v in doc.items() if k != "cases"}
    lines = ["{"] + ["  %s: %s," % (json.dumps(k), json.dumps(v)) for k, v in head.items()]
    lines.append('  "cases": [')
    body = [json.dumps(c, separators=(",", ":")) for c in doc["cases"]]
    lines += ["    %s%s" % (c, "," if i + 1 < len(body) else "") for i, c in enumerate(body)]
    lines += ["  ]", "}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------ self-test

def _check(name, got, want, tol, failures):
    ok = isinstance(got, float) and math.isfinite(got) and abs(got - want) <= tol
    print("  %-4s %-48s got %.12f want %.12f" % ("ok" if ok else "FAIL", name, got, want))
    if not ok:
        failures.append(name)


def selftest():
    failures = []
    print("hand-computed cases:")
    # 1. A perfect, fully confident classifier: every metric at its ideal.
    P = np.eye(3)[[0, 1, 2, 0, 1, 2]]
    y = np.array([0, 1, 2, 0, 1, 2])
    _check("perfect macro_f1", macro_f1(P, y), 1.0, 0.0, failures)
    _check("perfect f_avg(1,2)", f_avg(P, y, [1, 2]), 1.0, 0.0, failures)
    _check("perfect ece", ece_top_label(P, y), 0.0, 0.0, failures)
    _check("perfect nll", nll(P, y), 0.0, 1e-12, failures)

    # 2. Always predicts class 0 on y = [0, 1, 2, 0]: F1(0) = 2*2/(2*2+2+0) = 2/3, F1(1) = F1(2) = 0,
    #    macro over {0, 1, 2} = 2/9; f_avg over (1, 2) = 0. conf = 0.6 for every row -> bin 9 of 15,
    #    acc = 0.5, ECE = |0.5 - 0.6| = 0.1. NLL = -(2 ln 0.6 + 2 ln 0.2) / 4.
    P = np.array([[0.6, 0.2, 0.2]] * 4)
    y = np.array([0, 1, 2, 0])
    _check("one-class macro_f1", macro_f1(P, y), 2.0 / 9.0, 1e-15, failures)
    _check("one-class f_avg(1,2)", f_avg(P, y, [1, 2]), 0.0, 0.0, failures)
    _check("one-class ece", ece_top_label(P, y), 0.1, 1e-12, failures)
    _check("one-class nll", nll(P, y), -(2 * math.log(0.6) + 2 * math.log(0.2)) / 4, 1e-12, failures)

    # 3. Four rows, each alone in its 15-bin (conf 0.9 -> 13, 0.75 -> 11, 0.62 -> 9, 0.7 -> 10, no
    #    confidence on an edge): ECE = (|1-0.9| + |0-0.75| + |1-0.62| + |1-0.7|) / 4 = 1.53 / 4.
    P = np.array([[0.9, 0.1], [0.75, 0.25], [0.62, 0.38], [0.3, 0.7]])
    y = np.array([0, 1, 0, 1])
    _check("4-row ece (15 bins)", ece_top_label(P, y), 1.53 / 4, 1e-12, failures)
    # pred = [0, 0, 0, 1]: F1(0) = 2*2/(4+1+0) = 0.8, F1(1) = 2*1/(2+0+1) = 2/3.
    _check("4-row macro_f1", macro_f1(P, y), (0.8 + 2.0 / 3.0) / 2, 1e-15, failures)

    # 4. The saturated row: conf = 1 gives floor(1 * 15) = 15, which must clamp to the top bin.
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0, 0])
    _check("saturated ece (conf 1 -> top bin)", ece_top_label(P, y), 0.5, 1e-12, failures)

    print("frozen house cases (%s):" % ECE_CASES.relative_to(REPO))
    cases = json.loads(ECE_CASES.read_text())["cases"]
    if not cases:
        failures.append("frozen cases: none found")
    for c in cases:
        got = ece_top_label(np.array(c["probabilities"]), np.array(c["labels"]), bins=int(c["n_bins"]))
        _check("frozen %s (bins=%d)" % (c["id"], c["n_bins"]), got, float(c["ece"]), FROZEN_TOL, failures)

    print("numeric agreement cases (%s, bit for bit):" % NUMERIC_CASES.relative_to(REPO))
    doc = json.loads(NUMERIC_CASES.read_text())
    ncases = doc.get("cases") or []
    if doc.get("schema") != NUMERIC_SCHEMA or not ncases:
        failures.append("numeric cases: missing or wrong schema")
    if doc.get("min_macro_f1_margin") != CASE_MIN_MARGIN or doc.get("rank_scale") != CASE_RANK_SCALE:
        failures.append("numeric cases: min_macro_f1_margin / rank_scale are not %r / %r"
                        % (CASE_MIN_MARGIN, CASE_RANK_SCALE))
    for c in ncases:
        got, want = case_expected(c), c["expected"]
        ok = got == want
        print("  %-4s %s %s" % ("ok" if ok else "FAIL", c["name"],
                                "" if ok else "got %s want %s" % (got, want)))
        if not ok:
            failures.append("numeric " + c["name"])

    if failures:
        print("METRICS SELFTEST FAILED: %s" % ", ".join(failures))
        return 1
    print("METRICS SELFTEST OK (%d frozen cases replayed within %g; %d numeric cases bit for bit)"
          % (len(cases), FROZEN_TOL, len(ncases)))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    if sys.argv[1:] == ["--write-numeric-cases"]:
        text = numeric_cases_text()
        NUMERIC_CASES.write_text(text)
        print("WROTE %s (%d cases)" % (NUMERIC_CASES.relative_to(REPO), len(json.loads(text)["cases"])))
        sys.exit(0)
    print("usage: python metrics.py --selftest | --write-numeric-cases", file=sys.stderr)
    sys.exit(2)
