"""Spike 027: aggregate results/runs/*.json -> results/summary.json, RESULTS.md tables, report.html.

    python3 .planning/spikes/027-laya-calibration-slice-and-tcap/aggregate.py   (stdlib only)
"""
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = sorted((HERE / "results" / "runs").glob("*.json"))
ORDER = ["fixed12", "es12", "r1", "es12m10"]
LABEL = {"fixed12": "fixed 12 ep (d0f4e40d family)", "es12": "early stop, monitor T<=5 (3d4b91da family)",
         "r1": "fixed 4 ep (spike-024 r1)", "es12m10": "early stop, monitor T<=10 (spike hook)"}


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def fmt(pair, d=3):
    m, s = pair
    return "–" if m is None else ("%.*f ± %.*f" % (d, m, d, s))


def main():
    allruns = [json.loads(p.read_text()) for p in RUNS]
    runs = [r for r in allruns if not r.get("replicate")]
    reps = [r for r in allruns if r.get("replicate")]
    cells = {}
    for r in runs:
        cells.setdefault((r["size"], r["recipe"]), []).append(r)
    keys = sorted(cells, key=lambda k: (k[0], ORDER.index(k[1])))
    summary = []
    for k in keys:
        rs = sorted(cells[k], key=lambda r: r["seed"])
        g = lambda f: [f(r) for r in rs]  # noqa: E731
        row = {"size": k[0], "recipe": k[1], "n": len(rs), "seeds": g(lambda r: r["seed"]),
               "calib_slice": rs[0]["calib_slice_size"], "zs_macro_f1": rs[0]["zs_macro_f1"]}
        for name, f in (("ft_macro_f1", lambda r: r["ft_macro_f1"]), ("margin", lambda r: r["margin"]),
                        ("ece_pre", lambda r: r["ece_pre"]), ("t_unc", lambda r: r["t_fitted_unclamped"]),
                        ("ece5", lambda r: r["ece_post_5"]), ("ece10", lambda r: r["ece_post_10"]),
                        ("ece_unc", lambda r: r["ece_post_unclamped"]),
                        ("oracle_t", lambda r: r["EVAL_ORACLE"]["t_min_ece"]),
                        ("oracle_ece", lambda r: r["EVAL_ORACLE"]["ece_at_t_min_ece"]),
                        ("oracle_nll_t", lambda r: r["EVAL_ORACLE"]["t_min_nll"]),
                        ("oracle_best_ece_le5", lambda r: r["EVAL_ORACLE"]["best_ece_T_le_5"]),
                        ("oracle_best_ece_le10", lambda r: r["EVAL_ORACLE"]["best_ece_T_le_10"]),
                        ("calib_acc", lambda r: r["calib_acc"]), ("eval_acc", lambda r: r["eval_acc"]),
                        ("epochs_run", lambda r: r["epochs_run"]), ("best_epoch", lambda r: r["best_epoch"]),
                        ("train_s", lambda r: r["train_seconds"]), ("wall_s", lambda r: r["wall_seconds"])):
            row[name] = ms(g(f))
            row[name + "_per_seed"] = g(f)
        for p in ("pass_5", "pass_10", "pass_unclamped", "margin_ok"):
            row[p] = sum(bool(r[p]) for r in rs)
        row["ece_curve_mean"] = {t: st.mean(r["ece_curve"][t] for r in rs) for t in rs[0]["ece_curve"]}
        summary.append(row)
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    L = []
    L.append("## Per cell (mean ± sd over seeds; ECE = house top-label ECE, 15 bins, 280 eval rows)\n")
    L.append("| size / cal slice | recipe | n | zs F1 | ft F1 | margin | ECE pre | T fit unclamped | ECE @T≤5 | ECE @T≤10 | ECE @T unclamped | EVAL-ORACLE T (ECE) | pass @5 | pass @10 | pass unclamped |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        L.append("| %s / %d | %s | %d | %.3f | %s | %s | %s | %s | %s | %s | %s | %s (%s) | %d/%d | %d/%d | %d/%d |" % (
            r["size"], r["calib_slice"], LABEL[r["recipe"]], r["n"], r["zs_macro_f1"], fmt(r["ft_macro_f1"]),
            fmt(r["margin"]), fmt(r["ece_pre"]), fmt(r["t_unc"], 2), fmt(r["ece5"]), fmt(r["ece10"]),
            fmt(r["ece_unc"]), fmt(r["oracle_t"], 1), fmt(r["oracle_ece"]), r["pass_5"], r["n"], r["pass_10"], r["n"],
            r["pass_unclamped"], r["n"]))
    L.append("\n## Calibration slice vs eval, epochs\n")
    L.append("| size | recipe | cal acc | eval acc | T fit unclamped | EVAL-ORACLE NLL-T | best ECE T≤5 (oracle) | best ECE T≤10 (oracle) | epochs run | best epoch | train s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["size"], r["recipe"], fmt(r["calib_acc"]), fmt(r["eval_acc"]), fmt(r["t_unc"], 2),
            fmt(r["oracle_nll_t"], 2), fmt(r["oracle_best_ece_le5"]), fmt(r["oracle_best_ece_le10"]),
            fmt(r["epochs_run"], 1), fmt(r["best_epoch"], 1), fmt(r["train_s"], 0)))
    L.append("\n## Per run\n")
    L.append("| run | margin | T unclamped | ECE pre | ECE @5 | ECE @10 | ECE @unc | oracle T | cal acc | eval acc | epochs run / best | pass @5 / @10 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (r["size"], ORDER.index(r["recipe"]), r["seed"])):
        L.append("| %s-%s-seed%d | %.4f | %.2f | %.3f | %.4f | %.4f | %.4f | %.1f | %.3f | %.3f | %s / %s | %s / %s |" % (
            r["size"], r["recipe"], r["seed"], r["margin"], r["t_fitted_unclamped"], r["ece_pre"], r["ece_post_5"],
            r["ece_post_10"], r["ece_post_unclamped"], r["EVAL_ORACLE"]["t_min_ece"], r["calib_acc"], r["eval_acc"],
            r["epochs_run"], r["best_epoch"] or "–", "PASS" if r["pass_5"] else "fail",
            "PASS" if r["pass_10"] else "fail"))
    L.append("\n## MPS replicates (same recipe, seed, data, split; re-run) — run-to-run noise\n")
    L.append("| run | margin orig → rep | ECE @T≤5 orig → rep | ECE @T≤10 orig → rep | T unclamped orig → rep | best epoch orig → rep | pass @5 orig → rep |")
    L.append("|---|---|---|---|---|---|---|")
    by = {(r["size"], r["recipe"], r["seed"]): r for r in runs}
    for q in sorted(reps, key=lambda r: (r["size"], r["recipe"], r["seed"])):
        o = by.get((q["size"], q["recipe"], q["seed"]))
        if o is None:
            continue
        L.append("| %s-%s-seed%d | %.3f → %.3f | %.3f → %.3f | %.3f → %.3f | %.2f → %.2f | %s → %s | %s → %s |" % (
            q["size"], q["recipe"], q["seed"], o["margin"], q["margin"], o["ece_post_5"], q["ece_post_5"],
            o["ece_post_10"], q["ece_post_10"], o["t_fitted_unclamped"], q["t_fitted_unclamped"],
            o["best_epoch"] or "–", q["best_epoch"] or "–", o["pass_5"], q["pass_5"]))
    pooled = [r for r in allruns if r["size"] == "s64" and r["recipe"] == "es12"]
    L.append("\n**s64 early-stop (the contract's default rule), originals + replicates pooled, n=%d:** margin %s, "
             "ECE @T≤5 %s, T unclamped %s, pass @5 %d/%d, pass @10 %d/%d.\n" % (
                 len(pooled), fmt(ms([r["margin"] for r in pooled])), fmt(ms([r["ece_post_5"] for r in pooled])),
                 fmt(ms([r["t_fitted_unclamped"] for r in pooled]), 2), sum(r["pass_5"] for r in pooled), len(pooled),
                 sum(r["pass_10"] for r in pooled), len(pooled)))
    (HERE / "RESULTS.md").write_text("# Spike 027 results (generated by aggregate.py)\n\n" + "\n".join(L) + "\n")
    print("\n".join(L))
    write_html(summary)


def write_html(summary):
    """ECE vs T (mean over seeds), one line per cell, log-T axis; inline SVG, no libraries."""
    W, H, P = 760, 420, 56
    ts = sorted({float(t) for r in summary for t in r["ece_curve_mean"]})
    x = lambda t: P + (math.log(t) - math.log(ts[0])) / (math.log(ts[-1]) - math.log(ts[0])) * (W - 2 * P)  # noqa: E731
    ymax = 0.5
    y = lambda e: H - P - min(e, ymax) / ymax * (H - 2 * P)  # noqa: E731
    cols = {"s16": ["#1f6feb", "#6ea8fe", "#0a3069", "#9ec5fe"], "s64": ["#d1242f", "#ff8182", "#82071e", "#ffb3b3"]}
    parts = []
    for i, r in enumerate(summary):
        c = cols[r["size"]][ORDER.index(r["recipe"])]
        pts = " ".join("%.1f,%.1f" % (x(float(t)), y(e)) for t, e in sorted(r["ece_curve_mean"].items(), key=lambda kv: float(kv[0])))
        parts.append('<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>' % (c, pts))
        parts.append('<text x="%d" y="%d" fill="%s" font-size="12">%s %s</text>' % (W - P - 250, P + 16 * i, c, r["size"], LABEL[r["recipe"]]))
    grid = ['<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#2da44e" stroke-dasharray="4"/>' % (P, W - P, y(0.10), y(0.10)),
            '<text x="%d" y="%.1f" font-size="11" fill="#2da44e">gate ECE 0.10</text>' % (P + 4, y(0.10) - 4)]
    for t in (5.0, 10.0):
        grid.append('<line x1="%.1f" x2="%.1f" y1="%d" y2="%d" stroke="#888" stroke-dasharray="2"/>' % (x(t), x(t), P, H - P))
        grid.append('<text x="%.1f" y="%d" font-size="11" fill="#888">T=%g cap</text>' % (x(t) + 3, P - 6, t))
    for t in ts:
        grid.append('<text x="%.1f" y="%d" font-size="10" fill="#666" text-anchor="middle">%g</text>' % (x(t), H - P + 14, t))
    for e in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        grid.append('<text x="%d" y="%.1f" font-size="10" fill="#666" text-anchor="end">%.1f</text>' % (P - 6, y(e) + 3, e))
    svg = '<svg viewBox="0 0 %d %d" width="100%%" style="max-width:%dpx">%s%s<text x="%d" y="%d" font-size="12" fill="#444">temperature T (log)</text></svg>' % (
        W, H, W, "".join(grid), "".join(parts), W // 2 - 50, H - 12)
    html = ("<!doctype html><meta charset=utf-8><title>Spike 027 ECE vs T</title><style>body{font:14px system-ui;margin:16px;"
            "background:#fff;color:#1f2328}@media(prefers-color-scheme:dark){body{background:#0d1117;color:#e6edf3}}</style>"
            "<h1>Spike 027: eval ECE vs applied temperature</h1><p>Mean over seeds 13/17/23 of the house top-label ECE on "
            "the 280 TweetEval stance test rows, as a function of T. This curve READS EVAL LABELS: it shows where the cap "
            "binds, it is not a way to choose T. Measurements, not gate runs.</p>%s</html>" % svg)
    (HERE / "report.html").write_text(html)


if __name__ == "__main__":
    main()
