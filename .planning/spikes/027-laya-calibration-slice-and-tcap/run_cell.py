"""Spike 027: one (size, recipe) cell over seeds -- a MEASUREMENT, not a gate run.

    uv run --frozen --project scripts/laya_train python \
        .planning/spikes/027-laya-calibration-slice-and-tcap/run_cell.py --size s64 --recipe es12 --seeds 13,17,23

Reuses scripts/laya_train unchanged: data.py (validation, overlap refusal, the contract's group-disjoint
calibration split with the DECLARED seed 13), contract.py (recipe values, base pin, early-stopping block),
train.SeedRun.train_and_score (train on MPS -> COMPLETE F16 checkpoint -> fp32 CPU reload -> bounded
T fit on the calibration slice -> reload -> eval probabilities -- exactly the path train.main runs for
every seed), train.probes_obj, gate.evaluate_gate / fit_temperature and metrics.py.

What the wrapper adds (spike-only hooks, production code untouched):
  * epochs are passed straight to SeedRun, so `r1` (4 epochs) runs at s16 too, where the contract's epoch
    rule fixes 12 -- a measurement outside the declared recipe space, labelled as such;
  * `es12m10`: the early-stopping monitor's T bound widened to [0.5, 10] by wrapping gate.calibration_monitor
    (what a cap-10 amendment would also change); every other value is the contract's;
  * the calibration-slice logits and the eval logits are kept, so T can be re-fitted unclamped, clamped to
    [0.5, 5] and [0.5, 10], and an EVAL-ORACLE T found (a diagnostic that reads eval labels -- never a
    selection rule);
  * the kept seed is the run's own seed (train.main keeps only seed 13), and the split stays seed 13.

Run dirs: models/decide/spike-027/<size>-<recipe>-seed<s>/ in train.py's run_dir_layout (checkpoint/,
recipe.json, gate-report.json at the contract's [0.5, 5] T, eval-probs.json, zero-shot-probs.json,
probes.json, task.json) plus spike-metrics.json and NOT-A-GATE-RUN.txt. Checkpoints are pruned to the best
KEEP by (pass@5, pass@10, margin ok, ECE@10) because the disk holds ~12 GB.
"""
import argparse
import json
import math
import shutil
import sys
import time
import types
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "laya_train"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import contract  # noqa: E402
import data  # noqa: E402
import gate  # noqa: E402
import metrics  # noqa: E402
import train  # noqa: E402
from common import f32_list, sha256_bytes, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_ROOT = REPO / "models" / "decide" / "spike-027"
import os
KEEP = int(os.environ.get("SPIKE027_KEEP", "5"))
REP = os.environ.get("SPIKE027_REP", "")   # "-rep": an MPS replicate of an existing run (noise measurement)
RECIPES = {  # name -> (epochs, stopping, monitor_tmax)
    "fixed12": (12, "fixed_epochs", None),
    "es12": (12, "early_stopping", None),
    "r1": (4, "fixed_epochs", None),
    "es12m10": (12, "early_stopping", 10.0),
}
T_WIDE = (0.05, 1000.0)       # "unclamped": NLL is convex in 1/T, so clamp(T_unc) == the bounded fit
T_GRID = np.exp(np.linspace(math.log(0.5), math.log(50.0), 461))
T_CURVE = (1.0, 1.76, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0)
KNOWN_RECIPE_IDS = {("s16", "fixed12", 13): "d0f4e40d39425e68d503f557f4f660eb9a73a4fb258da01f777b6f49362bcf20",
                    ("s16", "es12", 13): "3d4b91daf86772bcb23e5342c2dff4bb6467f5d9f254f7833ac7f61a8f2f5375"}


def log(m):
    print(m, flush=True)


def as64(P):
    return np.array([f32_list(r) for r in P], dtype=np.float64)


def zero_shot(size, base, eval_rows, question, labels):
    """The declared base on eval, scored once per size (same loader / path as train.main step 7)."""
    cache = OUT_ROOT / ("zero-shot-%s.json" % size)
    if cache.is_file():
        obj = json.loads(cache.read_text())
        return np.array([r["probabilities"] for r in obj["rows"]], dtype=np.float32), obj
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zs_agent = train.load_for_scoring(base.src, base.digest, base.revision)
    P_zs, _ = train.score_rows(zs_agent, eval_rows, question)
    del zs_agent
    obj = train.eval_probs_obj(labels, eval_rows, P_zs)
    write_json(cache, obj)
    return P_zs, obj


def ece(P, y, bins):
    return metrics.ece_top_label(np.asarray(P, dtype=np.float64), y, bins)


def measure(Zc, yc, Ze, y, P_zs, P_pre, P_ft, calib, bins, th, f_avg_labels):
    sm = gate.softmax
    t_unc = gate.fit_temperature(Zc, yc, *T_WIDE)[0]
    t5 = min(5.0, max(0.5, t_unc))
    t10 = min(10.0, max(0.5, t_unc))
    zs_f1 = metrics.macro_f1(np.asarray(P_zs, np.float64), y)
    ft_f1 = metrics.macro_f1(sm(Ze, 1.0), y)
    margin = ft_f1 - zs_f1
    e = {name: ece(sm(Ze, t), y, bins) for name, t in (("t5", t5), ("t10", t10), ("unclamped", t_unc))}
    grid_ece = np.array([ece(sm(Ze, t), y, bins) for t in T_GRID])
    oracle_ece_t = float(T_GRID[int(np.argmin(grid_ece))])
    oracle_nll_t = gate.fit_temperature(Ze, y, *T_WIDE)[0]
    best_in = lambda hi: float(grid_ece[T_GRID <= hi + 1e-9].min())  # noqa: E731
    ok_m = margin >= float(th["min_macro_f1_margin"])
    mx = float(th["max_ece"])
    acc = lambda Z, yy: float((np.argmax(Z, 1) == yy).mean())  # noqa: E731
    out = {
        "zs_macro_f1": zs_f1, "ft_macro_f1": ft_f1, "margin": margin, "margin_ok": ok_m,
        "ft_f_avg": None if f_avg_labels is None else metrics.f_avg(sm(Ze, 1.0), y, f_avg_labels),
        "ece_pre": ece(P_pre, y, bins), "t_pre": calib["t_pre"],
        "t_fitted_unclamped": t_unc, "t_fitted_5": t5, "t_fitted_10": t10,
        "clamp_hit_5": t_unc >= 5.0, "clamp_hit_10": t_unc >= 10.0,
        "ece_post_5": e["t5"], "ece_post_10": e["t10"], "ece_post_unclamped": e["unclamped"],
        "nll_eval_5": metrics.nll(sm(Ze, t5), y), "nll_eval_10": metrics.nll(sm(Ze, t10), y),
        "pass_5": bool(ok_m and e["t5"] <= mx), "pass_10": bool(ok_m and e["t10"] <= mx),
        "pass_unclamped": bool(ok_m and e["unclamped"] <= mx),
        "EVAL_ORACLE": {"note": "reads eval labels: a diagnostic of the cap, never a selection rule",
                        "t_min_ece": oracle_ece_t, "ece_at_t_min_ece": float(grid_ece.min()),
                        "t_min_nll": oracle_nll_t, "ece_at_t_min_nll": ece(sm(Ze, oracle_nll_t), y, bins),
                        "best_ece_T_le_5": best_in(5.0), "best_ece_T_le_10": best_in(10.0)},
        "ece_curve": {str(t): ece(sm(Ze, t), y, bins) for t in T_CURVE},
        "calib_slice_size": int(len(yc)), "calib_acc": acc(Zc, yc), "eval_acc": acc(Ze, y),
        "calib_nll_at_t5": metrics.nll(sm(Zc, t5), yc),
        # cross-check against the production path's own numbers (f32 files vs float64 logits)
        "check_prod_t_applied": calib["t_applied"],
        "check_prod_ece_post": ece(as64(P_ft), y, bins),
    }
    return out


def prune():
    """Keep the KEEP best checkpoints on disk (the disk holds ~12 GB); metadata files always stay."""
    runs = []
    for m in OUT_ROOT.glob("*/spike-metrics.json"):
        r = json.loads(m.read_text())
        if (m.parent / "checkpoint").is_dir():
            runs.append((r["pass_5"], r["pass_10"], r["margin_ok"], -r["ece_post_10"], m.parent))
    runs.sort(key=lambda x: x[:4], reverse=True)
    for r in runs[KEEP:]:
        shutil.rmtree(r[4] / "checkpoint", ignore_errors=True)
        (r[4] / "CHECKPOINT-PRUNED.txt").write_text("checkpoint deleted by spike-027 prune (disk); metrics kept\n")
        log("PRUNE %s" % r[4].name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=("s16", "s64"), required=True)
    ap.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    ap.add_argument("--seeds", default="13,17,23")
    a = ap.parse_args()
    epochs, stopping, monitor_tmax = RECIPES[a.recipe]
    c = contract.constants()
    th = contract.thresholds()
    bins = int(th["ece_bins"])
    split_seed = int(contract.seed_policy()["declared_seed"])
    data_dir = HERE / "data" / a.size
    task = data.load_task(data_dir / "task.json")
    train_rows = data.load_rows(data_dir / "train.jsonl", task, "train")
    eval_rows = data.load_rows(data_dir / "eval.jsonl", task, "eval")
    data.refuse_overlap(train_rows, eval_rows)
    labels, k = task["labels"], len(task["labels"])
    fit_ids, calib_ids, slice_ids, slice_sha = data.calibration_split(
        train_rows, c["calibration_slice_fraction"], c["calibration_slice_min_per_class"], split_seed, k)
    spc = max(data.class_counts(train_rows, k))
    question = data.laya_question(task)
    fit_rows = [train_rows[i] for i in fit_ids]
    calib_rows = [train_rows[i] for i in calib_ids]
    y = np.array([lab for _, lab in eval_rows])
    yc = np.array([lab for _, lab in calib_rows])
    f_avg_labels = [labels.index("against"), labels.index("favor")]
    log("CELL size=%s recipe=%s epochs=%d stopping=%s monitor_tmax=%s shots/class=%d fit=%d calib=%d slice_sha=%s"
        % (a.size, a.recipe, epochs, stopping, monitor_tmax, spc, len(fit_ids), len(calib_ids), slice_sha[:12]))
    base = train.Base(types.SimpleNamespace(variant="production", base=None, base_sha256=None))
    P_zs, zs_obj = zero_shot(a.size, base, eval_rows, question, labels)
    if monitor_tmax is not None:
        orig = gate.calibration_monitor
        gate.calibration_monitor = lambda z, yy, tmin, tmax: orig(z, yy, tmin, monitor_tmax)
    requested = train.request_device(None)
    for seed in [int(s) for s in a.seeds.split(",")]:
        out = OUT_ROOT / ("%s-%s-seed%d%s" % (a.size, a.recipe, seed, REP))
        if (out / "spike-metrics.json").is_file():
            log("SKIP %s (done)" % out.name)
            continue
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        t0 = time.time()
        recipe = contract.recipe_json("production", spc, epochs, seed, base.block, stopping)
        rb = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        (out / "recipe.json").write_bytes(rb)
        rid = sha256_bytes(rb)
        known = KNOWN_RECIPE_IDS.get((a.size, a.recipe, seed))
        if known is not None and known != rid:
            raise RuntimeError("recipe_id %s != the demo's %s: the wrapper diverged from train.main" % (rid, known))
        (out / "NOT-A-GATE-RUN.txt").write_text(
            "spike 027 measurement (recipe %s, seed %d, calibration split seed %d). Not a declared gate run (D-07).\n"
            % (a.recipe, seed, split_seed))
        run = train.SeedRun(base, requested, question, fit_rows, calib_rows, eval_rows, k, epochs, stopping, rid)
        ck = out / "checkpoint"
        agent, P_ft, P_pre, calib, info = run.train_and_score(seed, ck)
        train.write_json(out / "eval-probs.json", train.eval_probs_obj(labels, eval_rows, P_ft))
        train.assert_eval_after_checkpoint(out, ck)
        train.write_json(out / "probes.json", train.probes_obj(agent))
        _, Zc = train.score_rows(agent, calib_rows, question)
        del agent
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        Zc = Zc.astype(np.float64)
        Ze = calib["t_pre"] * np.log(np.asarray(P_pre, dtype=np.float64))   # softmax-equivalent raw logits
        write_json(out / "zero-shot-probs.json", zs_obj)
        (out / "task.json").write_bytes((data_dir / "task.json").read_bytes())
        g = gate.evaluate_gate(as64(P_zs), as64(P_ft), P_pre, y, f_avg_labels)
        report = {"schema": "laya-gate-report-v1", "pass": g["pass"], "thresholds": g["thresholds"],
                  "zero_shot": g["zero_shot"], "fine_tuned": g["fine_tuned"], "margin": g["margin"],
                  "calibration": {"bucket": calib["bucket"], "t_fitted": calib["t_fitted"],
                                  "t_applied": calib["t_applied"], "clamp_hit": calib["clamp_hit"],
                                  "slice_size": len(slice_ids), "slice_ids": slice_ids, "slice_ids_sha256": slice_sha},
                  "seeds": {"declared": seed, "n": 1, "label": contract.seeds_label(1)},
                  "device_used": info["device_used"], "device_is_cpu": info["device_used"] == "cpu",
                  "torch_version": torch.__version__, "recipe_id": rid,
                  "inputs_sha256": {"task_json": data.sha256_file(data_dir / "task.json"),
                                    "train_jsonl": data.sha256_file(data_dir / "train.jsonl"),
                                    "eval_jsonl": data.sha256_file(data_dir / "eval.jsonl"),
                                    "base_model": base.digest,
                                    "tokenizer_json": data.sha256_file(ck / "tokenizer" / "tokenizer.json")},
                  "eval_probs_sha256": data.sha256_file(out / "eval-probs.json"),
                  "zero_shot_probs_sha256": data.sha256_file(out / "zero-shot-probs.json"),
                  "probes_sha256": data.sha256_file(out / "probes.json")}
        write_json(out / "gate-report.json", report)
        m = measure(Zc, yc, Ze, y, P_zs, P_pre, P_ft, calib, bins, th, f_avg_labels)
        if abs(m["check_prod_ece_post"] - m["ece_post_5"]) > 1e-4 or abs(m["t_fitted_5"] - calib["t_applied"]) > 1e-3:
            log("WARN cross-check: prod ece %.6f vs %.6f, prod T %.6f vs %.6f" % (
                m["check_prod_ece_post"], m["ece_post_5"], calib["t_applied"], m["t_fitted_5"]))
        st = info["stopping_record"] or {}
        m.update({"replicate": bool(REP), "size": a.size, "recipe": a.recipe, "seed": seed, "split_seed": split_seed, "recipe_id": rid,
                  "epochs_max": epochs, "stopping": stopping, "monitor_tmax": monitor_tmax or c["calibration_temp_max"],
                  "epochs_run": info["epochs_run"], "best_epoch": st.get("best_epoch"),
                  "es_trace": st.get("per_epoch"), "stop_reason": st.get("reason"),
                  "fit_rows": info["fit_rows"], "ce_last": info["ce_last"], "device_used": info["device_used"],
                  "train_seconds": info["train_seconds"], "wall_seconds": round(time.time() - t0, 1),
                  "gate_report_pass_at_contract_T": g["pass"], "run_dir": str(out.relative_to(REPO)),
                  "calib_logits": [[float(v) for v in row] for row in Zc], "calib_labels": [int(v) for v in yc]})
        write_json(out / "spike-metrics.json", m)
        (HERE / "results" / "runs").mkdir(parents=True, exist_ok=True)
        (HERE / "results" / "runs" / (out.name + ".json")).write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
        log("RESULT %s margin=%.4f ece_pre=%.3f T_unc=%.3f ece@5=%.4f ece@10=%.4f ece@unc=%.4f oracleT=%.2f "
            "(ece %.4f) calib_acc=%.3f eval_acc=%.3f epochs_run=%s best=%s pass5=%s pass10=%s wall=%.0fs"
            % (out.name, m["margin"], m["ece_pre"], m["t_fitted_unclamped"], m["ece_post_5"], m["ece_post_10"],
               m["ece_post_unclamped"], m["EVAL_ORACLE"]["t_min_ece"], m["EVAL_ORACLE"]["ece_at_t_min_ece"],
               m["calib_acc"], m["eval_acc"], m["epochs_run"], m["best_epoch"], m["pass_5"], m["pass_10"],
               m["wall_seconds"]))
        prune()


if __name__ == "__main__":
    main()
