"""The fail-closed quality gate and the bounded temperature fit (D-06, D-07).

No torch import (module level or anywhere): `python gate.py --selftest` runs with numpy + pyyaml only.

    fit_temperature(z, y, t_min, t_max)
        laya-finetune-gate-v1 `calibration_fit_bounded`: T_fitted = argmin over T in [t_min, t_max] of
        NLL(softmax(z / T), y). NLL is convex in beta = 1 / T (a log-sum-exp minus a linear term), so
        dNLL/dbeta = mean(E_p[z] - z_y) is monotone and its root is found by bisection in float64. When
        the derivative does not change sign over the interval the optimum lies outside it: T_fitted is
        the bound and clamp_hit is true. T_applied = clamp(T_fitted) (identical inside the interval),
        which is what Laya's loader would serve anyway (Pitfall 6).

    EarlyStopper(decl, max_epochs), calibration_monitor(z, y, t_min, t_max)
        the `early_stopping` rule (laya-finetune-gate-v1 1.1.0) on calibration-slice logits only.

    evaluate_gate(zs_probs, ft_probs, ft_probs_pre, y, calibration, f_avg_labels)
        macro-F1 / F_avg / house top-label ECE / NLL through metrics.py, thresholds READ from the
        contract (contract.thresholds()), pass = margin >= min_macro_f1_margin AND ece_post <= max_ece.
        Returns the metric blocks of the gate report. A non-finite metric can never pass.

    rank_key(ece_post, rank_scale), select_median_seed(rows, rank_scale)
        seed_policy.rank_rule (A3, 1.4.0): floor(ece_post x rank_scale) as an int, ordered by (rank_key,
        seed); the shipped seed is the middle of an odd N. rank_scale is read from the contract.

    verify_report(report) -> {pass, margin, failed}
        re-decides gate_pass from a report's REPORTED metrics under the contract's thresholds: a report
        whose `thresholds` differ from the contract (value, missing or extra key) is refused, and so is a
        reported `pass` that disagrees with the re-decided one. Under seeds.policy median_ece the shipped
        seed is re-derived from the report's per_seed rows and the top-level pass must be that row's. This is the Python-side decision rule and
        what the self-test's fail-closed vectors run through. It is NOT the forgery defence: a rule applied
        to reported numbers passes an edited report, so the Rust verifier (plan 08-09) recomputes every
        metric from verified probabilities before deciding.

    python gate.py --selftest    fabricated reports, threshold refusals, the two FAILING D-19 demo runs
                                 (contract demo.fail_closed_vectors) decided FAIL -- and, when the
                                 gitignored run dirs are present, recomputed from their probability
                                 files -- the bounded T fit and the early-stopping rule. numpy + pyyaml.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contract  # noqa: E402
import data  # noqa: E402  (torch-free: DataError, the back office's typed refusal)
import metrics  # noqa: E402


def softmax(z, t):
    z = np.asarray(z, dtype=np.float64) / float(t)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _dnll_dbeta(z, y, beta):
    p = softmax(z, 1.0 / beta)
    return float(((p * z).sum(1) - z[np.arange(len(y)), y]).mean())


def fit_temperature(z, y, t_min, t_max, iters=200):
    """(t_fitted, t_applied, clamp_hit) for logits z [N, K] and labels y [N]."""
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if z.ndim != 2 or len(y) != z.shape[0] or len(y) == 0:
        raise ValueError("fit_temperature needs z [N, K] and y [N], N >= 1")
    if not np.isfinite(z).all():
        raise ValueError("fit_temperature: non-finite logits")
    t_min, t_max = float(t_min), float(t_max)
    b_lo, b_hi = 1.0 / t_max, 1.0 / t_min           # beta range
    d_lo, d_hi = _dnll_dbeta(z, y, b_lo), _dnll_dbeta(z, y, b_hi)
    if d_lo >= 0.0:                                  # NLL rising already at T = t_max: optimum T >= t_max
        t_fit = t_max
    elif d_hi <= 0.0:                                # still falling at T = t_min: optimum T <= t_min
        t_fit = t_min
    else:
        lo, hi = b_lo, b_hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if _dnll_dbeta(z, y, mid) < 0.0:
                lo = mid
            else:
                hi = mid
        t_fit = 1.0 / (0.5 * (lo + hi))
    t_applied = min(t_max, max(t_min, t_fit))
    clamp_hit = bool(t_fit <= t_min or t_fit >= t_max)
    return float(t_fit), float(t_applied), clamp_hit


class EarlyStopper:
    """laya-finetune-gate-v1 `early_stopping` / equation early_stopping_train_side, torch-free.

    Fed one calibration-slice monitor value per evaluated epoch (never an eval number). The BEST-ANCHORED
    rule: epoch e improves iff m_e is finite and m_e < m_best - min_delta, m_best being the monitor of the
    last improving epoch (+inf before the first) and updated only on improvement -- so an epoch within
    min_delta of the current best never replaces it (ties keep the EARLIER epoch). Epochs before
    first_candidate_epoch (epoch 0 = the untrained base) are never candidates. Stops when
    e - best_epoch >= patience_epochs, or at max_epochs. This is the rule the 08-16 run of record used
    (RUN_OF_RECORD_STOPPING); record() with no finite monitor is REFUSED early-stopping (a DataError)."""

    def __init__(self, decl, max_epochs):
        if (decl["monitor"], decl["mode"], decl["restore"], decl["tie_break"]) != (
                "calibration_nll_at_fitted_t", "min", "best", "earliest"):
            raise ValueError("EarlyStopper implements monitor calibration_nll_at_fitted_t, mode min, restore best, "
                             "tie_break earliest; the recipe declares %s" % (decl,))
        self.every = int(decl["eval_every_epochs"])
        self.first = int(decl["first_candidate_epoch"])
        self.patience = int(decl["patience_epochs"])
        self.min_delta = float(decl["min_delta"])
        self.max_epochs = int(max_epochs)
        if self.every < 1 or self.first < 1 or self.patience < 1 or self.min_delta < 0.0:
            raise ValueError("early_stopping needs eval_every_epochs, first_candidate_epoch, patience_epochs >= 1 "
                             "and min_delta >= 0")
        self.best_epoch, self.best = None, math.inf
        self.trace = []
        self.reason = None

    def evaluates(self, epoch):
        return epoch >= self.first and (epoch % self.every == 0 or epoch == self.max_epochs)

    def update(self, epoch, monitor, t_star=None):
        """Record epoch `epoch`; returns (improved, stop)."""
        m = float(monitor)
        improved = math.isfinite(m) and m < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch = m, int(epoch)
        self.trace.append({"epoch": int(epoch), "monitor": m if math.isfinite(m) else None, "t_star": t_star})
        anchor = self.best_epoch if self.best_epoch is not None else self.first - 1
        if epoch >= self.max_epochs:
            self.reason = "max_epochs"
        elif epoch - anchor >= self.patience:
            self.reason = "patience"
        return improved, self.reason is not None

    def record(self, epochs_run):
        if self.best_epoch is None:
            raise data.DataError("early-stopping", "no evaluated epoch produced a finite calibration monitor "
                                 "value (%d epoch(s) run), so there is no best epoch to restore" % int(epochs_run))
        return {"rule": "early_stopping", "monitor": "calibration_nll_at_fitted_t", "per_epoch": list(self.trace),
                "best_epoch": self.best_epoch, "best_monitor": self.best, "epochs_run": int(epochs_run),
                "reason": self.reason or "max_epochs"}


def calibration_monitor(z, y, t_min, t_max):
    """(m, T*) = (NLL at the bounded fitted temperature, that temperature) -- `early_stopping.monitor`."""
    z = np.asarray(z, dtype=np.float64)
    if not np.isfinite(z).all():
        return float("nan"), None
    _, t_star, _ = fit_temperature(z, y, t_min, t_max)
    return metrics.nll(softmax(z, t_star), y), t_star


def _finite(x):
    return x is not None and isinstance(x, float) and math.isfinite(x)


class GateError(ValueError):
    """A refused gate report; the message starts `REFUSED <field>:`."""


def verify_report(report):
    """Re-decide gate_pass from a report's reported metrics (see the module docstring for the limits)."""
    th = contract.thresholds()
    got = report.get("thresholds")
    if (not isinstance(got, dict) or sorted(got) != sorted(th)
            or any(float(got[k]) != float(th[k]) for k in th)):
        raise GateError("REFUSED thresholds: the report carries %r, the contract declares %r (D-07: thresholds "
                        "are read from the contract, never from a report)" % (got, th))
    margin = float(report["fine_tuned"]["macro_f1"]) - float(report["zero_shot"]["macro_f1"])
    ece = float(report["fine_tuned"]["ece_post"])
    failed = [clause for clause, ok in (
        ("margin", _finite(margin) and margin >= float(th["min_macro_f1_margin"])),
        ("ece_post", _finite(ece) and ece <= float(th["max_ece"]))) if not ok]
    passed = not failed
    if "pass" in report and report["pass"] is not passed:
        raise GateError("REFUSED pass: the report says pass=%r but its metrics decide %r (failed: %s)"
                        % (report["pass"], passed, failed or "none"))
    seeds = report.get("seeds") or {}
    if seeds.get("policy") == contract.seed_selection_decl()["policy"]:
        # A3: the shipped seed is re-derived from the report's own per_seed rows, and the gate is that seed's.
        per = seeds.get("per_seed") or []
        median = select_median_seed(per, contract.seed_selection_decl()["rank_scale"])
        if seeds.get("shipped") != median:
            raise GateError("REFUSED seeds: the report ships seed %r, the median-ECE seed of its per_seed rows is %d"
                            % (seeds.get("shipped"), median))
        row = [r for r in per if int(r["seed"]) == median][0]
        if row.get("pass") is not passed:
            raise GateError("REFUSED seeds: the top-level gate decides pass=%r but the shipped seed %d's row says %r "
                            "(the pass rule reads only the shipped seed)" % (passed, median, row.get("pass")))
    return {"pass": passed, "margin": margin, "failed": failed}


def rank_key(ece_post, rank_scale):
    """seed_policy.rank_rule: floor(ece_post x rank_scale) as an int. rank_scale is the INTEGER the
    contract declares (10000), multiplied in, never a division by a float grid step. A non-finite
    ece_post has no rank and is refused (it would otherwise sort arbitrarily)."""
    x = float(ece_post)
    if not math.isfinite(x):
        raise GateError("REFUSED seeds: ece_post %r is not finite, so it has no rank key" % (ece_post,))
    return metrics.rank_key(x, rank_scale)


def select_median_seed(rows, rank_scale):
    """The SHIPPED seed under seed_policy.selection median_ece (A3): rows carry `seed` and `ece_post`;
    ordered by (rank_key ascending, seed ascending) -- tie_break smaller_seed --, the seed at 0-based
    index (N - 1) // 2. An empty or even N, or a repeated seed, is refused (no median to ship)."""
    rows = list(rows)
    n = len(rows)
    if n == 0 or n % 2 == 0:
        raise GateError("REFUSED seeds: the median rule needs an odd, non-empty number of seeds, got %d" % n)
    seeds = [int(r["seed"]) for r in rows]
    if len(set(seeds)) != n:
        raise GateError("REFUSED seeds: a seed appears more than once in %s" % seeds)
    order = sorted(rows, key=lambda r: (rank_key(r["ece_post"], rank_scale), int(r["seed"])))
    return int(order[(n - 1) // 2]["seed"])


def evaluate_gate(zs_probs, ft_probs, ft_probs_pre, y, f_avg_labels=None):
    """The metric blocks, margin and pass of a gate report (thresholds from the contract)."""
    th = contract.thresholds()
    bins = int(th["ece_bins"])
    y = np.asarray(y, dtype=np.int64)
    zs_P, ft_P, pre_P = (np.asarray(p, dtype=np.float64) for p in (zs_probs, ft_probs, ft_probs_pre))

    def fav(P):
        return None if f_avg_labels is None else metrics.f_avg(P, y, list(f_avg_labels))

    zs = {"macro_f1": metrics.macro_f1(zs_P, y), "f_avg": fav(zs_P),
          "ece": metrics.ece_top_label(zs_P, y, bins), "n": int(len(y))}
    ft = {"macro_f1": metrics.macro_f1(ft_P, y), "f_avg": fav(ft_P),
          "ece_pre": metrics.ece_top_label(pre_P, y, bins), "ece_post": metrics.ece_top_label(ft_P, y, bins),
          "nll": metrics.nll(ft_P, y), "n": int(len(y))}
    margin = ft["macro_f1"] - zs["macro_f1"]
    passed = bool(_finite(margin) and _finite(ft["ece_post"])
                  and margin >= float(th["min_macro_f1_margin"]) and ft["ece_post"] <= float(th["max_ece"]))
    return {"pass": passed, "thresholds": th, "zero_shot": zs, "fine_tuned": ft, "margin": margin}


# ------------------------------------------------------------------------------------------ self-test

# The D-19 demo's two FAILING runs (laya-finetune-gate-v1 1.2.0 `demo.fail_closed_vectors`), their
# REPORTED numbers copied exactly from each gate-report.json. They are fail-closed test vectors: the
# gate must decide FAIL on them, on the ece_post clause alone (the margin clause passes in both).
FAIL_CLOSED_VECTORS = (
    {"name": "fixed_epochs", "run_dir": "models/decide/tweet-stance-16-fixed-epochs",
     "recipe_id": "d0f4e40d39425e68d503f557f4f660eb9a73a4fb258da01f777b6f49362bcf20",
     "zs_macro_f1": 0.34021730836081837, "ft_macro_f1": 0.4701016893133163, "ece_post": 0.37732241111142295,
     "t_applied": 5.0, "clamp_hit": True},
    {"name": "early_stopping", "run_dir": "models/decide/tweet-stance-16",
     "recipe_id": "3d4b91daf86772bcb23e5342c2dff4bb6467f5d9f254f7833ac7f61a8f2f5375",
     "zs_macro_f1": 0.34021730836081837, "ft_macro_f1": 0.4458312929908441, "ece_post": 0.2224078384893281,
     "t_applied": 3.143624512876524, "clamp_hit": False},
)
DEMO_DATA_DIR = "data/decide/tweet-stance-16"


def _fabricated(zs_f1, ft_f1, ece_post, th=None, passed=None):
    r = {"thresholds": dict(contract.thresholds()) if th is None else th,
         "zero_shot": {"macro_f1": zs_f1}, "fine_tuned": {"macro_f1": ft_f1, "ece_post": ece_post}}
    if passed is not None:
        r["pass"] = passed
    return r


def _recompute_run_dir(vec, case):
    """Recompute a fail-closed vector's gate from its probability files and the demo eval labels.
    Local evidence only (both dirs are gitignored); absent -> an explicit SKIP line, never a pass."""
    import data  # torch-free
    repo = contract.REPO
    run, data_dir = repo / vec["run_dir"], repo / DEMO_DATA_DIR
    need = [run / f for f in ("gate-report.json", "recipe.json", "eval-probs.json", "zero-shot-probs.json",
                              "task.json")] + [data_dir / "eval.jsonl"]
    missing = [str(q.relative_to(repo)) for q in need if not q.is_file()]
    if missing:
        print("  SKIP run-dir recompute [%s]: %s not present (gitignored local evidence)"
              % (vec["name"], ", ".join(missing)))
        return
    rep = json.loads((run / "gate-report.json").read_text())
    tag = "[%s] " % vec["name"]
    case(tag + "recipe_id == sha256(recipe.json) == the vector's",
         data.sha256_file(run / "recipe.json") == rep["recipe_id"] == vec["recipe_id"])
    case(tag + "probability files and eval.jsonl match the report's sha256s",
         data.sha256_file(run / "eval-probs.json") == rep["eval_probs_sha256"]
         and data.sha256_file(run / "zero-shot-probs.json") == rep["zero_shot_probs_sha256"]
         and data.sha256_file(data_dir / "eval.jsonl") == rep["inputs_sha256"]["eval_jsonl"])
    task = data.load_task(run / "task.json")
    rows = data.load_rows(data_dir / "eval.jsonl", task, "eval")
    y = np.array([lab for _, lab in rows])

    def probs(name):
        obj = json.loads((run / name).read_text())
        ok = (obj["labels"] == task["labels"] and [r["row"] for r in obj["rows"]] == list(range(len(rows)))
              and all(r["text_sha256"] == data.exact_sha256(t) for r, (t, _) in zip(obj["rows"], rows)))
        case(tag + "%s rows align with eval.jsonl (index + text sha256)" % name, ok)
        return np.array([r["probabilities"] for r in obj["rows"]], dtype=np.float64)
    ft, zs = probs("eval-probs.json"), probs("zero-shot-probs.json")
    labels = task["labels"]
    fav = [labels.index("against"), labels.index("favor")] if "against" in labels and "favor" in labels else None
    g = evaluate_gate(zs, ft, ft, y, fav)
    tol = contract.constant("gate_metric_recompute_abs", "float")
    deltas = [abs(g["zero_shot"]["macro_f1"] - rep["zero_shot"]["macro_f1"]),
              abs(g["fine_tuned"]["macro_f1"] - rep["fine_tuned"]["macro_f1"]),
              abs(g["fine_tuned"]["ece_post"] - rep["fine_tuned"]["ece_post"]), abs(g["margin"] - rep["margin"])]
    case(tag + "recomputed macro_f1 (zs, ft), ece_post, margin within gate_metric_recompute_abs",
         max(deltas) <= tol, "max |d| = %.3g (tol %g)" % (max(deltas), tol))
    case(tag + "gate recomputed from the probability files decides FAIL",
         g["pass"] is False and rep["pass"] is False,
         "margin=%.4f ece_post=%.4f" % (g["margin"], g["fine_tuned"]["ece_post"]))


# The run of record (plan 08-16, models/decide/laya-stance-64, shipped seed 17, deployed as 24a44d7e...):
# checkpoint/rl_agent_config.json `training.stopping`, copied exactly. Replayed through EarlyStopper it
# must give the recorded best_epoch / epochs_run / reason.
RUN_OF_RECORD_STOPPING = {
    "run_dir": "models/decide/laya-stance-64", "seed": 17,
    "monitors": [0.756670608641795, 0.7521509716564435, 0.8539618408615052, 0.8706240461957222, 0.9254170977443964],
    "best_epoch": 2, "epochs_run": 5, "reason": "patience", "max_epochs": 12}


def _running_min_rule(trace, first, min_delta):
    """The rule early_stopping_train_side USED to state (laya-finetune-gate-v1 <= 4.0.0): e improves iff
    m_e < min_{first <= j < e} m_j - min_delta (vacuous at e = first); the LAST improving epoch is best."""
    best = None
    for e, m in enumerate(trace, first):
        prior = trace[:e - first]
        if not prior or m < min(prior) - min_delta:
            best = e
    return best


def _within_min_delta_of_minimum_rule(trace, first, min_delta):
    """The rule FALSIFY-LAYA-GATE-009 and tie_break_rule USED to state: the earliest epoch whose monitor is
    within min_delta of the minimum."""
    lo = min(trace)
    return min(e for e, m in enumerate(trace, first) if m - lo <= min_delta)


def _best_anchored_cases(case, decl):
    """V13-c: the early-stopping contract states ONE rule, the best-anchored rule the trainer implements."""
    md = float(decl["min_delta"])
    first = int(decl["first_candidate_epoch"])
    trace = [1.0, 0.9993, 0.9988]                # epochs 1..3
    st = EarlyStopper(decl, 12)
    for e, m in enumerate(trace, first):
        st.update(e, m)
    got = st.best_epoch
    rmin, wmin = _running_min_rule(trace, first, md), _within_min_delta_of_minimum_rule(trace, first, md)
    case("best-anchored: trace %s (min_delta %g) -> epoch 3, since 0.9988 < m_best 1.0 - min_delta" % (trace, md),
         got == 3 and rmin == 1 and wmin == 2,
         "best-anchored %s; running-min %s; earliest-within-min_delta-of-min %s: the trace distinguishes all three, "
         "and tie_break_rule's old 'among epochs within min_delta of each other the earliest is kept' would keep "
         "epoch 2 (2 and 3 are within min_delta)" % (got, rmin, wmin))
    c = contract.gate_contract()
    formula = c["equations"]["early_stopping_train_side"]["formula"]
    f009 = [t for t in c["falsification_tests"] if t["id"] == "FALSIFY-LAYA-GATE-009"][0]["prediction"]
    tbr = c["early_stopping"]["tie_break_rule"]
    stale = [where for where, text, gone in (
        ("early_stopping_train_side.formula", formula, "min_{first_candidate_epoch <= j < e}"),
        ("FALSIFY-LAYA-GATE-009.prediction", f009, "earliest epoch within min_delta of the minimum"),
        ("early_stopping.tie_break_rule", tbr, "among epochs within min_delta of each other the earliest is kept"))
        if gone in text]
    missing = [where for where, text in (("early_stopping_train_side.formula", formula),
                                         ("FALSIFY-LAYA-GATE-009.prediction", f009),
                                         ("early_stopping.tie_break_rule", tbr)) if "best-anchored" not in text.lower()]
    case("best-anchored: the equation, FALSIFY-LAYA-GATE-009 and tie_break_rule all state the best-anchored rule",
         not stale and not missing and "m_best - min_delta" in formula
         and c["early_stopping"]["improvement_rule"].startswith("epoch e improves iff m_e < m_best - min_delta"),
         "stale wording in %s; no 'best-anchored' in %s" % (stale or "none", missing or "none"))
    rec = RUN_OF_RECORD_STOPPING
    st = EarlyStopper(decl, rec["max_epochs"])
    stopped = None
    for e, m in enumerate(rec["monitors"], first):
        _, stop = st.update(e, m)
        if stop:
            stopped = e
            break
    r = st.record(stopped)
    case("best-anchored: the run of record's seed-%d trace replays to best_epoch %d, epochs_run %d, %s"
         % (rec["seed"], rec["best_epoch"], rec["epochs_run"], rec["reason"]),
         (r["best_epoch"], r["epochs_run"], r["reason"]) == (rec["best_epoch"], rec["epochs_run"], rec["reason"]),
         "got %s / %s / %s" % (r["best_epoch"], r["epochs_run"], r["reason"]))
    cfg = contract.REPO / rec["run_dir"] / "checkpoint" / "rl_agent_config.json"
    if not cfg.is_file():
        print("  SKIP run-of-record stopping record: %s not present (gitignored local evidence)"
              % cfg.relative_to(contract.REPO))
        return
    stp = json.loads(cfg.read_text())["training"]["stopping"]
    case("best-anchored: the constants above equal the run dir's recorded training.stopping",
         [p["monitor"] for p in stp["per_epoch"]] == rec["monitors"]
         and (stp["best_epoch"], stp["epochs_run"], stp["reason"]) == (rec["best_epoch"], rec["epochs_run"], rec["reason"]))


def selftest():
    failures = []

    def case(name, ok, detail=""):
        print("  %-4s %-74s %s" % ("ok" if ok else "FAIL", name, detail))
        if not ok:
            failures.append(name)

    th = contract.thresholds()
    print("gate decision on fabricated reports (thresholds from the contract: %s):" % th)
    # 0.55 - 0.5 = 0.050000000000000044 in f64: the same subtraction the Rust verifier does, which is true
    # because both sides compute margin as ONE f64 subtraction of exactly-summed macro-F1s
    # (laya-finetune-gate-v1 numeric_agreement.quantities.margin, plan 08-27).
    for name, zs, ft, ece, want in (
            ("margin 0.049 fails", 0.5, 0.549, 0.05, False),
            ("margin 0.05 with ECE 0.10 passes (both bounds inclusive)", 0.5, 0.55, 0.10, True),
            ("ECE 0.1001 fails", 0.5, 0.55, 0.1001, False),
            ("margin NaN fails", float("nan"), 0.55, 0.05, False),
            ("ECE NaN fails", 0.5, 0.55, float("nan"), False),
            ("ECE +inf fails", 0.5, 0.55, float("inf"), False)):
        try:
            got = verify_report(_fabricated(zs, ft, ece))["pass"]
            case(name, got is want, "pass=%s" % got)
        except GateError as e:
            case(name, False, "unexpected refusal: %s" % e)
    for name, mutate in (
            ("thresholds.max_ece 0.2 refused", lambda t: t.update(max_ece=0.2)),
            ("thresholds.min_macro_f1_margin 0.0 refused", lambda t: t.update(min_macro_f1_margin=0.0)),
            ("thresholds.ece_bins 10 refused", lambda t: t.update(ece_bins=10)),
            ("thresholds missing a key refused", lambda t: t.pop("ece_bins")),
            ("thresholds with an extra key refused", lambda t: t.update(min_f_avg=0.5))):
        t = dict(th)
        mutate(t)
        try:
            verify_report(_fabricated(0.5, 0.9, 0.01, t))
            case(name, False, "accepted")
        except GateError as e:
            case(name, str(e).startswith("REFUSED thresholds"), str(e)[:90])
    try:
        verify_report(_fabricated(0.5, 0.9, 0.5, passed=True))
        case("reported pass=true on failing metrics refused", False, "accepted")
    except GateError as e:
        case("reported pass=true on failing metrics refused", str(e).startswith("REFUSED pass"), str(e)[:90])

    print("fail-closed vectors: the two FAILING D-19 demo runs (contract demo.fail_closed_vectors):")
    listed = " ".join(contract.demo().get("fail_closed_vectors", []))
    for v in FAIL_CLOSED_VECTORS:
        tag = "[%s] " % v["name"]
        case(tag + "recipe_id is listed in the contract's fail_closed_vectors", v["recipe_id"] in listed)
        out = verify_report(_fabricated(v["zs_macro_f1"], v["ft_macro_f1"], v["ece_post"]))
        case(tag + "decides FAIL on the ece_post clause alone (margin passes)",
             out["pass"] is False and out["failed"] == ["ece_post"],
             "margin=%.4f ece_post=%.4f failed=%s" % (out["margin"], v["ece_post"], out["failed"]))
        try:
            verify_report(_fabricated(v["zs_macro_f1"], v["ft_macro_f1"], v["ece_post"], passed=True))
            case(tag + "the same report with pass flipped to true is refused", False, "accepted")
        except GateError as e:
            case(tag + "the same report with pass flipped to true is refused", True, str(e)[:70])
        _recompute_run_dir(v, case)

    print("median-ECE seed selection (seed_policy A3, FALSIFY-LAYA-GATE-011; rank_scale from the contract):")
    scale = contract.seed_selection_decl()["rank_scale"]

    def rows_of(*pairs):
        return [{"seed": sd, "ece_post": e} for sd, e in pairs]

    for name, rows, want in (
            ("distinct ECEs 0.15 / 0.05 / 0.12 -> the 0.12 seed (23) ships", rows_of((13, 0.15), (17, 0.05), (23, 0.12)), 23),
            ("rank-key tie 0.10009 / 0.10001 (both key %d): smaller seed first -> 17 ships"
             % rank_key(0.10001, scale), rows_of((13, 0.10009), (17, 0.10001), (23, 0.2)), 17),
            ("exact tie on all three -> seed order -> 17 ships", rows_of((13, 0.08), (17, 0.08), (23, 0.08)), 17),
            ("the median can be the declared seed 13", rows_of((13, 0.07), (17, 0.03), (23, 0.09)), 13)):
        try:
            got = select_median_seed(rows, scale)
            case(name, got == want, "shipped=%s" % got)
        except GateError as e:
            case(name, False, "unexpected refusal: %s" % e)
    case("rank_key(0.10001) == rank_key(0.10009) == floor(x * scale), an int",
         rank_key(0.10001, scale) == rank_key(0.10009, scale) == int(math.floor(0.10001 * scale))
         and isinstance(rank_key(0.1, scale), int))
    for name, rows in (("a NaN ece_post is refused", rows_of((13, float("nan")), (17, 0.05), (23, 0.12))),
                       ("an infinite ece_post is refused", rows_of((13, float("inf")), (17, 0.05), (23, 0.12))),
                       ("N = 2 is refused (no median)", rows_of((13, 0.05), (17, 0.12))),
                       ("N = 0 is refused", []),
                       ("a repeated seed is refused", rows_of((13, 0.05), (13, 0.12), (23, 0.2)))):
        try:
            select_median_seed(rows, scale)
            case(name, False, "accepted")
        except GateError as e:
            case(name, str(e).startswith("REFUSED seeds"), str(e)[:80])

    def median_report(shipped, top_pass=None):
        """seed 17 (median, margin 0.10 but ECE 0.12: FAILS), seed 23 (ECE 0.05: PASSES), seed 13 (ECE 0.15)."""
        per = [{"seed": 13, "macro_f1": 0.60, "ece_post": 0.15, "pass": False},
               {"seed": 17, "macro_f1": 0.60, "ece_post": 0.12, "pass": False},
               {"seed": 23, "macro_f1": 0.62, "ece_post": 0.05, "pass": True}]
        top = [r for r in per if r["seed"] == shipped][0]
        rep = _fabricated(0.5, top["macro_f1"], top["ece_post"], passed=top["pass"] if top_pass is None else top_pass)
        rep["seeds"] = {"policy": contract.seed_selection_decl()["policy"], "shipped": shipped, "per_seed": per}
        return rep
    try:
        out = verify_report(median_report(17))
        case("a failing median fails the gate even when another seed passes", out["pass"] is False,
             "pass=%s failed=%s" % (out["pass"], out["failed"]))
    except GateError as e:
        case("a failing median fails the gate even when another seed passes", False, "unexpected refusal: %s" % e)
    for name, rep in (("a report shipping the passing non-median seed (23) is refused", median_report(23)),
                      ("a report whose seeds.shipped is edited to 13 is refused", median_report(13))):
        try:
            verify_report(rep)
            case(name, False, "accepted")
        except GateError as e:
            case(name, str(e).startswith("REFUSED seeds"), str(e)[:80])
    rep = median_report(17)
    rep["seeds"]["per_seed"][1]["pass"] = True
    try:
        verify_report(rep)
        case("a shipped row whose pass disagrees with the top-level gate is refused", False, "accepted")
    except GateError as e:
        case("a shipped row whose pass disagrees with the top-level gate is refused",
             str(e).startswith("REFUSED seeds"), str(e)[:80])

    print("evaluate_gate on probabilities:")
    y = np.array([0, 1, 2, 0, 1, 2])
    good = np.eye(3)[y] * 0.9 + 0.1 / 3
    chance = np.full((6, 3), 1 / 3)
    g = evaluate_gate(chance, good, good, y)
    case("confident-correct vs chance passes", g["pass"] is True,
         "margin=%.4f ece=%.4f" % (g["margin"], g["fine_tuned"]["ece_post"]))
    g = evaluate_gate(good, good, good, y)
    case("equal models fail (margin 0)", g["pass"] is False and g["margin"] == 0.0)
    try:
        evaluate_gate(chance, np.full((6, 3), np.nan), good, y)
        case("NaN probabilities refused", False, "accepted")
    except ValueError as e:
        case("NaN probabilities refused", True, str(e)[:70])

    print("bounded temperature fit (FALSIFY-LAYA-GATE-004):")
    lo, hi = contract.constant("calibration_temp_min", "float"), contract.constant("calibration_temp_max", "float")
    rng = np.random.RandomState(0)
    yy = rng.randint(0, 3, size=40)        # memorised and confidently wrong half the time: optimum T >> 5
    wrong = (yy + 1) % 3
    zz = np.where(rng.rand(40)[:, None] < 0.5, 20.0 * np.eye(3)[yy], 20.0 * np.eye(3)[wrong])
    tf, ta, hit = fit_temperature(zz, yy, lo, hi)
    case("optimum T > 5 -> t_fitted = t_applied = 5.0, clamp_hit", tf == hi and ta == hi and hit,
         "t=%s clamp=%s" % (tf, hit))
    zz = 0.3 * np.eye(3)[yy]               # always right and under-confident: optimum T < 0.5
    tf, ta, hit = fit_temperature(zz, yy, lo, hi)
    case("optimum T < 0.5 -> t_fitted = t_applied = 0.5, clamp_hit", tf == lo and ta == lo and hit,
         "t=%s clamp=%s" % (tf, hit))
    zz = rng.randn(60, 3) * 2.0 + 1.5 * np.eye(3)[rng.randint(0, 3, 60)]
    yy = zz.argmax(1).copy()
    yy[:15] = (yy[:15] + 1) % 3
    tf, ta, hit = fit_temperature(zz, yy, lo, hi)
    grid = np.linspace(lo, hi, 45001)
    tg = float(grid[int(np.argmin([metrics.nll(softmax(zz, t), yy) for t in grid]))])
    case("interior optimum matches a 1e-4 grid, no clamp", (not hit) and abs(tf - tg) <= 2e-4 and ta == tf,
         "t=%.6f grid=%.6f" % (tf, tg))

    print("early stopping on a synthetic calibration-monitor trace (FALSIFY-LAYA-GATE-009):")
    decl = contract.early_stopping_decl()
    md = float(decl["min_delta"])
    st = EarlyStopper(decl, 12)
    case("epoch 0 (the untrained base) is never evaluated", not st.evaluates(0) and st.evaluates(1))
    trace = [0.90, 0.80, 0.80 - md / 2, 0.80 - md * 0.8, 0.85, 0.70]
    stopped_at = None
    for e, m in enumerate(trace, 1):
        _, stop = st.update(e, m)
        if stop:
            stopped_at = e
            break
    rec = st.record(stopped_at)
    case("an epoch within min_delta of the CURRENT BEST never replaces it (epoch 2 kept, not 3 or 4)",
         rec["best_epoch"] == 2, "best_epoch=%s" % rec["best_epoch"])
    case("stops patience_epochs after the best epoch (at 2 + %d), never sees epoch 6" % decl["patience_epochs"],
         stopped_at == 2 + int(decl["patience_epochs"]) and rec["reason"] == "patience",
         "stopped_at=%s reason=%s" % (stopped_at, rec["reason"]))
    st = EarlyStopper(decl, 12)
    st.update(1, float("nan"))
    st.update(2, 0.5)
    rec = st.record(2)
    case("a non-finite monitor never improves", rec["best_epoch"] == 2 and rec["per_epoch"][0]["monitor"] is None)
    st = EarlyStopper(decl, 12)
    for e in range(1, 4):
        st.update(e, float("nan"))
    import data                                  # torch-free; DataError is the back office's typed refusal
    name = "early-stopping: no finite monitor at all -> REFUSED early-stopping (typed), nothing to restore"
    try:
        st.record(3)
        case(name, False, "accepted")
    except data.DataError as e:
        case(name, e.rule == "early-stopping" and str(e).startswith("REFUSED early-stopping: "), str(e)[:80])
    except Exception as e:
        case(name, False, "raised %s, not the typed refusal: %s" % (type(e).__name__, str(e)[:60]))
    st = EarlyStopper(decl, 2)
    st.update(1, 0.9)
    _, stop = st.update(2, 0.8)
    case("an improving run stops at max_epochs", stop and st.record(2)["reason"] == "max_epochs")
    _best_anchored_cases(case, decl)

    if failures:
        print("GATE SELFTEST FAILED: %s" % ", ".join(failures))
        return 1
    print("GATE SELFTEST OK")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    print("usage: python gate.py --selftest", file=sys.stderr)
    sys.exit(2)
