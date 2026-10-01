#!/usr/bin/env python3
"""Freeze the CLAIMS-LAYER numeric reference fixtures (plan 05-04, D-05/D-06/D-07).

Run:  cd scripts/setfit_fixtures && uv run python gen_claims_fixtures.py

Developer workflow only -- never wired into CI, exactly like generate_fixtures.py.

WHAT THIS IS, AND WHAT IT IS NOT
-------------------------------
This generator is INDEPENDENT of the Phase 1 encoder-conformance corpus. It writes into
its OWN directory (``scripts/setfit_fixtures/claims_stats/``) with its OWN
``manifest.sha256`` and never opens, reads or rewrites anything under
``crates/aprender-core/tests/fixtures/setfit/``. Running this script therefore cannot
re-baseline a Phase 1 fixture even by accident (T-05-04-04).

It needs no torch model, no network and no new dependency pins: scipy 1.18.0 and
scikit-learn 1.9.0 already resolve in the committed ``uv.lock``.

WHY NOT jsonfmt.py
------------------
The Phase 1 writer emits floats through ``%.9g`` because every Phase 1 fixture records an
f32. The claims layer is f64 by decision (D-06: the paired statistics are the numbers that
get published), and 9 significant digits would silently truncate the t critical value in
the 10th digit. This module uses ``json.dumps`` instead, whose float repr is the shortest
form that round-trips binary64 exactly.

NON-FINITE VALUES ARE UNREPRESENTABLE HERE (review finding, 05-04 T1)
--------------------------------------------------------------------
A paired sample whose ten differences are all identical has zero difference variance, so
the t statistic is undefined and its CI has no finite bounds. ``NaN``/``Infinity`` are not
valid JSON, and ``serde_json`` renders a non-finite f64 as ``null`` (the Ph3 CR-03 lesson),
so a fixture that recorded one would become a MISSING number in a published claims row.

Such a case is therefore a TYPED degenerate case: it carries ``"kind":
"degenerate_zero_variance"``, records ``"expect": "ZeroVarianceDifferences"`` -- the typed
error the Rust side must return -- and records NO statistic, p-value or CI numbers at all.
``json.dumps`` is called with ``allow_nan=False`` and every payload additionally walks
``assert_all_finite`` first, which names the offending JSON path in its FATAL message.

ASSUMPTIONS RESOLVED IN THIS ENVIRONMENT (not from memory)
----------------------------------------------------------
A1  ``scipy.stats.t.ppf(0.975, 9)`` -- computed here and recorded at full f64 repr. The
    ENV value is the authority; the planning documents' 2.2621571628 is a guess that this
    file either confirms or replaces.
A3  Whether sklearn exposes a multiclass ECE -- probed at run time (``probe_sklearn_ece``)
    and the probe result is written into the ECE fixture header.
A4  The one-vs-rest Brier identity ``BS = sum_k brier_score_loss(y == k, p[:, k])`` --
    cross-checked numerically for every Brier case; a deviation beyond 1e-12 is FATAL.

THREE-WAY DISCIPLINE (Phase 2 fixture precedent)
------------------------------------------------
Where a closed form, a library value and an independent hand value all exist, all three
must agree or this generator aborts:

  t statistic  scipy.ttest_rel | scipy.ttest_1samp(diffs) | the closed form in numpy
  t p-value    scipy.ttest_rel | 2 * scipy.stats.t.sf(|t|, df)
  t critical   scipy.t.ppf(0.975, df) | scipy.t.cdf(value, df) == 0.975 (inverse check)
  ECE          vectorised numpy | an independent pure-python loop | an analytic value
                                                                    for the uniform case
  Brier        vectorised numpy | an independent pure-python loop | the sklearn OvR sum
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy import stats
from sklearn.metrics import brier_score_loss

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "claims_stats"

GENERATOR = "gen_claims_fixtures.py"

# The 10-seed benchmark design fixes n = 10, hence df = 9, for every paired comparison.
PAIRED_N = 10
PAIRED_DF = PAIRED_N - 1
CI_Q = 0.975

# The claims usage bins into ten equal-width bins over [0, 1]; n_bins stays a parameter
# on the Rust side, so the fixtures record it explicitly per case.
N_BINS = 10
N_CLASSES = 3

# Absolute agreement bound for every cross-check below. Chosen far tighter than the 1e-6
# parity band the Rust tests use, so a disagreement is a real formula defect and not
# float noise.
CROSS_CHECK_TOL = 1e-12

# A confidence that lands within this distance of a bin boundary would let f32 (Rust) and
# f64 (here) fall into different bins, turning a parity test into a coin flip. Fixture
# inputs are rejected if they get that close; conf == 1.0 is exempt because it is the
# CLAMPED case (bin index n_bins is pulled back to n_bins - 1) and is deliberate coverage.
BIN_BOUNDARY_MARGIN = 1e-2


# ====================================================================================
# environment / integrity helpers
# ====================================================================================


def env_versions() -> dict:
    """Record the versions the fixtures are a numerical artifact OF, read at run time."""
    return {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def assert_all_finite(obj: object, path: str = "$") -> None:
    """Abort naming the JSON path of any non-finite float before anything is written."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_all_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_all_finite(v, f"{path}[{i}]")
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            sys.exit(f"FATAL: non-finite float at {path} (value {obj!r}) -- refusing to write")


def write_fixture(name: str, payload: dict) -> None:
    """Write one fixture, f64-lossless, after proving every float in it is finite."""
    assert_all_finite(payload, f"${name}")
    text = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=True) + "\n"
    (OUT_DIR / name).write_text(text, encoding="utf-8")
    print(f"  wrote claims_stats/{name} ({len(text)} bytes)")


def agree(label: str, *values: float, tol: float = CROSS_CHECK_TOL) -> float:
    """Assert every cross-check of one quantity agrees, and return the first value."""
    base = float(values[0])
    for i, v in enumerate(values[1:], start=1):
        if not math.isfinite(v) or abs(float(v) - base) > tol:
            sys.exit(
                f"FATAL: cross-check disagreement for {label}: "
                f"value[0]={base!r} vs value[{i}]={float(v)!r} (tol {tol})"
            )
    return base


def write_manifest(directory: Path = OUT_DIR) -> None:
    """SHA-256 manifest over every emitted fixture, then verify it with shasum.

    Same rule as the Phase 1 corpus (Ph1 D-13): entries are BARE FILENAMES resolved
    against the directory the manifest was read from, which is why the self-verification
    runs with `cwd=directory`. This manifest covers ONLY this directory -- the Phase 1
    manifest is a separate file and is not touched.
    """
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != "manifest.sha256")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in files]
    (directory / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")

    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", "manifest.sha256"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"FATAL: manifest self-verification failed\n{proc.stdout}\n{proc.stderr}")
    print(f"{directory.name}/manifest.sha256 covers {len(files)} files; shasum -c passed")


# ====================================================================================
# (a) t critical values -- VERIFIES A1
# ====================================================================================


def gen_t_critical() -> None:
    rows = []
    for df in (5, PAIRED_DF, 19):
        value = float(stats.t.ppf(CI_Q, df))
        # Inverse consistency: the quantile function and the CDF must be mutual inverses.
        agree(f"t.ppf/cdf round trip at df={df}", CI_Q, float(stats.t.cdf(value, df)), tol=1e-12)
        rows.append({"df": df, "q": CI_Q, "value": value})

    primary = next(r for r in rows if r["df"] == PAIRED_DF)

    write_fixture(
        "t_critical.json",
        {
            "generator": GENERATOR,
            "versions": env_versions(),
            "formula": "t_crit = scipy.stats.t.ppf(0.975, df)",
            "note": (
                "A1 VERIFIED IN THE PINNED ENV. df is fixed at 9 by the ten-seed benchmark "
                "design, so the Rust side freezes ONE constant and implements no inverse CDF. "
                "The df=5 and df=19 rows are context only -- nothing consumes them."
            ),
            "cross_checks": [
                "scipy.stats.t.cdf(value, df) == 0.975 to 1e-12 for every row"
            ],
            "primary": primary,
            "context": rows,
        },
    )


# ====================================================================================
# (b) paired t cases -- the D-05/D-06 closed-form claims arithmetic
# ====================================================================================


def _closed_form_paired(diffs: np.ndarray, t_crit: float) -> dict:
    """The exact arithmetic the Rust side must reproduce, written out in numpy f64."""
    n = diffs.size
    d_bar = float(diffs.mean())
    s_d = float(diffs.std(ddof=1))
    se = s_d / math.sqrt(n)
    statistic = d_bar / se
    half_width = t_crit * se
    return {
        "n": n,
        "mean_diff": d_bar,
        "std_diff": s_d,
        "std_err": se,
        "statistic": statistic,
        "ci_half_width": half_width,
        "ci_low": d_bar - half_width,
        "ci_high": d_bar + half_width,
    }


def _finite_case(case_id: str, note: str, a: list[float], b: list[float], t_crit: float) -> dict:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diffs = arr_a - arr_b

    closed = _closed_form_paired(diffs, t_crit)
    if closed["std_diff"] <= 0.0:
        sys.exit(f"FATAL: case '{case_id}' is declared finite but has zero difference variance")

    rel = stats.ttest_rel(arr_a, arr_b)
    one = stats.ttest_1samp(diffs, 0.0)

    statistic = agree(
        f"{case_id}.statistic",
        closed["statistic"],
        float(rel.statistic),
        float(one.statistic),
    )
    pvalue = agree(
        f"{case_id}.pvalue",
        float(rel.pvalue),
        float(one.pvalue),
        float(2.0 * stats.t.sf(abs(statistic), PAIRED_DF)),
    )

    return {
        "id": case_id,
        "kind": "finite",
        "note": note,
        "a": [float(x) for x in a],
        "b": [float(x) for x in b],
        "expect": "Ok",
        "df": PAIRED_DF,
        "t_crit": t_crit,
        "statistic": statistic,
        "pvalue": pvalue,
        **{k: closed[k] for k in ("n", "mean_diff", "std_diff", "std_err",
                                  "ci_half_width", "ci_low", "ci_high")},
        # RED values: what a named WRONG implementation produces on this exact input.
        # A test that only asserts the GREEN value passes for any implementation that
        # happens to land close; asserting the RED value is DIFFERENT is what makes the
        # test falsifying (glm_tests.rs:274-298 precedent).
        "red_statistic_population_std": float(
            closed["mean_diff"] / (float(diffs.std(ddof=0)) / math.sqrt(diffs.size))
        ),
        "red_ci_half_width_no_sqrt_n": float(t_crit * closed["std_diff"]),
    }


def _degenerate_case(case_id: str, note: str, a: list[float], b: list[float]) -> dict:
    """A zero-difference-variance case: raw samples + the typed error, no numbers.

    Deliberately records NO statistic, p-value or CI. There is no finite value to record:
    se = 0, so t = d_bar / 0. The contracted behaviour is a typed refusal, and a fixture
    that carried a placeholder number here would be inviting exactly the silent
    `null`-in-a-published-row failure this case exists to prevent.
    """
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diffs = arr_a - arr_b
    s_d = float(diffs.std(ddof=1))
    if s_d != 0.0:
        sys.exit(
            f"FATAL: case '{case_id}' is declared degenerate but s_d = {s_d!r} != 0.0; "
            "pick inputs whose differences are exactly identical in binary64"
        )
    unique = sorted({float(d) for d in diffs})
    if len(unique) != 1:
        sys.exit(f"FATAL: case '{case_id}' differences are not all identical: {unique}")

    return {
        "id": case_id,
        "kind": "degenerate_zero_variance",
        "note": note,
        "a": [float(x) for x in a],
        "b": [float(x) for x in b],
        "expect": "ZeroVarianceDifferences",
        "n": int(diffs.size),
        "df": PAIRED_DF,
        "std_diff": s_d,
        "constant_difference": unique[0],
    }


def gen_paired_t_cases() -> None:
    t_crit = float(stats.t.ppf(CI_Q, PAIRED_DF))

    # F_avg-shaped values in [0, 1]. Ten entries per arm, mirroring the ten contracted
    # seeds; the arms stand in for (SetFit, LoRA) evaluated on the same sampled IDs.
    setfit = [0.612, 0.640, 0.598, 0.671, 0.655, 0.603, 0.688, 0.629, 0.647, 0.618]
    lora = [0.581, 0.622, 0.604, 0.639, 0.610, 0.599, 0.651, 0.618, 0.634, 0.596]

    # Deliberately NOT symmetric: an exactly-zero mean delta would make the statistic
    # exactly 0.0 and the p-value exactly 1.0, which tests the sign convention and the
    # tail machinery not at all. A small non-zero mean keeps the case a near tie while
    # still exercising a mid-range p-value.
    near_tie_a = [0.5210, 0.5330, 0.5170, 0.5405, 0.5288, 0.5122, 0.5461, 0.5233, 0.5309, 0.5195]
    near_tie_b = [0.5207, 0.5327, 0.5174, 0.5401, 0.5291, 0.5119, 0.5458, 0.5236, 0.5306, 0.5198]

    mixed_a = [0.44, 0.71, 0.52, 0.66, 0.38, 0.75, 0.49, 0.61, 0.57, 0.69]
    mixed_b = [0.51, 0.63, 0.58, 0.60, 0.47, 0.66, 0.44, 0.70, 0.50, 0.72]

    neg_a = [0.402, 0.418, 0.395, 0.431, 0.409, 0.388, 0.442, 0.415, 0.427, 0.399]
    neg_b = [0.446, 0.451, 0.438, 0.470, 0.455, 0.429, 0.481, 0.452, 0.468, 0.441]

    # Exact binary fractions (denominator 16) so every difference is bit-identical in
    # binary64 AND in binary32 -- the degenerate cases must be degenerate on both sides.
    exact = [0.5, 0.75, 0.25, 0.625, 0.375, 0.875, 0.125, 0.6875, 0.4375, 0.5625]
    shifted = [x - 0.125 for x in exact]

    cases = [
        _finite_case(
            "clear_positive_delta",
            "SetFit ahead on every seed; the CI excludes zero.",
            setfit, lora, t_crit,
        ),
        _finite_case(
            "near_tie",
            "Mean delta near zero with tiny alternating differences; the CI straddles zero. "
            "This is the shape PF-007 warns about and D-08 refuses to call a verdict.",
            near_tie_a, near_tie_b, t_crit,
        ),
        _finite_case(
            "mixed_signs",
            "Per-seed deltas change sign; large dispersion relative to the mean.",
            mixed_a, mixed_b, t_crit,
        ),
        _finite_case(
            "small_negative_delta",
            "Arm A behind on every seed -- the sign of the statistic must follow.",
            neg_a, neg_b, t_crit,
        ),
        _degenerate_case(
            "constant_difference_nonzero",
            "Every difference is exactly +0.125: a real, non-zero effect with zero "
            "dispersion. t = d_bar / 0 is undefined, so the contracted result is the typed "
            "ZeroVarianceDifferences refusal, NOT a large-or-infinite statistic.",
            exact, shifted,
        ),
        _degenerate_case(
            "all_zero_difference",
            "The two arms are identical. s_d = 0 AND d_bar = 0, so the statistic is 0/0. "
            "Same typed refusal -- the zero-variance guard must not special-case d_bar == 0 "
            "into a spuriously 'fine' answer.",
            exact, list(exact),
        ),
    ]

    n_finite = sum(1 for c in cases if c["kind"] == "finite")
    n_degen = sum(1 for c in cases if c["kind"] == "degenerate_zero_variance")
    if n_finite < 2 or n_degen < 2:
        sys.exit(f"FATAL: need >= 2 finite and >= 2 degenerate cases, got {n_finite}/{n_degen}")

    write_fixture(
        "paired_t_cases.json",
        {
            "generator": GENERATOR,
            "versions": env_versions(),
            "formula": (
                "d_i = a_i - b_i; d_bar = mean(d); s_d = std(d, ddof=1); "
                "se = s_d / sqrt(n); t = d_bar / se; df = n - 1; "
                "p = 2 * t.sf(|t|, df); CI95 = d_bar +/- t_crit * se"
            ),
            "t_crit_source": "scipy.stats.t.ppf(0.975, 9), see t_critical.json",
            "t_crit": t_crit,
            "degenerate_policy": (
                "A case with s_d == 0.0 has an undefined statistic. It carries "
                "kind=degenerate_zero_variance and expect=ZeroVarianceDifferences and records "
                "no statistic, no p-value and no CI. Both the paired t function and the "
                "paired-CI helper must return that typed error rather than a non-finite f64."
            ),
            "cross_checks": [
                "statistic: closed form vs scipy.ttest_rel vs scipy.ttest_1samp(diffs) to 1e-12",
                "pvalue: scipy.ttest_rel vs scipy.ttest_1samp vs 2*scipy.stats.t.sf to 1e-12",
            ],
            "red_implementations": {
                "red_statistic_population_std": (
                    "t computed with the POPULATION std (ddof=0) instead of the (n-1) "
                    "sample std. Off by sqrt(n/(n-1)) ~ 1.054 at n=10 -- close enough to "
                    "survive a loose tolerance, wrong in every published CI."
                ),
                "red_ci_half_width_no_sqrt_n": (
                    "CI half-width computed as t_crit * s_d, forgetting the / sqrt(n). "
                    "Inflates every interval by a factor of sqrt(10) ~ 3.162."
                ),
            },
            "n_finite": n_finite,
            "n_degenerate": n_degen,
            "cases": cases,
        },
    )


# ====================================================================================
# (b2) seed-dispersion one-sample CI95 -- the ACTIVE-scope statistic of claims 2.0.0
# ====================================================================================


def gen_seed_dispersion_ci_cases() -> None:
    """Reference endpoints for `ci95_one_sample_df9` (claims 2.0.0 seed_dispersion_ci95).

    The active, single-method scope has no second arm to difference against, so its
    uncertainty is the dispersion of one method's score across the ten contracted seeds
    at fixed data and protocol. The Rust side computes it by delegating to the PAIRED
    helper against an all-zero comparator, so there is exactly one mean and one (n-1)
    std in the claims layer (OPS-03). This fixture pins the endpoints that delegation
    must reproduce, computed here independently by scipy -- so a mistake in the
    delegation is caught by a reference rather than by the delegation's own arithmetic.
    """
    t_crit = float(stats.t.ppf(CI_Q, PAIRED_DF))

    # F_avg-shaped, ten entries per case, one per contracted seed.
    spread = [0.612, 0.640, 0.598, 0.671, 0.655, 0.603, 0.688, 0.629, 0.647, 0.618]
    tight = [0.7031, 0.7028, 0.7034, 0.7029, 0.7033, 0.7030, 0.7035, 0.7027, 0.7032, 0.7031]
    low = [0.311, 0.289, 0.402, 0.256, 0.377, 0.298, 0.341, 0.266, 0.388, 0.303]
    # Exact binary fractions: identical in binary64 AND binary32, so the degenerate
    # case is degenerate on both sides rather than by rounding.
    constant = [0.6875] * PAIRED_N

    def finite(case_id: str, note: str, values: list[float]) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        n = int(arr.size)
        if n != PAIRED_N:
            sys.exit(f"FATAL: {case_id} has {n} values, the design is {PAIRED_N}")
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        se = std / math.sqrt(n)
        half = t_crit * se

        # CROSS-CHECK against scipy's own interval machinery, not just the closed form.
        sp_low, sp_high = stats.t.interval(0.95, n - 1, loc=mean, scale=se)
        # And against the one-sample t-test's sem, which is the same quantity by a
        # different route.
        sp_se = float(stats.sem(arr, ddof=1))

        return {
            "id": case_id,
            "kind": "finite",
            "note": note,
            "values": [float(v) for v in values],
            "n": n,
            "df": PAIRED_DF,
            "mean": agree(f"{case_id}.mean", mean, float(np.mean(arr))),
            "std": agree(f"{case_id}.std", std, float(np.std(arr, ddof=1))),
            "std_err": agree(f"{case_id}.std_err", se, sp_se),
            "half_width": half,
            "ci95_low": agree(f"{case_id}.ci95_low", mean - half, float(sp_low)),
            "ci95_high": agree(f"{case_id}.ci95_high", mean + half, float(sp_high)),
        }

    cases = [
        finite(
            "seed_spread_typical",
            "Ten seeds with the dispersion a few-shot run actually shows; the interval "
            "is wide enough that quoting the mean alone would overstate precision, "
            "which is the whole reason EVAL-04 asks for uncertainty and not just a mean.",
            spread,
        ),
        finite(
            "seed_spread_tight",
            "Barely-varying seeds. The interval is narrow but PRESENT -- narrowness is "
            "a measurement, not a licence to drop the interval.",
            tight,
        ),
        finite(
            "seed_spread_wide_low_scores",
            "Low scores with large seed-to-seed swing: at 8 shots the sampled subset is "
            "a larger source of variation than anything else, which is PF-007's point.",
            low,
        ),
        {
            "id": "all_seeds_identical",
            "kind": "degenerate_zero_variance",
            "note": (
                "All ten seeds scored exactly the same. There is no dispersion to "
                "interval over, so the contracted result is the typed "
                "ZeroVarianceDifferences refusal with the constant reported -- never a "
                "NaN and never a serde null that reads as a missing measurement (CR-03)."
            ),
            "values": [float(v) for v in constant],
            "n": PAIRED_N,
            "df": PAIRED_DF,
            "expect": "ZeroVarianceDifferences",
            "constant_value": float(constant[0]),
        },
    ]

    n_finite = sum(1 for c in cases if c["kind"] == "finite")
    n_degen = sum(1 for c in cases if c["kind"] == "degenerate_zero_variance")
    if n_finite < 2 or n_degen < 1:
        sys.exit(f"FATAL: need >= 2 finite and >= 1 degenerate cases, got {n_finite}/{n_degen}")

    write_fixture(
        "seed_dispersion_ci_cases.json",
        {
            "generator": GENERATOR,
            "versions": env_versions(),
            "formula": (
                "x_bar = mean(x); s = std(x, ddof=1); se = s / sqrt(n); df = n - 1; "
                "CI95 = x_bar +/- t_crit * se"
            ),
            "t_crit_source": "scipy.stats.t.ppf(0.975, 9), see t_critical.json",
            "t_crit": t_crit,
            "scope": (
                "ACTIVE scope of setfit-benchmark-claims-v1 2.0.0 "
                "(equations.claims_statistics.seed_dispersion_ci95). A seed-dispersion "
                "interval at fixed data and protocol -- NOT a population interval and "
                "NOT a comparison."
            ),
            "degenerate_policy": (
                "A case whose ten values are all identical has no dispersion. It carries "
                "kind=degenerate_zero_variance and expect=ZeroVarianceDifferences and "
                "records no interval. ci95_one_sample_df9 must return that typed error "
                "rather than a non-finite f64."
            ),
            "cross_checks": [
                "mean: closed form vs numpy.mean to 1e-12",
                "std: closed form vs numpy.std(ddof=1) to 1e-12",
                "std_err: s/sqrt(n) vs scipy.stats.sem(ddof=1) to 1e-12",
                "endpoints: closed form vs scipy.stats.t.interval(0.95, df, loc, scale) to 1e-12",
            ],
            "cases": cases,
        },
    )


# ====================================================================================
# (c) top-label ECE -- D-07, VERIFIES A3
# ====================================================================================


def probe_sklearn_ece() -> dict:
    """A3: does sklearn 1.9.0 expose a multiclass ECE? Probed, not assumed."""
    import sklearn.calibration as calib
    import sklearn.metrics as metrics

    metric_names = sorted(
        n for n in dir(metrics)
        if not n.startswith("_") and ("calib" in n.lower() or "brier" in n.lower())
    )
    calib_names = sorted(
        n for n in dir(calib)
        if not n.startswith("_") and ("calib" in n.lower() or "curve" in n.lower())
    )
    has_ece = any("expected_calibration" in n or n.lower() == "ece" for n in metric_names + calib_names)
    return {
        "sklearn_version": sklearn.__version__,
        "sklearn_metrics_calibration_names": metric_names,
        "sklearn_calibration_names": calib_names,
        "exposes_multiclass_ece": has_ece,
        "conclusion": (
            "A3 CONFIRMED: sklearn 1.9.0 exposes no expected-calibration-error API at all "
            "(multiclass or binary); calibration_curve is a binary reliability-curve helper. "
            "The reference ECE below is therefore an independent numpy implementation, "
            "cross-checked against a separate pure-python loop and, for the uniform case, "
            "against a closed-form analytic value."
        ) if not has_ece else "A3 REFUTED: sklearn exposes a calibration-error API; prefer it.",
    }


def _softmax_rows(logits: list[list[float]]) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64)
    shifted = arr - arr.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _ece_numpy(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    idx = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)

    n = float(probs.shape[0])
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(float(conf[sel].mean()) - float(correct[sel].mean()))
    return ece


def _ece_red_true_label_conf(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """RED: confidence taken from the TRUE label's column instead of the maximum.

    The plan phrases this mutation as "bin by predicted-class probability instead of
    max". Those two are the SAME quantity -- the predicted class IS the argmax -- so that
    mutation is a no-op and could never turn a test red. The real confusion the wording
    is reaching for is between the PREDICTED class's probability and the LABELLED class's
    probability, which is a genuinely different number whenever the prediction is wrong.
    """
    conf = probs[np.arange(probs.shape[0]), labels]
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    idx = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)

    n = float(probs.shape[0])
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        if count == 0:
            continue
        ece += (count / n) * abs(float(conf[sel].mean()) - float(correct[sel].mean()))
    return ece


def _ece_red_unweighted_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """RED: bins averaged unweighted instead of weighted by occupancy (n_b / N)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    idx = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)

    gaps = []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        gaps.append(abs(float(conf[sel].mean()) - float(correct[sel].mean())))
    return float(sum(gaps) / len(gaps)) if gaps else 0.0


def _ece_pure_python(probs: list[list[float]], labels: list[int], n_bins: int) -> float:
    """Independent implementation: explicit loops, no numpy, no vectorised reductions."""
    bin_conf_sum = [0.0] * n_bins
    bin_correct = [0.0] * n_bins
    bin_count = [0] * n_bins

    for row, label in zip(probs, labels):
        conf = max(row)
        pred = row.index(conf)
        b = int(conf * n_bins)
        if b > n_bins - 1:
            b = n_bins - 1
        bin_conf_sum[b] += conf
        bin_correct[b] += 1.0 if pred == label else 0.0
        bin_count[b] += 1

    n = float(len(labels))
    ece = 0.0
    for b in range(n_bins):
        if bin_count[b] == 0:
            continue
        avg_conf = bin_conf_sum[b] / bin_count[b]
        avg_acc = bin_correct[b] / bin_count[b]
        ece += (bin_count[b] / n) * abs(avg_conf - avg_acc)
    return ece


def _check_bin_margins(case_id: str, probs: np.ndarray, n_bins: int) -> None:
    """Refuse inputs whose top confidence sits on a bin boundary.

    f32 (Rust) and f64 (here) disagree in the ~1e-8 range. A confidence sitting on a
    boundary would therefore bin differently on the two sides and the parity test would
    become a coin flip that occasionally, mysteriously, goes red. conf == 1.0 is exempt:
    it is the CLAMPED case (index n_bins pulled back to n_bins - 1), it is exact in both
    precisions, and it is deliberate coverage of the .min(n_bins - 1) guard.
    """
    for i, conf in enumerate(probs.max(axis=1)):
        conf = float(conf)
        if conf == 1.0:
            continue
        scaled = conf * n_bins
        frac = scaled - math.floor(scaled)
        if min(frac, 1.0 - frac) < BIN_BOUNDARY_MARGIN:
            sys.exit(
                f"FATAL: case '{case_id}' row {i} has top confidence {conf!r}, which is "
                f"within {BIN_BOUNDARY_MARGIN} of a bin boundary; pick a different input"
            )


def _ece_case(case_id: str, note: str, probs_list: list[list[float]], labels: list[int],
              analytic: float | None = None) -> dict:
    probs = np.asarray(probs_list, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)

    if probs.shape[1] != N_CLASSES:
        sys.exit(f"FATAL: case '{case_id}' has K={probs.shape[1]}, expected {N_CLASSES}")
    if probs.shape[0] != labels_arr.size:
        sys.exit(f"FATAL: case '{case_id}' row/label count mismatch")
    row_sums = probs.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-9):
        sys.exit(f"FATAL: case '{case_id}' has a row not summing to 1: {row_sums.tolist()}")
    if labels_arr.min() < 0 or labels_arr.max() >= N_CLASSES:
        sys.exit(f"FATAL: case '{case_id}' has a label outside 0..{N_CLASSES - 1}")
    _check_bin_margins(case_id, probs, N_BINS)

    values = [
        _ece_numpy(probs, labels_arr, N_BINS),
        _ece_pure_python([[float(x) for x in row] for row in probs], labels, N_BINS),
    ]
    if analytic is not None:
        values.append(analytic)
    ece = agree(f"{case_id}.ece", *values)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    return {
        "id": case_id,
        "note": note,
        "n_classes": N_CLASSES,
        "n_bins": N_BINS,
        "probabilities": [[float(x) for x in row] for row in probs],
        "labels": [int(x) for x in labels],
        "confidences": [float(x) for x in conf],
        "predictions": [int(x) for x in pred],
        "accuracy": float((pred == labels_arr).mean()),
        "ece": ece,
        "red_ece_true_label_conf": _ece_red_true_label_conf(probs, labels_arr, N_BINS),
        "red_ece_unweighted_bins": _ece_red_unweighted_bins(probs, labels_arr, N_BINS),
    }


def gen_ece_cases() -> None:
    probe = probe_sklearn_ece()

    calibrated = _softmax_rows([
        [2.10, 0.30, 0.05], [0.20, 1.95, 0.40], [0.15, 0.35, 2.05], [1.70, 0.60, 0.25],
        [0.40, 1.55, 0.35], [0.25, 0.45, 1.80], [1.35, 0.85, 0.30], [0.55, 1.25, 0.45],
        [0.30, 0.50, 1.45], [1.20, 0.70, 0.55], [0.45, 1.15, 0.60], [0.35, 0.65, 1.30],
    ])
    calibrated_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2]

    overconfident = _softmax_rows([
        [5.10, 0.20, 0.10], [0.10, 4.85, 0.15], [0.05, 0.20, 5.25], [4.70, 0.30, 0.20],
        [0.15, 5.05, 0.10], [0.20, 0.10, 4.60], [5.35, 0.15, 0.25], [0.25, 4.95, 0.20],
        [0.10, 0.30, 5.15], [4.80, 0.20, 0.15],
    ])
    # Six of ten predictions are wrong while confidence stays ~0.99: heavy miscalibration.
    overconfident_labels = [0, 2, 1, 1, 1, 0, 2, 1, 0, 0]

    third = 1.0 / 3.0
    uniform = [[third, third, third] for _ in range(9)]
    uniform_labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    # Analytic: every row has conf = 1/3 and argmax = 0 (first maximum wins ties), so all
    # nine land in one bin; accuracy is 3/9. ECE = |1/3 - 1/3| = 0 exactly. The tie-break
    # is load-bearing, and the Rust argmax must break ties the same way.
    uniform_analytic = abs(third - 3.0 / 9.0)

    saturated = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],
        [0.955, 0.030, 0.015], [0.020, 0.965, 0.015], [0.015, 0.025, 0.960],
        [0.945, 0.035, 0.020],
    ]
    # Three of the four saturated (conf == 1.0) rows are wrong, so the clamped top bin
    # carries real signal rather than being a formality.
    saturated_labels = [0, 2, 1, 2, 0, 1, 2, 0]

    mixed = _softmax_rows([
        [1.10, 0.90, 0.35], [0.80, 1.05, 0.55], [0.45, 0.75, 1.15], [2.30, 0.40, 0.20],
        [0.30, 2.15, 0.45], [0.25, 0.35, 2.45], [0.95, 1.20, 0.60], [1.45, 0.55, 0.85],
        [0.65, 1.35, 0.40], [0.50, 0.95, 1.25], [3.05, 0.25, 0.15], [0.20, 3.20, 0.30],
        [0.15, 0.40, 2.95], [1.60, 1.10, 0.30], [0.70, 0.45, 1.55],
    ])
    mixed_labels = [0, 1, 2, 0, 1, 2, 1, 0, 1, 2, 0, 0, 2, 1, 2]

    cases = [
        _ece_case("underconfident_moderate",
                  "Moderate confidences spread across several bins with a much higher hit "
                  "rate: accuracy sits ABOVE confidence in most bins. ECE is an absolute "
                  "gap, so under- and over-confidence must contribute with the same sign.",
                  [[float(x) for x in r] for r in calibrated], calibrated_labels),
        _ece_case("overconfident_sharp",
                  "Near-certain confidences with a 40% hit rate: the largest ECE here.",
                  [[float(x) for x in r] for r in overconfident], overconfident_labels),
        _ece_case("uniform_uncertain",
                  "Every row is the uniform distribution. All nine rows land in one bin and "
                  "accuracy equals confidence exactly, so ECE is analytically zero.",
                  uniform, uniform_labels, analytic=uniform_analytic),
        _ece_case("saturated_top_bin",
                  "Includes rows with confidence EXACTLY 1.0. conf * n_bins == 10 must be "
                  "clamped to bin index 9 by the .min(n_bins - 1) guard; without the clamp "
                  "the implementation indexes out of bounds.",
                  saturated, saturated_labels),
        _ece_case("mixed_hard",
                  "Fifteen rows spanning low to high confidence with mixed correctness.",
                  [[float(x) for x in r] for r in mixed], mixed_labels),
    ]

    if len(cases) < 4:
        sys.exit("FATAL: need >= 4 ECE cases")

    write_fixture(
        "ece_top_label_cases.json",
        {
            "generator": GENERATOR,
            "versions": env_versions(),
            "formula": (
                "conf_i = max_k p_ik; pred_i = argmax_k p_ik; "
                "bin(i) = min(floor(conf_i * n_bins), n_bins - 1); "
                "ECE = sum_b (n_b / N) * |acc_b - conf_b|"
            ),
            "a3_probe": probe,
            "cross_checks": [
                "ece: vectorised numpy vs an independent pure-python loop to 1e-12",
                "ece: plus a closed-form analytic value for uniform_uncertain",
                "every input row sums to 1 within 1e-9",
                "no top confidence within 1e-2 of a bin boundary (except exactly 1.0)",
            ],
            "red_implementations": {
                "red_ece_true_label_conf": (
                    "Confidence read from the TRUE label's column instead of the maximum. "
                    "NOTE: the mutation 'bin by predicted-class probability instead of max' "
                    "is a NO-OP -- the predicted class IS the argmax, so those two are the "
                    "same number. This is the mutation that wording is reaching for."
                ),
                "red_ece_unweighted_bins": (
                    "Bin gaps averaged unweighted instead of weighted by occupancy n_b/N. "
                    "Silently over-weights sparsely-populated bins."
                ),
            },
            "cases": cases,
        },
    )


# ====================================================================================
# (d) multiclass Brier -- D-07, VERIFIES A4
# ====================================================================================


def _brier_numpy(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(probs.shape[0]), labels] = 1.0
    return float(((probs - onehot) ** 2).sum() / probs.shape[0])


def _brier_pure_python(probs: list[list[float]], labels: list[int]) -> float:
    total = 0.0
    for row, label in zip(probs, labels):
        for k, p in enumerate(row):
            y = 1.0 if k == label else 0.0
            total += (p - y) ** 2
    return total / len(labels)


def _brier_ovr_sklearn(probs: np.ndarray, labels: np.ndarray) -> float:
    """A4: the one-vs-rest identity, evaluated with sklearn's BINARY brier_score_loss.

    sklearn 1.9's `brier_score_loss` takes `scale_by_half='auto'`, which is True for
    binary input -- i.e. it returns mean((p - y)^2) per class, exactly the per-class term
    of the unnormalised multiclass sum. Summing over k therefore reproduces
    BS = (1/N) sum_i sum_k (p_ik - y_ik)^2 with no rescaling.
    """
    total = 0.0
    for k in range(probs.shape[1]):
        y_k = (labels == k).astype(np.int64)
        total += float(brier_score_loss(y_k, probs[:, k], labels=[0, 1], scale_by_half=True))
    return total


def _brier_case(case_id: str, note: str, probs_list: list[list[float]], labels: list[int]) -> dict:
    probs = np.asarray(probs_list, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)

    if probs.shape[1] != N_CLASSES:
        sys.exit(f"FATAL: case '{case_id}' has K={probs.shape[1]}, expected {N_CLASSES}")
    row_sums = probs.sum(axis=1)
    if np.any(np.abs(row_sums - 1.0) > 1e-9):
        sys.exit(f"FATAL: case '{case_id}' has a row not summing to 1")

    direct = _brier_numpy(probs, labels_arr)
    loop = _brier_pure_python([[float(x) for x in r] for r in probs], labels)
    ovr = _brier_ovr_sklearn(probs, labels_arr)
    brier = agree(f"{case_id}.brier", direct, loop, ovr)

    return {
        "id": case_id,
        "note": note,
        "n_classes": N_CLASSES,
        "probabilities": [[float(x) for x in row] for row in probs],
        "labels": [int(x) for x in labels],
        "brier_multiclass": brier,
        "ovr_identity_sklearn": ovr,
        "ovr_identity_abs_deviation": abs(direct - ovr),
        # RED: the silent divide-by-K renormalisation the 05-REVIEWS finding warns about.
        # It is exactly the change a reader would make to force the (impossible) K=2
        # equality with the binary brier_score, so the fixture pins its value too.
        "red_brier_divided_by_k": brier / N_CLASSES,
        # RED: only the true class's term kept, dropping the other K-1 columns entirely.
        "red_brier_true_class_only": float(
            sum((float(probs[i, labels[i]]) - 1.0) ** 2 for i in range(probs.shape[0]))
            / probs.shape[0]
        ),
    }


def gen_brier_cases() -> None:
    sharp_correct = [
        [0.96, 0.03, 0.01], [0.02, 0.95, 0.03], [0.01, 0.04, 0.95], [0.94, 0.04, 0.02],
        [0.03, 0.93, 0.04], [0.02, 0.03, 0.95], [0.97, 0.02, 0.01], [0.01, 0.96, 0.03],
    ]
    sharp_correct_labels = [0, 1, 2, 0, 1, 2, 0, 1]

    sharp_wrong = [
        [0.96, 0.03, 0.01], [0.02, 0.95, 0.03], [0.01, 0.04, 0.95], [0.94, 0.04, 0.02],
        [0.03, 0.93, 0.04], [0.02, 0.03, 0.95],
    ]
    # Every prediction is wrong: this is the upper end of the K=3 range, which is [0, 2].
    sharp_wrong_labels = [1, 2, 0, 2, 0, 1]

    third = 1.0 / 3.0
    uniform = [[third, third, third] for _ in range(6)]
    uniform_labels = [0, 1, 2, 0, 1, 2]

    mixed = [
        [0.55, 0.30, 0.15], [0.20, 0.62, 0.18], [0.25, 0.15, 0.60], [0.40, 0.35, 0.25],
        [0.30, 0.45, 0.25], [0.10, 0.20, 0.70], [0.80, 0.12, 0.08], [0.15, 0.25, 0.60],
        [0.45, 0.40, 0.15], [0.33, 0.34, 0.33],
    ]
    mixed_labels = [0, 1, 2, 1, 1, 2, 0, 0, 2, 1]

    # A K=3 case whose third column is identically zero: the two-class situation embedded
    # in the multiclass surface, which is where the "multiclass == 2 x binary" relation is
    # easiest to see. It is recorded for coverage; the Rust K=2 consistency test builds its
    # own [1-p, p] pair rather than reading this case.
    two_class_embedded = [
        [0.70, 0.30, 0.0], [0.20, 0.80, 0.0], [0.55, 0.45, 0.0], [0.90, 0.10, 0.0],
        [0.35, 0.65, 0.0], [0.48, 0.52, 0.0],
    ]
    two_class_labels = [0, 1, 0, 0, 1, 1]

    cases = [
        _brier_case("sharp_correct", "Confident and right: near the [0, 2] floor.",
                    sharp_correct, sharp_correct_labels),
        _brier_case("sharp_wrong", "Confident and wrong on every row: near the [0, 2] ceiling. "
                                   "This case alone refutes any [0, 1] range assertion.",
                    sharp_wrong, sharp_wrong_labels),
        _brier_case("uniform", "Uniform predictions: BS = (1/N) sum_i [(1-1/3)^2 + 2*(1/3)^2].",
                    uniform, uniform_labels),
        _brier_case("mixed", "Ten rows of moderate, mixed-quality predictions.",
                    mixed, mixed_labels),
        _brier_case("two_class_embedded",
                    "K=3 with an identically-zero third column -- the two-class case living "
                    "inside the multiclass surface.",
                    two_class_embedded, two_class_labels),
    ]

    if len(cases) < 4:
        sys.exit("FATAL: need >= 4 Brier cases")

    worst = max(c["ovr_identity_abs_deviation"] for c in cases)
    if worst > CROSS_CHECK_TOL:
        sys.exit(f"FATAL: A4 OvR identity deviates by {worst!r} > {CROSS_CHECK_TOL}")

    write_fixture(
        "brier_multiclass_cases.json",
        {
            "generator": GENERATOR,
            "versions": env_versions(),
            "formula": "BS = (1/N) sum_i sum_k (p_ik - y_ik)^2 with one-hot y",
            "normalization": (
                "UNNORMALISED over classes: the class sum is NOT divided by K. For K=3 the "
                "range is [0, 2], not [0, 1]. For K=2 built from a binary p as [1-p, p], the "
                "class sum is (p-y)^2 + ((1-p)-(1-y))^2 = 2(p-y)^2, so this quantity is "
                "EXACTLY twice the binary brier_score -- never equal to it."
            ),
            "a4_identity": (
                "BS == sum_k sklearn.metrics.brier_score_loss(y == k, p[:, k], "
                "scale_by_half=True); VERIFIED for every case below"
            ),
            "a4_max_abs_deviation": worst,
            "a4_tolerance": CROSS_CHECK_TOL,
            "cross_checks": [
                "brier: vectorised numpy vs an independent pure-python loop vs the sklearn "
                "OvR sum, all to 1e-12",
            ],
            "red_implementations": {
                "red_brier_divided_by_k": (
                    "The class sum divided by K. This is the silent renormalisation that a "
                    "reader would reach for in order to make the multiclass score EQUAL the "
                    "binary one at K=2 -- which is mathematically impossible, since the K=2 "
                    "class sum is definitionally 2x the binary term."
                ),
                "red_brier_true_class_only": (
                    "Only the true class's squared error kept; the other K-1 columns "
                    "dropped. Equals the multiclass value only when the other columns are "
                    "all zero."
                ),
            },
            "cases": cases,
        },
    )


# ====================================================================================


def main() -> None:
    versions = env_versions()
    print(f"claims-stats fixture generator: {versions}")
    if versions["scipy"] != "1.18.0" or versions["scikit_learn"] != "1.9.0":
        sys.exit(
            "FATAL: this generator is pinned to scipy 1.18.0 / scikit-learn 1.9.0 "
            f"(uv.lock); got {versions['scipy']} / {versions['scikit_learn']}"
        )

    OUT_DIR.mkdir(exist_ok=True)
    print(f"output directory: {OUT_DIR}")

    gen_t_critical()
    gen_paired_t_cases()
    gen_seed_dispersion_ci_cases()
    gen_ece_cases()
    gen_brier_cases()

    write_manifest()
    print("claims-stats fixture set complete.")


if __name__ == "__main__":
    main()
