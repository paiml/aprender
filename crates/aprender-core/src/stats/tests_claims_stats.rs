// =========================================================================
// CLAIMS-LAYER PAIRED STATISTICS (plan 05-04 T3, D-05/D-06)
//
// Fixture parity for the f64 paired-t mirrors and the frozen t critical
// constant, against scipy 1.18.0 in the pinned uv environment.
//
// Regenerate the fixtures with:
//   cd scripts/setfit_fixtures && uv run python gen_claims_fixtures.py
//
// `include_str!` rather than a runtime read: a test that silently skips when a
// file is absent is a test that proves nothing, and these bytes are the ONLY
// evidence that the numbers the benchmark publishes are the numbers scipy
// computes (thresholds.rs rule).
//
// This file is NEW and additive. `tests_hypothesis_contract.rs` (the FALSIFY-HT
// suite) is untouched.
// =========================================================================

use super::*;
use crate::error::AprenderError;

const T_CRITICAL_FIXTURE: &str =
    include_str!("../../../../scripts/setfit_fixtures/claims_stats/t_critical.json");
const PAIRED_T_FIXTURE: &str =
    include_str!("../../../../scripts/setfit_fixtures/claims_stats/paired_t_cases.json");
const SEED_DISPERSION_FIXTURE: &str =
    include_str!("../../../../scripts/setfit_fixtures/claims_stats/seed_dispersion_ci_cases.json");

/// Statistic parity band. Both sides do the same IEEE-754 binary64 operations on the
/// same stored inputs, so agreement is expected to the last few ulp.
const STATISTIC_TOL: f64 = 1e-10;

/// p-value parity band. Looser than the statistic because the two-tailed tail is an
/// incomplete-beta evaluation here versus scipy's own special-function implementation.
const PVALUE_TOL: f64 = 1e-9;

/// A RED value closer than this to GREEN is not a usable discriminator.
const RED_DISCRIMINATION_MARGIN: f64 = 1e-6;

fn fixture_doc(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).expect("fixture JSON parses")
}

fn fixture_cases(raw: &str) -> Vec<serde_json::Value> {
    fixture_doc(raw)["cases"]
        .as_array()
        .expect("fixture has a cases array")
        .clone()
}

fn f64_array(case: &serde_json::Value, key: &str) -> Vec<f64> {
    case[key]
        .as_array()
        .unwrap_or_else(|| panic!("fixture case missing array field '{key}'"))
        .iter()
        .map(|v| v.as_f64().expect("array entry is a number"))
        .collect()
}

fn f64_field(case: &serde_json::Value, key: &str) -> f64 {
    case[key]
        .as_f64()
        .unwrap_or_else(|| panic!("fixture case missing numeric field '{key}'"))
}

fn case_id(case: &serde_json::Value) -> String {
    case["id"].as_str().expect("case has an id").to_string()
}

// ---- the frozen constant -----------------------------------------------------------

#[test]
fn t_crit_975_df9_equals_the_pinned_env_fixture_exactly() {
    let doc = fixture_doc(T_CRITICAL_FIXTURE);
    let primary = &doc["primary"];
    assert_eq!(
        primary["df"].as_u64().expect("primary df"),
        9,
        "the ten-seed design fixes df = 9; a fixture with another df is the wrong fixture"
    );
    let recorded = primary["value"].as_f64().expect("primary value");

    // Bit equality, not a tolerance. The constant is FROZEN: it is copied from this
    // fixture and nothing is allowed to drift it. serde_json's `float_roundtrip`
    // feature (enabled in aprender-core's manifest) makes the parse exact, so any
    // difference at all is a real change to a published CI half-width.
    assert_eq!(
        T_CRIT_975_DF9, recorded,
        "T_CRIT_975_DF9 has drifted from scipy.stats.t.ppf(0.975, 9): \
         constant {T_CRIT_975_DF9:?}, fixture {recorded:?}"
    );
}

#[test]
fn t_crit_975_df9_is_the_value_the_planning_documents_guessed() {
    // 05-RESEARCH.md assumption A1 carried 2.2621571628 from training knowledge and
    // required that it be CONFIRMED in the pinned environment before being frozen.
    // It was: the env value agrees to every digit A1 recorded. This test records the
    // resolution so the assumption cannot quietly come back as an open question.
    assert!(
        (T_CRIT_975_DF9 - 2.262_157_162_8).abs() < 1e-10,
        "A1 REFUTED: env-computed t critical is {T_CRIT_975_DF9:?}"
    );
}

// ---- the ACTIVE-scope seed-dispersion interval (claims 2.0.0) -----------------------
//
// `ci95_one_sample_df9` is what the narrowed, single-method scope reports uncertainty
// with. It DELEGATES to `paired_ci` against an all-zero comparator so there is exactly
// one mean and one (n-1) std in the claims layer (OPS-03) — and that delegation is
// exactly what needs an independent reference: checking it against its own arithmetic
// would prove nothing. scipy computed the endpoints below.

#[test]
fn ci95_one_sample_df9_matches_scipy_for_every_finite_case() {
    let cases = fixture_cases(SEED_DISPERSION_FIXTURE);
    let finite: Vec<_> = cases.iter().filter(|c| c["kind"] == "finite").collect();
    assert!(
        finite.len() >= 2,
        "fixture set shrank below the contracted 2 finite cases"
    );

    for case in finite {
        let id = case_id(case);
        let values = f64_array(case, "values");
        let got = ci95_one_sample_df9(&values).expect("finite case must produce an interval");

        for (label, got_v, want_v) in [
            ("mean", got.mean_diff, f64_field(case, "mean")),
            ("std", got.std_diff, f64_field(case, "std")),
            ("std_err", got.std_err, f64_field(case, "std_err")),
            ("half_width", got.half_width, f64_field(case, "half_width")),
            ("ci95_low", got.low, f64_field(case, "ci95_low")),
            ("ci95_high", got.high, f64_field(case, "ci95_high")),
        ] {
            assert!(
                (got_v - want_v).abs() < STATISTIC_TOL,
                "case {id}: {label} is {got_v:?}, scipy says {want_v:?}"
            );
        }

        assert_eq!(got.n, 10, "case {id}: the design is ten contracted seeds");
    }
}

#[test]
fn ci95_one_sample_df9_refuses_a_seed_set_with_no_dispersion() {
    let cases = fixture_cases(SEED_DISPERSION_FIXTURE);
    let degenerate: Vec<_> = cases
        .iter()
        .filter(|c| c["kind"] == "degenerate_zero_variance")
        .collect();
    assert!(
        !degenerate.is_empty(),
        "the degenerate case is the CR-03 guard; a fixture without it proves nothing"
    );

    for case in degenerate {
        let id = case_id(case);
        let values = f64_array(case, "values");
        match ci95_one_sample_df9(&values) {
            Err(AprenderError::ZeroVarianceDifferences { n, constant_value }) => {
                assert_eq!(n, 10, "case {id}");
                assert!(
                    (constant_value - f64_field(case, "constant_value")).abs() < STATISTIC_TOL,
                    "case {id}: the refusal must report the constant, not a NaN"
                );
            }
            other => {
                panic!("case {id}: ten identical seeds must be the typed refusal, got {other:?}")
            }
        }
    }
}

#[test]
fn ci95_one_sample_df9_is_the_paired_helper_against_zero_not_a_second_definition() {
    // OPS-03 made falsifiable. If someone re-implements the one-sample interval with its
    // own mean/std instead of delegating, this goes red the moment the two implementations
    // differ by a single ulp — which is what "one definition of the moments" has to mean
    // to be worth asserting.
    let values = f64_array(
        fixture_cases(SEED_DISPERSION_FIXTURE)
            .iter()
            .find(|c| c["kind"] == "finite")
            .expect("at least one finite case"),
        "values",
    );
    let zeros = vec![0.0_f64; values.len()];

    let direct = ci95_one_sample_df9(&values).expect("interval exists");
    let through_paired = paired_ci95_df9(&values, &zeros).expect("interval exists");

    assert_eq!(
        direct.mean_diff.to_bits(),
        through_paired.mean_diff.to_bits(),
        "the mean must be BIT-identical, not merely close"
    );
    assert_eq!(direct.std_diff.to_bits(), through_paired.std_diff.to_bits());
    assert_eq!(direct.low.to_bits(), through_paired.low.to_bits());
    assert_eq!(direct.high.to_bits(), through_paired.high.to_bits());
}

#[test]
fn ci95_one_sample_df9_refuses_any_sample_size_but_the_frozen_design() {
    // The frozen t is only correct at df = 9. Applied to another n it would produce a
    // plausible, WRONG interval — the exact failure freezing the constant prevents.
    for n in [2_usize, 9, 11, 20] {
        let values = vec![0.5_f64; n]
            .iter()
            .enumerate()
            .map(|(i, v)| v + i as f64 * 1e-3)
            .collect::<Vec<f64>>();
        assert!(
            matches!(
                ci95_one_sample_df9(&values),
                Err(AprenderError::DimensionMismatch { .. })
            ),
            "n = {n} is not the ten-seed design and must be refused"
        );
    }
}

// ---- paired t parity ----------------------------------------------------------------

#[test]
fn ttest_rel_f64_matches_scipy_for_every_finite_case() {
    let cases = fixture_cases(PAIRED_T_FIXTURE);
    let finite: Vec<_> = cases.iter().filter(|c| c["kind"] == "finite").collect();
    assert!(
        finite.len() >= 2,
        "fixture set shrank below the contracted 2 finite cases"
    );

    for case in finite {
        let id = case_id(case);
        let a = f64_array(case, "a");
        let b = f64_array(case, "b");

        let got = ttest_rel_f64(&a, &b).expect("finite case must produce a result");

        let want_stat = f64_field(case, "statistic");
        let want_p = f64_field(case, "pvalue");

        assert!(
            (got.statistic - want_stat).abs() < STATISTIC_TOL,
            "case '{id}': statistic {} vs scipy {want_stat} (tol {STATISTIC_TOL})",
            got.statistic
        );
        assert!(
            (got.pvalue - want_p).abs() < PVALUE_TOL,
            "case '{id}': p-value {} vs scipy {want_p} (tol {PVALUE_TOL})",
            got.pvalue
        );
        assert!(
            (got.df - 9.0).abs() < STATISTIC_TOL,
            "case '{id}': df {} is not 9",
            got.df
        );

        // RED (population std, ddof=0, instead of the (n-1) sample std): off by
        // sqrt(n/(n-1)) ~ 1.054 at n = 10 -- close enough to survive a loose
        // tolerance, wrong in every published interval.
        //   clear_positive_delta   RED 4.369676063  vs  GREEN 4.145438698
        //   small_negative_delta   RED -36.296967911 vs GREEN -34.434327227
        let red = f64_field(case, "red_statistic_population_std");
        if (want_stat - red).abs() > RED_DISCRIMINATION_MARGIN {
            assert!(
                (got.statistic - red).abs() > RED_DISCRIMINATION_MARGIN,
                "case '{id}': reproduced the RED population-std statistic {red}; \
                 the (n-1) divisor is missing"
            );
        }
    }
}

#[test]
fn ttest_1samp_f64_on_the_differences_is_the_same_number() {
    // ttest_rel_f64 must be a thin wrapper over ttest_1samp_f64(diffs, 0.0), not a
    // second implementation of the same closed form (OPS-03).
    let cases = fixture_cases(PAIRED_T_FIXTURE);
    for case in cases.iter().filter(|c| c["kind"] == "finite") {
        let id = case_id(case);
        let a = f64_array(case, "a");
        let b = f64_array(case, "b");
        let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

        let paired = ttest_rel_f64(&a, &b).expect("finite case");
        let one = ttest_1samp_f64(&diffs, 0.0).expect("finite case");

        assert_eq!(
            paired.statistic, one.statistic,
            "case '{id}': the paired and one-sample statistics must be bit-identical"
        );
        assert_eq!(
            paired.pvalue, one.pvalue,
            "case '{id}': p-values must match"
        );
    }
}

#[test]
fn paired_ci95_matches_the_fixture_bounds() {
    let cases = fixture_cases(PAIRED_T_FIXTURE);
    for case in cases.iter().filter(|c| c["kind"] == "finite") {
        let id = case_id(case);
        let a = f64_array(case, "a");
        let b = f64_array(case, "b");

        let ci = paired_ci(&a, &b, T_CRIT_975_DF9).expect("finite case must produce a CI");

        for (label, got, want) in [
            ("mean_diff", ci.mean_diff, f64_field(case, "mean_diff")),
            ("std_diff", ci.std_diff, f64_field(case, "std_diff")),
            ("std_err", ci.std_err, f64_field(case, "std_err")),
            (
                "half_width",
                ci.half_width,
                f64_field(case, "ci_half_width"),
            ),
            ("low", ci.low, f64_field(case, "ci_low")),
            ("high", ci.high, f64_field(case, "ci_high")),
        ] {
            assert!(
                (got - want).abs() < STATISTIC_TOL,
                "case '{id}': {label} {got} vs reference {want} (tol {STATISTIC_TOL})"
            );
        }
        assert_eq!(ci.n, 10, "case '{id}': n must be 10");

        // RED (half-width computed as t_crit * s_d, forgetting the / sqrt(n)):
        // inflates every interval by sqrt(10) ~ 3.162.
        //   clear_positive_delta  RED 0.035720919  vs  GREEN 0.011295946
        let red = f64_field(case, "red_ci_half_width_no_sqrt_n");
        assert!(
            (ci.half_width - red).abs() > RED_DISCRIMINATION_MARGIN,
            "case '{id}': reproduced the RED half-width {red}; the / sqrt(n) is missing"
        );
    }
}

#[test]
fn paired_ci95_df9_uses_the_frozen_constant() {
    let cases = fixture_cases(PAIRED_T_FIXTURE);
    let case = cases
        .iter()
        .find(|c| c["id"] == "clear_positive_delta")
        .expect("clear_positive_delta fixture case present");
    let a = f64_array(case, "a");
    let b = f64_array(case, "b");

    let explicit = paired_ci(&a, &b, T_CRIT_975_DF9).expect("finite case");
    let frozen = paired_ci95_df9(&a, &b).expect("finite case");
    assert_eq!(
        explicit.half_width, frozen.half_width,
        "paired_ci95_df9 must pass T_CRIT_975_DF9 and nothing else"
    );
}

// ---- the typed degenerate case -------------------------------------------------------

#[test]
fn zero_variance_paired_input_is_a_typed_refusal() {
    let cases = fixture_cases(PAIRED_T_FIXTURE);
    let degenerate: Vec<_> = cases
        .iter()
        .filter(|c| c["kind"] == "degenerate_zero_variance")
        .collect();
    assert_eq!(
        degenerate.len(),
        2,
        "both the constant-nonzero-difference and all-zero-difference shapes must be pinned"
    );

    for case in degenerate {
        let id = case_id(case);
        assert_eq!(
            case["expect"].as_str(),
            Some("ZeroVarianceDifferences"),
            "case '{id}': fixture must name the typed error"
        );
        let a = f64_array(case, "a");
        let b = f64_array(case, "b");
        let constant = f64_field(case, "constant_difference");

        for (surface, err) in [
            ("ttest_rel_f64", ttest_rel_f64(&a, &b).err()),
            ("paired_ci", paired_ci(&a, &b, T_CRIT_975_DF9).err()),
            ("paired_ci95_df9", paired_ci95_df9(&a, &b).err()),
        ] {
            let err = err.unwrap_or_else(|| {
                panic!("case '{id}': {surface} returned Ok on a zero-variance input")
            });
            match err {
                AprenderError::ZeroVarianceDifferences { n, constant_value } => {
                    assert_eq!(n, 10, "case '{id}': {surface} reported n = {n}");
                    assert_eq!(
                        constant_value, constant,
                        "case '{id}': {surface} reported the wrong constant difference"
                    );
                }
                other => {
                    panic!("case '{id}': {surface} returned {other:?}, not ZeroVarianceDifferences")
                }
            }
        }
    }
}

#[test]
fn zero_variance_refusal_covers_the_all_zero_shape_specifically() {
    // The all-zero case is 0/0, not "a big number over a tiny one". A guard written as
    // `if d_bar != 0.0 && s_d == 0.0` would let this one through as a spuriously fine
    // answer, so it gets its own named test rather than only riding the loop above.
    let same = [
        0.5_f64, 0.75, 0.25, 0.625, 0.375, 0.875, 0.125, 0.6875, 0.4375, 0.5625,
    ];
    let err = ttest_rel_f64(&same, &same).expect_err("identical arms must refuse");
    assert!(
        matches!(err, AprenderError::ZeroVarianceDifferences { .. }),
        "identical arms produced {err:?}, not ZeroVarianceDifferences"
    );
}

#[test]
fn numerically_constant_input_also_refuses() {
    // The SECOND degenerate shape: the values are not bit-identical, but their squared
    // deviations underflow to zero at this scale, so the computed (n-1) variance is
    // exactly 0. Without the post-computation `std == 0.0` check the statistic would be
    // x/0. Reached by real inputs, so it gets a real test rather than a comment.
    let tiny = [1e-200_f64, 2e-200, 3e-200];
    let err = ttest_1samp_f64(&tiny, 0.0).expect_err("underflowed variance must refuse");
    match err {
        AprenderError::ZeroVarianceDifferences { n, constant_value } => {
            assert_eq!(n, 3);
            assert!(
                (constant_value - 2e-200).abs() < 1e-210,
                "constant_value should report the mean, got {constant_value:?}"
            );
        }
        other => panic!("underflowed variance produced {other:?}"),
    }
}

// ---- the f64 special functions the p-value rests on ----------------------------------

#[test]
fn ln_gamma_f64_matches_closed_form_values() {
    // The Lanczos g=7 coefficients are otherwise only validated INDIRECTLY, through the
    // p-value parity assertions. A transposed digit in the table would show up there as
    // a confusing tail mismatch; here it shows up as what it is.
    //
    //   ln Γ(1/2) = ln √π          ln Γ(1) = 0            ln Γ(5) = ln 24
    //   ln Γ(9/2) is the a = df/2 argument the df = 9 t-tail actually uses.
    for (z, want) in [
        (0.5_f64, std::f64::consts::PI.sqrt().ln()),
        (1.0, 0.0),
        (5.0, 24.0_f64.ln()),
        (4.5, 11.631_728_396_567_448_f64.ln()),
    ] {
        let got = ln_gamma_f64(z);
        assert!(
            (got - want).abs() < 1e-12,
            "ln_gamma_f64({z}) = {got}, expected {want}"
        );
    }
}

#[test]
fn incomplete_beta_f64_satisfies_the_symmetry_identity() {
    // I_x(a,b) + I_{1-x}(b,a) = 1 exercises BOTH continued-fraction branches on the same
    // pair, so a defect confined to the `1 - bt * cf(b,a,1-x) / b` arm cannot hide.
    for (a, b, x) in [
        (4.5_f64, 0.5_f64, 0.3_f64),
        (4.5, 0.5, 0.9),
        (2.0, 3.0, 0.45),
    ] {
        let lhs = incomplete_beta_f64(a, b, x) + incomplete_beta_f64(b, a, 1.0 - x);
        assert!(
            (lhs - 1.0).abs() < 1e-12,
            "I_{x}({a},{b}) + I_{{1-x}}({b},{a}) = {lhs}, expected 1"
        );
    }
}

#[test]
fn t_distribution_pvalue_f64_is_more_accurate_than_the_f32_path() {
    // Why the claims layer does not simply call the existing f32 tail: at the scale of a
    // published p-value the f32 path's own rounding is larger than the 1e-9 band the
    // fixtures are asserted at. This records the size of that gap rather than asserting
    // it from principle.
    let cases = fixture_cases(PAIRED_T_FIXTURE);
    let case = cases
        .iter()
        .find(|c| c["id"] == "near_tie")
        .expect("near_tie fixture case present");
    let a = f64_array(case, "a");
    let b = f64_array(case, "b");
    let want = f64_field(case, "pvalue");

    let f64_p = ttest_rel_f64(&a, &b).expect("finite case").pvalue;
    let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let f32_p = f64::from(ttest_rel(&a32, &b32).expect("finite case").pvalue);

    assert!(
        (f64_p - want).abs() < PVALUE_TOL,
        "f64 p-value {f64_p} vs scipy {want}"
    );
    assert!(
        (f64_p - want).abs() < (f32_p - want).abs(),
        "the f64 mirror must be strictly closer to scipy than the f32 path: \
         f64 err {}, f32 err {}",
        (f64_p - want).abs(),
        (f32_p - want).abs()
    );
}

// ---- typed errors on the structural paths -------------------------------------------

#[test]
fn ttest_rel_f64_refuses_length_mismatch() {
    let err =
        ttest_rel_f64(&[1.0, 2.0, 3.0], &[1.0, 2.0]).expect_err("length mismatch must refuse");
    assert!(
        matches!(err, AprenderError::DimensionMismatch { .. }),
        "length mismatch produced {err:?}, not DimensionMismatch"
    );
}

#[test]
fn ttest_1samp_f64_refuses_fewer_than_two_samples() {
    assert!(ttest_1samp_f64(&[1.0], 0.0).is_err());
    assert!(ttest_1samp_f64(&[], 0.0).is_err());
}

#[test]
fn paired_ci95_df9_refuses_a_sample_that_is_not_ten_long() {
    // The constant is frozen at df = 9. Handing this helper any other n would silently
    // apply the wrong critical value, so it refuses rather than computing.
    let a = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let b = [1.5_f64, 2.5, 2.5, 4.5, 5.5];
    assert!(
        paired_ci95_df9(&a, &b).is_err(),
        "paired_ci95_df9 must refuse n != 10; the frozen constant only applies at df = 9"
    );
}

// ---- aggregation helpers the report reuses -------------------------------------------

#[test]
fn aggregation_helpers_match_the_fixture_moments() {
    let cases = fixture_cases(PAIRED_T_FIXTURE);
    for case in cases.iter().filter(|c| c["kind"] == "finite") {
        let id = case_id(case);
        let a = f64_array(case, "a");
        let b = f64_array(case, "b");
        let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

        let mean = mean_f64(&diffs).expect("non-empty");
        let std = sample_std_f64(&diffs).expect("n >= 2");
        assert!(
            (mean - f64_field(case, "mean_diff")).abs() < STATISTIC_TOL,
            "case '{id}': mean_f64 {mean}"
        );
        assert!(
            (std - f64_field(case, "std_diff")).abs() < STATISTIC_TOL,
            "case '{id}': sample_std_f64 {std}"
        );

        let (lo, hi) = min_max_f64(&diffs).expect("non-empty");
        assert!(
            lo <= mean && mean <= hi,
            "case '{id}': mean outside [min, max]"
        );
    }
}

#[test]
fn sample_std_f64_uses_the_n_minus_one_divisor() {
    // [1, 2, 3, 4]: (n-1) std = sqrt(5/3) = 1.290994...; population std = sqrt(5/4)
    // = 1.118034... A test that only asserted "positive" would pass for both.
    let std = sample_std_f64(&[1.0, 2.0, 3.0, 4.0]).expect("n >= 2");
    assert!(
        (std - (5.0_f64 / 3.0).sqrt()).abs() < 1e-12,
        "sample_std_f64 = {std}, expected sqrt(5/3) = {}",
        (5.0_f64 / 3.0).sqrt()
    );
    assert!(
        (std - (5.0_f64 / 4.0).sqrt()).abs() > 1e-6,
        "sample_std_f64 used the POPULATION divisor"
    );
}

// ---- source guards: no RNG, no non-finite escape hatch --------------------------------

const HYPOTHESIS_SOURCE: &str = include_str!("hypothesis.rs");

/// Non-comment source lines, so a guard cannot be satisfied or tripped by prose.
fn source_without_comments() -> String {
    HYPOTHESIS_SOURCE
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            !(t.starts_with("//") || t.starts_with("///") || t.starts_with("//!"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn no_rng_enters_the_claims_statistics_path() {
    // D-06: "exactly recompute" is bit-level, so a resampling stream anywhere in this
    // module would make the published numbers unreproducible. T-05-04-02.
    //
    // The plan's criterion lists `sample` as an RNG symbol. It is not one HERE: it is
    // the statistical noun, and it is already the parameter name of the pre-existing
    // f32 `ttest_1samp(sample: &[f32], ...)`. Banning it would ban the existing API and
    // catch nothing. These are the symbols that would actually indicate a random stream.
    let src = source_without_comments();
    for banned in [
        "rand::",
        "rand_chacha",
        "thread_rng",
        "SeedableRng",
        "StdRng",
        "gen_range",
        "bootstrap",
        "resample",
        "shuffle",
        "random(",
    ] {
        assert!(
            !src.contains(banned),
            "RNG symbol '{banned}' has entered aprender-core::stats::hypothesis; the \
             claims path is closed-form only (D-06)"
        );
    }
}

#[test]
fn no_non_finite_f64_literal_escapes_through_this_module() {
    // The zero-variance case must be a typed refusal, never a serialised non-finite
    // f64: serde_json renders NaN/Infinity as `null`, so one would become a silently
    // MISSING number in a published claims row (Ph3 CR-03).
    let src = source_without_comments();
    assert!(
        !src.contains("f64::NAN"),
        "f64::NAN appears in non-comment source; the degenerate case is a typed error"
    );
    assert!(
        !src.contains("f64::INFINITY"),
        "f64::INFINITY appears in non-comment source; the degenerate case is a typed error"
    );
}

#[test]
fn no_finite_result_is_ever_non_finite() {
    // The positive half of the guard above: every Ok result over the whole fixture set
    // is finite, so "typed error instead of NaN" is proven by behaviour and not only by
    // the absence of a literal.
    for case in fixture_cases(PAIRED_T_FIXTURE) {
        let a = f64_array(&case, "a");
        let b = f64_array(&case, "b");
        if let Ok(t) = ttest_rel_f64(&a, &b) {
            assert!(t.statistic.is_finite() && t.pvalue.is_finite() && t.df.is_finite());
        }
        if let Ok(ci) = paired_ci(&a, &b, T_CRIT_975_DF9) {
            assert!(ci.mean_diff.is_finite());
            assert!(ci.std_diff.is_finite());
            assert!(ci.std_err.is_finite());
            assert!(ci.half_width.is_finite());
            assert!(ci.low.is_finite() && ci.high.is_finite());
        }
    }
}
