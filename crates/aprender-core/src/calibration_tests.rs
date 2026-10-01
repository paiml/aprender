pub(crate) use super::*;

#[test]
fn test_temperature_scaling_new() {
    let ts = TemperatureScaling::new();
    assert_eq!(ts.temperature(), 1.0);
}

#[test]
fn test_temperature_scaling_calibrate() {
    let mut ts = TemperatureScaling::new();
    ts.temperature = 2.0;

    let logits = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let calibrated = ts.calibrate(&logits);

    assert_eq!(calibrated.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_temperature_scaling_predict_proba() {
    let ts = TemperatureScaling::new();
    let logits = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let probs = ts.predict_proba(&logits);

    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_temperature_scaling_fit() {
    let mut ts = TemperatureScaling::new();

    let logits = vec![
        Vector::from_slice(&[2.0, 1.0]),
        Vector::from_slice(&[1.0, 2.0]),
        Vector::from_slice(&[3.0, 0.0]),
    ];
    let labels = vec![0, 1, 0];

    ts.fit(&logits, &labels);
    assert!(ts.temperature() > 0.0);
}

#[test]
fn test_platt_scaling_new() {
    let ps = PlattScaling::new();
    assert_eq!(ps.params(), (1.0, 0.0));
}

#[test]
fn test_platt_scaling_predict_proba() {
    let ps = PlattScaling::new();
    let prob = ps.predict_proba(0.0);
    assert!((prob - 0.5).abs() < 1e-5);
}

#[test]
fn test_platt_scaling_fit() {
    let mut ps = PlattScaling::new();
    let logits = vec![2.0, 1.0, -1.0, -2.0, 0.5, -0.5];
    let labels = vec![true, true, false, false, true, false];

    ps.fit(&logits, &labels);
    // After fitting, higher logits should give higher probability
    assert!(ps.predict_proba(2.0) > ps.predict_proba(-2.0));
}

#[test]
fn test_ece_perfect_calibration() {
    let predictions = vec![0.9, 0.9, 0.1, 0.1];
    let labels = vec![true, true, false, false];

    let ece = expected_calibration_error(&predictions, &labels, 10);
    assert!(ece < 0.2);
}

#[test]
fn test_ece_poor_calibration() {
    let predictions = vec![0.9, 0.9, 0.9, 0.9];
    let labels = vec![true, false, false, false];

    let ece = expected_calibration_error(&predictions, &labels, 10);
    assert!(ece > 0.5);
}

#[test]
fn test_mce() {
    let predictions = vec![0.9, 0.9, 0.1, 0.1];
    let labels = vec![true, true, false, false];

    let mce = maximum_calibration_error(&predictions, &labels, 10);
    assert!(mce < 0.2);
}

#[test]
fn test_softmax() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = softmax(&logits);

    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    assert!(probs[2] > probs[1]);
    assert!(probs[1] > probs[0]);
}

#[test]
fn test_sigmoid() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-5);
    assert!(sigmoid(10.0) > 0.99);
    assert!(sigmoid(-10.0) < 0.01);
}

#[test]
fn test_isotonic_new() {
    let iso = IsotonicRegression::new();
    assert!(iso.thresholds.is_empty());
    assert!(iso.values.is_empty());
}

#[test]
fn test_isotonic_fit() {
    let mut iso = IsotonicRegression::new();
    let predictions = vec![0.1, 0.4, 0.6, 0.9];
    let labels = vec![false, false, true, true];

    iso.fit(&predictions, &labels);

    assert!(!iso.thresholds.is_empty());
    assert!(!iso.values.is_empty());
}

#[test]
fn test_isotonic_predict() {
    let mut iso = IsotonicRegression::new();
    let predictions = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let labels = vec![false, false, true, true, true];

    iso.fit(&predictions, &labels);

    // Test predictions at various points
    let p1 = iso.predict(0.2);
    let p2 = iso.predict(0.8);

    // Low prediction should give low calibrated value
    // High prediction should give high calibrated value
    assert!(
        p2 >= p1,
        "Higher predictions should give higher calibrated values"
    );
    assert!((0.0..=1.0).contains(&p1));
    assert!((0.0..=1.0).contains(&p2));
}

#[test]
fn test_isotonic_monotonic() {
    let mut iso = IsotonicRegression::new();
    // Non-monotonic accuracy pattern
    let predictions = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let labels = vec![false, true, false, true, true, false, true, true, true];

    iso.fit(&predictions, &labels);

    // Calibrated values should be monotonically non-decreasing
    let mut prev = iso.predict(0.0);
    for i in 1..=10 {
        let x = i as f32 / 10.0;
        let curr = iso.predict(x);
        assert!(
            curr >= prev - 1e-6,
            "Isotonic should be monotonic: {curr} < {prev}"
        );
        prev = curr;
    }
}

// PMAT-870: PAV pooled-block flatness (sklearn IsotonicRegression parity).
//
// After Pool-Adjacent-Violators pooling, the fitted function must be FLAT at the
// pooled value across the WHOLE x-range of a pooled block. The bug recorded only
// the block's min-x as a knot, discarding the max-x, so a query strictly inside a
// pooled block was linearly interpolated toward the NEXT block instead of returning
// the constant pooled value.
//
// Reference: sklearn.isotonic.IsotonicRegression
//   X=[0.0,0.1,0.2,0.3], y=[0,1,0,1] → PAV pools x=0.1,0.2 to value 0.5
//   X_thresholds_=[0.0,0.1,0.2,0.3], y_thresholds_=[0.0,0.5,0.5,1.0]
//   predict(0.2)=0.5, predict(0.15)=0.5, predict(0.25)=0.75
#[test]
fn test_isotonic_pav_block_is_flat_pmat_870() {
    let mut iso = IsotonicRegression::new();
    let predictions = vec![0.0, 0.1, 0.2, 0.3];
    let labels = vec![false, true, false, true]; // y = [0, 1, 0, 1]

    iso.fit(&predictions, &labels);

    // Inside the pooled [0.1, 0.2] block → flat at the pooled value 0.5.
    // RED (bug): predict(0.2)=0.75, predict(0.15)=0.625
    // GREEN (fix): both = 0.5
    let p_max_edge = iso.predict(0.2);
    let p_mid = iso.predict(0.15);
    assert!(
        (p_max_edge - 0.5).abs() < 1e-5,
        "PMAT-870 FALSIFIED: predict(0.2)={p_max_edge}, expected 0.5 (flat pooled block)"
    );
    assert!(
        (p_mid - 0.5).abs() < 1e-5,
        "PMAT-870 FALSIFIED: predict(0.15)={p_mid}, expected 0.5 (flat pooled block)"
    );

    // Between the pooled block (val 0.5 at x=0.2) and the final block (val 1.0 at
    // x=0.3): linear interpolation. predict(0.25) = 0.5 + 0.5*(1.0-0.5) = 0.75.
    // RED (bug): 0.875
    let p_interp = iso.predict(0.25);
    assert!(
        (p_interp - 0.75).abs() < 1e-5,
        "PMAT-870 FALSIFIED: predict(0.25)={p_interp}, expected 0.75 (interp between blocks)"
    );
}

// PMAT-870: NO-POOL guard — when no adjacent violators exist, behavior must be
// unchanged. Each x is its own block, so interpolation between distinct single-point
// blocks still applies. X=[0,0.2,0.4,0.6,0.8,1.0], y=[0,0,0,1,1,1].
// PAV pools the two halves: block A (x 0..0.4, val 0) and block B (x 0.6..1.0, val 1).
// predict(0.5) interpolates between A's max-edge (x=0.4, val 0) and B's min-edge
// (x=0.6, val 1): 0.0 + 0.5*(1.0-0.0) = 0.5.
#[test]
fn test_isotonic_no_pool_guard_pmat_870() {
    let mut iso = IsotonicRegression::new();
    let predictions = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let labels = vec![false, false, false, true, true, true];

    iso.fit(&predictions, &labels);

    let p = iso.predict(0.5);
    assert!(
        (p - 0.5).abs() < 1e-5,
        "PMAT-870 FALSIFIED (no-pool guard): predict(0.5)={p}, expected 0.5"
    );
    // Endpoints stay clamped at the block values.
    assert!((iso.predict(0.1) - 0.0).abs() < 1e-5);
    assert!((iso.predict(0.9) - 1.0).abs() < 1e-5);
}

#[test]
fn test_reliability_diagram() {
    let predictions = vec![0.1, 0.2, 0.8, 0.9];
    let labels = vec![false, false, true, true];

    let diagram = reliability_diagram(&predictions, &labels, 5);

    assert_eq!(diagram.len(), 5);
    for bin in &diagram {
        assert!(bin.0 >= 0.0 && bin.0 <= 1.0);
        assert!(bin.1 >= 0.0 && bin.1 <= 1.0);
    }
}

#[test]
fn test_brier_score() {
    // Perfect predictions
    let predictions = vec![1.0, 0.0, 1.0, 0.0];
    let labels = vec![true, false, true, false];
    let brier = brier_score(&predictions, &labels);
    assert!((brier - 0.0).abs() < 1e-6);

    // Worst predictions
    let predictions = vec![0.0, 1.0, 0.0, 1.0];
    let labels = vec![true, false, true, false];
    let brier = brier_score(&predictions, &labels);
    assert!((brier - 1.0).abs() < 1e-6);
}

// ====================================================================================
// MULTICLASS CALIBRATION (plan 05-04 T2, D-07) -- fixture parity against the pinned
// uv environment (scipy 1.18.0 / scikit-learn 1.9.0).
// ====================================================================================
//
// The fixtures are embedded with `include_str!` rather than read at run time. A test that
// silently skips when a file is absent is a test that proves nothing (the thresholds.rs
// rule), and these bytes are the ONLY evidence that the two new metrics compute what
// their contract equations say they compute.
//
// Regenerate with:
//   cd scripts/setfit_fixtures && uv run python gen_claims_fixtures.py
//
// Each fixture case additionally carries RED values -- what a NAMED wrong implementation
// produces on that exact input. Asserting only the GREEN value would pass for any
// implementation that happens to land nearby; asserting that the result is NOT the RED
// value is what makes these tests falsifying (glm_tests.rs:274-298 precedent).

const ECE_FIXTURES: &str =
    include_str!("../../../scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json");
const BRIER_FIXTURES: &str =
    include_str!("../../../scripts/setfit_fixtures/claims_stats/brier_multiclass_cases.json");

/// Fixture parity band. The reference is f64 numpy; these functions are f32 to match the
/// binary pair they sit beside, so ~1e-7 of representation drift is expected and 1e-6 is
/// the band the plan contracts.
const FIXTURE_TOL: f32 = 1e-6;

/// A RED value closer than this to the GREEN value is not a usable discriminator (it
/// happens when a case has a single occupied bin, or when every value is zero), so the
/// not-equal half of the assertion is skipped for that case rather than weakened.
const RED_DISCRIMINATION_MARGIN: f32 = 1e-4;

fn fixture_cases(raw: &str) -> Vec<serde_json::Value> {
    let doc: serde_json::Value = serde_json::from_str(raw).expect("fixture JSON parses");
    doc["cases"]
        .as_array()
        .expect("fixture has a cases array")
        .clone()
}

fn flat_probabilities(case: &serde_json::Value) -> (Vec<f32>, usize) {
    let rows = case["probabilities"]
        .as_array()
        .expect("case has probabilities");
    let n_classes = rows[0].as_array().expect("row is an array").len();
    let mut flat = Vec::with_capacity(rows.len() * n_classes);
    for row in rows {
        let row = row.as_array().expect("row is an array");
        assert_eq!(row.len(), n_classes, "ragged probability matrix in fixture");
        for v in row {
            flat.push(v.as_f64().expect("probability is a number") as f32);
        }
    }
    (flat, n_classes)
}

fn fixture_labels(case: &serde_json::Value) -> Vec<usize> {
    case["labels"]
        .as_array()
        .expect("case has labels")
        .iter()
        .map(|v| v.as_u64().expect("label is a non-negative integer") as usize)
        .collect()
}

fn fixture_f32(case: &serde_json::Value, key: &str) -> f32 {
    case[key]
        .as_f64()
        .unwrap_or_else(|| panic!("fixture case missing numeric field '{key}'")) as f32
}

/// Assert the computed value matches GREEN and is distinguishable from a named RED value.
fn assert_green_not_red(id: &str, metric: &str, got: f32, green: f32, red: f32, red_name: &str) {
    assert!(
        (got - green).abs() < FIXTURE_TOL,
        "{metric} fixture parity FAILED for case '{id}': got {got}, reference {green} \
         (tolerance {FIXTURE_TOL})"
    );
    if (green - red).abs() > RED_DISCRIMINATION_MARGIN {
        assert!(
            (got - red).abs() > RED_DISCRIMINATION_MARGIN,
            "{metric} case '{id}' reproduced the RED value of '{red_name}' ({red}); \
             the implementation is the WRONG one, not merely imprecise"
        );
    }
}

#[test]
fn top_label_ece_matches_pinned_env_fixtures() {
    let cases = fixture_cases(ECE_FIXTURES);
    assert!(
        cases.len() >= 4,
        "fixture set shrank below the contracted 4 cases"
    );

    for case in &cases {
        let id = case["id"].as_str().expect("case has an id");
        let (probs, n_classes) = flat_probabilities(case);
        let labels = fixture_labels(case);
        let n_bins = case["n_bins"].as_u64().expect("case has n_bins") as usize;

        let got = expected_calibration_error_top_label(&probs, n_classes, &labels, n_bins);

        // RED (confidence read from the TRUE label's column instead of the maximum):
        //   overconfident_sharp  0.011659185  vs  GREEN 0.583457378
        //   saturated_top_bin    0.021875000  vs  GREEN 0.353125000
        // RED (bin gaps averaged unweighted instead of by occupancy n_b/N):
        //   mixed_hard           0.410361899  vs  GREEN 0.363084624
        assert_green_not_red(
            id,
            "top-label ECE",
            got,
            fixture_f32(case, "ece"),
            fixture_f32(case, "red_ece_true_label_conf"),
            "true-label confidence",
        );
        assert_green_not_red(
            id,
            "top-label ECE",
            got,
            fixture_f32(case, "ece"),
            fixture_f32(case, "red_ece_unweighted_bins"),
            "unweighted bin average",
        );
    }
}

#[test]
fn top_label_ece_clamps_confidence_of_exactly_one() {
    // `conf * n_bins` is exactly 10.0 for a saturated row, so without the
    // `.min(n_bins - 1)` guard the bin index is out of range. This is the concrete
    // refutation of the "may index out of bounds at conf == 1.0" review concern:
    // the binary pair's indexing rule already handles it, and this test proves it.
    let probs = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let labels = vec![0usize, 1usize];
    let ece = expected_calibration_error_top_label(&probs, 3, &labels, 10);
    assert!(
        (ece - 0.0).abs() < FIXTURE_TOL,
        "two saturated, correct rows are perfectly calibrated; got {ece}"
    );
}

#[test]
fn brier_score_multiclass_matches_pinned_env_fixtures() {
    let cases = fixture_cases(BRIER_FIXTURES);
    assert!(
        cases.len() >= 4,
        "fixture set shrank below the contracted 4 cases"
    );

    for case in &cases {
        let id = case["id"].as_str().expect("case has an id");
        let (probs, n_classes) = flat_probabilities(case);
        let labels = fixture_labels(case);

        let got = brier_score_multiclass(&probs, n_classes, &labels);

        // RED (class sum divided by K -- the silent renormalisation the 05-REVIEWS
        // finding warns about):  sharp_wrong 0.615966667 vs GREEN 1.847900000
        // RED (only the true class's term kept):
        //                        sharp_wrong 0.950683333 vs GREEN 1.847900000
        assert_green_not_red(
            id,
            "multiclass Brier",
            got,
            fixture_f32(case, "brier_multiclass"),
            fixture_f32(case, "red_brier_divided_by_k"),
            "divide-by-K renormalisation",
        );
        assert_green_not_red(
            id,
            "multiclass Brier",
            got,
            fixture_f32(case, "brier_multiclass"),
            fixture_f32(case, "red_brier_true_class_only"),
            "true-class term only",
        );
    }
}

#[test]
fn brier_score_multiclass_exceeds_one_for_confidently_wrong_k3() {
    // The unnormalised multiclass Brier has range [0, 2], NOT [0, 1]. A [0,1] bound
    // assertion anywhere in this surface would be false; the `sharp_wrong` fixture
    // (1.8479) is the standing counterexample and this test states it in-line.
    let cases = fixture_cases(BRIER_FIXTURES);
    let case = cases
        .iter()
        .find(|c| c["id"] == "sharp_wrong")
        .expect("sharp_wrong fixture case present");
    let (probs, n_classes) = flat_probabilities(case);
    let labels = fixture_labels(case);
    let got = brier_score_multiclass(&probs, n_classes, &labels);
    assert!(
        got > 1.0,
        "confidently-wrong K=3 predictions must exceed 1.0 under the unnormalised \
         definition; got {got}"
    );
    assert!(
        got <= 2.0,
        "multiclass Brier is bounded above by 2.0; got {got}"
    );
}

#[test]
fn brier_score_multiclass_is_exactly_twice_binary_at_k2() {
    // NORMALIZATION CONSISTENCY (05-REVIEWS consensus item 3).
    //
    // For K = 2 with p_i = [1 - p, p] and one-hot y_i = [1 - y, y]:
    //     sum_k (p_ik - y_ik)^2 = (p - y)^2 + ((1 - p) - (1 - y))^2
    //                           = (p - y)^2 + (y - p)^2
    //                           = 2 (p - y)^2
    // so the declared UNNORMALISED multiclass score is definitionally TWICE the binary
    // `brier_score`, and can never equal it for a non-zero score. The test asserts the
    // factor of 2 AND asserts the two are not equal, so a future silent divide-by-K
    // renormalisation turns this red in both directions.
    let binary_p = [0.9_f32, 0.15, 0.62, 0.05, 0.48, 0.77];
    let binary_labels = [true, false, true, false, false, true];

    let mut flat = Vec::with_capacity(binary_p.len() * 2);
    let mut labels = Vec::with_capacity(binary_p.len());
    for (&p, &y) in binary_p.iter().zip(binary_labels.iter()) {
        flat.push(1.0 - p);
        flat.push(p);
        labels.push(usize::from(y));
    }

    let binary = brier_score(&binary_p, &binary_labels);
    let multiclass = brier_score_multiclass(&flat, 2, &labels);

    assert!(
        (multiclass - 2.0 * binary).abs() < FIXTURE_TOL,
        "K=2 multiclass Brier must equal EXACTLY 2 x the binary brier_score: \
         multiclass {multiclass}, binary {binary}, 2 x binary {}",
        2.0 * binary
    );
    assert!(
        (multiclass - binary).abs() > RED_DISCRIMINATION_MARGIN,
        "K=2 multiclass Brier must NOT equal the binary brier_score -- an equality here \
         means someone renormalised the class sum by K. multiclass {multiclass}, \
         binary {binary}"
    );
}

// ---- contracted precondition failures (not garbage values) ------------------------

#[test]
#[should_panic(expected = "n_bins")]
fn top_label_ece_refuses_zero_bins() {
    let _ = expected_calibration_error_top_label(&[0.5, 0.3, 0.2], 3, &[0], 0);
}

#[test]
#[should_panic(expected = "label")]
fn top_label_ece_refuses_label_out_of_range() {
    let _ = expected_calibration_error_top_label(&[0.5, 0.3, 0.2], 3, &[3], 10);
}

#[test]
#[should_panic(expected = "row")]
fn top_label_ece_refuses_row_not_summing_to_one() {
    let _ = expected_calibration_error_top_label(&[0.5, 0.3, 0.9], 3, &[0], 10);
}

#[test]
#[should_panic(expected = "precondition violated")]
fn top_label_ece_refuses_empty_input() {
    // The CONTRACT's `input.len() > 0` precondition fires here, ahead of the structural
    // asserts -- that ordering is the point, so the expected substring names the
    // contract's message rather than the local one.
    let _ = expected_calibration_error_top_label(&[], 3, &[], 10);
}

#[test]
#[should_panic(expected = "label")]
fn brier_multiclass_refuses_label_out_of_range() {
    let _ = brier_score_multiclass(&[0.5, 0.3, 0.2], 3, &[7]);
}

#[test]
#[should_panic(expected = "row")]
fn brier_multiclass_refuses_row_not_summing_to_one() {
    let _ = brier_score_multiclass(&[0.5, 0.3, 0.9], 3, &[0]);
}

#[test]
#[should_panic(expected = "precondition violated")]
fn brier_multiclass_refuses_empty_input() {
    let _ = brier_score_multiclass(&[], 3, &[]);
}

#[test]
#[should_panic(expected = "labels")]
fn brier_multiclass_refuses_label_count_mismatch() {
    // Two rows of probabilities, one label.
    let _ = brier_score_multiclass(&[0.5, 0.3, 0.2, 0.1, 0.2, 0.7], 3, &[0]);
}

/// Plan 08-27: the f64 exactly-summed top-label ECE (the Laya gate's) matches every frozen house
/// case within the fixture band and the hand-computed cases exactly, and leaves the f32
/// function's results where they were (frozen bit for bit in metrics::f32_bits_tests).
#[test]
fn top_label_ece_f64_matches_house_cases_and_hand_values() {
    for case in &fixture_cases(ECE_FIXTURES) {
        let id = case["id"].as_str().expect("case has an id");
        let (probs, n_classes) = flat_probabilities(case);
        let labels = fixture_labels(case);
        let n_bins = case["n_bins"].as_u64().expect("case has n_bins") as usize;
        let got = expected_calibration_error_top_label_f64(&probs, n_classes, &labels, n_bins);
        let want = case["ece"].as_f64().expect("ece");
        assert!(
            (got - want).abs() < f64::from(FIXTURE_TOL),
            "f64 top-label ECE on '{id}': got {got}, reference {want}"
        );
    }
    // Always predicts class 0 at 0.6 on y = [0, 1, 2, 0]: one bin, acc 0.5, conf 0.6 (as f32).
    let probs = [0.6_f32, 0.2, 0.2].repeat(4);
    let ece = expected_calibration_error_top_label_f64(&probs, 3, &[0, 1, 2, 0], 15);
    assert_eq!(ece.to_bits(), (f64::from(0.6_f32) - 0.5).to_bits());
    // A saturated row lands in the top bin (the clamp), as in the f32 function.
    let ece = expected_calibration_error_top_label_f64(&[1.0, 0.0, 0.0, 1.0], 2, &[0, 0], 15);
    assert_eq!(ece, 0.5);
}
