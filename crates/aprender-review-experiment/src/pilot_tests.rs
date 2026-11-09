use super::*;

#[test]
fn the_design_point_sits_just_under_the_rule() {
    // §2.2 [C] quotes about 0.117 for 70 defect items at p = 0.5; that is the
    // Wald width. The frozen Wilson width the rule reads is about 0.114.
    let pr = project(Ratio::new(0, 0), &[], &[], 70, 140);
    assert_eq!(pr.basis, Basis::DesignWorstCase);
    let h = pr.half_width.expect("n > 0");
    assert!((h - 0.114).abs() < 0.001, "{h}");
    assert!(!pr.grow_corpus);
}

#[test]
fn falsify_rxp_001_a_wide_projection_grows_the_corpus() {
    // 40 defect items at the pilot's p = 0.5: half-width about 0.15 > 0.12.
    let pr = project(Ratio::new(3, 6), &[1.0], &[], 40, 80);
    assert_eq!(pr.basis, Basis::Pilot);
    assert!(pr.half_width.expect("n > 0") > MAX_HALF_WIDTH);
    assert!(pr.grow_corpus, "a > 0.12 projection must grow the corpus");
    // An unknown width (no test defects) is never a reason to keep the corpus.
    assert!(project(Ratio::new(3, 6), &[1.0], &[], 0, 80).grow_corpus);
}

#[test]
fn falsify_rxp_002_the_projection_uses_the_pilot_estimate() {
    // A high pilot recall narrows the interval below the rule at n = 50.
    let pr = project(Ratio::new(9, 10), &[1.0], &[], 50, 100);
    assert_eq!(pr.p, 0.9);
    assert!(!pr.grow_corpus, "{:?}", pr.half_width);
    // The same n at the design p = 0.5 would grow it: the estimate matters.
    assert!(project(Ratio::new(5, 10), &[1.0], &[], 50, 100).grow_corpus);
}

#[test]
fn falsify_rxp_003_runtime_counts_the_rerun_and_nothing_warm_is_unknown() {
    let pr = project(Ratio::new(1, 2), &[1000.0, 3000.0], &[9000.0], 70, 145);
    assert_eq!(pr.projected_requests, 145 + 15);
    assert_eq!(pr.projected_runtime_s, Some(2.0 * 160.0));
    let w = pr.warm.expect("warm ran");
    assert_eq!((w.n, w.p50, w.p95), (2, 1000.0, 3000.0));
    assert_eq!(pr.cold.map(|c| c.n), Some(1));
    let none = project(Ratio::new(1, 2), &[], &[], 70, 145);
    assert_eq!(
        none.projected_runtime_s, None,
        "no warm sample is not zero seconds"
    );
    assert!(Spread::of(&[1.0, f64::NAN]).is_none());
}
