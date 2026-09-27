//! Falsifiers for the tracing CRUX contracts (crux-F-22..25, TRACE-001 TR-11).
//!
//! Each `falsify_*` test plants the defect its contract names and asserts
//! the metric turns it RED; each `specimen_*` test is the positive case
//! that must stay GREEN.

use super::*;

fn counts(pairs: &[(&str, u64)]) -> BTreeMap<String, u64> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

fn set(names: &[&str]) -> BTreeSet<String> {
    names.iter().map(|s| (*s).to_string()).collect()
}

/// W1-shaped strace runs with ±2 jitter on `read`/`write` only.
fn strace_runs() -> Vec<BTreeMap<String, u64>> {
    (0..10u64)
        .map(|i| {
            counts(&[
                ("openat", 40),
                ("read", 1000 + i % 3),
                ("write", 500 + (i + 1) % 3),
                ("mmap", 60),
            ])
        })
        .collect()
}

fn self_noise(runs: &[BTreeMap<String, u64>]) -> Vec<f64> {
    runs.windows(2)
        .filter_map(|w| syscall_divergence(&w[0], &w[1]))
        .collect()
}

// ── crux-F-22 syscall-trace (E-S1, E-S2, KANI-CRUX-D) ──────────────────────

#[test]
fn falsify_crux_f22_001_renacer_dropping_openat_fails_parity() {
    let strace = strace_runs();
    let noise = self_noise(&strace);
    let cross: Vec<f64> = strace
        .iter()
        .filter_map(|s| {
            let mut dropped = s.clone();
            dropped.remove("openat");
            syscall_divergence(&dropped, s)
        })
        .collect();
    assert_eq!(parity_ok(&cross, &noise), Some(false));
}

#[test]
fn specimen_crux_f22_001_faithful_renacer_passes_parity() {
    let strace = strace_runs();
    let noise = self_noise(&strace);
    let cross: Vec<f64> = strace
        .iter()
        .rev()
        .zip(strace.iter())
        .filter_map(|(r, s)| syscall_divergence(r, s))
        .collect();
    assert_eq!(parity_ok(&cross, &noise), Some(true));
}

#[test]
fn falsify_crux_f22_002_unreported_overhead_is_not_a_number() {
    assert_eq!(overhead_gap_effect(f64::NAN, 1.2), None);
    assert_eq!(overhead_gap_effect(1.3, 0.0), None);
    assert_eq!(parity_ok(&[], &[0.1]), None, "no cross runs is not a pass");
    assert_eq!(
        parity_ok(&[0.0], &[]),
        None,
        "no noise sample is not a pass"
    );
}

#[test]
fn specimen_crux_f22_002_gap_effect_sign() {
    let cheaper = overhead_gap_effect(1.1, 2.2).expect("finite ratios");
    assert!(cheaper < 0.0, "renacer cheaper => gap_effect < 0");
    assert_eq!(overhead_gap_effect(1.5, 1.5), Some(0.0));
}

#[test]
fn falsify_crux_f22_003_d_identity_and_order() {
    let a = counts(&[("read", 7), ("write", 3), ("openat", 2)]);
    assert_eq!(syscall_divergence(&a, &a), Some(0.0));
    let mut b = a.clone();
    b.insert("fsync".into(), 1);
    let d = syscall_divergence(&b, &a).expect("den > 0");
    assert!(d > 0.0);
    assert_eq!(divergence_parts(&[1, 2, 3], &[3, 2, 1]), (4, 6));
    assert_eq!(divergence_parts(&[3, 2, 1], &[1, 2, 3]), (4, 6));
    assert_eq!(
        divergence_parts(&[2, 1, 3], &[2, 3, 1]),
        divergence_parts(&[1, 2, 3], &[3, 2, 1]),
        "same permutation on both vectors leaves D unchanged"
    );
    assert_eq!(syscall_divergence(&a, &BTreeMap::new()), None);
}

// ── crux-F-23 golden-trace (§2.5) ──────────────────────────────────────────

fn golden_baseline() -> BTreeMap<String, Vec<u64>> {
    let runs = strace_runs();
    let mut base: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for r in &runs {
        for (k, v) in r {
            base.entry(k.clone()).or_default().push(*v);
        }
    }
    base
}

#[test]
fn falsify_crux_f23_001_plus_10000_write_is_red_on_first_run() {
    let base = golden_baseline();
    let mut run = strace_runs().swap_remove(0);
    *run.get_mut("write").expect("write baselined") += 10_000;
    let v = golden_verdict(&base, &run);
    assert!(
        matches!(&v, GoldenVerdict::Breach { syscall, .. } if syscall == "write"),
        "{v:?}"
    );
    assert!(v.is_red());
}

#[test]
fn falsify_crux_f23_002_new_syscall_class_is_red() {
    let base = golden_baseline();
    let mut run = strace_runs().swap_remove(0);
    run.insert("fsync".into(), 1);
    assert_eq!(
        golden_verdict(&base, &run),
        GoldenVerdict::NewClass("fsync".into())
    );
}

#[test]
fn falsify_crux_f23_003_empty_baseline_cannot_pass() {
    let mut base = golden_baseline();
    base.insert("brk".into(), Vec::new());
    let run = strace_runs().swap_remove(0);
    assert!(golden_verdict(&base, &run).is_red());
}

#[test]
fn specimen_crux_f23_001_unchanged_build_green_on_every_baseline_run() {
    let base = golden_baseline();
    for run in strace_runs() {
        assert_eq!(golden_verdict(&base, &run), GoldenVerdict::Green);
    }
}

#[test]
fn specimen_crux_f23_002_deterministic_class_tolerates_one_call() {
    let (ell, theta) = golden_theta(&[40; 10]).expect("non-empty");
    assert!(
        (41f64.ln_1p() - ell).abs() <= theta + 1e-12,
        "one-call jitter"
    );
    assert!((43f64.ln_1p() - ell).abs() > theta, "three calls exceed θ");
}

// ── crux-F-24 kernel-profile (E-S3, KANI-CRUX-J) ───────────────────────────

fn nsys_times() -> BTreeMap<String, f64> {
    [
        ("gemv_q4k", 120.0),
        ("rmsnorm", 8.0),
        ("rope", 4.0),
        ("softmax", 6.0),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), *v))
    .collect()
}

const NSYS_NOISE: [f64; 8] = [0.01, 0.02, 0.015, 0.03, 0.005, 0.02, 0.025, 0.01];

#[test]
fn falsify_crux_f24_001_cgp_missing_one_kernel_fails() {
    let nsys = nsys_times();
    // Every position, including the last: a positional zip over the two
    // maps would silently stop early there and report agreement.
    for missing in nsys.keys() {
        let mut cgp = nsys.clone();
        cgp.remove(missing);
        let ks: BTreeSet<String> = cgp.keys().cloned().collect();
        let kn: BTreeSet<String> = nsys.keys().cloned().collect();
        assert!(kernel_jaccard(&ks, &kn) < 1.0, "{missing}");
        assert_eq!(
            kernel_times_ok(&cgp, &nsys, &NSYS_NOISE),
            Some(false),
            "cgp missing {missing}"
        );
        assert_eq!(
            kernel_times_ok(&nsys, &cgp, &NSYS_NOISE),
            Some(false),
            "nsys missing {missing}"
        );
    }
}

#[test]
fn falsify_crux_f24_002_time_outside_nsys_noise_fails() {
    let nsys = nsys_times();
    let mut cgp = nsys.clone();
    *cgp.get_mut("gemv_q4k").expect("present") *= 1.5;
    assert_eq!(kernel_times_ok(&cgp, &nsys, &NSYS_NOISE), Some(false));
    assert_eq!(kernel_times_ok(&cgp, &nsys, &[]), None);
}

#[test]
fn specimen_crux_f24_001_equal_sets_within_noise_pass() {
    let nsys = nsys_times();
    let cgp: BTreeMap<String, f64> = nsys.iter().map(|(k, v)| (k.clone(), v * 1.01)).collect();
    assert_eq!(kernel_times_ok(&cgp, &nsys, &NSYS_NOISE), Some(true));
    assert_eq!(kernel_jaccard(&set(&["a", "b"]), &set(&["b", "a"])), 1.0);
    assert_eq!(kernel_jaccard(&set(&[]), &set(&[])), 1.0);
}

#[test]
fn falsify_crux_f24_003_j_bounds() {
    assert_eq!(jaccard_parts(0b1011, 0b1011), (3, 3));
    assert_eq!(jaccard_parts(0b0001, 0b0010), (0, 2));
    let j = kernel_jaccard(&set(&["a", "b"]), &set(&["b", "c"]));
    assert!((0.0..1.0).contains(&j));
}

// ── crux-F-25 op-attribution (R-1) ─────────────────────────────────────────

#[test]
fn falsify_crux_f25_001_sum_exceeds_wall() {
    assert_eq!(
        op_attribution_verdict(Provenance::Measured, &[Some(600), Some(500)], 1000),
        Err(AttributionDefect::SumExceedsWall {
            sum_us: 1100,
            wall_us: 1000
        })
    );
}

#[test]
fn falsify_crux_f25_002_zero_fill_under_not_instrumented() {
    assert_eq!(
        op_attribution_verdict(Provenance::NotInstrumented, &[None, Some(0)], 1000),
        Err(AttributionDefect::ComponentUnderUnmeasured(1))
    );
    assert_eq!(
        op_attribution_verdict(Provenance::WallClockTotal, &[Some(1000)], 1000),
        Err(AttributionDefect::ComponentUnderUnmeasured(0)),
        "compute_us == duration_us under WallClockTotal is the F1 defect"
    );
    assert_eq!(
        op_attribution_verdict(Provenance::Measured, &[None, None], 1000),
        Err(AttributionDefect::MeasuredWithoutComponents)
    );
}

#[test]
fn specimen_crux_f25_001_measured_breakdown_under_wall() {
    assert_eq!(
        op_attribution_verdict(Provenance::Measured, &[Some(700), None, Some(250)], 1000),
        Ok(())
    );
    assert_eq!(
        op_attribution_verdict(Provenance::NotInstrumented, &[None, None], 1000),
        Ok(())
    );
}

#[test]
fn quantile_and_median_reject_non_finite() {
    assert_eq!(median(&[1.0, f64::NAN]), None);
    assert_eq!(quantile(&[1.0, 2.0], 0.0), None);
    assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.95), Some(4.0));
    assert_eq!(median(&[3.0, 1.0, 2.0, 4.0]), Some(2.5));
}
