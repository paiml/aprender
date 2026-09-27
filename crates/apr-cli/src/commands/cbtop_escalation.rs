// TR-02 (#4557, TRACE-001): cbtop anomaly escalation.
//
// The old `check_renacer_escalation` built a `BrickTracer` it never used and
// printed "BrickTracer: Enabled for syscall breakdown" — nothing was traced.
// It also escalated on "efficiency" = tok/s ÷ a hardcoded 976 tok/s (a 1.5B
// target applied to every model and GPU), so any slower model always
// "escalated". Now the only trigger is the CV of the run's own latency
// samples, and the message says plainly that no trace was taken.
//
// Contract: `contracts/no-false-escalation-v1.yaml`.

/// CV (%) above which a run is flagged as unstable. Same default as renacer's
/// `BrickEscalationThresholds::cv_percent` (Curtsinger & Berger 2013).
const ESCALATION_CV_PERCENT: f64 = 15.0;

/// Why a run escalated. There is only one arm: the run's own sample CV.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Escalation {
    cv_percent: f64,
    threshold_percent: f64,
    samples: usize,
}

impl Escalation {
    /// The line cbtop prints. It never claims a trace was taken.
    fn message(&self) -> String {
        format!(
            "cbtop: escalation: not traced (CV {:.1}% > {:.1}% over {} samples; no tracer attached)",
            self.cv_percent, self.threshold_percent, self.samples
        )
    }
}

/// Decide from the run's own latency samples alone. Fewer than two samples,
/// or a non-positive / non-finite mean, is not decidable and never escalates.
#[cfg_attr(not(all(feature = "inference", feature = "cuda")), allow(dead_code))]
fn escalation_decision(latencies_us: &[f64], threshold_percent: f64) -> Option<Escalation> {
    let n = latencies_us.len();
    if n < 2 {
        return None;
    }
    let mean = latencies_us.iter().sum::<f64>() / n as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return None;
    }
    let variance = latencies_us.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let cv_percent = variance.sqrt() / mean * 100.0;
    (cv_percent.is_finite() && cv_percent > threshold_percent).then_some(Escalation {
        cv_percent,
        threshold_percent,
        samples: n,
    })
}

/// Print the escalation line, if the run's samples call for one.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn report_escalation(latencies_us: &[f64]) {
    if let Some(e) = escalation_decision(latencies_us, ESCALATION_CV_PERCENT) {
        eprintln!();
        eprintln!("{}", e.message());
        eprintln!();
    }
}

#[cfg(test)]
mod escalation_tests {
    use super::*;

    /// FALSIFY-NFE-001: a stable run never escalates, however slow it is —
    /// there is no throughput/efficiency arm any more.
    #[test]
    fn falsify_nfe_001_stable_slow_run_does_not_escalate() {
        // ~10 tok/s (100 ms/token), CV ≈ 1%: the old efficiency arm fired here.
        let samples = [100_000.0, 101_000.0, 99_000.0, 100_500.0, 99_500.0];
        assert_eq!(escalation_decision(&samples, ESCALATION_CV_PERCENT), None);
    }

    /// FALSIFY-NFE-002: an unstable run escalates on its own CV, and the
    /// message says it was not traced.
    #[test]
    fn falsify_nfe_002_unstable_run_escalates_not_traced() {
        let samples = [1_000.0, 3_000.0, 1_000.0, 3_000.0];
        let e = escalation_decision(&samples, ESCALATION_CV_PERCENT).expect("CV 50% escalates");
        assert!((e.cv_percent - 50.0).abs() < 1e-9, "cv {}", e.cv_percent);
        assert_eq!(e.samples, 4);
        let msg = e.message();
        assert!(msg.contains("escalation: not traced"), "{msg}");
        assert!(!msg.contains("Enabled"), "{msg}");
        assert!(!msg.contains("efficiency"), "{msg}");
    }

    /// FALSIFY-NFE-003: undecidable inputs never escalate and never yield NaN.
    #[test]
    fn falsify_nfe_003_degenerate_samples_do_not_escalate() {
        for s in [&[][..], &[5.0][..], &[0.0, 0.0][..], &[f64::NAN, 1.0][..]] {
            assert_eq!(escalation_decision(s, ESCALATION_CV_PERCENT), None, "{s:?}");
        }
    }

    /// The threshold is strict: CV exactly at the threshold does not escalate.
    #[test]
    fn falsify_nfe_004_threshold_is_strict() {
        let samples = [1_000.0, 3_000.0, 1_000.0, 3_000.0]; // CV = 50% exactly
        assert_eq!(escalation_decision(&samples, 50.0), None);
        assert!(escalation_decision(&samples, 49.999).is_some());
    }
}
