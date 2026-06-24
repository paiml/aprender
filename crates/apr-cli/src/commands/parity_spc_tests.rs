
// ═══════════════════════════════════════════════════════════════════════════════
// Tests for spc_color.rs functions (included into parity.rs)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod spc_tests {
    use super::*;

    // ── Helper: construct SpcMetrics with specified fields ──────────────────
    fn make_metrics(
        cosine_similarity: f32,
        kl_divergence: f64,
        max_abs_diff: f32,
        mean_abs_diff: f32,
        rmse: f32,
        sigma_level: f64,
        cpu_argmax: u32,
        gpu_argmax: u32,
        cpu_nan: usize,
        gpu_nan: usize,
        out_of_spec_count: usize,
        vocab_size: usize,
    ) -> SpcMetrics {
        SpcMetrics {
            position: 0,
            token_id: 0,
            cpu_argmax,
            gpu_argmax,
            _cpu_top_logit: 1.0,
            _gpu_top_logit: 1.0,
            max_abs_diff,
            _max_diff_idx: 0,
            mean_abs_diff,
            rmse,
            cosine_similarity,
            kl_divergence,
            sigma_level,
            cpu_nan,
            gpu_nan,
            out_of_spec_count,
            vocab_size,
        }
    }

    fn perfect_metrics() -> SpcMetrics {
        make_metrics(1.0, 0.0, 0.0, 0.0, 0.0, 99.0, 42, 42, 0, 0, 0, 32000)
    }

    // ── Verdict tests ──────────────────────────────────────────────────────

    #[test]
    fn test_verdict_pass() {
        let m = perfect_metrics();
        assert_eq!(m.verdict(), Verdict::Pass);
        assert!(m.verdict().is_pass());
        assert!(!m.verdict().is_fail());
    }

    #[test]
    fn test_verdict_fail_nan_cpu() {
        let m = make_metrics(1.0, 0.0, 0.0, 0.0, 0.0, 99.0, 42, 42, 1, 0, 0, 32000);
        assert_eq!(m.verdict(), Verdict::FailNan);
        assert!(m.verdict().is_fail());
    }

    #[test]
    fn test_verdict_fail_nan_gpu() {
        let m = make_metrics(1.0, 0.0, 0.0, 0.0, 0.0, 99.0, 42, 42, 0, 3, 0, 32000);
        assert_eq!(m.verdict(), Verdict::FailNan);
    }

    #[test]
    fn test_verdict_fail_catastrophic() {
        let m = make_metrics(0.5, 0.0, 0.0, 0.0, 0.0, 99.0, 42, 42, 0, 0, 0, 32000);
        assert_eq!(m.verdict(), Verdict::FailCatastrophic);
        assert!(m.verdict().is_fail());
    }

    #[test]
    fn test_verdict_fail_divergent() {
        // PMAT-919: cosine >= 0.9 (not catastrophic) but < COSINE_SIM_MIN (0.95).
        let m = make_metrics(0.92, 0.0, 0.0, 0.0, 0.0, 99.0, 42, 42, 0, 0, 0, 32000);
        assert_eq!(m.verdict(), Verdict::FailDivergent);
        assert!(m.verdict().is_fail());
    }

    #[test]
    fn test_verdict_correct_quant_path_is_not_fail() {
        // PMAT-919 (reconciled): a CORRECT quantized GPU Q4_K path (the production
        // fp32-Mwv default on Blackwell, or HwDp4a on discrete sm_89+) scores cosine
        // ~0.97-1.0 (NOT bit-exact) and may flip the argmax on a near-tie. With the
        // old bit-exact COSINE_SIM_MIN=0.999 this aggregate verdict was FailDivergent
        // ("different function"). Quant-aware thresholds must NOT fail it. (The
        // degraded-HwDp4a mid-context divergence is caught by the F2 per-position
        // gate's argmax assertion, not by this coarse aggregate SPC verdict.)
        let m = make_metrics(0.97, 0.0, 2.5, 0.0, 0.0, 99.0, 42, 43, 0, 0, 0, 32000);
        assert!(!m.verdict().is_fail(), "correct quantized path must not be a FAIL");
    }

    #[test]
    fn test_verdict_warn_out_of_spec() {
        // PMAT-919: cosine good, but max_abs_diff > TOLERANCE_ABS (4.0).
        let m = make_metrics(0.9999, 0.0, 5.0, 0.0, 0.0, 99.0, 42, 42, 0, 0, 0, 32000);
        assert_eq!(m.verdict(), Verdict::WarnOutOfSpec);
        assert!(!m.verdict().is_pass());
        assert!(!m.verdict().is_fail());
    }

    #[test]
    fn test_verdict_warn_argmax() {
        // cosine good, max_abs_diff within tolerance, but argmax differs
        let m = make_metrics(0.9999, 0.0, 0.5, 0.0, 0.0, 99.0, 42, 43, 0, 0, 0, 32000);
        assert_eq!(m.verdict(), Verdict::WarnArgmax);
    }

    // ── Cpk tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_cpk_perfect() {
        let m = perfect_metrics();
        assert!((m.cpk() - 99.9).abs() < 0.01);
    }

    #[test]
    fn test_cpk_normal() {
        // PMAT-919: TOLERANCE_ABS relaxed 1.0 → 4.0. rmse = 0.1, mean_abs_diff = 0.01
        // cpk = (4.0 - 0.01) / (3.0 * 0.1) = 3.99 / 0.3 = 13.3
        let m = make_metrics(0.9999, 0.0, 0.5, 0.01, 0.1, 3.0, 42, 42, 0, 0, 0, 32000);
        assert!((m.cpk() - 13.3).abs() < 0.01);
    }

    // ── color_lower_is_better / color_higher_is_better ─────────────────────

    #[test]
    fn test_color_lower_is_better_below_first() {
        let color = color_lower_is_better(0.0005, &KL_THRESHOLDS, SpcColor::RedBold);
        assert!(matches!(color, SpcColor::GreenBold));
    }

    #[test]
    fn test_color_lower_is_better_between() {
        let color = color_lower_is_better(0.005, &KL_THRESHOLDS, SpcColor::RedBold);
        assert!(matches!(color, SpcColor::Green));
    }

    #[test]
    fn test_color_lower_is_better_high() {
        let color = color_lower_is_better(0.5, &KL_THRESHOLDS, SpcColor::RedBold);
        assert!(matches!(color, SpcColor::Red));
    }

    #[test]
    fn test_color_lower_is_better_fallback() {
        let color = color_lower_is_better(999.0, &KL_THRESHOLDS, SpcColor::RedBold);
        assert!(matches!(color, SpcColor::RedBold));
    }

    #[test]
    fn test_color_higher_is_better_above_first() {
        let color = color_higher_is_better(7.0, &SIGMA_THRESHOLDS, SpcColor::RedBold);
        assert!(matches!(color, SpcColor::GreenBold));
    }

    #[test]
    fn test_color_higher_is_better_mid() {
        let color = color_higher_is_better(2.5, &SIGMA_THRESHOLDS, SpcColor::RedBold);
        assert!(matches!(color, SpcColor::Yellow));
    }

    #[test]
    fn test_color_higher_is_better_fallback() {
        let color = color_higher_is_better(0.5, &SIGMA_THRESHOLDS, SpcColor::RedBold);
        assert!(matches!(color, SpcColor::RedBold));
    }

    // ── format_kl ──────────────────────────────────────────────────────────

    #[test]
    fn test_format_kl_tiny() {
        let s = format_kl(0.00001);
        // Precision 5 for values < 0.1
        assert!(s.contains("0.00001"));
    }

    #[test]
    fn test_format_kl_medium() {
        let s = format_kl(0.5);
        // Precision 3 for values in [0.1, 1.0)
        assert!(s.contains("0.500"));
    }

    #[test]
    fn test_format_kl_large() {
        let s = format_kl(5.0);
        // Precision 2 for values >= 1.0
        assert!(s.contains("5.00"));
    }

    // ── format_sigma ───────────────────────────────────────────────────────

    #[test]
    fn test_format_sigma_high() {
        let s = format_sigma(6.5);
        assert!(s.contains("6.5"));
        assert!(s.contains("\u{03c3}")); // sigma symbol
    }

    #[test]
    fn test_format_sigma_low() {
        let s = format_sigma(0.5);
        assert!(s.contains("0.5"));
    }

    // ── format_sigma_summary ───────────────────────────────────────────────

    #[test]
    fn test_format_sigma_summary_world_class() {
        let s = format_sigma_summary(6.5);
        assert!(s.contains("world class"));
    }

    #[test]
    fn test_format_sigma_summary_three_sigma() {
        let s = format_sigma_summary(3.5);
        assert!(s.contains("99.73% yield"));
    }

    #[test]
    fn test_format_sigma_summary_out_of_control() {
        let s = format_sigma_summary(0.3);
        assert!(s.contains("out of control"));
        assert!(s.contains("VIOLATED"));
    }

    #[test]
    fn test_format_sigma_summary_below_two_violated() {
        let s = format_sigma_summary(1.5);
        assert!(s.contains("VIOLATED"));
    }

    // ── format_cpk ─────────────────────────────────────────────────────────

    #[test]
    fn test_format_cpk_world_class() {
        let s = format_cpk(2.5);
        assert!(s.contains("world class"));
    }

    #[test]
    fn test_format_cpk_capable() {
        let s = format_cpk(1.5);
        assert!(s.contains("capable"));
    }

    #[test]
    fn test_format_cpk_marginal() {
        let s = format_cpk(1.1);
        assert!(s.contains("marginal"));
    }

    #[test]
    fn test_format_cpk_incapable() {
        let s = format_cpk(0.5);
        assert!(s.contains("incapable"));
        assert!(s.contains("VIOLATED"));
    }

    // ── color_vs_threshold ─────────────────────────────────────────────────

    #[test]
    fn test_color_vs_threshold_higher_is_better_pass() {
        assert!(matches!(
            color_vs_threshold(0.999, 0.99, true),
            SpcColor::Green
        ));
    }

    #[test]
    fn test_color_vs_threshold_higher_is_better_fail() {
        assert!(matches!(
            color_vs_threshold(0.98, 0.99, true),
            SpcColor::RedBold
        ));
    }

    #[test]
    fn test_color_vs_threshold_lower_is_better_pass() {
        assert!(matches!(
            color_vs_threshold(0.001, 0.01, false),
            SpcColor::Green
        ));
    }

    #[test]
    fn test_color_vs_threshold_lower_is_better_fail() {
        assert!(matches!(
            color_vs_threshold(0.05, 0.01, false),
            SpcColor::RedBold
        ));
    }

    // ── format_metric_range ────────────────────────────────────────────────

    #[test]
    fn test_format_metric_range_pass() {
        let s = format_metric_range(0.9999, 0.999, 0.999, true);
        assert!(!s.contains("VIOLATED"));
    }

    #[test]
    fn test_format_metric_range_fail() {
        let s = format_metric_range(0.9999, 0.998, 0.999, true);
        assert!(s.contains("VIOLATED"));
    }

    // ── format_metric_single ───────────────────────────────────────────────

    #[test]
    fn test_format_metric_single_pass_lower() {
        let s = format_metric_single(0.001, 0.01, true);
        assert!(!s.contains("VIOLATED"));
    }

    #[test]
    fn test_format_metric_single_fail_lower() {
        let s = format_metric_single(0.05, 0.01, true);
        assert!(s.contains("VIOLATED"));
    }

    // ── print_summary ──────────────────────────────────────────────────────

    #[test]
    fn test_print_summary_empty() {
        // Should not panic on empty slice
        print_summary(&[]);
    }

    #[test]
    fn test_print_summary_all_pass() {
        let metrics = vec![
            perfect_metrics(),
            perfect_metrics(),
            perfect_metrics(),
        ];
        // Should print without panicking; prints "PARITY PROVEN"
        print_summary(&metrics);
    }

    #[test]
    fn test_print_summary_with_warnings() {
        let mut m = perfect_metrics();
        m.cpu_argmax = 100;
        m.gpu_argmax = 101; // WarnArgmax
        let metrics = vec![perfect_metrics(), m];
        print_summary(&metrics);
    }

    #[test]
    fn test_print_summary_with_failure() {
        let fail = make_metrics(0.5, 5.0, 10.0, 5.0, 3.0, 0.5, 42, 100, 0, 0, 1000, 32000);
        let metrics = vec![perfect_metrics(), fail];
        print_summary(&metrics);
    }

    #[test]
    fn test_print_summary_single_metric() {
        let metrics = vec![perfect_metrics()];
        print_summary(&metrics);
    }

    // ── compute_metrics ────────────────────────────────────────────────────

    #[test]
    fn test_compute_metrics_identical() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = compute_metrics(&logits, &logits, 0, 0);
        assert!((m.cosine_similarity - 1.0).abs() < 1e-5);
        assert!(m.kl_divergence < 1e-10);
        assert_eq!(m.max_abs_diff, 0.0);
        assert_eq!(m.cpu_argmax, 4);
        assert_eq!(m.gpu_argmax, 4);
        assert_eq!(m.out_of_spec_count, 0);
        assert_eq!(m.verdict(), Verdict::Pass);
    }

    #[test]
    fn test_compute_metrics_small_diff() {
        let cpu = vec![1.0, 2.0, 3.0];
        let gpu = vec![1.001, 2.001, 3.001];
        let m = compute_metrics(&cpu, &gpu, 0, 0);
        assert!(m.cosine_similarity > 0.9999);
        assert!(m.max_abs_diff < 0.01);
        assert_eq!(m.cpu_argmax, 2);
        assert_eq!(m.gpu_argmax, 2);
    }

    #[test]
    fn test_compute_metrics_divergent() {
        let cpu = vec![1.0, 0.0, 0.0];
        let gpu = vec![0.0, 0.0, 1.0];
        let m = compute_metrics(&cpu, &gpu, 0, 0);
        assert!(m.cosine_similarity < 0.5);
        assert!(m.verdict().is_fail());
    }

    // ── cosine_sim ─────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_opposite() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        assert!((cosine_sim(&a, &b) + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_sim(&a, &b), 0.0);
    }

    // ── kl_div_softmax ─────────────────────────────────────────────────────

    #[test]
    fn test_kl_div_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let kl = kl_div_softmax(&v, &v);
        assert!(kl < 1e-10);
    }

    #[test]
    fn test_kl_div_shifted() {
        // Shifting all logits by a constant doesn't change softmax
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![101.0, 102.0, 103.0];
        let kl = kl_div_softmax(&a, &b);
        assert!(kl < 1e-10);
    }

    #[test]
    fn test_kl_div_different() {
        let a = vec![10.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 10.0];
        let kl = kl_div_softmax(&a, &b);
        assert!(kl > 1.0); // Very different distributions
    }

    // ── argmax ─────────────────────────────────────────────────────────────

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn test_argmax_first_element() {
        assert_eq!(argmax(&[10.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn test_argmax_empty() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn test_argmax_single() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    #[test]
    fn test_argmax_negative() {
        assert_eq!(argmax(&[-5.0, -1.0, -3.0]), 1);
    }
}
