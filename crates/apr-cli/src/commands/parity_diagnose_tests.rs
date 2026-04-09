
// ════════════════════════════════════════════════════════════════════
// Coverage tests for auto_diagnose (PMAT coverage gap)
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod diagnose_tests {
    use super::*;

    // Reuse the make_metrics helper from parity_spc_tests
    fn diagnose_metrics(
        cosine_similarity: f32,
        max_abs_diff: f32,
        cpu_argmax: u32,
        gpu_argmax: u32,
        cpu_nan: usize,
        gpu_nan: usize,
        position: usize,
    ) -> SpcMetrics {
        SpcMetrics {
            position,
            token_id: position as u32,
            cpu_argmax,
            gpu_argmax,
            _cpu_top_logit: 1.0,
            _gpu_top_logit: 1.0,
            max_abs_diff,
            _max_diff_idx: 0,
            mean_abs_diff: max_abs_diff * 0.5,
            rmse: max_abs_diff * 0.3,
            cosine_similarity,
            kl_divergence: if cosine_similarity > 0.999 { 0.001 } else { 5.0 },
            sigma_level: if cosine_similarity > 0.999 { 6.0 } else { 0.5 },
            cpu_nan,
            gpu_nan,
            out_of_spec_count: 0,
            vocab_size: 32000,
        }
    }

    // ── auto_diagnose: no failures ─────────────────────────────────────

    #[test]
    fn test_auto_diagnose_no_failures() {
        // All passing — should return early without diagnosis
        let metrics = vec![
            diagnose_metrics(1.0, 0.0, 42, 42, 0, 0, 0),
            diagnose_metrics(1.0, 0.0, 42, 42, 0, 0, 1),
            diagnose_metrics(1.0, 0.0, 42, 42, 0, 0, 2),
        ];
        // Should not panic, returns early
        auto_diagnose(&metrics, 3584, 28, 4);
    }

    #[test]
    fn test_auto_diagnose_empty_metrics() {
        auto_diagnose(&[], 3584, 28, 4);
    }

    // ── Pattern 1: Position 0 fails (core layer bug) ───────────────────

    #[test]
    fn test_auto_diagnose_pos0_catastrophic() {
        let metrics = vec![
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 0), // catastrophic at pos 0
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 1),
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 2),
        ];
        // Should output WHY 1, WHY 2 (catastrophic), WHY 3 (all fail + pos0)
        // + likely root causes + falsification tests
        auto_diagnose(&metrics, 3584, 28, 4);
    }

    // ── Pattern 2: Growing divergence (KV cache accumulation) ──────────

    #[test]
    fn test_auto_diagnose_growing_divergence() {
        let metrics = vec![
            diagnose_metrics(0.9999, 0.1, 42, 42, 0, 0, 0),  // pass
            diagnose_metrics(0.998, 0.5, 42, 42, 0, 0, 1),    // divergent
            diagnose_metrics(0.995, 1.5, 42, 43, 0, 0, 2),    // divergent, growing
            diagnose_metrics(0.990, 3.0, 42, 50, 0, 0, 3),    // more divergent
            diagnose_metrics(0.980, 6.0, 42, 100, 0, 0, 4),   // much more
            diagnose_metrics(0.950, 12.0, 42, 200, 0, 0, 5),  // catastrophic
        ];
        // Should detect growing pattern
        auto_diagnose(&metrics, 4096, 32, 8);
    }

    // ── Pattern 3: High cosine but wrong argmax ────────────────────────

    #[test]
    fn test_auto_diagnose_high_cos_wrong_argmax() {
        let metrics = vec![
            diagnose_metrics(0.9999, 0.1, 42, 42, 0, 0, 0),  // pass
            diagnose_metrics(0.995, 0.5, 42, 43, 0, 0, 1),    // divergent, wrong argmax
        ];
        // Should trigger Pattern 3 (high cosine wrong argmax)
        auto_diagnose(&metrics, 2048, 16, 4);
    }

    // ── Pattern 5: All positions fail uniformly ────────────────────────

    #[test]
    fn test_auto_diagnose_all_fail_uniformly() {
        let metrics = vec![
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 0),
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 1),
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 2),
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 3),
        ];
        // Pattern 5: all_fail + pos0_fails
        auto_diagnose(&metrics, 1536, 12, 4);
    }

    // ── NaN failures ───────────────────────────────────────────────────

    #[test]
    fn test_auto_diagnose_nan_failures() {
        let metrics = vec![
            diagnose_metrics(1.0, 0.0, 42, 42, 3, 0, 0), // NaN on CPU
            diagnose_metrics(1.0, 0.0, 42, 42, 0, 5, 1), // NaN on GPU
        ];
        auto_diagnose(&metrics, 3584, 28, 4);
    }

    // ── Edge case: zero kv_heads ───────────────────────────────────────

    #[test]
    fn test_auto_diagnose_zero_kv_heads() {
        let metrics = vec![
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 0),
        ];
        // Should not divide by zero
        auto_diagnose(&metrics, 3584, 28, 0);
    }

    // ── Edge case: zero num_heads ──────────────────────────────────────

    #[test]
    fn test_auto_diagnose_zero_num_heads() {
        let metrics = vec![
            diagnose_metrics(0.5, 10.0, 42, 100, 0, 0, 0),
        ];
        // Should not divide by zero
        auto_diagnose(&metrics, 3584, 0, 0);
    }

    // ── Mixed pass/fail ────────────────────────────────────────────────

    #[test]
    fn test_auto_diagnose_mixed_pass_fail() {
        let metrics = vec![
            diagnose_metrics(1.0, 0.0, 42, 42, 0, 0, 0),   // pass
            diagnose_metrics(1.0, 0.0, 42, 42, 0, 0, 1),   // pass
            diagnose_metrics(0.995, 0.5, 42, 43, 0, 0, 2),  // divergent
            diagnose_metrics(1.0, 0.0, 42, 42, 0, 0, 3),   // pass
        ];
        // Some fail, pos0 passes — should NOT trigger pattern 1 or 5
        auto_diagnose(&metrics, 4096, 32, 8);
    }

    // ── Verdict helper coverage ────────────────────────────────────────

    #[test]
    fn test_verdict_symbol_all_variants() {
        // Exercise all verdict symbol paths
        let _pass = Verdict::Pass.symbol();
        let _warn_argmax = Verdict::WarnArgmax.symbol();
        let _warn_oos = Verdict::WarnOutOfSpec.symbol();
        let _fail_div = Verdict::FailDivergent.symbol();
        let _fail_cat = Verdict::FailCatastrophic.symbol();
        let _fail_nan = Verdict::FailNan.symbol();
    }

    #[test]
    fn test_format_diff_ranges() {
        // Exercise all format_diff color paths
        let _tiny = format_diff(0.05);
        let _medium = format_diff(0.5);
        let _high = format_diff(2.5);
        let _very_high = format_diff(10.0);
    }

    #[test]
    fn test_format_cosine_ranges() {
        // Exercise all format_cosine color paths
        let _perfect = format_cosine(0.99999);
        let _good = format_cosine(0.9995);
        let _warning = format_cosine(0.995);
        let _bad = format_cosine(0.95);
        let _catastrophic = format_cosine(0.5);
    }

    // ── print_header / print_footer / print_row ────────────────────────

    #[test]
    fn test_print_header_no_panic() {
        print_header();
    }

    #[test]
    fn test_print_footer_no_panic() {
        print_footer();
    }

    #[test]
    fn test_print_row_pass() {
        let m = SpcMetrics {
            position: 0,
            token_id: 42,
            cpu_argmax: 100,
            gpu_argmax: 100,
            _cpu_top_logit: 5.0,
            _gpu_top_logit: 5.0,
            max_abs_diff: 0.001,
            _max_diff_idx: 0,
            mean_abs_diff: 0.0005,
            rmse: 0.0003,
            cosine_similarity: 0.99999,
            kl_divergence: 0.0001,
            sigma_level: 6.5,
            cpu_nan: 0,
            gpu_nan: 0,
            out_of_spec_count: 0,
            vocab_size: 32000,
        };
        print_row(&m);
    }

    #[test]
    fn test_print_row_fail() {
        let m = SpcMetrics {
            position: 5,
            token_id: 999,
            cpu_argmax: 100,
            gpu_argmax: 200,
            _cpu_top_logit: 5.0,
            _gpu_top_logit: -3.0,
            max_abs_diff: 10.0,
            _max_diff_idx: 42,
            mean_abs_diff: 5.0,
            rmse: 3.0,
            cosine_similarity: 0.5,
            kl_divergence: 5.0,
            sigma_level: 0.5,
            cpu_nan: 0,
            gpu_nan: 0,
            out_of_spec_count: 1000,
            vocab_size: 32000,
        };
        print_row(&m);
    }

    #[test]
    fn test_print_row_nan_verdict() {
        let m = SpcMetrics {
            position: 0,
            token_id: 1,
            cpu_argmax: 0,
            gpu_argmax: 0,
            _cpu_top_logit: 0.0,
            _gpu_top_logit: 0.0,
            max_abs_diff: 0.0,
            _max_diff_idx: 0,
            mean_abs_diff: 0.0,
            rmse: 0.0,
            cosine_similarity: 1.0,
            kl_divergence: 0.0,
            sigma_level: 99.0,
            cpu_nan: 5,
            gpu_nan: 0,
            out_of_spec_count: 0,
            vocab_size: 32000,
        };
        print_row(&m);
    }
}
