
#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PARITY-090: TUI Configuration Tests
    // =========================================================================

    /// PARITY-090a: Test TuiConfig defaults
    #[test]
    fn test_parity_090a_tui_config_defaults() {
        println!("PARITY-090a: TuiConfig Default Values");

        let config = TuiConfig::default();

        println!("  refresh_rate_ms: {}", config.refresh_rate_ms);
        println!("  m4_target_tok_per_sec: {}", config.m4_target_tok_per_sec);
        println!("  width: {}", config.width);

        assert_eq!(config.refresh_rate_ms, 100);
        assert_eq!(config.m4_target_tok_per_sec, 192.0);
        assert!(config.show_throughput_sparkline);
        assert!(config.show_latency_sparkline);
    }

    /// PARITY-090b: Test InferenceMetrics creation
    #[test]
    fn test_parity_090b_inference_metrics() {
        println!("PARITY-090b: InferenceMetrics");

        let metrics = InferenceMetrics {
            throughput_tok_per_sec: 64.0,
            latency_ms: 15.6,
            latency_p95_ms: 23.4,
            gpu_memory_bytes: 4_200_000_000,
            gpu_memory_total_bytes: 24_000_000_000,
            batch_size: 4,
            queue_size: 12,
            total_tokens: 1234,
            total_requests: 42,
            running: true,
            using_gpu: true,
        };

        println!("  throughput: {:.1} tok/s", metrics.throughput_tok_per_sec);
        println!("  achieves_m4: {}", metrics.achieves_m4_parity());
        println!("  gap_to_m4: {:.2}x", metrics.gap_to_m4());
        println!("  gpu_memory: {}", metrics.format_gpu_memory());

        assert!(!metrics.achieves_m4_parity());
        assert!((metrics.gap_to_m4() - 3.0).abs() < 0.1);
        assert!(metrics.format_gpu_memory().contains("4.2 GB"));
    }

    /// PARITY-090c: Test M4 parity detection
    #[test]
    fn test_parity_090c_m4_parity_detection() {
        println!("PARITY-090c: M4 Parity Detection");

        let test_cases = [
            (64.0, false, "Baseline - not M4"),
            (150.0, false, "Batch threshold - not M4"),
            (192.0, true, "Exactly M4"),
            (256.0, true, "Above M4"),
        ];

        for (throughput, expected, description) in test_cases {
            let metrics = InferenceMetrics {
                throughput_tok_per_sec: throughput,
                ..Default::default()
            };
            let achieves = metrics.achieves_m4_parity();
            println!("  {}: {} tok/s → M4={}", description, throughput, achieves);
            assert_eq!(achieves, expected, "{}", description);
        }
    }

    // =========================================================================
    // PARITY-091: TUI Rendering Tests
    // =========================================================================

    /// PARITY-091a: Test TUI creation and update
    #[test]
    fn test_parity_091a_tui_creation_update() {
        println!("PARITY-091a: TUI Creation and Update");

        let config = TuiConfig::default();
        let mut tui = InferenceTui::new(config);

        let metrics = InferenceMetrics {
            throughput_tok_per_sec: 64.0,
            latency_ms: 15.6,
            running: true,
            ..Default::default()
        };

        tui.update(&metrics);

        assert_eq!(tui.metrics().throughput_tok_per_sec, 64.0);
        assert_eq!(tui.throughput_history().len(), 1);
    }

    /// PARITY-091b: Test sparkline generation
    #[test]
    fn test_parity_091b_sparkline_generation() {
        println!("PARITY-091b: Sparkline Generation");

        let mut history = VecDeque::new();
        for i in 0..20 {
            history.push_back((i as f64) * 10.0);
        }

        let sparkline = InferenceTui::sparkline(&history, 20);
        println!("  Sparkline: {}", sparkline);

        assert_eq!(sparkline.chars().count(), 20);
        assert!(sparkline.contains('▁')); // Low values
        assert!(sparkline.contains('█')); // High values
    }

    /// PARITY-091c: Test TUI render output structure
    #[test]
    fn test_parity_091c_tui_render_structure() {
        println!("PARITY-091c: TUI Render Output Structure");

        let config = TuiConfig::default();
        let mut tui = InferenceTui::new(config);

        // Add some history
        for i in 0..10 {
            let metrics = InferenceMetrics {
                throughput_tok_per_sec: 50.0 + (i as f64) * 5.0,
                latency_ms: 20.0 - (i as f64),
                batch_size: 4,
                queue_size: 12,
                total_tokens: 1234,
                total_requests: 42,
                running: true,
                using_gpu: true,
                ..Default::default()
            };
            tui.update(&metrics);
        }

        let output = tui.render_to_string();
        println!("{}", output);

        // Verify structure
        assert!(output.contains("╭"), "Should have top border");
        assert!(output.contains("╰"), "Should have bottom border");
        assert!(
            output.contains("realizar Inference Monitor"),
            "Should have title"
        );
        assert!(output.contains("Throughput:"), "Should show throughput");
        assert!(output.contains("Latency:"), "Should show latency");
        assert!(output.contains("tok/s"), "Should show tok/s unit");
        assert!(output.contains("● Running"), "Should show running status");
    }

    /// PARITY-091d: Test TUI visual regression baseline
    #[test]
    fn test_parity_091d_visual_regression_baseline() {
        println!("PARITY-091d: Visual Regression Baseline");

        let config = TuiConfig {
            width: 65,
            ..Default::default()
        };
        let mut tui = InferenceTui::new(config);

        let metrics = InferenceMetrics {
            throughput_tok_per_sec: 64.2,
            latency_ms: 15.6,
            latency_p95_ms: 23.4,
            gpu_memory_bytes: 4_200_000_000,
            gpu_memory_total_bytes: 24_000_000_000,
            batch_size: 4,
            queue_size: 12,
            total_tokens: 1234,
            total_requests: 42,
            running: true,
            using_gpu: true,
        };

        tui.update(&metrics);
        let output = tui.render_to_string();

        println!("=== GOLDEN BASELINE ===");
        println!("{}", output);
        println!("=== END BASELINE ===");

        // Verify key visual elements
        let lines: Vec<&str> = output.lines().collect();
        assert!(lines.len() >= 10, "Should have at least 10 lines");

        // Check border consistency
        assert!(lines[0].starts_with('╭'));
        assert!(lines[0].ends_with('╮'));
        assert!(lines.last().expect("test").starts_with('╰'));
        assert!(lines.last().expect("test").ends_with('╯'));

        // Check content
        assert!(
            output.contains("64.2 tok/s"),
            "Should show throughput value"
        );
        assert!(output.contains("15.6 ms/tok"), "Should show latency value");
        assert!(output.contains("1234"), "Should show token count");
    }

    /// PARITY-091e: Test history accumulation
    #[test]
    fn test_parity_091e_history_accumulation() {
        println!("PARITY-091e: History Accumulation");

        let config = TuiConfig::default();
        let mut tui = InferenceTui::new(config);

        // Add more than max_history items
        for i in 0..50 {
            let metrics = InferenceMetrics {
                throughput_tok_per_sec: (i as f64) * 2.0,
                latency_ms: 100.0 - (i as f64),
                ..Default::default()
            };
            tui.update(&metrics);
        }

        // Should be capped at max_history (40)
        assert_eq!(tui.throughput_history().len(), 40);
        assert_eq!(tui.latency_history().len(), 40);

        // Most recent should be last
        assert!((tui.throughput_history().back().expect("test") - 98.0).abs() < 0.1);
    }

    /// PARITY-091f: Test empty sparkline handling
    #[test]
    fn test_parity_091f_empty_sparkline() {
        println!("PARITY-091f: Empty Sparkline Handling");

        let empty: VecDeque<f64> = VecDeque::new();
        let sparkline = InferenceTui::sparkline(&empty, 20);

        println!("  Empty sparkline: '{}'", sparkline);
        assert_eq!(sparkline.len(), 20);
        assert!(sparkline.chars().all(|c| c == ' '));
    }
}
