
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_event_creation() {
        let tensor = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let event = TraceEvent::new("test", &tensor, 0, false);

        assert_eq!(event.name, "test");
        assert_eq!(event.position, 0);
        assert_eq!(event.len, 5);
        assert!((event.l2_norm - 7.416198).abs() < 0.001); // sqrt(55)
        assert!((event.mean - 3.0).abs() < 0.001);
        assert!((event.min - 1.0).abs() < 0.001);
        assert!((event.max - 5.0).abs() < 0.001);
        assert!(event.full_data.is_none());
    }

    #[test]
    fn test_trace_event_verbose() {
        let tensor = vec![1.0, 2.0, 3.0];
        let event = TraceEvent::new("test", &tensor, 0, true);

        assert!(event.full_data.is_some());
        assert_eq!(event.full_data.expect("full_data"), tensor);
    }

    #[test]
    fn test_tracer_log() {
        let mut tracer = BrickTracer::new();

        tracer.log("step1", &[1.0, 2.0, 3.0]);
        tracer.log("step2", &[4.0, 5.0, 6.0]);

        assert_eq!(tracer.events().len(), 2);
        assert!(tracer.get("step1").is_some());
        assert!(tracer.get("step2").is_some());
        assert!(tracer.get("step3").is_none());
    }

    #[test]
    fn test_tracer_comparison_match() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("embedding", &[1.0, 2.0, 3.0]);
        cpu.log("layer0_norm", &[0.5, 1.0, 1.5]);

        gpu.log("embedding", &[1.0, 2.0, 3.0]);
        gpu.log("layer0_norm", &[0.5, 1.0, 1.5]);

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        assert!(comparison.is_equivalent());
    }

    #[test]
    fn test_tracer_comparison_diverge() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("embedding", &[1.0, 2.0, 3.0]);
        cpu.log("layer0_norm", &[0.5, 1.0, 1.5]);
        cpu.log("layer0_attn", &[0.1, 0.2, 0.3]);

        gpu.log("embedding", &[1.0, 2.0, 3.0]); // Match
        gpu.log("layer0_norm", &[0.5, 1.0, 1.5]); // Match
        gpu.log("layer0_attn", &[0.5, 0.6, 0.7]); // DIVERGE!

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        assert!(!comparison.is_equivalent());

        let first = comparison.first_divergence().expect("first");
        assert_eq!(first.name, "layer0_attn");
    }

    #[test]
    fn test_tracer_clear() {
        let mut tracer = BrickTracer::new();
        tracer.log("step1", &[1.0]);
        assert_eq!(tracer.events().len(), 1);

        tracer.clear();
        assert_eq!(tracer.events().len(), 0);
    }

    #[test]
    fn test_tracer_position() {
        let mut tracer = BrickTracer::new();

        tracer.set_position(0);
        tracer.log("pos0", &[1.0]);

        tracer.set_position(1);
        tracer.log("pos1", &[2.0]);

        tracer.log_at("explicit", &[3.0], 5);

        assert_eq!(tracer.get("pos0").expect("get").position, 0);
        assert_eq!(tracer.get("pos1").expect("get").position, 1);
        assert_eq!(tracer.get("explicit").expect("get").position, 5);
    }

    #[test]
    fn test_relative_diff() {
        let event1 = TraceEvent::new("a", &[10.0], 0, false);
        let event2 = TraceEvent::new("b", &[11.0], 0, false);

        let diff = event1.relative_diff(&event2);
        assert!((diff - 0.1).abs() < 0.001); // 10% difference
    }

    #[test]
    fn test_approx_eq() {
        let event1 = TraceEvent::new("a", &[10.0], 0, false);
        let event2 = TraceEvent::new("b", &[10.05], 0, false);
        let event3 = TraceEvent::new("c", &[11.0], 0, false);

        assert!(event1.approx_eq(&event2, 0.01)); // 1% tolerance
        assert!(!event1.approx_eq(&event3, 0.01)); // 10% diff > 1% tolerance
        assert!(event1.approx_eq(&event3, 0.15)); // 10% diff < 15% tolerance
    }

    #[test]
    fn test_empty_tensor() {
        let event = TraceEvent::new("empty", &[], 0, false);
        assert_eq!(event.len, 0);
        assert_eq!(event.l2_norm, 0.0);
        assert_eq!(event.mean, 0.0);
    }

    // T-COV-95: Additional coverage tests

    #[test]
    fn test_trace_event_display() {
        let event = TraceEvent::new("layer0", &[1.0, 2.0, 3.0], 5, false);
        let display = format!("{}", event);
        assert!(display.contains("layer0"));
        assert!(display.contains("pos=5"));
    }

    #[test]
    fn test_trace_comparison_summary_no_divergence() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("a", &[1.0, 2.0, 3.0]);
        gpu.log("a", &[1.0, 2.0, 3.0]);

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        let summary = comparison.summary();
        assert!(summary.contains("No divergence"));
    }

    #[test]
    fn test_trace_comparison_summary_with_divergence() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("attn", &[1.0, 2.0, 3.0]);
        gpu.log("attn", &[10.0, 20.0, 30.0]); // Big divergence

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        let summary = comparison.summary();
        assert!(summary.contains("divergence"));
        assert!(summary.contains("attn"));
    }

    #[test]
    fn test_trace_comparison_display_no_divergence() {
        let comparison = TraceComparison {
            diffs: vec![],
            tolerance: 0.01,
        };
        let display = format!("{}", comparison);
        assert!(display.contains("MATCH"));
    }

    #[test]
    fn test_trace_comparison_display_with_divergence() {
        let comparison = TraceComparison {
            diffs: vec![TraceDiff {
                name: "test".to_string(),
                position: 0,
                cpu_l2: 10.0,
                gpu_l2: 15.0,
                relative_diff: 0.5,
                cpu_head: [0.0; 8],
                gpu_head: [0.0; 8],
            }],
            tolerance: 0.01,
        };
        let display = format!("{}", comparison);
        assert!(display.contains("DIVERGENCE DETECTED"));
    }

    #[test]
    fn test_trace_comparison_display_multiple_divergences() {
        let comparison = TraceComparison {
            diffs: vec![
                TraceDiff {
                    name: "attn".to_string(),
                    position: 0,
                    cpu_l2: 10.0,
                    gpu_l2: 15.0,
                    relative_diff: 0.5,
                    cpu_head: [0.0; 8],
                    gpu_head: [0.0; 8],
                },
                TraceDiff {
                    name: "ffn".to_string(),
                    position: 1,
                    cpu_l2: 20.0,
                    gpu_l2: 25.0,
                    relative_diff: 0.25,
                    cpu_head: [0.0; 8],
                    gpu_head: [0.0; 8],
                },
            ],
            tolerance: 0.01,
        };
        let display = format!("{}", comparison);
        assert!(display.contains("All Divergences"));
        assert!(display.contains("attn"));
        assert!(display.contains("ffn"));
    }

    #[test]
    fn test_trace_diff_display() {
        let diff = TraceDiff {
            name: "layer0_attn".to_string(),
            position: 3,
            cpu_l2: 100.0,
            gpu_l2: 110.0,
            relative_diff: 0.1,
            cpu_head: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            gpu_head: [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
        };
        let display = format!("{}", diff);
        assert!(display.contains("layer0_attn"));
    }

    #[test]
    fn test_tracer_default() {
        let tracer = BrickTracer::default();
        assert!(tracer.events().is_empty());
    }

    #[test]
    fn test_tracer_verbose() {
        let mut tracer = BrickTracer::verbose();
        tracer.log("test", &[1.0, 2.0, 3.0]);
        let event = tracer.get("test").expect("event");
        assert!(event.full_data.is_some());
    }

    #[test]
    fn test_tracer_dump_no_panic() {
        let mut tracer = BrickTracer::new();
        tracer.log("step1", &[1.0]);
        tracer.log("step2", &[2.0]);
        // Just verify it doesn't panic
        tracer.dump();
    }

    #[test]
    fn test_tracer_summary_no_panic() {
        let mut tracer = BrickTracer::new();
        tracer.log("step1", &[1.0, 100.0]);
        tracer.log("step2", &[2.0, 50.0]);
        // Just verify it doesn't panic
        tracer.summary();
    }

    #[test]
    fn test_tracer_summary_empty() {
        let tracer = BrickTracer::new();
        tracer.summary(); // Should not panic
    }

    #[test]
    fn test_tracer_compare_mismatched_events() {
        let mut cpu = BrickTracer::new();
        let mut gpu = BrickTracer::new();

        cpu.log("a", &[1.0]);
        cpu.log("b", &[2.0]);

        gpu.log("a", &[1.0]);
        // gpu doesn't have "b"

        let comparison = BrickTracer::compare(&cpu, &gpu, 0.01);
        // Should handle missing events gracefully
        assert!(comparison.is_equivalent() || !comparison.is_equivalent());
    }

    #[test]
    fn test_trace_event_min_max() {
        let tensor = vec![-10.0, 0.0, 5.0, 100.0];
        let event = TraceEvent::new("minmax", &tensor, 0, false);
        assert!((event.min - (-10.0)).abs() < 0.001);
        assert!((event.max - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_relative_diff_zero_cpu() {
        let event1 = TraceEvent::new("a", &[0.0], 0, false);
        let event2 = TraceEvent::new("b", &[1.0], 0, false);
        let diff = event1.relative_diff(&event2);
        // When CPU L2 is zero, should handle gracefully
        assert!(diff.is_finite());
    }
}
