
    #[test]
    fn test_filter_mlp_up_proj_keyword() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![
                Hotspot {
                    name: "up_proj".to_string(),
                    time_us: 300.0,
                    percent: 30.0,
                    count: 1,
                    avg_us: 300.0,
                    min_us: 300.0,
                    max_us: 300.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "down_proj".to_string(),
                    time_us: 300.0,
                    percent: 30.0,
                    count: 1,
                    avg_us: 300.0,
                    min_us: 300.0,
                    max_us: 300.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let filtered = filter_results_by_focus(&results, ProfileFocus::Mlp);
        assert_eq!(filtered.hotspots.len(), 2);
    }

    #[test]
    fn test_filter_matmul_mm_keyword() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "mm_q4k".to_string(),
                time_us: 500.0,
                percent: 50.0,
                count: 1,
                avg_us: 500.0,
                min_us: 500.0,
                max_us: 500.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let filtered = filter_results_by_focus(&results, ProfileFocus::Matmul);
        assert_eq!(filtered.hotspots.len(), 1);
    }

    #[test]
    fn test_filter_embedding_vocab_keyword() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![Hotspot {
                name: "vocab_lookup".to_string(),
                time_us: 200.0,
                percent: 20.0,
                count: 1,
                avg_us: 200.0,
                min_us: 200.0,
                max_us: 200.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            }],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };
        let filtered = filter_results_by_focus(&results, ProfileFocus::Embedding);
        assert_eq!(filtered.hotspots.len(), 1);
    }

    // ========================================================================
    // Case-insensitive hotspot filtering
    // ========================================================================

    #[test]
    fn test_filter_case_insensitive() {
        let results = RealProfileResults {
            model_path: "test.gguf".to_string(),
            architecture: "llama".to_string(),
            num_layers: 4,
            vocab_size: 1000,
            hidden_dim: 256,
            warmup_passes: 1,
            measure_passes: 1,
            total_inference_us: 1000.0,
            throughput_tok_s: 1000.0,
            tokens_per_pass: 1,
            hotspots: vec![
                Hotspot {
                    name: "ATTENTION_QKV".to_string(),
                    time_us: 500.0,
                    percent: 50.0,
                    count: 1,
                    avg_us: 500.0,
                    min_us: 500.0,
                    max_us: 500.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "MATMUL_F16".to_string(),
                    time_us: 300.0,
                    percent: 30.0,
                    count: 1,
                    avg_us: 300.0,
                    min_us: 300.0,
                    max_us: 300.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
                Hotspot {
                    name: "MLP_Gate".to_string(),
                    time_us: 200.0,
                    percent: 20.0,
                    count: 1,
                    avg_us: 200.0,
                    min_us: 200.0,
                    max_us: 200.0,
                    bottleneck: None,
                    efficiency_pct: None,
                    category: None,
                    bandwidth_gbs: None,
                    data_bytes_per_call: None,
                },
            ],
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        };

        // Attention filter should match ATTENTION_QKV (case-insensitive)
        let attn_filtered = filter_results_by_focus(&results, ProfileFocus::Attention);
        assert_eq!(attn_filtered.hotspots.len(), 1);
        assert_eq!(attn_filtered.hotspots[0].name, "ATTENTION_QKV");

        // Matmul filter should match MATMUL_F16
        let mm_filtered = filter_results_by_focus(&results, ProfileFocus::Matmul);
        assert_eq!(mm_filtered.hotspots.len(), 1);
        assert_eq!(mm_filtered.hotspots[0].name, "MATMUL_F16");

        // MLP filter should match MLP_Gate (contains "gate")
        let mlp_filtered = filter_results_by_focus(&results, ProfileFocus::Mlp);
        assert_eq!(mlp_filtered.hotspots.len(), 1);
        assert_eq!(mlp_filtered.hotspots[0].name, "MLP_Gate");
    }

    // ================================================================
    // F-PROFILE-011: Cross-format comparison tests
    // ================================================================

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(detect_format(Path::new("model.gguf")), "gguf");
    }

    #[test]
    fn test_detect_format_apr() {
        assert_eq!(detect_format(Path::new("model.apr")), "apr");
    }

    #[test]
    fn test_detect_format_safetensors() {
        assert_eq!(
            detect_format(Path::new("weights.safetensors")),
            "safetensors"
        );
    }

    #[test]
    fn test_detect_format_bin_and_txt() {
        assert_eq!(detect_format(Path::new("data.bin")), "pytorch");
        assert_eq!(detect_format(Path::new("data.txt")), "unknown");
    }

    #[test]
    fn test_print_comparison_row_does_not_panic() {
        // Smoke test: ensure formatting works for various values
        print_comparison_row("Test metric", 100.5, 200.3);
        print_comparison_row("Zero case", 0.0, 100.0);
        print_comparison_row("Both zero", 0.0, 0.0);
    }

    #[test]
    fn test_cross_format_comparison_nonexistent_files() {
        let result = run_cross_format_comparison(
            Path::new("/tmp/nonexistent_a.gguf"),
            Path::new("/tmp/nonexistent_b.apr"),
            1,
            1,
            8,
            true,
        );
        assert!(result.is_err(), "Should fail with nonexistent files");
    }

    // ========================================================================
    // F-PROFILE-007/008/009: Per-Kernel Profiling Tests
    // ========================================================================

    #[test]
    fn test_estimate_kernel_data_bytes_q_proj() {
        let bytes = estimate_kernel_data_bytes("q_proj", 4096, 151936);
        assert!(bytes.is_some());
        let b = bytes.expect("q_proj should have data estimate");
        // Q4K weight: 4096*4096 * 0.5625 ≈ 9.4M, plus activation RW
        assert!(b > 9_000_000, "Q_proj should move >9MB: got {b}");
        assert!(b < 20_000_000, "Q_proj should move <20MB: got {b}");
    }

    #[test]
    fn test_estimate_kernel_data_bytes_gate_proj() {
        let bytes = estimate_kernel_data_bytes("gate_proj", 4096, 151936);
        assert!(bytes.is_some());
        let b = bytes.expect("gate_proj should have data estimate");
        // FFN gate: 4096 * (4096*4) * 0.5625 ≈ 37.7M, plus activation RW
        assert!(b > 30_000_000, "gate_proj should move >30MB: got {b}");
    }

    #[test]
    fn test_estimate_kernel_data_bytes_lm_head() {
        let bytes = estimate_kernel_data_bytes("lm_head", 4096, 151936);
        assert!(bytes.is_some());
        let b = bytes.expect("lm_head should have data estimate");
        // LM head: 4096*151936 * 0.5625 ≈ 350M
        assert!(b > 300_000_000, "lm_head should move >300MB: got {b}");
    }

    #[test]
    fn test_estimate_kernel_data_bytes_rmsnorm() {
        let bytes = estimate_kernel_data_bytes("rmsnorm", 4096, 151936);
        assert!(bytes.is_some());
        let b = bytes.expect("rmsnorm should have data estimate");
        // Norm: activation RW (4096*8) + weight (4096*4) ≈ 48KB
        assert!(b > 40_000, "rmsnorm should move >40KB: got {b}");
        assert!(b < 200_000, "rmsnorm should move <200KB: got {b}");
    }

    #[test]
    fn test_estimate_kernel_data_bytes_unknown() {
        let bytes = estimate_kernel_data_bytes("random_op_xyz", 4096, 151936);
        assert!(bytes.is_none(), "Unknown ops should return None");
    }

    #[test]
    fn test_compute_kernel_launch_overhead_basic() {
        let hotspots = vec![
            Hotspot {
                name: "q_proj".to_string(),
                time_us: 500.0,
                percent: 50.0,
                count: 10,
                avg_us: 50.0,
                min_us: 45.0,
                max_us: 55.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            },
            Hotspot {
                name: "k_proj".to_string(),
                time_us: 300.0,
                percent: 30.0,
                count: 10,
                avg_us: 30.0,
                min_us: 25.0,
                max_us: 35.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            },
        ];
        // Total kernel time = 800µs, total decode = 1000µs → 200µs overhead = 20%
        let (overhead_us, overhead_pct) = compute_kernel_launch_overhead(&hotspots, 1000.0);
        assert!(
            (overhead_us - 200.0).abs() < 1.0,
            "Expected ~200µs overhead, got {overhead_us}"
        );
        assert!(
            (overhead_pct - 20.0).abs() < 1.0,
            "Expected ~20% overhead, got {overhead_pct}"
        );
    }

    #[test]
    fn test_compute_kernel_launch_overhead_zero_decode() {
        let hotspots = vec![];
        let (overhead_us, overhead_pct) = compute_kernel_launch_overhead(&hotspots, 0.0);
        assert!(
            overhead_us.abs() < 0.001,
            "Zero decode should yield zero overhead"
        );
        assert!(
            overhead_pct.abs() < 0.001,
            "Zero decode should yield zero percent"
        );
    }

    #[test]
    fn test_hotspot_bandwidth_fields() {
        let h = Hotspot {
            name: "test".to_string(),
            time_us: 100.0,
            percent: 50.0,
            count: 5,
            avg_us: 20.0,
            min_us: 15.0,
            max_us: 25.0,
            bottleneck: None,
            efficiency_pct: Some(45.0),
            category: Some("FFN".to_string()),
            bandwidth_gbs: Some(500.0),
            data_bytes_per_call: Some(10_000_000),
        };
        assert!(
            (h.bandwidth_gbs.expect("should have bw") - 500.0).abs() < 0.1,
            "Bandwidth should be preserved"
        );
        assert_eq!(
            h.data_bytes_per_call.expect("should have data bytes"),
            10_000_000
        );
    }

    #[test]
    fn test_real_profile_results_has_launch_overhead() {
        let results = RealProfileResults {
            kernel_launch_overhead_pct: 15.0,
            kernel_launch_overhead_us: 1500.0,
            ..Default::default()
        };
        assert!(
            (results.kernel_launch_overhead_pct - 15.0).abs() < 0.01,
            "Launch overhead percent preserved"
        );
        assert!(
            (results.kernel_launch_overhead_us - 1500.0).abs() < 0.01,
            "Launch overhead µs preserved"
        );
    }
