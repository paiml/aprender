    // ========================================================================
    // GH-2395: `apr profile` defects found by dogfooding crates.io 0.63.0.
    //
    // These assert BEHAVIOUR against the operation names the profiler actually
    // emits. The pre-existing focus tests fed synthetic hotspots literally named
    // "up_proj" / "down_proj" / "mm_q4k", which the snake_case keyword table
    // matched — so they passed for two releases while `--focus mlp` on a real
    // model kept 1 of 4 FFN operations.
    // ========================================================================

    /// The CamelCase operation names `BrickProfiler` emits for a qwen2 decode,
    /// taken verbatim from `apr profile <gguf>` on the 1.5B fixture.
    fn real_decode_hotspot_names() -> Vec<(&'static str, f64)> {
        vec![
            ("GateProjection", 23.2),
            ("UpProjection", 23.0),
            ("DownProjection", 20.6),
            ("QkvProjection", 15.2),
            ("OutputProjection", 13.2),
            ("LmHead", 2.4),
            ("Activation", 1.4),
            ("RopeEmbedding", 0.5),
            ("Embedding", 0.1),
            ("RmsNorm", 0.4),
            ("AttentionScore", 0.3),
        ]
    }

    fn results_with_real_names() -> RealProfileResults {
        let hotspots = real_decode_hotspot_names()
            .into_iter()
            .map(|(name, percent)| Hotspot {
                name: name.to_string(),
                time_us: percent * 1000.0,
                percent,
                count: 280,
                avg_us: percent * 10.0,
                min_us: 0.0,
                max_us: 0.0,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            })
            .collect();
        RealProfileResults {
            model_path: "qwen2-1.5b.gguf".to_string(),
            architecture: "qwen2".to_string(),
            num_layers: 28,
            vocab_size: 151_936,
            hidden_dim: 1536,
            warmup_passes: 3,
            measure_passes: 10,
            total_inference_us: 100_000.0,
            throughput_tok_s: 18.9,
            tokens_per_pass: 1,
            hotspots,
            per_layer_us: vec![],
            is_real_data: true,
            roofline: None,
            category_summary: None,
            backend: "cpu".to_string(),
            ..Default::default()
        }
    }

    fn filtered_names(focus: ProfileFocus) -> Vec<String> {
        filter_results_by_focus(&results_with_real_names(), focus)
            .hotspots
            .iter()
            .map(|h| h.name.clone())
            .collect()
    }

    #[test]
    fn test_gh2395_focus_mlp_keeps_all_four_real_ffn_ops() {
        // 0.63.0 kept ONLY GateProjection: "upprojection" does not contain
        // "up_proj" and "downprojection" does not contain "down_proj".
        let names = filtered_names(ProfileFocus::Mlp);
        for expected in ["GateProjection", "UpProjection", "DownProjection", "Activation"] {
            assert!(
                names.iter().any(|n| n == expected),
                "--focus mlp dropped {expected}; kept {names:?}"
            );
        }
        assert!(
            !names.iter().any(|n| n == "QkvProjection"),
            "--focus mlp must not keep attention ops; kept {names:?}"
        );
    }

    #[test]
    fn test_gh2395_focus_matmul_is_not_empty() {
        // 0.63.0 returned an EMPTY table with exit 0: no emitted name contains
        // "matmul", "gemm" or "linear".
        let names = filtered_names(ProfileFocus::Matmul);
        assert!(
            !names.is_empty(),
            "--focus matmul returned an empty hotspot table"
        );
        for expected in ["QkvProjection", "UpProjection", "DownProjection", "LmHead"] {
            assert!(
                names.iter().any(|n| n == expected),
                "--focus matmul dropped the weight-matrix op {expected}; kept {names:?}"
            );
        }
        assert!(
            !names.iter().any(|n| n == "Activation" || n == "RmsNorm"),
            "--focus matmul must not keep elementwise ops; kept {names:?}"
        );
    }

    #[test]
    fn test_gh2395_focus_attention_keeps_output_projection() {
        // 0.63.0 kept only QkvProjection + AttentionScore; OutputProjection (13.2%
        // of decode) was silently dropped from the attention view.
        let names = filtered_names(ProfileFocus::Attention);
        for expected in ["QkvProjection", "AttentionScore", "OutputProjection"] {
            assert!(
                names.iter().any(|n| n == expected),
                "--focus attention dropped {expected}; kept {names:?}"
            );
        }
    }

    #[test]
    fn test_gh2395_focus_embedding_keeps_lm_head_and_drops_rope() {
        // 0.63.0 inverted this: it captured RopeEmbedding (positional encoding)
        // and missed LmHead (the vocabulary projection, 27402µs — the largest
        // genuinely embedding-adjacent op).
        let names = filtered_names(ProfileFocus::Embedding);
        assert!(
            names.iter().any(|n| n == "LmHead"),
            "--focus embedding dropped LmHead; kept {names:?}"
        );
        assert!(
            names.iter().any(|n| n == "Embedding"),
            "--focus embedding dropped Embedding; kept {names:?}"
        );
        assert!(
            !names.iter().any(|n| n == "RopeEmbedding"),
            "RopeEmbedding is positional encoding, not an embedding-table op; kept {names:?}"
        );
    }

    #[test]
    fn test_gh2395_focused_table_prints_share_of_total_not_of_survivors() {
        // This is the number the hotspot TABLE prints. 0.63.0 divided each row's
        // time by the sum over the rows it was about to print, so after --focus
        // dropped every row but one, the survivor printed as "100.0%" of a model
        // whose Category Summary put FFN at 70.5%.
        let filtered = filter_results_by_focus(&results_with_real_names(), ProfileFocus::Mlp);
        let percents = hotspot_display_percents(&filtered.hotspots);
        let sum: f64 = percents.iter().sum();
        assert!(
            (sum - 68.2).abs() < 0.5,
            "the focused table must still sum to the FFN share of total (~68.2%), got {sum:.1}%"
        );
        assert!(
            percents.iter().all(|p| *p < 99.0),
            "no single focused row may print as ~100%: {percents:?}"
        );

        // Degenerate case: --focus down to exactly one row.
        let one = RealProfileResults {
            hotspots: vec![filtered.hotspots[0].clone()],
            ..results_with_real_names()
        };
        let p = hotspot_display_percents(&one.hotspots);
        assert!(
            (p[0] - 23.2).abs() < 1e-6,
            "a lone surviving row must keep its 23.2% share of total, printed {:.1}%",
            p[0]
        );
    }

    #[test]
    fn test_gh2395_table_falls_back_to_local_share_when_no_percent_recorded() {
        // Results built without recorded percentages (older snapshots, synthetic
        // instrumentation) must still produce a usable table.
        let hotspots: Vec<Hotspot> = [("A", 750.0), ("B", 250.0)]
            .iter()
            .map(|(name, t)| Hotspot {
                name: (*name).to_string(),
                time_us: *t,
                percent: 0.0,
                count: 1,
                avg_us: *t,
                min_us: *t,
                max_us: *t,
                bottleneck: None,
                efficiency_pct: None,
                category: None,
                bandwidth_gbs: None,
                data_bytes_per_call: None,
            })
            .collect();
        let p = hotspot_display_percents(&hotspots);
        assert!((p[0] - 75.0).abs() < 1e-6, "{p:?}");
        assert!((p[1] - 25.0).abs() < 1e-6, "{p:?}");
        assert!(hotspot_display_percents(&[]).is_empty());
    }

    #[test]
    fn test_gh2395_unknown_focus_is_rejected_not_ignored() {
        // 0.63.0 fell back to the full unfiltered report with exit 0.
        assert!(resolve_focus(Some("bogus")).is_err());
        assert!(resolve_focus(Some("mlpp")).is_err());
        assert!(resolve_focus(Some("")).is_err());
        // Documented values still resolve.
        for ok in ["all", "attention", "attn", "mlp", "ffn", "matmul", "gemm", "embedding"] {
            assert!(resolve_focus(Some(ok)).is_ok(), "--focus {ok} must be accepted");
        }
        assert!(matches!(resolve_focus(None), Ok(ProfileFocus::All)));
    }

    #[test]
    fn test_gh2395_global_json_flag_selects_json_output() {
        // 0.63.0 parsed the global --json (it is `global = true` on `Cli`) and
        // printed the human table anyway, while `apr bench --json` honoured it.
        assert!(matches!(
            resolve_output_format("human", true),
            Ok(OutputFormat::Json)
        ));
        assert!(matches!(
            resolve_output_format("human", false),
            Ok(OutputFormat::Human)
        ));
        assert!(matches!(
            resolve_output_format("flamegraph", false),
            Ok(OutputFormat::Flamegraph)
        ));
    }

    #[test]
    fn test_gh2395_unknown_format_is_rejected_not_silently_human() {
        assert!(resolve_output_format("jsonn", false).is_err());
        assert!(resolve_output_format("", false).is_err());
    }

    // ------------------------------------------------------------------
    // CI mode percentiles
    // ------------------------------------------------------------------

    fn ci_results_with_spread() -> RealProfileResults {
        // Numbers from the issue's non-CI run on the same model/binary:
        // p50=120.2 p95=132.6 p99=136.3 min=99.2 max=137.2, mean 105.96ms.
        RealProfileResults {
            total_inference_us: 105_960.0,
            throughput_tok_s: 18.9,
            latency_p50_ms: 120.2,
            latency_p95_ms: 132.6,
            latency_p99_ms: 136.3,
            latency_min_ms: 99.2,
            latency_max_ms: 137.2,
            measure_passes: 10,
            ..Default::default()
        }
    }

    #[test]
    fn test_gh2395_ci_p99_is_the_tail_not_the_mean() {
        // 0.63.0 assigned the mean to BOTH percentiles, so `--assert-p99` was
        // asserting on a single averaged sample and p50 always equalled p99.
        let report = CiProfileReport::from_results(
            &ci_results_with_spread(),
            &CiAssertions {
                min_throughput: None,
                max_p99_ms: None,
                max_p50_ms: None,
                max_memory_mb: None,
            },
        );
        assert!(
            (report.latency_p99_ms - 136.3).abs() < 1e-6,
            "CI p99 must be the measured tail, got {}",
            report.latency_p99_ms
        );
        assert!(
            (report.latency_p50_ms - 120.2).abs() < 1e-6,
            "CI p50 must be the measured median, got {}",
            report.latency_p50_ms
        );
        assert!(
            report.latency_p99_ms > report.latency_p50_ms,
            "p99 ({}) collapsed onto p50 ({})",
            report.latency_p99_ms,
            report.latency_p50_ms
        );
    }

    #[test]
    fn test_gh2395_assert_p99_fails_on_a_tail_only_regression() {
        // The behavioural consequence: a budget of 130ms sits ABOVE the mean
        // (105.96) and BELOW the real tail (136.3). 0.63.0 compared against the
        // mean and passed, so a tail-only regression could never fail the gate.
        let report = CiProfileReport::from_results(
            &ci_results_with_spread(),
            &CiAssertions {
                min_throughput: None,
                max_p99_ms: Some(130.0),
                max_p50_ms: None,
                max_memory_mb: None,
            },
        );
        assert!(
            !report.passed,
            "--assert-p99 130 must FAIL when the measured p99 is 136.3ms"
        );
        let a = report
            .assertions
            .iter()
            .find(|a| a.name == "latency_p99")
            .expect("p99 assertion recorded");
        assert!(
            a.actual.starts_with("136.3"),
            "p99 assertion reported {} instead of the tail",
            a.actual
        );
    }

    #[test]
    fn test_gh2395_ci_falls_back_to_mean_when_no_percentiles_recorded() {
        // The static-analysis backend records no per-pass times; the mean is then
        // the only honest number available.
        let results = RealProfileResults {
            total_inference_us: 50_000.0,
            ..Default::default()
        };
        let report = CiProfileReport::from_results(
            &results,
            &CiAssertions {
                min_throughput: None,
                max_p99_ms: None,
                max_p50_ms: None,
                max_memory_mb: None,
            },
        );
        assert!((report.latency_p99_ms - 50.0).abs() < 1e-6);
        assert!((report.latency_p50_ms - 50.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // --detect-naive / --fail-on-naive
    // ------------------------------------------------------------------

    fn results_with_achieved_gflops(gflops: f64) -> RealProfileResults {
        let mut r = results_with_real_names();
        r.roofline = Some(RooflineAnalysis {
            peak_compute: 2304.0,
            peak_bandwidth_gbps: 80.0,
            achieved_gflops: gflops,
            achieved_bandwidth_gbps: 60.0,
            arithmetic_intensity: 0.3,
            ai_threshold: 28.8,
            bottleneck: "MEMORY BOUND".to_string(),
            hardware_model: "AMD Ryzen (24 cores, 512-bit SIMD)".to_string(),
            ..Default::default()
        });
        r
    }

    #[test]
    fn test_gh2395_threshold_drives_the_naive_verdict() {
        // 0.63.0 did `let _ = naive_threshold;` — `--threshold 100000` against a
        // run achieving 22.1 GFLOPS still reported "No obvious naive
        // implementations detected".
        let r = results_with_achieved_gflops(22.1);
        assert!(
            detect_naive_implementations(&r, 10.0).is_empty(),
            "22.1 GFLOPS clears the default 10.0 floor"
        );
        let flagged = detect_naive_implementations(&r, 100_000.0);
        assert!(
            !flagged.is_empty(),
            "--threshold 100000 must flag a 22.1 GFLOPS run"
        );
        assert!(
            flagged[0].reason.contains("22.1")&& flagged[0].reason.contains("100000"),
            "the reason must name both numbers, got {:?}",
            flagged[0].reason
        );
    }

    #[test]
    fn test_gh2395_dominant_op_still_flagged_without_a_roofline() {
        let mut r = results_with_real_names();
        r.roofline = None;
        r.hotspots[0].avg_us = r.total_inference_us * 0.9;
        assert!(!detect_naive_implementations(&r, 0.0).is_empty());
    }

    // ------------------------------------------------------------------
    // Hardware identity in the roofline block
    // ------------------------------------------------------------------

    #[test]
    #[cfg(feature = "inference")]
    fn test_gh2395_cpuinfo_vendor_and_model_are_resolved() {
        // 0.63.0 printed `Hardware: Unknown Unknown (24 cores, 512)` beside the
        // peak GFLOPS/bandwidth figures that justify the MEMORY BOUND verdict.
        let amd = "processor\t: 0\nvendor_id\t: AuthenticAMD\ncpu family\t: 25\n\
                   model name\t: AMD Ryzen 9 7900X 12-Core Processor\n";
        assert_eq!(
            trueno::hardware::parse_cpuinfo(amd),
            (
                Some("AMD".to_string()),
                Some("AMD Ryzen 9 7900X 12-Core Processor".to_string())
            )
        );

        let intel = "vendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i9-13900K\n";
        assert_eq!(
            trueno::hardware::parse_cpuinfo(intel).0,
            Some("Intel".to_string())
        );

        // aarch64 kernels expose no vendor_id / model name.
        let arm = "processor\t: 0\nCPU implementer\t: 0x41\nCPU part\t: 0xd4c\n";
        assert_eq!(trueno::hardware::parse_cpuinfo(arm).0, Some("ARM".to_string()));

        // Nothing parseable stays honest rather than inventing a vendor.
        assert_eq!(trueno::hardware::parse_cpuinfo(""), (None, None));
    }

    #[test]
    fn test_gh2395_hardware_label_names_the_cpu_and_units_the_simd_width() {
        // 0.63.0: "Unknown Unknown (24 cores, 512)".
        assert_eq!(
            cpu_hardware_label("AMD", "AMD Ryzen Threadripper 7960X 24-Cores", 24, 512),
            "AMD Ryzen Threadripper 7960X 24-Cores (24 cores, 512-bit SIMD)"
        );
        // Vendor is prepended only when the model does not already carry it.
        assert_eq!(
            cpu_hardware_label("Intel", "Core i9-13900K", 24, 256),
            "Intel Core i9-13900K (24 cores, 256-bit SIMD)"
        );
        // Unresolvable model must not print "Unknown Unknown".
        assert_eq!(
            cpu_hardware_label("ARM", "Unknown", 8, 128),
            "ARM (8 cores, 128-bit SIMD)"
        );
        assert!(!cpu_hardware_label("Unknown", "Unknown", 24, 512).contains("Unknown Unknown"));
    }
