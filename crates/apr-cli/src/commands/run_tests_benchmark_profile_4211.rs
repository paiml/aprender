    // #4211: `apr run --benchmark --json` stdout must be ONE JSON document, and
    // `--profile` must not call a GPU run CPU-bound or tell it to use --gpu.

    fn bench_result_4211(tok_per_sec: Option<f64>, used_gpu: Option<bool>) -> RunResult {
        RunResult {
            text: "hello".to_string(),
            duration_secs: 8.892,
            cached: true,
            tokens_generated: Some(56),
            tok_per_sec,
            used_gpu,
            gpu_attempted: used_gpu,
            generated_tokens: None,
            token_texts: None,
            usage: Default::default(),
        }
    }

    #[test]
    fn benchmark_json_stdout_is_exactly_one_json_document_4211() {
        let r = bench_result_4211(Some(6.3), Some(true));
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "json", 32);
        // The whole of stdout parses as one document: a human preamble, or a
        // second document after it, makes this fail (the `jq .` of the ticket).
        let v: serde_json::Value = serde_json::from_str(stdout.trim())
            .unwrap_or_else(|e| panic!("stdout is not one JSON document ({e}): {stdout:?}"));
        assert!(v.is_object(), "stdout JSON is not an object: {v}");
        assert_eq!(stdout.trim().lines().count(), 1, "stdout: {stdout:?}");
        // The human block still reaches the user, on stderr.
        assert!(stderr.contains("Benchmark Results"), "stderr: {stderr:?}");
        assert!(!stdout.contains("Benchmark Results"), "stdout: {stdout:?}");
    }

    #[test]
    fn benchmark_reports_one_throughput_with_its_basis_4211() {
        // Engine figure present: stdout reports it — the number stderr's
        // "Generated N tokens in X ms (Y tok/s)" line prints — not the wall-clock
        // 56 / 8.892 = 6.3. 9.7 is distinct from 6.3 so the two cannot alias.
        // The backend marked where generation began: setup is excluded.
        let mut r = bench_result_4211(Some(9.7), Some(true));
        r.usage.generation_ms = Some(5773);
        r.usage.setup_ms = Some(3119);
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "json", 32);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
        assert_eq!(v["tok_s"].as_f64(), Some(9.7), "{v}");
        assert_eq!(v["tok_s_basis"], "generation", "{v}");
        assert_eq!(v["generation_ms"].as_u64(), Some(5773), "{v}");
        assert_eq!(v["latency_basis"], "wall", "{v}");
        assert!(stderr.contains("setup excluded"), "stderr: {stderr:?}");

        // The path did not mark it (wgpu, for one): the engine figure still
        // includes weight upload + F2, and must not be called setup-excluded.
        let r = bench_result_4211(Some(9.7), Some(true));
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "json", 32);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
        assert_eq!(v["tok_s"].as_f64(), Some(9.7), "{v}");
        assert_eq!(v["tok_s_basis"], "inference_incl_setup", "{v}");
        assert!(v["generation_ms"].is_null(), "{v}");
        assert!(!stderr.contains("excluded"), "stderr: {stderr:?}");

        // No engine figure: wall clock, and labelled as such.
        let r = bench_result_4211(None, Some(false));
        let (stdout, _) = render_benchmark_results(&r, "m.gguf", "json", 32);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).expect("json");
        assert_eq!(v["tok_s"].as_f64(), Some(6.3), "{v}"); // 56 / 8.892
        assert_eq!(v["tok_s_basis"], "wall", "{v}");
    }

    #[test]
    fn benchmark_human_format_keeps_the_block_on_stdout_4211() {
        let r = bench_result_4211(Some(6.3), None);
        let (stdout, stderr) = render_benchmark_results(&r, "m.gguf", "text", 32);
        assert!(stdout.contains("Benchmark Results"), "stdout: {stdout:?}");
        assert!(stdout.contains("tok/s: 6.3"), "stdout: {stdout:?}");
        assert!(stderr.is_empty(), "stderr: {stderr:?}");
    }

    #[test]
    fn roofline_on_a_gpu_run_never_says_cpu_or_use_gpu_4211() {
        // Every tier the throughput-only table had, including the two the
        // ticket quotes (6.x tok/s and <5 tok/s on CUDA).
        for tps in [0.0, 1.0, 4.1, 5.0, 6.3, 12.0, 20.0, 35.0, 50.0, 93.9, 250.0] {
            let (c, m, bottleneck, rec) = classify_roofline(tps, Some(true));
            assert_eq!(u32::from(c) + u32::from(m), 100, "tps {tps}");
            assert!(!bottleneck.contains("CPU"), "tps {tps}: {bottleneck}");
            assert!(!bottleneck.contains("DRAM"), "tps {tps}: {bottleneck}");
            assert!(!rec.contains("--gpu"), "tps {tps}: {rec}");
            assert!(!rec.contains("CPU"), "tps {tps}: {rec}");
        }
    }

    #[test]
    fn roofline_on_a_cpu_run_never_claims_tensor_cores_4211() {
        for tps in [0.0, 4.1, 6.3, 35.0, 50.1, 93.9, 250.0] {
            let (_, _, bottleneck, rec) = classify_roofline(tps, Some(false));
            assert!(!bottleneck.contains("GPU"), "tps {tps}: {bottleneck}");
            assert!(!rec.contains("GPU-accelerated"), "tps {tps}: {rec}");
        }
        // The CPU-bound readings the ticket saw on CUDA still apply to a CPU run.
        assert!(classify_roofline(6.3, Some(false)).3.contains("--gpu"));
        assert!(classify_roofline(4.1, Some(false)).2.contains("CPU"));
    }

    #[test]
    fn roofline_with_unreported_backend_names_no_backend_4211() {
        for tps in [0.0, 4.1, 6.3, 35.0, 50.1, 93.9, 250.0] {
            let (_, _, bottleneck, rec) = classify_roofline(tps, None);
            assert!(!bottleneck.contains("GPU"), "tps {tps}: {bottleneck}");
            assert!(!bottleneck.contains("tensor cores"), "tps {tps}: {bottleneck}");
            assert!(!rec.contains("GPU-accelerated"), "tps {tps}: {rec}");
        }
    }
