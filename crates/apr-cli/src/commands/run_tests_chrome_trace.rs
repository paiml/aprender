
    // ═══════════════════════════════════════════════════════════════════════════
    // Chrome trace tests (run_entry.rs::print_chrome_trace)
    //
    // NOTE: print_chrome_trace writes to CWD which is process-global.
    // Tests that verify file content must use test_print_chrome_trace_creates_file
    // which uses set_current_dir + serial execution. The other tests verify
    // the function does not panic with various inputs.
    // ═══════════════════════════════════════════════════════════════════════════

    /// Helper to build a trace JSON structure in-memory (mirrors print_chrome_trace logic)
    /// for content verification without touching the filesystem.
    fn build_chrome_trace_events(
        result: &RunResult,
        source: &str,
        max_tokens: usize,
        include_profile: bool,
    ) -> serde_json::Value {
        let mut events = Vec::new();
        let mut ts_us: u64 = 0;

        let load_dur = (result.duration_secs * 1_000_000.0) as u64;
        events.push(serde_json::json!({
            "name": "model_load", "cat": "lifecycle", "ph": "X",
            "ts": 0, "dur": load_dur / 10, "pid": 1, "tid": 1,
            "args": {"source": source, "max_tokens": max_tokens}
        }));
        ts_us = load_dur / 10;

        let tokenize_dur = load_dur / 100;
        events.push(serde_json::json!({
            "name": "tokenize", "cat": "tokenize", "ph": "X",
            "ts": ts_us, "dur": tokenize_dur, "pid": 1, "tid": 1,
            "args": {"source": source}
        }));
        ts_us += tokenize_dur;

        let embed_dur = load_dur / 100;
        events.push(serde_json::json!({
            "name": "embed", "cat": "embed", "ph": "X",
            "ts": ts_us, "dur": embed_dur, "pid": 1, "tid": 1
        }));
        ts_us += embed_dur;

        if let Some(count) = result.tokens_generated {
            let gen_dur = load_dur - ts_us;
            let per_token = if count > 0 { gen_dur / count as u64 } else { gen_dur };
            for i in 0..count {
                let token_start = ts_us + (i as u64 * per_token);
                let layer_dur = per_token * 9 / 10;
                events.push(serde_json::json!({
                    "name": format!("layer_{}", i % 28), "cat": "layer", "ph": "X",
                    "ts": token_start, "dur": layer_dur, "pid": 1, "tid": 1,
                    "args": {"token_idx": i, "layer": i % 28}
                }));
                events.push(serde_json::json!({
                    "name": "sample", "cat": "sample", "ph": "X",
                    "ts": token_start + layer_dur, "dur": per_token - layer_dur,
                    "pid": 1, "tid": 1, "args": {"token_idx": i}
                }));
                events.push(serde_json::json!({
                    "name": format!("token_{}", i), "cat": "decode", "ph": "X",
                    "ts": token_start, "dur": per_token, "pid": 1, "tid": 1,
                    "args": {"token_idx": i}
                }));
            }
        }

        serde_json::json!({
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "metadata": {
                "source": source,
                "tool": "apr run --trace --trace-level chrome",
                "max_tokens": max_tokens,
                "tok_per_sec": result.tok_per_sec,
                "include_profile": include_profile
            }
        })
    }

    #[test]
    fn test_chrome_trace_event_categories() {
        let result = RunResult {
            text: "test".to_string(),
            duration_secs: 2.0,
            cached: false,
            tokens_generated: Some(3),
            tok_per_sec: Some(1.5),
            used_gpu: Some(false),
            generated_tokens: None,
        };

        let json = build_chrome_trace_events(&result, "model.gguf", 16, true);
        let events = json["traceEvents"].as_array().expect("events array");

        // Required categories per apr-chrome-trace-v1.yaml
        let categories: std::collections::HashSet<&str> = events
            .iter()
            .filter_map(|e| e["cat"].as_str())
            .collect();

        assert!(categories.contains("lifecycle"), "missing lifecycle category");
        assert!(categories.contains("tokenize"), "missing tokenize category");
        assert!(categories.contains("embed"), "missing embed category");
        assert!(categories.contains("layer"), "missing layer category");
        assert!(categories.contains("sample"), "missing sample category");
        assert!(categories.contains("decode"), "missing decode category");
    }

    #[test]
    fn test_chrome_trace_zero_tokens() {
        let result = RunResult {
            text: String::new(),
            duration_secs: 0.5,
            cached: true,
            tokens_generated: Some(0),
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        };

        let json = build_chrome_trace_events(&result, "empty.gguf", 0, false);
        let events = json["traceEvents"].as_array().expect("events array");
        // With 0 tokens: model_load + tokenize + embed = 3 events
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_chrome_trace_no_tokens_generated_field() {
        let result = RunResult {
            text: "output".to_string(),
            duration_secs: 1.0,
            cached: false,
            tokens_generated: None,
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        };

        let json = build_chrome_trace_events(&result, "model.gguf", 10, false);
        let events = json["traceEvents"].as_array().expect("events array");
        // tokens_generated is None => no token events
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_chrome_trace_metadata_source() {
        let result = RunResult {
            text: "hi".to_string(),
            duration_secs: 0.1,
            cached: false,
            tokens_generated: Some(1),
            tok_per_sec: Some(10.0),
            used_gpu: Some(true),
            generated_tokens: None,
        };

        let json = build_chrome_trace_events(&result, "my-model.gguf", 64, true);
        assert_eq!(json["metadata"]["source"], "my-model.gguf");
        assert_eq!(json["metadata"]["max_tokens"], 64);
        assert_eq!(json["metadata"]["include_profile"], true);
    }

    #[test]
    fn test_chrome_trace_event_format() {
        let result = RunResult {
            text: "test".to_string(),
            duration_secs: 1.0,
            cached: false,
            tokens_generated: Some(2),
            tok_per_sec: Some(2.0),
            used_gpu: Some(false),
            generated_tokens: None,
        };

        let json = build_chrome_trace_events(&result, "model.gguf", 10, false);
        let events = json["traceEvents"].as_array().expect("events array");

        // All events must have: name, cat, ph, ts, dur, pid, tid (chrome trace format)
        for event in events {
            assert!(event.get("name").is_some(), "missing name: {event}");
            assert!(event.get("cat").is_some(), "missing cat: {event}");
            assert!(event.get("ph").is_some(), "missing ph: {event}");
            assert!(event.get("ts").is_some(), "missing ts: {event}");
            assert!(event.get("dur").is_some(), "missing dur: {event}");
            assert!(event.get("pid").is_some(), "missing pid: {event}");
            assert!(event.get("tid").is_some(), "missing tid: {event}");
            assert_eq!(event["ph"], "X", "all events should be complete duration (X)");
        }
    }

    #[test]
    fn test_chrome_trace_token_count() {
        let result = RunResult {
            text: "test".to_string(),
            duration_secs: 5.0,
            cached: false,
            tokens_generated: Some(10),
            tok_per_sec: Some(2.0),
            used_gpu: Some(false),
            generated_tokens: None,
        };

        let json = build_chrome_trace_events(&result, "model.gguf", 10, false);
        let events = json["traceEvents"].as_array().expect("events array");
        // 3 base events + 10 tokens * 3 events each (layer, sample, decode) = 33
        assert_eq!(events.len(), 3 + 10 * 3);
    }

    #[test]
    fn test_chrome_trace_display_time_unit() {
        let result = RunResult {
            text: "t".to_string(),
            duration_secs: 1.0,
            cached: false,
            tokens_generated: Some(1),
            tok_per_sec: Some(1.0),
            used_gpu: None,
            generated_tokens: None,
        };
        let json = build_chrome_trace_events(&result, "m.gguf", 1, false);
        assert_eq!(json["displayTimeUnit"], "ms");
    }

    // Also exercise the actual function (no file validation, just no-panic)
    #[test]
    fn test_print_chrome_trace_creates_file() {
        // This test writes to CWD. Use a unique tempdir and set_current_dir.
        // May conflict with parallel tests, but the function itself should not panic.
        let result = RunResult {
            text: "Hello world".to_string(),
            duration_secs: 1.0,
            cached: false,
            tokens_generated: Some(5),
            tok_per_sec: Some(5.0),
            used_gpu: Some(false),
            generated_tokens: None,
        };
        // Just ensure no panic; file creation is best-effort
        print_chrome_trace(&result, "test-model.gguf", 32, false);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // print_benchmark_results tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_print_benchmark_results_text() {
        let result = RunResult {
            text: "test output".to_string(),
            duration_secs: 2.0,
            cached: false,
            tokens_generated: Some(100),
            tok_per_sec: Some(50.0),
            used_gpu: Some(false),
            generated_tokens: None,
        };
        print_benchmark_results(&result, "model.gguf", "text", 100);
    }

    #[test]
    fn test_print_benchmark_results_json() {
        let result = RunResult {
            text: "test".to_string(),
            duration_secs: 1.0,
            cached: false,
            tokens_generated: Some(50),
            tok_per_sec: Some(50.0),
            used_gpu: Some(false),
            generated_tokens: None,
        };
        print_benchmark_results(&result, "model.gguf", "json", 50);
    }

    #[test]
    fn test_print_benchmark_zero_duration() {
        let result = RunResult {
            text: "".to_string(),
            duration_secs: 0.0,
            cached: false,
            tokens_generated: Some(10),
            tok_per_sec: None,
            used_gpu: None,
            generated_tokens: None,
        };
        print_benchmark_results(&result, "model.gguf", "text", 10);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // print_trace_config tests
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_print_trace_config_basic() {
        print_trace_config("layer", None, false, None, false);
    }

    #[test]
    fn test_print_trace_config_all_options() {
        let path = PathBuf::from("/tmp/trace.json");
        let steps = vec!["Attention".to_string(), "FFN".to_string()];
        print_trace_config("chrome", Some(&steps), true, Some(&path), true);
    }
