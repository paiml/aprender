
    // ═══════════════════════════════════════════════════════════════════════════
    // Chrome trace tests (run_entry.rs::print_chrome_trace)
    //
    // NOTE: print_chrome_trace writes to CWD which is process-global.
    // Tests that verify file content must use test_print_chrome_trace_creates_file
    // which uses set_current_dir + serial execution. The other tests verify
    // the function does not panic with various inputs.
    // ═══════════════════════════════════════════════════════════════════════════

    /// These tests used to assert against a hand-maintained COPY of
    /// `print_chrome_trace`'s body kept here in the test file, which could only
    /// ever prove the copy agreed with itself. They now call the real builder.
    use super::build_chrome_trace as build_chrome_trace_events;

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
            token_texts: None,
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
            token_texts: None,
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
            token_texts: None,
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
            token_texts: None,
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
            token_texts: None,
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
            token_texts: None,
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
            token_texts: None,
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
            token_texts: None,
        };
        // Just ensure no panic; file creation is best-effort
        print_chrome_trace(&result, "test-model.gguf", 32, false, None);
    }

    /// `--trace-level chrome --trace-output FILE` must put the chrome trace in
    /// FILE.
    ///
    /// Before the fix the chrome writer ignored `--trace-output` entirely: it
    /// wrote `trace-<epoch>.json` into the process CWD, and the path the user
    /// named was left holding the summary stub written by the inference layer.
    /// A scripted consumer read the file it asked for and found no
    /// `traceEvents` at all.
    #[test]
    fn chrome_trace_honours_requested_output_path() {
        let dir = std::env::temp_dir().join(format!(
            "apr-chrome-trace-{}-{}",
            std::process::id(),
            line!()
        ));
        std::fs::create_dir_all(&dir).expect("tempdir");
        let target = dir.join("requested.json");

        let result = RunResult {
            text: "Hello world".to_string(),
            duration_secs: 1.0,
            cached: false,
            tokens_generated: Some(3),
            tok_per_sec: Some(3.0),
            used_gpu: Some(false),
            generated_tokens: None,
            token_texts: None,
        };
        print_chrome_trace(&result, "test-model.gguf", 32, false, Some(&target));

        let body = std::fs::read_to_string(&target)
            .unwrap_or_else(|e| panic!("chrome trace must be written to {}: {e}", target.display()));
        let v: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
        let events = v["traceEvents"].as_array().expect("traceEvents array");
        assert!(
            !events.is_empty(),
            "chrome trace at the requested path must carry events, got: {body}"
        );
        let _ = std::fs::remove_dir_all(&dir);
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
            token_texts: None,
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
            token_texts: None,
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
            token_texts: None,
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
