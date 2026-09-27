    // ═══════════════════════════════════════════════════════════════════════════
    // Layer trace = the apr-trace-v1 document (TR-09, #4564)
    //
    // `apr run --trace --trace-level layer` used to print an 8-step table whose
    // per-step values were `wall_ms / tokens * <fixed share>` — the same
    // 85/8/2/1.7 split for every model and every prompt — while serve answered
    // `X-Trace-Level: layer` for the same request with a different schema. Both
    // now come from `realizar::api::apr_trace`.
    // ═══════════════════════════════════════════════════════════════════════════

    fn layer_trace_result(duration_secs: f64, tokens: usize) -> RunResult {
        RunResult {
            text: "hi".to_string(),
            duration_secs,
            cached: true,
            tokens_generated: Some(tokens),
            tok_per_sec: Some(tokens as f64 / duration_secs),
            used_gpu: Some(false),
            gpu_attempted: None,
            generated_tokens: None,
            token_texts: None,
            usage: RunUsage {
                prompt_tokens: Some(4),
                completion_tokens: Some(tokens),
                num_layers: Some(2),
                ..RunUsage::default()
            },
        }
    }

    /// Zero out the fields that are timings, so two documents compare on schema
    /// and content only ("byte-identical modulo timestamps").
    #[cfg(feature = "inference")]
    fn untimed(t: &realizar::api::TraceData) -> String {
        let mut v = serde_json::to_value(t).expect("apr-trace-v1 serializes");
        v["total_time_us"] = 0.into();
        for row in v["breakdown"].as_array_mut().expect("breakdown") {
            row["time_us"] = 0.into();
        }
        serde_json::to_string(&v).expect("serializes")
    }

    /// FALSIFY-TRACE-004: the same request through `apr run` and through serve
    /// yields byte-identical apr-trace-v1, modulo timestamps. Serve's input here
    /// is what its handler passes for an untraced backend (no events, no layer
    /// timings) with the same token and layer counts.
    #[cfg(feature = "inference")]
    #[test]
    fn falsify_trace_004_run_and_serve_emit_one_schema() {
        for level in ["brick", "step", "layer"] {
            let run = run_apr_trace(&layer_trace_result(2.0, 3), level).expect("run document");
            let serve = realizar::api::apr_trace(&realizar::api::ServeTrace {
                level: Some(level),
                events: &[],
                layers: None,
                wall_us: 1_234,
                prompt_tokens: 4,
                completion_tokens: 3,
                num_layers: 2,
            })
            .expect("serve document");
            assert_eq!(untimed(&run), untimed(&serve), "level {level}");
        }
    }

    /// The table is the document: no fixed-share row survives, the provenance is
    /// printed, and the last line is the document itself.
    #[cfg(feature = "inference")]
    #[test]
    fn layer_trace_renders_the_document_and_invents_nothing() {
        let result = layer_trace_result(2.0, 4);
        let out = render_layer_trace(&result);
        for invented in ["TRANSFORMER", "LM_HEAD", "85.0%", "Est. Time"] {
            assert!(!out.contains(invented), "`{invented}` is a fabricated row; got:\n{out}");
        }
        assert!(out.contains("provenance: wall_clock_total"), "got:\n{out}");
        let doc = run_apr_trace(&result, "layer").expect("layer document");
        let line = format!("apr-trace-v1: {}", serde_json::to_string(&doc).expect("json"));
        assert!(out.contains(&line), "the document line must be emitted verbatim; got:\n{out}");
    }

    /// Zero tokens must not divide by zero or print NaN.
    #[test]
    fn layer_trace_zero_tokens_is_finite() {
        let out = render_layer_trace(&layer_trace_result(1.0, 0));
        // Check the rendered numbers, not prose: the row `total_inference`
        // contains "inf" and is not a non-finite value.
        for bad in ["NaNms", "infms", "NaN ", "-0.00ms"] {
            assert!(!out.contains(bad), "`{bad}` in:\n{out}");
        }
    }
