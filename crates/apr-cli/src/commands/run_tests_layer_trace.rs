
    // ═══════════════════════════════════════════════════════════════════════════
    // Layer trace honesty (run_trace_print.rs::render_layer_trace)
    //
    // `apr run --trace --trace-level layer` printed a table headed `Time` whose
    // per-step values were `wall_ms / tokens * <fixed share>` — the same
    // 85/8/2/1.7 split for every model and every prompt. TOKENIZE, EMBED and
    // DECODE came out identical to the hundredth of a millisecond in every run,
    // and the table "proved" TRANSFORMER was 85% while the real [BRICK-PROFILE]
    // block in the same output said FFN 42% / Qkv 21% / LmHead 4.5%.
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
            usage: Default::default(),
        }
    }

    /// A derived number must never be printed as if it were measured.
    #[test]
    fn layer_trace_marks_derived_timings_as_estimates() {
        let out = render_layer_trace(&layer_trace_result(2.0, 4), 4);

        assert!(
            out.contains("ESTIMATED"),
            "the table must say the per-step values are estimated; got:\n{out}"
        );
        assert!(
            out.contains("Est. Time"),
            "the column heading must not be a bare `Time`; got:\n{out}"
        );
        assert!(
            out.contains("~"),
            "each derived value must be marked approximate; got:\n{out}"
        );
        assert!(
            out.contains("Share"),
            "the fixed share used to derive each value must be shown, so the \
             three equal rows are explicable; got:\n{out}"
        );
        assert!(
            out.contains("85.0%"),
            "TRANSFORMER's assumed 85% share must be visible; got:\n{out}"
        );
    }

    /// The run total and the decode rate must be labelled, or the table
    /// contradicts the profiler block printed a few lines above it: the same
    /// run reports an end-to-end rate and a decode rate that differ by more
    /// than an order of magnitude, and neither one says which it is. The
    /// assertions below are the check; no rate literal belongs in this
    /// comment, because a number here is a claim no measurement resolves.
    #[test]
    fn layer_trace_labels_wall_clock_and_end_to_end_rate() {
        let out = render_layer_trace(&layer_trace_result(2.0, 4), 4);

        assert!(
            out.contains("incl. model load"),
            "TOTAL must state that it includes model load; got:\n{out}"
        );
        assert!(
            out.contains("end-to-end"),
            "the rate must be labelled end-to-end, not left to be read as decode \
             throughput; got:\n{out}"
        );
        assert!(
            out.contains("BRICK-PROFILE"),
            "the table must point at the measured decode-rate figure; got:\n{out}"
        );
    }

    /// The estimate itself must still be arithmetically what it claims: the
    /// stated share of per-token wall time.
    #[test]
    fn layer_trace_estimates_match_their_stated_share() {
        // 2.0s wall / 4 tokens = 500ms per token; TRANSFORMER's share is 85%.
        let out = render_layer_trace(&layer_trace_result(2.0, 4), 4);
        assert!(
            out.contains("425.00ms"),
            "TRANSFORMER must be 0.85 * 500ms = 425.00ms; got:\n{out}"
        );
        // 1.7% of 500ms = 8.50ms, shared by TOKENIZE/EMBED/DECODE.
        assert!(
            out.contains("8.50ms"),
            "the 1.7%-share steps must be 8.50ms; got:\n{out}"
        );
    }

    /// Zero tokens must not divide by zero or print NaN.
    #[test]
    fn layer_trace_zero_tokens_is_finite() {
        let out = render_layer_trace(&layer_trace_result(1.0, 0), 0);
        assert!(!out.contains("NaN"), "got:\n{out}");
        assert!(!out.contains("inf"), "got:\n{out}");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Layer trace on the GPU: refused by name, never estimated
    //
    // On the GPU path nothing times the steps: the table is the same fixed
    // 85/8/2/1.7 split of wall time, and kernel launches are asynchronous, so a
    // host-side split of one wall-clock total says nothing about where the
    // device spent it. An ESTIMATED label is not enough there — the run is
    // refused by name, so a script cannot mistake the table for a measurement.
    // ═══════════════════════════════════════════════════════════════════════════

    fn layer_trace_result_on(used_gpu: Option<bool>) -> RunResult {
        RunResult {
            used_gpu,
            ..layer_trace_result(2.0, 4)
        }
    }

    #[test]
    fn layer_trace_on_gpu_is_refused_by_name() {
        let err = print_layer_trace(&layer_trace_result_on(Some(true)), 4)
            .expect_err("a GPU run must not print estimated layer times");
        let CliError::NotImplemented(msg) = &err else {
            panic!("expected NotImplemented, got {err:?}");
        };
        for needle in ["--trace-level layer", "GPU", "not measured"] {
            assert!(msg.contains(needle), "refusal must name {needle:?}; got: {msg}");
        }
        assert_eq!(err.exit_code_value(), 12, "NotImplemented exit code");
    }

    /// The refusal points somewhere real, and never at a number it cannot back.
    #[test]
    fn layer_trace_gpu_refusal_names_no_timing() {
        let msg = layer_trace_refusal(&layer_trace_result_on(Some(true)))
            .map(|e| e.to_string())
            .expect("GPU run must be refused");
        assert!(
            !msg.chars().any(|c| c.is_ascii_digit()),
            "a refusal must not carry a number; got: {msg}"
        );
        assert!(msg.contains("--no-gpu"), "must name the CPU fallback; got: {msg}");
    }

    /// CPU and an unknown backend keep the labelled ESTIMATED table.
    #[test]
    fn layer_trace_off_gpu_still_renders() {
        for used_gpu in [Some(false), None] {
            assert!(
                layer_trace_refusal(&layer_trace_result_on(used_gpu)).is_none(),
                "used_gpu={used_gpu:?} must not be refused"
            );
            assert!(print_layer_trace(&layer_trace_result_on(used_gpu), 4).is_ok());
        }
    }
