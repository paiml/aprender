    // =========================================================================
    // `--json` must emit JSON, not human-formatted text.
    //
    // apr 0.63.0 (from crates.io) ignored `--json` on `apr pull` and `apr rm`:
    //
    //   $ apr pull qwen2.5-coder --dry-run --json     # rc=0
    //   === APR Pull ===
    //   Model:    qwen2.5-coder
    //   ...
    //   json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
    //
    // These tests pipe the exact stdout each command produces in `--json` mode
    // through a real JSON parse, which is what a consumer does.
    // =========================================================================

    /// Parse the bytes a `--json` consumer would read, failing loudly with the
    /// offending stdout when they are not JSON.
    fn parse_as_a_consumer_would(command: &str, stdout: &str) -> serde_json::Value {
        match serde_json::from_str::<serde_json::Value>(stdout) {
            Ok(v) => v,
            Err(e) => panic!(
                "`{command} --json` must write parseable JSON to stdout, but a consumer got \
                 {e}. Actual stdout was:\n{stdout}"
            ),
        }
    }

    #[test]
    fn pull_dry_run_json_stdout_parses_as_json() {
        let report =
            build_dry_run_report("qwen2.5-coder", None, false).expect("known alias must resolve");
        let stdout = report.stdout(true);

        let parsed = parse_as_a_consumer_would("apr pull --dry-run", &stdout);
        assert_eq!(parsed["model"], "qwen2.5-coder");
        assert_eq!(parsed["mode"], "dry-run");
        assert_eq!(parsed["revision"], "main");
        // A bool, not the "false" the human rendering paints yellow. (The value
        // itself depends on APR_OFFLINE/HF_HUB_OFFLINE in the environment, so
        // only the type is asserted here; the flag is pinned in the next test.)
        assert!(
            parsed["offline"].is_boolean(),
            "offline must be a JSON bool: {parsed:?}"
        );
        assert!(
            parsed["resolved"]
                .as_str()
                .is_some_and(|s| s.starts_with("hf://")),
            "resolved must carry the canonical URI: {parsed:?}"
        );
    }

    #[test]
    fn pull_dry_run_json_stdout_carries_no_human_decoration() {
        let report = build_dry_run_report("qwen2.5-coder", Some("v1.0"), true)
            .expect("known alias must resolve");
        let stdout = report.stdout(true);

        // The banner and the aligned key/value block are what broke consumers.
        for leak in ["=== APR Pull ===", "Model:    ", "Mode:     ", "(no network I/O)"] {
            assert!(
                !stdout.contains(leak),
                "human decoration {leak:?} leaked into `--json` stdout:\n{stdout}"
            );
        }
        let parsed = parse_as_a_consumer_would("apr pull --dry-run", &stdout);
        assert_eq!(parsed["revision"], "v1.0");
        assert_eq!(parsed["offline"], serde_json::Value::Bool(true));
    }

    #[test]
    fn pull_dry_run_human_mode_is_still_human() {
        let report =
            build_dry_run_report("qwen2.5-coder", None, false).expect("known alias must resolve");
        let stdout = report.stdout(false);
        assert!(
            stdout.contains("Mode:") && stdout.contains("(no network I/O)"),
            "default (non-json) rendering must stay human-readable:\n{stdout}"
        );
        // The banner moved into this renderer; without it the default output
        // would silently lose a line it has always had.
        assert!(
            stdout.contains("=== APR Pull ==="),
            "default (non-json) rendering must keep its banner:\n{stdout}"
        );
        assert!(
            serde_json::from_str::<serde_json::Value>(&stdout).is_err(),
            "human mode must not be silently emitting JSON:\n{stdout}"
        );
    }

    #[test]
    fn rm_json_stdout_parses_as_json_on_success() {
        let stdout = remove_stdout("qwen2.5-coder", true, true)
            .expect("a successful --json rm must write a document to stdout");

        let parsed = parse_as_a_consumer_would("apr rm", &stdout);
        assert_eq!(parsed["model"], "qwen2.5-coder");
        assert_eq!(parsed["removed"], serde_json::Value::Bool(true));
        assert!(
            !stdout.contains("=== APR Remove ==="),
            "banner leaked into `--json` stdout:\n{stdout}"
        );
    }

    #[test]
    fn rm_json_writes_nothing_to_stdout_when_the_model_is_absent() {
        // GH-601 keeps the non-zero exit code; `--json` must not put a
        // human "not found" line where a document is expected. Partial or
        // non-JSON stdout is exactly what breaks `apr rm ... --json | jq`.
        assert!(
            remove_stdout("no-such-model", false, true).is_none(),
            "a failed --json rm must leave stdout empty for the consumer"
        );
    }

    #[test]
    fn rm_human_mode_is_still_human() {
        let stdout = remove_stdout("qwen2.5-coder", true, false).expect("human output");
        assert!(stdout.contains("=== APR Remove ==="), "stdout: {stdout}");
        assert!(stdout.contains("Model removed from cache"), "stdout: {stdout}");
        let missing = remove_stdout("no-such-model", false, false).expect("human output");
        assert!(
            missing.contains("Model not found in cache"),
            "stdout: {missing}"
        );
    }
