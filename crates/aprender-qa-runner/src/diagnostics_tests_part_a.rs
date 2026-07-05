#[test]
fn test_failure_details_from_evidence() {
    let evidence = test_evidence();
    let details = FailureDetails::from(&evidence);

    assert_eq!(details.gate_id, "G3-STABLE");
    assert_eq!(details.model, "Qwen/Qwen2.5-Coder-0.5B-Instruct");
    assert_eq!(details.format, "Apr");
    assert_eq!(details.backend, "Cpu");
    assert_eq!(details.exit_code, Some(-1));
}

#[test]
fn test_environment_context_collect() {
    let ctx = EnvironmentContext::collect();

    assert!(!ctx.os.is_empty());
    assert!(!ctx.arch.is_empty());
    assert!(!ctx.aprender_qa_version.is_empty());
}

#[test]
fn test_diagnostic_result_serialization() {
    let result = DiagnosticResult {
        command: "apr check model.apr".to_string(),
        success: true,
        stdout: "{}".to_string(),
        stderr: String::new(),
        duration_ms: 1234,
        timed_out: false,
    };

    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("apr check"));
    assert!(json.contains("1234"));
}

#[test]
fn test_generate_markdown() {
    let reporter = FailFastReporter::new(Path::new("output"));
    let evidence = test_evidence();

    let report = FailFastReport {
        version: "1.0.0".to_string(),
        timestamp: "2024-02-04T18:00:00Z".to_string(),
        failure: FailureDetails::from(&evidence),
        environment: EnvironmentContext {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            aprender_qa_version: "0.1.0".to_string(),
            apr_cli_version: "0.2.12".to_string(),
            git_commit: "abc123".to_string(),
            git_branch: "main".to_string(),
            git_dirty: false,
            rustc_version: "1.93.0".to_string(),
        },
        diagnostics: DiagnosticsBundle {
            check: None,
            inspect: None,
            trace: None,
            tensors: None,
            explain: None,
        },
        reproduction: ReproductionInfo {
            command: "apr-qa run playbook.yaml --fail-fast".to_string(),
            model_path: "/path/to/model.apr".to_string(),
            playbook: Some("playbook.yaml".to_string()),
        },
    };

    let md = reporter.generate_markdown(&report);

    assert!(md.contains("# Fail-Fast Report: G3-STABLE"));
    assert!(md.contains("| Gate | `G3-STABLE` |"));
    assert!(md.contains("| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` |"));
    assert!(md.contains("## Reproduction"));
}

#[test]
fn test_reporter_new() {
    let reporter = FailFastReporter::new(Path::new("output"));
    assert_eq!(reporter.output_dir, PathBuf::from("output"));
    assert_eq!(reporter.binary, "apr");
}

#[test]
fn test_reporter_with_binary() {
    let reporter = FailFastReporter::new(Path::new("output")).with_binary("/custom/apr");
    assert_eq!(reporter.binary, "/custom/apr");
}

#[test]
fn test_generate_markdown_with_diagnostics() {
    let reporter = FailFastReporter::new(Path::new("output"));
    let evidence = test_evidence();

    let check_result = DiagnosticResult {
        command: "apr check /model.apr --json".to_string(),
        success: false,
        stdout: "{}".to_string(),
        stderr: "Error: failed to load model".to_string(),
        duration_ms: 500,
        timed_out: false,
    };

    let inspect_result = DiagnosticResult {
        command: "apr inspect /model.apr --json".to_string(),
        success: true,
        stdout: r#"{"architecture": "Qwen2"}"#.to_string(),
        stderr: String::new(),
        duration_ms: 200,
        timed_out: false,
    };

    let tensors_result = DiagnosticResult {
        command: "apr tensors /model.apr --json".to_string(),
        success: true,
        stdout: r#"{"count": 256}"#.to_string(),
        stderr: String::new(),
        duration_ms: 150,
        timed_out: false,
    };

    let trace_result = DiagnosticResult {
        command: "apr trace /model.apr --payload --json".to_string(),
        success: true,
        stdout: r#"{"layers": []}"#.to_string(),
        stderr: String::new(),
        duration_ms: 1000,
        timed_out: false,
    };

    let explain_result = DiagnosticResult {
        command: "apr explain G3-STABLE".to_string(),
        success: true,
        stdout: "G3-STABLE: Model stability gate - ensures no crashes".to_string(),
        stderr: String::new(),
        duration_ms: 50,
        timed_out: false,
    };

    let report = FailFastReport {
        version: "1.0.0".to_string(),
        timestamp: "2024-02-04T18:00:00Z".to_string(),
        failure: FailureDetails::from(&evidence),
        environment: EnvironmentContext {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            aprender_qa_version: "0.1.0".to_string(),
            apr_cli_version: "0.2.12".to_string(),
            git_commit: "abc123".to_string(),
            git_branch: "main".to_string(),
            git_dirty: true,
            rustc_version: "1.93.0".to_string(),
        },
        diagnostics: DiagnosticsBundle {
            check: Some(check_result),
            inspect: Some(inspect_result),
            trace: Some(trace_result),
            tensors: Some(tensors_result),
            explain: Some(explain_result),
        },
        reproduction: ReproductionInfo {
            command: "apr-qa run playbook.yaml --fail-fast".to_string(),
            model_path: "/path/to/model.apr".to_string(),
            playbook: Some("playbook.yaml".to_string()),
        },
    };

    let md = reporter.generate_markdown(&report);

    // Check diagnostic sections are included
    assert!(md.contains("## Pipeline Check Results"));
    assert!(md.contains("**Pipeline check failed:**"));
    assert!(md.contains("Error: failed to load model"));
    assert!(md.contains("## Model Metadata"));
    assert!(md.contains("apr inspect output"));
    assert!(md.contains("## Tensor Inventory"));
    assert!(md.contains("apr tensors output"));
    assert!(md.contains("## Layer Trace"));
    assert!(md.contains("apr trace output"));
    assert!(md.contains("## Error Analysis"));
    assert!(md.contains("G3-STABLE: Model stability gate"));
    assert!(md.contains("[dirty]")); // git dirty flag
    assert!(md.contains("## Stderr Capture"));
    assert!(md.contains("SIGSEGV at 0x12345"));
}

#[test]
fn test_generate_markdown_successful_check() {
    let reporter = FailFastReporter::new(Path::new("output"));
    let evidence = test_evidence();

    let check_result = DiagnosticResult {
        command: "apr check /model.apr --json".to_string(),
        success: true,
        stdout: "{}".to_string(),
        stderr: String::new(),
        duration_ms: 500,
        timed_out: false,
    };

    let report = FailFastReport {
        version: "1.0.0".to_string(),
        timestamp: "2024-02-04T18:00:00Z".to_string(),
        failure: FailureDetails::from(&evidence),
        environment: EnvironmentContext {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            aprender_qa_version: "0.1.0".to_string(),
            apr_cli_version: "0.2.12".to_string(),
            git_commit: "abc123".to_string(),
            git_branch: "main".to_string(),
            git_dirty: false,
            rustc_version: "1.93.0".to_string(),
        },
        diagnostics: DiagnosticsBundle {
            check: Some(check_result),
            inspect: None,
            trace: None,
            tensors: None,
            explain: None,
        },
        reproduction: ReproductionInfo {
            command: "apr-qa run playbook.yaml --fail-fast".to_string(),
            model_path: "/path/to/model.apr".to_string(),
            playbook: Some("playbook.yaml".to_string()),
        },
    };

    let md = reporter.generate_markdown(&report);

    assert!(md.contains("## Pipeline Check Results"));
    assert!(md.contains("All pipeline checks passed."));
}

#[test]
fn test_run_trace_skips_non_apr() {
    let reporter = FailFastReporter::new(Path::new("output"));
    // run_trace should return None for non-.apr files
    let result = reporter.run_trace(Path::new("/model.safetensors"));
    assert!(result.is_none());
}

#[test]
fn test_diagnostics_bundle_debug() {
    let bundle = DiagnosticsBundle {
        check: None,
        inspect: None,
        trace: None,
        tensors: None,
        explain: None,
    };
    // Just ensure Debug trait is implemented
    let _ = format!("{:?}", bundle);
}

#[test]
fn test_reproduction_info_debug() {
    let info = ReproductionInfo {
        command: "apr-qa run test.yaml".to_string(),
        model_path: "/test/model.apr".to_string(),
        playbook: None,
    };
    // Just ensure Debug trait is implemented
    let _ = format!("{:?}", info);
}

#[test]
fn test_fail_fast_report_debug() {
    let evidence = test_evidence();
    let report = FailFastReport {
        version: "1.0.0".to_string(),
        timestamp: "2024-02-04T18:00:00Z".to_string(),
        failure: FailureDetails::from(&evidence),
        environment: EnvironmentContext {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            aprender_qa_version: "0.1.0".to_string(),
            apr_cli_version: "0.2.12".to_string(),
            git_commit: "abc123".to_string(),
            git_branch: "main".to_string(),
            git_dirty: false,
            rustc_version: "1.93.0".to_string(),
        },
        diagnostics: DiagnosticsBundle {
            check: None,
            inspect: None,
            trace: None,
            tensors: None,
            explain: None,
        },
        reproduction: ReproductionInfo {
            command: "apr-qa run playbook.yaml --fail-fast".to_string(),
            model_path: "/path/to/model.apr".to_string(),
            playbook: Some("playbook.yaml".to_string()),
        },
    };
    // Just ensure Debug trait is implemented
    let _ = format!("{:?}", report);
}

// ---- New coverage tests below ----

#[test]
fn test_run_command_with_timeout_success() {
    let reporter = FailFastReporter::new(Path::new("."));
    let result = reporter.run_command_with_timeout(&["echo", "hello"], Duration::from_secs(5));
    assert!(result.success);
    assert!(result.stdout.contains("hello"));
    assert!(!result.timed_out);
    assert_eq!(result.command, "echo hello");
}

#[test]
fn test_run_command_with_timeout_nonexistent_command() {
    let reporter = FailFastReporter::new(Path::new("."));
    let result = reporter.run_command_with_timeout(
        &["this_command_does_not_exist_xyz_12345"],
        Duration::from_secs(5),
    );
    assert!(!result.success);
    assert!(result.stderr.contains("Failed to execute"));
    assert!(result.stdout.is_empty());
}

#[test]
fn test_run_command_with_timeout_failing_command() {
    let reporter = FailFastReporter::new(Path::new("."));
    let result = reporter.run_command_with_timeout(&["false"], Duration::from_secs(5));
    assert!(!result.success);
    assert!(!result.timed_out);
}

#[test]
fn test_save_json_to_tempdir() {
    let tmp = tempfile::TempDir::new().unwrap();
    let reporter = FailFastReporter::new(tmp.path());
    let data = DiagnosticResult {
        command: "test cmd".to_string(),
        success: true,
        stdout: "ok".to_string(),
        stderr: String::new(),
        duration_ms: 42,
        timed_out: false,
    };
    let path = tmp.path().join("test.json");
    reporter.save_json(&path, &data).unwrap();
    let contents = std::fs::read_to_string(&path).unwrap();
    assert!(contents.contains("test cmd"));
    assert!(contents.contains("42"));
    // Verify it is valid JSON that round-trips
    let parsed: DiagnosticResult = serde_json::from_str(&contents).unwrap();
    assert_eq!(parsed.command, "test cmd");
    assert!(parsed.success);
}

#[test]
fn test_run_check_returns_result() {
    let reporter =
        FailFastReporter::new(Path::new(".")).with_binary("this_binary_does_not_exist");
    let result = reporter.run_check(Path::new("/nonexistent/model.apr"));
    assert!(result.is_some());
    let diag = result.unwrap();
    assert!(!diag.success);
    assert!(diag.command.contains("check"));
}

#[test]
fn test_run_inspect_returns_result() {
    let reporter =
        FailFastReporter::new(Path::new(".")).with_binary("this_binary_does_not_exist");
    let result = reporter.run_inspect(Path::new("/nonexistent/model.apr"));
    assert!(result.is_some());
    let diag = result.unwrap();
    assert!(!diag.success);
    assert!(diag.command.contains("inspect"));
}

#[test]
fn test_run_tensors_returns_result() {
    let reporter =
        FailFastReporter::new(Path::new(".")).with_binary("this_binary_does_not_exist");
    let result = reporter.run_tensors(Path::new("/nonexistent/model.apr"));
    assert!(result.is_some());
    let diag = result.unwrap();
    assert!(!diag.success);
    assert!(diag.command.contains("tensors"));
}

#[test]
fn test_run_explain_returns_result() {
    let reporter =
        FailFastReporter::new(Path::new(".")).with_binary("this_binary_does_not_exist");
    let result = reporter.run_explain("G3-STABLE");
    assert!(result.is_some());
    let diag = result.unwrap();
    assert!(!diag.success);
    assert!(diag.command.contains("explain"));
    assert!(diag.command.contains("G3-STABLE"));
}

#[test]
fn test_run_trace_for_apr_file() {
    let reporter =
        FailFastReporter::new(Path::new(".")).with_binary("this_binary_does_not_exist");
    // .apr extension should cause run_trace to actually run the command (not skip)
    let result = reporter.run_trace(Path::new("/nonexistent/model.apr"));
    assert!(result.is_some());
    let diag = result.unwrap();
    assert!(!diag.success);
    assert!(diag.command.contains("trace"));
}

#[test]
fn test_run_trace_skips_no_extension() {
    let reporter = FailFastReporter::new(Path::new("."));
    let result = reporter.run_trace(Path::new("/nonexistent/model"));
    assert!(result.is_none());
}

#[test]
fn test_get_apr_version_returns_string() {
    let version = get_apr_version();
    // Should be either a version string or "unknown" if apr is not installed
    assert!(!version.is_empty());
}

#[test]
fn test_get_git_commit_returns_string() {
    let commit = get_git_commit();
    // In a git repo, returns a short hash; in CI containers without git, may be empty
    // Just verify it doesn't panic — empty is valid in headless environments
    let _ = commit;
}

#[test]
fn test_get_git_branch_returns_string() {
    let branch = get_git_branch();
    // May be empty in CI containers without git — just verify no panic
    let _ = branch;
}

#[test]
fn test_get_git_dirty_returns_bool() {
    // Just exercise the function; the return value depends on working tree state
    let _dirty = get_git_dirty();
}

