#[test]
fn test_get_rustc_version_returns_string() {
    let version = get_rustc_version();
    // rustc should be installed in any Rust dev environment
    assert!(!version.is_empty());
    assert_ne!(version, "unknown");
}

#[test]
fn test_generate_report_creates_files() {
    let tmp = tempfile::TempDir::new().unwrap();
    let reporter = FailFastReporter::new(tmp.path()).with_binary("this_binary_does_not_exist");
    let evidence = test_evidence();

    let report = reporter
        .generate_report(
            &evidence,
            Path::new("/nonexistent/model.apr"),
            Some("test.yaml"),
        )
        .unwrap();

    let report_dir = tmp.path().join("fail-fast-report");
    assert!(report_dir.exists());
    // Check that expected files were created
    assert!(report_dir.join("diagnostics.json").exists());
    assert!(report_dir.join("environment.json").exists());
    assert!(report_dir.join("summary.md").exists());
    assert!(report_dir.join("stderr.log").exists());
    assert!(report_dir.join("check.json").exists());
    assert!(report_dir.join("inspect.json").exists());
    assert!(report_dir.join("tensors.json").exists());

    // Verify the report structure
    assert_eq!(report.version, "1.0.0");
    assert_eq!(report.failure.gate_id, "G3-STABLE");
    assert_eq!(report.reproduction.playbook, Some("test.yaml".to_string()));
    assert!(report.reproduction.command.contains("test.yaml"));
    assert!(report.diagnostics.check.is_some());
    assert!(report.diagnostics.inspect.is_some());
    assert!(report.diagnostics.tensors.is_some());

    // Verify the summary markdown was written
    let summary = std::fs::read_to_string(report_dir.join("summary.md")).unwrap();
    assert!(summary.contains("# Fail-Fast Report: G3-STABLE"));
}

#[test]
fn test_generate_report_without_playbook() {
    let tmp = tempfile::TempDir::new().unwrap();
    let reporter = FailFastReporter::new(tmp.path()).with_binary("this_binary_does_not_exist");
    let evidence = test_evidence();

    let report = reporter
        .generate_report(&evidence, Path::new("/nonexistent/model.apr"), None)
        .unwrap();

    // When playbook is None, it should use "playbook.yaml" as default
    assert!(report.reproduction.command.contains("playbook.yaml"));
    assert!(report.reproduction.playbook.is_none());
}

#[test]
fn test_generate_report_no_stderr_skips_log() {
    let tmp = tempfile::TempDir::new().unwrap();
    let reporter = FailFastReporter::new(tmp.path()).with_binary("this_binary_does_not_exist");
    let mut evidence = test_evidence();
    evidence.stderr = None;

    reporter
        .generate_report(&evidence, Path::new("/nonexistent/model.apr"), None)
        .unwrap();

    let report_dir = tmp.path().join("fail-fast-report");
    // stderr.log should NOT be created when evidence.stderr is None
    assert!(!report_dir.join("stderr.log").exists());
}

#[test]
fn test_generate_markdown_no_stderr() {
    let reporter = FailFastReporter::new(Path::new("output"));
    let mut evidence = test_evidence();
    evidence.stderr = None;
    let details = FailureDetails::from(&evidence);

    let report = FailFastReport {
        version: "1.0.0".to_string(),
        timestamp: "2024-02-04T18:00:00Z".to_string(),
        failure: details,
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

    // Should NOT contain stderr section when stderr is None
    assert!(!md.contains("## Stderr Capture"));
}

#[test]
fn test_generate_markdown_empty_stderr() {
    let reporter = FailFastReporter::new(Path::new("output"));
    let mut evidence = test_evidence();
    evidence.stderr = Some(String::new());
    let details = FailureDetails::from(&evidence);

    let report = FailFastReport {
        version: "1.0.0".to_string(),
        timestamp: "2024-02-04T18:00:00Z".to_string(),
        failure: details,
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

    // Empty stderr should also NOT produce the stderr section (line 505: !stderr.is_empty())
    assert!(!md.contains("## Stderr Capture"));
}

#[test]
fn test_failure_details_no_exit_code() {
    let mut evidence = test_evidence();
    evidence.exit_code = None;
    let details = FailureDetails::from(&evidence);
    assert!(details.exit_code.is_none());

    // Also verify that the markdown omits the Exit Code row
    let reporter = FailFastReporter::new(Path::new("output"));
    let report = FailFastReport {
        version: "1.0.0".to_string(),
        timestamp: "2024-02-04T18:00:00Z".to_string(),
        failure: details,
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
    assert!(!md.contains("Exit Code"));
}

#[test]
fn test_generate_markdown_explain_empty_stdout() {
    // When explain has empty stdout, the "Error Analysis" section should be omitted
    let reporter = FailFastReporter::new(Path::new("output"));
    let evidence = test_evidence();

    let explain_result = DiagnosticResult {
        command: "apr explain G3-STABLE".to_string(),
        success: true,
        stdout: String::new(), // empty
        stderr: String::new(),
        duration_ms: 10,
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
            check: None,
            inspect: None,
            trace: None,
            tensors: None,
            explain: Some(explain_result),
        },
        reproduction: ReproductionInfo {
            command: "apr-qa run playbook.yaml --fail-fast".to_string(),
            model_path: "/path/to/model.apr".to_string(),
            playbook: Some("playbook.yaml".to_string()),
        },
    };

    let md = reporter.generate_markdown(&report);
    // Line 496: if !explain.stdout.is_empty() - with empty stdout, this section is skipped
    assert!(!md.contains("## Error Analysis"));
}

#[test]
fn test_environment_context_collect_has_all_fields() {
    let ctx = EnvironmentContext::collect();
    // apr_cli_version exercises get_apr_version()
    // git_commit exercises get_git_commit()
    // git_branch exercises get_git_branch()
    // git_dirty exercises get_git_dirty()
    // rustc_version exercises get_rustc_version()
    assert!(!ctx.apr_cli_version.is_empty());
    // git_commit and git_branch may be empty in CI containers without git
    let _ = ctx.git_commit;
    let _ = ctx.git_branch;
    assert!(!ctx.rustc_version.is_empty());
    // git_dirty is just a bool, any value is fine
}

#[test]
fn test_run_command_with_timeout_captures_stderr() {
    let reporter = FailFastReporter::new(Path::new("."));
    // Use sh -c to write to stderr
    let result = reporter.run_command_with_timeout(
        &["sh", "-c", "echo error_output >&2; exit 1"],
        Duration::from_secs(5),
    );
    assert!(!result.success);
    assert!(result.stderr.contains("error_output"));
}

#[test]
fn test_generate_report_trace_json_created_for_apr() {
    let tmp = tempfile::TempDir::new().unwrap();
    let reporter = FailFastReporter::new(tmp.path()).with_binary("this_binary_does_not_exist");
    let evidence = test_evidence();

    // model path ends in .apr, so trace should run and trace.json should be saved
    reporter
        .generate_report(&evidence, Path::new("/nonexistent/model.apr"), None)
        .unwrap();

    let report_dir = tmp.path().join("fail-fast-report");
    // trace.json should exist because the model path ends in .apr
    assert!(report_dir.join("trace.json").exists());
}

#[test]
fn test_generate_report_no_trace_json_for_non_apr() {
    let tmp = tempfile::TempDir::new().unwrap();
    let reporter = FailFastReporter::new(tmp.path()).with_binary("this_binary_does_not_exist");
    let mut evidence = test_evidence();
    // Change format but we really just need the path to be non-.apr
    evidence.scenario = QaScenario::new(
        ModelId::new("Qwen", "Qwen2.5-Coder-0.5B-Instruct"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "What is 2+2?".to_string(),
        0,
    );

    reporter
        .generate_report(&evidence, Path::new("/nonexistent/model.safetensors"), None)
        .unwrap();

    let report_dir = tmp.path().join("fail-fast-report");
    // trace.json should NOT exist because the model path is .safetensors
    assert!(!report_dir.join("trace.json").exists());
}
