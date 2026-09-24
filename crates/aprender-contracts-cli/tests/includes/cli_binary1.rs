    #[test]
    fn pv_validate_softmax() {
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Contract is valid"));
    }

    #[test]
    fn pv_validate_invalid_file() {
        let dir = tempfile::tempdir().expect("temp dir is creatable");
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, "{{invalid").expect("fixture file is writable");
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg(&path)
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
    }

    /// #3347: the single-file form of `--strict-test-binding` reported EVERY
    /// cited test as missing, because the gate roots its source index at the
    /// contract's parent (`contracts/`, which has no `crates/`). Measured on
    /// `pv-artifact-kinds-v1.yaml` — a contract whose 8 refs all resolve in
    /// the directory form — it reported 8 refs, 0 existing, 8 missing. A
    /// control that fails identically to a broken contract cannot
    /// discriminate, so the invocation is refused.
    #[test]
    fn pv_lint_single_file_refuses_strict_test_binding() {
        let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
        let output = Command::new(pv_bin())
            .current_dir(scratch.path())
            .arg("lint")
            .arg(contract_path("pv-artifact-kinds-v1.yaml"))
            .arg("--strict-test-binding")
            .output()
            .expect("failed to run pv");
        assert!(
            !output.status.success(),
            "the single-file form must be refused, not reported over"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--strict-test-binding"),
            "the refusal must name the flag it refuses: {stderr}"
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !stdout.contains("Result: PASS"),
            "a PASS was printed over a gate that could not resolve one ref: {stdout}"
        );
        assert!(
            !stdout.contains("Dangling test reference"),
            "the false negatives were printed anyway: {stdout}"
        );
    }

    /// Discrimination for the row above: the refusal is specific to the
    /// single-FILE form, so a directory is still linted with the same flag.
    #[test]
    fn pv_lint_directory_form_accepts_strict_test_binding() {
        let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
        let dir = scratch.path().join("contracts");
        std::fs::create_dir_all(&dir).expect("fixture dir is creatable");
        std::fs::copy(
            contract_path("pv-artifact-kinds-v1.yaml"),
            dir.join("pv-artifact-kinds-v1.yaml"),
        )
        .expect("fixture contract is copyable");

        let output = Command::new(pv_bin())
            .current_dir(scratch.path())
            .arg("lint")
            .arg(&dir)
            .arg("--strict-test-binding")
            .output()
            .expect("failed to run pv");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("cannot run over a single contract file"),
            "the directory form must not be refused: {stderr}"
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("strict-test-binding") || stdout.contains("Result:"),
            "the directory form ran the gate and reported: {stdout}"
        );
    }

    #[test]
    fn pv_scaffold_softmax() {
        let output = Command::new(pv_bin())
            .arg("scaffold")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Trait Definition"));
    }

    #[test]
    fn pv_kani_softmax() {
        let output = Command::new(pv_bin())
            .arg("kani")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("kani"));
    }

    #[test]
    fn pv_probar_softmax() {
        let output = Command::new(pv_bin())
            .arg("probar")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("#[test]"));
    }

    #[test]
    fn pv_probar_with_binding() {
        let output = Command::new(pv_bin())
            .arg("probar")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--binding")
            .arg(binding_path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("proptest!"));
    }

    #[test]
    fn pv_status_softmax() {
        let output = Command::new(pv_bin())
            .arg("status")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Contract:"));
        assert!(stdout.contains("QA gate:"));
    }

    #[test]
    fn pv_audit_softmax() {
        let output = Command::new(pv_bin())
            .arg("audit")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Traceability Audit"));
    }

    #[test]
    fn pv_audit_with_binding() {
        let output = Command::new(pv_bin())
            .arg("audit")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--binding")
            .arg(binding_path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Binding Audit"));
    }

    #[test]
    fn pv_validate_nonexistent() {
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg("/nonexistent.yaml")
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
    }

    #[test]
    fn pv_status_matmul() {
        let output = Command::new(pv_bin())
            .arg("status")
            .arg(contract_path("matmul-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
    }

    #[test]
    fn pv_validate_with_warnings() {
        // Use a contract that has no qa_gate (generates SCHEMA-013 warning)
        let dir = tempfile::tempdir().expect("temp dir is creatable");
        let path = dir.path().join("warn.yaml");
        std::fs::write(
            &path,
            r#"
metadata:
  version: "1.0.0"
  description: "Warn test"
  registry: true
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .expect("the test fixture is well-formed");
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg(&path)
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("0 error(s)"));
        assert!(stdout.contains("warning(s)"));
    }

    #[test]
    fn pv_validate_with_errors() {
        let dir = tempfile::tempdir().expect("temp dir is creatable");
        let path = dir.path().join("errors.yaml");
        std::fs::write(
            &path,
            r#"
metadata:
  version: "1.0.0"
  description: "Error test"
  references: []
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .expect("the test fixture is well-formed");
        let output = Command::new(pv_bin())
            .arg("validate")
            .arg(&path)
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("error"));
    }

    #[test]
    fn pv_status_no_qa_gate() {
        let dir = tempfile::tempdir().expect("temp dir is creatable");
        let path = dir.path().join("no_qa.yaml");
        std::fs::write(
            &path,
            r#"
metadata:
  version: "1.0.0"
  description: "No QA"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
        )
        .expect("the test fixture is well-formed");
        let output = Command::new(pv_bin())
            .arg("status")
            .arg(&path)
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("QA gate: not defined"));
    }

    #[test]
    fn pv_audit_with_errors() {
        let dir = tempfile::tempdir().expect("temp dir is creatable");
        let contract = dir.path().join("audit_err.yaml");
        std::fs::write(
            &contract,
            r#"
metadata:
  version: "1.0.0"
  description: "Audit error"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
kani_harnesses:
  - id: KANI-001
    obligation: ""
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "works"
    if_fails: "broken"
"#,
        )
        .expect("the test fixture is well-formed");
        let output = Command::new(pv_bin())
            .arg("audit")
            .arg(&contract)
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
    }

    #[test]
    fn pv_scaffold_activation() {
        let output = Command::new(pv_bin())
            .arg("scaffold")
            .arg(contract_path("activation-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Contract Tests"));
    }

    #[test]
    fn pv_probar_rope() {
        let output = Command::new(pv_bin())
            .arg("probar")
            .arg(contract_path("rope-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
    }
