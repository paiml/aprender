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
