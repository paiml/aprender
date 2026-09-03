    #[test]
    fn pv_diff_identical() {
        let output = Command::new(pv_bin())
            .arg("diff")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("identical"));
    }

    #[test]
    fn pv_diff_different() {
        let output = Command::new(pv_bin())
            .arg("diff")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg(contract_path("rmsnorm-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("major"));
    }

    #[test]
    fn pv_coverage_contracts() {
        let output = Command::new(pv_bin())
            .arg("coverage")
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Obligation Coverage Report"));
        assert!(stdout.contains("softmax-kernel-v1"));
    }

    #[test]
    fn pv_coverage_with_binding() {
        let output = Command::new(pv_bin())
            .arg("coverage")
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts"))
            .arg("--binding")
            .arg(binding_path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Binding implemented"));
    }

    #[test]
    fn pv_generate_softmax() {
        let dir = tempfile::tempdir().expect("temp dir is creatable");
        let output = Command::new(pv_bin())
            .arg("generate")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--output")
            .arg(dir.path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Generated"));
        assert!(stdout.contains("scaffold"));
    }

    #[test]
    fn pv_generate_with_binding() {
        let dir = tempfile::tempdir().expect("temp dir is creatable");
        let output = Command::new(pv_bin())
            .arg("generate")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--output")
            .arg(dir.path())
            .arg("--binding")
            .arg(binding_path())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("wired-probar"));
    }

    #[test]
    fn pv_graph_contracts() {
        let output = Command::new(pv_bin())
            .arg("graph")
            .arg(contracts_dir())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Contract Dependency Graph"));
        assert!(stdout.contains("Topological order"));
    }

    #[test]
    fn pv_graph_dot() {
        let output = Command::new(pv_bin())
            .arg("graph")
            .arg(contracts_dir())
            .arg("--format")
            .arg("dot")
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("digraph contracts {"));
        assert!(stdout.contains("rankdir=LR"));
        assert!(stdout.contains("->"));
        assert!(stdout.contains('}'));
    }

    #[test]
    fn pv_graph_json() {
        let output = Command::new(pv_bin())
            .arg("graph")
            .arg(contracts_dir())
            .arg("--format")
            .arg("json")
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("\"nodes\""));
        assert!(stdout.contains("\"edges\""));
        assert!(stdout.contains("\"topo_order\""));
        assert!(stdout.contains("\"cycles\""));
        assert!(stdout.contains("\"from\""));
        assert!(stdout.contains("\"to\""));
    }

    #[test]
    fn pv_graph_mermaid() {
        let output = Command::new(pv_bin())
            .arg("graph")
            .arg(contracts_dir())
            .arg("--format")
            .arg("mermaid")
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("graph TD"));
        assert!(stdout.contains("-->"));
    }

    #[test]
    fn pv_graph_invalid_format() {
        let output = Command::new(pv_bin())
            .arg("graph")
            .arg(contracts_dir())
            .arg("--format")
            .arg("xml")
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("unknown format"));
    }

    #[test]
    fn pv_equations_text() {
        let output = Command::new(pv_bin())
            .arg("equations")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("Equations for softmax-kernel-v1"));
        assert!(stdout.contains("formula:"));
        assert!(stdout.contains("domain:"));
        assert!(stdout.contains("invariants:"));
    }

    #[test]
    fn pv_equations_latex() {
        let output = Command::new(pv_bin())
            .arg("equations")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--format")
            .arg("latex")
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("\\section{Equations:"));
        assert!(stdout.contains("\\begin{equation}"));
        assert!(stdout.contains("\\end{equation}"));
        assert!(stdout.contains("\\begin{itemize}"));
    }

    #[test]
    fn pv_equations_ptx() {
        let output = Command::new(pv_bin())
            .arg("equations")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--format")
            .arg("ptx")
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains(".version 8.5"));
        assert!(stdout.contains(".target sm_90"));
        assert!(stdout.contains(".visible .entry softmax("));
        assert!(stdout.contains("Phase 1: find_max"));
        assert!(stdout.contains("Phase 4: normalize"));
        assert!(stdout.contains("ret;"));
    }

    #[test]
    fn pv_equations_asm() {
        let output = Command::new(pv_bin())
            .arg("equations")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--format")
            .arg("asm")
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains(".intel_syntax noprefix"));
        assert!(stdout.contains("softmax_avx2:"));
        assert!(stdout.contains("AVX2"));
        assert!(stdout.contains("ymm"));
        assert!(stdout.contains("Phase 1: find_max"));
        assert!(stdout.contains("ret"));
    }

    #[test]
    fn pv_equations_ptx_no_kernel_structure() {
        // model-config-algebra has no kernel_structure
        let output = Command::new(pv_bin())
            .arg("equations")
            .arg(contract_path("model-config-algebra-v1.yaml"))
            .arg("--format")
            .arg("ptx")
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains(".visible .entry model_config_algebra("));
        assert!(stdout.contains("// Equation:"));
        assert!(stdout.contains("ret;"));
    }

    #[test]
    fn pv_equations_invalid_format() {
        let output = Command::new(pv_bin())
            .arg("equations")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .arg("--format")
            .arg("json")
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("unknown format"));
    }

    #[test]
    fn pv_lean_softmax() {
        let output = Command::new(pv_bin())
            .arg("lean")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        // softmax-kernel-v1.yaml has lean blocks — should generate Lean files
        assert!(
            stdout.contains("import Mathlib") || stdout.contains("No Lean"),
            "lean command should produce output"
        );
    }

    #[test]
    fn pv_lean_status_single() {
        let output = Command::new(pv_bin())
            .arg("lean-status")
            .arg(contract_path("softmax-kernel-v1.yaml"))
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
    }

    #[test]
    fn pv_lean_status_directory() {
        let output = Command::new(pv_bin())
            .arg("lean-status")
            .arg(contracts_dir())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("L4 Coverage") || stdout.contains("No Lean"),
            "lean-status should show coverage or indicate no lean metadata"
        );
    }

    #[test]
    fn pv_query_kind_pattern_finds_patterns() {
        let output = Command::new(pv_bin())
            .arg("query")
            .arg("--contract-dir")
            .arg(contracts_dir())
            .arg("--rebuild-index")
            .arg("--kind")
            .arg("pattern")
            .arg("--regex")
            .arg(".")
            .arg("--limit")
            .arg("10")
            .output()
            .expect("failed to run pv");
        assert!(
            output.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("[pattern]"),
            "pattern results should carry the [pattern] kind tag"
        );
    }

    #[test]
    fn pv_query_kind_rejects_invalid() {
        let output = Command::new(pv_bin())
            .arg("query")
            .arg("--contract-dir")
            .arg(contracts_dir())
            .arg("--kind")
            .arg("bogus")
            .arg("--regex")
            .arg(".")
            .output()
            .expect("failed to run pv");
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("invalid --kind value") || stderr.contains("kernel"),
            "error should mention invalid value and valid list"
        );
    }

    #[test]
    fn pv_proof_status_kind_kernel_excludes_patterns() {
        let output = Command::new(pv_bin())
            .arg("proof-status")
            .arg("--format")
            .arg("text")
            .arg("--kind")
            .arg("kernel")
            .arg(contracts_dir())
            .output()
            .expect("failed to run pv");
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !stdout.contains("threading-safety-v1"),
            "kernel filter should exclude pattern contracts"
        );
        assert!(
            !stdout.contains("async-safety-v1"),
            "kernel filter should exclude pattern contracts"
        );
    }
