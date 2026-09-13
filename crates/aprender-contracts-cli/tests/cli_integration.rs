//! SIDE EFFECT, DISCLOSED: this target DIRTIES TRACKED GENERATED CACHES.
//!
//! Running it writes `.pv/contracts.idx`, `.pv/contracts.idx.mtime` and
//! `.pv/lint-previous.json` — all three are tracked in git. Isolated by
//! bisection (`git checkout .pv/`, run this target, `git status`), so the
//! attribution is measured rather than inferred.
//!
//! This matters because the PR that wires this target into ci.yml's integration
//! step makes CI run it for the first time: a CI working tree that goes dirty
//! can red an unrelated guard, or produce a diff nobody wrote. Restore with
//! `git checkout .pv/` — never hand-edit those files.
//!
//! NOT CONTAINED, deliberately, and here is the reason rather than a shrug:
//! `pv_surface_gate.rs` solves the same problem with `pv_in(dir, ..)`, which
//! spawns the `pv` BINARY with `.current_dir(scratch)`. That fix does not
//! transfer here, because this file does not spawn a subprocess at all — it
//! calls `parse_contract` / `validate_contract` IN-PROCESS, so the cache path
//! follows the test process's own cwd. Changing that cwd is process-wide and
//! would race every other test in the binary, which cargo runs in threads.
//! Containing it properly means giving the library an explicit cache-root
//! parameter; that is a real change and does not belong in this PR.
//!
//! Note also that `pv_bin()` below is currently UNUSED — a leftover from when
//! this file did shell out. Left in place rather than deleted so the deletion
//! is not confused with the disclosure above.

use std::path::Path;
use std::process::Command;

/// Helper to get the path to a contract fixture.
fn contract_path(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts")
        .join(name)
}

/// Helper to get the contracts directory path.
fn contracts_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

/// Helper to get the binding registry path.
fn binding_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/aprender/binding.yaml")
}

/// Helper to get the pv binary path.
fn pv_bin() -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(env!("CARGO_BIN_EXE_pv"));
    // Fallback for test environments
    if !path.exists() {
        path = std::path::PathBuf::from("target/debug/pv");
    }
    path
}

// ================================================================
// validate command
// ================================================================

mod validate {
    use provable_contracts::error::Severity;
    use provable_contracts::schema::{parse_contract, validate_contract};

    #[test]
    fn valid_softmax_contract() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_rmsnorm_contract() {
        let path = super::contract_path("rmsnorm-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_rope_contract() {
        let path = super::contract_path("rope-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_activation_contract() {
        let path = super::contract_path("activation-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_attention_contract() {
        let path = super::contract_path("attention-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_matmul_contract() {
        let path = super::contract_path("matmul-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn valid_flash_attention_contract() {
        let path = super::contract_path("flash-attention-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let violations = validate_contract(&contract);
        let errors: Vec<_> = violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn invalid_yaml_fails() {
        let dir = tempfile::tempdir().expect("temp dir is creatable");
        let path = dir.path().join("bad.yaml");
        std::fs::write(&path, "not: [valid: {{").expect("fixture file is writable");
        let result = parse_contract(&path);
        assert!(result.is_err());
    }

    #[test]
    fn nonexistent_file_fails() {
        let result = parse_contract(std::path::Path::new("/nonexistent.yaml"));
        assert!(result.is_err());
    }
}

// ================================================================
// scaffold command
// ================================================================

mod scaffold {
    use provable_contracts::scaffold::{generate_contract_tests, generate_trait};
    use provable_contracts::schema::parse_contract;

    #[test]
    fn scaffold_softmax() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let trait_code = generate_trait(&contract);
        let test_code = generate_contract_tests(&contract);
        assert!(trait_code.contains("trait"));
        assert!(test_code.contains("#[test]"));
    }

    #[test]
    fn scaffold_activation() {
        let path = super::contract_path("activation-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let trait_code = generate_trait(&contract);
        assert!(trait_code.contains("fn"));
    }
}

// ================================================================
// kani command
// ================================================================

mod kani {
    use provable_contracts::kani_gen::generate_kani_harnesses;
    use provable_contracts::schema::parse_contract;

    #[test]
    fn kani_softmax() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let code = generate_kani_harnesses(&contract);
        assert!(code.contains("kani"));
    }

    #[test]
    fn kani_matmul() {
        let path = super::contract_path("matmul-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let code = generate_kani_harnesses(&contract);
        assert!(!code.is_empty());
    }
}

// ================================================================
// probar command
// ================================================================

mod probar {
    use provable_contracts::binding::parse_binding;
    use provable_contracts::probar_gen::{generate_probar_tests, generate_wired_probar_tests};
    use provable_contracts::schema::parse_contract;

    #[test]
    fn probar_softmax_plain() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let code = generate_probar_tests(&contract);
        assert!(code.contains("probar_tests"));
        assert!(code.contains("#[test]"));
    }

    #[test]
    fn probar_softmax_wired() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let bp = super::binding_path();
        let contract = parse_contract(&path).expect("fixture contract parses");
        let binding = parse_binding(&bp).expect("fixture binding registry parses");
        let code = generate_wired_probar_tests(&contract, "softmax-kernel-v1.yaml", &binding);
        assert!(code.contains("proptest!"));
        assert!(code.contains("softmax"));
    }

    #[test]
    fn probar_activation_wired() {
        let path = super::contract_path("activation-kernel-v1.yaml");
        let bp = super::binding_path();
        let contract = parse_contract(&path).expect("fixture contract parses");
        let binding = parse_binding(&bp).expect("fixture binding registry parses");
        let code = generate_wired_probar_tests(&contract, "activation-kernel-v1.yaml", &binding);
        assert!(code.contains("CONTRACT: activation"));
    }

    #[test]
    fn probar_rmsnorm_plain() {
        let path = super::contract_path("rmsnorm-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let code = generate_probar_tests(&contract);
        assert!(code.contains("#[test]"));
    }
}

// ================================================================
// status command
// ================================================================

mod status {
    use provable_contracts::schema::parse_contract;

    #[test]
    fn status_softmax_has_qa_gate() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        assert!(contract.qa_gate.is_some());
        assert_eq!(contract.equations.len(), 1);
        assert_eq!(contract.proof_obligations.len(), 9);
        assert_eq!(contract.falsification_tests.len(), 9);
        assert_eq!(contract.kani_harnesses.len(), 12);
    }

    #[test]
    fn status_matmul_has_no_qa_gate() {
        let path = super::contract_path("matmul-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        // matmul may or may not have qa_gate, just check it parses
        assert!(!contract.equations.is_empty());
    }

    #[test]
    fn status_all_contracts_parse() {
        let contracts = [
            "softmax-kernel-v1.yaml",
            "rmsnorm-kernel-v1.yaml",
            "rope-kernel-v1.yaml",
            "activation-kernel-v1.yaml",
            "attention-kernel-v1.yaml",
            "matmul-kernel-v1.yaml",
            "flash-attention-v1.yaml",
        ];
        for name in contracts {
            let path = super::contract_path(name);
            let contract = parse_contract(&path).expect("fixture contract parses");
            assert!(
                !contract.metadata.description.is_empty(),
                "{name} has empty description"
            );
        }
    }
}

// ================================================================
// audit command
// ================================================================

mod audit {
    use provable_contracts::audit::{audit_binding, audit_contract};
    use provable_contracts::binding::parse_binding;
    use provable_contracts::error::Severity;
    use provable_contracts::schema::parse_contract;

    #[test]
    fn audit_softmax_contract() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let report = audit_contract(&contract);
        assert_eq!(report.equations, 1);
        assert_eq!(report.obligations, 9);
        assert_eq!(report.falsification_tests, 9);
        assert_eq!(report.kani_harnesses, 12);
    }

    #[test]
    fn audit_softmax_with_binding() {
        let path = super::contract_path("softmax-kernel-v1.yaml");
        let bp = super::binding_path();
        let contract = parse_contract(&path).expect("fixture contract parses");
        let binding = parse_binding(&bp).expect("fixture binding registry parses");
        let report = audit_binding(&[("softmax-kernel-v1.yaml", &contract)], &binding);
        assert_eq!(report.total_equations, 1);
        assert_eq!(report.implemented, 1);
        assert!(report
            .violations
            .iter()
            .all(|v| v.severity != Severity::Error));
    }

    #[test]
    fn audit_activation_with_binding() {
        let path = super::contract_path("activation-kernel-v1.yaml");
        let bp = super::binding_path();
        let contract = parse_contract(&path).expect("fixture contract parses");
        let binding = parse_binding(&bp).expect("fixture binding registry parses");
        let report = audit_binding(&[("activation-kernel-v1.yaml", &contract)], &binding);
        // activation has 3+ equations, silu is not_implemented
        assert!(report.total_equations >= 2);
    }

    #[test]
    fn audit_flash_attention_no_binding() {
        let path = super::contract_path("flash-attention-v1.yaml");
        let contract = parse_contract(&path).expect("fixture contract parses");
        let report = audit_contract(&contract);
        assert!(report.equations > 0);
    }
}

// ================================================================
// Binary integration tests (exercise CLI main + commands)
// ================================================================

mod binary {
    use super::*;

    // Part 1: validate, scaffold, kani, probar, status, audit CLI tests
    include!("includes/cli_binary1.rs");

    // Part 2: diff, coverage, generate, graph, equations, lean CLI tests
    include!("includes/cli_binary2.rs");
}
