//! Book contract integration tests
//!
//! Verifies that all 27 chapter examples exist, all 27 contract YAMLs exist,
//! and namespace discipline is maintained (zero legacy names).

use std::path::Path;

const CHAPTERS: &[(&str, &str)] = &[
    ("ch01_hello_apr", "apr-book-ch01-v1.yaml"),
    ("ch02_tensors", "apr-book-ch02-v1.yaml"),
    ("ch03_apr_format", "apr-book-ch03-v1.yaml"),
    ("ch04_supervised", "apr-book-ch04-v1.yaml"),
    ("ch05_unsupervised", "apr-book-ch05-v1.yaml"),
    ("ch06_ensembles", "apr-book-ch06-v1.yaml"),
    ("ch07_model_selection", "apr-book-ch07-v1.yaml"),
    ("ch08_transformer", "apr-book-ch08-v1.yaml"),
    ("ch09_inference", "apr-book-ch09-v1.yaml"),
    ("ch10_training", "apr-book-ch10-v1.yaml"),
    ("ch11_formats", "apr-book-ch11-v1.yaml"),
    ("ch12_serving", "apr-book-ch12-v1.yaml"),
    ("ch13_profiling", "apr-book-ch13-v1.yaml"),
    ("ch14_contracts", "apr-book-ch14-v1.yaml"),
    ("ch15_orchestrate", "apr-book-ch15-v1.yaml"),
    ("ch16_timeseries", "apr-book-ch16-v1.yaml"),
    ("ch17_bayesian", "apr-book-ch17-v1.yaml"),
    ("ch18_graphs", "apr-book-ch18-v1.yaml"),
    ("ch19_text", "apr-book-ch19-v1.yaml"),
    ("ch20_rag", "apr-book-ch20-v1.yaml"),
    ("ch21_vs_candle", "apr-book-ch21-v1.yaml"),
    ("ch22_vs_llamacpp", "apr-book-ch22-v1.yaml"),
    ("ch23_training_bench", "apr-book-ch23-v1.yaml"),
    ("ch24_switch_pytorch", "apr-book-ch24-v1.yaml"),
    ("ch25_switch_ollama", "apr-book-ch25-v1.yaml"),
    ("ch26_switch_ndarray", "apr-book-ch26-v1.yaml"),
    ("ch27_switch_unsloth", "apr-book-ch27-v1.yaml"),
];

/// Workspace root (two levels up from aprender-core)
fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .parent()
        .expect("workspace root")
}

#[test]
fn all_chapter_examples_exist() {
    let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let mut missing = Vec::new();
    for (example, _) in CHAPTERS {
        let path = examples_dir.join(format!("{example}.rs"));
        if !path.exists() {
            missing.push(example.to_string());
        }
    }
    assert!(
        missing.is_empty(),
        "Missing chapter examples: {:?}",
        missing
    );
}

#[test]
fn all_chapter_contracts_exist() {
    let contracts_dir = workspace_root().join("contracts");
    let mut missing = Vec::new();
    for (_, contract) in CHAPTERS {
        let path = contracts_dir.join(contract);
        if !path.exists() {
            missing.push(contract.to_string());
        }
    }
    assert!(
        missing.is_empty(),
        "Missing chapter contracts: {:?}",
        missing
    );
}

#[test]
fn chapter_count_is_27() {
    assert_eq!(CHAPTERS.len(), 27, "Book must have exactly 27 chapters");
}

#[test]
fn namespace_discipline_in_examples() {
    let examples_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
    let legacy_names = [
        "trueno", "realizar", "entrenar", "batuta", "presentar", "renacer",
    ];
    let mut violations = Vec::new();

    for (example, _) in CHAPTERS {
        let path = examples_dir.join(format!("{example}.rs"));
        if let Ok(content) = std::fs::read_to_string(&path) {
            for name in &legacy_names {
                // Skip comments that reference the old→new mapping in ch14
                if content.contains(&format!("\"{name}\""))
                    && example == &"ch14_contracts"
                {
                    continue;
                }
                // Check for `use <legacy>::` or standalone usage
                if content.contains(&format!("use {name}::"))
                    || content.contains(&format!("{name} ="))
                    || content.contains(&format!("cargo install {name}"))
                {
                    violations.push(format!("{example}: contains legacy name '{name}'"));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "Namespace violations: {:?}",
        violations
    );
}

#[test]
fn contracts_have_falsification_conditions() {
    let contracts_dir = workspace_root().join("contracts");
    let mut missing_falsification = Vec::new();

    for (_, contract) in CHAPTERS {
        let path = contracts_dir.join(contract);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if !content.contains("falsification:") {
                missing_falsification.push(contract.to_string());
            }
            // Each contract must have at least 5 falsification conditions
            let condition_count = content.matches("condition:").count();
            if condition_count < 5 {
                missing_falsification.push(format!(
                    "{contract}: only {condition_count} conditions (need 5)"
                ));
            }
        }
    }
    assert!(
        missing_falsification.is_empty(),
        "Contracts missing falsification: {:?}",
        missing_falsification
    );
}

#[test]
fn spec_file_exists() {
    // Book spec may be at either location (moved during monorepo consolidation)
    let spec_paths = [
        workspace_root().join("docs/specifications/apr-book-spec.md"),
        workspace_root().join("docs/specifications/aprender-monorepo-consolidation.md"),
    ];
    let spec = spec_paths
        .iter()
        .find(|p| p.exists())
        .expect("Book spec must exist at one of the known paths");
    let content = std::fs::read_to_string(spec).expect("read spec");
    assert!(
        content.contains("falsification") || content.contains("Falsification"),
        "Spec must reference falsification"
    );
}
