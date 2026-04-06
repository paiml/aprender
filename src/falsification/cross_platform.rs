//! Section 9: Cross-Platform & API Completeness (CP-01 to CP-05)
//!
//! Portability and API coverage verification.
//!
//! # TPS Principles
//!
//! - **Portability**: Multi-platform support
//! - **API completeness**: NumPy/sklearn coverage

use super::types::{CheckItem, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// Evaluate all Cross-Platform & API checks.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_linux_compatibility(project_path),
        check_macos_windows_compatibility(project_path),
        check_wasm_browser_compatibility(project_path),
        check_numpy_api_coverage(project_path),
        check_sklearn_coverage(project_path),
    ]
}

/// CP-01: Linux Distribution Compatibility
pub fn check_linux_compatibility(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "CP-01",
        "Linux Distribution Compatibility",
        "Stack runs on major Linux distributions",
    )
    .with_severity(Severity::Major)
    .with_tps("Portability");

    let has_linux_ci = check_ci_for_pattern(project_path, &["ubuntu", "linux"]);
    let has_glibc_docs = check_for_pattern(project_path, &["glibc", "musl", "linux"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Linux: ci={}, docs={}", has_linux_ci, has_glibc_docs),
        data: None,
        files: Vec::new(),
    });

    if has_linux_ci {
        item = item.pass();
    } else {
        item = item.partial("No Linux CI testing");
    }

    item.finish_timed(start)
}

/// CP-02: macOS/Windows Compatibility
pub fn check_macos_windows_compatibility(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("CP-02", "macOS/Windows Compatibility", "Stack runs on macOS and Windows")
            .with_severity(Severity::Major)
            .with_tps("Portability");

    let has_macos_ci = check_ci_for_pattern(project_path, &["macos", "darwin"]);
    let has_windows_ci = check_ci_for_pattern(project_path, &["windows"]);
    let has_cross_platform_code =
        check_for_pattern(project_path, &["cfg(target_os", "cfg!(windows)", "cfg!(macos)"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Cross-platform: macos_ci={}, windows_ci={}, code={}",
            has_macos_ci, has_windows_ci, has_cross_platform_code
        ),
        data: None,
        files: Vec::new(),
    });

    if has_macos_ci && has_windows_ci {
        item = item.pass();
    } else if has_macos_ci || has_windows_ci {
        item = item.partial("Partial cross-platform CI");
    } else if has_cross_platform_code {
        item = item.partial("Cross-platform code (no CI)");
    } else {
        item = item.partial("Linux-only testing");
    }

    item.finish_timed(start)
}

/// CP-03: WASM Browser Compatibility
pub fn check_wasm_browser_compatibility(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("CP-03", "WASM Browser Compatibility", "WASM build works in major browsers")
            .with_severity(Severity::Major)
            .with_tps("Edge deployment");

    let has_wasm_build = check_for_pattern(project_path, &["wasm32", "wasm-bindgen", "wasm-pack"]);
    let has_browser_tests = check_for_pattern(
        project_path,
        &["wasm-bindgen-test", "browser_test", "chrome", "firefox"],
    );

    // Check for WASM feature in Cargo.toml
    let cargo_toml = project_path.join("Cargo.toml");
    let has_wasm_feature = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| c.contains("wasm") || c.contains("wasm32"))
        .unwrap_or(false);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "WASM: build={}, tests={}, feature={}",
            has_wasm_build, has_browser_tests, has_wasm_feature
        ),
        data: None,
        files: Vec::new(),
    });

    if has_wasm_build && has_browser_tests {
        item = item.pass();
    } else if has_wasm_build || has_wasm_feature {
        item = item.partial("WASM support (verify browser testing)");
    } else {
        item = item.partial("No WASM browser support");
    }

    item.finish_timed(start)
}

/// CP-04: NumPy API Coverage
pub fn check_numpy_api_coverage(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("CP-04", "NumPy API Coverage", "Supports >90% of NumPy operations")
            .with_severity(Severity::Major)
            .with_tps("API completeness");

    // Check for array/tensor operations that mirror NumPy
    let numpy_ops = [
        "reshape",
        "transpose",
        "dot",
        "matmul",
        "sum",
        "mean",
        "std",
        "var",
        "min",
        "max",
        "argmin",
        "argmax",
        "zeros",
        "ones",
        "eye",
        "linspace",
        "concatenate",
        "stack",
        "split",
    ];

    let mut found_ops = 0;
    if let Ok(entries) = glob::glob(&format!("{}/src/**/*.rs", project_path.display())) {
        for entry in entries.flatten() {
            if let Ok(content) = std::fs::read_to_string(&entry) {
                for op in &numpy_ops {
                    if content.contains(op) {
                        found_ops += 1;
                    }
                }
            }
        }
    }

    let coverage = (found_ops as f64 / numpy_ops.len() as f64 * 100.0) as u32;

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("NumPy coverage: ~{}% ({}/{})", coverage, found_ops, numpy_ops.len()),
        data: None,
        files: Vec::new(),
    });

    let is_numeric = check_for_pattern(project_path, &["ndarray", "tensor", "Array"]);
    if !is_numeric || found_ops >= numpy_ops.len() * 80 / 100 {
        item = item.pass();
    } else if found_ops >= numpy_ops.len() / 2 {
        item = item.partial(format!("Partial NumPy coverage (~{}%)", coverage));
    } else {
        item = item.partial("Limited NumPy-like API coverage");
    }

    item.finish_timed(start)
}

/// CP-05: sklearn Estimator Coverage
pub fn check_sklearn_coverage(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "CP-05",
        "sklearn Estimator Coverage",
        "Supports >80% of sklearn estimators",
    )
    .with_severity(Severity::Major)
    .with_tps("API completeness");

    // Check for common sklearn estimator equivalents
    let sklearn_estimators = [
        "LinearRegression",
        "LogisticRegression",
        "Ridge",
        "Lasso",
        "RandomForest",
        "GradientBoosting",
        "DecisionTree",
        "KMeans",
        "DBSCAN",
        "PCA",
        "StandardScaler",
        "SVM",
        "KNeighbors",
        "NaiveBayes",
    ];

    let found_estimators = sklearn_estimators
        .iter()
        .filter(|est| {
            super::helpers::source_contains_pattern(project_path, &[est])
                || super::helpers::files_contain_pattern_ci(project_path, &["src/**/*.rs"], &[est])
        })
        .count();

    let coverage = (found_estimators as f64 / sklearn_estimators.len() as f64 * 100.0) as u32;

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "sklearn coverage: ~{}% ({}/{})",
            coverage,
            found_estimators,
            sklearn_estimators.len()
        ),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["train", "fit", "predict", "classifier"]);
    if !is_ml || found_estimators >= sklearn_estimators.len() * 70 / 100 {
        item = item.pass();
    } else if found_estimators >= sklearn_estimators.len() / 3 {
        item = item.partial(format!("Partial sklearn coverage (~{}%)", coverage));
    } else {
        item = item.partial("Limited sklearn-like estimator coverage");
    }

    item.finish_timed(start)
}

fn check_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::source_contains_pattern(project_path, patterns)
}

fn check_ci_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::ci_contains_pattern(project_path, patterns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_evaluate_all_returns_5_items() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn test_all_items_have_tps_principle() {
        let path = PathBuf::from(".");
        for item in evaluate_all(&path) {
            assert!(!item.tps_principle.is_empty(), "Item {} missing TPS", item.id);
        }
    }

    #[test]
    fn test_all_items_have_evidence() {
        let path = PathBuf::from(".");
        for item in evaluate_all(&path) {
            assert!(!item.evidence.is_empty(), "Item {} missing evidence", item.id);
        }
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_cp01_linux_compatibility_id() {
        let result = check_linux_compatibility(Path::new("."));
        assert_eq!(result.id, "CP-01");
        assert_eq!(result.severity, Severity::Major);
        assert_eq!(result.tps_principle, "Portability");
    }

    #[test]
    fn test_cp02_macos_windows_compatibility_id() {
        let result = check_macos_windows_compatibility(Path::new("."));
        assert_eq!(result.id, "CP-02");
        assert_eq!(result.severity, Severity::Major);
        assert_eq!(result.tps_principle, "Portability");
    }

    #[test]
    fn test_cp03_wasm_browser_compatibility_id() {
        let result = check_wasm_browser_compatibility(Path::new("."));
        assert_eq!(result.id, "CP-03");
        assert_eq!(result.severity, Severity::Major);
        assert_eq!(result.tps_principle, "Edge deployment");
    }

    #[test]
    fn test_cp04_numpy_api_coverage_id() {
        let result = check_numpy_api_coverage(Path::new("."));
        assert_eq!(result.id, "CP-04");
        assert_eq!(result.severity, Severity::Major);
        assert_eq!(result.tps_principle, "API completeness");
    }

    #[test]
    fn test_cp05_sklearn_coverage_id() {
        let result = check_sklearn_coverage(Path::new("."));
        assert_eq!(result.id, "CP-05");
        assert_eq!(result.severity, Severity::Major);
        assert_eq!(result.tps_principle, "API completeness");
    }

    #[test]
    fn test_cp_nonexistent_path() {
        let path = Path::new("/nonexistent/path/for/testing");
        let items = evaluate_all(path);
        // Should still return 5 items
        assert_eq!(items.len(), 5);
    }

    #[test]
    fn test_linux_compat_with_ci_dir() {
        let temp_dir = std::env::temp_dir().join("test_cp_linux");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join(".github/workflows")).expect("mkdir failed");

        // Create workflow with ubuntu
        std::fs::write(temp_dir.join(".github/workflows/ci.yml"), "runs-on: ubuntu-latest")
            .expect("unexpected failure");

        let result = check_linux_compatibility(&temp_dir);
        assert_eq!(result.id, "CP-01");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_macos_windows_compat_with_ci() {
        let temp_dir = std::env::temp_dir().join("test_cp_macos");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join(".github/workflows")).expect("mkdir failed");

        // Create workflow with macos
        std::fs::write(temp_dir.join(".github/workflows/ci.yml"), "runs-on: macos-latest")
            .expect("unexpected failure");

        let result = check_macos_windows_compatibility(&temp_dir);
        assert_eq!(result.id, "CP-02");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_wasm_compat_with_cargo_toml() {
        let temp_dir = std::env::temp_dir().join("test_cp_wasm");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("mkdir failed");

        // Create Cargo.toml with wasm-bindgen
        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"[package]
name = "test"
version = "0.1.0"

[dependencies]
wasm-bindgen = "0.2"
"#,
        )
        .expect("unexpected failure");

        let result = check_wasm_browser_compatibility(&temp_dir);
        assert_eq!(result.id, "CP-03");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_numpy_coverage_with_converter() {
        let temp_dir = std::env::temp_dir().join("test_cp_numpy");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).expect("mkdir failed");

        // Create file with numpy converter reference
        std::fs::write(
            temp_dir.join("src/converter.rs"),
            "// numpy converter using trueno operations",
        )
        .expect("unexpected failure");

        let result = check_numpy_api_coverage(&temp_dir);
        assert_eq!(result.id, "CP-04");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_sklearn_coverage_with_converter() {
        let temp_dir = std::env::temp_dir().join("test_cp_sklearn");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).expect("mkdir failed");

        // Create file with sklearn converter reference
        std::fs::write(
            temp_dir.join("src/sklearn_converter.rs"),
            "// sklearn to aprender conversion",
        )
        .expect("unexpected failure");

        let result = check_sklearn_coverage(&temp_dir);
        assert_eq!(result.id, "CP-05");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_all_items_have_reasonable_duration() {
        let path = PathBuf::from(".");
        for item in evaluate_all(&path) {
            // Duration should be reasonable (less than 1 minute per check)
            assert!(
                item.duration_ms < 60_000,
                "Item {} took unreasonably long: {}ms",
                item.id,
                item.duration_ms
            );
        }
    }

    // =========================================================================
    // Coverage Gap: check_numpy_api_coverage partial/limited branches
    // =========================================================================

    #[test]
    fn test_numpy_coverage_partial_with_numeric_project() {
        // Create a project that has tensor/ndarray but only ~50% numpy ops
        let temp_dir = std::env::temp_dir().join("test_cp04_partial");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).expect("mkdir failed");

        // Include ndarray reference (is_numeric=true) and ~10 numpy ops (above 50%)
        std::fs::write(
            temp_dir.join("src/ops.rs"),
            concat!(
                "use ndarray::Array;\n",
                "pub fn reshape() {}\n",
                "pub fn transpose() {}\n",
                "pub fn dot() {}\n",
                "pub fn matmul() {}\n",
                "pub fn sum() {}\n",
                "pub fn mean() {}\n",
                "pub fn std() {}\n",
                "pub fn var() {}\n",
                "pub fn min() {}\n",
                "pub fn max() {}\n",
            ),
        )
        .expect("unexpected failure");

        let result = check_numpy_api_coverage(&temp_dir);
        assert_eq!(result.id, "CP-04");
        // With is_numeric=true and ~10/18 ops (55%), should be partial
        assert_eq!(result.status, super::super::types::CheckStatus::Partial);
        assert!(result.rejection_reason.as_deref().unwrap_or("").contains("NumPy"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_numpy_coverage_limited_with_numeric_project() {
        // Create a project that has ndarray (is_numeric=true) but very few numpy ops (< 50%)
        let temp_dir = std::env::temp_dir().join("test_cp04_limited");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).expect("mkdir failed");

        // Include ndarray reference for is_numeric=true, but only 2 numpy ops
        std::fs::write(
            temp_dir.join("src/lib.rs"),
            "use ndarray::Array2;\npub fn reshape() {}\npub fn dot() {}\n",
        )
        .expect("unexpected failure");

        let result = check_numpy_api_coverage(&temp_dir);
        assert_eq!(result.id, "CP-04");
        // With is_numeric=true and only ~2/18 ops (11%), should be partial "Limited"
        assert_eq!(result.status, super::super::types::CheckStatus::Partial);
        assert!(result.rejection_reason.as_deref().unwrap_or("").contains("Limited"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    // =========================================================================
    // Coverage Gap: check_sklearn_coverage partial/limited branches
    // =========================================================================

    #[test]
    fn test_sklearn_coverage_partial_with_ml_project() {
        // ML project with some sklearn estimators (>= 33%, < 70%)
        let temp_dir = std::env::temp_dir().join("test_cp05_partial");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).expect("mkdir failed");

        // has train/fit/predict (is_ml=true) + ~6 estimators
        std::fs::write(
            temp_dir.join("src/ml.rs"),
            concat!(
                "pub fn train() {}\npub fn fit() {}\npub fn predict() {}\n",
                "pub struct LinearRegression;\n",
                "pub struct LogisticRegression;\n",
                "pub struct Ridge;\n",
                "pub struct Lasso;\n",
                "pub struct RandomForest;\n",
                "pub struct GradientBoosting;\n",
            ),
        )
        .expect("unexpected failure");

        let result = check_sklearn_coverage(&temp_dir);
        assert_eq!(result.id, "CP-05");
        // 6/14 ~= 42%, above 33% threshold, should be partial
        assert_eq!(result.status, super::super::types::CheckStatus::Partial);
        assert!(result.rejection_reason.as_deref().unwrap_or("").contains("sklearn"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_sklearn_coverage_limited_with_ml_project() {
        // ML project with very few sklearn estimators (< 33%)
        let temp_dir = std::env::temp_dir().join("test_cp05_limited");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).expect("mkdir failed");

        // has train/fit (is_ml=true) but only 1 estimator
        std::fs::write(
            temp_dir.join("src/ml.rs"),
            "pub fn train() {}\npub fn fit() {}\npub fn classifier() {}\npub struct LinearRegression;\n",
        )
        .expect("unexpected failure");

        let result = check_sklearn_coverage(&temp_dir);
        assert_eq!(result.id, "CP-05");
        // 1/14 ~= 7%, below 33% threshold, should be partial "Limited"
        assert_eq!(result.status, super::super::types::CheckStatus::Partial);
        assert!(result.rejection_reason.as_deref().unwrap_or("").contains("Limited"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
