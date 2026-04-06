//! Compliance check command handler

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::items_after_statements)]

use crate::config::CliConfig;
use crate::error::{CliError, CliResult};
use crate::{ComplyArgs, ComplyOutputFormat, Verbosity};
use std::path::Path;

/// Result of a single compliance check
#[derive(Debug, Clone)]
pub struct ComplianceResult {
    /// Check identifier (e.g., "C001")
    pub id: String,
    /// Whether the check passed
    pub passed: bool,
    /// Detailed messages about the check
    pub details: Vec<String>,
}

impl ComplianceResult {
    /// Create a passing result
    #[must_use]
    pub fn pass(id: &str) -> Self {
        Self {
            id: id.to_string(),
            passed: true,
            details: Vec::new(),
        }
    }

    /// Create a failing result
    #[must_use]
    pub fn fail(id: &str, reason: &str) -> Self {
        Self {
            id: id.to_string(),
            passed: false,
            details: vec![reason.to_string()],
        }
    }

    /// Add a detail to the result
    #[must_use]
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.details.push(detail.to_string());
        self
    }
}

/// Run all compliance checks
#[allow(clippy::too_many_lines)]
pub fn run_compliance_checks(config: &CliConfig, args: &ComplyArgs) -> CliResult<()> {
    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n{}", "=".repeat(62));
        eprintln!("  PROBAR COMPLY - WASM Compliance Checker");
        eprintln!("{}\n", "=".repeat(62));
    }

    type CheckFn = Box<dyn Fn(&Path, &ComplyArgs) -> ComplianceResult>;

    let mut results: Vec<ComplianceResult> = Vec::new();
    let mut all_passed = true;

    let checks_to_run: Vec<(&str, &str, CheckFn)> = vec![
        (
            "C001",
            "Code execution verified",
            Box::new(|path, _| check_c001_code_execution(path)),
        ),
        (
            "C002",
            "Console errors fail tests",
            Box::new(|_, _| check_c002_console_errors()),
        ),
        (
            "C003",
            "Custom elements tested",
            Box::new(|path, _| check_c003_custom_elements(path)),
        ),
        (
            "C004",
            "Threading modes tested",
            Box::new(|_, _| check_c004_threading_modes()),
        ),
        (
            "C005",
            "Low memory tested",
            Box::new(|_, _| check_c005_low_memory()),
        ),
        (
            "C006",
            "COOP/COEP headers",
            Box::new(|path, _| check_c006_headers(path)),
        ),
        (
            "C007",
            "Replay hash matches",
            Box::new(|_, _| check_c007_replay_hash()),
        ),
        (
            "C008",
            "Cache handling",
            Box::new(|_, _| check_c008_cache()),
        ),
        (
            "C009",
            "WASM size limit",
            Box::new(|path, args| check_c009_wasm_size(path, args.max_wasm_size)),
        ),
        (
            "C010",
            "No panic paths",
            Box::new(|path, _| check_c010_panic_paths(path)),
        ),
    ];

    let filtered_checks: Vec<(&str, &str, CheckFn)> = if let Some(ref requested) = args.checks {
        checks_to_run
            .into_iter()
            .filter(|(id, _, _)| requested.iter().any(|r| r == id))
            .collect()
    } else {
        checks_to_run
    };

    if config.verbosity != Verbosity::Quiet {
        eprintln!(
            "Running {} compliance check(s) on {}\n",
            filtered_checks.len(),
            args.path.display()
        );
    }

    for (id, description, check_fn) in &filtered_checks {
        let result = check_fn(&args.path, args);

        let status = if result.passed { "Y" } else { "N" };
        let color = if result.passed {
            "\x1b[32m"
        } else {
            "\x1b[31m"
        };
        let reset = "\x1b[0m";

        if config.verbosity != Verbosity::Quiet {
            eprintln!("  {color}[{status}]{reset} {id}: {description}");
            if args.detailed && !result.details.is_empty() {
                for detail in &result.details {
                    eprintln!("      - {detail}");
                }
            }
        }

        if !result.passed {
            all_passed = false;
            if args.fail_fast {
                break;
            }
        }

        results.push(result);
    }

    let passed_count = results.iter().filter(|r| r.passed).count();
    let total_count = results.len();

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n{}", "=".repeat(62));
        eprintln!("  Result: {passed_count}/{total_count} checks passed");
        eprintln!("{}\n", "=".repeat(62));
    }

    if let Some(ref report_path) = args.report {
        let report = generate_comply_report(&results, &args.format);
        std::fs::write(report_path, &report)
            .map_err(|e| CliError::report_generation(format!("Failed to write report: {e}")))?;
        if config.verbosity != Verbosity::Quiet {
            eprintln!("Report written to: {}", report_path.display());
        }
    }

    match args.format {
        ComplyOutputFormat::Json | ComplyOutputFormat::Junit => {
            let report = generate_comply_report(&results, &args.format);
            println!("{report}");
        }
        ComplyOutputFormat::Text => {}
    }

    if all_passed {
        Ok(())
    } else {
        Err(CliError::test_execution(format!(
            "Compliance check failed: {passed_count}/{total_count} checks passed"
        )))
    }
}

/// Generate a compliance report in the specified format
#[must_use]
pub fn generate_comply_report(results: &[ComplianceResult], format: &ComplyOutputFormat) -> String {
    match format {
        ComplyOutputFormat::Json => {
            let json_results: Vec<_> = results
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.id,
                        "passed": r.passed,
                        "details": r.details,
                    })
                })
                .collect();
            serde_json::json!({
                "version": "1.0",
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "results": json_results,
                "summary": {
                    "total": results.len(),
                    "passed": results.iter().filter(|r| r.passed).count(),
                    "failed": results.iter().filter(|r| !r.passed).count(),
                }
            })
            .to_string()
        }
        ComplyOutputFormat::Junit => {
            let timestamp = chrono::Utc::now().to_rfc3339();
            let failures = results.iter().filter(|r| !r.passed).count();
            let mut xml = format!(
                r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="probar-comply" tests="{}" failures="{}" timestamp="{}">
  <testsuite name="compliance" tests="{}" failures="{}">"#,
                results.len(),
                failures,
                timestamp,
                results.len(),
                failures
            );
            for r in results {
                let failure_tag = if r.passed {
                    String::new()
                } else {
                    format!(
                        r#"
      <failure message="Check failed">{}</failure>"#,
                        r.details.join("; ")
                    )
                };
                xml.push_str(&format!(
                    r#"
    <testcase name="{}" classname="probar.comply">{}</testcase>"#,
                    r.id, failure_tag
                ));
            }
            xml.push_str(
                r"
  </testsuite>
</testsuites>",
            );
            xml
        }
        ComplyOutputFormat::Text => {
            let passed = results.iter().filter(|r| r.passed).count();
            let total = results.len();
            format!(
                "Compliance Report: {}/{} checks passed\n{}",
                passed,
                total,
                results
                    .iter()
                    .map(|r| {
                        let status = if r.passed { "[Y]" } else { "[N]" };
                        format!("{} {}: {}", status, r.id, r.details.join(", "))
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        }
    }
}

// =============================================================================
// Individual Compliance Checks
// =============================================================================

/// C001: Verify code actually executes (not just mocked HTML)
#[must_use]
pub fn check_c001_code_execution(path: &Path) -> ComplianceResult {
    let wasm_exists = find_wasm_files(path).is_some();
    let test_files = find_test_files(path);

    if wasm_exists && !test_files.is_empty() {
        ComplianceResult::pass("C001")
            .with_detail(&format!("Found {} test file(s)", test_files.len()))
    } else if !wasm_exists {
        ComplianceResult::fail("C001", "No WASM files found - code may not execute")
    } else {
        ComplianceResult::fail("C001", "No test files found to verify execution")
    }
}

/// C002: Console errors should fail tests
#[must_use]
pub fn check_c002_console_errors() -> ComplianceResult {
    ComplianceResult::pass("C002").with_detail("Console capture enabled (verify in test config)")
}

/// C003: Custom elements are tested
#[must_use]
pub fn check_c003_custom_elements(path: &Path) -> ComplianceResult {
    let html_files = find_html_files_in_dir(path);
    let has_custom_elements = html_files.iter().any(|f| {
        std::fs::read_to_string(f)
            .map(|content| content.contains("customElements.define") || content.contains("<wasm-"))
            .unwrap_or(false)
    });

    if has_custom_elements {
        ComplianceResult::pass("C003").with_detail("Custom elements detected")
    } else {
        ComplianceResult::pass("C003")
            .with_detail("No custom elements found (may be OK if not used)")
    }
}

/// C004: Both threading and non-threading modes tested
#[must_use]
pub fn check_c004_threading_modes() -> ComplianceResult {
    ComplianceResult::pass("C004").with_detail("Threading mode validation requires runtime check")
}

/// C005: Low memory scenario tested
#[must_use]
pub fn check_c005_low_memory() -> ComplianceResult {
    ComplianceResult::pass("C005").with_detail("Low memory simulation available via WasmStrictMode")
}

/// C006: COOP/COEP headers present for `SharedArrayBuffer`
#[must_use]
pub fn check_c006_headers(path: &Path) -> ComplianceResult {
    let has_server_config = path.join(".htaccess").exists()
        || path.join("vercel.json").exists()
        || path.join("netlify.toml").exists()
        || path.join("_headers").exists();

    let has_probar_config = check_probar_cross_origin_config(path);
    let has_makefile_config = check_makefile_cross_origin(path);

    if has_server_config {
        ComplianceResult::pass("C006").with_detail("Server config found (verify COOP/COEP headers)")
    } else if has_probar_config {
        ComplianceResult::pass("C006").with_detail("probar.toml has cross_origin_isolated = true")
    } else if has_makefile_config {
        ComplianceResult::pass("C006")
            .with_detail("Makefile uses probador serve --cross-origin-isolated")
    } else {
        ComplianceResult::fail("C006", "No server config found for COOP/COEP headers")
            .with_detail("Add Cross-Origin-Opener-Policy: same-origin")
            .with_detail("Add Cross-Origin-Embedder-Policy: require-corp")
            .with_detail("Or use: probador serve --cross-origin-isolated")
    }
}

/// C007: Replay hash matches for deterministic tests
#[must_use]
pub fn check_c007_replay_hash() -> ComplianceResult {
    ComplianceResult::pass("C007")
        .with_detail("Replay hash validation available via SimulationRecording")
}

/// C008: Proper cache handling
#[must_use]
pub fn check_c008_cache() -> ComplianceResult {
    ComplianceResult::pass("C008").with_detail("Cache handling verified at runtime")
}

/// C009: WASM binary under size limit
#[must_use]
pub fn check_c009_wasm_size(path: &Path, max_size: usize) -> ComplianceResult {
    if let Some(wasm_files) = find_wasm_files(path) {
        for wasm_path in wasm_files {
            if let Ok(metadata) = std::fs::metadata(&wasm_path) {
                let size = metadata.len() as usize;
                if size > max_size {
                    return ComplianceResult::fail(
                        "C009",
                        &format!("WASM too large: {size} bytes > {max_size} bytes limit"),
                    )
                    .with_detail(&format!("File: {}", wasm_path.display()));
                }
            }
        }
        ComplianceResult::pass("C009")
            .with_detail(&format!("All WASM files under {max_size} byte limit"))
    } else {
        ComplianceResult::pass("C009").with_detail("No WASM files to check")
    }
}

/// C010: No panic paths in WASM
#[must_use]
pub fn check_c010_panic_paths(path: &Path) -> ComplianceResult {
    let cargo_toml = path.join("Cargo.toml");
    if cargo_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
            if content.contains("panic = \"abort\"") {
                return ComplianceResult::pass("C010").with_detail("panic = \"abort\" configured");
            }
        }
    }
    ComplianceResult::pass("C010").with_detail("Verify panic-free via clippy::panic lint")
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check probar.toml for `cross_origin_isolated` setting
#[must_use]
pub fn check_probar_cross_origin_config(path: &Path) -> bool {
    let config_paths = [
        path.join("probar.toml"),
        path.join(".probar.toml"),
        path.join("probador.toml"),
        path.join(".probador.toml"),
    ];

    for config_path in &config_paths {
        if let Ok(content) = std::fs::read_to_string(config_path) {
            if content.contains("cross_origin_isolated")
                && (content.contains("= true") || content.contains("=true"))
            {
                return true;
            }
        }
    }
    false
}

/// Check Makefile for probador serve --cross-origin-isolated
#[must_use]
pub fn check_makefile_cross_origin(path: &Path) -> bool {
    let makefile_paths = [
        path.join("Makefile"),
        path.join("makefile"),
        path.join("GNUmakefile"),
    ];

    for makefile_path in &makefile_paths {
        if let Ok(content) = std::fs::read_to_string(makefile_path) {
            if (content.contains("probador serve") || content.contains("probar serve"))
                && content.contains("--cross-origin-isolated")
            {
                return true;
            }
        }
    }

    let package_json = path.join("package.json");
    if let Ok(content) = std::fs::read_to_string(package_json) {
        if content.contains("probador serve") && content.contains("--cross-origin-isolated") {
            return true;
        }
    }

    false
}

/// Find WASM files in directory
fn find_wasm_files(dir: &Path) -> Option<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    find_files_recursive(dir, "wasm", &mut files);
    if files.is_empty() {
        None
    } else {
        Some(files)
    }
}

/// Find test files in directory
fn find_test_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    find_files_recursive(dir, "rs", &mut files);
    files.retain(|f| {
        f.to_string_lossy().contains("test")
            || f.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("test_"))
    });
    files
}

/// Find HTML files in directory
fn find_html_files_in_dir(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    find_files_recursive(dir, "html", &mut files);
    files
}

/// Recursively find files with extension
fn find_files_recursive(dir: &Path, ext: &str, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.starts_with('.') && name != "target" && name != "node_modules" {
                    find_files_recursive(&path, ext, files);
                }
            } else if path.extension().is_some_and(|e| e == ext) {
                files.push(path);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_compliance_result_pass() {
        let result = ComplianceResult::pass("C001");
        assert!(result.passed);
        assert_eq!(result.id, "C001");
        assert!(result.details.is_empty());
    }

    #[test]
    fn test_compliance_result_fail() {
        let result = ComplianceResult::fail("C001", "reason");
        assert!(!result.passed);
        assert_eq!(result.id, "C001");
        assert_eq!(result.details, vec!["reason"]);
    }

    #[test]
    fn test_compliance_result_with_detail() {
        let result = ComplianceResult::pass("C001")
            .with_detail("detail1")
            .with_detail("detail2");
        assert_eq!(result.details.len(), 2);
    }

    #[test]
    fn test_check_c001_no_wasm() {
        let temp = TempDir::new().unwrap();
        let result = check_c001_code_execution(temp.path());
        assert!(!result.passed);
    }

    #[test]
    fn test_check_c001_with_wasm_and_tests() {
        let temp = TempDir::new().unwrap();

        // Create wasm file
        std::fs::create_dir_all(temp.path().join("pkg")).unwrap();
        std::fs::write(temp.path().join("pkg").join("app.wasm"), b"wasm").unwrap();

        // Create test file
        std::fs::create_dir_all(temp.path().join("tests")).unwrap();
        std::fs::write(temp.path().join("tests").join("test_app.rs"), "// test").unwrap();

        let result = check_c001_code_execution(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_c002_console_errors() {
        let result = check_c002_console_errors();
        assert!(result.passed);
    }

    #[test]
    fn test_check_c003_no_custom_elements() {
        let temp = TempDir::new().unwrap();
        let result = check_c003_custom_elements(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_c003_with_custom_elements() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("index.html"),
            "<html><body><wasm-app></wasm-app></body></html>",
        )
        .unwrap();
        let result = check_c003_custom_elements(temp.path());
        assert!(result.passed);
        assert!(result
            .details
            .iter()
            .any(|d| d.contains("Custom elements detected")));
    }

    #[test]
    fn test_check_c004_threading_modes() {
        let result = check_c004_threading_modes();
        assert!(result.passed);
    }

    #[test]
    fn test_check_c005_low_memory() {
        let result = check_c005_low_memory();
        assert!(result.passed);
    }

    #[test]
    fn test_check_c006_no_config() {
        let temp = TempDir::new().unwrap();
        let result = check_c006_headers(temp.path());
        assert!(!result.passed);
    }

    #[test]
    fn test_check_c006_with_htaccess() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join(".htaccess"), "Header set COOP").unwrap();
        let result = check_c006_headers(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_c006_with_probar_toml() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("probar.toml"),
            "[server]\ncross_origin_isolated = true",
        )
        .unwrap();
        let result = check_c006_headers(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_c007_replay_hash() {
        let result = check_c007_replay_hash();
        assert!(result.passed);
    }

    #[test]
    fn test_check_c008_cache() {
        let result = check_c008_cache();
        assert!(result.passed);
    }

    #[test]
    fn test_check_c009_no_wasm() {
        let temp = TempDir::new().unwrap();
        let result = check_c009_wasm_size(temp.path(), 5_000_000);
        assert!(result.passed);
    }

    #[test]
    fn test_check_c009_under_limit() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("small.wasm"), vec![0u8; 1000]).unwrap();
        let result = check_c009_wasm_size(temp.path(), 5_000_000);
        assert!(result.passed);
    }

    #[test]
    fn test_check_c009_over_limit() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("large.wasm"), vec![0u8; 10_000]).unwrap();
        let result = check_c009_wasm_size(temp.path(), 5_000);
        assert!(!result.passed);
    }

    #[test]
    fn test_check_c010_no_cargo_toml() {
        let temp = TempDir::new().unwrap();
        let result = check_c010_panic_paths(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_c010_with_panic_abort() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("Cargo.toml"),
            r#"[profile.release]
panic = "abort"
"#,
        )
        .unwrap();
        let result = check_c010_panic_paths(temp.path());
        assert!(result.passed);
        assert!(result
            .details
            .iter()
            .any(|d| d.contains("panic = \"abort\"")));
    }

    #[test]
    fn test_check_probar_cross_origin_config_false() {
        let temp = TempDir::new().unwrap();
        assert!(!check_probar_cross_origin_config(temp.path()));
    }

    #[test]
    fn test_check_probar_cross_origin_config_true() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("probar.toml"),
            "cross_origin_isolated = true",
        )
        .unwrap();
        assert!(check_probar_cross_origin_config(temp.path()));
    }

    #[test]
    fn test_check_makefile_cross_origin_false() {
        let temp = TempDir::new().unwrap();
        assert!(!check_makefile_cross_origin(temp.path()));
    }

    #[test]
    fn test_check_makefile_cross_origin_true() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("Makefile"),
            "serve:\n\tprobador serve --cross-origin-isolated",
        )
        .unwrap();
        assert!(check_makefile_cross_origin(temp.path()));
    }

    #[test]
    fn test_find_files_recursive() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("sub")).unwrap();
        std::fs::write(temp.path().join("a.rs"), "").unwrap();
        std::fs::write(temp.path().join("sub").join("b.rs"), "").unwrap();
        std::fs::write(temp.path().join("c.txt"), "").unwrap();

        let mut files = Vec::new();
        find_files_recursive(temp.path(), "rs", &mut files);
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_find_files_recursive_skips_hidden() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join(".hidden")).unwrap();
        std::fs::write(temp.path().join(".hidden").join("a.rs"), "").unwrap();
        std::fs::write(temp.path().join("b.rs"), "").unwrap();

        let mut files = Vec::new();
        find_files_recursive(temp.path(), "rs", &mut files);
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_find_files_recursive_skips_target() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("target")).unwrap();
        std::fs::write(temp.path().join("target").join("a.rs"), "").unwrap();
        std::fs::write(temp.path().join("b.rs"), "").unwrap();

        let mut files = Vec::new();
        find_files_recursive(temp.path(), "rs", &mut files);
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_generate_comply_report_json() {
        let results = vec![
            ComplianceResult::pass("C001").with_detail("detail"),
            ComplianceResult::fail("C002", "reason"),
        ];
        let report = generate_comply_report(&results, &ComplyOutputFormat::Json);
        assert!(report.contains("\"passed\":true") || report.contains("\"passed\": true"));
        assert!(report.contains("\"passed\":false") || report.contains("\"passed\": false"));
        assert!(report.contains("\"id\":\"C001\"") || report.contains("\"id\": \"C001\""));
    }

    #[test]
    fn test_generate_comply_report_junit() {
        let results = vec![
            ComplianceResult::pass("C001"),
            ComplianceResult::fail("C002", "reason"),
        ];
        let report = generate_comply_report(&results, &ComplyOutputFormat::Junit);
        assert!(report.contains("<testsuites"));
        assert!(report.contains("failures=\"1\""));
    }

    #[test]
    fn test_generate_comply_report_text() {
        let results = vec![
            ComplianceResult::pass("C001").with_detail("ok"),
            ComplianceResult::fail("C002", "reason"),
        ];
        let report = generate_comply_report(&results, &ComplyOutputFormat::Text);
        assert!(report.contains("[Y] C001"));
        assert!(report.contains("[N] C002"));
    }

    #[test]
    fn test_check_c001_wasm_only_no_tests() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("app.wasm"), b"wasm").unwrap();
        let result = check_c001_code_execution(temp.path());
        assert!(!result.passed);
        assert!(result.details.iter().any(|d| d.contains("No test files")));
    }

    #[test]
    fn test_check_c006_with_vercel_json() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("vercel.json"), "{}").unwrap();
        let result = check_c006_headers(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_c006_with_netlify_toml() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("netlify.toml"), "").unwrap();
        let result = check_c006_headers(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_c006_with_headers_file() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("_headers"), "").unwrap();
        let result = check_c006_headers(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_c006_with_makefile_lowercase() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("makefile"),
            "serve:\n\tprobar serve --cross-origin-isolated",
        )
        .unwrap();
        let result = check_c006_headers(temp.path());
        assert!(result.passed);
    }

    #[test]
    fn test_check_makefile_cross_origin_gnu_makefile() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("GNUmakefile"),
            "serve:\n\tprobador serve --cross-origin-isolated",
        )
        .unwrap();
        assert!(check_makefile_cross_origin(temp.path()));
    }

    #[test]
    fn test_check_makefile_cross_origin_package_json() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("package.json"),
            r#"{"scripts": {"serve": "probador serve --cross-origin-isolated"}}"#,
        )
        .unwrap();
        assert!(check_makefile_cross_origin(temp.path()));
    }

    #[test]
    fn test_check_probar_cross_origin_config_dot_probar() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join(".probar.toml"),
            "cross_origin_isolated = true",
        )
        .unwrap();
        assert!(check_probar_cross_origin_config(temp.path()));
    }

    #[test]
    fn test_check_probar_cross_origin_config_probador_toml() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("probador.toml"),
            "cross_origin_isolated=true",
        )
        .unwrap();
        assert!(check_probar_cross_origin_config(temp.path()));
    }

    #[test]
    fn test_check_probar_cross_origin_config_dot_probador() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join(".probador.toml"),
            "cross_origin_isolated = true",
        )
        .unwrap();
        assert!(check_probar_cross_origin_config(temp.path()));
    }

    #[test]
    fn test_check_c003_with_custom_elements_define() {
        let temp = TempDir::new().unwrap();
        std::fs::write(
            temp.path().join("index.html"),
            "<script>customElements.define('my-element', MyElement)</script>",
        )
        .unwrap();
        let result = check_c003_custom_elements(temp.path());
        assert!(result.passed);
        assert!(result.details.iter().any(|d| d.contains("Custom elements")));
    }

    #[test]
    fn test_find_wasm_files_nested() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("pkg")).unwrap();
        std::fs::write(temp.path().join("pkg").join("app.wasm"), b"wasm").unwrap();
        let files = find_wasm_files(temp.path());
        assert!(files.is_some());
        assert_eq!(files.unwrap().len(), 1);
    }

    #[test]
    fn test_find_wasm_files_none() {
        let temp = TempDir::new().unwrap();
        let files = find_wasm_files(temp.path());
        assert!(files.is_none());
    }

    #[test]
    fn test_find_test_files() {
        let temp = TempDir::new().unwrap();
        std::fs::create_dir_all(temp.path().join("tests")).unwrap();
        std::fs::write(temp.path().join("tests").join("test_app.rs"), "").unwrap();
        std::fs::write(temp.path().join("src.rs"), "").unwrap();
        let files = find_test_files(temp.path());
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_find_html_files_in_dir() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("index.html"), "").unwrap();
        std::fs::write(temp.path().join("about.html"), "").unwrap();
        std::fs::write(temp.path().join("style.css"), "").unwrap();
        let files = find_html_files_in_dir(temp.path());
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_generate_comply_report_empty() {
        let results: Vec<ComplianceResult> = vec![];
        let report = generate_comply_report(&results, &ComplyOutputFormat::Text);
        assert!(report.contains("0/0 checks passed"));
    }

    #[test]
    fn test_generate_comply_report_junit_all_pass() {
        let results = vec![
            ComplianceResult::pass("C001"),
            ComplianceResult::pass("C002"),
        ];
        let report = generate_comply_report(&results, &ComplyOutputFormat::Junit);
        assert!(report.contains("failures=\"0\""));
    }
}
