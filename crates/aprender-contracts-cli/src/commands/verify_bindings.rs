//! Verify that functions named in binding.yaml exist in crate source.
//!
//! Layer 2 enforcement: cross-references function names from binding.yaml
//! against `pub fn` declarations found in the crate's `src/` directory.
//! Ghost bindings (claimed implemented but function missing) are reported.
//!
//! This runs in CI as a test — no build.rs modification needed.

use std::collections::HashSet;
use std::path::Path;

pub fn run(
    binding_path: &Path,
    output: Option<&Path>,
    crate_name: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(binding_path)?;
    let label = crate_name.unwrap_or("unknown");

    let expected = parse_expected_functions(&content);
    if expected.is_empty() {
        println!("{label}: no function names in binding — nothing to verify");
        return Ok(());
    }

    let found = scan_all_sources(binding_path, label);
    let missing = compute_missing(&expected, &found);

    if let Some(out_path) = output {
        write_report(out_path, label, expected.len(), found.len(), &missing)?;
    }

    let verified = expected.len() - missing.len();
    println!(
        "{label}: {verified}/{} binding functions verified in source",
        expected.len()
    );

    if missing.is_empty() {
        return Ok(());
    }
    report_ghost_bindings(label, &missing);
    Err(format!("{} ghost binding(s) detected", missing.len()).into())
}

/// Extract lowercased short-function-names from `function:` lines in a binding yaml.
fn parse_expected_functions(content: &str) -> HashSet<String> {
    let mut expected: HashSet<String> = HashSet::new();
    for line in content.lines() {
        let Some(rest) = line.trim().strip_prefix("function:") else {
            continue;
        };
        let func = rest.trim().trim_matches('"').trim_matches('\'').trim();
        if func.is_empty() || func == "N/A" {
            continue;
        }
        let short = func.rsplit("::").next().unwrap_or(func).to_lowercase();
        if !short.is_empty() {
            expected.insert(short);
        }
    }
    expected
}

/// Scan the crate's `src/`, `crates/`, and the current-dir `src/` (if different)
/// for `fn` declarations.
fn scan_all_sources(binding_path: &Path, label: &str) -> HashSet<String> {
    let src_dir = derive_src_root(binding_path, label);
    let mut found: HashSet<String> = HashSet::new();
    let src = src_dir.join("src");
    if src.exists() {
        scan_fns(&src, &mut found);
    }
    let crates = src_dir.join("crates");
    if crates.exists() {
        scan_fns(&crates, &mut found);
    }
    let local_src = Path::new("src");
    if local_src.exists() && local_src != src {
        scan_fns(local_src, &mut found);
    }
    found
}

/// binding.yaml lives in `contracts/<repo>/` — source is `../../<repo>/`.
/// Falls back to `.` when the path has no usable parent chain.
fn derive_src_root(binding_path: &Path, label: &str) -> std::path::PathBuf {
    let Some(parent) = binding_path.parent() else {
        return Path::new(".").to_path_buf();
    };
    parent
        .parent()
        .and_then(|p| p.parent())
        .map_or_else(|| Path::new(".").to_path_buf(), |p| p.join(label))
}

/// Sort the expected names missing from `found` for stable reporting.
fn compute_missing<'a>(
    expected: &'a HashSet<String>,
    found: &HashSet<String>,
) -> Vec<&'a String> {
    let mut missing: Vec<&String> = expected
        .iter()
        .filter(|n| !found.contains(n.as_str()))
        .collect();
    missing.sort();
    missing
}

/// Write the binding-verification markdown report.
fn write_report(
    out_path: &Path,
    label: &str,
    expected: usize,
    found: usize,
    missing: &[&String],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut report = format!("# Binding Verification Report: {label}\n\n");
    report.push_str(&format!("Expected: {} functions\n", expected));
    report.push_str(&format!("Found in source: {} functions\n", found));
    report.push_str(&format!("Missing: {}\n\n", missing.len()));
    if !missing.is_empty() {
        report.push_str("## Missing Functions\n\n");
        for m in missing {
            report.push_str(&format!("- `{m}`\n"));
        }
    }
    std::fs::write(out_path, &report)?;
    println!("Report written to {}", out_path.display());
    Ok(())
}

fn report_ghost_bindings(label: &str, missing: &[&String]) {
    eprintln!(
        "{label}: {} ghost binding(s) — function not found in source:",
        missing.len()
    );
    for m in missing.iter().take(20) {
        eprintln!("  - {m}");
    }
    if missing.len() > 20 {
        eprintln!("  ... and {} more", missing.len() - 20);
    }
}

fn scan_fns(dir: &Path, found: &mut HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name != "target" && name != ".git" && name != "tests" {
                scan_fns(&path, found);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                extract_fn_names(&content, found);
            }
        }
    }
}

/// Extract lowercased `fn`/`pub fn`/`pub async fn`/`pub(crate) fn` names from source.
fn extract_fn_names(content: &str, found: &mut HashSet<String>) {
    for line in content.lines() {
        let t = line.trim();
        if !(t.starts_with("pub fn ")
            || t.starts_with("pub async fn ")
            || t.starts_with("pub(crate) fn ")
            || t.starts_with("fn "))
        {
            continue;
        }
        let part = t
            .trim_start_matches("pub async fn ")
            .trim_start_matches("pub(crate) fn ")
            .trim_start_matches("pub fn ")
            .trim_start_matches("fn ");
        let name = part
            .split('(')
            .next()
            .unwrap_or("")
            .split('<')
            .next()
            .unwrap_or("")
            .trim()
            .to_lowercase();
        if !name.is_empty() {
            found.insert(name);
        }
    }
}
