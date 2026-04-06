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

    // Extract function names from binding.yaml
    let mut expected: HashSet<String> = HashSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("function:") {
            let func = rest.trim().trim_matches('"').trim_matches('\'').trim();
            if func.is_empty() || func == "N/A" {
                continue;
            }
            // Extract just the function name (after last ::)
            let short = func.rsplit("::").next().unwrap_or(func).to_lowercase();
            if !short.is_empty() {
                expected.insert(short);
            }
        }
    }

    if expected.is_empty() {
        println!("{label}: no function names in binding — nothing to verify");
        return Ok(());
    }

    // Determine source directory
    let src_dir = if let Some(parent) = binding_path.parent() {
        // binding.yaml is in contracts/<repo>/ — source is in ../../<repo>/src/
        // or we can accept --crate-dir
        parent
            .parent()
            .and_then(|p| p.parent())
            .map_or_else(|| Path::new(".").to_path_buf(), |p| p.join(label))
    } else {
        Path::new(".").to_path_buf()
    };

    // Scan source for pub fn declarations
    let mut found: HashSet<String> = HashSet::new();
    let src = src_dir.join("src");
    if src.exists() {
        scan_fns(&src, &mut found);
    }
    let crates = src_dir.join("crates");
    if crates.exists() {
        scan_fns(&crates, &mut found);
    }
    // Also scan current directory's src/ if different
    let local_src = Path::new("src");
    if local_src.exists() && local_src != src {
        scan_fns(local_src, &mut found);
    }

    let mut missing: Vec<&String> = expected
        .iter()
        .filter(|n| !found.contains(n.as_str()))
        .collect();
    missing.sort();

    if let Some(out_path) = output {
        // Write report to file
        let mut report = format!("# Binding Verification Report: {label}\n\n");
        report.push_str(&format!("Expected: {} functions\n", expected.len()));
        report.push_str(&format!("Found in source: {} functions\n", found.len()));
        report.push_str(&format!("Missing: {}\n\n", missing.len()));
        if !missing.is_empty() {
            report.push_str("## Missing Functions\n\n");
            for m in &missing {
                report.push_str(&format!("- `{m}`\n"));
            }
        }
        std::fs::write(out_path, &report)?;
        println!("Report written to {}", out_path.display());
    }

    let verified = expected.len() - missing.len();
    println!(
        "{label}: {verified}/{} binding functions verified in source",
        expected.len()
    );

    if !missing.is_empty() {
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
        return Err(format!("{} ghost binding(s) detected", missing.len()).into());
    }

    Ok(())
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
                for line in content.lines() {
                    let t = line.trim();
                    if t.starts_with("pub fn ")
                        || t.starts_with("pub async fn ")
                        || t.starts_with("pub(crate) fn ")
                        || t.starts_with("fn ")
                    {
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
            }
        }
    }
}
