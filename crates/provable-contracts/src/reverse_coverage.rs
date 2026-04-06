//! Reverse coverage: detect public functions without contract bindings.
//!
//! Forward coverage checks: does every binding have an implementation?
//! Reverse coverage checks: does every implementation have a binding?
//!
//! This closes the "whack-a-mole" gap where new functions escape the
//! contract system silently.

use std::collections::HashSet;
use std::path::Path;

/// A public function found in a crate's source code.
#[derive(Debug, Clone)]
pub struct PubFn {
    /// Fully qualified path (e.g., `aprender::nn::ssm::ssm_scan`)
    pub path: String,
    /// File where the function is defined
    pub file: String,
    /// Line number
    pub line: usize,
    /// Whether it has a #[contract] annotation
    pub has_contract_macro: bool,
    /// Feature gate if the function is behind `#[cfg(feature = "...")]`
    pub feature_gate: Option<String>,
}

/// Result of reverse coverage analysis.
#[derive(Debug)]
pub struct ReverseCoverageReport {
    /// Total public functions found in the crate
    pub total_pub_fns: usize,
    /// Functions that have a binding entry
    pub bound_fns: usize,
    /// Functions that have a #[contract] annotation
    pub annotated_fns: usize,
    /// Functions marked exempt (trivial, don't need contracts)
    pub exempt_fns: usize,
    /// Functions without any binding
    pub unbound: Vec<PubFn>,
    /// Reverse coverage percentage (bound + exempt / total)
    pub coverage_pct: f64,
}

/// Scan a crate directory for `pub fn` declarations and diff against binding.yaml.
pub fn reverse_coverage(crate_dir: &Path, binding_path: &Path) -> ReverseCoverageReport {
    let bound_names = extract_bound_functions(binding_path);
    let pub_fns = scan_pub_fns(crate_dir);

    let total = pub_fns.len();
    let mut bound = 0usize;
    let mut annotated = 0usize;
    let mut exempt = 0usize;
    let mut unbound = Vec::new();

    for f in &pub_fns {
        let fn_name = f.path.rsplit("::").next().unwrap_or(&f.path).to_lowercase();

        if f.has_contract_macro {
            annotated += 1;
            bound += 1;
        } else if bound_names.contains(&fn_name) {
            bound += 1;
        } else if crate::auto_exempt::is_auto_exempt(&fn_name) {
            exempt += 1;
        } else {
            unbound.push(f.clone());
        }
    }

    let covered = bound + exempt;
    let coverage_pct = if total > 0 {
        #[allow(clippy::cast_precision_loss)]
        {
            (covered as f64 / total as f64) * 100.0
        }
    } else {
        100.0
    };

    ReverseCoverageReport {
        total_pub_fns: total,
        bound_fns: bound,
        annotated_fns: annotated,
        exempt_fns: exempt,
        unbound,
        coverage_pct,
    }
}

/// Extract function names from binding.yaml.
fn extract_bound_functions(binding_path: &Path) -> HashSet<String> {
    let mut names = HashSet::new();
    if let Ok(content) = std::fs::read_to_string(binding_path) {
        for line in content.lines() {
            let trimmed = line.trim();
            let func_line = trimmed.strip_prefix("- ").unwrap_or(trimmed);
            if let Some(rest) = func_line.strip_prefix("function:") {
                let fname = rest.trim().trim_matches('"').trim_matches('\'').trim();
                let short = fname.rsplit("::").next().unwrap_or(fname).to_lowercase();
                names.insert(short);
            }
        }
    }
    names
}

/// Scan .rs files for `pub fn` declarations.
fn scan_pub_fns(crate_dir: &Path) -> Vec<PubFn> {
    let mut results = Vec::new();
    let src_dirs = [crate_dir.join("src"), crate_dir.join("crates")];
    for dir in &src_dirs {
        if dir.exists() {
            scan_dir(dir, &mut results);
        }
    }
    results
}

fn scan_dir(dir: &Path, results: &mut Vec<PubFn>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name != "target" && name != "tests" && name != ".git" {
                scan_dir(&path, results);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            scan_file(&path, results);
        }
    }
}

fn scan_file(path: &Path, results: &mut Vec<PubFn>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    let file_str = path.display().to_string();
    let mut prev_line_has_contract = false;
    // Track feature gates from #[cfg(feature = "...")] or #[cfg(all(test, feature = "..."))]
    let mut module_feature_gate: Option<String> = None;
    let mut pending_feature_gate: Option<String> = None;
    let mut brace_depth: usize = 0;
    let mut gate_depth: Option<usize> = None;

    for (i, line) in content.lines().enumerate() {
        let trimmed = line.trim();

        // Track brace depth for module-level cfg gates
        for ch in trimmed.chars() {
            if ch == '{' {
                brace_depth += 1;
            } else if ch == '}' {
                brace_depth = brace_depth.saturating_sub(1);
                // If we exit the gated block, clear the module gate
                if gate_depth.is_some_and(|d| brace_depth < d) {
                    module_feature_gate = None;
                    gate_depth = None;
                }
            }
        }

        // Detect #[cfg(feature = "...")] on functions or modules
        if trimmed.starts_with("#[cfg(") {
            if let Some(feat) = extract_feature_gate(trimmed) {
                pending_feature_gate = Some(feat);
                continue;
            }
        }

        // Detect cfg-gated module declarations: `mod foo {`
        if trimmed.starts_with("mod ") && pending_feature_gate.is_some() {
            if trimmed.contains('{') {
                module_feature_gate = pending_feature_gate.take();
                gate_depth = Some(brace_depth);
            }
            pending_feature_gate = None;
            continue;
        }

        if trimmed.contains("#[contract(") {
            prev_line_has_contract = true;
            continue;
        }

        if trimmed.starts_with("pub fn ") || trimmed.starts_with("pub async fn ") {
            let fn_part = trimmed
                .trim_start_matches("pub async fn ")
                .trim_start_matches("pub fn ");
            let fn_name = fn_part
                .split('(')
                .next()
                .unwrap_or("")
                .split('<')
                .next()
                .unwrap_or("")
                .trim();

            if !fn_name.is_empty() && fn_name != "main" && fn_name != "new" {
                let gate = pending_feature_gate
                    .take()
                    .or_else(|| module_feature_gate.clone());
                results.push(PubFn {
                    path: fn_name.to_string(),
                    file: file_str.clone(),
                    line: i + 1,
                    has_contract_macro: prev_line_has_contract,
                    feature_gate: gate,
                });
            }
            prev_line_has_contract = false;
            pending_feature_gate = None;
        } else if !trimmed.starts_with("//") && !trimmed.starts_with('#') {
            prev_line_has_contract = false;
            pending_feature_gate = None;
        }
    }
}

/// Extract the feature name from a `#[cfg(feature = "...")]` or
/// `#[cfg(all(test, feature = "..."))]` attribute.
fn extract_feature_gate(cfg_line: &str) -> Option<String> {
    let feat_pos = cfg_line.find("feature")?;
    let rest = &cfg_line[feat_pos..];
    let quote_start = rest.find('"')?;
    let after_quote = &rest[quote_start + 1..];
    let quote_end = after_quote.find('"')?;
    Some(after_quote[..quote_end].to_string())
}

#[cfg(test)]
#[allow(clippy::all)]
#[path = "reverse_coverage_tests.rs"]
mod tests;
