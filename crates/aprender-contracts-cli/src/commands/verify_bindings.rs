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
        if let Some(short) = short_name(rest) {
            expected.insert(short);
        }
    }
    expected
}

/// The name a `function:` value is resolved by: its last `::` segment, lowercased.
/// `None` for an empty value or `N/A`. ONE normalization, shared with
/// `pv proof-status --binding` (PVL-001 EV-2), so the two commands cannot disagree.
pub(crate) fn short_name(function: &str) -> Option<String> {
    let func = function.trim().trim_matches('"').trim_matches('\'').trim();
    if func.is_empty() || func == "N/A" {
        return None;
    }
    let short = func.rsplit("::").next().unwrap_or(func).to_lowercase();
    (!short.is_empty()).then_some(short)
}

/// Scan the crate's `src/`, `crates/`, and the current-dir `src/` (if different)
/// for `fn` declarations.
pub(crate) fn scan_all_sources(binding_path: &Path, label: &str) -> HashSet<String> {
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

/// Where a binding's source lives.
///
/// The multi-repo layout (`contracts/<repo>/binding.yaml`, source at `../../<repo>/`)
/// is used when that directory exists. In this monorepo it does not
/// (`contracts/aprender/binding.yaml` -> `./aprender/`, absent), and scanning an
/// absent root made nearly every binding a ghost: under PVL-001 EV-2's reject that
/// is a false reject, and one that depended on the caller's cwd. So otherwise the
/// root is the nearest ancestor of the binding file holding a `crates/` or `src/`
/// tree (the workspace root), and `.` only when there is none.
fn derive_src_root(binding_path: &Path, label: &str) -> std::path::PathBuf {
    let has_tree = |d: &Path| d.join("src").is_dir() || d.join("crates").is_dir();
    let legacy = binding_path
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(|p| p.join(label));
    if let Some(l) = legacy.filter(|l| has_tree(l)) {
        return l;
    }
    let abs = std::fs::canonicalize(binding_path).unwrap_or_else(|_| binding_path.to_path_buf());
    abs.ancestors()
        .skip(1)
        .find(|d| has_tree(d))
        .map_or_else(|| Path::new(".").to_path_buf(), Path::to_path_buf)
}

/// Sort the expected names missing from `found` for stable reporting.
fn compute_missing<'a>(expected: &'a HashSet<String>, found: &HashSet<String>) -> Vec<&'a String> {
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

/// Extract lowercased function names from source: every `fn` item, whatever its
/// visibility (`pub`, `pub(crate)`, `pub(super)`, `pub(in path)`) and qualifiers
/// (`const`, `async`, `unsafe`, `extern "ABI"`). PVL-001 EV-2: `pv proof-status`
/// now REJECTS on a ghost, so a real function the scanner cannot see is a false
/// reject. Measured on aprender's contracts/binding.yaml: `compute_mse` is
/// `pub(super) fn` (crates/aprender-core/src/tree/regression_helpers.rs:27) and was
/// reported a ghost by the old four-prefix scanner (scripts/dogfood.sh records the
/// same defect for rmedia's `apply_loudnorm`).
fn extract_fn_names(content: &str, found: &mut HashSet<String>) {
    for line in content.lines() {
        if let Some(name) = fn_item_name(line) {
            found.insert(name);
        }
    }
}

/// The lowercased name of the item (`fn`, `struct`, `enum`, `type`, `trait`) a source
/// line declares, if it declares one.
fn fn_item_name(line: &str) -> Option<String> {
    let mut t = line.trim_start();
    // visibility: `pub` or `pub(...)`
    if let Some(rest) = t.strip_prefix("pub") {
        let rest_trim = rest.trim_start();
        if let Some(inner) = rest_trim.strip_prefix('(') {
            t = inner.split_once(')')?.1.trim_start();
        } else if rest.starts_with(char::is_whitespace) {
            t = rest_trim;
        } else {
            return None; // `pubfoo`, `pub_x`: an identifier, not a visibility
        }
    }
    // qualifiers, in any order the grammar allows them to appear
    loop {
        let before = t;
        for q in ["const ", "async ", "unsafe ", "default "] {
            if let Some(rest) = t.strip_prefix(q) {
                t = rest.trim_start();
            }
        }
        if let Some(rest) = t.strip_prefix("extern ") {
            let rest = rest.trim_start();
            t = match rest.strip_prefix('"') {
                Some(abi) => abi.split_once('"')?.1.trim_start(),
                None => rest,
            };
        }
        if t == before {
            break;
        }
    }
    // A binding may name a function or a type (setfit-apr-v1 binds
    // `SetFitArtifactDoc` and `ClassifyResponse`, both `pub struct`): the resolver
    // sees every item kind a binding can name.
    let part = ["fn ", "struct ", "enum ", "type ", "trait "]
        .iter()
        .find_map(|kw| t.strip_prefix(kw))?;
    let name = part
        .split(|c: char| matches!(c, '(' | '<' | ';' | '{' | ':' | '=') || c.is_whitespace())
        .next()
        .unwrap_or("")
        .trim()
        .to_lowercase();
    (!name.is_empty()).then_some(name)
}

#[cfg(test)]
mod tests {
    use super::fn_item_name;

    /// The resolver's case table: every declaration form a binding can name must be
    /// seen (a miss is a false GHOST reject), and non-declarations must not be.
    #[test]
    fn fn_item_name_case_table() {
        let must_match = [
            ("fn plain() {}", "plain"),
            ("pub fn public(x: u8) -> u8 {", "public"),
            ("pub(crate) fn in_crate() {", "in_crate"),
            (
                "pub(super) fn compute_mse(y_left: &[f32], y_right: &[f32]) -> f32 {",
                "compute_mse",
            ),
            ("pub(in crate::tree) fn scoped() {", "scoped"),
            ("pub async fn serve() {", "serve"),
            ("pub(crate) async fn load() {", "load"),
            ("pub const fn size() -> usize {", "size"),
            ("pub unsafe fn raw() {", "raw"),
            ("pub const unsafe fn both() {", "both"),
            ("pub extern \"C\" fn ffi() {", "ffi"),
            ("    fn indented<T: Clone>(t: T) {", "indented"),
            ("fn Mixed_Case() {", "mixed_case"),
            ("pub struct SetFitArtifactDoc {", "setfitartifactdoc"),
            ("pub struct ClassifyResponse {", "classifyresponse"),
            ("pub(crate) enum Mode {", "mode"),
            ("pub type Alias<T> = Vec<T>;", "alias"),
            ("pub trait Estimator {", "estimator"),
            ("pub unsafe trait Marker {}", "marker"),
            ("pub struct Fn;", "fn"),
        ];
        for (line, want) in must_match {
            assert_eq!(fn_item_name(line).as_deref(), Some(want), "{line}");
        }
        let must_not_match = [
            "// fn commented_out() {}",
            "let f = fn_pointer;",
            "pubfn not_a_decl() {}",
            "impl Fn for X {}",
            "call(fn_like);",
            "let structure = 3;",
            "// struct Commented {}",
        ];
        for line in must_not_match {
            assert_eq!(fn_item_name(line), None, "{line}");
        }
    }
}
