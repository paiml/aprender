//! pmat work-ticket stubs live in `docs/work-tickets/`, not in `contracts/`.
//!
//! pmat's work-ticket tool writes one stub per ticket (header
//! `# Auto-generated from .pmat-work/…`, `surface: work-contract`). A stub states
//! no equation, obligation or test: it is a tracking record, not a provable
//! contract, so it does not belong in the `contracts/` corpus that pv walks and
//! the #5291 no-test debt counts. The 263 stubs moved out together (quorum
//! receipt QUORUM-WORKSTUBS-1113). Two rules keep it that way:
//!
//! - (a) every file in [`WORK_TICKETS_DIR`] is a stub ([`is_work_stub`]); a real
//!   contract is never parked there.
//! - (b) no file under `contracts/` carries a work-ticket marker
//!   ([`has_work_marker`]) unless it carries at least one falsification test.
//!   pmat keeps writing new stubs to `contracts/work/`; this rule is what
//!   catches one.

use std::path::Path;

use serde_yaml::{Mapping, Value};

/// Where the stubs live, relative to the workspace root.
pub const WORK_TICKETS_DIR: &str = "docs/work-tickets";

/// The first line pmat writes into every stub.
pub const STUB_HEADER: &str = "# Auto-generated from .pmat-work/";

/// The `surface:` value pmat writes into every stub.
pub const WORK_SURFACE: &str = "work-contract";

/// The template: each top-level key and the keys allowed under it
/// (`None` for a scalar).
const TEMPLATE: &[(&str, Option<&[&str]>)] = &[
    ("name", None),
    (
        "metadata",
        Some(&["version", "kind", "description", "references"]),
    ),
    ("surface", None),
    (
        "verification_summary",
        Some(&["target_level", "current_level", "total_obligations"]),
    ),
];

/// Does `text` read as a work-ticket file: pmat's header on line 1, or a
/// top-level `surface: work-contract`?
#[must_use]
pub fn has_work_marker(text: &str) -> bool {
    text.lines()
        .next()
        .is_some_and(|l| l.starts_with(STUB_HEADER))
        || text.lines().any(|l| {
            l.trim_end()
                .strip_prefix("surface:")
                .is_some_and(|v| v.trim() == WORK_SURFACE)
        })
}

/// The shape predicate for a stub: pmat's header, `surface: work-contract`,
/// only the template keys, and so no `equations`, `proof_obligations` or
/// `falsification_tests`. `Err` names the first rule `text` breaks.
pub fn is_work_stub(text: &str) -> Result<(), String> {
    if !text
        .lines()
        .next()
        .is_some_and(|l| l.starts_with(STUB_HEADER))
    {
        return Err(format!("line 1 is not `{STUB_HEADER}…`"));
    }
    let doc: Value = serde_yaml::from_str(text).map_err(|e| format!("not YAML: {e}"))?;
    let top = doc.as_mapping().ok_or("not a mapping")?;
    if top.get("surface").and_then(Value::as_str) != Some(WORK_SURFACE) {
        return Err(format!("`surface` is not `{WORK_SURFACE}`"));
    }
    for key in ["equations", "proof_obligations", "falsification_tests"] {
        if top.contains_key(key) {
            return Err(format!("carries `{key}`: a contract, not a stub"));
        }
    }
    only_template_keys(top)
}

fn only_template_keys(top: &Mapping) -> Result<(), String> {
    for (k, v) in top {
        let key = k.as_str().ok_or("a non-string top-level key")?;
        let Some((_, sub)) = TEMPLATE.iter().find(|(t, _)| *t == key) else {
            return Err(format!("key `{key}` is not in the stub template"));
        };
        if let Some(allowed) = sub {
            let m = v
                .as_mapping()
                .ok_or_else(|| format!("`{key}` is not a mapping"))?;
            for sk in m.keys() {
                let s = sk.as_str().unwrap_or("<non-string>");
                if !allowed.contains(&s) {
                    return Err(format!("key `{key}.{s}` is not in the stub template"));
                }
            }
        }
    }
    Ok(())
}

/// Number of `falsification_tests` `text` declares (0 when absent or unparsable).
#[must_use]
pub fn falsification_test_count(text: &str) -> usize {
    serde_yaml::from_str::<Value>(text)
        .ok()
        .and_then(|d| {
            d.get("falsification_tests")
                .and_then(Value::as_sequence)
                .map(Vec::len)
        })
        .unwrap_or(0)
}

/// Rule (a) over `dir`: every `.yaml`/`.yml` file is a stub. Returns
/// (files read, offenders).
#[must_use]
pub fn non_stubs_in(dir: &Path) -> (usize, Vec<String>) {
    let mut files = Vec::new();
    every_yaml(dir, &mut files);
    files.sort();
    let mut bad = Vec::new();
    for f in &files {
        let why = match std::fs::read_to_string(f) {
            Ok(t) => is_work_stub(&t).err(),
            Err(e) => Some(format!("unreadable: {e}")),
        };
        if let Some(why) = why {
            bad.push(format!("{}: {why}", f.display()));
        }
    }
    (files.len(), bad)
}

/// Rule (b) over `dir`: files carrying a work-ticket marker and no test.
#[must_use]
pub fn untested_work_files_in(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    every_yaml(dir, &mut files);
    files.sort();
    files
        .iter()
        .filter(|f| {
            std::fs::read_to_string(f)
                .is_ok_and(|t| has_work_marker(&t) && falsification_test_count(&t) == 0)
        })
        .map(|f| f.display().to_string())
        .collect()
}

/// Every `.yaml`/`.yml` under `dir`, every subdirectory included (unlike the
/// pv walker, nothing is skipped: a stub hidden in `quarantine/` is still one).
fn every_yaml(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for path in entries.flatten().map(|e| e.path()) {
        if path.is_dir() {
            every_yaml(&path, out);
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("yaml" | "yml")
        ) {
            out.push(path);
        }
    }
}

#[cfg(test)]
#[path = "work_tickets_tests.rs"]
mod tests;
