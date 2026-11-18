//! Active-tool registry rows live in `docs/registry/apr-tools/`, not in `contracts/`.
//!
//! An `apr-tool-*-v1.yaml` row (`registry: true`, `category: active-tool`) records
//! that a sibling repo exists, is documented and has CI. It states no equation,
//! obligation or test, so it is a registry entry, not a provable contract; the 21
//! rows moved out together (#7235, quorum receipt quorum-7236-relocate-1739Z).
//! This guard stops one from coming back under `contracts/`: a row there must
//! carry at least one falsification test.

use std::path::Path;

use serde_yaml::Value;

/// Where the registry rows live, relative to the workspace root.
pub const ACTIVE_TOOL_DIR: &str = "docs/registry/apr-tools";

/// Does `text` declare `registry: true` with `category: active-tool`, in
/// `metadata` or at the top level?
#[must_use]
pub fn is_active_tool_row(text: &str) -> bool {
    let Ok(doc) = serde_yaml::from_str::<Value>(text) else {
        return false;
    };
    let found = [doc.get("metadata"), Some(&doc)]
        .into_iter()
        .flatten()
        .any(|m| {
            m.get("registry").and_then(Value::as_bool) == Some(true)
                && m.get("category").and_then(Value::as_str) == Some("active-tool")
        });
    found
}

/// Active-tool rows under `dir` (every subdirectory) that carry no
/// falsification test.
#[must_use]
pub fn active_tool_rows_in(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();
    every_yaml(dir, &mut files);
    files.sort();
    files
        .iter()
        .filter(|f| {
            std::fs::read_to_string(f).is_ok_and(|t| {
                is_active_tool_row(&t) && super::work_tickets::falsification_test_count(&t) == 0
            })
        })
        .map(|f| f.display().to_string())
        .collect()
}

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
mod tests {
    use super::*;

    const ROW: &str = "metadata:\n  registry: true\n  category: active-tool\n";

    #[test]
    fn is_active_tool_row_case_table() {
        let top = "registry: true\ncategory: active-tool\n";
        let not_reg = ROW.replace("true", "false");
        let other = ROW.replace("active-tool", "kernel");
        let rows: [(&str, &str, bool); 5] = [
            ("metadata row", ROW, true),
            ("top-level row", top, true),
            ("registry false", &not_reg, false),
            ("other category", &other, false),
            ("not yaml", ":\t[", false),
        ];
        for (row, text, want) in rows {
            assert_eq!(is_active_tool_row(text), want, "{row}");
        }
    }

    #[test]
    fn guard_fires_on_planted_row_and_spares_tested_one() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let sub = tmp.path().join("sub");
        std::fs::create_dir_all(&sub).expect("mkdir");
        let tested = format!("{ROW}falsification_tests:\n  - id: F-1\n");
        std::fs::write(tmp.path().join("ok.yaml"), tested).expect("write");
        assert!(active_tool_rows_in(tmp.path()).is_empty());
        std::fs::write(sub.join("apr-tool-x-v1.yml"), ROW).expect("write");
        assert_eq!(active_tool_rows_in(tmp.path()).len(), 1);
    }

    /// GUARD: no untested active-tool registry row under `contracts/`.
    #[test]
    fn no_active_tool_row_under_contracts() {
        let Some(dir) = crate::tree::workspace_path_or_skip_at(
            "no_active_tool_row_under_contracts",
            Path::new(env!("CARGO_MANIFEST_DIR")),
            "contracts",
        ) else {
            return;
        };
        let bad = active_tool_rows_in(&dir);
        assert!(
            bad.is_empty(),
            "active-tool registry row(s) under contracts/ with no falsification test: {bad:?}. \
             Move each with `git mv` to {ACTIVE_TOOL_DIR}/ (do not add it to no_test_debt.txt)."
        );
    }
}
