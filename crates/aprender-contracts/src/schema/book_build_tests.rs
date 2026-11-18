//! #7235: structural claims of `apr-book-build-v1` and `apr-book-schema-v1`, checked in `--lib`.

use std::path::{Path, PathBuf};

fn tree_path(test: &str, rel: &str) -> Option<PathBuf> {
    crate::tree::workspace_path_or_skip_at(test, Path::new(env!("CARGO_MANIFEST_DIR")), rel)
}

/// Targets of `[text](path.md)` links in a markdown string.
fn md_link_targets(md: &str) -> Vec<String> {
    let mut out = Vec::new();
    for part in md.split("](").skip(1) {
        if let Some(end) = part.find(')') {
            let t = &part[..end];
            if t.ends_with(".md") && !t.contains("://") {
                out.push(t.to_string());
            }
        }
    }
    out
}

/// `contracts/...` path named by a leading `<!-- PCU: id | contract: path -->` comment.
fn pcu_contract(md: &str) -> Option<String> {
    let first = md.lines().next()?;
    let rest = first.strip_prefix("<!-- PCU: ")?;
    let (_, tail) = rest.split_once("| contract: ")?;
    Some(tail.trim_end_matches("-->").trim().to_string())
}

fn md_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            md_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "md") {
            out.push(p);
        }
    }
}

#[test]
fn md_link_targets_case_table() {
    let md = "[a](x.md) [b](https://e.com/y.md) [c](z.html) [d](d/e.md)";
    assert_eq!(md_link_targets(md), vec!["x.md", "d/e.md"]);
    assert_eq!(
        pcu_contract("<!-- PCU: a | contract: contracts/a-v1.yaml -->\n").as_deref(),
        Some("contracts/a-v1.yaml")
    );
    assert_eq!(pcu_contract("# no header\n"), None);
}

/// apr-book-build FALSIFY-BOOK-002/005 and the structure invariants: every `.md` that
/// SUMMARY.md lists exists, book.toml and book.yml exist, the repo URL is paiml/aprender.
#[test]
fn book_build_structure_holds() {
    let t = "book_build_structure_holds";
    let Some(summary) = tree_path(t, "book/src/SUMMARY.md") else {
        return;
    };
    let src = summary.parent().expect("book/src");
    let text = std::fs::read_to_string(&summary).expect("SUMMARY.md reads");
    let missing: Vec<String> = md_link_targets(&text)
        .into_iter()
        .filter(|p| !src.join(p).exists())
        .collect();
    assert!(
        missing.is_empty(),
        "SUMMARY.md lists missing pages: {missing:?}"
    );
    let toml = tree_path(t, "book/book.toml").expect("in tree");
    let toml = std::fs::read_to_string(toml).expect("book.toml reads");
    assert!(toml.contains("git-repository-url = \"https://github.com/paiml/aprender\""));
    assert!(tree_path(t, ".github/workflows/book.yml").is_some());
}

/// apr-book-schema: every page's PCU header names a contract that exists.
#[test]
fn book_pages_pcu_contracts_exist() {
    let t = "book_pages_pcu_contracts_exist";
    let Some(summary) = tree_path(t, "book/src/SUMMARY.md") else {
        return;
    };
    let src = summary.parent().expect("book/src");
    let root = src.parent().and_then(Path::parent).expect("workspace root");
    let mut files = Vec::new();
    md_files(src, &mut files);
    assert!(files.len() > 100, "book has pages: {}", files.len());
    let mut bad = Vec::new();
    for f in files {
        let md = std::fs::read_to_string(&f).unwrap_or_default();
        if let Some(c) = pcu_contract(&md) {
            if !root.join(&c).exists() {
                bad.push(format!("{}: {c}", f.display()));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "PCU header names a missing contract: {bad:?}"
    );
}
