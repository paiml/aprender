//! #7235: the book-page contracts' structural claims, checked in `--lib`.
//!
//! Each `apr-page-*` contract declares `page.path` and a `sections` list, and its
//! falsification conditions say a missing page, or a `##` section on the page that the
//! contract does not list (or the reverse), is a P0. This reads the contract and the page
//! and checks both directions.

use std::path::Path;

/// `##` headings of a markdown page, outside code fences.
fn h2_headings(md: &str) -> Vec<String> {
    let mut fenced = false;
    let mut out = Vec::new();
    for line in md.lines() {
        if line.starts_with("```") {
            fenced = !fenced;
        } else if !fenced {
            if let Some(h) = line.strip_prefix("## ") {
                out.push(h.trim().to_string());
            }
        }
    }
    out
}

/// Errors (empty when the page matches its contract).
fn check_page(contract_rel: &str) -> Vec<String> {
    let test = "page_contracts_tests";
    let Some(text) = super::workspace_contract_or_skip(test, contract_rel) else {
        return Vec::new();
    };
    let doc: serde_yaml::Value = serde_yaml::from_str(&text).expect("page contract parses");
    let mut errs = Vec::new();
    let Some(page_path) = doc["page"]["path"].as_str() else {
        return vec![format!("{contract_rel}: page.path missing")];
    };
    let Some(md) = crate::tree::workspace_file_or_skip_at(
        test,
        Path::new(env!("CARGO_MANIFEST_DIR")),
        page_path,
    ) else {
        return errs;
    };
    let declared: Vec<String> = doc["sections"]
        .as_sequence()
        .map(|s| {
            s.iter()
                .filter_map(|e| e["heading"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let actual = h2_headings(&md);
    for h in &declared {
        if !actual.contains(h) {
            errs.push(format!(
                "{contract_rel}: section {h:?} is not on {page_path}"
            ));
        }
    }
    for h in &actual {
        if !declared.contains(h) {
            errs.push(format!(
                "{contract_rel}: {page_path} has section {h:?} the contract omits"
            ));
        }
    }
    errs
}

#[test]
fn h2_headings_case_table() {
    let md = "# T\n## A\ntext\n```\n## not a heading\n```\n### deeper\n## B \n";
    assert_eq!(h2_headings(md), vec!["A".to_string(), "B".to_string()]);
    assert!(h2_headings("no headings\n").is_empty());
}

#[test]
fn book_page_contracts_match_their_pages() {
    let pages = [
        "apr-page-examples-bench-bpe-v1.yaml",
        "apr-page-examples-gpu-fallback-dogfood-v1.yaml",
        "apr-page-examples-qa-falsification-v1.yaml",
        "apr-page-examples-qa-run-v1.yaml",
        "apr-page-examples-qa-serve-v1.yaml",
        "apr-page-examples-qa-verify-v1.yaml",
        "apr-page-examples-shell-encryption-demo-v1.yaml",
        "apr-page-introduction-v1.yaml",
        "apr-page-methodology-zero-tolerance-v1.yaml",
    ];
    let errs: Vec<String> = pages.iter().flat_map(|p| check_page(p)).collect();
    assert!(
        errs.is_empty(),
        "book page drifted from its contract:\n{}",
        errs.join("\n")
    );
}

#[test]
fn a_page_with_an_unlisted_section_is_reported() {
    // Mutation check: the comparison itself, on in-memory sections.
    let declared = ["Run".to_string(), "Source".to_string()];
    let actual = h2_headings("## Run\n## Source\n## Extra\n");
    assert!(actual.iter().any(|h| !declared.contains(h)));
}

/// archive-repos-v1: every repo listed for archival has its new home in the monorepo, so
/// the redirect the archived README points at is not a dead end.
#[test]
fn archive_repos_new_homes_exist_in_the_monorepo() {
    let test = "archive_repos_new_homes_exist_in_the_monorepo";
    let Some(text) = super::workspace_contract_or_skip(test, "archive-repos-v1.yaml") else {
        return;
    };
    let doc: serde_yaml::Value = serde_yaml::from_str(&text).expect("archive contract parses");
    let repos = doc["repos_to_archive"]
        .as_sequence()
        .expect("repos_to_archive is a list");
    assert_eq!(repos.len(), 20, "the contract archives 20 repositories");
    let mut errs = Vec::new();
    for r in repos {
        let repo = r["repo"].as_str().unwrap_or("?");
        let krate = r["new_crate"].as_str().unwrap_or("");
        let manifest = format!("crates/{krate}/Cargo.toml");
        match crate::tree::workspace_file_or_skip_at(
            test,
            Path::new(env!("CARGO_MANIFEST_DIR")),
            &manifest,
        ) {
            Some(m) if m.contains(&format!("name = \"{krate}\"")) => {}
            Some(_) => errs.push(format!("{repo}: {manifest} does not declare {krate}")),
            None => return,
        }
    }
    assert!(
        errs.is_empty(),
        "archived repos without a monorepo home:\n{}",
        errs.join("\n")
    );
}
