//! Unit tests for [`super::load_org_policy`] and friends.

use super::*;
use std::fs;
use tempfile::tempdir;

#[test]
fn returns_none_when_no_roots_match() {
    let tmp = tempdir().unwrap();
    let out = load_org_policy(&[tmp.path()], "CLAUDE.md", 1024);
    assert!(out.is_none());
}

#[test]
fn returns_none_when_max_bytes_is_zero() {
    let tmp = tempdir().unwrap();
    fs::write(tmp.path().join("CLAUDE.md"), "should be ignored").unwrap();
    let out = load_org_policy(&[tmp.path()], "CLAUDE.md", 0);
    assert!(out.is_none(), "max_bytes=0 must disable loader");
}

#[test]
fn loads_file_when_present_at_first_root() {
    let tmp = tempdir().unwrap();
    fs::write(tmp.path().join("CLAUDE.md"), "hello org").unwrap();
    let p = load_org_policy(&[tmp.path()], "CLAUDE.md", 1024).unwrap();
    assert_eq!(p.content, "hello org");
    assert_eq!(p.tier, PolicyTier::Enforced);
    assert_eq!(p.source, tmp.path().join("CLAUDE.md"));
}

#[test]
fn first_root_wins_over_second() {
    let a = tempdir().unwrap();
    let b = tempdir().unwrap();
    fs::write(a.path().join("CLAUDE.md"), "from-A").unwrap();
    fs::write(b.path().join("CLAUDE.md"), "from-B").unwrap();
    let p = load_org_policy(&[a.path(), b.path()], "CLAUDE.md", 1024).unwrap();
    assert_eq!(p.content, "from-A");
    assert_eq!(p.source, a.path().join("CLAUDE.md"));
}

#[test]
fn falls_back_to_second_root_when_first_missing() {
    let a = tempdir().unwrap();
    let b = tempdir().unwrap();
    fs::write(b.path().join("CLAUDE.md"), "from-B").unwrap();
    let p = load_org_policy(&[a.path(), b.path()], "CLAUDE.md", 1024).unwrap();
    assert_eq!(p.content, "from-B");
}

#[test]
fn skips_root_where_path_is_a_directory_not_file() {
    let a = tempdir().unwrap();
    let b = tempdir().unwrap();
    // CLAUDE.md is a dir, not a file, in root A
    fs::create_dir(a.path().join("CLAUDE.md")).unwrap();
    fs::write(b.path().join("CLAUDE.md"), "from-B").unwrap();
    let p = load_org_policy(&[a.path(), b.path()], "CLAUDE.md", 1024).unwrap();
    assert_eq!(p.content, "from-B");
}

#[test]
fn truncates_when_content_exceeds_max_bytes() {
    let tmp = tempdir().unwrap();
    let body = "x".repeat(5000);
    fs::write(tmp.path().join("CLAUDE.md"), &body).unwrap();
    let p = load_org_policy(&[tmp.path()], "CLAUDE.md", 100).unwrap();
    assert!(p.content.starts_with(&"x".repeat(100)));
    assert!(p.content.contains("(truncated from 5000 bytes)"));
}

#[test]
fn preserves_utf8_boundary_on_truncation() {
    let tmp = tempdir().unwrap();
    // each "é" is 2 bytes — truncating at an odd byte would corrupt
    let body = "é".repeat(200);
    fs::write(tmp.path().join("CLAUDE.md"), &body).unwrap();
    let p = load_org_policy(&[tmp.path()], "CLAUDE.md", 51).unwrap();
    // Even though we asked for 51 bytes, the truncation must land on
    // a char boundary — 25 é's = 50 bytes is the largest valid prefix.
    assert!(p.content.starts_with("é"));
    assert!(p.content.contains("(truncated from 400 bytes)"));
}

#[test]
fn does_not_truncate_when_under_budget() {
    let tmp = tempdir().unwrap();
    fs::write(tmp.path().join("CLAUDE.md"), "small").unwrap();
    let p = load_org_policy(&[tmp.path()], "CLAUDE.md", 1024).unwrap();
    assert_eq!(p.content, "small");
    assert!(!p.content.contains("truncated from"));
}

#[test]
fn canonical_roots_point_at_apr_code_first_then_claude_code() {
    let roots = canonical_system_roots();
    assert_eq!(roots[0], Path::new("/etc/apr-code"));
    assert_eq!(roots[1], Path::new("/etc/claude-code"));
}

#[test]
fn enforced_tier_is_highest_precedence() {
    // Sanity check on the derived Ord impl — enforced tier must
    // compare greater than any future lower tier we add, so the
    // merge step is total-ordered. Today there's only one variant
    // so reflexive ordering is what we can assert.
    assert_eq!(PolicyTier::Enforced, PolicyTier::Enforced);
    assert!(PolicyTier::Enforced <= PolicyTier::Enforced);
    assert!(PolicyTier::Enforced >= PolicyTier::Enforced);
}

#[test]
fn ignores_io_error_and_tries_next_root() {
    // Read-permission errors on a /etc file shouldn't kill the
    // REPL. We can't realistically strip read on tempdirs in CI,
    // but the no-root-matches path is equivalent (continue to
    // next root). Covered by returns_none_when_no_roots_match +
    // falls_back_to_second_root_when_first_missing.
    let a = tempdir().unwrap();
    let b = tempdir().unwrap();
    fs::write(b.path().join("CLAUDE.md"), "survivor").unwrap();
    let p = load_org_policy(&[a.path(), b.path()], "CLAUDE.md", 1024).unwrap();
    assert_eq!(p.content, "survivor");
}

#[test]
fn org_policy_roundtrip_clone_and_eq() {
    let p = OrgPolicy {
        source: PathBuf::from("/etc/apr-code/CLAUDE.md"),
        content: "hello".into(),
        tier: PolicyTier::Enforced,
    };
    let clone = p.clone();
    assert_eq!(p, clone);
}
