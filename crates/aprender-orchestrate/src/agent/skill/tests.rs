//! Unit tests for [`super::parse_skill_md`], [`super::load_skills_from`],
//! [`super::discover_skills`], and [`super::SkillRegistry`].

use std::fs;
use std::path::Path;

use super::*;

fn write(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, body).expect("write file");
}

const VALID: &str = "---\n\
name: rust-test\n\
description: Run the Rust test suite and interpret failures\n\
when_to_use: User asks about failing cargo tests\n\
allowed-tools: shell, grep\n\
---\n\
Run cargo test from the repo root.\n";

#[test]
fn parse_skill_md_happy_path() {
    let s = parse_skill_md(VALID).expect("parse");
    assert_eq!(s.name, "rust-test");
    assert_eq!(s.description, "Run the Rust test suite and interpret failures");
    assert_eq!(s.when_to_use.as_deref(), Some("User asks about failing cargo tests"));
    assert_eq!(s.allowed_tools, vec!["shell", "grep"]);
    assert!(s.instructions.contains("cargo test"));
}

#[test]
fn parse_skill_md_allowed_tools_space_separated() {
    let src = "---\n\
name: x\n\
description: y\n\
allowed-tools: shell grep file_read\n\
---\nbody\n";
    let s = parse_skill_md(src).expect("parse");
    assert_eq!(s.allowed_tools, vec!["shell", "grep", "file_read"]);
}

#[test]
fn parse_skill_md_when_to_use_optional() {
    let src = "---\nname: x\ndescription: y\n---\nbody\n";
    let s = parse_skill_md(src).expect("parse");
    assert!(s.when_to_use.is_none(), "absent key → None");
    assert!(s.allowed_tools.is_empty());
}

#[test]
fn parse_skill_md_accepts_underscore_aliases() {
    let src = "---\n\
name: x\n\
description: y\n\
when-to-use: trigger phrase\n\
allowed_tools: shell\n\
---\nbody\n";
    let s = parse_skill_md(src).expect("parse");
    assert_eq!(s.when_to_use.as_deref(), Some("trigger phrase"));
    assert_eq!(s.allowed_tools, vec!["shell"]);
}

#[test]
fn parse_skill_md_tolerates_context_fork_key() {
    // Claude-compat: `context: fork` must not error.
    let src = "---\nname: x\ndescription: y\ncontext: fork\n---\nbody\n";
    let s = parse_skill_md(src).expect("parse");
    assert_eq!(s.name, "x");
}

#[test]
fn parse_skill_md_strips_bom() {
    let bom = "\u{feff}---\nname: b\ndescription: d\n---\nbody\n";
    let s = parse_skill_md(bom).expect("parse");
    assert_eq!(s.name, "b");
}

#[test]
fn parse_skill_md_accepts_crlf() {
    let crlf = "---\r\nname: x\r\ndescription: y\r\n---\r\nbody\r\n";
    let s = parse_skill_md(crlf).expect("parse");
    assert_eq!(s.instructions, "body");
}

#[test]
fn parse_skill_md_missing_frontmatter_errors() {
    let err = parse_skill_md("no fence\njust text\n").unwrap_err();
    assert!(matches!(err, SkillError::MissingFrontmatter));
}

#[test]
fn parse_skill_md_missing_name_errors() {
    let err = parse_skill_md("---\ndescription: d\n---\nbody\n").unwrap_err();
    assert!(matches!(err, SkillError::MissingName));
}

#[test]
fn parse_skill_md_missing_description_errors() {
    let err = parse_skill_md("---\nname: x\n---\nbody\n").unwrap_err();
    assert!(matches!(err, SkillError::MissingDescription));
}

#[test]
fn parse_skill_md_empty_body_errors() {
    let err = parse_skill_md("---\nname: x\ndescription: d\n---\n   \n").unwrap_err();
    assert!(matches!(err, SkillError::EmptyBody));
}

#[test]
fn load_skills_flat_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    write(&dir.path().join("a.md"), VALID);
    write(
        &dir.path().join("b.md"),
        "---\nname: b\ndescription: db\n---\ninstructions-b\n",
    );
    write(&dir.path().join("readme.txt"), "skip");

    let skills = load_skills_from(dir.path());
    assert_eq!(skills.len(), 2);
}

#[test]
fn load_skills_subdir_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    write(&dir.path().join("rust-test").join("SKILL.md"), VALID);
    // Wrong filename in subdir — ignored.
    write(&dir.path().join("ignored").join("notes.md"), "nope");

    let skills = load_skills_from(dir.path());
    assert_eq!(skills.len(), 1);
    assert_eq!(skills[0].name, "rust-test");
}

#[test]
fn load_skills_silently_skips_malformed() {
    let dir = tempfile::tempdir().expect("tempdir");
    write(&dir.path().join("good.md"), VALID);
    write(&dir.path().join("bad.md"), "no fence\n");

    let skills = load_skills_from(dir.path());
    assert_eq!(skills.len(), 1);
}

#[test]
fn discover_skills_prefers_apr_over_claude() {
    let cwd = tempfile::tempdir().expect("tempdir");
    write(
        &cwd.path().join(".apr").join("skills").join("x.md"),
        "---\nname: x\ndescription: FROM_APR\n---\nbody\n",
    );
    write(
        &cwd.path().join(".claude").join("skills").join("x.md"),
        "---\nname: x\ndescription: FROM_CLAUDE\n---\nbody\n",
    );

    let skills = discover_skills(cwd.path());
    let x = skills.iter().find(|s| s.name == "x").expect("found");
    assert_eq!(x.description, "FROM_APR");
}

#[test]
fn discover_skills_falls_back_to_claude_dir() {
    let cwd = tempfile::tempdir().expect("tempdir");
    write(&cwd.path().join(".claude").join("skills").join("y.md"), VALID);
    let skills = discover_skills(cwd.path());
    assert!(skills.iter().any(|s| s.name == "rust-test"));
}

#[test]
fn discover_skills_missing_dir_returns_empty() {
    let cwd = tempfile::tempdir().expect("tempdir");
    assert!(discover_skills(cwd.path()).is_empty());
}

#[test]
fn registry_register_resolve_names() {
    let mut r = SkillRegistry::new();
    assert!(r.is_empty());
    let s = parse_skill_md(VALID).unwrap();
    r.register(s);
    assert_eq!(r.len(), 1);
    assert!(r.resolve("rust-test").is_some());
    assert_eq!(r.names(), vec!["rust-test"]);
    assert!(r.resolve("nonexistent").is_none());
}

#[test]
fn registry_register_replaces_by_name() {
    let mut r = SkillRegistry::new();
    let s1 = parse_skill_md(VALID).unwrap();
    let s2 = Skill {
        name: "rust-test".into(),
        description: "overridden".into(),
        when_to_use: None,
        allowed_tools: Vec::new(),
        instructions: "override".into(),
    };
    r.register(s1);
    r.register(s2);
    assert_eq!(r.len(), 1);
    assert_eq!(r.resolve("rust-test").unwrap().description, "overridden");
}

#[test]
fn registry_auto_match_by_when_to_use() {
    let mut r = SkillRegistry::new();
    r.register(parse_skill_md(VALID).unwrap());
    // Matches "failing cargo tests" via substring "failing cargo"
    let matched = r.auto_match("My failing cargo tests are broken");
    assert!(matched.is_some());
    assert_eq!(matched.unwrap().name, "rust-test");
}

#[test]
fn registry_auto_match_returns_none_when_no_when_to_use() {
    let mut r = SkillRegistry::new();
    let s = Skill {
        name: "x".into(),
        description: "d".into(),
        when_to_use: None,
        allowed_tools: Vec::new(),
        instructions: "body".into(),
    };
    r.register(s);
    assert!(r.auto_match("anything").is_none());
}

#[test]
fn registry_auto_match_case_insensitive() {
    let mut r = SkillRegistry::new();
    let s = Skill {
        name: "x".into(),
        description: "d".into(),
        when_to_use: Some("Deployment Issue".into()),
        allowed_tools: Vec::new(),
        instructions: "body".into(),
    };
    r.register(s);
    assert!(r.auto_match("seeing a deployment issue in prod").is_some());
}

#[test]
fn register_discovered_skills_into_counts() {
    let cwd = tempfile::tempdir().expect("tempdir");
    write(&cwd.path().join(".apr").join("skills").join("a.md"), VALID);
    write(
        &cwd.path().join(".apr").join("skills").join("b.md"),
        "---\nname: b\ndescription: d\n---\nbody\n",
    );

    let mut r = SkillRegistry::new();
    let n = register_discovered_skills_into(&mut r, cwd.path());
    assert_eq!(n, 2);
    assert!(r.resolve("rust-test").is_some());
    assert!(r.resolve("b").is_some());
}

#[test]
fn register_discovered_skills_into_empty_dir_returns_zero() {
    let cwd = tempfile::tempdir().expect("tempdir");
    let mut r = SkillRegistry::new();
    assert_eq!(register_discovered_skills_into(&mut r, cwd.path()), 0);
}

#[test]
fn error_display_messages() {
    assert!(SkillError::MissingFrontmatter.to_string().contains("frontmatter"));
    assert!(SkillError::MissingName.to_string().contains("name"));
    assert!(SkillError::MissingDescription.to_string().contains("description"));
    assert!(SkillError::EmptyBody.to_string().contains("empty"));
    assert!(SkillError::Io("boom".into()).to_string().contains("boom"));
}
