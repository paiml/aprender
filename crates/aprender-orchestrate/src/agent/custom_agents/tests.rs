//! Unit tests for [`super::parse_agent_md`], [`super::load_custom_agents_from`],
//! and [`super::discover_standard_locations`].

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
name: code-reviewer\n\
description: Reviews code for bugs\n\
max_iterations: 5\n\
---\n\
You are a code-review subagent.\n";

#[test]
fn parse_agent_md_happy_path() {
    let spec = parse_agent_md(VALID).expect("parse");
    assert_eq!(spec.name, "code-reviewer");
    assert_eq!(spec.description, "Reviews code for bugs");
    assert_eq!(spec.max_iterations, 5);
    assert!(spec.system_prompt.contains("code-review subagent"));
}

#[test]
fn parse_agent_md_accepts_crlf() {
    let crlf = "---\r\nname: x\r\ndescription: y\r\n---\r\nbody\r\n";
    let spec = parse_agent_md(crlf).expect("parse CRLF");
    assert_eq!(spec.name, "x");
    assert_eq!(spec.description, "y");
    assert_eq!(spec.system_prompt, "body");
}

#[test]
fn parse_agent_md_strips_bom() {
    let bom = "\u{feff}---\nname: bom\ndescription: d\n---\nbody\n";
    let spec = parse_agent_md(bom).expect("parse BOM");
    assert_eq!(spec.name, "bom");
}

#[test]
fn parse_agent_md_tolerates_unknown_keys() {
    // Claude-compat: tools/model ignored, not rejected
    let src = "---\n\
name: x\n\
description: y\n\
tools: read,write\n\
model: haiku\n\
---\nprompt\n";
    let spec = parse_agent_md(src).expect("parse");
    assert_eq!(spec.name, "x");
    assert_eq!(spec.max_iterations, 8, "default when absent");
}

#[test]
fn parse_agent_md_quoted_values() {
    let src = "---\n\
name: \"quoted-name\"\n\
description: 'single-quoted'\n\
---\nbody\n";
    let spec = parse_agent_md(src).expect("parse");
    assert_eq!(spec.name, "quoted-name");
    assert_eq!(spec.description, "single-quoted");
}

#[test]
fn parse_agent_md_missing_frontmatter_errors() {
    let err = parse_agent_md("no fence here\njust text\n").unwrap_err();
    assert!(matches!(err, CustomAgentError::MissingFrontmatter));
}

#[test]
fn parse_agent_md_unclosed_frontmatter_errors() {
    // Open `---\n` but no closing fence
    let err = parse_agent_md("---\nname: x\ndescription: y\n").unwrap_err();
    assert!(matches!(err, CustomAgentError::MissingFrontmatter));
}

#[test]
fn parse_agent_md_missing_name_errors() {
    let src = "---\ndescription: d\n---\nbody\n";
    let err = parse_agent_md(src).unwrap_err();
    assert!(matches!(err, CustomAgentError::MissingName));
}

#[test]
fn parse_agent_md_missing_description_errors() {
    let src = "---\nname: x\n---\nbody\n";
    let err = parse_agent_md(src).unwrap_err();
    assert!(matches!(err, CustomAgentError::MissingDescription));
}

#[test]
fn parse_agent_md_empty_body_errors() {
    let src = "---\nname: x\ndescription: d\n---\n   \n";
    let err = parse_agent_md(src).unwrap_err();
    assert!(matches!(err, CustomAgentError::EmptyBody));
}

#[test]
fn parse_agent_md_zero_max_iterations_keeps_default() {
    let src = "---\nname: x\ndescription: d\nmax_iterations: 0\n---\nbody\n";
    let spec = parse_agent_md(src).expect("parse");
    assert_eq!(spec.max_iterations, 8, "0 rejected, default kept");
}

#[test]
fn parse_agent_md_skips_comment_lines() {
    let src = "---\n# comment\nname: x\n# another\ndescription: d\n---\nbody\n";
    let spec = parse_agent_md(src).expect("parse");
    assert_eq!(spec.name, "x");
    assert_eq!(spec.description, "d");
}

#[test]
fn load_custom_agents_from_flat_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    write(&dir.path().join("reviewer.md"), VALID);
    write(&dir.path().join("planner.md"), "---\nname: planner\ndescription: plans\n---\nbody\n");
    // Non-md files are ignored
    write(&dir.path().join("README.txt"), "not an agent");

    let specs = load_custom_agents_from(dir.path());
    assert_eq!(specs.len(), 2);
    let names: Vec<_> = specs.iter().map(|s| s.name.clone()).collect();
    assert!(names.contains(&"code-reviewer".to_string()));
    assert!(names.contains(&"planner".to_string()));
}

#[test]
fn load_custom_agents_from_subdir_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    write(&dir.path().join("code-reviewer").join("AGENT.md"), VALID);
    // An unrelated file in a subdirectory is ignored
    write(&dir.path().join("empty-dir").join("notes.txt"), "nothing");

    let specs = load_custom_agents_from(dir.path());
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].name, "code-reviewer");
}

#[test]
fn load_custom_agents_silently_skips_malformed() {
    let dir = tempfile::tempdir().expect("tempdir");
    write(&dir.path().join("good.md"), VALID);
    // Malformed: missing frontmatter → skipped, not a hard error
    write(&dir.path().join("broken.md"), "no fence\njust text\n");

    let specs = load_custom_agents_from(dir.path());
    assert_eq!(specs.len(), 1, "only the well-formed file loads");
    assert_eq!(specs[0].name, "code-reviewer");
}

#[test]
fn load_custom_agents_on_missing_dir_returns_empty() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("does-not-exist");
    let specs = load_custom_agents_from(&missing);
    assert!(specs.is_empty());
}

#[test]
fn discover_standard_locations_finds_project_dir() {
    let cwd = tempfile::tempdir().expect("tempdir");
    write(&cwd.path().join(".apr").join("agents").join("x.md"), VALID);

    let specs = discover_standard_locations(cwd.path());
    assert!(specs.iter().any(|s| s.name == "code-reviewer"));
}

#[test]
fn discover_standard_locations_falls_back_to_claude_dir() {
    let cwd = tempfile::tempdir().expect("tempdir");
    // No .apr/agents — only .claude/agents
    write(&cwd.path().join(".claude").join("agents").join("x.md"), VALID);

    let specs = discover_standard_locations(cwd.path());
    assert!(
        specs.iter().any(|s| s.name == "code-reviewer"),
        "must fall back to .claude/agents when .apr/agents is absent"
    );
}

#[test]
fn discover_standard_locations_apr_dir_takes_precedence() {
    // When both layouts exist, .apr/agents wins (matches lookup order).
    let cwd = tempfile::tempdir().expect("tempdir");
    write(
        &cwd.path().join(".apr").join("agents").join("x.md"),
        "---\nname: code-reviewer\ndescription: FROM_APR\n---\nbody\n",
    );
    write(
        &cwd.path().join(".claude").join("agents").join("x.md"),
        "---\nname: code-reviewer\ndescription: FROM_CLAUDE\n---\nbody\n",
    );

    let specs = discover_standard_locations(cwd.path());
    let reviewer = specs.iter().find(|s| s.name == "code-reviewer").expect("found");
    assert_eq!(reviewer.description, "FROM_APR", ".apr/ wins over .claude/");
}

#[test]
fn register_discovered_into_counts_and_registers() {
    use crate::agent::task_tool::{default_registry, SubagentRegistry};

    let cwd = tempfile::tempdir().expect("tempdir");
    write(&cwd.path().join(".apr").join("agents").join("reviewer.md"), VALID);
    write(
        &cwd.path().join(".apr").join("agents").join("planner.md"),
        "---\nname: planner\ndescription: plans\n---\nbody\n",
    );

    let mut registry = SubagentRegistry::new();
    let n = register_discovered_into(&mut registry, cwd.path());
    assert_eq!(n, 2);
    assert!(registry.resolve("code-reviewer").is_some());
    assert!(registry.resolve("planner").is_some());

    // And: overrides built-ins if names collide
    let mut r2 = default_registry();
    write(
        &cwd.path().join(".apr").join("agents").join("explore.md"),
        "---\nname: explore\ndescription: CUSTOM\n---\nbody\n",
    );
    register_discovered_into(&mut r2, cwd.path());
    assert_eq!(r2.resolve("explore").unwrap().description, "CUSTOM");
}

#[test]
fn register_discovered_into_on_missing_dir_returns_zero() {
    use crate::agent::task_tool::SubagentRegistry;

    let cwd = tempfile::tempdir().expect("tempdir");
    let mut registry = SubagentRegistry::new();
    let n = register_discovered_into(&mut registry, cwd.path());
    assert_eq!(n, 0);
    assert!(registry.is_empty());
}

#[test]
fn error_display_messages() {
    assert!(CustomAgentError::MissingFrontmatter.to_string().contains("frontmatter"));
    assert!(CustomAgentError::MissingName.to_string().contains("name"));
    assert!(CustomAgentError::MissingDescription.to_string().contains("description"));
    assert!(CustomAgentError::EmptyBody.to_string().contains("empty"));
    assert!(CustomAgentError::Io("boom".into()).to_string().contains("boom"));
}
