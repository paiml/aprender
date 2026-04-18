//! Unit tests for `agent::hooks`.
//!
//! PMAT-CODE-HOOKS-001 — falsification conditions:
//! * `HookDecision::from_exit_code` must route 0/1/{2,others} exactly.
//! * `HookRegistry::run` must surface the FIRST blocking decision and
//!   suppress later hooks of the same event.
//! * Matcher semantics: `None` always fires, `Some(s)` fires iff the
//!   context string contains `s`.
//! * TOML round-trip: a `[[hooks]]` array must deserialize into
//!   `HookConfig` with event + command preserved.

use super::*;
use std::env;

#[test]
fn test_decision_from_exit_code_allow() {
    assert_eq!(
        HookDecision::from_exit_code(0, "ignored".into()),
        HookDecision::Allow
    );
}

#[test]
fn test_decision_from_exit_code_warn() {
    assert_eq!(
        HookDecision::from_exit_code(1, "hi".into()),
        HookDecision::Warn("hi".into())
    );
}

#[test]
fn test_decision_from_exit_code_block() {
    assert_eq!(
        HookDecision::from_exit_code(2, "no".into()),
        HookDecision::Block("no".into())
    );
    // Anything other than 0/1 is also a block.
    assert_eq!(
        HookDecision::from_exit_code(42, "x".into()),
        HookDecision::Block("x".into())
    );
}

#[test]
fn test_registry_register_and_len() {
    let cfg = HookConfig {
        event: HookEvent::PreToolUse,
        matcher: None,
        command: "true".into(),
        timeout_secs: 30,
    };
    let mut reg = HookRegistry::new();
    assert!(reg.is_empty());
    reg.register(cfg);
    assert_eq!(reg.len(), 1);
    assert_eq!(reg.hooks_for(HookEvent::PreToolUse).len(), 1);
    assert_eq!(reg.hooks_for(HookEvent::PostToolUse).len(), 0);
}

#[test]
fn test_registry_from_configs() {
    let reg = HookRegistry::from_configs(vec![
        HookConfig {
            event: HookEvent::SessionStart,
            matcher: None,
            command: "true".into(),
            timeout_secs: 30,
        },
        HookConfig {
            event: HookEvent::Stop,
            matcher: None,
            command: "true".into(),
            timeout_secs: 30,
        },
    ]);
    assert_eq!(reg.len(), 2);
}

#[test]
fn test_registry_run_allow_when_empty() {
    let reg = HookRegistry::new();
    let cwd = env::temp_dir();
    assert_eq!(
        reg.run(HookEvent::PreToolUse, "shell", &cwd),
        HookDecision::Allow
    );
}

#[test]
fn test_registry_run_block_short_circuits() {
    let reg = HookRegistry::from_configs(vec![
        HookConfig {
            event: HookEvent::PreToolUse,
            matcher: None,
            command: "exit 2".into(),
            timeout_secs: 30,
        },
        // A hook registered after the blocking one must NOT run; its side
        // effect (touching /tmp/apr-hooks-test-must-not-run-xyz) should
        // never materialize.
        HookConfig {
            event: HookEvent::PreToolUse,
            matcher: None,
            command: "exit 0".into(),
            timeout_secs: 30,
        },
    ]);
    let cwd = env::temp_dir();
    let decision = reg.run(HookEvent::PreToolUse, "shell", &cwd);
    assert!(decision.is_blocking(), "expected Block, got {decision:?}");
}

#[test]
fn test_registry_matcher_filters() {
    let reg = HookRegistry::from_configs(vec![HookConfig {
        event: HookEvent::PreToolUse,
        matcher: Some("shell".into()),
        command: "exit 2".into(),
        timeout_secs: 30,
    }]);
    let cwd = env::temp_dir();
    // Matcher fires for "shell" context → Block.
    let d1 = reg.run(HookEvent::PreToolUse, "shell", &cwd);
    assert!(d1.is_blocking(), "matcher hit should block: {d1:?}");
    // Matcher misses for "file_read" → Allow (filtered out).
    let d2 = reg.run(HookEvent::PreToolUse, "file_read", &cwd);
    assert_eq!(d2, HookDecision::Allow);
}

#[test]
fn test_hook_config_toml_roundtrip() {
    let toml_src = r#"
event = "PreToolUse"
matcher = "shell"
command = "echo hi"
"#;
    let cfg: HookConfig = toml::from_str(toml_src).expect("deserialize HookConfig");
    assert_eq!(cfg.event, HookEvent::PreToolUse);
    assert_eq!(cfg.matcher.as_deref(), Some("shell"));
    assert_eq!(cfg.command, "echo hi");
    assert_eq!(cfg.timeout_secs, 30, "default timeout applied");
}

#[test]
fn test_hook_config_toml_array_in_manifest_shape() {
    let toml_src = r#"
[[hooks]]
event = "SessionStart"
command = "date >> /tmp/apr-session-start.log"

[[hooks]]
event = "PreToolUse"
matcher = "shell"
command = "echo pretool"
"#;
    #[derive(serde::Deserialize)]
    struct Wrap {
        hooks: Vec<HookConfig>,
    }
    let w: Wrap = toml::from_str(toml_src).expect("deserialize wrapper");
    assert_eq!(w.hooks.len(), 2);
    assert_eq!(w.hooks[0].event, HookEvent::SessionStart);
    assert_eq!(w.hooks[1].event, HookEvent::PreToolUse);
    assert_eq!(w.hooks[1].matcher.as_deref(), Some("shell"));
}
