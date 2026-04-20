//! Unit tests for [`super::PermissionMode`] and [`super::PermissionVerdict`].

use super::*;

fn read_cap() -> Capability {
    Capability::FileRead { allowed_paths: vec!["*".into()] }
}
fn write_cap() -> Capability {
    Capability::FileWrite { allowed_paths: vec!["*".into()] }
}
fn shell_cap() -> Capability {
    Capability::Shell { allowed_commands: vec!["*".into()] }
}
fn net_cap() -> Capability {
    Capability::Network { allowed_hosts: vec!["*".into()] }
}

#[test]
fn default_is_default_mode() {
    let m: PermissionMode = Default::default();
    assert_eq!(m, PermissionMode::Default);
}

#[test]
fn as_str_matches_claude_canonical_identifiers() {
    assert_eq!(PermissionMode::Default.as_str(), "default");
    assert_eq!(PermissionMode::Plan.as_str(), "plan");
    assert_eq!(PermissionMode::AcceptEdits.as_str(), "acceptEdits");
    assert_eq!(PermissionMode::BypassPermissions.as_str(), "bypassPermissions");
}

#[test]
fn display_round_trips_with_as_str() {
    for m in [
        PermissionMode::Default,
        PermissionMode::Plan,
        PermissionMode::AcceptEdits,
        PermissionMode::BypassPermissions,
    ] {
        assert_eq!(m.to_string(), m.as_str());
    }
}

#[test]
fn parse_accepts_canonical_camel_case() {
    assert_eq!(PermissionMode::parse("default"), Some(PermissionMode::Default));
    assert_eq!(PermissionMode::parse("plan"), Some(PermissionMode::Plan));
    assert_eq!(PermissionMode::parse("acceptEdits"), Some(PermissionMode::AcceptEdits));
    assert_eq!(PermissionMode::parse("bypassPermissions"), Some(PermissionMode::BypassPermissions));
}

#[test]
fn parse_accepts_kebab_and_snake_aliases() {
    assert_eq!(PermissionMode::parse("accept-edits"), Some(PermissionMode::AcceptEdits));
    assert_eq!(PermissionMode::parse("accept_edits"), Some(PermissionMode::AcceptEdits));
    assert_eq!(
        PermissionMode::parse("bypass-permissions"),
        Some(PermissionMode::BypassPermissions)
    );
    assert_eq!(
        PermissionMode::parse("bypass_permissions"),
        Some(PermissionMode::BypassPermissions)
    );
}

#[test]
fn parse_trims_whitespace() {
    assert_eq!(PermissionMode::parse("  plan  "), Some(PermissionMode::Plan));
}

#[test]
fn parse_rejects_unknown_string() {
    assert!(PermissionMode::parse("yolo").is_none());
    assert!(PermissionMode::parse("").is_none());
}

#[test]
fn next_cycles_in_claude_order() {
    assert_eq!(PermissionMode::Default.next(), PermissionMode::Plan);
    assert_eq!(PermissionMode::Plan.next(), PermissionMode::AcceptEdits);
    assert_eq!(PermissionMode::AcceptEdits.next(), PermissionMode::BypassPermissions);
    assert_eq!(PermissionMode::BypassPermissions.next(), PermissionMode::Default);
}

#[test]
fn bypass_allows_every_capability() {
    let m = PermissionMode::BypassPermissions;
    for cap in [read_cap(), write_cap(), shell_cap(), net_cap()] {
        assert_eq!(m.verdict(&cap), PermissionVerdict::Allow);
    }
}

#[test]
fn default_allows_read_and_asks_on_write_shell_network() {
    let m = PermissionMode::Default;
    assert_eq!(m.verdict(&read_cap()), PermissionVerdict::Allow);
    assert_eq!(m.verdict(&write_cap()), PermissionVerdict::Ask);
    assert_eq!(m.verdict(&shell_cap()), PermissionVerdict::Ask);
    assert_eq!(m.verdict(&net_cap()), PermissionVerdict::Ask);
}

#[test]
fn plan_allows_read_and_blocks_everything_mutating() {
    let m = PermissionMode::Plan;
    assert_eq!(m.verdict(&read_cap()), PermissionVerdict::Allow);
    assert_eq!(m.verdict(&write_cap()), PermissionVerdict::Block);
    assert_eq!(m.verdict(&shell_cap()), PermissionVerdict::Block);
    assert_eq!(m.verdict(&net_cap()), PermissionVerdict::Block);
}

#[test]
fn accept_edits_allows_read_and_write_but_asks_on_shell() {
    let m = PermissionMode::AcceptEdits;
    assert_eq!(m.verdict(&read_cap()), PermissionVerdict::Allow);
    assert_eq!(m.verdict(&write_cap()), PermissionVerdict::Allow);
    assert_eq!(m.verdict(&shell_cap()), PermissionVerdict::Ask);
    assert_eq!(m.verdict(&net_cap()), PermissionVerdict::Ask);
}

#[test]
fn would_run_unattended_matches_allow_verdict() {
    assert!(PermissionMode::BypassPermissions.would_run_unattended(&shell_cap()));
    assert!(PermissionMode::Default.would_run_unattended(&read_cap()));
    assert!(!PermissionMode::Default.would_run_unattended(&shell_cap()));
    assert!(!PermissionMode::Plan.would_run_unattended(&write_cap()));
}

#[test]
fn serde_round_trip_uses_camel_case() {
    let m = PermissionMode::AcceptEdits;
    let json = serde_json::to_string(&m).expect("serialize");
    assert_eq!(json, "\"acceptEdits\"");
    let back: PermissionMode = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back, m);
}

#[test]
fn memory_and_rag_auto_allowed_everywhere_except_plan_blocks_rag_none() {
    // Memory + Rag are internal substrates (local), so non-Plan modes
    // auto-allow them; Plan also auto-allows them since they don't
    // mutate the filesystem.
    for m in [
        PermissionMode::Default,
        PermissionMode::Plan,
        PermissionMode::AcceptEdits,
        PermissionMode::BypassPermissions,
    ] {
        assert_eq!(m.verdict(&Capability::Memory), PermissionVerdict::Allow, "mode={m}");
        assert_eq!(m.verdict(&Capability::Rag), PermissionVerdict::Allow, "mode={m}");
    }
}
