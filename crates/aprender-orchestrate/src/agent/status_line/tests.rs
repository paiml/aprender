//! Unit tests for [`super::StatusLine`] and [`super::short_cwd`].

use std::path::{Path, PathBuf};

use super::*;

fn baseline() -> StatusLine {
    StatusLine {
        model: "opus-4-7".into(),
        mode: "default".into(),
        cost_usd: 0.0,
        branch: None,
        cwd_short: None,
    }
}

#[test]
fn render_shows_model_mode_and_cost_when_optionals_absent() {
    let r = baseline().render();
    assert_eq!(r, "opus-4-7 | [default] | $0.00");
}

#[test]
fn render_includes_branch_when_present() {
    let mut s = baseline();
    s.branch = Some("feat/foo".into());
    assert_eq!(s.render(), "opus-4-7 | [default] | $0.00 | feat/foo");
}

#[test]
fn render_includes_cwd_short_when_present() {
    let mut s = baseline();
    s.cwd_short = Some("~/src/aprender".into());
    assert_eq!(s.render(), "opus-4-7 | [default] | $0.00 | ~/src/aprender");
}

#[test]
fn render_orders_columns_as_model_mode_cost_branch_cwd() {
    let s = StatusLine {
        model: "qwen2.5-coder-7b".into(),
        mode: "bypassPermissions".into(),
        cost_usd: 0.12345,
        branch: Some("main".into()),
        cwd_short: Some("~/src/aprender".into()),
    };
    assert_eq!(
        s.render(),
        "qwen2.5-coder-7b | [bypassPermissions] | $0.12 | main | ~/src/aprender"
    );
}

#[test]
fn render_truncates_cost_to_two_decimals() {
    let mut s = baseline();
    s.cost_usd = 1.23789;
    assert!(s.render().contains("$1.24"));
    s.cost_usd = 0.0034;
    assert!(s.render().contains("$0.00"));
    s.cost_usd = 12.5;
    assert!(s.render().contains("$12.50"));
}

#[test]
fn build_converts_permission_mode_to_canonical_string() {
    let s = StatusLine::build("m", PermissionMode::AcceptEdits, 0.0, None, None);
    assert_eq!(s.mode, "acceptEdits");
}

#[test]
fn build_passes_through_optionals_untouched() {
    let s =
        StatusLine::build("m", PermissionMode::Plan, 0.5, Some("br".into()), Some("~/x".into()));
    assert_eq!(s.branch.as_deref(), Some("br"));
    assert_eq!(s.cwd_short.as_deref(), Some("~/x"));
    assert_eq!(s.cost_usd, 0.5);
}

#[test]
fn short_cwd_replaces_home_prefix_with_tilde_slash() {
    let home = PathBuf::from("/home/alice");
    let cwd = PathBuf::from("/home/alice/src/aprender");
    assert_eq!(short_cwd(&cwd, Some(&home)), "~/src/aprender");
}

#[test]
fn short_cwd_returns_lone_tilde_when_cwd_is_home() {
    let home = PathBuf::from("/home/alice");
    let cwd = PathBuf::from("/home/alice");
    assert_eq!(short_cwd(&cwd, Some(&home)), "~");
}

#[test]
fn short_cwd_leaves_non_home_paths_verbatim() {
    let home = PathBuf::from("/home/alice");
    let cwd = PathBuf::from("/etc/nginx");
    assert_eq!(short_cwd(&cwd, Some(&home)), "/etc/nginx");
}

#[test]
fn short_cwd_falls_back_when_home_is_none() {
    let cwd = PathBuf::from("/var/log");
    assert_eq!(short_cwd(&cwd, None), "/var/log");
}

#[test]
fn render_is_pure_and_does_not_mutate_self() {
    let s = baseline();
    let a = s.render();
    let b = s.render();
    assert_eq!(a, b);
}

#[test]
fn roundtrip_preserves_fields() {
    let s = StatusLine {
        model: "m".into(),
        mode: "plan".into(),
        cost_usd: 0.42,
        branch: Some("b".into()),
        cwd_short: Some("c".into()),
    };
    let clone = s.clone();
    assert_eq!(s, clone);
}

#[test]
fn short_cwd_handles_trailing_separator_in_home() {
    let home = PathBuf::from("/home/alice");
    let cwd: &Path = Path::new("/home/alice/src");
    assert_eq!(short_cwd(cwd, Some(&home)), "~/src");
}
