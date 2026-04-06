use super::*;
use crate::playbook::cache;
use crate::playbook::types::*;
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_PB005_run_config_defaults() {
    let config = RunConfig {
        playbook_path: PathBuf::from("/tmp/test.yaml"),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };
    assert!(!config.force);
    assert!(!config.dry_run);
    assert!(config.stage_filter.is_none());
}

#[tokio::test]
async fn test_PB005_execute_command_success() {
    let result = execute_command("echo hello").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_PB005_execute_command_failure() {
    let result = execute_command("exit 1").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    let cmd_err = err.downcast_ref::<CommandError>().expect("downcast failed");
    assert_eq!(cmd_err.exit_code, Some(1));
}

#[tokio::test]
async fn test_PB005_execute_command_with_output() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out = dir.path().join("out.txt");
    let cmd = format!("echo hello > {}", out.display());
    execute_command(&cmd).await.expect("async operation failed");
    let content = std::fs::read_to_string(&out).expect("fs read failed");
    assert_eq!(content.trim(), "hello");
}

#[test]
fn test_PB005_validate_only() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        r#"
version: "1.0"
name: test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs:
      - path: /tmp/out.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
    )
    .expect("unexpected failure");

    let (pb, warnings) = validate_only(&yaml_path).expect("unexpected failure");
    assert_eq!(pb.name, "test");
    assert!(warnings.is_empty());
}

#[test]
fn test_PB005_validate_only_with_cycle() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        r#"
version: "1.0"
name: cycle
params: {}
targets: {}
stages:
  a:
    cmd: "echo a"
    deps:
      - path: /tmp/b.txt
    outs:
      - path: /tmp/a.txt
  b:
    cmd: "echo b"
    deps:
      - path: /tmp/a.txt
    outs:
      - path: /tmp/b.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
    )
    .expect("unexpected failure");

    let err = validate_only(&yaml_path).unwrap_err();
    assert!(err.to_string().contains("cycle"));
}

#[tokio::test]
async fn test_PB005_run_simple_pipeline() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_dir = dir.path().join("outputs");
    std::fs::create_dir(&out_dir).expect("unexpected failure");

    let out_file = out_dir.join("hello.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: simple-test
params: {{}}
targets: {{}}
stages:
  hello:
    cmd: "echo hello > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
            out_file.display(),
            out_file.display(),
        ),
    )
    .expect("unexpected failure");

    let config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };

    let result = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(result.stages_run, 1);
    assert_eq!(result.stages_cached, 0);
    assert_eq!(result.stages_failed, 0);

    // Output file should exist
    assert!(out_file.exists());
    assert_eq!(std::fs::read_to_string(&out_file).expect("fs read failed").trim(), "hello");

    // Lock file should exist
    let lock_path = cache::lock_file_path(&yaml_path);
    assert!(lock_path.exists());
}

#[tokio::test]
async fn test_PB005_cached_rerun() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_file = dir.path().join("out.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: cache-test
params: {{}}
targets: {{}}
stages:
  write:
    cmd: "echo cached > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
            out_file.display(),
            out_file.display(),
        ),
    )
    .expect("unexpected failure");

    let config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };

    // First run: executes
    let r1 = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(r1.stages_run, 1);
    assert_eq!(r1.stages_cached, 0);

    // Second run: cached
    let r2 = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(r2.stages_run, 0);
    assert_eq!(r2.stages_cached, 1);
}

#[tokio::test]
async fn test_PB005_jidoka_stop_on_failure() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        r#"
version: "1.0"
name: fail-test
params: {}
targets: {}
stages:
  fail:
    cmd: "exit 1"
    deps: []
    outs: []
  after_fail:
    cmd: "echo should-not-run"
    deps: []
    outs: []
    after:
      - fail
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
    )
    .expect("unexpected failure");

    let config = RunConfig {
        playbook_path: yaml_path,
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };

    let err = run_playbook(&config).await.unwrap_err();
    assert!(err.to_string().contains("Jidoka"));
}

#[test]
fn test_PB005_show_status_no_lock() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        r#"
version: "1.0"
name: status-test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs: []
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
    )
    .expect("unexpected failure");

    // No lock file exists — should print "No lock file found"
    let result = show_status(&yaml_path);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_PB005_show_status_with_lock() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_file = dir.path().join("out.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: status-lock-test
params: {{}}
targets: {{}}
stages:
  hello:
    cmd: "echo hi > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
            out_file.display(),
            out_file.display(),
        ),
    )
    .expect("unexpected failure");

    // Run to create lock file
    let config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };
    run_playbook(&config).await.expect("async operation failed");

    // Now show_status should display lock info
    let result = show_status(&yaml_path);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_PB005_frozen_stage_cached() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_file = dir.path().join("frozen_out.txt");
    // Pre-create the output to avoid "output missing" issues
    std::fs::write(&out_file, "frozen content").expect("fs write failed");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: frozen-test
params: {{}}
targets: {{}}
stages:
  frozen_stage:
    cmd: "echo should-not-run > {}"
    deps: []
    outs:
      - path: {}
    frozen: true
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
            out_file.display(),
            out_file.display(),
        ),
    )
    .expect("unexpected failure");

    let config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };

    let result = run_playbook(&config).await.expect("async operation failed");
    // Frozen stage should be cached without ever running
    assert_eq!(result.stages_cached, 1);
    assert_eq!(result.stages_run, 0);
    // Output should still have original content (command never ran)
    assert_eq!(std::fs::read_to_string(&out_file).expect("fs read failed"), "frozen content");
}

#[tokio::test]
async fn test_PB005_frozen_stage_force_overrides() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_file = dir.path().join("frozen_force.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: frozen-force-test
params: {{}}
targets: {{}}
stages:
  frozen_stage:
    cmd: "echo forced > {}"
    deps: []
    outs:
      - path: {}
    frozen: true
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
            out_file.display(),
            out_file.display(),
        ),
    )
    .expect("unexpected failure");

    let config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: true,
        dry_run: false,
        param_overrides: HashMap::new(),
    };

    let result = run_playbook(&config).await.expect("async operation failed");
    // --force should override frozen
    assert_eq!(result.stages_run, 1);
    assert_eq!(result.stages_cached, 0);
    assert_eq!(std::fs::read_to_string(&out_file).expect("fs read failed").trim(), "forced");
}
