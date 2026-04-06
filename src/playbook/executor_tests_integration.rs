//! Integration tests for the playbook executor (targets, filters, force, status).

use super::*;
use crate::playbook::cache;
use crate::playbook::types::*;
use indexmap::IndexMap;
use std::collections::HashMap;

#[tokio::test]
async fn test_PB005_remote_target_rejected() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        r#"
version: "1.0"
name: remote-test
params: {}
targets:
  gpu-box:
    host: "gpu-box.local"
    ssh_user: noah
stages:
  remote_stage:
    cmd: "echo remote"
    deps: []
    outs: []
    target: gpu-box
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
    assert!(err.to_string().contains("Remote execution requires Phase 2"));
}

#[tokio::test]
async fn test_PB005_localhost_target_allowed() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_file = dir.path().join("local.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: localhost-test
params: {{}}
targets:
  local:
    host: "localhost"
stages:
  local_stage:
    cmd: "echo local > {}"
    deps: []
    outs:
      - path: {}
    target: local
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
        playbook_path: yaml_path,
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };

    let result = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(result.stages_run, 1);
    assert_eq!(std::fs::read_to_string(&out_file).expect("fs read failed").trim(), "local");
}

#[tokio::test]
async fn test_PB005_127_0_0_1_target_allowed() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_file = dir.path().join("loopback.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: loopback-test
params: {{}}
targets:
  loopback:
    host: "127.0.0.1"
stages:
  loop_stage:
    cmd: "echo loopback > {}"
    deps: []
    outs:
      - path: {}
    target: loopback
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
        playbook_path: yaml_path,
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };

    let result = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(result.stages_run, 1);
}

#[tokio::test]
async fn test_PB005_stage_filter() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out1 = dir.path().join("one.txt");
    let out2 = dir.path().join("two.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: filter-test
params: {{}}
targets: {{}}
stages:
  one:
    cmd: "echo one > {}"
    deps: []
    outs:
      - path: {}
  two:
    cmd: "echo two > {}"
    deps: []
    outs:
      - path: {}
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
            out1.display(),
            out1.display(),
            out2.display(),
            out2.display(),
        ),
    )
    .expect("unexpected failure");

    let config = RunConfig {
        playbook_path: yaml_path,
        stage_filter: Some(vec!["one".to_string()]),
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };

    let result = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(result.stages_run, 1);
    assert!(out1.exists());
    assert!(!out2.exists()); // Stage "two" was filtered out
}

#[tokio::test]
async fn test_PB005_force_rerun() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_file = dir.path().join("force.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: force-test
params: {{}}
targets: {{}}
stages:
  write:
    cmd: "echo force > {}"
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

    // First run
    let config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };
    let r1 = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(r1.stages_run, 1);

    // Second run without force — cached
    let r2 = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(r2.stages_cached, 1);
    assert_eq!(r2.stages_run, 0);

    // Third run with force — re-runs
    let force_config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: true,
        dry_run: false,
        param_overrides: HashMap::new(),
    };
    let r3 = run_playbook(&force_config).await.expect("async operation failed");
    assert_eq!(r3.stages_run, 1);
    assert_eq!(r3.stages_cached, 0);
}

#[tokio::test]
async fn test_PB005_execute_command_stderr() {
    let result = execute_command("echo error >&2 && exit 42").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    let cmd_err = err.downcast_ref::<CommandError>().expect("downcast failed");
    assert_eq!(cmd_err.exit_code, Some(42));
    assert!(cmd_err.stderr.contains("error"));
}

#[test]
fn test_PB005_show_status_all_stage_statuses() {
    // Test that show_status handles all StageStatus variants in lock
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        r#"
version: "1.0"
name: multi-status
params: {}
targets: {}
stages:
  completed:
    cmd: "echo a"
    deps: []
    outs: []
  failed:
    cmd: "echo b"
    deps: []
    outs: []
  running:
    cmd: "echo c"
    deps: []
    outs: []
  pending:
    cmd: "echo d"
    deps: []
    outs: []
  hashing:
    cmd: "echo e"
    deps: []
    outs: []
  validating:
    cmd: "echo f"
    deps: []
    outs: []
  not_run:
    cmd: "echo g"
    deps: []
    outs: []
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#,
    )
    .expect("unexpected failure");

    // Manually create a lock file with various statuses
    let lock = LockFile {
        schema: "1.0".to_string(),
        playbook: "multi-status".to_string(),
        generated_at: "2026-02-16T14:00:00Z".to_string(),
        generator: "batuta test".to_string(),
        blake3_version: "1.8".to_string(),
        params_hash: None,
        stages: IndexMap::from([
            (
                "completed".to_string(),
                StageLock {
                    status: StageStatus::Completed,
                    started_at: None,
                    completed_at: None,
                    duration_seconds: Some(1.5),
                    target: None,
                    deps: vec![],
                    params_hash: None,
                    outs: vec![],
                    cmd_hash: None,
                    cache_key: None,
                },
            ),
            (
                "failed".to_string(),
                StageLock {
                    status: StageStatus::Failed,
                    started_at: None,
                    completed_at: None,
                    duration_seconds: Some(0.3),
                    target: None,
                    deps: vec![],
                    params_hash: None,
                    outs: vec![],
                    cmd_hash: None,
                    cache_key: None,
                },
            ),
            (
                "running".to_string(),
                StageLock {
                    status: StageStatus::Running,
                    started_at: None,
                    completed_at: None,
                    duration_seconds: None,
                    target: None,
                    deps: vec![],
                    params_hash: None,
                    outs: vec![],
                    cmd_hash: None,
                    cache_key: None,
                },
            ),
            (
                "pending".to_string(),
                StageLock {
                    status: StageStatus::Pending,
                    started_at: None,
                    completed_at: None,
                    duration_seconds: None,
                    target: None,
                    deps: vec![],
                    params_hash: None,
                    outs: vec![],
                    cmd_hash: None,
                    cache_key: None,
                },
            ),
            (
                "hashing".to_string(),
                StageLock {
                    status: StageStatus::Hashing,
                    started_at: None,
                    completed_at: None,
                    duration_seconds: None,
                    target: None,
                    deps: vec![],
                    params_hash: None,
                    outs: vec![],
                    cmd_hash: None,
                    cache_key: None,
                },
            ),
            (
                "validating".to_string(),
                StageLock {
                    status: StageStatus::Validating,
                    started_at: None,
                    completed_at: None,
                    duration_seconds: None,
                    target: None,
                    deps: vec![],
                    params_hash: None,
                    outs: vec![],
                    cmd_hash: None,
                    cache_key: None,
                },
            ),
        ]),
    };

    cache::save_lock_file(&lock, &yaml_path).expect("unexpected failure");
    let result = show_status(&yaml_path);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_PB005_param_overrides() {
    let dir = tempfile::tempdir().expect("tempdir creation failed");
    let out_file = dir.path().join("param_out.txt");
    let yaml_path = dir.path().join("test.yaml");
    std::fs::write(
        &yaml_path,
        format!(
            r#"
version: "1.0"
name: param-override-test
params:
  greeting: "default"
targets: {{}}
stages:
  write:
    cmd: "echo '{{{{params.greeting}}}}' > {}"
    deps: []
    outs:
      - path: {}
    params:
      - greeting
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

    // First run with default param
    let config = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: HashMap::new(),
    };
    let r1 = run_playbook(&config).await.expect("async operation failed");
    assert_eq!(r1.stages_run, 1);
    assert_eq!(std::fs::read_to_string(&out_file).expect("fs read failed").trim(), "default");

    // Second run with override
    let mut overrides = HashMap::new();
    overrides
        .insert("greeting".to_string(), serde_yaml_ng::Value::String("overridden".to_string()));
    let config2 = RunConfig {
        playbook_path: yaml_path.clone(),
        stage_filter: None,
        force: false,
        dry_run: false,
        param_overrides: overrides,
    };
    let r2 = run_playbook(&config2).await.expect("async operation failed");
    // Should re-run because param changed
    assert_eq!(r2.stages_run, 1);
    assert_eq!(std::fs::read_to_string(&out_file).expect("fs read failed").trim(), "overridden");
}
