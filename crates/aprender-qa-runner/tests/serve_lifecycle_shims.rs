//! `ToolExecutor::execute_serve_lifecycle` (F-INTEG-003) against shim `apr` and `curl`.
//!
//! The function spawns `apr serve` and two `curl` calls by bare name, so each case puts
//! a throwaway directory of shell shims first (and only) on `PATH`. `PATH` is process
//! global, so the cases serialise on a lock.
//!
//! Refs: #5722

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Mutex;

use aprender_qa_runner::ToolExecutor;

static PATH_LOCK: Mutex<()> = Mutex::new(());

fn shim(dir: &Path, name: &str, body: &str) {
    let p = dir.join(name);
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

/// Run the lifecycle with `PATH` = `dir` alone.
fn run(dir: &Path, no_gpu: bool) -> aprender_qa_runner::ToolTestResult {
    let _g = PATH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old = std::env::var_os("PATH");
    std::env::set_var("PATH", dir);
    let r = ToolExecutor::new("model.gguf".to_string(), no_gpu, 5000).execute_serve_lifecycle();
    match old {
        Some(p) => std::env::set_var("PATH", p),
        None => std::env::remove_var("PATH"),
    }
    r
}

/// An `apr` that records its argv to `args` and announces readiness on stderr.
fn ready_apr(dir: &Path) -> std::path::PathBuf {
    let args = dir.join("args.txt");
    shim(
        dir,
        "apr",
        &format!(
            "echo \"$@\" > {}\necho 'Listening on 18080' >&2\nexec sleep 30",
            args.display()
        ),
    );
    args
}

#[test]
fn a_missing_apr_binary_is_reported_as_failed_to_start() {
    let dir = tempfile::tempdir().unwrap();
    let r = run(dir.path(), false);
    assert!(!r.passed);
    assert_eq!(r.exit_code, -1);
    assert!(
        r.stderr.starts_with("Failed to start server:"),
        "{}",
        r.stderr
    );
    assert!(r.stdout.is_empty());
    assert_eq!(
        (r.tool.as_str(), r.gate_id.as_str()),
        ("serve-lifecycle", "F-INTEG-003")
    );
}

#[test]
fn health_and_inference_both_ok_passes() {
    let dir = tempfile::tempdir().unwrap();
    let args = ready_apr(dir.path());
    shim(dir.path(), "curl", "exit 0");
    let r = run(dir.path(), false);
    assert!(r.passed, "{}", r.stderr);
    assert_eq!(r.exit_code, 0);
    assert_eq!(r.stdout, "Health check: OK\nInference: OK");
    assert!(r.stderr.is_empty());
    let argv = std::fs::read_to_string(args).unwrap();
    assert_eq!(argv.trim(), "serve model.gguf --port 18080");
}

#[test]
fn no_gpu_is_forwarded_to_apr_serve() {
    let dir = tempfile::tempdir().unwrap();
    let args = ready_apr(dir.path());
    shim(dir.path(), "curl", "exit 0");
    let _ = run(dir.path(), true);
    let argv = std::fs::read_to_string(args).unwrap();
    assert_eq!(argv.trim(), "serve model.gguf --port 18080 --no-gpu");
}

#[test]
fn health_ok_but_inference_failing_fails_with_the_split_in_stderr() {
    let dir = tempfile::tempdir().unwrap();
    ready_apr(dir.path());
    shim(
        dir.path(),
        "curl",
        "case \"$*\" in *chat/completions*) exit 22;; esac\nexit 0",
    );
    let r = run(dir.path(), false);
    assert!(!r.passed);
    assert_eq!(r.exit_code, 1);
    assert_eq!(r.stdout, "Health check: OK\nInference: FAILED");
    assert_eq!(
        r.stderr,
        "Serve lifecycle incomplete: health=true, inference=false"
    );
}

#[test]
fn both_probes_failing_reports_both_failed() {
    let dir = tempfile::tempdir().unwrap();
    ready_apr(dir.path());
    shim(dir.path(), "curl", "exit 22");
    let r = run(dir.path(), false);
    assert!(!r.passed);
    assert_eq!(r.stdout, "Health check: FAILED\nInference: FAILED");
    assert_eq!(
        r.stderr,
        "Serve lifecycle incomplete: health=false, inference=false"
    );
}

#[test]
fn a_server_that_never_says_listening_still_gets_probed_after_the_grace_wait() {
    let dir = tempfile::tempdir().unwrap();
    shim(dir.path(), "apr", "echo booting >&2\nexec sleep 30");
    shim(dir.path(), "curl", "exit 0");
    let r = run(dir.path(), false);
    assert!(r.passed, "{}", r.stderr);
    assert!(
        r.duration_ms >= 2000,
        "grace sleep skipped: {} ms",
        r.duration_ms
    );
}
