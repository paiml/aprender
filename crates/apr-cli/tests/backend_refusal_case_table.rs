//! R-0b (#3002, PMAT-1073, PP-066 claim 1): the resolution case table.
//!
//! Rows are generated from {host facts, via APR_REGISTRY_FIXTURE} × BACKEND_VALUES
//! ∪ {--gpu, --no-gpu, default}. The rule under test, over `apr_cli::registry`:
//!
//!   * a forced accelerator (`--gpu`, `--backend cuda|wgpu`) never resolves to
//!     cpu — it returns the selected kind or an Err;
//!   * a request the build did not compile is `FeatureDisabled` (exit 9);
//!   * a request the build compiled but this host has not Ready is
//!     `BackendUnavailable` (exit 14);
//!   * `--no-gpu` / `--backend cpu` / default-on-a-cpu-host resolve to cpu.
//!
//! The fixtures are R-0a's: cpu-only (nothing Ready but cpu), one-cuda (a Ready
//! cuda), two-vendors (a Ready cuda and a Ready wgpu). A build without the
//! `inference` feature has no registry crate; the stub answers cpu / refuse and
//! is covered by the unit tests in registry.rs.
#![cfg(feature = "inference")]

use apr_cli::error::CliError;
use apr_cli::registry::{resolve_in, Request};
use trueno::registry::BackendRegistry;

fn reg(fixture: &str) -> BackendRegistry {
    let path = format!(
        "{}/tests/fixtures/registry/{fixture}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    BackendRegistry::from_fixture_json(&text, &path).expect("fixture parses")
}

fn req(gpu: bool, no_gpu: bool, backend: Option<&str>) -> Request<'_> {
    Request {
        gpu,
        no_gpu,
        backend,
        layers_want_accelerator: false,
    }
}

#[test]
fn a_forced_accelerator_never_resolves_to_cpu() {
    // one-cuda: cuda is Ready → --gpu and --backend cuda select cuda, not cpu.
    let r = reg("one-cuda");
    for (asked, request) in [
        ("--gpu", req(true, false, None)),
        ("--backend cuda", req(false, false, Some("cuda"))),
    ] {
        let sel = resolve_in(&request, asked, &r).expect("cuda is Ready in one-cuda");
        assert_eq!(sel.kind, "cuda", "{asked} must select cuda, never cpu");
    }
}

#[test]
fn a_forced_accelerator_with_none_ready_refuses_and_never_downgrades() {
    // cpu-only: cuda dlopen present but no device; wgpu compiled; nothing Ready.
    let r = reg("cpu-only");
    for (asked, request) in [
        ("--gpu", req(true, false, None)),
        ("--backend cuda", req(false, false, Some("cuda"))),
        ("--backend wgpu", req(false, false, Some("wgpu"))),
    ] {
        let e = resolve_in(&request, asked, &r).expect_err("nothing Ready — must refuse, not cpu");
        assert!(
            matches!(
                e,
                CliError::FeatureDisabled(_) | CliError::BackendUnavailable(_)
            ),
            "{asked} must refuse with a backend error, got {e}"
        );
        // exit code is 9 (not compiled) or 14 (not Ready) — never 0, never a cpu run.
        assert!(
            matches!(e.exit_code_value(), 9 | 14),
            "{asked}: exit {} not a refusal",
            e.exit_code_value()
        );
    }
}

#[test]
fn cpu_and_no_gpu_and_default_resolve_to_cpu() {
    let r = reg("cpu-only");
    for (asked, request) in [
        ("--no-gpu", req(true, true, None)), // --no-gpu wins over --gpu
        ("--backend cpu", req(false, false, Some("cpu"))),
        ("default", req(false, false, None)), // no Ready accelerator → cpu
    ] {
        let sel = resolve_in(&request, asked, &r).expect("cpu always resolves");
        assert_eq!(sel.kind, "cpu", "{asked} must resolve to cpu");
        assert!(
            !sel.reason.is_empty(),
            "{asked}: the cpu selection must say why"
        );
    }
}

#[test]
fn the_default_takes_a_ready_accelerator_when_there_is_one() {
    // two-vendors: cuda Ready first → default selects it (REG-8), no flag needed.
    let sel =
        resolve_in(&req(false, false, None), "default", &reg("two-vendors")).expect("resolves");
    assert_ne!(
        sel.kind, "cpu",
        "a Ready accelerator is the default when present"
    );
}

#[test]
fn every_backend_value_is_resolvable_on_every_fixture_and_forced_gpu_is_never_cpu() {
    for fixture in ["cpu-only", "one-cuda", "two-vendors"] {
        let r = reg(fixture);
        for backend in apr_cli::BACKEND_VALUES {
            let asked = format!("--backend {backend}");
            match resolve_in(&req(false, false, Some(backend)), &asked, &r) {
                Ok(sel) if backend == "cpu" => assert_eq!(sel.kind, "cpu"),
                Ok(sel) => assert_eq!(sel.kind, backend, "{fixture}/{asked}: Ready ⇒ that kind"),
                Err(e) => {
                    assert_ne!(backend, "cpu", "{fixture}: cpu never refuses");
                    assert!(
                        matches!(
                            e,
                            CliError::FeatureDisabled(_) | CliError::BackendUnavailable(_)
                        ),
                        "{fixture}/{asked}: {e}"
                    );
                }
            }
        }
    }
}
