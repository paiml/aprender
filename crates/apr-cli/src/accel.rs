//! PERF-021: the accelerator refusal, shared by every CLI surface.
//!
//! WHY THIS MODULE EXISTS. `ensure_accelerator_available` lived inside
//! `commands::serve` and took a `&ServerConfig`, so only `apr serve` could call
//! it — and only `apr serve` did. `apr run` and `apr chat` accept the same
//! accelerator request, verify nothing, and run on CPU.
//!
//! That is not a hypothetical: #2696's headline measurement was taken through
//! `apr run` — the surface with NO guard. The jidoka fix landed on the one
//! surface where the defect had not been measured.
//!
//! The figures live in #2696 and are deliberately NOT repeated here. This is a
//! doc comment on shipped code, so `cargo doc` renders it to users, which makes
//! it a user-facing surface under §9 — and `check_no_claim_literals.sh` caught
//! an earlier draft that quoted the tok/s and the comparator ratio inline. That
//! guard is right: a measured number reachable from `cargo doc` is a published
//! claim, and a claim belongs where its receipt is.
//!
//! A refusal that lives in one command's private module is a refusal one
//! surface deep. This takes plain values instead of a config type so all three
//! surfaces can share exactly one implementation, and one message.

use crate::error::{CliError, Result};

/// True when this build carries a GPU backend that could honour a request.
#[must_use]
pub(crate) fn build_has_accelerator() -> bool {
    // R-0b (#3002): the registry says what this build compiled; never `cfg!`.
    crate::registry::build_has_accelerator()
}

/// Refuse an accelerator request this build cannot honour.
///
/// `asked` is the flag the USER typed, quoted back verbatim. Telling someone
/// who typed `--gpu` about `--gpu-layers` sends them to a flag they did not
/// use, so the caller passes what it saw.
///
/// I-17, EXPLICIT WINS: an explicit request is refused loudly rather than
/// quietly downgraded. Automation overriding an explicit user instruction is
/// the v2.2 root cause of defect #1 (§7.5 N5), and a silent CPU fallback is
/// exactly that override wearing a performance number.
///
/// # Errors
/// [`CliError::FeatureDisabled`] when `wants_accelerator` and the build has none.
/// R-0b "selected: always": resolve the request exactly as `apr run` / `apr
/// chat` honour it — GH-326 `--gpu` overrides `--no-gpu`, and `--gpu` also
/// overrides `--backend cpu` (that is what `effective_no_gpu` does downstream,
/// so the line must say the same) — announce the selection, and refuse a forced
/// accelerator this host cannot honour. Nothing can refuse a cpu request.
pub(crate) fn ensure_available_for(gpu: bool, no_gpu: bool, backend: Option<&str>) -> Result<()> {
    let backend = if gpu {
        backend.filter(|b| *b != "cpu")
    } else {
        backend
    };
    let no_gpu = no_gpu && !gpu;
    let wants = gpu || matches!(backend, Some("cuda" | "wgpu" | "gpu"));
    let asked = if wants {
        asked_flag(gpu, backend)
    } else if no_gpu {
        "--no-gpu".to_string()
    } else if backend == Some("cpu") {
        "--backend cpu".to_string()
    } else {
        "default".to_string()
    };
    let req = crate::registry::Request {
        gpu,
        no_gpu,
        backend,
        layers_want_accelerator: false,
    };
    crate::registry::announce(&req, &asked).map(|_| ())
}

pub(crate) fn ensure_available(wants_accelerator: bool, asked: &str) -> Result<()> {
    if !wants_accelerator {
        return Ok(());
    }
    // R-0b: resolve the request the user typed against the registry. A forced
    // kind that is not Ready refuses (FeatureDisabled when not compiled,
    // BackendUnavailable when compiled but absent here); it never downgrades.
    let req = request_from_asked(asked);
    crate::registry::announce(&req, asked).map(|_| ())
}

/// The request behind the flag text a caller quotes back (`--gpu`,
/// `--gpu-layers`, `--backend <kind>`).
pub(crate) fn request_from_asked(asked: &str) -> crate::registry::Request<'_> {
    match asked.strip_prefix("--backend ") {
        Some(kind) => crate::registry::Request {
            backend: Some(kind.trim()),
            ..Default::default()
        },
        None => crate::registry::Request {
            gpu: true,
            ..Default::default()
        },
    }
}

/// Which flag the user actually typed, for quoting back.
#[must_use]
pub(crate) fn asked_flag(gpu: bool, backend: Option<&str>) -> String {
    if gpu {
        "--gpu".to_string()
    } else if let Some(b) = backend.filter(|b| *b != "cpu") {
        format!("--backend {b}")
    } else {
        "--gpu".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_this_build_cannot_honour_is_refused() {
        if build_has_accelerator() {
            // On a GPU build the request IS honourable; assert that, so this
            // test says something on both builds rather than silently skipping.
            assert!(ensure_available(true, "--gpu").is_ok());
            return;
        }
        let e = ensure_available(true, "--gpu").expect_err("must refuse");
        let m = e.to_string();
        assert!(m.contains("2696"), "the refusal must cite the defect: {m}");
        assert!(
            m.contains("--no-gpu"),
            "and offer the deliberate CPU path: {m}"
        );
    }

    #[test]
    fn no_request_is_never_refused() {
        assert!(ensure_available(false, "--gpu").is_ok());
    }

    /// I-18 / THE THREE-SURFACE GATE.
    ///
    /// Every accelerator-accepting surface must call the refusal. The original
    /// defect was not that the refusal was wrong — it was correct — but that it
    /// existed on ONE of three surfaces, and not the one #2696 was measured
    /// through. This repo has a documented "CLI 3-surface drift" defect class;
    /// this is that class, with a measurement attached.
    ///
    /// A source scan, for the same reason `the_guard_is_actually_wired_into_run`
    /// in serve/mod.rs is one: `dispatch_runtime_commands` needs a parsed CLI
    /// and a model file, so a unit test cannot drive it, and every unit test in
    /// this module passes with the call sites deleted.
    #[test]
    fn every_accelerator_surface_calls_the_refusal() {
        // S3b (#3041): run and chat call the REGISTRY-resolving entry point BY
        // NAME. The old needle `accel::ensure_available` is deliberately not
        // accepted: it is a substring of `ensure_available_for`, so a scan for
        // it passed identically before and after this slice and witnessed
        // nothing. serve keeps its own named wrapper.
        let surfaces: [(&str, &str, &str); 3] = [
            (
                "apr run (dispatch.rs)",
                include_str!("dispatch.rs"),
                "accel::ensure_available_for(",
            ),
            (
                "apr chat (dispatch_analysis.rs)",
                include_str!("dispatch_analysis.rs"),
                "accel::ensure_available_for(",
            ),
            (
                "apr serve (commands/serve/mod.rs)",
                include_str!("commands/serve/mod.rs"),
                "ensure_accelerator_available(config)?",
            ),
        ];
        let mut missing = Vec::new();
        for (name, src, needle) in surfaces {
            if !src.contains(needle) {
                missing.push(name);
            }
        }
        assert!(
            missing.is_empty(),
            "these surfaces accept an accelerator request and never verify a \
             backend exists, so `--gpu` is silently ignored there (#2696 was \
             measured through `apr run`, which was one of them): {missing:?}"
        );
    }

    /// R-0b/S3b: NOTHING that asks for CPU may be refused, in any spelling.
    /// `ensure_available_for` is now the one preflight `apr run` and `apr chat`
    /// share, so a regression here is a refusal on a plain `apr run model.gguf`.
    #[test]
    fn a_cpu_or_default_request_is_never_refused_in_any_spelling() {
        for (gpu, no_gpu, backend) in [
            (false, false, None),
            (false, true, None),
            (false, false, Some("cpu")),
            (false, true, Some("cpu")),
        ] {
            assert!(
                ensure_available_for(gpu, no_gpu, backend).is_ok(),
                "a cpu/default request must never be refused: \
                 gpu={gpu} no_gpu={no_gpu} backend={backend:?}"
            );
        }
    }

    /// GH-326 (`--gpu` beats `--no-gpu`) and its twin (`--gpu` beats `--backend
    /// cpu`), asserted at the resolution boundary rather than at the four call
    /// sites that used to each re-derive it.
    ///
    /// Both outcomes are asserted so the test says something on EVERY host: a
    /// build/host with an accelerator resolves, one without refuses. The one
    /// thing it may never do is quietly become a cpu run — that is #2696.
    #[test]
    fn a_forced_gpu_request_is_honoured_or_refused_never_quietly_made_cpu() {
        for backend in [None, Some("cpu"), Some("gpu")] {
            match ensure_available_for(true, true, backend) {
                Ok(()) => assert!(
                    build_has_accelerator(),
                    "--gpu resolved on a build the registry says has no accelerator"
                ),
                Err(e) => {
                    let m = e.to_string();
                    assert!(
                        m.contains("--gpu"),
                        "the refusal quotes the flag typed: {m}"
                    );
                    assert!(
                        m.contains("--no-gpu"),
                        "and offers the deliberate CPU path: {m}"
                    );
                }
            }
        }
    }

    /// `ensure_available` keeps taking the flag TEXT a caller quotes back, so
    /// the text has to map onto the same request the registry resolves.
    #[test]
    fn the_flag_text_maps_back_onto_the_request_it_came_from() {
        assert!(request_from_asked("--gpu").gpu);
        assert_eq!(request_from_asked("--gpu").backend, None);
        assert!(request_from_asked("--gpu-layers all").gpu);
        assert_eq!(request_from_asked("--backend cuda").backend, Some("cuda"));
        assert!(!request_from_asked("--backend cuda").gpu);
        assert_eq!(request_from_asked("--backend wgpu ").backend, Some("wgpu"));
    }

    #[test]
    fn the_flag_quoted_back_is_the_one_the_user_typed() {
        assert_eq!(asked_flag(true, None), "--gpu");
        assert_eq!(asked_flag(false, Some("cuda")), "--backend cuda");
        assert_eq!(asked_flag(false, Some("cpu")), "--gpu");
    }
}
