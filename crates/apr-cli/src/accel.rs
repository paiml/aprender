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
    cfg!(any(feature = "cuda", feature = "wgpu"))
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
pub(crate) fn ensure_available(wants_accelerator: bool, asked: &str) -> Result<()> {
    if !wants_accelerator || build_has_accelerator() {
        return Ok(());
    }
    Err(CliError::FeatureDisabled(format!(
        "{asked} was requested, but this build has no GPU backend compiled in, \n\
         so it would have run on CPU without telling you. On a 7B Q4_K_M \n\
         model that is roughly a tenth of the decode rate and several seconds of \n\
         extra latency to the first token (aprender#2696).\n\
         \n\
         Install a build that has one:\n\
         \n\
        \x20    cargo install aprender --features cuda    # NVIDIA\n\
        \x20    cargo install aprender --features wgpu    # portable GPU backend\n\
         \n\
         Or pass --no-gpu to run on CPU deliberately."
    )))
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
        let surfaces: [(&str, &str); 3] = [
            ("apr run (dispatch.rs)", include_str!("dispatch.rs")),
            (
                "apr chat (dispatch_analysis.rs)",
                include_str!("dispatch_analysis.rs"),
            ),
            (
                "apr serve (commands/serve/mod.rs)",
                include_str!("commands/serve/mod.rs"),
            ),
        ];
        let mut missing = Vec::new();
        for (name, src) in surfaces {
            // serve keeps its own named wrapper; run and chat call accel directly.
            let guarded = src.contains("accel::ensure_available")
                || src.contains("ensure_accelerator_available(config)?");
            if !guarded {
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

    #[test]
    fn the_flag_quoted_back_is_the_one_the_user_typed() {
        assert_eq!(asked_flag(true, None), "--gpu");
        assert_eq!(asked_flag(false, Some("cuda")), "--backend cuda");
        assert_eq!(asked_flag(false, Some("cpu")), "--gpu");
    }
}
