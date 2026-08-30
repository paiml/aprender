//! PERF-062 / #2790 — the compute REQUEST in, the compute RESOLUTION out.
//!
//! # Why a latch and not a parameter
//!
//! `apr run --gpu` reaches the engine through `dispatch_run` → `run::run` →
//! `RunOptions` → `execute_with_realizar`, and every one of those hops carries
//! the request as a single `no_gpu: bool`. `dispatch.rs` collapses it:
//!
//! ```text
//! let effective_no_gpu = if *gpu { false } else { *no_gpu || backend_forces_cpu };
//! ```
//!
//! After that line `--gpu` and a bare `apr run` are the SAME VALUE. Nothing
//! downstream can report a refused accelerator request, because nothing
//! downstream can see that a request was made. That is the root of #2790's
//! silent fallback: not a missing `eprintln!`, a missing fact.
//!
//! Threading a new parameter through those four hops is the design that already
//! failed twice in this crate — [`crate::commands::offline`] (three commands
//! forgot to forward `--offline` and the control was inert) and
//! [`crate::verbosity`] (`--quiet` was byte-inert on 14 of 16 sampled
//! commands). Both were fixed by latching once in `execute_command`. This is
//! the same shape, for the same reason, and it carries the answer back the same
//! way: a surface cannot disarm the record by forgetting to plumb a field,
//! because it never receives one.
//!
//! # Fail-closed in both directions
//!
//! * The request defaults to [`ComputeRequest::Auto`] — a run that never
//!   latched claims nothing.
//! * The resolution starts EMPTY, and a caller that reads it before an engine
//!   recorded one gets `None`. It never defaults to a class, because a
//!   defaulted class is the fabricated provenance this ticket removes.

use realizar::infer::{ComputeRequest, ComputeResolution};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// The request, as typed. `AtomicU8` because it is set once and read from the
/// engine-facing side without contention.
static REQUEST: AtomicU8 = AtomicU8::new(REQ_AUTO);

const REQ_AUTO: u8 = 0;
const REQ_ACCELERATOR: u8 = 1;
const REQ_CPU: u8 = 2;
const REQ_CUDA: u8 = 3;
const REQ_WGPU: u8 = 4;
const REQ_METAL: u8 = 5;

fn encode(request: ComputeRequest) -> u8 {
    use realizar::infer::ComputeClass as C;
    match request {
        ComputeRequest::Auto => REQ_AUTO,
        ComputeRequest::Accelerator => REQ_ACCELERATOR,
        ComputeRequest::Cpu => REQ_CPU,
        ComputeRequest::Named(C::Cuda) => REQ_CUDA,
        ComputeRequest::Named(C::Wgpu) => REQ_WGPU,
        ComputeRequest::Named(C::Metal) => REQ_METAL,
        ComputeRequest::Named(C::Cpu) => REQ_CPU,
    }
}

fn decode(code: u8) -> ComputeRequest {
    use realizar::infer::ComputeClass as C;
    match code {
        REQ_ACCELERATOR => ComputeRequest::Accelerator,
        REQ_CPU => ComputeRequest::Cpu,
        REQ_CUDA => ComputeRequest::Named(C::Cuda),
        REQ_WGPU => ComputeRequest::Named(C::Wgpu),
        REQ_METAL => ComputeRequest::Named(C::Metal),
        _ => ComputeRequest::Auto,
    }
}

/// Record what the operator asked for. Called once, from the dispatcher.
pub fn latch_request(request: ComputeRequest) {
    REQUEST.store(encode(request), Ordering::Relaxed);
}

/// What the operator asked for.
#[must_use]
pub fn request() -> ComputeRequest {
    decode(REQUEST.load(Ordering::Relaxed))
}

fn slot() -> &'static Mutex<Option<ComputeResolution>> {
    static SLOT: OnceLock<Mutex<Option<ComputeResolution>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

/// Record what the engine actually resolved to.
pub fn record_resolution(resolution: &ComputeResolution) {
    if let Ok(mut guard) = slot().lock() {
        *guard = Some(resolution.clone());
    }
}

/// What the engine resolved to, or `None` when no engine reported one.
///
/// `None` is NOT `cpu`. A printer that turns "nobody measured" into a class
/// name is the defect; it must say the field is absent instead.
#[must_use]
pub fn resolution() -> Option<ComputeResolution> {
    slot().lock().ok().and_then(|g| g.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use realizar::infer::ComputeClass;

    /// Every request survives the round-trip through the latch's encoding.
    ///
    /// Mutation: collapse `REQ_ACCELERATOR` and `REQ_AUTO` to the same code —
    /// which is exactly what `effective_no_gpu` does in `dispatch.rs` — and
    /// this goes RED.
    #[test]
    fn every_request_round_trips() {
        for want in [
            ComputeRequest::Auto,
            ComputeRequest::Accelerator,
            ComputeRequest::Cpu,
            ComputeRequest::Named(ComputeClass::Cuda),
            ComputeRequest::Named(ComputeClass::Wgpu),
            ComputeRequest::Named(ComputeClass::Metal),
        ] {
            assert_eq!(decode(encode(want)), want, "{want:?} did not survive");
        }
    }

    /// THE DISCRIMINATION CASE. Without it the encoding could map everything
    /// to one code and `every_request_round_trips` would still pass if `decode`
    /// mapped it back to the same thing — so assert the codes are DISTINCT
    /// where the defect conflated them.
    #[test]
    fn accelerator_and_auto_do_not_share_a_code() {
        assert_ne!(
            encode(ComputeRequest::Accelerator),
            encode(ComputeRequest::Auto),
            "`--gpu` and no flag must not encode to the same value; that \
             collapse is #2790"
        );
        assert_ne!(
            encode(ComputeRequest::Accelerator),
            encode(ComputeRequest::Cpu)
        );
    }

    /// An unknown code decodes to the claim-nothing value, never to an
    /// accelerator.
    #[test]
    fn an_unknown_code_decodes_to_auto() {
        assert_eq!(decode(200), ComputeRequest::Auto);
    }

    /// A resolution nobody recorded is ABSENT, not `cpu`.
    ///
    /// The slot is process-wide and other tests in this binary may have filled
    /// it, so the assertion is on the MODULE'S SOURCE rather than on the live
    /// slot: nothing outside the tests may turn absence into a class.
    ///
    /// The scan stops at the test module. A whole-file `include_str!` matches
    /// the needle written in the test itself and fails on a correct module,
    /// which is exactly what the first draft of this test did.
    #[test]
    fn an_unrecorded_resolution_is_none_not_a_class() {
        let src = include_str!("compute_latch.rs");
        let production = src
            .split_once("#[cfg(test)]")
            .map_or(src, |(before, _)| before);
        for forbidden in ["unwrap_or(", "unwrap_or_else(", "unwrap_or_default("] {
            assert!(
                !production.contains(forbidden),
                "compute_latch must not default an absent resolution into a class; \
                 found {forbidden} in the production half"
            );
        }
        assert!(
            production.contains("pub fn resolution() -> Option<ComputeResolution>"),
            "resolution() must return an Option so absence is expressible at all"
        );
    }
}
