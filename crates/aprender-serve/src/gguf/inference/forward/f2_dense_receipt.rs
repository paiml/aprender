//! The F2 receipt for the DENSE GGUF path (#3602), on the #3604 table.
//!
//! # Why
//!
//! `validate_gpu_first_token` proves the dense CUDA forward against a CPU
//! reference forward of the whole prompt before the GPU may serve. It re-ran on
//! every `apr run --gpu`. Measured on lambda (RTX 4090, qwen2.5-coder-0.5b
//! Q4_K_M, apr 0.70.0 @ 11bcf6c2ba, 30-token prompt): `setup_ms` 28 193 on the
//! GPU run against a whole `--no-gpu` run of 33 851–36 541 ms. The guard IS a
//! CPU prefill, so `--gpu` could never reach its first token before `--no-gpu`
//! finished its prompt. The hybrid path already validates once per
//! (model sha256, apr build, device) — the operator ruling of 2026-09-20 on
//! #3604 — and this is that receipt, not a second design.
//!
//! # The one thing the dense path adds: precision
//!
//! On the dense path the guard can CHANGE what runs. #3807: an FP8 batched
//! prefill miss is re-measured on FP16, and when FP16 passes, FP8 stays off for
//! the model. A receipt that recorded only "passed" would let the next run skip
//! the guard with FP8 back on — serving exactly the precision the guard
//! rejected. So the device key carries the prefill precision that passed:
//!
//! | tag                        | written when                                  |
//! |----------------------------|-----------------------------------------------|
//! | `prefill=fp8`              | FP8 was on and passed                         |
//! | `prefill=fp16`             | FP8 was off from the start and FP16 passed    |
//! | `prefill=fp16;fp8-rejected`| FP8 missed, the FP16 re-measure passed        |
//!
//! and [`decide_dense`] reads it:
//!
//! - an exact tag match skips;
//! - FP8 on + `fp16;fp8-rejected` skips AND forces FP16, which is what the
//!   validating run ended on;
//! - FP8 off + `fp16;fp8-rejected` skips: FP16 is what runs, and it passed;
//! - everything else validates — in particular FP8 on + plain `fp16`: a run
//!   under `FP8_PREFILL=0` proved nothing about FP8, and must not switch a
//!   model's default precision off through the cache.
//!
//! Pure: no CUDA, no filesystem, no env, so the table is unit-tested on every
//! build (`f2_dense_receipt_tests.rs`).

use super::f2_receipt::{decide, F2Decision, F2Receipt, F2ReceiptKey, F2ValidateReason};

/// The prefill precision a dense F2 validation ended on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensePrecision {
    /// FP8 (E4M3) batched prefill was on and passed.
    Fp8,
    /// FP16 prefill, FP8 never on for this run.
    Fp16,
    /// FP8 missed and the FP16 re-measure passed (#3807); FP8 is off from here.
    Fp16Fp8Rejected,
}

impl DensePrecision {
    /// What a validation ended on, from the FP8 prefill flag before and after it.
    #[must_use]
    pub fn after_validation(fp8_before: bool, fp8_after: bool) -> Self {
        match (fp8_before, fp8_after) {
            (true, true) => Self::Fp8,
            (true, false) => Self::Fp16Fp8Rejected,
            // FP8 is never switched ON by the guard; off-before is plain FP16.
            (false, _) => Self::Fp16,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Fp8 => "prefill=fp8",
            Self::Fp16 => "prefill=fp16",
            Self::Fp16Fp8Rejected => "prefill=fp16;fp8-rejected",
        }
    }
}

/// The receipt's `device` key on the dense path: the driver's device name and
/// the precision that passed.
#[must_use]
pub fn dense_device_key(device: &str, precision: DensePrecision) -> String {
    format!("{device} {}", precision.tag())
}

/// The dense decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseDecision {
    /// Read the receipt; skip the CPU reference forward.
    Skip {
        /// The receipt that matched.
        receipt: F2Receipt,
        /// Turn FP8 prefill (and decode) off before serving: the receipt
        /// vouches for FP16 only, because FP8 was rejected on this triple.
        force_fp16: bool,
    },
    /// Run the guard.
    Validate(F2ValidateReason),
}

/// THE DENSE TABLE. `device` is the bare driver name; `fp8_prefill_on` is the
/// precision this run would serve with if nothing changed it.
#[must_use]
pub fn decide_dense(
    found: Result<Option<F2Receipt>, String>,
    model_sha256: &str,
    apr_version: &str,
    device: &str,
    fp8_prefill_on: bool,
    revalidate: bool,
) -> DenseDecision {
    let key = |precision| F2ReceiptKey {
        model_sha256: model_sha256.to_string(),
        apr_version: apr_version.to_string(),
        device: dense_device_key(device, precision),
    };
    let current = if fp8_prefill_on {
        DensePrecision::Fp8
    } else {
        DensePrecision::Fp16
    };
    let first = decide(found.clone(), &key(current), revalidate);
    let reason = match first {
        F2Decision::Skip { receipt } => {
            return DenseDecision::Skip {
                receipt,
                force_fp16: false,
            }
        },
        F2Decision::Validate(reason) => reason,
    };
    // Only a device-key mismatch can be the rejected-FP8 receipt: every other
    // reason (no file, schema, model, build, --revalidate) stands as it is.
    if !matches!(reason, F2ValidateReason::DeviceMismatch { .. }) {
        return DenseDecision::Validate(reason);
    }
    match decide(found, &key(DensePrecision::Fp16Fp8Rejected), revalidate) {
        F2Decision::Skip { receipt } => DenseDecision::Skip {
            receipt,
            // Already off: nothing to force. On: the receipt says FP8 lost.
            force_fp16: fp8_prefill_on,
        },
        // Report the mismatch against what this run would actually serve.
        F2Decision::Validate(_) => DenseDecision::Validate(reason),
    }
}

#[cfg(test)]
#[path = "f2_dense_receipt_tests.rs"]
mod f2_dense_receipt_tests;
