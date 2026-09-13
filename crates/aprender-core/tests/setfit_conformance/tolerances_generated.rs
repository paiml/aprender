//! GENERATED FILE — DO NOT HAND-EDIT.
//!
//! Tolerance constants for the SetFit conformance harness (plan 01-08).
//!
//! Source contract: contracts/setfit-encoder-conformance-v1.yaml
//! Contract metadata.version: 2.0.0
//! Contract sha256: 16a6591788a6c693ad3d08845a20267e31d4a86ee663310a943c841d9e7b2b93
//!
//! Regenerate with:
//!
//! ```text
//! APRENDER_REGEN_TOLERANCES=1 cargo test -p aprender-core \
//!   --features setfit,conformance-fixtures --test setfit_conformance \
//!   conformance_tolerances_regenerate -- --ignored
//! ```
//!
//! The emitter is rustfmt-stable; if a future edit breaks that, run
//! `cargo fmt -p aprender-core` after regenerating, so a drift check on
//! this file reports semantics rather than whitespace (D7).
//!
//! D-14: these numbers exist in ONE place, the versioned contract. Widening
//! one requires a contract edit `pv diff` flags with a semver bump. The
//! agreement test in tests/setfit_conformance.rs fails the build if this file
//! and the contract ever disagree in a workspace checkout.

/// From `OBLIG-ENC-03-PER-LAYER-FORWARD-PARITY`.
pub const FORWARD_PER_LAYER: f32 = 1.52587891e-5;
/// From `OBLIG-ENC-03-POOLED-EMBEDDING-PARITY`.
pub const POOLING_NORMALIZE: f32 = 7.62939453e-6;
/// From `OBLIG-ENC-03-ACTIVATION-PARITY`.
pub const ACTIVATION: f32 = 4.47239421e-6;
/// From `OBLIG-ENC-03-PADDING-INVARIANCE`.
pub const BATCH_INVARIANCE: f32 = 7.62939453e-6;
/// From `OBLIG-ENC-04-NAMED-GRADIENT-PARITY`.
pub const GRADIENTS: f32 = 3.05175781e-5;
/// From `OBLIG-ENC-04-GRADIENT-AND-STEP-GATE`.
pub const ZERO_GRAD_FLOOR: f32 = 6.12913980e-5;
/// From `OBLIG-ENC-06-LOSS-FORWARD-PARITY`.
pub const LOSS_PAIR: f32 = 7.62939453e-6;
/// From `OBLIG-ENC-04-POST-STEP-PARAMETER-PARITY`.
pub const OPTIMIZER_STEP: f32 = 1.89172753e-6;
/// From `OBLIG-ENC-04-MULTISTEP-TRAJECTORY-PARITY`.
pub const OPTIMIZER_MULTISTEP: f32 = 7.62939453e-6;
/// From `OBLIG-ENC-01-FULL-MODEL-REFERENCE-PARITY`.
pub const FULL_MODEL_REFERENCE: f32 = 3.73762473e-5;

/// sha256 of the source contract at generation time.
pub const CONTRACT_SHA256: &str =
    "16a6591788a6c693ad3d08845a20267e31d4a86ee663310a943c841d9e7b2b93";

/// `metadata.version` of the source contract at generation time.
pub const CONTRACT_VERSION: &str = "2.0.0";

/// The command that regenerates this file.
pub const REGENERATE_COMMAND: &str = "APRENDER_REGEN_TOLERANCES=1 cargo test -p aprender-core --features setfit,conformance-fixtures --test setfit_conformance conformance_tolerances_regenerate -- --ignored";
