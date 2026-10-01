//! The credential the lock doors are typed against (04-17 G2, D-11, TRN-07).
//!
//! # What this module exists to fix, and what it deliberately does NOT do
//!
//! `create_selection_lock`, `SelectionLock::mint_test_token` and
//! `CanonicalTestAccess::grant` all required a `SetFitRun<ArtifactReloadedAndVerified>`.
//! That state's only producer is `SetFitRun::<HeadFitted>::verify_artifact`, which
//! consumes a live `SetFitRun<HeadFitted>` — so the three doors were reachable only from
//! inside the process that did the training. `apr eval --split test`, running a day later
//! against a `.apr` file, had no way in.
//!
//! Plan 04-16 measured, with rustc, that the train-time state is NOT reconstructible from
//! artifact bytes: `HeadFittedEvidence` has seven fields of which five are unrecoverable,
//! and `tune::PassedEvidence`'s fields are private (`E0451`) with `tune::validate_evidence`
//! — which needs the full `UpdateEvidence` table — its only producer. The artifact records
//! the evidence SUMMARY, not the table (bundle field 19), on purpose.
//!
//! Then it measured the thing that makes this tractable: **not one of the three doors
//! touches the evidence table.** Between them they read exactly three values —
//! `artifact_hash()`, `selection_semantic_hash()` and `selection().ledger_hash()`. So the
//! blocker was never missing data. It was TYPING.
//!
//! [`SetFitCredential`] is those three values and nothing else. Retyping the doors against
//! it changes no behaviour for the train-time run (which implements it by delegating to the
//! accessors it already had) and opens the doors to a fresh-process credential that carries
//! the same three facts — which is 04-CONTEXT.md D-11 as written: *a fresh process's
//! verified state is integrity + probe replay*.
//!
//! **No field of `HeadFittedEvidence` or `PassedEvidence` is fabricated, defaulted or
//! reconstructed here, and no second minting policy is introduced.** This module contains
//! no `Default`, no placeholder and no evidence type. `tune::validate_evidence` remains the
//! only producer of `PassedEvidence`; that is asserted by
//! `credential_module_fabricates_no_evidence`.
//!
//! # Why the trait is SEALED
//!
//! A credential is what the lock chain checks identity against. An out-of-crate type
//! implementing this trait could return any three strings it liked — and then
//! `mint_test_token` would compare the lock's recorded hash against a number the caller
//! chose, `grant` would re-check it against the same number, and canonical test access
//! would be granted to nothing in particular. That is precisely the substitution TRN-07
//! exists to block, and it is the same argument that seals `LifecycleState` and
//! `SetFitCodec`.
//!
//! The seal is proved by the compiler, not by this paragraph:
//! `tests/ui/setfit_external_credential_impl.rs` is a complete out-of-crate program that
//! implements [`SetFitCredential`] and must fail to compile, with rustc's diagnostic pinned
//! by a committed `.stderr`.
//!
//! # Why three METHODS and not a struct of three fields
//!
//! A `struct Credential { artifact_hash: String, .. }` would have a public constructor or
//! public fields, and either one is a door for supplying the three values rather than
//! reading them off an object — the exact defect `SelectionCandidate::from_evaluation`
//! avoids by reading the artifact hash out of the evaluation instead of accepting it
//! beside it. A sealed trait has no constructor at all: the only way to hold a credential
//! is to hold something this crate minted.

use super::apr_reload::ReloadedSetFitCredential;
use super::{ArtifactReloadedAndVerified, SetFitRun};

/// The seal. Private module, public-in-private trait — the idiom `LifecycleState` and
/// `SetFitCodec` both use, kept local so implementing one does not implement another.
mod sealed {
    /// Implemented only by this crate's credential-bearing types.
    pub trait Sealed {}
}

/// The three values every lock, token and grant door actually reads.
///
/// # This is the WHOLE surface, and that is the point
///
/// Three methods, all `&self`, all returning owned values a caller cannot write back
/// through. There is no accessor here for the evidence table, the encode ledger, the loss
/// trace or the head — not because those are unavailable on the train-time run, but
/// because the doors do not read them, and a credential that carried more than the doors
/// consume would be a claim nothing checks.
///
/// # Errors
///
/// None. Every value is already recorded by the time a credential exists; there is nothing
/// left to fail. A fallible accessor here would mean a credential could exist without one
/// of the three facts, which is the state this trait is defined to make unrepresentable.
pub trait SetFitCredential: sealed::Sealed {
    /// SHA-256 of the artifact's canonical bytes, lowercase hex.
    ///
    /// The value `SelectionLock` locks onto and both later doors re-check against.
    fn artifact_hash(&self) -> String;

    /// The selection's semantic hash — ordered ids, label map and content hashes — as hex.
    fn selection_semantic_hash(&self) -> String;

    /// The access ledger's hash, as the raw digest.
    ///
    /// RAW rather than hex, because `create_selection_lock`'s single implementation is the
    /// one place that renders it and a second rendering is a second chance to render it
    /// differently.
    fn selection_ledger_hash(&self) -> [u8; 32];
}

impl sealed::Sealed for SetFitRun<ArtifactReloadedAndVerified> {}

/// The train-time run is a credential, by DELEGATION to the accessors it already had.
///
/// Every method below is a forwarding call. Nothing is recomputed, re-derived or defaulted,
/// so a train-time call site that was passing `&run` before this trait existed observes
/// byte-identical behaviour — which is what makes the retype in `lock.rs` a retype rather
/// than a change.
///
/// The calls are written in fully-qualified inherent form on purpose. `self.artifact_hash()`
/// would resolve to the inherent method today (inherent methods win over trait methods), but
/// it would silently become an infinite recursion the day the inherent one is removed or
/// renamed. Spelling out which method is meant costs a line and cannot rot into a stack
/// overflow.
impl SetFitCredential for SetFitRun<ArtifactReloadedAndVerified> {
    fn artifact_hash(&self) -> String {
        SetFitRun::<ArtifactReloadedAndVerified>::artifact_hash(self)
    }

    fn selection_semantic_hash(&self) -> String {
        SetFitRun::<ArtifactReloadedAndVerified>::selection_semantic_hash(self)
    }

    fn selection_ledger_hash(&self) -> [u8; 32] {
        self.selection().ledger_hash()
    }
}

impl sealed::Sealed for ReloadedSetFitCredential {}

/// The FRESH-PROCESS credential (04-16), and the reason this trait exists.
///
/// # Why the seal impl is HERE and the type is in `apr_reload`
///
/// `mod sealed` is private to this module, so `Sealed` is not nameable from a sibling —
/// which is the seal working as intended. A type elsewhere in the crate becomes a
/// credential by an impl written in THIS file, so the complete answer to "what satisfies
/// the seal" is one grep in one place, and `credential_seal_is_a_private_supertrait` counts
/// it. `ReloadedSetFitCredential` itself lives beside the door that mints it, because the
/// provenance gate and the type are one argument.
///
/// # This is what 04-16's refusal bought
///
/// `reload_verified_run_from_apr` runs `aprender-core`'s eight-rung production loader,
/// including the six-probe replay, and then requires the artifact's recorded selection
/// semantic hash, selection ledger hash and dataset fingerprint to equal the values
/// computed from the caller's own `Selection` and `PreparedDataset`. Those three facts are
/// exactly what the doors read. Nothing about the optimizer run is claimed, recomputed or
/// defaulted — which is the difference between this credential and the state 04-16 measured
/// to be unreconstructible.
impl SetFitCredential for ReloadedSetFitCredential {
    fn artifact_hash(&self) -> String {
        ReloadedSetFitCredential::artifact_hash(self).to_string()
    }

    fn selection_semantic_hash(&self) -> String {
        ReloadedSetFitCredential::selection_semantic_hash(self).to_string()
    }

    fn selection_ledger_hash(&self) -> [u8; 32] {
        ReloadedSetFitCredential::selection_ledger_hash(self)
    }
}

#[cfg(test)]
#[path = "credential_tests.rs"]
mod credential_tests;
