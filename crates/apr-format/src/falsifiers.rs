//! RED-on-purpose falsifier stubs for the apr-format extraction contracts
//! (issue #2231). One `#[test]` per proof obligation in
//! `contracts/apr-format-extraction-v1.yaml`, named exactly as that contract's
//! `falsification_tests[].test` fields cite them so `pv lint` Gate-4 /
//! strict-test-binding resolves the refs immediately (no dangling-ref errors).
//!
//! These are deliberately FAILING until Stage 2 wires the real oracle (the
//! golden byte fixtures + the `cargo tree` deny guard + the core re-export). A
//! falsifier going GREEN here means its obligation has been discharged. Stage 1
//! ships them RED so the obligation exists before the implementation.
//!
//! Mechanics: each stub panics with `unimplemented!` carrying the obligation id,
//! so `cargo test` reports a clean RED with a self-describing message.

#![allow(clippy::panic, clippy::unimplemented)]

#[cfg(test)]
mod tests {
    /// FALSIFY-APRF-BYTE-IDENTITY: re-saving the captured golden model with the
    /// pinned `SaveOptions` must reproduce `golden_v1.apr` / `golden_v2.apr`
    /// byte-for-byte (full file incl. CRC trailer).
    #[test]
    #[ignore = "Stage 1 RED stub — golden byte-identity oracle wired in Stage 2"]
    fn test_falsify_aprf_byte_identity_golden_roundtrip() {
        unimplemented!(
            "FALSIFY-APRF-BYTE-IDENTITY: compare apr_format save() output against \
             tests/fixtures/golden_v1.apr + golden_v2.apr (wired Stage 2)"
        );
    }

    /// FALSIFY-APRF-SOVEREIGN-DEPS: `apr-format`'s dependency graph must contain
    /// zero ML/GPU/tokenizer crates (no trueno / wgpu / cuda* / candle / tch).
    #[test]
    #[ignore = "Stage 1 RED stub — `cargo tree` deny guard wired in Stage 2"]
    fn test_falsify_aprf_sovereign_deps_no_ml_gpu() {
        unimplemented!(
            "FALSIFY-APRF-SOVEREIGN-DEPS: assert `cargo tree -p apr-format` excludes \
             trueno/wgpu/cuda*/candle/tch (wired Stage 2)"
        );
    }

    /// FALSIFY-APRF-CRC-INTEGRITY: the deduplicated `crc32` must be byte-identical
    /// to the legacy `core_io.rs` + `v2/mod.rs` implementations on a shared vector set.
    #[test]
    #[ignore = "Stage 1 RED stub — cross-impl CRC parity oracle wired in Stage 2"]
    fn test_falsify_aprf_crc_integrity_matches_legacy() {
        unimplemented!(
            "FALSIFY-APRF-CRC-INTEGRITY: assert apr_format::crc32::crc32 == legacy \
             core/v2 crc32 over a shared corpus (wired Stage 2)"
        );
    }

    /// FALSIFY-APRF-METADATA-FIDELITY: a save→load round-trip must preserve every
    /// metadata field (`created_at`, version, hyperparameters, license, …) exactly.
    #[test]
    #[ignore = "Stage 1 RED stub — metadata-fidelity oracle wired in Stage 2"]
    fn test_falsify_aprf_metadata_fidelity_roundtrip() {
        unimplemented!(
            "FALSIFY-APRF-METADATA-FIDELITY: assert load(save(meta)) == meta for all \
             populated metadata fields (wired Stage 2)"
        );
    }

    /// FALSIFY-APRF-API-COMPAT: `aprender_core::format::*` re-exports must keep
    /// resolving (no downstream API break) after the leaf extraction.
    #[test]
    #[ignore = "Stage 1 RED stub — re-export compat oracle wired in Stage 2"]
    fn test_falsify_aprf_api_compat_reexport_resolves() {
        unimplemented!(
            "FALSIFY-APRF-API-COMPAT: assert aprender_core::format re-exports the leaf \
             types and the full aprender test suite stays green (wired Stage 2)"
        );
    }

    /// FALSIFY-APRF-QUALITY-GATE: the Jidoka quality gate must still REFUSE a
    /// `quality_score == Some(0)` save and ACCEPT a known-good save, identically
    /// to the pre-extraction behavior.
    #[test]
    #[ignore = "Stage 1 RED stub — quality-gate-preserved oracle wired in Stage 2"]
    fn test_falsify_aprf_quality_gate_preserved() {
        unimplemented!(
            "FALSIFY-APRF-QUALITY-GATE: assert save(score=Some(0)) is Err and a \
             known-good save is Ok, matching captured verdicts (wired Stage 2)"
        );
    }
}
