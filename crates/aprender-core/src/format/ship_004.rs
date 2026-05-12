// SHIP-TWO-001 AC-SHIP1-004 / FALSIFY-SHIP-004 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md
// Contract: contracts/qwen2-e2e-verification-v1.yaml (FALSIFY-QW2E-SHIP-004 —
// wired in the same PR as this file lands).
//
// AC-SHIP1-004 states that the MODEL-1 teacher must survive an
// `apr export --format gguf` round-trip: the emitted .gguf must load
// in upstream llama.cpp. The spec's falsification rule is "llama.cpp
// exit ≠ 0" on the exported file.
//
// This file discharges the *decision rules* at `PARTIAL_ALGORITHM_LEVEL`
// by binding three pure verdict fns — one per format / tool boundary:
//
//   1. `verdict_from_llama_cli_exit(code) -> Ship004Verdict` —
//      Pass iff `code == 0`. The POSIX success-exit convention: any
//      non-zero exit (crash, llama.cpp format-reject, missing metadata,
//      OOM, SIGTERM) is a Fail. No noise allowance — the round-trip is
//      either consumable or it is not.
//
//   2. `verdict_from_gguf_magic_bytes(bytes) -> Ship004Verdict` —
//      Pass iff the first four bytes of the exported file are `b"GGUF"`
//      (the GGUF v2/v3 magic). Conservative Fail on short slices
//      (`len < 4`) or any single-byte flip. Catches a serializer that
//      silently writes the SafeTensors / APR header into a .gguf file
//      before llama.cpp has a chance to reject it.
//
//   3. `verdict_from_gguf_version(ver) -> Ship004Verdict` —
//      Pass iff `ver ∈ {2, 3}` (the only GGUF versions llama.cpp
//      currently accepts; v1 was the predecessor GGJT format). Fail on
//      0, 1, 4+, or `u32::MAX`. Catches a serializer that writes the
//      wrong version u32 after the magic.
//
// The compute-heavy portion of the AC (shelling out to a real
// `llama-cli` binary on the exported file) is intentionally out of
// scope here. The three decision rules are what the compute harness
// must emit a Pass on, and changing any of the three bound constants
// (`AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE`,
// `AC_SHIP1_004_GGUF_MAGIC_BYTES`, `AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS`)
// breaks this test before a single export runs.
//
// Mirrors the integer-threshold shape of SHIP-017 / SHIP-002
// (`verdict_from_syntax_error_count(usize)`, exit-code semantics mirror
// zero-tolerance) and the byte-literal-magic shape of SHIP-010
// (`REQUIRED_URL_SCHEME = b"https://"`). Unlike the float-threshold
// ships (SHIP-003 cosine, SHIP-005 pass@1, SHIP-007 decode-tps), this
// discharge has *three* verdict fns rather than one or two, because
// GGUF round-trip validity is decomposable into three independent
// format-boundary gates. MODEL-1 is at 7/10 AC-SHIP1 items touched
// (SHIP-008 + SHIP-009 + SHIP-006 + SHIP-007 + SHIP-005 + SHIP-010 +
// SHIP-003) before this lands; SHIP-004 brings it to 8/10.

/// The POSIX success-exit code. `llama-cli <exported>.gguf` must emit
/// exit code `0` on success; any non-zero exit (including SIGTERM,
/// SIGKILL, or format-reject) is a FALSIFY-SHIP-004 failure.
///
/// Lockstep with `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §4.2 row AC-SHIP1-004: "Export to GGUF via `apr export --format
/// gguf`; loads in llama.cpp".
pub const AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE: i32 = 0;

/// The four-byte ASCII magic at offset 0 of every GGUF file. Pinned as
/// `b"GGUF"` = `[0x47, 0x47, 0x55, 0x46]`. Any single-byte flip is a
/// FALSIFY-SHIP-004 failure; the mutation survey exercises three
/// classic off-by-one permutations (`b"GGUE"`, `b"FUGG"`, `b"GGU "`)
/// and a short-slice rejection.
pub const AC_SHIP1_004_GGUF_MAGIC_BYTES: &[u8; 4] = b"GGUF";

/// GGUF versions currently accepted by upstream llama.cpp. Pinned as
/// `&[2, 3]`: V1 was the GGJT predecessor (llama.cpp rejects), V2 is
/// the current stable, V3 is the modern revision. Any version outside
/// this set is a FALSIFY-SHIP-004 failure.
///
/// Upstream reference: `crates/aprender-serve/src/gguf/types.rs`
/// `pub const GGUF_VERSION_V2: u32 = 2; pub const GGUF_VERSION_V3: u32 = 3;`
pub const AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS: &[u32] = &[2, 3];

/// Binary verdict for FALSIFY-SHIP-004 / AC-SHIP1-004.
/// `Pass` iff the specific rule Passes. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship004Verdict {
    /// The boundary gate Passes: `llama-cli` exit code is 0 (round-trip
    /// consumable) OR the GGUF magic matches `b"GGUF"` exactly OR the
    /// GGUF version is one of the supported set `{2, 3}`.
    Pass,
    /// The boundary gate Fails: any non-zero exit code; any single-byte
    /// magic flip or short slice; any version outside `{2, 3}`.
    Fail,
}

/// Pure decision rule for FALSIFY-SHIP-004 / AC-SHIP1-004 exit-code
/// boundary: `Pass` iff `code == AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE`.
///
/// POSIX-convention zero-tolerance: any non-zero exit from `llama-cli`
/// on the exported `.gguf` is a ship-blocker. No noise allowance.
///
/// # Examples
///
/// ```
/// use aprender::format::ship_004::{verdict_from_llama_cli_exit, Ship004Verdict};
///
/// assert_eq!(verdict_from_llama_cli_exit(0), Ship004Verdict::Pass);
/// assert_eq!(verdict_from_llama_cli_exit(1), Ship004Verdict::Fail);
/// assert_eq!(verdict_from_llama_cli_exit(-1), Ship004Verdict::Fail);
/// assert_eq!(verdict_from_llama_cli_exit(i32::MAX), Ship004Verdict::Fail);
/// ```
#[must_use]
pub const fn verdict_from_llama_cli_exit(code: i32) -> Ship004Verdict {
    if code == AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE {
        Ship004Verdict::Pass
    } else {
        Ship004Verdict::Fail
    }
}

/// Pure decision rule for FALSIFY-SHIP-004 / AC-SHIP1-004 GGUF-magic
/// boundary: `Pass` iff `bytes.len() >= 4` AND the first four bytes
/// equal `AC_SHIP1_004_GGUF_MAGIC_BYTES` byte-for-byte.
///
/// Conservative Fail on short slices: a `.gguf` file shorter than the
/// magic header is a serializer truncation bug, not a parseable GGUF.
///
/// # Examples
///
/// ```
/// use aprender::format::ship_004::{verdict_from_gguf_magic_bytes, Ship004Verdict};
///
/// assert_eq!(verdict_from_gguf_magic_bytes(b"GGUF"), Ship004Verdict::Pass);
/// assert_eq!(verdict_from_gguf_magic_bytes(b"GGUF\x02\x00\x00\x00"), Ship004Verdict::Pass);
/// assert_eq!(verdict_from_gguf_magic_bytes(b"GGUE"), Ship004Verdict::Fail);
/// assert_eq!(verdict_from_gguf_magic_bytes(b"GGU"), Ship004Verdict::Fail);
/// assert_eq!(verdict_from_gguf_magic_bytes(&[]), Ship004Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_gguf_magic_bytes(bytes: &[u8]) -> Ship004Verdict {
    if bytes.len() < AC_SHIP1_004_GGUF_MAGIC_BYTES.len() {
        return Ship004Verdict::Fail;
    }
    if &bytes[..AC_SHIP1_004_GGUF_MAGIC_BYTES.len()] == AC_SHIP1_004_GGUF_MAGIC_BYTES.as_slice() {
        Ship004Verdict::Pass
    } else {
        Ship004Verdict::Fail
    }
}

/// Pure decision rule for FALSIFY-SHIP-004 / AC-SHIP1-004 GGUF-version
/// boundary: `Pass` iff `ver` is one of `AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS`.
///
/// V1 is the GGJT predecessor format (pre-GGUF) and is Fail. V4+ is
/// an unspecified future revision and Fail-closed until the support
/// set is updated. `u32::MAX` is a telemetry corruption sentinel and
/// Fail.
///
/// # Examples
///
/// ```
/// use aprender::format::ship_004::{verdict_from_gguf_version, Ship004Verdict};
///
/// assert_eq!(verdict_from_gguf_version(2), Ship004Verdict::Pass);
/// assert_eq!(verdict_from_gguf_version(3), Ship004Verdict::Pass);
/// assert_eq!(verdict_from_gguf_version(1), Ship004Verdict::Fail);
/// assert_eq!(verdict_from_gguf_version(0), Ship004Verdict::Fail);
/// assert_eq!(verdict_from_gguf_version(4), Ship004Verdict::Fail);
/// assert_eq!(verdict_from_gguf_version(u32::MAX), Ship004Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_gguf_version(ver: u32) -> Ship004Verdict {
    let mut i = 0;
    while i < AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS.len() {
        if AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS[i] == ver {
            return Ship004Verdict::Pass;
        }
        i += 1;
    }
    Ship004Verdict::Fail
}

#[cfg(test)]
mod ship_004_tests {
    //! FALSIFY-SHIP-004 algorithm-level mutation survey.
    //!
    //! Each of the three verdict fns gets its own dedicated test section
    //! covering: the happy path, adjacent-value rejection, provenance
    //! pins on the bound constants, and conservative-Fail guards.
    //!
    //! The compute-heavy portion of AC-SHIP1-004 (shell out to a real
    //! `llama-cli` binary on the exported `.gguf` and assert exit = 0)
    //! is out of scope; see `full_discharge_blocks_on` in
    //! `contracts/qwen2-e2e-verification-v1.yaml` FALSIFY-QW2E-SHIP-004.
    use super::*;

    #[test]
    fn falsify_ship_004_llama_cli_exit_code_logic() {
        // Section 1: POSIX-success boundary — exit code 0 ↔ Pass.
        assert_eq!(
            verdict_from_llama_cli_exit(0),
            Ship004Verdict::Pass,
            "POSIX success exit (0) must Pass"
        );

        // Section 2: Adjacent-value Fail — the neighbours of 0 on both
        // sides Fail.
        assert_eq!(
            verdict_from_llama_cli_exit(1),
            Ship004Verdict::Fail,
            "Exit code 1 must Fail"
        );
        assert_eq!(
            verdict_from_llama_cli_exit(-1),
            Ship004Verdict::Fail,
            "Exit code -1 must Fail"
        );

        // Section 3: Classic non-zero exit bands — format reject, OOM,
        // SIGSEGV (128+11=139), SIGKILL (128+9=137), shell "command not
        // found" (127), shell general error (2).
        for bad_code in [2_i32, 127, 137, 139, 255] {
            assert_eq!(
                verdict_from_llama_cli_exit(bad_code),
                Ship004Verdict::Fail,
                "Classic failure exit code {bad_code} must Fail"
            );
        }

        // Section 4: i32 extrema — saturation bounds must Fail.
        assert_eq!(
            verdict_from_llama_cli_exit(i32::MIN),
            Ship004Verdict::Fail,
            "i32::MIN must Fail"
        );
        assert_eq!(
            verdict_from_llama_cli_exit(i32::MAX),
            Ship004Verdict::Fail,
            "i32::MAX must Fail"
        );

        // Section 5: Monotonicity — anything non-zero is Fail. Exhaustive
        // sweep over a representative band.
        for code in -256_i32..=256 {
            let v = verdict_from_llama_cli_exit(code);
            let expected = if code == 0 {
                Ship004Verdict::Pass
            } else {
                Ship004Verdict::Fail
            };
            assert_eq!(v, expected, "Mismatch at exit code {code}");
        }

        // Section 6: Provenance pin on the success sentinel.
        assert_eq!(
            AC_SHIP1_004_LLAMA_CLI_SUCCESS_EXIT_CODE, 0_i32,
            "POSIX success sentinel drifted — spec §4.2 AC-SHIP1-004 binds 0"
        );
    }

    #[test]
    fn falsify_ship_004_gguf_magic_bytes_logic() {
        // Section 1: Canonical magic — exact `b"GGUF"` Passes.
        assert_eq!(
            verdict_from_gguf_magic_bytes(b"GGUF"),
            Ship004Verdict::Pass,
            "Canonical GGUF magic must Pass"
        );

        // Section 2: Magic + trailing data — a real .gguf has the 4-byte
        // version u32 immediately after; Pass must hold on the full
        // prefix.
        assert_eq!(
            verdict_from_gguf_magic_bytes(b"GGUF\x02\x00\x00\x00"),
            Ship004Verdict::Pass,
            "GGUF magic + V2 LE version header must Pass (first 4 bytes match)"
        );
        assert_eq!(
            verdict_from_gguf_magic_bytes(b"GGUF\x03\x00\x00\x00garbage_after"),
            Ship004Verdict::Pass,
            "GGUF magic + V3 LE version + trailing must Pass (first 4 bytes match)"
        );

        // Section 3: Single-byte flips — any one-byte off magic Fails.
        // Classic off-by-one permutations plus a padding-space flip.
        for bad_magic in [
            &b"GGUE"[..], // last byte off-by-one
            &b"GGTF"[..], // third byte flipped
            &b"GFUF"[..], // second byte flipped
            &b"FUGG"[..], // reversed
            &b"GGU "[..], // last byte → space
            &b"gguf"[..], // case flip
        ] {
            assert_eq!(
                verdict_from_gguf_magic_bytes(bad_magic),
                Ship004Verdict::Fail,
                "Byte-flipped magic {bad_magic:?} must Fail"
            );
        }

        // Section 4: Short slice — conservative Fail when the reader
        // didn't even have enough bytes to compare the magic.
        assert_eq!(
            verdict_from_gguf_magic_bytes(&[]),
            Ship004Verdict::Fail,
            "Empty slice must conservatively Fail"
        );
        assert_eq!(
            verdict_from_gguf_magic_bytes(b"G"),
            Ship004Verdict::Fail,
            "1-byte slice must Fail"
        );
        assert_eq!(
            verdict_from_gguf_magic_bytes(b"GG"),
            Ship004Verdict::Fail,
            "2-byte slice must Fail"
        );
        assert_eq!(
            verdict_from_gguf_magic_bytes(b"GGU"),
            Ship004Verdict::Fail,
            "3-byte slice must Fail"
        );

        // Section 5: Wrong-format magics — SafeTensors, APR, binary zero.
        assert_eq!(
            verdict_from_gguf_magic_bytes(b"APR\0"),
            Ship004Verdict::Fail,
            "APR v2 magic must Fail (a .gguf file with APR header is broken serializer)"
        );
        assert_eq!(
            verdict_from_gguf_magic_bytes(b"APRN"),
            Ship004Verdict::Fail,
            "APR v1 magic must Fail"
        );
        assert_eq!(
            verdict_from_gguf_magic_bytes(&[0x00, 0x00, 0x00, 0x00]),
            Ship004Verdict::Fail,
            "Zero-filled first 4 bytes must Fail"
        );

        // Section 6: Provenance pin on the magic byte literal. Locking
        // each byte (G, G, U, F) = (0x47, 0x47, 0x55, 0x46).
        assert_eq!(
            AC_SHIP1_004_GGUF_MAGIC_BYTES, b"GGUF",
            "GGUF magic constant drifted — spec AC-SHIP1-004 binds b\"GGUF\""
        );
        assert_eq!(
            AC_SHIP1_004_GGUF_MAGIC_BYTES[0], 0x47,
            "First magic byte must be 'G' = 0x47"
        );
        assert_eq!(
            AC_SHIP1_004_GGUF_MAGIC_BYTES[1], 0x47,
            "Second magic byte must be 'G' = 0x47"
        );
        assert_eq!(
            AC_SHIP1_004_GGUF_MAGIC_BYTES[2], 0x55,
            "Third magic byte must be 'U' = 0x55"
        );
        assert_eq!(
            AC_SHIP1_004_GGUF_MAGIC_BYTES[3], 0x46,
            "Fourth magic byte must be 'F' = 0x46"
        );
    }

    #[test]
    fn falsify_ship_004_gguf_version_logic() {
        // Section 1: Supported versions — V2 and V3 Pass.
        assert_eq!(
            verdict_from_gguf_version(2),
            Ship004Verdict::Pass,
            "GGUF V2 must Pass"
        );
        assert_eq!(
            verdict_from_gguf_version(3),
            Ship004Verdict::Pass,
            "GGUF V3 must Pass"
        );

        // Section 2: Predecessor / zero / below-band Fail.
        assert_eq!(
            verdict_from_gguf_version(0),
            Ship004Verdict::Fail,
            "GGUF version 0 must Fail (un-initialized header)"
        );
        assert_eq!(
            verdict_from_gguf_version(1),
            Ship004Verdict::Fail,
            "GGUF V1 must Fail (GGJT predecessor — llama.cpp rejects)"
        );

        // Section 3: Above-band Fail — future versions are Fail-closed
        // until the support set is explicitly widened.
        for bad_ver in [4_u32, 5, 10, 100, 1_000_000, u32::MAX] {
            assert_eq!(
                verdict_from_gguf_version(bad_ver),
                Ship004Verdict::Fail,
                "Above-band GGUF version {bad_ver} must Fail (Fail-closed)"
            );
        }

        // Section 4: Exhaustive sweep over a representative band — only
        // {2, 3} Pass.
        for ver in 0_u32..=64 {
            let v = verdict_from_gguf_version(ver);
            let expected = if ver == 2 || ver == 3 {
                Ship004Verdict::Pass
            } else {
                Ship004Verdict::Fail
            };
            assert_eq!(v, expected, "Mismatch at GGUF version {ver}");
        }

        // Section 5: Provenance pin on the supported-versions set.
        assert_eq!(
            AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS,
            &[2_u32, 3_u32],
            "GGUF supported versions drifted — spec §4.2 AC-SHIP1-004 binds {{2, 3}}"
        );
        assert_eq!(
            AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS.len(),
            2,
            "Supported version count must be exactly 2"
        );
        assert_eq!(AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS[0], 2_u32);
        assert_eq!(AC_SHIP1_004_GGUF_SUPPORTED_VERSIONS[1], 3_u32);
    }

    /// FALSIFY-QW2E-SHIP-004 YAML binding: parses
    /// `qwen2-e2e-verification-v1.yaml` and asserts the
    /// FALSIFY-QW2E-SHIP-004 falsification block has been promoted from
    /// `PARTIAL_ALGORITHM_LEVEL` (v1.5.0) → `DISCHARGED` (v1.8.0) via
    /// live `apr export → llama-cli` round-trip on noah-Lambda-Vector
    /// RTX 4090. Falsifier: if the contract is edited to drop the
    /// live-evidence block or downgrade the discharge marker, this
    /// test fails before any network/compute I/O is launched.
    #[test]
    fn falsify_ship_004_yaml_binding_pins_discharged_status() {
        const CONTRACT_YAML: &str =
            include_str!("../../../../contracts/qwen2-e2e-verification-v1.yaml");

        let doc: serde_yaml::Value = serde_yaml::from_str(CONTRACT_YAML)
            .expect("qwen2-e2e-verification-v1.yaml must parse as YAML");

        let falsifications = doc["falsification_tests"]
            .as_sequence()
            .expect("falsification_tests must be a sequence");
        let block = falsifications
            .iter()
            .find(|b| b["id"].as_str() == Some("FALSIFY-QW2E-SHIP-004"))
            .expect("FALSIFY-QW2E-SHIP-004 must exist in qwen2-e2e-verification-v1");

        assert_eq!(
            block["discharge_status"].as_str(),
            Some("DISCHARGED"),
            "FALSIFY-QW2E-SHIP-004 must advertise DISCHARGED \
             (live `apr export → llama-cli` round-trip on canonical teacher \
             at v1.8.0; previous PARTIAL_ALGORITHM_LEVEL at v1.5.0)",
        );
        assert!(
            block["discharged_evidence"].is_mapping(),
            "FALSIFY-QW2E-SHIP-004 DISCHARGED status requires a discharged_evidence \
             block documenting the host, command chain, and per-step verdicts",
        );
        assert_eq!(
            block["discharged_evidence"]["host"].as_str(),
            Some("noah-Lambda-Vector"),
            "discharged_evidence.host must pin to the lambda-labs RTX 4090 host",
        );
        assert_eq!(
            block["discharged_evidence"]["overall"].as_str(),
            Some("PASS"),
            "discharged_evidence.overall must equal PASS",
        );
        assert_eq!(
            block["discharged_evidence"]["magic_verdict"].as_str(),
            Some("PASS"),
            "discharged_evidence.magic_verdict must equal PASS",
        );
        assert_eq!(
            block["discharged_evidence"]["version"].as_u64(),
            Some(3),
            "discharged_evidence.version must equal 3 (in {{2, 3}} supported set)",
        );
        assert_eq!(
            block["discharged_evidence"]["llama_cli_exit_verdict"].as_str(),
            Some("PASS"),
            "discharged_evidence.llama_cli_exit_verdict must equal PASS",
        );
        let live_evidence = block["discharged_evidence"]["evidence_discharged_by_live"]
            .as_sequence()
            .expect(
                "FALSIFY-QW2E-SHIP-004 DISCHARGED requires \
                 discharged_evidence.evidence_discharged_by_live (live RTX 4090 evidence list)",
            );
        assert!(
            !live_evidence.is_empty(),
            "evidence_discharged_by_live must list at least one live observation",
        );
    }
}
