// SHIP-TWO-001 AC-SHIP1-001 / FALSIFY-SHIP-001 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md
// Contract: contracts/qwen2-e2e-verification-v1.yaml (FALSIFY-QW2E-SHIP-001 —
// wired in the same PR as this file lands).
//
// AC-SHIP1-001 states that the MODEL-1 teacher weights must load via
// `realizar::Model::load_safetensors(path)` returning `Ok(_)`. The
// spec's falsification rule is "`Err(_)` returned".
//
// This file discharges the *decision rules* at `PARTIAL_ALGORITHM_LEVEL`
// by binding three pure verdict fns — one per format / boundary gate:
//
//   1. `verdict_from_load_result(is_ok) -> Ship001Verdict` —
//      the Result boundary. `Pass` iff `is_ok`. Any `Err(_)` (file not
//      found, permission denied, corrupted header, tensor shape mismatch,
//      dtype unsupported) is a Fail. No noise allowance — the load
//      either succeeds or it does not.
//
//   2. `verdict_from_safetensors_header_size(header_len, file_len) -> Ship001Verdict`
//      — the safetensors format's fundamental length invariant. The
//      safetensors file layout is:
//        [bytes 0..8]: u64 LE = header JSON length N
//        [bytes 8..8+N]: JSON header object
//        [bytes 8+N..]: tensor data
//      `Pass` iff `0 < header_len <= file_len - 8`. Fail on zero-length
//      header (a deserialzer refusing to emit metadata), Fail on a
//      header that claims more bytes than the file physically has
//      (silent truncation), Fail on file_len < 8 (too short to even
//      store the length prefix).
//
//   3. `verdict_from_safetensors_json_open_byte(byte) -> Ship001Verdict`
//      — the JSON-object-start boundary. The safetensors header must
//      begin with the ASCII open-brace `b'{'` (0x7B). `Pass` iff the
//      first byte of the JSON header region equals 0x7B. Catches a
//      serializer that writes a JSON array, a bare string, or binary
//      tensor data into the header region.
//
// The compute-heavy portion of the AC (actually calling
// `realizar::Model::load_safetensors(path)` on the real 7B teacher
// safetensors file and asserting the Result is Ok with the expected
// tensor count ≥ 339 for Qwen2.5-Coder-7B) is intentionally out of
// scope here. The three decision rules are what the compute harness
// must emit a Pass on, and changing any of the three bound constants
// (`AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN`,
// `AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE`, or the Result→Pass
// correspondence) breaks this test before a single model load runs.
//
// Mirrors the triple-verdict shape of SHIP-004 (GGUF export → llama.cpp):
// both ACs decompose the "tool X accepted the artifact" rule into the
// underlying format-boundary gates that the tool necessarily checks,
// so a harness-level Ok/Err cannot mask a silent upstream drift. Unlike
// SHIP-004's exec-tool boundary (exit code 0), this AC's boundary is
// an in-process Rust `Result<Model, Error>`; the pure verdict fn
// collapses that to a `bool` and then pins the format invariants
// separately. MODEL-1 is at 8/10 AC-SHIP1 items touched (SHIP-008 +
// SHIP-009 + SHIP-006 + SHIP-007 + SHIP-005 + SHIP-010 + SHIP-003 +
// SHIP-004) before this lands; SHIP-001 brings it to 9/10.

/// Length in bytes of the u64 little-endian prefix that stores the
/// JSON header size in every safetensors file. The first 8 bytes of
/// any valid safetensors file are the header length, so any file
/// shorter than 8 bytes fails the format check unconditionally.
///
/// Lockstep with `docs/specifications/aprender-train/ship-two-models-spec.md`
/// §4.2 row AC-SHIP1-001: "Student weights load via
/// `realizar::Model::load_safetensors`".
///
/// Upstream reference: <https://github.com/huggingface/safetensors>
/// file format specification section "Format".
pub const AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN: u64 = 8;

/// The ASCII open-brace byte (`b'{'` = 0x7B) that the safetensors
/// JSON header region must begin with. Any other first byte is a
/// serializer bug that FALSIFY-SHIP-001 must reject conservatively.
///
/// Locked to a single byte literal rather than a `&str` so mutation
/// on the character level (`{`→`[`, `{`→`(`, `{`→`<`, etc.) is
/// caught atomically.
pub const AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE: u8 = b'{';

/// Binary verdict for FALSIFY-SHIP-001 / AC-SHIP1-001.
/// `Pass` iff the specific rule Passes. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship001Verdict {
    /// The boundary gate Passes: `Result` was `Ok(_)` OR the header
    /// length invariant holds OR the JSON open byte is `b'{'`.
    Pass,
    /// The boundary gate Fails: `Result` was `Err(_)`; the header
    /// length is zero, exceeds the file, or the file is shorter than
    /// the 8-byte prefix; the JSON open byte is anything other than
    /// `b'{'`.
    Fail,
}

/// Pure decision rule for FALSIFY-SHIP-001 / AC-SHIP1-001 Result
/// boundary: `Pass` iff `is_ok` (the `realizar::Model::load_safetensors`
/// call returned `Ok(_)`).
///
/// The spec rule is phrased directly over a `Result<Model, _>`, but
/// the algorithm-level discharge collapses the outcome to a single
/// `bool` so the decision rule is observable without a concrete
/// `Error` type from the realizar crate. Any `Err(_)` variant —
/// file-not-found, IO error, format parse error, dtype mismatch,
/// tensor-shape mismatch — is uniformly a Fail.
///
/// # Examples
///
/// ```
/// use aprender::format::ship_001::{verdict_from_load_result, Ship001Verdict};
///
/// assert_eq!(verdict_from_load_result(true), Ship001Verdict::Pass);
/// assert_eq!(verdict_from_load_result(false), Ship001Verdict::Fail);
/// ```
#[must_use]
pub const fn verdict_from_load_result(is_ok: bool) -> Ship001Verdict {
    if is_ok {
        Ship001Verdict::Pass
    } else {
        Ship001Verdict::Fail
    }
}

/// Pure decision rule for FALSIFY-SHIP-001 / AC-SHIP1-001
/// safetensors-header-size boundary:
/// `Pass` iff `file_len >= AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN`
/// AND `header_len > 0`
/// AND `header_len <= file_len - AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN`.
///
/// The safetensors format reserves the first 8 bytes as a u64 LE
/// header length `N`. The JSON header then occupies the next `N`
/// bytes, after which tensor data follows. So any valid safetensors
/// file satisfies `8 + header_len <= file_len`, i.e.
/// `header_len <= file_len - 8`. A header_len of zero is a malformed
/// serializer output (no metadata), also Fail.
///
/// Conservative Fail on `file_len < 8`: a file shorter than the
/// length prefix cannot be safetensors.
///
/// # Examples
///
/// ```
/// use aprender::format::ship_001::{verdict_from_safetensors_header_size, Ship001Verdict};
///
/// // 100-byte file: 8 prefix + 50 header + 42 data.
/// assert_eq!(verdict_from_safetensors_header_size(50, 100), Ship001Verdict::Pass);
/// // Edge: header fills the entire remainder of the file (no data region).
/// assert_eq!(verdict_from_safetensors_header_size(92, 100), Ship001Verdict::Pass);
/// // Fail: header claims more bytes than the file has.
/// assert_eq!(verdict_from_safetensors_header_size(200, 100), Ship001Verdict::Fail);
/// // Fail: zero-length header.
/// assert_eq!(verdict_from_safetensors_header_size(0, 100), Ship001Verdict::Fail);
/// // Fail: file shorter than the 8-byte prefix.
/// assert_eq!(verdict_from_safetensors_header_size(0, 4), Ship001Verdict::Fail);
/// ```
#[must_use]
pub const fn verdict_from_safetensors_header_size(
    header_len: u64,
    file_len: u64,
) -> Ship001Verdict {
    if file_len < AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN {
        return Ship001Verdict::Fail;
    }
    if header_len == 0 {
        return Ship001Verdict::Fail;
    }
    let max_header = file_len - AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN;
    if header_len <= max_header {
        Ship001Verdict::Pass
    } else {
        Ship001Verdict::Fail
    }
}

/// Pure decision rule for FALSIFY-SHIP-001 / AC-SHIP1-001
/// safetensors-JSON-object-start boundary:
/// `Pass` iff `byte == AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE`.
///
/// The first byte of the JSON header region (at file offset 8) MUST
/// be `b'{'` (0x7B) — a safetensors header is always a JSON object.
/// An open-bracket `b'['`, a quote `b'"'`, or any other byte is a
/// serializer bug caught conservatively.
///
/// # Examples
///
/// ```
/// use aprender::format::ship_001::{verdict_from_safetensors_json_open_byte, Ship001Verdict};
///
/// assert_eq!(verdict_from_safetensors_json_open_byte(b'{'), Ship001Verdict::Pass);
/// assert_eq!(verdict_from_safetensors_json_open_byte(b'['), Ship001Verdict::Fail);
/// assert_eq!(verdict_from_safetensors_json_open_byte(b'"'), Ship001Verdict::Fail);
/// assert_eq!(verdict_from_safetensors_json_open_byte(0), Ship001Verdict::Fail);
/// ```
#[must_use]
pub const fn verdict_from_safetensors_json_open_byte(byte: u8) -> Ship001Verdict {
    if byte == AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE {
        Ship001Verdict::Pass
    } else {
        Ship001Verdict::Fail
    }
}

#[cfg(test)]
mod ship_001_tests {
    //! FALSIFY-SHIP-001 algorithm-level mutation survey.
    //!
    //! Each of the three verdict fns gets its own dedicated test section
    //! covering: the happy path, adjacent-value rejection, provenance
    //! pins on the bound constants, and conservative-Fail guards.
    //!
    //! The compute-heavy portion of AC-SHIP1-001 (actually calling
    //! `realizar::Model::load_safetensors(path)` on the real 7B teacher
    //! safetensors file) is out of scope; see `full_discharge_blocks_on`
    //! in `contracts/qwen2-e2e-verification-v1.yaml` FALSIFY-QW2E-SHIP-001.
    use super::*;

    #[test]
    fn falsify_ship_001_load_result_logic() {
        // Section 1: Ok → Pass.
        assert_eq!(
            verdict_from_load_result(true),
            Ship001Verdict::Pass,
            "Ok(Model) must Pass"
        );

        // Section 2: Err → Fail.
        assert_eq!(
            verdict_from_load_result(false),
            Ship001Verdict::Fail,
            "Err(_) must Fail"
        );

        // Section 3: Exhaustive bool — only Pass on true, only Fail on
        // false. No third state possible.
        for is_ok in [true, false] {
            let v = verdict_from_load_result(is_ok);
            let expected = if is_ok {
                Ship001Verdict::Pass
            } else {
                Ship001Verdict::Fail
            };
            assert_eq!(v, expected, "Mismatch at is_ok={is_ok}");
        }
    }

    #[test]
    fn falsify_ship_001_safetensors_header_size_logic() {
        // Section 1: Nominal — 8-byte prefix + 128-byte header + 1024
        // bytes of tensor data = 1160-byte file with header_len=128.
        assert_eq!(
            verdict_from_safetensors_header_size(128, 1160),
            Ship001Verdict::Pass,
            "Nominal safetensors layout must Pass"
        );

        // Section 2: Edge — header claims every remaining byte after
        // the 8-byte prefix. Valid but tensor-less (a metadata-only
        // safetensors file is legal).
        assert_eq!(
            verdict_from_safetensors_header_size(92, 100),
            Ship001Verdict::Pass,
            "Header filling exactly file_len - 8 must Pass"
        );

        // Section 3: Minimum valid file — 8-byte prefix + 1-byte
        // header. The JSON header `{}` is 2 bytes, so header_len=1 is
        // below a parseable JSON, but the *size invariant* (the thing
        // this verdict fn checks) holds. JSON validity is the job of
        // a separate gate.
        assert_eq!(
            verdict_from_safetensors_header_size(1, 9),
            Ship001Verdict::Pass,
            "header_len=1, file_len=9 must Pass at the size-invariant level"
        );

        // Section 4: Off-by-one Fail — header claims one more byte
        // than the file physically has.
        assert_eq!(
            verdict_from_safetensors_header_size(93, 100),
            Ship001Verdict::Fail,
            "header_len = file_len - 7 (one byte too many) must Fail"
        );

        // Section 5: Zero-length header Fail — malformed serializer
        // output with no metadata at all.
        assert_eq!(
            verdict_from_safetensors_header_size(0, 100),
            Ship001Verdict::Fail,
            "Zero-length header must Fail (malformed serializer output)"
        );

        // Section 6: File shorter than the 8-byte prefix — conservative
        // Fail, cannot even store the length prefix.
        for short_file in 0_u64..8 {
            assert_eq!(
                verdict_from_safetensors_header_size(0, short_file),
                Ship001Verdict::Fail,
                "file_len={short_file} (below prefix minimum) must Fail"
            );
            assert_eq!(
                verdict_from_safetensors_header_size(1, short_file),
                Ship001Verdict::Fail,
                "file_len={short_file} with nonzero header must Fail"
            );
        }

        // Section 7: Egregious over-claim — header claims u64::MAX
        // bytes on a modest file.
        assert_eq!(
            verdict_from_safetensors_header_size(u64::MAX, 1_000_000),
            Ship001Verdict::Fail,
            "header_len=u64::MAX on 1MB file must Fail"
        );

        // Section 8: Exhaustive small-file sweep. For file_len in
        // [8, 20], header_len in [0, 20], verify every pair matches
        // the invariant `0 < header_len <= file_len - 8`.
        for file_len in 8_u64..=20 {
            for header_len in 0_u64..=20 {
                let v = verdict_from_safetensors_header_size(header_len, file_len);
                let expected = if header_len > 0 && header_len <= file_len - 8 {
                    Ship001Verdict::Pass
                } else {
                    Ship001Verdict::Fail
                };
                assert_eq!(
                    v, expected,
                    "Mismatch at header_len={header_len}, file_len={file_len}"
                );
            }
        }

        // Section 9: Provenance pin on the 8-byte prefix constant.
        assert_eq!(
            AC_SHIP1_001_SAFETENSORS_HEADER_PREFIX_LEN, 8_u64,
            "safetensors prefix length drifted — format spec binds u64 LE = 8 bytes"
        );
    }

    #[test]
    fn falsify_ship_001_safetensors_json_open_byte_logic() {
        // Section 1: Canonical — `b'{'` (0x7B) Passes.
        assert_eq!(
            verdict_from_safetensors_json_open_byte(b'{'),
            Ship001Verdict::Pass,
            "ASCII open-brace `{{` must Pass"
        );
        assert_eq!(
            verdict_from_safetensors_json_open_byte(0x7B),
            Ship001Verdict::Pass,
            "Hex 0x7B = b'{{' must Pass"
        );

        // Section 2: Adjacent-ASCII Fail — bytes one off in either
        // direction Fail.
        assert_eq!(
            verdict_from_safetensors_json_open_byte(0x7A), // 'z'
            Ship001Verdict::Fail,
            "0x7A (one below `{{`) must Fail"
        );
        assert_eq!(
            verdict_from_safetensors_json_open_byte(0x7C), // '|'
            Ship001Verdict::Fail,
            "0x7C (one above `{{`) must Fail"
        );

        // Section 3: Classic confusable-bracket Fails — `[`, `(`, `<`.
        // A serializer that emitted a JSON array or a tuple literal or
        // an XML-ish tag would trip these.
        for bad_open in [b'[', b'(', b'<', b']', b')', b'>'] {
            assert_eq!(
                verdict_from_safetensors_json_open_byte(bad_open),
                Ship001Verdict::Fail,
                "Confusable-bracket byte {:?} must Fail",
                bad_open as char
            );
        }

        // Section 4: JSON-structural-adjacent Fails — quote, colon,
        // comma, whitespace. Each would indicate a serializer emitting
        // something other than a JSON object at offset 8.
        for bad_open in [b'"', b':', b',', b' ', b'\n', b'\t', b'\r'] {
            assert_eq!(
                verdict_from_safetensors_json_open_byte(bad_open),
                Ship001Verdict::Fail,
                "JSON-adjacent byte {:?} must Fail",
                bad_open as char
            );
        }

        // Section 5: Binary Fails — NUL, high bit set, etc. Catches a
        // serializer writing raw tensor bytes straight into the header
        // region.
        for bad_open in [0_u8, 0x01, 0x7F, 0x80, 0xFF] {
            assert_eq!(
                verdict_from_safetensors_json_open_byte(bad_open),
                Ship001Verdict::Fail,
                "Binary byte 0x{bad_open:02X} must Fail"
            );
        }

        // Section 6: Exhaustive u8 sweep — only 0x7B Passes.
        for b in 0_u8..=255 {
            let v = verdict_from_safetensors_json_open_byte(b);
            let expected = if b == 0x7B {
                Ship001Verdict::Pass
            } else {
                Ship001Verdict::Fail
            };
            assert_eq!(v, expected, "Mismatch at byte 0x{b:02X}");
        }

        // Section 7: Provenance pin on the open-byte constant.
        assert_eq!(
            AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE, b'{',
            "safetensors JSON open byte drifted — format binds ASCII `{{`"
        );
        assert_eq!(
            AC_SHIP1_001_SAFETENSORS_JSON_OPEN_BYTE, 0x7B_u8,
            "safetensors JSON open byte must be 0x7B"
        );
    }

    /// FALSIFY-QW2E-SHIP-001 YAML binding: parses
    /// `qwen2-e2e-verification-v1.yaml` and asserts the
    /// FALSIFY-QW2E-SHIP-001 falsification block has been wired
    /// (it was missing on disk in v1.7.0 despite the v1.6.0 changelog
    /// claim) AND advertises `discharge_status: DISCHARGED` (v1.8.0
    /// promotion via live `apr inspect` on the canonical teacher
    /// safetensors at noah-Lambda-Vector RTX 4090).
    /// Falsifier: if the contract is edited to drop the YAML block,
    /// downgrade the discharge marker, or remove the live evidence,
    /// this test fails.
    #[test]
    fn falsify_ship_001_yaml_binding_pins_discharged_status() {
        const CONTRACT_YAML: &str =
            include_str!("../../../../contracts/qwen2-e2e-verification-v1.yaml");

        let doc: serde_yaml::Value = serde_yaml::from_str(CONTRACT_YAML)
            .expect("qwen2-e2e-verification-v1.yaml must parse as YAML");

        let falsifications = doc["falsification_tests"]
            .as_sequence()
            .expect("falsification_tests must be a sequence in qwen2-e2e-verification-v1");
        let block = falsifications
            .iter()
            .find(|b| b["id"].as_str() == Some("FALSIFY-QW2E-SHIP-001"))
            .expect("FALSIFY-QW2E-SHIP-001 must exist in qwen2-e2e-verification-v1 (v1.8.0)");

        assert_eq!(
            block["discharge_status"].as_str(),
            Some("DISCHARGED"),
            "FALSIFY-QW2E-SHIP-001 must advertise DISCHARGED \
             (live `apr inspect` on canonical lambda-labs teacher safetensors at v1.8.0)",
        );
        assert!(
            block["ship_blocking"].as_bool().unwrap_or(false),
            "FALSIFY-QW2E-SHIP-001 must be ship_blocking=true",
        );
        assert!(
            block["discharged_evidence"].is_mapping(),
            "FALSIFY-QW2E-SHIP-001 DISCHARGED status requires a discharged_evidence \
             block documenting host, command, file size, tensor count, and live verdict",
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
            block["discharged_evidence"]["tensor_count"].as_u64(),
            Some(339),
            "discharged_evidence.tensor_count must equal 339 (Qwen2.5-Coder-7B canonical)",
        );
        assert_eq!(
            block["discharged_evidence"]["total_params"].as_u64(),
            Some(7_615_616_512),
            "discharged_evidence.total_params must equal 7,615,616,512 (Qwen2.5-Coder-7B canonical)",
        );
        let live_evidence = block["discharged_evidence"]["evidence_discharged_by_live"]
            .as_sequence()
            .expect(
                "FALSIFY-QW2E-SHIP-001 DISCHARGED requires \
                 discharged_evidence.evidence_discharged_by_live (live RTX 4090 evidence list)",
            );
        assert!(
            !live_evidence.is_empty(),
            "evidence_discharged_by_live must list at least one live observation",
        );
    }
}
