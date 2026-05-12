// SHIP-TWO-001 AC-SHIP1-010 / FALSIFY-SHIP-010 algorithm-level PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §4.2 row
// `AC-SHIP1-010  | Published artifact URL resolves; SHA-256 matches manifest`.
// Spec FALSIFY-SHIP-010 rule: "curl + sha256sum against manifest / hash
// mismatch or 404".
//
// Contract: contracts/publish-manifest-v1.yaml v1.3.0 → v1.4.0 (GATE-PM-010
// PARTIAL — binds AC-SHIP1-010 to the pure decision rule discharged here
// while leaving the live-network discharge to FALSIFY-PM-002 + FALSIFY-PM-003
// at publish time).
//
// AC-SHIP1-010 states that the MODEL-1 teacher artifact
// (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) — in safetensors, gguf, and apr
// form — MUST be reachable at its HF URL AND its downloaded bytes MUST
// sha256-match the manifest's declared hash. Any URL that 404s, any byte
// drift from CDN caching or re-upload, any manifest-hash staleness is a
// ship-blocker: downstream tools cannot tell whether they got the claimed
// bytes or a swapped replacement.
//
// This file discharges the *decision rules* at `PARTIAL_ALGORITHM_LEVEL`:
//
//   1. `verdict_from_sha256_match(expected, actual) -> Ship010Verdict`
//      — Pass iff both strings are exactly 64 lowercase hex chars AND
//      byte-identical. Fail for any length drift, any non-hex character,
//      any uppercase input (sha256sum output is canonical lowercase), any
//      mismatch.
//
//   2. `verdict_from_manifest_url(url) -> Ship010Verdict`
//      — Pass iff URL starts with `https://`, has a non-empty host
//      component (at least one char past the scheme), and contains no
//      whitespace or control characters. Fail for plaintext http://, bare
//      hostnames, whitespace injection.
//
// The compute-heavy portion of the AC (running `curl -sI $url` for the
// 200 OK + running `sha256sum model.gguf` on a ~5GiB download) is
// intentionally out of scope here; the threshold rule is what
// `apr validate-manifest --strict` must emit a Pass on, and changing either
// side of the bind (the 64 constant, the lowercase-hex shape, the https://
// prefix) breaks this test before any network I/O is launched.
//
// Mirrors the two-function pattern emerging across MODEL-1 PARTIAL
// discharges: SHIP-005 (HumanEval three-constant threshold), SHIP-007
// (decode-tps single threshold). SHIP-010 is the 6th MODEL-1 AC-SHIP1
// item touched (SHIP-002 + SHIP-005 + SHIP-006 + SHIP-007 + SHIP-008 +
// SHIP-009 + SHIP-010 — 7/10 once this lands).

/// Canonical SHA-256 digest length in lowercase hex characters. Binds
/// together the "64 hex chars" contract on every manifest's `sha256`
/// field with the `sha256sum` CLI output format consumed by
/// FALSIFY-PM-002 / FALSIFY-PM-003.
///
/// Derivation: SHA-256 produces a 256-bit digest = 32 bytes = 64 hex
/// characters. Pinned here so that contract drift in either direction
/// (accepting truncated hashes, accepting Base64 encoding, accepting
/// uppercase) is caught at compile+test time, not at a production publish.
/// Lockstep with `contracts/publish-manifest-v1.yaml` §schema.required_fields
/// (sha256 "64-char hex string, lowercase").
pub const AC_SHIP1_010_SHA256_HEX_LEN: usize = 64;

/// Required URL scheme prefix for a well-formed manifest `artifact_url`.
/// Plain-HTTP publishes are ship-blockers: without TLS the content-length
/// check in FALSIFY-PM-003 is trivially spoofable on public networks.
pub const AC_SHIP1_010_REQUIRED_URL_SCHEME: &str = "https://";

/// Binary verdict for FALSIFY-SHIP-010 / GATE-PM-010 / AC-SHIP1-010.
/// `Pass` iff every ship-gate decision rule bound here returns Pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ship010Verdict {
    /// Input conforms to every contract-bound decision rule.
    Pass,
    /// Input violates at least one decision rule (ship-blocker).
    Fail,
}

/// Algorithm-level verdict rule for the SHA-256 identity half of
/// FALSIFY-SHIP-010: given the manifest's declared digest and the
/// computed digest of the downloaded artifact, Pass iff both are
/// canonical 64-char lowercase-hex strings AND byte-identical.
///
/// Every failure mode is explicit:
/// - Wrong length on either side → Fail (catches truncation, Base64 slip,
///   accidental whitespace).
/// - Any non-hex character (incl. uppercase A-F, `sha256sum` emits
///   lowercase only) → Fail.
/// - Any bit of difference → Fail.
///
/// Returns [`Ship010Verdict::Fail`] conservatively for every malformed
/// input so a parser bug can never silently promote a garbage comparison
/// to Pass.
#[must_use]
pub fn verdict_from_sha256_match(expected_hex: &str, actual_hex: &str) -> Ship010Verdict {
    if expected_hex.len() != AC_SHIP1_010_SHA256_HEX_LEN
        || actual_hex.len() != AC_SHIP1_010_SHA256_HEX_LEN
    {
        return Ship010Verdict::Fail;
    }
    if !is_canonical_lowercase_hex(expected_hex) || !is_canonical_lowercase_hex(actual_hex) {
        return Ship010Verdict::Fail;
    }
    if expected_hex.as_bytes() == actual_hex.as_bytes() {
        Ship010Verdict::Pass
    } else {
        Ship010Verdict::Fail
    }
}

/// Algorithm-level verdict rule for the URL-resolves half of
/// FALSIFY-SHIP-010: given the manifest's `artifact_url`, Pass iff the
/// URL is syntactically well-formed enough to hand to `curl -sI` without
/// being rejected at the schema layer (plaintext HTTP, whitespace
/// injection, empty host, control chars are all Fail).
///
/// Returns [`Ship010Verdict::Fail`] conservatively; the live 200-OK check
/// against HF Hub is the full-discharge path in FALSIFY-PM-003, not here.
#[must_use]
pub fn verdict_from_manifest_url(url: &str) -> Ship010Verdict {
    if !url.starts_with(AC_SHIP1_010_REQUIRED_URL_SCHEME) {
        return Ship010Verdict::Fail;
    }
    let host_and_path = &url[AC_SHIP1_010_REQUIRED_URL_SCHEME.len()..];
    if host_and_path.is_empty() {
        return Ship010Verdict::Fail;
    }
    for b in url.bytes() {
        if b.is_ascii_whitespace() || b.is_ascii_control() {
            return Ship010Verdict::Fail;
        }
    }
    let host = host_and_path.split('/').next().unwrap_or("");
    if host.is_empty() {
        return Ship010Verdict::Fail;
    }
    Ship010Verdict::Pass
}

/// Helper: canonical `sha256sum` lowercase-hex — 0-9 a-f only. Uppercase
/// and mixed-case hex is rejected to preserve byte-identical comparison
/// semantics with the `sha256sum` CLI output that FALSIFY-PM-002
/// specifies.
#[must_use]
const fn is_canonical_lowercase_hex(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let ok = b.is_ascii_digit() || matches!(b, b'a'..=b'f');
        if !ok {
            return false;
        }
        i += 1;
    }
    true
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-SHIP-010 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod ship_010_tests {
    use super::*;

    /// FALSIFY-SHIP-010 algorithm-level PARTIAL discharge (SHA-256 half):
    /// prove the byte-identical comparison rule binding an expected
    /// manifest digest to a computed `sha256sum` digest. Any edit that
    /// relaxes the length check, silently accepts uppercase, or makes the
    /// comparison case-insensitive must break this test before a real
    /// `apr validate-manifest` publish gate runs.
    #[test]
    fn falsify_ship_010_sha256_match_threshold_logic() {
        // Section 1: byte-identical canonical hash — baseline Pass.
        let canonical = "4e8b3f2a9c1d0000000000000000000000000000000000000000000000000001";
        assert_eq!(canonical.len(), AC_SHIP1_010_SHA256_HEX_LEN);
        assert_eq!(
            verdict_from_sha256_match(canonical, canonical),
            Ship010Verdict::Pass,
            "byte-identical canonical lowercase hex must Pass",
        );

        // Section 2: single-hex-char flip — sharpest possible Fail
        // counter-example. Any mutation that turns `!=` into
        // "close-enough" comparison flips this to Pass.
        let flipped = "4e8b3f2a9c1d0000000000000000000000000000000000000000000000000002";
        assert_eq!(
            verdict_from_sha256_match(canonical, flipped),
            Ship010Verdict::Fail,
            "one-byte (last-nibble) difference must Fail",
        );

        // Section 3: wrong-length inputs on either side. 63 chars
        // (truncated CDN body), 65 chars (trailing newline from bad
        // yq pipeline), empty, and a length of 32 (raw-bytes-instead-
        // of-hex confusion) all must Fail.
        let too_short = "4e8b3f2a9c1d000000000000000000000000000000000000000000000000000";
        let too_long = "4e8b3f2a9c1d00000000000000000000000000000000000000000000000000012";
        let raw_bytes_like = "4e8b3f2a9c1d00000000000000000000";
        assert_eq!(too_short.len(), AC_SHIP1_010_SHA256_HEX_LEN - 1);
        assert_eq!(too_long.len(), AC_SHIP1_010_SHA256_HEX_LEN + 1);
        assert_eq!(
            verdict_from_sha256_match(too_short, canonical),
            Ship010Verdict::Fail,
            "63-char truncated expected must Fail",
        );
        assert_eq!(
            verdict_from_sha256_match(canonical, too_long),
            Ship010Verdict::Fail,
            "65-char padded actual must Fail",
        );
        assert_eq!(
            verdict_from_sha256_match("", canonical),
            Ship010Verdict::Fail,
            "empty expected must Fail",
        );
        assert_eq!(
            verdict_from_sha256_match(raw_bytes_like, canonical),
            Ship010Verdict::Fail,
            "32-char raw-bytes slip must Fail",
        );

        // Section 4: uppercase / mixed-case hex rejected. sha256sum
        // emits lowercase only; accepting uppercase here would let a
        // manifest declaring the "wrong" casing silently compare-equal
        // to real CLI output (it doesn't — this gate rejects pre-
        // comparison).
        let uppercase = "4E8B3F2A9C1D0000000000000000000000000000000000000000000000000001";
        let mixed_case = "4e8B3f2A9c1D0000000000000000000000000000000000000000000000000001";
        assert_eq!(
            verdict_from_sha256_match(uppercase, canonical),
            Ship010Verdict::Fail,
            "all-uppercase expected must Fail (sha256sum is lowercase)",
        );
        assert_eq!(
            verdict_from_sha256_match(canonical, mixed_case),
            Ship010Verdict::Fail,
            "mixed-case actual must Fail (sha256sum is lowercase)",
        );

        // Section 5: non-hex characters (g-z, punctuation, whitespace).
        // A yq extraction bug that leaves `sha256: ` in the captured
        // field, or a sha256sum output that retains `  filename` must
        // not silently Pass.
        let non_hex_g = "ge8b3f2a9c1d0000000000000000000000000000000000000000000000000001";
        let with_space = "4e8b3f2a9c1d000000000000000000000000000000000000000000000000000 ";
        let with_punct = "4e8b3f2a9c1d000000000000000000000000000000000000000000000000000!";
        assert_eq!(
            verdict_from_sha256_match(non_hex_g, canonical),
            Ship010Verdict::Fail,
            "non-hex `g` prefix must Fail",
        );
        assert_eq!(
            verdict_from_sha256_match(canonical, with_space),
            Ship010Verdict::Fail,
            "trailing whitespace in actual must Fail",
        );
        assert_eq!(
            verdict_from_sha256_match(canonical, with_punct),
            Ship010Verdict::Fail,
            "trailing punctuation in actual must Fail",
        );

        // Section 6: all-zero digest (often used as a sentinel / default
        // in bad code paths). A stale-default manifest MUST NOT compare
        // equal to a real sha256sum — this guards against an "empty
        // pipeline" failure mode where both sides were initialized to
        // zeros and nothing ever wrote them.
        let all_zero = "0000000000000000000000000000000000000000000000000000000000000000";
        assert_eq!(
            verdict_from_sha256_match(all_zero, all_zero),
            Ship010Verdict::Pass,
            "two equal zero strings must Pass per byte-identity rule \
             (the defect is elsewhere: why did both sides produce zeros?)",
        );
        assert_eq!(
            verdict_from_sha256_match(all_zero, canonical),
            Ship010Verdict::Fail,
            "zero vs real digest must Fail",
        );

        // Section 7: provenance pin — the length constant is load-
        // bearing. If AC-SHIP1-010 ever adopts SHA-384 (96 hex chars)
        // or SHA-512 (128 hex chars), this constant and every test that
        // manually constructed a 64-char fixture must move together.
        assert_eq!(
            AC_SHIP1_010_SHA256_HEX_LEN, 64,
            "SHA-256 canonical hex length is 64 \
             (spec §4.2 AC-SHIP1-010; publish-manifest-v1 schema)",
        );
    }

    /// FALSIFY-SHIP-010 algorithm-level PARTIAL discharge (URL half):
    /// prove the URL well-formedness rule binding an `artifact_url`
    /// manifest field to the shape that `curl -sI` can handle without
    /// schema-layer rejection. Any edit that relaxes the https:// prefix,
    /// accepts whitespace injection, or allows empty hosts must break
    /// this test before a real publish gate runs.
    #[test]
    fn falsify_ship_010_manifest_url_well_formedness() {
        // Section 1: canonical HF-Hub-style URL — baseline Pass.
        // Mirrors the published teacher URL shape:
        // https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1/resolve/main/model.gguf
        assert_eq!(
            verdict_from_manifest_url(
                "https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1/resolve/main/model.gguf"
            ),
            Ship010Verdict::Pass,
            "canonical HF Hub resolve URL must Pass",
        );
        assert_eq!(
            verdict_from_manifest_url("https://paiml-public.s3.amazonaws.com/models/m.apr"),
            Ship010Verdict::Pass,
            "canonical S3 public URL must Pass",
        );

        // Section 2: plaintext HTTP rejected. Catches the class where
        // a manifest shipped with http:// gets silently accepted, and
        // the content-length check in FALSIFY-PM-003 becomes trivially
        // MITM-spoofable on any public network.
        assert_eq!(
            verdict_from_manifest_url(
                "http://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1/resolve/main/model.gguf"
            ),
            Ship010Verdict::Fail,
            "plaintext http:// must Fail (TLS is required for tamper-proof publish)",
        );

        // Section 3: scheme-less / bare hostname / wrong-scheme must
        // Fail. Catches manifests where someone pasted a file path,
        // an HF repo slug, or an ftp://-style legacy URL.
        assert_eq!(
            verdict_from_manifest_url("huggingface.co/paiml/model.gguf"),
            Ship010Verdict::Fail,
            "scheme-less URL must Fail",
        );
        assert_eq!(
            verdict_from_manifest_url("paiml/qwen2.5-coder-7b-apache-q4k-v1"),
            Ship010Verdict::Fail,
            "bare HF repo slug must Fail",
        );
        assert_eq!(
            verdict_from_manifest_url("ftp://paiml.com/model.gguf"),
            Ship010Verdict::Fail,
            "ftp:// scheme must Fail",
        );
        assert_eq!(
            verdict_from_manifest_url("/absolute/path/to/model.gguf"),
            Ship010Verdict::Fail,
            "absolute file path must Fail",
        );

        // Section 4: empty host after https:// must Fail. Catches the
        // trivially-malformed https:// and https:/// shapes.
        assert_eq!(
            verdict_from_manifest_url("https://"),
            Ship010Verdict::Fail,
            "https:// with no host must Fail",
        );
        assert_eq!(
            verdict_from_manifest_url("https:///paiml/model.gguf"),
            Ship010Verdict::Fail,
            "https:/// with empty host must Fail",
        );

        // Section 5: whitespace / control-char injection. A manifest
        // that crossed a yq | cat boundary and picked up a trailing
        // `\n` or a stray space must not silently Pass — that's how
        // curl gets a URL it can't resolve at publish time.
        assert_eq!(
            verdict_from_manifest_url(
                "https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1/resolve/main/m.gguf\n"
            ),
            Ship010Verdict::Fail,
            "trailing newline must Fail",
        );
        assert_eq!(
            verdict_from_manifest_url("https://huggingface.co/paiml /model.gguf"),
            Ship010Verdict::Fail,
            "embedded space must Fail",
        );
        assert_eq!(
            verdict_from_manifest_url("https://huggingface.co/paiml\tmodel.gguf"),
            Ship010Verdict::Fail,
            "embedded tab must Fail",
        );
        assert_eq!(
            verdict_from_manifest_url("https://huggingface.co/paiml\rmodel.gguf"),
            Ship010Verdict::Fail,
            "embedded CR must Fail",
        );
        assert_eq!(
            verdict_from_manifest_url("https://huggingface.co/paiml\x00model.gguf"),
            Ship010Verdict::Fail,
            "embedded NUL must Fail",
        );

        // Section 6: empty string. Degenerate input; must Fail Pass-
        // unreachable to confirm no special-case Pass hides in the
        // prefix check.
        assert_eq!(
            verdict_from_manifest_url(""),
            Ship010Verdict::Fail,
            "empty URL must Fail",
        );

        // Section 7: provenance pin — the required scheme constant is
        // load-bearing and lockstepped with the spec's TLS floor. If
        // AC-SHIP1-010 ever relaxes this (e.g. to allow https:// or
        // http:// against an internal mirror), this constant and this
        // test must move together.
        assert_eq!(
            AC_SHIP1_010_REQUIRED_URL_SCHEME, "https://",
            "TLS floor is https:// (spec §4.2 AC-SHIP1-010; no plaintext publish)",
        );
    }

    /// FALSIFY-SHIP-010 helper proof: `is_canonical_lowercase_hex`
    /// rejects every non-0-9-a-f byte and accepts every valid one.
    /// Guards against a refactor that relaxes to
    /// `is_ascii_hexdigit()` (which would silently accept A-F).
    #[test]
    fn is_canonical_lowercase_hex_rejects_uppercase() {
        assert!(is_canonical_lowercase_hex("0123456789abcdef"));
        assert!(!is_canonical_lowercase_hex("0123456789ABCDEF"));
        assert!(!is_canonical_lowercase_hex("0123456789abcdeF"));
        assert!(!is_canonical_lowercase_hex("xyz"));
        assert!(!is_canonical_lowercase_hex("abcdef "));
        assert!(is_canonical_lowercase_hex(""));
    }

    /// FALSIFY-SHIP-010 YAML binding: parses publish-manifest-v1.yaml
    /// and asserts the FALSIFY-SHIP-010 falsification block has been
    /// promoted from `PARTIAL_ALGORITHM_LEVEL` (v1.4.0) → `DISCHARGED`
    /// (v1.5.0) via live `apr validate-manifest --live --json` evidence
    /// on noah-Lambda-Vector RTX 4090.
    /// Falsifier: if the contract is edited to drop the live-evidence
    /// block or downgrade the discharge marker, this test fails before
    /// any network I/O is launched.
    #[test]
    fn falsify_ship_010_yaml_binding_pins_discharged_status() {
        const CONTRACT_YAML: &str = include_str!("../../../../contracts/publish-manifest-v1.yaml");

        let doc: serde_yaml::Value = serde_yaml::from_str(CONTRACT_YAML)
            .expect("publish-manifest-v1.yaml must parse as YAML");

        let falsifications = doc["falsification_tests"]
            .as_sequence()
            .expect("falsification_tests must be a sequence in publish-manifest-v1");
        let block = falsifications
            .iter()
            .find(|b| b["id"].as_str() == Some("FALSIFY-SHIP-010"))
            .expect("FALSIFY-SHIP-010 must exist in publish-manifest-v1");

        assert_eq!(
            block["binds_to"].as_str(),
            Some("AC-SHIP1-010"),
            "FALSIFY-SHIP-010 must bind AC-SHIP1-010 (Published artifact URL + SHA-256)",
        );
        assert_eq!(
            block["discharge_status"].as_str(),
            Some("DISCHARGED"),
            "FALSIFY-SHIP-010 must advertise DISCHARGED \
             (live `apr validate-manifest --live` PASS on all 3 paiml \
             manifests at v1.5.0; previous PARTIAL_ALGORITHM_LEVEL at v1.4.0)",
        );
        assert!(
            block["discharged_evidence"].is_mapping(),
            "FALSIFY-SHIP-010 DISCHARGED status requires discharged_evidence \
             block documenting the host, command, and per-manifest live verdicts",
        );
        assert_eq!(
            block["discharged_evidence"]["host"].as_str(),
            Some("noah-Lambda-Vector"),
            "discharged_evidence.host must pin to the lambda-labs RTX 4090 host",
        );
        let manifests = block["discharged_evidence"]["manifests"]
            .as_sequence()
            .expect("discharged_evidence.manifests must be a sequence");
        assert_eq!(
            manifests.len(),
            3,
            "all 3 paiml-qwen2.5-coder-7b-apache-q4k-v1 manifests \
             (apr/gguf/safetensors) must show live PASS",
        );
        for m in manifests {
            assert_eq!(
                m["overall"].as_str(),
                Some("PASS"),
                "every manifest in discharged_evidence.manifests must have overall=PASS",
            );
        }
    }
}
