// SHIP-TWO-001 §6 Compound Ship Gates — GATE-SHIP-005 algorithm-level
// PARTIAL discharge.
//
// Spec: docs/specifications/aprender-train/ship-two-models-spec.md §6 row
// `GATE-SHIP-005 | License metadata is present AND matches upstream
// declaration | publish`.
// Contract: contracts/compound-ship-gates-v1.yaml v1.0.0 PROPOSED
// (FALSIFY-GATE-SHIP-005 — wired in the same PR as this file lands).
//
// GATE-SHIP-005 states that every published artifact MUST carry a
// `license` field in its metadata AND that value MUST byte-equally
// match the upstream declaration (the license in the parent-model
// card or in the distillation/pretraining source's HF repo). Any
// drift — case change, trailing whitespace, missing field — is a
// compliance ship-blocker: a downstream consumer that reads the
// apr/gguf metadata and dispatches on license strings ("apache-2.0"
// vs "Apache-2.0") MUST see the canonical upstream form.
//
// This file discharges the *decision rule* at `PARTIAL_ALGORITHM_LEVEL`:
// given the model's declared license string and the upstream source's
// declared license string, the verdict is `Pass` iff both are
// non-empty, both are ASCII-printable (rejects emoji, control chars,
// BOM), AND byte-equal (case-sensitive). The compute-heavy portion
// (actually fetching the upstream HF card, parsing YAML front-matter,
// resolving the license field) is intentionally out of scope here.
//
// Case-sensitivity rationale: SPDX license IDs are case-sensitive
// (`Apache-2.0` is the canonical form; `apache-2.0` is non-canonical
// per SPDX). A drift in casing means someone silently normalized the
// string, which invalidates downstream consumers that compare against
// the canonical SPDX list. We enforce the upstream-declared casing
// verbatim — case drift is drift.

/// Name of the metadata field that MUST carry the license string.
/// Lockstep with `apr-provenance-v1.yaml` §required_fields and the
/// SPDX-licenses-list consumer conventions in
/// `crates/aprender-core/src/format/model_card.rs`.
pub const AC_GATE_SHIP_005_REQUIRED_LICENSE_FIELD: &str = "license";

/// Binary verdict for FALSIFY-GATE-SHIP-005 / GATE-SHIP-005.
/// `Pass` iff both inputs are non-empty, ASCII-printable, AND
/// byte-equal. `Fail` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateShip005Verdict {
    /// Both license strings are non-empty, ASCII-printable, and
    /// byte-equal. The published artifact's license metadata matches
    /// the upstream declaration verbatim.
    Pass,
    /// Any of: empty string on either side; non-printable byte
    /// (control char, non-ASCII); byte mismatch (incl. case drift or
    /// trailing whitespace). Compliance gate fails; publish blocked.
    Fail,
}

/// Algorithm-level verdict rule for FALSIFY-GATE-SHIP-005 /
/// GATE-SHIP-005: license metadata well-formedness + upstream-parity.
///
/// Conservative-Fail guards:
///
///   - Empty string on either side → Fail (missing license is a
///     compliance ship-blocker).
///   - Any non-ASCII-printable byte on either side → Fail (license
///     strings are SPDX-style ASCII; emoji / NUL / tab / BOM are
///     harness bugs or injection attempts).
///   - Byte mismatch → Fail (case drift, trailing whitespace,
///     punctuation normalization are all drift classes).
///
/// # Examples
///
/// ```
/// use aprender::format::gate_ship_005::{
///     verdict_from_license_metadata, GateShip005Verdict,
/// };
///
/// // Canonical upstream license match → Pass.
/// assert_eq!(
///     verdict_from_license_metadata("Apache-2.0", "Apache-2.0"),
///     GateShip005Verdict::Pass
/// );
///
/// // Case drift → Fail (SPDX IDs are case-sensitive).
/// assert_eq!(
///     verdict_from_license_metadata("apache-2.0", "Apache-2.0"),
///     GateShip005Verdict::Fail
/// );
/// ```
#[must_use]
pub fn verdict_from_license_metadata(
    model_license: &str,
    upstream_license: &str,
) -> GateShip005Verdict {
    if model_license.is_empty() || upstream_license.is_empty() {
        return GateShip005Verdict::Fail;
    }
    if !is_ascii_printable(model_license) || !is_ascii_printable(upstream_license) {
        return GateShip005Verdict::Fail;
    }
    if model_license.as_bytes() == upstream_license.as_bytes() {
        GateShip005Verdict::Pass
    } else {
        GateShip005Verdict::Fail
    }
}

/// Helper: ASCII-printable means every byte is in the range `0x20..=0x7E`
/// (space through `~`). Rejects control chars (incl. NUL, tab, CR, LF),
/// DEL (0x7F), and all non-ASCII bytes (emoji, UTF-8 multi-byte).
///
/// Whitespace-before-newline is NOT printable (tab, CR, LF are all
/// control chars < 0x20). Trailing space IS printable (0x20) but will
/// cause a byte-equal mismatch at the next check, which is the
/// desired outcome — a trailing space in one side but not the other
/// is drift.
#[must_use]
const fn is_ascii_printable(s: &str) -> bool {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x20 || b > 0x7E {
            return false;
        }
        i += 1;
    }
    true
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GATE-SHIP-005 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod gate_ship_005_tests {
    use super::*;

    /// FALSIFY-GATE-SHIP-005 algorithm-level PARTIAL discharge: prove
    /// the license-metadata byte-equal comparison rule. Any edit that
    /// relaxes to case-insensitive, trims whitespace, or silently
    /// normalizes must break this test.
    #[test]
    fn falsify_gate_ship_005_license_metadata_match() {
        // Section 1: happy path — canonical SPDX IDs match verbatim.
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0", "Apache-2.0"),
            GateShip005Verdict::Pass,
            "canonical Apache-2.0 match must Pass",
        );
        assert_eq!(
            verdict_from_license_metadata("MIT", "MIT"),
            GateShip005Verdict::Pass,
            "canonical MIT match must Pass",
        );
        assert_eq!(
            verdict_from_license_metadata("Qwen-License-Agreement-v1", "Qwen-License-Agreement-v1"),
            GateShip005Verdict::Pass,
            "custom upstream-declared license (verbatim match) must Pass",
        );

        // Section 2: case drift — SPDX IDs are case-sensitive. Any
        // silent normalization (upper→lower or title-case shift) Fails.
        assert_eq!(
            verdict_from_license_metadata("apache-2.0", "Apache-2.0"),
            GateShip005Verdict::Fail,
            "case drift (lower vs canonical) must Fail — SPDX IDs are case-sensitive",
        );
        assert_eq!(
            verdict_from_license_metadata("APACHE-2.0", "Apache-2.0"),
            GateShip005Verdict::Fail,
            "case drift (upper vs canonical) must Fail",
        );
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0", "apache-2.0"),
            GateShip005Verdict::Fail,
            "case drift (canonical vs lower) must Fail (symmetric)",
        );

        // Section 3: empty string on either side → Fail (missing
        // license is a compliance ship-blocker).
        assert_eq!(
            verdict_from_license_metadata("", "Apache-2.0"),
            GateShip005Verdict::Fail,
            "empty model license must Fail — compliance ship-blocker",
        );
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0", ""),
            GateShip005Verdict::Fail,
            "empty upstream license must Fail",
        );
        assert_eq!(
            verdict_from_license_metadata("", ""),
            GateShip005Verdict::Fail,
            "both empty must Fail (no evidence of license declaration)",
        );

        // Section 4: non-ASCII-printable bytes — emoji, control chars,
        // BOM, NUL all Fail. Catches harness bugs or injection
        // attempts.
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0\n", "Apache-2.0"),
            GateShip005Verdict::Fail,
            "trailing newline (0x0A control char) must Fail",
        );
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0\0", "Apache-2.0"),
            GateShip005Verdict::Fail,
            "embedded NUL must Fail",
        );
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0\t", "Apache-2.0"),
            GateShip005Verdict::Fail,
            "embedded tab must Fail",
        );
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0", "\u{FEFF}Apache-2.0"),
            GateShip005Verdict::Fail,
            "leading BOM must Fail (non-ASCII)",
        );
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0", "Apache-2.0"),
            GateShip005Verdict::Pass,
            "harness sanity: ASCII-only canonical must still Pass",
        );

        // Section 5: trailing-whitespace drift — space (0x20) IS
        // ASCII-printable so it passes the first guard, but then fails
        // the byte-equal check. This is the subtle drift class where
        // `"Apache-2.0" != "Apache-2.0 "` looks the same visually.
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0 ", "Apache-2.0"),
            GateShip005Verdict::Fail,
            "trailing space (only on model side) must Fail — drift guard",
        );
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0", " Apache-2.0"),
            GateShip005Verdict::Fail,
            "leading space (only on upstream side) must Fail",
        );
        assert_eq!(
            verdict_from_license_metadata("Apache-2.0  ", "Apache-2.0 "),
            GateShip005Verdict::Fail,
            "differing-amount-of-trailing-space must Fail",
        );

        // Section 6: provenance pin — the required-field constant is
        // load-bearing and lockstepped with apr-provenance-v1.yaml. If
        // the metadata field is ever renamed (`license` → `spdx_id`
        // or `license_id`), this constant and every consumer must
        // move together.
        assert_eq!(
            AC_GATE_SHIP_005_REQUIRED_LICENSE_FIELD, "license",
            "required metadata field is `license` \
             (spec §6 GATE-SHIP-005; apr-provenance-v1.yaml)",
        );
    }
}
