// SHIP-TWO-001 — `apr-cli-publish-extra-v1` algorithm-level
// PARTIAL discharge for FALSIFY-PUB-EXTRA-002 + 006.
//
// Contract: `contracts/apr-cli-publish-extra-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// Both gates protect MODEL-1 / MODEL-2 ship integrity:
// - PUB-EXTRA-002 is the **CRITICAL** sha256-mismatch guard
//   ("the whole point of manifest-first publish").
// - PUB-EXTRA-006 is the post-upload byte-integrity round-trip
//   ("byte-integrity lost in transit").

// ===========================================================================
// PUB-EXTRA-002 — sha256 mismatch must abort BEFORE upload
// ===========================================================================

/// Binary verdict for `FALSIFY-PUB-EXTRA-002`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtra002Verdict {
    /// `apr publish` exited non-zero AND made zero HF API calls.
    Pass,
    /// `apr publish` either exited 0 (didn't catch mismatch) OR
    /// made >= 1 HF API call (uploaded garbage before validating).
    Fail,
}

/// Pure verdict function for `FALSIFY-PUB-EXTRA-002`.
///
/// Inputs:
/// - `exit_code`: process exit from `apr publish` with corrupted
///   manifest sha256.
/// - `hf_api_call_count`: number of network requests made to
///   huggingface.co (0 means upload aborted pre-flight).
///
/// Pass iff `exit_code != 0 AND hf_api_call_count == 0`.
#[must_use]
pub fn verdict_from_sha256_mismatch_abort(
    exit_code: i32,
    hf_api_call_count: u64,
) -> PubExtra002Verdict {
    if exit_code == 0 {
        return PubExtra002Verdict::Fail;
    }
    if hf_api_call_count != 0 {
        return PubExtra002Verdict::Fail;
    }
    PubExtra002Verdict::Pass
}

// ===========================================================================
// PUB-EXTRA-006 — post-upload sha256 round-trip
// ===========================================================================

/// SHA-256 digest length in bytes.
pub const AC_PUB_EXTRA_006_SHA256_BYTES: usize = 32;

/// Binary verdict for `FALSIFY-PUB-EXTRA-006`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PubExtra006Verdict {
    /// Both digests are 32 bytes long AND byte-identical.
    Pass,
    /// Wrong digest length OR digests differ.
    Fail,
}

/// Pure verdict function for `FALSIFY-PUB-EXTRA-006`.
///
/// Inputs:
/// - `manifest_sha256`: the sha256 declared in publish-manifest.yaml.
/// - `pulled_sha256`: sha256 of the artifact after `apr pull` from HF.
///
/// Pass iff both 32 bytes long AND byte-identical.
#[must_use]
pub fn verdict_from_post_upload_roundtrip(
    manifest_sha256: &[u8],
    pulled_sha256: &[u8],
) -> PubExtra006Verdict {
    if manifest_sha256.len() != AC_PUB_EXTRA_006_SHA256_BYTES {
        return PubExtra006Verdict::Fail;
    }
    if pulled_sha256.len() != AC_PUB_EXTRA_006_SHA256_BYTES {
        return PubExtra006Verdict::Fail;
    }
    if manifest_sha256 == pulled_sha256 {
        PubExtra006Verdict::Pass
    } else {
        PubExtra006Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // PUB-EXTRA-002 tests
    // -------------------------------------------------------------------------
    #[test]
    fn p002_pass_aborted_pre_upload() {
        // Sha256 mismatch detected; apr exits 1 with zero HF calls.
        let v = verdict_from_sha256_mismatch_abort(1, 0);
        assert_eq!(v, PubExtra002Verdict::Pass);
    }

    #[test]
    fn p002_pass_clap_error_exit_2() {
        let v = verdict_from_sha256_mismatch_abort(2, 0);
        assert_eq!(v, PubExtra002Verdict::Pass);
    }

    #[test]
    fn p002_fail_silent_upload_exit_zero() {
        // CRITICAL regression: apr exited 0 despite the sha256
        // mismatch. The whole point of manifest-first publish
        // bypassed.
        let v = verdict_from_sha256_mismatch_abort(0, 0);
        assert_eq!(
            v,
            PubExtra002Verdict::Fail,
            "exit==0 with sha256 mismatch must Fail (CRITICAL)"
        );
    }

    #[test]
    fn p002_fail_uploaded_before_aborting() {
        // apr made HF API calls before catching the mismatch.
        let v = verdict_from_sha256_mismatch_abort(1, 1);
        assert_eq!(
            v,
            PubExtra002Verdict::Fail,
            "any HF call before abort must Fail (uploaded garbage)"
        );
    }

    #[test]
    fn p002_fail_many_uploads_before_abort() {
        let v = verdict_from_sha256_mismatch_abort(1, 100);
        assert_eq!(v, PubExtra002Verdict::Fail);
    }

    #[test]
    fn p002_fail_zero_exit_with_uploads() {
        let v = verdict_from_sha256_mismatch_abort(0, 5);
        assert_eq!(v, PubExtra002Verdict::Fail);
    }

    #[test]
    fn p002_pass_panic_exit_101_no_calls() {
        // Panic-driven exit (101) is still a valid abort if no HF
        // calls were made.
        let v = verdict_from_sha256_mismatch_abort(101, 0);
        assert_eq!(v, PubExtra002Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // PUB-EXTRA-006 tests
    // -------------------------------------------------------------------------
    #[test]
    fn p006_provenance_sha256_byte_length_is_32() {
        assert_eq!(AC_PUB_EXTRA_006_SHA256_BYTES, 32);
    }

    #[test]
    fn p006_pass_identical_digests() {
        let manifest = [0xab_u8; 32];
        let pulled = [0xab_u8; 32];
        let v = verdict_from_post_upload_roundtrip(&manifest, &pulled);
        assert_eq!(v, PubExtra006Verdict::Pass);
    }

    #[test]
    fn p006_pass_realistic_sha256() {
        let digest = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        let v = verdict_from_post_upload_roundtrip(&digest, &digest);
        assert_eq!(v, PubExtra006Verdict::Pass);
    }

    #[test]
    fn p006_fail_corruption_in_transit() {
        let manifest = [0xab_u8; 32];
        let mut pulled = [0xab_u8; 32];
        pulled[15] = 0xac;
        let v = verdict_from_post_upload_roundtrip(&manifest, &pulled);
        assert_eq!(
            v,
            PubExtra006Verdict::Fail,
            "in-transit corruption must Fail"
        );
    }

    #[test]
    fn p006_fail_first_byte_differs() {
        let manifest = [0xab_u8; 32];
        let mut pulled = [0xab_u8; 32];
        pulled[0] = 0xac;
        let v = verdict_from_post_upload_roundtrip(&manifest, &pulled);
        assert_eq!(v, PubExtra006Verdict::Fail);
    }

    #[test]
    fn p006_fail_last_byte_differs() {
        let manifest = [0xab_u8; 32];
        let mut pulled = [0xab_u8; 32];
        pulled[31] = 0xac;
        let v = verdict_from_post_upload_roundtrip(&manifest, &pulled);
        assert_eq!(v, PubExtra006Verdict::Fail);
    }

    #[test]
    fn p006_fail_manifest_too_short() {
        let v = verdict_from_post_upload_roundtrip(&[0u8; 31], &[0u8; 32]);
        assert_eq!(v, PubExtra006Verdict::Fail);
    }

    #[test]
    fn p006_fail_pulled_too_short() {
        let v = verdict_from_post_upload_roundtrip(&[0u8; 32], &[0u8; 16]);
        assert_eq!(v, PubExtra006Verdict::Fail);
    }

    #[test]
    fn p006_fail_both_empty() {
        let v = verdict_from_post_upload_roundtrip(&[], &[]);
        assert_eq!(v, PubExtra006Verdict::Fail);
    }

    #[test]
    fn p006_fail_both_wrong_length_but_equal() {
        // 16-byte arrays match each other → still Fail (wrong hash
        // function, e.g., MD5 substituted).
        let v = verdict_from_post_upload_roundtrip(&[0xab_u8; 16], &[0xab_u8; 16]);
        assert_eq!(
            v,
            PubExtra006Verdict::Fail,
            "matching-but-wrong-length must Fail"
        );
    }

    #[test]
    fn p006_fail_at_every_byte_position() {
        for pos in 0..32 {
            let manifest = [0xab_u8; 32];
            let mut pulled = [0xab_u8; 32];
            pulled[pos] ^= 0x01;
            let v = verdict_from_post_upload_roundtrip(&manifest, &pulled);
            assert_eq!(
                v,
                PubExtra006Verdict::Fail,
                "drift at position {pos} must Fail"
            );
        }
    }
}
