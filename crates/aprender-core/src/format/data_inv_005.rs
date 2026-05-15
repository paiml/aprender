// SHIP-TWO-001 MODEL-2 — `dataset-thestack-python-v1` (C-DATA-THESTACK-PYTHON)
// algorithm-level PARTIAL discharge for INV-DATA-005.
//
// Contract: `contracts/dataset-thestack-python-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pipeline (§26.2), AC-SHIP2-002.
//
// ## What INV-DATA-005 says
//
//   description: corpus_sha256 declared in manifest matches the
//                recomputed merkle-style sha256 over sorted shard
//                sha256s. Anyone re-ingesting the same source.revision_sha
//                with the same seed gets the same corpus_sha256.
//   falsifier:   On a second host, re-run ingest with the same
//                revision_sha and same seed; compare corpus_sha256.
//                Mismatch → FAIL.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two recomputed corpus_sha256 byte-arrays from
// two independent ingest hosts using the same source.revision_sha
// and same seed, Pass iff:
//
//   host_a == host_b (byte-identical, all 32 bytes)
//
// AND both digests are well-formed (32 bytes each — SHA-256 output
// length). Composes byte-equality with provenance pinning the
// expected SHA-256 digest length. The contract falsifier ("Mismatch
// → FAIL") admits no near-equality band; reproducibility is binary.

/// Expected length of a SHA-256 digest in bytes.
///
/// Per RFC 6234 / FIPS 180-4: SHA-256 emits a 256-bit (32-byte)
/// output. Pinning this constant catches a regression where the
/// scanner truncates or pads digests, OR where a future drift to
/// SHA-3-256 or BLAKE3 silently changes the manifest representation
/// without bumping the contract.
pub const AC_DATA_INV_005_SHA256_BYTES: usize = 32;

/// Binary verdict for `INV-DATA-005`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataInv005Verdict {
    /// Both digests are 32 bytes long AND byte-identical.
    Pass,
    /// One or more of:
    /// - Either digest is not 32 bytes long (caller error — wrong
    ///   hash function, truncation, padding bug).
    /// - Digests differ in any byte (reproducibility violation).
    Fail,
}

/// Pure verdict function for `INV-DATA-005`.
///
/// Inputs:
/// - `host_a`: corpus_sha256 from the first ingest host.
/// - `host_b`: corpus_sha256 from the second ingest host (re-running
///   with same source.revision_sha + same seed).
///
/// Pass iff:
/// 1. `host_a.len() == AC_DATA_INV_005_SHA256_BYTES` (32),
/// 2. `host_b.len() == AC_DATA_INV_005_SHA256_BYTES` (32),
/// 3. `host_a == host_b` (byte-identical).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Two independent hosts produce identical 32-byte digest — `Pass`:
/// ```
/// use aprender::format::data_inv_005::{
///     verdict_from_corpus_sha256_pair, DataInv005Verdict,
/// };
/// let host_a = [0xab_u8; 32];
/// let host_b = [0xab_u8; 32];
/// let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
/// assert_eq!(v, DataInv005Verdict::Pass);
/// ```
///
/// Single-byte mismatch — `Fail`:
/// ```
/// use aprender::format::data_inv_005::{
///     verdict_from_corpus_sha256_pair, DataInv005Verdict,
/// };
/// let host_a = [0xab_u8; 32];
/// let mut host_b = [0xab_u8; 32];
/// host_b[0] = 0xac;
/// let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
/// assert_eq!(v, DataInv005Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_corpus_sha256_pair(host_a: &[u8], host_b: &[u8]) -> DataInv005Verdict {
    if host_a.len() != AC_DATA_INV_005_SHA256_BYTES {
        return DataInv005Verdict::Fail;
    }
    if host_b.len() != AC_DATA_INV_005_SHA256_BYTES {
        return DataInv005Verdict::Fail;
    }
    if host_a == host_b {
        DataInv005Verdict::Pass
    } else {
        DataInv005Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — SHA-256 is exactly 32 bytes.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_sha256_byte_length_is_32() {
        assert_eq!(AC_DATA_INV_005_SHA256_BYTES, 32);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — identical digests.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_two_identical_digests_all_zeros() {
        let host_a = [0u8; 32];
        let host_b = [0u8; 32];
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Pass);
    }

    #[test]
    fn pass_two_identical_digests_all_ones() {
        let host_a = [0xff_u8; 32];
        let host_b = [0xff_u8; 32];
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Pass);
    }

    #[test]
    fn pass_realistic_sha256_pattern() {
        // A plausible non-trivial SHA-256 digest.
        let digest = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        let v = verdict_from_corpus_sha256_pair(&digest, &digest);
        assert_eq!(v, DataInv005Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — single-byte mismatch (reproducibility violation).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_first_byte_differs() {
        let host_a = [0xab_u8; 32];
        let mut host_b = [0xab_u8; 32];
        host_b[0] = 0xac;
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Fail, "single-byte mismatch must Fail");
    }

    #[test]
    fn fail_last_byte_differs() {
        let host_a = [0xab_u8; 32];
        let mut host_b = [0xab_u8; 32];
        host_b[31] = 0xac;
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Fail);
    }

    #[test]
    fn fail_middle_byte_differs() {
        let host_a = [0xab_u8; 32];
        let mut host_b = [0xab_u8; 32];
        host_b[15] = 0xac;
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Fail);
    }

    #[test]
    fn fail_one_bit_differs() {
        // Smallest possible mismatch: one bit flipped at byte 0.
        let host_a = [0u8; 32];
        let mut host_b = [0u8; 32];
        host_b[0] = 0x01;
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Fail);
    }

    #[test]
    fn fail_completely_different() {
        let host_a = [0x00_u8; 32];
        let host_b = [0xff_u8; 32];
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — caller errors (wrong digest length).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_host_a_too_short() {
        let host_a = [0u8; 31]; // SHA-1 length
        let host_b = [0u8; 32];
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Fail);
    }

    #[test]
    fn fail_host_b_too_short() {
        let host_a = [0u8; 32];
        let host_b = [0u8; 16]; // MD5 length
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Fail);
    }

    #[test]
    fn fail_both_zero_length() {
        let v = verdict_from_corpus_sha256_pair(&[], &[]);
        assert_eq!(
            v,
            DataInv005Verdict::Fail,
            "empty digests must Fail (caller error)"
        );
    }

    #[test]
    fn fail_host_a_too_long() {
        let host_a = [0u8; 64]; // SHA-512 length
        let host_b = [0u8; 32];
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(v, DataInv005Verdict::Fail);
    }

    #[test]
    fn fail_both_wrong_length_but_equal() {
        // Two 16-byte arrays that match each other — must still
        // Fail because they aren't SHA-256 length. Catches a
        // regression that would silently accept MD5 collisions.
        let host_a = [0xab_u8; 16];
        let host_b = [0xab_u8; 16];
        let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        assert_eq!(
            v,
            DataInv005Verdict::Fail,
            "matching but wrong-length digests must Fail"
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep — every byte position differing.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_at_every_byte_position() {
        // For each of the 32 byte positions, flipping one byte must
        // Fail. Catches a regression that compares only a prefix or
        // a hash of the digest.
        for pos in 0..32 {
            let host_a = [0xab_u8; 32];
            let mut host_b = [0xab_u8; 32];
            host_b[pos] ^= 0x01;
            let v = verdict_from_corpus_sha256_pair(&host_a, &host_b);
            assert_eq!(
                v,
                DataInv005Verdict::Fail,
                "byte position {pos} flip must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry — verdict is symmetric in (a, b).
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_is_symmetric() {
        let host_a = [0u8; 32];
        let mut host_b = [0u8; 32];
        host_b[7] = 0xff;

        let ab = verdict_from_corpus_sha256_pair(&host_a, &host_b);
        let ba = verdict_from_corpus_sha256_pair(&host_b, &host_a);
        assert_eq!(ab, ba, "verdict must be symmetric in (a, b)");
        assert_eq!(ab, DataInv005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — well-known SHA-256 of empty string.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_well_known_sha256_empty_string() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        // Two independent ingest hosts both producing this digest →
        // Pass.
        let empty_sha256 = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        let v = verdict_from_corpus_sha256_pair(&empty_sha256, &empty_sha256);
        assert_eq!(v, DataInv005Verdict::Pass);
    }
}
