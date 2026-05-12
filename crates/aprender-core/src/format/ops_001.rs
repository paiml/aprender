// SHIP-TWO-001 — `apr-cli-operations-v1` algorithm-level PARTIAL
// discharge for FALSIFY-OPS-001.
//
// Contract: `contracts/apr-cli-operations-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What FALSIFY-OPS-001 says
//
//   rule: ReadOnly commands have no side effects
//   prediction: apr check does not modify any file (before/after
//               hash identical)
//   test: Snapshot filesystem, run apr check, diff snapshot
//   if_fails: Read-only command has hidden file mutation
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two filesystem-snapshot SHA-256 digests
// (before / after running a "read-only" command), Pass iff:
//
//   before is 32 bytes long AND
//   after is 32 bytes long AND
//   before == after (byte-identical)
//
// Mirrors `data_inv_005`'s shape (corpus_sha256 reproducibility),
// applied to filesystem snapshots. The 32-byte SHA-256 length
// pin catches hash-function drift. Byte-identical match catches
// any file mutation (mtime/size/content) that a read-only
// command should not introduce.

/// SHA-256 digest length in bytes.
pub const AC_OPS_001_SHA256_BYTES: usize = 32;

/// Binary verdict for `FALSIFY-OPS-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ops001Verdict {
    /// Both digests are 32 bytes long AND byte-identical.
    Pass,
    /// One or more of:
    /// - Either digest is not 32 bytes (caller error — wrong
    ///   hash function, truncation, padding bug).
    /// - Digests differ in any byte (read-only command mutated
    ///   the filesystem).
    Fail,
}

/// Pure verdict function for `FALSIFY-OPS-001`.
///
/// Inputs:
/// - `before_snapshot`: SHA-256 of the filesystem state before
///   running the command.
/// - `after_snapshot`: SHA-256 of the filesystem state after the
///   command exited.
///
/// Pass iff:
/// 1. `before_snapshot.len() == 32`,
/// 2. `after_snapshot.len() == 32`,
/// 3. `before_snapshot == after_snapshot` (byte-identical).
///
/// Otherwise `Fail`.
#[must_use]
pub fn verdict_from_filesystem_snapshot_pair(
    before_snapshot: &[u8],
    after_snapshot: &[u8],
) -> Ops001Verdict {
    if before_snapshot.len() != AC_OPS_001_SHA256_BYTES {
        return Ops001Verdict::Fail;
    }
    if after_snapshot.len() != AC_OPS_001_SHA256_BYTES {
        return Ops001Verdict::Fail;
    }
    if before_snapshot == after_snapshot {
        Ops001Verdict::Pass
    } else {
        Ops001Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — SHA-256 32-byte length.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_sha256_byte_length_is_32() {
        assert_eq!(AC_OPS_001_SHA256_BYTES, 32);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — read-only commands leave filesystem unchanged.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_identical_digests_all_zeros() {
        let snapshot = [0u8; 32];
        let v = verdict_from_filesystem_snapshot_pair(&snapshot, &snapshot);
        assert_eq!(v, Ops001Verdict::Pass);
    }

    #[test]
    fn pass_realistic_sha256_digest() {
        let digest = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        let v = verdict_from_filesystem_snapshot_pair(&digest, &digest);
        assert_eq!(v, Ops001Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — single-byte mutation (read-only contract violated).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_first_byte_differs() {
        let before = [0xab_u8; 32];
        let mut after = [0xab_u8; 32];
        after[0] = 0xac;
        let v = verdict_from_filesystem_snapshot_pair(&before, &after);
        assert_eq!(
            v,
            Ops001Verdict::Fail,
            "single-byte mutation must Fail"
        );
    }

    #[test]
    fn fail_last_byte_differs() {
        let before = [0xab_u8; 32];
        let mut after = [0xab_u8; 32];
        after[31] = 0xac;
        let v = verdict_from_filesystem_snapshot_pair(&before, &after);
        assert_eq!(v, Ops001Verdict::Fail);
    }

    #[test]
    fn fail_one_bit_differs() {
        let before = [0u8; 32];
        let mut after = [0u8; 32];
        after[0] = 0x01;
        let v = verdict_from_filesystem_snapshot_pair(&before, &after);
        assert_eq!(v, Ops001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — wrong digest length (caller error).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_before_too_short() {
        let v = verdict_from_filesystem_snapshot_pair(&[0u8; 31], &[0u8; 32]);
        assert_eq!(v, Ops001Verdict::Fail);
    }

    #[test]
    fn fail_after_too_short() {
        let v = verdict_from_filesystem_snapshot_pair(&[0u8; 32], &[0u8; 16]);
        assert_eq!(v, Ops001Verdict::Fail);
    }

    #[test]
    fn fail_both_empty() {
        let v = verdict_from_filesystem_snapshot_pair(&[], &[]);
        assert_eq!(v, Ops001Verdict::Fail);
    }

    #[test]
    fn fail_wrong_length_but_equal() {
        // 16-byte arrays that match — must still Fail (wrong hash
        // function, e.g., MD5 substituted).
        let v = verdict_from_filesystem_snapshot_pair(&[0xab_u8; 16], &[0xab_u8; 16]);
        assert_eq!(
            v,
            Ops001Verdict::Fail,
            "matching but wrong-length must Fail"
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep — every byte position differing.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_at_every_byte_position() {
        for pos in 0..32 {
            let before = [0xab_u8; 32];
            let mut after = [0xab_u8; 32];
            after[pos] ^= 0x01;
            let v = verdict_from_filesystem_snapshot_pair(&before, &after);
            assert_eq!(
                v,
                Ops001Verdict::Fail,
                "byte position {pos} flip must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry property.
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_is_symmetric() {
        let a = [0u8; 32];
        let mut b = [0u8; 32];
        b[7] = 0xff;
        let ab = verdict_from_filesystem_snapshot_pair(&a, &b);
        let ba = verdict_from_filesystem_snapshot_pair(&b, &a);
        assert_eq!(ab, ba);
        assert_eq!(ab, Ops001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — apr check / inspect read-only scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_apr_check_clean_run() {
        // `apr check model.gguf` exits without mutating filesystem.
        let snapshot = [0xa5_u8; 32];
        let v = verdict_from_filesystem_snapshot_pair(&snapshot, &snapshot);
        assert_eq!(v, Ops001Verdict::Pass);
    }

    #[test]
    fn fail_apr_check_wrote_lock_file() {
        // Regression: apr check accidentally wrote a .lock file or
        // updated mtime on the model — filesystem hash differs.
        let before = [0xa5_u8; 32];
        let mut after = [0xa5_u8; 32];
        after[15] = 0xb6; // mid-digest mutation
        let v = verdict_from_filesystem_snapshot_pair(&before, &after);
        assert_eq!(v, Ops001Verdict::Fail);
    }

    #[test]
    fn fail_apr_inspect_cached_metadata_write() {
        // Regression: `apr inspect` writes a metadata cache as a
        // side effect — filesystem mutated.
        let before = [0xa5_u8; 32];
        let after = [0xb6_u8; 32]; // completely different
        let v = verdict_from_filesystem_snapshot_pair(&before, &after);
        assert_eq!(v, Ops001Verdict::Fail);
    }
}
