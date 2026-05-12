// SHIP-TWO-001 — `apr-publish-hf-large-file-v1` algorithm-level
// PARTIAL discharge for FALSIFY-PUB-LFS-003..008 (closes 6 unbound
// gates; 001/002/009/010 already bound at runtime in `hf_hub/xet.rs`).
//
// Contract: `contracts/apr-publish-hf-large-file-v1.yaml`.
// Spec: SHIP-TWO-001 §12.8 (large-file Xet upload path).
//
// Each verdict is a pure decision rule isolated from the live HF
// upload runtime so the algorithm-level rule is testable offline.

// ===========================================================================
// LFS-003 — Chunk size bound: 8 KiB ≤ len ≤ 128 KiB except possibly last
// ===========================================================================

pub const AC_LFS_003_MIN_CHUNK_BYTES: u64 = 8 * 1024;
pub const AC_LFS_003_MAX_CHUNK_BYTES: u64 = 128 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfs003Verdict { Pass, Fail }

/// Pass iff every chunk except the last has `8192 ≤ len ≤ 131072`,
/// and the last chunk has `len ≤ 131072` (the last may be smaller).
/// Empty input fails (no chunks emitted is treated as Fail).
#[must_use]
pub fn verdict_from_chunk_sizes(chunk_lens: &[u64]) -> Lfs003Verdict {
    if chunk_lens.is_empty() { return Lfs003Verdict::Fail; }
    let last_idx = chunk_lens.len() - 1;
    for (i, &len) in chunk_lens.iter().enumerate() {
        if len == 0 { return Lfs003Verdict::Fail; }
        if len > AC_LFS_003_MAX_CHUNK_BYTES { return Lfs003Verdict::Fail; }
        if i != last_idx && len < AC_LFS_003_MIN_CHUNK_BYTES {
            return Lfs003Verdict::Fail;
        }
    }
    Lfs003Verdict::Pass
}

// ===========================================================================
// LFS-004 — Xorb size bound: serialized ≤ 64 MiB
// ===========================================================================

pub const AC_LFS_004_MAX_XORB_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfs004Verdict { Pass, Fail }

/// Pass iff every xorb has `serialized_size ≤ 64 MiB`.
#[must_use]
pub fn verdict_from_xorb_sizes(xorb_serialized_sizes: &[u64]) -> Lfs004Verdict {
    for &size in xorb_serialized_sizes {
        if size > AC_LFS_004_MAX_XORB_BYTES { return Lfs004Verdict::Fail; }
    }
    Lfs004Verdict::Pass
}

// ===========================================================================
// LFS-005 — Shard upload only fires after all referenced xorbs are 2xx
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadEventKind {
    /// 2xx response received from CAS for a xorb upload.
    XorbAck,
    /// Shard upload request initiated.
    ShardSent,
}

#[derive(Debug, Clone, Copy)]
pub struct UploadEvent {
    pub kind: UploadEventKind,
    /// Monotonic timestamp (e.g., wall-clock micros) when event was logged.
    pub at_micros: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfs005Verdict { Pass, Fail }

/// Pass iff for every `ShardSent` event there exists at least
/// `expected_xorb_count` `XorbAck` events whose `at_micros` is
/// strictly less than the shard's `at_micros`. Empty event list
/// fails (no shard sent ⇒ no upload happened).
#[must_use]
pub fn verdict_from_upload_ordering(events: &[UploadEvent], expected_xorb_count: u64) -> Lfs005Verdict {
    if events.is_empty() { return Lfs005Verdict::Fail; }
    let mut shard_seen = false;
    for (i, event) in events.iter().enumerate() {
        if matches!(event.kind, UploadEventKind::ShardSent) {
            shard_seen = true;
            let acks_before = events[..i]
                .iter()
                .filter(|e| matches!(e.kind, UploadEventKind::XorbAck))
                .filter(|e| e.at_micros < event.at_micros)
                .count() as u64;
            if acks_before < expected_xorb_count {
                return Lfs005Verdict::Fail;
            }
        }
    }
    if !shard_seen { return Lfs005Verdict::Fail; }
    Lfs005Verdict::Pass
}

// ===========================================================================
// LFS-006 — Idempotent replay: was_inserted:false ⇒ success
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasReplayResponse {
    /// First upload — server inserted the xorb; `was_inserted: true`.
    InsertedTrue,
    /// Repeat upload — server already had the xorb; `was_inserted: false`.
    InsertedFalse,
    /// Server returned a non-success status.
    HttpError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfs006Verdict { Pass, Fail }

/// Pass iff `response in {InsertedTrue, InsertedFalse}` (both treated
/// as upload success). Only `HttpError` should propagate as Fail.
#[must_use]
pub fn verdict_from_idempotent_replay(response: CasReplayResponse) -> Lfs006Verdict {
    match response {
        CasReplayResponse::InsertedTrue | CasReplayResponse::InsertedFalse => Lfs006Verdict::Pass,
        CasReplayResponse::HttpError => Lfs006Verdict::Fail,
    }
}

// ===========================================================================
// LFS-007 — Retry policy classification by HTTP status
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryDecision {
    /// Retry with exponential backoff.
    Retry,
    /// Refresh token once, then retry; abort if still failing.
    RefreshThenRetryOnce,
    /// Abort immediately — permanent failure.
    Abort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfs007Verdict { Pass, Fail }

/// Returns the canonical retry decision for a given HTTP status code
/// per the contract's retry-policy table:
///   - 429, 500, 503, 504 → Retry
///   - 401                → RefreshThenRetryOnce
///   - 400, 403, 404      → Abort
///   - all other          → Abort (conservative default)
#[must_use]
pub const fn classify_retry(status: u16) -> RetryDecision {
    match status {
        429 | 500 | 503 | 504 => RetryDecision::Retry,
        401 => RetryDecision::RefreshThenRetryOnce,
        400 | 403 | 404 => RetryDecision::Abort,
        _ => RetryDecision::Abort,
    }
}

/// Pass iff `classify_retry(status) == expected`.
#[must_use]
pub fn verdict_from_retry_classification(status: u16, expected: RetryDecision) -> Lfs007Verdict {
    if classify_retry(status) == expected { Lfs007Verdict::Pass } else { Lfs007Verdict::Fail }
}

// ===========================================================================
// LFS-008 — Hash-in-URL Xet 8-byte-reversed hex encoding
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lfs008Verdict { Pass, Fail }

/// Encode a 32-byte hash as Xet's URL-path encoding: each 8-byte
/// little-endian group is reversed before hex-encoding.
///
/// Returns `None` if `hash.len() != 32`.
#[must_use]
pub fn xet_url_hash_encode(hash: &[u8]) -> Option<String> {
    if hash.len() != 32 { return None; }
    let mut out = String::with_capacity(64);
    for group in hash.chunks_exact(8) {
        for byte in group.iter().rev() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    Some(out)
}

/// Pass iff `xet_url_hash_encode(hash)` equals `expected_hex` AND
/// `hash.len() == 32` AND `expected_hex.len() == 64`.
#[must_use]
pub fn verdict_from_url_hash_encoding(hash: &[u8], expected_hex: &str) -> Lfs008Verdict {
    if hash.len() != 32 || expected_hex.len() != 64 { return Lfs008Verdict::Fail; }
    match xet_url_hash_encode(hash) {
        Some(actual) if actual == expected_hex => Lfs008Verdict::Pass,
        _ => Lfs008Verdict::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- LFS-003 -----------------------------------------------------------

    #[test]
    fn lfs003_pass_uniform_chunks() {
        let chunks = vec![64 * 1024_u64; 10];
        assert_eq!(verdict_from_chunk_sizes(&chunks), Lfs003Verdict::Pass);
    }

    #[test]
    fn lfs003_pass_small_last_chunk() {
        let chunks = vec![64 * 1024_u64, 64 * 1024, 1024];
        assert_eq!(verdict_from_chunk_sizes(&chunks), Lfs003Verdict::Pass);
    }

    #[test]
    fn lfs003_pass_at_min_boundary() {
        let chunks = vec![AC_LFS_003_MIN_CHUNK_BYTES, AC_LFS_003_MIN_CHUNK_BYTES];
        assert_eq!(verdict_from_chunk_sizes(&chunks), Lfs003Verdict::Pass);
    }

    #[test]
    fn lfs003_pass_at_max_boundary() {
        let chunks = vec![AC_LFS_003_MAX_CHUNK_BYTES, AC_LFS_003_MAX_CHUNK_BYTES];
        assert_eq!(verdict_from_chunk_sizes(&chunks), Lfs003Verdict::Pass);
    }

    #[test]
    fn lfs003_fail_below_min_in_middle() {
        // Middle chunk is 4 KiB, below the 8 KiB minimum.
        let chunks = vec![64 * 1024_u64, 4 * 1024, 64 * 1024];
        assert_eq!(verdict_from_chunk_sizes(&chunks), Lfs003Verdict::Fail);
    }

    #[test]
    fn lfs003_fail_above_max() {
        let chunks = vec![64 * 1024_u64, 256 * 1024];
        assert_eq!(verdict_from_chunk_sizes(&chunks), Lfs003Verdict::Fail);
    }

    #[test]
    fn lfs003_fail_zero_length_chunk() {
        let chunks = vec![64 * 1024_u64, 0];
        assert_eq!(verdict_from_chunk_sizes(&chunks), Lfs003Verdict::Fail);
    }

    #[test]
    fn lfs003_fail_empty() {
        assert_eq!(verdict_from_chunk_sizes(&[]), Lfs003Verdict::Fail);
    }

    #[test]
    fn lfs003_provenance_bounds() {
        assert_eq!(AC_LFS_003_MIN_CHUNK_BYTES, 8192);
        assert_eq!(AC_LFS_003_MAX_CHUNK_BYTES, 131_072);
    }

    // ----- LFS-004 -----------------------------------------------------------

    #[test]
    fn lfs004_pass_under_limit() {
        let xorbs = vec![32 * 1024 * 1024_u64, 60 * 1024 * 1024];
        assert_eq!(verdict_from_xorb_sizes(&xorbs), Lfs004Verdict::Pass);
    }

    #[test]
    fn lfs004_pass_at_exact_limit() {
        let xorbs = vec![AC_LFS_004_MAX_XORB_BYTES];
        assert_eq!(verdict_from_xorb_sizes(&xorbs), Lfs004Verdict::Pass);
    }

    #[test]
    fn lfs004_pass_empty_set() {
        // Vacuously true: no xorbs ⇒ no violation.
        assert_eq!(verdict_from_xorb_sizes(&[]), Lfs004Verdict::Pass);
    }

    #[test]
    fn lfs004_fail_above_limit() {
        let xorbs = vec![AC_LFS_004_MAX_XORB_BYTES + 1];
        assert_eq!(verdict_from_xorb_sizes(&xorbs), Lfs004Verdict::Fail);
    }

    #[test]
    fn lfs004_fail_one_oversized_in_batch() {
        let xorbs = vec![1024_u64, 100 * 1024 * 1024, 1024];
        assert_eq!(verdict_from_xorb_sizes(&xorbs), Lfs004Verdict::Fail);
    }

    #[test]
    fn lfs004_provenance_limit() {
        assert_eq!(AC_LFS_004_MAX_XORB_BYTES, 67_108_864);
    }

    // ----- LFS-005 -----------------------------------------------------------

    fn ev(kind: UploadEventKind, at: u64) -> UploadEvent {
        UploadEvent { kind, at_micros: at }
    }

    #[test]
    fn lfs005_pass_canonical_order() {
        let events = vec![
            ev(UploadEventKind::XorbAck, 100),
            ev(UploadEventKind::XorbAck, 200),
            ev(UploadEventKind::XorbAck, 300),
            ev(UploadEventKind::ShardSent, 400),
        ];
        assert_eq!(verdict_from_upload_ordering(&events, 3), Lfs005Verdict::Pass);
    }

    #[test]
    fn lfs005_fail_shard_before_acks() {
        let events = vec![
            ev(UploadEventKind::ShardSent, 50),
            ev(UploadEventKind::XorbAck, 100),
            ev(UploadEventKind::XorbAck, 200),
        ];
        assert_eq!(verdict_from_upload_ordering(&events, 2), Lfs005Verdict::Fail);
    }

    #[test]
    fn lfs005_fail_only_one_ack_when_two_required() {
        let events = vec![
            ev(UploadEventKind::XorbAck, 100),
            ev(UploadEventKind::ShardSent, 200),
        ];
        assert_eq!(verdict_from_upload_ordering(&events, 2), Lfs005Verdict::Fail);
    }

    #[test]
    fn lfs005_fail_no_shard_sent() {
        let events = vec![
            ev(UploadEventKind::XorbAck, 100),
            ev(UploadEventKind::XorbAck, 200),
        ];
        assert_eq!(verdict_from_upload_ordering(&events, 2), Lfs005Verdict::Fail);
    }

    #[test]
    fn lfs005_fail_empty() {
        assert_eq!(verdict_from_upload_ordering(&[], 0), Lfs005Verdict::Fail);
    }

    // ----- LFS-006 -----------------------------------------------------------

    #[test]
    fn lfs006_pass_inserted_true() {
        assert_eq!(
            verdict_from_idempotent_replay(CasReplayResponse::InsertedTrue),
            Lfs006Verdict::Pass
        );
    }

    #[test]
    fn lfs006_pass_inserted_false() {
        // The key invariant: was_inserted:false on replay is success.
        assert_eq!(
            verdict_from_idempotent_replay(CasReplayResponse::InsertedFalse),
            Lfs006Verdict::Pass
        );
    }

    #[test]
    fn lfs006_fail_http_error() {
        assert_eq!(
            verdict_from_idempotent_replay(CasReplayResponse::HttpError),
            Lfs006Verdict::Fail
        );
    }

    // ----- LFS-007 -----------------------------------------------------------

    #[test]
    fn lfs007_retry_band() {
        for status in [429_u16, 500, 503, 504] {
            assert_eq!(classify_retry(status), RetryDecision::Retry, "status {status}");
            assert_eq!(
                verdict_from_retry_classification(status, RetryDecision::Retry),
                Lfs007Verdict::Pass
            );
        }
    }

    #[test]
    fn lfs007_abort_band() {
        for status in [400_u16, 403, 404] {
            assert_eq!(classify_retry(status), RetryDecision::Abort, "status {status}");
            assert_eq!(
                verdict_from_retry_classification(status, RetryDecision::Abort),
                Lfs007Verdict::Pass
            );
        }
    }

    #[test]
    fn lfs007_refresh_band() {
        assert_eq!(classify_retry(401), RetryDecision::RefreshThenRetryOnce);
        assert_eq!(
            verdict_from_retry_classification(401, RetryDecision::RefreshThenRetryOnce),
            Lfs007Verdict::Pass
        );
    }

    #[test]
    fn lfs007_unknown_status_aborts_conservatively() {
        // Unknown / novel status — treat as Abort to avoid infinite loops.
        for status in [200_u16, 418, 499, 599, 600] {
            assert_eq!(classify_retry(status), RetryDecision::Abort, "status {status}");
        }
    }

    #[test]
    fn lfs007_fail_wrong_classification() {
        // 429 must be Retry, not Abort.
        assert_eq!(
            verdict_from_retry_classification(429, RetryDecision::Abort),
            Lfs007Verdict::Fail
        );
    }

    // ----- LFS-008 -----------------------------------------------------------

    #[test]
    fn lfs008_canonical_example_from_spec() {
        // Per contract: hash = [0,1,2,...,31] → 8-byte LE-reversed
        // groups, hex-encoded. The contract's example string drops one
        // byte ("10") between groups 2/3 — a 62-char typo of the
        // 64-char output. Pin against the mathematically correct
        // value: byte 0x10 must appear in the middle, between the
        // group-2 reverse "17 16 15 14 13 12 11 10" and group-3's
        // "1f 1e 1d 1c 1b 1a 19 18".
        let hash: Vec<u8> = (0u8..32).collect();
        let encoded = xet_url_hash_encode(&hash).unwrap();
        assert_eq!(encoded.len(), 64);
        assert_eq!(
            encoded,
            "07060504030201000f0e0d0c0b0a090817161514131211101f1e1d1c1b1a1918"
        );
        assert_eq!(
            verdict_from_url_hash_encoding(
                &hash,
                "07060504030201000f0e0d0c0b0a090817161514131211101f1e1d1c1b1a1918"
            ),
            Lfs008Verdict::Pass
        );
    }

    #[test]
    fn lfs008_fail_naive_hex() {
        // The naive hex encoding (no group reversal) must be rejected.
        let hash: Vec<u8> = (0u8..32).collect();
        let naive_hex = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";
        assert_eq!(
            verdict_from_url_hash_encoding(&hash, naive_hex),
            Lfs008Verdict::Fail
        );
    }

    #[test]
    fn lfs008_fail_short_hash() {
        let hash = vec![0u8; 31];
        let encoded = xet_url_hash_encode(&hash);
        assert!(encoded.is_none());
        assert_eq!(
            verdict_from_url_hash_encoding(&hash, &"0".repeat(64)),
            Lfs008Verdict::Fail
        );
    }

    #[test]
    fn lfs008_fail_long_hash() {
        let hash = vec![0u8; 33];
        assert!(xet_url_hash_encode(&hash).is_none());
    }

    #[test]
    fn lfs008_fail_short_expected_hex() {
        let hash: Vec<u8> = (0u8..32).collect();
        assert_eq!(
            verdict_from_url_hash_encoding(&hash, "0102"),
            Lfs008Verdict::Fail
        );
    }

    #[test]
    fn lfs008_zero_hash_is_well_formed() {
        let hash = vec![0u8; 32];
        let encoded = xet_url_hash_encode(&hash).unwrap();
        assert_eq!(encoded, "0".repeat(64));
        assert_eq!(
            verdict_from_url_hash_encoding(&hash, &"0".repeat(64)),
            Lfs008Verdict::Pass
        );
    }
}
