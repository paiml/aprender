//! ULID run ids (EXT-04, aprender#4386).
//!
//! EXT-001 §3.1: run ids are ULIDs, and host-local autoincrement ids never
//! leave the host. `run-{n}` from a per-process counter collides as soon as
//! two trainers write to one pacha registry.
//!
//! A ULID is 48 bits of milliseconds since the Unix epoch followed by 80
//! random bits, written as 26 Crockford base32 characters. Ids from later
//! milliseconds sort after earlier ones as plain strings.

use std::time::{SystemTime, UNIX_EPOCH};

const CROCKFORD: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";

/// Length of a ULID string.
pub const ULID_LEN: usize = 26;

/// A new ULID for the current time.
#[must_use]
pub fn new_ulid() -> String {
    let ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
    ulid_from_parts(ms as u64, rand::random::<u128>())
}

/// The ULID for `ms` since the epoch and the low 80 bits of `random`.
pub(crate) fn ulid_from_parts(ms: u64, random: u128) -> String {
    let value = (u128::from(ms & 0xFFFF_FFFF_FFFF) << 80) | (random & ((1u128 << 80) - 1));
    // 26 × 5 = 130 bits; the top two are always zero.
    (0..ULID_LEN)
        .map(|i| {
            let shift = 5 * (ULID_LEN - 1 - i);
            char::from(CROCKFORD[((value >> shift) & 0x1F) as usize])
        })
        .collect()
}

/// Whether `id` is a canonical (upper-case) ULID.
#[must_use]
pub fn is_ulid(id: &str) -> bool {
    id.len() == ULID_LEN
        && id.bytes().all(|b| CROCKFORD.contains(&b))
        // The first character carries only 3 bits: `7ZZZ…` is the maximum.
        && id.as_bytes()[0] <= b'7'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulid_matches_the_spec_vector() {
        // ulid-js encodeTime test vector: 1469918176385 encodes as 01ARYZ6S41.
        assert!(ulid_from_parts(1_469_918_176_385, 0).starts_with("01ARYZ6S41"));
        assert_eq!(ulid_from_parts(0, 0), "0".repeat(ULID_LEN));
        assert_eq!(ulid_from_parts(u64::MAX, u128::MAX), format!("7{}", "Z".repeat(25)));
    }

    #[test]
    fn ulids_are_valid_distinct_and_time_ordered() {
        let a = ulid_from_parts(1_000, u128::MAX);
        let b = ulid_from_parts(1_001, 0);
        assert!(a < b, "a later millisecond must sort after: {a} {b}");
        let fresh: std::collections::BTreeSet<String> = (0..1000).map(|_| new_ulid()).collect();
        assert_eq!(fresh.len(), 1000);
        assert!(fresh.iter().all(|u| is_ulid(u)));
    }

    #[test]
    fn is_ulid_rejects_host_local_and_malformed_ids() {
        for bad in
            ["run-1", "", "01ARYZ6S41", "8ZZZZZZZZZZZZZZZZZZZZZZZZZ", "01aryz6s41tsv4rrffq69g5fav"]
        {
            assert!(!is_ulid(bad), "{bad}");
        }
        assert!(is_ulid("01ARYZ6S41TSV4RRFFQ69G5FAV"));
        assert!(!is_ulid("01ARYZ6S41TSV4RRFFQ69G5FAI"), "I is not Crockford");
    }
}
