//! Content manifest: a flat tree of digests reduced to one root hash.
//!
//! APEX-001 EV-2a rule 4. Moved from `rmedia-types::lock` — rmedia is never published to
//! crates.io (F-07), so the code is copied rather than imported, and the tests come with it.
//! The `#[provable_contracts_macros::contract(...)]` attributes the originals carry are bound to
//! rmedia's own contract YAML and are **not** reproduced here; `verify` returns this crate's
//! [`Error`] instead of `RmediaError`.
//!
//! Flat, not a Merkle DAG: a recursive tree buys partial re-verification and subtree dedup,
//! neither of which pays for its indexing complexity at this scale. Verifying is O(files) —
//! recompute each descriptor, re-sort, re-hash, compare one root.

use crate::error::{Error, Result};
use sha2::{Digest, Sha256};

/// SHA-256 of `bytes`, lowercase hex.
///
/// **Over raw bytes, never over a canonicalisation.** See
/// `falsify_lock_001_digest_is_over_raw_bytes` for why that is load-bearing rather than
/// incidental.
///
/// ```
/// use trueno_viz::manifest::digest_bytes;
/// assert_eq!(
///     digest_bytes(b"abc"),
///     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
/// );
/// ```
#[must_use]
pub fn digest_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// One entry of a [`Manifest`]: the OCI descriptor triple, minus the media type.
///
/// OCI mandates verifying retrieved content against **both** digest and size, so `size` is
/// carried and covered by the root even though it is redundant with the digest — a length
/// mismatch lets a verifier distrust content without paying for a full hash.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Descriptor {
    /// Path relative to the locked tree's root. Unique within a manifest.
    pub path: String,
    /// Byte length of the content.
    pub size: u64,
    /// [`digest_bytes`] of the content.
    pub digest: String,
}

/// A flat manifest over a tree, reduced to a single root hash.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Manifest {
    entries: Vec<Descriptor>,
}

impl Manifest {
    /// An empty manifest.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `bytes` under `path`, computing size and digest.
    ///
    /// Re-inserting a path replaces the previous descriptor, so a manifest never carries two
    /// entries for one path.
    pub fn insert(&mut self, path: impl Into<String>, bytes: &[u8]) {
        self.insert_descriptor(Descriptor {
            path: path.into(),
            size: bytes.len() as u64,
            digest: digest_bytes(bytes),
        });
    }

    /// Record a descriptor computed elsewhere (e.g. read back from a lock).
    pub fn insert_descriptor(&mut self, d: Descriptor) {
        match self.entries.iter().position(|e| e.path == d.path) {
            Some(i) => self.entries[i] = d,
            None => self.entries.push(d),
        }
    }

    /// The descriptors, sorted by path.
    #[must_use]
    pub fn entries(&self) -> Vec<Descriptor> {
        let mut v = self.entries.clone();
        v.sort();
        v
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the manifest has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The canonical manifest text: entries sorted by path, one `{digest}  {size}  {path}` line
    /// each, LF-terminated.
    ///
    /// Sorting here is what makes [`Manifest::root`] independent of insertion order.
    #[must_use]
    pub fn canonical(&self) -> String {
        let mut out = String::new();
        for e in self.entries() {
            out.push_str(&e.digest);
            out.push_str("  ");
            out.push_str(&e.size.to_string());
            out.push_str("  ");
            out.push_str(&e.path);
            out.push('\n');
        }
        out
    }

    /// The single root hash over the whole tree.
    ///
    /// [`digest_bytes`] over the canonical text — raw bytes again, so the root inherits every
    /// property of the digest.
    #[must_use]
    pub fn root(&self) -> String {
        digest_bytes(self.canonical().as_bytes())
    }

    /// Verify this manifest against the root recorded when the tree was locked.
    ///
    /// **Refusal, not repair.** On mismatch this returns `Err` and stops. It never rewrites the
    /// manifest, never re-resolves an entry, never downgrades to a warning. The `&self` receiver
    /// and the absence of any I/O make that structural rather than merely intended.
    ///
    /// A matching root proves *integrity* — the bytes are unchanged since locking. It proves
    /// nothing about *authority*.
    ///
    /// # Errors
    ///
    /// [`Error::ManifestMismatch`] when the computed root differs from `expected_root`.
    pub fn verify(&self, expected_root: &str) -> Result<()> {
        let actual = self.root();
        if actual == expected_root {
            return Ok(());
        }
        Err(Error::ManifestMismatch {
            expected: expected_root.to_string(),
            actual,
            entries: self.entries.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FALSIFY-LOCK-001 ────────────────────────────────────────────────
    //
    // The digest is over RAW BYTES, never over a canonicalization.
    //
    // Mutation that must turn this RED: canonicalize JSON-shaped preimages inside
    // `digest_bytes` before hashing (drop all but the last occurrence of each duplicate key,
    // sort by key, re-emit). That is exactly the `hash_mode: canonical` design rmedia measured
    // collapsing the two documents below onto one digest.
    #[test]
    fn falsify_lock_001_digest_is_over_raw_bytes() {
        // The collision witness. Under RFC 8785 these share a digest.
        let with_dup = br#"{"a":1,"a":2,"b":2}"#;
        let without_dup = br#"{"a":2,"b":2}"#;
        assert_ne!(
            digest_bytes(with_dup),
            digest_bytes(without_dup),
            "digest_bytes canonicalised its input: two distinct byte strings hashed the same"
        );
    }

    #[test]
    fn digest_matches_the_published_sha256_of_abc() {
        assert_eq!(
            digest_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn the_root_is_independent_of_insertion_order() {
        let mut a = Manifest::new();
        a.insert("z.svg", b"zzz");
        a.insert("a.svg", b"aaa");

        let mut b = Manifest::new();
        b.insert("a.svg", b"aaa");
        b.insert("z.svg", b"zzz");

        assert_eq!(a.root(), b.root());
        assert_eq!(a.canonical(), b.canonical());
    }

    #[test]
    fn reinserting_a_path_replaces_rather_than_duplicates() {
        let mut m = Manifest::new();
        m.insert("f.png", b"one");
        m.insert("f.png", b"two");
        assert_eq!(m.len(), 1);
        assert_eq!(m.entries()[0].digest, digest_bytes(b"two"));
    }

    #[test]
    fn verify_refuses_a_mismatched_root_and_does_not_repair() {
        let mut m = Manifest::new();
        m.insert("f.png", b"one");
        let before = m.root();

        let err = m.verify("not the root").unwrap_err();
        assert!(matches!(err, Error::ManifestMismatch { .. }), "got {err:?}");
        // Refusal, not repair: the manifest is unchanged by a failed verification.
        assert_eq!(m.root(), before);
        assert!(m.verify(&before).is_ok());
    }

    #[test]
    fn canonical_is_one_lf_terminated_line_per_entry_sorted_by_path() {
        let mut m = Manifest::new();
        m.insert("b", b"2");
        m.insert("a", b"1");
        let canonical = m.canonical();
        let lines: Vec<&str> = canonical.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].ends_with("  1  a"));
        assert!(lines[1].ends_with("  1  b"));
        assert!(m.canonical().ends_with('\n'));
    }

    #[test]
    fn an_empty_manifest_is_empty_and_still_has_a_root() {
        let m = Manifest::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
        assert_eq!(m.canonical(), "");
        assert_eq!(m.root(), digest_bytes(b""));
    }
}
