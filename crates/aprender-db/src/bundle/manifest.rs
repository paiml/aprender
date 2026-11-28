//! Canonical sha256 manifest over a bundle's named parts.
//!
//! Text encoding (UTF-8, LF line endings, trailing newline):
//!
//! ```text
//! aprender-bundle-manifest v1
//! <name>\t<eager|lazy>\t<len>\t<sha256, 64 lowercase hex>    one line per part, sorted by name
//! digest\t<sha256, 64 lowercase hex, of every byte above>
//! ```
//!
//! The encoding depends only on the part names, kinds and bytes, never on the
//! order the parts were supplied in, so two builds of the same parts are the
//! same bytes.

use std::fmt::Write;

use sha2::{Digest, Sha256};

use super::BundleError;

/// First line of every encoded manifest.
pub const HEADER: &str = "aprender-bundle-manifest v1";

/// Manifest format version recorded in [`Manifest::version`].
pub const VERSION: u32 = 1;

const DIGEST_KEY: &str = "digest";

/// Whether a part is needed for first paint or fetched on first use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartKind {
    /// Needed for first paint; never fetched lazily.
    Eager,
    /// Fetched on first use through [`super::LazyTables`].
    Lazy,
}

impl PartKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Lazy => "lazy",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "eager" => Some(Self::Eager),
            "lazy" => Some(Self::Lazy),
            _ => None,
        }
    }
}

/// One part as recorded in a manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartEntry {
    /// Unique, non-empty part name.
    pub name: String,
    /// Eager or lazy.
    pub kind: PartKind,
    /// Byte length of the part.
    pub len: u64,
    /// sha256 of the part bytes.
    pub sha256: [u8; 32],
}

/// The manifest: every part, sorted by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Format version; [`VERSION`] for every manifest this module produces.
    pub version: u32,
    /// Parts sorted by name, names unique.
    pub parts: Vec<PartEntry>,
}

impl Manifest {
    /// Build a manifest by hashing each `(name, kind, bytes)` part.
    ///
    /// The result does not depend on the iteration order of `parts`.
    ///
    /// # Errors
    ///
    /// Returns [`BundleError::EmptyName`], [`BundleError::InvalidName`] for a
    /// name with a control character, or [`BundleError::DuplicatePart`] when a
    /// name is repeated.
    pub fn from_parts<'a, I>(parts: I) -> Result<Self, BundleError>
    where
        I: IntoIterator<Item = (&'a str, PartKind, &'a [u8])>,
    {
        let mut entries = Vec::new();
        for (name, kind, bytes) in parts {
            check_name(name)?;
            entries.push(PartEntry {
                name: name.to_owned(),
                kind,
                len: bytes.len() as u64,
                sha256: sha256(bytes),
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        ensure_strictly_sorted(&entries)?;
        Ok(Self { version: VERSION, parts: entries })
    }

    /// The entry for `name`, if the manifest has one.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&PartEntry> {
        self.parts
            .binary_search_by(|p| p.name.as_str().cmp(name))
            .ok()
            .and_then(|i| self.parts.get(i))
    }

    /// Check `bytes` against the entry for `name`: length, then sha256.
    ///
    /// # Errors
    ///
    /// [`BundleError::UnknownPart`], [`BundleError::LengthMismatch`] or
    /// [`BundleError::HashMismatch`], each naming the part.
    pub fn verify_part(&self, name: &str, bytes: &[u8]) -> Result<(), BundleError> {
        let entry =
            self.entry(name).ok_or_else(|| BundleError::UnknownPart { part: name.to_owned() })?;
        let actual = bytes.len() as u64;
        if actual != entry.len {
            return Err(BundleError::LengthMismatch {
                part: name.to_owned(),
                expected: entry.len,
                actual,
            });
        }
        if sha256(bytes) != entry.sha256 {
            return Err(BundleError::HashMismatch { part: name.to_owned() });
        }
        Ok(())
    }

    /// Encode to the canonical text form. Same parts, same bytes.
    #[must_use]
    pub fn encode(&self) -> String {
        let mut body = format!("{HEADER}\n");
        for p in &self.parts {
            writeln!(body, "{}\t{}\t{}\t{}", p.name, p.kind.as_str(), p.len, to_hex(&p.sha256))
                .expect("writing to a String cannot fail");
        }
        let digest = sha256(body.as_bytes());
        writeln!(body, "{DIGEST_KEY}\t{}", to_hex(&digest))
            .expect("writing to a String cannot fail");
        body
    }

    /// Decode the canonical text form.
    ///
    /// The self-digest is checked before any entry line is parsed, so a flipped
    /// byte anywhere in the manifest is refused as
    /// [`BundleError::ManifestDigest`].
    ///
    /// # Errors
    ///
    /// Any [`BundleError`] describing the first defect found.
    pub fn decode(text: &str) -> Result<Self, BundleError> {
        let (head, digest_line) = split_digest_line(text)?;
        let recorded = parse_digest_line(digest_line, head.lines().count() + 1)?;
        let computed = sha256(head.as_bytes());
        if computed != recorded {
            return Err(BundleError::ManifestDigest {
                expected: to_hex(&recorded),
                actual: to_hex(&computed),
            });
        }
        let mut lines = head.lines();
        let header = lines.next().unwrap_or_default();
        if header != HEADER {
            return Err(BundleError::BadHeader { found: header.to_owned() });
        }
        let mut parts = Vec::new();
        for (i, line) in lines.enumerate() {
            parts.push(parse_entry(line, i + 2)?);
        }
        ensure_strictly_sorted(&parts)?;
        Ok(Self { version: VERSION, parts })
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn check_name(name: &str) -> Result<(), BundleError> {
    if name.is_empty() {
        return Err(BundleError::EmptyName);
    }
    if name.chars().any(char::is_control) {
        return Err(BundleError::InvalidName { name: name.to_owned() });
    }
    Ok(())
}

fn ensure_strictly_sorted(parts: &[PartEntry]) -> Result<(), BundleError> {
    for pair in parts.windows(2) {
        let (before, after) = (&pair[0].name, &pair[1].name);
        if before == after {
            return Err(BundleError::DuplicatePart { part: after.clone() });
        }
        if before > after {
            return Err(BundleError::UnsortedPart { part: after.clone() });
        }
    }
    Ok(())
}

/// Split the text into (every byte above the digest line, the digest line).
fn split_digest_line(text: &str) -> Result<(&str, &str), BundleError> {
    let body = text
        .strip_suffix('\n')
        .ok_or(BundleError::Malformed { reason: "missing trailing newline" })?;
    let cut = body.rfind('\n').map_or(0, |i| i + 1);
    Ok((&text[..cut], &body[cut..]))
}

fn parse_digest_line(line: &str, line_no: usize) -> Result<[u8; 32], BundleError> {
    let hex = line
        .strip_prefix(DIGEST_KEY)
        .and_then(|rest| rest.strip_prefix('\t'))
        .ok_or(BundleError::Malformed { reason: "missing digest line" })?;
    from_hex(hex).ok_or(BundleError::BadHex { line: line_no })
}

fn parse_entry(line: &str, line_no: usize) -> Result<PartEntry, BundleError> {
    let fields: Vec<&str> = line.split('\t').collect();
    let &[name, kind, len, hex] = fields.as_slice() else {
        return Err(BundleError::BadLine {
            line: line_no,
            reason: "expected 4 tab-separated fields",
        });
    };
    check_name(name)?;
    let kind = PartKind::parse(kind)
        .ok_or(BundleError::BadLine { line: line_no, reason: "kind is not eager or lazy" })?;
    if len.is_empty() || !len.bytes().all(|b| b.is_ascii_digit()) {
        return Err(BundleError::BadLine {
            line: line_no,
            reason: "length is not an unsigned integer",
        });
    }
    let len: u64 = len
        .parse()
        .map_err(|_| BundleError::BadLine { line: line_no, reason: "length is out of range" })?;
    let sha256 = from_hex(hex).ok_or(BundleError::BadHex { line: line_no })?;
    Ok(PartEntry { name: name.to_owned(), kind, len, sha256 })
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
        write!(out, "{b:02x}").expect("writing to a String cannot fail");
        out
    })
}

fn from_hex(text: &str) -> Option<[u8; 32]> {
    let digits = text.as_bytes();
    let well_formed = digits.len() == 64
        && digits.iter().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c));
    if !well_formed {
        return None;
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(digits.chunks_exact(2)) {
        *slot = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Some(out)
}

/// Value of one lowercase hex digit; the caller has already validated it.
fn nibble(c: u8) -> u8 {
    if c.is_ascii_digit() {
        c - b'0'
    } else {
        c - b'a' + 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn fixture() -> Vec<(&'static str, PartKind, &'static [u8])> {
        vec![
            ("zones", PartKind::Lazy, &b"zone-rows"[..]),
            ("meta", PartKind::Eager, &b"eager-bytes"[..]),
            ("cells", PartKind::Lazy, &b"cell-rows"[..]),
        ]
    }

    fn manifest() -> Manifest {
        Manifest::from_parts(fixture()).expect("fixture names are unique")
    }

    /// Append a digest line over `body`, so structural errors can be reached
    /// past the self-digest check.
    fn with_digest(body: &str) -> String {
        format!("{body}{DIGEST_KEY}\t{}\n", to_hex(&sha256(body.as_bytes())))
    }

    /// Flip the final hex digit of the first line starting with `prefix`.
    fn flip_last_char_of_line(text: &str, prefix: &str) -> String {
        let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
        let line = lines.iter_mut().find(|l| l.starts_with(prefix)).expect("prefix line present");
        let last = line.pop().expect("line non-empty");
        line.push(if last == '0' { '1' } else { '0' });
        lines.iter().fold(String::new(), |mut out, l| {
            out.push_str(l);
            out.push('\n');
            out
        })
    }

    #[test]
    fn encoding_is_independent_of_insertion_order() {
        let forward = manifest();
        let mut reversed_parts = fixture();
        reversed_parts.reverse();
        let reversed = Manifest::from_parts(reversed_parts).expect("unique names");
        assert_eq!(forward.encode(), reversed.encode());
        assert_eq!(forward.parts[0].name, "cells");
        assert_eq!(forward.version, VERSION);
    }

    #[test]
    fn decode_round_trips_encode() {
        let m = manifest();
        let text = m.encode();
        assert!(text.starts_with(HEADER));
        assert!(text.ends_with('\n'));
        assert_eq!(Manifest::decode(&text).expect("decodes"), m);
    }

    #[test]
    fn flipped_manifest_hash_byte_is_refused_as_digest() {
        let text = manifest().encode();
        let tampered = flip_last_char_of_line(&text, "meta\t");
        let err = Manifest::decode(&tampered).expect_err("tampered manifest");
        assert!(matches!(err, BundleError::ManifestDigest { .. }), "{err}");
        assert!(err.to_string().contains("digest"));
    }

    #[test]
    fn flipped_data_byte_names_the_part() {
        let m = manifest();
        let mut bytes = b"eager-bytes".to_vec();
        bytes[0] ^= 0x01;
        let err = m.verify_part("meta", &bytes).expect_err("flipped byte");
        assert_eq!(err, BundleError::HashMismatch { part: "meta".to_owned() });
        assert!(err.to_string().contains("`meta`"));
    }

    #[test]
    fn verify_part_accepts_the_original_bytes() {
        let m = manifest();
        assert!(m.verify_part("zones", b"zone-rows").is_ok());
        assert!(m.entry("meta").is_some());
        assert!(m.entry("absent").is_none());
    }

    #[test]
    fn verify_part_reports_length_and_unknown_name() {
        let m = manifest();
        assert_eq!(
            m.verify_part("zones", b"short"),
            Err(BundleError::LengthMismatch { part: "zones".to_owned(), expected: 9, actual: 5 })
        );
        assert_eq!(
            m.verify_part("absent", b""),
            Err(BundleError::UnknownPart { part: "absent".to_owned() })
        );
    }

    #[test]
    fn from_parts_refuses_empty_control_and_duplicate_names() {
        let empty = Manifest::from_parts([("", PartKind::Eager, &b"x"[..])]);
        assert_eq!(empty, Err(BundleError::EmptyName));
        let control = Manifest::from_parts([("a\tb", PartKind::Eager, &b"x"[..])]);
        assert!(matches!(control, Err(BundleError::InvalidName { .. })));
        let dup = Manifest::from_parts([
            ("a", PartKind::Eager, &b"x"[..]),
            ("a", PartKind::Lazy, &b"y"[..]),
        ]);
        assert_eq!(dup, Err(BundleError::DuplicatePart { part: "a".to_owned() }));
    }

    #[test]
    fn decode_refuses_missing_newline_and_digest() {
        let text = manifest().encode();
        let no_newline = text.strip_suffix('\n').expect("ends with newline");
        assert!(matches!(Manifest::decode(no_newline), Err(BundleError::Malformed { .. })));
        assert_eq!(
            Manifest::decode("garbage\n"),
            Err(BundleError::Malformed { reason: "missing digest line" })
        );
    }

    #[test]
    fn decode_refuses_bad_digest_hex() {
        assert_eq!(Manifest::decode("digest\tzz\n"), Err(BundleError::BadHex { line: 1 }));
    }

    #[test]
    fn decode_refuses_bad_header() {
        let err = Manifest::decode(&with_digest("aprender-bundle-manifest v2\n"))
            .expect_err("wrong version");
        assert_eq!(err, BundleError::BadHeader { found: "aprender-bundle-manifest v2".to_owned() });
        assert!(Manifest::decode(&with_digest("")).is_err());
    }

    #[test]
    fn decode_refuses_malformed_entry_lines() {
        let hex = to_hex(&sha256(b"x"));
        let cases = [
            format!("{HEADER}\na\teager\t1\n"),
            format!("{HEADER}\na\tweird\t1\t{hex}\n"),
            format!("{HEADER}\na\teager\t+1\t{hex}\n"),
            format!("{HEADER}\na\teager\t\t{hex}\n"),
            format!("{HEADER}\na\teager\t99999999999999999999999\t{hex}\n"),
            format!("{HEADER}\n\teager\t1\t{hex}\n"),
            format!("{HEADER}\na\teager\t1\t{}\n", hex.to_uppercase()),
        ];
        for body in cases {
            let err = Manifest::decode(&with_digest(&body)).expect_err("malformed");
            assert!(
                matches!(
                    err,
                    BundleError::BadLine { .. }
                        | BundleError::BadHex { .. }
                        | BundleError::EmptyName
                ),
                "{err}"
            );
        }
    }

    #[test]
    fn decode_refuses_unsorted_and_duplicate_names() {
        let hex = to_hex(&sha256(b"x"));
        let unsorted = format!("{HEADER}\nb\teager\t1\t{hex}\na\teager\t1\t{hex}\n");
        assert_eq!(
            Manifest::decode(&with_digest(&unsorted)),
            Err(BundleError::UnsortedPart { part: "a".to_owned() })
        );
        let dup = format!("{HEADER}\na\teager\t1\t{hex}\na\tlazy\t1\t{hex}\n");
        assert_eq!(
            Manifest::decode(&with_digest(&dup)),
            Err(BundleError::DuplicatePart { part: "a".to_owned() })
        );
    }

    #[test]
    fn every_error_displays_a_message() {
        let errors = [
            BundleError::EmptyName,
            BundleError::InvalidName { name: "a\n".into() },
            BundleError::DuplicatePart { part: "p".into() },
            BundleError::UnsortedPart { part: "p".into() },
            BundleError::UnknownPart { part: "p".into() },
            BundleError::EagerPart { part: "p".into() },
            BundleError::LengthMismatch { part: "p".into(), expected: 1, actual: 2 },
            BundleError::HashMismatch { part: "p".into() },
            BundleError::ManifestDigest { expected: "a".into(), actual: "b".into() },
            BundleError::BadHeader { found: "h".into() },
            BundleError::BadLine { line: 3, reason: "why" },
            BundleError::BadHex { line: 3 },
            BundleError::Malformed { reason: "why" },
        ];
        for err in &errors {
            assert!(!err.to_string().is_empty());
            let _: &dyn std::error::Error = err;
        }
    }

    proptest! {
        #[test]
        fn encode_decode_round_trip_and_order_free(
            parts in proptest::collection::btree_map(
                "[a-z][a-z0-9_]{0,11}",
                (any::<bool>(), proptest::collection::vec(any::<u8>(), 0..64)),
                0..8,
            )
        ) {
            let input: Vec<(&str, PartKind, &[u8])> = parts
                .iter()
                .map(|(name, (lazy, bytes))| {
                    let kind = if *lazy { PartKind::Lazy } else { PartKind::Eager };
                    (name.as_str(), kind, bytes.as_slice())
                })
                .collect();
            let m = Manifest::from_parts(input.clone()).expect("unique names");
            let text = m.encode();
            prop_assert_eq!(Manifest::decode(&text).expect("decodes"), m);
            let mut reversed = input;
            reversed.reverse();
            prop_assert_eq!(Manifest::from_parts(reversed).expect("unique").encode(), text);
        }
    }
}
