//! Labeled-example schema and strict JSONL parse/encode over `&[u8]`.
//!
//! Deserialization is `deny_unknown_fields` at the bytes -> typed boundary: bytes that
//! arrive from object storage are untrusted, and a silently ignored extra field is how a
//! schema drift becomes a silent data change.
//!
//! # Field names and order are load-bearing
//!
//! [`LabeledExample`]'s five fields are `id, input, label, label_text, source_split`, in
//! that order, matching the field order of the D-06 baseline row struct this type absorbed
//! (`StanceSample`, formerly in `crates/apr-cli/src/commands/data_tweeteval.rs`, removed in
//! plan 02-06 once the adapter read through this type instead). `serde_json` emits struct
//! fields in declaration order, so keeping the order means relocated JSONL output stays
//! **byte-compatible** with datasets already produced on developer machines. Reordering
//! the fields is a manifest `schema_version` bump, never a silent edit.
//!
//! # Why the byte round-trip matters
//!
//! [`encode_jsonl`] is the canonical re-encoding of accepted rows, and
//! `encode_jsonl(parse_jsonl_bytes(b)?)? == b` holds for canonical input. The split
//! constructors rely on that: one derives its `source_hash` from the ingested buffer and
//! the other from the re-encoding of typed rows, and the two must agree or a dataset
//! fingerprint would depend on which door the caller came through.

use serde::{Deserialize, Serialize};

use crate::error::ContrastiveDataError;

/// One labeled example, the atom of every split in this protocol.
///
/// `deny_unknown_fields` is the untrusted-input control: a mirror that grew an extra
/// column is a schema change, and a schema change that deserializes silently is a data
/// change nobody reviewed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LabeledExample {
    /// Stable row identifier, conventionally `{split}:{index}`.
    pub id: String,
    /// The raw text. Never trimmed or normalized in place — normalization is a hash-time
    /// concern (`hash::normalized_hash`), because the stored bytes are the provenance.
    pub input: String,
    /// Zero-based class index into the declared label map.
    pub label: usize,
    /// Human-readable label. Must equal `label_names[label]`; a disagreement is what a
    /// hand-edited mirror looks like and is a typed error at ingest.
    pub label_text: String,
    /// The split role these bytes claim to belong to. Claimed, not trusted.
    pub source_split: String,
}

/// Parse a JSONL buffer into typed rows, preserving input order.
///
/// `split` names the role for error messages only; it is not compared against the rows'
/// embedded `source_split` here. That comparison is the split constructor's Gate 2,
/// because only the constructor knows which role is being built.
///
/// # Errors
///
/// [`ContrastiveDataError::InvalidUtf8`], [`ContrastiveDataError::MalformedRow`], or
/// [`ContrastiveDataError::EmptyInput`], each naming the split and the zero-based row
/// index.
pub fn parse_jsonl_bytes(
    bytes: &[u8],
    split: &str,
) -> Result<Vec<LabeledExample>, ContrastiveDataError> {
    let mut rows = Vec::new();
    for (index, line) in jsonl_lines(bytes).enumerate() {
        // Gate 1a: the row must be UTF-8 before it can be anything else.
        let text = core::str::from_utf8(line).map_err(|_| ContrastiveDataError::InvalidUtf8 {
            split: split.to_string(),
            index,
        })?;
        // Gate 1b: strict schema. An unknown field is a schema change, not a nuisance.
        let row: LabeledExample =
            serde_json::from_str(text).map_err(|error| ContrastiveDataError::MalformedRow {
                split: split.to_string(),
                index,
                reason: error.to_string(),
            })?;
        // Gate 1c: a whitespace-only example carries no signal and would hash to the
        // normalization of the empty string, colliding with every other blank row.
        if row.input.trim().is_empty() {
            return Err(ContrastiveDataError::EmptyInput {
                split: split.to_string(),
                index,
            });
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Split a JSONL buffer into row slices.
///
/// A single trailing `\n` terminates the last row rather than introducing an empty one —
/// `encode_jsonl` always emits it, so treating it as a row would make the round-trip
/// asymmetric. Any OTHER blank line is left in place and fails as a malformed row, because
/// silently skipping blanks would let a truncated upload look like a shorter dataset.
fn jsonl_lines(bytes: &[u8]) -> impl Iterator<Item = &[u8]> {
    let trimmed = match bytes.split_last() {
        Some((b'\n', head)) => head,
        _ => bytes,
    };
    let empty = trimmed.is_empty();
    trimmed.split(|byte| *byte == b'\n').filter(move |_| !empty)
}

/// Canonically encode rows as JSONL: `serde_json::to_writer` per row, then `b'\n'`.
///
/// Written straight into one buffer rather than joining per-row `String`s, so there is no
/// intermediate allocation whose formatting could drift from the parser's expectations.
///
/// # Errors
///
/// [`ContrastiveDataError::Serialization`] if a row cannot be serialized.
pub fn encode_jsonl(rows: &[LabeledExample]) -> Result<Vec<u8>, ContrastiveDataError> {
    let mut bytes = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut bytes, row).map_err(|error| {
            ContrastiveDataError::Serialization {
                context: format!("encode_jsonl row {:?}", row.id),
                detail: error.to_string(),
            }
        })?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}

#[cfg(test)]
mod schema_tests {
    use super::{encode_jsonl, parse_jsonl_bytes, LabeledExample};
    use crate::error::ContrastiveDataError;

    fn row(id: &str, input: &str, label: usize, label_text: &str, split: &str) -> LabeledExample {
        LabeledExample {
            id: id.to_string(),
            input: input.to_string(),
            label,
            label_text: label_text.to_string(),
            source_split: split.to_string(),
        }
    }

    fn canonical_two_rows() -> Vec<LabeledExample> {
        vec![
            row("train:0", "first post", 0, "none", "train"),
            row("train:1", "second post", 1, "against", "train"),
        ]
    }

    #[test]
    fn schema_parse_valid_rows_preserves_input_order() {
        let bytes = encode_jsonl(&canonical_two_rows()).expect("encode must succeed");
        let parsed = parse_jsonl_bytes(&bytes, "train").expect("parse must succeed");
        assert_eq!(parsed, canonical_two_rows());
    }

    #[test]
    fn schema_parse_rejects_unknown_field() {
        let bytes = concat!(
            "{\"id\":\"train:0\",\"input\":\"a\",\"label\":0,\"label_text\":\"none\",\"source_split\":\"train\"}\n",
            "{\"id\":\"train:1\",\"input\":\"b\",\"label\":0,\"label_text\":\"none\",\"source_split\":\"train\",\"extra\":1}\n",
        );
        let err =
            parse_jsonl_bytes(bytes.as_bytes(), "train").expect_err("unknown field must fail");
        match err {
            ContrastiveDataError::MalformedRow {
                split,
                index,
                reason,
            } => {
                assert_eq!(split, "train");
                assert_eq!(index, 1);
                assert!(
                    reason.contains("extra"),
                    "reason must name the field: {reason}"
                );
            }
            other => panic!("expected MalformedRow, got {other:?}"),
        }
    }

    #[test]
    fn schema_parse_rejects_missing_field() {
        let bytes = "{\"id\":\"train:0\",\"input\":\"a\",\"label\":0,\"label_text\":\"none\"}\n";
        let err =
            parse_jsonl_bytes(bytes.as_bytes(), "train").expect_err("missing field must fail");
        assert!(matches!(
            err,
            ContrastiveDataError::MalformedRow { index: 0, .. }
        ));
    }

    #[test]
    fn schema_parse_rejects_non_utf8_bytes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(
            b"{\"id\":\"train:0\",\"input\":\"a\",\"label\":0,\"label_text\":\"none\",\"source_split\":\"train\"}\n",
        );
        bytes.extend_from_slice(&[0xff, 0xfe, b'\n']);
        let err = parse_jsonl_bytes(&bytes, "train").expect_err("non-UTF-8 must fail");
        match err {
            ContrastiveDataError::InvalidUtf8 { split, index } => {
                assert_eq!(split, "train");
                assert_eq!(index, 1);
            }
            other => panic!("expected InvalidUtf8, got {other:?}"),
        }
    }

    #[test]
    fn schema_parse_rejects_whitespace_only_input() {
        let bytes = concat!(
            "{\"id\":\"train:0\",\"input\":\"a\",\"label\":0,\"label_text\":\"none\",\"source_split\":\"train\"}\n",
            "{\"id\":\"train:1\",\"input\":\"   \\t \",\"label\":0,\"label_text\":\"none\",\"source_split\":\"train\"}\n",
        );
        let err =
            parse_jsonl_bytes(bytes.as_bytes(), "validation").expect_err("empty input must fail");
        match err {
            ContrastiveDataError::EmptyInput { split, index } => {
                assert_eq!(split, "validation");
                assert_eq!(index, 1);
            }
            other => panic!("expected EmptyInput, got {other:?}"),
        }
    }

    #[test]
    fn schema_encode_parse_round_trip_is_byte_exact() {
        let rows = canonical_two_rows();
        let bytes = encode_jsonl(&rows).expect("encode must succeed");
        let parsed = parse_jsonl_bytes(&bytes, "train").expect("parse must succeed");
        let reencoded = encode_jsonl(&parsed).expect("re-encode must succeed");
        assert_eq!(reencoded, bytes, "encode(parse(b)) must equal b");
    }

    #[test]
    fn schema_encode_emits_one_newline_terminated_line_per_row() {
        let bytes = encode_jsonl(&canonical_two_rows()).expect("encode must succeed");
        assert_eq!(bytes.split(|byte| *byte == b'\n').count() - 1, 2);
        assert_eq!(bytes.last().copied(), Some(b'\n'));
    }

    #[test]
    fn schema_field_order_matches_the_committed_baseline() {
        let bytes =
            encode_jsonl(&[row("train:0", "a", 2, "favor", "train")]).expect("encode must succeed");
        let line = String::from_utf8(bytes).expect("encoded JSONL is UTF-8");
        assert_eq!(
            line,
            "{\"id\":\"train:0\",\"input\":\"a\",\"label\":2,\"label_text\":\"favor\",\"source_split\":\"train\"}\n"
        );
    }

    #[test]
    fn schema_parse_accepts_a_buffer_with_no_trailing_newline() {
        let bytes =
            "{\"id\":\"t:0\",\"input\":\"a\",\"label\":0,\"label_text\":\"none\",\"source_split\":\"train\"}";
        let parsed = parse_jsonl_bytes(bytes.as_bytes(), "train").expect("parse must succeed");
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn schema_parse_of_an_empty_buffer_yields_no_rows() {
        assert!(parse_jsonl_bytes(b"", "train")
            .expect("empty buffer parses")
            .is_empty());
    }
}
