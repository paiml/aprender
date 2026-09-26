//! ONT-001 §3.7, §5 ONT-4c1 — `extract:apr-model`: an aprender `.apr` (v2 container) becomes a `model:Model` node.
//!
//! In-house reader of exactly what the vocabulary needs (R-13): the 64-byte header (`APR\0`, version, flags,
//! `tensor_count`, the section offsets, a CRC32 over the header with the checksum field zeroed) and the tensor
//! index, walked entry by entry between `tensor_index_offset` and `data_offset` so the header's `tensor_count`
//! is checked against the entries the file actually holds. A header that lies about its count — the
//! `pc_extract["apr-model"]` positive control, R-3 — is a rejection naming the file, never a node.
//!
//! The layout is `crates/apr-format/src/v2/{header_impl,tensor_index_impl}.rs`'s; this module does not depend
//! on that crate (the contracts crate stays the leaf `pv` links) and reads only the fields it emits.

use std::path::Path;

use sha2::{Digest, Sha256};

use crate::ontology::rdf::{iri, Graph, Term, RDF_TYPE};

use super::gguf::{hex, model, ExtractError};
use super::pv_contract::scalar;

pub const APR_MAGIC: &[u8; 4] = b"APR\0";
pub const HEADER_SIZE: usize = 64;

/// What the header and the index agree on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AprHeader {
    pub version: (u8, u8),
    pub tensor_count: u32,
    pub metadata_offset: u64,
    pub metadata_size: u32,
    pub tensor_index_offset: u64,
    pub data_offset: u64,
    /// Entries the index section actually holds (equal to `tensor_count`, or the file is refused).
    pub index_entries: u32,
}

/// Parse the header and walk the tensor index. Refuses a bad magic, a checksum that does not match, a header
/// whose `tensor_count` disagrees with the index section, and an index that runs past the file.
pub fn parse(file: &str, bytes: &[u8]) -> Result<AprHeader, ExtractError> {
    let err = |what: String| ExtractError {
        file: file.to_string(),
        what,
    };
    if bytes.len() < HEADER_SIZE {
        return Err(err(format!(
            "shorter than an APR v2 header ({HEADER_SIZE} bytes)"
        )));
    }
    if &bytes[0..4] != APR_MAGIC {
        return Err(err("magic is not APR\\0".into()));
    }
    let version = (bytes[4], bytes[5]);
    let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let metadata_offset = u64::from_le_bytes(bytes[12..20].try_into().unwrap_or([0; 8]));
    let metadata_size = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    let tensor_index_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap_or([0; 8]));
    let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap_or([0; 8]));
    let checksum = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
    if checksum != header_checksum(&bytes[..HEADER_SIZE]) {
        return Err(err("header checksum does not match the header bytes".into()));
    }
    let index_entries =
        count_index_entries(bytes, tensor_index_offset, data_offset).map_err(&err)?;
    if index_entries != tensor_count {
        return Err(err(format!(
            "header says {tensor_count} tensor(s), the index holds {index_entries}"
        )));
    }
    Ok(AprHeader {
        version,
        tensor_count,
        metadata_offset,
        metadata_size,
        tensor_index_offset,
        data_offset,
        index_entries,
    })
}

/// CRC32 (IEEE) over header bytes 0..40 ++ 44..64 — the checksum field left out, exactly as
/// `AprV2Header::compute_checksum` concatenates the two regions.
#[must_use]
pub fn header_checksum(header: &[u8]) -> u32 {
    let mut data = Vec::with_capacity(60);
    data.extend_from_slice(&header[0..40]);
    data.extend_from_slice(&header[44..HEADER_SIZE]);
    crc32(&data)
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Walk `[tensor_index_offset, data_offset)` entry by entry: `name_len u16, name, dtype u8, ndim u8, ndim×u64,
/// offset u64, size u64`. Returns how many entries fit exactly.
fn count_index_entries(bytes: &[u8], start: u64, end: u64) -> Result<u32, String> {
    let start =
        usize::try_from(start).map_err(|_| "tensor_index_offset overflows usize".to_string())?;
    let end = usize::try_from(end).map_err(|_| "data_offset overflows usize".to_string())?;
    if end > bytes.len() || start > end {
        return Err(format!(
            "tensor index [{start}, {end}) does not fit a {}-byte file",
            bytes.len()
        ));
    }
    let mut pos = start;
    let mut n: u32 = 0;
    while pos < end {
        let name_len = usize::from(u16::from_le_bytes([
            *bytes.get(pos).ok_or("index truncated")?,
            *bytes.get(pos + 1).ok_or("index truncated")?,
        ]));
        pos += 2 + name_len;
        let ndim = usize::from(*bytes.get(pos + 1).ok_or("index truncated")?);
        pos += 2 + 8 * ndim + 16;
        if pos > end {
            return Err("an index entry runs past data_offset".into());
        }
        n += 1;
        // the index is padded to the alignment boundary after the last entry: stop when the rest is zero
        if bytes[pos..end].iter().all(|b| *b == 0) {
            break;
        }
    }
    Ok(n)
}

/// What one run extracted.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AprStats {
    pub files_read: usize,
    pub errors: Vec<ExtractError>,
}

/// Emit one node from a `.apr` file.
pub fn emit_file(g: &mut Graph, stem: &str, rel: &str, bytes: &[u8]) -> Result<(), ExtractError> {
    let h = parse(rel, bytes)?;
    let sha = hex(&Sha256::digest(bytes));
    let s = iri("model", &sha);
    g.insert(s.clone(), RDF_TYPE, Term::iri(model("Model")));
    g.insert(s.clone(), model("sha256"), Term::string(&sha));
    g.insert(s.clone(), model("format"), Term::string("apr"));
    g.insert(s.clone(), model("file"), Term::string(rel));
    g.insert(
        s.clone(),
        model("tensorCount"),
        Term::integer(u64::from(h.tensor_count)),
    );
    g.insert(
        s.clone(),
        model("aprVersion"),
        Term::string(format!("{}.{}", h.version.0, h.version.1)),
    );
    g.insert(s, model("contract"), Term::iri(iri("contract", stem)));
    Ok(())
}

/// Every contract under `contract_dir` whose `entity.type` is `apr-model` and whose `entity.ref` names a file.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> AprStats {
    let mut stats = AprStats::default();
    let root = contract_dir.parent().unwrap_or(contract_dir);
    for (stem, _rel, doc) in super::pv_contract::documents(contract_dir) {
        let entity = doc.get("entity");
        if scalar(entity.and_then(|e| e.get("type"))).as_deref() != Some("apr-model") {
            continue;
        }
        let Some(r) = scalar(entity.and_then(|e| e.get("ref"))) else {
            continue;
        };
        let path = root.join(&r);
        match std::fs::read(&path) {
            Ok(bytes) => match emit_file(g, &stem, &r, &bytes) {
                Ok(()) => stats.files_read += 1,
                Err(e) => stats.errors.push(e),
            },
            Err(e) => stats.errors.push(ExtractError {
                file: r,
                what: format!("unreadable: {e}"),
            }),
        }
    }
    stats
}

/// The positive control (R-3): a header whose `tensor_count` disagrees with its index — with the checksum
/// recomputed so the lie is internally consistent — must be refused, every run.
#[must_use]
pub fn positive_control(sample: &[u8]) -> bool {
    let Some(lying) = lie_about_tensor_count(sample) else {
        return false;
    };
    parse("__pc_apr_model__", &lying).is_err() && parse("__pc_apr_model_sample__", sample).is_ok()
}

/// A copy of `sample` whose header claims one more tensor than the index holds, checksum recomputed.
#[must_use]
pub fn lie_about_tensor_count(sample: &[u8]) -> Option<Vec<u8>> {
    if sample.len() < HEADER_SIZE {
        return None;
    }
    let mut b = sample.to_vec();
    let n = u32::from_le_bytes([b[8], b[9], b[10], b[11]]).checked_add(1)?;
    b[8..12].copy_from_slice(&n.to_le_bytes());
    let c = header_checksum(&b[..HEADER_SIZE]);
    b[40..44].copy_from_slice(&c.to_le_bytes());
    Some(b)
}

/// A tiny valid `.apr` v2 container with `n` zero-sized f32 tensors, for fixtures and the positive control when
/// no sample is at hand: header, empty metadata, an index of `n` entries, no data.
#[must_use]
pub fn minimal_container(n: u32) -> Vec<u8> {
    let mut index = Vec::new();
    for i in 0..n {
        let name = format!("t{i}");
        index.extend_from_slice(&(name.len() as u16).to_le_bytes());
        index.extend_from_slice(name.as_bytes());
        index.push(1); // dtype byte (any value the reader does not interpret)
        index.push(1); // ndim
        index.extend_from_slice(&0u64.to_le_bytes());
        index.extend_from_slice(&0u64.to_le_bytes()); // offset
        index.extend_from_slice(&0u64.to_le_bytes()); // size
    }
    let metadata = b"{}";
    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata.len() as u64;
    let data_offset = tensor_index_offset + index.len() as u64;
    let mut h = [0u8; HEADER_SIZE];
    h[0..4].copy_from_slice(APR_MAGIC);
    h[4] = 2;
    h[5] = 0;
    h[8..12].copy_from_slice(&n.to_le_bytes());
    h[12..20].copy_from_slice(&metadata_offset.to_le_bytes());
    h[20..24].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
    h[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    h[32..40].copy_from_slice(&data_offset.to_le_bytes());
    let c = header_checksum(&h);
    h[40..44].copy_from_slice(&c.to_le_bytes());
    let mut out = h.to_vec();
    out.extend_from_slice(metadata);
    out.extend_from_slice(&index);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_minimal_container_parses_and_its_lying_copy_is_refused_naming_the_counts() {
        let good = minimal_container(2);
        let h = parse("m.apr", &good).expect("parses");
        assert_eq!(h.tensor_count, 2);
        assert_eq!(h.index_entries, 2);
        let lie = lie_about_tensor_count(&good).expect("copy");
        let e = parse("m.apr", &lie).expect_err("refused");
        assert!(e.what.contains("header says 3"), "{e}");
        assert!(e.what.contains("index holds 2"), "{e}");
        assert!(positive_control(&good));
    }

    #[test]
    fn a_stale_checksum_and_a_bad_magic_are_each_refused() {
        let mut stale = minimal_container(1);
        stale[8] = 5; // tensor_count changed, checksum not
        let e = parse("m.apr", &stale).expect_err("refused");
        assert!(e.what.contains("checksum"), "{e}");
        let mut bad = minimal_container(1);
        bad[0] = b'X';
        let e = parse("m.apr", &bad).expect_err("refused");
        assert!(e.what.contains("magic"), "{e}");
    }

    #[test]
    fn crc32_matches_the_ieee_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn a_node_carries_the_measured_sha_and_the_tensor_count() {
        let bytes = minimal_container(3);
        let mut g = Graph::new();
        emit_file(&mut g, "c", "models/m.apr", &bytes).expect("emitted");
        let sha = hex(&Sha256::digest(&bytes));
        let s = iri("model", &sha);
        assert_eq!(
            g.objects(&s, &model("tensorCount"))[0]
                .as_literal()
                .map(|l| l.0),
            Some("3")
        );
        assert_eq!(
            g.objects(&s, &model("format"))[0].as_literal().map(|l| l.0),
            Some("apr")
        );
    }
}

#[cfg(test)]
mod golden_tests {
    use super::*;

    /// The real fixture the row names: `crates/apr-format/tests/fixtures/golden_v2.apr`, 2 tensors.
    #[test]
    fn golden_v2_parses_with_two_tensors_and_its_lying_copy_is_refused() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../apr-format/tests/fixtures/golden_v2.apr");
        let bytes = std::fs::read(&p).expect("golden_v2.apr is in the tree");
        let h = parse("golden_v2.apr", &bytes).expect("the golden container parses");
        assert_eq!(h.tensor_count, 2);
        assert_eq!(h.index_entries, 2);
        assert!(positive_control(&bytes));
    }
}

#[cfg(test)]
mod fixture_tests {
    use super::*;

    /// The committed lying fixture IS `lie_about_tensor_count(minimal_container(2))`, byte for byte — a drifted
    /// fixture fails here, not in the gate that reads it.
    #[test]
    fn the_lying_apr_fixture_is_the_generated_bytes() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont/ladder-lyingapr/models/lying.apr");
        let on_disk = std::fs::read(&p).expect("fixture tracked");
        let generated = lie_about_tensor_count(&minimal_container(2)).expect("generated");
        assert_eq!(on_disk, generated);
        assert!(parse("lying.apr", &on_disk).is_err());
    }
}
