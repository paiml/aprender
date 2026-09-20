//! ONT-001 §3.7, §5 ONT-4c1 — `extract:gguf`: a GGUF model becomes a `model:Model` node.
//!
//! Two inputs, one vocabulary. (1) The model-capability ladder: a contract whose `entity.type` is `gguf` and
//! that carries `ladder.rungs[]` yields one node per rung, keyed by the rung's `sha256` — the release ladder's
//! focus objects ARE the rungs (`gguf`, `sha256`, `arch`, `backends`, optional `hosts`, `required`), and the
//! multi-gigabyte files they name live on hosts, not in the tree. (2) A contract whose `entity.ref` names a
//! readable GGUF file yields a node from the file itself: the header's `general.architecture`,
//! `general.file_type`, the tensor count, and the file's measured sha256.
//!
//! The header reader is in-house (R-13): magic, version, tensor count, and the KV stream walked far enough to
//! read the two keys the vocabulary needs. A corrupt magic is a rejection naming the file (`pc_extract.gguf`,
//! R-3) — never a node with holes in it.
//!
//! IRIs: `https://ont.paiml.dev/v1alpha1/model/<sha256>`; classes `model:Model` and, for a required rung,
//! `model:RequiredModel` (so a shape may target required rungs alone). Zero blank nodes; two extractions of the
//! same corpus are byte-identical (R-15) because everything is read from files and sorted by the graph.

use std::collections::BTreeSet;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::ontology::rdf::{iri, Graph, Term, RDF_TYPE};

use super::pv_contract::scalar;

/// The vocabulary root for models: `https://ont.paiml.dev/v1alpha1/model/<name>`.
#[must_use]
pub fn model(name: &str) -> String {
    format!("{}model/{name}", crate::ontology::rdf::ONT_BASE)
}

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// What the header says. `arch` and `file_type` are `None` when the KV stream does not carry them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub arch: Option<String>,
    pub file_type: Option<u32>,
}

/// Why a file is not a GGUF this extractor will speak for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractError {
    pub file: String,
    pub what: String,
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.what)
    }
}

impl std::error::Error for ExtractError {}

/// One rung of the ladder, as the graph sees it. Public because the receipt resolver keys on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rung {
    pub id: String,
    pub sha256: String,
    pub arch: String,
    pub gguf: String,
    pub backends: Vec<String>,
    /// `hosts:` when the rung lists them; empty = every host with a tracked receipt (ONT-4c1).
    pub hosts: Vec<String>,
    pub required: bool,
    /// The contract stem the rung came from.
    pub contract: String,
}

/// What one run extracted, for the gate's report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GgufStats {
    /// `model:Model` nodes emitted from rungs.
    pub rungs: Vec<Rung>,
    /// `model:Model` nodes emitted from GGUF files read in the tree.
    pub files_read: usize,
    /// Files a contract named that this extractor refused, each naming why.
    pub errors: Vec<ExtractError>,
}

/// Parse a GGUF header (magic, version, counts, and the KV stream up to the two keys the vocabulary needs).
pub fn parse_header(file: &str, bytes: &[u8]) -> Result<GgufHeader, ExtractError> {
    let err = |what: &str| ExtractError {
        file: file.to_string(),
        what: what.to_string(),
    };
    if bytes.len() < 24 {
        return Err(err("shorter than a GGUF header (24 bytes)"));
    }
    if &bytes[0..4] != GGUF_MAGIC {
        return Err(err("magic is not GGUF"));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if !(2..=3).contains(&version) {
        return Err(err(&format!("GGUF version {version} is not 2 or 3")));
    }
    let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap_or([0; 8]));
    let kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap_or([0; 8]));
    let mut cur = Cursor {
        bytes,
        pos: 24,
        file,
    };
    let mut arch = None;
    let mut file_type = None;
    for _ in 0..kv_count {
        let key = cur.string()?;
        let ty = cur.u32()?;
        match (key.as_str(), ty) {
            ("general.architecture", 8) => arch = Some(cur.string()?),
            ("general.file_type", 4) => file_type = Some(cur.u32()?),
            _ => cur.skip_value(ty)?,
        }
        if arch.is_some() && file_type.is_some() {
            break;
        }
    }
    Ok(GgufHeader {
        version,
        tensor_count,
        kv_count,
        arch,
        file_type,
    })
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    file: &'a str,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], ExtractError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| self.err("length overflow"))?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| self.err("KV stream ends inside a value"))?;
        self.pos = end;
        Ok(s)
    }
    fn err(&self, what: &str) -> ExtractError {
        ExtractError {
            file: self.file.to_string(),
            what: format!("{what} (at byte {})", self.pos),
        }
    }
    fn u32(&mut self) -> Result<u32, ExtractError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, ExtractError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap_or([0; 8])))
    }
    fn string(&mut self) -> Result<String, ExtractError> {
        let n = self.u64()?;
        let n = usize::try_from(n).map_err(|_| self.err("string length overflows usize"))?;
        let b = self.take(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
    /// Skip one value of GGUF type `ty` (the enum of the format: 0 u8 … 12 f64).
    fn skip_value(&mut self, ty: u32) -> Result<(), ExtractError> {
        let width = match ty {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            8 => {
                self.string()?;
                return Ok(());
            }
            9 => {
                let elem_ty = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.skip_value(elem_ty)?;
                }
                return Ok(());
            }
            other => return Err(self.err(&format!("unknown GGUF value type {other}"))),
        };
        self.take(width)?;
        Ok(())
    }
}

/// Read the ladder rungs of a contract document (`ladder.rungs[]`), in file order.
#[must_use]
pub fn rungs_of(stem: &str, doc: &serde_yaml::Value) -> Vec<Rung> {
    let mut out = Vec::new();
    let Some(list) = doc
        .get("ladder")
        .and_then(|l| l.get("rungs"))
        .and_then(serde_yaml::Value::as_sequence)
    else {
        return out;
    };
    for r in list {
        let (Some(id), Some(sha)) = (scalar(r.get("id")), scalar(r.get("sha256"))) else {
            continue;
        };
        let strings = |k: &str| -> Vec<String> {
            r.get(k)
                .and_then(serde_yaml::Value::as_sequence)
                .map(|s| s.iter().filter_map(|x| scalar(Some(x))).collect())
                .unwrap_or_default()
        };
        out.push(Rung {
            id,
            sha256: sha.to_ascii_lowercase(),
            arch: scalar(r.get("arch")).unwrap_or_default(),
            gguf: scalar(r.get("gguf")).unwrap_or_default(),
            backends: strings("backends"),
            hosts: strings("hosts"),
            required: r
                .get("required")
                .and_then(serde_yaml::Value::as_bool)
                .unwrap_or(false),
            contract: stem.to_string(),
        });
    }
    out
}

/// Emit one `model:Model` node per rung.
pub fn emit_rung(g: &mut Graph, rung: &Rung) {
    let s = iri("model", &rung.sha256);
    g.insert(s.clone(), RDF_TYPE, Term::iri(model("Model")));
    if rung.required {
        g.insert(s.clone(), RDF_TYPE, Term::iri(model("RequiredModel")));
    }
    g.insert(s.clone(), model("id"), Term::string(&rung.id));
    g.insert(s.clone(), model("sha256"), Term::string(&rung.sha256));
    g.insert(s.clone(), model("format"), Term::string("gguf"));
    if !rung.arch.is_empty() {
        g.insert(s.clone(), model("arch"), Term::string(&rung.arch));
    }
    if !rung.gguf.is_empty() {
        g.insert(s.clone(), model("file"), Term::string(&rung.gguf));
    }
    for b in &rung.backends {
        g.insert(s.clone(), model("backend"), Term::string(b));
    }
    for h in &rung.hosts {
        g.insert(s.clone(), model("host"), Term::string(h));
    }
    g.insert(s.clone(), model("required"), Term::boolean(rung.required));
    g.insert(
        s,
        model("contract"),
        Term::iri(iri("contract", &rung.contract)),
    );
}

/// Emit one `model:Model` node from a GGUF file read in the tree.
pub fn emit_file(g: &mut Graph, stem: &str, rel: &str, bytes: &[u8]) -> Result<(), ExtractError> {
    let h = parse_header(rel, bytes)?;
    let sha = hex(&Sha256::digest(bytes));
    let s = iri("model", &sha);
    g.insert(s.clone(), RDF_TYPE, Term::iri(model("Model")));
    g.insert(s.clone(), model("sha256"), Term::string(&sha));
    g.insert(s.clone(), model("format"), Term::string("gguf"));
    g.insert(s.clone(), model("file"), Term::string(rel));
    g.insert(
        s.clone(),
        model("tensorCount"),
        Term::integer(h.tensor_count),
    );
    g.insert(
        s.clone(),
        model("ggufVersion"),
        Term::integer(u64::from(h.version)),
    );
    if let Some(a) = h.arch {
        g.insert(s.clone(), model("arch"), Term::string(a));
    }
    if let Some(ft) = h.file_type {
        g.insert(s.clone(), model("fileType"), Term::integer(u64::from(ft)));
    }
    g.insert(s, model("contract"), Term::iri(iri("contract", stem)));
    Ok(())
}

/// Every contract under `contract_dir` whose `entity.type` is `gguf`: its rungs, and its `entity.ref` when
/// that names a readable file. Files are resolved against the repository root (`contract_dir`'s parent).
pub fn extract(contract_dir: &Path, g: &mut Graph) -> GgufStats {
    let mut stats = GgufStats::default();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (stem, rel, doc) in super::pv_contract::documents(contract_dir) {
        let ty = scalar(doc.get("entity").and_then(|e| e.get("type")));
        if ty.as_deref() != Some("gguf") {
            continue;
        }
        for rung in rungs_of(&stem, &doc) {
            if seen.insert(rung.sha256.clone()) {
                emit_rung(g, &rung);
            }
            stats.rungs.push(rung);
        }
        if let Some(r) = scalar(doc.get("entity").and_then(|e| e.get("ref"))) {
            let root = contract_dir.parent().unwrap_or(contract_dir);
            let path = root.join(&r);
            if path.is_file() {
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
        }
        let _ = rel;
    }
    stats
}

/// The positive control for this extractor (R-3): a corrupt magic must be refused, every run.
#[must_use]
pub fn positive_control() -> bool {
    let mut bad = minimal_header("qwen2", 15);
    bad[0..4].copy_from_slice(b"GGUX");
    parse_header("__pc_gguf__", &bad).is_err()
}

/// A minimal, valid GGUF header carrying `general.architecture` and `general.file_type` (for fixtures and the
/// positive control). Version 3, zero tensors.
#[must_use]
pub fn minimal_header(arch: &str, file_type: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(GGUF_MAGIC);
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&2u64.to_le_bytes());
    let put_str = |b: &mut Vec<u8>, s: &str| {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    };
    put_str(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    put_str(&mut b, arch);
    put_str(&mut b, "general.file_type");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&file_type.to_le_bytes());
    b
}

#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::rdf::ont;

    #[test]
    fn a_minimal_header_parses_and_a_corrupt_magic_is_refused_by_name() {
        let h = parse_header("m.gguf", &minimal_header("qwen3", 15)).expect("parses");
        assert_eq!(h.version, 3);
        assert_eq!(h.arch.as_deref(), Some("qwen3"));
        assert_eq!(h.file_type, Some(15));
        assert_eq!(h.tensor_count, 0);
        let mut bad = minimal_header("qwen3", 15);
        bad[1] = b'X';
        let e = parse_header("m.gguf", &bad).expect_err("refused");
        assert_eq!(e.file, "m.gguf");
        assert!(e.what.contains("magic"), "{e}");
        assert!(positive_control());
    }

    #[test]
    fn a_truncated_kv_stream_is_refused_not_read_as_nothing() {
        let full = minimal_header("qwen3", 15);
        let cut = &full[..full.len() - 3];
        let e = parse_header("m.gguf", cut).expect_err("refused");
        assert!(e.what.contains("ends inside"), "{e}");
    }

    #[test]
    fn an_unknown_value_type_is_refused_by_number() {
        let mut b = Vec::new();
        b.extend_from_slice(GGUF_MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&(1u64).to_le_bytes());
        b.push(b'k');
        b.extend_from_slice(&99u32.to_le_bytes());
        let e = parse_header("m.gguf", &b).expect_err("refused");
        assert!(e.what.contains("value type 99"), "{e}");
    }

    #[test]
    fn rungs_become_typed_nodes_keyed_by_sha256_and_required_rungs_carry_the_second_class() {
        let doc: serde_yaml::Value = serde_yaml::from_str(
            "entity: {type: gguf}\nladder:\n  rungs:\n    - {id: a, sha256: AAAA, arch: qwen2, gguf: a.gguf, backends: [cpu, cuda], required: true}\n    - {id: b, sha256: bbbb, arch: qwen3, gguf: b.gguf, backends: [cpu], hosts: [gx10], required: false}\n",
        )
        .expect("yaml");
        let rungs = rungs_of("ladder", &doc);
        assert_eq!(rungs.len(), 2);
        assert_eq!(rungs[0].sha256, "aaaa", "lower-cased");
        assert_eq!(rungs[1].hosts, vec!["gx10".to_string()]);
        let mut g = Graph::new();
        for r in &rungs {
            emit_rung(&mut g, r);
        }
        let a = iri("model", "aaaa");
        assert_eq!(g.instances_of(&model("Model")).len(), 2);
        assert_eq!(g.instances_of(&model("RequiredModel")), vec![a.as_str()]);
        assert_eq!(g.objects(&a, &model("backend")).len(), 2);
        assert_eq!(
            g.objects(&a, &model("contract"))[0].as_iri(),
            Some(iri("contract", "ladder").as_str())
        );
        assert!(!g.to_ntriples().contains("_:"));
        let _ = ont("Contract");
    }

    #[test]
    fn a_file_node_carries_the_measured_sha256_and_the_header_facts() {
        let bytes = minimal_header("qwen35", 15);
        let mut g = Graph::new();
        emit_file(&mut g, "c", "models/m.gguf", &bytes).expect("emitted");
        let sha = hex(&Sha256::digest(&bytes));
        let s = iri("model", &sha);
        assert_eq!(g.instances_of(&model("Model")), vec![s.as_str()]);
        assert_eq!(
            g.objects(&s, &model("arch"))[0].as_literal().map(|l| l.0),
            Some("qwen35")
        );
        assert_eq!(
            g.objects(&s, &model("tensorCount"))[0]
                .as_literal()
                .map(|l| l.0),
            Some("0")
        );
    }
}
