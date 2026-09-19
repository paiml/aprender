//! ONT-001 §5 ONT-4c1 — `resolves: receipt`: the tracked model-capability-ladder receipts, read once, joined
//! to the rungs by sha256, and materialized as edges a shape can count.
//!
//! One reader, one schema: `evidence/dogfood/models/<version>/<host>.json` with
//! `schema: apr-model-ladder-receipt/v1` (aprender `scripts/model_ladder.sh`). Any other `schema` value is
//! refused BY NAME — a receipt this reader does not understand is not silently read as empty. A rung row with
//! `sha256` equal to the contract's is a **witness**; a row WITHOUT `sha256` is not a witness (the 0.68.1
//! receipts committed before aprender#3507 have none — "no measurement", never a match); a row with a
//! different hex is a **hex mismatch**, materialized so a shape rejects it naming rung, receipt and both hexes.
//!
//! What the resolver writes onto a rung node (all literals, all sorted by the graph — R-15):
//!
//! | edge | meaning |
//! |---|---|
//! | `model:parityReceipt <receipt>` | one per witness row, to a `model:Receipt` node |
//! | `model:receiptHexMismatch "<file>: <hex>"` | one per row whose hex differs |
//! | `model:unmeasuredRow "<file>"` | one per row with no `sha256` |
//! | `model:greenOn "<host>"` | one per witness row that is green ∧ capability passed ∧ every claimed backend ran without fallback |
//! | `model:missingGreenHost "<host>"` | one per host the rung lists (or, with no `hosts:`, every host with a receipt) that has no `greenOn` |
//!
//! and onto a receipt node (`…/receipt/<version>/<host>/<rung id>`): `model:receiptHost`, `model:receiptVersion`,
//! `model:cc` (a STRING — `"8.9"`, `"12.1"`), `model:gpu`, `model:green`, `model:capabilityPassed`
//! (`passed ∧ ¬skipped`: a skipped gate reports `passed: true`), `model:backendsOk`, `model:receiptSha256`,
//! `model:receiptFile`. The shapes then say `minCount 1` on `parityReceipt`, `maxCount 0` on
//! `receiptHexMismatch` / `missingGreenHost`, `in:` on `cc` — the whole quantification lives here, in Rust that
//! the fixtures falsify, and the shapes stay inside §3.6's subset.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::ontology::extract::gguf::{model, Rung};
use crate::ontology::rdf::{iri, Graph, Term, RDF_TYPE};

pub const SCHEMA: &str = "apr-model-ladder-receipt/v1";
/// Where the receipts live, relative to the repository root.
pub const EVIDENCE_DIR: &str = "evidence/dogfood/models";

/// One rung row of one receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub id: String,
    pub present: bool,
    pub sha_ok: bool,
    pub sha256: Option<String>,
    pub required: bool,
    pub green: bool,
    /// `capability_match.passed && !capability_match.skipped`.
    pub capability_passed: bool,
    /// backend → (ran, fallback)
    pub backends: BTreeMap<String, (bool, bool)>,
}

/// One receipt file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    /// Path relative to the repository root, `/`-separated.
    pub file: String,
    pub host: String,
    pub version: String,
    pub sha: String,
    pub cc: String,
    pub gpu: String,
    pub rows: Vec<Row>,
}

/// A receipt file this reader refuses, by name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptError {
    pub file: String,
    pub what: String,
}

impl std::fmt::Display for ReceiptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.what)
    }
}

impl std::error::Error for ReceiptError {}

/// Read every `*.json` under `<root>/evidence/dogfood/models/**`, in byte order. A file whose `schema` is not
/// [`SCHEMA`] is an error naming the file and the schema it carries.
pub fn read_all(root: &Path) -> Result<Vec<Receipt>, ReceiptError> {
    let dir = root.join(EVIDENCE_DIR);
    let mut files = Vec::new();
    walk(&dir, &mut files);
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let rel = f
            .strip_prefix(root)
            .unwrap_or(&f)
            .to_string_lossy()
            .replace('\\', "/");
        let text = std::fs::read_to_string(&f).map_err(|e| ReceiptError {
            file: rel.clone(),
            what: format!("unreadable: {e}"),
        })?;
        out.push(parse(&rel, &text)?);
    }
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("json") {
            out.push(p);
        }
    }
}

/// Parse one receipt document.
pub fn parse(file: &str, text: &str) -> Result<Receipt, ReceiptError> {
    let err = |what: String| ReceiptError {
        file: file.to_string(),
        what,
    };
    let v: serde_json::Value =
        serde_json::from_str(text).map_err(|e| err(format!("not JSON: {e}")))?;
    let schema = v
        .get("schema")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if schema != SCHEMA {
        return Err(err(format!(
            "schema {schema:?} is not {SCHEMA} — refused by name"
        )));
    }
    let s = |k: &str| {
        v.get(k)
            .map(|x| match x {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default()
    };
    let mut rows = Vec::new();
    for r in v
        .get("rungs")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
    {
        let b = |k: &str| {
            r.get(k)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        };
        let cap = r.get("capability_match");
        let capability_passed = cap
            .and_then(|c| c.get("passed"))
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
            && !cap
                .and_then(|c| c.get("skipped"))
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
        let mut backends = BTreeMap::new();
        if let Some(bs) = r.get("backends").and_then(serde_json::Value::as_object) {
            for (name, st) in bs {
                let ran = st
                    .get("ran")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                let fallback = st
                    .get("fallback")
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false);
                backends.insert(name.clone(), (ran, fallback));
            }
        }
        rows.push(Row {
            id: r
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
            present: b("present"),
            sha_ok: b("sha_ok"),
            sha256: r
                .get("sha256")
                .and_then(serde_json::Value::as_str)
                .map(str::to_ascii_lowercase),
            required: b("required"),
            green: b("green"),
            capability_passed,
            backends,
        });
    }
    Ok(Receipt {
        file: file.to_string(),
        host: s("host"),
        version: s("version"),
        sha: s("sha"),
        cc: s("cc"),
        gpu: s("gpu"),
        rows,
    })
}

/// What resolution found, for the gate's report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResolveStats {
    pub receipts: usize,
    pub witnesses: usize,
    pub hex_mismatches: usize,
    pub unmeasured_rows: usize,
    pub green_on: usize,
    pub missing_green_hosts: usize,
}

/// Join `receipts` to `rungs` by sha256 and write the edges described in the module doc.
pub fn resolve(g: &mut Graph, rungs: &[Rung], receipts: &[Receipt]) -> ResolveStats {
    let mut stats = ResolveStats {
        receipts: receipts.len(),
        ..ResolveStats::default()
    };
    let all_hosts: BTreeSet<String> = receipts.iter().map(|r| r.host.clone()).collect();
    for rung in rungs {
        let s = iri("model", &rung.sha256);
        let green_hosts = witness_rows(g, &s, rung, receipts, &mut stats);
        for h in &green_hosts {
            stats.green_on += 1;
            g.insert(s.clone(), model("greenOn"), Term::string(h));
        }
        let expected: BTreeSet<String> = if rung.hosts.is_empty() {
            all_hosts.clone()
        } else {
            rung.hosts.iter().cloned().collect()
        };
        for h in expected.difference(&green_hosts) {
            stats.missing_green_hosts += 1;
            g.insert(s.clone(), model("missingGreenHost"), Term::string(h));
        }
    }
    stats
}

/// One rung's rows across every receipt: witness / hex mismatch / unmeasured, and the hosts it is green on.
fn witness_rows(
    g: &mut Graph,
    s: &str,
    rung: &Rung,
    receipts: &[Receipt],
    stats: &mut ResolveStats,
) -> BTreeSet<String> {
    let mut green_hosts: BTreeSet<String> = BTreeSet::new();
    for rec in receipts {
        for row in rec.rows.iter().filter(|r| r.id == rung.id) {
            let Some(hex) = &row.sha256 else {
                stats.unmeasured_rows += 1;
                g.insert(
                    s.to_string(),
                    model("unmeasuredRow"),
                    Term::string(&rec.file),
                );
                continue;
            };
            if hex != &rung.sha256 {
                stats.hex_mismatches += 1;
                g.insert(
                    s.to_string(),
                    model("receiptHexMismatch"),
                    Term::string(format!("{}: {hex}", rec.file)),
                );
                continue;
            }
            stats.witnesses += 1;
            let node = emit_receipt_node(g, rec, row, rung);
            g.insert(s.to_string(), model("parityReceipt"), Term::iri(node));
            if row.green && row.capability_passed && backends_ok(rung, row) {
                green_hosts.insert(rec.host.clone());
            }
        }
    }
    green_hosts
}

/// Every backend the rung claims ran on this row without falling back.
fn backends_ok(rung: &Rung, row: &Row) -> bool {
    rung.backends.iter().all(|b| {
        row.backends
            .get(b)
            .is_some_and(|(ran, fallback)| *ran && !*fallback)
    })
}

fn emit_receipt_node(g: &mut Graph, rec: &Receipt, row: &Row, rung: &Rung) -> String {
    let node = iri(
        "receipt",
        &format!("{}/{}/{}", rec.version, rec.host, row.id),
    );
    g.insert(node.clone(), RDF_TYPE, Term::iri(model("Receipt")));
    g.insert(node.clone(), model("receiptHost"), Term::string(&rec.host));
    g.insert(
        node.clone(),
        model("receiptVersion"),
        Term::string(&rec.version),
    );
    g.insert(node.clone(), model("receiptFile"), Term::string(&rec.file));
    g.insert(node.clone(), model("cc"), Term::string(&rec.cc));
    g.insert(node.clone(), model("gpu"), Term::string(&rec.gpu));
    g.insert(node.clone(), model("green"), Term::boolean(row.green));
    g.insert(
        node.clone(),
        model("capabilityPassed"),
        Term::boolean(row.capability_passed),
    );
    g.insert(
        node.clone(),
        model("backendsOk"),
        Term::boolean(backends_ok(rung, row)),
    );
    if let Some(h) = &row.sha256 {
        g.insert(node.clone(), model("receiptSha256"), Term::string(h));
    }
    for (b, (ran, fallback)) in &row.backends {
        g.insert(
            node.clone(),
            model("backendRan"),
            Term::string(format!(
                "{b}={}",
                if *ran && !*fallback {
                    "ok"
                } else if *ran {
                    "fallback"
                } else {
                    "no"
                }
            )),
        );
    }
    g.insert(
        node.clone(),
        model("receiptOf"),
        Term::iri(iri("model", &rung.sha256)),
    );
    node
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rung(id: &str, sha: &str, hosts: &[&str], required: bool) -> Rung {
        Rung {
            id: id.into(),
            sha256: sha.into(),
            arch: "qwen2".into(),
            gguf: format!("{id}.gguf"),
            backends: vec!["cpu".into(), "cuda".into()],
            hosts: hosts.iter().map(|h| (*h).to_string()).collect(),
            required,
            contract: "ladder".into(),
        }
    }

    fn receipt(host: &str, cc: &str, rows: &str) -> Receipt {
        parse(
            &format!("evidence/dogfood/models/0.68.1/{host}.json"),
            &format!(
                r#"{{"schema":"apr-model-ladder-receipt/v1","host":"{host}","version":"0.68.1","sha":"abc","cc":"{cc}","gpu":"g","rungs":[{rows}]}}"#
            ),
        )
        .expect("parses")
    }

    const OK: &str = r#"{"id":"a","present":true,"sha_ok":true,"sha256":"aaaa","required":true,"green":true,"capability_match":{"passed":true,"skipped":false},"backends":{"cpu":{"ran":true,"fallback":false},"cuda":{"ran":true,"fallback":false}}}"#;

    #[test]
    fn a_foreign_schema_is_refused_by_name() {
        let e = parse("x.json", r#"{"schema":"something/v9","rungs":[]}"#).expect_err("refused");
        assert!(e.what.contains("something/v9"), "{e}");
        assert!(e.what.contains(SCHEMA), "{e}");
    }

    #[test]
    fn a_witness_yields_a_receipt_edge_and_a_green_host() {
        let mut g = Graph::new();
        let r = rung("a", "aaaa", &[], true);
        let st = resolve(&mut g, &[r], &[receipt("lambda", "8.9", OK)]);
        assert_eq!(st.witnesses, 1);
        assert_eq!(st.green_on, 1);
        assert_eq!(st.missing_green_hosts, 0);
        let s = iri("model", "aaaa");
        assert_eq!(g.objects(&s, &model("parityReceipt")).len(), 1);
        let node = g.objects(&s, &model("parityReceipt"))[0]
            .as_iri()
            .expect("iri")
            .to_string();
        assert_eq!(
            g.objects(&node, &model("cc"))[0].as_literal().map(|l| l.0),
            Some("8.9")
        );
        assert!(!g.to_ntriples().contains("_:"));
    }

    #[test]
    fn a_row_without_sha256_is_not_a_witness_and_a_different_hex_is_a_mismatch() {
        let mut g = Graph::new();
        let r = rung("a", "aaaa", &[], true);
        let no_sha = OK.replace(r#""sha256":"aaaa","#, "");
        let bad = OK.replace(r#""sha256":"aaaa""#, r#""sha256":"bbbb""#);
        let st = resolve(
            &mut g,
            &[r],
            &[
                receipt("lambda", "8.9", &no_sha),
                receipt("gx10", "12.1", &bad),
            ],
        );
        assert_eq!(st.witnesses, 0);
        assert_eq!(st.unmeasured_rows, 1);
        assert_eq!(st.hex_mismatches, 1);
        let s = iri("model", "aaaa");
        let m = g.objects(&s, &model("receiptHexMismatch"));
        assert_eq!(m.len(), 1);
        assert!(m[0]
            .as_literal()
            .map(|l| l.0)
            .unwrap_or("")
            .contains("gx10.json: bbbb"));
        // two hosts, none green
        assert_eq!(st.missing_green_hosts, 2);
    }

    #[test]
    fn a_skipped_capability_gate_or_a_fallback_backend_is_not_green() {
        let skipped = OK.replace(r#""skipped":false"#, r#""skipped":true"#);
        let fallback = OK.replace(
            r#""cuda":{"ran":true,"fallback":false}"#,
            r#""cuda":{"ran":true,"fallback":true}"#,
        );
        for (name, rows) in [("skipped", skipped), ("fallback", fallback)] {
            let mut g = Graph::new();
            let st = resolve(
                &mut g,
                &[rung("a", "aaaa", &[], true)],
                &[receipt("lambda", "8.9", &rows)],
            );
            assert_eq!(st.witnesses, 1, "{name}: still a witness");
            assert_eq!(st.green_on, 0, "{name}: not green");
            assert_eq!(st.missing_green_hosts, 1, "{name}");
        }
    }

    #[test]
    fn hosts_scope_the_green_requirement_and_absence_means_every_host_with_a_receipt() {
        let scoped = rung("a", "aaaa", &["gx10"], true);
        let open = rung("a", "aaaa", &[], true);
        let lambda_green = receipt("lambda", "8.9", OK);
        let gx10_fallback = receipt(
            "gx10",
            "12.1",
            &OK.replace(
                r#""cuda":{"ran":true,"fallback":false}"#,
                r#""cuda":{"ran":true,"fallback":true}"#,
            ),
        );
        let mut g = Graph::new();
        let st = resolve(
            &mut g,
            &[scoped],
            &[lambda_green.clone(), gx10_fallback.clone()],
        );
        assert_eq!(st.missing_green_hosts, 1, "gx10 is listed and not green");
        let s = iri("model", "aaaa");
        assert_eq!(
            g.objects(&s, &model("missingGreenHost"))[0]
                .as_literal()
                .map(|l| l.0),
            Some("gx10")
        );
        let mut g = Graph::new();
        let st = resolve(&mut g, &[open], &[lambda_green, gx10_fallback]);
        assert_eq!(st.green_on, 1);
        assert_eq!(
            st.missing_green_hosts, 1,
            "no hosts: → both hosts expected, gx10 missing"
        );
    }
}
