//! ONT-001 §5 ONT-4c1 — `resolves: receipt`: the tracked model-capability-ladder receipts, read once, joined
//! to the rungs by sha256, and materialized as edges a shape can count.
//!
//! One reader, two versions of one schema: `evidence/dogfood/models/<version>/<host>.json` with
//! `schema: apr-model-ladder-receipt/v1` or `/v2` (aprender `scripts/model_ladder.sh`; v2 is #3712's, and adds the
//! host's measured `inventory[]`, the full `apr_sha`, and — with the widened 0.69.1 bar — `cells[]`, one row per
//! (model × verb × context rung), which the release-readiness resolver reads, #3715). Any other `schema` value is
//! refused BY NAME — a receipt this reader does not understand is not silently read as empty. v2 rows are v1 rows
//! plus `file`/`inventory_only`, so the rung join below reads both versions the same way. A rung row with
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
/// #3712's successor: v1 plus `inventory[]`, `apr_sha` and `cells[]`.
pub const SCHEMA_V2: &str = "apr-model-ladder-receipt/v2";
/// Every schema this reader understands, in version order.
pub const SCHEMAS: [&str; 2] = [SCHEMA, SCHEMA_V2];
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

/// One model the host HOLDS (v2 `inventory[]`): the universe the release must prove, measured on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InventoryItem {
    pub file: String,
    /// `None` when the row carries no hash: no measurement, never a match.
    pub sha256: Option<String>,
    pub arch: Option<String>,
    pub quant: Option<String>,
    /// The GGUF's own `*.context_length`, read from the file on the host (#3710: "not picked here").
    pub context_length: Option<u64>,
    /// The thinking modes the model's chat template supports, DERIVED from `tokenizer.chat_template` (#3723,
    /// the cop's rule): `enable_thinking` → `["on","off"]`; a generation prompt that always opens `<think>` →
    /// `["on"]`; otherwise `["off"]`. `None` (no template read) owes BOTH modes.
    pub thinking_modes: Option<Vec<String>>,
    /// The evidence for `thinking_modes`: the markers found in `tokenizer.chat_template`, and that template's hash.
    pub thinking_markers: Option<Vec<String>>,
    pub chat_template_sha256: Option<String>,
    /// The producer's ECHO of whether this model owes the long rungs; pv re-derives it from the ladder contract
    /// and a disagreement is a violation (#3712 amendment 2).
    pub owes_long_rungs: Option<bool>,
    /// The declared memory arithmetic (#3710 rule, operator "a"): weights + KV per token (at the chosen
    /// precision) + workspace, compared with the host's measured `gpu_mem_total_bytes`.
    pub weights_bytes: Option<u64>,
    pub kv_bytes_per_token: Option<u64>,
    pub workspace_bytes: Option<u64>,
    /// The KV precision `kv_bytes_per_token` was computed at (#3712, 62: device KV was measured as f32).
    pub kv_dtype: Option<String>,
}

/// One (model × verb × thinking × context rung) row of v2 `cells[]` (#3712 / #3715).
#[derive(Debug, Clone, PartialEq)]
pub struct CellRow {
    pub sha256: Option<String>,
    pub file: String,
    pub verb: String,
    /// `on` / `off` (#3710, operator: "chat with and without thinking, ditto run, ditto code").
    pub thinking: String,
    pub context: String,
    pub prompt_tokens: Option<u64>,
    /// The output budget the request asked for; with `prompt_tokens`, what fills the declared context.
    pub max_tokens: Option<u64>,
    /// Thinking ON: did the think block close? `None` on a thinking-OFF row (and on a row that did not say).
    pub think_closed: Option<bool>,
    /// Characters of answer AFTER any think block. `Some(0)` is the empty answer #3720 forbids.
    pub answer_chars: Option<u64>,
    pub ttft_ms: Option<f64>,
    /// A pre-load refusal's arithmetic: what the cell needs, and what the host has. `verdict: "refused"`.
    pub required_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub verdict: String,
    pub backend: String,
    /// `None` when the row does not say — which is not `false`.
    pub fallback: Option<bool>,
    pub rc: Option<i64>,
    pub reason: String,
}

/// One receipt file.
#[derive(Debug, Clone, PartialEq)]
pub struct Receipt {
    /// Path relative to the repository root, `/`-separated (or as given, for a receipt dir outside the tree).
    pub file: String,
    pub schema: String,
    pub host: String,
    pub version: String,
    pub sha: String,
    /// v2: the full 40-hex commit the measured `apr` was built from. `None` on v1.
    pub apr_sha: Option<String>,
    pub cc: String,
    pub gpu: String,
    /// v2: the host's measured TOTAL GPU memory (unified on GB10), the right-hand side of the fit arithmetic.
    /// Owed iff required ≤ total; a refusal is honest only when required > total (#3712, 62): a co-tenant that
    /// holds memory must not shrink the owed set.
    pub gpu_mem_total_bytes: Option<u64>,
    /// v2: free GPU memory when measured: carried for the reader, never part of the fit.
    pub gpu_mem_free_bytes: Option<u64>,
    pub rows: Vec<Row>,
    /// v2 `inventory[]`; empty on v1 (which cannot say what the host holds).
    pub inventory: Vec<InventoryItem>,
    /// v2 `cells[]`; empty when the measuring side wrote none.
    pub cells: Vec<CellRow>,
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
/// one of [`SCHEMAS`] is an error naming the file and the schema it carries.
pub fn read_all(root: &Path) -> Result<Vec<Receipt>, ReceiptError> {
    read_dir(&root.join(EVIDENCE_DIR), root)
}

/// Read every `*.json` under `dir/**`, in byte order, naming each file relative to `root` when it lies under it
/// (a T-1 receipt dir such as `$AP/models-t1` does not, and is named as given).
pub fn read_dir(dir: &Path, root: &Path) -> Result<Vec<Receipt>, ReceiptError> {
    let mut files = Vec::new();
    walk(dir, &mut files);
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
    if !SCHEMAS.contains(&schema) {
        return Err(err(format!(
            "schema {schema:?} is not {} — refused by name",
            SCHEMAS.join(" or ")
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
    Ok(Receipt {
        file: file.to_string(),
        schema: schema.to_string(),
        host: s("host"),
        version: s("version"),
        sha: s("sha"),
        apr_sha: str_of(&v, "apr_sha").map(|x| x.to_ascii_lowercase()),
        cc: s("cc"),
        gpu: s("gpu"),
        gpu_mem_total_bytes: v
            .get("gpu_mem_total_bytes")
            .and_then(serde_json::Value::as_u64),
        gpu_mem_free_bytes: v
            .get("gpu_mem_free_bytes")
            .and_then(serde_json::Value::as_u64),
        rows: list(&v, "rungs").map(parse_row).collect(),
        inventory: list(&v, "inventory").map(parse_inventory_item).collect(),
        cells: list(&v, "cells").map(parse_cell).collect(),
    })
}

/// The elements of array `key`, or none.
fn list<'a>(v: &'a serde_json::Value, key: &str) -> impl Iterator<Item = &'a serde_json::Value> {
    v.get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
}

fn str_of(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

fn bool_of(v: &serde_json::Value, key: &str) -> bool {
    v.get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// `passed ∧ ¬skipped`: a skipped gate reports `passed: true`, and that is not a pass.
fn gate_passed(cap: Option<&serde_json::Value>) -> bool {
    cap.is_some_and(|c| bool_of(c, "passed") && !bool_of(c, "skipped"))
}

fn parse_row(r: &serde_json::Value) -> Row {
    let mut backends = BTreeMap::new();
    if let Some(bs) = r.get("backends").and_then(serde_json::Value::as_object) {
        for (name, st) in bs {
            backends.insert(name.clone(), (bool_of(st, "ran"), bool_of(st, "fallback")));
        }
    }
    Row {
        id: str_of(r, "id").unwrap_or_default(),
        present: bool_of(r, "present"),
        sha_ok: bool_of(r, "sha_ok"),
        sha256: str_of(r, "sha256").map(|x| x.to_ascii_lowercase()),
        required: bool_of(r, "required"),
        green: bool_of(r, "green"),
        capability_passed: gate_passed(r.get("capability_match")),
        backends,
    }
}

fn parse_inventory_item(r: &serde_json::Value) -> InventoryItem {
    InventoryItem {
        file: str_of(r, "file").unwrap_or_default(),
        sha256: str_of(r, "sha256").map(|x| x.to_ascii_lowercase()),
        arch: str_of(r, "arch"),
        quant: str_of(r, "quant"),
        context_length: r.get("context_length").and_then(serde_json::Value::as_u64),
        thinking_modes: strings(r, "thinking_modes"),
        thinking_markers: strings(r, "thinking_markers"),
        chat_template_sha256: str_of(r, "chat_template_sha256"),
        owes_long_rungs: r
            .get("owes_long_rungs")
            .and_then(serde_json::Value::as_bool),
        weights_bytes: r.get("weights_bytes").and_then(serde_json::Value::as_u64),
        kv_bytes_per_token: r
            .get("kv_bytes_per_token")
            .and_then(serde_json::Value::as_u64),
        workspace_bytes: r.get("workspace_bytes").and_then(serde_json::Value::as_u64),
        kv_dtype: str_of(r, "kv_dtype"),
    }
}

fn parse_cell(r: &serde_json::Value) -> CellRow {
    CellRow {
        sha256: str_of(r, "sha256").map(|x| x.to_ascii_lowercase()),
        file: str_of(r, "file").unwrap_or_default(),
        verb: str_of(r, "verb").unwrap_or_default(),
        thinking: str_of(r, "thinking").unwrap_or_default(),
        context: str_of(r, "context").unwrap_or_default(),
        prompt_tokens: r.get("prompt_tokens").and_then(serde_json::Value::as_u64),
        max_tokens: r.get("max_tokens").and_then(serde_json::Value::as_u64),
        think_closed: r.get("think_closed").and_then(serde_json::Value::as_bool),
        answer_chars: r.get("answer_chars").and_then(serde_json::Value::as_u64),
        ttft_ms: r.get("ttft_ms").and_then(serde_json::Value::as_f64),
        required_bytes: r.get("required_bytes").and_then(serde_json::Value::as_u64),
        available_bytes: r.get("available_bytes").and_then(serde_json::Value::as_u64),
        verdict: str_of(r, "verdict").unwrap_or_default(),
        backend: str_of(r, "backend").unwrap_or_default(),
        fallback: r.get("fallback").and_then(serde_json::Value::as_bool),
        rc: r.get("rc").and_then(serde_json::Value::as_i64),
        reason: str_of(r, "reason").unwrap_or_default(),
    }
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

/// An array of strings at `key`, or `None` when the key is absent (which is not an empty list).
fn strings(r: &serde_json::Value, key: &str) -> Option<Vec<String>> {
    r.get(key).and_then(serde_json::Value::as_array).map(|a| {
        a.iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_string)
            .collect()
    })
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
        assert!(e.what.contains(SCHEMA_V2), "{e}");
    }

    #[test]
    fn a_v2_receipt_joins_its_rungs_like_v1_and_carries_inventory_apr_sha_and_cells() {
        // #3712's v2: the rung rows are v1 rows plus `file`/`inventory_only`, so the ladder join is unchanged —
        // the arm that goes RED when v2 is refused (the collision #3715 closes) or read as empty.
        let row = OK.replace(
            r#""id":"a","#,
            r#""id":"a","file":"a.gguf","inventory_only":false,"#,
        );
        let text = format!(
            r#"{{"schema":"apr-model-ladder-receipt/v2","host":"lambda","version":"0.69.1","sha":"abc",
                "apr_sha":"ABCDEF0123456789ABCDEF0123456789ABCDEF01","cc":"8.9","gpu":"g",
                "inventory":[{{"file":"a.gguf","sha256":"AAAA","bytes":4}}],
                "cells":[{{"sha256":"aaaa","file":"a.gguf","verb":"chat","context":"golden","prompt_tokens":40,
                           "verdict":"pass","backend":"cuda","fallback":false,"rc":0,"reason":""}}],
                "rungs":[{row}]}}"#
        );
        let rec = parse("evidence/dogfood/models/0.69.1/lambda.json", &text).expect("v2 parses");
        assert_eq!(rec.schema, SCHEMA_V2);
        assert_eq!(
            rec.apr_sha.as_deref(),
            Some("abcdef0123456789abcdef0123456789abcdef01")
        );
        assert_eq!(rec.inventory.len(), 1);
        assert_eq!(rec.inventory[0].sha256.as_deref(), Some("aaaa"));
        assert_eq!(rec.cells.len(), 1);
        assert_eq!(rec.cells[0].verb, "chat");
        assert_eq!(rec.cells[0].fallback, Some(false));
        let mut g = Graph::new();
        let st = resolve(&mut g, &[rung("a", "aaaa", &[], true)], &[rec]);
        assert_eq!((st.witnesses, st.green_on), (1, 1));
        let v3 =
            parse("x.json", r#"{"schema":"apr-model-ladder-receipt/v3"}"#).expect_err("v3 refused");
        assert!(v3.what.contains("v3"), "{v3}");
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
        // listed on lambda ONLY: gx10's red receipt must not count against it — the arm that goes RED when
        // `hosts:` is ignored (measured: an `if true` in resolve() left the earlier gx10-listed version green)
        let scoped = rung("a", "aaaa", &["lambda"], true);
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
        assert_eq!(
            st.missing_green_hosts, 0,
            "gx10 is not listed, so its red receipt is not this rung's"
        );
        assert_eq!(st.green_on, 1, "lambda, the one listed host, is green");
        let s = iri("model", "aaaa");
        assert!(g.objects(&s, &model("missingGreenHost")).is_empty());
        let mut g = Graph::new();
        let st = resolve(&mut g, &[open], &[lambda_green, gx10_fallback]);
        assert_eq!(st.green_on, 1);
        assert_eq!(
            st.missing_green_hosts, 1,
            "no hosts: → both hosts expected, gx10 missing"
        );
    }
}
