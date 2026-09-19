//! ONT-001 §3.6, §3.8, §5 ONT-4b2 — the differential oracle, out of the gate path (R-13).
//!
//! Two arms, one question: does the in-house validator agree with a real SHACL processor?
//!
//! 1. **W3C arm.** For every vendored case under `w3c/<group>/<name>.ttl` that the subset claims (the ids come
//!    from `crates/aprender-contracts/w3c/*.yaml`, the translations the gate runs), the pinned `shacl` crate
//!    validates the ORIGINAL case file — `sh:targetNode`, blank nodes and all — and its `(focus, component)`
//!    multiset is compared with the expectation the translation states. The in-house side already ran the same
//!    expectation in `cargo test` (`ontology::w3c`), so agreement here means the translation is faithful and
//!    both validators agree with the standard; a disagreement names the case.
//! 2. **Corpus arm.** The oracle validates `contracts/contracts.nt` against `contracts/shapes.ttl` — both
//!    written by `pv extract` — and its violations are compared with pv's own, read from the JSON report
//!    `make oracle` passes in.
//!
//! Output: `tests/oracle/differential.json` — `{cases, disagreements, …}` — which ONT-4b2's probe reads. A
//! disagreement is never resolved here: it is printed, counted, and the row's gate declines `Unknown{Differential}`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rudof_rdf::rdf_core::RDFFormat;
use rudof_rdf::rdf_impl::ReaderMode;
use shacl::ir::IRSchema;
use shacl::validator::processor::{GraphValidation, ShaclProcessor};
use shacl::validator::{ShaclConfig, ShaclValidationMode};

/// `sh:MinCountConstraintComponent` → `minCount`, the name the in-house report uses.
fn short_component(iri: &str) -> String {
    let local = iri.rsplit(['#', '/']).next().unwrap_or(iri);
    let base = local.strip_suffix("ConstraintComponent").unwrap_or(local);
    let mut c = base.chars();
    match c.next() {
        Some(f) => f.to_lowercase().chain(c).collect(),
        None => base.to_string(),
    }
}

/// One `(focus, component)` pair, sorted for multiset comparison.
type Pair = (String, String);

/// A `file:` base for a case, because the W3C cases name their own manifest node `<>` — a relative IRI that has
/// no scheme until a base supplies one, which the Turtle reader refuses outright ("No scheme found in an
/// absolute IRI"). The base never appears in a result: the cases' own terms are all `ex:`-prefixed.
fn base_of(path: &Path) -> String {
    let abs = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    format!("file://{}", abs.display())
}

fn oracle_pairs(data: &Path, shapes: &Path) -> Result<Vec<Pair>, String> {
    let shapes_base = base_of(shapes);
    let schema = IRSchema::from_reader(
        &mut std::fs::File::open(shapes).map_err(|e| format!("{}: {e}", shapes.display()))?,
        &shapes.display().to_string(),
        &RDFFormat::Turtle,
        Some(&shapes_base),
        &ReaderMode::Lax,
    )
    .map_err(|e| format!("{}: {e}", shapes.display()))?;
    let format = if data.extension().is_some_and(|e| e == "nt") {
        RDFFormat::NTriples
    } else {
        RDFFormat::Turtle
    };
    let data_base = base_of(data);
    let mut validation = GraphValidation::from_path(data, format, Some(&data_base))
        .map_err(|e| format!("{}: {e}", data.display()))?;
    let report = validation
        .validate(&schema, &ShaclValidationMode::Native, &ShaclConfig::new())
        .map_err(|e| format!("validate {}: {e}", data.display()))?;
    let mut pairs: Vec<Pair> = report
        .results()
        .iter()
        .map(|r| {
            (
                r.focus_node().to_string().trim_matches(['<', '>']).to_string(),
                short_component(&r.constraint_component().to_string()),
            )
        })
        .collect();
    pairs.sort();
    Ok(pairs)
}

/// The expectation a translation states: `(focus, component)` pairs, expanded with the case's prefix.
fn expected_pairs(yaml: &Path) -> Result<(String, Vec<Pair>), String> {
    let text = std::fs::read_to_string(yaml).map_err(|e| format!("{}: {e}", yaml.display()))?;
    let doc: serde_yaml::Value =
        serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", yaml.display()))?;
    let id = doc
        .get("id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{}: no id", yaml.display()))?
        .to_string();
    let prefix = doc
        .get("prefix")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{}: no prefix", yaml.display()))?;
    let mut pairs = Vec::new();
    if let Some(list) = doc
        .get("expect")
        .and_then(|e| e.get("results"))
        .and_then(|r| r.as_sequence())
    {
        for r in list {
            let focus = r.get("focus").and_then(|v| v.as_str()).unwrap_or_default();
            let component = r
                .get("component")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let focus = focus
                .strip_prefix("ex:")
                .map_or_else(|| focus.to_string(), |l| format!("{prefix}{l}"));
            pairs.push((focus, component));
        }
    }
    pairs.sort();
    Ok((id, pairs))
}

fn main() {
    let root = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    let pv_report = std::env::args().nth(2).map(PathBuf::from);
    let oracle_dir = root.join("tests/oracle");
    let translations = root.join("crates/aprender-contracts/w3c");

    let mut cases = 0usize;
    let mut disagreements: Vec<String> = Vec::new();
    let mut per_case: BTreeMap<String, String> = BTreeMap::new();

    let mut files: Vec<PathBuf> = std::fs::read_dir(&translations)
        .unwrap_or_else(|e| panic!("{}: {e}", translations.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "yaml"))
        .collect();
    files.sort();

    for yaml in &files {
        let (id, expected) = match expected_pairs(yaml) {
            Ok(x) => x,
            Err(e) => {
                disagreements.push(format!("translation unreadable: {e}"));
                continue;
            }
        };
        let case_file = oracle_dir.join("w3c").join(format!("{id}.ttl"));
        // A case whose data and shapes are separate files (datatype-ill-formed) names them by suffix.
        let data_alt = oracle_dir.join("w3c").join(format!("{id}-data.ttl"));
        let shapes_alt = oracle_dir.join("w3c").join(format!("{id}-shapes.ttl"));
        let (data, shapes) = if data_alt.is_file() && shapes_alt.is_file() {
            (data_alt, shapes_alt)
        } else {
            (case_file.clone(), case_file.clone())
        };
        cases += 1;
        match oracle_pairs(&data, &shapes) {
            Ok(got) => {
                if got == expected {
                    per_case.insert(id, "agree".to_string());
                } else {
                    per_case.insert(id.clone(), "DISAGREE".to_string());
                    disagreements.push(format!(
                        "{id}: oracle {got:?} vs the vendored expectation {expected:?}"
                    ));
                }
            }
            Err(e) => {
                per_case.insert(id.clone(), "ERROR".to_string());
                disagreements.push(format!("{id}: oracle could not run: {e}"));
            }
        }
    }

    // Corpus arm.
    let nt = root.join("contracts/contracts.nt");
    let ttl = root.join("contracts/shapes.ttl");
    let mut corpus = serde_json::json!({ "ran": false });
    if nt.is_file() && ttl.is_file() {
        cases += 1;
        match oracle_pairs(&nt, &ttl) {
            Ok(got) => {
                let pv: Vec<Pair> = pv_report
                    .as_ref()
                    .and_then(|p| std::fs::read_to_string(p).ok())
                    .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                    .map(|v| pv_violations(&v))
                    .unwrap_or_default();
                let agree = got == pv;
                corpus = serde_json::json!({
                    "ran": true,
                    "oracle_violations": got.len(),
                    "pv_violations": pv.len(),
                    "agree": agree,
                });
                if !agree {
                    disagreements.push(format!(
                        "corpus: oracle {} violation(s), pv {} — first difference: {:?}",
                        got.len(),
                        pv.len(),
                        first_difference(&got, &pv)
                    ));
                }
            }
            Err(e) => {
                corpus = serde_json::json!({ "ran": false, "error": e });
                disagreements.push(format!("corpus: oracle could not run: {e}"));
            }
        }
    }

    let out = serde_json::json!({
        "schema": "ont-oracle-differential/v1",
        "oracle": "shacl 0.3.21 (pinned, out of gate — ONT-0, R-13)",
        "cases": cases,
        "disagreements": disagreements.len(),
        "detail": disagreements,
        "w3c": per_case,
        "corpus": corpus,
    });
    let path = oracle_dir.join("differential.json");
    std::fs::write(&path, format!("{}\n", serde_json::to_string_pretty(&out).expect("json")))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    println!(
        "ont-oracle: {} case(s), {} disagreement(s) → {}",
        cases,
        disagreements.len(),
        path.display()
    );
    if !disagreements.is_empty() {
        for d in &disagreements {
            eprintln!("  {d}");
        }
        std::process::exit(1);
    }
}

/// pv's own violations from `pv lint … --gate shapes --format json`. Each PV-ONT-011 finding's message is
/// `<focus> [(rung …)] violates shape `<id>` [not armed] (<component>): …`, so focus and component are read
/// from it; the short `ont:x` form expands to the IRI the oracle prints. Warnings are violations of UNARMED
/// shapes — the oracle knows no arming, so both arms compare the same set.
fn pv_violations(v: &serde_json::Value) -> Vec<Pair> {
    let mut out = Vec::new();
    for f in v
        .get("findings")
        .and_then(|f| f.as_array())
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let Some(msg) = f.get("message").and_then(|m| m.as_str()) else {
            continue;
        };
        let Some((left, right)) = msg.split_once(" violates shape ") else {
            continue;
        };
        let focus = left.split_whitespace().next().unwrap_or_default();
        let focus = focus
            .strip_prefix("ont:")
            .map_or_else(|| focus.to_string(), |l| format!("https://ont.paiml.dev/v1alpha1/{l}"));
        let Some((before_colon, _)) = right.split_once("): ") else {
            continue;
        };
        let Some((_, component)) = before_colon.rsplit_once('(') else {
            continue;
        };
        out.push((focus, component.to_string()));
    }
    out.sort();
    out
}

fn first_difference(a: &[Pair], b: &[Pair]) -> String {
    for (i, x) in a.iter().enumerate() {
        match b.get(i) {
            Some(y) if y == x => {}
            other => return format!("at {i}: oracle {x:?} vs pv {other:?}"),
        }
    }
    format!("pv has {} extra", b.len().saturating_sub(a.len()))
}
