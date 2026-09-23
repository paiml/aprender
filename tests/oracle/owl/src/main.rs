//! ONT-001 §3.8, row ONT-2c: the OWL oracle, out of the gate path (R-13), in two arms.
//!
//! 1. **Round-trip (horned-owl 3.0.0, pinned; LGPL, oracle only).**
//!    - `roundtrip <ofn> <expected>`: horned-owl re-parses the in-house writer's output, and its axiom set
//!      must EQUAL a hand-written, neutral axiom list (`tests/fixtures/ont/owl/axioms.txt`). Writing
//!      `acyclic` as `TransitiveObjectProperty` + `IrreflexiveObjectProperty` fails here: the mutation this
//!      row names.
//!    - `kinds <ofn>`: every axiom in the live `contracts/ontology.ofn` is one of the kinds under which
//!      told-closure is EL classification. That is the in-tree precondition, confirmed by a parser that is
//!      not ours.
//! 2. **TBox differential (ELK 0.4.3, pinned by sha256; Apache-2.0; needs a JVM).** `elk <repo>` classifies
//!    `contracts/ontology.ofn` with ELK. Its `consistent` and entailed subsumptions must EQUAL the in-tree
//!    told-closure report (`contracts/tbox-report.json`), and equal entailment means equal
//!    `unintended_subsumptions`, since both sides share Σ's intent. A POSITIVE CONTROL runs every time: a
//!    copy of the ontology with `SubClassOf(Contract Symbol)` planted must be entailed by ELK and must make
//!    the differential disagree. If it does not, the oracle is blind and the result is RED.
//!    - Measured: ELK ignores `ObjectPropertyRange` and `SymmetricObjectProperty`
//!      (`[reasoner.indexing.axiomIgnored]`). Any OTHER ignored kind is RED, because the oracle did not
//!      reason over it. Neither ignored kind can entail an atomic subsumption.
//!    - ELK exits 0 on an inconsistent ontology. Inconsistency is read from its log (`Ontology is
//!      inconsistent`) and from `EquivalentClasses(owl:Nothing …)` in the taxonomy, never from its exit code.
//!    - Writes `tests/oracle/tbox-differential.json`.
//!
//! Exit codes, as `pv lint`: 0 agree · 1 disagree / a planted control not seen · 2 `decline:` NOT MEASURED
//! (no JVM, jar unreachable). A decline is RED at the RELEASE gate and is never skipped (cop ruling
//! 2026-09-23). The agreement is required at release only, never per PR.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use horned_owl::io::ofn::reader::read;
use horned_owl::io::ParserConfiguration;
use horned_owl::model::{ClassExpression, Component, ObjectPropertyExpression, RcStr};
use horned_owl::ontology::set::SetOntology;

const BASE: &str = "https://ont.paiml.dev/v1alpha1/";
const ELK_URL: &str =
    "https://repo1.maven.org/maven2/org/semanticweb/elk/elk-distribution/0.4.3/elk-distribution-0.4.3-standalone-executable.zip";
const ELK_ZIP_SHA256: &str = "965ad946eb566ed9db0160a10ea6252606557f5cadd275469647e5908d964d4a";
const ELK_JAR_SHA256: &str = "de1fffafbe0bb19656335b53b4a5a44d8b8feb75e077ad1677ba8a2fb91cac7c";
/// The axiom kinds ELK may ignore without blinding the differential (measured, module doc).
const ELK_MAY_IGNORE: [&str; 2] = ["ObjectPropertyRange", "SymmetricObjectProperty"];
/// Neutral kinds under which told-closure is EL classification (mirrors `ontology::owl::ADMITTED_AXIOM_KINDS`).
const ADMITTED: [&str; 6] = ["Class", "ObjectProperty", "SubClassOf", "Domain", "Range", "Symmetric"];

fn local(iri: &str) -> String {
    iri.strip_prefix(BASE).unwrap_or(iri).to_string()
}

fn ce(c: &ClassExpression<RcStr>) -> String {
    match c {
        ClassExpression::Class(k) => local(&k.0.to_string()),
        other => format!("NONATOMIC[{other:?}]"),
    }
}

fn ope(p: &ObjectPropertyExpression<RcStr>) -> String {
    match p {
        ObjectPropertyExpression::ObjectProperty(o) => local(&o.0.to_string()),
        other => format!("NONATOMIC[{other:?}]"),
    }
}

/// One axiom as a neutral line, the vocabulary of `tests/fixtures/ont/owl/axioms.txt`. `None` for the ontology id.
fn neutral(c: &Component<RcStr>) -> Option<String> {
    Some(match c {
        Component::OntologyID(_) | Component::DocIRI(_) => return None,
        Component::DeclareClass(d) => format!("Class {}", local(&d.0 .0.to_string())),
        Component::DeclareObjectProperty(d) => format!("ObjectProperty {}", local(&d.0 .0.to_string())),
        Component::SubClassOf(s) => format!("SubClassOf {} {}", ce(&s.sub), ce(&s.sup)),
        Component::ObjectPropertyDomain(d) => format!("Domain {} {}", ope(&d.ope), ce(&d.ce)),
        Component::ObjectPropertyRange(r) => format!("Range {} {}", ope(&r.ope), ce(&r.ce)),
        Component::SymmetricObjectProperty(s) => format!("Symmetric {}", ope(&s.0)),
        Component::TransitiveObjectProperty(t) => format!("Transitive {}", ope(&t.0)),
        Component::IrreflexiveObjectProperty(i) => format!("Irreflexive {}", ope(&i.0)),
        other => format!("OTHER {other:?}"),
    })
}

fn parse(path: &Path) -> Result<BTreeSet<String>, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (ont, _): (SetOntology<RcStr>, _) = read(std::io::BufReader::new(f), ParserConfiguration::default())
        .map_err(|e| format!("horned-owl cannot parse {}: {e}", path.display()))?;
    Ok(ont.iter().filter_map(|ac| neutral(&ac.component)).collect())
}

fn roundtrip(ofn: &Path, expected: &Path) -> i32 {
    let got = match parse(ofn) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("reject: {e}");
            return 1;
        }
    };
    let want: BTreeSet<String> = std::fs::read_to_string(expected)
        .unwrap_or_else(|e| panic!("{}: {e}", expected.display()))
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect();
    if want.is_empty() {
        eprintln!("decline: {} holds no axioms: an empty expectation proves nothing", expected.display());
        return 2;
    }
    if got == want {
        println!("roundtrip: {} axioms, equal to {}", got.len(), expected.display());
        return 0;
    }
    for x in got.difference(&want) {
        eprintln!("reject: written but not expected: {x}");
    }
    for x in want.difference(&got) {
        eprintln!("reject: expected but not written: {x}");
    }
    1
}

fn kinds(ofn: &Path) -> i32 {
    match parse(ofn) {
        Err(e) => {
            eprintln!("reject: {e}");
            1
        }
        Ok(axioms) => {
            let bad: Vec<&String> = axioms
                .iter()
                .filter(|a| {
                    let kind = a.split(' ').next().unwrap_or("");
                    !ADMITTED.contains(&kind) || a.contains("NONATOMIC")
                })
                .collect();
            if bad.is_empty() {
                println!("kinds: {} axioms, every one an admitted kind", axioms.len());
                0
            } else {
                for b in bad {
                    eprintln!("reject: outside the told-closure precondition: {b}");
                }
                1
            }
        }
    }
}

// ---- ELK ------------------------------------------------------------------------------------------------

fn sha256(path: &Path) -> Option<String> {
    let out = Command::new("sha256sum").arg(path).output().ok()?;
    String::from_utf8_lossy(&out.stdout).split_whitespace().next().map(String::from)
}

/// The pinned jar, fetched once into the cache and verified by sha256 every time. `Err` is NOT MEASURED.
fn elk_jar() -> Result<PathBuf, String> {
    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .ok_or("no cache dir")?
        .join("ont-oracle/elk-0.4.3");
    std::fs::create_dir_all(&cache).map_err(|e| e.to_string())?;
    let zip = cache.join("elk-distribution-0.4.3-standalone-executable.zip");
    let jar = cache.join("elk-distribution-0.4.3-standalone-executable/elk-standalone.jar");
    if sha256(&zip).as_deref() != Some(ELK_ZIP_SHA256) {
        let ok = Command::new("curl").args(["-sfL", "-o"]).arg(&zip).arg(ELK_URL).status().is_ok_and(|s| s.success());
        if !ok || sha256(&zip).as_deref() != Some(ELK_ZIP_SHA256) {
            return Err(format!("ELK zip unreachable or its sha256 is not {ELK_ZIP_SHA256}"));
        }
        let ok = Command::new("unzip").args(["-qo"]).arg(&zip).arg("-d").arg(&cache).status().is_ok_and(|s| s.success());
        if !ok {
            return Err("unzip failed".into());
        }
    }
    if sha256(&jar).as_deref() != Some(ELK_JAR_SHA256) {
        return Err(format!("ELK jar sha256 is not {ELK_JAR_SHA256}"));
    }
    Ok(jar)
}

struct Classification {
    consistent: bool,
    entailed: BTreeSet<(String, String)>,
    ignored_kinds: BTreeSet<String>,
}

/// Run ELK once: `(log, taxonomy text)`. ELK exits 0 even on an inconsistent ontology, so the caller reads
/// consistency from the log and the taxonomy, never from this status.
fn run_elk(java: &str, jar: &Path, ofn: &Path, work: &Path) -> Result<(String, String), String> {
    let name = ofn.file_name().and_then(|n| n.to_str()).unwrap_or("x");
    let out = work.join(format!("{name}.taxonomy"));
    let run = Command::new(java)
        .arg("-jar")
        .arg(jar)
        .arg("-i")
        .arg(ofn)
        .args(["-c", "-o"])
        .arg(&out)
        .output()
        .map_err(|e| format!("java: {e}"))?;
    let log = format!("{}{}", String::from_utf8_lossy(&run.stdout), String::from_utf8_lossy(&run.stderr));
    if !run.status.success() {
        return Err(format!("ELK exited {:?}: {}", run.status.code(), log.lines().last().unwrap_or("")));
    }
    let tax = std::fs::read_to_string(&out).map_err(|e| format!("no taxonomy written: {e}"))?;
    Ok((log, tax))
}

/// The axiom kinds ELK logged as `axiomIgnored`.
fn ignored_kinds(log: &str) -> BTreeSet<String> {
    log.lines()
        .filter_map(|l| l.split("ELK does not support ").nth(1))
        .filter_map(|r| r.split('.').next())
        .map(String::from)
        .collect()
}

/// The IRIs inside one `Keyword(<a> <b> …)` taxonomy line, or `None` for another keyword.
fn taxonomy_args(line: &str, keyword: &str) -> Option<Vec<String>> {
    let body = line.strip_prefix(keyword)?.strip_prefix('(')?.strip_suffix(')')?;
    Some(body.split_whitespace().map(|t| t.trim_matches(|c| c == '<' || c == '>').to_string()).collect())
}

/// ELK's taxonomy as direct edges (both directions for an equivalence), and whether any class is `≡ ⊥`.
fn parse_taxonomy(tax: &str) -> (BTreeSet<(String, String)>, bool) {
    let mut direct = BTreeSet::new();
    let mut unsatisfiable = false;
    for line in tax.lines() {
        if let Some(v) = taxonomy_args(line, "SubClassOf").filter(|v| v.len() == 2) {
            direct.insert((local(&v[0]), local(&v[1])));
        } else if let Some(v) = taxonomy_args(line, "EquivalentClasses") {
            unsatisfiable |= v.iter().any(|x| x.ends_with("owl#Nothing"));
            for a in &v {
                direct.extend(v.iter().filter(|b| *b != a).map(|b| (local(a), local(b))));
            }
        }
    }
    (direct, unsatisfiable)
}

fn is_top_or_bottom(x: &str) -> bool {
    x.ends_with("owl#Thing") || x.ends_with("owl#Nothing")
}

/// Every strict subsumption between named classes the direct edges entail (transitive closure).
fn closure(direct: &BTreeSet<(String, String)>) -> BTreeSet<(String, String)> {
    let mut entailed = BTreeSet::new();
    for a in direct.iter().map(|(a, _)| a).filter(|a| !is_top_or_bottom(a)) {
        let mut stack = vec![a.clone()];
        let mut seen = BTreeSet::new();
        while let Some(n) = stack.pop() {
            let next: Vec<String> = direct.iter().filter(|(x, _)| *x == n).map(|(_, y)| y.clone()).collect();
            stack.extend(next.into_iter().filter(|y| seen.insert(y.clone())));
        }
        entailed.extend(seen.into_iter().filter(|y| y != a && !is_top_or_bottom(y)).map(|y| (a.clone(), y)));
    }
    entailed
}

fn classify(java: &str, jar: &Path, ofn: &Path, work: &Path) -> Result<Classification, String> {
    let (log, tax) = run_elk(java, jar, ofn, work)?;
    let (direct, unsatisfiable) = parse_taxonomy(&tax);
    Ok(Classification {
        consistent: !log.contains("Ontology is inconsistent") && !unsatisfiable,
        entailed: closure(&direct),
        ignored_kinds: ignored_kinds(&log),
    })
}

fn pairs(v: &serde_json::Value, key: &str) -> BTreeSet<(String, String)> {
    v[key]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| Some((p[0].as_str()?.to_string(), p[1].as_str()?.to_string())))
        .collect()
}

fn elk(repo: &Path) -> i32 {
    let ofn = repo.join("contracts/ontology.ofn");
    let report_path = repo.join("contracts/tbox-report.json");
    let out_path = repo.join("tests/oracle/tbox-differential.json");
    let mut doc = serde_json::json!({
        "schema": "ont-tbox-differential/v1",
        "oracle": format!("ELK 0.4.3 (jar sha256 {ELK_JAR_SHA256}; out of gate, release only)"),
        "measured": false,
    });
    let decline = |doc: &mut serde_json::Value, why: String| -> i32 {
        doc["decline"] = serde_json::Value::String(why.clone());
        let _ = std::fs::write(&out_path, format!("{}\n", serde_json::to_string_pretty(doc).unwrap_or_default()));
        eprintln!("decline: NOT MEASURED: {why}");
        2
    };
    let java = std::env::var("ONT_ORACLE_JAVA").unwrap_or_else(|_| "java".into());
    match Command::new(&java).arg("-version").output() {
        Ok(o) if o.status.success() => {
            let v = String::from_utf8_lossy(&o.stderr).lines().next().unwrap_or("").to_string();
            doc["jvm"] = serde_json::Value::String(v);
        }
        _ => return decline(&mut doc, format!("no JVM (`{java} -version` failed)")),
    }
    let jar = match elk_jar() {
        Ok(j) => j,
        Err(e) => return decline(&mut doc, e),
    };
    let report: serde_json::Value = match std::fs::read_to_string(&report_path).ok().and_then(|t| serde_json::from_str(&t).ok()) {
        Some(r) => r,
        None => return decline(&mut doc, format!("{} unreadable", report_path.display())),
    };
    let work = std::env::temp_dir().join(format!("ont-oracle-elk-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&work);
    let live = match classify(&java, &jar, &ofn, &work) {
        Ok(c) => c,
        Err(e) => return decline(&mut doc, e),
    };
    // POSITIVE CONTROL: a planted unintended subsumption must be entailed, and must break agreement.
    let planted = work.join("planted.ofn");
    let text = std::fs::read_to_string(&ofn).unwrap_or_default();
    let plant = format!("SubClassOf(<{BASE}Contract> <{BASE}Symbol>)\n)\n");
    let _ = std::fs::write(&planted, text.trim_end().trim_end_matches(')').to_string() + &plant);
    let control = match classify(&java, &jar, &planted, &work) {
        Ok(c) => c,
        Err(e) => return decline(&mut doc, format!("positive control did not run: {e}")),
    };
    let _ = std::fs::remove_dir_all(&work);

    let report_consistent = report["consistent"].as_bool();
    let report_entailed = pairs(&report, "entailed_subsumptions");
    let blind: Vec<&String> = live.ignored_kinds.iter().filter(|k| !ELK_MAY_IGNORE.contains(&k.as_str())).collect();
    let control_seen = control.entailed.contains(&("Contract".to_string(), "Symbol".to_string()))
        && control.entailed != report_entailed;
    let agree = report_consistent == Some(live.consistent) && report_entailed == live.entailed;
    doc["measured"] = true.into();
    doc["elk"] = serde_json::json!({
        "consistent": live.consistent,
        "entailed_subsumptions": live.entailed,
        "ignored_axiom_kinds": live.ignored_kinds,
    });
    doc["report"] = serde_json::json!({"consistent": report_consistent, "entailed_subsumptions": report_entailed});
    doc["blind_to"] = serde_json::json!(blind);
    doc["positive_control"] = serde_json::json!({
        "planted": "SubClassOf(Contract Symbol)",
        "elk_entailed_it": control.entailed.contains(&("Contract".to_string(), "Symbol".to_string())),
        "differential_went_red": control.entailed != report_entailed,
    });
    doc["agree"] = agree.into();
    let _ = std::fs::write(&out_path, format!("{}\n", serde_json::to_string_pretty(&doc).unwrap_or_default()));
    let mut rc = 0;
    if !blind.is_empty() {
        eprintln!("reject: ELK ignored axiom kind(s) outside {ELK_MAY_IGNORE:?}: {blind:?} — the oracle is blind there");
        rc = 1;
    }
    if !control_seen {
        eprintln!("reject: POSITIVE CONTROL not seen: the planted SubClassOf(Contract Symbol) did not turn the differential RED");
        rc = 1;
    }
    if !agree {
        eprintln!(
            "reject: ELK disagrees with tbox-report.json: consistent {} vs {:?}, entailed {:?} vs {:?}",
            live.consistent, report_consistent, live.entailed, report_entailed
        );
        rc = 1;
    }
    if rc == 0 {
        println!(
            "elk: agree — consistent={}, {} entailed subsumptions; positive control RED as required; ignored {:?}",
            live.consistent,
            live.entailed.len(),
            live.ignored_kinds
        );
    }
    rc
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rc = match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["roundtrip", ofn, expected] => roundtrip(Path::new(ofn), Path::new(expected)),
        ["kinds", ofn] => kinds(Path::new(ofn)),
        ["elk", repo] => elk(Path::new(repo)),
        _ => {
            eprintln!("usage: owl-oracle roundtrip <ofn> <expected-axioms.txt> | kinds <ofn> | elk <repo-root>");
            2
        }
    };
    std::process::exit(rc);
}

#[cfg(test)]
mod tests {
    //! The taxonomy shapes below are ELK 0.4.3's own output, measured on lambda 2026-09-23.
    use super::*;

    fn p(a: &str, b: &str) -> (String, String) {
        (a.to_string(), b.to_string())
    }

    #[test]
    fn a_chain_is_closed_transitively() {
        let tax = format!(
            "Ontology(\nSubClassOf(<{BASE}A> <{BASE}B>)\nSubClassOf(<{BASE}B> <{BASE}C>)\n)\n"
        );
        let (d, unsat) = parse_taxonomy(&tax);
        assert!(!unsat);
        assert_eq!(closure(&d), [p("A", "B"), p("A", "C"), p("B", "C")].into_iter().collect());
    }

    #[test]
    fn an_equivalence_entails_both_directions() {
        let tax = format!("Ontology(\nEquivalentClasses(<{BASE}A> <{BASE}B>)\n)\n");
        let (d, _) = parse_taxonomy(&tax);
        assert_eq!(closure(&d), [p("A", "B"), p("B", "A")].into_iter().collect());
    }

    #[test]
    fn a_class_equivalent_to_nothing_is_unsatisfiable_and_top_bottom_are_not_pairs() {
        let tax = format!(
            "Ontology(\nEquivalentClasses(<http://www.w3.org/2002/07/owl#Nothing> <{BASE}A>)\n)\n"
        );
        let (d, unsat) = parse_taxonomy(&tax);
        assert!(unsat, "A ≡ ⊥ must read as inconsistent");
        assert!(closure(&d).is_empty(), "owl:Nothing is never reported as a Σ subsumption");
    }

    #[test]
    fn ignored_kinds_are_read_from_the_log() {
        let log = "90 [main] WARN x - [reasoner.indexing.axiomIgnored]ELK does not support ObjectPropertyRange. Axiom ignored:\n";
        assert_eq!(ignored_kinds(log), ["ObjectPropertyRange".to_string()].into_iter().collect());
    }
}
