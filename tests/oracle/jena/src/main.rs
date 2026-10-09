//! CRUX category R against Apache Jena 5.6.0 (contract `contracts/crux-shacl-jena-v1.yaml`,
//! spec §14). Out of every gate: `make oracle-jena` runs it on demand; no PR, merge-queue or release job does.
//!
//! - `jena-oracle --self-test` runs the planted controls, one per FALSIFY-CRUXSHACL row this harness implements, and
//!   prints `self-test: N/N ok`. The controls act on the comparator's inputs, never on pv, so they prove the harness
//!   whatever pv does. Six of them need the JVM.
//! - `jena-oracle run <repo> <pv-shapes.json> --pv <pv>` checks the pins, runs every control, measures the cells and
//!   writes `tests/oracle/jena/receipt.json`. R-INPUT and the R-RETURN round trip run `<pv> ontology read` (S3).
//!
//! Exit codes follow the OWL oracle: 0 GREEN, 1 RED, 2 NOT MEASURED, by the contract's `verdict` equation. Until S4
//! lands, the cells whose pv side needs it are NOT MEASURED, so a run whose every Jena-side check is green
//! exits 2, never 0. The JVM is `ONT_ORACLE_JAVA`, else `java` on PATH.

mod jena;
mod nt;
mod pv;
mod report;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use jena::Jena;
use nt::{Term, Triple};
use report::{Graph, Report};

const RECEIPT_SCHEMA: &str = "crux-shacl-jena-receipt/1";
/// sha256 of `std/shacl-shacl.ttl` inside the pinned `jena-shacl-5.6.0.jar` (subject S-SHSH), measured 2026-10-09.
const SHSH_SHA256: &str = "af9e8c2968aff88aa8895fca951e2cd14891b27bd2ef90a10ad2c2452332a7e1";
/// The base riot resolved W3C minCount-001 against when `controls/minCount-001.nt` was written.
const MINCOUNT_BASE: &str = "file:///w3c/property/minCount-001.ttl";
const MF: &str = "http://www.w3.org/2001/sw/DataAccess/tests/test-manifest#";
const SHT: &str = "http://www.w3.org/ns/shacl-test#";
const EX: &str = "http://ex.org/";

const BASE_NT: &str = include_str!("../controls/base.nt");
const SSK_NT: &str = include_str!("../controls/source-shape-key.nt");
const MINCOUNT_NT: &str = include_str!("../controls/minCount-001.nt");
const MINCOUNT_TTL: &str = include_str!("../../w3c/property/minCount-001.ttl");
const LINE7_TTL: &str = include_str!("../controls/syntax-error-line-7.ttl");
const ESCAPES_TTL: &str = include_str!("../controls/escapes-langtags.ttl");
const SHSH_OK_TTL: &str = include_str!("../controls/shsh-ok.ttl");

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum V {
    Green,
    Red,
    Nm,
}

impl V {
    fn name(self) -> &'static str {
        match self {
            V::Green => "GREEN",
            V::Red => "RED",
            V::Nm => "NOT MEASURED",
        }
    }
}

/// One verdict line. Every RED line names one cell, one case and one comparator (F-CRUXSHACL-QA-001).
struct Line {
    cell: &'static str,
    case: String,
    comparator: &'static str,
    v: V,
    detail: String,
}

impl Line {
    fn new(cell: &'static str, case: &str, comparator: &'static str, v: V, detail: String) -> Line {
        Line {
            cell,
            case: case.to_string(),
            comparator,
            v,
            detail,
        }
    }

    fn print(&self) {
        println!(
            "{} {} {} [{}]: {}",
            self.v.name(),
            self.cell,
            self.case,
            self.comparator,
            self.detail
        );
    }
}

/// The inputs of the contract's `verdict` equation, plus `nm_cell`: a run that would exit 0 while a cell is still
/// NOT MEASURED (its pv side waits for S4) exits 2.
#[derive(Clone, Copy, Default)]
struct Facts {
    pins_ok: bool,
    complete: bool,
    self_compare: bool,
    red_w3c: bool,
    jvm: bool,
    red_jena: bool,
    control_unseen: bool,
    nm_cell: bool,
}

/// `exit = 2 if ¬pins_ok ∨ ¬complete ∨ self_compare; else 1 if red_w3c; else 2 if ¬jvm; else 1 if red_jena ∨
/// control_unseen; else 2 if nm_cell; else 0`.
fn verdict(f: Facts) -> u8 {
    if !f.pins_ok || !f.complete || f.self_compare {
        2
    } else if f.red_w3c {
        1
    } else if !f.jvm {
        2
    } else if f.red_jena || f.control_unseen {
        1
    } else if f.nm_cell {
        2
    } else {
        0
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Pieces the controls and the run share.
// ---------------------------------------------------------------------------------------------------------------

fn parse_fixture(name: &str, text: &str) -> Vec<Triple> {
    nt::parse(text).unwrap_or_else(|e| panic!("embedded fixture {name}: {e}"))
}

/// Every blank-node label renamed and the triple order reversed.
fn relabel(t: &[Triple]) -> Vec<Triple> {
    let r = |x: &Term| match x {
        Term::Blank(l) => Term::Blank(format!("z{}", l.chars().rev().collect::<String>())),
        other => other.clone(),
    };
    t.iter().rev().map(|[s, p, o]| [r(s), r(p), r(o)]).collect()
}

fn nt_text(t: &[Triple]) -> String {
    t.iter()
        .map(|[s, p, o]| format!("{} {} {} .\n", s.nt(), p.nt(), o.nt()))
        .collect()
}

/// 0 isomorphic, 1 different, 2 inconclusive (a split class; the run confirms with `jena.rdfcompare`).
fn iso_code(iso: &nt::Iso) -> u8 {
    if iso.equal {
        0
    } else if iso.inconclusive {
        2
    } else {
        1
    }
}

fn report_of(t: &[Triple]) -> Result<Report, String> {
    let g = Graph::new(t);
    let node = report::find_report(&g)?;
    Ok(report::report_at(&g, &node))
}

/// The `sh:message` each shape declares, keyed like a result's source shape.
fn declared_messages(t: &[Triple]) -> BTreeMap<String, Vec<String>> {
    let g = Graph::new(t);
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for [s, p, o] in t {
        if *p != Term::Iri(format!("{}message", report::SH)) {
            continue;
        }
        let Term::Lit { lex, .. } = o else { continue };
        let path = g
            .obj(s, &Term::Iri(format!("{}path", report::SH)))
            .map(|x| g.render(x, 0))
            .unwrap_or_default();
        let (key, _) = report::source_shape_key(&g, s, &path);
        out.entry(key).or_default().push(lex.clone());
    }
    out
}

/// R-VALIDATE on one S-W3C case for the side under test. W3C's expected report decides (011/012). A refusal, such
/// as pv's exit 3 on a construct outside its subset, is RED, never a skip (010). Returns the line and, when Jena is
/// the one that differs from W3C, the `jena≠w3c` note for the receipt.
fn validate_line(
    case: &str,
    under_test: Result<&Report, &str>,
    jena: Option<&Report>,
    w3c: &Report,
) -> (Line, Option<String>) {
    let ut = match under_test {
        Ok(r) => r,
        Err(why) => {
            return (
                Line::new("R-VALIDATE", case, "w3c", V::Red, format!("refused: {why}")),
                None,
            )
        }
    };
    let res = match jena {
        Some(j) => report::arbitrate(case, ut, j, w3c),
        None => {
            let d = report::diff(ut, w3c);
            if d.equal() {
                Ok(None)
            } else {
                Err(format!(
                    "{case}: under test ≠ w3c: {}",
                    d.lines("under-test", "w3c").join("; ")
                ))
            }
        }
    };
    match res {
        Ok(note) => (
            Line::new(
                "R-VALIDATE",
                case,
                "w3c",
                V::Green,
                format!("{} results as W3C expects", ut.results.len()),
            ),
            note,
        ),
        Err(e) => (Line::new("R-VALIDATE", case, "w3c", V::Red, e), None),
    }
}

/// S-SHSH: the shape graph conforms to the SHACL-SHACL graph.
fn shsh_line(case: &str, r: &Report) -> Line {
    if r.conforms == Some(true) && r.results.is_empty() {
        return Line::new(
            "S-SHSH",
            case,
            "shacl-shacl",
            V::Green,
            "conforms to SHACL-SHACL".into(),
        );
    }
    let first: Vec<String> = r
        .results
        .iter()
        .take(5)
        .map(|x| format!("{} {}", x.key.focus, x.key.component))
        .collect();
    Line::new(
        "S-SHSH",
        case,
        "shacl-shacl",
        V::Red,
        format!(
            "conforms {:?}, {} results: {}",
            r.conforms,
            r.results.len(),
            first.join("; ")
        ),
    )
}

fn same_file(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (std::fs::metadata(a), std::fs::metadata(b)) {
        (Ok(x), Ok(y)) => x.dev() == y.dev() && x.ino() == y.ino(),
        _ => false,
    }
}

/// FALSIFY-CRUXSHACL-007: `Some(why)` when the two sides are one file (same path or inode, a symlink included) or
/// the written side was not created by this run.
fn self_compare(read: &Path, written: &Path, started: SystemTime) -> Option<String> {
    if same_file(read, written) {
        return Some(format!(
            "{} and {} are one file",
            read.display(),
            written.display()
        ));
    }
    match std::fs::symlink_metadata(written).and_then(|m| m.modified()) {
        Ok(t) if t >= started => None,
        Ok(_) => Some(format!("{} predates this run", written.display())),
        Err(e) => Some(format!("{}: {e}", written.display())),
    }
}

/// The pinned files of a suite: each `sha256  path` line verified, and every `*.ttl` under `dir` listed (an
/// unlisted file is an unpinned input). Returns the listed paths.
fn check_manifest(root: &Path, manifest: &str, dir: &str) -> Result<Vec<String>, String> {
    let mut listed = Vec::new();
    for l in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let (sha, rel) = l
            .split_once("  ")
            .ok_or_else(|| format!("bad manifest line {l:?}"))?;
        jena::verify_pin(&root.join(rel), sha)?;
        listed.push(rel.to_string());
    }
    let mut on_disk = Vec::new();
    let mut stack = vec![root.join(dir)];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).map_err(|e| format!("{}: {e}", d.display()))? {
            let p = e.map_err(|e| format!("{}: {e}", d.display()))?.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "ttl") {
                on_disk.push(p.strip_prefix(root).unwrap_or(&p).display().to_string());
            }
        }
    }
    let l: BTreeSet<&String> = listed.iter().collect();
    if let Some(extra) = on_disk.iter().find(|p| !l.contains(p)) {
        return Err(format!("{extra} is not pinned in the manifest"));
    }
    if listed.is_empty() {
        return Err("the manifest pins no file".into());
    }
    Ok(listed)
}

/// FALSIFY-CRUXSHACL-024: every listed case produced a verdict, and there is at least one.
fn complete(listed: &[String], seen: &BTreeSet<String>) -> Result<(), String> {
    if listed.is_empty() {
        return Err("the subject has no case".into());
    }
    match listed.iter().find(|c| !seen.contains(*c)) {
        Some(c) => Err(format!("case {c} produced no verdict")),
        None => Ok(()),
    }
}

/// `sh:MinCountConstraintComponent` → `minCount`, the name pv's report uses.
fn short_component(iri: &str) -> String {
    let iri = iri.trim_start_matches('<').trim_end_matches('>');
    let local = iri.rsplit(['#', '/']).next().unwrap_or(iri);
    let base = local.strip_suffix("ConstraintComponent").unwrap_or(local);
    let mut c = base.chars();
    match c.next() {
        Some(f) => f.to_lowercase().chain(c).collect(),
        None => base.to_string(),
    }
}

type Pair = (String, String);

/// pv's violations from `pv lint contracts --gate shapes --format json`, as `(focus IRI, component)`. Each
/// PV-ONT-011 message is `<focus> [(rung …)] violates shape `<id>` [not armed] (<component>): …`. `None` when the
/// JSON has no `findings` array: pv did not run, which is an incomplete subject, never zero violations.
fn pv_violations(v: &serde_json::Value) -> Option<Vec<Pair>> {
    let mut out = Vec::new();
    for f in v.get("findings")?.as_array()? {
        let Some(msg) = f.get("message").and_then(|m| m.as_str()) else {
            continue;
        };
        let Some((left, right)) = msg.split_once(" violates shape ") else {
            continue;
        };
        let focus = left.split_whitespace().next().unwrap_or_default();
        let focus = focus.strip_prefix("ont:").map_or_else(
            || focus.to_string(),
            |l| format!("https://ont.paiml.dev/v1alpha1/{l}"),
        );
        let Some((before, _)) = right.split_once("): ") else {
            continue;
        };
        let Some((_, component)) = before.rsplit_once('(') else {
            continue;
        };
        out.push((focus, component.to_string()));
    }
    out.sort();
    Some(out)
}

fn jena_pairs(r: &Report) -> Vec<Pair> {
    let mut out: Vec<Pair> = r
        .results
        .iter()
        .map(|x| {
            (
                x.key
                    .focus
                    .trim_start_matches('<')
                    .trim_end_matches('>')
                    .to_string(),
                short_component(&x.key.component),
            )
        })
        .collect();
    out.sort();
    out
}

/// Multiset difference, as `(pair, jena count − pv count)` for every pair whose counts differ.
fn pair_diff(a: &[Pair], b: &[Pair]) -> Vec<(Pair, isize)> {
    let mut m: BTreeMap<&Pair, isize> = BTreeMap::new();
    for x in a {
        *m.entry(x).or_insert(0) += 1;
    }
    for x in b {
        *m.entry(x).or_insert(0) -= 1;
    }
    m.into_iter()
        .filter(|&(_, n)| n != 0)
        .map(|(p, n)| (p.clone(), n))
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------
// Controls.
// ---------------------------------------------------------------------------------------------------------------

/// A planted control: the exit the harness must give on it, and the exit it gave (`None`: needs the JVM, none ran).
struct Ctl {
    id: &'static str,
    name: &'static str,
    want: u8,
    got: Option<u8>,
    detail: String,
}

/// The harness went RED but its line did not name what the control changed, or a precondition of the control
/// failed. Never equal to a `want`.
const UNNAMED: u8 = 3;

fn red_named(red: bool, named: bool) -> u8 {
    match (red, named) {
        (false, _) => 0,
        (true, true) => 1,
        (true, false) => UNNAMED,
    }
}

fn ctl(id: &'static str, name: &'static str, want: u8, got: u8, detail: String) -> Ctl {
    Ctl {
        id,
        name,
        want,
        got: Some(got),
        detail,
    }
}

/// The FALSIFY-CRUXSHACL rows whose control needs the JVM, in the order `jvm_controls` returns them.
const JVM_CONTROLS: [(&str, &str, u8); 6] = [
    ("FALSIFY-CRUXSHACL-004", "input/syntax-error-line-7", 0),
    ("FALSIFY-CRUXSHACL-005", "return/escapes-langtags", 0),
    ("FALSIFY-CRUXSHACL-006", "return/drop-one-triple", 1),
    ("FALSIFY-CRUXSHACL-008", "validate/minCount-001", 0),
    ("FALSIFY-CRUXSHACL-009", "validate/minCount-001-min2", 1),
    ("FALSIFY-CRUXSHACL-026", "shsh/minCount-string", 1),
];

fn write(work: &Path, name: &str, text: &str) -> PathBuf {
    let p = work.join(name);
    std::fs::write(&p, text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    p
}

/// A case or file name as a file-name fragment.
fn tag(s: &str) -> String {
    s.replace(['/', '#', ' '], "-")
}

/// `N triples`, or `N triples (D distinct)` when the file states a triple more than once: a graph is a set, so
/// the reader that keeps one copy and the one that streams both agree.
fn counted(t: &[Triple]) -> String {
    let distinct = t.iter().collect::<std::collections::BTreeSet<_>>().len();
    if distinct == t.len() {
        format!("{} triples", t.len())
    } else {
        format!("{} triples ({distinct} distinct)", t.len())
    }
}

/// Are `a` (riot's) and `b` (pv's side) one graph? The harness's comparator; when it is inconclusive (a split
/// class of blank nodes), `jena.rdfcompare` on both written out decides.
fn same_graph(j: &Jena, work: &Path, tag: &str, a: &[Triple], b: &[Triple]) -> (V, String) {
    let iso = nt::compare(a, b);
    let counts = format!("riot {}, pv {}", counted(a), counted(b));
    if iso.equal {
        return (V::Green, format!("{counts}, isomorphic"));
    }
    let diff: Vec<String> = iso
        .only_a
        .iter()
        .map(|l| format!("riot only {l}"))
        .chain(iso.only_b.iter().map(|l| format!("pv only {l}")))
        .take(3)
        .collect();
    if !iso.inconclusive {
        return (V::Red, format!("{counts}; {}", diff.join("; ")));
    }
    let pa = write(work, &format!("{tag}-riot.nt"), &nt_text(a));
    let pb = write(work, &format!("{tag}-pv.nt"), &nt_text(b));
    match j.rdfcompare(&pa, &pb) {
        Ok(true) => (V::Green, format!("{counts}, isomorphic by rdfcompare")),
        Ok(false) => (
            V::Red,
            format!("{counts}; rdfcompare: not isomorphic; {}", diff.join("; ")),
        ),
        Err(e) => (V::Nm, format!("{counts}; refinement inconclusive and {e}")),
    }
}

fn with_result(r: &Report, i: usize, f: impl Fn(&mut report::Res)) -> Report {
    let mut results = r.results.clone();
    f(&mut results[i]);
    Report {
        conforms: r.conforms,
        results,
    }
}

fn pure_controls(work: &Path) -> Vec<Ctl> {
    let mut out = input_controls();
    out.extend(self_compare_controls(work));
    out.extend(validate_controls());
    out.extend(feedback_controls());
    out.extend(harness_controls(work));
    out
}

/// 001–003: the N-Triples comparator.
fn input_controls() -> Vec<Ctl> {
    let mut out = Vec::new();
    let base = parse_fixture("base.nt", BASE_NT);

    // 001: relabelled blank nodes (two symmetric pairs and two 2-cycles force individualisation) compare equal.
    let iso = nt::compare(&base, &relabel(&base));
    out.push(ctl(
        "FALSIFY-CRUXSHACL-001",
        "input/bnode-relabel",
        0,
        iso_code(&iso),
        format!("{iso:?}").chars().take(200).collect(),
    ));

    // 002: one triple dropped. A ground triple must be named exactly; a blank-to-blank one must still be RED.
    let g = base
        .iter()
        .position(|t| !t[0].is_blank() && !t[2].is_blank())
        .expect("base.nt has a ground triple");
    let mut b = base.clone();
    let gone = b.remove(g);
    let iso = nt::compare(&base, &b);
    let named = iso.only_a == [nt_text(&[gone]).trim_end().to_string()] && iso.only_b.is_empty();
    let k = base
        .iter()
        .position(|t| t[0].is_blank() && t[2].is_blank())
        .expect("base.nt has a blank-blank triple");
    let mut b2 = base.clone();
    b2.remove(k);
    let iso2 = nt::compare(&base, &b2);
    let got = if iso_code(&iso2) == 1 {
        red_named(!iso.equal, named)
    } else {
        UNNAMED
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-002",
        "input/drop-one-triple",
        1,
        got,
        format!("only_a {:?}; blank drop {}", iso.only_a, iso_code(&iso2)),
    ));

    // 003: "01"^^xsd:integer → "1"^^xsd:integer. Value-equal literals are different terms.
    let text = BASE_NT.replacen("\"01\"^^", "\"1\"^^", 1);
    let iso = nt::compare(&base, &parse_fixture("base.nt (003)", &text));
    let named = iso.only_a.iter().any(|l| l.contains("\"01\"^^"))
        && iso.only_b.iter().any(|l| l.contains("\"1\"^^"));
    let got = if text == BASE_NT {
        UNNAMED
    } else {
        red_named(!iso.equal, named)
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-003",
        "input/literal-lexical-form",
        1,
        got,
        format!("{:?} vs {:?}", iso.only_a, iso.only_b),
    ));
    out
}

/// 007: self-comparison.
fn self_compare_controls(work: &Path) -> Vec<Ctl> {
    let mut out = Vec::new();
    // 007: one file twice, a symlink to it, and a file older than the run are self-comparisons; a fresh copy is not.
    let started = SystemTime::now() - Duration::from_secs(2);
    let x = write(work, "self-x.nt", BASE_NT);
    let copy = write(work, "self-copy.nt", BASE_NT);
    let link = work.join("self-link.nt");
    let linked = std::os::unix::fs::symlink(&x, &link).is_ok();
    let stale = write(work, "self-stale.nt", BASE_NT);
    let aged = std::fs::File::options()
        .write(true)
        .open(&stale)
        .and_then(|f| f.set_modified(UNIX_EPOCH + Duration::from_secs(1)))
        .is_ok();
    let caught = [
        self_compare(&x, &x, started),
        self_compare(&x, &link, started),
        self_compare(&x, &stale, started),
    ];
    let got = if !linked || !aged || self_compare(&x, &copy, started).is_some() {
        UNNAMED
    } else if caught.iter().all(Option::is_some) {
        2
    } else {
        0
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-007",
        "return/self-compare",
        2,
        got,
        format!("{caught:?}"),
    ));
    out
}

/// 010–012: arbitration between the side under test, Jena and W3C.
fn validate_controls() -> Vec<Ctl> {
    let mut out = Vec::new();
    // The S-W3C minCount-001 expectation, as riot read it (008 checks the fixture still matches the case file).
    let w3c = report_of(&parse_fixture("minCount-001.nt", MINCOUNT_NT))
        .expect("minCount-001.nt holds W3C's expected report");
    let wrong = with_result(&w3c, 0, |r| {
        r.key.severity = format!("<{}Warning>", report::SH)
    });

    // 010: a refusal is RED and names the case.
    let (l, _) = validate_line(
        "property/datatype-003",
        Err("pv exit 3: sh:or is outside the subset"),
        None,
        &w3c,
    );
    out.push(ctl(
        "FALSIFY-CRUXSHACL-010",
        "validate/datatype-003-refused",
        1,
        red_named(l.v == V::Red, l.case == "property/datatype-003"),
        l.detail,
    ));

    // 011: the side under test and Jena agree with each other and not with W3C: RED.
    let (l, _) = validate_line("property/minCount-001", Ok(&wrong), Some(&wrong), &w3c);
    out.push(ctl(
        "FALSIFY-CRUXSHACL-011",
        "selftest/arbitrate-agree-wrong",
        1,
        red_named(l.v == V::Red, l.detail.contains("severity")),
        l.detail,
    ));

    // 012: the side under test equals W3C and Jena differs: GREEN, with exactly one jena≠w3c note.
    let (l, note) = validate_line("property/minCount-001", Ok(&w3c), Some(&wrong), &w3c);
    let noted = note
        .as_deref()
        .is_some_and(|n| n.starts_with("jena≠w3c property/minCount-001"));
    let got = match (l.v, noted) {
        (V::Green, true) => 0,
        (V::Green, false) => UNNAMED,
        _ => 1,
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-012",
        "selftest/arbitrate-jena-wrong",
        0,
        got,
        format!("{} {note:?}", l.v.name()),
    ));
    out
}

/// 013–021: the report comparator.
fn feedback_controls() -> Vec<Ctl> {
    let mut out = Vec::new();
    // 013: blank-node property shapes are keyed by (parent IRI, sh:path), so relabelled reports agree.
    let ssk = parse_fixture("source-shape-key.nt", SSK_NT);
    let r1 = report_of(&ssk).expect("source-shape-key.nt holds a report");
    let r2 = report_of(&relabel(&ssk)).expect("relabelled report");
    let want_key = format!("<{EX}PersonShape> / <{EX}firstName>");
    let keyed = r1.results.iter().any(|r| r.key.source_shape == want_key)
        && r1.results.iter().all(|r| !r.anonymous);
    let d = report::diff(&r1, &r2);
    let got = match (keyed, d.equal()) {
        (true, true) => 0,
        (true, false) => 1,
        (false, _) => UNNAMED,
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-013",
        "feedback/source-shape-key",
        0,
        got,
        format!(
            "keys {:?}",
            r1.results
                .iter()
                .map(|r| &r.key.source_shape)
                .collect::<Vec<_>>()
        ),
    ));

    // 014–019: one tuple field changed in one result: RED, naming that field.
    let with_value = r1
        .results
        .iter()
        .position(|r| !r.key.value.is_empty())
        .expect("a result with sh:value");
    let mutants: [(&'static str, &'static str, usize, usize, String); 6] = [
        (
            "FALSIFY-CRUXSHACL-014",
            "feedback/mutant-focus",
            0,
            0,
            format!("<{EX}Alice>"),
        ),
        (
            "FALSIFY-CRUXSHACL-015",
            "feedback/mutant-path",
            0,
            1,
            format!("<{EX}lastName>"),
        ),
        (
            "FALSIFY-CRUXSHACL-016",
            "feedback/mutant-value",
            with_value,
            2,
            "\"y\"".into(),
        ),
        (
            "FALSIFY-CRUXSHACL-017",
            "feedback/mutant-source-shape",
            1 - with_value.min(1),
            3,
            r1.results[with_value].key.source_shape.clone(),
        ),
        (
            "FALSIFY-CRUXSHACL-018",
            "feedback/mutant-component",
            0,
            4,
            format!("<{}MaxCountConstraintComponent>", report::SH),
        ),
        (
            "FALSIFY-CRUXSHACL-019",
            "feedback/mutant-severity",
            0,
            5,
            format!("<{}Warning>", report::SH),
        ),
    ];
    for (id, name, i, field, val) in mutants {
        let m = with_result(&r1, i, |r| *r.key.field_mut(field) = val.clone());
        let changed = m.results[i].key != r1.results[i].key;
        let lines = report::diff(&m, &r1).lines("under-test", "jena").join("; ");
        let got = if changed {
            red_named(
                !report::diff(&m, &r1).equal(),
                lines.contains(report::FIELDS[field]),
            )
        } else {
            UNNAMED
        };
        out.push(ctl(id, name, 1, got, lines));
    }

    // 020: an empty message, and a declared sh:message the result does not carry, are each RED; the fixture as is
    // is clean.
    let declared = declared_messages(&ssk);
    let empty = with_result(&r1, 1, |r| r.messages = vec![String::new()]);
    let missing = with_result(&r1, 0, |r| r.messages = vec!["some other text".into()]);
    let (fe, fm, f0) = (
        report::message_faults(&empty, &declared),
        report::message_faults(&missing, &declared),
        report::message_faults(&r1, &declared),
    );
    let got = if !f0.is_empty() || declared.is_empty() {
        UNNAMED
    } else {
        red_named(
            !fe.is_empty() && !fm.is_empty(),
            fe.iter().any(|l| l.contains("message")) && fm.iter().any(|l| l.contains("message")),
        )
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-020",
        "feedback/message",
        1,
        got,
        format!("{fe:?} {fm:?} clean {f0:?}"),
    ));

    // 021: sh:value removed where the reference has one: RED, naming value. An absent field matches nothing.
    let m = with_result(&r1, with_value, |r| r.key.value.clear());
    let lines = report::diff(&m, &r1).lines("under-test", "jena").join("; ");
    out.push(ctl(
        "FALSIFY-CRUXSHACL-021",
        "feedback/missing-value",
        1,
        red_named(!report::diff(&m, &r1).equal(), lines.contains("value")),
        lines,
    ));
    out
}

/// 022–025: the harness itself: no JVM, a bad pin, a partial subject, an incomplete receipt.
fn harness_controls(work: &Path) -> Vec<Ctl> {
    let mut out = Vec::new();
    // 022: no JVM is NOT MEASURED, or RED if the side under test already differs from W3C; never 0.
    let ok = Facts {
        pins_ok: true,
        complete: true,
        ..Facts::default()
    };
    let nojvm = jena::find_java("/nonexistent/jena-oracle-no-java");
    let got = match (
        nojvm.is_err(),
        verdict(ok),
        verdict(Facts {
            red_w3c: true,
            ..ok
        }),
    ) {
        (true, 2, 1) => 2,
        (true, v, _) => v,
        (false, _, _) => UNNAMED,
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-022",
        "selftest/no-jvm",
        2,
        got,
        format!("{nojvm:?}"),
    ));

    // 023: a pin one hex digit off is NOT MEASURED, for the Jena archive digest and for a suite manifest.
    let pinned = write(work, "pin.txt", "pinned\n");
    let sha = jena::sha256(&pinned).unwrap_or_default();
    let flip = |s: &str| {
        let mut c: Vec<char> = s.chars().collect();
        if let Some(x) = c.first_mut() {
            *x = if *x == '0' { '1' } else { '0' };
        }
        c.into_iter().collect::<String>()
    };
    let _ = std::fs::create_dir_all(work.join("suite"));
    std::fs::rename(&pinned, work.join("suite/pin.ttl")).ok();
    let good = format!("{sha}  suite/pin.ttl\n");
    let bad = format!("{}  suite/pin.ttl\n", flip(&sha));
    let all = Facts {
        complete: true,
        jvm: true,
        red_w3c: true,
        red_jena: true,
        ..Facts::default()
    };
    let got = match (
        check_manifest(work, &good, "suite").is_ok(),
        check_manifest(work, &bad, "suite").is_err(),
        jena::verify_pin(&work.join("suite/pin.ttl"), &flip(&sha)).is_err(),
        verdict(all),
    ) {
        (true, true, true, 2) => 2,
        (true, _, _, v) => v.min(1),
        _ => UNNAMED,
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-023",
        "selftest/pin-mismatch",
        2,
        got,
        format!("{:?}", check_manifest(work, &bad, "suite")),
    ));

    // 024: a listed case with no verdict, or a subject with no case, is NOT MEASURED.
    let listed: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
    let seen: BTreeSet<String> = ["a", "b"].map(String::from).into();
    let all3: BTreeSet<String> = listed.iter().cloned().collect();
    let good = Facts {
        pins_ok: true,
        jvm: true,
        ..Facts::default()
    };
    let got = match (
        complete(&listed, &seen).is_err(),
        complete(&[], &seen).is_err(),
        complete(&listed, &all3).is_ok(),
        verdict(good),
    ) {
        (true, true, true, 2) => 2,
        (true, true, true, v) => v,
        _ => 0,
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-024",
        "selftest/partial-subject",
        2,
        got,
        format!("{:?}", complete(&listed, &seen)),
    ));

    // 025: a receipt missing any required field is RED.
    let full = receipt(&Meta::dummy(), &[], &[], &[], 2);
    let missing: Vec<&str> = REQUIRED
        .iter()
        .copied()
        .filter(|f| check_receipt(&without(&full, f)).is_ok())
        .collect();
    let got = match (check_receipt(&full), missing.is_empty()) {
        (Ok(()), true) => 1,
        (Ok(()), false) => 0,
        (Err(_), _) => UNNAMED,
    };
    out.push(ctl(
        "FALSIFY-CRUXSHACL-025",
        "selftest/receipt-missing-field",
        1,
        got,
        format!("not caught: {missing:?}; full: {:?}", check_receipt(&full)),
    ));

    out
}

fn jvm_controls(j: &Jena, work: &Path) -> Vec<Ctl> {
    let started = SystemTime::now() - Duration::from_secs(2);
    let c004 = riot_line7(j, work);
    let (c005, written) = roundtrip(j, work, started);
    let c006 = dropped_triple(j, work, written);
    // The S-W3C minCount-001 expectation, shared by 008 and 009.
    let _ = std::fs::create_dir_all(work.join("w3c"));
    let fixture = parse_fixture("minCount-001.nt", MINCOUNT_NT);
    let w3c = report_of(&fixture).expect("minCount-001.nt holds W3C's expected report");
    let c008 = jena_mincount(j, work, &fixture, &w3c);
    let c009 = mincount_min2(j, work, &w3c);
    let c026 = shsh_mincount_string(j, work);
    JVM_CONTROLS
        .iter()
        .zip([c004, c005, c006, c008, c009, c026])
        .map(|(&(id, name, want), (got, detail))| ctl(id, name, want, got, detail))
        .collect()
}

/// 004: riot rejects the file and names line 7 (pv's side is the R-INPUT cell on the same file).
fn riot_line7(j: &Jena, work: &Path) -> (u8, String) {
    let p = write(work, "line7.ttl", LINE7_TTL);
    match j.riot(&p, None) {
        Ok((ok, clean, _, err)) if !ok || !clean => (
            if err.contains("[line: 7,") {
                0
            } else {
                UNNAMED
            },
            err.lines().next().unwrap_or("").to_string(),
        ),
        Ok(_) => (1, "riot accepted it".to_string()),
        Err(e) => (UNNAMED, e),
    }
}

/// The mechanism every strict read leans on: a file riot only WARNs about (an ill-formed xsd:integer) is refused
/// strict and read non-strict.
fn strict_bites(j: &Jena, work: &Path) -> Result<(), String> {
    let warn = write(
        work,
        "warn.ttl",
        "<http://ex.org/s> <http://ex.org/p> \"x\"^^<http://www.w3.org/2001/XMLSchema#integer> .\n",
    );
    match (j.read(&warn, None, true), j.read(&warn, None, false)) {
        (Err(e), Ok(t)) if e.contains("WARN") && t.len() == 1 => Ok(()),
        (s, n) => Err(format!(
            "strict read of a WARN file: {:?}, non-strict: {:?}",
            s.map(|t| t.len()),
            n.map(|t| t.len())
        )),
    }
}

/// 005: Jena reads X; the triples are written back by this harness's N-Triples writer; Jena reads that file to an
/// isomorphic graph, and `jena.rdfcompare` agrees. Returns the triples read, for 006.
fn roundtrip(j: &Jena, work: &Path, started: SystemTime) -> ((u8, String), Option<Vec<Triple>>) {
    if let Err(e) = strict_bites(j, work) {
        return ((UNNAMED, e), None);
    }
    let a = match j.read(&write(work, "esc.ttl", ESCAPES_TTL), None, true) {
        Ok(a) => a,
        Err(e) => return ((UNNAMED, e), None),
    };
    let riot_nt = write(work, "esc-riot.nt", &nt_text(&a));
    let w = write(work, "esc-written.nt", &nt_text(&relabel(&a)));
    let sc = self_compare(&riot_nt, &w, started);
    match (j.read(&w, None, true), j.rdfcompare(&riot_nt, &w), sc) {
        (Ok(b), Ok(true), None) => {
            let iso = nt::compare(&a, &b);
            let lits = a
                .iter()
                .filter(|t| matches!(t[2], Term::Lit { .. }))
                .count();
            let got = if iso.equal && lits >= 9 { 0 } else { 1 };
            let detail = format!("{} triples, {lits} literals, iso {}", a.len(), iso.equal);
            ((got, detail), Some(a))
        }
        (b, r, sc) => (
            (
                1,
                format!(
                    "read {:?}, rdfcompare {r:?}, self {sc:?}",
                    b.map(|b| b.len())
                ),
            ),
            None,
        ),
    }
}

/// 006: the written file loses one triple before Jena reads it: RED, naming the triple.
fn dropped_triple(j: &Jena, work: &Path, written: Option<Vec<Triple>>) -> (u8, String) {
    let Some(a) = written else {
        return (UNNAMED, "005 did not produce a written file".into());
    };
    let i = a
        .iter()
        .position(|t| matches!(&t[2], Term::Lit { lang, .. } if lang == "fr"))
        .unwrap_or(0);
    let mut w = a.clone();
    let gone = w.remove(i);
    let wp = write(work, "esc-dropped.nt", &nt_text(&w));
    match j.read(&wp, None, true) {
        Ok(b) => {
            let iso = nt::compare(&a, &b);
            let lex = match &gone[2] {
                Term::Lit { lex, .. } => lex.clone(),
                t => t.nt(),
            };
            (
                red_named(!iso.equal, iso.only_a.iter().any(|l| l.contains(&lex))),
                format!("{:?}", iso.only_a),
            )
        }
        Err(e) => (UNNAMED, e),
    }
}

/// 008: Jena on W3C minCount-001 gives W3C's expected report. The committed fixture must still be what riot reads
/// from the case file, or the pure controls test a stale expectation.
fn jena_mincount(j: &Jena, work: &Path, fixture: &[Triple], w3c: &Report) -> (u8, String) {
    let case = write(work, "w3c/minCount-001.ttl", MINCOUNT_TTL);
    match (
        j.read(&case, Some(MINCOUNT_BASE), true),
        j.validate(&case, &case, work, "mincount"),
    ) {
        (Ok(t), Ok(rep)) if nt::compare(&t, fixture).equal => match report_of(&rep) {
            Ok(jr) => {
                let d = report::diff(&jr, w3c);
                (
                    if d.equal() { 0 } else { 1 },
                    format!(
                        "{} results; {}",
                        jr.results.len(),
                        d.lines("jena", "w3c").join("; ")
                    ),
                )
            }
            Err(e) => (UNNAMED, e),
        },
        (Ok(_), Ok(_)) => (
            UNNAMED,
            "controls/minCount-001.nt no longer matches the W3C case: regenerate it".into(),
        ),
        (a, b) => (UNNAMED, format!("{:?} {:?}", a.err(), b.err())),
    }
}

/// 009: the copy under test has sh:minCount 2: one result more than W3C expects. RED, naming the case.
fn mincount_min2(j: &Jena, work: &Path, w3c: &Report) -> (u8, String) {
    let min2 = MINCOUNT_TTL.replacen("sh:minCount 1 ;", "sh:minCount 2 ;", 1);
    let p = write(work, "w3c/minCount-001-min2.ttl", &min2);
    match j.validate(&p, &p, work, "min2").and_then(|t| report_of(&t)) {
        Ok(jr) if min2 != MINCOUNT_TTL => {
            let (l, _) = validate_line("property/minCount-001", Ok(&jr), None, w3c);
            (
                red_named(
                    l.v == V::Red,
                    l.case == "property/minCount-001" && jr.results.len() == w3c.results.len() + 1,
                ),
                l.detail,
            )
        }
        Ok(_) => (UNNAMED, "sh:minCount 1 ; not found in the case".into()),
        Err(e) => (UNNAMED, e),
    }
}

/// 026: a shape graph with sh:minCount "one" does not conform to SHACL-SHACL; the well-formed one does.
fn shsh_mincount_string(j: &Jena, work: &Path) -> (u8, String) {
    let bad_text = SHSH_OK_TTL.replacen("sh:minCount 1 .", "sh:minCount \"one\" .", 1);
    let shsh = match j.shsh(work) {
        Ok(s) => s,
        Err(e) => return (UNNAMED, e),
    };
    let ok = write(work, "shsh-ok.ttl", SHSH_OK_TTL);
    let bad = write(work, "shsh-bad.ttl", &bad_text);
    match (
        j.validate(&shsh, &ok, work, "shsh-ok")
            .and_then(|t| report_of(&t)),
        j.validate(&shsh, &bad, work, "shsh-bad")
            .and_then(|t| report_of(&t)),
    ) {
        (Ok(g), Ok(b)) if bad_text != SHSH_OK_TTL && shsh_line("ok", &g).v == V::Green => {
            let l = shsh_line("shsh/minCount-string", &b);
            (
                red_named(l.v == V::Red, l.detail.contains(&format!("<{EX}S-p>"))),
                l.detail,
            )
        }
        (g, b) => (
            UNNAMED,
            format!(
                "ok {:?} / bad {:?}",
                g.map(|r| r.conforms),
                b.map(|r| r.conforms)
            ),
        ),
    }
}

fn unrun_jvm_controls(why: &str) -> Vec<Ctl> {
    JVM_CONTROLS
        .iter()
        .map(|&(id, name, want)| Ctl {
            id,
            name,
            want,
            got: None,
            detail: format!("NOT MEASURED: {why}"),
        })
        .collect()
}

fn print_controls(c: &[Ctl]) -> (usize, usize, usize) {
    let mut sorted: Vec<&Ctl> = c.iter().collect();
    sorted.sort_by_key(|c| c.id);
    let (mut ok, mut failed, mut unrun) = (0, 0, 0);
    for c in sorted {
        let tag = match c.got {
            None => {
                unrun += 1;
                "NOT MEASURED"
            }
            Some(g) if g == c.want => {
                ok += 1;
                "ok"
            }
            Some(_) => {
                failed += 1;
                "FAIL"
            }
        };
        let detail: String = c.detail.chars().take(240).collect();
        println!(
            "  {tag} {} {} want {} got {:?}: {detail}",
            c.id, c.name, c.want, c.got
        );
    }
    (ok, failed, unrun)
}

// ---------------------------------------------------------------------------------------------------------------
// Receipt.
// ---------------------------------------------------------------------------------------------------------------

/// Everything a verdict is tied to. A field this run could not measure holds `NOT MEASURED: <why>`, never "".
struct Meta {
    jena_version: String,
    zip_sha256: String,
    shsh_sha256: String,
    java_version: String,
    pv_version: String,
    pv_sha256: String,
    git_head: String,
    git_dirty: String,
    inputs: BTreeMap<String, String>,
}

impl Meta {
    fn dummy() -> Meta {
        let s = |x: &str| x.to_string();
        Meta {
            jena_version: s(jena::JENA_VERSION),
            zip_sha256: s(jena::JENA_ZIP_SHA256),
            shsh_sha256: s(SHSH_SHA256),
            java_version: s("openjdk version \"17\""),
            pv_version: s("pv 0.0.0"),
            pv_sha256: s("0"),
            git_head: s("0"),
            git_dirty: s("false"),
            inputs: BTreeMap::from([(s("x"), s("0"))]),
        }
    }
}

/// The receipt fields FALSIFY-CRUXSHACL-025 requires, as JSON pointer paths.
const REQUIRED: [&str; 16] = [
    "/schema",
    "/utc",
    "/jena/version",
    "/jena/zip_sha256",
    "/jena/shsh_sha256",
    "/java/version",
    "/pv/version",
    "/pv/sha256",
    "/git/head",
    "/git/dirty",
    "/inputs",
    "/cells",
    "/controls/total",
    "/controls/seen",
    "/controls/rows",
    "/exit",
];

/// A JSON object from (key, value) pairs. `serde_json::json!` with interpolated values expands to `unwrap`, which
/// the repository's clippy configuration disallows, so the receipt is built from `Value::from` instead.
fn obj<const N: usize>(pairs: [(&str, serde_json::Value); N]) -> serde_json::Value {
    serde_json::Value::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn receipt(
    m: &Meta,
    lines: &[Line],
    controls: &[Ctl],
    notes: &[String],
    exit: u8,
) -> serde_json::Value {
    use serde_json::Value;
    let utc = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let cell = |cell: &str, case: &str, comparator: &str, verdict: &str, detail: &str| {
        obj([
            ("cell", cell.into()),
            ("case", case.into()),
            ("comparator", comparator.into()),
            ("verdict", verdict.into()),
            ("detail", detail.into()),
        ])
    };
    let mut cells: Vec<Value> = lines
        .iter()
        .map(|l| cell(l.cell, &l.case, l.comparator, l.v.name(), &l.detail))
        .collect();
    if cells.is_empty() {
        cells.push(cell("-", "-", "-", "NOT MEASURED", "no cell ran"));
    }
    let rows: Vec<Value> = controls
        .iter()
        .map(|c| {
            obj([
                ("id", c.id.into()),
                ("name", c.name.into()),
                ("want", c.want.into()),
                ("got", c.got.map_or(Value::Null, Value::from)),
                ("seen", (c.got == Some(c.want)).into()),
            ])
        })
        .collect();
    let seen = controls.iter().filter(|c| c.got == Some(c.want)).count();
    let s = |x: &str| Value::from(x);
    obj([
        ("schema", RECEIPT_SCHEMA.into()),
        ("utc", utc.into()),
        (
            "jena",
            obj([
                ("version", s(&m.jena_version)),
                ("zip_sha256", s(&m.zip_sha256)),
                ("shsh_sha256", s(&m.shsh_sha256)),
            ]),
        ),
        ("java", obj([("version", s(&m.java_version))])),
        (
            "pv",
            obj([("version", s(&m.pv_version)), ("sha256", s(&m.pv_sha256))]),
        ),
        (
            "git",
            obj([("head", s(&m.git_head)), ("dirty", s(&m.git_dirty))]),
        ),
        (
            "inputs",
            Value::Object(m.inputs.iter().map(|(k, v)| (k.clone(), s(v))).collect()),
        ),
        ("cells", cells.into()),
        (
            "controls",
            obj([
                ("total", controls.len().into()),
                ("seen", seen.into()),
                ("rows", rows.into()),
            ]),
        ),
        ("notes", notes.to_vec().into()),
        ("exit", exit.into()),
    ])
}

fn without(v: &serde_json::Value, pointer: &str) -> serde_json::Value {
    let mut v = v.clone();
    let (parent, key) = pointer.rsplit_once('/').unwrap_or(("", pointer));
    if let Some(serde_json::Value::Object(o)) = v.pointer_mut(parent) {
        o.remove(key);
    }
    v
}

/// FALSIFY-CRUXSHACL-025: every required field is present and non-empty.
fn check_receipt(v: &serde_json::Value) -> Result<(), String> {
    for p in REQUIRED {
        let ok = match v.pointer(p) {
            None | Some(serde_json::Value::Null) => false,
            Some(serde_json::Value::String(s)) => !s.trim().is_empty(),
            Some(serde_json::Value::Object(o)) => !o.is_empty(),
            Some(serde_json::Value::Array(a)) => p == "/controls/rows" || !a.is_empty(),
            Some(_) => true,
        };
        if !ok {
            return Err(format!("receipt field {p} is missing or empty"));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------
// Modes.
// ---------------------------------------------------------------------------------------------------------------

fn java_cmd() -> String {
    std::env::var("ONT_ORACLE_JAVA")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "java".into())
}

/// Find the JVM, verify and unpack the pin. `(jena, java_version_or_why, pins_ok)`.
fn open_jena(work: &Path) -> (Option<Jena>, Result<String, String>, Result<(), String>) {
    let cmd = java_cmd();
    let java = jena::find_java(&cmd);
    if java.is_err() {
        return (None, java, Ok(()));
    }
    let zip = match jena::pinned_zip() {
        Ok(z) => z,
        Err(e) => return (None, java, Err(e)),
    };
    match Jena::open(&cmd, &zip, work) {
        Ok(j) => (Some(j), java, Ok(())),
        Err(e) => (None, java, Err(e)),
    }
}

fn self_test() -> u8 {
    let work = match jena::fresh_dir("selftest") {
        Ok(w) => w,
        Err(e) => {
            println!("decline: NOT MEASURED ({e})");
            return 2;
        }
    };
    let mut c = pure_controls(&work);
    let (j, java, pins) = open_jena(&work);
    match (&j, &java, &pins) {
        (Some(j), _, _) => c.extend(jvm_controls(j, &work)),
        (None, Err(e), _) => c.extend(unrun_jvm_controls(&format!("no JVM: {e}"))),
        (None, _, Err(e)) => c.extend(unrun_jvm_controls(&format!("pin: {e}"))),
        (None, Ok(_), Ok(())) => c.extend(unrun_jvm_controls("Jena did not open")),
    }
    let (ok, failed, unrun) = print_controls(&c);
    println!(
        "self-test: {ok}/{} ok{}",
        c.len(),
        if unrun > 0 {
            format!(", {unrun} NOT MEASURED")
        } else {
            String::new()
        }
    );
    let _ = std::fs::remove_dir_all(&work);
    if failed > 0 {
        1
    } else if unrun > 0 {
        2
    } else {
        0
    }
}

fn first_line(cmd: &mut Command) -> Option<String> {
    let o = cmd.output().ok()?;
    let s = String::from_utf8_lossy(&o.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();
    (o.status.success() && !s.is_empty()).then_some(s)
}

fn nm(why: impl std::fmt::Display) -> String {
    format!("NOT MEASURED: {why}")
}

/// The S-W3C cases of one pinned file: `(case name, test node's report, data path, shapes path)`.
fn w3c_cases(rel: &str, t: &[Triple]) -> Vec<(String, Report, PathBuf, PathBuf)> {
    let g = Graph::new(t);
    let name = rel
        .trim_start_matches("tests/oracle/w3c/")
        .trim_end_matches(".ttl")
        .to_string();
    let path_of = |x: Option<&Term>| match x {
        Some(Term::Iri(i)) => i.strip_prefix("file://").map(PathBuf::from),
        _ => None,
    };
    let mut out = Vec::new();
    let tests: Vec<(&Term, &Term)> = t
        .iter()
        .filter(|[_, p, _]| *p == Term::Iri(format!("{MF}result")))
        .map(|[s, _, o]| (s, o))
        .collect();
    for (k, (test, rep)) in tests.iter().enumerate() {
        let action = g.obj(test, &Term::Iri(format!("{MF}action")));
        let data = action.and_then(|a| path_of(g.obj(a, &Term::Iri(format!("{SHT}dataGraph")))));
        let shapes =
            action.and_then(|a| path_of(g.obj(a, &Term::Iri(format!("{SHT}shapesGraph")))));
        if let (Some(d), Some(s)) = (data, shapes) {
            let case = if tests.len() > 1 {
                format!("{name}#{k}")
            } else {
                name.clone()
            };
            out.push((case, report::report_at(&g, rep), d, s));
        }
    }
    out
}

/// What one `run` accumulates: its cell lines, notes, facts for the verdict, and the receipt's metadata.
struct Run<'a> {
    repo: &'a Path,
    pv: &'a Path,
    started: SystemTime,
    work: PathBuf,
    nt_path: PathBuf,
    ttl_path: PathBuf,
    lines: Vec<Line>,
    notes: Vec<String>,
    f: Facts,
    meta: Meta,
}

fn run_meta(repo: &Path, pv: &Path) -> Meta {
    let git = |args: &[&str]| first_line(Command::new("git").arg("-C").arg(repo).args(args));
    let dirty = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .map(|o| (!o.stdout.is_empty()).to_string())
        .unwrap_or_else(nm);
    Meta {
        jena_version: jena::JENA_VERSION.into(),
        zip_sha256: nm("Jena not opened"),
        shsh_sha256: nm("Jena not opened"),
        java_version: nm("not probed"),
        pv_version: first_line(Command::new(pv).arg("--version"))
            .unwrap_or_else(|| nm(format!("{} --version", pv.display()))),
        pv_sha256: jena::sha256(pv).unwrap_or_else(|| nm(pv.display())),
        git_head: git(&["rev-parse", "HEAD"]).unwrap_or_else(|| nm("git rev-parse")),
        git_dirty: dirty,
        inputs: BTreeMap::new(),
    }
}

impl Run<'_> {
    fn line(
        &mut self,
        cell: &'static str,
        case: &str,
        comparator: &'static str,
        v: V,
        detail: String,
    ) {
        self.lines
            .push(Line::new(cell, case, comparator, v, detail));
    }

    /// Subjects: pv's two files, pv's report, and the pinned W3C suite. Returns pv's (focus, component) pairs and
    /// the suite's case files.
    fn subjects(&mut self, pv_json: &Path) -> (Option<Vec<Pair>>, Vec<String>) {
        for p in [
            self.nt_path.clone(),
            self.ttl_path.clone(),
            pv_json.to_path_buf(),
        ] {
            let sha = jena::sha256(&p).unwrap_or_else(|| nm("unreadable"));
            if std::fs::metadata(&p).map_or(true, |m| m.len() == 0) {
                self.f.complete = false;
                self.line(
                    "subject",
                    &p.display().to_string(),
                    "-",
                    V::Nm,
                    "missing or empty".into(),
                );
            }
            let key = p
                .strip_prefix(self.repo)
                .unwrap_or(&p)
                .display()
                .to_string();
            self.meta.inputs.insert(key, sha);
        }
        let pv_pairs = std::fs::read_to_string(pv_json)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| pv_violations(&v));
        if pv_pairs.is_none() {
            self.f.complete = false;
            let why = format!(
                "{} has no findings array: pv did not run",
                pv_json.display()
            );
            self.line("subject", "pv-report", "-", V::Nm, why);
        }
        let manifest_path = self.repo.join("tests/oracle/jena/w3c.sha256");
        let suite = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("{}: {e}", manifest_path.display()))
            .and_then(|m| check_manifest(self.repo, &m, "tests/oracle/w3c"));
        self.meta.inputs.insert(
            "tests/oracle/jena/w3c.sha256".into(),
            jena::sha256(&manifest_path).unwrap_or_else(|| nm("unreadable")),
        );
        let suite = suite.unwrap_or_else(|e| {
            self.f.pins_ok = false;
            self.line("pins", "S-W3C", "sha256", V::Nm, e);
            Vec::new()
        });
        (pv_pairs, suite)
    }

    /// The engine: the JVM, Jena's pinned zip and the pinned SHACL-SHACL graph.
    fn engine(&mut self) -> (Option<Jena>, Option<Result<PathBuf, String>>) {
        let (j, java, pins) = open_jena(&self.work);
        self.meta.java_version = java.clone().unwrap_or_else(nm);
        self.f.jvm = java.is_ok();
        if let Err(e) = java {
            println!("decline: NOT MEASURED (no JVM): {e}");
            self.line("engine", "jvm", "-", V::Nm, e);
        }
        if let Err(e) = pins {
            self.f.pins_ok = false;
            self.line("pins", "jena", "sha256", V::Nm, e);
        }
        if j.is_some() {
            self.meta.zip_sha256 = jena::JENA_ZIP_SHA256.into();
        }
        let shsh = j.as_ref().map(|j| {
            j.shsh(&self.work)
                .and_then(|p| jena::verify_pin(&p, SHSH_SHA256).map(|()| p))
        });
        match &shsh {
            Some(Ok(_)) => self.meta.shsh_sha256 = SHSH_SHA256.into(),
            Some(Err(e)) => {
                self.f.pins_ok = false;
                self.line("pins", "S-SHSH", "sha256", V::Nm, e.clone());
            }
            None => {}
        }
        (j, shsh)
    }

    /// R-RETURN, read side: riot reads pv's two files without a warning; contracts.nt reads to the same graph as
    /// the neutral N-Triples parser here. Then pv's side (S3): R-INPUT and the round trip on both files and on the
    /// escapes control, and R-INPUT on the malformed control.
    fn return_cells(&mut self, j: &Jena) {
        let (nt_path, ttl_path) = (self.nt_path.clone(), self.ttl_path.clone());
        let nt_jena = j.read(&nt_path, Some(&self.base(&nt_path)), true);
        let ttl_jena = j.read(&ttl_path, Some(&self.base(&ttl_path)), true);
        let parsed = std::fs::read_to_string(&self.nt_path)
            .map_err(|e| e.to_string())
            .and_then(|s| nt::parse(&s));
        let (v, detail) = match (&nt_jena, parsed) {
            (Ok(a), Ok(b)) => {
                let iso = nt::compare(a, &b);
                let diff: Vec<_> = iso.only_a.iter().chain(&iso.only_b).take(3).collect();
                let v = if iso.equal { V::Green } else { V::Red };
                (v, format!("{} triples; {diff:?}", a.len()))
            }
            (a, b) => (V::Red, format!("{:?} {:?}", a.as_ref().err(), b.err())),
        };
        self.line("R-RETURN", "contracts.nt", "riot-vs-ntparse", v, detail);
        let (v, detail) = match &ttl_jena {
            Ok(t) => (V::Green, format!("{} triples, no WARN", t.len())),
            Err(e) => (V::Red, e.clone()),
        };
        self.line("R-RETURN", "shapes.ttl", "riot", v, detail);
        let esc = self
            .repo
            .join("tests/oracle/jena/controls/escapes-langtags.ttl");
        let esc_jena = j.read(&esc, Some(&self.base(&esc)), true);
        for (x, jena) in [(nt_path, nt_jena), (ttl_path, ttl_jena), (esc, esc_jena)] {
            self.pv_cells(j, &x, jena.as_ref());
        }
        self.malformed_cell(j);
    }

    /// X relative to the repo, as a case name.
    fn rel(&self, x: &Path) -> String {
        x.strip_prefix(self.repo).unwrap_or(x).display().to_string()
    }

    /// The base both engines resolve X against: fixed, so a receipt does not depend on where the repo sits.
    fn base(&self, x: &Path) -> String {
        format!("file:///{}", self.rel(x))
    }

    /// Record the sha256 of an input the cells read beyond the subjects.
    fn pin_input(&mut self, x: &Path) {
        let sha = jena::sha256(x).unwrap_or_else(|| nm("unreadable"));
        self.meta.inputs.insert(self.rel(x), sha);
    }

    /// pv's dump of X against riot's parse `a`; the dump comes back for the round trip.
    fn pv_vs_riot(&self, j: &Jena, x: &Path, a: &[Triple]) -> (V, String, Option<String>) {
        match pv::read(self.pv, x, &self.base(x)) {
            Ok(pv::Read::Dump(text, b)) => {
                let (v, d) = same_graph(j, &self.work, &tag(&self.rel(x)), a, &b);
                let d = format!("{d}; pv {} blank nodes", pv::blank_nodes(&b));
                (v, d, Some(text))
            }
            Ok(pv::Read::Refused(_, msg)) => {
                (V::Red, format!("pv refuses what riot reads: {msg}"), None)
            }
            Ok(pv::Read::Broken(e)) => (V::Red, e, None),
            Err(e) => (V::Nm, e, None),
        }
    }

    /// R-INPUT and the R-RETURN round trip on X, which riot read as `jena`.
    fn pv_cells(&mut self, j: &Jena, x: &Path, jena: Result<&Vec<Triple>, &String>) {
        self.pin_input(x);
        let case = self.rel(x);
        let trip = format!("roundtrip {case}");
        let a = match jena {
            Ok(a) => a,
            Err(e) => {
                let why = format!("riot refused X: {e}");
                self.line("R-INPUT", &case, "pv-vs-riot", V::Nm, why.clone());
                return self.line("R-RETURN", &trip, "riot-roundtrip", V::Nm, why);
            }
        };
        let (v, detail, dump) = self.pv_vs_riot(j, x, a);
        self.line("R-INPUT", &case, "pv-vs-riot", v, detail);
        match dump {
            Some(d) => self.roundtrip_cell(j, x, &trip, a, &d),
            None => self.line(
                "R-RETURN",
                &trip,
                "riot-roundtrip",
                V::Nm,
                "pv wrote no dump".into(),
            ),
        }
    }

    /// R-RETURN round trip: Jena(X) ≅ Jena(pv_write(pv_read(X))). pv's dump goes to a file of its own, made by this
    /// run; X itself or a stale file is refused (FALSIFY-CRUXSHACL-007).
    fn roundtrip_cell(&mut self, j: &Jena, x: &Path, trip: &str, a: &[Triple], dump: &str) {
        let w = write(&self.work, &format!("pv-{}.nt", tag(trip)), dump);
        if let Some(why) = self_compare(x, &w, self.started) {
            self.f.self_compare = true;
            return self.line("R-RETURN", trip, "riot-roundtrip", V::Nm, why);
        }
        let (v, detail) = match j.read(&w, None, true) {
            Ok(b) => same_graph(j, &self.work, &tag(trip), a, &b),
            Err(e) => (V::Red, format!("riot refuses the file pv wrote: {e}")),
        };
        self.line("R-RETURN", trip, "riot-roundtrip", v, detail);
    }

    /// R-INPUT on a malformed X: riot and pv both refuse it, at the same line.
    fn malformed_cell(&mut self, j: &Jena) {
        let x = self
            .repo
            .join("tests/oracle/jena/controls/syntax-error-line-7.ttl");
        self.pin_input(&x);
        let base = self.base(&x);
        let riot = match j.riot(&x, Some(&base)) {
            Ok((ok, clean, _, err)) if !ok || !clean => {
                pv::riot_line(&err).ok_or_else(|| format!("riot names no line: {err}"))
            }
            Ok(_) => Err("riot accepted it".into()),
            Err(e) => Err(e),
        };
        let (v, detail) = match (riot, pv::read(self.pv, &x, &base)) {
            (Err(e), _) | (_, Err(e)) => (V::Nm, e),
            (Ok(r), Ok(pv::Read::Refused(Some(p), msg))) => {
                let v = if p == r { V::Green } else { V::Red };
                (v, format!("riot line {r}, pv line {p}: {msg}"))
            }
            (Ok(r), Ok(pv::Read::Refused(None, msg))) => {
                (V::Red, format!("riot line {r}, pv names no line: {msg}"))
            }
            (Ok(r), Ok(pv::Read::Dump(_, b))) => (
                V::Red,
                format!("riot refuses at line {r}, pv reads {} triples", b.len()),
            ),
            (Ok(_), Ok(pv::Read::Broken(e))) => (V::Red, e),
        };
        let case = self.rel(&x);
        self.line("R-INPUT", &case, "line", v, detail);
    }

    /// R-INPUT on S-W3C: pv's dump of every pinned case file against riot's, read non-strict (some cases hold
    /// ill-formed literals on purpose). A line for each file that does not agree, and one for the files that do.
    fn w3c_input_cells(&mut self, j: &Jena, suite: &[String]) {
        let mut agree = 0;
        for rel in suite {
            let x = self.repo.join(rel);
            let (v, detail) = match j.read(&x, Some(&self.base(&x)), false) {
                Ok(a) => {
                    let (v, d, _) = self.pv_vs_riot(j, &x, &a);
                    (v, d)
                }
                Err(e) => (V::Nm, e),
            };
            if v == V::Green {
                agree += 1;
            } else {
                self.line("R-INPUT", rel, "pv-vs-riot", v, detail);
            }
        }
        let detail = format!("{agree} of {} pinned case files isomorphic", suite.len());
        if suite.is_empty() || agree < suite.len() {
            return self.notes.push(format!("R-INPUT S-W3C: {detail}"));
        }
        self.line("R-INPUT", "S-W3C", "pv-vs-riot", V::Green, detail);
    }

    /// R-VALIDATE, corpus: Jena's (focus, component) multiset against pv's.
    fn corpus_cell(&mut self, j: &Jena, pv_pairs: Option<&Vec<Pair>>) {
        let jr = j
            .validate(&self.ttl_path, &self.nt_path, &self.work, "corpus")
            .and_then(|t| report_of(&t));
        let (jr, pvp) = match (jr, pv_pairs) {
            (Ok(jr), Some(pvp)) => (jr, pvp),
            (Err(e), _) => {
                self.f.complete = false;
                return self.line("R-VALIDATE", "corpus", "jena-vs-pv", V::Nm, e);
            }
            (_, None) => return,
        };
        let jp = jena_pairs(&jr);
        let d = pair_diff(&jp, pvp);
        let v = if d.is_empty() { V::Green } else { V::Red };
        let all: Vec<String> = d
            .iter()
            .map(|((fo, c), n)| format!("{fo} {c} jena{n:+}"))
            .collect();
        self.notes
            .extend(all.iter().map(|x| format!("corpus jena≠pv: {x}")));
        let detail = format!(
            "jena {} results (conforms {:?}), pv {}; {} pairs differ: {}",
            jp.len(),
            jr.conforms,
            pvp.len(),
            d.len(),
            all[..all.len().min(8)].join("; ")
        );
        self.line("R-VALIDATE", "corpus", "jena-vs-pv", v, detail);
    }

    /// Jena on one W3C case against its expected report. A Jena disagreement is a note: W3C decides, and Jena is
    /// not under test. Returns whether the source shape had to be dropped from the comparison.
    fn w3c_case(
        &mut self,
        j: &Jena,
        case: &str,
        w3c: &Report,
        data: &Path,
        shapes: &Path,
    ) -> Option<bool> {
        let tag = case.replace(['/', '#'], "-");
        match j
            .validate(shapes, data, &self.work, &tag)
            .and_then(|r| report_of(&r))
        {
            Ok(jr) => {
                let d = report::diff(&jr, w3c);
                if !d.equal() {
                    let lines = d.lines("jena", "w3c").join("; ");
                    self.notes.push(format!("jena≠w3c {case}: {lines}"));
                }
                Some(d.degraded > 0)
            }
            Err(e) => {
                self.line("R-VALIDATE", case, "jena-vs-w3c", V::Nm, e);
                None
            }
        }
    }

    /// R-VALIDATE and R-FEEDBACK on S-W3C. pv's side waits for S4.
    fn w3c_cells(&mut self, j: &Jena, suite: &[String]) {
        let mut listed = Vec::new();
        let mut seen = BTreeSet::new();
        let mut degraded = 0;
        for rel in suite {
            let t = match j.read(&self.repo.join(rel), None, false) {
                Ok(t) => t,
                Err(e) => {
                    self.line("R-VALIDATE", rel, "jena-vs-w3c", V::Nm, e);
                    listed.push(rel.clone());
                    continue;
                }
            };
            for (case, w3c, data, shapes) in w3c_cases(rel, &t) {
                listed.push(case.clone());
                if let Some(d) = self.w3c_case(j, &case, &w3c, &data, &shapes) {
                    degraded += usize::from(d);
                    seen.insert(case);
                }
            }
        }
        if let Err(e) = complete(&listed, &seen) {
            self.f.complete = false;
            self.line("R-VALIDATE", "S-W3C", "jena-vs-w3c", V::Nm, e);
        }
        let pv_side = format!(
            "S4: pv writes no sh:ValidationReport; {} cases, Jena ran {}, {degraded} compared without source shape",
            listed.len(),
            seen.len()
        );
        self.line("R-VALIDATE", "S-W3C", "pv-vs-w3c", V::Nm, pv_side);
        let feedback = "S4: pv's report has no value, source shape or W3C severity".into();
        self.line(
            "R-FEEDBACK",
            "S-W3C+S-CORPUS",
            "full-tuple",
            V::Nm,
            feedback,
        );
    }

    /// S-SHSH: pv's shapes conform to the SHACL-SHACL graph Jena ships.
    fn shsh_cell(&mut self, j: &Jena, shsh: &Path) {
        match j
            .validate(shsh, &self.ttl_path, &self.work, "shsh")
            .and_then(|t| report_of(&t))
        {
            Ok(r) => self.lines.push(shsh_line("shapes.ttl", &r)),
            Err(e) => {
                self.f.complete = false;
                self.line("S-SHSH", "shapes.ttl", "shacl-shacl", V::Nm, e);
            }
        }
    }

    /// Print the lines, controls and notes, write the receipt, and return the exit code.
    fn finish(mut self, controls: &[Ctl]) -> u8 {
        let f = &mut self.f;
        f.red_w3c = self
            .lines
            .iter()
            .any(|l| l.v == V::Red && l.comparator == "w3c");
        f.red_jena = self
            .lines
            .iter()
            .any(|l| l.v == V::Red && l.comparator != "w3c");
        f.nm_cell = self.lines.iter().any(|l| l.v == V::Nm);
        let f = self.f;
        let mut exit = verdict(f);
        for l in &self.lines {
            l.print();
        }
        let (ok, failed, unrun) = print_controls(controls);
        println!(
            "controls: {ok}/{} seen, {failed} failed, {unrun} NOT MEASURED",
            controls.len()
        );
        for n in &self.notes {
            println!("note: {n}");
        }
        let r = receipt(&self.meta, &self.lines, controls, &self.notes, exit);
        if let Err(e) = check_receipt(&r) {
            println!("RED receipt: {e}");
            exit = 1;
        }
        let out = self.repo.join("tests/oracle/jena/receipt.json");
        let text = serde_json::to_string_pretty(&receipt(
            &self.meta,
            &self.lines,
            controls,
            &self.notes,
            exit,
        ))
        .unwrap_or_default();
        if let Err(e) = std::fs::write(&out, text + "\n") {
            println!("RED receipt: {}: {e}", out.display());
            exit = 1;
        }
        let _ = std::fs::remove_dir_all(&self.work);
        println!(
            "verdict: exit {exit} (pins_ok {}, complete {}, jvm {}, red_w3c {}, red_jena {}, control_unseen {}, nm_cell {}) → {}",
            f.pins_ok,
            f.complete,
            f.jvm,
            f.red_w3c,
            f.red_jena,
            f.control_unseen,
            f.nm_cell,
            out.display()
        );
        exit
    }
}

fn run(repo: &Path, pv_json: &Path, pv: &Path) -> u8 {
    let work = match jena::fresh_dir("run") {
        Ok(w) => w,
        Err(e) => {
            println!("decline: NOT MEASURED ({e})");
            return 2;
        }
    };
    let mut r = Run {
        repo,
        pv,
        started: SystemTime::now() - Duration::from_secs(2),
        work,
        nt_path: repo.join("contracts/contracts.nt"),
        ttl_path: repo.join("contracts/shapes.ttl"),
        lines: Vec::new(),
        notes: Vec::new(),
        f: Facts {
            pins_ok: true,
            complete: true,
            ..Facts::default()
        },
        meta: run_meta(repo, pv),
    };
    let (pv_pairs, suite) = r.subjects(pv_json);
    let (j, shsh) = r.engine();

    // Controls, every run.
    let mut controls = pure_controls(&r.work);
    match &j {
        Some(j) if r.f.pins_ok => controls.extend(jvm_controls(j, &r.work)),
        _ => controls.extend(unrun_jvm_controls("no JVM or a pin failed")),
    }
    r.f.control_unseen = controls.iter().any(|c| c.got != Some(c.want));

    // The cells: only on matching pins (023 runs no comparison otherwise).
    if let (Some(j), true, Some(Ok(shsh))) = (&j, r.f.pins_ok, &shsh) {
        r.return_cells(j);
        r.corpus_cell(j, pv_pairs.as_ref());
        r.w3c_cells(j, &suite);
        r.w3c_input_cells(j, &suite);
        r.shsh_cell(j, shsh);
    }
    r.finish(&controls)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["--self-test"] => self_test(),
        ["run", repo, pv_json, "--pv", pv] => {
            run(Path::new(repo), Path::new(pv_json), Path::new(pv))
        }
        _ => {
            eprintln!("usage: jena-oracle --self-test | jena-oracle run <repo> <pv-shapes.json> --pv <pv>");
            2
        }
    };
    std::process::exit(i32::from(code));
}
