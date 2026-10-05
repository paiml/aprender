//! ONT-001 §3.6, §5 ONT-4b2 — the vendored W3C SHACL-Core cases, run by the shapes gate every time it runs.
//!
//! The originals live in `tests/oracle/w3c/<group>/<name>.ttl` (w3c/data-shapes@gh-pages,
//! `data-shapes-test-suite/tests/core/`) for the out-of-gate oracle. The gate path has no Turtle reader (R-13,
//! §3.5), so each case runnable within the subset is **translated by hand** into `w3c/<group>-<name>.yaml` —
//! the shape in the contract dialect [`crate::ontology::shapes`] parses, the data as N-Triples with the case's
//! prefixes, the expected report as W3C states it — and embedded here at compile time. The translation is checked
//! by the oracle differential (`make oracle`): the same data and the exported shape through the pinned `shacl`
//! crate must agree with the expected report too, so a mistranslation is caught from the other side.
//!
//! Two translations are semantic, and each case that uses one says so in its `targeting:` line: (1) a
//! `sh:targetNode` set becomes a class — every target node is typed `<case>#Focus` and the shape targets it,
//! which yields the same focus nodes; (2) a data blank node is skolemized to a named node under the case prefix,
//! which changes no result the case expects. What is NOT vendored is listed in [`NOT_VENDORED`], one line per
//! case with the component or form that puts it outside the subset — so the count reported is a count over the
//! cases the subset claims, and a case the validator refuses by design is never scored as a failure.
//!
//! A case fails when `conforms` or the multiset of `(focus, path, component)` differs from the expected report.
//! A failing case is a differential, not a corpus verdict: the gate declines `Unknown{Differential}` naming it.

use std::collections::BTreeMap;

use crate::ontology::rdf::{Graph, Term, XSD_BOOLEAN, XSD_INTEGER, XSD_STRING};
use crate::ontology::shapes::{self, NodeShape, Severity, RDF_NS, XSD_NS};

/// The embedded translations: `(id, yaml)`.
pub const CASES: &[(&str, &str)] = &[
    (
        "property/class-001",
        include_str!("../../w3c/property-class-001.yaml"),
    ),
    (
        "property/datatype-001",
        include_str!("../../w3c/property-datatype-001.yaml"),
    ),
    (
        "property/datatype-002",
        include_str!("../../w3c/property-datatype-002.yaml"),
    ),
    (
        "property/datatype-ill-formed",
        include_str!("../../w3c/property-datatype-ill-formed.yaml"),
    ),
    (
        "property/minCount-001",
        include_str!("../../w3c/property-minCount-001.yaml"),
    ),
    (
        "property/minCount-002",
        include_str!("../../w3c/property-minCount-002.yaml"),
    ),
    (
        "property/maxCount-001",
        include_str!("../../w3c/property-maxCount-001.yaml"),
    ),
    (
        "property/maxCount-002",
        include_str!("../../w3c/property-maxCount-002.yaml"),
    ),
    (
        "property/in-001",
        include_str!("../../w3c/property-in-001.yaml"),
    ),
    (
        "property/pattern-001",
        include_str!("../../w3c/property-pattern-001.yaml"),
    ),
    (
        "property/minLength-001",
        include_str!("../../w3c/property-minLength-001.yaml"),
    ),
    (
        "property/maxLength-001",
        include_str!("../../w3c/property-maxLength-001.yaml"),
    ),
    (
        "property/node-001",
        include_str!("../../w3c/property-node-001.yaml"),
    ),
    (
        "property/node-002",
        include_str!("../../w3c/property-node-002.yaml"),
    ),
    (
        "node/closed-002",
        include_str!("../../w3c/node-closed-002.yaml"),
    ),
    (
        "targets/targetClass-001",
        include_str!("../../w3c/targets-targetClass-001.yaml"),
    ),
    // #3611: the property-pair components, added after ONT-0's table of 32
    (
        "property/lessThan-001",
        include_str!("../../w3c/property-lessThan-001.yaml"),
    ),
    (
        "property/lessThan-002",
        include_str!("../../w3c/property-lessThan-002.yaml"),
    ),
    (
        "property/lessThanOrEquals-001",
        include_str!("../../w3c/property-lessThanOrEquals-001.yaml"),
    ),
    (
        "property/minExclusive-001",
        include_str!("../../w3c/property-minExclusive-001.yaml"),
    ),
    (
        "property/minExclusive-002",
        include_str!("../../w3c/property-minExclusive-002.yaml"),
    ),
    (
        "property/maxExclusive-001",
        include_str!("../../w3c/property-maxExclusive-001.yaml"),
    ),
    (
        "property/maxInclusive-001",
        include_str!("../../w3c/property-maxInclusive-001.yaml"),
    ),
    (
        "property/equals-001",
        include_str!("../../w3c/property-equals-001.yaml"),
    ),
    (
        "property/disjoint-001",
        include_str!("../../w3c/property-disjoint-001.yaml"),
    ),
    (
        "property/hasValue-001",
        include_str!("../../w3c/property-hasValue-001.yaml"),
    ),
];

/// Every case id of the W3C SHACL Core suite, as vendored in `w3c/core-suite.txt` (the header says how it was
/// derived from the suite's manifests).
pub const CORE_SUITE: &str = include_str!("../../w3c/core-suite.txt");

/// The ids of [`CORE_SUITE`], comment lines skipped.
pub fn core_suite_ids() -> impl Iterator<Item = &'static str> {
    CORE_SUITE
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
}

/// The cases of the W3C Core suite that the subset cannot run, each with the reason — measured at this row, not
/// guessed: ONT-0 counted 32 by component name; 16 of them use a FORM the subset does not attach. #4814 added the
/// other 63 suite cases, so every id in [`CORE_SUITE`] is in [`CASES`] or here, and this list only shrinks.
pub const NOT_VENDORED: &[(&str, &str)] = &[
    ("node/class-001", "sh:class on the node shape itself; the subset attaches constraints to property shapes"),
    ("node/class-002", "sh:class on the node shape itself"),
    ("node/class-003", "sh:class on the node shape itself"),
    ("node/datatype-001", "sh:datatype on the node shape itself"),
    ("node/datatype-002", "sh:datatype on the node shape itself"),
    ("node/in-001", "sh:in on the node shape itself"),
    ("node/pattern-001", "sh:pattern on the node shape itself"),
    ("node/pattern-002", "sh:pattern on the node shape itself"),
    ("node/minLength-001", "sh:minLength on the node shape itself"),
    ("node/maxLength-001", "sh:maxLength on the node shape itself"),
    ("node/nodeKind-001", "sh:nodeKind on the node shape itself, with sh:BlankNode variants"),
    ("node/node-001", "sh:node on the node shape itself"),
    ("node/closed-001", "expects an rdf:type violation under sh:closed; the subset always admits rdf:type on a closed shape (every extracted node is typed, §3.6) — closed-002, with sh:ignoredProperties (rdf:type), is the vendored form"),
    ("property/datatype-003", "sh:or — outside the subset, refused by name"),
    ("property/nodeKind-001", "data blank nodes and sh:BlankNode / sh:IRIOrLiteral kinds — the graph has no blank node (R-15) and the subset knows IRI and Literal"),
    ("property/pattern-002", "sh:flags — outside the subset, refused by name"),
    // #4814 slice 0: the other 63 cases of the W3C Core suite (98 in all), so every suite case is in one list or the
    // other. The parenthesised slice is the #4814 step-3 slice that would vendor it (evidence/4814/step3-design.md).
    ("complex/personexample", "sh:inversePath (slice 6)"),
    ("complex/shacl-shacl", "the SHACL-for-SHACL shapes graph, read as RDF; the gate path has no Turtle reader (R-13) — permanent"),
    ("misc/deactivated-001", "sh:deactivated — refused by name (slice 11)"),
    ("misc/deactivated-002", "sh:deactivated on the node shape itself (slices 5, 11)"),
    ("misc/message-001", "sh:message and sh:resultMessage — refused by name (slice 11)"),
    ("misc/severity-001", "sh:datatype on the node shape itself (slice 5)"),
    ("misc/severity-002", "sh:Info and sh:BlankNode — refused by name (slice 11)"),
    ("node/and-001", "sh:and — refused by name (slice 8)"),
    ("node/and-002", "sh:and — refused by name (slice 8)"),
    ("node/disjoint-001", "sh:disjoint on the node shape itself (slices 2, 5)"),
    ("node/equals-001", "sh:equals on the node shape itself (slices 2, 5)"),
    ("node/hasValue-001", "sh:hasValue on the node shape itself (slices 3, 5)"),
    ("node/languageIn-001", "sh:languageIn on the node shape itself; the term model drops language tags (slices 5, 10)"),
    ("node/maxExclusive-001", "sh:maxExclusive on the node shape itself (slice 5; the component is slice 1)"),
    ("node/maxInclusive-001", "sh:maxInclusive on the node shape itself (slice 5; the component is slice 1)"),
    ("node/minExclusive-001", "sh:minExclusive on the node shape itself (slice 5; the component is slice 1)"),
    ("node/minInclusive-001", "sh:minInclusive on the node shape itself (slice 5; the component is slice 1)"),
    ("node/minInclusive-002", "sh:minInclusive on the node shape itself (slice 5; the component is slice 1)"),
    ("node/minInclusive-003", "sh:minInclusive on the node shape itself (slice 5; the component is slice 1)"),
    ("node/not-001", "sh:not — refused by name (slice 8)"),
    ("node/not-002", "sh:not — refused by name (slice 8)"),
    ("node/or-001", "sh:or — refused by name (slice 8)"),
    ("node/xone-001", "sh:xone — refused by name (slice 8)"),
    ("node/xone-duplicate", "sh:xone — refused by name (slice 8)"),
    ("node/qualified-001", "sh:qualifiedValueShape on the node shape itself (slice 9)"),
    ("path/path-alternative-001", "sh:alternativePath (slice 7)"),
    ("path/path-complex-001", "sh:zeroOrMorePath and sh:hasValue (slices 3, 7)"),
    ("path/path-complex-002", "sh:inversePath (slice 6)"),
    ("path/path-inverse-001", "sh:inversePath (slice 6)"),
    ("path/path-oneOrMore-001", "sh:oneOrMorePath (slice 7)"),
    ("path/path-sequence-001", "a sequence path (slice 7)"),
    ("path/path-sequence-002", "a sequence path (slice 7)"),
    ("path/path-sequence-duplicate-001", "a sequence path (slice 7)"),
    ("path/path-strange-001", "sh:inversePath (slice 6)"),
    ("path/path-strange-002", "sh:inversePath (slice 6)"),
    ("path/path-zeroOrMore-001", "sh:zeroOrMorePath (slice 7)"),
    ("path/path-zeroOrOne-001", "sh:zeroOrOnePath (slice 7)"),
    ("path/path-unused-001", "expects an ill-formed path in an unused shape to be ignored; pv refuses every ill-formed path at parse, by design (fail closed) — permanent"),
    ("property/and-001", "sh:and — refused by name (slice 8)"),
    ("property/languageIn-001", "sh:languageIn; the term model drops language tags (slice 10)"),
    ("property/not-001", "sh:not — refused by name (slice 8)"),
    ("property/or-001", "sh:or — refused by name (slice 8)"),
    ("property/or-datatypes-001", "sh:or — refused by name (slice 8)"),
    ("property/property-001", "sh:property nested in a property shape; the subset nests only sh:node, one level (slice 8)"),
    ("property/qualifiedMinCountDisjoint-001", "sh:qualifiedValueShape — refused by name (slice 9)"),
    ("property/qualifiedValueShape-001", "sh:qualifiedValueShape — refused by name (slice 9)"),
    ("property/qualifiedValueShapesDisjoint-001", "sh:qualifiedValueShape — refused by name (slice 9)"),
    ("property/uniqueLang-001", "sh:uniqueLang; the term model drops language tags (slice 10)"),
    ("property/uniqueLang-002", "sh:uniqueLang; the term model drops language tags (slice 10)"),
    ("targets/multipleTargets-001", "sh:targetSubjectsOf (slice 4)"),
    ("targets/targetClassImplicit-001", "an implicit class target (a shape that is also an rdfs:Class) (slice 4)"),
    ("targets/targetNode-001", "tests sh:targetNode itself; the targetNode-to-class rewrite would make it vacuous (slice 4)"),
    ("targets/targetObjectsOf-001", "sh:targetObjectsOf (slice 4)"),
    ("targets/targetSubjectsOf-001", "sh:targetSubjectsOf (slice 4)"),
    ("targets/targetSubjectsOf-002", "sh:targetSubjectsOf (slice 4)"),
    ("validation-reports/shared", "one property shape reached from two node shapes; the YAML dialect has no shared shape reference (slice 8)"),
];

/// One expected `sh:ValidationResult`, as the case states it (component as W3C's short name, lower camel).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Expected {
    pub focus: String,
    pub path: Option<String>,
    pub component: String,
}

/// The outcome of one case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseResult {
    pub id: String,
    pub passed: bool,
    /// What differed, when it did.
    pub detail: String,
}

/// The run over every embedded case.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct W3cRun {
    pub results: Vec<CaseResult>,
}

impl W3cRun {
    #[must_use]
    pub fn passed(&self) -> usize {
        self.results.iter().filter(|r| r.passed).count()
    }
    #[must_use]
    pub fn failed(&self) -> Vec<String> {
        self.results
            .iter()
            .filter(|r| !r.passed)
            .map(|r| format!("{}: {}", r.id, r.detail))
            .collect()
    }
}

/// Why a case could not even be set up (a translation error, never a validator verdict).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CaseError(pub String);

impl std::fmt::Display for CaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A parsed case.
pub struct Case {
    pub id: String,
    pub prefix: String,
    pub shapes: Vec<NodeShape>,
    pub graph: Graph,
    pub conforms: bool,
    pub expected: Vec<Expected>,
}

/// The fixed prefixes every case may use beside its own `ex:`.
fn fixed_prefix(p: &str) -> Option<&'static str> {
    match p {
        "rdf" => Some(RDF_NS),
        "rdfs" => Some("http://www.w3.org/2000/01/rdf-schema#"),
        "xsd" => Some(XSD_NS),
        "owl" => Some("http://www.w3.org/2002/07/owl#"),
        "sh" => Some("http://www.w3.org/ns/shacl#"),
        _ => None,
    }
}

/// `ex:x` → `<prefix>x`; `rdf:type` → its IRI; a full IRI as is; anything else unchanged.
fn expand_term(s: &str, prefix: &str) -> String {
    if s.starts_with("http://") || s.starts_with("https://") {
        return s.to_string();
    }
    match s.split_once(':') {
        Some(("ex", local)) => format!("{prefix}{local}"),
        Some((p, local)) => {
            fixed_prefix(p).map_or_else(|| s.to_string(), |ns| format!("{ns}{local}"))
        }
        None => s.to_string(),
    }
}

/// Expand `ex:` / `rdfs:` / `owl:` / `sh:` inside a shape document's string values, recursively, so the shape
/// parser (which knows `ont:`, `xsd:`, `rdf:`, `prov:`) sees full IRIs for the case's own terms.
fn expand_doc(v: &serde_yaml::Value, prefix: &str) -> serde_yaml::Value {
    match v {
        serde_yaml::Value::String(s) => {
            let e = expand_term(s, prefix);
            serde_yaml::Value::String(if e == *s && s.starts_with("xsd:") {
                s.clone()
            } else {
                e
            })
        }
        serde_yaml::Value::Sequence(seq) => {
            serde_yaml::Value::Sequence(seq.iter().map(|x| expand_doc(x, prefix)).collect())
        }
        serde_yaml::Value::Mapping(m) => serde_yaml::Value::Mapping(
            m.iter()
                .map(|(k, x)| (k.clone(), expand_doc(x, prefix)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// One N-Triples-shaped line with the case's prefixes: `<s> <p> <o> .` where a term is `ex:x`, `rdf:type`, a
/// full `<iri>`, `"literal"`, `"literal"^^xsd:t`, `"literal"@lang`, a bare integer, decimal or boolean.
fn parse_line(line: &str, prefix: &str) -> Result<Option<(String, String, Term)>, CaseError> {
    let t = line.trim();
    if t.is_empty() || t.starts_with('#') {
        return Ok(None);
    }
    let body = t.strip_suffix('.').unwrap_or(t).trim();
    let mut it = body.splitn(3, char::is_whitespace);
    let (Some(s), Some(p), Some(o)) = (it.next(), it.next(), it.next()) else {
        return Err(CaseError(format!(
            "data line has fewer than three terms: `{t}`"
        )));
    };
    let subject = iri_term(s, prefix);
    let predicate = iri_term(p, prefix);
    let object = object_term(o.trim(), prefix)?;
    Ok(Some((subject, predicate, object)))
}

fn iri_term(s: &str, prefix: &str) -> String {
    let s = s.trim();
    let s = s
        .strip_prefix('<')
        .and_then(|x| x.strip_suffix('>'))
        .unwrap_or(s);
    expand_term(s, prefix)
}

fn object_term(o: &str, prefix: &str) -> Result<Term, CaseError> {
    if let Some(rest) = o.strip_prefix('"') {
        let Some(end) = rest.rfind('"') else {
            return Err(CaseError(format!("unterminated literal `{o}`")));
        };
        let value = rest[..end].to_string();
        let tail = &rest[end + 1..];
        let datatype = if let Some(dt) = tail.strip_prefix("^^") {
            iri_term(dt, prefix)
        } else if tail.starts_with('@') {
            format!("{RDF_NS}langString")
        } else {
            XSD_STRING.to_string()
        };
        return Ok(Term::Literal { value, datatype });
    }
    if o == "true" || o == "false" {
        return Ok(Term::Literal {
            value: o.to_string(),
            datatype: XSD_BOOLEAN.to_string(),
        });
    }
    if o.parse::<i64>().is_ok() {
        return Ok(Term::Literal {
            value: o.to_string(),
            datatype: XSD_INTEGER.to_string(),
        });
    }
    if o.parse::<f64>().is_ok() && o.contains('.') {
        return Ok(Term::Literal {
            value: o.to_string(),
            datatype: format!("{XSD_NS}decimal"),
        });
    }
    Ok(Term::iri(iri_term(o, prefix)))
}

/// Parse one embedded case.
pub fn parse_case(id: &str, yaml: &str) -> Result<Case, CaseError> {
    let doc: serde_yaml::Value =
        serde_yaml::from_str(yaml).map_err(|e| CaseError(format!("{id}: {e}")))?;
    let prefix = doc
        .get("prefix")
        .and_then(serde_yaml::Value::as_str)
        .ok_or_else(|| CaseError(format!("{id}: no `prefix`")))?
        .to_string();
    let shapes_doc = doc
        .get("shapes")
        .ok_or_else(|| CaseError(format!("{id}: no `shapes`")))?;
    let shapes_doc = expand_doc(shapes_doc, &prefix);
    let mut wrapper = serde_yaml::Mapping::new();
    wrapper.insert("shapes".into(), shapes_doc);
    let shapes = shapes::parse_shapes(id, &serde_yaml::Value::Mapping(wrapper))
        .map_err(|e| CaseError(format!("{id}: {e}")))?;
    let data = doc
        .get("data")
        .and_then(serde_yaml::Value::as_str)
        .ok_or_else(|| CaseError(format!("{id}: no `data`")))?;
    let mut graph = Graph::new();
    for line in data.lines() {
        if let Some((s, p, o)) = parse_line(line, &prefix)? {
            graph.insert(s, p, o);
        }
    }
    let expect = doc
        .get("expect")
        .ok_or_else(|| CaseError(format!("{id}: no `expect`")))?;
    let conforms = expect
        .get("conforms")
        .and_then(serde_yaml::Value::as_bool)
        .ok_or_else(|| CaseError(format!("{id}: `expect.conforms` is not a boolean")))?;
    let mut expected = Vec::new();
    if let Some(list) = expect
        .get("results")
        .and_then(serde_yaml::Value::as_sequence)
    {
        for r in list {
            let focus = r
                .get("focus")
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or_default();
            let path = r.get("path").and_then(serde_yaml::Value::as_str);
            let component = r
                .get("component")
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or_default();
            expected.push(Expected {
                focus: expand_term(focus, &prefix),
                path: path.map(|p| expand_term(p, &prefix)),
                component: component.to_string(),
            });
        }
    }
    expected.sort();
    Ok(Case {
        id: id.to_string(),
        prefix,
        shapes,
        graph,
        conforms,
        expected,
    })
}

/// Run one case through the in-house validator.
#[must_use]
pub fn run_case(case: &Case) -> CaseResult {
    let report = shapes::validate(&case.graph, &case.shapes);
    let mut got: Vec<Expected> = report
        .results
        .iter()
        .filter(|r| r.severity == Severity::Violation)
        .map(|r| Expected {
            focus: r.focus.clone(),
            path: r.path.clone(),
            component: r.component.to_string(),
        })
        .collect();
    got.sort();
    let conforms = got.is_empty();
    if conforms != case.conforms {
        return CaseResult {
            id: case.id.clone(),
            passed: false,
            detail: format!(
                "conforms {} (expected {}); got {}",
                conforms,
                case.conforms,
                render(&got, &case.prefix)
            ),
        };
    }
    if got != case.expected {
        return CaseResult {
            id: case.id.clone(),
            passed: false,
            detail: format!(
                "results differ: got {} expected {}",
                render(&got, &case.prefix),
                render(&case.expected, &case.prefix)
            ),
        };
    }
    CaseResult {
        id: case.id.clone(),
        passed: true,
        detail: String::new(),
    }
}

fn render(list: &[Expected], prefix: &str) -> String {
    let short = |s: &str| {
        s.strip_prefix(prefix)
            .map_or_else(|| s.to_string(), |l| format!("ex:{l}"))
    };
    let items: Vec<String> = list
        .iter()
        .map(|e| {
            format!(
                "({}, {}, {})",
                short(&e.focus),
                e.path.as_deref().map_or("-".to_string(), short),
                e.component
            )
        })
        .collect();
    format!("[{}]", items.join(" "))
}

/// Every embedded case. A case that does not parse is a failed case naming the translation error — a
/// vendored file that cannot be read must never count as passed.
#[must_use]
pub fn run_all() -> W3cRun {
    let mut run = W3cRun::default();
    for (id, yaml) in CASES {
        match parse_case(id, yaml) {
            Ok(case) => run.results.push(run_case(&case)),
            Err(e) => run.results.push(CaseResult {
                id: (*id).to_string(),
                passed: false,
                detail: e.to_string(),
            }),
        }
    }
    run
}

/// The per-component tally over the embedded cases, for the receipt.
#[must_use]
pub fn components_covered() -> BTreeMap<String, usize> {
    let mut m = BTreeMap::new();
    for (id, _) in CASES {
        let component = id
            .rsplit('/')
            .next()
            .unwrap_or(id)
            .split('-')
            .next()
            .unwrap_or(id)
            .to_string();
        *m.entry(component).or_insert(0) += 1;
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_embedded_case_parses_and_passes() {
        let run = run_all();
        assert_eq!(run.results.len(), CASES.len());
        assert_eq!(run.failed(), Vec::<String>::new());
        assert_eq!(run.passed(), CASES.len());
    }

    #[test]
    fn the_table_of_ont_0_is_accounted_for_case_by_case() {
        // #4814 slice 0: the denominator is the W3C suite itself (98 cases at 976ed12ad3), read from the vendored
        // id list — no longer the 35 cases of ONT-0 + #3611. Every suite case is vendored or excused with a reason,
        // never both, and neither list names a case the suite does not have.
        let suite: Vec<&str> = core_suite_ids().collect();
        let mut sorted = suite.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            suite.len(),
            "the suite list names no case twice"
        );
        assert_eq!(suite.len(), 98, "w3c/data-shapes 976ed12ad3: complex 2, misc 5, node 32, path 13, property 38, targets 7, validation-reports 1");
        let mut ids: Vec<&str> = CASES
            .iter()
            .map(|(id, _)| *id)
            .chain(NOT_VENDORED.iter().map(|(id, _)| *id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(
            ids.len(),
            CASES.len() + NOT_VENDORED.len(),
            "no case is both vendored and excused, nor listed twice"
        );
        assert_eq!(
            ids, sorted,
            "CASES and NOT_VENDORED together are exactly the suite"
        );
        for (id, why) in NOT_VENDORED {
            assert!(!why.trim().is_empty(), "{id} is excused without a reason");
        }
        // the ratchet (#4814 plan step 4): the vendored count only grows from its measured value
        assert!(
            CASES.len() >= 26,
            "vendored W3C cases dropped below 26: the #4814 baseline 19, slice 1's four value-range cases, slice 2's equals and disjoint, slice 3's hasValue"
        );
    }

    #[test]
    fn a_data_line_reads_every_term_form() {
        let p = "http://x/#";
        let (s, pr, o) = parse_line("ex:a rdf:type ex:B .", p)
            .expect("ok")
            .expect("a triple");
        assert_eq!(s, "http://x/#a");
        assert_eq!(pr, format!("{RDF_NS}type"));
        assert_eq!(o, Term::iri("http://x/#B"));
        let (_, _, o) = parse_line("ex:a ex:p \"A\"@en .", p)
            .expect("ok")
            .expect("t");
        assert_eq!(
            o.as_literal().map(|(_, d)| d),
            Some(format!("{RDF_NS}langString").as_str())
        );
        let (_, _, o) = parse_line("ex:a ex:p 11.1 .", p).expect("ok").expect("t");
        assert_eq!(
            o.as_literal(),
            Some(("11.1", format!("{XSD_NS}decimal").as_str()))
        );
        let (_, _, o) = parse_line("ex:a ex:p \"300\"^^xsd:byte .", p)
            .expect("ok")
            .expect("t");
        assert_eq!(
            o.as_literal().map(|(_, d)| d),
            Some(format!("{XSD_NS}byte").as_str())
        );
        assert!(parse_line("ex:a ex:p", p).is_err());
        assert!(parse_line("# comment", p).expect("ok").is_none());
    }

    #[test]
    fn a_mistranslated_expectation_fails_the_case_and_names_the_difference() {
        let (id, yaml) = CASES[0];
        let broken = yaml.replace("conforms: false", "conforms: true");
        let case = parse_case(id, &broken).expect("parses");
        let r = run_case(&case);
        assert!(!r.passed);
        assert!(
            r.detail.contains("conforms false (expected true)"),
            "{}",
            r.detail
        );
    }
}
