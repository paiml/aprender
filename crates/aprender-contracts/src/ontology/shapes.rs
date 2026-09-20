//! ONT-001 §3.6, §5 ONT-4b — the in-house shapes validator: the SHACL Core subset, exactly.
//!
//! Shapes are **authored in a contract's `shape:` block** (YAML — the only syntax the gate path parses) and applied
//! to the graph `extract:pv-contract` produces. The subset is §3.6's table and nothing more at v1alpha1:
//!
//! | implemented | refused at parse (`error: shape uses unsupported <x>`, exit 3) |
//! |---|---|
//! | `targetClass` | `targetNode`, `targetSubjectsOf`, `targetObjectsOf` |
//! | `minCount`, `maxCount` | the `qualifiedValueShape` family |
//! | `datatype`, `class`, `nodeKind` | — |
//! | `in`, `pattern`, `minLength`, `maxLength` | `languageIn`, `uniqueLang` |
//! | `node` (one level) | recursive / cyclic shapes |
//! | `closed`, `ignoredProperties` | — |
//! | a single predicate `path` | sequence, alternative, inverse, `*`/`+` paths |
//! | — | `and`/`or`/`not`/`xone`, `sparql`, and every component not in this table |
//!
//! A key the table does not name is REFUSED, not ignored: an ignored constraint is a shape that reports "conforms"
//! for something it never checked, which is the vacuity R-2 exists to end. `resolves:` is ours, not SHACL — it is
//! accepted here and checked by the extractor (§3.6), so a shape may carry it.
//!
//! Prefixes: `ont:` → `https://ont.paiml.dev/v1alpha1/`, `xsd:`, `rdf:`, `prov:` → their namespaces, and any other
//! `p:name` → `https://ont.paiml.dev/v1alpha1/p/name` (the per-entity-type vocabularies of §3.7: `readme:`, `model:`).
//! An IRI written in full (`http…`) is taken as is.
//!
//! The report is `sh:ValidationReport`-shaped: one result per (focus node, shape, path, component), each naming the
//! focus node and the shape that fired, so a corpus violation is a line a person can act on. Severity is
//! `Violation` unless the property shape says `severity: warning`.

use std::collections::{BTreeMap, BTreeSet};

use crate::ontology::rdf::{ont, Graph, Term, RDF_TYPE};

pub const XSD_NS: &str = "http://www.w3.org/2001/XMLSchema#";
pub const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
pub const PROV_NS: &str = "http://www.w3.org/ns/prov#";

/// Why a `shape:` block could not be read. Both are the DECLARATION's fault (exit 3), never a corpus verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// A component outside §3.6's implemented set.
    Unsupported { shape: String, component: String },
    /// A component in the set, written wrong (a non-integer `minCount`, a `pattern` that does not compile, …).
    Malformed { shape: String, what: String },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported { shape, component } => {
                write!(f, "shape {shape} uses unsupported {component}")
            }
            Self::Malformed { shape, what } => write!(f, "shape {shape} is malformed: {what}"),
        }
    }
}

impl std::error::Error for ShapeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Iri,
    Literal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Warning,
    Violation,
}

/// `xsd:string`, spelled once.
pub const XSD_STRING_IRI: &str = "http://www.w3.org/2001/XMLSchema#string";

/// One entry of a `sh:in` list, as a TERM: its lexical form and the datatype the YAML scalar gave it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InEntry {
    pub lexical: String,
    pub datatype: String,
}

impl InEntry {
    /// Does `value` (a literal's lexical form and datatype) equal this term?
    #[must_use]
    pub fn matches_literal(&self, value: &str, datatype: &str) -> bool {
        self.lexical == value && self.datatype == datatype
    }
    /// Does `iri` equal this entry, taken as an IRI (prefixed or written in full)?
    #[must_use]
    pub fn matches_iri(&self, iri: &str) -> bool {
        self.lexical == iri || expand(&self.lexical) == iri
    }
}

impl std::fmt::Display for InEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.datatype == XSD_STRING_IRI {
            write!(f, "{:?}", self.lexical)
        } else {
            write!(f, "{:?}^^{}", self.lexical, short(&self.datatype))
        }
    }
}

/// One `properties[]` entry: a single-predicate path and its constraints.
#[derive(Debug, Clone)]
pub struct PropertyShape {
    pub path: String,
    pub min_count: Option<usize>,
    pub max_count: Option<usize>,
    pub datatype: Option<String>,
    pub class: Option<String>,
    pub node_kind: Option<NodeKind>,
    pub r#in: Option<Vec<InEntry>>,
    pub pattern: Option<(String, regex::Regex)>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub node: Option<Box<NodeShape>>,
    pub resolves: Option<String>,
    pub severity: Severity,
}

/// A node shape: a target class, an open/closed switch, and its property shapes.
#[derive(Debug, Clone)]
pub struct NodeShape {
    /// The contract that declares it (its stem); nested `node` shapes are `<stem>/node`.
    pub id: String,
    pub target_class: String,
    pub closed: bool,
    pub ignored_properties: Vec<String>,
    pub properties: Vec<PropertyShape>,
}

/// One `sh:ValidationResult`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ValidationResult {
    pub severity: Severity,
    pub focus: String,
    pub shape: String,
    pub path: Option<String>,
    pub component: &'static str,
    pub message: String,
}

/// The `sh:ValidationReport`: `conforms` iff no result has `Violation` severity.
#[derive(Debug, Clone, Default)]
pub struct Report {
    pub results: Vec<ValidationResult>,
    pub focus_nodes_n: usize,
}

impl Report {
    #[must_use]
    pub fn violations(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.severity == Severity::Violation)
            .count()
    }
    #[must_use]
    pub fn warnings(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.severity == Severity::Warning)
            .count()
    }
    #[must_use]
    pub fn conforms(&self) -> bool {
        self.violations() == 0
    }
}

/// Expand a prefixed name to an IRI (see the module doc for the rule).
#[must_use]
pub fn expand(name: &str) -> String {
    if name.starts_with("http://") || name.starts_with("https://") {
        return name.to_string();
    }
    match name.split_once(':') {
        Some(("ont", local)) => ont(local),
        Some(("xsd", local)) => format!("{XSD_NS}{local}"),
        Some(("rdf", local)) => format!("{RDF_NS}{local}"),
        Some(("prov", local)) => format!("{PROV_NS}{local}"),
        Some((prefix, local)) => format!("{}{prefix}/{local}", crate::ontology::rdf::ONT_BASE),
        None => ont(name),
    }
}

const NODE_KEYS: &[&str] = &["targetClass", "closed", "ignoredProperties", "properties"];
const PROPERTY_KEYS: &[&str] = &[
    "path",
    "minCount",
    "maxCount",
    "datatype",
    "class",
    "nodeKind",
    "in",
    "pattern",
    "minLength",
    "maxLength",
    "node",
    "resolves",
    "severity",
];

/// Read a contract's `shape:` block. `None` when the contract has no `shape:`. The target class defaults to
/// `ont:Contract` when the contract's `entity.type` is `pv-contract`; otherwise `targetClass` is required.
pub fn parse_shape(stem: &str, doc: &serde_yaml::Value) -> Result<Option<NodeShape>, ShapeError> {
    let Some(block) = doc.get("shape") else {
        return Ok(None);
    };
    let Some(map) = block.as_mapping() else {
        return Err(ShapeError::Malformed {
            shape: stem.to_string(),
            what: "`shape:` is not a mapping".into(),
        });
    };
    parse_node_shape(stem, map, default_target(doc), 0).map(Some)
}

/// Every shape a contract declares: its `shape:` block (id = the stem) and each entry of its `shapes:` list
/// (id = the entry's own `id`, required, so a contract may hold several shapes that are armed one by one —
/// ONT-4c1's `ladder-measured` and `ladder-green`). An entry without `id`, or an `id` that repeats within the
/// contract, is malformed.
pub fn parse_shapes(stem: &str, doc: &serde_yaml::Value) -> Result<Vec<NodeShape>, ShapeError> {
    let mut out = Vec::new();
    if let Some(s) = parse_shape(stem, doc)? {
        out.push(s);
    }
    let Some(list) = doc.get("shapes") else {
        return Ok(out);
    };
    let seq = list.as_sequence().ok_or_else(|| ShapeError::Malformed {
        shape: stem.to_string(),
        what: "`shapes:` is not a list".into(),
    })?;
    for (i, entry) in seq.iter().enumerate() {
        let map = entry.as_mapping().ok_or_else(|| ShapeError::Malformed {
            shape: stem.to_string(),
            what: format!("shapes[{i}] is not a mapping"),
        })?;
        let id = map
            .get("id")
            .and_then(serde_yaml::Value::as_str)
            .ok_or_else(|| ShapeError::Malformed {
                shape: stem.to_string(),
                what: format!("shapes[{i}] has no `id`"),
            })?;
        if out.iter().any(|s| s.id == id) {
            return Err(ShapeError::Malformed {
                shape: stem.to_string(),
                what: format!("shape id `{id}` repeats"),
            });
        }
        let mut body = map.clone();
        body.remove(serde_yaml::Value::String("id".into()));
        out.push(parse_node_shape(id, &body, default_target(doc), 0)?);
    }
    Ok(out)
}

fn default_target(doc: &serde_yaml::Value) -> Option<String> {
    doc.get("entity")
        .and_then(|e| e.get("type"))
        .and_then(serde_yaml::Value::as_str)
        .filter(|t| *t == "pv-contract")
        .map(|_| ont("Contract"))
}

fn parse_node_shape(
    id: &str,
    map: &serde_yaml::Mapping,
    default_target: Option<String>,
    depth: usize,
) -> Result<NodeShape, ShapeError> {
    for key in map.keys() {
        let k = key.as_str().unwrap_or("?");
        if !NODE_KEYS.contains(&k) {
            return Err(ShapeError::Unsupported {
                shape: id.to_string(),
                component: k.to_string(),
            });
        }
    }
    let target_class = match map.get("targetClass").and_then(serde_yaml::Value::as_str) {
        Some(t) => expand(t),
        None => match (depth, default_target) {
            (0, Some(t)) => t,
            (0, None) => {
                return Err(ShapeError::Malformed {
                    shape: id.to_string(),
                    what: "no `targetClass`, and the contract's entity is not pv-contract".into(),
                })
            }
            // a nested `node` shape applies to the value, whatever its class
            (_, _) => String::new(),
        },
    };
    let closed = match map.get("closed") {
        None => false,
        Some(v) => v.as_bool().ok_or_else(|| ShapeError::Malformed {
            shape: id.to_string(),
            what: "`closed` is not a bool".into(),
        })?,
    };
    let ignored_properties = match map.get("ignoredProperties") {
        None => Vec::new(),
        Some(v) => v
            .as_sequence()
            .ok_or_else(|| ShapeError::Malformed {
                shape: id.to_string(),
                what: "`ignoredProperties` is not a list".into(),
            })?
            .iter()
            .filter_map(serde_yaml::Value::as_str)
            .map(expand)
            .collect(),
    };
    let mut properties = Vec::new();
    if let Some(list) = map.get("properties") {
        let seq = list.as_sequence().ok_or_else(|| ShapeError::Malformed {
            shape: id.to_string(),
            what: "`properties` is not a list".into(),
        })?;
        for (i, p) in seq.iter().enumerate() {
            let pm = p.as_mapping().ok_or_else(|| ShapeError::Malformed {
                shape: id.to_string(),
                what: format!("properties[{i}] is not a mapping"),
            })?;
            properties.push(parse_property(id, pm, depth)?);
        }
    }
    Ok(NodeShape {
        id: id.to_string(),
        target_class,
        closed,
        ignored_properties,
        properties,
    })
}

fn parse_property(
    shape: &str,
    pm: &serde_yaml::Mapping,
    depth: usize,
) -> Result<PropertyShape, ShapeError> {
    let malformed = |what: String| ShapeError::Malformed {
        shape: shape.to_string(),
        what,
    };
    for key in pm.keys() {
        let k = key.as_str().unwrap_or("?");
        if !PROPERTY_KEYS.contains(&k) {
            return Err(ShapeError::Unsupported {
                shape: shape.to_string(),
                component: k.to_string(),
            });
        }
    }
    let path = pm
        .get("path")
        .and_then(serde_yaml::Value::as_str)
        .ok_or_else(|| malformed("a property has no `path`".into()))?;
    if path.contains(['/', '|', '^', '*', '+']) && !path.starts_with("http") {
        return Err(ShapeError::Unsupported {
            shape: shape.to_string(),
            component: format!("path `{path}` (only a single predicate is a path here)"),
        });
    }
    let count = |k: &str| -> Result<Option<usize>, ShapeError> {
        match pm.get(k) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| malformed(format!("`{k}` is not a non-negative integer"))),
        }
    };
    let iri_opt = |k: &str| pm.get(k).and_then(serde_yaml::Value::as_str).map(expand);
    let node_kind = parse_node_kind(
        shape,
        pm.get("nodeKind").and_then(serde_yaml::Value::as_str),
    )?;
    // `sh:in` is TERM equality (SHACL §4.5.1), and a term carries its datatype. The YAML scalar's own type is
    // what gives it one: `in: [true]` is `"true"^^xsd:boolean`, `in: [1]` is `xsd:integer`, `in: [a, b]` is
    // `xsd:string`. This used to collapse every entry to its lexical form and compare strings, so a shape
    // `in: ["true"]` accepted `"true"^^xsd:boolean` — which the pinned oracle refuses, and which `make oracle`
    // caught on this row's own shapes (490 results of difference on the real corpus). An entry that expands to
    // an IRI still matches an IRI value, because a `sh:in` over `nodeKind: IRI` is a list of IRIs.
    let r#in = match pm.get("in") {
        None => None,
        Some(v) => Some(
            v.as_sequence()
                .ok_or_else(|| malformed("`in` is not a list".into()))?
                .iter()
                .map(|x| match x {
                    serde_yaml::Value::String(s) => InEntry {
                        lexical: s.clone(),
                        datatype: XSD_STRING_IRI.to_string(),
                    },
                    serde_yaml::Value::Number(n) => InEntry {
                        lexical: n.to_string(),
                        datatype: if n.is_f64() {
                            format!("{XSD_NS}double")
                        } else {
                            format!("{XSD_NS}integer")
                        },
                    },
                    serde_yaml::Value::Bool(b) => InEntry {
                        lexical: b.to_string(),
                        datatype: format!("{XSD_NS}boolean"),
                    },
                    _ => InEntry {
                        lexical: String::new(),
                        datatype: XSD_STRING_IRI.to_string(),
                    },
                })
                .collect(),
        ),
    };
    let pattern = match pm.get("pattern").and_then(serde_yaml::Value::as_str) {
        None => None,
        Some(p) => Some((
            p.to_string(),
            regex::Regex::new(p)
                .map_err(|e| malformed(format!("`pattern` does not compile: {e}")))?,
        )),
    };
    let node = match pm.get("node") {
        None => None,
        Some(_) if depth >= 1 => {
            return Err(ShapeError::Unsupported {
                shape: shape.to_string(),
                component: "node (nested more than one level)".into(),
            })
        }
        Some(v) => {
            let nm = v
                .as_mapping()
                .ok_or_else(|| malformed("`node` is not a mapping".into()))?;
            Some(Box::new(parse_node_shape(
                &format!("{shape}/node"),
                nm,
                None,
                depth + 1,
            )?))
        }
    };
    let severity = parse_severity(
        shape,
        pm.get("severity").and_then(serde_yaml::Value::as_str),
    )?;
    Ok(PropertyShape {
        path: expand(path),
        min_count: count("minCount")?,
        max_count: count("maxCount")?,
        datatype: iri_opt("datatype"),
        class: iri_opt("class"),
        node_kind,
        r#in,
        pattern,
        min_length: count("minLength")?,
        max_length: count("maxLength")?,
        node,
        resolves: pm
            .get("resolves")
            .and_then(serde_yaml::Value::as_str)
            .map(String::from),
        severity,
    })
}

/// `nodeKind`: `IRI` or `Literal` (with or without the `sh:` prefix); anything else is outside the subset.
fn parse_node_kind(shape: &str, v: Option<&str>) -> Result<Option<NodeKind>, ShapeError> {
    match v {
        None => Ok(None),
        Some("IRI" | "sh:IRI") => Ok(Some(NodeKind::Iri)),
        Some("Literal" | "sh:Literal") => Ok(Some(NodeKind::Literal)),
        Some(other) => Err(ShapeError::Unsupported {
            shape: shape.to_string(),
            component: format!("nodeKind {other}"),
        }),
    }
}

/// `severity`: `violation` (the default) or `warning`; anything else is outside the subset.
fn parse_severity(shape: &str, v: Option<&str>) -> Result<Severity, ShapeError> {
    match v {
        None | Some("violation" | "Violation") => Ok(Severity::Violation),
        Some("warning" | "Warning") => Ok(Severity::Warning),
        Some(other) => Err(ShapeError::Unsupported {
            shape: shape.to_string(),
            component: format!("severity {other}"),
        }),
    }
}

/// `rdfs:subClassOf`.
pub const RDFS_SUBCLASS_OF: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";

/// The SHACL instance relation: `node rdf:type T` for some `T` that is `class` or an `rdfs:subClassOf`-ancestor
/// of it (SHACL §2.1.1, the W3C `class-001` cases). A graph without `subClassOf` triples — every graph the
/// extractors emit today — reduces to a direct `rdf:type` test.
#[must_use]
pub fn is_instance(graph: &Graph, node: &str, class: &str) -> bool {
    graph
        .objects(node, RDF_TYPE)
        .iter()
        .filter_map(|t| t.as_iri())
        .any(|t| t == class || is_subclass_of(graph, t, class))
}

/// `sub rdfs:subClassOf* class`, cycle-safe.
fn is_subclass_of(graph: &Graph, sub: &str, class: &str) -> bool {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut stack = vec![sub.to_string()];
    while let Some(c) = stack.pop() {
        if !seen.insert(c.clone()) {
            continue;
        }
        for sup in graph.objects(&c, RDFS_SUBCLASS_OF) {
            if let Some(s) = sup.as_iri() {
                if s == class {
                    return true;
                }
                stack.push(s.to_string());
            }
        }
    }
    false
}

/// The focus nodes of a class: its instances under [`is_instance`] (subclass instances included), in byte order.
/// One pass: the classes at or below `class` (the inverse `subClassOf` closure), then every `rdf:type` triple
/// whose object is one of them.
#[must_use]
pub fn instances_closed(graph: &Graph, class: &str) -> Vec<String> {
    let mut subs: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for t in graph.iter() {
        if t.predicate == RDFS_SUBCLASS_OF {
            if let Some(sup) = t.object.as_iri() {
                subs.entry(sup).or_default().push(t.subject.as_str());
            }
        }
    }
    let mut at_or_below: BTreeSet<&str> = BTreeSet::new();
    let mut stack = vec![class];
    while let Some(c) = stack.pop() {
        if !at_or_below.insert(c) {
            continue;
        }
        if let Some(children) = subs.get(c) {
            stack.extend(children.iter().copied());
        }
    }
    let out: BTreeSet<String> = graph
        .iter()
        .filter(|t| {
            t.predicate == RDF_TYPE && t.object.as_iri().is_some_and(|o| at_or_below.contains(o))
        })
        .map(|t| t.subject.clone())
        .collect();
    out.into_iter().collect()
}

/// Validate `graph` against `shapes`. Focus nodes of a shape are the instances of its target class.
#[must_use]
pub fn validate(graph: &Graph, shapes: &[NodeShape]) -> Report {
    let mut report = Report::default();
    let mut focus_seen: BTreeSet<String> = BTreeSet::new();
    for shape in shapes {
        for focus in instances_closed(graph, &shape.target_class) {
            let focus = focus.as_str();
            focus_seen.insert(focus.to_string());
            validate_focus(graph, shape, focus, &mut report.results);
        }
    }
    report.focus_nodes_n = focus_seen.len();
    report.results.sort();
    report.results.dedup();
    report
}

fn validate_focus(graph: &Graph, shape: &NodeShape, focus: &str, out: &mut Vec<ValidationResult>) {
    let mut push =
        |severity: Severity, path: Option<&str>, component: &'static str, message: String| {
            out.push(ValidationResult {
                severity,
                focus: focus.to_string(),
                shape: shape.id.clone(),
                path: path.map(String::from),
                component,
                message,
            });
        };
    for p in &shape.properties {
        let values = graph.objects(focus, &p.path);
        if let Some(min) = p.min_count {
            if values.len() < min {
                push(
                    p.severity,
                    Some(&p.path),
                    "minCount",
                    format!(
                        "has {} value(s) of {}, minCount is {min}",
                        values.len(),
                        short(&p.path)
                    ),
                );
            }
        }
        if let Some(max) = p.max_count {
            if values.len() > max {
                // name the values (up to five): a `maxCount 0` on a materialized edge — `missingGreenHost`,
                // `receiptHexMismatch` (ONT-4c1) — is only actionable when the message says WHICH host, WHICH file
                let named: Vec<String> = values
                    .iter()
                    .take(5)
                    .map(|v| match v {
                        Term::Iri(i) => short(i),
                        Term::Literal { value, .. } => value.clone(),
                    })
                    .collect();
                push(
                    p.severity,
                    Some(&p.path),
                    "maxCount",
                    format!(
                        "has {} value(s) of {}, maxCount is {max}: {}",
                        values.len(),
                        short(&p.path),
                        named.join(", ")
                    ),
                );
            }
        }
        for v in values {
            check_value(graph, p, v, &mut push);
        }
    }
    if shape.closed {
        let allowed: BTreeSet<&str> = shape
            .properties
            .iter()
            .map(|p| p.path.as_str())
            .chain(shape.ignored_properties.iter().map(String::as_str))
            .chain(std::iter::once(RDF_TYPE))
            .collect();
        for pred in graph.predicates_of(focus) {
            if !allowed.contains(pred) {
                push(
                    Severity::Violation,
                    Some(pred),
                    "closed",
                    format!(
                        "carries {}, which the closed shape does not declare",
                        short(pred)
                    ),
                );
            }
        }
    }
}

fn check_value(
    graph: &Graph,
    p: &PropertyShape,
    v: &Term,
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
    check_kind_and_type(graph, p, v, push);
    check_lexical(p, v, push);
    if let Some(inner) = &p.node {
        match v.as_iri() {
            Some(i) => {
                // One `sh:NodeConstraintComponent` result per VALUE that fails the nested shape, on the outer
                // path, with the nested findings as its detail (SHACL §4.6.2; W3C property/node-001, -002).
                let mut nested = Vec::new();
                validate_focus(graph, inner, i, &mut nested);
                let violations: Vec<String> = nested
                    .iter()
                    .filter(|r| r.severity == Severity::Violation)
                    .map(|r| r.message.clone())
                    .collect();
                if !violations.is_empty() {
                    push(
                        p.severity,
                        Some(&p.path),
                        "node",
                        format!(
                            "value {i} fails the nested shape: {}",
                            violations.join("; ")
                        ),
                    );
                }
            }
            None => push(
                p.severity,
                Some(&p.path),
                "node",
                format!("{v} is a literal; `node` needs an IRI"),
            ),
        }
    }
}

/// Is `value` a well-formed lexical form of the XSD `datatype`? A literal typed `xsd:byte` with the lexical
/// form `300` (or `c`) is ill-formed and violates `sh:datatype` (SHACL §4.1.2; W3C `datatype-ill-formed`). The
/// types checked are the ones the corpus and the vendored cases use; any other datatype is taken as
/// well-formed, because refusing what is not understood would be a verdict about a value never measured.
#[must_use]
pub fn well_formed(value: &str, datatype: &str) -> bool {
    let Some(local) = datatype.strip_prefix(XSD_NS) else {
        return true;
    };
    let int_in = |lo: i128, hi: i128| value.parse::<i128>().is_ok_and(|n| n >= lo && n <= hi);
    match local {
        "string" | "anyURI" => true,
        "boolean" => matches!(value, "true" | "false" | "1" | "0"),
        "integer" => value.parse::<i128>().is_ok(),
        "long" => int_in(i128::from(i64::MIN), i128::from(i64::MAX)),
        "int" => int_in(i128::from(i32::MIN), i128::from(i32::MAX)),
        "short" => int_in(i128::from(i16::MIN), i128::from(i16::MAX)),
        "byte" => int_in(i128::from(i8::MIN), i128::from(i8::MAX)),
        "nonNegativeInteger" => int_in(0, i128::MAX),
        "positiveInteger" => int_in(1, i128::MAX),
        "nonPositiveInteger" => int_in(i128::MIN, 0),
        "negativeInteger" => int_in(i128::MIN, -1),
        "unsignedLong" => int_in(0, i128::from(u64::MAX)),
        "unsignedInt" => int_in(0, i128::from(u32::MAX)),
        "unsignedShort" => int_in(0, i128::from(u16::MAX)),
        "unsignedByte" => int_in(0, i128::from(u8::MAX)),
        "decimal" => {
            !value.is_empty()
                && value
                    .strip_prefix(['+', '-'])
                    .unwrap_or(value)
                    .chars()
                    .all(|c| c.is_ascii_digit() || c == '.')
                && value.chars().filter(|c| *c == '.').count() <= 1
                && value.chars().any(|c| c.is_ascii_digit())
        }
        "double" | "float" => {
            matches!(value, "INF" | "-INF" | "NaN") || value.parse::<f64>().is_ok()
        }
        "date" => is_date(value),
        "dateTime" => value
            .split_once('T')
            .is_some_and(|(d, t)| is_date(d) && t.len() >= 8 && t.as_bytes()[2] == b':'),
        _ => true,
    }
}

/// `YYYY-MM-DD` with an optional timezone suffix.
fn is_date(s: &str) -> bool {
    let core = s.split(['Z', '+']).next().unwrap_or(s);
    let core = if core.len() > 10 { &core[..10] } else { core };
    let b = core.as_bytes();
    b.len() == 10
        && b[4] == b'-'
        && b[7] == b'-'
        && [0, 1, 2, 3, 5, 6, 8, 9]
            .iter()
            .all(|&i| b[i].is_ascii_digit())
        && (1..=12).contains(&core[5..7].parse::<u8>().unwrap_or(0))
        && (1..=31).contains(&core[8..10].parse::<u8>().unwrap_or(0))
}

/// `nodeKind`, `datatype`, `class` — the constraints about what KIND of term the value is.
fn check_kind_and_type(
    graph: &Graph,
    p: &PropertyShape,
    v: &Term,
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
    let path = Some(p.path.as_str());
    if let Some(kind) = p.node_kind {
        let ok = match kind {
            NodeKind::Iri => v.as_iri().is_some(),
            NodeKind::Literal => v.as_literal().is_some(),
        };
        if !ok {
            push(
                p.severity,
                path,
                "nodeKind",
                format!("{v} is not of nodeKind {kind:?}"),
            );
        }
    }
    if let Some(dt) = &p.datatype {
        match v.as_literal() {
            Some((value, actual)) if actual == dt && well_formed(value, dt) => {}
            Some((value, actual)) if actual == dt => push(
                p.severity,
                path,
                "datatype",
                format!("\"{value}\" is not a well-formed {}", short(dt)),
            ),
            _ => push(
                p.severity,
                path,
                "datatype",
                format!("{v} is not a {}", short(dt)),
            ),
        }
    }
    if let Some(class) = &p.class {
        let ok = v.as_iri().is_some_and(|i| is_instance(graph, i, class));
        if !ok {
            push(
                p.severity,
                path,
                "class",
                format!("{v} is not an instance of {}", short(class)),
            );
        }
    }
}

/// `in`, `pattern`, `minLength`, `maxLength` — the constraints on the value's lexical form.
fn check_lexical(
    p: &PropertyShape,
    v: &Term,
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
    let path = Some(p.path.as_str());
    let lexical: &str = match v {
        Term::Iri(i) => i.as_str(),
        Term::Literal { value, .. } => value.as_str(),
    };
    if let Some(allowed) = &p.r#in {
        let ok = match v {
            Term::Iri(i) => allowed.iter().any(|a| a.matches_iri(i)),
            Term::Literal { value, datatype } => {
                allowed.iter().any(|a| a.matches_literal(value, datatype))
            }
        };
        if !ok {
            // The PATH is in the message, not only in the result's `path` field: a reader who gets one line
            // ("quadratic is not one of …") cannot act on it without being told which property said it, and a
            // shape with nine properties produces nine indistinguishable lines (measured on apex's EV-21).
            // The value and the list are TERMS (ONT-4b2: `sh:in` is term equality, so `"1"^^xsd:integer` and
            // `"1"` are different entries), rendered as terms, not as bare lexical forms.
            let listed: Vec<String> = allowed.iter().map(ToString::to_string).collect();
            push(
                p.severity,
                path,
                "in",
                format!(
                    "{}: {} is not one of [{}]",
                    short(&p.path),
                    term_short(v),
                    listed.join(", ")
                ),
            );
        }
    }
    if let Some((src, re)) = &p.pattern {
        if !re.is_match(lexical) {
            push(
                p.severity,
                path,
                "pattern",
                format!("{}: {lexical:?} does not match /{src}/", short(&p.path)),
            );
        }
    }
    let len = lexical.chars().count();
    if p.min_length.is_some_and(|m| len < m) {
        push(
            p.severity,
            path,
            "minLength",
            format!("{}: length {len} is below minLength", short(&p.path)),
        );
    }
    if p.max_length.is_some_and(|m| len > m) {
        push(
            p.severity,
            path,
            "maxLength",
            format!("{}: length {len} is above maxLength", short(&p.path)),
        );
    }
}

/// A term as a message says it: `"ok"` for an xsd:string literal, `"300"^^xsd:byte` for any other typed one,
/// `ont:id` for an IRI. The datatype is shown only when it carries information — `sh:in` is term equality, so
/// a message that hid the datatype would name two different terms the same way.
#[must_use]
pub fn term_short(v: &Term) -> String {
    match v {
        Term::Iri(i) => short(i),
        Term::Literal { value, datatype } if datatype == XSD_STRING_IRI => format!("{value:?}"),
        Term::Literal { value, datatype } => format!("{value:?}^^{}", short(datatype)),
    }
}

/// `<https://ont.paiml.dev/v1alpha1/id>` → `ont:id`, for messages.
#[must_use]
pub fn short(iri: &str) -> String {
    for (ns, prefix) in [
        (crate::ontology::rdf::ONT_BASE, "ont"),
        (XSD_NS, "xsd"),
        (RDF_NS, "rdf"),
        (PROV_NS, "prov"),
    ] {
        if let Some(rest) = iri.strip_prefix(ns) {
            return format!("{prefix}:{rest}");
        }
    }
    iri.to_string()
}

/// The shapes as SHACL Turtle, for the oracle and for anyone running a real processor (§3.6). Deterministic.
#[must_use]
pub fn to_turtle(shapes: &[NodeShape]) -> String {
    let mut out = String::from(
        "@prefix sh: <http://www.w3.org/ns/shacl#> .\n@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n@prefix ont: <https://ont.paiml.dev/v1alpha1/> .\n\n",
    );
    for s in shapes {
        out.push_str(&turtle_node(
            s,
            &format!("<{}shape/{}>", crate::ontology::rdf::ONT_BASE, s.id),
        ));
    }
    out
}

fn turtle_node(s: &NodeShape, subject: &str) -> String {
    let mut o = format!("{subject} a sh:NodeShape ;\n");
    if !s.target_class.is_empty() {
        o.push_str(&format!("    sh:targetClass <{}> ;\n", s.target_class));
    }
    if s.closed {
        o.push_str("    sh:closed true ;\n");
        if !s.ignored_properties.is_empty() {
            let list: Vec<String> = s
                .ignored_properties
                .iter()
                .map(|p| format!("<{p}>"))
                .collect();
            o.push_str(&format!(
                "    sh:ignoredProperties ( {} ) ;\n",
                list.join(" ")
            ));
        }
    }
    for p in &s.properties {
        o.push_str(&turtle_property(p));
    }
    o.push_str(".\n\n");
    for inner in s.properties.iter().filter_map(|p| p.node.as_deref()) {
        o.push_str(&turtle_node(
            inner,
            &format!("<{}shape/{}>", crate::ontology::rdf::ONT_BASE, inner.id),
        ));
    }
    o
}

/// One `sh:property [ … ] ;` block. Every implemented component has a line; nothing else is emitted.
fn turtle_property(p: &PropertyShape) -> String {
    let mut o = String::from("    sh:property [\n");
    let mut line = |s: String| o.push_str(&format!("        {s} ;\n"));
    line(format!("sh:path <{}>", p.path));
    if let Some(n) = p.min_count {
        line(format!("sh:minCount {n}"));
    }
    if let Some(n) = p.max_count {
        line(format!("sh:maxCount {n}"));
    }
    if let Some(d) = &p.datatype {
        line(format!("sh:datatype <{d}>"));
    }
    if let Some(c) = &p.class {
        line(format!("sh:class <{c}>"));
    }
    if let Some(k) = p.node_kind {
        line(format!(
            "sh:nodeKind sh:{}",
            match k {
                NodeKind::Iri => "IRI",
                NodeKind::Literal => "Literal",
            }
        ));
    }
    if let Some(list) = &p.r#in {
        // Typed, because `sh:in` is term equality: an untyped `"true"` is an xsd:string and would not match
        // the xsd:boolean the extractor writes — the difference `make oracle` measures.
        let items: Vec<String> = list
            .iter()
            .map(|v| {
                if v.datatype == XSD_STRING_IRI {
                    format!("\"{}\"", v.lexical)
                } else {
                    format!("\"{}\"^^<{}>", v.lexical, v.datatype)
                }
            })
            .collect();
        line(format!("sh:in ( {} )", items.join(" ")));
    }
    if let Some((src, _)) = &p.pattern {
        line(format!(
            "sh:pattern \"{}\"",
            src.replace('\\', "\\\\").replace('"', "\\\"")
        ));
    }
    if let Some(n) = p.min_length {
        line(format!("sh:minLength {n}"));
    }
    if let Some(n) = p.max_length {
        line(format!("sh:maxLength {n}"));
    }
    if p.severity == Severity::Warning {
        line("sh:severity sh:Warning".to_string());
    }
    if let Some(inner) = &p.node {
        line(format!(
            "sh:node <{}shape/{}>",
            crate::ontology::rdf::ONT_BASE,
            inner.id
        ));
    }
    o.push_str("    ] ;\n");
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::rdf::{iri, Term};

    fn shape(yaml: &str) -> NodeShape {
        let doc: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        parse_shape("t", &doc).unwrap().unwrap()
    }

    fn graph_with(id: &str, kind: Option<&str>) -> Graph {
        let mut g = Graph::new();
        let s = iri("contract", id);
        g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Contract")));
        g.insert(s.clone(), ont("id"), Term::string(id));
        if let Some(k) = kind {
            g.insert(s, ont("kind"), Term::string(k));
        }
        g
    }

    const BASE: &str = "entity: {type: pv-contract}\nshape:\n  properties:\n    - {path: ont:id, minCount: 1, maxCount: 1, pattern: '^[a-z0-9-]+$'}\n    - {path: ont:kind, maxCount: 1, in: [kernel, pattern]}\n";

    #[test]
    fn a_conforming_focus_node_yields_no_result() {
        let r = validate(&graph_with("a-1", Some("kernel")), &[shape(BASE)]);
        assert!(r.conforms(), "{:?}", r.results);
        assert_eq!(r.focus_nodes_n, 1);
    }

    #[test]
    fn each_implemented_component_fires_and_names_focus_and_shape() {
        let r = validate(&graph_with("Bad Id", Some("novel")), &[shape(BASE)]);
        let mut comps: Vec<&str> = r.results.iter().map(|x| x.component).collect();
        comps.sort_unstable();
        assert_eq!(comps, vec!["in", "pattern"], "{:?}", r.results);
        assert!(r.results[0].focus.ends_with("/contract/Bad%20Id"));
        assert_eq!(r.results[0].shape, "t");
        let mut g = Graph::new();
        g.insert(iri("contract", "x"), RDF_TYPE, Term::iri(ont("Contract"))); // no ont:id at all
        let r = validate(&g, &[shape(BASE)]);
        assert_eq!(r.results[0].component, "minCount");
    }

    #[test]
    fn closed_rejects_an_undeclared_predicate_and_ignores_rdf_type() {
        let mut g = graph_with("a", None);
        g.insert(iri("contract", "a"), ont("extra"), Term::string("x"));
        let s = shape("entity: {type: pv-contract}\nshape:\n  closed: true\n  properties:\n    - {path: ont:id}\n");
        let r = validate(&g, &[s]);
        assert_eq!(r.violations(), 1);
        assert_eq!(r.results[0].component, "closed");
    }

    #[test]
    fn datatype_class_nodekind_and_lengths_fire() {
        let mut g = graph_with("a", None);
        let a = iri("contract", "a");
        g.insert(a.clone(), ont("n"), Term::string("7"));
        g.insert(a.clone(), ont("dep"), Term::string("not-an-iri"));
        let s = shape("entity: {type: pv-contract}\nshape:\n  properties:\n    - {path: ont:n, datatype: xsd:integer, maxLength: 0}\n    - {path: ont:dep, nodeKind: IRI, class: ont:Contract}\n");
        let r = validate(&g, &[s]);
        let mut comps: Vec<&str> = r.results.iter().map(|x| x.component).collect();
        comps.sort_unstable();
        assert_eq!(
            comps,
            vec!["class", "datatype", "maxLength", "nodeKind"],
            "{:?}",
            r.results
        );
    }

    #[test]
    fn a_warning_only_report_conforms_but_counts_the_warning() {
        let g = graph_with("a", Some("novel"));
        let s = shape("entity: {type: pv-contract}\nshape:\n  properties:\n    - {path: ont:kind, in: [kernel], severity: warning}\n");
        let r = validate(&g, &[s]);
        assert!(r.conforms());
        assert_eq!(r.warnings(), 1);
    }

    #[test]
    fn node_one_level_validates_the_value_and_two_levels_are_refused() {
        let mut g = graph_with("a", None);
        let a = iri("contract", "a");
        let b = iri("contract", "b");
        g.insert(a.clone(), ont("depends_on"), Term::iri(b.clone()));
        g.insert(b.clone(), RDF_TYPE, Term::iri(ont("Contract")));
        let s = shape("entity: {type: pv-contract}\nshape:\n  properties:\n    - {path: ont:depends_on, node: {properties: [{path: ont:id, minCount: 1}]}}\n");
        let r = validate(&g, &[s]);
        assert_eq!(r.results.len(), 1, "{:?}", r.results);
        assert_eq!(r.results[0].component, "node");
        let doc: serde_yaml::Value = serde_yaml::from_str("entity: {type: pv-contract}\nshape:\n  properties:\n    - {path: ont:x, node: {properties: [{path: ont:y, node: {properties: []}}]}}\n").unwrap();
        assert!(matches!(
            parse_shape("t", &doc),
            Err(ShapeError::Unsupported { .. })
        ));
    }

    #[test]
    fn unsupported_components_are_refused_at_parse_by_name() {
        for (yaml, want) in [
            ("shape:\n  targetNode: x\n", "targetNode"),
            ("entity: {type: pv-contract}\nshape:\n  properties: [{path: ont:x, qualifiedValueShape: {}}]\n", "qualifiedValueShape"),
            ("entity: {type: pv-contract}\nshape:\n  properties: [{path: ont:x, languageIn: [en]}]\n", "languageIn"),
            ("entity: {type: pv-contract}\nshape:\n  properties: [{path: 'ont:a/ont:b'}]\n", "path `ont:a/ont:b`"),
            ("entity: {type: pv-contract}\nshape:\n  or: []\n", "or"),
        ] {
            let doc: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
            match parse_shape("t", &doc) {
                Err(ShapeError::Unsupported { component, .. }) => assert!(component.starts_with(want), "{component} vs {want}"),
                other => panic!("{yaml}: expected Unsupported, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_shape_with_no_target_and_no_pv_contract_entity_is_malformed() {
        let doc: serde_yaml::Value = serde_yaml::from_str("shape:\n  properties: []\n").unwrap();
        assert!(matches!(
            parse_shape("t", &doc),
            Err(ShapeError::Malformed { .. })
        ));
        let doc: serde_yaml::Value =
            serde_yaml::from_str("shape:\n  targetClass: ont:Contract\n  properties: []\n")
                .unwrap();
        assert_eq!(
            parse_shape("t", &doc).unwrap().unwrap().target_class,
            ont("Contract")
        );
    }

    #[test]
    fn turtle_export_is_deterministic_and_names_every_component() {
        let s = shape(BASE);
        let t1 = to_turtle(std::slice::from_ref(&s));
        let t2 = to_turtle(std::slice::from_ref(&s));
        assert_eq!(t1, t2);
        for want in [
            "sh:NodeShape",
            "sh:targetClass",
            "sh:minCount 1",
            "sh:maxCount 1",
            "sh:pattern",
            "sh:in ( \"kernel\" \"pattern\" )",
        ] {
            assert!(t1.contains(want), "{want}\n{t1}");
        }
    }

    #[test]
    fn prefixes_expand_by_the_stated_rule() {
        assert_eq!(expand("ont:id"), "https://ont.paiml.dev/v1alpha1/id");
        assert_eq!(
            expand("xsd:integer"),
            "http://www.w3.org/2001/XMLSchema#integer"
        );
        assert_eq!(
            expand("readme:kind"),
            "https://ont.paiml.dev/v1alpha1/readme/kind"
        );
        assert_eq!(expand("https://x/y"), "https://x/y");
        assert_eq!(short("https://ont.paiml.dev/v1alpha1/id"), "ont:id");
    }
}
