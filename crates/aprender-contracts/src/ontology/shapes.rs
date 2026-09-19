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

use std::collections::BTreeSet;

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

/// One `properties[]` entry: a single-predicate path and its constraints.
#[derive(Debug, Clone)]
pub struct PropertyShape {
    pub path: String,
    pub min_count: Option<usize>,
    pub max_count: Option<usize>,
    pub datatype: Option<String>,
    pub class: Option<String>,
    pub node_kind: Option<NodeKind>,
    pub r#in: Option<Vec<String>>,
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
    let r#in = match pm.get("in") {
        None => None,
        Some(v) => Some(
            v.as_sequence()
                .ok_or_else(|| malformed("`in` is not a list".into()))?
                .iter()
                .map(|x| match x {
                    serde_yaml::Value::String(s) => s.clone(),
                    serde_yaml::Value::Number(n) => n.to_string(),
                    serde_yaml::Value::Bool(b) => b.to_string(),
                    _ => String::new(),
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

/// Validate `graph` against `shapes`. Focus nodes of a shape are the instances of its target class.
#[must_use]
pub fn validate(graph: &Graph, shapes: &[NodeShape]) -> Report {
    let mut report = Report::default();
    let mut focus_seen: BTreeSet<String> = BTreeSet::new();
    for shape in shapes {
        for focus in graph.instances_of(&shape.target_class) {
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
                let mut nested = Vec::new();
                validate_focus(graph, inner, i, &mut nested);
                for r in nested {
                    push(
                        r.severity,
                        r.path.as_deref(),
                        "node",
                        format!("value {i}: {}", r.message),
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
            Some((_, actual)) if actual == dt => {}
            _ => push(
                p.severity,
                path,
                "datatype",
                format!("{v} is not a {}", short(dt)),
            ),
        }
    }
    if let Some(class) = &p.class {
        let ok = v
            .as_iri()
            .is_some_and(|i| graph.instances_of(class).contains(&i));
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
        let ok = allowed.iter().any(|a| a == lexical || expand(a) == lexical);
        if !ok {
            push(
                p.severity,
                path,
                "in",
                format!("{lexical} is not one of {allowed:?}"),
            );
        }
    }
    if let Some((src, re)) = &p.pattern {
        if !re.is_match(lexical) {
            push(
                p.severity,
                path,
                "pattern",
                format!("{lexical:?} does not match /{src}/"),
            );
        }
    }
    let len = lexical.chars().count();
    if p.min_length.is_some_and(|m| len < m) {
        push(
            p.severity,
            path,
            "minLength",
            format!("length {len} is below minLength"),
        );
    }
    if p.max_length.is_some_and(|m| len > m) {
        push(
            p.severity,
            path,
            "maxLength",
            format!("length {len} is above maxLength"),
        );
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
        let items: Vec<String> = list.iter().map(|v| format!("\"{v}\"")).collect();
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
