//! ONT-001 §3.6, §5 ONT-4b — the in-house shapes validator: the SHACL Core subset, exactly.
//!
//! Shapes are **authored in a contract's `shape:` block** (YAML — the only syntax the gate path parses) and applied
//! to the graph `extract:pv-contract` produces. The subset is §3.6's table and nothing more at v1alpha1:
//!
//! | implemented | refused at parse (`error: shape uses unsupported <x>`, exit 3) |
//! |---|---|
//! | `targetClass`; `targetNode`, `targetSubjectsOf`, `targetObjectsOf` (a string or a list of them, #4814) | the implicit class target; a non-class target in a CONTRACT shape (the gate's plant reads only `targetClass`) |
//! | `minCount`, `maxCount` | the `qualifiedValueShape` family |
//! | `datatype`, `class`, `nodeKind` | — |
//! | `in`, `pattern`, `minLength`, `maxLength` | `languageIn`, `uniqueLang` |
//! | `minExclusive`, `minInclusive`, `maxExclusive`, `maxInclusive` (a YAML scalar bound, #4814) | a typed bound (`"2026-01-01"^^xsd:date`) |
//! | `node` (one level) | recursive / cyclic shapes |
//! | `closed`, `ignoredProperties` | — |
//! | `hasValue` (a YAML scalar, #4814) | an IRI value |
//! | `lessThan`, `lessThanOrEquals`, `equals`, `disjoint` (property pairs) | a language-tagged value (F9: the tag is not kept) |
//! | a single predicate `path` | sequence, alternative, inverse, `*`/`+` paths |
//! | — | `and`/`or`/`not`/`xone`, `sparql`, and every component not in this table |
//!
//! A key the table does not name is REFUSED, not ignored: an ignored constraint is a shape that reports "conforms"
//! for something it never checked, which is the vacuity R-2 exists to end. A key the table DOES name, holding a
//! value of the wrong YAML type (`datatype: 5`, `targetClass: [x]`), is malformed for the same reason (#4814). `resolves:` is ours, not SHACL — it is
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

use crate::ontology::rdf::{ont, Graph, Term, RDF_LANG_STRING, RDF_TYPE};

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
    /// This entry as a literal term (a range bound is compared as one).
    #[must_use]
    pub fn term(&self) -> Term {
        Term::Literal {
            value: self.lexical.clone(),
            datatype: self.datatype.clone(),
        }
    }
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
    /// `sh:lessThan`: every value must be `<` every value of this predicate on the same focus node.
    pub less_than: Option<String>,
    /// `sh:lessThanOrEquals`: every value must be `<=` every value of this predicate on the same focus node.
    pub less_than_or_equals: Option<String>,
    /// `sh:equals`: the value set must equal this predicate's value set on the same focus node (#4814).
    pub equals: Option<String>,
    /// `sh:disjoint`: the value set must share no term with this predicate's value set on the same focus node.
    pub disjoint: Option<String>,
    /// `sh:hasValue`: at least one value must be exactly this term (#4814). A literal only, typed as an `in` entry is.
    pub has_value: Option<InEntry>,
    /// `sh:minExclusive` / `sh:minInclusive` / `sh:maxExclusive` / `sh:maxInclusive` (SHACL §4.3): each value must
    /// compare, by SPARQL `<`, `>` / `>=` / `<` / `<=` the bound. A value that does not compare (an IRI, a string
    /// against a number) is a result, never a skip (#4814).
    pub min_exclusive: Option<InEntry>,
    pub min_inclusive: Option<InEntry>,
    pub max_exclusive: Option<InEntry>,
    pub max_inclusive: Option<InEntry>,
    pub resolves: Option<String>,
    pub severity: Severity,
}

/// A node shape: a target class, an open/closed switch, and its property shapes.
#[derive(Debug, Clone)]
pub struct NodeShape {
    /// The contract that declares it (its stem); nested `node` shapes are `<stem>/node`.
    pub id: String,
    pub target_class: String,
    /// The targets beside `targetClass` (#4814 slice 4). Their focus nodes are added to the class's.
    pub targets: Targets,
    pub closed: bool,
    pub ignored_properties: Vec<String>,
    pub properties: Vec<PropertyShape>,
    /// The constraints written on the node shape itself (#4814 slice 5), if any.
    pub own: Option<Box<NodeLevel>>,
    /// `allowEmpty: "<why>"` (#3610): this shape's target class is empty BY DESIGN in the good state — a
    /// `release:RefusalCell` exists only when a cell does not fit. Zero focus nodes then does not refuse the
    /// verdict, but the shape is still named in the gate's `declines[]`. Not SHACL; pv's own key, and it
    /// must carry its reason: an exemption with no reason is a silent one.
    pub allow_empty: Option<String>,
}

/// Constraints on the node shape itself (SHACL §2.1, #4814 slice 5): the value set is `{focus}`, and a result has
/// no `resultPath`. Held as a [`PropertyShape`] whose `path` is empty and never read as a predicate, beside the keys
/// as written, so that a refusal can name the key it refuses.
#[derive(Debug, Clone)]
pub struct NodeLevel {
    pub keys: Vec<String>,
    pub constraints: PropertyShape,
}

/// `sh:targetNode` (an IRI, or since #4814 slice 5 a literal), then `sh:targetSubjectsOf` and `sh:targetObjectsOf`
/// (full IRIs), in the order written.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Targets {
    pub nodes: Vec<Term>,
    pub subjects_of: Vec<String>,
    pub objects_of: Vec<String>,
}

impl Targets {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.subjects_of.is_empty() && self.objects_of.is_empty()
    }
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

const NODE_KEYS: &[&str] = &[
    "targetClass",
    "targetNode",
    "targetSubjectsOf",
    "targetObjectsOf",
    "closed",
    "ignoredProperties",
    "properties",
    "allowEmpty",
];
/// The value components a node shape may carry on itself (#4814 slice 5). Each one means what it means on a
/// property shape, applied to the value set `{focus}`. `languageIn` stays refused until a term keeps its tag (F9,
/// slice 10). The counts and the property pairs (`lessThan`, `lessThanOrEquals`) are property-shape components.
const NODE_LEVEL_KEYS: &[&str] = &[
    "class",
    "datatype",
    "nodeKind",
    "in",
    "pattern",
    "minLength",
    "maxLength",
    "minExclusive",
    "minInclusive",
    "maxExclusive",
    "maxInclusive",
    "hasValue",
    "equals",
    "disjoint",
    "node",
];
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
    "lessThan",
    "lessThanOrEquals",
    "equals",
    "disjoint",
    "hasValue",
    "minExclusive",
    "minInclusive",
    "maxExclusive",
    "maxInclusive",
    "resolves",
    "severity",
];

/// Σ `entity_type_target_class` (v4.16 D-T1, qd4c4): `entity.type` → the class a shape with no `targetClass`
/// targets, as a prefixed name (`readme:Readme`), expanded like any other.
pub type TargetMap = std::collections::BTreeMap<String, String>;

/// Read a contract's `shape:` block with NO Σ map — so a shape without `targetClass` is malformed. The gate
/// reads Σ and calls [`parse_shape_with`].
pub fn parse_shape(stem: &str, doc: &serde_yaml::Value) -> Result<Option<NodeShape>, ShapeError> {
    parse_shape_with(stem, doc, &TargetMap::new())
}

/// Read a contract's `shape:` block. `None` when the contract has no `shape:`. With no `targetClass`, the
/// target is `targets[entity.type]` (Σ `entity_type_target_class`); an unmapped type is malformed, named.
pub fn parse_shape_with(
    stem: &str,
    doc: &serde_yaml::Value,
    targets: &TargetMap,
) -> Result<Option<NodeShape>, ShapeError> {
    let Some(block) = doc.get("shape") else {
        return Ok(None);
    };
    let Some(map) = block.as_mapping() else {
        return Err(ShapeError::Malformed {
            shape: stem.to_string(),
            what: "`shape:` is not a mapping".into(),
        });
    };
    parse_node_shape(stem, map, default_target(doc, targets), 0).map(Some)
}

/// [`parse_shapes_with`] with no Σ map.
pub fn parse_shapes(stem: &str, doc: &serde_yaml::Value) -> Result<Vec<NodeShape>, ShapeError> {
    parse_shapes_with(stem, doc, &TargetMap::new())
}

/// Every shape a contract declares: its `shape:` block (id = the stem) and each entry of its `shapes:` list
/// (id = the entry's own `id`, required, so a contract may hold several shapes that are armed one by one —
/// ONT-4c1's `ladder-measured` and `ladder-green`). An entry without `id`, or an `id` that repeats within the
/// contract, is malformed.
pub fn parse_shapes_with(
    stem: &str,
    doc: &serde_yaml::Value,
    targets: &TargetMap,
) -> Result<Vec<NodeShape>, ShapeError> {
    let mut out = Vec::new();
    if let Some(s) = parse_shape_with(stem, doc, targets)? {
        out.push(s);
    }
    let Some(list) = doc.get("shapes") else {
        apply_allow_empty(stem, doc, &mut out)?;
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
        out.push(parse_node_shape(
            id,
            &body,
            default_target(doc, targets),
            0,
        )?);
    }
    apply_allow_empty(stem, doc, &mut out)?;
    Ok(out)
}

/// The contract-level `allow_empty: {<shape id>: "<why>"}` map — the same exemption as a shape's own
/// `allowEmpty`, declared OUTSIDE the shape so a released pv that predates the key still parses the shape
/// (0.69.1 refuses an unknown shape key; the fleet-pinned shapes gate runs it, #4587). An id the contract
/// does not declare, a blank reason, or a shape that also says `allowEmpty` is malformed.
fn apply_allow_empty(
    stem: &str,
    doc: &serde_yaml::Value,
    shapes: &mut [NodeShape],
) -> Result<(), ShapeError> {
    let Some(v) = doc.get("allow_empty") else {
        return Ok(());
    };
    let malformed = |what: String| ShapeError::Malformed {
        shape: stem.to_string(),
        what,
    };
    let map = v
        .as_mapping()
        .ok_or_else(|| malformed("`allow_empty:` is not a mapping".into()))?;
    for (k, reason) in map {
        let id = k
            .as_str()
            .ok_or_else(|| malformed("an `allow_empty:` key is not a shape id".into()))?;
        let reason = match reason.as_str().map(str::trim) {
            Some(r) if !r.is_empty() => r.to_string(),
            _ => {
                return Err(malformed(format!(
                    "`allow_empty.{id}` must be a non-empty reason string"
                )))
            }
        };
        let shape = shapes.iter_mut().find(|s| s.id == id).ok_or_else(|| {
            malformed(format!(
                "`allow_empty.{id}` names no shape in this contract"
            ))
        })?;
        if shape.allow_empty.is_some() {
            return Err(malformed(format!("`{id}` declares allowEmpty twice")));
        }
        shape.allow_empty = Some(reason);
    }
    Ok(())
}

/// Σ `entity_type_target_class[entity.type]`, expanded. No entity type is special-cased (v4.16 D-T1 replaced
/// the `pv-contract`-only rule): a type the map does not carry has no default.
fn default_target(doc: &serde_yaml::Value, targets: &TargetMap) -> Option<String> {
    doc.get("entity")
        .and_then(|e| e.get("type"))
        .and_then(serde_yaml::Value::as_str)
        .and_then(|t| targets.get(t))
        .map(|c| expand(c))
}

fn parse_node_shape(
    id: &str,
    map: &serde_yaml::Mapping,
    default_target: Option<String>,
    depth: usize,
) -> Result<NodeShape, ShapeError> {
    for key in map.keys() {
        let k = key.as_str().unwrap_or("?");
        if !NODE_KEYS.contains(&k) && !NODE_LEVEL_KEYS.contains(&k) {
            return Err(ShapeError::Unsupported {
                shape: id.to_string(),
                component: k.to_string(),
            });
        }
    }
    let targets = targets_of(id, map, depth)?;
    let target_class = match str_key(id, map, "targetClass")? {
        Some(t) => expand(t),
        None => match (depth, default_target) {
            // explicit targets name the focus nodes; the entity-type default class does not widen them
            (0, _) if !targets.is_empty() => String::new(),
            (0, Some(t)) => t,
            (0, None) => {
                return Err(ShapeError::Malformed {
                    shape: id.to_string(),
                    what: "no targetClass (entity.type has no Σ entity_type_target_class mapping)"
                        .into(),
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
            .map(|x| {
                x.as_str().map(expand).ok_or_else(|| ShapeError::Malformed {
                    shape: id.to_string(),
                    what: "an `ignoredProperties` entry is not a string".into(),
                })
            })
            .collect::<Result<_, _>>()?,
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
        targets,
        closed,
        ignored_properties,
        properties,
        own: node_level_of(id, map, depth)?,
        allow_empty: allow_empty_of(id, map, depth)?,
    })
}

/// The [`NODE_LEVEL_KEYS`] written on the node shape itself, parsed as a property shape's would be, with no path.
fn node_level_of(
    id: &str,
    map: &serde_yaml::Mapping,
    depth: usize,
) -> Result<Option<Box<NodeLevel>>, ShapeError> {
    let own: serde_yaml::Mapping = map
        .iter()
        .filter(|(k, _)| k.as_str().is_some_and(|k| NODE_LEVEL_KEYS.contains(&k)))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if own.is_empty() {
        return Ok(None);
    }
    Ok(Some(Box::new(NodeLevel {
        keys: own
            .keys()
            .filter_map(|k| k.as_str().map(String::from))
            .collect(),
        constraints: parse_constraints(id, &own, depth, String::new())?,
    })))
}

/// `targetNode` / `targetSubjectsOf` / `targetObjectsOf`: each one entry or a non-empty list of them. An entry is a
/// string, expanded to an IRI; a `targetNode` entry may instead be a literal, written `{literal: "7", datatype:
/// xsd:integer}` (#4814 slice 5). A bare YAML scalar stays an IRI, because in `in` and `hasValue` the same scalar
/// is a literal typed by its YAML type, and one spelling cannot mean both.
/// A nested `node` shape has no focus set of its own, so a target there is malformed, not ignored.
fn targets_of(id: &str, map: &serde_yaml::Mapping, depth: usize) -> Result<Targets, ShapeError> {
    let list = |k: &str, literal_ok: bool| -> Result<Vec<Term>, ShapeError> {
        let Some(v) = map.get(k) else {
            return Ok(Vec::new());
        };
        if depth > 0 {
            return Err(malformed_in(
                id,
                format!("`{k}` on a nested `node` shape, which has no focus set"),
            ));
        }
        let bad = || {
            malformed_in(
                id,
                format!(
                    "`{k}` is not {} or a non-empty list of them",
                    if literal_ok {
                        "a string, a {literal, datatype} mapping,"
                    } else {
                        "a string"
                    }
                ),
            )
        };
        let entry = |x: &serde_yaml::Value| -> Result<Term, ShapeError> {
            match x {
                serde_yaml::Value::String(s) => Ok(Term::iri(expand(s))),
                serde_yaml::Value::Mapping(m) if literal_ok => target_literal(m).ok_or_else(bad),
                _ => Err(bad()),
            }
        };
        match v {
            serde_yaml::Value::Sequence(seq) if !seq.is_empty() => seq.iter().map(entry).collect(),
            serde_yaml::Value::Sequence(_) => Err(bad()),
            one => Ok(vec![entry(one)?]),
        }
    };
    let iris = |k: &str| -> Result<Vec<String>, ShapeError> {
        Ok(list(k, false)?
            .into_iter()
            .filter_map(|t| t.as_iri().map(String::from))
            .collect())
    };
    Ok(Targets {
        nodes: list("targetNode", true)?,
        subjects_of: iris("targetSubjectsOf")?,
        objects_of: iris("targetObjectsOf")?,
    })
}

/// `{literal: "<lexical>", datatype: <prefixed or full IRI>}`, exactly those two keys, both strings. No default
/// datatype: a focus literal written without one would be an `xsd:string` by accident, as F11 found for numbers.
fn target_literal(m: &serde_yaml::Mapping) -> Option<Term> {
    if m.len() != 2 {
        return None;
    }
    let value = m.get("literal")?.as_str()?;
    let datatype = m.get("datatype")?.as_str()?;
    Some(Term::Literal {
        value: value.to_string(),
        datatype: expand(datatype),
    })
}

/// `allowEmpty` is a reason string on a TOP-LEVEL shape; a nested `node` shape has no focus set of its own.
fn allow_empty_of(
    id: &str,
    map: &serde_yaml::Mapping,
    depth: usize,
) -> Result<Option<String>, ShapeError> {
    let Some(v) = map.get("allowEmpty") else {
        return Ok(None);
    };
    let malformed = |what: &str| ShapeError::Malformed {
        shape: id.to_string(),
        what: what.into(),
    };
    if depth > 0 {
        return Err(malformed("`allowEmpty` on a nested node shape"));
    }
    match v.as_str().map(str::trim) {
        Some(reason) if !reason.is_empty() => Ok(Some(reason.to_string())),
        _ => Err(malformed("`allowEmpty` must be a non-empty reason string")),
    }
}

fn parse_property(
    shape: &str,
    pm: &serde_yaml::Mapping,
    depth: usize,
) -> Result<PropertyShape, ShapeError> {
    check_property_keys(shape, pm)?;
    let path = expand(parse_path(shape, pm)?);
    parse_constraints(shape, pm, depth, path)
}

/// Every constraint of one property shape, or of a node shape itself (`path` empty, #4814 slice 5).
fn parse_constraints(
    shape: &str,
    pm: &serde_yaml::Mapping,
    depth: usize,
    path: String,
) -> Result<PropertyShape, ShapeError> {
    let malformed = |what: String| ShapeError::Malformed {
        shape: shape.to_string(),
        what,
    };
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
    let iri_opt = |k: &str| str_key(shape, pm, k).map(|v| v.map(expand));
    let node_kind = parse_node_kind(shape, str_key(shape, pm, "nodeKind")?)?;
    // `sh:in` is TERM equality (SHACL §4.5.1), and a term carries its datatype. The YAML scalar's own type is
    // what gives it one: `in: [true]` is `"true"^^xsd:boolean`, `in: [1]` is `xsd:integer`, `in: [a, b]` is
    // `xsd:string`. This used to collapse every entry to its lexical form and compare strings, so a shape
    // `in: ["true"]` accepted `"true"^^xsd:boolean` — which the pinned oracle refuses, and which `make oracle`
    // caught on this row's own shapes (490 results of difference on the real corpus). An entry that expands to
    // an IRI still matches an IRI value, because a `sh:in` over `nodeKind: IRI` is a list of IRIs.
    let r#in = parse_in(shape, pm)?;
    let pattern = parse_pattern(shape, pm)?;
    let node = parse_nested_node(shape, pm, depth)?;
    let severity = parse_severity(shape, str_key(shape, pm, "severity")?)?;
    Ok(PropertyShape {
        path,
        min_count: count("minCount")?,
        max_count: count("maxCount")?,
        datatype: iri_opt("datatype")?,
        class: iri_opt("class")?,
        node_kind,
        r#in,
        pattern,
        min_length: count("minLength")?,
        max_length: count("maxLength")?,
        node,
        less_than: iri_opt("lessThan")?,
        less_than_or_equals: iri_opt("lessThanOrEquals")?,
        equals: iri_opt("equals")?,
        disjoint: iri_opt("disjoint")?,
        has_value: bound(shape, pm, "hasValue")?,
        min_exclusive: bound(shape, pm, "minExclusive")?,
        min_inclusive: bound(shape, pm, "minInclusive")?,
        max_exclusive: bound(shape, pm, "maxExclusive")?,
        max_inclusive: bound(shape, pm, "maxInclusive")?,
        resolves: str_key(shape, pm, "resolves")?.map(String::from),
        severity,
    })
}

fn malformed_in(shape: &str, what: String) -> ShapeError {
    ShapeError::Malformed {
        shape: shape.to_string(),
        what,
    }
}

/// A key whose value must be a string: absent is `None`, any other YAML type is malformed (#4814). Reading a
/// `datatype: 5` or a `targetClass: [x]` as absent drops the constraint — the shape then reports conforms for
/// what it never checked, the same vacuity as an ignored unknown key.
fn str_key<'a>(
    shape: &str,
    m: &'a serde_yaml::Mapping,
    k: &str,
) -> Result<Option<&'a str>, ShapeError> {
    match m.get(k) {
        None => Ok(None),
        Some(v) => v
            .as_str()
            .map(Some)
            .ok_or_else(|| malformed_in(shape, format!("`{k}` is not a string"))),
    }
}

/// A property mapping may carry only the keys of the supported subset.
fn check_property_keys(shape: &str, pm: &serde_yaml::Mapping) -> Result<(), ShapeError> {
    for key in pm.keys() {
        let k = key.as_str().unwrap_or("?");
        if !PROPERTY_KEYS.contains(&k) {
            return Err(ShapeError::Unsupported {
                shape: shape.to_string(),
                component: k.to_string(),
            });
        }
    }
    Ok(())
}

/// `path`: a single predicate; a SHACL property path expression is outside the subset.
fn parse_path<'a>(shape: &str, pm: &'a serde_yaml::Mapping) -> Result<&'a str, ShapeError> {
    let path = pm
        .get("path")
        .and_then(serde_yaml::Value::as_str)
        .ok_or_else(|| malformed_in(shape, "a property has no `path`".into()))?;
    // A full IRI keeps its `/`, `*` and `+`, but it can never hold whitespace or `|^<>"{}\` (RFC 3987): such a
    // string is a path expression or a list written with full IRIs, and reading it as one predicate checks a
    // property no data carries (#4814). A prefixed name with whitespace is the same thing.
    let iri = path.starts_with("http");
    let expression = if iri {
        path.contains(|c: char| c.is_whitespace() || "|^<>\"{}\\`".contains(c))
    } else {
        path.contains(|c: char| c.is_whitespace() || "/|^*+".contains(c))
    };
    if expression {
        return Err(ShapeError::Unsupported {
            shape: shape.to_string(),
            component: format!("path `{path}` (only a single predicate is a path here)"),
        });
    }
    Ok(path)
}

// `sh:in` is TERM equality (SHACL §4.5.1), and a term carries its datatype. The YAML scalar's own type is
// what gives it one: `in: [true]` is `"true"^^xsd:boolean`, `in: [1]` is `xsd:integer`, `in: [a, b]` is
// `xsd:string`. This used to collapse every entry to its lexical form and compare strings, so a shape
// `in: ["true"]` accepted `"true"^^xsd:boolean` — which the pinned oracle refuses, and which `make oracle`
// caught on this row's own shapes (490 results of difference on the real corpus). An entry that expands to
// an IRI still matches an IRI value, because a `sh:in` over `nodeKind: IRI` is a list of IRIs.
fn parse_in(shape: &str, pm: &serde_yaml::Mapping) -> Result<Option<Vec<InEntry>>, ShapeError> {
    let Some(v) = pm.get("in") else {
        return Ok(None);
    };
    let seq = v
        .as_sequence()
        .ok_or_else(|| malformed_in(shape, "`in` is not a list".into()))?;
    seq.iter()
        .map(|x| {
            in_entry(x).ok_or_else(|| malformed_in(shape, "an `in` entry is not a scalar".into()))
        })
        .collect::<Result<_, _>>()
        .map(Some)
}

/// A value-range bound (`minExclusive: 40`): a YAML scalar, typed as an `in` entry is. Anything else is malformed,
/// never absent (#4814).
fn bound(shape: &str, pm: &serde_yaml::Mapping, k: &str) -> Result<Option<InEntry>, ShapeError> {
    pm.get(k)
        .map(|v| in_entry(v).ok_or_else(|| malformed_in(shape, format!("`{k}` is not a scalar"))))
        .transpose()
}

/// A YAML scalar as a term; `None` for a mapping, a list or null, which name no term (#4814: these were read as
/// the empty string, so a malformed list was accepted).
fn in_entry(x: &serde_yaml::Value) -> Option<InEntry> {
    Some(match x {
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
        _ => return None,
    })
}

fn parse_pattern(
    shape: &str,
    pm: &serde_yaml::Mapping,
) -> Result<Option<(String, regex::Regex)>, ShapeError> {
    let Some(p) = str_key(shape, pm, "pattern")? else {
        return Ok(None);
    };
    let re = regex::Regex::new(p)
        .map_err(|e| malformed_in(shape, format!("`pattern` does not compile: {e}")))?;
    Ok(Some((p.to_string(), re)))
}

/// `node`: one level of nesting is in the subset, deeper is not.
fn parse_nested_node(
    shape: &str,
    pm: &serde_yaml::Mapping,
    depth: usize,
) -> Result<Option<Box<NodeShape>>, ShapeError> {
    match pm.get("node") {
        None => Ok(None),
        Some(_) if depth >= 1 => Err(ShapeError::Unsupported {
            shape: shape.to_string(),
            component: "node (nested more than one level)".into(),
        }),
        Some(v) => {
            let nm = v
                .as_mapping()
                .ok_or_else(|| malformed_in(shape, "`node` is not a mapping".into()))?;
            Ok(Some(Box::new(parse_node_shape(
                &format!("{shape}/node"),
                nm,
                None,
                depth + 1,
            )?)))
        }
    }
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

/// A shape's focus nodes, unique, in byte order: the instances of its target class (subclasses included), each
/// `targetNode` (whether or not the graph mentions it, as SHACL says), each subject of a `targetSubjectsOf`
/// predicate and each object of a `targetObjectsOf` predicate. A literal object is a focus node too; it is named by
/// its N-Triples form, which no IRI can equal, and it has no values on any path.
#[must_use]
pub fn focus_nodes(graph: &Graph, shape: &NodeShape) -> Vec<String> {
    focus_terms(graph, shape).iter().map(focus_name).collect()
}

/// The focus nodes of a shape as terms: [`focus_nodes`] before naming, so that a literal focus keeps its datatype
/// for the node shape's own constraints (#4814 slice 5).
fn focus_terms(graph: &Graph, shape: &NodeShape) -> BTreeSet<Term> {
    let mut out: BTreeSet<Term> = BTreeSet::new();
    if !shape.target_class.is_empty() {
        out.extend(
            instances_closed(graph, &shape.target_class)
                .into_iter()
                .map(Term::Iri),
        );
    }
    out.extend(shape.targets.nodes.iter().cloned());
    for t in graph.iter() {
        if shape.targets.subjects_of.contains(&t.predicate) {
            out.insert(Term::Iri(t.subject.clone()));
        }
        if shape.targets.objects_of.contains(&t.predicate) {
            out.insert(t.object.clone());
        }
    }
    out
}

/// How a focus node is named in a result: an IRI as itself, a literal by its N-Triples form, which no IRI equals.
fn focus_name(t: &Term) -> String {
    match t {
        Term::Iri(s) => s.clone(),
        lit @ Term::Literal { .. } => lit.to_string(),
    }
}

/// Validate `graph` against `shapes`. Focus nodes of a shape are [`focus_nodes`].
#[must_use]
pub fn validate(graph: &Graph, shapes: &[NodeShape]) -> Report {
    let mut report = Report::default();
    let mut focus_seen: BTreeSet<String> = BTreeSet::new();
    for shape in shapes {
        for focus in focus_terms(graph, shape) {
            focus_seen.insert(focus_name(&focus));
            validate_focus(graph, shape, &focus, &mut report.results);
        }
    }
    report.focus_nodes_n = focus_seen.len();
    report.results.sort();
    report.results.dedup();
    report
}

fn validate_focus(graph: &Graph, shape: &NodeShape, focus: &Term, out: &mut Vec<ValidationResult>) {
    // A literal focus is no subject, so it has no values on any path; its name matches no triple.
    let name = focus_name(focus);
    let mut push =
        |severity: Severity, path: Option<&str>, component: &'static str, message: String| {
            out.push(ValidationResult {
                severity,
                focus: name.clone(),
                shape: shape.id.clone(),
                path: path.map(String::from),
                component,
                message,
            });
        };
    let focus_s = name.as_str();
    if let Some(own) = &shape.own {
        // SHACL §2.1: the value set of a node shape is {focus}, and its results carry no resultPath
        let mut no_path =
            |s: Severity, _: Option<&str>, c: &'static str, m: String| push(s, None, c, m);
        let p = &own.constraints;
        let values = [focus];
        check_value(graph, p, focus, &mut no_path);
        check_sets(graph, focus_s, p, &values, &mut no_path);
        check_has_value(p, &values, &mut no_path);
    }
    for p in &shape.properties {
        let focus = focus_s;
        let values = graph.objects(focus, &p.path);
        check_counts(p, &values, &mut push);
        for v in &values {
            check_value(graph, p, v, &mut push);
        }
        check_pairs(graph, focus, p, &values, &mut push);
        check_sets(graph, focus, p, &values, &mut push);
        check_has_value(p, &values, &mut push);
    }
    if shape.closed {
        check_closed(graph, shape, focus_s, &mut push);
    }
}

/// `sh:minCount` / `sh:maxCount` on one property of one focus node.
fn check_counts(
    p: &PropertyShape,
    values: &[&Term],
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
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
}

/// `sh:lessThan` / `sh:lessThanOrEquals` (SHACL §4.5.3/§4.5.4) on one property of one focus node.
fn check_pairs(
    graph: &Graph,
    focus: &str,
    p: &PropertyShape,
    values: &[&Term],
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
    for (other, strict, component) in [
        (p.less_than.as_deref(), true, "lessThan"),
        (p.less_than_or_equals.as_deref(), false, "lessThanOrEquals"),
    ] {
        let Some(other) = other else { continue };
        // SHACL §4.5.3/§4.5.4: one result per (value, other value) PAIR that is not ordered — the W3C case
        // lessThan-002 expects four results from two values against two, so the pair is named in the message
        // (which also keeps the pairs distinct through `validate`'s dedup). A pair that SPARQL `<` cannot
        // compare (an IRI, a string against a number) is NOT ordered, so it is a result, never a skip.
        let others = graph.objects(focus, other);
        for v in values {
            for w in &others {
                let ordered = match compare_terms(v, w) {
                    Some(std::cmp::Ordering::Less) => true,
                    Some(std::cmp::Ordering::Equal) => !strict,
                    _ => false,
                };
                if !ordered {
                    push(
                        p.severity,
                        Some(&p.path),
                        component,
                        format!(
                            "{}: {} is not {} {} of {}",
                            short(&p.path),
                            term_short(v),
                            if strict { "<" } else { "<=" },
                            term_short(w),
                            short(other)
                        ),
                    );
                }
            }
        }
    }
}

/// `sh:equals` / `sh:disjoint` (SHACL §4.5.1/§4.5.2) on one property of one focus node: one result per VALUE that
/// breaks the relation (W3C property/equals-001 expects five from four resources), named in the message so
/// `validate`'s dedup keeps them apart. Equality is RDF term equality, (lexical, datatype) exact. A language-tagged
/// value is a result on both components, never a pass: `Term` drops the tag (F9), so `"a"@en` and `"a"@fr` would
/// read as one term, and pv refuses to decide a comparison it cannot see until the tag is kept (#4814 slice 10).
fn check_sets(
    graph: &Graph,
    focus: &str,
    p: &PropertyShape,
    values: &[&Term],
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
    let lang =
        |t: &Term| matches!(t, Term::Literal { datatype, .. } if datatype == RDF_LANG_STRING);
    if let Some(other) = p.equals.as_deref() {
        let others = graph.objects(focus, other);
        let mut unmatched = |t: &Term, from: &str, to: &str| {
            push(
                p.severity,
                Some(&p.path),
                "equals",
                format!(
                    "{}: {} of {} has no equal in {}",
                    short(&p.path),
                    term_short(t),
                    short(from),
                    short(to)
                ),
            );
        };
        for v in values.iter().filter(|v| lang(v) || !others.contains(v)) {
            unmatched(v, &p.path, other);
        }
        for w in others.iter().filter(|w| lang(w) || !values.contains(w)) {
            unmatched(w, other, &p.path);
        }
    }
    if let Some(other) = p.disjoint.as_deref() {
        let others = graph.objects(focus, other);
        for v in values.iter().filter(|v| lang(v) || others.contains(v)) {
            push(
                p.severity,
                Some(&p.path),
                "disjoint",
                format!(
                    "{}: {} is also a value of {}",
                    short(&p.path),
                    term_short(v),
                    short(other)
                ),
            );
        }
    }
}

/// `sh:hasValue` (SHACL §4.8.2) on one property of one focus node: one result when no value is exactly the term,
/// none for a value that differs (W3C property/hasValue-001: `"female"` beside `"male"` conforms). A
/// language-tagged value never matches, since the term is never `rdf:langString`, so F9 cannot make it pass.
fn check_has_value(
    p: &PropertyShape,
    values: &[&Term],
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
    let Some(want) = &p.has_value else { return };
    let want = want.term();
    if !values.iter().any(|v| **v == want) {
        push(
            p.severity,
            Some(&p.path),
            "hasValue",
            format!("{}: no value is {}", short(&p.path), term_short(&want)),
        );
    }
}

/// `sh:closed`: every predicate the focus carries is declared, ignored, or `rdf:type`.
fn check_closed(
    graph: &Graph,
    shape: &NodeShape,
    focus: &str,
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
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

fn check_value(
    graph: &Graph,
    p: &PropertyShape,
    v: &Term,
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
    check_kind_and_type(graph, p, v, push);
    check_lexical(p, v, push);
    check_range(p, v, push);
    if let Some(inner) = &p.node {
        match v.as_iri() {
            Some(i) => {
                // One `sh:NodeConstraintComponent` result per VALUE that fails the nested shape, on the outer
                // path, with the nested findings as its detail (SHACL §4.6.2; W3C property/node-001, -002).
                let mut nested = Vec::new();
                validate_focus(graph, inner, &Term::iri(i), &mut nested);
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

/// `sh:minExclusive` / `sh:minInclusive` / `sh:maxExclusive` / `sh:maxInclusive` (SHACL §4.3) on one value. SHACL
/// counts a value as conforming only when the SPARQL comparison against the bound is TRUE, so a value it cannot
/// compare (an IRI, a string against a number: W3C property/minExclusive-002, maxExclusive-001) is a result.
fn check_range(
    p: &PropertyShape,
    v: &Term,
    push: &mut impl FnMut(Severity, Option<&str>, &'static str, String),
) {
    use std::cmp::Ordering::{Equal, Greater, Less};
    for (bound, component, op, holds) in [
        (&p.min_exclusive, "minExclusive", ">", &[Greater][..]),
        (
            &p.min_inclusive,
            "minInclusive",
            ">=",
            &[Greater, Equal][..],
        ),
        (&p.max_exclusive, "maxExclusive", "<", &[Less][..]),
        (&p.max_inclusive, "maxInclusive", "<=", &[Less, Equal][..]),
    ] {
        let Some(b) = bound else { continue };
        if !compare_terms(v, &b.term()).is_some_and(|o| holds.contains(&o)) {
            push(
                p.severity,
                Some(&p.path),
                component,
                format!("{}: {} is not {op} {b}", short(&p.path), term_short(v)),
            );
        }
    }
}

/// The XSD numeric datatypes SPARQL `<` compares by value, across types (`4 < 4.5` holds for an integer and a
/// decimal).
const NUMERIC_TYPES: &[&str] = &[
    "integer",
    "decimal",
    "double",
    "float",
    "long",
    "int",
    "short",
    "byte",
    "nonNegativeInteger",
    "nonPositiveInteger",
    "positiveInteger",
    "negativeInteger",
    "unsignedLong",
    "unsignedInt",
    "unsignedShort",
    "unsignedByte",
];

/// SPARQL `<` over two terms, as `sh:lessThan` uses it (SHACL §4.5.3): numbers by value across the numeric
/// types; strings, booleans, dates and dateTimes within their own type. `None` for every pair SPARQL cannot
/// order — an IRI, a type mismatch, an ill-formed number — which the caller treats as NOT ordered.
/// Dates and dateTimes compare by lexical form, so this is exact for the corpus's zone-less ISO forms and not for
/// two values in different time zones.
#[must_use]
pub fn compare_terms(a: &Term, b: &Term) -> Option<std::cmp::Ordering> {
    let (
        Term::Literal {
            value: av,
            datatype: ad,
        },
        Term::Literal {
            value: bv,
            datatype: bd,
        },
    ) = (a, b)
    else {
        return None;
    };
    let numeric = |d: &str| {
        d.strip_prefix(XSD_NS)
            .is_some_and(|local| NUMERIC_TYPES.contains(&local))
    };
    if numeric(ad) && numeric(bd) {
        if let (Ok(x), Ok(y)) = (av.trim().parse::<i128>(), bv.trim().parse::<i128>()) {
            return Some(x.cmp(&y));
        }
        let (x, y) = (
            av.trim().parse::<f64>().ok()?,
            bv.trim().parse::<f64>().ok()?,
        );
        return x.partial_cmp(&y);
    }
    if ad != bd {
        return None;
    }
    match ad.strip_prefix(XSD_NS) {
        Some("string" | "date" | "dateTime") => Some(av.cmp(bv)),
        Some("boolean") => {
            let bool_of = |s: &str| match s {
                "true" | "1" => Some(true),
                "false" | "0" => Some(false),
                _ => None,
            };
            Some(bool_of(av)?.cmp(&bool_of(bv)?))
        }
        _ => None,
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
    if let Some((lo, hi)) = integer_range(local) {
        return value.parse::<i128>().is_ok_and(|n| n >= lo && n <= hi);
    }
    match local {
        "boolean" => matches!(value, "true" | "false" | "1" | "0"),
        "decimal" => is_decimal(value),
        "double" | "float" => {
            matches!(value, "INF" | "-INF" | "NaN") || value.parse::<f64>().is_ok()
        }
        "date" => is_date(value),
        "dateTime" => is_date_time(value),
        // "string", "anyURI", and every datatype outside the checked set.
        _ => true,
    }
}

/// The value range of each XSD integer type (`integer` itself is unbounded, so i128's).
fn integer_range(local: &str) -> Option<(i128, i128)> {
    Some(match local {
        "integer" => (i128::MIN, i128::MAX),
        "long" => (i128::from(i64::MIN), i128::from(i64::MAX)),
        "int" => (i128::from(i32::MIN), i128::from(i32::MAX)),
        "short" => (i128::from(i16::MIN), i128::from(i16::MAX)),
        "byte" => (i128::from(i8::MIN), i128::from(i8::MAX)),
        "nonNegativeInteger" => (0, i128::MAX),
        "positiveInteger" => (1, i128::MAX),
        "nonPositiveInteger" => (i128::MIN, 0),
        "negativeInteger" => (i128::MIN, -1),
        "unsignedLong" => (0, i128::from(u64::MAX)),
        "unsignedInt" => (0, i128::from(u32::MAX)),
        "unsignedShort" => (0, i128::from(u16::MAX)),
        "unsignedByte" => (0, i128::from(u8::MAX)),
        _ => return None,
    })
}

fn is_decimal(value: &str) -> bool {
    !value.is_empty()
        && value
            .strip_prefix(['+', '-'])
            .unwrap_or(value)
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.')
        && value.chars().filter(|c| *c == '.').count() <= 1
        && value.chars().any(|c| c.is_ascii_digit())
}

fn is_date_time(value: &str) -> bool {
    value
        .split_once('T')
        .is_some_and(|(d, t)| is_date(d) && t.len() >= 8 && t.as_bytes()[2] == b':')
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
    for t in &s.targets.nodes {
        // a Term displays in N-Triples form, `<iri>` or `"lexical"^^<datatype>`, which is valid Turtle
        o.push_str(&format!("    sh:targetNode {t} ;\n"));
    }
    for (pred, list) in [
        ("targetSubjectsOf", &s.targets.subjects_of),
        ("targetObjectsOf", &s.targets.objects_of),
    ] {
        for t in list {
            o.push_str(&format!("    sh:{pred} <{t}> ;\n"));
        }
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
    if let Some(own) = &s.own {
        for l in constraint_lines(&own.constraints) {
            o.push_str(&format!("    {l} ;\n"));
        }
    }
    for p in &s.properties {
        o.push_str(&turtle_property(p));
    }
    o.push_str(".\n\n");
    let own_node = s.own.as_ref().and_then(|n| n.constraints.node.as_deref());
    for inner in s
        .properties
        .iter()
        .filter_map(|p| p.node.as_deref())
        .chain(own_node)
    {
        o.push_str(&turtle_node(
            inner,
            &format!("<{}shape/{}>", crate::ontology::rdf::ONT_BASE, inner.id),
        ));
    }
    o
}

/// The string-based (`sh:pattern`, `sh:minLength`, `sh:maxLength`) and property-pair (`sh:lessThan`,
/// `sh:lessThanOrEquals`) lines of one `sh:property` block, in emission order.
fn turtle_string_and_pair_lines(p: &PropertyShape) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some((src, _)) = &p.pattern {
        lines.push(format!(
            "sh:pattern \"{}\"",
            src.replace('\\', "\\\\").replace('"', "\\\"")
        ));
    }
    if let Some(n) = p.min_length {
        lines.push(format!("sh:minLength {n}"));
    }
    if let Some(n) = p.max_length {
        lines.push(format!("sh:maxLength {n}"));
    }
    if let Some(o) = &p.less_than {
        lines.push(format!("sh:lessThan <{o}>"));
    }
    if let Some(o) = &p.less_than_or_equals {
        lines.push(format!("sh:lessThanOrEquals <{o}>"));
    }
    if let Some(o) = &p.equals {
        lines.push(format!("sh:equals <{o}>"));
    }
    if let Some(o) = &p.disjoint {
        lines.push(format!("sh:disjoint <{o}>"));
    }
    if let Some(v) = &p.has_value {
        lines.push(format!("sh:hasValue {}", turtle_entry(v)));
    }
    for (b, k) in [
        (&p.min_exclusive, "minExclusive"),
        (&p.min_inclusive, "minInclusive"),
        (&p.max_exclusive, "maxExclusive"),
        (&p.max_inclusive, "maxInclusive"),
    ] {
        if let Some(b) = b {
            lines.push(format!("sh:{k} {}", turtle_entry(b)));
        }
    }
    lines
}

/// A YAML-typed term as a Turtle literal: typed unless it is an `xsd:string`, because `sh:in` is term equality and a
/// range bound compares by its datatype — an untyped `"40"` would be a string the oracle cannot order against 39.
fn turtle_entry(v: &InEntry) -> String {
    // escaped as `sh:pattern` is: a `"` or `\` in the lexical form would otherwise end or break the literal
    let lexical = v.lexical.replace('\\', "\\\\").replace('"', "\\\"");
    if v.datatype == XSD_STRING_IRI {
        format!("\"{lexical}\"")
    } else {
        format!("\"{lexical}\"^^<{}>", v.datatype)
    }
}

/// One `sh:property [ … ] ;` block. Every implemented component has a line; nothing else is emitted.
fn turtle_property(p: &PropertyShape) -> String {
    let mut o = String::from("    sh:property [\n");
    o.push_str(&format!("        sh:path <{}> ;\n", p.path));
    for l in constraint_lines(p) {
        o.push_str(&format!("        {l} ;\n"));
    }
    o.push_str("    ] ;\n");
    o
}

/// Every constraint line of a property shape, or of a node shape itself, without the path and without the `;`.
fn constraint_lines(p: &PropertyShape) -> Vec<String> {
    let mut o = Vec::new();
    let mut line = |s: String| o.push(s);
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
        let items: Vec<String> = list.iter().map(turtle_entry).collect();
        line(format!("sh:in ( {} )", items.join(" ")));
    }
    for l in turtle_string_and_pair_lines(p) {
        line(l);
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
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::rdf::{iri, Term};

    fn pv_map() -> TargetMap {
        TargetMap::from([("pv-contract".to_string(), "ont:Contract".to_string())])
    }

    fn shape(yaml: &str) -> NodeShape {
        let doc: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        parse_shape_with("t", &doc, &pv_map()).unwrap().unwrap()
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
    fn allow_empty_carries_its_reason_and_refuses_a_blank_or_nested_one() {
        let with = |v: &str| {
            format!("entity: {{type: pv-contract}}\nshape:\n  allowEmpty: {v}\n  properties: []\n")
        };
        assert_eq!(
            shape(&with("\"none fit\"")).allow_empty.as_deref(),
            Some("none fit")
        );
        assert_eq!(shape(BASE).allow_empty, None);
        for bad in ["\"  \"", "true", "[a]"] {
            let doc: serde_yaml::Value = serde_yaml::from_str(&with(bad)).unwrap();
            assert!(
                matches!(
                    parse_shape_with("t", &doc, &pv_map()),
                    Err(ShapeError::Malformed { what, .. }) if what.contains("non-empty reason")
                ),
                "allowEmpty: {bad} must be refused for its reason"
            );
        }
        let nested = "entity: {type: pv-contract}\nshape:\n  properties:\n    - {path: ont:id, node: {allowEmpty: x, properties: []}}\n";
        let doc: serde_yaml::Value = serde_yaml::from_str(nested).unwrap();
        assert!(matches!(
            parse_shape_with("t", &doc, &pv_map()),
            Err(ShapeError::Malformed { what, .. }) if what.contains("nested node shape")
        ));
    }

    #[test]
    fn a_contract_level_allow_empty_map_exempts_by_id_and_refuses_a_stray_or_double_one() {
        let list = |extra: &str, own: &str| {
            format!("entity: {{type: pv-contract}}\n{extra}shapes:\n  - {{id: a, {own}properties: []}}\n  - {{id: b, properties: []}}\n")
        };
        let parse = |yaml: &str| {
            let doc: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
            parse_shapes_with("t", &doc, &pv_map())
        };
        let got = parse(&list("allow_empty: {a: \"none fit\"}\n", "")).unwrap();
        assert_eq!(got[0].allow_empty.as_deref(), Some("none fit"));
        assert_eq!(got[1].allow_empty, None, "only the named shape is exempt");
        for bad in [
            list("allow_empty: {z: why}\n", ""),
            list("allow_empty: {a: \"  \"}\n", ""),
            list("allow_empty: [a]\n", ""),
            list("allow_empty: {a: why}\n", "allowEmpty: why, "),
        ] {
            assert!(
                matches!(parse(&bad), Err(ShapeError::Malformed { .. })),
                "must be refused:\n{bad}"
            );
        }
    }

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
            parse_shape_with("t", &doc, &pv_map()),
            Err(ShapeError::Unsupported { .. })
        ));
    }

    #[test]
    fn less_than_pairs_fire_per_unordered_pair_and_an_incomparable_pair_is_a_result() {
        // #3611: the slk-post case — `part: {i, n}` with i <= n
        let s = shape(
            "shape:\n  targetClass: ont:Part\n  properties:\n    - {path: ont:i, lessThanOrEquals: ont:n}\n    - {path: ont:lo, lessThan: ont:hi}\n",
        );
        let lit = |v: &str, t: &str| Term::Literal {
            value: v.into(),
            datatype: format!("{XSD_NS}{t}"),
        };
        let part = |id: &str, pairs: &[(&str, Term)]| {
            let mut g = Graph::new();
            let s = iri("part", id);
            g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Part")));
            for (p, o) in pairs {
                g.insert(s.clone(), ont(p), o.clone());
            }
            g
        };
        let comps = |g: &Graph| -> Vec<&'static str> {
            let mut c: Vec<_> = validate(g, std::slice::from_ref(&s))
                .results
                .iter()
                .map(|r| r.component)
                .collect();
            c.sort_unstable();
            c
        };
        let ok = part(
            "ok",
            &[
                ("i", lit("3", "integer")),
                ("n", lit("3", "integer")),
                ("lo", lit("2.5", "decimal")),
                ("hi", lit("3", "integer")),
            ],
        );
        assert!(comps(&ok).is_empty(), "3 <= 3 and 2.5 < 3 hold");
        let bad = part(
            "bad",
            &[
                ("i", lit("4", "integer")),
                ("n", lit("3", "integer")),
                ("lo", lit("3", "integer")),
                ("hi", lit("3", "integer")),
            ],
        );
        assert_eq!(
            comps(&bad),
            vec!["lessThan", "lessThanOrEquals"],
            "4 <= 3 and 3 < 3 fail"
        );
        // numbers against a string, and an IRI: not comparable, so NOT ordered — a result, never a skip
        let mixed = part(
            "mixed",
            &[
                ("i", lit("1", "integer")),
                ("n", Term::string("a")),
                ("lo", Term::iri(ont("x"))),
                ("hi", lit("3", "integer")),
            ],
        );
        assert_eq!(comps(&mixed), vec!["lessThan", "lessThanOrEquals"]);
        // no value on the other side: no pair, no result (W3C lessThan-001 ValidResource2)
        let lone = part("lone", &[("i", lit("9", "integer"))]);
        assert!(comps(&lone).is_empty());
        // the exported Turtle carries both components, so the oracle sees what the gate checked
        let t = to_turtle(std::slice::from_ref(&s));
        assert!(
            t.contains(&format!("sh:lessThanOrEquals <{}>", ont("n"))),
            "{t}"
        );
        assert!(t.contains(&format!("sh:lessThan <{}>", ont("hi"))), "{t}");
    }

    #[test]
    fn equals_and_disjoint_compare_whole_value_sets_and_refuse_a_language_tag() {
        // #4814 slice 2 (#4600): equals is set equality, disjoint is an empty intersection, both by exact term
        let s = shape(
            "shape:\n  targetClass: ont:Part\n  properties:\n    - {path: ont:a, equals: ont:b}\n    - {path: ont:c, disjoint: ont:d}\n",
        );
        let part = |id: &str, pairs: &[(&str, Term)]| {
            let mut g = Graph::new();
            let s = iri("part", id);
            g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Part")));
            for (p, o) in pairs {
                g.insert(s.clone(), ont(p), o.clone());
            }
            g
        };
        let comps = |g: &Graph| -> Vec<&'static str> {
            let mut c: Vec<_> = validate(g, std::slice::from_ref(&s))
                .results
                .iter()
                .map(|r| r.component)
                .collect();
            c.sort_unstable();
            c
        };
        let int = |v: &str| Term::Literal {
            value: v.into(),
            datatype: format!("{XSD_NS}integer"),
        };
        let ok = part(
            "ok",
            &[
                ("a", Term::string("x")),
                ("a", Term::string("y")),
                ("b", Term::string("y")),
                ("b", Term::string("x")),
                ("c", Term::string("x")),
                ("d", Term::string("z")),
            ],
        );
        assert!(
            comps(&ok).is_empty(),
            "{{x,y}} = {{y,x}}; {{x}} and {{z}} share nothing"
        );
        assert!(
            comps(&part("empty", &[])).is_empty(),
            "two empty sets are equal and disjoint"
        );
        // one value each way, and a shared term: "1"^^integer is not "1"^^string, so that pair is not shared
        let bad = part(
            "bad",
            &[
                ("a", Term::string("x")),
                ("b", Term::string("w")),
                ("c", Term::string("s")),
                ("d", Term::string("s")),
                ("c", int("1")),
                ("d", Term::string("1")),
            ],
        );
        assert_eq!(comps(&bad), vec!["disjoint", "equals", "equals"]);
        // F9: a language-tagged value is never read as equal (the tag is dropped), so it is a result on both
        let lang = Term::Literal {
            value: "x".into(),
            datatype: RDF_LANG_STRING.into(),
        };
        let tagged = part(
            "tagged",
            &[
                ("a", lang.clone()),
                ("b", lang.clone()),
                ("c", lang),
                ("d", Term::string("q")),
            ],
        );
        assert_eq!(comps(&tagged), vec!["disjoint", "equals", "equals"]);
        let t = to_turtle(std::slice::from_ref(&s));
        assert!(t.contains(&format!("sh:equals <{}>", ont("b"))), "{t}");
        assert!(t.contains(&format!("sh:disjoint <{}>", ont("d"))), "{t}");
    }

    #[test]
    fn has_value_needs_one_exact_term_and_ignores_the_others() {
        // #4814 slice 3 (#3715): at least one value is the term; other values do not matter
        let s = shape(
            "shape:\n  targetClass: ont:Part\n  properties:\n    - {path: ont:g, hasValue: male}\n    - {path: ont:n, hasValue: 3}\n",
        );
        let part = |id: &str, pairs: &[(&str, Term)]| {
            let mut g = Graph::new();
            let s = iri("part", id);
            g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Part")));
            for (p, o) in pairs {
                g.insert(s.clone(), ont(p), o.clone());
            }
            g
        };
        let comps = |g: &Graph| -> Vec<&'static str> {
            validate(g, std::slice::from_ref(&s))
                .results
                .iter()
                .map(|r| r.component)
                .collect()
        };
        let int = |v: &str| Term::Literal {
            value: v.into(),
            datatype: format!("{XSD_NS}integer"),
        };
        let ok = part(
            "ok",
            &[
                ("g", Term::string("female")),
                ("g", Term::string("male")),
                ("n", int("3")),
            ],
        );
        assert!(comps(&ok).is_empty(), "one exact value is enough");
        // absent; the right lexical form with the wrong datatype; a language-tagged "male" (F9)
        let tagged = Term::Literal {
            value: "male".into(),
            datatype: RDF_LANG_STRING.into(),
        };
        let bad = part("bad", &[("g", tagged), ("n", Term::string("3"))]);
        assert_eq!(comps(&bad), vec!["hasValue", "hasValue"]);
        assert_eq!(comps(&part("none", &[])), vec!["hasValue", "hasValue"]);
        let t = to_turtle(std::slice::from_ref(&s));
        assert!(t.contains("sh:hasValue \"male\""), "{t}");
        assert!(
            t.contains(&format!("sh:hasValue \"3\"^^<{XSD_NS}integer>")),
            "{t}"
        );
        // a quote or backslash in the term is escaped, so the exported literal still parses
        let q = shape("shape:\n  targetClass: ont:Part\n  properties:\n    - {path: ont:g, hasValue: 'a\"b\\c'}\n");
        let t = to_turtle(std::slice::from_ref(&q));
        assert!(t.contains(r#"sh:hasValue "a\"b\\c""#), "{t}");
    }

    #[test]
    fn targets_beside_target_class_parse_as_a_string_or_a_list_and_never_widen_to_the_default() {
        // #4814 slice 4: an explicit target names the focus nodes; the entity-type default class is not added
        let s = shape("entity: {type: pv-contract}\nshape:\n  targetNode: ont:a\n  targetSubjectsOf: [ont:p, ont:q]\n  targetObjectsOf: ont:r\n  properties: []\n");
        assert_eq!(s.target_class, "");
        assert_eq!(s.targets.nodes, vec![Term::iri(ont("a"))]);
        assert_eq!(s.targets.subjects_of, vec![ont("p"), ont("q")]);
        assert_eq!(s.targets.objects_of, vec![ont("r")]);
        // beside an explicit targetClass both are kept
        let both =
            shape("shape:\n  targetClass: ont:A\n  targetSubjectsOf: ont:p\n  properties: []\n");
        assert_eq!(both.target_class, ont("A"));
        assert_eq!(both.targets.subjects_of, vec![ont("p")]);
        assert!(shape("shape:\n  targetClass: ont:A\n  properties: []\n")
            .targets
            .is_empty());
        let parse =
            |y: &str| parse_shape_with("t", &serde_yaml::from_str(y).expect("yaml"), &pv_map());
        for (y, want) in [
            ("shape:\n  targetNode: []\n  properties: []\n", "not a string"),
            ("shape:\n  targetNode: [ont:a, 3]\n  properties: []\n", "not a string"),
            ("shape:\n  targetSubjectsOf: {a: b}\n  properties: []\n", "not a string"),
            ("shape:\n  targetObjectsOf: ~\n  properties: []\n", "not a string"),
            (
                "entity: {type: pv-contract}\nshape:\n  properties: [{path: ont:x, node: {targetNode: ont:a, properties: []}}]\n",
                "nested",
            ),
        ] {
            match parse(y) {
                Err(ShapeError::Malformed { what, .. }) => assert!(what.contains(want), "{y}: {what}"),
                other => panic!("{y}: expected Malformed, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_literal_target_node_is_a_typed_mapping_and_a_bare_scalar_stays_an_iri() {
        // #4814 slice 5: `{literal, datatype}`, exactly those two keys, and only under targetNode
        let s = shape("shape:\n  targetNode: [ont:a, {literal: '7', datatype: xsd:integer}]\n  minInclusive: 8\n");
        assert_eq!(s.targets.nodes, vec![Term::iri(ont("a")), Term::integer(7)]);
        let parse =
            |y: &str| parse_shape_with("t", &serde_yaml::from_str(y).expect("yaml"), &pv_map());
        for y in [
            "shape:\n  targetNode: {literal: '7'}\n",
            "shape:\n  targetNode: {literal: '7', datatype: xsd:integer, lang: en}\n",
            "shape:\n  targetNode: {literal: 7, datatype: xsd:integer}\n",
            "shape:\n  targetSubjectsOf: {literal: '7', datatype: xsd:integer}\n",
        ] {
            match parse(y) {
                Err(ShapeError::Malformed { what, .. }) => {
                    assert!(what.contains("not a string"), "{y}: {what}")
                }
                other => panic!("{y}: expected Malformed, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_node_shape_constrains_its_focus_itself_with_no_result_path() {
        // #4814 slice 5: the value set is {focus}; a literal focus is named by its N-Triples form
        let s = shape("shape:\n  targetNode: [ont:a, {literal: '7', datatype: xsd:integer}, {literal: '9', datatype: xsd:integer}]\n  minInclusive: 8\n  nodeKind: Literal\n");
        let own = s.own.as_ref().expect("node-level constraints");
        assert_eq!(own.keys, vec!["minInclusive", "nodeKind"]);
        let r = validate(&Graph::new(), std::slice::from_ref(&s));
        let got: Vec<(&str, Option<&str>, &str)> = r
            .results
            .iter()
            .map(|x| (x.focus.as_str(), x.path.as_deref(), x.component))
            .collect();
        let seven = Term::integer(7).to_string();
        let a = ont("a");
        let mut want = vec![
            (a.as_str(), None, "minInclusive"),
            (a.as_str(), None, "nodeKind"),
            (seven.as_str(), None, "minInclusive"),
        ];
        want.sort_unstable();
        let mut got = got;
        got.sort_unstable();
        assert_eq!(got, want, "{:?}", r.results);
        assert_eq!(r.focus_nodes_n, 3);
        // the export writes the literal target and the constraints on the node shape itself
        let t = to_turtle(std::slice::from_ref(&s));
        assert!(
            t.contains(&format!("sh:targetNode \"7\"^^<{XSD_NS}integer> ;")),
            "{t}"
        );
        assert!(t.contains("    sh:nodeKind sh:Literal ;\n"), "{t}");
        // languageIn stays refused on a node shape too
        let doc: serde_yaml::Value =
            serde_yaml::from_str("shape:\n  targetNode: ont:a\n  languageIn: [en]\n")
                .expect("yaml");
        assert!(matches!(
            parse_shape_with("t", &doc, &pv_map()),
            Err(ShapeError::Unsupported { component, .. }) if component == "languageIn"
        ));
    }

    #[test]
    fn focus_nodes_are_the_union_of_every_target_without_repeats() {
        let s = shape("shape:\n  targetClass: ont:A\n  targetNode: [ont:a, ont:ghost]\n  targetSubjectsOf: ont:p\n  targetObjectsOf: ont:r\n  properties: []\n");
        let mut g = Graph::new();
        g.insert(ont("a"), RDF_TYPE, Term::iri(ont("A")));
        g.insert(ont("b"), ont("p"), Term::string("x"));
        g.insert(ont("a"), ont("p"), Term::string("y"));
        g.insert(ont("c"), ont("r"), Term::iri(ont("d")));
        g.insert(ont("c"), ont("r"), Term::string("lit"));
        g.insert(ont("e"), ont("q"), Term::iri(ont("f")));
        let mut want = vec![
            ont("a"),
            ont("b"),
            ont("d"),
            ont("ghost"),
            Term::string("lit").to_string(),
        ];
        want.sort();
        assert_eq!(
            focus_nodes(&g, &s),
            want,
            "a once; ghost though absent; c and e never"
        );
        // a targetNode absent from the graph is still checked, so its minCount fires
        let m = shape("entity: {type: pv-contract}\nshape:\n  targetNode: ont:ghost\n  properties: [{path: ont:x, minCount: 1}]\n");
        let r = validate(&g, std::slice::from_ref(&m));
        assert_eq!(r.results.len(), 1);
        assert_eq!(r.results[0].focus, ont("ghost"));
        let t = to_turtle(std::slice::from_ref(&s));
        for line in [
            format!("sh:targetNode <{}>", ont("ghost")),
            format!("sh:targetSubjectsOf <{}>", ont("p")),
            format!("sh:targetObjectsOf <{}>", ont("r")),
        ] {
            assert!(t.contains(&line), "{line} in {t}");
        }
    }

    #[test]
    fn compare_terms_orders_numbers_across_types_and_refuses_mixed_kinds() {
        use std::cmp::Ordering::{Equal, Greater, Less};
        let lit = |v: &str, t: &str| Term::Literal {
            value: v.into(),
            datatype: format!("{XSD_NS}{t}"),
        };
        assert_eq!(
            compare_terms(&lit("4", "integer"), &lit("4.5", "decimal")),
            Some(Less)
        );
        assert_eq!(
            compare_terms(&lit("10", "int"), &lit("9", "integer")),
            Some(Greater)
        );
        assert_eq!(
            compare_terms(&lit("1e1", "double"), &lit("10", "integer")),
            Some(Equal)
        );
        assert_eq!(
            compare_terms(&lit("2026-09-01", "date"), &lit("2026-09-24", "date")),
            Some(Less)
        );
        assert_eq!(
            compare_terms(&Term::string("a"), &Term::string("b")),
            Some(Less)
        );
        assert_eq!(
            compare_terms(&lit("false", "boolean"), &lit("true", "boolean")),
            Some(Less)
        );
        assert_eq!(
            compare_terms(&lit("1", "integer"), &Term::string("1")),
            None
        );
        assert_eq!(
            compare_terms(&lit("2026-09-01", "date"), &Term::string("2026-09-02")),
            None
        );
        assert_eq!(
            compare_terms(&Term::iri(ont("a")), &Term::iri(ont("b"))),
            None
        );
        assert_eq!(
            compare_terms(&lit("x", "integer"), &lit("1", "integer")),
            None
        );
    }

    #[test]
    fn unsupported_components_are_refused_at_parse_by_name() {
        for (yaml, want) in [
            ("entity: {type: pv-contract}\nshape:\n  properties: [{path: ont:x, qualifiedValueShape: {}}]\n", "qualifiedValueShape"),
            ("entity: {type: pv-contract}\nshape:\n  properties: [{path: ont:x, languageIn: [en]}]\n", "languageIn"),
            ("entity: {type: pv-contract}\nshape:\n  properties: [{path: 'ont:a/ont:b'}]\n", "path `ont:a/ont:b`"),
            ("entity: {type: pv-contract}\nshape:\n  or: []\n", "or"),
        ] {
            let doc: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
            match parse_shape_with("t", &doc, &pv_map()) {
                Err(ShapeError::Unsupported { component, .. }) => assert!(component.starts_with(want), "{component} vs {want}"),
                other => panic!("{yaml}: expected Unsupported, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_known_key_with_a_wrong_typed_value_is_refused_never_read_as_absent() {
        // #4814 fail-closed: each row was dropped without an error before (F1–F8 of evidence/4814/baseline.md).
        // `ok` is the same key, well typed, so a row that passes only because the shape never parses is caught.
        let p = |prop: &str| {
            format!(
                "entity: {{type: pv-contract}}\nshape:\n  properties: [{{path: ont:x, {prop}}}]\n"
            )
        };
        let rows: [(&str, String, String); 18] = [
            ("F1 targetClass", "shape:\n  targetClass: [ont:A]\n  properties: []\n".into(), "shape:\n  targetClass: ont:A\n  properties: []\n".into()),
            ("F2 datatype", p("datatype: 5"), p("datatype: xsd:string")),
            ("F2 class", p("class: [ont:A]"), p("class: ont:A")),
            ("F2 lessThan", p("lessThan: {a: 1}"), p("lessThan: ont:y")),
            ("F2 lessThanOrEquals", p("lessThanOrEquals: 1"), p("lessThanOrEquals: ont:y")),
            ("F3 pattern", p("pattern: 7"), p("pattern: '^a'")),
            ("F4 nodeKind", p("nodeKind: [IRI]"), p("nodeKind: IRI")),
            ("F5 severity", p("severity: 1"), p("severity: warning")),
            ("F6 ignoredProperties", "entity: {type: pv-contract}\nshape:\n  closed: true\n  ignoredProperties: [rdf:type, 3]\n  properties: []\n".into(), "entity: {type: pv-contract}\nshape:\n  closed: true\n  ignoredProperties: [rdf:type]\n  properties: []\n".into()),
            ("F7 in", p("in: [a, {b: 1}]"), p("in: [a, b]")),
            ("F2 resolves", p("resolves: [x]"), p("resolves: x")),
            ("slice 1 minExclusive", p("minExclusive: [1]"), p("minExclusive: 1")),
            ("slice 1 minInclusive", p("minInclusive: {a: 1}"), p("minInclusive: 1.5")),
            ("slice 1 maxExclusive", p("maxExclusive: ~"), p("maxExclusive: 'z'")),
            ("slice 1 maxInclusive", p("maxInclusive: [true]"), p("maxInclusive: true")),
            ("slice 2 equals", p("equals: [ont:y]"), p("equals: ont:y")),
            ("slice 2 disjoint", p("disjoint: 3"), p("disjoint: ont:y")),
            ("slice 3 hasValue", p("hasValue: [male]"), p("hasValue: male")),
        ];
        for (row, bad, ok) in rows {
            let parse =
                |y: &str| parse_shape_with("t", &serde_yaml::from_str(y).expect("yaml"), &pv_map());
            assert!(
                parse(&ok).is_ok(),
                "{row}: control must parse: {:?}",
                parse(&ok)
            );
            match parse(&bad) {
                Err(ShapeError::Malformed { what, .. }) => {
                    assert!(what.contains("not a"), "{row}: {what}")
                }
                other => panic!("{row}: expected Malformed, got {other:?}"),
            }
        }
        // F8: a path expression or two predicates written so they look like one predicate
        for path in [
            "'http://x/a|http://x/b'",
            "'^http://x/a'",
            "'http://x/a http://x/b'",
            "'ont:a ont:b'",
        ] {
            let doc: serde_yaml::Value =
                serde_yaml::from_str(&p(&format!("minCount: 0}}, {{path: {path}"))).expect("yaml");
            assert!(
                matches!(
                    parse_shape_with("t", &doc, &pv_map()),
                    Err(ShapeError::Unsupported { .. })
                ),
                "F8 {path}"
            );
        }
        let doc: serde_yaml::Value =
            serde_yaml::from_str(&p("minCount: 0}, {path: 'https://x.org/a/b*c+d'")).expect("yaml");
        assert!(
            parse_shape_with("t", &doc, &pv_map()).is_ok(),
            "F8 control: a full IRI keeps / * +"
        );
    }

    #[test]
    fn a_shape_with_no_target_and_no_pv_contract_entity_is_malformed() {
        let doc: serde_yaml::Value = serde_yaml::from_str("shape:\n  properties: []\n").unwrap();
        assert!(matches!(
            parse_shape_with("t", &doc, &pv_map()),
            Err(ShapeError::Malformed { .. })
        ));
        let doc: serde_yaml::Value =
            serde_yaml::from_str("shape:\n  targetClass: ont:Contract\n  properties: []\n")
                .unwrap();
        assert_eq!(
            parse_shape_with("t", &doc, &pv_map())
                .unwrap()
                .unwrap()
                .target_class,
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
    fn in_entry_types_each_yaml_scalar_by_its_own_kind() {
        let e = |y: &str| {
            let v: serde_yaml::Value = serde_yaml::from_str(y).expect("yaml");
            let x = in_entry(&v).expect("a scalar");
            (x.lexical, x.datatype)
        };
        let xsd = |t: &str| format!("{XSD_NS}{t}");
        assert_eq!(e("7"), ("7".to_string(), xsd("integer")));
        assert_eq!(e("1.5"), ("1.5".to_string(), xsd("double")));
        assert_eq!(e("true"), ("true".to_string(), xsd("boolean")));
        assert_eq!(e("false"), ("false".to_string(), xsd("boolean")));
        assert_eq!(e("a"), ("a".to_string(), XSD_STRING_IRI.to_string()));
        for y in ["[1]", "{a: 1}", "~"] {
            let v: serde_yaml::Value = serde_yaml::from_str(y).expect("yaml");
            assert!(in_entry(&v).is_none(), "{y} names no term");
        }
    }

    #[test]
    fn well_formed_checks_each_xsd_type_it_names() {
        let wf = |v: &str, t: &str| well_formed(v, &format!("{XSD_NS}{t}"));
        // boolean
        assert!(wf("true", "boolean") && wf("0", "boolean"));
        assert!(!wf("yes", "boolean"));
        // decimal
        assert!(wf("1.5", "decimal") && wf("-2", "decimal"));
        assert!(!wf("abc", "decimal"));
        // double / float
        for t in ["double", "float"] {
            assert!(wf("1.5e3", t) && wf("INF", t) && wf("-INF", t) && wf("NaN", t));
            assert!(!wf("abc", t) && !wf("", t));
        }
        // date / dateTime
        assert!(wf("2026-01-02", "date"));
        assert!(!wf("nope", "date"));
        assert!(wf("2026-01-02T10:11:12", "dateTime"));
        assert!(!wf("nope", "dateTime"));
        // non-xsd datatype and unknown local names are taken as well-formed
        assert!(well_formed("x", "http://example.org/t"));
        assert!(wf("anything", "anyURI"));
    }

    #[test]
    fn integer_types_enforce_their_exact_bounds() {
        let wf = |v: &str, t: &str| well_formed(v, &format!("{XSD_NS}{t}"));
        let cases: [(&str, &str, &str); 14] = [
            ("integer", "170141183460469231731687303715884105727", "x"),
            ("long", "9223372036854775807", "9223372036854775808"),
            ("int", "2147483647", "2147483648"),
            ("short", "32767", "32768"),
            ("byte", "127", "128"),
            ("nonNegativeInteger", "0", "-1"),
            ("positiveInteger", "1", "0"),
            ("nonPositiveInteger", "0", "1"),
            ("negativeInteger", "-1", "0"),
            (
                "unsignedLong",
                "18446744073709551615",
                "18446744073709551616",
            ),
            ("unsignedInt", "4294967295", "4294967296"),
            ("unsignedShort", "65535", "65536"),
            ("unsignedByte", "255", "256"),
            ("unsignedByte", "0", "-1"),
        ];
        for (t, ok, bad) in cases {
            assert!(wf(ok, t), "{ok} is a valid xsd:{t}");
            assert!(!wf(bad, t), "{bad} is not a valid xsd:{t}");
        }
        assert!(wf("-32768", "short") && !wf("-32769", "short"));
        assert!(wf("-128", "byte") && !wf("-129", "byte"));
        assert!(wf("-9223372036854775808", "long") && !wf("-9223372036854775809", "long"));
        assert!(wf("-2147483648", "int") && !wf("-2147483649", "int"));
        assert!(wf("-5", "negativeInteger") && wf("-5", "nonPositiveInteger"));
        assert!(wf("-170141183460469231731687303715884105728", "integer"));
        assert!(wf(
            "170141183460469231731687303715884105727",
            "nonNegativeInteger"
        ));
        assert!(wf(
            "170141183460469231731687303715884105727",
            "positiveInteger"
        ));
        assert!(wf(
            "-170141183460469231731687303715884105728",
            "nonPositiveInteger"
        ));
        assert!(wf(
            "-170141183460469231731687303715884105728",
            "negativeInteger"
        ));
    }

    #[test]
    fn is_decimal_requires_digits_one_dot_and_only_digits() {
        for ok in ["1", "1.5", ".5", "5.", "+1", "-1.25"] {
            assert!(is_decimal(ok), "{ok}");
        }
        for bad in [
            "", ".", "+", "-", "1.2.3", "a", "1a", "1.a", "--1", "a.5", "1,5",
        ] {
            assert!(!is_decimal(bad), "{bad}");
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
