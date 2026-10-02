//! ONT-001 R-19, row ONT-4d — shapes inherit down Σ's subsumption hierarchy.
//!
//! Two things the `shapes` gate reports from Σ's `subsumes`:
//!
//! - **inheritance** ([`inherited`]): a shape whose `targetClass` is a Σ concept `C` applies to the instances of
//!   every strict sub-concept `D ⊑ C`. The validator already sees them: `extract::all` materializes the rdf:type
//!   closure, so each instance of `D` is also typed `C`. This counts those applications, so a report shows the
//!   hierarchy applied, never merely declared. On aprender's corpus every `Kernel` is ALSO asserted
//!   `ont:Contract` directly, so there the count shows application, not necessity. In the
//!   `subsumption-inherit` fixture, the closure is the ONLY way a `Kernel` instance reaches the `Code` shape.
//!   Drop the closure there, and that fixture's violation stops firing: the row's mutation.
//! - **weakening** ([`weakenings`]): "a sub-concept may add constraints and may not remove any". For shapes
//!   `S_sub` on `D` and `S_sup` on `C` with `D ⊑ C`, every property path both constrain must keep each of
//!   `S_sup`'s components at least as strict in `S_sub`. A component that is dropped, or loosened, is
//!   `reject: <S_sub> weakens <S_sup>.<path>.<component>`, a violation (exit 1). A sub-shape that does not
//!   mention the path removes nothing: `S_sup` still applies through inheritance.
//!
//! **Why weakening is checked at all.** The inherited super-shape still validates every sub-instance, so a
//! looser sub-shape cannot make a violation pass. It can MISLEAD a reader, who sees `minCount 0` on the
//! sub-shape and concludes the property is optional there when it is not. R-19 refuses that
//! contradiction at the declaration.

use std::collections::BTreeMap;

use crate::ontology::rdf::{ont, Graph};
use crate::ontology::shapes::{NodeShape, PropertyShape};
use crate::ontology::sigma::Sigma;

/// The Σ concept a `targetClass` IRI names, if it names one.
fn concept_of<'a>(sigma: &'a Sigma, target: &str) -> Option<&'a str> {
    sigma
        .concepts
        .keys()
        .map(String::as_str)
        .find(|c| ont(c) == target)
}

/// `(total, ["<shape> <- <sub>=<n>", …])`: focus nodes reached through a strict sub-concept, per shape.
#[must_use]
pub fn inherited(graph: &Graph, shapes: &[NodeShape], sigma: &Sigma) -> (usize, Vec<String>) {
    let mut total = 0;
    let mut rows = Vec::new();
    for s in shapes {
        let Some(c) = concept_of(sigma, &s.target_class) else {
            continue;
        };
        for d in sigma.subs(c) {
            let n = graph.instances_of(&ont(&d)).len();
            if n > 0 {
                total += n;
                rows.push(format!("{} <- {d}={n}", s.id));
            }
        }
    }
    rows.sort();
    (total, rows)
}

/// `minCount` weakens iff `sub` allows fewer occurrences than `sup` required.
fn weakened_min_count(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    let ok = match (sup.min_count, sub.min_count) {
        (Some(a), Some(b)) => b >= a,
        (Some(_), None) => false,
        (None, _) => true,
    };
    (!ok).then_some("minCount")
}

/// `maxCount` weakens iff `sub` allows more occurrences than `sup` capped.
fn weakened_max_count(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    let ok = match (sup.max_count, sub.max_count) {
        (Some(a), Some(b)) => b <= a,
        (Some(_), None) => false,
        (None, _) => true,
    };
    (!ok).then_some("maxCount")
}

/// `datatype` weakens iff `sup` constrains it and `sub` drops or changes it.
fn weakened_datatype(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    (sup.datatype.is_some() && sup.datatype != sub.datatype).then_some("datatype")
}

/// `class` weakens iff `sup` constrains it and `sub` drops or changes it.
fn weakened_class(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    (sup.class.is_some() && sup.class != sub.class).then_some("class")
}

/// `nodeKind` weakens iff `sup` constrains it and `sub` drops or changes it.
fn weakened_node_kind(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    (sup.node_kind.is_some() && sup.node_kind != sub.node_kind).then_some("nodeKind")
}

/// `pattern` weakens iff `sup` constrains it and `sub` drops or changes the source regex.
fn weakened_pattern(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    let src = |p: &Option<(String, regex::Regex)>| p.as_ref().map(|(s, _)| s.clone());
    (sup.pattern.is_some() && src(&sup.pattern) != src(&sub.pattern)).then_some("pattern")
}

/// `in` weakens iff `sup` constrains it and `sub`'s allowed set is not a subset of `sup`'s.
fn weakened_in(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    let Some(allowed) = &sup.r#in else {
        return None;
    };
    let narrower = sub
        .r#in
        .as_ref()
        .is_some_and(|s| s.iter().all(|x| allowed.contains(x)));
    (!narrower).then_some("in")
}

/// `minLength` weakens iff `sup` required a floor and `sub` allows shorter (or none).
fn weakened_min_length(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    sup.min_length
        .is_some_and(|a| sub.min_length.is_none_or(|b| b < a))
        .then_some("minLength")
}

/// `maxLength` weakens iff `sup` capped it and `sub` allows longer (or none).
fn weakened_max_length(sup: &PropertyShape, sub: &PropertyShape) -> Option<&'static str> {
    sup.max_length
        .is_some_and(|a| sub.max_length.is_none_or(|b| b > a))
        .then_some("maxLength")
}

/// The components of `sup` that `sub` drops or loosens, on the same path.
fn weakened(sup: &PropertyShape, sub: &PropertyShape) -> Vec<&'static str> {
    let checks: [fn(&PropertyShape, &PropertyShape) -> Option<&'static str>; 9] = [
        weakened_min_count,
        weakened_max_count,
        weakened_datatype,
        weakened_class,
        weakened_node_kind,
        weakened_pattern,
        weakened_in,
        weakened_min_length,
        weakened_max_length,
    ];
    checks.iter().filter_map(|f| f(sup, sub)).collect()
}

/// Every `reject: <S_sub> weakens <S_sup>.<path>.<component>`, sorted.
#[must_use]
pub fn weakenings(shapes: &[NodeShape], sigma: &Sigma) -> Vec<String> {
    let by_concept: BTreeMap<&str, Vec<&NodeShape>> =
        shapes.iter().fold(BTreeMap::new(), |mut m, s| {
            if let Some(c) = concept_of(sigma, &s.target_class) {
                m.entry(c).or_default().push(s);
            }
            m
        });
    let mut out = Vec::new();
    for (&d, subs) in &by_concept {
        for c in sigma.supers(d) {
            let Some(sups) = by_concept.get(c.as_str()) else {
                continue;
            };
            for sub in subs {
                for sup in sups {
                    for sp in &sup.properties {
                        let Some(bp) = sub.properties.iter().find(|p| p.path == sp.path) else {
                            continue;
                        };
                        let local = sp.path.rsplit(['/', '#']).next().unwrap_or(&sp.path);
                        out.extend(
                            weakened(sp, bp)
                                .into_iter()
                                .map(|k| format!("{} weakens {}.{local}.{k}", sub.id, sup.id)),
                        );
                    }
                }
            }
        }
    }
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One case-table row: the component the mutation must name, and the mutation.
    type Case = (&'static str, fn(&mut PropertyShape));

    fn p(path: &str) -> PropertyShape {
        PropertyShape {
            path: path.into(),
            min_count: None,
            max_count: None,
            datatype: None,
            class: None,
            node_kind: None,
            r#in: None,
            pattern: None,
            min_length: None,
            max_length: None,
            node: None,
            less_than: None,
            less_than_or_equals: None,
            resolves: None,
            severity: crate::ontology::shapes::Severity::Violation,
        }
    }

    /// The component table: each row loosens or drops ONE component and must name exactly that one; the
    /// strictly-tighter and the equal rows must name none (a sub-concept MAY add constraints).
    #[test]
    fn ont4d_weakened_names_exactly_the_loosened_component() {
        let mut sup = p("x");
        sup.min_count = Some(1);
        sup.max_count = Some(2);
        sup.datatype = Some("xsd:string".into());
        sup.min_length = Some(3);
        sup.max_length = Some(9);
        let same = sup.clone();
        assert!(weakened(&sup, &same).is_empty(), "equal is not weaker");
        let mut tighter = sup.clone();
        tighter.min_count = Some(2);
        tighter.max_count = Some(2);
        tighter.min_length = Some(4);
        tighter.max_length = Some(8);
        assert!(weakened(&sup, &tighter).is_empty(), "stricter is allowed");
        let cases: [Case; 9] = [
            ("minCount", |b| b.min_count = Some(0)),
            ("minCount", |b| b.min_count = None),
            ("maxCount", |b| b.max_count = Some(5)),
            ("maxCount", |b| b.max_count = None),
            ("datatype", |b| b.datatype = Some("xsd:integer".into())),
            ("datatype", |b| b.datatype = None),
            ("minLength", |b| b.min_length = Some(1)),
            ("maxLength", |b| b.max_length = Some(99)),
            ("maxLength", |b| b.max_length = None),
        ];
        for (want, mutate) in cases {
            let mut sub = sup.clone();
            mutate(&mut sub);
            assert_eq!(weakened(&sup, &sub), vec![want], "loosening {want}");
        }
    }

    /// Quorum lane 1 (PMAT-4070): the class / nodeKind / pattern / in branches had no case. Each is loosened or
    /// dropped once and must be named alone; each equal or narrowed form must name nothing.
    #[test]
    fn ont4d_weakened_covers_class_nodekind_pattern_and_in() {
        use crate::ontology::shapes::{InEntry, NodeKind};
        let e = |s: &str| InEntry {
            lexical: s.into(),
            datatype: "xsd:string".into(),
        };
        let rx = |s: &str| Some((s.to_string(), regex::Regex::new(s).expect("regex")));
        let mut sup = p("x");
        sup.class = Some("ont:Code".into());
        sup.node_kind = Some(NodeKind::Iri);
        sup.pattern = rx("^a");
        sup.r#in = Some(vec![e("a"), e("b")]);
        assert!(
            weakened(&sup, &sup.clone()).is_empty(),
            "equal is not weaker"
        );
        let mut narrower = sup.clone();
        narrower.r#in = Some(vec![e("a")]);
        assert!(
            weakened(&sup, &narrower).is_empty(),
            "a subset of `in` is stricter, allowed"
        );
        let cases: [Case; 8] = [
            ("class", |b| b.class = Some("ont:Contract".into())),
            ("class", |b| b.class = None),
            ("nodeKind", |b| b.node_kind = Some(NodeKind::Literal)),
            ("nodeKind", |b| b.node_kind = None),
            ("pattern", |b| {
                b.pattern = Some(("^b".into(), regex::Regex::new("^b").expect("regex")))
            }),
            ("pattern", |b| b.pattern = None),
            ("in", |b| {
                b.r#in = Some(vec![InEntry {
                    lexical: "c".into(),
                    datatype: "xsd:string".into(),
                }])
            }),
            ("in", |b| b.r#in = None),
        ];
        for (want, mutate) in cases {
            let mut sub = sup.clone();
            mutate(&mut sub);
            assert_eq!(weakened(&sup, &sub), vec![want], "loosening {want}");
        }
    }

    #[test]
    fn ont4d_a_super_with_no_constraint_cannot_be_weakened() {
        let sup = p("x");
        let mut sub = p("x");
        sub.min_count = Some(1);
        assert!(weakened(&sup, &sub).is_empty());
    }

    fn shape(id: &str, target: &str) -> NodeShape {
        NodeShape {
            id: id.into(),
            target_class: target.into(),
            closed: false,
            ignored_properties: vec![],
            properties: vec![],
            allow_empty: None,
        }
    }

    fn sigma_of(concepts: &[&str], edges: &[(&str, &str)]) -> Sigma {
        let mut y = String::from("schema: ont-test\nconcepts:\n");
        for c in concepts {
            y.push_str(&format!("  {c}:\n    doc: d\n"));
        }
        y.push_str("subsumes:\n");
        for (sub, sup) in edges {
            y.push_str(&format!("  - sub: {sub}\n    sup: {sup}\n"));
        }
        serde_yaml::from_str(&y).expect("sigma yaml")
    }

    /// `concept_of` names the concept whose IRI is the target and no other; `inherited` totals the instances of
    /// every strict sub-concept per shape (empty sub-concepts are not rows) and sorts its rows.
    #[test]
    fn ont4d_inherited_counts_instances_of_strict_sub_concepts() {
        use crate::ontology::rdf::{Term, RDF_TYPE};
        let sigma = sigma_of(
            &["A", "B", "C", "D", "E", "F"],
            &[("B", "A"), ("C", "A"), ("D", "A"), ("F", "E")],
        );
        assert_eq!(concept_of(&sigma, &ont("A")), Some("A"));
        assert_eq!(concept_of(&sigma, &ont("F")), Some("F"));
        assert_eq!(concept_of(&sigma, "http://example.org/Other"), None);
        assert_eq!(concept_of(&sigma, ""), None);
        let mut g = Graph::new();
        for (subj, class) in [("x1", "B"), ("x2", "B"), ("x3", "C"), ("x4", "F")] {
            g.insert(subj, RDF_TYPE, Term::iri(ont(class)));
        }
        let shapes = [
            shape("s-e", &ont("E")),
            shape("s-a", &ont("A")),
            shape("s-none", "http://example.org/Other"),
        ];
        let (total, rows) = inherited(&g, &shapes, &sigma);
        assert_eq!(total, 4);
        assert_eq!(rows, ["s-a <- B=2", "s-a <- C=1", "s-e <- F=1"]);
        assert_eq!(inherited(&g, &shapes[2..], &sigma), (0, vec![]));
        assert_eq!(
            inherited(&g, &[shape("s-d", &ont("D"))], &sigma),
            (0, vec![])
        );
    }
}
