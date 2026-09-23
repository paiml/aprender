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

/// The components of `sup` that `sub` drops or loosens, on the same path.
fn weakened(sup: &PropertyShape, sub: &PropertyShape) -> Vec<&'static str> {
    let mut out = Vec::new();
    let min_ok = match (sup.min_count, sub.min_count) {
        (Some(a), Some(b)) => b >= a,
        (Some(_), None) => false,
        (None, _) => true,
    };
    if !min_ok {
        out.push("minCount");
    }
    let max_ok = match (sup.max_count, sub.max_count) {
        (Some(a), Some(b)) => b <= a,
        (Some(_), None) => false,
        (None, _) => true,
    };
    if !max_ok {
        out.push("maxCount");
    }
    let same = |a: &Option<String>, b: &Option<String>| a.is_none() || a == b;
    if !same(&sup.datatype, &sub.datatype) {
        out.push("datatype");
    }
    if !same(&sup.class, &sub.class) {
        out.push("class");
    }
    if sup.node_kind.is_some() && sup.node_kind != sub.node_kind {
        out.push("nodeKind");
    }
    let pat = |p: &Option<(String, regex::Regex)>| p.as_ref().map(|(s, _)| s.clone());
    if sup.pattern.is_some() && pat(&sup.pattern) != pat(&sub.pattern) {
        out.push("pattern");
    }
    if let Some(allowed) = &sup.r#in {
        let narrower = sub
            .r#in
            .as_ref()
            .is_some_and(|s| s.iter().all(|x| allowed.contains(x)));
        if !narrower {
            out.push("in");
        }
    }
    if sup
        .min_length
        .is_some_and(|a| sub.min_length.is_none_or(|b| b < a))
    {
        out.push("minLength");
    }
    if sup
        .max_length
        .is_some_and(|a| sub.max_length.is_none_or(|b| b > a))
    {
        out.push("maxLength");
    }
    out
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
        let cases: [(&str, fn(&mut PropertyShape)); 9] = [
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

    #[test]
    fn ont4d_a_super_with_no_constraint_cannot_be_weakened() {
        let sup = p("x");
        let mut sub = p("x");
        sub.min_count = Some(1);
        assert!(weakened(&sup, &sub).is_empty());
    }
}
