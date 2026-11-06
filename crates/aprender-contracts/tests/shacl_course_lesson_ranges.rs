//! SHACL numeric ranges against the real course-lesson facts (COURSE-SHAPES-001 §4): m1-05 must fail on
//! `globalMotionFraction <= 0.20`; m1-01 passes every range and fails only the plot rule. The gate has no
//! Turtle reader, so the fixtures (a flat `ads:` subset) are read by the small parser below.

use provable_contracts::ontology::rdf::{Graph, Term, RDF_TYPE};
use provable_contracts::ontology::shapes::{
    parse_shape_with, validate, TargetMap, ValidationResult, XSD_NS,
};

const ADS: &str = "https://paiml.com/ns/ads#";
const M1_01: &str = include_str!("fixtures/course-shapes/m1-01-a5633bebda56.ttl");
const M1_05: &str = include_str!("fixtures/course-shapes/m1-05-c51e06e80dd9.ttl");

fn term(raw: &str) -> Term {
    let raw = raw.trim().trim_end_matches([';', '.']).trim();
    if let Some(rest) = raw.strip_prefix('"') {
        let (lex, tail) = rest.split_once('"').unwrap_or((rest, ""));
        return match tail.strip_prefix("^^xsd:") {
            Some(t) => Term::Literal {
                value: lex.to_string(),
                datatype: format!("{XSD_NS}{t}"),
            },
            None => Term::string(lex),
        };
    }
    match raw {
        "true" => Term::boolean(true),
        "false" => Term::boolean(false),
        other => Term::iri(other.replace("ads:", ADS)),
    }
}

fn load(ttl: &str) -> Graph {
    let mut g = Graph::new();
    let mut subject = String::new();
    for line in ttl.lines().filter(|l| !l.starts_with("@prefix")) {
        let mut body = line.trim();
        if let Some((s, rest)) = body.split_once(" a ads:RenderedLesson") {
            subject = s.replace("ads:", ADS);
            g.insert(
                subject.clone(),
                RDF_TYPE,
                Term::iri(format!("{ADS}RenderedLesson")),
            );
            body = rest;
        }
        if let Some((p, o)) = body.trim().split_once(' ') {
            if let Some(local) = p.strip_prefix("ads:") {
                g.insert(subject.clone(), format!("{ADS}{local}"), term(o));
            }
        }
    }
    g
}

const SHAPE: &str = "
shape:
  targetClass: https://paiml.com/ns/ads#RenderedLesson
  properties:
    - {path: https://paiml.com/ns/ads#loudnessLUFS, minCount: 1, minInclusive: -19, maxInclusive: -17}
    - {path: https://paiml.com/ns/ads#maxStillS, minCount: 1, maxInclusive: 5.0}
    - {path: https://paiml.com/ns/ads#globalMotionFraction, minCount: 1, maxInclusive: 0.20}
    - {path: https://paiml.com/ns/ads#endCardSSIM, minCount: 1, minInclusive: 0.90}
    - {path: https://paiml.com/ns/ads#plotWritten, lessThanOrEquals: https://paiml.com/ns/ads#plotShown}
";

fn results(ttl: &str) -> Result<Vec<ValidationResult>, String> {
    let doc: serde_yaml::Value = serde_yaml::from_str(SHAPE).map_err(|e| e.to_string())?;
    let shape = parse_shape_with("course-lesson-v1", &doc, &TargetMap::new())
        .map_err(|e| format!("{e:?}"))?
        .ok_or("no shape")?;
    let g = load(ttl);
    assert_eq!(g.instances_of(&format!("{ADS}RenderedLesson")).len(), 1);
    Ok(validate(&g, std::slice::from_ref(&shape)).results)
}

fn components(rs: &[ValidationResult]) -> Vec<String> {
    let mut c: Vec<String> = rs
        .iter()
        .map(|r| {
            format!(
                "{}:{}",
                r.component,
                r.path.as_deref().map_or("", |p| &p[ADS.len()..])
            )
        })
        .collect();
    c.sort();
    c
}

#[test]
fn m1_05_fails_on_global_motion_and_no_other_range() -> Result<(), String> {
    let c = components(&results(M1_05)?);
    assert!(
        c.contains(&"maxInclusive:globalMotionFraction".to_string()),
        "{c:?}"
    );
    let ranges: Vec<_> = c.iter().filter(|x| x.contains("Inclusive")).collect();
    assert_eq!(ranges.len(), 1, "motion is the only range violation: {c:?}");
    Ok(())
}

#[test]
fn m1_01_passes_every_range_and_fails_only_the_plot_rule() -> Result<(), String> {
    let c = components(&results(M1_01)?);
    assert_eq!(c, vec!["lessThanOrEquals:plotWritten".to_string()], "{c:?}");
    Ok(())
}

/// m1-01 with `plotWritten` / `plotShown` replaced by the given booleans.
fn with_plot(written: bool, shown: bool) -> String {
    M1_01
        .replace(
            "ads:plotWritten true",
            &format!("ads:plotWritten {written}"),
        )
        .replace(
            "ads:plotShown \"unknown\"",
            &format!("ads:plotShown {shown}"),
        )
}

#[test]
fn plot_rule_boolean_truth_table() -> Result<(), String> {
    let plot = |w: bool, s: bool| -> Result<Vec<String>, String> {
        let ttl = with_plot(w, s);
        assert!(ttl.contains(&format!("ads:plotWritten {w}")), "{ttl}");
        assert!(ttl.contains(&format!("ads:plotShown {s}")), "{ttl}");
        Ok(components(&results(&ttl)?))
    };
    assert!(plot(false, false)?.is_empty());
    assert!(plot(true, true)?.is_empty());
    assert!(plot(false, true)?.is_empty());
    assert_eq!(
        plot(true, false)?,
        vec!["lessThanOrEquals:plotWritten".to_string()]
    );
    Ok(())
}

#[test]
fn m1_05_unknown_plot_manifest_fires_the_plot_rule() -> Result<(), String> {
    let c = components(&results(M1_05)?);
    assert!(
        c.contains(&"lessThanOrEquals:plotWritten".to_string()),
        "{c:?}"
    );
    Ok(())
}
