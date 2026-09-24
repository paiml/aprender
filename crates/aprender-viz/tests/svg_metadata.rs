//! APEX-001 EV-2e: `SvgEncoder::metadata` emits one deterministic `<metadata>` block
//! (paiml/aprender#4067).
//!
//! - **RED (a)**: exactly one `<metadata>` block, the first child of `<svg>` (before the
//!   background), and its bytes are the caller's payload plus the fixed wrapping and nothing else.
//! - **RED (b)**: two encoders built independently from the same inputs render byte-identical SVG
//!   with the block present. Built independently, not rendered twice from one encoder, so a
//!   timestamp taken at build time is caught as surely as one taken at render time.
//! - **RED (c)**: a payload that is not well-formed XML is refused, never emitted, including one
//!   that closes the wrapper early to break out into the drawing.

use trueno_viz::color::Rgba;
use trueno_viz::error::Error;
use trueno_viz::output::SvgEncoder;

const PAYLOAD: &str = "<provenance><contract-iri>urn:sha256:0f</contract-iri>\
<evidence-sha256>ab</evidence-sha256><verdict>DISCHARGED</verdict></provenance>";

fn figure() -> SvgEncoder {
    SvgEncoder::new(64, 48).rect(4.0, 4.0, 20.0, 10.0, Rgba::rgb(200, 30, 30)).circle(
        40.0,
        24.0,
        6.0,
        Rgba::rgb(30, 30, 200),
    )
}

fn with_metadata(xml: &str) -> String {
    figure().metadata(xml).expect("a well-formed payload is accepted").render()
}

#[test]
fn exactly_one_metadata_block_placed_first() {
    let svg = with_metadata(PAYLOAD);
    assert_eq!(svg.matches("<metadata>").count(), 1, "{svg}");
    assert_eq!(svg.matches("</metadata>").count(), 1, "{svg}");

    let mut lines = svg.lines();
    assert!(lines.next().is_some_and(|l| l.starts_with("<svg ")), "{svg}");
    // The whole second line is the block: payload plus fixed wrapping, nothing else in it.
    assert_eq!(lines.next(), Some(format!("  <metadata>{PAYLOAD}</metadata>").as_str()), "{svg}");
    assert!(lines.next().is_some_and(|l| l.contains(r#"<rect width="100%""#)), "{svg}");
}

#[test]
fn a_second_call_replaces_the_block_rather_than_adding_one() {
    let svg =
        figure().metadata("<a/>").and_then(|e| e.metadata(PAYLOAD)).expect("accepted").render();
    assert_eq!(svg.matches("<metadata>").count(), 1, "{svg}");
    assert!(svg.contains(PAYLOAD) && !svg.contains("<a/>"), "{svg}");
}

#[test]
fn two_independent_builds_render_byte_identical_with_the_block_present() {
    let a = with_metadata(PAYLOAD);
    std::thread::sleep(std::time::Duration::from_millis(5));
    let b = with_metadata(PAYLOAD);
    assert!(a.contains("<metadata>"), "the block must be present for (b) to mean anything");
    assert_eq!(a.as_bytes(), b.as_bytes());
}

#[test]
fn without_metadata_the_document_is_unchanged() {
    let svg = figure().render();
    assert!(!svg.contains("<metadata"), "{svg}");
    assert!(svg.lines().nth(1).is_some_and(|l| l.contains(r#"<rect width="100%""#)), "{svg}");
}

#[test]
fn a_malformed_payload_is_refused_not_emitted() {
    for bad in [
        "<provenance>",                                   // unclosed
        "<a></b>",                                        // mismatched
        "a & b",                                          // bare ampersand
        "a < b",                                          // bare less-than
        "</metadata><script>alert(1)</script><metadata>", // breaks out of the wrapper
        "<!DOCTYPE x [<!ENTITY e \"boom\">]><x>&e;</x>",  // DTD / entity expansion
        "<?xml version=\"1.0\"?><x/>",                    // a declaration mid-document
    ] {
        match figure().metadata(bad) {
            Err(Error::SvgMetadataRefused { .. }) => {}
            Err(e) => panic!("{bad:?}: refused with the wrong error: {e}"),
            Ok(enc) => panic!("{bad:?}: accepted and would emit:\n{}", enc.render()),
        }
    }
}

#[test]
fn well_formed_payloads_of_other_shapes_are_accepted_verbatim() {
    for good in [
        "",
        "plain text &amp; an escaped entity",
        "<rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description/></rdf:RDF>",
        "<a/><b>two roots are fine inside the wrapper</b><!-- a comment -->",
    ] {
        let svg = with_metadata(good);
        assert!(svg.contains(&format!("  <metadata>{good}</metadata>\n")), "{good:?}:\n{svg}");
    }
}
