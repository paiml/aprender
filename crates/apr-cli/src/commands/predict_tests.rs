//! Tests for the generic `apr predict` surface.
//!
//! What these can and cannot reach is a MEASURED boundary, recorded here so a
//! reader does not mistake its shape for a gap in the suite. These UNIT tests
//! cannot construct a `setfit-apr-v1` artifact that passes the load ladder: the
//! conformance slice's 97-row vocabulary closure cannot compute two of the six
//! contract-resident probes, and core's APR-capable view fixture is
//! `#[cfg(all(test, feature = "setfit"))] pub(crate)`, so it is unreachable from
//! this crate. Every rung up to the artifact door is therefore tested here by
//! behaviour; the rung past it is tested by the typed refusal that arrives, which
//! is what proves the routing engaged rather than merely that a message was
//! printed.
//!
//! **The boundary is a fixture boundary, not a capability one.** Before Phase 5's
//! 05-03 calibration edit (commit `a63bb130b`) no user-reachable path produced a
//! `setfit-apr-v1` (finding F-10). That is closed: `apr predict` is exercised
//! against a REAL trained artifact, with its full eight-rung load ladder and probe
//! replay, by 05-07's spawned production chain
//! (`crates/apr-cli/tests/setfit_cli_lifecycle.rs`,
//! `setfit_cli_production_chain_completes_after_the_calibration_edit`). That test
//! needs the 86.7 MB production checkout and trains an encoder, which is why the
//! positive rung lives there and not in this default-suite module.

use super::*;
use crate::setfit_tag::test_support::write_setfit_shaped_apr;
use tempfile::TempDir;

/// This command's own source, for the source assertions below.
const PREDICT_SOURCE: &str = include_str!("predict.rs");

/// Assemble a needle at RUNTIME so it cannot match the scan's own source.
fn needle(fragments: &[&str]) -> String {
    fragments.concat()
}

/// The four inputs review M2 is about, in one ordered set.
///
/// A newline is the one that a line-delimited `--input` format cannot carry at all;
/// a tab and a leading/trailing space are the ones a naive trimmer eats; the empty
/// string is the one a `filter(|s| !s.is_empty())` silently drops, which would
/// change the ORDER of everything after it; and the non-ASCII entries are the ones a
/// byte-oriented reader can split mid-character.
fn m2_texts() -> Vec<String> {
    vec![
        "plain ascii".to_string(),
        "line one\nline two".to_string(),
        "tabbed\tvalue".to_string(),
        String::new(),
        "el zorro café — naïve π".to_string(),
        "  padded  ".to_string(),
    ]
}

// ===========================================================================
// The request, before anything is read
// ===========================================================================

#[test]
fn predict_refuses_text_and_input_together_naming_both_flags() {
    let error = RequestSource::choose(&["a".to_string()], Some(Path::new("/tmp/doc.json")))
        .expect_err("two ordered input sets cannot both be classified");
    assert!(
        matches!(error, CliError::ValidationFailed(_)),
        "a conflicting invocation is a request error (exit 5); got: {error}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains("--text") && rendered.contains("--input"),
        "the refusal must name BOTH flags so the operator knows which to drop; got: {rendered}"
    );
}

#[test]
fn predict_refuses_an_invocation_with_no_texts_and_names_the_document_shape() {
    let error = RequestSource::choose(&[], None).expect_err("there is nothing to classify");
    let rendered = error.to_string();
    assert!(
        rendered.contains("texts") && rendered.contains("include_logits"),
        "the remedy must name the document's shape, not merely say 'invalid arguments'; got: \
         {rendered}"
    );
}

#[test]
fn predict_judges_the_request_before_it_opens_the_model_path() {
    // ORDERING. The model path here does not exist, so a run that opened it first
    // would report FileNotFound — sending the operator to look at a file that is not
    // the problem. `Fail on the REQUEST before blaming the data` (04-06's pattern).
    let error = run(
        Path::new("/nonexistent/definitely-not-here.apr"),
        &["a".to_string()],
        Some(Path::new("/nonexistent/doc.json")),
        false,
        true,
    )
    .expect_err("a conflicting invocation must be refused");
    assert!(
        matches!(error, CliError::ValidationFailed(_)),
        "the FLAG conflict must win over the missing file; got: {error}"
    );
}

// ===========================================================================
// Detection (D-04) at the command boundary
// ===========================================================================

/// WR-08 at the `apr predict` surface: the operator must not be told something false.
///
/// Measured at HEAD `81652bb50`, this exact file — a real APR v2 container carrying
/// `model_type = "setfit"` — made `apr predict` answer "not a SetFit classifier … run
/// `apr inspect <FILE>` to see what it actually is". It IS tagged. `predict` simply
/// could not read the block, and the fail-open turned "I could not tell" into a
/// confident denial that then routed the operator to a tool which rendered the full
/// APR-05 section for the same bytes.
///
/// BOTH directions are asserted in one test on purpose: an implementation that
/// merely appended the cap text to the old denial would satisfy the positive half
/// alone while leaving the false statement in place.
#[test]
fn predict_surfaces_the_cap_refusal_instead_of_denying_the_tag() {
    let temp = TempDir::new().expect("tempdir");
    let declared = crate::setfit_tag::MAX_TAG_METADATA_BYTES + 1;
    let path = crate::setfit_tag::test_support::write_over_cap_apr(
        temp.path(),
        "over-cap.apr",
        crate::setfit_tag::SETFIT_MODEL_TYPE,
        declared,
    );

    let error = run(&path, &["hello".to_string()], None, false, true)
        .expect_err("a metadata block that cannot be read cannot be classified");
    let rendered = error.to_string();
    assert!(
        rendered.contains("16 MiB"),
        "predict must surface the CAP that fired, so the operator learns what to fix; \
         got: {rendered}"
    );
    assert!(
        !rendered.contains("not a SetFit classifier"),
        "predict must NOT deny the tag it never read — that is the false statement \
         WR-08 names; got: {rendered}"
    );
}

#[test]
fn predict_refuses_a_setfit_shaped_but_untagged_apr() {
    let temp = TempDir::new().expect("tempdir");
    let path = write_setfit_shaped_apr(
        temp.path(),
        "untagged.apr",
        "",
        Some(r#"{"schema":"setfit-apr-v1","schema_version":1}"#),
    );

    let error = run(&path, &["hello".to_string()], None, false, true)
        .expect_err("an untagged APR is not a classifier, whatever its tensors are named");
    assert!(
        matches!(error, CliError::InvalidFormat(_)),
        "an untagged APR must be InvalidFormat (exit 4), never sniffed into the SetFit path; \
         got: {error}"
    );
    assert!(
        error.to_string().contains("setfit-apr-v1"),
        "the refusal must name what predict DOES support"
    );
}

#[test]
fn predict_refuses_a_non_apr_file() {
    let temp = TempDir::new().expect("tempdir");
    let path = temp.path().join("notes.txt");
    std::fs::write(&path, b"not a model").expect("fixture is writable");
    let error = run(&path, &["hello".to_string()], None, false, true)
        .expect_err("a text file is not a model");
    assert!(matches!(error, CliError::InvalidFormat(_)), "got: {error}");
}

#[test]
fn predict_reports_an_absent_model_path_as_file_not_found() {
    let temp = TempDir::new().expect("tempdir");
    let error = run(
        &temp.path().join("absent.apr"),
        &["hello".to_string()],
        None,
        false,
        true,
    )
    .expect_err("an absent model has nothing to classify with");
    assert!(matches!(error, CliError::FileNotFound(_)), "got: {error}");
}

// ===========================================================================
// The classifier path
// ===========================================================================

#[cfg(feature = "setfit")]
mod with_setfit {
    use super::*;
    use aprender::setfit::{ClassifyRequestDocument, MAX_REQUEST_BODY_BYTES};

    fn write_document(dir: &Path, name: &str, body: &str) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, body.as_bytes()).expect("fixture document is writable");
        path
    }

    #[test]
    fn predict_input_document_carries_multiline_tabbed_empty_and_non_ascii_texts_in_order() {
        // THE M2 WITNESS AT THE CLI BOUNDARY. The document is built from the same
        // core type the HTTP surface parses, so this asserts that the ordered input
        // set the model receives is byte-identical to what was written on disk —
        // including the newline that no line-delimited format could have carried.
        let temp = TempDir::new().expect("tempdir");
        let expected = m2_texts();
        let body = serde_json::to_string(&ClassifyRequestDocument {
            texts: expected.clone(),
            include_logits: false,
        })
        .expect("the request document serializes");
        let path = write_document(temp.path(), "req.json", &body);

        let document = setfit::build_request(&RequestSource::Document(path), false)
            .expect("a well-formed request document must parse");

        assert_eq!(
            document.texts, expected,
            "every text must arrive byte-identical AND in the order written; a line-delimited \
             reader would have split entry 1 into two and shifted every later index"
        );
        assert_eq!(
            document.texts.len(),
            6,
            "the empty string must survive as an ENTRY, not be filtered away"
        );
        assert!(
            !document.include_logits,
            "the document said false and no flag overrode it"
        );
    }

    #[test]
    fn predict_text_flags_build_the_same_document_in_order() {
        let expected = m2_texts();
        let document = setfit::build_request(&RequestSource::Texts(expected.clone()), true)
            .expect("--text values always build a document");
        assert_eq!(document.texts, expected, "--text order is response order");
        assert!(
            document.include_logits,
            "--logits sets include_logits on the --text path too"
        );
    }

    #[test]
    fn predict_logits_flag_can_only_turn_the_documents_request_on() {
        let temp = TempDir::new().expect("tempdir");
        let path = write_document(
            temp.path(),
            "logits.json",
            r#"{"texts":["a"],"include_logits":true}"#,
        );
        let document = setfit::build_request(&RequestSource::Document(path), false)
            .expect("the document parses");
        assert!(
            document.include_logits,
            "omitting --logits must not silently DOWNGRADE a document that asked for them"
        );
    }

    #[test]
    fn predict_refuses_an_unknown_key_in_the_request_document() {
        let temp = TempDir::new().expect("tempdir");
        let path = write_document(
            temp.path(),
            "unknown.json",
            r#"{"texts":["a"],"temperature":0.7}"#,
        );
        let error = setfit::build_request(&RequestSource::Document(path), false)
            .expect_err("deny_unknown_fields is what makes a typo a refusal");
        assert!(
            matches!(error, CliError::ValidationFailed(_)),
            "got: {error}"
        );
        assert!(
            error.to_string().contains("include_logits"),
            "the refusal must show the shape it wanted"
        );
    }

    #[test]
    fn predict_refuses_an_over_cap_request_document_before_parsing_it() {
        // T-04-50 at this surface, and the half F-14(4) recorded as still owed:
        // core's MAX_BATCH_TEXTS is checked AFTER the document is materialised, so a
        // body with ten million strings is fully allocated before it fires. The
        // stat'd length is what makes this refusal free.
        let temp = TempDir::new().expect("tempdir");
        let path = temp.path().join("huge.json");
        let file = std::fs::File::create(&path).expect("fixture is creatable");
        // Sparse: an attacker does not have to spend a megabyte to make a reader
        // spend one, and this costs no disk here either.
        file.set_len(MAX_REQUEST_BODY_BYTES + 1)
            .expect("a sparse over-cap file is creatable");
        drop(file);

        let error = setfit::build_request(&RequestSource::Document(path), false)
            .expect_err("a document past the contracted body bound must be refused");
        assert!(
            matches!(error, CliError::InvalidFormat(_)),
            "an over-cap body is a format refusal (exit 4); got: {error}"
        );
        let rendered = error.to_string();
        assert!(
            rendered.contains(&MAX_REQUEST_BODY_BYTES.to_string())
                && rendered.contains(&(MAX_REQUEST_BODY_BYTES + 1).to_string()),
            "the refusal must name BOTH the bound and the observed length; got: {rendered}"
        );
    }

    #[test]
    fn predict_reports_a_tagged_artifact_the_loader_refuses_as_model_load_failed() {
        // This is what proves the ROUTING engaged rather than that a message was
        // printed: the file is tagged, so detection sends it to `load_setfit_apr`,
        // and the error that comes back is the LOADER's — not "unsupported format".
        // The container is real; its tensors are not a model, so a rung refuses it.
        let temp = TempDir::new().expect("tempdir");
        let path = write_setfit_shaped_apr(
            temp.path(),
            "tagged-but-broken.apr",
            crate::setfit_tag::SETFIT_MODEL_TYPE,
            Some(r#"{"schema":"setfit-apr-v1","schema_version":1}"#),
        );

        let error = run(&path, &["hello".to_string()], None, false, true)
            .expect_err("a tagged file that is not a valid artifact must be refused");
        assert!(
            matches!(error, CliError::ModelLoadFailed(_)),
            "a tagged artifact the ladder rejects is ModelLoadFailed (exit 6), which is a \
             DIFFERENT answer from the untagged case's InvalidFormat — that difference is the \
             evidence that the tag routed; got: {error}"
        );
    }

    #[test]
    fn predict_reports_an_over_cap_artifact_before_allocating_it() {
        // The bounded artifact door (review B5), reached through this command.
        let temp = TempDir::new().expect("tempdir");
        let honest = write_setfit_shaped_apr(
            temp.path(),
            "honest.apr",
            crate::setfit_tag::SETFIT_MODEL_TYPE,
            None,
        );
        let bytes = std::fs::read(&honest).expect("the fixture is readable");
        let path = temp.path().join("over-cap.apr");
        let file = std::fs::File::create(&path).expect("fixture is creatable");
        drop(file);
        std::fs::write(&path, &bytes).expect("fixture is writable");
        let handle = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("fixture is openable");
        handle
            .set_len(aprender::setfit::MAX_ARTIFACT_BYTES + 1)
            .expect("a sparse over-cap artifact is creatable");
        drop(handle);

        let error = run(&path, &["hello".to_string()], None, false, true)
            .expect_err("an artifact past the contracted cap must be refused");
        assert!(matches!(error, CliError::InvalidFormat(_)), "got: {error}");
        assert!(
            error.to_string().contains("declared_length"),
            "the refusal must name the check that fired, so a reader can tell NO byte was read"
        );
    }
}

// ===========================================================================
// Source assertions
// ===========================================================================

#[test]
fn predict_declares_no_response_type() {
    // D-08: the CLI serializes core's envelope and defines nothing of its own. A
    // local response struct would be a second wire form for OPS-04's envelope, and
    // the day core gained a field the two surfaces would diverge with every test in
    // both crates still green.
    let struct_decl = needle(&["struct ", ""]);
    let offenders: Vec<&str> = PREDICT_SOURCE
        .lines()
        .filter(|line| line.trim_start().starts_with(&struct_decl))
        .filter(|line| line.contains("Response") || line.contains("Result {"))
        .collect();
    assert!(
        offenders.is_empty(),
        "predict.rs must declare no response type; found: {offenders:?}"
    );
}

#[test]
fn predict_reads_artifacts_only_through_the_one_bounded_door() {
    // Review B5, at this file. The needle is assembled at runtime so this assertion
    // does not fail on its own text.
    let banned = needle(&["fs::", "read("]);
    assert_eq!(
        PREDICT_SOURCE.matches(&banned).count(),
        0,
        "no unbounded whole-file read may appear in this command"
    );
    assert!(
        PREDICT_SOURCE.contains("read_setfit_apr_file_bounded"),
        "artifact bytes must come through the ONE bounded door"
    );
    assert!(
        PREDICT_SOURCE.contains("load_setfit_apr"),
        "prediction must go through the full load ladder, never a bundle-tier shortcut"
    );
}

#[test]
fn predict_input_is_a_document_and_never_a_line_iterator() {
    // Review M2, stated on the source: a `.lines()` over the input file is the exact
    // shape that cannot carry `m2_texts()[1]`, and the behavioural witness above
    // would not catch a SECOND reader added beside the document one.
    let iterator = needle(&[".", "lines()"]);
    assert_eq!(
        PREDICT_SOURCE.matches(&iterator).count(),
        0,
        "--input must parse as one ClassifyRequestDocument, not as one text per line"
    );
    assert!(
        PREDICT_SOURCE.contains("ClassifyRequestDocument"),
        "the shared request document is the input type"
    );
    assert!(
        PREDICT_SOURCE.contains("MAX_REQUEST_BODY_BYTES"),
        "the contracted body bound must be applied by this reading surface"
    );
}
