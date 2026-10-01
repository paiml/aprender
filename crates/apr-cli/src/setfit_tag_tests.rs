//! Tests for the typed-tag detector (D-04's negative, stated as tests).

use super::test_support::{write_over_cap_apr, write_setfit_shaped_apr};
use super::*;
use tempfile::TempDir;

/// This module's subject, for the source assertions below.
const SETFIT_TAG_SOURCE: &str = include_str!("setfit_tag.rs");

/// Assemble a needle at RUNTIME so it cannot match the scan's own source.
fn needle(fragments: &[&str]) -> String {
    fragments.concat()
}

#[test]
fn setfit_tag_recognizes_a_tagged_artifact_and_returns_its_document() {
    let temp = TempDir::new().expect("tempdir");
    let path = write_setfit_shaped_apr(
        temp.path(),
        "tagged.apr",
        SETFIT_MODEL_TYPE,
        Some(r#"{"schema":"setfit-apr-v1","schema_version":1}"#),
    );

    let tag = read_setfit_tag(&path)
        .expect("a well-formed container is readable")
        .expect("a container tagged `setfit` must be recognized");
    let doc = tag.doc.expect("the fixture carries the one custom key");
    assert_eq!(
        doc.get("schema").and_then(serde_json::Value::as_str),
        Some("setfit-apr-v1"),
        "the detector must hand back the custom key's value verbatim"
    );
}

#[test]
fn setfit_tag_refuses_to_sniff_a_setfit_shaped_but_untagged_apr() {
    // D-04's NEGATIVE. This file carries `setfit.head.weight` and `setfit.head.bias`
    // and the artifact document at the very key a SetFit artifact uses. The ONLY
    // difference from the positive above is `model_type`. A detector that looked at
    // tensor names — or at the presence of the custom key — would say yes here.
    let temp = TempDir::new().expect("tempdir");
    let path = write_setfit_shaped_apr(
        temp.path(),
        "untagged.apr",
        "",
        Some(r#"{"schema":"setfit-apr-v1","schema_version":1}"#),
    );

    assert!(
        read_setfit_tag(&path)
            .expect("a well-formed container is readable")
            .is_none(),
        "an APR whose model_type is empty is a PLAIN APR, whatever its tensors are named"
    );
}

#[test]
fn setfit_tag_treats_a_differently_tagged_apr_as_plain() {
    let temp = TempDir::new().expect("tempdir");
    let path = write_setfit_shaped_apr(temp.path(), "qwen.apr", "qwen2", None);
    assert!(
        read_setfit_tag(&path)
            .expect("a well-formed container is readable")
            .is_none(),
        "only the exact tag routes to the SetFit path"
    );
}

#[test]
fn setfit_tag_returns_none_for_a_non_apr_file() {
    let temp = TempDir::new().expect("tempdir");
    let path = temp.path().join("notes.txt");
    std::fs::write(&path, b"this is not a model at all").expect("fixture is writable");
    assert!(
        read_setfit_tag(&path)
            .expect("a readable file is not an error")
            .is_none(),
        "a file with the wrong magic is not a SetFit artifact"
    );
}

#[test]
fn setfit_tag_reports_an_absent_path_and_a_directory_typed() {
    let temp = TempDir::new().expect("tempdir");
    let absent = read_setfit_tag(&temp.path().join("nope.apr"))
        .expect_err("an absent path cannot be classified");
    assert!(
        matches!(absent, CliError::FileNotFound(_)),
        "an absent path must be FileNotFound (exit 3); got: {absent}"
    );
    let dir = read_setfit_tag(temp.path()).expect_err("a directory cannot be classified");
    assert!(
        matches!(dir, CliError::NotAFile(_)),
        "a directory must be NotAFile (exit 3); got: {dir}"
    );
}

#[test]
fn setfit_tag_refuses_a_metadata_block_longer_than_its_own_file() {
    // The T-04-50 half this module owns. `metadata_size` is a u32 read out of the
    // file under judgement, so a hostile container can declare up to 4 GiB and make
    // a naive reader allocate it. The refusal must land BEFORE the allocation, which
    // is why it is expressed against the stat'd length.
    //
    // The hostile file is produced by TRUNCATING an honest one rather than by
    // patching a byte at a guessed header offset: truncation is layout-independent,
    // so this test cannot silently stop testing anything the day a header field
    // moves. (A first draft did patch offset 28 — the detector accepted the file, so
    // the test was measuring nothing.)
    let temp = TempDir::new().expect("tempdir");
    let honest = write_setfit_shaped_apr(temp.path(), "honest.apr", SETFIT_MODEL_TYPE, None);
    let bytes = std::fs::read(&honest).expect("the fixture is readable");
    assert!(
        read_setfit_tag(&honest)
            .expect("the honest fixture is readable")
            .is_some(),
        "non-vacuity: the UNtruncated fixture must be recognized, or this test proves nothing \
         about truncation"
    );

    // Long enough to hold the header, far too short to hold the metadata block the
    // header declares.
    let truncated_len = HEADER_SIZE_V2 + 8;
    assert!(
        bytes.len() > truncated_len,
        "the honest fixture must be longer than the truncation point"
    );
    let path = temp.path().join("truncated.apr");
    std::fs::write(&path, &bytes[..truncated_len]).expect("fixture is writable");

    let error = read_setfit_tag(&path)
        .expect_err("a metadata block that runs past the end of the file is malformed");
    assert!(
        matches!(error, CliError::InvalidFormat(_)),
        "an impossible metadata length is a format refusal (exit 4); got: {error}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains(&truncated_len.to_string()),
        "the refusal must name the file length that bounds it; got: {rendered}"
    );
}

// ===========================================================================
// WR-08: an over-cap block is a TYPED REFUSAL, not a fail-open `Ok(None)`
// ===========================================================================

/// The over-cap arm is `Err`, and the message names the declared size and the cap.
///
/// Measured at HEAD `81652bb50`, this returned `Ok(None)`. That single value made
/// `apr predict` tell an operator a genuinely tagged classifier "is not a SetFit
/// classifier … run `apr inspect`", and `apr inspect` then rendered the full APR-05
/// section for the same bytes. The fail-open was the contradiction's source.
#[test]
fn setfit_tag_refuses_an_over_cap_metadata_block_by_name() {
    let temp = TempDir::new().expect("tempdir");
    let declared = MAX_TAG_METADATA_BYTES + 1;
    let path = write_over_cap_apr(
        temp.path(),
        "over-cap-tagged.apr",
        SETFIT_MODEL_TYPE,
        declared,
    );

    let error = read_setfit_tag(&path)
        .expect_err("a block over the identification cap must be a typed refusal");
    assert!(
        matches!(error, CliError::InvalidFormat(_)),
        "an over-cap block is a format refusal (exit 4); got: {error}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains(&declared.to_string()),
        "the refusal must name the DECLARED size so the operator can see what the file \
         asked for; got: {rendered}"
    );
    assert!(
        rendered.contains("16 MiB"),
        "the refusal must name the cap in the units an operator reads; got: {rendered}"
    );
}

/// The refusal is about the BLOCK, not about the tag.
///
/// `read_setfit_tag` cannot know the tag without reading the block, so a container
/// tagged `qwen2` earns exactly the same refusal. A message that mentioned SetFit
/// either way would be asserting something this code has no evidence for — which is
/// the precise defect WR-08 names, re-introduced one layer down.
#[test]
fn setfit_tag_refuses_an_over_cap_block_whatever_the_tag_says() {
    let temp = TempDir::new().expect("tempdir");
    let declared = MAX_TAG_METADATA_BYTES + 1;
    let path = write_over_cap_apr(temp.path(), "over-cap-qwen.apr", "qwen2", declared);

    let error = read_setfit_tag(&path).expect_err("the block is over the cap whatever it says");
    assert!(
        matches!(error, CliError::InvalidFormat(_)),
        "an over-cap block is a format refusal (exit 4); got: {error}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains("16 MiB"),
        "the same cap refusal must arrive for an untagged container; got: {rendered}"
    );
    assert!(
        !rendered.contains("SetFit") && !rendered.contains("setfit-apr-v1"),
        "the refusal must say NOTHING about whether this is a SetFit artifact — the \
         block was never read, so that fact is unknown here; got: {rendered}"
    );
}

/// NON-VACUITY. The new `Err` must not swallow the two ordinary dispositions.
///
/// An arm that refused everything would satisfy both tests above and break every
/// real invocation of `apr predict`, `apr serve` and `apr eval --task classify`.
#[test]
fn setfit_tag_keeps_both_in_bounds_dispositions_after_the_over_cap_error() {
    let temp = TempDir::new().expect("tempdir");
    let tagged = write_setfit_shaped_apr(
        temp.path(),
        "in-bounds-tagged.apr",
        SETFIT_MODEL_TYPE,
        Some(r#"{"schema":"setfit-apr-v1","schema_version":1}"#),
    );
    let plain = write_setfit_shaped_apr(temp.path(), "in-bounds-plain.apr", "qwen2", None);

    assert!(
        read_setfit_tag(&tagged)
            .expect("an in-bounds tagged artifact is still readable")
            .is_some(),
        "the tagged disposition must be unchanged"
    );
    assert!(
        read_setfit_tag(&plain)
            .expect("an in-bounds plain APR is still readable")
            .is_none(),
        "the plain-APR disposition must be unchanged"
    );
}

/// TWO malformations, TWO diagnoses. Collapsing them would lose the ORDER.
///
/// "Longer than its own file" and "fits, but too large to identify cheaply" are
/// different facts about a container, and the checks run in that order. A fixture
/// that tripped the first could never prove anything about the second, so the
/// over-cap fixture is deliberately extended past its own declared end.
#[test]
fn setfit_tag_keeps_the_two_metadata_malformations_distinguishable() {
    let temp = TempDir::new().expect("tempdir");

    // (a) the block runs PAST THE END of its own file.
    let honest = write_setfit_shaped_apr(temp.path(), "honest.apr", SETFIT_MODEL_TYPE, None);
    let bytes = std::fs::read(&honest).expect("the fixture is readable");
    let truncated_len = HEADER_SIZE_V2 + 8;
    assert!(
        bytes.len() > truncated_len,
        "the honest fixture must be longer than the truncation point"
    );
    let past_eof_path = temp.path().join("past-eof.apr");
    std::fs::write(&past_eof_path, &bytes[..truncated_len]).expect("fixture is writable");
    let past_eof = read_setfit_tag(&past_eof_path)
        .expect_err("a block past its own EOF is malformed")
        .to_string();

    // (b) the block FITS the file but is over the identification cap.
    let declared = MAX_TAG_METADATA_BYTES + 1;
    let over_cap_path =
        write_over_cap_apr(temp.path(), "over-cap.apr", SETFIT_MODEL_TYPE, declared);
    let over_cap = read_setfit_tag(&over_cap_path)
        .expect_err("a block over the cap is refused")
        .to_string();

    assert_ne!(
        past_eof, over_cap,
        "two different malformations must not collapse into one diagnosis"
    );
    assert!(
        past_eof.contains(&truncated_len.to_string()),
        "the past-EOF refusal names the FILE LENGTH that bounds it; got: {past_eof}"
    );
    assert!(
        !past_eof.contains("16 MiB"),
        "the past-EOF refusal must NOT name the cap — the cap is not what fired, and \
         naming it would send the operator after the wrong fact; got: {past_eof}"
    );
    assert!(
        over_cap.contains("16 MiB"),
        "the over-cap refusal names the CAP; got: {over_cap}"
    );
}

/// WR-08 stated as ONE table over ONE file: no two tools may contradict each other.
///
/// Measured at HEAD `81652bb50`, before this plan, for this exact container — a real
/// APR v2 file carrying `model_type = "setfit"` whose metadata block is in-bounds for
/// the file but over the identification cap:
///
/// * `apr predict` said "not a SetFit classifier … run `apr inspect <FILE>` to see
///   what it actually is" — a denial of a tag it had not read, plus a pointer;
/// * `apr inspect`, followed to, rendered the FULL APR-05 section for the same bytes;
/// * `apr eval --task classify` refused its SetFit-only flags with "does not carry
///   the SetFit tag", which is false of a file that carries it.
///
/// CLAUDE.md rule 5: a guard that does not scan the surface where the DECISION is made
/// is theater. The decision surfaces here are exactly four — the shared detector,
/// `inspect`'s own reader, `predict`, and the `apr eval --task classify` router — so
/// all four are driven, and two `pub(crate)` seams exist for no other purpose than to
/// make the last two reachable. That this function COMPILES is the reachability proof;
/// the assertions below are the agreement proof.
///
/// `apr serve` is asserted too, and asserted to be DIFFERENT: its fall-through is
/// deliberate and documented, and it is driven through the very expression
/// `serve/handlers.rs:774` uses so that an edit there turns this red.
#[cfg(feature = "training")]
#[test]
fn four_consumers_agree_about_one_over_cap_file() {
    /// A statement no surface may make about this file. It carries the tag; the only
    /// honest position is "I could not read the block", never "it is not one".
    const FALSE_DENIALS: [&str; 3] = [
        "not a SetFit classifier",
        "does not carry the SetFit tag",
        "is not a SetFit",
    ];

    let temp = TempDir::new().expect("tempdir");
    let declared = MAX_TAG_METADATA_BYTES + 1;
    let path = write_over_cap_apr(temp.path(), "over-cap.apr", SETFIT_MODEL_TYPE, declared);

    // ---- surface 1: the SHARED DETECTOR -----------------------------------
    let detector = read_setfit_tag(&path)
        .expect_err("the shared detector refuses a block it cannot cheaply read");
    let detector_msg = detector.to_string();

    // ---- surface 2: `apr inspect`'s OWN reader ----------------------------
    // Not `read_setfit_tag` again: `inspect` has a SEPARATE reader, and WR-08 is
    // exactly the case where the two disagreed about one file.
    let mut reader =
        std::io::BufReader::new(std::fs::File::open(&path).expect("the fixture is readable"));
    let header = crate::commands::inspect::read_and_parse_header(&mut reader)
        .expect("the fixture is a real APR v2 container");
    let info = crate::commands::inspect::read_metadata(&mut reader, &header);

    // ---- surface 3: `apr predict` -----------------------------------------
    let predict = crate::commands::predict::run(&path, &["hello".to_string()], None, false, true)
        .expect_err("predict cannot classify a block it could not read");
    let predict_msg = predict.to_string();

    // ---- surface 4: `apr eval --task classify` ----------------------------
    let eval = crate::dispatch_classify_eval(
        &path,
        "sst2",
        None,
        None,
        2,
        false,
        true,
        None,
        "validation",
        None,
        None,
        &[],
        false,
    )
    .expect_err("the classify router cannot route a block it could not read");
    let eval_msg = eval.to_string();

    // ---- the CONJUNCTION WR-08 names --------------------------------------
    for (surface, message) in [
        ("read_setfit_tag", &detector_msg),
        ("predict", &predict_msg),
        ("dispatch_classify_eval", &eval_msg),
    ] {
        assert!(
            message.contains("16 MiB"),
            "{surface} must name the cap that fired, so all surfaces cite ONE cause; \
             got: {message}"
        );
        for denial in FALSE_DENIALS {
            assert!(
                !message.contains(denial),
                "{surface} must not state or imply the file is not a SetFit artifact — it \
                 carries the tag, and the block was never read; found {denial:?} in: {message}"
            );
        }
    }

    // `apr inspect` must not render a populated APR-05 section for these bytes, and
    // must disclose WHY rather than defaulting to a silent "no model type".
    assert_eq!(
        info.metadata_over_cap_bytes,
        Some(declared),
        "inspect must DISCLOSE the refusal, not render the file as an ordinary APR"
    );
    assert_eq!(
        info.model_type, None,
        "inspect must derive no model type from a block it never read"
    );
    assert!(
        info.setfit_doc.is_none(),
        "inspect must render NO APR-05 section for a file whose document it never read — \
         rendering one while predict denied the tag is WR-08 itself"
    );

    // ---- `apr serve`: DIFFERENT, and deliberately so ----------------------
    let serve_routes_to_setfit = read_setfit_tag(&path).ok().flatten().is_some();
    assert!(
        !serve_routes_to_setfit,
        "serve keeps its documented fall-through to the plain-APR path via \
         `.ok().flatten()`; that is a considered disposition, not a contradiction"
    );
}

/// The CODE LINES of the production half of `setfit_tag.rs`.
///
/// Two filters, and each was earned by a real red:
///
/// * the file is truncated at the test-support boundary, because the fixture builder
///   there deliberately writes tensors NAMED `setfit.head.weight` — a whole-file scan
///   turns red on the very fixture that proves the detector ignores tensor names;
/// * comment lines are dropped, because the module header EXPLAINS the rule using the
///   forbidden names, and a guard that fails on its own documentation is the F-05
///   defect. The explanation is worth more than the names' absence from prose.
///
/// Both filters carry a non-vacuity assertion, because a filter that ate the module
/// would make every claim below trivially true.
fn production_code_lines() -> String {
    let boundary = needle(&["#[cfg(test)]\n", "pub(crate) mod test_support"]);
    let end = SETFIT_TAG_SOURCE
        .find(&boundary)
        .expect("the test-support boundary marker must exist; if it moved, fix this scan");
    let half = &SETFIT_TAG_SOURCE[..end];
    assert!(
        half.contains("pub(crate) fn read_setfit_tag"),
        "non-vacuity: the truncation must leave the function under scan in place"
    );
    let code: String = half
        .lines()
        .filter(|line| !line.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        code.contains("AprV2Metadata::from_json"),
        "non-vacuity: the comment filter must not have eaten the module's code"
    );
    code
}

#[test]
fn setfit_tag_detection_never_consults_a_tensor_name() {
    // The source half of D-04. The behavioural half is
    // `setfit_tag_refuses_to_sniff_a_setfit_shaped_but_untagged_apr`; this one
    // catches a future edit that adds a name check the fixtures happen not to cover.
    let production = production_code_lines();
    let reader = needle(&["AprV2", "Reader"]);
    assert_eq!(
        production.matches(&reader).count(),
        0,
        "detection must not open the tensor index: the tag is in the metadata record"
    );
    for banned in ["tensor_names", "head.weight", "tokenizer.blob"] {
        assert!(
            !production.contains(banned),
            "detection must not name a tensor ({banned}) — D-04 is tag-only"
        );
    }
}

/// The local tag literal and core's `MODEL_TYPE_TAG` must agree.
///
/// `setfit_tag` is deliberately UNGATED while `aprender::setfit` is `#[cfg(feature = "setfit")]`,
/// so this module cannot name core's constant in non-test code and must carry a literal. Nothing
/// bound the two until now: `serve/handlers.rs` compared against core's constant while this module
/// compared against the literal, and a divergence would have silently split detection between
/// `apr serve` and `apr predict`. The gate is one line, so the literal cannot drift unnoticed.
#[cfg(feature = "setfit")]
#[test]
fn the_local_tag_literal_equals_cores_model_type_tag() {
    assert_eq!(
        super::SETFIT_MODEL_TYPE,
        aprender::setfit::artifact::MODEL_TYPE_TAG,
        "apr-cli's ungated tag literal must track the value core's writer stamps"
    );
}
