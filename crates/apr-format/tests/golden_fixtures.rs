#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::float_cmp
)]
//! Golden-fixture regression tests for the sovereign `apr-format` leaf (#2231).
//!
//! `golden_v1.apr` / `golden_v2.apr` were captured by the temporary harness in
//! `aprender-core` (`format/golden_capture_tmp.rs`) while the format code still
//! lived in core — i.e. they are the byte-identity oracle produced by the
//! pre-extraction code. These tests prove the EXTRACTED leaf reads the SAME bytes.
//!
//! Stage 2 scope: the v1 (`APRN`) container AND the v2 (`APR\0`) container both
//! live in the leaf now, so the leaf reads `golden_v1.apr` AND `golden_v2.apr`
//! and round-trips their F32 tensors against the captured oracle bytes.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct GoldenModel {
    name: String,
    weights: Vec<f32>,
    bias: f32,
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

#[test]
fn golden_v1_loads_in_leaf() {
    let path = fixtures().join("golden_v1.apr");
    let model: GoldenModel =
        apr_format::load(&path, apr_format::ModelType::LinearRegression).expect("load golden v1");
    assert_eq!(model.name, "golden_v1");
    assert_eq!(
        model.weights,
        vec![1.0, 2.0, 0.5, -0.5, 4.0, -2.0, 0.25, 8.0]
    );
    assert_eq!(model.bias, 0.125);
}

#[test]
fn golden_v1_crc_and_header_are_consistent() {
    // The leaf's crc32 must validate the core-produced trailer (CRC-integrity).
    let bytes = std::fs::read(fixtures().join("golden_v1.apr")).expect("read v1");
    assert!(bytes.len() > apr_format::HEADER_SIZE + 4);
    let stored = u32::from_le_bytes([
        bytes[bytes.len() - 4],
        bytes[bytes.len() - 3],
        bytes[bytes.len() - 2],
        bytes[bytes.len() - 1],
    ]);
    let computed = apr_format::crc32(&bytes[..bytes.len() - 4]);
    assert_eq!(
        stored, computed,
        "leaf crc32 must match the core-written trailer"
    );

    let header = apr_format::Header::from_bytes(&bytes[..apr_format::HEADER_SIZE]).expect("hdr");
    assert_eq!(header.magic, apr_format::MAGIC);
    assert_eq!(header.quality_score, 85);
}

#[test]
fn golden_v2_loads_in_leaf() {
    // Stage 2: the leaf's v2 reader parses the pre-extraction golden_v2.apr bytes
    // and round-trips its F32 tensors against the captured oracle values.
    let bytes = std::fs::read(fixtures().join("golden_v2.apr")).expect("read v2");
    assert_eq!(&bytes[0..4], &[0x41, 0x50, 0x52, 0x00], "APR\\0 magic");
    assert_eq!(bytes.len(), 1092, "v2 golden size pinned");

    let reader = apr_format::v2::AprV2Reader::from_bytes(&bytes).expect("leaf parses golden v2");
    assert!(reader.header().verify_checksum(), "v2 header CRC valid");
    assert_eq!(reader.metadata().model_type, "linear_regression");
    assert_eq!(reader.metadata().name.as_deref(), Some("golden-v2"));

    let mut names = reader.tensor_names();
    names.sort_unstable();
    assert_eq!(names, vec!["bias", "weights"]);

    // F32 tensors read back exactly (no dequant needed — leaf's get_f32_tensor).
    assert_eq!(reader.get_f32_tensor("bias").expect("bias"), vec![0.125]);
    assert_eq!(
        reader.get_f32_tensor("weights").expect("weights"),
        vec![1.0, 2.0, 0.5, -0.5, 4.0, -2.0, 0.25, 8.0]
    );
}

#[test]
fn golden_v2_f32_writer_is_byte_identical() {
    // BYTE IDENTITY IS AGAINST golden_v2_current.apr, NOT golden_v2.apr, AND THE
    // SPLIT IS THE POINT (#3235).
    //
    // golden_v2.apr is the PRE-EXTRACTION oracle: bytes captured by the format
    // code while it still lived in aprender-core. `golden_v2_loads_in_leaf`
    // above is what it is for -- the leaf must still READ artifacts written
    // before the extraction, and it does.
    //
    // This row asked something different: that today's writer REPRODUCES those
    // bytes. That stopped being true, deliberately, in #2254
    // (`fix(finetune): apr finetune --merge produces a directly-runnable .apr`),
    // which added `#[serde(default, skip_serializing_if = "Option::is_none")]`
    // to AprV2Metadata. The oracle predates it (#2236) and spells 26 optional
    // fields as explicit `null`; today's writer omits them. 1092 bytes vs 516.
    //
    // Regenerating golden_v2.apr would have been the easy fix and the wrong one:
    // it would delete the only artifact in the tree written by the pre-extraction
    // code, and `golden_v2_loads_in_leaf` would then prove nothing but that the
    // leaf reads its own output. So the compatibility oracle STAYS at 1092 bytes,
    // and byte-identity gets a fixture of its own.
    //
    // `the_two_v2_goldens_carry_the_same_model` below is what keeps this honest:
    // it proves the new fixture is the SAME MODEL as the oracle, not a different
    // artifact quietly substituted for one that would not match.
    let golden = std::fs::read(fixtures().join("golden_v2_current.apr")).expect("read v2 current");

    let produced = write_golden_v2_model();

    if produced != golden {
        // SAY WHICH KIND OF FAILURE THIS IS. A byte diff has two very different
        // causes and they have opposite fixes: the model changed (a defect), or
        // only its serialization did (regenerate the fixture, as #2254 should
        // have). Reporting "bytes differ" makes the reader re-derive that every
        // time, which is how this one sat unresolved.
        let p = apr_format::v2::AprV2Reader::from_bytes(&produced);
        let g = apr_format::v2::AprV2Reader::from_bytes(&golden);
        let same_model = match (&p, &g) {
            (Ok(p), Ok(g)) => same_v2_model(p, g),
            _ => false,
        };
        assert!(
            !same_model,
            "v2 F32 SERIALIZATION drift: produced {} bytes vs golden {}, but both \
             carry the same model_type, metadata values, tensor names, shapes and \
             F32 values. The format's spelling changed, not its content -- \
             regenerate tests/fixtures/golden_v2_current.apr in the same commit as \
             the change that moved it. (golden_v2.apr, the pre-extraction \
             compatibility oracle, must NOT be regenerated.)",
            produced.len(),
            golden.len()
        );
        panic!(
            "v2 F32 CONTENT drift: produced {} bytes vs golden {}, and the two do \
             NOT carry the same model. This is a writer defect, not a spelling \
             change.",
            produced.len(),
            golden.len()
        );
    }

    // Length is asserted separately so a size-only regression names itself before
    // the byte compare's diff has to be read.
    assert_eq!(produced.len(), golden.len(), "v2 F32 byte length");
}

/// THE ROW THAT MAKES THE TWO-FIXTURE SPLIT HONEST (#3235).
///
/// A second golden is a licence to smuggle: regenerate it from whatever the code
/// happens to emit and the byte-identity row above passes forever. This proves
/// `golden_v2_current.apr` is the SAME MODEL as the pre-extraction oracle
/// `golden_v2.apr` -- same metadata values, same tensors, same F32 payload --
/// so the only thing that changed between 1092 and 516 bytes is how the optional
/// metadata fields are spelled.
#[test]
fn the_two_v2_goldens_carry_the_same_model() {
    let oracle = std::fs::read(fixtures().join("golden_v2.apr")).expect("read v2 oracle");
    let current = std::fs::read(fixtures().join("golden_v2_current.apr")).expect("read v2 current");
    assert_eq!(oracle.len(), 1092, "the pre-extraction oracle is pinned");
    assert_ne!(
        oracle.len(),
        current.len(),
        "if these ever become the same size the split has no reason to exist"
    );

    let o = apr_format::v2::AprV2Reader::from_bytes(&oracle).expect("oracle parses");
    let c = apr_format::v2::AprV2Reader::from_bytes(&current).expect("current parses");
    assert!(
        same_v2_model(&o, &c),
        "the two v2 goldens do not carry the same model -- the 516-byte fixture is \
         not a re-spelling of the 1092-byte oracle and must not be used as one"
    );
}

/// `same_v2_model` IS THE THING THE SPLIT RESTS ON, so it gets its own rows
/// rather than only whatever the two committed fixtures happen to exercise.
///
/// The committed pair differs only by fields the oracle spells as `null`, which
/// exercises exactly ONE of the two symmetric loops. These rows drive both, and
/// a real difference in each direction.
#[test]
fn same_v2_model_discriminates_a_real_difference() {
    use apr_format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer};

    let build = |name: &str, w: f32, arch: Option<&str>| -> Vec<u8> {
        let mut m = AprV2Metadata::new("linear_regression");
        m.name = Some(name.to_string());
        m.version = Some("0.0.0-golden".to_string());
        m.created_at = Some("1700000000".to_string());
        m.param_count = 8;
        m.architecture = arch.map(str::to_string);
        let mut wr = AprV2Writer::new(m);
        wr.add_f32_tensor(
            "weights",
            vec![8],
            &[1.0, 2.0, 0.5, -0.5, 4.0, -2.0, 0.25, w],
        );
        wr.add_f32_tensor("bias", vec![1], &[0.125]);
        wr.write().expect("v2 write")
    };

    let base = build("golden-v2", 8.0, None);
    let same = build("golden-v2", 8.0, None);
    let diff_tensor = build("golden-v2", 8.5, None);
    let diff_meta = build("golden-v2-other", 8.0, None);
    let extra_field = build("golden-v2", 8.0, Some("llama"));

    let r = |b: &[u8]| AprV2Reader::from_bytes(b).expect("parses");

    assert!(
        same_v2_model(&r(&base), &r(&same)),
        "control: two writes of the same model must compare equal"
    );
    assert!(
        !same_v2_model(&r(&base), &r(&diff_tensor)),
        "a different F32 value is a different model"
    );
    assert!(
        !same_v2_model(&r(&base), &r(&diff_meta)),
        "a different metadata VALUE is a different model"
    );
    // This is the row the committed fixtures cannot reach: one side carries a
    // NON-NULL field the other omits entirely. `skip_serializing_if` makes those
    // two indistinguishable at the byte level unless the comparison looks both
    // ways -- deleting either loop in same_v2_model turns exactly this red.
    assert!(
        !same_v2_model(&r(&base), &r(&extra_field)),
        "a non-null field present on only one side is a different model"
    );
    assert!(
        !same_v2_model(&r(&extra_field), &r(&base)),
        "...and it must be caught from the other direction too"
    );
}

/// Write the model both v2 goldens encode. One definition, so the byte-identity
/// row and any future regeneration cannot drift apart.
fn write_golden_v2_model() -> Vec<u8> {
    use apr_format::v2::{AprV2Metadata, AprV2Writer};
    let mut metadata = AprV2Metadata::new("linear_regression");
    metadata.name = Some("golden-v2".to_string());
    metadata.version = Some("0.0.0-golden".to_string());
    metadata.created_at = Some("1700000000".to_string());
    metadata.param_count = 8;

    let mut writer = AprV2Writer::new(metadata);
    // Index is sorted by name on write; insertion order does not matter.
    writer.add_f32_tensor(
        "weights",
        vec![8],
        &[1.0, 2.0, 0.5, -0.5, 4.0, -2.0, 0.25, 8.0],
    );
    writer.add_f32_tensor("bias", vec![1], &[0.125]);
    writer.write().expect("v2 write")
}

/// Same model, ignoring how the metadata is spelled: every field the two agree
/// on must be equal, and neither may carry a NON-NULL field the other lacks.
fn same_v2_model(a: &apr_format::v2::AprV2Reader, b: &apr_format::v2::AprV2Reader) -> bool {
    metadata_agrees(a, b) && tensors_agree(a, b)
}

/// One direction of the metadata comparison. A field present in `one` and absent
/// from `other` is only forgivable when it is null -- that is exactly what
/// `skip_serializing_if = "Option::is_none"` drops, and it is the ONLY difference
/// the two committed goldens are allowed to have.
fn fields_agree(
    one: &serde_json::Map<String, serde_json::Value>,
    other: &serde_json::Map<String, serde_json::Value>,
) -> bool {
    one.iter().all(|(k, v)| match other.get(k) {
        Some(w) => w == v,
        None => v.is_null(),
    })
}

fn metadata_agrees(a: &apr_format::v2::AprV2Reader, b: &apr_format::v2::AprV2Reader) -> bool {
    let ja = serde_json::to_value(a.metadata()).expect("metadata to json");
    let jb = serde_json::to_value(b.metadata()).expect("metadata to json");
    let (Some(oa), Some(ob)) = (ja.as_object(), jb.as_object()) else {
        return false;
    };
    // BOTH directions. One loop cannot see a non-null field that only the OTHER
    // side carries, and `same_v2_model_discriminates_a_real_difference` drives
    // exactly that case in each direction.
    fields_agree(oa, ob) && fields_agree(ob, oa)
}

fn tensors_agree(a: &apr_format::v2::AprV2Reader, b: &apr_format::v2::AprV2Reader) -> bool {
    let mut na = a.tensor_names();
    let mut nb = b.tensor_names();
    na.sort_unstable();
    nb.sort_unstable();
    na == nb && na.iter().all(|n| one_tensor_agrees(a, b, n))
}

fn one_tensor_agrees(
    a: &apr_format::v2::AprV2Reader,
    b: &apr_format::v2::AprV2Reader,
    name: &str,
) -> bool {
    if a.get_f32_tensor(name) != b.get_f32_tensor(name) {
        return false;
    }
    match (a.get_tensor(name), b.get_tensor(name)) {
        (Some(x), Some(y)) => x.shape == y.shape,
        _ => false,
    }
}
