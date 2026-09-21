//! #3601: `get_tensor_raw` sizes every live ggml type — and the two commands that read
//! those sizes (`apr inspect`, `apr tensors`) accept an IQ/TQ GGUF instead of calling it
//! corrupt. And #3656: sized is not readable — `get_tensor_f32` refuses what it has no
//! dequantizer for instead of approximating it.
//!
//! The sizes below are typed from ggml-common.h's block structs on purpose, NOT read
//! back from `trueno_quant::TRAITS`: a test that asked the table what the table says
//! could not fail. `TRAITS` has its own upstream-fixture test (PMAT-3430); these pin
//! what this reader does with it.

use super::builder::build_synthetic_gguf_with_tensor;
use super::*;

/// (ggml id, name, elements per block, bytes per block) — every IQ*/TQ* type, which is
/// every id the pre-#3601 match refused although ggml defines a layout for it.
const IQ_AND_TQ: [(u32, &str, usize, usize); 11] = [
    (16, "IQ2_XXS", 256, 66), // d + qs[QK_K/8] u16
    (17, "IQ2_XS", 256, 74),  // d + qs[QK_K/8] u16 + scales[QK_K/32]
    (18, "IQ3_XXS", 256, 98), // d + qs[3*QK_K/8]
    (19, "IQ1_S", 256, 50),   // d + qs[QK_K/8] + qh[QK_K/32] u16
    (20, "IQ4_NL", 32, 18),   // d + qs[QK4_NL/2]
    (21, "IQ3_S", 256, 110),  // d + qs[QK_K/4] + qh[QK_K/32] + signs[QK_K/8] + scales[4]
    (22, "IQ2_S", 256, 82),   // d + qs[QK_K/4] + qh[QK_K/32] + scales[QK_K/32]
    (23, "IQ4_XS", 256, 136), // d + scales_h u16 + scales_l[QK_K/64] + qs[QK_K/2]
    (29, "IQ1_M", 256, 56),   // qs[QK_K/8] + qh[QK_K/16] + scales[QK_K/32]
    (34, "TQ1_0", 256, 54),   // qs[(QK_K - 4*QK_K/64)/5] + qh[QK_K/64] + d
    (35, "TQ2_0", 256, 66),   // qs[QK_K/4] + d
];

/// A reader over one tensor of `dtype` with `dims`, whose data section holds `payload`
/// followed by `trailing` bytes of 0xEE — so a reader that sizes too large reads 0xEE,
/// and one that sizes too small returns fewer bytes than the payload.
fn one_tensor(dims: &[u64], dtype: u32, payload: &[u8], trailing: usize) -> GgufReader {
    let mut data = payload.to_vec();
    data.extend(std::iter::repeat_n(0xEE_u8, trailing));
    let bytes = build_synthetic_gguf_with_tensor("t.weight", dims, dtype, &data, &[]);
    GgufReader::from_bytes(bytes).expect("synthetic GGUF parses")
}

/// A payload whose bytes are all distinct from the 0xEE tail.
fn payload(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 200) as u8).collect()
}

/// done_when 2: every IQ/TQ id is sized, at exactly two blocks' worth of bytes.
#[test]
fn test_3601_get_tensor_raw_sizes_every_iq_and_tq_type() {
    for (id, name, blck, type_size) in IQ_AND_TQ {
        let want = payload(2 * type_size);
        let reader = one_tensor(&[2 * blck as u64], id, &want, 64);
        let (bytes, shape, dtype) = reader
            .get_tensor_raw("t.weight")
            .unwrap_or_else(|e| panic!("{name} (id {id}) must be sized, got: {e}"));
        assert_eq!(dtype, id, "{name}: dtype id must round-trip");
        assert_eq!(shape, vec![2 * blck], "{name}: shape");
        assert_eq!(bytes, want, "{name}: exactly two blocks, no tail byte");
    }
}

/// A 2-D tensor is sized from its full element count: 3 rows of one block each.
#[test]
fn test_3601_two_d_iq4_xs_tensor_is_sized_over_all_rows() {
    let want = payload(3 * 136);
    let reader = one_tensor(&[256, 3], 23, &want, 64);
    let (bytes, _, _) = reader.get_tensor_raw("t.weight").expect("IQ4_XS sized");
    assert_eq!(bytes, want);
}

/// Behaviour preservation: the 15 ids the hand-typed match sized keep their byte count
/// for every element count, including counts that are not a whole number of blocks
/// (the arms rounded DOWN, and so does the table-driven path).
#[test]
fn test_3601_ids_sized_before_keep_their_byte_counts() {
    // The pre-#3601 arms, transcribed: (id, divisor, multiplier).
    const PRE_3601: [(u32, usize, usize); 15] = [
        (0, 1, 4),
        (1, 1, 2),
        (2, 32, 18),
        (3, 32, 20),
        (6, 32, 22),
        (7, 32, 24),
        (8, 32, 34),
        (9, 32, 36),
        (10, 256, 84),
        (11, 256, 110),
        (12, 256, 144),
        (13, 256, 176),
        (14, 256, 210),
        (15, 256, 292),
        (30, 1, 2),
    ];
    for (id, div, mul) in PRE_3601 {
        for n in [1, div.saturating_sub(1).max(1), div, div + 1, 2 * div, 1000] {
            let expected = (n / div) * mul;
            let reader = one_tensor(&[n as u64], id, &payload(expected), 512);
            let (bytes, _, _) = reader
                .get_tensor_raw("t.weight")
                .unwrap_or_else(|e| panic!("id {id} at {n} elements: {e}"));
            assert_eq!(bytes.len(), expected, "id {id} at {n} elements");
        }
    }
}

/// done_when 3, removed id: refused by NAME, as a removed type — not as corruption, and
/// not as the bare "Unsupported dtype N" it used to be.
#[test]
fn test_3601_removed_id_is_refused_by_name() {
    let reader = one_tensor(&[64], 32, &payload(64), 0);
    let err = reader
        .get_tensor_raw("t.weight")
        .expect_err("Q4_0_4_8 has no layout upstream")
        .to_string();
    for needle in ["t.weight", "32", "Q4_0_4_8", "removed upstream", "ggml-type-v1.yaml"] {
        assert!(err.contains(needle), "missing {needle:?} in: {err}");
    }
    assert!(!err.contains("Unsupported dtype"), "bare old message: {err}");
}

/// done_when 3, unknown id: refused as a gap in apr's table, with somewhere to report it.
#[test]
fn test_3601_unknown_id_is_refused_as_a_gap_in_apr() {
    for id in [trueno_quant::GGML_TYPE_COUNT, 4242] {
        let reader = one_tensor(&[64], id, &payload(64), 0);
        let err = reader
            .get_tensor_raw("t.weight")
            .expect_err("not a ggml type at the pin")
            .to_string();
        for needle in [
            &id.to_string(),
            "newer than the ggml type table",
            "not a corrupt file",
            "github.com/paiml/aprender/issues",
        ] {
            assert!(err.contains(needle), "id {id}: missing {needle:?} in: {err}");
        }
        // `try_gguf_raw_import` (GH-375) falls back to the dequant path on these two
        // phrases; a sizing refusal must not trip it.
        assert!(!err.contains("cannot represent exactly") && !err.contains("not yet supported"));
    }
}

/// #3656: sized is not readable. `get_tensor_f32` used to return `(b - 128) * 0.01` per
/// raw byte for ids 16..=23 and `Ok`; every IQ/TQ tensor must now be refused by type.
/// With #3601 sizing these ids, raw import reaches the GH-375 fallback — so this refusal
/// is what stops that fallback from importing invented weights.
#[test]
fn test_3656_get_tensor_f32_refuses_iq_and_tq_instead_of_approximating() {
    for (id, name, blck, type_size) in IQ_AND_TQ {
        let reader = one_tensor(&[blck as u64], id, &payload(type_size), 0);
        let err = reader
            .get_tensor_f32("t.weight")
            .expect_err("no dequantizer here: must refuse, never approximate")
            .to_string();
        for needle in ["t.weight", name, &format!("ggml type {id}"), "no dequantizer"] {
            assert!(err.contains(needle), "{name}: missing {needle:?} in: {err}");
        }
        assert!(
            !err.contains("cannot represent exactly") && !err.contains("not yet supported"),
            "{name}: a dequant refusal must not read as a GH-375 fallback trigger: {err}"
        );
    }
}

/// done_when 1 at the library boundary `apr inspect` calls: RosettaStone inspects an
/// IQ4_XS GGUF, names the dtype, and reports its true on-disk size.
#[test]
fn test_3601_rosetta_inspect_accepts_an_iq4_xs_gguf() {
    use std::io::Write;

    let bytes = build_synthetic_gguf_with_tensor("blk.0.ffn_up.weight", &[512], 23, &payload(272), &[]);
    let mut file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp file");
    file.write_all(&bytes).expect("write GGUF");
    file.flush().expect("flush");

    let report = crate::format::rosetta::RosettaStone::new()
        .inspect(file.path())
        .expect("an IQ4_XS GGUF must inspect");
    let t = report
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up.weight")
        .expect("tensor listed");
    assert_eq!(t.dtype, "IQ4_XS");
    assert_eq!(t.size_bytes, 2 * 136);
}

/// `apr tensors` checks every tensor's extent against the file before listing. With
/// IQ2_XXS weighted at 144/256 bytes/element it declared a correctly sized file
/// "Truncated" — measured on Qwen3.5-0.8B-UD-IQ2_XXS.gguf. Sized right, it lists it.
#[test]
fn test_3601_tensor_listing_accepts_a_correctly_sized_iq2_xxs_gguf() {
    let bytes = build_synthetic_gguf_with_tensor("t.weight", &[512], 16, &payload(2 * 66), &[]);
    let listing = crate::format::tensors::list_tensors_from_bytes(
        &bytes,
        crate::format::tensors::TensorListOptions::default(),
    )
    .expect("a correctly sized IQ2_XXS GGUF must list");
    assert_eq!(listing.tensors.len(), 1);
    assert_eq!(listing.tensors[0].dtype, "IQ2_XXS");
    assert_eq!(listing.tensors[0].size_bytes, 2 * 66);
}
