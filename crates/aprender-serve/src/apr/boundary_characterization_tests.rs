//! CHARACTERIZATION — the four remaining serve boundaries that turn an id or a
//! name into a quant type, snapshotted before PMAT-3430 Phases 3-4 move them
//! onto `trueno_quant::GgmlType`.
//!
//! Written against UNCHANGED code; must pass UNCHANGED afterwards. Together
//! with `gguf/dtype_characterization_tests.rs` this covers all seven boundaries
//! #3430 Q1-c enumerates, which is the only reviewable evidence that the
//! admitted sets were REPRODUCED rather than chosen.
//!
//! `apr/mod.rs`'s boundary is not a function — it is an arm inside
//! `TensorEntry::from_binary`'s `match dtype_byte`. It is therefore exercised
//! through the real parser with a minimal entry, not through a copy of the
//! expression: a test of a re-typed copy would pass while the real path changed.

use super::{MappedAprModel, TensorEntry};
use crate::apr::dequant::dtype_to_ggml_qtype;
use crate::infer::qtype_to_dtype_str;

/// The 16 (id, name) pairs serve admitted before M1.
const ADMITTED_TODAY: [(u32, &str); 16] = [
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (6, "Q5_0"),
    (7, "Q5_1"),
    (8, "Q8_0"),
    (9, "Q8_1"),
    (10, "Q2_K"),
    (11, "Q3_K"),
    (12, "Q4_K"),
    (13, "Q5_K"),
    (14, "Q6_K"),
    (16, "IQ2_XXS"),
    (17, "IQ2_XS"),
    (30, "BF16"),
];

fn admitted_name(id: u32) -> Option<&'static str> {
    ADMITTED_TODAY
        .iter()
        .find(|(candidate, _)| *candidate == id)
        .map(|(_, name)| *name)
}

/// A minimal v2 tensor entry: name "t", the dtype byte under test, 0 dims,
/// offset 0, size 0. Shaped from `TensorEntry::from_binary`'s own doc comment.
fn entry_with_dtype(dtype_byte: u8) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&1u16.to_le_bytes()); // name_len
    v.push(b't'); // name
    v.push(dtype_byte); // dtype
    v.push(0u8); // ndim
    v.extend_from_slice(&0u64.to_le_bytes()); // offset
    v.extend_from_slice(&0u64.to_le_bytes()); // size
    v
}

#[test]
fn infer_qtype_to_dtype_str_names_exactly_the_admitted_ids() {
    for id in 0u32..=255 {
        let got = qtype_to_dtype_str(id);
        match admitted_name(id) {
            Some(name) => assert_eq!(got, name, "qtype {id} renamed"),
            None => assert_eq!(got, "Unknown", "qtype {id} is newly named {got}"),
        }
    }
}

#[test]
fn apr_tensor_entry_dtype_byte_mapping_is_unchanged_over_the_whole_byte_range() {
    for byte in 0u8..=255 {
        let (entry, _) = TensorEntry::from_binary(&entry_with_dtype(byte))
            .unwrap_or_else(|e| panic!("dtype byte {byte} no longer parses: {e}"));
        let expected: String = match byte {
            // APR-native ids, which are NOT ggml and must stay that way (GH-438).
            128 => "q4".to_string(),
            129 => "q8".to_string(),
            8 => "q4".to_string(),
            9 => "q8".to_string(),
            other => admitted_name(u32::from(other)).unwrap_or("F32").to_string(),
        };
        assert_eq!(
            entry.dtype, expected,
            "APR dtype byte {byte} changed meaning"
        );
    }
}

#[test]
fn special_tokens_dtype_to_qtype_maps_every_admitted_name_and_zero_otherwise() {
    for (id, name) in ADMITTED_TODAY {
        assert_eq!(MappedAprModel::dtype_to_qtype(name), id, "{name} moved");
        assert_eq!(
            MappedAprModel::dtype_to_qtype(&name.to_ascii_lowercase()),
            id,
            "{name} moved in lower case"
        );
    }
    // Unknown names collapse to 0 (= F32). Lossy, and characterized as such.
    for name in [
        "", "nonsense", "Q9_9", "q4_K", "IQ3_S", "TQ1_0", "MXFP4", "NVFP4",
    ] {
        assert_eq!(
            MappedAprModel::dtype_to_qtype(name),
            0,
            "dtype {name:?} is newly recognised"
        );
    }
}

#[test]
fn dequant_dtype_to_ggml_qtype_admits_only_the_quantized_subset() {
    // This boundary deliberately admits LESS than the enum knows: F32/F16/BF16
    // are not quantized and APR-native q4/q8 are a different layout, so both
    // are None. That narrowing is the property under test.
    for (id, name) in ADMITTED_TODAY {
        let got = dtype_to_ggml_qtype(name);
        let is_quantized_here = !matches!(name, "F32" | "F16" | "BF16");
        if is_quantized_here {
            assert_eq!(got, Some(id), "{name} is no longer dequantizable");
        } else {
            assert_eq!(got, None, "{name} is newly treated as quantized");
        }
    }
    for name in ["q4", "q8", "", "nonsense", "IQ3_S", "TQ1_0", "MXFP4"] {
        assert_eq!(
            dtype_to_ggml_qtype(name),
            None,
            "dtype {name:?} is newly dequantizable"
        );
    }
}
