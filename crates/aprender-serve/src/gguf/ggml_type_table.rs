//! #3432: ONE ggml type table, mirroring llama.cpp's `ggml.c` `type_traits[]`.
//!
//! Every place that needs "how many bytes does a tensor of ggml type N occupy"
//! used to answer it with its own hand-written `match`, and each of those
//! matches listed only the types its author had met. The loader's match
//! (`QuantizedGGUFTransformer::tensor_byte_size`) carried 11 arms, so real
//! unsloth files — `Qwen3.5-0.8B-UD-IQ2_XXS.gguf` (type 16) and
//! `Qwen3.5-0.8B-IQ4_XS.gguf` (type 23) — were refused at *size* time with
//! "Unsupported quantization type" before any kernel could be consulted.
//!
//! Sizing a type is NOT a claim that a kernel can run it: the dequant/matmul
//! refusal is downstream and unchanged (`fused_matmul_into.rs` fails loud for
//! any qtype it has no dequantizer for). This table answers one question only,
//! the one the GGUF byte layout already fixes.
//!
//! Each row is `(id, name, blck_size, type_size)` where `type_size` is
//! `sizeof(block_*)` for a block of `blck_size` elements, exactly as in ggml.

/// Static layout traits of one ggml tensor type.
///
/// Mirrors `ggml_type_traits` in llama.cpp's `ggml.c`: a block of `blck_size`
/// elements is stored in `type_size` bytes. Non-quantized types have
/// `blck_size == 1`, so `type_size` is simply bytes-per-element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlTypeTraits {
    /// The ggml type id as written in the GGUF tensor info record.
    pub id: u32,
    /// The ggml name of the type (`Q4_K`, `IQ2_XXS`, …), for error messages.
    pub name: &'static str,
    /// Elements per stored block (1 for the plain scalar types).
    pub blck_size: usize,
    /// Bytes occupied by one block of `blck_size` elements.
    pub type_size: usize,
}

/// Every ggml type id with a defined block layout, in id order.
///
/// Sizes are `sizeof(block_*)` from ggml's headers, e.g. `block_q4_K` is
/// `{ half d; half dmin; uint8_t scales[12]; uint8_t qs[128]; }` = 144 bytes
/// per 256 elements.
pub const GGML_TYPES: &[GgmlTypeTraits] = &[
    GgmlTypeTraits {
        id: 0,
        name: "F32",
        blck_size: 1,
        type_size: 4,
    },
    GgmlTypeTraits {
        id: 1,
        name: "F16",
        blck_size: 1,
        type_size: 2,
    },
    GgmlTypeTraits {
        id: 2,
        name: "Q4_0",
        blck_size: 32,
        type_size: 18,
    },
    GgmlTypeTraits {
        id: 3,
        name: "Q4_1",
        blck_size: 32,
        type_size: 20,
    },
    GgmlTypeTraits {
        id: 6,
        name: "Q5_0",
        blck_size: 32,
        type_size: 22,
    },
    GgmlTypeTraits {
        id: 7,
        name: "Q5_1",
        blck_size: 32,
        type_size: 24,
    },
    GgmlTypeTraits {
        id: 8,
        name: "Q8_0",
        blck_size: 32,
        type_size: 34,
    },
    GgmlTypeTraits {
        id: 9,
        name: "Q8_1",
        blck_size: 32,
        type_size: 36,
    },
    GgmlTypeTraits {
        id: 10,
        name: "Q2_K",
        blck_size: 256,
        type_size: 84,
    },
    GgmlTypeTraits {
        id: 11,
        name: "Q3_K",
        blck_size: 256,
        type_size: 110,
    },
    GgmlTypeTraits {
        id: 12,
        name: "Q4_K",
        blck_size: 256,
        type_size: 144,
    },
    GgmlTypeTraits {
        id: 13,
        name: "Q5_K",
        blck_size: 256,
        type_size: 176,
    },
    GgmlTypeTraits {
        id: 14,
        name: "Q6_K",
        blck_size: 256,
        type_size: 210,
    },
    GgmlTypeTraits {
        id: 15,
        name: "Q8_K",
        blck_size: 256,
        type_size: 292,
    },
    GgmlTypeTraits {
        id: 16,
        name: "IQ2_XXS",
        blck_size: 256,
        type_size: 66,
    },
    GgmlTypeTraits {
        id: 17,
        name: "IQ2_XS",
        blck_size: 256,
        type_size: 74,
    },
    GgmlTypeTraits {
        id: 18,
        name: "IQ3_XXS",
        blck_size: 256,
        type_size: 98,
    },
    GgmlTypeTraits {
        id: 19,
        name: "IQ1_S",
        blck_size: 256,
        type_size: 50,
    },
    GgmlTypeTraits {
        id: 20,
        name: "IQ4_NL",
        blck_size: 32,
        type_size: 18,
    },
    GgmlTypeTraits {
        id: 21,
        name: "IQ3_S",
        blck_size: 256,
        type_size: 110,
    },
    GgmlTypeTraits {
        id: 22,
        name: "IQ2_S",
        blck_size: 256,
        type_size: 82,
    },
    GgmlTypeTraits {
        id: 23,
        name: "IQ4_XS",
        blck_size: 256,
        type_size: 136,
    },
    GgmlTypeTraits {
        id: 24,
        name: "I8",
        blck_size: 1,
        type_size: 1,
    },
    GgmlTypeTraits {
        id: 25,
        name: "I16",
        blck_size: 1,
        type_size: 2,
    },
    GgmlTypeTraits {
        id: 26,
        name: "I32",
        blck_size: 1,
        type_size: 4,
    },
    GgmlTypeTraits {
        id: 27,
        name: "I64",
        blck_size: 1,
        type_size: 8,
    },
    GgmlTypeTraits {
        id: 28,
        name: "F64",
        blck_size: 1,
        type_size: 8,
    },
    GgmlTypeTraits {
        id: 29,
        name: "IQ1_M",
        blck_size: 256,
        type_size: 56,
    },
    GgmlTypeTraits {
        id: 30,
        name: "BF16",
        blck_size: 1,
        type_size: 2,
    },
    GgmlTypeTraits {
        id: 34,
        name: "TQ1_0",
        blck_size: 256,
        type_size: 54,
    },
    GgmlTypeTraits {
        id: 35,
        name: "TQ2_0",
        blck_size: 256,
        type_size: 66,
    },
];

/// Ids ggml has *named* but whose layout this table deliberately does not carry:
/// the removed/deprecated repack types, plus any type whose block layout we have
/// not verified. Naming them turns "Unsupported quantization type: 32" into a
/// message a reader can act on, while still refusing to invent a size.
const NAMED_UNSIZED_TYPES: &[(u32, &str)] = &[
    (4, "Q4_2 (removed from ggml)"),
    (5, "Q4_3 (removed from ggml)"),
    (31, "Q4_0_4_4 (removed from ggml: repacked at load time)"),
    (32, "Q4_0_4_8 (removed from ggml: repacked at load time)"),
    (33, "Q4_0_8_8 (removed from ggml: repacked at load time)"),
    (36, "IQ4_NL_4_4 (removed from ggml: repacked at load time)"),
    (37, "IQ4_NL_4_8 (removed from ggml: repacked at load time)"),
    (38, "IQ4_NL_8_8 (removed from ggml: repacked at load time)"),
    (39, "MXFP4 (layout not verified against ggml)"),
];

/// Layout traits for a ggml type id, or `None` if this table has no size for it.
#[must_use]
pub fn traits(id: u32) -> Option<&'static GgmlTypeTraits> {
    GGML_TYPES.iter().find(|t| t.id == id)
}

/// The ggml name of a type id, including ids that carry no size here.
#[must_use]
pub fn name(id: u32) -> Option<&'static str> {
    traits(id).map(|t| t.name).or_else(|| {
        NAMED_UNSIZED_TYPES
            .iter()
            .find(|(unsized_id, _)| *unsized_id == id)
            .map(|(_, unsized_name)| *unsized_name)
    })
}

/// The refusal text for an id with no size in this table.
fn unsupported(id: u32) -> String {
    match name(id) {
        Some(known) => format!("Unsupported quantization type: {id} ({known})"),
        None => format!("Unsupported quantization type: {id}"),
    }
}

/// Bytes a tensor of ggml type `id` with shape `dims` occupies in a GGUF file.
///
/// A 2-D tensor pads EACH ROW to a whole block, which is what the file layout
/// does and what the loader's `k_quant_bytes` has always done; 1-D (and higher)
/// shapes are sized from the flat element count. All arithmetic is checked, so
/// a corrupt header claiming absurd dims returns `Err` instead of wrapping into
/// a small, plausible, wrong byte count.
///
/// # Errors
///
/// Returns the refusal text when `id` has no layout here, or when the size
/// overflows `usize`.
pub fn byte_size(id: u32, dims: &[u64]) -> Result<usize, String> {
    let t = traits(id).ok_or_else(|| unsupported(id))?;
    if dims.len() == 2 {
        let rows = usize::try_from(dims[0]).map_err(|_| overflow(id, dims))?;
        let cols = usize::try_from(dims[1]).map_err(|_| overflow(id, dims))?;
        return cols
            .div_ceil(t.blck_size)
            .checked_mul(t.type_size)
            .and_then(|row_bytes| row_bytes.checked_mul(rows))
            .ok_or_else(|| overflow(id, dims));
    }
    let mut n: usize = 1;
    for &d in dims {
        let d = usize::try_from(d).map_err(|_| overflow(id, dims))?;
        n = n.checked_mul(d).ok_or_else(|| overflow(id, dims))?;
    }
    flat_byte_size(id, n)
}

/// Bytes for `num_elements` elements of type `id`, ignoring row structure.
///
/// This is the pre-#3432 sizing of the legacy arms (Q4_0/Q4_1/Q5_0/Q8_0/Q2_K
/// and the scalar types) — kept as its own entry point so the loader can
/// preserve those numbers exactly rather than silently re-sizing every model.
///
/// # Errors
///
/// Returns the refusal text when `id` has no layout here, or on overflow.
pub fn flat_byte_size(id: u32, num_elements: usize) -> Result<usize, String> {
    let t = traits(id).ok_or_else(|| unsupported(id))?;
    num_elements
        .div_ceil(t.blck_size)
        .checked_mul(t.type_size)
        .ok_or_else(|| {
            format!(
                "Tensor of ggml type {} with {num_elements} elements overflows usize",
                t.name
            )
        })
}

/// Overflow refusal naming the shape that produced it.
fn overflow(id: u32, dims: &[u64]) -> String {
    format!("Tensor of ggml type {id} with dims {dims:?} overflows usize")
}

#[cfg(test)]
mod ggml_type_table_tests {
    use super::*;

    /// A duplicate id would make `traits()` return whichever row came first —
    /// a silently wrong size for one of the two types.
    #[test]
    fn ids_are_unique_and_ascending() {
        let mut previous = None;
        for t in GGML_TYPES {
            if let Some(prev) = previous {
                assert!(
                    t.id > prev,
                    "table must be unique and ascending by id: {} after {prev}",
                    t.id
                );
            }
            previous = Some(t.id);
        }
    }

    /// A zero in either column would produce a zero-byte tensor (or a division
    /// by zero) rather than a refusal.
    #[test]
    fn every_row_has_a_nonzero_block_and_size() {
        for t in GGML_TYPES {
            assert!(t.blck_size > 0, "{} has blck_size 0", t.name);
            assert!(t.type_size > 0, "{} has type_size 0", t.name);
            assert!(!t.name.is_empty(), "id {} has no name", t.id);
        }
    }

    /// The rows this repo's existing kernels already agree on — if a transcription
    /// slip changed one of these, every model of that type would mis-load.
    #[test]
    fn spot_checks_against_ggml_sizes() {
        let cases: [(u32, &str, usize, usize); 8] = [
            (0, "F32", 1, 4),
            (1, "F16", 1, 2),
            (2, "Q4_0", 32, 18),
            (12, "Q4_K", 256, 144),
            (14, "Q6_K", 256, 210),
            (16, "IQ2_XXS", 256, 66),
            (23, "IQ4_XS", 256, 136),
            (30, "BF16", 1, 2),
        ];
        for (id, name, blck, size) in cases {
            let t = traits(id).expect("id must be in the table");
            assert_eq!(t.name, name, "id {id} name");
            assert_eq!(t.blck_size, blck, "{name} block size");
            assert_eq!(t.type_size, size, "{name} type size");
        }
    }

    /// 2-D tensors pad each ROW, which is the file layout — sizing them from the
    /// flat element count under-counts whenever a row is not a whole block.
    #[test]
    fn two_d_tensors_pad_each_row() {
        // 4 rows x 300 cols of Q4_K: each row is 2 super-blocks (300 -> 512).
        assert_eq!(byte_size(12, &[4, 300]).expect("Q4_K"), 4 * 2 * 144);
        // Aligned rows: row padding and flat sizing agree.
        assert_eq!(byte_size(12, &[4, 512]).expect("Q4_K"), 4 * 2 * 144);
    }

    #[test]
    fn one_d_tensors_are_sized_from_the_element_count() {
        assert_eq!(byte_size(0, &[1000]).expect("F32"), 4000);
        assert_eq!(byte_size(16, &[512]).expect("IQ2_XXS"), 2 * 66);
        assert_eq!(byte_size(23, &[512]).expect("IQ4_XS"), 2 * 136);
    }

    /// An id outside the table must refuse, naming the id — never fall back to
    /// a "close enough" size.
    #[test]
    fn an_unknown_id_is_refused_by_id() {
        let err = byte_size(4242, &[16]).expect_err("4242 is not a ggml type");
        assert!(err.contains("4242"), "got: {err}");
        assert!(err.contains("Unsupported quantization type"), "got: {err}");
        assert_eq!(name(4242), None);
    }

    /// A removed/unverified id is refused too, but by NAME — the message is the
    /// only thing that tells a user their file uses a repacked legacy type.
    #[test]
    fn a_named_but_unsized_id_is_refused_by_name() {
        let err = byte_size(32, &[16]).expect_err("Q4_0_4_8 carries no size here");
        assert!(err.contains("Q4_0_4_8"), "got: {err}");
        assert!(traits(32).is_none(), "32 must not be sized");
        assert_eq!(name(39), Some("MXFP4 (layout not verified against ggml)"));
    }

    /// Checked arithmetic: a corrupt header must produce an error, not a small
    /// wrapped byte count that then reads inside the file.
    #[test]
    fn absurd_dims_overflow_into_an_error() {
        let err = byte_size(12, &[u64::MAX, u64::MAX]).expect_err("must not wrap");
        assert!(err.contains("overflows usize"), "got: {err}");
        // Q8_K: 292 bytes per 256 elements, the widest row in the table — the
        // only one whose flat size overflows at `usize::MAX` elements.
        let err = flat_byte_size(15, usize::MAX).expect_err("must not wrap");
        assert!(err.contains("overflows usize"), "got: {err}");
    }
}
