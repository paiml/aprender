//! ONE ggml tensor-type enum, and ONE table of what each type weighs.
//!
//! # Why this module exists (PMAT-3430, PP-QUANT-001 M1)
//!
//! aprender named ggml tensor types in three independent enums — serve's
//! `GgmlQuantType` (16 ids), core's `GgmlType` (12), compute's `GgmlType` (15)
//! — plus a fourth private label table, against 35 live ids upstream. They
//! disagreed about which ids exist, about spelling, and about SIZES. Nothing
//! compared any of them to upstream, so `NVFP4` (40), `Q1_0` (41) and `Q2_0`
//! (42) appeared upstream with no signal here at all.
//!
//! # The oracle is upstream, by reference
//!
//! Operator ruling 2026-09-20, clauses 2' and 3: upstream ggml at
//! `scripts/llama_pin.toml` `build_commit` is the reference; in-tree values are
//! the defendant. Every number in [`TRAITS`] is extracted from ggml's own
//! compiled `type_traits` table by `scripts/extract_ggml_traits.sh` into
//! `fixtures/ggml_traits.json`, and `tests/ggml_traits_fixture.rs` fails if a
//! single row here stops matching that fixture. The fixture records the sha it
//! was read from; a repo-level invariant (`monorepo_invariants.rs`) fails if
//! that sha stops being the pin — so bumping the comparator turns this RED in
//! the bump's own PR rather than silently later (clause 2a, generalised by
//! #3563).
//!
//! # What this module is NOT
//!
//! It is not an admission policy. [`GgmlType::from_id`] knows all 35 live ids;
//! whether a given crate should ACCEPT a given id at a parse boundary is that
//! crate's decision, and M1 preserves each one exactly as it was (#3430 Q1-c).
//! It also carries no dequantiser: knowing that `IQ3_S` is 110 bytes per 256
//! elements is not the same as being able to read one.

use core::fmt;

/// How a type stores its numbers.
///
/// This is aprender's classification, not upstream's: ggml's table has no
/// family column, only an `is_quantized` flag. The link back to upstream that
/// CAN fail is exactly that flag — [`GgmlFamily::Float`] and [`GgmlFamily::Int`]
/// are the non-quantized families, every other live family is quantized, and
/// `tests/ggml_traits_fixture.rs` checks that against the fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlFamily {
    /// IEEE / bfloat floats stored one per "block": `F32`, `F16`, `F64`, `BF16`.
    Float,
    /// Plain integers, one per "block": `I8`, `I16`, `I32`, `I64`.
    Int,
    /// The legacy fixed-block quants — a scale (and for the `_1` forms a min)
    /// plus packed weights: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, and
    /// upstream's newer `Q1_0` (128/block) and `Q2_0` (64/block).
    Affine,
    /// K-quants: 256-element super-blocks with per-sub-block scales.
    KQuant,
    /// The importance-matrix / codebook quants (`IQ*`), 256-element super-blocks
    /// except `IQ4_NL`, which is 32.
    Codebook,
    /// Ternary packings `TQ1_0` and `TQ2_0`.
    Ternary,
    /// Microscaling floats: `MXFP4` (E8M0 scale, 32/block) and `NVFP4`
    /// (UE4M3 sub-block scales, 64/block).
    MicroscaleFp,
    /// An id upstream has removed. No [`GgmlType`] variant constructs one; the
    /// slot exists so [`TRAITS`] can stay indexed by id.
    Removed,
}

/// What one ggml tensor type weighs, and what to call it.
///
/// `blck_size` and `type_size` are ggml's own field names, and ggml's own
/// values: `type_size` bytes hold exactly `blck_size` elements. Both are 0 for
/// a removed id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantTraits {
    /// The upstream ENUM name without its `GGML_TYPE_` prefix — `"Q4_K"`,
    /// `"IQ2_XXS"`, `"BF16"`. This is the GGUF-facing spelling and what
    /// [`GgmlType::as_str`] returns.
    pub name: &'static str,
    /// The name `ggml_type_name()` returns — `"q4_K"`, `"iq2_xxs"`, `"bf16"`.
    /// Differs from [`Self::name`] in case only, which the fixture test asserts.
    pub ggml_name: &'static str,
    /// Elements per block. 1 for the unquantized types, 0 for a removed id.
    pub blck_size: u32,
    /// Bytes per block. 0 for a removed id.
    pub type_size: u32,
    /// aprender's family classification (see [`GgmlFamily`]).
    pub family: GgmlFamily,
}

/// Why an id is not a [`GgmlType`].
///
/// The distinction is the point: `4` is not an unknown id, it is `Q4_2`, which
/// upstream removed. A loader that says "unknown tensor type 4" sends its
/// reader looking for a type that was deleted in 2023.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlTypeError {
    /// The id names a type upstream has removed.
    Removed {
        /// The removed id.
        id: u32,
        /// Its upstream name, e.g. `"Q4_0_4_8"`.
        name: &'static str,
    },
    /// The id is not in `0..GGML_TYPE_COUNT` at the pinned commit.
    Unknown {
        /// The id that was offered.
        id: u32,
    },
}

impl fmt::Display for GgmlTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Removed { id, name } => {
                write!(f, "ggml tensor type {id} ({name}) was removed upstream")
            }
            Self::Unknown { id } => write!(f, "unknown ggml tensor type {id}"),
        }
    }
}

impl std::error::Error for GgmlTypeError {}

/// The number of id slots upstream defines (`GGML_TYPE_COUNT`).
pub const GGML_TYPE_COUNT: u32 = 43;

// GENERATED ROWS: the `GgmlType` variants and the `TRAITS` table below are
// emitted from fixtures/ggml_traits.json, which scripts/extract_ggml_traits.sh
// extracted from upstream ggml at the pinned comparator commit. Editing a row
// by hand turns tests/ggml_traits_fixture.rs RED.

/// A ggml tensor type, by its upstream id.
///
/// 35 variants: every LIVE id at the pinned commit, and only those. The 8 ids
/// upstream removed are not constructible — see [`GgmlType::try_from_id`], which
/// tells a removed id apart from one that never existed — but they keep their
/// slots in [`TRAITS`] so the table is indexed by id.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)] // upstream ids: Q4_0, TQ1_0 — the tree's spelling
pub enum GgmlType {
    /// `F32` — ggml `f32`: `blck_size` 1, `type_size` 4.
    F32 = 0,
    /// `F16` — ggml `f16`: `blck_size` 1, `type_size` 2.
    F16 = 1,
    /// `Q4_0` — ggml `q4_0`: `blck_size` 32, `type_size` 18.
    Q4_0 = 2,
    /// `Q4_1` — ggml `q4_1`: `blck_size` 32, `type_size` 20.
    Q4_1 = 3,
    /// `Q5_0` — ggml `q5_0`: `blck_size` 32, `type_size` 22.
    Q5_0 = 6,
    /// `Q5_1` — ggml `q5_1`: `blck_size` 32, `type_size` 24.
    Q5_1 = 7,
    /// `Q8_0` — ggml `q8_0`: `blck_size` 32, `type_size` 34.
    Q8_0 = 8,
    /// `Q8_1` — ggml `q8_1`: `blck_size` 32, `type_size` 36.
    Q8_1 = 9,
    /// `Q2_K` — ggml `q2_K`: `blck_size` 256, `type_size` 84.
    Q2K = 10,
    /// `Q3_K` — ggml `q3_K`: `blck_size` 256, `type_size` 110.
    Q3K = 11,
    /// `Q4_K` — ggml `q4_K`: `blck_size` 256, `type_size` 144.
    Q4K = 12,
    /// `Q5_K` — ggml `q5_K`: `blck_size` 256, `type_size` 176.
    Q5K = 13,
    /// `Q6_K` — ggml `q6_K`: `blck_size` 256, `type_size` 210.
    Q6K = 14,
    /// `Q8_K` — ggml `q8_K`: `blck_size` 256, `type_size` 292.
    Q8K = 15,
    /// `IQ2_XXS` — ggml `iq2_xxs`: `blck_size` 256, `type_size` 66.
    IQ2XXS = 16,
    /// `IQ2_XS` — ggml `iq2_xs`: `blck_size` 256, `type_size` 74.
    IQ2XS = 17,
    /// `IQ3_XXS` — ggml `iq3_xxs`: `blck_size` 256, `type_size` 98.
    IQ3XXS = 18,
    /// `IQ1_S` — ggml `iq1_s`: `blck_size` 256, `type_size` 50.
    IQ1S = 19,
    /// `IQ4_NL` — ggml `iq4_nl`: `blck_size` 32, `type_size` 18.
    IQ4NL = 20,
    /// `IQ3_S` — ggml `iq3_s`: `blck_size` 256, `type_size` 110.
    IQ3S = 21,
    /// `IQ2_S` — ggml `iq2_s`: `blck_size` 256, `type_size` 82.
    IQ2S = 22,
    /// `IQ4_XS` — ggml `iq4_xs`: `blck_size` 256, `type_size` 136.
    IQ4XS = 23,
    /// `I8` — ggml `i8`: `blck_size` 1, `type_size` 1.
    I8 = 24,
    /// `I16` — ggml `i16`: `blck_size` 1, `type_size` 2.
    I16 = 25,
    /// `I32` — ggml `i32`: `blck_size` 1, `type_size` 4.
    I32 = 26,
    /// `I64` — ggml `i64`: `blck_size` 1, `type_size` 8.
    I64 = 27,
    /// `F64` — ggml `f64`: `blck_size` 1, `type_size` 8.
    F64 = 28,
    /// `IQ1_M` — ggml `iq1_m`: `blck_size` 256, `type_size` 56.
    IQ1M = 29,
    /// `BF16` — ggml `bf16`: `blck_size` 1, `type_size` 2.
    BF16 = 30,
    /// `TQ1_0` — ggml `tq1_0`: `blck_size` 256, `type_size` 54.
    TQ1_0 = 34,
    /// `TQ2_0` — ggml `tq2_0`: `blck_size` 256, `type_size` 66.
    TQ2_0 = 35,
    /// `MXFP4` — ggml `mxfp4`: `blck_size` 32, `type_size` 17.
    MXFP4 = 39,
    /// `NVFP4` — ggml `nvfp4`: `blck_size` 64, `type_size` 36.
    NVFP4 = 40,
    /// `Q1_0` — ggml `q1_0`: `blck_size` 128, `type_size` 18.
    Q1_0 = 41,
    /// `Q2_0` — ggml `q2_0`: `blck_size` 64, `type_size` 18.
    Q2_0 = 42,
}

/// The ggml tensor-type table, INDEXED BY ID, all 43 slots including the 8 removed.
///
/// Dense so that `TRAITS[t.as_id() as usize]` is the lookup and no id needs a
/// search. A removed slot carries `family: GgmlFamily::Removed` and upstream's
/// own zeroes, which is what ggml reports for it.
pub const TRAITS: [QuantTraits; 43] = [
    QuantTraits {
        name: "F32",
        ggml_name: "f32",
        blck_size: 1,
        type_size: 4,
        family: GgmlFamily::Float,
    }, // 0
    QuantTraits {
        name: "F16",
        ggml_name: "f16",
        blck_size: 1,
        type_size: 2,
        family: GgmlFamily::Float,
    }, // 1
    QuantTraits {
        name: "Q4_0",
        ggml_name: "q4_0",
        blck_size: 32,
        type_size: 18,
        family: GgmlFamily::Affine,
    }, // 2
    QuantTraits {
        name: "Q4_1",
        ggml_name: "q4_1",
        blck_size: 32,
        type_size: 20,
        family: GgmlFamily::Affine,
    }, // 3
    QuantTraits {
        name: "Q4_2",
        ggml_name: "DEPRECATED",
        blck_size: 0,
        type_size: 0,
        family: GgmlFamily::Removed,
    }, // 4
    QuantTraits {
        name: "Q4_3",
        ggml_name: "DEPRECATED",
        blck_size: 0,
        type_size: 0,
        family: GgmlFamily::Removed,
    }, // 5
    QuantTraits {
        name: "Q5_0",
        ggml_name: "q5_0",
        blck_size: 32,
        type_size: 22,
        family: GgmlFamily::Affine,
    }, // 6
    QuantTraits {
        name: "Q5_1",
        ggml_name: "q5_1",
        blck_size: 32,
        type_size: 24,
        family: GgmlFamily::Affine,
    }, // 7
    QuantTraits {
        name: "Q8_0",
        ggml_name: "q8_0",
        blck_size: 32,
        type_size: 34,
        family: GgmlFamily::Affine,
    }, // 8
    QuantTraits {
        name: "Q8_1",
        ggml_name: "q8_1",
        blck_size: 32,
        type_size: 36,
        family: GgmlFamily::Affine,
    }, // 9
    QuantTraits {
        name: "Q2_K",
        ggml_name: "q2_K",
        blck_size: 256,
        type_size: 84,
        family: GgmlFamily::KQuant,
    }, // 10
    QuantTraits {
        name: "Q3_K",
        ggml_name: "q3_K",
        blck_size: 256,
        type_size: 110,
        family: GgmlFamily::KQuant,
    }, // 11
    QuantTraits {
        name: "Q4_K",
        ggml_name: "q4_K",
        blck_size: 256,
        type_size: 144,
        family: GgmlFamily::KQuant,
    }, // 12
    QuantTraits {
        name: "Q5_K",
        ggml_name: "q5_K",
        blck_size: 256,
        type_size: 176,
        family: GgmlFamily::KQuant,
    }, // 13
    QuantTraits {
        name: "Q6_K",
        ggml_name: "q6_K",
        blck_size: 256,
        type_size: 210,
        family: GgmlFamily::KQuant,
    }, // 14
    QuantTraits {
        name: "Q8_K",
        ggml_name: "q8_K",
        blck_size: 256,
        type_size: 292,
        family: GgmlFamily::KQuant,
    }, // 15
    QuantTraits {
        name: "IQ2_XXS",
        ggml_name: "iq2_xxs",
        blck_size: 256,
        type_size: 66,
        family: GgmlFamily::Codebook,
    }, // 16
    QuantTraits {
        name: "IQ2_XS",
        ggml_name: "iq2_xs",
        blck_size: 256,
        type_size: 74,
        family: GgmlFamily::Codebook,
    }, // 17
    QuantTraits {
        name: "IQ3_XXS",
        ggml_name: "iq3_xxs",
        blck_size: 256,
        type_size: 98,
        family: GgmlFamily::Codebook,
    }, // 18
    QuantTraits {
        name: "IQ1_S",
        ggml_name: "iq1_s",
        blck_size: 256,
        type_size: 50,
        family: GgmlFamily::Codebook,
    }, // 19
    QuantTraits {
        name: "IQ4_NL",
        ggml_name: "iq4_nl",
        blck_size: 32,
        type_size: 18,
        family: GgmlFamily::Codebook,
    }, // 20
    QuantTraits {
        name: "IQ3_S",
        ggml_name: "iq3_s",
        blck_size: 256,
        type_size: 110,
        family: GgmlFamily::Codebook,
    }, // 21
    QuantTraits {
        name: "IQ2_S",
        ggml_name: "iq2_s",
        blck_size: 256,
        type_size: 82,
        family: GgmlFamily::Codebook,
    }, // 22
    QuantTraits {
        name: "IQ4_XS",
        ggml_name: "iq4_xs",
        blck_size: 256,
        type_size: 136,
        family: GgmlFamily::Codebook,
    }, // 23
    QuantTraits {
        name: "I8",
        ggml_name: "i8",
        blck_size: 1,
        type_size: 1,
        family: GgmlFamily::Int,
    }, // 24
    QuantTraits {
        name: "I16",
        ggml_name: "i16",
        blck_size: 1,
        type_size: 2,
        family: GgmlFamily::Int,
    }, // 25
    QuantTraits {
        name: "I32",
        ggml_name: "i32",
        blck_size: 1,
        type_size: 4,
        family: GgmlFamily::Int,
    }, // 26
    QuantTraits {
        name: "I64",
        ggml_name: "i64",
        blck_size: 1,
        type_size: 8,
        family: GgmlFamily::Int,
    }, // 27
    QuantTraits {
        name: "F64",
        ggml_name: "f64",
        blck_size: 1,
        type_size: 8,
        family: GgmlFamily::Float,
    }, // 28
    QuantTraits {
        name: "IQ1_M",
        ggml_name: "iq1_m",
        blck_size: 256,
        type_size: 56,
        family: GgmlFamily::Codebook,
    }, // 29
    QuantTraits {
        name: "BF16",
        ggml_name: "bf16",
        blck_size: 1,
        type_size: 2,
        family: GgmlFamily::Float,
    }, // 30
    QuantTraits {
        name: "Q4_0_4_4",
        ggml_name: "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
        blck_size: 0,
        type_size: 0,
        family: GgmlFamily::Removed,
    }, // 31
    QuantTraits {
        name: "Q4_0_4_8",
        ggml_name: "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
        blck_size: 0,
        type_size: 0,
        family: GgmlFamily::Removed,
    }, // 32
    QuantTraits {
        name: "Q4_0_8_8",
        ggml_name: "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
        blck_size: 0,
        type_size: 0,
        family: GgmlFamily::Removed,
    }, // 33
    QuantTraits {
        name: "TQ1_0",
        ggml_name: "tq1_0",
        blck_size: 256,
        type_size: 54,
        family: GgmlFamily::Ternary,
    }, // 34
    QuantTraits {
        name: "TQ2_0",
        ggml_name: "tq2_0",
        blck_size: 256,
        type_size: 66,
        family: GgmlFamily::Ternary,
    }, // 35
    QuantTraits {
        name: "IQ4_NL_4_4",
        ggml_name: "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
        blck_size: 0,
        type_size: 0,
        family: GgmlFamily::Removed,
    }, // 36
    QuantTraits {
        name: "IQ4_NL_4_8",
        ggml_name: "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
        blck_size: 0,
        type_size: 0,
        family: GgmlFamily::Removed,
    }, // 37
    QuantTraits {
        name: "IQ4_NL_8_8",
        ggml_name: "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
        blck_size: 0,
        type_size: 0,
        family: GgmlFamily::Removed,
    }, // 38
    QuantTraits {
        name: "MXFP4",
        ggml_name: "mxfp4",
        blck_size: 32,
        type_size: 17,
        family: GgmlFamily::MicroscaleFp,
    }, // 39
    QuantTraits {
        name: "NVFP4",
        ggml_name: "nvfp4",
        blck_size: 64,
        type_size: 36,
        family: GgmlFamily::MicroscaleFp,
    }, // 40
    QuantTraits {
        name: "Q1_0",
        ggml_name: "q1_0",
        blck_size: 128,
        type_size: 18,
        family: GgmlFamily::Affine,
    }, // 41
    QuantTraits {
        name: "Q2_0",
        ggml_name: "q2_0",
        blck_size: 64,
        type_size: 18,
        family: GgmlFamily::Affine,
    }, // 42
];

const fn from_id_inner(id: u32) -> Option<GgmlType> {
    match id {
        0 => Some(GgmlType::F32),
        1 => Some(GgmlType::F16),
        2 => Some(GgmlType::Q4_0),
        3 => Some(GgmlType::Q4_1),
        6 => Some(GgmlType::Q5_0),
        7 => Some(GgmlType::Q5_1),
        8 => Some(GgmlType::Q8_0),
        9 => Some(GgmlType::Q8_1),
        10 => Some(GgmlType::Q2K),
        11 => Some(GgmlType::Q3K),
        12 => Some(GgmlType::Q4K),
        13 => Some(GgmlType::Q5K),
        14 => Some(GgmlType::Q6K),
        15 => Some(GgmlType::Q8K),
        16 => Some(GgmlType::IQ2XXS),
        17 => Some(GgmlType::IQ2XS),
        18 => Some(GgmlType::IQ3XXS),
        19 => Some(GgmlType::IQ1S),
        20 => Some(GgmlType::IQ4NL),
        21 => Some(GgmlType::IQ3S),
        22 => Some(GgmlType::IQ2S),
        23 => Some(GgmlType::IQ4XS),
        24 => Some(GgmlType::I8),
        25 => Some(GgmlType::I16),
        26 => Some(GgmlType::I32),
        27 => Some(GgmlType::I64),
        28 => Some(GgmlType::F64),
        29 => Some(GgmlType::IQ1M),
        30 => Some(GgmlType::BF16),
        34 => Some(GgmlType::TQ1_0),
        35 => Some(GgmlType::TQ2_0),
        39 => Some(GgmlType::MXFP4),
        40 => Some(GgmlType::NVFP4),
        41 => Some(GgmlType::Q1_0),
        42 => Some(GgmlType::Q2_0),
        _ => None,
    }
}

/// The 8 ids upstream has removed: constructible by no one, but distinguishable
/// from an id that never existed. `// GGML_TYPE_Q4_0_4_8 = 32,` in ggml.h says
/// nothing about being removed, so membership here is positional, never textual.
const REMOVED_IDS: [u32; 8] = [4, 5, 31, 32, 33, 36, 37, 38];

/// Every live variant, in id order — the iteration surface for tests and callers.
pub const ALL: [GgmlType; 35] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
    GgmlType::Q8_1,
    GgmlType::Q2K,
    GgmlType::Q3K,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Q8K,
    GgmlType::IQ2XXS,
    GgmlType::IQ2XS,
    GgmlType::IQ3XXS,
    GgmlType::IQ1S,
    GgmlType::IQ4NL,
    GgmlType::IQ3S,
    GgmlType::IQ2S,
    GgmlType::IQ4XS,
    GgmlType::I8,
    GgmlType::I16,
    GgmlType::I32,
    GgmlType::I64,
    GgmlType::F64,
    GgmlType::IQ1M,
    GgmlType::BF16,
    GgmlType::TQ1_0,
    GgmlType::TQ2_0,
    GgmlType::MXFP4,
    GgmlType::NVFP4,
    GgmlType::Q1_0,
    GgmlType::Q2_0,
];

impl GgmlType {
    /// The type with this id, or `None` if no live type has it.
    ///
    /// Returns `None` for both a removed id and an id that never existed; use
    /// [`Self::try_from_id`] when the difference should reach the user. The
    /// signature is `Option` because that is what serve's `GgmlQuantType::from_id`
    /// returned, and M1 does not move its callers.
    #[must_use]
    pub const fn from_id(id: u32) -> Option<Self> {
        from_id_inner(id)
    }

    /// The type with this id, naming WHY it is not one when it is not.
    ///
    /// # Errors
    ///
    /// [`GgmlTypeError::Removed`] for the 8 ids upstream has deleted,
    /// [`GgmlTypeError::Unknown`] for anything else outside the live set.
    pub const fn try_from_id(id: u32) -> Result<Self, GgmlTypeError> {
        match from_id_inner(id) {
            Some(t) => Ok(t),
            None => {
                if id < GGML_TYPE_COUNT {
                    // In range and not live => one of the 8 removed slots, whose
                    // TRAITS row still carries upstream's name for it.
                    Err(GgmlTypeError::Removed {
                        id,
                        name: TRAITS[id as usize].name,
                    })
                } else {
                    Err(GgmlTypeError::Unknown { id })
                }
            }
        }
    }

    /// This type's upstream id.
    #[must_use]
    pub const fn as_id(self) -> u32 {
        self as u32
    }

    /// This type's upstream id as a byte, for the APR tensor header.
    ///
    /// Total, with no truncation to hide: the largest id at the pin is 42.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    /// The GGUF-facing name: `"Q4_K"`, `"IQ2_XXS"`, `"BF16"`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        self.traits().name
    }

    /// The name ggml itself prints: `"q4_K"`, `"iq2_xxs"`, `"bf16"`.
    #[must_use]
    pub const fn ggml_name(self) -> &'static str {
        self.traits().ggml_name
    }

    /// This type's row in [`TRAITS`].
    #[must_use]
    pub const fn traits(self) -> &'static QuantTraits {
        &TRAITS[self as usize]
    }

    /// This type's family.
    #[must_use]
    pub const fn family(self) -> GgmlFamily {
        self.traits().family
    }

    /// Bytes per block — ggml's `type_size`.
    #[must_use]
    pub const fn block_bytes(&self) -> usize {
        self.traits().type_size as usize
    }

    /// Elements per block — ggml's `blck_size`.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.traits().blck_size as usize
    }

    /// Bytes for `n_elements` weights, rounding a partial block UP.
    ///
    /// This is compute's arithmetic, unchanged, because its callers depend on
    /// the rounding. It says nothing about whether a partial block is LEGAL —
    /// [`Self::checked_tensor_bytes`] is the one that refuses.
    #[must_use]
    pub const fn tensor_bytes(&self, n_elements: usize) -> usize {
        let bs = self.block_size();
        let n_blocks = n_elements.div_ceil(bs);
        n_blocks * self.block_bytes()
    }

    /// Bytes for `n_elements` weights, or `None` if that is not a whole number
    /// of blocks or the product overflows.
    ///
    /// A quantized tensor whose element count is not a multiple of `blck_size`
    /// is not a short tensor, it is a malformed one; rounding it up produces a
    /// plausible byte count for a file that cannot be read.
    #[must_use]
    pub const fn checked_tensor_bytes(self, n_elements: usize) -> Option<usize> {
        let bs = self.block_size();
        if bs == 0 || !n_elements.is_multiple_of(bs) {
            return None;
        }
        match (n_elements / bs).checked_mul(self.block_bytes()) {
            Some(bytes) => Some(bytes),
            None => None,
        }
    }

    /// Parse a GGUF-facing name, accepting its exact spelling or its all-lowercase
    /// form — `"Q4_K"` and `"q4_k"`, as serve's `from_str_lossy` accepted them.
    ///
    /// Note that ggml's own `"q4_K"` is neither, and is deliberately still not
    /// accepted: M1 changes no parser's admitted set.
    #[must_use]
    pub fn from_str_lossy(s: &str) -> Option<Self> {
        ALL.into_iter().find(|t| {
            let name = t.as_str();
            s == name
                || (s.eq_ignore_ascii_case(name) && s.chars().all(|c| !c.is_ascii_uppercase()))
        })
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Const assertions: these fail the BUILD, not a test run.
// ---------------------------------------------------------------------------

/// `TRAITS` is indexed by id, so every variant must find its own row there.
const _: () = {
    let mut i = 0;
    while i < ALL.len() {
        let t = ALL[i];
        assert!(
            TRAITS[t.as_id() as usize].blck_size > 0,
            "a live type indexes a removed TRAITS row"
        );
        i += 1;
    }
};

/// Every removed id is a removed row, and there are exactly 8 of them.
const _: () = {
    let mut i = 0;
    while i < REMOVED_IDS.len() {
        let id = REMOVED_IDS[i];
        assert!(
            TRAITS[id as usize].type_size == 0,
            "a removed id has a non-zero type_size"
        );
        assert!(from_id_inner(id).is_none(), "a removed id is constructible");
        i += 1;
    }
    assert!(
        REMOVED_IDS.len() + ALL.len() == TRAITS.len(),
        "live + removed != 43"
    );
};

// ---------------------------------------------------------------------------
// Kani harness (KANI-GGMLT-001, contracts/ggml-type-v1.yaml)
// ---------------------------------------------------------------------------

/// Exhaustive over the id space: resolution agrees with the table, and no
/// method panics or overflows on a live type.
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(44)]
fn verify_ggml_type_id_space() {
    let id: u32 = kani::any();
    kani::assume(id < GGML_TYPE_COUNT);

    let live = TRAITS[id as usize].blck_size > 0;
    assert_eq!(GgmlType::from_id(id).is_some(), live);
    assert_eq!(GgmlType::try_from_id(id).is_ok(), live);

    if let Some(t) = GgmlType::from_id(id) {
        assert_eq!(t.as_id(), id);
        assert!(t.block_size() > 0);
        assert!(t.block_bytes() > 0);
        assert_eq!(t.tensor_bytes(t.block_size()), t.block_bytes());
        assert_eq!(
            t.checked_tensor_bytes(t.block_size()),
            Some(t.block_bytes())
        );
    }
}
