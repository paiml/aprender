//! The scheme a GGUF declares: llama.cpp's `enum llama_ftype`, stored in `general.file_type`.
//!
//! # Why this module exists (#3762)
//!
//! `apr inspect` named the dtype holding the most parameters as a file's scheme, so a
//! Q4_K_M Qwen3.5 reported Q6_K: its 248k-vocab embedding outweighs every Q4_K matrix.
//! The file says what it is. `general.file_type` 15 is `LLAMA_FTYPE_MOSTLY_Q4_K_M`, and
//! the names of those ids belong to upstream.
//!
//! # The oracle is upstream, by reference
//!
//! As for [`crate::ggml_type`] (PMAT-3430 M1): every row of [`LLAMA_FTYPES`] is extracted
//! from `include/llama.h` at `scripts/llama_pin.toml` `build_commit` by
//! `scripts/extract_llama_ftypes.sh` into `fixtures/llama_ftypes.json`, and
//! `tests/llama_ftypes_fixture.rs` fails if a row here stops matching that fixture. A pin
//! bump that moves the enum turns it RED in the bump's own PR.
//!
//! The names are upstream's enum names without the `LLAMA_FTYPE_MOSTLY_` / `LLAMA_FTYPE_ALL_`
//! prefix: `Q4_K_M`, `F32`. Upstream's `llama_ftype_name` prose ("Q4_K - Medium") is not
//! used, because these names are also what `apr` prints for a tensor dtype.

/// Every live `llama_ftype`: (id, name). Removed ids (4-6, 33-35) and the
/// `LLAMA_FTYPE_GUESSED` sentinel (1024, "not specified in the model file") are absent:
/// neither is a scheme a file can declare.
pub const LLAMA_FTYPES: &[(u32, &str)] = &[
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (7, "Q8_0"),
    (8, "Q5_0"),
    (9, "Q5_1"),
    (10, "Q2_K"),
    (11, "Q3_K_S"),
    (12, "Q3_K_M"),
    (13, "Q3_K_L"),
    (14, "Q4_K_S"),
    (15, "Q4_K_M"),
    (16, "Q5_K_S"),
    (17, "Q5_K_M"),
    (18, "Q6_K"),
    (19, "IQ2_XXS"),
    (20, "IQ2_XS"),
    (21, "Q2_K_S"),
    (22, "IQ3_XS"),
    (23, "IQ3_XXS"),
    (24, "IQ1_S"),
    (25, "IQ4_NL"),
    (26, "IQ3_S"),
    (27, "IQ3_M"),
    (28, "IQ2_S"),
    (29, "IQ2_M"),
    (30, "IQ4_XS"),
    (31, "IQ1_M"),
    (32, "BF16"),
    (36, "TQ1_0"),
    (37, "TQ2_0"),
    (38, "MXFP4_MOE"),
    (39, "NVFP4"),
    (40, "Q1_0"),
    (41, "Q2_0"),
];

/// The scheme named by a `general.file_type` value, or `None` for an id upstream removed
/// or never defined.
#[must_use]
pub fn llama_ftype_name(id: u32) -> Option<&'static str> {
    LLAMA_FTYPES
        .iter()
        .find(|&&(ftype, _)| ftype == id)
        .map(|&(_, name)| name)
}
