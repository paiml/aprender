//! The one YAML reader for pv (#5209 S4).
//!
//! serde-saphyr replaces serde_yaml (unmaintained, RUSTSEC-2024-0449). The options keep the
//! old reader's semantics: `yes`/`no` stay strings, a long comment run is not a budget
//! refusal, and `.inf`/`.nan` load in typeless slots.

/// Parse `s` as YAML into `T`, `<<` left as an ordinary key.
///
/// # Errors
///
/// Returns the serde-saphyr error if `s` is not valid YAML or does not match `T`.
pub fn from_str<'de, T: serde::Deserialize<'de>>(s: &'de str) -> Result<T, serde_saphyr::Error> {
    serde_saphyr::from_str_with_options(
        s,
        serde_saphyr::options! {
            strict_booleans: true,
            merge_keys: serde_saphyr::MergeKeyPolicy::AsOrdinary,
            emit_comments: false,
            reject_non_finite_typeless_float: false,
        },
    )
}

/// Parse `s` as YAML into `T`, `<<` merge keys expanded as PyYAML expands them.
///
/// # Errors
///
/// Returns the serde-saphyr error if `s` is not valid YAML or does not match `T`.
pub fn from_str_merging<'de, T: serde::Deserialize<'de>>(
    s: &'de str,
) -> Result<T, serde_saphyr::Error> {
    serde_saphyr::from_str_with_options(
        s,
        serde_saphyr::options! {
            strict_booleans: true,
            merge_keys: serde_saphyr::MergeKeyPolicy::Merge,
            emit_comments: false,
            reject_non_finite_typeless_float: false,
        },
    )
}
