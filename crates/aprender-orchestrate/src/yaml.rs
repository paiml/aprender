//! The one YAML reader for batuta (#5205 S4).
//!
//! serde-saphyr replaces serde_yaml_ng 0.10 (which pulls unsafe-libyaml,
//! RUSTSEC-2024-0449). The options keep the old reader's semantics:
//! `yes`/`no` stay strings, `<<` stays an ordinary key, a long comment run is
//! not a budget refusal, and `.inf`/`.nan` load in typeless slots.

/// Parse `s` as YAML into `T` with the #5205 stage options.
///
/// # Errors
///
/// Returns the serde-saphyr error if `s` is not valid YAML or does not match `T`.
pub(crate) fn from_str<'de, T: serde::Deserialize<'de>>(
    s: &'de str,
) -> Result<T, serde_saphyr::Error> {
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
