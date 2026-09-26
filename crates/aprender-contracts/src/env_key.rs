// `include!`d by `build_helper.rs` (which carries the equality test against the
// macro) and by this crate's own build.rs, which cannot build-depend on the
// crate it builds (#4369).
/// The env var key `#[contract(contract, equation = equation)]` reads.
///
/// A producer (a consuming crate's `build.rs`) must emit its
/// `cargo:rustc-env` vars under exactly this key, plus `_PRE_COUNT`,
/// `_PRE_<i>`, `_POST_COUNT` and `_POST_<i>`. Call this instead of
/// re-deriving the format: a hand-rolled copy that drifts makes every
/// condition land under a name the macro never reads (#2699 §4). It is
/// tested equal to the macro's own `contract_env_key!`.
pub fn env_key(contract: &str, equation: &str) -> String {
    let contract_part = contract.to_uppercase().replace(['-', '.'], "_");
    let equation_part = equation.to_uppercase().replace(['-', '.'], "_");
    format!("CONTRACT_{contract_part}_{equation_part}")
}
