//! The `CONTRACT_*` env-var key builder (#4369, #5909). Dependency-free on purpose.

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
