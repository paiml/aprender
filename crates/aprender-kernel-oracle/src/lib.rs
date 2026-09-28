//! KTEST-02 (KTEST-001 §3.2): the f64 scalar oracle and the per-element margin harness.
//!
//! A kernel is judged element by element: mᵢ = |ŷᵢ − yᵢ| / Bᵢ, where yᵢ is the oracle's f64 value
//! and Bᵢ is the bound the kernel's declared error model gives for that element. The kernel passes
//! iff max mᵢ ≤ 1 (plus the NaN/Inf rules in [`margin`]). A global NMSE is reported next to it so
//! receipts stay comparable with llama.cpp's `test-backend-ops`, but NMSE never decides: one wrong
//! tile in a million outputs moves it by ~1e-11 (F-12, tested below).
//!
//! What a kernel is run on comes from [`shapes`] (§4: dimension classes derived from the
//! registered tile and vector width) and [`inputs`] (§3.3: typical and adversarial values, each
//! reproducible from its seed). F-1 and F-6 are planted in `falsifiers`.
//!
//! The oracle is independent by construction. This crate has no dependencies, so it cannot call
//! the optimized backend it judges (§0.3 "the producer is never the gate");
//! `oracle_has_no_dependencies` refuses a manifest that adds one.
//!
//! Quantization error is not the kernel's (§0.4): feed the oracle the dequantized weights and, for
//! a kernel that quantizes its activations, the round-tripped activations it actually consumed.

pub mod error_model;
#[cfg(test)]
mod falsifiers;
pub mod inputs;
pub mod margin;
pub mod oracle;
pub mod rng;
pub mod shapes;

pub use error_model::{Bound, Dtype, ErrorModel, Refusal};
pub use inputs::{generate, Input, InputClass};
pub use margin::{judge, screen, Failure, Report, Verdict};
pub use shapes::{Dim, ShapeClass};

#[cfg(test)]
mod tests {
    /// Every entry under a `[*dependencies*]` table of a Cargo manifest.
    fn dependency_entries(manifest: &str) -> Vec<String> {
        let mut in_deps = false;
        let mut entries = Vec::new();
        for line in manifest.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                in_deps = t.contains("dependencies");
                continue;
            }
            if in_deps && !t.is_empty() && !t.starts_with('#') {
                entries.push(t.to_string());
            }
        }
        entries
    }

    /// §0.3: the oracle must not be able to reach an optimized backend. Any `[dependencies]`
    /// entry (trueno, aprender-serve, …) would make that possible, so the manifest must list none.
    #[test]
    fn oracle_has_no_dependencies() {
        let entries = dependency_entries(include_str!("../Cargo.toml"));
        assert!(
            entries.is_empty(),
            "KTEST-02 §0.3: the oracle crate must stay dependency-free, found {entries:?}"
        );
    }

    /// Anti-vacuity: the check sees a planted dependency in every dependency table.
    #[test]
    fn dependency_check_sees_a_planted_dependency() {
        for table in [
            "[dependencies]",
            "[dev-dependencies]",
            "[build-dependencies]",
        ] {
            let planted = format!(
                "[package]\nname = \"x\"\n\n{table}\n# why\ntrueno = {{ workspace = true }}\n"
            );
            assert_eq!(
                dependency_entries(&planted),
                vec!["trueno = { workspace = true }"],
                "{table}"
            );
        }
        assert!(
            dependency_entries("[package]\nname = \"x\"\n[lints]\nworkspace = true\n").is_empty()
        );
    }
}
