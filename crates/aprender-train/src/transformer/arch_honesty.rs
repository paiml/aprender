//! T1-R1 (#4552, contract `train-arch-honesty-v1`): the training crate refuses
//! an architecture it does not model, by name, before any weight is read.
//!
//! Before this, `qwen3_5` / `qwen3.5` were aliased to the dense `qwen3` family
//! and the 3.5 presets built a dense `Decoder` with no Gated DeltaNet (GDN)
//! layers. `apr finetune` / `apr distill` / `apr pretrain --init` on a Qwen3.5
//! model therefore trained a different model than `apr serve` runs, silently.
//! A named refusal is the honest interim state until GDN forward and backward
//! land in this crate (#4000 T1); R2/R3 remove the refusal, not this check.

use std::fmt;

/// What the training crate is missing to model a given architecture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedArch {
    /// The architecture string as the caller supplied it.
    pub arch: String,
    /// The layer kind aprender-train has no forward/backward for.
    pub missing: &'static str,
}

impl fmt::Display for UnsupportedArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "UnsupportedArch: aprender-train does not model architecture '{}' — it has no {}. \
             Training it as a dense transformer would train a different model than `apr serve` \
             runs, so it is refused (train-arch-honesty-v1, #4552; GDN training is #4000 T1).",
            self.arch, self.missing
        )
    }
}

impl std::error::Error for UnsupportedArch {}

const GDN: &str = "Gated DeltaNet (linear-attention) forward/backward";

/// The layer kind this crate lacks for `arch`, or `None` if it is modelled.
///
/// Matches every spelling the field takes — family slug (`qwen3_5`, `qwen3.5`,
/// `qwen35`), HF class (`Qwen3_5ForCausalLM`, `Qwen3NextForCausalLM`) and size
/// preset (`qwen3.5-9b`) — by comparing with case and `_ . -` removed.
#[must_use]
pub fn unmodelled_layer(arch: &str) -> Option<&'static str> {
    let norm: String = arch
        .chars()
        .filter(|c| !matches!(c, '_' | '.' | '-'))
        .flat_map(char::to_lowercase)
        .collect();
    if norm.starts_with("qwen35") || norm.starts_with("qwen3next") {
        return Some(GDN);
    }
    None
}

/// `Err(UnsupportedArch)` if aprender-train does not model `arch`.
///
/// # Errors
/// Returns the named refusal for a hybrid/GDN architecture.
pub fn check_trainable_arch(arch: &str) -> Result<(), UnsupportedArch> {
    match unmodelled_layer(arch) {
        Some(missing) => Err(UnsupportedArch { arch: arch.to_string(), missing }),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-TAH-001 case table: every spelling of a hybrid is refused, and
    /// no dense family that neighbours it by prefix is.
    #[test]
    fn falsify_tah_001_hybrid_spellings_refused_dense_neighbours_pass() {
        for arch in [
            "qwen3_5",
            "qwen3.5",
            "qwen35",
            "Qwen3_5ForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForCausalLM",
            "qwen3.5-9b",
            "Qwen3.5-4B",
            "Qwen3NextForCausalLM",
            "qwen3_next",
        ] {
            let err = check_trainable_arch(arch).expect_err(arch);
            assert_eq!(err.arch, arch);
            assert!(err.to_string().contains("Gated DeltaNet"), "{arch}: {err}");
        }
        for arch in [
            "qwen3",
            "Qwen3ForCausalLM",
            "Qwen3MoeForCausalLM",
            "qwen2",
            "qwen2.5",
            "Qwen2ForCausalLM",
            "llama",
            "mistral",
            "unknown",
            "",
        ] {
            assert!(check_trainable_arch(arch).is_ok(), "{arch} must not be refused");
        }
    }
}
