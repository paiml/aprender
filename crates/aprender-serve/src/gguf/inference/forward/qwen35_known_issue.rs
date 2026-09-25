//! 0.69.1 load-time warning for Qwen3.5-0.8B (#4032, known issue #4030).
//!
//! Operator ruling 2026-09-23 (operator selection "your rec", wording by the cop): 0.69.1
//! ships Qwen3.5-0.8B as a known RED and WARNS at load, never refuses. The model is
//! recognised from GGUF metadata (architecture + parameter count summed from the tensor
//! table), never from the filename. The general mechanism (CRUX status on every load) is
//! #4031.

use crate::gguf::GGUFModel;

/// The warning text, verbatim from #4032.
pub const QWEN35_0_8B_WARNING: &str = "warning: Qwen3.5-0.8B is not CRUX-certified. Known issue #4030: with thinking on it may\nreason past the token budget without answering; use thinking off or a certified model\n(Qwen3.5-2B/4B).";

/// The 0.8B class, in parameters. Measured from the GGUF tensor tables on lambda: every
/// Qwen3.5-0.8B quant counts 0.752B, the nearest neighbour (2B) counts 1.882B.
const QWEN35_0_8B_PARAMS: std::ops::Range<u64> = 600_000_000..1_000_000_000;

/// Total parameter count from the tensor table: the product of each tensor's dims, summed.
#[must_use]
pub fn gguf_param_count(model: &GGUFModel) -> u64 {
    model
        .tensors
        .iter()
        .map(|t| t.dims.iter().product::<u64>())
        .sum()
}

/// The warning for this model, if it is Qwen3.5-0.8B: dense `qwen35` in the 0.8B class.
#[must_use]
pub fn known_issue_warning(architecture: Option<&str>, param_count: u64) -> Option<&'static str> {
    (architecture == Some("qwen35") && QWEN35_0_8B_PARAMS.contains(&param_count))
        .then_some(QWEN35_0_8B_WARNING)
}

/// Print the warning to stderr at most once per process, if `model` is Qwen3.5-0.8B.
pub fn warn_if_known_issue(model: &GGUFModel) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if let Some(w) = known_issue_warning(model.architecture(), gguf_param_count(model)) {
        ONCE.call_once(|| eprintln!("{w}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Param counts measured from the real GGUF tensor tables (gguf-py, lambda, 2026-09-23).
    const P_0_8B: u64 = 752_393_024;
    const P_2B: u64 = 1_881_825_088;
    const P_4B: u64 = 4_205_751_296;

    #[test]
    fn qwen35_0_8b_must_warn() {
        assert_eq!(
            known_issue_warning(Some("qwen35"), P_0_8B),
            Some(QWEN35_0_8B_WARNING)
        );
    }

    #[test]
    fn qwen35_2b_and_4b_must_not_warn() {
        assert_eq!(known_issue_warning(Some("qwen35"), P_2B), None);
        assert_eq!(known_issue_warning(Some("qwen35"), P_4B), None);
    }

    #[test]
    fn another_architecture_of_the_same_size_must_not_warn() {
        assert_eq!(known_issue_warning(Some("qwen2"), P_0_8B), None);
        assert_eq!(known_issue_warning(Some("qwen35moe"), P_0_8B), None);
        assert_eq!(known_issue_warning(None, P_0_8B), None);
    }

    #[test]
    fn the_warning_is_the_ticket_text() {
        assert!(QWEN35_0_8B_WARNING.starts_with("warning: Qwen3.5-0.8B is not CRUX-certified."));
        assert!(QWEN35_0_8B_WARNING.contains("Known issue #4030"));
        assert!(QWEN35_0_8B_WARNING.ends_with("(Qwen3.5-2B/4B)."));
    }
}
