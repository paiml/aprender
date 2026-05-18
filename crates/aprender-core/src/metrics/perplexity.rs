//! Perplexity classifier (CRUX-E-02).
//!
//! Pure-function perplexity computation over per-token natural-log probabilities.
//! Mirrors llama.cpp `examples/perplexity` convention: `PPL = exp(-mean(log p))`.
//!
//! Contract: `contracts/crux-E-02-v1.yaml`.
//!
//! This module deliberately takes pre-computed log-probs as input rather than
//! doing inference itself — the classifier is the pure math half of the
//! perplexity pipeline. The live-model half (computing per-token log-probs
//! from a real GGUF/APR and held-out corpus) is dispatched elsewhere and
//! discharged as PARTIAL_ALGORITHM_LEVEL under BLOCKER-UPSTREAM-MISSING
//! until a stable log-probs extraction path lands.

/// Outcome of `compute_perplexity` — no silent pass on ill-formed input.
#[derive(Debug, Clone, PartialEq)]
pub enum PerplexityOutcome {
    /// Perplexity, mean NLL in nats, and token count.
    Ok {
        ppl: f64,
        mean_nll: f64,
        num_tokens: usize,
    },
    /// Input slice was empty — cannot compute mean NLL.
    EmptyLogProbs,
    /// A log-prob was NaN or ±∞.
    NonFiniteLogProb,
    /// A log-prob was strictly positive — probability > 1 is impossible.
    PositiveLogProb(f64),
}

/// Computes perplexity from per-token natural-log probabilities.
///
/// `PPL = exp(-mean(log p_i))` where `log p_i` are natural-log probabilities
/// of the observed next tokens under the model. Each `log p_i` must be in
/// the closed interval `(-∞, 0]` — strictly positive values indicate a bug
/// (probability > 1).
///
/// Invariants (see contract `crux-E-02-v1` §equations):
/// - `PPL >= 1.0` for any non-empty valid input (mean NLL is non-negative).
/// - Empty → `EmptyLogProbs` (distinct variant).
/// - NaN/±∞ log-prob → `NonFiniteLogProb`.
/// - Strictly positive log-prob → `PositiveLogProb(value)`.
pub fn compute_perplexity(log_probs: &[f64]) -> PerplexityOutcome {
    if log_probs.is_empty() {
        return PerplexityOutcome::EmptyLogProbs;
    }

    for &lp in log_probs {
        if !lp.is_finite() {
            return PerplexityOutcome::NonFiniteLogProb;
        }
        if lp > 0.0 {
            return PerplexityOutcome::PositiveLogProb(lp);
        }
    }

    let n = log_probs.len();
    let sum: f64 = log_probs.iter().sum();
    let mean_nll = -sum / (n as f64);
    let ppl = mean_nll.exp();

    PerplexityOutcome::Ok {
        ppl,
        mean_nll,
        num_tokens: n,
    }
}

/// Classifier for FALSIFY-CRUX-E-02-004: PPL is bounded below by 1.0
/// and is finite for any valid input.
pub fn classify_ppl_at_least_one(log_probs: &[f64]) -> bool {
    match compute_perplexity(log_probs) {
        PerplexityOutcome::Ok { ppl, .. } => ppl >= 1.0 && ppl.is_finite(),
        _ => false,
    }
}

/// Classifier for no-silent-pass: empty input produces a distinct variant.
pub fn classify_empty_distinct() -> bool {
    matches!(compute_perplexity(&[]), PerplexityOutcome::EmptyLogProbs)
}

/// Classifier for no-silent-pass: NaN log-prob is rejected.
pub fn classify_nan_rejected() -> bool {
    matches!(
        compute_perplexity(&[-1.0, f64::NAN, -2.0]),
        PerplexityOutcome::NonFiniteLogProb
    )
}

/// Classifier for no-silent-pass: +∞ log-prob is rejected.
pub fn classify_inf_rejected() -> bool {
    matches!(
        compute_perplexity(&[-1.0, f64::INFINITY, -2.0]),
        PerplexityOutcome::NonFiniteLogProb
    )
}

/// Classifier for no-silent-pass: positive log-prob (prob > 1) is rejected.
pub fn classify_positive_log_prob_rejected() -> bool {
    matches!(
        compute_perplexity(&[-1.0, 0.5, -2.0]),
        PerplexityOutcome::PositiveLogProb(_)
    )
}

/// Classifier: perfect prediction (log p = 0 for all tokens) ⇒ PPL = 1.0.
pub fn classify_perfect_prediction_is_one() -> bool {
    matches!(
        compute_perplexity(&[0.0, 0.0, 0.0, 0.0]),
        PerplexityOutcome::Ok { ppl, .. } if (ppl - 1.0).abs() < 1e-12
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log_probs_distinct_outcome() {
        assert!(matches!(
            compute_perplexity(&[]),
            PerplexityOutcome::EmptyLogProbs
        ));
    }

    #[test]
    fn perfect_prediction_ppl_is_one() {
        match compute_perplexity(&[0.0, 0.0, 0.0]) {
            PerplexityOutcome::Ok {
                ppl,
                mean_nll,
                num_tokens,
            } => {
                assert!((ppl - 1.0).abs() < 1e-12, "ppl={ppl} expected 1.0");
                assert!(mean_nll.abs() < 1e-12);
                assert_eq!(num_tokens, 3);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn uniform_log_prob_gives_correct_ppl() {
        // If every token has log p = -ln(2) (probability 0.5), then
        // mean NLL = ln(2), and PPL = exp(ln(2)) = 2.
        let ln_half = -std::f64::consts::LN_2;
        let samples = vec![ln_half; 16];
        match compute_perplexity(&samples) {
            PerplexityOutcome::Ok {
                ppl,
                mean_nll,
                num_tokens,
            } => {
                assert!((ppl - 2.0).abs() < 1e-12, "ppl={ppl}");
                assert!((mean_nll - std::f64::consts::LN_2).abs() < 1e-12);
                assert_eq!(num_tokens, 16);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn ppl_at_least_one_invariant_holds() {
        let samples = [-1.0_f64, -0.5, -2.3, -0.01];
        assert!(classify_ppl_at_least_one(&samples));
        if let PerplexityOutcome::Ok { ppl, .. } = compute_perplexity(&samples) {
            assert!(ppl >= 1.0, "ppl={ppl} must be >= 1.0");
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn nan_log_prob_rejected() {
        assert!(matches!(
            compute_perplexity(&[-1.0, f64::NAN]),
            PerplexityOutcome::NonFiniteLogProb
        ));
    }

    #[test]
    fn positive_infinity_rejected() {
        assert!(matches!(
            compute_perplexity(&[-1.0, f64::INFINITY]),
            PerplexityOutcome::NonFiniteLogProb
        ));
    }

    #[test]
    fn negative_infinity_rejected() {
        // -inf log p means prob=0 at a seen token — model is infinitely
        // surprised. We reject rather than silently returning +inf PPL.
        assert!(matches!(
            compute_perplexity(&[-1.0, f64::NEG_INFINITY]),
            PerplexityOutcome::NonFiniteLogProb
        ));
    }

    #[test]
    fn positive_log_prob_rejected() {
        match compute_perplexity(&[-1.0, 0.25, -2.0]) {
            PerplexityOutcome::PositiveLogProb(v) => assert!((v - 0.25).abs() < 1e-12),
            other => panic!("expected PositiveLogProb, got {other:?}"),
        }
    }

    #[test]
    fn single_log_prob_works() {
        match compute_perplexity(&[-1.0]) {
            PerplexityOutcome::Ok {
                ppl,
                mean_nll,
                num_tokens,
            } => {
                assert!((ppl - std::f64::consts::E).abs() < 1e-12);
                assert!((mean_nll - 1.0).abs() < 1e-12);
                assert_eq!(num_tokens, 1);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn ppl_monotone_in_mean_nll() {
        // Higher mean NLL ⇒ higher PPL.
        let a = [-0.5_f64, -0.5, -0.5];
        let b = [-1.0_f64, -1.0, -1.0];
        let ppl_a = match compute_perplexity(&a) {
            PerplexityOutcome::Ok { ppl, .. } => ppl,
            _ => panic!(),
        };
        let ppl_b = match compute_perplexity(&b) {
            PerplexityOutcome::Ok { ppl, .. } => ppl,
            _ => panic!(),
        };
        assert!(ppl_a < ppl_b, "ppl({ppl_a}) should be < ppl({ppl_b})");
    }

    #[test]
    fn classifier_functions_all_pass() {
        assert!(classify_empty_distinct());
        assert!(classify_nan_rejected());
        assert!(classify_inf_rejected());
        assert!(classify_positive_log_prob_rejected());
        assert!(classify_perfect_prediction_is_one());
    }

    #[test]
    fn num_tokens_matches_input_length() {
        for n in [1usize, 5, 100, 1000] {
            let samples = vec![-0.7_f64; n];
            match compute_perplexity(&samples) {
                PerplexityOutcome::Ok { num_tokens, .. } => assert_eq!(num_tokens, n),
                other => panic!("n={n}: expected Ok, got {other:?}"),
            }
        }
    }

    #[test]
    fn known_wikitext_ballpark_ppl() {
        // A realistic mean NLL for a decent LM on WikiText is ~1.8 (natural log),
        // yielding PPL ≈ 6.05. Smoke-check the formula bends that way.
        let mean_nll = 1.8_f64;
        let log_probs = vec![-mean_nll; 256];
        match compute_perplexity(&log_probs) {
            PerplexityOutcome::Ok { ppl, .. } => {
                assert!((ppl - mean_nll.exp()).abs() < 1e-9);
                assert!((5.5..=7.5).contains(&ppl), "ppl={ppl}");
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }
}
