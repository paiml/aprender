//! Reinforcement Learning on Verifiable Rewards (RLVR) (GH-450)
//!
//! Implements RLVR for training language models using verifiable reward signals
//! (math correctness, code execution, format compliance) instead of learned
//! reward models.
//!
//! Key insight: When rewards are verifiable (binary correct/incorrect), we can
//! skip reward model training entirely and use REINFORCE with verifiable
//! reward functions directly.
//!
//! # References
//!
//! - [Lambert et al. 2024] "RLVR: Reinforcement Learning from Verifiable Rewards"
//! - [Williams 1992] "Simple statistical gradient-following algorithms for
//!   connectionist reinforcement learning" (REINFORCE)
//!
//! # Toyota Way Principles
//!
//! - **Poka-Yoke**: Verifiable rewards eliminate reward model errors
//! - **Jidoka**: Binary correctness signals stop bad gradient updates automatically
//! - **Kaizen**: Continuous improvement through verifiable feedback loops

use crate::error::{AprenderError, Result};

/// Configuration for RLVR training.
#[derive(Debug, Clone)]
pub struct RlvrConfig {
    /// Learning rate for policy gradient updates.
    pub learning_rate: f64,
    /// KL penalty coefficient. Controls divergence from reference policy.
    /// Higher values keep the policy closer to the reference.
    pub kl_coeff: f64,
    /// Reward scaling factor. Multiplied with raw reward before gradient computation.
    pub reward_scale: f64,
    /// Maximum response length in tokens.
    pub max_response_len: usize,
    /// Number of samples to draw per prompt for reward estimation.
    pub num_samples: usize,
}

impl Default for RlvrConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            kl_coeff: 0.1,
            reward_scale: 1.0,
            max_response_len: 512,
            num_samples: 4,
        }
    }
}

impl RlvrConfig {
    /// Validate configuration constraints.
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 || !self.learning_rate.is_finite() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "learning_rate must be positive finite, got {}",
                    self.learning_rate
                ),
            });
        }
        if self.kl_coeff < 0.0 || !self.kl_coeff.is_finite() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "kl_coeff must be non-negative finite, got {}",
                    self.kl_coeff
                ),
            });
        }
        if self.reward_scale <= 0.0 || !self.reward_scale.is_finite() {
            return Err(AprenderError::FormatError {
                message: format!(
                    "reward_scale must be positive finite, got {}",
                    self.reward_scale
                ),
            });
        }
        if self.max_response_len == 0 {
            return Err(AprenderError::FormatError {
                message: "max_response_len must be > 0".to_string(),
            });
        }
        if self.num_samples == 0 {
            return Err(AprenderError::FormatError {
                message: "num_samples must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// Result of verifying a model response against a ground truth reward.
#[derive(Debug, Clone)]
pub struct RewardResult {
    /// Reward score in [0.0, 1.0] (0 = incorrect, 1 = fully correct).
    pub score: f64,
    /// Whether the response was deemed correct by the verifier.
    pub correct: bool,
    /// Optional explanation of the verification outcome.
    pub explanation: Option<String>,
}

/// Trait for verifiable reward functions.
///
/// Implementors provide deterministic, binary-verifiable reward signals
/// for model responses. Unlike learned reward models, verifiable rewards
/// have zero noise — the answer is either correct or it is not.
pub trait VerifiableReward {
    /// Verify a model response against the prompt and return a reward.
    ///
    /// # Arguments
    /// * `prompt` - The original prompt/question
    /// * `response` - The model's generated response
    ///
    /// # Returns
    /// A `RewardResult` with score, correctness, and optional explanation.
    fn verify(&self, prompt: &str, response: &str) -> RewardResult;
}

/// RLVR loss calculator using REINFORCE policy gradient.
///
/// Computes:
/// - Policy gradient: -mean(reward_i * log_prob_i)
/// - KL penalty: kl_coeff * mean(policy_logprob_i - ref_logprob_i)
/// - Total loss: policy_gradient + kl_coeff * kl_penalty
#[derive(Debug, Clone)]
pub struct RlvrLoss {
    config: RlvrConfig,
}

impl RlvrLoss {
    /// Create a new RLVR loss calculator.
    #[must_use]
    pub fn new(config: RlvrConfig) -> Self {
        Self { config }
    }

    /// Compute REINFORCE policy gradient loss.
    ///
    /// L_pg = -1/N * sum(reward_i * log_prob_i)
    ///
    /// Rewards are scaled by `config.reward_scale` before computation.
    ///
    /// # Arguments
    /// * `log_probs` - Per-sample log-probabilities under the current policy
    /// * `rewards` - Per-sample reward scores from the verifier
    #[must_use]
    pub fn compute_policy_gradient(&self, log_probs: &[f64], rewards: &[f64]) -> f64 {
        if log_probs.is_empty() || rewards.is_empty() {
            return 0.0;
        }
        let n = log_probs.len().min(rewards.len());
        let sum: f64 = log_probs[..n]
            .iter()
            .zip(&rewards[..n])
            .map(|(&lp, &r)| r * self.config.reward_scale * lp)
            .sum();
        -sum / n as f64
    }

    /// Compute mean KL divergence penalty between policy and reference.
    ///
    /// KL = 1/N * sum(policy_logprob_i - ref_logprob_i)
    ///
    /// This is the per-token KL approximation: KL(pi || pi_ref) ~ E_pi[log pi - log pi_ref].
    ///
    /// # Arguments
    /// * `policy_logprobs` - Log-probabilities under the current policy
    /// * `ref_logprobs` - Log-probabilities under the reference (frozen) policy
    #[must_use]
    pub fn compute_kl_penalty(&self, policy_logprobs: &[f64], ref_logprobs: &[f64]) -> f64 {
        if policy_logprobs.is_empty() || ref_logprobs.is_empty() {
            return 0.0;
        }
        let n = policy_logprobs.len().min(ref_logprobs.len());
        let sum: f64 = policy_logprobs[..n]
            .iter()
            .zip(&ref_logprobs[..n])
            .map(|(&p, &r)| p - r)
            .sum();
        sum / n as f64
    }

    /// Compute total RLVR loss: policy_gradient + kl_coeff * kl_penalty.
    ///
    /// # Arguments
    /// * `log_probs` - Per-sample log-probabilities under the current policy
    /// * `rewards` - Per-sample reward scores from the verifier
    /// * `ref_logprobs` - Log-probabilities under the reference policy
    #[must_use]
    pub fn compute_total_loss(
        &self,
        log_probs: &[f64],
        rewards: &[f64],
        ref_logprobs: &[f64],
    ) -> f64 {
        let pg = self.compute_policy_gradient(log_probs, rewards);
        let kl = self.compute_kl_penalty(log_probs, ref_logprobs);
        pg + self.config.kl_coeff * kl
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &RlvrConfig {
        &self.config
    }
}

/// Aggregate metrics for an RLVR training batch.
#[derive(Debug, Clone, Default)]
pub struct RlvrMetrics {
    /// Average reward across all samples.
    pub avg_reward: f64,
    /// Fraction of samples that were correct (reward > 0.5).
    pub accuracy: f64,
    /// Average KL divergence from reference policy.
    pub avg_kl: f64,
    /// Average total loss.
    pub avg_loss: f64,
    /// Number of samples in the batch.
    pub num_samples: usize,
}

impl RlvrMetrics {
    /// Compute metrics from a batch of RLVR results.
    ///
    /// # Arguments
    /// * `loss` - The RLVR loss calculator
    /// * `log_probs` - Per-sample log-probabilities under the current policy
    /// * `rewards` - Per-sample reward scores
    /// * `ref_logprobs` - Log-probabilities under the reference policy
    #[must_use]
    pub fn from_batch(
        loss: &RlvrLoss,
        log_probs: &[f64],
        rewards: &[f64],
        ref_logprobs: &[f64],
    ) -> Self {
        if log_probs.is_empty() || rewards.is_empty() || ref_logprobs.is_empty() {
            return Self::default();
        }

        let n = log_probs.len().min(rewards.len()).min(ref_logprobs.len());

        let avg_reward = rewards[..n].iter().sum::<f64>() / n as f64;
        let correct_count = rewards[..n].iter().filter(|&&r| r > 0.5).count();
        let accuracy = correct_count as f64 / n as f64;
        let avg_kl = loss.compute_kl_penalty(&log_probs[..n], &ref_logprobs[..n]);
        let avg_loss = loss.compute_total_loss(&log_probs[..n], &rewards[..n], &ref_logprobs[..n]);

        Self {
            avg_reward,
            accuracy,
            avg_kl,
            avg_loss,
            num_samples: n,
        }
    }
}

/// Verifiable reward for math problems.
///
/// Checks whether the response contains the expected numeric answer.
/// Supports simple pattern matching against integer and decimal answers
/// extracted from the response text.
#[derive(Debug, Clone)]
pub struct MathReward;

impl MathReward {
    /// Extract the last numeric value from a response string.
    ///
    /// Scans for patterns like "= 42", "answer is 3.14", or boxed answers "\\boxed{7}".
    fn extract_answer(response: &str) -> Option<f64> {
        // Try boxed answer first (common in math formatting)
        if let Some(start) = response.find("\\boxed{") {
            let rest = &response[start + 7..];
            if let Some(end) = rest.find('}') {
                if let Ok(val) = rest[..end].trim().parse::<f64>() {
                    return Some(val);
                }
            }
        }

        // Try "answer is <number>" pattern
        let lower = response.to_lowercase();
        if let Some(idx) = lower.rfind("answer is ") {
            let rest = &response[idx + 10..];
            let num_str: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            if let Ok(val) = num_str.parse::<f64>() {
                return Some(val);
            }
        }

        // Try "= <number>" pattern (last occurrence)
        if let Some(idx) = response.rfind('=') {
            let rest = response[idx + 1..].trim();
            let num_str: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                .collect();
            if let Ok(val) = num_str.parse::<f64>() {
                return Some(val);
            }
        }

        None
    }

    /// Extract the expected answer from a prompt.
    ///
    /// Looks for "expected: <number>" or "answer: <number>" patterns.
    fn extract_expected(prompt: &str) -> Option<f64> {
        let lower = prompt.to_lowercase();
        for prefix in &["expected: ", "answer: ", "expected answer: "] {
            if let Some(idx) = lower.find(prefix) {
                let rest = &prompt[idx + prefix.len()..];
                let num_str: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                    .collect();
                if let Ok(val) = num_str.parse::<f64>() {
                    return Some(val);
                }
            }
        }
        None
    }
}

impl VerifiableReward for MathReward {
    fn verify(&self, prompt: &str, response: &str) -> RewardResult {
        let expected = match Self::extract_expected(prompt) {
            Some(v) => v,
            None => {
                return RewardResult {
                    score: 0.0,
                    correct: false,
                    explanation: Some("No expected answer found in prompt".to_string()),
                };
            }
        };

        match Self::extract_answer(response) {
            Some(actual) => {
                let correct = (actual - expected).abs() < 1e-6;
                RewardResult {
                    score: if correct { 1.0 } else { 0.0 },
                    correct,
                    explanation: Some(format!("Expected {expected}, got {actual}")),
                }
            }
            None => RewardResult {
                score: 0.0,
                correct: false,
                explanation: Some("No numeric answer found in response".to_string()),
            },
        }
    }
}

/// Verifiable reward for code generation.
///
/// Checks whether the generated code contains expected patterns such as
/// function definitions, return statements, or specific keywords.
#[derive(Debug, Clone)]
pub struct CodeReward;

impl CodeReward {
    /// Extract expected patterns from a prompt.
    ///
    /// Looks for "must contain: pattern1, pattern2" or "expected output: text".
    fn extract_requirements(prompt: &str) -> Vec<String> {
        let lower = prompt.to_lowercase();
        let mut requirements = Vec::new();

        // "must contain: x, y, z" pattern
        if let Some(idx) = lower.find("must contain: ") {
            let rest = &prompt[idx + 14..];
            let end = rest.find('\n').unwrap_or(rest.len());
            for part in rest[..end].split(',') {
                let trimmed = part.trim().to_string();
                if !trimmed.is_empty() {
                    requirements.push(trimmed);
                }
            }
        }

        // "expected output: text" pattern
        if let Some(idx) = lower.find("expected output: ") {
            let rest = &prompt[idx + 17..];
            let end = rest.find('\n').unwrap_or(rest.len());
            let trimmed = rest[..end].trim().to_string();
            if !trimmed.is_empty() {
                requirements.push(trimmed);
            }
        }

        requirements
    }
}

impl VerifiableReward for CodeReward {
    fn verify(&self, prompt: &str, response: &str) -> RewardResult {
        let requirements = Self::extract_requirements(prompt);

        if requirements.is_empty() {
            return RewardResult {
                score: 0.0,
                correct: false,
                explanation: Some("No requirements found in prompt".to_string()),
            };
        }

        let matched: Vec<&String> = requirements
            .iter()
            .filter(|req| response.contains(req.as_str()))
            .collect();

        let score = matched.len() as f64 / requirements.len() as f64;
        let correct = (score - 1.0).abs() < f64::EPSILON;

        let explanation = if correct {
            format!("All {} requirements satisfied", requirements.len())
        } else {
            let missing: Vec<&String> = requirements
                .iter()
                .filter(|req| !response.contains(req.as_str()))
                .collect();
            format!(
                "Missing {}/{} requirements: {:?}",
                missing.len(),
                requirements.len(),
                missing
            )
        };

        RewardResult {
            score,
            correct,
            explanation: Some(explanation),
        }
    }
}

#[cfg(test)]
#[path = "rlvr_tests.rs"]
mod tests;
