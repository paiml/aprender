//! Adversarial Robustness Testing
//!
//! Implements FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent)
//! attacks for testing model robustness.
//!
//! # References
//! - Goodfellow et al. (2015) "Explaining and Harnessing Adversarial Examples"
//! - Carlini & Wagner (2017) "Towards Evaluating the Robustness of Neural Networks"

use super::{CategoryScore, QaCategory, QaIssue, Severity, TestResult};
use std::time::{Duration, Instant};

/// Adversarial attack configuration
#[derive(Debug, Clone)]
pub struct AdversarialConfig {
    /// FGSM epsilon (perturbation magnitude)
    pub fgsm_epsilon: f32,
    /// PGD number of steps
    pub pgd_steps: u32,
    /// PGD step size
    pub pgd_step_size: f32,
    /// PGD epsilon bound
    pub pgd_epsilon: f32,
    /// Gaussian noise standard deviation
    pub noise_sigma: f32,
    /// Maximum allowed accuracy drop
    pub max_accuracy_drop: f32,
}

impl Default for AdversarialConfig {
    fn default() -> Self {
        Self {
            fgsm_epsilon: 0.1,
            pgd_steps: 10,
            pgd_step_size: 0.01,
            pgd_epsilon: 0.03,
            noise_sigma: 0.05,
            max_accuracy_drop: 0.05, // 5% max drop
        }
    }
}

/// Result of an adversarial attack
#[derive(Debug, Clone)]
pub struct AttackResult {
    /// Attack name
    pub attack_name: String,
    /// Original accuracy
    pub original_accuracy: f32,
    /// Accuracy under attack
    pub attacked_accuracy: f32,
    /// Accuracy drop
    pub accuracy_drop: f32,
    /// Whether the model is robust (drop < threshold)
    pub is_robust: bool,
    /// Duration of the attack
    pub duration: Duration,
}

impl AttackResult {
    /// Create a new attack result
    #[must_use]
    pub fn new(
        attack_name: impl Into<String>,
        original_accuracy: f32,
        attacked_accuracy: f32,
        max_drop: f32,
        duration: Duration,
    ) -> Self {
        let accuracy_drop = (original_accuracy - attacked_accuracy).max(0.0);
        Self {
            attack_name: attack_name.into(),
            original_accuracy,
            attacked_accuracy,
            accuracy_drop,
            is_robust: accuracy_drop <= max_drop,
            duration,
        }
    }
}

/// FGSM attack implementation
///
/// Generates adversarial examples using the Fast Gradient Sign Method.
/// `x_adv` = x + epsilon * `sign(grad_x(loss))`
#[derive(Debug, Clone)]
pub struct FgsmAttack {
    /// Perturbation magnitude
    pub epsilon: f32,
}

impl FgsmAttack {
    /// Create a new FGSM attack
    #[must_use]
    pub const fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }

    /// Apply FGSM perturbation to input
    ///
    /// # Arguments
    /// * `input` - Original input vector
    /// * `gradient` - Gradient of loss w.r.t. input
    ///
    /// # Returns
    /// Adversarial example
    #[must_use]
    pub fn perturb(&self, input: &[f32], gradient: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(gradient.iter())
            .map(|(&x, &g)| {
                let sign = if g >= 0.0 { 1.0 } else { -1.0 };
                x + self.epsilon * sign
            })
            .collect()
    }

    /// Run FGSM attack on a batch of inputs
    ///
    /// # Arguments
    /// * `inputs` - Batch of input vectors
    /// * `gradients` - Gradients for each input
    ///
    /// # Returns
    /// Batch of adversarial examples
    #[must_use]
    pub fn attack_batch(&self, inputs: &[Vec<f32>], gradients: &[Vec<f32>]) -> Vec<Vec<f32>> {
        inputs
            .iter()
            .zip(gradients.iter())
            .map(|(input, grad)| self.perturb(input, grad))
            .collect()
    }
}

/// PGD attack implementation
///
/// Projected Gradient Descent attack with multiple iterations.
#[derive(Debug, Clone)]
pub struct PgdAttack {
    /// Number of iterations
    pub steps: u32,
    /// Step size per iteration
    pub step_size: f32,
    /// Maximum perturbation (L-infinity bound)
    pub epsilon: f32,
}

impl PgdAttack {
    /// Create a new PGD attack
    #[must_use]
    pub const fn new(steps: u32, step_size: f32, epsilon: f32) -> Self {
        Self {
            steps,
            step_size,
            epsilon,
        }
    }

    /// Apply single PGD step
    fn step(&self, current: &[f32], original: &[f32], gradient: &[f32]) -> Vec<f32> {
        current
            .iter()
            .zip(original.iter())
            .zip(gradient.iter())
            .map(|((&c, &o), &g)| {
                let sign = if g >= 0.0 { 1.0 } else { -1.0 };
                let new_val = c + self.step_size * sign;
                // Project back to epsilon-ball around original
                let delta = (new_val - o).clamp(-self.epsilon, self.epsilon);
                o + delta
            })
            .collect()
    }

    /// Run full PGD attack
    ///
    /// # Arguments
    /// * `input` - Original input vector
    /// * `gradient_fn` - Function that computes gradient at current point
    ///
    /// # Returns
    /// Adversarial example after all steps
    pub fn attack<F>(&self, input: &[f32], mut gradient_fn: F) -> Vec<f32>
    where
        F: FnMut(&[f32]) -> Vec<f32>,
    {
        let mut current = input.to_vec();

        for _ in 0..self.steps {
            let gradient = gradient_fn(&current);
            current = self.step(&current, input, &gradient);
        }

        current
    }
}

/// Gaussian noise attack for robustness testing
#[derive(Debug, Clone)]
pub struct GaussianNoiseAttack {
    /// Standard deviation of noise
    pub sigma: f32,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl GaussianNoiseAttack {
    /// Create a new Gaussian noise attack
    #[must_use]
    pub const fn new(sigma: f32, seed: u64) -> Self {
        Self { sigma, seed }
    }

    /// Add Gaussian noise to input
    ///
    /// Uses Box-Muller transform for Gaussian sampling
    #[must_use]
    pub fn perturb(&self, input: &[f32]) -> Vec<f32> {
        // Simple deterministic pseudo-random for reproducibility
        let mut state = self.seed;

        input
            .iter()
            .map(|&x| {
                // LCG random
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let u1 = ((state >> 33) as f32 / u32::MAX as f32).max(1e-10);
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let u2 = (state >> 33) as f32 / u32::MAX as f32;

                // Box-Muller transform
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                x + self.sigma * z
            })
            .collect()
    }
}

/// Run all adversarial robustness tests
///
/// # Returns
/// Tuple of (`CategoryScore`, `Vec<QaIssue>`)
#[must_use]
pub fn run_robustness_tests(config: &AdversarialConfig) -> (CategoryScore, Vec<QaIssue>) {
    let start = Instant::now();
    let mut score = CategoryScore::new(20); // 20 points for robustness
    let mut issues = Vec::new();

    // Test 1: FGSM attack
    let fgsm_result = test_fgsm_robustness(config);
    if fgsm_result.is_robust {
        score.add_result(TestResult::pass("FGSM robustness", fgsm_result.duration));
    } else {
        score.add_result(TestResult::fail(
            "FGSM robustness",
            format!(
                "Accuracy drop {:.1}% > {:.1}%",
                fgsm_result.accuracy_drop * 100.0,
                config.max_accuracy_drop * 100.0
            ),
            fgsm_result.duration,
        ));
        issues.push(QaIssue::new(
            QaCategory::Robustness,
            Severity::Warning,
            format!(
                "FGSM attack causes {:.1}% accuracy drop",
                fgsm_result.accuracy_drop * 100.0
            ),
            "Consider adversarial training or input preprocessing",
        ));
    }

    // Test 2: PGD attack
    let pgd_result = test_pgd_robustness(config);
    if pgd_result.is_robust {
        score.add_result(TestResult::pass("PGD robustness", pgd_result.duration));
    } else {
        score.add_result(TestResult::fail(
            "PGD robustness",
            format!(
                "Accuracy drop {:.1}% > {:.1}%",
                pgd_result.accuracy_drop * 100.0,
                config.max_accuracy_drop * 100.0
            ),
            pgd_result.duration,
        ));
        issues.push(QaIssue::new(
            QaCategory::Robustness,
            Severity::Critical,
            format!(
                "PGD attack causes {:.1}% accuracy drop",
                pgd_result.accuracy_drop * 100.0
            ),
            "PGD is a strong attack; consider certified defenses",
        ));
    }

    // Test 3: Gaussian noise
    let noise_result = test_noise_robustness(config);
    if noise_result.is_robust {
        score.add_result(TestResult::pass("Noise robustness", noise_result.duration));
    } else {
        score.add_result(TestResult::fail(
            "Noise robustness",
            format!(
                "Accuracy drop {:.1}% > {:.1}%",
                noise_result.accuracy_drop * 100.0,
                config.max_accuracy_drop * 100.0
            ),
            noise_result.duration,
        ));
    }

    score.finalize();

    let _elapsed = start.elapsed();
    (score, issues)
}

/// Test FGSM robustness (mock implementation)
fn test_fgsm_robustness(config: &AdversarialConfig) -> AttackResult {
    let start = Instant::now();
    // Mock: In real implementation, would run model on adversarial examples
    let original_acc = 0.95;
    let attacked_acc = 0.92; // 3% drop

    AttackResult::new(
        "FGSM",
        original_acc,
        attacked_acc,
        config.max_accuracy_drop,
        start.elapsed(),
    )
}

/// Test PGD robustness (mock implementation)
fn test_pgd_robustness(config: &AdversarialConfig) -> AttackResult {
    let start = Instant::now();
    let original_acc = 0.95;
    let attacked_acc = 0.88; // 7% drop - stronger attack

    AttackResult::new(
        "PGD",
        original_acc,
        attacked_acc,
        config.max_accuracy_drop,
        start.elapsed(),
    )
}

/// Test Gaussian noise robustness (mock implementation)
fn test_noise_robustness(config: &AdversarialConfig) -> AttackResult {
    let start = Instant::now();
    let original_acc = 0.95;
    let attacked_acc = 0.93; // 2% drop

    AttackResult::new(
        "GaussianNoise",
        original_acc,
        attacked_acc,
        config.max_accuracy_drop,
        start.elapsed(),
    )
}

#[cfg(test)]
#[path = "adversarial_tests.rs"]
mod tests;
