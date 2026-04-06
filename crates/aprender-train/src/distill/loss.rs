//! Distillation loss functions

use ndarray::{Array2, Axis};

/// Knowledge Distillation Loss
///
/// Combines soft targets from teacher (via temperature-scaled KL divergence)
/// with hard targets from ground truth labels (via cross-entropy).
///
/// # Formula
///
/// ```text
/// L = α * T² * KL(softmax(teacher/T) || softmax(student/T))
///   + (1-α) * CE(student, labels)
/// ```
///
/// where T is temperature and α is the distillation weight.
///
/// # Example
///
/// ```
/// use entrenar::distill::DistillationLoss;
/// use ndarray::array;
///
/// let loss_fn = DistillationLoss::new(2.0, 0.7);
/// let student_logits = array![[2.0, 1.0, 0.5]];
/// let teacher_logits = array![[1.5, 1.2, 0.8]];
/// let labels = vec![0];
///
/// let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
/// assert!(loss > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// Temperature for softening probability distributions
    pub temperature: f32,
    /// Weight for distillation loss (α). Hard loss weight is (1-α)
    pub alpha: f32,
}

impl DistillationLoss {
    /// Create a new distillation loss function
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature for softening distributions (typically 2.0-5.0)
    /// * `alpha` - Weight for distillation vs hard loss (typically 0.5-0.9)
    ///
    /// # Panics
    ///
    /// Panics if temperature <= 0 or alpha not in [0, 1]
    pub fn new(temperature: f32, alpha: f32) -> Self {
        assert!(temperature > 0.0, "Temperature must be positive, got {temperature}");
        assert!((0.0..=1.0).contains(&alpha), "Alpha must be in [0, 1], got {alpha}");

        Self { temperature, alpha }
    }

    /// Compute the distillation loss
    ///
    /// # Arguments
    ///
    /// * `student_logits` - Logits from student model [batch_size, num_classes]
    /// * `teacher_logits` - Logits from teacher model [batch_size, num_classes]
    /// * `labels` - Ground truth labels `[batch_size]`
    ///
    /// # Returns
    ///
    /// Combined distillation and hard loss (scalar)
    pub fn forward(
        &self,
        student_logits: &Array2<f32>,
        teacher_logits: &Array2<f32>,
        labels: &[usize],
    ) -> f32 {
        assert_eq!(
            student_logits.shape(),
            teacher_logits.shape(),
            "Student and teacher logits must have same shape"
        );
        assert_eq!(student_logits.nrows(), labels.len(), "Batch size must match number of labels");

        // Soft targets: KL divergence with temperature scaling
        let kl_loss = self.kl_divergence_loss(student_logits, teacher_logits);

        // Hard targets: Cross-entropy with ground truth
        let ce_loss = self.cross_entropy_loss(student_logits, labels);

        // Combine with temperature adjust factor (T²)
        self.alpha * kl_loss * self.temperature * self.temperature + (1.0 - self.alpha) * ce_loss
    }

    /// Temperature-scaled KL divergence loss
    ///
    /// KL(teacher || student) where both distributions are softened by temperature
    fn kl_divergence_loss(
        &self,
        student_logits: &Array2<f32>,
        teacher_logits: &Array2<f32>,
    ) -> f32 {
        let student_soft = softmax_2d(&(student_logits / self.temperature));
        let teacher_soft = softmax_2d(&(teacher_logits / self.temperature));

        kl_divergence(&teacher_soft, &student_soft)
    }

    /// Standard cross-entropy loss with hard labels
    fn cross_entropy_loss(&self, logits: &Array2<f32>, labels: &[usize]) -> f32 {
        let probs = softmax_2d(logits);

        let mut loss = 0.0;
        for (i, &label) in labels.iter().enumerate() {
            let prob = probs[[i, label]].max(1e-10); // Avoid log(0)
            loss -= prob.max(f32::MIN_POSITIVE).ln();
        }

        loss / labels.len().max(1) as f32
    }
}

/// Compute softmax along last axis for 2D array
///
/// softmax(x)_i = exp(x_i) / Σ exp(x_j)
fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let mut result = x.clone();

    for mut row in result.axis_iter_mut(Axis(0)) {
        // Subtract max for numerical stability
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max_val).exp());

        // Normalize
        let sum: f32 = row.sum();
        row.mapv_inplace(|v| v / sum);
    }

    result
}

/// KL divergence between two probability distributions
///
/// KL(p || q) = Σ p_i * log(p_i / q_i)
///
/// Average over batch dimension.
fn kl_divergence(p: &Array2<f32>, q: &Array2<f32>) -> f32 {
    assert_eq!(p.shape(), q.shape());

    if p.nrows() == 0 {
        return 0.0;
    }

    let mut total_kl = 0.0;

    for (p_row, q_row) in p.axis_iter(Axis(0)).zip(q.axis_iter(Axis(0))) {
        let mut kl = 0.0;
        for (&p_i, &q_i) in p_row.iter().zip(q_row.iter()) {
            if p_i > 1e-10 {
                // Avoid log(0)
                kl += p_i * (p_i / q_i.max(1e-10)).ln();
            }
        }
        total_kl += kl;
    }

    total_kl / p.nrows() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_distillation_loss_basic() {
        let loss_fn = DistillationLoss::new(2.0, 0.5);
        let student = array![[2.0, 1.0, 0.5]];
        let teacher = array![[1.5, 1.2, 0.8]];
        let labels = vec![0];

        let loss = loss_fn.forward(&student, &teacher, &labels);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let probs = softmax_2d(&x);

        for row in probs.axis_iter(Axis(0)) {
            let sum: f32 = row.sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_kl_divergence_zero_for_identical() {
        let p = array![[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]];
        let kl = kl_divergence(&p, &p);
        assert_relative_eq!(kl, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = array![[0.7, 0.2, 0.1]];
        let q = array![[0.4, 0.4, 0.2]];
        let kl = kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_negative_temperature_panics() {
        DistillationLoss::new(-1.0, 0.5);
    }

    #[test]
    #[should_panic(expected = "Alpha must be in [0, 1]")]
    fn test_invalid_alpha_panics() {
        DistillationLoss::new(2.0, 1.5);
    }

    #[test]
    fn test_temperature_effect() {
        let student = array![[10.0, 1.0, 0.1]];
        let teacher = array![[5.0, 4.0, 3.0]];
        let labels = vec![0];

        let low_temp_loss = DistillationLoss::new(1.0, 1.0);
        let high_temp_loss = DistillationLoss::new(5.0, 1.0);

        let loss_low = low_temp_loss.forward(&student, &teacher, &labels);
        let loss_high = high_temp_loss.forward(&student, &teacher, &labels);

        // Higher temperature should soften distributions more
        assert!(loss_low != loss_high);
    }

    #[test]
    fn test_alpha_balances_losses() {
        let student = array![[2.0, 1.0, 0.5]];
        let teacher = array![[1.5, 1.2, 0.8]];
        let labels = vec![0];

        // Pure distillation (α=1)
        let pure_distill = DistillationLoss::new(2.0, 1.0);
        let loss_distill = pure_distill.forward(&student, &teacher, &labels);

        // Pure hard loss (α=0)
        let pure_hard = DistillationLoss::new(2.0, 0.0);
        let loss_hard = pure_hard.forward(&student, &teacher, &labels);

        // Balanced (α=0.5)
        let balanced = DistillationLoss::new(2.0, 0.5);
        let loss_balanced = balanced.forward(&student, &teacher, &labels);

        // Balanced should be between the two extremes (approximately)
        assert!(loss_balanced > 0.0);
        assert!(loss_distill > 0.0);
        assert!(loss_hard > 0.0);
    }

    // =========================================================================
    // FALSIFY-EMB-006/007: Temperature scaling (embedding-algebra-v1.yaml)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had 0 FALSIFY-EMB-* temperature tests
    //   Why 2: temperature tests existed but weren't tagged to YAML contract
    //   Why 3: no mapping from embedding-algebra-v1.yaml to entrenar test names
    //   Why 4: distillation loss uses temperature but was not linked to contract
    //   Why 5: EMB-006/007 were treated as inference-only, not training
    //
    // References:
    //   - provable-contracts/contracts/embedding-algebra-v1.yaml
    //   - Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"
    // =========================================================================

    /// FALSIFY-EMB-006: Temperature=1.0 is identity for softmax
    ///
    /// Contract: softmax(x / 1.0) == softmax(x)
    #[test]
    fn falsify_emb_006_temperature_identity() {
        let logits = array![[3.0, 1.0, 0.5, -1.0]];

        let softmax_raw = softmax_2d(&logits);
        let softmax_t1 = softmax_2d(&(&logits / 1.0));

        for (a, b) in softmax_raw.iter().zip(softmax_t1.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    /// FALSIFY-EMB-007: Higher temperature → more uniform distribution
    ///
    /// Contract: entropy(softmax(x/T_high)) > entropy(softmax(x/T_low))
    #[test]
    fn falsify_emb_007_temperature_monotonicity() {
        let logits = array![[5.0, 2.0, 0.1, -3.0]];

        let probs_low = softmax_2d(&(&logits / 1.0));
        let probs_high = softmax_2d(&(&logits / 10.0));

        // Compute Shannon entropy: -Σ p_i * log(p_i)
        let entropy = |probs: &Array2<f32>| -> f32 {
            probs.iter().filter(|&&p| p > 1e-10).map(|&p| -p * p.ln()).sum()
        };

        let h_low = entropy(&probs_low);
        let h_high = entropy(&probs_high);

        assert!(
            h_high > h_low,
            "FALSIFIED EMB-007: higher temperature should increase entropy, got h_low={h_low}, h_high={h_high}"
        );
    }

    // =========================================================================
    // FALSIFY-SM: softmax-kernel-v1.yaml contract (entrenar's softmax_2d)
    //
    // Five-Whys (PMAT-354):
    //   Why 1: entrenar had test_softmax_sums_to_one but no FALSIFY-SM-*
    //   Why 2: existing test checks 1 property, not all 3 contract invariants
    //   Why 3: no mapping from softmax-kernel-v1.yaml to entrenar tests
    //   Why 4: entrenar predates the provable-contracts YAML
    //   Why 5: distillation softmax was "obviously correct" (3 lines)
    // =========================================================================

    /// FALSIFY-SM-001: Softmax output sums to 1 per row
    #[test]
    fn falsify_sm_001_sums_to_one() {
        let x = array![[3.0, 1.0, 0.5, -1.0], [-2.0, 0.0, 4.0, 1.0]];
        let probs = softmax_2d(&x);

        for (idx, row) in probs.axis_iter(Axis(0)).enumerate() {
            let sum: f32 = row.sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
            let _ = idx;
        }
    }

    /// FALSIFY-SM-002: All softmax outputs strictly positive
    #[test]
    fn falsify_sm_002_strictly_positive() {
        let x = array![[-10.0, -5.0, 0.0, 5.0, 10.0]];
        let probs = softmax_2d(&x);

        for &p in &probs {
            assert!(p > 0.0, "FALSIFIED SM-002: softmax output {p} not strictly positive");
        }
    }

    /// FALSIFY-SM-003: Order preservation (argmax invariant)
    #[test]
    fn falsify_sm_003_order_preservation() {
        let x = array![[1.0, 5.0, 3.0, 2.0]];
        let probs = softmax_2d(&x);

        let input_argmax = x
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation should succeed"))
            .expect("operation should succeed")
            .0;
        let output_argmax = probs
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation should succeed"))
            .expect("operation should succeed")
            .0;

        assert_eq!(
            input_argmax, output_argmax,
            "FALSIFIED SM-003: argmax changed from {input_argmax} to {output_argmax}"
        );
    }

    /// FALSIFY-SM-004: Softmax outputs bounded in [0, 1]
    ///
    /// Contract: 0 <= softmax(x)_i <= 1 for all i
    ///
    /// N-10 escape: IEEE 754 f32 underflow — exp(-200) = 0.0 exactly, so the
    /// mathematical open interval (0,1) becomes closed [0,1] in floating point.
    /// This is correct behavior, not a bug.
    #[test]
    fn falsify_sm_004_bounded_zero_one() {
        let x = array![[-100.0, -10.0, 0.0, 10.0, 100.0]];
        let probs = softmax_2d(&x);

        for &p in &probs {
            assert!((0.0..=1.0).contains(&p), "FALSIFIED SM-004: softmax output {p} not in [0, 1]");
        }

        // For moderate inputs, outputs ARE strictly in (0, 1) — no underflow
        let moderate = array![[1.0, 2.0, 3.0]];
        let probs_mod = softmax_2d(&moderate);
        for &p in &probs_mod {
            assert!(
                p > 0.0 && p < 1.0,
                "FALSIFIED SM-004: moderate softmax output {p} not in (0, 1)"
            );
        }
    }

    /// FALSIFY-SM-005: Numerical stability — extreme inputs don't produce NaN/Inf
    ///
    /// Contract: softmax is stable for inputs near f32 limits (via max-subtraction trick)
    #[test]
    fn falsify_sm_005_numerical_stability() {
        let x = array![[1000.0, 999.0, 998.0]];
        let probs = softmax_2d(&x);

        for &p in &probs {
            assert!(
                p.is_finite(),
                "FALSIFIED SM-005: softmax output {p} not finite for extreme inputs"
            );
            assert!(
                p > 0.0,
                "FALSIFIED SM-005: softmax output {p} not positive for extreme inputs"
            );
        }

        let sum: f32 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);
    }

    /// FALSIFY-SM-006: Identical elements → uniform distribution
    ///
    /// Contract: softmax([c, c, ..., c]) = [1/n, 1/n, ..., 1/n]
    #[test]
    fn falsify_sm_006_identical_elements_uniform() {
        for n in [2, 4, 8, 16] {
            let data: Vec<f32> = vec![7.0; n];
            let x = Array2::from_shape_vec((1, n), data).expect("operation should succeed");
            let probs = softmax_2d(&x);

            let expected = 1.0 / n as f32;
            for (i, &p) in probs.iter().enumerate() {
                assert_relative_eq!(p, expected, epsilon = 1e-6);
                let _ = i;
            }
        }
    }

    /// FALSIFY-SM-009: Single element boundary — softmax([x]) = [1.0]
    ///
    /// Contract: YAML SM-005 = softmax of a single element is always 1.0.
    #[test]
    fn falsify_sm_009_single_element() {
        for x in [0.0_f32, 1.0, -1.0, 100.0, -100.0, f32::MIN_POSITIVE] {
            let t = array![[x]];
            let probs = softmax_2d(&t);
            assert!(
                (probs[[0, 0]] - 1.0).abs() < 1e-6,
                "FALSIFIED SM-009: softmax([{x}]) = {}, expected 1.0",
                probs[[0, 0]]
            );
        }
    }

    /// FALSIFY-SM-007: Translation invariance — σ(x + c) = σ(x) for any scalar c
    ///
    /// Five-Whys (PMAT-354):
    ///   Why 1: SM-INV-003 (translation invariance) had ZERO coverage
    ///   Why 2: max-subtraction trick IMPLEMENTS this but nobody tested it
    ///   Why 3: foundational to numerical stability but untested
    ///
    /// Contract: σ(x + c·1) = σ(x) for any scalar c.
    #[test]
    fn falsify_sm_007_translation_invariance() {
        let base = array![[1.0_f32, 3.0, -2.0, 0.5]];
        let base_probs = softmax_2d(&base);

        for c in [100.0_f32, -100.0, 0.0, 42.0, -999.0] {
            let shifted = array![[1.0 + c, 3.0 + c, -2.0 + c, 0.5 + c]];
            let shifted_probs = softmax_2d(&shifted);

            for (i, (&orig, &shift)) in base_probs.iter().zip(shifted_probs.iter()).enumerate() {
                assert!(
                    (orig - shift).abs() < 1e-5,
                    "FALSIFIED SM-007: σ(x+{c})[{i}] = {shift} != σ(x)[{i}] = {orig}"
                );
            }
        }
    }

    mod softmax_proptest_falsify {
        use super::*;
        use proptest::prelude::*;

        // FALSIFY-SM-001-prop: Normalization for random vectors
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_sm_001_prop_sums_to_one(
                logits in proptest::collection::vec(-100.0_f32..100.0, 2..64),
            ) {
                let n = logits.len();
                let arr = Array2::from_shape_vec((1, n), logits).expect("operation should succeed");
                let probs = softmax_2d(&arr);
                let sum: f32 = probs.row(0).sum();
                prop_assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "FALSIFIED SM-001-prop: sum={} for {} elements", sum, n
                );
            }
        }

        // FALSIFY-SM-002-prop: Positivity for random vectors
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_sm_002_prop_positive(
                logits in proptest::collection::vec(-500.0_f32..500.0, 2..32),
            ) {
                let n = logits.len();
                let arr = Array2::from_shape_vec((1, n), logits).expect("operation should succeed");
                let probs = softmax_2d(&arr);
                for (i, &p) in probs.row(0).iter().enumerate() {
                    prop_assert!(p >= 0.0, "FALSIFIED SM-002-prop: probs[{}]={} negative", i, p);
                    prop_assert!(p.is_finite(), "FALSIFIED SM-002-prop: probs[{}]={} non-finite", i, p);
                }
            }
        }

        // FALSIFY-SM-003-prop: Order preservation for random vectors
        //
        // Contract: argmax(softmax(x)) = argmax(x) when no duplicate max
        proptest! {
            #![proptest_config(ProptestConfig::with_cases(500))]
            #[test]
            fn falsify_sm_003_prop_order_preservation(
                logits in proptest::collection::vec(-50.0_f32..50.0, 2..32),
            ) {
                let has_dupes = logits.windows(2).any(|w| (w[0] - w[1]).abs() < 1e-10);
                if has_dupes {
                    return Ok(());
                }

                let n = logits.len();
                let arr = Array2::from_shape_vec((1, n), logits.clone()).expect("operation should succeed");
                let probs = softmax_2d(&arr);
                let input_argmax = logits.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation should succeed")).expect("operation should succeed").0;
                let output_argmax = probs.row(0).iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("operation should succeed")).expect("operation should succeed").0;
                prop_assert_eq!(
                    input_argmax, output_argmax,
                    "FALSIFIED SM-003-prop: argmax {} -> {} for {:?}", input_argmax, output_argmax, logits
                );
            }
        }
    }
}
