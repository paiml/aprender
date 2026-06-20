//! Knowledge-distillation training step orchestration.
//!
//! # SPEC-DISTILL-001 Phase 2 (PMAT-691)
//!
//! This module wires together the teacher logits provider (Phase 1/1b)
//! and the student trainer's loss computation, producing a per-step KD
//! signal that the pipeline can log and (in Phase 2b) feed back into the
//! student's gradient update.
//!
//! ## Scope of Phase 2 vs Phase 2b
//!
//! **Phase 2 (this module, landed in this PR)**: orchestrates the data
//! path. For each batch:
//!
//! 1. Call `teacher.logits_for_batch(input_ids)` → teacher logits.
//! 2. Compute the student's predictions via a caller-supplied closure
//!    (so the implementation isn't coupled to `CudaTransformerTrainer`
//!    today — the caller can pass any function that returns student
//!    logits per batch element).
//! 3. Apply `DistillationLoss::forward` to produce the combined CE+KL
//!    scalar for logging.
//! 4. Compute the KD-aware logit-space gradient via `kd_logit_gradient`
//!    (made available here as a Phase 2b primer, but not yet pushed
//!    through `CudaTransformerTrainer.backward` — that wiring is
//!    Phase 2b).
//!
//! **Phase 2b (separate ticket, PMAT-694)**: extends
//! `CudaTransformerTrainer` with `forward_backward_kd_batch(batch,
//! teacher_logits)` that uses the KD logit gradient (not CE alone) as
//! the back-prop seed. With that in place, the pipeline switches from
//! "CE training with KD telemetry" to "real KD training".
//!
//! Splitting Phase 2 into 2a/2b lets us land the orchestration layer
//! and its tests now — without needing to extend a complex piece of
//! GPU code in the same PR.
//!
//! ## Falsifiers pinned here
//!
//! - **F-DISTILL-KDSTEP-001** — `kd_logit_gradient` reduces to plain
//!   softmax-CE gradient when `alpha = 1.0`. (CE-only sanity bound.)
//! - **F-DISTILL-KDSTEP-002** — when student logits equal teacher logits
//!   (perfect agreement), the KL portion of the loss is zero and the
//!   KL portion of the gradient is zero.
//! - **F-DISTILL-KDSTEP-003** — `kd_loss_for_batch` produces a scalar
//!   that strictly increases when student logits move away from a fixed
//!   teacher target.
//!
//! These pin the orchestration math now so Phase 2b only has to wire
//! the GPU backward — it doesn't have to also re-derive the math.

use crate::teacher_provider::TeacherLogitsProvider;
use entrenar_common::Result;

/// Softmax over a 1D logits slice (numerically stable via max-shift).
fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
    let z: f32 = exps.iter().sum();
    if z > 0.0 {
        exps.into_iter().map(|x| x / z).collect()
    } else {
        // Pathological: all -inf. Fall back to uniform.
        vec![1.0 / logits.len() as f32; logits.len()]
    }
}

/// One-hot gradient component: softmax(student) - one_hot(label).
///
/// This is the gradient of CE loss w.r.t. logits (assuming softmax-then-NLL
/// reduces to this clean form, which is the standard derivation).
fn ce_logit_gradient(student_logits: &[f32], label: usize) -> Vec<f32> {
    let mut grad = softmax(student_logits);
    if label < grad.len() {
        grad[label] -= 1.0;
    }
    grad
}

/// Compute the combined KD logit gradient.
///
/// ```text
///   ∂L/∂s = α · (softmax(s) - one_hot(label))
///         + (1-α) · T · (softmax(s/T) - softmax(t/T))
/// ```
///
/// where `s` is student logits, `t` is teacher logits, `T` is the
/// distillation temperature, and `α` is the CE-vs-KD weight.
///
/// The T factor (instead of T²) is correct because the gradient of
/// the T²-scaled KL with respect to student logits absorbs one of the
/// T factors — see Hinton et al. 2015 §2 footnote 2 for the derivation.
///
/// **Phase 2b plug point**: this is the gradient that
/// `CudaTransformerTrainer.forward_backward_kd_batch` will seed its
/// backward pass with (instead of the CE-only gradient).
pub fn kd_logit_gradient(
    student_logits: &[f32],
    teacher_logits: &[f32],
    label: usize,
    temperature: f32,
    alpha: f32,
) -> Vec<f32> {
    assert_eq!(
        student_logits.len(),
        teacher_logits.len(),
        "kd_logit_gradient: student and teacher logits must have the same vocab size"
    );
    let n = student_logits.len();
    let ce_grad = ce_logit_gradient(student_logits, label);

    if alpha >= 1.0 {
        // Pure CE: KD term contributes nothing.
        return ce_grad;
    }

    // Temperature-scaled softmaxes.
    let t_safe = temperature.max(1e-6);
    let scaled_s: Vec<f32> = student_logits.iter().map(|x| x / t_safe).collect();
    let scaled_t: Vec<f32> = teacher_logits.iter().map(|x| x / t_safe).collect();
    let p_s = softmax(&scaled_s);
    let p_t = softmax(&scaled_t);

    let mut out = vec![0.0_f32; n];
    for i in 0..n {
        let kd_term = t_safe * (p_s[i] - p_t[i]);
        out[i] = alpha * ce_grad[i] + (1.0 - alpha) * kd_term;
    }
    out
}

/// Compute the scalar combined KD loss for a single (student, teacher,
/// label) triple.
///
/// ```text
///   L = α · CE(softmax(s), label)
///     + (1-α) · T² · KL(softmax(t/T) || softmax(s/T))
/// ```
///
/// The soft-target term is FORWARD KL `KL(teacher || student)` — the
/// knowledge-distillation objective of Hinton et al. 2015 and PyTorch
/// `KLDivLoss(log_softmax(student/T), softmax(teacher/T))`. This is the
/// antiderivative of `kd_logit_gradient`'s KD term `T·(p_s − p_t)`, so the
/// logged loss is consistent with the gradient that trains the model.
///
/// Returned for logging / telemetry only — the gradient that goes back
/// through the model is `kd_logit_gradient`, not the symbolic derivative
/// of this scalar.
pub fn kd_loss(
    student_logits: &[f32],
    teacher_logits: &[f32],
    label: usize,
    temperature: f32,
    alpha: f32,
) -> f32 {
    let p_s_hard = softmax(student_logits);
    let ce = if label < p_s_hard.len() {
        -(p_s_hard[label].max(1e-9).ln())
    } else {
        0.0
    };

    if alpha >= 1.0 {
        return ce;
    }

    let t_safe = temperature.max(1e-6);
    let scaled_s: Vec<f32> = student_logits.iter().map(|x| x / t_safe).collect();
    let scaled_t: Vec<f32> = teacher_logits.iter().map(|x| x / t_safe).collect();
    let p_s = softmax(&scaled_s);
    let p_t = softmax(&scaled_t);

    // FORWARD KL — KL(P_t || P_s) = sum_i p_t[i] * (log p_t[i] - log p_s[i]).
    // This is the knowledge-distillation objective from Hinton et al. 2015 and
    // PyTorch `KLDivLoss(log_softmax(student/T), softmax(teacher/T))` (which
    // computes `sum p_t·(ln p_t − ln p_s)`). It is mean-seeking — the student
    // is pulled to cover all of the teacher's mass — whereas reverse KL
    // `KL(P_s || P_t)` is mode-seeking and is the WRONG direction for KD.
    // It is also the antiderivative of `kd_logit_gradient`'s KD term
    // `T·(p_s − p_t)`, keeping the logged loss consistent with the gradient.
    let mut kl = 0.0_f32;
    for i in 0..p_t.len() {
        if p_t[i] > 0.0 {
            kl += p_t[i] * (p_t[i].max(1e-9).ln() - p_s[i].max(1e-9).ln());
        }
    }

    alpha * ce + (1.0 - alpha) * t_safe * t_safe * kl
}

/// Run a single KD orchestration step.
///
/// Returns `(combined_loss, per_batch_logit_gradients)`:
/// - `combined_loss` is the scalar `L` averaged over the batch.
/// - `per_batch_logit_gradients` is `Vec<Vec<f32>>` shape `[batch, vocab]`
///   — the gradient that Phase 2b will feed into the student trainer's
///   backward pass.
///
/// `compute_student_logits` is a closure the caller supplies. In Phase 2a
/// tests this is a fixture; in Phase 4 production runs it'll be backed by
/// the student `CudaTransformerTrainer`'s `forward_logits` method.
///
/// # Errors
///
/// Propagates errors from the teacher provider. Returns
/// `EntrenarError::Internal` if `student_compute` produces logits whose
/// length doesn't match the teacher's vocab size.
pub fn kd_step<F>(
    teacher: &mut dyn TeacherLogitsProvider,
    input_ids: &[Vec<u32>],
    labels: &[usize],
    temperature: f32,
    alpha: f32,
    mut compute_student_logits: F,
) -> Result<(f32, Vec<Vec<f32>>)>
where
    F: FnMut(&[u32]) -> Vec<f32>,
{
    assert_eq!(
        input_ids.len(),
        labels.len(),
        "kd_step: input_ids and labels must have the same batch size"
    );
    let teacher_logits = teacher.logits_for_batch(input_ids)?;
    let vocab = teacher.vocab_size();

    let mut total_loss = 0.0_f32;
    let mut grads = Vec::with_capacity(input_ids.len());
    for ((ids, t_logits), &label) in input_ids
        .iter()
        .zip(teacher_logits.iter())
        .zip(labels.iter())
    {
        let s_logits = compute_student_logits(ids);
        if s_logits.len() != vocab {
            return Err(entrenar_common::EntrenarError::Internal {
                message: format!(
                    "kd_step: student logits len {} != teacher vocab_size {}",
                    s_logits.len(),
                    vocab
                ),
            });
        }
        total_loss += kd_loss(&s_logits, t_logits, label, temperature, alpha);
        grads.push(kd_logit_gradient(
            &s_logits,
            t_logits,
            label,
            temperature,
            alpha,
        ));
    }
    let avg_loss = if input_ids.is_empty() {
        0.0
    } else {
        total_loss / input_ids.len() as f32
    };
    Ok((avg_loss, grads))
}

/// Run a per-position KD orchestration step (full-sequence distillation).
///
/// Where [`kd_step`] trains on ONE target per window (the next token after
/// the window), this trains on EVERY position: position `p` of each row
/// predicts the token at `p+1`. That is up to `seq_len`× more KD signal per
/// forward pass.
///
/// - `labels` is `[batch][position]` — the shifted target sequence per row
///   (from `BatchSource::next_batch_per_position`).
/// - `teacher` supplies `[batch][position][vocab]` via
///   [`TeacherLogitsProvider::logits_per_position`].
/// - `compute_student_logits_per_position(row)` returns `[position][vocab]`
///   for one input row (typically the student provider's per-position
///   forward applied to that row).
///
/// Returns `(avg_loss, grads)` where `avg_loss` is averaged over ALL
/// (batch × position) predictions and `grads` is `[batch][position][vocab]`
/// (feed to [`crate::student_provider::StudentLogitsProvider::apply_kd_gradient_per_position`]).
///
/// Per-row position counts are reconciled as `min(teacher, student, labels)`
/// so a row with fewer student/label positions simply trains on the prefix
/// it has — no panic on ragged batches.
///
/// # Errors
///
/// Propagates teacher errors; returns `EntrenarError::Internal` if any
/// student/teacher logit row length doesn't match the teacher's vocab size.
pub fn kd_step_per_position<F>(
    teacher: &mut dyn TeacherLogitsProvider,
    input_ids: &[Vec<u32>],
    labels: &[Vec<usize>],
    temperature: f32,
    alpha: f32,
    mut compute_student_logits_per_position: F,
) -> Result<(f32, Vec<Vec<Vec<f32>>>)>
where
    F: FnMut(&[u32]) -> Vec<Vec<f32>>,
{
    assert_eq!(
        input_ids.len(),
        labels.len(),
        "kd_step_per_position: input_ids and labels must have the same batch size"
    );
    let teacher_pp = teacher.logits_per_position(input_ids)?;
    let vocab = teacher.vocab_size();

    let mut total_loss = 0.0_f32;
    let mut prediction_count = 0usize;
    let mut grads: Vec<Vec<Vec<f32>>> = Vec::with_capacity(input_ids.len());

    for ((ids, t_rows), row_labels) in input_ids.iter().zip(teacher_pp.iter()).zip(labels.iter()) {
        let s_rows = compute_student_logits_per_position(ids);
        let n_pos = t_rows.len().min(s_rows.len()).min(row_labels.len());
        let mut row_grads = Vec::with_capacity(n_pos);
        for p in 0..n_pos {
            let s = &s_rows[p];
            let t = &t_rows[p];
            if s.len() != vocab || t.len() != vocab {
                return Err(entrenar_common::EntrenarError::Internal {
                    message: format!(
                        "kd_step_per_position: logit len mismatch at position {p} \
                         (student {}, teacher {}, vocab {vocab})",
                        s.len(),
                        t.len()
                    ),
                });
            }
            let label = row_labels[p];
            total_loss += kd_loss(s, t, label, temperature, alpha);
            prediction_count += 1;
            row_grads.push(kd_logit_gradient(s, t, label, temperature, alpha));
        }
        grads.push(row_grads);
    }

    let avg_loss = if prediction_count == 0 {
        0.0
    } else {
        total_loss / prediction_count as f32
    };
    Ok((avg_loss, grads))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::teacher_provider::FixtureTeacher;

    #[test]
    fn softmax_is_unit_sum_and_nonnegative() {
        let logits = vec![1.0_f32, 2.0, 3.0, -1.0];
        let p = softmax(&logits);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sums to 1 (got {sum})");
        for v in &p {
            assert!(*v >= 0.0, "softmax outputs are non-negative");
        }
    }

    #[test]
    fn ce_gradient_correct_sign() {
        // The label has probability 1; non-label tokens have ~0. After CE,
        // the gradient at the label position should be negative (decrease
        // logit pushes prob up, which we don't want — wait, opposite:
        // gradient is softmax-1, so at the label, grad = p - 1 < 0; we
        // subtract grad to MAXIMIZE p[label].
        let logits = vec![0.0_f32, 0.0, 0.0, 0.0];
        let g = ce_logit_gradient(&logits, 2);
        // softmax of uniform = 0.25 everywhere; grad[label] = 0.25 - 1 = -0.75
        assert!((g[2] - (-0.75)).abs() < 1e-6);
        // grad at non-label = 0.25 (positive)
        for (i, &val) in g.iter().enumerate() {
            if i != 2 {
                assert!((val - 0.25).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn falsify_kdstep_001_alpha_1_reduces_to_pure_ce() {
        // F-DISTILL-KDSTEP-001
        let s = vec![1.0_f32, 0.5, -0.3, 2.0];
        let t = vec![3.0_f32, -0.1, 0.4, 0.0]; // teacher should NOT affect output when alpha=1.0
        let g_kd = kd_logit_gradient(&s, &t, 3, 4.0, 1.0);
        let g_ce = ce_logit_gradient(&s, 3);
        for (a, b) in g_kd.iter().zip(g_ce.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "alpha=1 must collapse KD gradient to CE: {a} vs {b}"
            );
        }
    }

    #[test]
    fn falsify_kdstep_002_perfect_agreement_zero_kl_term() {
        // F-DISTILL-KDSTEP-002 — student == teacher → KL portion is 0.
        // With alpha=0, the gradient is purely KL, so it must equal zero.
        let s = vec![1.0_f32, 0.5, -0.3, 2.0, 0.1];
        let t = s.clone();
        let g = kd_logit_gradient(&s, &t, 0, 4.0, 0.0);
        for &val in &g {
            assert!(
                val.abs() < 1e-6,
                "student==teacher with alpha=0 must produce zero gradient, got {val}"
            );
        }
    }

    #[test]
    fn falsify_kdstep_003_loss_increases_as_student_diverges() {
        // F-DISTILL-KDSTEP-003 — KD loss increases monotonically as student
        // logits drift further from a fixed teacher reference.
        let t = vec![1.0_f32, 2.0, 3.0, 0.5];

        let s_close = t.clone();
        let s_far: Vec<f32> = t.iter().map(|x| -x).collect(); // mirror across zero

        let loss_close = kd_loss(&s_close, &t, 0, 4.0, 0.0);
        let loss_far = kd_loss(&s_far, &t, 0, 4.0, 0.0);

        assert!(
            loss_far > loss_close,
            "KD loss with diverged student ({loss_far}) must exceed loss with matched student ({loss_close})"
        );
    }

    /// PMAT-868 / F-KD-FORWARD-KL-001: the soft-target term of `kd_loss`
    /// must be FORWARD KL — `KL(teacher ‖ student) = Σ p_t·(ln p_t − ln p_s)`
    /// scaled by T² — matching Hinton (2015) and PyTorch
    /// `KLDivLoss(log_softmax(student/T), softmax(teacher/T))`.
    ///
    /// Reverse KL `KL(student ‖ teacher)` is mode-seeking and is the WRONG
    /// objective for knowledge distillation; it also makes the logged loss
    /// inconsistent with `kd_logit_gradient` (whose KD term is the gradient
    /// of forward KL: `T·(p_s − p_t)`).
    ///
    /// RED (pre-fix): equals the reverse-KL value `T²·Σ p_s·(ln p_s − ln p_t)`.
    /// GREEN (post-fix): equals the forward-KL value `T²·Σ p_t·(ln p_t − ln p_s)`.
    #[test]
    fn pmat_868_kd_loss_soft_term_is_forward_kl() {
        // Asymmetric teacher/student so forward KL != reverse KL.
        let s = vec![2.0_f32, 0.0, -1.0, 0.5];
        let t = vec![0.3_f32, 1.7, -0.2, 2.4];
        let temperature = 3.0_f32;

        // Reference temperature-scaled softmaxes (independent of impl path).
        let p_s = softmax(&s.iter().map(|x| x / temperature).collect::<Vec<_>>());
        let p_t = softmax(&t.iter().map(|x| x / temperature).collect::<Vec<_>>());

        // Hand-computed forward KL: KL(teacher ‖ student) = Σ p_t·(ln p_t − ln p_s).
        let mut forward_kl = 0.0_f32;
        for i in 0..p_t.len() {
            if p_t[i] > 0.0 {
                forward_kl += p_t[i] * (p_t[i].max(1e-9).ln() - p_s[i].max(1e-9).ln());
            }
        }
        // Reverse KL: KL(student ‖ teacher) = Σ p_s·(ln p_s − ln p_t) — the BUG.
        let mut reverse_kl = 0.0_f32;
        for i in 0..p_s.len() {
            if p_s[i] > 0.0 {
                reverse_kl += p_s[i] * (p_s[i].max(1e-9).ln() - p_t[i].max(1e-9).ln());
            }
        }
        let t2 = temperature * temperature;
        let expected_forward = t2 * forward_kl;
        let expected_reverse = t2 * reverse_kl;

        // Sanity: the two directions genuinely differ on this fixture, so the
        // assertion below actually discriminates (not a degenerate symmetric case).
        assert!(
            (expected_forward - expected_reverse).abs() > 1e-3,
            "fixture must make forward != reverse KL (fwd {expected_forward}, rev {expected_reverse})"
        );

        // alpha = 0.0 → pure soft-target term, no CE contamination.
        let loss = kd_loss(&s, &t, 0, temperature, 0.0);

        assert!(
            (loss - expected_forward).abs() < 1e-5,
            "kd_loss soft term must be FORWARD KL·T² ({expected_forward}), got {loss} \
             (reverse-KL·T² would be {expected_reverse})"
        );
        assert!(
            (loss - expected_reverse).abs() > 1e-3,
            "kd_loss must NOT equal reverse-KL·T² ({expected_reverse}); got {loss}"
        );

        // KL is a non-negative divergence.
        assert!(
            loss >= 0.0,
            "forward KL·T² must be non-negative, got {loss}"
        );

        // Zero soft-target loss when student == teacher (KL of a dist with itself).
        let loss_eq = kd_loss(&s, &s, 0, temperature, 0.0);
        assert!(
            loss_eq.abs() < 1e-5,
            "student==teacher → zero soft-target loss, got {loss_eq}"
        );
    }

    #[test]
    fn kd_loss_alpha_1_is_pure_ce() {
        // Sanity: with alpha=1, loss is just CE.
        let s = vec![0.0_f32, 0.0, 0.0, 0.0];
        let t = vec![10.0_f32, 0.0, 0.0, 0.0]; // teacher should be ignored
        let loss = kd_loss(&s, &t, 2, 4.0, 1.0);
        // Uniform softmax → p[label] = 0.25 → CE = -ln(0.25) = ln(4)
        assert!((loss - 4.0_f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn kd_step_orchestrates_teacher_and_student() {
        let mut teacher = FixtureTeacher::new(8);
        // Student returns the same logits regardless of input (constant model).
        let compute_student = |_ids: &[u32]| -> Vec<f32> { vec![0.5_f32; 8] };

        let input_ids = vec![vec![1, 2, 3], vec![4, 5]];
        let labels = vec![3, 5];
        let (loss, grads) =
            kd_step(&mut teacher, &input_ids, &labels, 4.0, 0.5, compute_student).unwrap();

        assert!(loss.is_finite() && loss > 0.0, "loss is finite + positive");
        assert_eq!(grads.len(), 2, "one gradient vec per batch element");
        for g in &grads {
            assert_eq!(g.len(), 8, "gradient is vocab-sized");
            for &val in g {
                assert!(val.is_finite(), "gradient is finite");
            }
        }
    }

    #[test]
    fn kd_step_returns_zero_loss_on_empty_batch() {
        let mut teacher = FixtureTeacher::new(8);
        let compute_student = |_ids: &[u32]| vec![0.0_f32; 8];
        let (loss, grads) = kd_step(&mut teacher, &[], &[], 4.0, 0.5, compute_student).unwrap();
        assert_eq!(loss, 0.0);
        assert!(grads.is_empty());
    }

    #[test]
    fn kd_step_errors_on_vocab_size_mismatch() {
        let mut teacher = FixtureTeacher::new(16);
        let compute_student = |_ids: &[u32]| vec![0.0_f32; 8]; // wrong size
        let result = kd_step(&mut teacher, &[vec![1]], &[0], 4.0, 0.5, compute_student);
        assert!(
            result.is_err(),
            "vocab size mismatch must error, not silently corrupt"
        );
    }

    // ===== Per-position (full-sequence) KD — contract distill-per-position-kd-v1 =====

    /// FT-PERPOS-001: per-position KD trains on EVERY position. For a batch of
    /// B rows each of length L, it makes B×L predictions and returns grads
    /// shaped [B][L] — strictly more signal than the per-row path's B.
    #[test]
    fn pmat_perpos_001_trains_on_all_positions() {
        let vocab = 16;
        let mut teacher = FixtureTeacher::new(vocab);
        // 2 rows, length 4 → 8 predictions (vs 2 for per-row).
        let inputs = vec![vec![1u32, 2, 3, 4], vec![5u32, 6, 7, 8]];
        let labels = vec![vec![2usize, 3, 4, 4], vec![6usize, 7, 8, 8]];
        let student = |ids: &[u32]| vec![vec![0.0_f32; vocab]; ids.len()];
        let (_loss, grads) =
            kd_step_per_position(&mut teacher, &inputs, &labels, 4.0, 0.5, student)
                .expect("per-position step");
        assert_eq!(grads.len(), 2, "one grad-block per row");
        assert_eq!(grads[0].len(), 4, "FT-PERPOS-001: grads at ALL 4 positions");
        assert_eq!(grads[1].len(), 4);
        let predictions: usize = grads.iter().map(Vec::len).sum();
        assert_eq!(predictions, 8, "B×L = 2×4 predictions, not B=2");
    }

    /// FT-PERPOS-002: per-position yields strictly more predictions than the
    /// per-row path on the same data (the whole point of full-sequence KD).
    #[test]
    fn pmat_perpos_002_more_signal_than_per_row() {
        let vocab = 16;
        let inputs = vec![vec![1u32, 2, 3, 4]];
        // per-row
        let mut t1 = FixtureTeacher::new(vocab);
        let (_l, per_row_grads) =
            kd_step(&mut t1, &inputs, &[4], 4.0, 0.5, |_| vec![0.0_f32; vocab]).expect("per-row");
        // per-position
        let mut t2 = FixtureTeacher::new(vocab);
        let (_l2, pp_grads) = kd_step_per_position(
            &mut t2,
            &inputs,
            &[vec![2usize, 3, 4, 4]],
            4.0,
            0.5,
            |ids| vec![vec![0.0_f32; vocab]; ids.len()],
        )
        .expect("per-position");
        let per_row: usize = per_row_grads.len();
        let per_pos: usize = pp_grads.iter().map(Vec::len).sum();
        assert_eq!(per_row, 1);
        assert_eq!(per_pos, 4);
        assert!(per_pos > per_row, "per-position must produce more signal");
    }

    /// FT-PERPOS-003: when the student equals the teacher at every position
    /// (and alpha=0, pure KD), the loss and all gradients are ~zero — the math
    /// is correct per-position (mirror of F-DISTILL-KDSTEP-002).
    #[test]
    fn pmat_perpos_003_zero_loss_when_student_equals_teacher() {
        let vocab = 8;
        let mut teacher = FixtureTeacher::new(vocab);
        let inputs = vec![vec![1u32, 2, 3]];
        let labels = vec![vec![2usize, 3, 3]];
        // Student returns exactly the teacher's per-position logits.
        let teacher_pp = teacher.logits_per_position(&inputs).expect("teacher pp");
        let tp = teacher_pp.clone();
        let (loss, grads) = kd_step_per_position(
            &mut teacher,
            &inputs,
            &labels,
            4.0,
            0.0, // pure KD → no CE term
            move |_ids| tp[0].clone(),
        )
        .expect("per-position step");
        assert!(
            loss.abs() < 1e-4,
            "alpha=0 + student==teacher → ~0 loss (got {loss})"
        );
        for row in &grads {
            for g in row {
                let max = g.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
                assert!(max < 1e-4, "KD grad ~0 when student==teacher (got {max})");
            }
        }
    }

    /// FT-PERPOS-004: ragged rows (fewer student/label positions) train on the
    /// common prefix without panicking — robustness on uneven batches.
    #[test]
    fn pmat_perpos_004_ragged_rows_use_min_positions() {
        let vocab = 16;
        let mut teacher = FixtureTeacher::new(vocab);
        let inputs = vec![vec![1u32, 2, 3, 4]]; // teacher gives 4 positions
        let labels = vec![vec![2usize, 3]]; // only 2 labels
        let student = |_ids: &[u32]| vec![vec![0.0_f32; vocab]; 3]; // only 3 positions
        let (_loss, grads) =
            kd_step_per_position(&mut teacher, &inputs, &labels, 4.0, 0.5, student)
                .expect("ragged step must not panic");
        assert_eq!(grads[0].len(), 2, "min(4 teacher, 3 student, 2 labels) = 2");
    }
}
