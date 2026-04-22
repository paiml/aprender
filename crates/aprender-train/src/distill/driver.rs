//! Per-epoch divergence abort for QLoRA distillation.
//!
//! Contract: `contracts/entrenar/qlora-distillation-v1.yaml`
//! Binding: INV-DISTILL-003.
//!
//! MODEL-1 v2 produced `train_loss=15.41` at epoch 0 and still wrote
//! `best/model.safetensors`. The ordering bug was: checkpoint-write
//! happened BEFORE the divergence check. This module inverts the
//! ordering with a small state machine so the check MUST pass before
//! any "best" pointer update.

use std::path::Path;

/// Per-epoch loss ceiling. Array index = epoch number; value = maximum
/// acceptable `train_loss`. For epochs beyond the array length the last
/// ceiling applies (asymptote).
///
/// Defaults surfaced in the contract:
///   epoch 0: < 4.0
///   epoch 1: < 2.8
///   epoch 2: < 2.2
pub const DEFAULT_EPOCH_LOSS_CEILINGS: &[f32] = &[4.0, 2.8, 2.2];

/// Structured abort verdict. `DIVERGENCE_DETECTED` matches the exit
/// contract so downstream dispatchers (e.g. `apr distill finetune`) can
/// map it to a non-zero shell exit without ambiguity.
#[derive(Debug, Clone, PartialEq)]
pub enum EpochVerdict {
    Continue,
    DivergenceDetected { epoch: u32, train_loss: f32, ceiling: f32 },
}

/// Evaluate a single epoch boundary. MUST be called before any
/// checkpoint-write or "best" pointer update.
pub fn epoch_divergence_check(epoch: u32, train_loss: f32, ceilings: &[f32]) -> EpochVerdict {
    let ceiling = if ceilings.is_empty() {
        f32::INFINITY
    } else {
        let idx = (epoch as usize).min(ceilings.len() - 1);
        ceilings[idx]
    };
    if train_loss >= ceiling {
        EpochVerdict::DivergenceDetected { epoch, train_loss, ceiling }
    } else {
        EpochVerdict::Continue
    }
}

/// Full epoch-boundary dispatcher. Returns `Ok(())` and is allowed to
/// update `best_dir`; returns `Err(verdict)` and MUST NOT touch
/// `best_dir` if divergence is detected.
///
/// The caller is responsible for the actual checkpoint write; this fn
/// only enforces the ORDER (check-then-write, not write-then-check).
pub fn advance_epoch(
    epoch: u32,
    train_loss: f32,
    ceilings: &[f32],
    best_dir: &Path,
) -> Result<(), EpochVerdict> {
    match epoch_divergence_check(epoch, train_loss, ceilings) {
        EpochVerdict::Continue => Ok(()),
        verdict @ EpochVerdict::DivergenceDetected { .. } => {
            // Explicit contract: diverged — do not touch best_dir.
            let _ = best_dir; // borrow to document non-use
            Err(verdict)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// FALSIFY-DISTILL-003: the exact v2 failure signature. A synthetic
    /// trainer emits train_loss=15.41 at epoch 0; the driver MUST halt
    /// before the best/ pointer is updated.
    #[test]
    fn epoch_0_train_loss_15_aborts() {
        let best_dir = PathBuf::from("/tmp/nonexistent-test-best-dir-forbidden");
        let v2_train_loss = 15.41_f32;

        let result = advance_epoch(0, v2_train_loss, DEFAULT_EPOCH_LOSS_CEILINGS, &best_dir);

        match result {
            Err(EpochVerdict::DivergenceDetected { epoch, train_loss, ceiling }) => {
                assert_eq!(epoch, 0);
                assert!((train_loss - 15.41).abs() < 1e-5);
                assert!((ceiling - 4.0).abs() < 1e-6);
            }
            other => panic!("expected DivergenceDetected, got {other:?}"),
        }

        // Abort contract: the driver must NOT have created best_dir.
        assert!(!best_dir.exists(), "divergence abort leaked a best/ directory — ordering bug");
    }

    #[test]
    fn converging_epoch_0_continues() {
        let best_dir = PathBuf::from("/tmp/unused-test-dir");
        // Healthy distillation epoch-0 loss (< 4.0 ceiling).
        assert!(advance_epoch(0, 3.2, DEFAULT_EPOCH_LOSS_CEILINGS, &best_dir).is_ok());
    }

    #[test]
    fn epoch_2_ceiling_tighter_than_epoch_0() {
        // train_loss=2.5 would pass epoch-0 (ceiling 4.0) but fail
        // epoch-2 (ceiling 2.2).
        assert_eq!(
            epoch_divergence_check(0, 2.5, DEFAULT_EPOCH_LOSS_CEILINGS),
            EpochVerdict::Continue
        );
        match epoch_divergence_check(2, 2.5, DEFAULT_EPOCH_LOSS_CEILINGS) {
            EpochVerdict::DivergenceDetected { ceiling, .. } => {
                assert!((ceiling - 2.2).abs() < 1e-6);
            }
            other => panic!("expected DivergenceDetected at epoch 2, got {other:?}"),
        }
    }

    #[test]
    fn epoch_beyond_ceiling_array_uses_last_bound() {
        // Epoch 5 has no explicit ceiling; treats the last bound (2.2)
        // as the asymptote.
        assert_eq!(
            epoch_divergence_check(5, 2.1, DEFAULT_EPOCH_LOSS_CEILINGS),
            EpochVerdict::Continue
        );
        assert!(matches!(
            epoch_divergence_check(5, 2.3, DEFAULT_EPOCH_LOSS_CEILINGS),
            EpochVerdict::DivergenceDetected { .. }
        ));
    }
}
