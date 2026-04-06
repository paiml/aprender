//! Epoch summary computation from training snapshots.

use super::super::state::TrainingSnapshot;

#[derive(Debug, Clone)]
pub struct EpochSummary {
    pub epoch: usize,
    pub avg_loss: f32,
    pub min_loss: f32,
    pub max_loss: f32,
    pub end_loss: f32,
    pub avg_grad: f32,
    pub lr: f32,
    pub tokens_per_sec: f32,
}

pub fn compute_epoch_summaries(snapshot: &TrainingSnapshot) -> Vec<EpochSummary> {
    if snapshot.steps_per_epoch == 0 || snapshot.loss_history.is_empty() {
        return Vec::new();
    }

    let steps = snapshot.steps_per_epoch;
    let mut summaries = Vec::new();

    for (epoch_idx, chunk) in snapshot.loss_history.chunks(steps).enumerate() {
        let valid: Vec<f32> = chunk.iter().copied().filter(|v| v.is_finite()).collect();
        if valid.is_empty() {
            continue;
        }

        // valid.len() <= steps_per_epoch, bounded by training config, safe for f32
        let valid_count = valid.len().min(usize::from(u16::MAX)) as f32;
        let avg_loss = valid.iter().sum::<f32>() / valid_count;
        let min_loss = valid.iter().copied().fold(f32::INFINITY, f32::min);
        let max_loss = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let end_loss = *valid.last().unwrap_or(&0.0);

        let lr = if snapshot.lr_history.is_empty() {
            snapshot.learning_rate
        } else {
            let lr_start = epoch_idx * steps;
            let lr_end = (lr_start + steps).min(snapshot.lr_history.len());
            if lr_start < snapshot.lr_history.len() {
                let lr_span = (lr_end - lr_start).max(1).min(usize::from(u16::MAX)) as f32;
                snapshot.lr_history[lr_start..lr_end].iter().sum::<f32>() / lr_span
            } else {
                snapshot.learning_rate
            }
        };

        summaries.push(EpochSummary {
            epoch: epoch_idx + 1,
            avg_loss,
            min_loss,
            max_loss,
            end_loss,
            avg_grad: snapshot.gradient_norm.max(0.0),
            lr,
            tokens_per_sec: snapshot.tokens_per_second.max(0.0),
        });
    }
    summaries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epoch_summaries() {
        let snapshot = TrainingSnapshot {
            steps_per_epoch: 4,
            loss_history: vec![10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5],
            ..Default::default()
        };

        let summaries = compute_epoch_summaries(&snapshot);
        assert_eq!(summaries.len(), 3);
        assert!((summaries[0].avg_loss - 9.25).abs() < 0.01);
        assert!((summaries[0].min_loss - 8.5).abs() < 0.01);
        assert!((summaries[0].max_loss - 10.0).abs() < 0.01);
    }
}
