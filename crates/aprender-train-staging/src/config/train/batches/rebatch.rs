//! Batch re-batching utilities

use crate::train::Batch;
use crate::Tensor;

/// Re-batch data into specified batch size
#[allow(dead_code)]
pub fn rebatch(batches: Vec<Batch>, batch_size: usize) -> Vec<Batch> {
    // Flatten all data
    let all_inputs: Vec<f32> =
        batches.iter().flat_map(|b| b.inputs.data().iter().copied()).collect();
    let all_targets: Vec<f32> =
        batches.iter().flat_map(|b| b.targets.data().iter().copied()).collect();

    if all_inputs.is_empty() {
        return Vec::new();
    }

    // Determine feature dimensions from first batch
    let input_dim = batches[0].inputs.len();
    let target_dim = batches[0].targets.len();

    // Re-batch
    let num_examples = all_inputs.len() / input_dim;
    let mut new_batches = Vec::new();

    for chunk_start in (0..num_examples).step_by(batch_size) {
        let chunk_end = (chunk_start + batch_size).min(num_examples);
        let input_start = chunk_start * input_dim;
        let input_end = chunk_end * input_dim;
        let target_start = chunk_start * target_dim;
        let target_end = chunk_end * target_dim;

        new_batches.push(Batch::new(
            Tensor::from_vec(all_inputs[input_start..input_end].to_vec(), false),
            Tensor::from_vec(all_targets[target_start..target_end].to_vec(), false),
        ));
    }

    new_batches
}
