//! Demo batch creation for testing

use crate::train::Batch;
use crate::Tensor;

/// Create demo batches for testing when no data file is available
pub fn create_demo_batches(batch_size: usize) -> Vec<Batch> {
    let num_batches = 2.max(8 / batch_size.max(1));
    (0..num_batches)
        .map(|i| {
            let input_data: Vec<f32> =
                (0..batch_size * 4).map(|j| ((i * batch_size + j) as f32) * 0.1).collect();
            let target_data: Vec<f32> =
                (0..batch_size * 4).map(|j| ((i * batch_size + j + 1) as f32) * 0.1).collect();
            Batch::new(Tensor::from_vec(input_data, false), Tensor::from_vec(target_data, false))
        })
        .collect()
}
