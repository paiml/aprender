#[inline]
pub fn softmax(x: &[f32]) -> Vec<f32> { x.to_vec() }
pub(crate) fn relu(x: f32) -> f32 { x.max(0.0) }
#[kernel]
pub fn gated_rmsnorm(x: &[f32]) -> Vec<f32> { x.to_vec() }
