
/// Apply Classifier-Free Guidance
///
/// Combines conditional and unconditional logits using the CFG formula:
/// output = unconditional + scale * (conditional - unconditional)
///
/// # Arguments
///
/// * `conditional_logits` - Logits from the model with the prompt
/// * `unconditional_logits` - Logits from the model with negative/empty prompt
/// * `scale` - Guidance scale
///
/// # Returns
///
/// Guided logits
///
/// # Errors
///
/// Returns error if conditional and unconditional logits have different shapes
pub fn apply_cfg(
    conditional_logits: &Tensor<f32>,
    unconditional_logits: &Tensor<f32>,
    scale: f32,
) -> Result<Tensor<f32>> {
    if conditional_logits.shape() != unconditional_logits.shape() {
        return Err(crate::error::RealizarError::ShapeMismatch {
            expected: conditional_logits.shape().to_vec(),
            actual: unconditional_logits.shape().to_vec(),
        });
    }

    let cond = conditional_logits.data();
    let uncond = unconditional_logits.data();

    // CFG formula: uncond + scale * (cond - uncond)
    let guided: Vec<f32> = cond
        .iter()
        .zip(uncond.iter())
        .map(|(&c, &u)| u + scale * (c - u))
        .collect();

    Tensor::from_vec(conditional_logits.shape().to_vec(), guided)
}
