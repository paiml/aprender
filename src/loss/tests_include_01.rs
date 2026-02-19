
#[test]
fn test_cross_entropy_loss_perfect_prediction() {
    // Large logit for the correct class → loss should be small
    let logits = Vector::from_slice(&[100.0, 0.0, 0.0]);
    let targets = Vector::from_slice(&[1.0, 0.0, 0.0]);

    let loss = cross_entropy_loss(&logits, &targets);
    assert!(loss < 0.01);
}

#[test]
fn test_cross_entropy_loss_wrong_prediction() {
    // Large logit for wrong class → loss should be large
    let logits = Vector::from_slice(&[0.0, 100.0, 0.0]);
    let targets = Vector::from_slice(&[1.0, 0.0, 0.0]); // target is class 0

    let loss = cross_entropy_loss(&logits, &targets);
    assert!(loss > 50.0);
}

#[test]
fn test_cross_entropy_loss_numerical_stability() {
    // Very large values that could overflow without max subtraction
    let logits = Vector::from_slice(&[1000.0, 1001.0, 1002.0]);
    let targets = Vector::from_slice(&[1.0, 0.0, 0.0]);

    let loss = cross_entropy_loss(&logits, &targets);
    assert!(loss.is_finite());
    assert!(loss > 0.0);
}

#[test]
fn test_cross_entropy_loss_soft_labels() {
    let logits = Vector::from_slice(&[2.0, 1.0]);
    let soft_targets = Vector::from_slice(&[0.8, 0.2]); // soft labels

    let loss = cross_entropy_loss(&logits, &soft_targets);
    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
#[should_panic(expected = "same length")]
fn test_cross_entropy_loss_mismatched_lengths() {
    let logits = Vector::from_slice(&[1.0, 2.0]);
    let targets = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let _ = cross_entropy_loss(&logits, &targets);
}

#[test]
#[should_panic(expected = "cannot be empty")]
fn test_cross_entropy_loss_empty() {
    let logits = Vector::from_slice(&[]);
    let targets = Vector::from_slice(&[]);
    let _ = cross_entropy_loss(&logits, &targets);
}

// ==========================================================================
// Loss Send + Sync Tests (GH-279)
// ==========================================================================

#[test]
fn test_loss_trait_send_sync() {
    // Verify Loss objects can be shared across threads
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MSELoss>();
    assert_send_sync::<MAELoss>();
    assert_send_sync::<HuberLoss>();
}

#[test]
fn test_loss_trait_arc() {
    use std::sync::Arc;

    let loss: Arc<dyn Loss> = Arc::new(MSELoss);
    let y_true = Vector::from_slice(&[1.0, 2.0]);
    let y_pred = Vector::from_slice(&[1.5, 2.5]);

    let result = loss.compute(&y_pred, &y_true);
    assert!(result > 0.0);
}

#[path = "tests_part_02.rs"]
mod tests_part_02;
