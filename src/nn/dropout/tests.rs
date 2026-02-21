pub(crate) use super::*;

#[test]
fn test_dropout_eval_mode() {
    let mut dropout = Dropout::new(0.5);
    dropout.eval();

    let x = Tensor::ones(&[10, 10]);
    let y = dropout.forward(&x);

    // In eval mode, output should equal input
    assert_eq!(y.data(), x.data());
}

#[test]
fn test_dropout_train_mode_zeros() {
    let dropout = Dropout::with_seed(0.5, 42);

    let x = Tensor::ones(&[100]);
    let y = dropout.forward(&x);

    // Should have some zeros
    let num_zeros = y.data().iter().filter(|&&v| v == 0.0).count();
    assert!(num_zeros > 0, "Expected some zeros in dropout output");
    assert!(num_zeros < 100, "Expected some non-zeros in dropout output");
}

#[test]
fn test_dropout_scaling() {
    let dropout = Dropout::with_seed(0.5, 42);

    let x = Tensor::ones(&[100]);
    let y = dropout.forward(&x);

    // Non-zero elements should be scaled by 2 (1 / (1 - 0.5))
    for &val in y.data() {
        assert!(val == 0.0 || (val - 2.0).abs() < 1e-5);
    }
}

#[test]
fn test_dropout_zero_probability() {
    let dropout = Dropout::new(0.0);

    let x = Tensor::ones(&[10, 10]);
    let y = dropout.forward(&x);

    // With p=0, output should equal input
    assert_eq!(y.data(), x.data());
}

#[test]
fn test_dropout_expected_value() {
    // With large samples, mean should be approximately preserved
    let dropout = Dropout::with_seed(0.3, 42);

    let x = Tensor::ones(&[10000]);
    let y = dropout.forward(&x);

    let mean: f32 = y.data().iter().sum::<f32>() / y.numel() as f32;

    // Expected value should be close to 1.0 (original value)
    assert!(
        (mean - 1.0).abs() < 0.1,
        "Mean {mean} should be close to 1.0"
    );
}

#[test]
fn test_dropout_reproducible() {
    let dropout1 = Dropout::with_seed(0.5, 42);
    let dropout2 = Dropout::with_seed(0.5, 42);

    let x = Tensor::ones(&[100]);
    let y1 = dropout1.forward(&x);
    let y2 = dropout2.forward(&x);

    assert_eq!(y1.data(), y2.data());
}

#[test]
fn test_dropout_train_eval_toggle() {
    let mut dropout = Dropout::new(0.5);

    assert!(dropout.training());

    dropout.eval();
    assert!(!dropout.training());

    dropout.train();
    assert!(dropout.training());
}

#[test]
#[should_panic(expected = "Dropout probability must be in [0, 1)")]
fn test_dropout_invalid_probability_high() {
    let _ = Dropout::new(1.0);
}

#[test]
#[should_panic(expected = "Dropout probability must be in [0, 1)")]
fn test_dropout_invalid_probability_negative() {
    let _ = Dropout::new(-0.1);
}

// Dropout2d tests

#[test]
fn test_dropout2d_eval_mode() {
    let mut dropout = Dropout2d::new(0.5);
    dropout.eval();

    let x = Tensor::ones(&[4, 64, 8, 8]);
    let y = dropout.forward(&x);

    assert_eq!(y.data(), x.data());
}

#[test]
fn test_dropout2d_shape() {
    let dropout = Dropout2d::with_seed(0.5, 42);

    let x = Tensor::ones(&[4, 64, 8, 8]);
    let y = dropout.forward(&x);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_dropout2d_drops_entire_channels() {
    let dropout = Dropout2d::with_seed(0.5, 42);

    let x = Tensor::ones(&[1, 16, 4, 4]); // 1 batch, 16 channels, 4x4
    let y = dropout.forward(&x);

    // Check that entire channels are either all zeros or all scaled
    let y_data = y.data();
    for c in 0..16 {
        let channel_start = c * 16;
        let channel_end = channel_start + 16;
        let channel_data = &y_data[channel_start..channel_end];

        // Either all zeros or all ~2.0 (scaled by 1/(1-0.5))
        let first_val = channel_data[0];
        for &val in channel_data {
            assert!(
                (val - first_val).abs() < 1e-5,
                "Channel should have uniform values, got {first_val} and {val}"
            );
        }
    }
}

#[test]
fn test_dropout2d_3d_input() {
    let dropout = Dropout2d::with_seed(0.5, 42);

    let x = Tensor::ones(&[4, 64, 100]); // 3D input (e.g., for Conv1d)
    let y = dropout.forward(&x);

    assert_eq!(y.shape(), &[4, 64, 100]);
}

#[test]
fn test_dropout2d_reproducible() {
    let dropout1 = Dropout2d::with_seed(0.5, 42);
    let dropout2 = Dropout2d::with_seed(0.5, 42);

    let x = Tensor::ones(&[4, 16, 8, 8]);
    let y1 = dropout1.forward(&x);
    let y2 = dropout2.forward(&x);

    assert_eq!(y1.data(), y2.data());
}

// AlphaDropout tests

#[test]
fn test_alpha_dropout_eval_mode() {
    let mut dropout = AlphaDropout::new(0.5);
    dropout.eval();

    let x = Tensor::ones(&[100]);
    let y = dropout.forward(&x);

    assert_eq!(y.data(), x.data());
}

#[test]
fn test_alpha_dropout_shape() {
    let dropout = AlphaDropout::with_seed(0.5, 42);

    let x = Tensor::ones(&[32, 64]);
    let y = dropout.forward(&x);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_alpha_dropout_not_zeros() {
    // AlphaDropout doesn't produce zeros, it uses negative saturation value
    let dropout = AlphaDropout::with_seed(0.5, 42);

    let x = Tensor::zeros(&[1000]);
    let y = dropout.forward(&x);

    // Some values should be non-zero (the dropped ones become negative saturation)
    let has_non_zero = y.data().iter().any(|&v| v != 0.0);
    assert!(
        has_non_zero,
        "AlphaDropout should produce non-zero dropped values"
    );
}

#[test]
fn test_alpha_dropout_train_eval_toggle() {
    let mut dropout = AlphaDropout::new(0.5);

    assert!(dropout.training());

    dropout.eval();
    assert!(!dropout.training());

    dropout.train();
    assert!(dropout.training());
}

// DropBlock tests

#[test]
fn test_dropblock_creation() {
    let db = DropBlock::new(3, 0.1);
    assert_eq!(db.block_size(), 3);
    assert_eq!(db.p(), 0.1);
}

#[test]
fn test_dropblock_eval_mode() {
    let mut db = DropBlock::new(3, 0.5);
    db.eval();

    let x = Tensor::ones(&[2, 4, 8, 8]);
    let y = db.forward(&x);

    assert_eq!(y.data(), x.data());
}

#[test]
fn test_dropblock_train_mode() {
    let db = DropBlock::with_seed(3, 0.3, 42);

    let x = Tensor::ones(&[1, 2, 8, 8]);
    let y = db.forward(&x);

    assert_eq!(y.shape(), x.shape());
    // Should have some zeros (blocks dropped)
    let num_zeros = y.data().iter().filter(|&&v| v == 0.0).count();
    assert!(num_zeros > 0);
}

#[test]
fn test_dropblock_train_eval_toggle() {
    let mut db = DropBlock::new(3, 0.1);

    assert!(db.training());
    db.eval();
    assert!(!db.training());
    db.train();
    assert!(db.training());
}

#[test]
fn test_dropblock_non_4d_fallback() {
    let db = DropBlock::with_seed(3, 0.3, 42);

    let x = Tensor::ones(&[10, 10]); // 2D, not 4D
    let y = db.forward(&x);

    assert_eq!(y.shape(), x.shape());
}

#[test]
fn test_dropblock_zero_prob() {
    let db = DropBlock::new(3, 0.0);

    let x = Tensor::ones(&[1, 2, 8, 8]);
    let y = db.forward(&x);

    assert_eq!(y.data(), x.data());
}

// DropConnect tests
#[test]
fn test_dropconnect_creation() {
    let dc = DropConnect::new(0.5);
    assert_eq!(dc.probability(), 0.5);
    assert!(dc.training());
}

#[test]
fn test_dropconnect_eval_mode() {
    let mut dc = DropConnect::new(0.5);
    dc.eval();

    let x = Tensor::ones(&[10, 10]);
    let y = dc.forward(&x);

    assert_eq!(y.data(), x.data());
}

#[test]
fn test_dropconnect_train_mode() {
    let dc = DropConnect::with_seed(0.5, 42);

    let x = Tensor::ones(&[100]);
    let y = dc.forward(&x);

    let num_zeros = y.data().iter().filter(|&&v| v == 0.0).count();
    assert!(num_zeros > 0);
    assert!(num_zeros < 100);
}

#[test]
fn test_dropconnect_apply_to_weights() {
    let dc = DropConnect::with_seed(0.5, 42);

    let weights = Tensor::ones(&[4, 4]);
    let masked = dc.apply_to_weights(&weights);

    assert_eq!(masked.shape(), weights.shape());
    let num_zeros = masked.data().iter().filter(|&&v| v == 0.0).count();
    assert!(num_zeros > 0);
}

#[test]
fn test_dropconnect_zero_prob() {
    let dc = DropConnect::new(0.0);

    let x = Tensor::ones(&[10]);
    let y = dc.forward(&x);

    assert_eq!(y.data(), x.data());
}

#[test]
fn test_dropconnect_train_eval_toggle() {
    let mut dc = DropConnect::new(0.5);

    assert!(dc.training());
    dc.eval();
    assert!(!dc.training());
    dc.train();
    assert!(dc.training());
}

// ==========================================================================
// Additional Coverage Tests
// ==========================================================================

#[test]
fn test_dropout_probability() {
    let dropout = Dropout::new(0.3);
    assert!((dropout.probability() - 0.3).abs() < 1e-6);
}

#[test]
fn test_dropout_debug() {
    let dropout = Dropout::new(0.5);
    let debug_str = format!("{:?}", dropout);
    assert!(debug_str.contains("Dropout"));
    assert!(debug_str.contains("0.5"));
}

#[test]
fn test_dropout2d_probability() {
    let dropout = Dropout2d::new(0.4);
    assert!((dropout.probability() - 0.4).abs() < 1e-6);
}

#[test]
fn test_dropout2d_zero_probability() {
    let dropout = Dropout2d::new(0.0);
    let x = Tensor::ones(&[2, 4, 8, 8]);
    let y = dropout.forward(&x);
    assert_eq!(y.data(), x.data());
}

#[test]
fn test_dropout2d_debug() {
    let dropout = Dropout2d::new(0.5);
    let debug_str = format!("{:?}", dropout);
    assert!(debug_str.contains("Dropout2d"));
}

#[test]
fn test_dropout2d_train_eval_toggle() {
    let mut dropout = Dropout2d::new(0.5);
    assert!(dropout.training());
    dropout.eval();
    assert!(!dropout.training());
    dropout.train();
    assert!(dropout.training());
}

#[test]
fn test_alpha_dropout_zero_probability() {
    let dropout = AlphaDropout::new(0.0);
    let x = Tensor::ones(&[100]);
    let y = dropout.forward(&x);
    assert_eq!(y.data(), x.data());
}

#[test]
fn test_alpha_dropout_debug() {
    let dropout = AlphaDropout::new(0.5);
    let debug_str = format!("{:?}", dropout);
    assert!(debug_str.contains("AlphaDropout"));
}

#[test]
fn test_dropblock_debug() {
    let db = DropBlock::new(3, 0.2);
    let debug_str = format!("{:?}", db);
    assert!(debug_str.contains("DropBlock"));
    assert!(debug_str.contains("block_size"));
}

#[test]
fn test_dropconnect_debug() {
    let dc = DropConnect::new(0.5);
    let debug_str = format!("{:?}", dc);
    assert!(debug_str.contains("DropConnect"));
}

#[test]
fn test_dropconnect_apply_to_weights_eval_mode() {
    let mut dc = DropConnect::new(0.5);
    dc.eval();

    let weights = Tensor::ones(&[4, 4]);
    let masked = dc.apply_to_weights(&weights);

    // In eval mode, should return unchanged
    assert_eq!(masked.data(), weights.data());
}

#[path = "tests_dropconnect.rs"]
mod tests_dropconnect;
