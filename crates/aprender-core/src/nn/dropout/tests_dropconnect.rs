use super::*;

#[test]
fn test_dropconnect_apply_to_weights_zero_prob() {
    let dc = DropConnect::new(0.0);

    let weights = Tensor::ones(&[4, 4]);
    let masked = dc.apply_to_weights(&weights);

    assert_eq!(masked.data(), weights.data());
}

#[test]
#[should_panic(expected = "Dropout probability must be in [0, 1)")]
fn test_dropout2d_invalid_probability() {
    let _ = Dropout2d::new(1.5);
}

#[test]
#[should_panic(expected = "Dropout probability must be in [0, 1)")]
fn test_alpha_dropout_invalid_probability() {
    let _ = AlphaDropout::new(1.0);
}

#[test]
#[should_panic(expected = "Drop probability must be in [0, 1)")]
fn test_dropblock_invalid_probability() {
    let _ = DropBlock::new(3, 1.0);
}

#[test]
#[should_panic(expected = "Block size must be > 0")]
fn test_dropblock_invalid_block_size() {
    let _ = DropBlock::new(0, 0.5);
}

#[test]
#[should_panic(expected = "Drop probability must be in [0, 1)")]
fn test_dropconnect_invalid_probability() {
    let _ = DropConnect::new(1.0);
}

#[test]
fn test_dropout_multidim() {
    let dropout = Dropout::with_seed(0.5, 42);
    let x = Tensor::ones(&[2, 3, 4, 5]);
    let y = dropout.forward(&x);
    assert_eq!(y.shape(), &[2, 3, 4, 5]);
}

#[test]
fn test_alpha_dropout_reproducible() {
    let dropout1 = AlphaDropout::with_seed(0.5, 42);
    let dropout2 = AlphaDropout::with_seed(0.5, 42);

    let x = Tensor::ones(&[100]);
    let y1 = dropout1.forward(&x);
    let y2 = dropout2.forward(&x);

    assert_eq!(y1.data(), y2.data());
}

#[test]
fn test_dropblock_small_spatial() {
    // Test with spatial dimensions smaller than block_size
    let db = DropBlock::with_seed(5, 0.3, 42);
    let x = Tensor::ones(&[1, 2, 3, 3]); // 3x3 spatial, block_size=5

    let y = db.forward(&x);
    assert_eq!(y.shape(), &[1, 2, 3, 3]);
}

#[test]
fn test_dropconnect_reproducible() {
    let dc1 = DropConnect::with_seed(0.5, 42);
    let dc2 = DropConnect::with_seed(0.5, 42);

    let x = Tensor::ones(&[100]);
    let y1 = dc1.forward(&x);
    let y2 = dc2.forward(&x);

    assert_eq!(y1.data(), y2.data());
}

#[test]
fn test_dropblock_larger_batch() {
    let db = DropBlock::with_seed(2, 0.2, 42);
    let x = Tensor::ones(&[4, 8, 16, 16]);
    let y = db.forward(&x);
    assert_eq!(y.shape(), &[4, 8, 16, 16]);
}
