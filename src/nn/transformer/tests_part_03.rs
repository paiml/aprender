
#[test]
fn test_transformer_decoder_layer_parameters_mut() {
    let mut layer = TransformerDecoderLayer::new(64, 8, 256);
    let params = layer.parameters_mut();
    assert!(params.len() > 0);
}

#[test]
fn test_add_tensors_shape_3d() {
    let a = Tensor::ones(&[2, 3, 4]);
    let b = Tensor::ones(&[2, 3, 4]);
    let c = add_tensors(&a, &b);
    assert_eq!(c.shape(), &[2, 3, 4]);
}

#[test]
fn test_softmax_last_dim_single_row() {
    let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let y = softmax_last_dim(&x);
    let sum: f32 = y.data().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_generate_causal_mask_small() {
    let mask = generate_causal_mask(2);
    assert_eq!(mask.shape(), &[2, 2]);
    // Lower triangle and diagonal should be 0
    assert_eq!(mask.data()[0], 0.0); // [0,0]
    assert_eq!(mask.data()[2], 0.0); // [1,0]
    assert_eq!(mask.data()[3], 0.0); // [1,1]
                                     // Upper triangle should be -inf
    assert!(mask.data()[1].is_infinite()); // [0,1]
}

#[test]
fn test_rope_head_dim() {
    let rope = RotaryPositionEmbedding::new(16, 100);
    assert_eq!(rope.head_dim(), 16);
}

#[test]
fn test_rope_single_position() {
    let rope = RotaryPositionEmbedding::new(8, 128);
    let x = Tensor::ones(&[1, 1, 2, 8]);
    let positions = vec![42_usize];
    let y = rope.apply(&x, &positions);
    assert_eq!(y.shape(), &[1, 1, 2, 8]);
}
