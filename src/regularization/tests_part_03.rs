
#[test]
fn test_cutmix_apply_empty_box() {
    let params = CutMixParams {
        lambda: 1.0,
        x1: 0,
        y1: 0,
        x2: 0,
        y2: 0,
    };
    let img1 = vec![1.0; 16];
    let img2 = vec![2.0; 16];
    let result = params.apply(&img1, &img2, 1, 4, 4);
    // With empty box, result should equal img1
    assert_eq!(result, img1);
}

#[test]
fn test_specaugment_very_large_masks() {
    // Test when mask params are larger than spec dimensions
    let sa = SpecAugment::with_params(2, 100, 2, 100); // Large mask params
    let spec = vec![1.0; 25]; // Small 5x5 spec
    let result = sa.apply(&spec, 5, 5);
    assert_eq!(result.len(), 25);
}

#[test]
fn test_specaugment_single_element() {
    let sa = SpecAugment::with_params(1, 1, 1, 1);
    let spec = vec![1.0]; // 1x1 spec
    let result = sa.apply(&spec, 1, 1);
    assert_eq!(result.len(), 1);
}

#[test]
fn test_randaugment_empty_augmentations() {
    let ra = RandAugment::new(5, 15).with_augmentations(vec![]);
    let sampled = ra.sample_augmentations();
    // With empty augmentations list, nothing can be sampled
    assert!(sampled.is_empty());
}

#[test]
fn test_cross_entropy_uniform_logits() {
    let logits = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let loss = cross_entropy_with_smoothing(&logits, 0, 0.1);
    // With uniform logits, each class has equal probability
    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_cross_entropy_extreme_logits() {
    let logits = Vector::from_slice(&[100.0, -100.0, -100.0]);
    let loss = cross_entropy_with_smoothing(&logits, 0, 0.0);
    // First class should have probability ~1.0, loss should be very small
    assert!(loss < 0.001);
}

#[test]
fn test_label_smoothing_boundary_epsilon() {
    // Test with epsilon at boundary (0.0)
    let ls = LabelSmoothing::new(0.0);
    let label = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let smoothed = ls.smooth(&label);
    // With epsilon=0, should return original labels
    assert_eq!(smoothed.as_slice(), &[1.0, 0.0, 0.0]);
}

#[test]
fn test_label_smoothing_high_epsilon() {
    let ls = LabelSmoothing::new(0.9);
    let smoothed = ls.smooth_index(0, 4);
    let sum: f32 = smoothed.as_slice().iter().sum();
    // Should still sum to 1
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_stochastic_depth_edge_drop_prob() {
    // Test with drop_prob = 0.0 in training mode
    let sd = StochasticDepth::new(0.0, DropMode::Row);
    for _ in 0..10 {
        assert!(sd.should_keep(true));
    }
}

#[test]
fn test_stochastic_depth_linear_decay_extremes() {
    // First layer (depth=0) should have survival=1.0
    let survival_first = StochasticDepth::linear_decay(0, 10, 0.5);
    assert!((survival_first - 1.0).abs() < 1e-6);

    // Last layer should have survival = 1 - max_drop
    let survival_last = StochasticDepth::linear_decay(10, 10, 0.5);
    assert!((survival_last - 0.5).abs() < 1e-6);
}

#[test]
fn test_rdrop_kl_divergence_with_zeros() {
    let rdrop = RDrop::new(1.0);
    // Test with very small probabilities (should use epsilon internally)
    let p = vec![0.001, 0.999];
    let q = vec![0.999, 0.001];
    let kl = rdrop.kl_divergence(&p, &q);
    assert!(kl > 0.0);
    assert!(kl.is_finite());
}

#[test]
fn test_specaugment_freq_mask_no_masks() {
    let sa = SpecAugment::with_params(0, 0, 0, 0);
    let spec = vec![1.0; 100];
    let result = sa.freq_mask(&spec, 10, 10);
    // With 0 freq masks, result should be unchanged
    assert_eq!(result, spec);
}

#[test]
fn test_specaugment_time_mask_no_masks() {
    let sa = SpecAugment::with_params(0, 0, 0, 0);
    let spec = vec![1.0; 100];
    let result = sa.time_mask(&spec, 10, 10);
    // With 0 time masks, result should be unchanged
    assert_eq!(result, spec);
}

#[test]
fn test_randaugment_brightness_zero_magnitude() {
    let ra = RandAugment::new(1, 0); // Zero magnitude
    let image = vec![0.5; 16];
    let result = ra.apply_single(&image, AugmentationType::Brightness, 4, 4);
    // With magnitude 0, factor = 1 + (0 - 0.5) * 2 = 0
    // All values should be clamped to 0
    for v in &result {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn test_randaugment_contrast_varying_image() {
    let ra = RandAugment::new(1, 15);
    let image: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
    let result = ra.apply_single(&image, AugmentationType::Contrast, 4, 4);
    // Verify clamping
    for v in &result {
        assert!(*v >= 0.0 && *v <= 1.0);
    }
}

#[test]
fn test_randaugment_translate_zero_shift() {
    // With magnitude = 15/30 = 0.5, shift should be 0
    let ra = RandAugment::new(1, 15);
    let image = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let result_x = ra.apply_single(&image, AugmentationType::TranslateX, 3, 3);
    let result_y = ra.apply_single(&image, AugmentationType::TranslateY, 3, 3);
    // With zero shift, should be unchanged
    assert_eq!(result_x, image);
    assert_eq!(result_y, image);
}

#[test]
fn test_cutmix_params_full_image() {
    // Test CutMix covering entire image
    let params = CutMixParams {
        lambda: 0.0, // All from img2
        x1: 0,
        y1: 0,
        x2: 4,
        y2: 4,
    };
    let img1 = vec![1.0; 16];
    let img2 = vec![2.0; 16];
    let result = params.apply(&img1, &img2, 1, 4, 4);
    // Entire image should be from img2
    assert_eq!(result, img2);
}

#[test]
fn test_mixup_large_alpha() {
    // Large alpha should still produce valid lambda in [0, 1]
    let mixup = Mixup::new(10.0);
    for _ in 0..20 {
        let lambda = mixup.sample_lambda();
        assert!(
            (0.0..=1.0).contains(&lambda),
            "Lambda out of range: {lambda}"
        );
    }
}

#[test]
fn test_all_augmentation_types_have_copy() {
    let aug = AugmentationType::Solarize;
    let copied = aug;
    assert_eq!(copied, AugmentationType::Solarize);
}

#[test]
fn test_drop_mode_debug() {
    let batch = DropMode::Batch;
    let row = DropMode::Row;
    let batch_debug = format!("{batch:?}");
    let row_debug = format!("{row:?}");
    assert!(batch_debug.contains("Batch"));
    assert!(row_debug.contains("Row"));
}
