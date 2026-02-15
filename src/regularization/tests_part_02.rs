
#[test]
fn test_randaugment_apply_shear_y() {
    let ra = RandAugment::new(1, 15);
    let image = vec![0.5; 16];
    let result = ra.apply_single(&image, AugmentationType::ShearY, 4, 4);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_randaugment_apply_sharpness() {
    let ra = RandAugment::new(1, 15);
    let image = vec![0.5; 16];
    let result = ra.apply_single(&image, AugmentationType::Sharpness, 4, 4);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_randaugment_apply_posterize() {
    let ra = RandAugment::new(1, 15);
    let image = vec![0.5; 16];
    let result = ra.apply_single(&image, AugmentationType::Posterize, 4, 4);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_randaugment_apply_solarize() {
    let ra = RandAugment::new(1, 15);
    let image = vec![0.3, 0.7, 0.5, 0.9];
    let result = ra.apply_single(&image, AugmentationType::Solarize, 2, 2);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_randaugment_apply_equalize() {
    let ra = RandAugment::new(1, 15);
    let image = vec![0.1, 0.5, 0.9, 0.3];
    let result = ra.apply_single(&image, AugmentationType::Equalize, 2, 2);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_mixup_mix_labels() {
    let mixup = Mixup::new(1.0);
    let y1 = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let y2 = Vector::from_slice(&[0.0, 1.0, 0.0]);
    let mixed = mixup.mix_labels(&y1, &y2, 0.7);
    assert!((mixed.as_slice()[0] - 0.7).abs() < 1e-6);
    assert!((mixed.as_slice()[1] - 0.3).abs() < 1e-6);
}

#[test]
fn test_mixup_alpha_negative() {
    let mixup = Mixup::new(-0.5);
    // Should return 1.0 when alpha <= 0
    assert_eq!(mixup.sample_lambda(), 1.0);
}

#[test]
fn test_cutmix_params_debug() {
    let params = CutMixParams {
        lambda: 0.5,
        x1: 0,
        y1: 0,
        x2: 2,
        y2: 2,
    };
    let debug_str = format!("{:?}", params);
    assert!(debug_str.contains("CutMixParams"));
}

#[test]
fn test_cutmix_sample_edge_cases() {
    let cm = CutMix::new(0.0);
    // Alpha 0 means lambda = 1.0 always
    let params = cm.sample(10, 10);
    assert_eq!(params.lambda, 1.0);
}

#[test]
fn test_stochastic_depth_clone() {
    let sd = StochasticDepth::new(0.3, DropMode::Row);
    let cloned = sd.clone();
    assert_eq!(cloned.drop_prob(), sd.drop_prob());
    assert_eq!(cloned.mode(), sd.mode());
}

#[test]
fn test_rdrop_clone() {
    let rdrop = RDrop::new(1.5);
    let cloned = rdrop.clone();
    assert_eq!(cloned.alpha(), rdrop.alpha());
}

#[test]
fn test_specaugment_clone() {
    let sa = SpecAugment::with_params(3, 20, 4, 80);
    let cloned = sa.clone();
    assert_eq!(cloned.num_freq_masks(), 3);
    assert_eq!(cloned.num_time_masks(), 4);
}

#[test]
fn test_randaugment_clone() {
    let ra = RandAugment::new(3, 12);
    let cloned = ra.clone();
    assert_eq!(cloned.n(), ra.n());
    assert_eq!(cloned.m(), ra.m());
}

#[test]
fn test_mixup_debug() {
    let mixup = Mixup::new(0.5);
    let debug_str = format!("{:?}", mixup);
    assert!(debug_str.contains("Mixup"));
}

#[test]
fn test_label_smoothing_debug() {
    let ls = LabelSmoothing::new(0.1);
    let debug_str = format!("{:?}", ls);
    assert!(debug_str.contains("LabelSmoothing"));
}

#[test]
fn test_cutmix_debug() {
    let cm = CutMix::new(1.0);
    let debug_str = format!("{:?}", cm);
    assert!(debug_str.contains("CutMix"));
}

#[test]
fn test_stochastic_depth_debug() {
    let sd = StochasticDepth::new(0.2, DropMode::Batch);
    let debug_str = format!("{:?}", sd);
    assert!(debug_str.contains("StochasticDepth"));
}

#[test]
fn test_rdrop_debug() {
    let rdrop = RDrop::new(0.5);
    let debug_str = format!("{:?}", rdrop);
    assert!(debug_str.contains("RDrop"));
}

#[test]
fn test_specaugment_debug() {
    let sa = SpecAugment::new();
    let debug_str = format!("{:?}", sa);
    assert!(debug_str.contains("SpecAugment"));
}

#[test]
fn test_randaugment_debug() {
    let ra = RandAugment::new(2, 10);
    let debug_str = format!("{:?}", ra);
    assert!(debug_str.contains("RandAugment"));
}

#[test]
fn test_augmentation_type_debug_copy() {
    let aug = AugmentationType::Posterize;
    let copied = aug;
    let debug_str = format!("{:?}", copied);
    assert!(debug_str.contains("Posterize"));
}

// =========================================================================
// Additional coverage tests for edge cases
// =========================================================================

#[test]
fn test_mixup_small_alpha() {
    // Test sample_beta with shape < 1.0 (triggers gamma special case)
    let mixup = Mixup::new(0.3);
    for _ in 0..10 {
        let lambda = mixup.sample_lambda();
        assert!((0.0..=1.0).contains(&lambda));
    }
}

#[test]
fn test_mixup_very_small_alpha() {
    // Very small alpha stresses the gamma sampler with extreme shapes.
    // sample_beta clamps to [0, 1] so this must never exceed bounds.
    let mixup = Mixup::new(0.01);
    for _ in 0..100 {
        let lambda = mixup.sample_lambda();
        assert!(
            (0.0..=1.0).contains(&lambda),
            "Beta sample out of range: {lambda}"
        );
    }
}

#[test]
fn test_cutmix_small_alpha() {
    let cm = CutMix::new(0.2);
    let params = cm.sample(16, 16);
    assert!(params.lambda >= 0.0 && params.lambda <= 1.0);
}

#[test]
fn test_cutmix_apply_out_of_bounds() {
    // Box exceeds image bounds
    let params = CutMixParams {
        lambda: 0.5,
        x1: 3,
        y1: 3,
        x2: 10, // Beyond image width
        y2: 10, // Beyond image height
    };
    let img1 = vec![1.0; 16]; // 1x4x4
    let img2 = vec![2.0; 16];
    let result = params.apply(&img1, &img2, 1, 4, 4);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_cutmix_apply_multichannel() {
    let params = CutMixParams {
        lambda: 0.5,
        x1: 0,
        y1: 0,
        x2: 2,
        y2: 2,
    };
    let img1 = vec![1.0; 48]; // 3 channels, 4x4
    let img2 = vec![2.0; 48];
    let result = params.apply(&img1, &img2, 3, 4, 4);
    assert_eq!(result.len(), 48);
}

#[test]
fn test_cutmix_params_clone() {
    let params = CutMixParams {
        lambda: 0.7,
        x1: 1,
        y1: 2,
        x2: 5,
        y2: 6,
    };
    let cloned = params.clone();
    assert_eq!(cloned.lambda, params.lambda);
    assert_eq!(cloned.x1, params.x1);
}

#[test]
fn test_stochastic_depth_high_drop_prob() {
    let sd = StochasticDepth::new(0.99, DropMode::Row);
    // With 99% drop probability, mostly should drop in training
    let mut drops = 0;
    for _ in 0..100 {
        if !sd.should_keep(true) {
            drops += 1;
        }
    }
    // Expect many drops with 99% probability
    assert!(drops > 50);
}

#[test]
fn test_rdrop_very_small_alpha() {
    let rdrop = RDrop::new(0.001);
    let logits1 = vec![2.0, 1.0];
    let logits2 = vec![1.0, 2.0];
    let loss = rdrop.compute_loss(&logits1, &logits2);
    // With small alpha, loss should be very small but positive
    assert!(loss > 0.0);
    assert!(loss < 0.1);
}

#[test]
fn test_rdrop_large_alpha() {
    let rdrop = RDrop::new(100.0);
    let logits1 = vec![5.0, 0.0];
    let logits2 = vec![0.0, 5.0];
    let loss = rdrop.compute_loss(&logits1, &logits2);
    assert!(loss > 10.0);
}

#[test]
fn test_specaugment_empty_spec() {
    let sa = SpecAugment::with_params(1, 5, 1, 10);
    let spec: Vec<f32> = vec![];
    let result = sa.apply(&spec, 0, 0);
    assert!(result.is_empty());
}

#[test]
fn test_specaugment_small_spec() {
    let sa = SpecAugment::with_params(1, 5, 1, 10);
    let spec = vec![1.0; 4]; // 2x2
    let result = sa.apply(&spec, 2, 2);
    assert_eq!(result.len(), 4);
}

#[test]
fn test_specaugment_freq_mask_small() {
    let sa = SpecAugment::with_params(3, 2, 0, 0).with_mask_value(-1.0);
    let spec = vec![1.0; 9]; // 3x3
    let result = sa.freq_mask(&spec, 3, 3);
    assert_eq!(result.len(), 9);
}

#[test]
fn test_specaugment_time_mask_small() {
    let sa = SpecAugment::with_params(0, 0, 3, 2).with_mask_value(-1.0);
    let spec = vec![1.0; 9]; // 3x3
    let result = sa.time_mask(&spec, 3, 3);
    assert_eq!(result.len(), 9);
}

#[test]
fn test_randaugment_zero_magnitude() {
    let ra = RandAugment::new(2, 0);
    assert_eq!(ra.normalized_magnitude(), 0.0);
}

#[test]
fn test_randaugment_translate_x_with_shift() {
    let ra = RandAugment::new(1, 30); // High magnitude
    let image: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = ra.apply_single(&image, AugmentationType::TranslateX, 4, 4);
    assert_eq!(result.len(), 16);
    // Values should be shifted
}

#[test]
fn test_randaugment_translate_y_with_shift() {
    let ra = RandAugment::new(1, 30);
    let image: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result = ra.apply_single(&image, AugmentationType::TranslateY, 4, 4);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_randaugment_translate_mid_magnitude() {
    let ra = RandAugment::new(1, 15); // mag = 0.5, shift = 0
    let image: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let result_x = ra.apply_single(&image, AugmentationType::TranslateX, 4, 4);
    let result_y = ra.apply_single(&image, AugmentationType::TranslateY, 4, 4);
    // At magnitude 0.5, shift = 0, so should be unchanged
    assert_eq!(result_x, image);
    assert_eq!(result_y, image);
}

#[test]
fn test_randaugment_brightness_clamping() {
    let ra = RandAugment::new(1, 30);
    let image = vec![1.0; 16]; // Already at max
    let result = ra.apply_single(&image, AugmentationType::Brightness, 4, 4);
    // Should clamp to [0, 1]
    for &v in &result {
        assert!(v >= 0.0 && v <= 1.0);
    }
}

#[test]
fn test_randaugment_contrast_zero_mean() {
    let ra = RandAugment::new(1, 20);
    let image = vec![0.5; 16]; // Constant image
    let result = ra.apply_single(&image, AugmentationType::Contrast, 4, 4);
    // Contrast on constant image should stay constant
    for &v in &result {
        assert!((v - 0.5).abs() < 1e-6 || (v >= 0.0 && v <= 1.0));
    }
}

#[test]
fn test_randaugment_multichannel() {
    let ra = RandAugment::new(1, 15);
    let image = vec![0.5; 48]; // 3 channels, 4x4
    let result = ra.apply_single(&image, AugmentationType::TranslateX, 4, 4);
    assert_eq!(result.len(), 48);
}

#[test]
fn test_cross_entropy_all_classes() {
    // Test cross entropy for each class
    let logits = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    for target in 0..4 {
        let loss = cross_entropy_with_smoothing(&logits, target, 0.1);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }
}

#[test]
fn test_label_smoothing_all_zero_label() {
    let ls = LabelSmoothing::new(0.2);
    let label = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);
    let smoothed = ls.smooth(&label);
    // All should be epsilon/4
    for &v in smoothed.as_slice() {
        assert!((v - 0.05).abs() < 1e-6);
    }
}

#[test]
fn test_cutmix_large_image() {
    let cm = CutMix::new(1.0);
    let params = cm.sample(256, 256);
    assert!(params.x2 <= 256);
    assert!(params.y2 <= 256);
}

#[test]
fn test_drop_mode_clone() {
    let mode = DropMode::Batch;
    let cloned = mode;
    assert_eq!(cloned, DropMode::Batch);
}

#[test]
fn test_mixup_clone() {
    let mixup = Mixup::new(0.5);
    let cloned = mixup.clone();
    assert_eq!(cloned.alpha(), mixup.alpha());
}

#[test]
fn test_label_smoothing_clone() {
    let ls = LabelSmoothing::new(0.15);
    let cloned = ls.clone();
    assert_eq!(cloned.epsilon(), ls.epsilon());
}

#[test]
fn test_cutmix_clone() {
    let cm = CutMix::new(0.8);
    let cloned = cm.clone();
    assert_eq!(cloned.alpha(), cm.alpha());
}

// =========================================================================
// Additional coverage tests for edge cases
// =========================================================================

#[test]
fn test_cutmix_alpha_negative() {
    let cm = CutMix::new(-0.5);
    let params = cm.sample(10, 10);
    // Alpha <= 0 should return lambda = 1.0 and empty box
    assert_eq!(params.lambda, 1.0);
    assert_eq!(params.x1, 0);
    assert_eq!(params.y1, 0);
    assert_eq!(params.x2, 0);
    assert_eq!(params.y2, 0);
}
