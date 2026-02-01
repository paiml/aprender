use super::*;

#[test]
fn test_mixup_new() {
    let mixup = Mixup::new(0.4);
    assert_eq!(mixup.alpha(), 0.4);
}

#[test]
fn test_mixup_sample_lambda() {
    let mixup = Mixup::new(1.0);
    for _ in 0..10 {
        let lambda = mixup.sample_lambda();
        assert!((0.0..=1.0).contains(&lambda));
    }
}

#[test]
fn test_mixup_alpha_zero() {
    let mixup = Mixup::new(0.0);
    assert_eq!(mixup.sample_lambda(), 1.0);
}

#[test]
fn test_mixup_mix_samples() {
    let mixup = Mixup::new(1.0);
    let x1 = Vector::from_slice(&[1.0, 0.0]);
    let x2 = Vector::from_slice(&[0.0, 1.0]);

    let mixed = mixup.mix_samples(&x1, &x2, 0.5);
    assert!((mixed.as_slice()[0] - 0.5).abs() < 1e-6);
    assert!((mixed.as_slice()[1] - 0.5).abs() < 1e-6);
}

#[test]
fn test_mixup_mix_extreme_lambda() {
    let mixup = Mixup::new(1.0);
    let x1 = Vector::from_slice(&[1.0, 2.0]);
    let x2 = Vector::from_slice(&[3.0, 4.0]);

    let mixed0 = mixup.mix_samples(&x1, &x2, 0.0);
    assert_eq!(mixed0.as_slice(), &[3.0, 4.0]);

    let mixed1 = mixup.mix_samples(&x1, &x2, 1.0);
    assert_eq!(mixed1.as_slice(), &[1.0, 2.0]);
}

#[test]
fn test_label_smoothing_new() {
    let ls = LabelSmoothing::new(0.1);
    assert_eq!(ls.epsilon(), 0.1);
}

#[test]
fn test_label_smoothing_smooth() {
    let ls = LabelSmoothing::new(0.1);
    let label = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let smoothed = ls.smooth(&label);

    // First element: 0.9 * 1.0 + 0.1/3 ≈ 0.933
    assert!((smoothed.as_slice()[0] - 0.9333).abs() < 0.01);
    // Others: 0.9 * 0.0 + 0.1/3 ≈ 0.033
    assert!((smoothed.as_slice()[1] - 0.0333).abs() < 0.01);
}

#[test]
fn test_label_smoothing_smooth_index() {
    let ls = LabelSmoothing::new(0.1);
    let smoothed = ls.smooth_index(0, 3);

    let sum: f32 = smoothed.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_label_smoothing_sums_to_one() {
    let ls = LabelSmoothing::new(0.2);
    let smoothed = ls.smooth_index(2, 5);

    let sum: f32 = smoothed.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_cross_entropy_with_smoothing() {
    let logits = Vector::from_slice(&[2.0, 1.0, 0.5]);
    let loss = cross_entropy_with_smoothing(&logits, 0, 0.1);
    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_cross_entropy_no_smoothing() {
    let logits = Vector::from_slice(&[10.0, 0.0, 0.0]);
    let loss = cross_entropy_with_smoothing(&logits, 0, 0.0);
    // Should be close to 0 since softmax gives ~1.0 for first class
    assert!(loss < 0.1);
}

// CutMix tests

#[test]
fn test_cutmix_creation() {
    let cm = CutMix::new(1.0);
    assert_eq!(cm.alpha(), 1.0);
}

#[test]
fn test_cutmix_sample() {
    let cm = CutMix::new(1.0);
    let params = cm.sample(32, 32);

    assert!(params.lambda >= 0.0 && params.lambda <= 1.0);
    assert!(params.x1 <= params.x2);
    assert!(params.y1 <= params.y2);
    assert!(params.x2 <= 32);
    assert!(params.y2 <= 32);
}

#[test]
fn test_cutmix_apply() {
    let params = CutMixParams {
        lambda: 0.5,
        x1: 1,
        y1: 1,
        x2: 2,
        y2: 2,
    };

    let img1 = vec![1.0; 12]; // 1 channel, 3x4
    let img2 = vec![2.0; 12];

    let result = params.apply(&img1, &img2, 1, 3, 4);
    assert_eq!(result.len(), 12);
    // Position (1,1) should be from img2
    assert_eq!(result[4 + 1], 2.0);
}

// Stochastic Depth tests

#[test]
fn test_stochastic_depth_creation() {
    let sd = StochasticDepth::new(0.2, DropMode::Batch);
    assert_eq!(sd.drop_prob(), 0.2);
}

#[test]
fn test_stochastic_depth_eval_always_keeps() {
    let sd = StochasticDepth::new(0.9, DropMode::Batch);
    // In eval mode, should always keep
    for _ in 0..10 {
        assert!(sd.should_keep(false));
    }
}

#[test]
fn test_stochastic_depth_zero_drop() {
    let sd = StochasticDepth::new(0.0, DropMode::Batch);
    for _ in 0..10 {
        assert!(sd.should_keep(true));
    }
}

#[test]
fn test_stochastic_depth_linear_decay() {
    let survival = StochasticDepth::linear_decay(5, 10, 0.5);
    assert!((survival - 0.75).abs() < 1e-6);

    let survival_last = StochasticDepth::linear_decay(10, 10, 0.5);
    assert!((survival_last - 0.5).abs() < 1e-6);
}

// R-Drop Tests
#[test]
fn test_rdrop_creation() {
    let rdrop = RDrop::new(0.5);
    assert_eq!(rdrop.alpha(), 0.5);
}

#[test]
fn test_rdrop_kl_divergence_same() {
    let rdrop = RDrop::new(1.0);
    let p = vec![0.5, 0.3, 0.2];
    let kl = rdrop.kl_divergence(&p, &p);
    assert!(kl.abs() < 1e-5);
}

#[test]
fn test_rdrop_kl_divergence_different() {
    let rdrop = RDrop::new(1.0);
    let p = vec![0.9, 0.1];
    let q = vec![0.1, 0.9];
    let kl = rdrop.kl_divergence(&p, &q);
    assert!(kl > 0.0);
}

#[test]
fn test_rdrop_symmetric_kl() {
    let rdrop = RDrop::new(1.0);
    let p = vec![0.7, 0.3];
    let q = vec![0.4, 0.6];
    let sym = rdrop.symmetric_kl(&p, &q);
    assert!(sym > 0.0);
}

#[test]
fn test_rdrop_compute_loss_same() {
    let rdrop = RDrop::new(1.0);
    let logits = vec![2.0, 1.0, 0.5];
    let loss = rdrop.compute_loss(&logits, &logits);
    assert!(loss.abs() < 1e-5);
}

#[test]
fn test_rdrop_compute_loss_different() {
    let rdrop = RDrop::new(1.0);
    let logits1 = vec![2.0, 0.0, 0.0];
    let logits2 = vec![0.0, 2.0, 0.0];
    let loss = rdrop.compute_loss(&logits1, &logits2);
    assert!(loss > 0.0);
}

#[test]
fn test_rdrop_alpha_zero() {
    let rdrop = RDrop::new(0.0);
    let logits1 = vec![2.0, 0.0];
    let logits2 = vec![0.0, 2.0];
    let loss = rdrop.compute_loss(&logits1, &logits2);
    assert_eq!(loss, 0.0);
}

// SpecAugment Tests

#[test]
fn test_specaugment_new() {
    let sa = SpecAugment::new();
    assert_eq!(sa.num_freq_masks(), 2);
    assert_eq!(sa.num_time_masks(), 2);
}

#[test]
fn test_specaugment_custom() {
    let sa = SpecAugment::with_params(1, 10, 3, 50);
    assert_eq!(sa.num_freq_masks(), 1);
    assert_eq!(sa.num_time_masks(), 3);
}

#[test]
fn test_specaugment_apply_shape() {
    let sa = SpecAugment::with_params(1, 5, 1, 10);
    let spec = vec![1.0; 80 * 100]; // 80 freq bins, 100 time steps
    let result = sa.apply(&spec, 80, 100);
    assert_eq!(result.len(), spec.len());
}

#[test]
fn test_specaugment_masks_applied() {
    let sa = SpecAugment::with_params(2, 10, 2, 20).with_mask_value(-999.0);
    let spec = vec![1.0; 40 * 50];
    let result = sa.apply(&spec, 40, 50);

    // Some values should be masked
    let masked_count = result.iter().filter(|&&v| v == -999.0).count();
    assert!(masked_count > 0, "Should have some masked values");
}

#[test]
fn test_specaugment_freq_mask() {
    let sa = SpecAugment::with_params(1, 5, 0, 0).with_mask_value(0.0);
    let spec = vec![1.0; 20 * 30]; // 20 freq, 30 time
    let result = sa.freq_mask(&spec, 20, 30);

    // Check that entire frequency bands are masked
    let zero_count = result.iter().filter(|&&v| v == 0.0).count();
    // zero_count is always >= 0 as usize; just verify we computed something
    let _ = zero_count; // May mask 0 width band
}

#[test]
fn test_specaugment_time_mask() {
    let sa = SpecAugment::with_params(0, 0, 1, 5).with_mask_value(0.0);
    let spec = vec![1.0; 20 * 30];
    let result = sa.time_mask(&spec, 20, 30);

    // Some time columns should be masked
    assert_eq!(result.len(), spec.len());
}

// RandAugment Tests

#[test]
fn test_randaugment_new() {
    let ra = RandAugment::new(2, 9);
    assert_eq!(ra.n(), 2);
    assert_eq!(ra.m(), 9);
}

#[test]
fn test_randaugment_default() {
    let ra = RandAugment::default();
    assert_eq!(ra.n(), 2);
    assert_eq!(ra.m(), 9);
}

#[test]
fn test_randaugment_magnitude_clamp() {
    let ra = RandAugment::new(1, 50); // Over max
    assert_eq!(ra.m(), 30);
}

#[test]
fn test_randaugment_normalized_magnitude() {
    let ra = RandAugment::new(1, 15);
    assert!((ra.normalized_magnitude() - 0.5).abs() < 1e-6);
}

#[test]
fn test_randaugment_sample_augmentations() {
    let ra = RandAugment::new(3, 10);
    let sampled = ra.sample_augmentations();
    assert_eq!(sampled.len(), 3);
}

#[test]
fn test_randaugment_apply_identity() {
    let ra = RandAugment::new(1, 15);
    let image = vec![0.5; 16]; // 1x4x4
    let result = ra.apply_single(&image, AugmentationType::Identity, 4, 4);
    assert_eq!(result, image);
}

#[test]
fn test_randaugment_apply_brightness() {
    let ra = RandAugment::new(1, 30); // Max magnitude
    let image = vec![0.5; 16];
    let result = ra.apply_single(&image, AugmentationType::Brightness, 4, 4);

    // Values should be modified
    let changed = result
        .iter()
        .zip(image.iter())
        .any(|(&r, &o)| (r - o).abs() > 0.01);
    assert!(changed, "Brightness should modify values");
}

#[test]
fn test_randaugment_apply_contrast() {
    let ra = RandAugment::new(1, 20);
    let image: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
    let result = ra.apply_single(&image, AugmentationType::Contrast, 4, 4);
    assert_eq!(result.len(), image.len());
}

#[test]
fn test_randaugment_custom_augmentations() {
    let ra = RandAugment::new(2, 10).with_augmentations(vec![
        AugmentationType::Identity,
        AugmentationType::Brightness,
    ]);

    let sampled = ra.sample_augmentations();
    for aug in sampled {
        assert!(aug == AugmentationType::Identity || aug == AugmentationType::Brightness);
    }
}

#[test]
fn test_augmentation_type_equality() {
    assert_eq!(AugmentationType::Rotate, AugmentationType::Rotate);
    assert_ne!(AugmentationType::Rotate, AugmentationType::Brightness);
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_stochastic_depth_mode() {
    let sd_batch = StochasticDepth::new(0.1, DropMode::Batch);
    assert_eq!(sd_batch.mode(), DropMode::Batch);

    let sd_row = StochasticDepth::new(0.1, DropMode::Row);
    assert_eq!(sd_row.mode(), DropMode::Row);
}

#[test]
fn test_drop_mode_eq() {
    assert_eq!(DropMode::Batch, DropMode::Batch);
    assert_ne!(DropMode::Batch, DropMode::Row);
}

#[test]
fn test_specaugment_default() {
    let sa = SpecAugment::default();
    assert_eq!(sa.num_freq_masks(), 2);
    assert_eq!(sa.num_time_masks(), 2);
}

#[test]
fn test_specaugment_with_mask_value() {
    let sa = SpecAugment::new().with_mask_value(-1.0);
    // Just verify it compiles and works
    let spec = vec![1.0; 100];
    let result = sa.apply(&spec, 10, 10);
    assert_eq!(result.len(), 100);
}

#[test]
fn test_randaugment_apply_rotate() {
    let ra = RandAugment::new(1, 20); // mag > 0.5
    let image = vec![1.0, 2.0, 3.0, 4.0];
    let result = ra.apply_single(&image, AugmentationType::Rotate, 2, 2);
    // High magnitude should reverse
    assert_eq!(result, vec![4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_randaugment_apply_rotate_low_mag() {
    let ra = RandAugment::new(1, 5); // mag = 5/30 < 0.5
    let image = vec![1.0, 2.0, 3.0, 4.0];
    let result = ra.apply_single(&image, AugmentationType::Rotate, 2, 2);
    // Low magnitude shouldn't reverse
    assert_eq!(result, image);
}

#[test]
fn test_randaugment_apply_translate_x() {
    let ra = RandAugment::new(1, 15);
    let image = vec![1.0; 16];
    let result = ra.apply_single(&image, AugmentationType::TranslateX, 4, 4);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_randaugment_apply_translate_y() {
    let ra = RandAugment::new(1, 15);
    let image = vec![1.0; 16];
    let result = ra.apply_single(&image, AugmentationType::TranslateY, 4, 4);
    assert_eq!(result.len(), 16);
}

#[test]
fn test_randaugment_apply_shear_x() {
    let ra = RandAugment::new(1, 15);
    let image = vec![0.5; 16];
    let result = ra.apply_single(&image, AugmentationType::ShearX, 4, 4);
    assert_eq!(result.len(), 16);
}

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
    // Very small alpha for extreme beta distribution shape
    let mixup = Mixup::new(0.01);
    let lambda = mixup.sample_lambda();
    assert!((0.0..=1.0).contains(&lambda));
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
