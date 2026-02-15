
    // ========================================================================
    // Coverage: VAE loss with beta > 1 (beta-VAE)
    // ========================================================================

    #[test]
    fn test_vae_loss_with_high_beta() {
        let vae = VAE::new(50, vec![32], 5).with_beta(10.0);
        let x = Tensor::ones(&[4, 50]);
        let output = vae.forward_vae(&x);
        let (total, recon, kl) = vae.loss(&output, &x);

        // With high beta, total should be recon + 10*kl
        let expected_total = recon + 10.0 * kl;
        assert!((total - expected_total).abs() < 1e-4);
    }

    // ========================================================================
    // Coverage: sample_standard_normal with multi-dim shape
    // ========================================================================

    #[test]
    fn test_sample_standard_normal_2d() {
        let samples = sample_standard_normal(&[10, 5]);
        assert_eq!(samples.shape(), &[10, 5]);
        assert_eq!(samples.data().len(), 50);
    }

    // ========================================================================
    // Coverage: lerp boundary values
    // ========================================================================

    #[test]
    fn test_lerp_boundary_alpha_zero() {
        let a = Tensor::new(&[1.0, 2.0], &[2]);
        let b = Tensor::new(&[10.0, 20.0], &[2]);
        let result = lerp(&a, &b, 0.0);
        assert_eq!(result.data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_lerp_boundary_alpha_one() {
        let a = Tensor::new(&[1.0, 2.0], &[2]);
        let b = Tensor::new(&[10.0, 20.0], &[2]);
        let result = lerp(&a, &b, 1.0);
        assert_eq!(result.data(), &[10.0, 20.0]);
    }

    // ========================================================================
    // Coverage: concat_one_hot with class_label 0
    // ========================================================================

    #[test]
    fn test_concat_one_hot_first_class() {
        let x = Tensor::new(&[1.0, 2.0], &[1, 2]);
        let result = concat_one_hot(&x, 0, 3);
        assert_eq!(result.shape(), &[1, 5]);
        // [1.0, 2.0, 1.0, 0.0, 0.0]
        assert_eq!(result.data()[2], 1.0);
        assert_eq!(result.data()[3], 0.0);
        assert_eq!(result.data()[4], 0.0);
    }

    // ========================================================================
    // Coverage: KL divergence with non-zero values
    // ========================================================================

    #[test]
    fn test_kl_divergence_nonzero() {
        // Non-trivial mu and log_var should give positive KL
        let mu = Tensor::new(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
        let log_var = Tensor::new(&[0.5, -0.5, 1.0, -1.0], &[2, 2]);
        let kl = kl_divergence_loss(&mu, &log_var);
        assert!(
            kl > 0.0,
            "KL should be positive for non-standard distribution"
        );
        assert!(kl.is_finite());
    }

    // ========================================================================
    // Coverage: CVAE sample
    // ========================================================================

    #[test]
    fn test_cvae_sample_multiple_classes() {
        let cvae = ConditionalVAE::new(50, 5, vec![32], 10);
        for class in 0..5 {
            let samples = cvae.sample(3, class);
            assert_eq!(samples.shape(), &[3, 50]);
        }
    }
