
    // ========================================================================
    // VAE Tests
    // ========================================================================

    #[test]
    fn test_vae_creation() {
        let vae = VAE::new(784, vec![256, 128], 20);

        assert_eq!(vae.input_dim(), 784);
        assert_eq!(vae.latent_dim(), 20);
        assert_eq!(vae.beta(), 1.0);
    }

    #[test]
    fn test_vae_with_beta() {
        let vae = VAE::new(784, vec![256], 20).with_beta(4.0);
        assert_eq!(vae.beta(), 4.0);
    }

    #[test]
    fn test_vae_encode() {
        let vae = VAE::new(100, vec![64], 10);

        let x = Tensor::ones(&[4, 100]);
        let (mu, log_var) = vae.encode(&x);

        assert_eq!(mu.shape(), &[4, 10]);
        assert_eq!(log_var.shape(), &[4, 10]);
    }

    #[test]
    fn test_vae_decode() {
        let vae = VAE::new(100, vec![64], 10);

        let z = Tensor::ones(&[4, 10]);
        let reconstruction = vae.decode(&z);

        assert_eq!(reconstruction.shape(), &[4, 100]);
    }

    #[test]
    fn test_vae_forward() {
        let vae = VAE::new(100, vec![64], 10);

        let x = Tensor::ones(&[4, 100]);
        let output = vae.forward_vae(&x);

        assert_eq!(output.reconstruction.shape(), &[4, 100]);
        assert_eq!(output.mu.shape(), &[4, 10]);
        assert_eq!(output.log_var.shape(), &[4, 10]);
        assert_eq!(output.z.shape(), &[4, 10]);
    }

    #[test]
    fn test_vae_module_forward() {
        let vae = VAE::new(100, vec![64], 10);

        let x = Tensor::ones(&[4, 100]);
        let output = vae.forward(&x);

        assert_eq!(output.shape(), &[4, 100]);
    }

    #[test]
    fn test_vae_loss() {
        let vae = VAE::new(100, vec![64], 10);

        let x = Tensor::ones(&[4, 100]);
        let output = vae.forward_vae(&x);
        let (total, recon, kl) = vae.loss(&output, &x);

        // Loss should be non-negative
        assert!(recon >= 0.0);
        // KL can be any real value but typically positive
        assert!(total.is_finite());
        assert!(kl.is_finite());
    }

    #[test]
    fn test_vae_sample() {
        let vae = VAE::new(100, vec![64], 10);

        let samples = vae.sample(8);

        assert_eq!(samples.shape(), &[8, 100]);
    }

    #[test]
    fn test_vae_interpolate() {
        let vae = VAE::new(100, vec![64], 10);

        let x1 = Tensor::ones(&[1, 100]);
        let x2 = Tensor::zeros(&[1, 100]);

        let interpolations = vae.interpolate(&x1, &x2, 5);

        assert_eq!(interpolations.len(), 5);
        for interp in &interpolations {
            assert_eq!(interp.shape(), &[1, 100]);
        }
    }

    #[test]
    fn test_vae_train_eval() {
        let mut vae = VAE::new(100, vec![64], 10);

        assert!(vae.training());

        vae.eval();
        assert!(!vae.training());

        vae.train();
        assert!(vae.training());
    }

    #[test]
    fn test_vae_parameters() {
        let vae = VAE::new(100, vec![64], 10);
        let params = vae.parameters();

        // encoder_layer (64) + fc_mu + fc_log_var + decoder_layer (64) + output
        // Each linear has weight + bias = 2 params
        // 1 encoder + 2 latent + 1 decoder + 1 output = 5 layers * 2 = 10 params
        assert_eq!(params.len(), 10);
    }

    #[test]
    fn test_vae_reparameterize_training() {
        let vae = VAE::new(100, vec![64], 10);

        let mu = Tensor::zeros(&[4, 10]);
        let log_var = Tensor::zeros(&[4, 10]);

        let z = vae.reparameterize(&mu, &log_var);

        assert_eq!(z.shape(), &[4, 10]);
        // z should have some variance due to sampling
    }

    #[test]
    fn test_vae_reparameterize_eval() {
        let mut vae = VAE::new(100, vec![64], 10);
        vae.eval();

        let mu = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let log_var = Tensor::zeros(&[2, 2]);

        let z = vae.reparameterize(&mu, &log_var);

        // In eval mode, z should equal mu
        assert_eq!(z.data(), mu.data());
    }

    // ========================================================================
    // Conditional VAE Tests
    // ========================================================================

    #[test]
    fn test_cvae_creation() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);
        assert_eq!(cvae.num_classes(), 10);
    }

    #[test]
    fn test_cvae_encode() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);

        let x = Tensor::ones(&[4, 100]);
        let (mu, log_var) = cvae.encode(&x, 5);

        assert_eq!(mu.shape(), &[4, 20]);
        assert_eq!(log_var.shape(), &[4, 20]);
    }

    #[test]
    fn test_cvae_sample() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);

        let samples = cvae.sample(8, 3);

        // Output should be the original input dimension (not including class)
        assert_eq!(samples.shape(), &[8, 100]);
    }

    #[test]
    fn test_cvae_forward() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);

        let x = Tensor::ones(&[4, 100]);
        let output = cvae.forward_cvae(&x, 5);

        assert_eq!(output.reconstruction.shape(), &[4, 100]);
        assert_eq!(output.mu.shape(), &[4, 20]);
        assert_eq!(output.log_var.shape(), &[4, 20]);
        assert_eq!(output.z.shape(), &[4, 20]);
    }

    #[test]
    fn test_cvae_getters() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);

        assert_eq!(cvae.input_dim(), 100);
        assert_eq!(cvae.latent_dim(), 20);
        assert_eq!(cvae.num_classes(), 10);
    }

    // ========================================================================
    // Helper Function Tests
    // ========================================================================

    #[test]
    fn test_sample_standard_normal() {
        let samples = sample_standard_normal(&[1000]);

        // Mean should be close to 0
        let mean: f32 = samples.data().iter().sum::<f32>() / 1000.0;
        assert!(mean.abs() < 0.2);

        // Variance should be close to 1
        let variance: f32 = samples
            .data()
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / 1000.0;
        assert!((variance - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_exp_half() {
        let log_var = Tensor::new(&[0.0, 2.0, -2.0], &[3]);
        let std = exp_half(&log_var);

        // exp(0.5 * 0) = 1, exp(0.5 * 2) = e, exp(0.5 * -2) = 1/e
        assert!((std.data()[0] - 1.0).abs() < 1e-6);
        assert!((std.data()[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((std.data()[2] - 1.0 / std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_add_mul() {
        let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(&[2.0, 2.0, 2.0], &[3]);
        let c = Tensor::new(&[1.0, 1.0, 1.0], &[3]);

        let result = add_mul(&a, &b, &c);

        // 1 + 2*1 = 3, 2 + 2*1 = 4, 3 + 2*1 = 5
        assert_eq!(result.data(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let target = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let loss = mse_loss(&pred, &target);
        assert_eq!(loss, 0.0);

        let pred2 = Tensor::new(&[2.0, 3.0, 4.0, 5.0], &[4]);
        let loss2 = mse_loss(&pred2, &target);
        assert_eq!(loss2, 1.0); // Each diff is 1, squared is 1, mean is 1
    }

    #[test]
    fn test_kl_divergence_loss() {
        // When mu=0 and log_var=0 (var=1), KL should be 0
        let mu = Tensor::zeros(&[2, 3]);
        let log_var = Tensor::zeros(&[2, 3]);

        let kl = kl_divergence_loss(&mu, &log_var);
        assert!(kl.abs() < 1e-6);
    }

    #[test]
    fn test_lerp() {
        let a = Tensor::new(&[0.0, 0.0], &[2]);
        let b = Tensor::new(&[10.0, 10.0], &[2]);

        let mid = lerp(&a, &b, 0.5);
        assert_eq!(mid.data(), &[5.0, 5.0]);

        let quarter = lerp(&a, &b, 0.25);
        assert_eq!(quarter.data(), &[2.5, 2.5]);
    }

    #[test]
    fn test_concat_one_hot() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = concat_one_hot(&x, 1, 3);

        assert_eq!(result.shape(), &[2, 5]); // 2 + 3 = 5

        // First sample: [1, 2, 0, 1, 0]
        assert_eq!(result.data()[0], 1.0);
        assert_eq!(result.data()[1], 2.0);
        assert_eq!(result.data()[2], 0.0);
        assert_eq!(result.data()[3], 1.0);
        assert_eq!(result.data()[4], 0.0);

        // Second sample: [3, 4, 0, 1, 0]
        assert_eq!(result.data()[5], 3.0);
        assert_eq!(result.data()[6], 4.0);
        assert_eq!(result.data()[7], 0.0);
        assert_eq!(result.data()[8], 1.0);
        assert_eq!(result.data()[9], 0.0);
    }

    #[test]
    fn test_vae_no_hidden_layers() {
        // Edge case: direct connection from input to latent
        let vae = VAE::new(100, vec![], 10);

        let x = Tensor::ones(&[4, 100]);
        let output = vae.forward_vae(&x);

        assert_eq!(output.reconstruction.shape(), &[4, 100]);
    }

    // ========================================================================
    // Coverage: Debug impls
    // ========================================================================

    #[test]
    fn test_vae_debug() {
        let vae = VAE::new(784, vec![256, 128], 20);
        let debug_str = format!("{:?}", vae);
        assert!(debug_str.contains("VAE"));
        assert!(debug_str.contains("input_dim"));
        assert!(debug_str.contains("784"));
        assert!(debug_str.contains("latent_dim"));
        assert!(debug_str.contains("20"));
        assert!(debug_str.contains("beta"));
    }

    #[test]
    fn test_vae_debug_with_beta() {
        let vae = VAE::new(100, vec![64], 10).with_beta(4.0);
        let debug_str = format!("{:?}", vae);
        assert!(debug_str.contains("4.0"));
    }

    #[test]
    fn test_cvae_debug() {
        let cvae = ConditionalVAE::new(100, 10, vec![64], 20);
        let debug_str = format!("{:?}", cvae);
        assert!(debug_str.contains("ConditionalVAE"));
        assert!(debug_str.contains("input_dim"));
        assert!(debug_str.contains("100"));
        assert!(debug_str.contains("latent_dim"));
        assert!(debug_str.contains("20"));
        assert!(debug_str.contains("num_classes"));
        assert!(debug_str.contains("10"));
    }

    #[test]
    fn test_vae_output_debug() {
        let vae = VAE::new(50, vec![32], 5);
        let x = Tensor::ones(&[2, 50]);
        let output = vae.forward_vae(&x);
        let debug_str = format!("{:?}", output);
        assert!(debug_str.contains("VAEOutput"));
    }

    // ========================================================================
    // Coverage: VAE parameters_mut
    // ========================================================================

    #[test]
    fn test_vae_parameters_mut() {
        let mut vae = VAE::new(100, vec![64], 10);
        let params = vae.parameters_mut();
        // Same count as parameters(): 1 encoder + 2 latent + 1 decoder + 1 output = 5 layers * 2 = 10
        assert_eq!(params.len(), 10);
    }

    // ========================================================================
    // Coverage: CVAE reparameterize in eval mode (training == false)
    // ========================================================================

    #[test]
    fn test_cvae_reparameterize_eval_mode() {
        let mut cvae = ConditionalVAE::new(50, 5, vec![32], 10);
        // Switch to eval to hit the !self.training path in reparameterize
        cvae.training = false;

        let x = Tensor::ones(&[2, 50]);
        let output = cvae.forward_cvae(&x, 2);

        // In eval mode, z should equal mu (no sampling noise)
        assert_eq!(output.z.data(), output.mu.data());
    }

    // ========================================================================
    // Coverage: CVAE with no hidden layers (unwrap_or path)
    // ========================================================================

    #[test]
    fn test_cvae_no_hidden_layers() {
        // When hidden_dims is empty, unwrap_or fallback is used
        let cvae = ConditionalVAE::new(50, 5, vec![], 10);

        let x = Tensor::ones(&[2, 50]);
        let output = cvae.forward_cvae(&x, 3);

        assert_eq!(output.reconstruction.shape(), &[2, 50]);
        assert_eq!(output.mu.shape(), &[2, 10]);
        assert_eq!(output.z.shape(), &[2, 10]);
    }

    // ========================================================================
    // Coverage: CVAE decode directly
    // ========================================================================

    #[test]
    fn test_cvae_decode() {
        let cvae = ConditionalVAE::new(50, 5, vec![32], 10);
        let z = Tensor::ones(&[2, 10]);
        let reconstruction = cvae.decode(&z, 1);
        assert_eq!(reconstruction.shape(), &[2, 50]);
    }

    // ========================================================================
    // Coverage: VAE Module::forward (wraps forward_vae)
    // ========================================================================

    #[test]
    fn test_vae_module_forward_returns_reconstruction() {
        let vae = VAE::new(50, vec![32], 10);
        let x = Tensor::ones(&[2, 50]);
        // Module::forward only returns reconstruction
        let output = Module::forward(&vae, &x);
        assert_eq!(output.shape(), &[2, 50]);
    }
