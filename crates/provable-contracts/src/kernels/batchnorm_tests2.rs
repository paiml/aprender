    // ── AVX2 parity tests ────────────────────────────────────────────────

    /// Verify AVX2 batchnorm matches scalar output during training
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_batchnorm_avx2_parity_training() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (0..16).map(|x| x as f32 * 0.5).collect();
        let gamma = vec![1.0_f32; 4];
        let beta = vec![0.0_f32; 4];

        let mut rm_scalar = vec![0.0_f32; 4];
        let mut rv_scalar = vec![1.0_f32; 4];
        let mut scalar_out = vec![0.0_f32; 16];
        batchnorm_scalar(
            &input,
            4,
            4,
            &gamma,
            &beta,
            1e-5,
            &mut rm_scalar,
            &mut rv_scalar,
            &mut scalar_out,
            0.1,
            true,
        );

        let mut rm_avx2 = vec![0.0_f32; 4];
        let mut rv_avx2 = vec![1.0_f32; 4];
        let mut avx2_out = vec![0.0_f32; 16];
        unsafe {
            batchnorm_avx2(
                &input,
                4,
                4,
                &gamma,
                &beta,
                1e-5,
                &mut rm_avx2,
                &mut rv_avx2,
                &mut avx2_out,
                0.1,
                true,
            );
        }

        assert_ulp_eq(&scalar_out, &avx2_out, 4);
        assert_ulp_eq(&rm_scalar, &rm_avx2, 4);
        assert_ulp_eq(&rv_scalar, &rv_avx2, 4);
    }

    /// Verify AVX2 batchnorm matches scalar output during inference
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_batchnorm_avx2_parity_inference() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let gamma = vec![1.0_f32; 3];
        let beta = vec![0.0_f32; 3];

        let mut rm_scalar = vec![2.0_f32; 3];
        let mut rv_scalar = vec![1.0_f32; 3];
        let mut scalar_out = vec![0.0_f32; 12];
        batchnorm_scalar(
            &input,
            4,
            3,
            &gamma,
            &beta,
            1e-5,
            &mut rm_scalar,
            &mut rv_scalar,
            &mut scalar_out,
            0.1,
            false,
        );

        let mut rm_avx2 = vec![2.0_f32; 3];
        let mut rv_avx2 = vec![1.0_f32; 3];
        let mut avx2_out = vec![0.0_f32; 12];
        unsafe {
            batchnorm_avx2(
                &input,
                4,
                3,
                &gamma,
                &beta,
                1e-5,
                &mut rm_avx2,
                &mut rv_avx2,
                &mut avx2_out,
                0.1,
                false,
            );
        }

        assert_ulp_eq(&scalar_out, &avx2_out, 4);
    }

    // ── PTX structural tests ─────────────────────────────────────────────

    /// Verify batchnorm PTX declares version 8.5
    #[test]
    fn test_batchnorm_ptx_version() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
    }

    /// Verify batchnorm PTX targets sm_90
    #[test]
    fn test_batchnorm_ptx_target() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
    }

    /// Verify batchnorm PTX contains the kernel entry point
    #[test]
    fn test_batchnorm_ptx_entry() {
        let ptx = batchnorm_ptx();
        assert!(
            ptx.contains(".entry batchnorm_kernel"),
            "missing entry point"
        );
    }

    /// Verify batchnorm PTX contains a ret instruction
    #[test]
    fn test_batchnorm_ptx_ret() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains("ret;"), "missing ret instruction");
    }

    /// Verify batchnorm PTX declares shared memory
    #[test]
    fn test_batchnorm_ptx_shared_memory() {
        let ptx = batchnorm_ptx();
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
    }

    /// Verify batchnorm PTX uses warp shuffle instructions for reduction
    #[test]
    fn test_batchnorm_ptx_warp_shuffle() {
        let ptx = batchnorm_ptx();
        assert!(
            ptx.contains("shfl.sync"),
            "missing warp shuffle instructions"
        );
    }

    /// Verify batchnorm PTX contains bar.sync for block synchronization
    #[test]
    fn test_batchnorm_ptx_bar_sync() {
        let ptx = batchnorm_ptx();
        assert!(
            ptx.contains("bar.sync"),
            "missing bar.sync for block synchronization"
        );
    }

    /// Verify batchnorm PTX has balanced curly braces
    #[test]
    fn test_batchnorm_ptx_balanced_braces() {
        let ptx = batchnorm_ptx();
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }
