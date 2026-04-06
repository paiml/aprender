    use super::super::ops::{patterned_floats, sequential_floats};
    use super::super::ulp::assert_ulp_eq;
    use super::*;
    use proptest::prelude::*;

    // ── Flash attention matches naive attention ─────────────────────────

    /// Verify flash attention matches naive attention on a small 4x3 input
    #[test]
    fn test_flash_matches_naive_small() {
        let n = 4;
        let d = 3;
        let tile_size = 2;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.15);
        let v = sequential_floats(n * d, 0.2);

        let mut flash_out = vec![0.0f32; n * d];
        let mut naive_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
        naive_attention(&q, &k, &v, n, d, &mut naive_out);

        for (i, (&f, &nv)) in flash_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (f - nv).abs() < 1e-5,
                "mismatch at index {i}: flash={f} naive={nv} (diff={})",
                (f - nv).abs()
            );
        }
    }

    /// Verify flash attention matches naive attention on a larger 8x4 patterned input
    #[test]
    fn test_flash_matches_naive_larger() {
        let n = 8;
        let d = 4;
        let tile_size = 3;

        let q = patterned_floats(n * d, 7, 3.0, 0.5);
        let k = patterned_floats(n * d, 5, 2.0, 0.3);
        let v = patterned_floats(n * d, 11, 5.0, 0.2);

        let mut flash_out = vec![0.0f32; n * d];
        let mut naive_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
        naive_attention(&q, &k, &v, n, d, &mut naive_out);

        for (i, (&f, &nv)) in flash_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (f - nv).abs() < 1e-4,
                "mismatch at index {i}: flash={f} naive={nv} (diff={})",
                (f - nv).abs()
            );
        }
    }

    // ── Single tile degrades to standard attention ──────────────────────

    /// Verify flash attention degrades to standard attention when tile covers all keys
    #[test]
    fn test_flash_single_tile() {
        let n = 4;
        let d = 3;
        let tile_size = n + 10;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.15);
        let v = sequential_floats(n * d, 0.2);

        let mut flash_out = vec![0.0f32; n * d];
        let mut naive_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
        naive_attention(&q, &k, &v, n, d, &mut naive_out);

        for (i, (&f, &nv)) in flash_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (f - nv).abs() < 1e-6,
                "mismatch at index {i}: flash={f} naive={nv}"
            );
        }
    }

    // ── Tile size = 1 (extreme tiling) ──────────────────────────────────

    /// Verify flash attention correctness with tile_size=1 (extreme tiling)
    #[test]
    fn test_flash_tile_size_one() {
        let n = 5;
        let d = 2;
        let tile_size = 1;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.2);
        let v = sequential_floats(n * d, 0.15);

        let mut flash_out = vec![0.0f32; n * d];
        let mut naive_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
        naive_attention(&q, &k, &v, n, d, &mut naive_out);

        for (i, (&f, &nv)) in flash_out.iter().zip(naive_out.iter()).enumerate() {
            assert!(
                (f - nv).abs() < 1e-5,
                "mismatch at index {i}: flash={f} naive={nv}"
            );
        }
    }

    // ── Single element ──────────────────────────────────────────────────

    /// Verify flash attention on a single element returns the value vector unchanged
    #[test]
    fn test_flash_single_element() {
        let n = 1;
        let d = 3;
        let tile_size = 1;

        let q = vec![1.0, 2.0, 3.0];
        let k = vec![1.0, 0.0, 0.0];
        let v = vec![10.0, 20.0, 30.0];
        let mut output = vec![0.0f32; d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut output);
        assert_ulp_eq(&output, &v, 0);
    }

    // ── Dimension assertions ────────────────────────────────────────────

    /// Verify flash attention panics on Q dimension mismatch
    #[test]
    #[should_panic(expected = "Q dimension mismatch")]
    fn test_flash_bad_q_dim() {
        let mut output = vec![0.0f32; 4];
        flash_attention_scalar(&[1.0], &[1.0; 4], &[1.0; 4], 2, 2, 1, &mut output);
    }

    /// Verify flash attention panics when tile_size is zero
    #[test]
    #[should_panic(expected = "tile_size must be > 0")]
    fn test_flash_zero_tile_size() {
        let mut output = vec![0.0f32; 4];
        flash_attention_scalar(&[1.0; 4], &[1.0; 4], &[1.0; 4], 2, 2, 0, &mut output);
    }

    // ── Property-based tests ────────────────────────────────────────────

    proptest! {
        #[test]
        fn prop_flash_matches_naive(
            n in 1usize..6,
            d in 1usize..5,
            tile_size in 1usize..8,
        ) {
            let q = patterned_floats(n*d, 7, 3.0, 0.3);
            let k = patterned_floats(n*d, 5, 2.0, 0.2);
            let v = patterned_floats(n*d, 11, 5.0, 0.15);

            let mut flash_out = vec![0.0f32; n * d];
            let mut naive_out = vec![0.0f32; n * d];

            flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut flash_out);
            naive_attention(&q, &k, &v, n, d, &mut naive_out);

            for idx in 0..n*d {
                let diff = (flash_out[idx] - naive_out[idx]).abs();
                prop_assert!(
                    diff < 1e-4,
                    "mismatch at {idx}: flash={} naive={} (diff={diff})",
                    flash_out[idx], naive_out[idx]
                );
            }
        }

        #[test]
        fn prop_flash_output_row_norms_bounded(
            n in 1usize..5,
            d in 1usize..4,
            tile_size in 1usize..6,
        ) {
            let q = sequential_floats(n*d, 0.1);
            let k = sequential_floats(n*d, 0.1);
            let v = sequential_floats(n*d, 0.1);
            let mut output = vec![0.0f32; n * d];

            flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut output);

            let max_v_norm: f32 = (0..n)
                .map(|r| {
                    (0..d).map(|c| v[r * d + c] * v[r * d + c]).sum::<f32>().sqrt()
                })
                .fold(0.0f32, f32::max);

            for i in 0..n {
                let row_norm: f32 = (0..d)
                    .map(|c| output[i * d + c] * output[i * d + c])
                    .sum::<f32>()
                    .sqrt();
                prop_assert!(
                    row_norm <= max_v_norm + 1e-4,
                    "output row {i} norm {row_norm} exceeds max V row norm {max_v_norm}"
                );
            }
        }
    }

    // ── AVX2 parity test ────────────────────────────────────────────────

    /// Verify AVX2 flash attention produces identical results to scalar implementation
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_flash_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 6;
        let d = 4;
        let tile_size = 2;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.15);
        let v = sequential_floats(n * d, 0.2);

        let mut scalar_out = vec![0.0f32; n * d];
        let mut avx2_out = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, tile_size, &mut scalar_out);
        unsafe { flash_attention_avx2(&q, &k, &v, n, d, tile_size, &mut avx2_out) };

        assert_ulp_eq(&scalar_out, &avx2_out, 8);
    }

    // ── PTX structural tests ────────────────────────────────────────────

    /// Verify flash attention PTX contains required instructions and balanced braces
    #[test]
    fn test_flash_attention_ptx_structure() {
        let ptx = flash_attention_ptx();
        assert!(ptx.contains(".version 8.5"), "missing PTX version");
        assert!(ptx.contains(".target sm_90"), "missing PTX target");
        assert!(
            ptx.contains(".entry flash_attention_kernel"),
            "missing entry point"
        );
        assert!(ptx.contains("ret;"), "missing ret instruction");
        assert!(ptx.contains(".shared"), "missing shared memory declaration");
        assert!(ptx.contains("bar.sync"), "missing barrier synchronization");
        assert!(ptx.contains("ex2.approx.f32"), "missing exp approximation");
        assert!(ptx.contains("fma.rn.f32"), "missing FMA instruction");
        assert!(
            ptx.contains("rcp.approx.f32"),
            "missing reciprocal for normalization"
        );
        let open = ptx.matches('{').count();
        let close = ptx.matches('}').count();
        assert_eq!(
            open, close,
            "unbalanced braces: {open} open vs {close} close"
        );
    }

    /// Verify flash attention PTX output is non-empty
    #[test]
    fn test_flash_attention_ptx_nonempty() {
        assert!(!flash_attention_ptx().is_empty());
    }

    // ── Different tile sizes produce same result ────────────────────────

    /// Verify different tile sizes produce identical flash attention results
    #[test]
    fn test_flash_tile_size_invariance() {
        let n = 6;
        let d = 3;

        let q = sequential_floats(n * d, 0.1);
        let k = sequential_floats(n * d, 0.15);
        let v = sequential_floats(n * d, 0.2);

        let mut out_t1 = vec![0.0f32; n * d];
        let mut out_t2 = vec![0.0f32; n * d];
        let mut out_t3 = vec![0.0f32; n * d];
        let mut out_tall = vec![0.0f32; n * d];

        flash_attention_scalar(&q, &k, &v, n, d, 1, &mut out_t1);
        flash_attention_scalar(&q, &k, &v, n, d, 2, &mut out_t2);
        flash_attention_scalar(&q, &k, &v, n, d, 3, &mut out_t3);
        flash_attention_scalar(&q, &k, &v, n, d, n, &mut out_tall);

        for i in 0..n * d {
            assert!(
                (out_t1[i] - out_tall[i]).abs() < 1e-5,
                "tile_size=1 vs full: index {i}: {} vs {}",
                out_t1[i],
                out_tall[i]
            );
            assert!(
                (out_t2[i] - out_tall[i]).abs() < 1e-5,
                "tile_size=2 vs full: index {i}: {} vs {}",
                out_t2[i],
                out_tall[i]
            );
            assert!(
                (out_t3[i] - out_tall[i]).abs() < 1e-5,
                "tile_size=3 vs full: index {i}: {} vs {}",
                out_t3[i],
                out_tall[i]
            );
        }
    }
