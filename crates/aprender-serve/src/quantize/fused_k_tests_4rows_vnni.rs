// ============================================================================
// FUSED Q4_K × Q8_K 4-ROW AVX-512 VNNI MICRO-KERNEL (#5689)
// ============================================================================
// fused_q4k_q8k_dot_4rows_avx512vnni shares one Q8K load across four weight
// rows. Each output lane must equal the single-row scalar dot of its own row.

#[cfg(target_arch = "x86_64")]
fn vnni_4rows_fixture(num_sb: usize, seed: u32) -> (Vec<Vec<u8>>, Vec<f32>, Vec<i8>) {
    let mut state = seed;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u8
    };
    let rows = (0..4u8)
        .map(|r| {
            let mut data = Vec::with_capacity(144 * num_sb);
            for _ in 0..num_sb {
                // Distinct d/dmin per row so a swapped lane cannot cancel out.
                let mut block = build_q4k_test_block(0.25 * f32::from(r + 1), 0.125, 0);
                for b in &mut block[4..] {
                    *b = next();
                }
                data.extend_from_slice(&block);
            }
            data
        })
        .collect();
    let scales = (0..num_sb).map(|i| 0.5 + 0.25 * i as f32).collect();
    let quants = (0..num_sb * 256).map(|_| next() as i8).collect();
    (rows, scales, quants)
}

#[cfg(target_arch = "x86_64")]
fn assert_4rows_match_scalar(num_sb: usize, seed: u32) {
    let (rows, scales, quants) = vnni_4rows_fixture(num_sb, seed);
    let ptrs = [
        rows[0].as_ptr(),
        rows[1].as_ptr(),
        rows[2].as_ptr(),
        rows[3].as_ptr(),
    ];
    // SAFETY: caller checked avx512f/vnni/bw; each row holds num_sb * 144 bytes
    // and quants/scales cover num_sb super-blocks.
    let got = unsafe { fused_q4k_q8k_dot_4rows_avx512vnni(ptrs, 144 * num_sb, &scales, &quants) };
    for (i, row) in rows.iter().enumerate() {
        let want = fused_q4k_q8k_dot(row, &scales, &quants).expect("scalar");
        let tol = want.abs().max(1.0) * 0.01;
        assert!(
            (got[i] - want).abs() < tol,
            "row {i}, {num_sb} sb: vnni={} vs scalar={want}",
            got[i]
        );
    }
}

#[cfg(target_arch = "x86_64")]
fn vnni_4rows_supported() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vnni")
        && is_x86_feature_detected!("avx512bw")
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_4rows_matches_scalar_single_superblock() {
    if !vnni_4rows_supported() {
        return;
    }
    assert_4rows_match_scalar(1, 0x5689);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_4rows_matches_scalar_multi_superblock_prefetch_paths() {
    if !vnni_4rows_supported() {
        return;
    }
    // 4 super-blocks exercises both the +2 and +1 prefetch branches and the tail.
    assert_4rows_match_scalar(4, 0x1234_5678);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_4rows_zero_superblocks_returns_zeros() {
    if !vnni_4rows_supported() {
        return;
    }
    let ptrs = [std::ptr::null::<u8>(); 4];
    // SAFETY: zero super-blocks means no row pointer or slice element is read.
    let got = unsafe { fused_q4k_q8k_dot_4rows_avx512vnni(ptrs, 0, &[], &[]) };
    assert_eq!(got, [0.0; 4]);
}
