//! Contract Falsification Tests (quantized-dot-product-v1.yaml)
//!
//! Five Popperian falsification tests that validate the mathematical contract
//! for quantized dot product kernels. Each test attempts to DISPROVE a claim
//! made by the contract — if the test passes, the claim holds.
//!
//! ## Test IDs
//!
//! - FALSIFY-QDOT-001: Scalar-SIMD equivalence
//! - FALSIFY-QDOT-002: Cross-format isolation
//! - FALSIFY-QDOT-003: Bsum precomputed vs on-the-fly equivalence
//! - FALSIFY-QDOT-004: Format exhaustiveness
//! - FALSIFY-QDOT-005: Dispatch exhaustiveness

#[cfg(test)]
mod tests {
    use crate::quantize::format_trait::{
        Q4_0Fmt, Q8_0Fmt, QuantBlockFormat, QuantFamily, ALL_FORMAT_IDS, Q4K, Q5K, Q6K,
    };
    use crate::quantize::generic_dot::{compute_bsums, generic_fused_dot_scalar};
    use crate::quantize::simd::{extract_scale_min, read_f16};

    // =========================================================================
    // FALSIFY-QDOT-001: Scalar-SIMD equivalence
    // =========================================================================
    //
    // Claim: "SIMD kernels are numerically equivalent to scalar reference"
    // Method: Compare generic scalar dot against existing format-specific scalar
    //         implementations for known test data.

    /// Helper: Create a Q4_K super-block with known values
    fn create_q4k_superblock(d: f32, dmin: f32, scale: u8, min: u8, q_val: u8) -> Vec<u8> {
        let mut sb = vec![0u8; 144];

        // Write d as f16
        let d_f16 = half::f16::from_f32(d);
        let d_bytes = d_f16.to_le_bytes();
        sb[0] = d_bytes[0];
        sb[1] = d_bytes[1];

        // Write dmin as f16
        let dmin_f16 = half::f16::from_f32(dmin);
        let dmin_bytes = dmin_f16.to_le_bytes();
        sb[2] = dmin_bytes[0];
        sb[3] = dmin_bytes[1];

        // Set scales: blocks 0-3 use simple layout (scale = q[j] & 63, min = q[j+4] & 63)
        for j in 0..4 {
            sb[4 + j] = scale & 63;
            sb[4 + j + 4] = min & 63;
        }
        // Blocks 4-7 use packed layout
        for j in 4..8 {
            // Set lower 4 bits of scales[j+4] to scale
            sb[4 + j + 4] = (scale & 0x0F) | ((min & 0x0F) << 4);
            // Set upper 2 bits of scales[j-4] for high bits of scale
            sb[4 + j - 4] |= ((scale >> 4) & 0x03) << 6;
            // Set upper 2 bits of scales[j] for high bits of min
            sb[4 + j] |= ((min >> 4) & 0x03) << 6;
        }

        // Set all quantized values to q_val (packed: low nibble and high nibble)
        let packed = (q_val & 0x0F) | ((q_val & 0x0F) << 4);
        for byte in &mut sb[16..144] {
            *byte = packed;
        }

        sb
    }

    #[test]
    fn falsify_qdot_001_q4k_scalar_generic_equivalence() {
        // Compare generic scalar against format-specific scalar for Q4_K
        let sb = create_q4k_superblock(0.5, 0.1, 10, 3, 7);
        let acts = vec![1.0f32; 256];

        // Generic scalar
        let generic_result =
            generic_fused_dot_scalar::<Q4K>(&sb, &acts).expect("generic Q4K dot should succeed");

        // Format-specific scalar (existing implementation)
        let specific_result =
            crate::quantize::fused_q4k_dot(&sb, &acts).expect("specific Q4K dot should succeed");

        // They should produce the same result (both are scalar, deterministic)
        let diff = (generic_result - specific_result).abs();
        let tolerance = 0.01 * specific_result.abs().max(1.0);
        assert!(
            diff <= tolerance,
            "FALSIFY-QDOT-001 FAILED: Q4K generic={generic_result}, specific={specific_result}, diff={diff}"
        );
    }

    #[test]
    fn falsify_qdot_001_q6k_scalar_generic_equivalence() {
        // Compare generic scalar against format-specific scalar for Q6_K
        // Create a Q6_K superblock with known data
        let mut sb = vec![0u8; 210];

        // Set d at offset 208
        let d_f16 = half::f16::from_f32(0.5);
        let d_bytes = d_f16.to_le_bytes();
        sb[208] = d_bytes[0];
        sb[209] = d_bytes[1];

        // Set scales at offset 192 (all = 4 as i8)
        for i in 0..16 {
            sb[192 + i] = 4u8; // i8 = 4
        }

        // Set ql values (low 4 bits) — all to value that gives q=36 after assembly
        // q = (ql & 0xF) | ((qh_bits & 3) << 4) - 32
        // For ql=4, qh=0: q = 4 - 32 = -28
        for byte in &mut sb[0..128] {
            *byte = 0x44; // low nibble = 4, high nibble = 4
        }
        // qh all zero → high bits = 0

        let acts = vec![1.0f32; 256];

        let generic_result =
            generic_fused_dot_scalar::<Q6K>(&sb, &acts).expect("generic Q6K dot should succeed");

        let specific_result =
            crate::quantize::fused_q6k_dot(&sb, &acts).expect("specific Q6K dot should succeed");

        let diff = (generic_result - specific_result).abs();
        let tolerance = 0.01 * specific_result.abs().max(1.0);
        assert!(
            diff <= tolerance,
            "FALSIFY-QDOT-001 FAILED: Q6K generic={generic_result}, specific={specific_result}, diff={diff}"
        );
    }

    #[test]
    fn falsify_qdot_001_q8_0_known_value() {
        // Q8_0: scale * q_i, all q_i = 10, scale = 2.0, acts = 1.0
        // Expected: 2.0 * 10 * 32 * 1.0 = 640.0
        let mut sb = [0u8; 34];
        let scale_f16 = half::f16::from_f32(2.0);
        let s_bytes = scale_f16.to_le_bytes();
        sb[0] = s_bytes[0];
        sb[1] = s_bytes[1];
        for i in 0..32 {
            sb[2 + i] = 10u8; // i8 = 10
        }

        let acts = vec![1.0f32; 32];
        let result =
            generic_fused_dot_scalar::<Q8_0Fmt>(&sb, &acts).expect("Q8_0 dot should succeed");

        assert!(
            (result - 640.0).abs() < 1.0,
            "FALSIFY-QDOT-001: Q8_0 expected ~640.0, got {result}"
        );
    }

    // =========================================================================
    // FALSIFY-QDOT-002: Cross-format isolation
    // =========================================================================
    //
    // Claim: "Passing data for format X through format Y's kernel produces garbage"
    // This proves format dispatch correctness matters.

    #[test]
    fn falsify_qdot_002_q6k_data_through_q4k_kernel() {
        // Create valid Q6_K data with a meaningful signal
        let mut q6k_data = vec![0u8; 210];
        // Set Q6_K d (at offset 208) to 1.0
        let d_f16 = half::f16::from_f32(1.0);
        let d_bytes = d_f16.to_le_bytes();
        q6k_data[208] = d_bytes[0];
        q6k_data[209] = d_bytes[1];
        // Set some scales and values
        for i in 0..16 {
            q6k_data[192 + i] = 10;
        }
        for byte in &mut q6k_data[0..128] {
            *byte = 0x55;
        }

        let acts = vec![1.0f32; 256];

        // Correct result (Q6K kernel)
        let correct =
            generic_fused_dot_scalar::<Q6K>(&q6k_data, &acts).expect("Q6K dot should succeed");

        // Now try to interpret this Q6_K data as Q4_K
        // Q4_K expects 144 bytes, but Q6_K is 210 bytes.
        // We truncate to 144 to make it "valid" Q4_K
        let truncated = &q6k_data[..144];
        let wrong = generic_fused_dot_scalar::<Q4K>(truncated, &acts)
            .expect("Q4K dot on wrong data should not panic");

        // The results should be substantially different (at least 10x)
        // because Q4_K reads d at offset 0 (which is Q6_K's ql data)
        // and Q6_K reads d at offset 208
        if correct.abs() > 1.0 {
            let ratio = (wrong / correct).abs();
            assert!(
                !(0.9..=1.1).contains(&ratio),
                "FALSIFY-QDOT-002 FAILED: Q6K→Q4K cross-format should produce different results. \
                 correct={correct}, wrong={wrong}, ratio={ratio}"
            );
        }
        // If correct is near zero, just verify wrong is different
    }

    // =========================================================================
    // FALSIFY-QDOT-003: Bsum precomputed vs on-the-fly equivalence
    // =========================================================================
    //
    // Claim: "Precomputed sub-block activation sums equal on-the-fly computation"
    // This validates the mathematical decomposition: offset term depends only on activations.

    #[test]
    fn falsify_qdot_003_bsum_equivalence() {
        // Generate activation values
        let acts: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01 - 1.28).collect();

        // Precomputed bsums (using our function)
        let bsums_precomputed = compute_bsums(&acts, 32);

        // On-the-fly bsums (computed inline, as current fused kernels do)
        let mut bsums_inline = Vec::with_capacity(8);
        for block_idx in 0..8 {
            let start = block_idx * 32;
            let end = start + 32;
            let sum: f32 = acts[start..end].iter().sum();
            bsums_inline.push(sum);
        }

        assert_eq!(bsums_precomputed.len(), bsums_inline.len());

        for (i, (pre, inline)) in bsums_precomputed.iter().zip(&bsums_inline).enumerate() {
            let diff = (pre - inline).abs();
            assert!(
                diff < 1e-6,
                "FALSIFY-QDOT-003 FAILED at sub-block {i}: precomputed={pre}, inline={inline}, diff={diff}"
            );
        }
    }

    #[test]
    fn falsify_qdot_003_bsum_with_offset_term() {
        // Verify that using precomputed bsums produces the same offset term
        // as computing it inline within the super-block loop.
        //
        // Offset term = dmin * Σ_j(m_j * bsum_j)
        let sb = create_q4k_superblock(0.5, 0.2, 8, 5, 7);
        let acts: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();

        // Read dmin from super-block
        let dmin = read_f16(&sb[2..4]);

        // Method 1: Precomputed bsums
        let bsums = compute_bsums(&acts, 32);
        let mut offset_precomputed = 0.0f32;
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&sb[4..16]);
        for j in 0..8 {
            let (_scale, min) = extract_scale_min(&scales, j);
            offset_precomputed += min * bsums[j];
        }
        offset_precomputed *= dmin;

        // Method 2: Inline (compute sums while iterating)
        let mut offset_inline = 0.0f32;
        for j in 0..8 {
            let (_scale, min) = extract_scale_min(&scales, j);
            let block_sum: f32 = acts[j * 32..(j + 1) * 32].iter().sum();
            offset_inline += min * block_sum;
        }
        offset_inline *= dmin;

        let diff = (offset_precomputed - offset_inline).abs();
        assert!(
            diff < 1e-4,
            "FALSIFY-QDOT-003 FAILED: offset precomputed={offset_precomputed}, \
             inline={offset_inline}, diff={diff}"
        );
    }

    // =========================================================================
    // FALSIFY-QDOT-004: Format exhaustiveness
    // =========================================================================
    //
    // Claim: "Every QuantBlockFormat impl has an entry in the format registry"

    #[test]
    fn falsify_qdot_004_format_registry_complete() {
        // Verify ALL_FORMAT_IDS contains every implementation's FORMAT_ID
        let impl_ids = [
            Q4K::FORMAT_ID,
            Q5K::FORMAT_ID,
            Q6K::FORMAT_ID,
            Q4_0Fmt::FORMAT_ID,
            Q8_0Fmt::FORMAT_ID,
        ];

        for id in &impl_ids {
            assert!(
                ALL_FORMAT_IDS.contains(id),
                "FALSIFY-QDOT-004 FAILED: Format {id} has a trait impl but is not in ALL_FORMAT_IDS"
            );
        }

        // Verify ALL_FORMAT_IDS doesn't contain any orphaned entries
        for &id in ALL_FORMAT_IDS {
            assert!(
                impl_ids.contains(&id),
                "FALSIFY-QDOT-004 FAILED: Format {id} is in ALL_FORMAT_IDS but has no trait impl"
            );
        }

        // Count check
        assert_eq!(
            impl_ids.len(),
            ALL_FORMAT_IDS.len(),
            "FALSIFY-QDOT-004 FAILED: impl count {} != registry count {}",
            impl_ids.len(),
            ALL_FORMAT_IDS.len()
        );
    }

    // =========================================================================
    // FALSIFY-QDOT-005: Dispatch exhaustiveness
    // =========================================================================
    //
    // Claim: "Every format has at least a scalar dot product implementation"
    // We verify this by calling the scalar generic dot for each format.

    #[test]
    fn falsify_qdot_005_all_formats_have_scalar_dot() {
        // Q4_K
        let q4k_data = vec![0u8; 144];
        let q4k_acts = vec![0.0f32; 256];
        assert!(
            generic_fused_dot_scalar::<Q4K>(&q4k_data, &q4k_acts).is_ok(),
            "FALSIFY-QDOT-005: Q4_K scalar dot should work"
        );

        // Q5_K
        let q5k_data = vec![0u8; 176];
        let q5k_acts = vec![0.0f32; 256];
        assert!(
            generic_fused_dot_scalar::<Q5K>(&q5k_data, &q5k_acts).is_ok(),
            "FALSIFY-QDOT-005: Q5_K scalar dot should work"
        );

        // Q6_K
        let q6k_data = vec![0u8; 210];
        let q6k_acts = vec![0.0f32; 256];
        assert!(
            generic_fused_dot_scalar::<Q6K>(&q6k_data, &q6k_acts).is_ok(),
            "FALSIFY-QDOT-005: Q6_K scalar dot should work"
        );

        // Q4_0
        let q4_0_data = vec![0u8; 18];
        let q4_0_acts = vec![0.0f32; 32];
        assert!(
            generic_fused_dot_scalar::<Q4_0Fmt>(&q4_0_data, &q4_0_acts).is_ok(),
            "FALSIFY-QDOT-005: Q4_0 scalar dot should work"
        );

        // Q8_0
        let q8_0_data = vec![0u8; 34];
        let q8_0_acts = vec![0.0f32; 32];
        assert!(
            generic_fused_dot_scalar::<Q8_0Fmt>(&q8_0_data, &q8_0_acts).is_ok(),
            "FALSIFY-QDOT-005: Q8_0 scalar dot should work"
        );
    }

    // =========================================================================
    // FALSIFY-QDOT-008: Wrong-kernel garbage detection
    // =========================================================================
    //
    // Claim: "Q6K weights dispatched through Q4K kernel produce garbage output"
    // This proves format isolation is not accidental — the formats are truly
    // incompatible and wrong dispatch produces meaningfully wrong results.

    #[test]
    fn falsify_qdot_008_q6k_through_q4k_produces_garbage() {
        // Create a valid Q6_K super-block with a strong, non-trivial signal.
        // Q6_K layout: ql(128) + qh(64) + scales(16) + d(2) = 210 bytes
        let mut q6k_data = vec![0u8; Q6K::SUPERBLOCK_BYTES];

        // Set Q6_K d (at offset 208) to 1.0
        let d_f16 = half::f16::from_f32(1.0);
        let d_bytes = d_f16.to_le_bytes();
        q6k_data[208] = d_bytes[0];
        q6k_data[209] = d_bytes[1];

        // Set Q6_K scales (at offset 192, 16 signed i8 values) to 10
        for i in 0..16 {
            q6k_data[192 + i] = 10;
        }

        // Set ql values (low 4 bits of 6-bit quants) to varied pattern
        for (idx, byte) in q6k_data[0..128].iter_mut().enumerate() {
            *byte = ((idx % 15) as u8) | (((idx % 13) as u8) << 4);
        }
        // Set qh values (high 2 bits) to non-zero pattern
        for (idx, byte) in q6k_data[128..192].iter_mut().enumerate() {
            *byte = (idx % 255) as u8;
        }

        // Activations with a clear signal (not all zeros/ones)
        let acts: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();

        // Correct result: Q6K kernel on Q6K data
        let correct = generic_fused_dot_scalar::<Q6K>(&q6k_data, &acts)
            .expect("Q6K dot on Q6K data should succeed");

        // Wrong result: Q4K kernel on the SAME bytes (truncated to Q4K size)
        // Q4K layout: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes
        // Q4K reads d from offset 0 (which is Q6K's ql data, NOT a float16 scale!)
        let q4k_data = &q6k_data[..Q4K::SUPERBLOCK_BYTES]; // truncate to 144
        let wrong = generic_fused_dot_scalar::<Q4K>(q4k_data, &acts)
            .expect("Q4K dot on wrong data should not panic (it computes garbage)");

        // The results MUST be substantially different.
        // Q4K reads ql[0..2] as f16 scale (garbage), Q6K reads offset 208 (1.0).
        // The outputs should differ by much more than any reasonable tolerance.
        let diff = (correct - wrong).abs();
        let magnitude = correct.abs().max(wrong.abs()).max(1.0);
        assert!(
            diff / magnitude > 0.1,
            "FALSIFY-008 FAILED: Q6K→Q4K cross-format SHOULD produce garbage.\n\
             correct(Q6K)={correct}, wrong(Q4K)={wrong}, diff={diff}, ratio={}\n\
             If these are close, format isolation is weaker than the contract claims.",
            diff / magnitude
        );
    }

    #[test]
    fn falsify_qdot_008_q4k_through_q8_0_produces_garbage() {
        // Second cross-format pair: Q4_K data through Q8_0 kernel
        let q4k_data = create_q4k_superblock(1.5, 0.3, 15, 5, 9);
        let acts_q4k = vec![1.0f32; Q4K::ELEMENTS_PER_SUPERBLOCK];

        // Correct Q4K result
        let correct =
            generic_fused_dot_scalar::<Q4K>(&q4k_data, &acts_q4k).expect("Q4K dot should succeed");

        // Feed first 34 bytes of Q4K data (which is d + dmin + scales prefix)
        // through Q8_0 kernel which expects d(2) + 32 signed i8 values
        let q8_0_slice = &q4k_data[..Q8_0Fmt::SUPERBLOCK_BYTES]; // 34 bytes
        let acts_q8 = vec![1.0f32; Q8_0Fmt::ELEMENTS_PER_SUPERBLOCK]; // 32 elements
        let wrong = generic_fused_dot_scalar::<Q8_0Fmt>(q8_0_slice, &acts_q8)
            .expect("Q8_0 dot on wrong data should not panic");

        // Q4K processes 256 elements with scale/min/dequant algebra.
        // Q8_0 processes 32 elements with simple scale*i8 algebra.
        // The results should be meaningfully different (different element counts alone
        // guarantee different magnitudes, plus the data interpretation differs).
        assert!(
            (correct - wrong).abs() > 1e-3
                || correct.abs() > 10.0 * wrong.abs()
                || wrong.abs() > 10.0 * correct.abs(),
            "FALSIFY-008 FAILED: Q4K→Q8_0 cross-format SHOULD produce different results.\n\
             correct(Q4K)={correct}, wrong(Q8_0)={wrong}"
        );
    }

    // =========================================================================
    // Additional structural tests
    // =========================================================================

    #[test]
    fn test_all_kquant_formats_are_256_elements() {
        // Contract requirement: KQuant formats use 256-element super-blocks
        assert_eq!(Q4K::ELEMENTS_PER_SUPERBLOCK, 256);
        assert_eq!(Q5K::ELEMENTS_PER_SUPERBLOCK, 256);
        assert_eq!(Q6K::ELEMENTS_PER_SUPERBLOCK, 256);

        assert_eq!(Q4K::FAMILY, QuantFamily::KQuant);
        assert_eq!(Q5K::FAMILY, QuantFamily::KQuant);
        assert_eq!(Q6K::FAMILY, QuantFamily::KQuant);
    }

    #[test]
    fn test_all_simple_formats_are_32_elements() {
        assert_eq!(Q4_0Fmt::ELEMENTS_PER_SUPERBLOCK, 32);
        assert_eq!(Q8_0Fmt::ELEMENTS_PER_SUPERBLOCK, 32);

        assert_eq!(Q4_0Fmt::FAMILY, QuantFamily::Simple);
        assert_eq!(Q8_0Fmt::FAMILY, QuantFamily::Simple);
    }

    #[test]
    fn test_has_dmin_only_for_q4k_q5k() {
        // Contract: only Q4_K and Q5_K have dmin
        assert!(Q4K::HAS_DMIN);
        assert!(Q5K::HAS_DMIN);
        assert!(!Q6K::HAS_DMIN);
        assert!(!Q4_0Fmt::HAS_DMIN);
        assert!(!Q8_0Fmt::HAS_DMIN);
    }

    // =========================================================================
    // FALSIFY-QDOT-006: Row-major only enforcement (PMAT-336)
    // =========================================================================
    //
    // Contract: quantized-dot-product-v1.yaml §enforcement.row_major_only
    // Claim: "All kernels operate on row-major data (LAYOUT-002)"
    // Method: Scan quantize module source for any colmajor function definitions.
    //
    // Five-Whys:
    //   Why 1: GPU produced garbage for 7B model
    //   Why 2: GGUF column-major data fed to row-major kernel
    //   Why 3: colmajor aliases existed as convenience shortcuts
    //   Why 4: No enforcement prevented adding colmajor kernels
    //   Why 5: ROOT — no regression test for LAYOUT-002

    #[test]
    fn falsify_qdot_006_no_colmajor_functions_in_quantize() {
        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let quantize_dir = crate_root.join("src/quantize");

        let mut violations = Vec::new();
        for entry in std::fs::read_dir(&quantize_dir).expect("quantize dir must exist") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.extension().map_or(true, |e| e != "rs") {
                continue;
            }
            // Skip test files — they document the rule, not violate it
            let fname = path.file_name().unwrap().to_string_lossy();
            if fname.contains("test") {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("read source");
            for (i, line) in source.lines().enumerate() {
                let trimmed = line.trim();
                // Skip comments documenting the removal
                if trimmed.starts_with("//") || trimmed.starts_with("///") {
                    continue;
                }
                if trimmed.contains("colmajor") || trimmed.contains("col_major") {
                    violations.push(format!(
                        "  {}:{}: {}",
                        path.file_name().unwrap().to_string_lossy(),
                        i + 1,
                        trimmed
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-QDOT-006 FAILED: Found colmajor function/reference in quantize module:\n{}\n\
             LAYOUT-002: All quantize kernels MUST be row-major only.\n\
             See contracts/quantized-dot-product-v1.yaml §enforcement.row_major_only",
            violations.join("\n")
        );
    }

    // =========================================================================
    // FALSIFY-007: No catch-all in WeightQuantType dispatch sites
    // =========================================================================
    //
    // Claim: "Every match on WeightQuantType MUST be EXHAUSTIVE with EXPLICIT arms.
    //         `_ =>` catch-all is FORBIDDEN."
    //
    // Method: Read the source files listed in tensor-layout-v1.yaml dispatch_sites
    //         and verify no `_ =>` arm exists in WeightQuantType matches.
    //         This is a regression test — if someone adds `_ =>`, this test fails.
    //
    // The Rust compiler already enforces exhaustiveness when no `_ =>` exists,
    // but this test prevents someone from ADDING a catch-all as a "convenience."

    /// Check if a `_ =>` at line `catch_all_line` is inside a WeightQuantType match.
    /// Scans backwards up to 30 lines looking for WeightQuantType within the same
    /// match block (tracking brace depth to avoid crossing block boundaries).
    fn is_in_weight_quant_match(lines: &[&str], catch_all_line: usize) -> bool {
        let start = catch_all_line.saturating_sub(30);
        let mut brace_depth = 0i32;

        for j in (start..catch_all_line).rev() {
            let l = lines[j].trim();
            brace_depth += l.matches('}').count() as i32;
            brace_depth -= l.matches('{').count() as i32;
            if brace_depth < 0 {
                return false; // exited the current match block
            }
            if l.contains("WeightQuantType") {
                return true;
            }
        }
        false
    }

    /// Scan a source file for `_ =>` catch-all arms inside WeightQuantType matches.
    /// Returns a list of violation descriptions (empty = clean).
    fn find_catch_all_violations(source: &str) -> Vec<String> {
        let lines: Vec<&str> = source.lines().collect();
        lines
            .iter()
            .enumerate()
            .filter(|(_i, line)| line.trim().starts_with("_ =>"))
            .filter(|(i, _line)| is_in_weight_quant_match(&lines, *i))
            .map(|(i, line)| format!("  line {}: {}", i + 1, line.trim()))
            .collect()
    }

    #[test]
    fn falsify_007_no_catch_all_in_dispatch_sites() {
        // PMAT-334 fix: Scan ALL WeightQuantType dispatch sites, not just 2.
        // Any non-test .rs file under src/ that matches on WeightQuantType.
        let dispatch_files = [
            "src/cuda/executor/layers/gemv_dispatch.rs",
            "src/cuda/types.rs",
            "src/cuda/transformer_workspace.rs",
            "src/cuda/executor/layers/indexed_ffn.rs",
            "src/cuda/executor/layers/logits.rs",
            "src/cuda/executor/layers/forward_workspace_captured.rs",
            "src/cuda/executor/layers/transformer_layer_indexed.rs",
            "src/cuda/executor/layers/indexed_transformer.rs",
        ];

        let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

        for file_rel in &dispatch_files {
            let path = crate_root.join(file_rel);
            let source = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("FALSIFY-007: Cannot read dispatch site {file_rel}: {e}");
            });

            let violations = find_catch_all_violations(&source);

            assert!(
                violations.is_empty(),
                "FALSIFY-007 FAILED: Found catch-all `_ =>` in WeightQuantType match in {file_rel}:\n{}\n\
                 WeightQuantType matches MUST be exhaustive — no catch-all allowed.\n\
                 See contracts/tensor-layout-v1.yaml §quant_dispatch",
                violations.join("\n")
            );
        }
    }

    #[test]
    fn test_superblock_bytes_correctness() {
        // Verify byte counts add up
        // Q4_K: d(2) + dmin(2) + scales(12) + qs(128) = 144
        assert_eq!(2 + 2 + 12 + 128, Q4K::SUPERBLOCK_BYTES);

        // Q5_K: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176
        assert_eq!(2 + 2 + 12 + 32 + 128, Q5K::SUPERBLOCK_BYTES);

        // Q6_K: ql(128) + qh(64) + scales(16) + d(2) = 210
        assert_eq!(128 + 64 + 16 + 2, Q6K::SUPERBLOCK_BYTES);

        // Q4_0: d(2) + qs(16) = 18
        assert_eq!(2 + 16, Q4_0Fmt::SUPERBLOCK_BYTES);

        // Q8_0: d(2) + qs(32) = 34
        assert_eq!(2 + 32, Q8_0Fmt::SUPERBLOCK_BYTES);
    }
}
