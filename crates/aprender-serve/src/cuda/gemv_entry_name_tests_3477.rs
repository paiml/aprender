// #3477: the launcher looks a kernel up BY NAME inside the module it just
// compiled. `kernel_name()` and the emitted PTX's `.visible .entry` are two
// strings in two files that must agree, and nothing checked that they did.
//
// A mismatch does not fail to compile and does not fail to load — the module
// builds, and the failure is a function lookup at launch time, on a GPU box
// only, reported as a CUDA error rather than as "these two strings disagree".
// The existing PTX tests in this crate assert `ptx.contains(".version")`, which
// no rename can violate.
//
// This is GPU-FREE: it generates PTX and reads it. It does need `--features
// cuda` to compile, so it runs in the cuda-unit CI job and NOT in the coverage
// target (which does not pass the feature).
#[cfg(test)]
mod gemv_entry_name_tests_3477 {
    use crate::cuda::{CudaKernels, KernelType};

    /// Every GEMV kernel the dispatch can reach, with the shape it is built at.
    fn every_gemv_kernel() -> Vec<KernelType> {
        let (k, n) = (4096u32, 4096u32);
        vec![
            KernelType::Gemv { k, n },
            KernelType::CoalescedGemv { k, n },
            KernelType::Q4KGemv { k, n },
            KernelType::Q5KGemv { k, n },
            KernelType::Q6KGemv { k, n },
            KernelType::Q8_0Gemv { k, n },
            KernelType::Q5_0Gemv { k, n },
            KernelType::Q4_0Gemv { k, n },
            KernelType::Q4_1Gemv { k, n },
            KernelType::F16Gemv { k, n },
            KernelType::Iq4XsGemv { k, n },
        ]
    }

    #[test]
    fn the_name_the_launcher_looks_up_exists_in_the_ptx_it_compiles() {
        let kernels = CudaKernels::new();
        let mut broken = Vec::new();
        for kt in every_gemv_kernel() {
            let name = kernels.kernel_name(&kt);
            let ptx = kernels.generate_ptx(&kt);
            // ptxas rejects a non-ASCII byte ANYWHERE, comments included, and
            // fails the whole module with `ptxas fatal : Unexpected non-ASCII
            // character`. The entry name can be perfectly correct and the module
            // still never assemble — which is exactly what an em dash in one of
            // MY comments did to the F16 and IQ4_XS kernels (#3477, caught by
            // aprender-45's A/B on its first real run, not by this guard).
            if let Some(bad) = ptx.chars().find(|c| !c.is_ascii()) {
                let line = ptx
                    .lines()
                    .find(|l| l.chars().any(|c| !c.is_ascii()))
                    .unwrap_or("")
                    .trim();
                broken.push(format!(
                    "\n  - {kt:?}: emitted PTX contains U+{:04X}, which ptxas refuses \
                     even inside a comment, so the module never assembles: {line:?}",
                    bad as u32
                ));
                continue;
            }
            let entry = format!(".visible .entry {name}(");
            if !ptx.contains(&entry) {
                let found: Vec<&str> = ptx
                    .lines()
                    .filter(|l| l.contains(".visible .entry"))
                    .map(str::trim)
                    .collect();
                broken.push(format!(
                    "\n  - {kt:?}: kernel_name() says {name:?}, but the PTX declares {found:?}"
                ));
            }
        }
        assert!(
            broken.is_empty(),
            "{} GEMV kernel(s) would fail the function lookup at launch — the name \
             the launcher asks for is not an entry in the PTX it just compiled:{}",
            broken.len(),
            broken.join("")
        );
    }

    /// The four places a new quant type has to be registered, asserted
    /// together. They were consistent before F16 and the compiler enforces the
    /// two `match`es; these are the ones it cannot check — a wrong BYTE COUNT
    /// compiles fine and reads the wrong bytes.
    #[test]
    fn f16_is_registered_consistently_in_every_table() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};

        assert_eq!(WeightQuantType::from_ggml_type(1), Some(WeightQuantType::F16));
        assert_eq!(WeightQuantType::F16.bytes_per_superblock(), 512, "256 elems x 2 bytes");
        assert_eq!(WeightQuantType::F16.bytes_per_block(), 64, "32 elems x 2 bytes");

        // A [2560, 32] ssm_alpha — the actual shape in Qwen3.5-4B-UD-Q4_K_XL.
        assert!(WeightQuantType::F16.matches_size(2560 * 32 * 2, 2560, 32));
        assert!(!WeightQuantType::F16.matches_size(2560 * 32 * 4, 2560, 32), "that is F32's size");

        // from_size must not let F32's check swallow F16, nor the reverse.
        assert_eq!(WeightQuantType::from_size(2560 * 32 * 2, 2560, 32), Some(WeightQuantType::F16));
        assert_eq!(WeightQuantType::from_size(2560 * 32 * 4, 2560, 32), Some(WeightQuantType::F32));

        assert_eq!(
            crate::cuda::types::BoundWeight::bind(0x1000, 2560 * 32 * 2, WeightQuantType::F16, 2560, 32)
                .kernel(),
            GemvKernel::F16,
            "binding F16 to any other kernel decodes 2-byte weights as something else"
        );
    }

    /// The whitelist is NOT widened here. #3477 lands the kernel; the
    /// `gpu_unsupported_quant_qtype` entry is a separate change behind a
    /// verified A/B, because a whitelist entry without a working kernel turns
    /// an honest refusal into silent Q4_K-decode garbage — the 0.68.1 defect.
    #[test]
    fn the_kernel_lands_before_the_whitelist_does() {
        assert!(
            crate::gguf::gpu_unsupported_quant_qtype(1),
            "F16 must still be refused until the A/B proves this kernel on real bytes"
        );
    }

    /// The IQ4_XS kernel's INDEX MATH, executed in Rust exactly as the PTX
    /// executes it, checked against the verified CPU decoder on the same bytes.
    ///
    /// WHAT THIS PROVES: the thread mapping is right — that `ib == m` and
    /// `jj == tid` (which holds only because `tid < 32`), that `tid >> 4`
    /// selects the nibble and `tid & 15` the byte, that the 6-bit scale is
    /// reassembled from `scales_l`/`scales_h` correctly, and that the codebook
    /// is indexed correctly. Index math is the likely bug class in a port and
    /// it is checkable with no GPU.
    ///
    /// WHAT IT DOES NOT PROVE: that the PTX *text* implements this model. That
    /// gap closes with the tensor-level A/B on real device bytes, and nothing
    /// on a CPU-only box can close it. Recorded rather than glossed.
    #[test]
    fn the_iq4_xs_thread_mapping_reproduces_the_cpu_decoder() {
        use crate::quantize::iq4_xs::{dequantize_iq4_xs_block, IQ4_XS_BLOCK_BYTES};

        // KVALUES_IQ4NL, the 16 non-linear levels, as the PTX embeds them.
        const KV: [i32; 16] =
            [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

        // A deterministic pseudo-random block; byte 0..2 is the f16 scale, kept
        // to a cleanly representable value so the comparison is about indexing.
        let mut block = [0u8; IQ4_XS_BLOCK_BYTES];
        let mut x: u32 = 0x1234_5678;
        for b in block.iter_mut() {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (x >> 24) as u8;
        }
        block[0] = 0x00;
        block[1] = 0x3c; // f16 1.0

        let mut expected = [0f32; 256];
        dequantize_iq4_xs_block(&block, &mut expected);

        // ---- exactly what the PTX does, one lane at a time ----
        let d = f32::from(half_from_le(block[0], block[1]));
        let scales_h = u32::from(u16::from_le_bytes([block[2], block[3]]));
        let mut got = [0f32; 256];
        for tid in 0..32usize {
            let jhalf = tid >> 4;
            let jlow = tid & 15;
            for m in 0..8usize {
                let ls_low = (u32::from(block[4 + (m >> 1)]) >> (4 * (m & 1))) & 0xf;
                let ls_high = ((scales_h >> (2 * m)) & 3) << 4;
                #[allow(clippy::cast_precision_loss)]
                let dl = d * ((ls_low | ls_high) as f32 - 32.0);
                let byte = u32::from(block[8 + 16 * m + jlow]);
                let nib = (byte >> (jhalf * 4)) & 15;
                #[allow(clippy::cast_precision_loss)]
                let w = dl * KV[nib as usize] as f32;
                got[32 * m + tid] = w;
            }
        }

        let mut wrong = Vec::new();
        for i in 0..256 {
            if (got[i] - expected[i]).abs() > 1e-6 {
                wrong.push(format!("\n  - element {i}: kernel {} vs decoder {}", got[i], expected[i]));
            }
        }
        assert!(
            wrong.is_empty(),
            "{} of 256 elements disagree with dequantize_iq4_xs_block — the kernel's \
             index math does not match the reference it was ported from:{}",
            wrong.len(),
            wrong.join("")
        );
    }

    /// f16 bits -> f32, for the test's own scale. Kept local so the test does
    /// not depend on a `pub(crate)` helper's visibility.
    fn half_from_le(lo: u8, hi: u8) -> f32 {
        let bits = u16::from_le_bytes([lo, hi]);
        let sign = f32::from_bits(u32::from(bits & 0x8000) << 16);
        let exp = i32::from((bits >> 10) & 0x1f);
        let frac = f32::from(bits & 0x3ff);
        let mag = if exp == 0 {
            frac * 2f32.powi(-24)
        } else {
            (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
        };
        if sign.is_sign_negative() { -mag } else { mag }
    }

    /// IQ4_XS must be registered with the right block size. 136 bytes per 256
    /// elements: a wrong number here walks the row at the wrong stride and
    /// every block after the first is misread.
    #[test]
    fn iq4_xs_is_registered_with_the_right_block_size() {
        use crate::cuda::types::{GemvKernel, WeightQuantType};
        assert_eq!(WeightQuantType::from_ggml_type(23), Some(WeightQuantType::IQ4XS));
        assert_eq!(WeightQuantType::IQ4XS.bytes_per_superblock(), 136);
        // [2560, 9216] ffn_gate — the real shape in the UD model.
        let nsb = 2560 * (9216 / 256);
        assert!(WeightQuantType::IQ4XS.matches_size(nsb * 136, 2560, 9216));
        assert_eq!(WeightQuantType::from_size(nsb * 136, 2560, 9216), Some(WeightQuantType::IQ4XS));
        assert_eq!(
            crate::cuda::types::BoundWeight::bind(0x1000, nsb * 136, WeightQuantType::IQ4XS, 2560, 9216)
                .kernel(),
            GemvKernel::IQ4XS
        );
        assert!(
            crate::gguf::gpu_unsupported_quant_qtype(23),
            "IQ4_XS must still be refused until the A/B proves this kernel on real bytes"
        );
    }

    /// #3477 specifically: F16 is the new one, and its PTX must carry the
    /// row-major 2-byte stride LAYOUT-001 requires. A 4-byte stride here reads
    /// every other weight and is the silent-garbage shape this ticket exists to
    /// avoid.
    #[test]
    fn the_f16_kernel_strides_two_bytes_per_weight() {
        let kernels = CudaKernels::new();
        let ptx = kernels.generate_ptx(&KernelType::F16Gemv { k: 4096, n: 4096 });
        assert!(
            ptx.contains("cvt.f32.f16"),
            "an F16 GEMV that never converts f16 to f32 is reading raw bits as floats"
        );
        assert!(
            ptx.contains("ld.global.b16"),
            "the weight load must be 16-bit; a 32-bit load reads two weights at once"
        );
        // row_base = w_ptr + row * (k << 1): the `<< 1` IS the 2-bytes-per-f16.
        assert!(
            ptx.contains("shl.b32 %r4, %r3, 1;"),
            "the row stride must be k*2 bytes — row-major, one f16 per weight"
        );
        assert!(
            !ptx.contains("colmajor") && !ptx.contains("col_major"),
            "LAYOUT-001: GGUF/APR data is row-major on this path"
        );
    }
}
