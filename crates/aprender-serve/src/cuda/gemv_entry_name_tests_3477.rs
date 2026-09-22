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
        ]
    }

    #[test]
    fn the_name_the_launcher_looks_up_exists_in_the_ptx_it_compiles() {
        let kernels = CudaKernels::new();
        let mut broken = Vec::new();
        for kt in every_gemv_kernel() {
            let name = kernels.kernel_name(&kt);
            let ptx = kernels.generate_ptx(&kt);
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
