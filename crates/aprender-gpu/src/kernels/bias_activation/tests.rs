use super::*;
use crate::kernels::Kernel;

#[test]
fn test_bias_activation_default_config() {
    let kernel = BiasActivationKernel::new(1024, 64);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.bias_size, 64);
    assert_eq!(kernel.activation, Activation::None);
}

// ===== Derived Trait Coverage Tests =====

#[test]
fn test_activation_default_trait() {
    // Test Default trait implementation
    let activation: Activation = Default::default();
    assert_eq!(activation, Activation::None);
    assert_eq!(Activation::default(), Activation::None);
}

#[test]
fn test_activation_clone_trait() {
    // Test Clone trait implementation
    let original = Activation::ReLU;
    let cloned = original.clone();
    assert_eq!(original, cloned);

    let gelu = Activation::GELU;
    let gelu_cloned = gelu.clone();
    assert_eq!(gelu, gelu_cloned);
}

#[test]
fn test_activation_copy_trait() {
    // Test Copy trait - assignment should copy, not move
    let original = Activation::GELU;
    let copied = original; // Copy, not move
    assert_eq!(original, copied); // original still usable
    assert_eq!(copied, Activation::GELU);
}

#[test]
fn test_activation_debug_trait() {
    // Test Debug trait implementation
    let debug_none = format!("{:?}", Activation::None);
    let debug_relu = format!("{:?}", Activation::ReLU);
    let debug_gelu = format!("{:?}", Activation::GELU);

    assert!(debug_none.contains("None"));
    assert!(debug_relu.contains("ReLU"));
    assert!(debug_gelu.contains("GELU"));
}

#[test]
fn test_activation_eq_trait() {
    // Test PartialEq and Eq traits
    assert_eq!(Activation::None, Activation::None);
    assert_eq!(Activation::ReLU, Activation::ReLU);
    assert_eq!(Activation::GELU, Activation::GELU);

    assert_ne!(Activation::None, Activation::ReLU);
    assert_ne!(Activation::ReLU, Activation::GELU);
    assert_ne!(Activation::None, Activation::GELU);
}

#[test]
fn test_kernel_clone_trait() {
    // Test Clone trait on BiasActivationKernel
    let original = BiasActivationKernel::new(2048, 128).with_gelu();
    let cloned = original.clone();

    assert_eq!(cloned.n, original.n);
    assert_eq!(cloned.bias_size, original.bias_size);
    assert_eq!(cloned.activation, original.activation);

    // Verify both produce valid PTX (register order may vary due to HashSet)
    let original_ptx = original.emit_ptx();
    let cloned_ptx = cloned.emit_ptx();

    assert!(original_ptx.contains(".entry bias_activation"));
    assert!(cloned_ptx.contains(".entry bias_activation"));
    assert!(original_ptx.contains("ex2.approx")); // GELU signature
    assert!(cloned_ptx.contains("ex2.approx")); // GELU signature
}

#[test]
fn test_kernel_debug_trait() {
    // Test Debug trait on BiasActivationKernel
    let kernel = BiasActivationKernel::new(512, 32).with_relu();
    let debug_output = format!("{:?}", kernel);

    assert!(debug_output.contains("BiasActivationKernel"));
    assert!(debug_output.contains("512"));
    assert!(debug_output.contains("32"));
    assert!(debug_output.contains("ReLU"));
}

// ===== Edge Case Tests =====

#[test]
fn test_minimum_sizes() {
    // Test with minimum valid sizes
    let kernel = BiasActivationKernel::new(1, 1);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry bias_activation"));
    assert!(ptx.contains("rem.u32"));
}

#[test]
fn test_large_sizes() {
    // Test with large sizes typical in ML workloads
    let kernel = BiasActivationKernel::new(1_000_000, 4096).with_gelu();
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".entry bias_activation"));
    assert!(ptx.contains("ex2")); // GELU uses exp
}

#[test]
fn test_bias_size_equals_n() {
    // Test when bias_size equals n (no modulo wrap)
    let kernel = BiasActivationKernel::new(64, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("rem.u32")); // Still uses modulo
}

#[test]
fn test_bias_size_larger_than_n() {
    // Test when bias_size > n (subset of bias vector used)
    let kernel = BiasActivationKernel::new(32, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("rem.u32"));
}

#[test]
fn test_with_activation_none() {
    // Test with_activation for None variant
    let kernel = BiasActivationKernel::new(1024, 64)
        .with_relu() // Set to ReLU first
        .with_activation(Activation::None); // Then override

    assert_eq!(kernel.activation, Activation::None);

    let ptx = kernel.emit_ptx();
    // Should not have max.f32 (ReLU) or ex2 (GELU)
    assert!(!ptx.contains("max.f32"));
}

#[test]
fn test_chained_activation_changes() {
    // Test that chaining activation changes works correctly
    let kernel = BiasActivationKernel::new(1024, 64)
        .with_relu()
        .with_gelu()
        .with_activation(Activation::None)
        .with_relu();

    assert_eq!(kernel.activation, Activation::ReLU);
}

#[test]
fn test_none_activation_ptx_structure() {
    // Verify None activation produces minimal PTX (no activation ops)
    let kernel = BiasActivationKernel::new(1024, 64).with_activation(Activation::None);
    let ptx = kernel.emit_ptx();

    // Should have bias addition
    assert!(ptx.contains("add.f32"));
    // Should NOT have ReLU max
    assert!(!ptx.contains("max.f32"));
}

#[test]
fn test_relu_activation_ptx_structure() {
    // Verify ReLU produces correct PTX structure
    let kernel = BiasActivationKernel::new(1024, 64).with_activation(Activation::ReLU);
    let ptx = kernel.emit_ptx();

    // ReLU needs: max.f32 and a zero constant
    assert!(ptx.contains("max.f32"));
    assert!(ptx.contains("mov.f32") || ptx.contains("0.0")); // Zero for ReLU
}

#[test]
fn test_gelu_activation_ptx_structure() {
    // Verify GELU produces correct PTX structure with all components
    let kernel = BiasActivationKernel::new(1024, 64).with_activation(Activation::GELU);
    let ptx = kernel.emit_ptx();

    // GELU needs: mul (for 1.702*x and x*sigmoid), ex2 (for exp), div (for 1/(1+exp))
    assert!(ptx.contains("mul.f32"));
    assert!(ptx.contains("ex2"));
    assert!(ptx.contains("div"));
    assert!(ptx.contains("sub.f32")); // For negation (0 - scaled)
}

#[test]
fn test_bias_activation_with_relu() {
    let kernel = BiasActivationKernel::new(1024, 64).with_relu();
    assert_eq!(kernel.activation, Activation::ReLU);
}

#[test]
fn test_bias_activation_with_gelu() {
    let kernel = BiasActivationKernel::new(1024, 64).with_gelu();
    assert_eq!(kernel.activation, Activation::GELU);
}

#[test]
fn test_bias_activation_kernel_name() {
    let kernel = BiasActivationKernel::new(1024, 64);
    assert_eq!(kernel.name(), "bias_activation");
}

#[test]
fn test_bias_activation_ptx_generation() {
    let kernel = BiasActivationKernel::new(1024, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".version 8.0"), "Missing PTX version");
    assert!(ptx.contains(".target sm_70"), "Missing target");
    assert!(ptx.contains(".visible .entry bias_activation"), "Missing entry point");
    assert!(ptx.contains(".param .u64 output"), "Missing output param");
    assert!(ptx.contains(".param .u64 bias"), "Missing bias param");
    assert!(ptx.contains(".param .u32 n"), "Missing n param");
    // bias_size is baked into kernel at generation time (for efficiency)
}

#[test]
fn test_bias_activation_relu_ptx() {
    let kernel = BiasActivationKernel::new(1024, 64).with_relu();
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("max.f32"), "ReLU should use max.f32");
}

#[test]
fn test_bias_activation_gelu_ptx() {
    let kernel = BiasActivationKernel::new(1024, 64).with_gelu();
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("ex2.approx") || ptx.contains("ex2.f32"), "GELU should use ex2 for exp");
    assert!(
        ptx.contains("div.rn.f32") || ptx.contains("div.f32"),
        "GELU should use div for sigmoid reciprocal"
    );
}

#[test]
fn test_bias_activation_contains_bias_addition() {
    let kernel = BiasActivationKernel::new(1024, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("add.f32"), "Should contain bias addition");
    assert!(ptx.contains("rem.u32"), "Should contain modulo for bias indexing");
}

#[test]
fn test_bias_activation_bounds_check() {
    let kernel = BiasActivationKernel::new(1024, 64);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("setp.ge.u32"), "Should have bounds check");
}

mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn bias_activation_always_valid(n in 64u32..8192, bias_size in 16u32..512) {
            let kernel = BiasActivationKernel::new(n, bias_size);
            let ptx = kernel.emit_ptx();

            prop_assert!(ptx.contains(".version"), "Missing PTX version");
            prop_assert!(ptx.contains(".entry"), "Missing entry point");
            prop_assert!(ptx.contains("bias_activation"), "Missing kernel name");
        }

        #[test]
        fn activation_variants_produce_valid_ptx(n in 64u32..4096, bias_size in 16u32..256) {
            for activation in [Activation::None, Activation::ReLU, Activation::GELU] {
                let kernel = BiasActivationKernel::new(n, bias_size)
                    .with_activation(activation);
                let ptx = kernel.emit_ptx();

                prop_assert!(ptx.contains(".version"), "Missing PTX version for {:?}", activation);
                prop_assert!(ptx.contains("bias_activation"), "Missing kernel name for {:?}", activation);
            }
        }

        /// Fast test: Power-of-2 sizes (common in ML)
        #[test]
        fn power_of_two_sizes_valid(exp_n in 6u32..14, exp_bias in 4u32..10) {
            let n = 1u32 << exp_n;  // 64 to 8192
            let bias_size = 1u32 << exp_bias;  // 16 to 512
            let kernel = BiasActivationKernel::new(n, bias_size);
            let ptx = kernel.emit_ptx();

            prop_assert!(ptx.contains(".entry"), "Power-of-2 size {} failed", n);
            prop_assert!(ptx.contains("rem.u32"), "Must have modulo for bias indexing");
        }

        /// Fast test: Non-aligned sizes (edge cases)
        #[test]
        fn non_aligned_sizes_valid(n in 1u32..1000, bias_size in 1u32..100) {
            // Skip n=0 which would be invalid
            let n = n.max(1);
            let bias_size = bias_size.max(1);
            let kernel = BiasActivationKernel::new(n, bias_size);
            let ptx = kernel.emit_ptx();

            prop_assert!(ptx.contains(".version"), "Non-aligned size n={} failed", n);
        }

        /// Fast test: PTX generation produces consistent instructions
        /// Note: Register declaration order may vary, but instructions must be identical
        #[test]
        fn ptx_generation_consistent(n in 64u32..1024, bias_size in 16u32..128) {
            let kernel1 = BiasActivationKernel::new(n, bias_size).with_gelu();
            let kernel2 = BiasActivationKernel::new(n, bias_size).with_gelu();

            let ptx1 = kernel1.emit_ptx();
            let ptx2 = kernel2.emit_ptx();

            // Extract instruction lines (skip register declarations which may vary in order)
            fn extract_instructions(ptx: &str) -> Vec<String> {
                ptx.lines()
                    .filter(|line| {
                        let trimmed = line.trim();
                        !trimmed.is_empty()
                            && !trimmed.starts_with("//")
                            && !trimmed.starts_with(".reg")
                    })
                    .map(|s| s.to_string())
                    .collect()
            }

            let instructions1 = extract_instructions(&ptx1);
            let instructions2 = extract_instructions(&ptx2);

            prop_assert_eq!(
                instructions1, instructions2,
                "PTX instructions must be consistent (excluding register declarations)"
            );
        }

        /// Fast test: ReLU always produces max instruction
        #[test]
        fn relu_always_has_max(n in 64u32..4096, bias_size in 16u32..256) {
            let kernel = BiasActivationKernel::new(n, bias_size).with_relu();
            let ptx = kernel.emit_ptx();

            prop_assert!(ptx.contains("max.f32"), "ReLU must use max.f32 instruction");
        }

        /// Fast test: GELU always uses exponential approximation
        #[test]
        fn gelu_always_has_exp(n in 64u32..4096, bias_size in 16u32..256) {
            let kernel = BiasActivationKernel::new(n, bias_size).with_gelu();
            let ptx = kernel.emit_ptx();

            prop_assert!(
                ptx.contains("ex2.approx") || ptx.contains("ex2.f32"),
                "GELU must use ex2 for exponential"
            );
        }
    }
}

/// Falsification tests - verify specific invariants hold
mod falsification_tests {
    use super::*;

    /// FALSIFY: PTX without bounds check would access out-of-bounds memory
    #[test]
    fn falsify_bounds_check_present() {
        // If bounds check is missing, kernel would crash on n < grid_size
        let kernel = BiasActivationKernel::new(1, 1);
        let ptx = kernel.emit_ptx();

        assert!(
            ptx.contains("setp.ge.u32") && ptx.contains("bra"),
            "FALSIFIED: Missing bounds check - kernel would crash on small inputs"
        );
    }

    /// FALSIFY: PTX without bias modulo would read wrong bias values
    #[test]
    fn falsify_bias_modulo_present() {
        // If rem.u32 is missing, bias indexing would be wrong when n > bias_size
        let kernel = BiasActivationKernel::new(1024, 64);
        let ptx = kernel.emit_ptx();

        assert!(
            ptx.contains("rem.u32"),
            "FALSIFIED: Missing rem.u32 - bias indexing would be incorrect"
        );
    }

    /// FALSIFY: ReLU without max would not clamp negative values
    #[test]
    fn falsify_relu_has_max() {
        let kernel = BiasActivationKernel::new(1024, 64).with_relu();
        let ptx = kernel.emit_ptx();

        assert!(
            ptx.contains("max.f32"),
            "FALSIFIED: ReLU without max.f32 - negative values would pass through"
        );
    }

    /// FALSIFY: GELU without sigmoid would not approximate correctly
    #[test]
    fn falsify_gelu_has_sigmoid_components() {
        let kernel = BiasActivationKernel::new(1024, 64).with_gelu();
        let ptx = kernel.emit_ptx();

        // GELU needs: exp (via ex2), division (for 1/(1+exp)), multiply
        assert!(ptx.contains("ex2"), "FALSIFIED: GELU missing exp component");
        assert!(ptx.contains("div"), "FALSIFIED: GELU missing division for sigmoid");
    }

    /// FALSIFY: None activation should not have ReLU or GELU instructions
    #[test]
    fn falsify_none_activation_minimal() {
        let kernel = BiasActivationKernel::new(1024, 64); // Default is None
        let ptx = kernel.emit_ptx();

        // None activation should not have max (ReLU) or ex2 (GELU)
        // It should only have add for bias
        assert!(!ptx.contains("max.f32"), "FALSIFIED: None activation has ReLU max instruction");
        // Note: We can't assert no ex2 since the PTX builder might use it elsewhere
        // But we verify the essential add is present
        assert!(ptx.contains("add.f32"), "FALSIFIED: None activation missing bias addition");
    }

    /// FALSIFY: Kernel without proper parameters would fail at runtime
    #[test]
    fn falsify_all_params_present() {
        let kernel = BiasActivationKernel::new(1024, 64);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".param .u64 output"), "Missing output pointer param");
        assert!(ptx.contains(".param .u64 bias"), "Missing bias pointer param");
        assert!(ptx.contains(".param .u32 n"), "Missing n param");
    }
}
