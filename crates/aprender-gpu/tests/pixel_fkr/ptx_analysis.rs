//! PTX Static Analysis Tests (using probar gpu_pixels)

#![cfg(feature = "gpu-pixels")]

use super::*;

/// ptx-pixel-fkr: GEMM tiled kernel has no shared memory bugs
#[test]
fn ptx_pixel_fkr_gemm_tiled_no_bugs() {
    let kernel = GemmKernel::tiled(32, 32, 128, 32);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "GEMM tiled kernel uses u64 for shared memory (should be u32)"
    );

    assert!(
        !result.has_bug(&PtxBugClass::MissingBarrierSync),
        "GEMM tiled kernel missing barrier synchronization"
    );

    println!("ptx_pixel_fkr_gemm_tiled: PASS (no shared memory bugs)");
}

/// ptx-pixel-fkr: Tensor core GEMM valid
#[test]
fn ptx_pixel_fkr_gemm_tensor_core() {
    let kernel = GemmKernel::tensor_core(32, 32, 64);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "Tensor core GEMM uses u64 for shared memory"
    );

    println!("ptx_pixel_fkr_gemm_tensor_core: PASS");
}

/// ptx-pixel-fkr: Attention kernel validation
#[test]
fn ptx_pixel_fkr_attention() {
    let kernel = AttentionKernel::new(64, 64);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::SharedMemU64Addressing),
        "Attention kernel uses u64 for shared memory"
    );

    assert!(ptx.contains("bar.sync"), "Attention kernel must have barrier synchronization");

    println!("ptx_pixel_fkr_attention: PASS");
}

/// ptx-pixel-fkr: Causal attention has correct name
#[test]
fn ptx_pixel_fkr_attention_causal() {
    let kernel = AttentionKernel::new(64, 64).with_causal();
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains("flash_attention_causal") || ptx.contains("causal"),
        "Causal attention should have _causal suffix"
    );

    println!("ptx_pixel_fkr_attention_causal: PASS");
}

/// ptx-pixel-fkr: Softmax kernel entry point
#[test]
fn ptx_pixel_fkr_softmax_entry() {
    let kernel = SoftmaxKernel::new(128);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::MissingEntryPoint),
        "Softmax kernel must have entry point"
    );

    println!("ptx_pixel_fkr_softmax: PASS");
}

/// ptx-pixel-fkr: LayerNorm kernel entry point
#[test]
fn ptx_pixel_fkr_layernorm_entry() {
    let kernel = LayerNormKernel::new(256);
    let ptx = kernel.emit_ptx();
    let result = validate_ptx(&ptx);

    assert!(
        !result.has_bug(&PtxBugClass::MissingEntryPoint),
        "LayerNorm kernel must have entry point"
    );

    println!("ptx_pixel_fkr_layernorm: PASS");
}

/// ptx-pixel-fkr: BiasActivation kernel entry point (all variants)
#[test]
fn ptx_pixel_fkr_bias_activation_entry() {
    for activation in [Activation::None, Activation::ReLU, Activation::GELU] {
        let kernel = BiasActivationKernel::new(1024, 64).with_activation(activation);
        let ptx = kernel.emit_ptx();
        let result = validate_ptx(&ptx);

        assert!(
            !result.has_bug(&PtxBugClass::MissingEntryPoint),
            "BiasActivation kernel ({:?}) must have entry point",
            activation
        );
    }

    println!("ptx_pixel_fkr_bias_activation: PASS (all variants)");
}

/// ptx-pixel-fkr: BiasActivation GELU uses approximation functions
#[test]
fn ptx_pixel_fkr_bias_activation_gelu_approx() {
    let kernel = BiasActivationKernel::new(1024, 64).with_gelu();
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains("ex2.approx") || ptx.contains("ex2.f32"),
        "GELU should use ex2 for fast exp approximation"
    );

    println!("ptx_pixel_fkr_bias_activation_gelu: PASS (uses ex2 approximation)");
}

/// ptx-pixel-fkr: BiasActivation ReLU uses max instruction
#[test]
fn ptx_pixel_fkr_bias_activation_relu_max() {
    let kernel = BiasActivationKernel::new(1024, 64).with_relu();
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("max.f32"), "ReLU should use max.f32 instruction");

    println!("ptx_pixel_fkr_bias_activation_relu: PASS (uses max.f32)");
}
