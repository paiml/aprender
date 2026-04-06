
// ============================================================================
// QWEN-007: Q8 Dequantization Kernel
// ============================================================================

/// Generate PTX for Q8 dequantization kernel
///
/// Dequantizes INT8 values to FP32 using per-block scales (block size = 32)
/// Formula: output[i] = quants[i] * scales[i / 32]
///
/// Parameters:
/// - quants: i8* input quantized values
/// - scales: f32* per-block scale factors
/// - output: f32* dequantized output
/// - n: u32 number of elements
fn generate_q8_dequant_ptx(_n: u32) -> String {
    // Note: n is used by caller for launch config, not embedded in PTX
    // The kernel uses n_param from arguments for bounds checking
    // PMAT-044: This raw PTX uses basic instructions only (no SM-specific features).
    // sm_70 forward-compiles to all sm_70+ devices via PTX JIT.
    // Unlike trueno-gpu generated kernels, this raw PTX cannot take a runtime target.
    r"
.version 8.0
.target sm_70
.address_size 64

.visible .entry q8_dequant(
    .param .u64 quants_ptr,
    .param .u64 scales_ptr,
    .param .u64 output_ptr,
    .param .u32 n_param
) {{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    .reg .b16 %h<2>;

    // Get global thread index
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;  // global_idx = blockIdx * blockDim + threadIdx

    // Load n parameter
    ld.param.u32 %r4, [n_param];

    // Bounds check
    setp.ge.u32 %p0, %r3, %r4;
    @%p0 bra $L_exit;

    // Load pointers
    ld.param.u64 %rd0, [quants_ptr];
    ld.param.u64 %rd1, [scales_ptr];
    ld.param.u64 %rd2, [output_ptr];

    // Calculate quants address: quants_ptr + global_idx
    cvt.u64.u32 %rd3, %r3;
    add.u64 %rd4, %rd0, %rd3;

    // Load quantized value (i8)
    ld.global.s8 %h0, [%rd4];
    cvt.rn.f32.s16 %f0, %h0;  // Convert i8 to f32

    // Calculate scale index: global_idx / 32
    shr.u32 %r5, %r3, 5;  // scale_idx = global_idx >> 5

    // Calculate scales address: scales_ptr + scale_idx * 4
    cvt.u64.u32 %rd5, %r5;
    shl.b64 %rd5, %rd5, 2;  // scale_idx * 4 (bytes)
    add.u64 %rd6, %rd1, %rd5;

    // Load scale (f32)
    ld.global.f32 %f1, [%rd6];

    // Dequantize: output = quant * scale
    mul.f32 %f2, %f0, %f1;

    // Calculate output address: output_ptr + global_idx * 4
    shl.b64 %rd3, %rd3, 2;  // global_idx * 4 (bytes)
    add.u64 %rd7, %rd2, %rd3;

    // Store result
    st.global.f32 [%rd7], %f2;

$L_exit:
    ret;
}}
"
    .to_string()
}

// ============================================================================
// Tests (PMAT-802: T-COV-95)
// ============================================================================

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod kernels_tests;
