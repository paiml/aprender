//! Metal Shading Language (MSL) compute kernels
//!
//! Pre-built Metal compute shaders for common operations.
//! These kernels are equivalent to the PTX kernels for CUDA.
//!
//! # Usage
//!
//! These are source strings only. This crate contains no Metal dispatcher, so
//! nothing here compiles or runs them -- pass a constant to a Metal API of your
//! own (`MTLDevice::newLibraryWithSource`), or use the `wgpu` feature, which
//! reaches Apple GPUs through wgpu's Metal backend and does execute.
//!
//! ```ignore
//! use aprender_gpu::backend::metal_shaders;
//!
//! let msl: &str = metal_shaders::ELEMENTWISE_ADD; // kernel name: "elementwise_add"
//! ```

/// Element-wise vector addition kernel
///
/// Computes: C[i] = A[i] + B[i]
///
/// # Buffers
/// - buffer(0): A - input vector (read-only)
/// - buffer(1): B - input vector (read-only)
/// - buffer(2): C - output vector (write-only)
/// - buffer(3): count - number of elements (u32)
pub const ELEMENTWISE_ADD: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        C[id] = A[id] + B[id];
    }
}
"#;

/// Element-wise vector multiplication kernel
///
/// Computes: C[i] = A[i] * B[i]
pub const ELEMENTWISE_MUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void elementwise_mul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        C[id] = A[id] * B[id];
    }
}
"#;

/// Scalar-vector multiplication kernel
///
/// Computes: B[i] = scalar * A[i]
pub const SCALAR_MUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scalar_mul(
    device const float* A [[buffer(0)]],
    device float* B [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        B[id] = scalar * A[id];
    }
}
"#;

/// Dot product kernel (partial reduction)
///
/// Computes partial sums for dot product. Final sum requires CPU reduction.
///
/// Uses SIMD group reduction for efficiency on Apple Silicon.
pub const DOT_PRODUCT: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void dot_product(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* partial_sums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    // Compute local product
    float sum = 0.0f;
    if (id < count) {
        sum = A[id] * B[id];
    }

    // SIMD group reduction
    sum = simd_sum(sum);

    // Write SIMD group result to shared memory
    if (simd_lane == 0) {
        shared[simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group does final reduction
    if (tid < 32) {
        float val = (tid < 8) ? shared[tid] : 0.0f;
        val = simd_sum(val);
        if (tid == 0) {
            partial_sums[0] = val;
        }
    }
}
"#;

/// Naive matrix multiplication kernel (for reference/small matrices)
///
/// Computes: C = A @ B
/// Assumes row-major storage.
///
/// # Buffers
/// - buffer(0): A - M x K matrix
/// - buffer(1): B - K x N matrix
/// - buffer(2): C - M x N matrix (output)
/// - buffer(3): M, N, K dimensions (uint3)
pub const GEMM_NAIVE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gemm_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
"#;

/// Tiled matrix multiplication kernel (optimized)
///
/// Uses shared memory tiles for better cache utilization.
/// Tile size: 16x16
pub const GEMM_TILED: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 16

kernel void gemm_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    threadgroup float* shared_A [[threadgroup(0)]],
    threadgroup float* shared_B [[threadgroup(1)]])
{
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint row = gid.y;
    uint col = gid.x;
    uint local_row = tid.y;
    uint local_col = tid.x;

    float sum = 0.0f;

    // Loop over tiles
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        if (row < M && (t * TILE_SIZE + local_col) < K) {
            shared_A[local_row * TILE_SIZE + local_col] = A[row * K + t * TILE_SIZE + local_col];
        } else {
            shared_A[local_row * TILE_SIZE + local_col] = 0.0f;
        }

        // Load tile of B into shared memory
        if ((t * TILE_SIZE + local_row) < K && col < N) {
            shared_B[local_row * TILE_SIZE + local_col] = B[(t * TILE_SIZE + local_row) * N + col];
        } else {
            shared_B[local_row * TILE_SIZE + local_col] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += shared_A[local_row * TILE_SIZE + k] * shared_B[k * TILE_SIZE + local_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
"#;

/// Numerically stable softmax kernel
///
/// Computes: softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
///
/// Uses max subtraction for numerical stability.
pub const SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    // Phase 1: Find max (use first thread for simplicity)
    if (tid == 0) {
        float max_val = input[0];
        for (uint i = 1; i < count; i++) {
            max_val = max(max_val, input[i]);
        }
        shared[0] = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared[0];

    // Phase 2: Compute exp(x - max) and sum
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < count; i++) {
            float exp_val = exp(input[i] - max_val);
            output[i] = exp_val;
            sum += exp_val;
        }
        shared[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = shared[0];

    // Phase 3: Normalize
    if (tid == 0) {
        for (uint i = 0; i < count; i++) {
            output[i] /= sum;
        }
    }
}
"#;

/// Layer normalization kernel
///
/// Computes: y = (x - mean) / sqrt(var + eps) * gamma + beta
///
/// # Buffers
/// - buffer(0): input
/// - buffer(1): output
/// - buffer(2): gamma (scale)
/// - buffer(3): beta (bias)
/// - buffer(4): params (count, eps)
pub const LAYERNORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void layernorm(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    constant float2& params [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    uint count = uint(params.x);
    float eps = params.y;

    // Phase 1: Compute mean (single thread for simplicity)
    if (tid == 0) {
        float sum = 0.0f;
        for (uint i = 0; i < count; i++) {
            sum += input[i];
        }
        shared[0] = sum / float(count);  // mean
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = shared[0];

    // Phase 2: Compute variance
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (uint i = 0; i < count; i++) {
            float diff = input[i] - mean;
            sum_sq += diff * diff;
        }
        shared[0] = sum_sq / float(count);  // variance
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = shared[0];
    float inv_std = rsqrt(var + eps);

    // Phase 3: Normalize and apply affine transform
    if (tid == 0) {
        for (uint i = 0; i < count; i++) {
            float normalized = (input[i] - mean) * inv_std;
            output[i] = normalized * gamma[i] + beta[i];
        }
    }
}
"#;

/// ReLU activation kernel
///
/// Computes: y = max(0, x)
pub const RELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        output[id] = max(0.0f, input[id]);
    }
}
"#;

/// GELU activation kernel (Gaussian Error Linear Unit)
///
/// Computes: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub const GELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        float x = input[id];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);  // sqrt(2/pi) = 0.7978845608
        output[id] = 0.5f * x * (1.0f + tanh(inner));
    }
}
"#;

/// SiLU (Swish) activation kernel
///
/// Computes: y = x * sigmoid(x)
pub const SILU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        float x = input[id];
        float sigmoid_x = 1.0f / (1.0f + exp(-x));
        output[id] = x * sigmoid_x;
    }
}
"#;

/// Fused add + ReLU kernel
///
/// Computes: y = max(0, a + b)
pub const FUSED_ADD_RELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void fused_add_relu(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        C[id] = max(0.0f, A[id] + B[id]);
    }
}
"#;

/// Copy kernel (for benchmarking memory bandwidth)
pub const COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void copy(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id < count) {
        output[id] = input[id];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shaders_not_empty() {
        assert!(!ELEMENTWISE_ADD.is_empty());
        assert!(!ELEMENTWISE_MUL.is_empty());
        assert!(!SCALAR_MUL.is_empty());
        assert!(!DOT_PRODUCT.is_empty());
        assert!(!GEMM_NAIVE.is_empty());
        assert!(!GEMM_TILED.is_empty());
        assert!(!SOFTMAX.is_empty());
        assert!(!LAYERNORM.is_empty());
        assert!(!RELU.is_empty());
        assert!(!GELU.is_empty());
        assert!(!SILU.is_empty());
        assert!(!FUSED_ADD_RELU.is_empty());
        assert!(!COPY.is_empty());
    }

    #[test]
    fn test_shaders_contain_kernel_declaration() {
        assert!(ELEMENTWISE_ADD.contains("kernel void"));
        assert!(GEMM_TILED.contains("kernel void"));
        assert!(SOFTMAX.contains("kernel void"));
    }

    #[test]
    fn test_gemm_tiled_has_tile_size() {
        assert!(GEMM_TILED.contains("#define TILE_SIZE"));
    }

    #[test]
    fn test_activations_use_metal_math() {
        assert!(GELU.contains("tanh("));
        assert!(SILU.contains("exp("));
        assert!(RELU.contains("max("));
    }
}
