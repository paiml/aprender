//! Cooperative matrix GEMM shader — hardware tensor core acceleration
//!
//! Uses VK_KHR_cooperative_matrix via wgpu 29's WGSL extension.
//! GB10 Blackwell: M=16, K=16, N=16, F16 input, F32 accumulation.
//!
//! Contract: cooperative-matrix-gemm-v1
//! Lean theorem: ProvableContracts.CooperativeMatrix.matmul_block_sum

/// Cooperative matrix GEMM: C = α * A[M,K] @ B[K,N]
///
/// Dispatch: ceil(N/16) × ceil(M/16) workgroups, @workgroup_size(32)
/// Fallback: if cooperative matrix not available, use TILED_GEMM_SHADER.
pub const COOPERATIVE_GEMM_SHADER: &str = r#"
enable f16;
enable wgpu_cooperative_matrix;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    alpha: f32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(32)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // No early return — cooperative ops require uniform control flow
    let tile_row = wg_id.y * 16u;
    let tile_col = wg_id.x * 16u;

    var acc = coop_mat16x16<f32, C>();
    let num_k_tiles = (dims.K + 15u) / 16u;
    for (var kt = 0u; kt < num_k_tiles; kt++) {
        let k_offset = kt * 16u;
        let a_tile = coopLoad<coop_mat16x16<f16, A>>(
            &a[tile_row * dims.K + k_offset], dims.K
        );
        let b_tile = coopLoad<coop_mat16x16<f16, B>>(
            &b[k_offset * dims.N + tile_col], dims.N
        );
        acc = coopMultiplyAdd(a_tile, b_tile, acc);
    }
    // Only store if within bounds (uniform control flow maintained above)
    if (tile_row < dims.M && tile_col < dims.N) {
        coopStore(acc, &c[tile_row * dims.N + tile_col], dims.N);
    }
}
"#;
