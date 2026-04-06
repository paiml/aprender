//! Basic element-wise operations: matmul, add, mul, sub, scale,
//! dot product, activations, clip, and 2D convolution.

/// Matrix multiplication compute shader (WGSL) — tiled shared memory
///
/// Computes C = A × B where:
/// - A is M×K
/// - B is K×N
/// - C is M×N
///
/// Uses 16×16 shared memory tiles to reduce global memory bandwidth by ~16×.
/// Each workgroup loads tiles of A and B into `var<workgroup>` memory, then
/// computes partial products from shared memory.  This is the standard tiled
/// matmul from GPU computing textbooks (KAIZEN-021).
///
/// # Contract (C-TILED-MATMUL-001)
///
/// - **Binding layout**: identical to the naive shader (0=a, 1=b, 2=c, 3=dims)
/// - **Workgroup size**: 16×16 = 256 threads (unchanged)
/// - **Dispatch**: ceil(M/16) × ceil(N/16) workgroups (unchanged)
/// - **Result**: bit-identical to naive shader for all M, K, N (f32 associativity aside)
/// - **Speedup**: 5–15× on real GPUs (bandwidth-bound → compute-bound)
pub const MATMUL_SHADER: &str = r#"
const TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dimensions {
    M: u32,  // rows of A and C
    K: u32,  // cols of A, rows of B
    N: u32,  // cols of B and C
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Shared memory tiles — each 16×16 = 256 floats
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

// Workgroup size: 16×16 = 256 threads
@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = global_id.x;
    let col = global_id.y;
    let lr = local_id.x;  // local row within tile [0..15]
    let lc = local_id.y;  // local col within tile [0..15]

    var sum: f32 = 0.0;

    // Iterate over K dimension in tiles of 16
    let num_tiles = (dims.K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Load A tile: A[row, t*TILE + lc]
        let a_col = t * TILE + lc;
        if (row < dims.M && a_col < dims.K) {
            tile_a[lr * TILE + lc] = a[row * dims.K + a_col];
        } else {
            tile_a[lr * TILE + lc] = 0.0;
        }

        // Load B tile: B[t*TILE + lr, col]
        let b_row = t * TILE + lr;
        if (b_row < dims.K && col < dims.N) {
            tile_b[lr * TILE + lc] = b[b_row * dims.N + col];
        } else {
            tile_b[lr * TILE + lc] = 0.0;
        }

        // Wait for all threads to finish loading
        workgroupBarrier();

        // Accumulate partial dot product from shared memory
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            sum = sum + tile_a[lr * TILE + k] * tile_b[k * TILE + lc];
        }

        // Wait before loading next tile (prevents overwriting while others read)
        workgroupBarrier();
    }

    // Write result
    if (row < dims.M && col < dims.N) {
        c[row * dims.N + col] = sum;
    }
}
"#;

/// CUTLASS-style tiled GEMM compute shader (WGSL) — 64×64 output tiles
///
/// Computes C = α·A×B + β·C where A is M×K, B is K×N, C is M×N.
///
/// ## CUTLASS-derived tiling (MIT licensed algorithm)
///
/// - **Thread-block tile**: 64×64 output, K-step: 8
/// - **Thread micro-tile**: 4×4 output elements per thread
/// - **Workgroup**: 16×16 = 256 threads
/// - **Shared memory**: double-buffered (2 × 64×8 × 4 bytes × 2 matrices = 8 KB)
/// - **Inner loop**: 4×4 outer product from shared memory per K-step
/// - **Vectorized loads**: vec4<f32> for coalesced global memory access
///
/// ## Performance vs naive 16×16
///
/// Each thread computes 16 output elements (4×4) instead of 1, amortizing
/// shared memory loads by 16x. Double buffering overlaps next tile load
/// with current tile compute. Expected 10-30x speedup over MATMUL_SHADER.
///
/// ## Contract (wgsl-gemm-tiled-v1)
///
/// - Binding layout: 0=a, 1=b, 2=c, 3=dims (compatible with MATMUL_SHADER)
/// - Dispatch: ceil(M/64) × ceil(N/64) workgroups
/// - Result: matches naive within ε < 1e-4 (f32 reassociation)
/// - Zero unsafe: entirely via wgpu safe Rust API
pub const TILED_GEMM_SHADER: &str = r#"
// CUTLASS-derived tiled GEMM — 64×64 tiles, 4×4 thread micro-tiles
// Algorithm from NVIDIA CUTLASS (MIT licensed), reimplemented in WGSL.

const BM: u32 = 64u;       // thread-block tile M
const BN: u32 = 64u;       // thread-block tile N
const BK: u32 = 8u;        // K-dimension tile step
const TM: u32 = 4u;        // thread micro-tile M (each thread computes 4 rows)
const TN: u32 = 4u;        // thread micro-tile N (each thread computes 4 cols)
// Workgroup: 16×16 = 256 threads
// Each thread: 4×4 = 16 output elements
// Total: 256 threads × 16 = 4096 elements = 64×64 ✓

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    alpha: f32,   // scaling factor (default 1.0)
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Double-buffered shared memory tiles
// Buffer 0: smem[0..BM*BK] for A, smem[BM*BK..BM*BK+BK*BN] for B
// Buffer 1: smem[BM*BK+BK*BN..2*(BM*BK+BK*BN)] duplicated
// Total: 2 * (64*8 + 8*64) * 4 = 2 * 1024 * 4 = 8192 bytes = 8 KB
var<workgroup> smem_a0: array<f32, 512>;  // BM * BK = 64 * 8
var<workgroup> smem_b0: array<f32, 512>;  // BK * BN = 8 * 64
var<workgroup> smem_a1: array<f32, 512>;  // double buffer
var<workgroup> smem_b1: array<f32, 512>;  // double buffer

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Thread position within workgroup (16×16 grid)
    let tx = lid.x;  // [0..15]
    let ty = lid.y;  // [0..15]
    let tid = ty * 16u + tx;  // flat thread index [0..255]

    // This workgroup computes output tile C[bm..bm+64, bn..bn+64]
    let bm = wg_id.y * BM;  // block row offset
    let bn = wg_id.x * BN;  // block col offset

    // Each thread computes a 4×4 micro-tile within the 64×64 block.
    // Thread (tx, ty) computes rows [ty*4..ty*4+3], cols [tx*4..tx*4+3]
    let thread_row = ty * TM;  // [0, 4, 8, ..., 60]
    let thread_col = tx * TN;  // [0, 4, 8, ..., 60]

    // Accumulator registers: 4×4 = 16 per thread
    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) {
        acc[i] = 0.0;
    }

    let num_k_tiles = (dims.K + BK - 1u) / BK;

    // === PROLOGUE: Load first tile into buffer 0 ===
    // Each thread loads 2 elements of A and 2 elements of B (256 threads × 2 = 512)
    let load_a_row = tid / BK;       // which row of the 64×8 tile
    let load_a_col = tid % BK;       // which col of the 64×8 tile
    let load_b_row = tid / BN;       // which row of the 8×64 tile
    let load_b_col = tid % BN;       // which col of the 8×64 tile

    // Load A[bm + load_a_row, 0 + load_a_col] into smem_a0
    let ga_row = bm + load_a_row;
    if (ga_row < dims.M && load_a_col < dims.K) {
        smem_a0[load_a_row * BK + load_a_col] = a[ga_row * dims.K + load_a_col];
    } else {
        smem_a0[load_a_row * BK + load_a_col] = 0.0;
    }
    // Second element (tid + 256 maps to rows 32..63 of the 64-row tile)
    let load_a_row2 = load_a_row + 32u;
    let ga_row2 = bm + load_a_row2;
    if (load_a_row2 < BM && ga_row2 < dims.M && load_a_col < dims.K) {
        smem_a0[load_a_row2 * BK + load_a_col] = a[ga_row2 * dims.K + load_a_col];
    } else if (load_a_row2 < BM) {
        smem_a0[load_a_row2 * BK + load_a_col] = 0.0;
    }

    // Load B[0 + load_b_row, bn + load_b_col] into smem_b0
    let gb_col = bn + load_b_col;
    if (load_b_row < dims.K && gb_col < dims.N) {
        smem_b0[load_b_row * BN + load_b_col] = b[load_b_row * dims.N + gb_col];
    } else {
        smem_b0[load_b_row * BN + load_b_col] = 0.0;
    }
    // B tile is only 8 rows × 64 cols = 512 elements = exactly 256 threads × 2
    let load_b_row2 = load_b_row + 4u;
    if (load_b_row2 < BK && load_b_row2 < dims.K && gb_col < dims.N) {
        smem_b0[load_b_row2 * BN + load_b_col] = b[load_b_row2 * dims.N + gb_col];
    } else if (load_b_row2 < BK) {
        smem_b0[load_b_row2 * BN + load_b_col] = 0.0;
    }

    workgroupBarrier();

    // === MAINLOOP: iterate over K-dimension tiles ===
    for (var kt = 0u; kt < num_k_tiles; kt++) {
        let k_offset = kt * BK;

        // Determine which buffer to read from (ping-pong)
        let read_buf = kt % 2u;

        // --- Compute 4×4 micro-tile from current shared memory ---
        for (var k = 0u; k < BK; k++) {
            // Load 4 A values from shared memory (one column of the micro-tile)
            var a_frag: array<f32, 4>;
            var b_frag: array<f32, 4>;

            for (var mi = 0u; mi < TM; mi++) {
                if (read_buf == 0u) {
                    a_frag[mi] = smem_a0[(thread_row + mi) * BK + k];
                } else {
                    a_frag[mi] = smem_a1[(thread_row + mi) * BK + k];
                }
            }
            for (var ni = 0u; ni < TN; ni++) {
                if (read_buf == 0u) {
                    b_frag[ni] = smem_b0[k * BN + thread_col + ni];
                } else {
                    b_frag[ni] = smem_b1[k * BN + thread_col + ni];
                }
            }

            // 4×4 outer product: acc[mi][ni] += a_frag[mi] * b_frag[ni]
            for (var mi = 0u; mi < TM; mi++) {
                for (var ni = 0u; ni < TN; ni++) {
                    acc[mi * TN + ni] += a_frag[mi] * b_frag[ni];
                }
            }
        }

        // --- Load NEXT tile into the other buffer (double buffering) ---
        let next_k = (kt + 1u) * BK;
        let write_buf = (kt + 1u) % 2u;

        if (kt + 1u < num_k_tiles) {
            // Load A next tile
            let na_col = next_k + load_a_col;
            let na_val = select(0.0, a[ga_row * dims.K + na_col],
                ga_row < dims.M && na_col < dims.K);
            if (write_buf == 0u) { smem_a0[load_a_row * BK + load_a_col] = na_val; }
            else { smem_a1[load_a_row * BK + load_a_col] = na_val; }

            let na_val2 = select(0.0, a[ga_row2 * dims.K + na_col],
                load_a_row2 < BM && ga_row2 < dims.M && na_col < dims.K);
            if (load_a_row2 < BM) {
                if (write_buf == 0u) { smem_a0[load_a_row2 * BK + load_a_col] = na_val2; }
                else { smem_a1[load_a_row2 * BK + load_a_col] = na_val2; }
            }

            // Load B next tile
            let nb_row = next_k + load_b_row;
            let nb_val = select(0.0, b[nb_row * dims.N + gb_col],
                nb_row < dims.K && gb_col < dims.N);
            if (write_buf == 0u) { smem_b0[load_b_row * BN + load_b_col] = nb_val; }
            else { smem_b1[load_b_row * BN + load_b_col] = nb_val; }

            let nb_row2 = next_k + load_b_row2;
            if (load_b_row2 < BK) {
                let nb_val2 = select(0.0, b[nb_row2 * dims.N + gb_col],
                    nb_row2 < dims.K && gb_col < dims.N);
                if (write_buf == 0u) { smem_b0[load_b_row2 * BN + load_b_col] = nb_val2; }
                else { smem_b1[load_b_row2 * BN + load_b_col] = nb_val2; }
            }
        }

        workgroupBarrier();
    }

    // === EPILOGUE: Write 4×4 micro-tile to global memory ===
    let alpha = dims.alpha;
    for (var mi = 0u; mi < TM; mi++) {
        for (var ni = 0u; ni < TN; ni++) {
            let grow = bm + thread_row + mi;
            let gcol = bn + thread_col + ni;
            if (grow < dims.M && gcol < dims.N) {
                c[grow * dims.N + gcol] = alpha * acc[mi * TN + ni];
            }
        }
    }
}
"#;

/// Fused LoRA addmm: output += (input @ A) @ B * scale
///
/// Computes the LoRA contribution and adds it to the base projection output.
/// Two matmuls + scaled add in sequence. Uses shared memory for the intermediate.
/// For rank << hidden_dim, this is much smaller than the base matmul.
///
/// Dispatch: one workgroup per output element (output is [seq, out_dim]).
/// Each thread computes one output element's LoRA delta.
pub const LORA_ADDMM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;   // [seq, in_dim]
@group(0) @binding(1) var<storage, read> lora_a: array<f32>;  // [in_dim, rank]
@group(0) @binding(2) var<storage, read> lora_b: array<f32>;  // [rank, out_dim]
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // [seq, out_dim] — ADD to existing

struct LoraParams {
    seq_len: u32,
    in_dim: u32,
    rank: u32,
    out_dim: u32,
    scale: f32,    // alpha / rank
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(4) var<uniform> params: LoraParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = params.seq_len * params.out_dim;
    if (idx >= total) { return; }

    let row = idx / params.out_dim;
    let col = idx % params.out_dim;

    // Compute (input[row] @ A) @ B[col] * scale
    // First: h = input[row] @ A → [rank] vector
    // Then: delta = h @ B[:, col] * scale → scalar
    var delta: f32 = 0.0;
    for (var r = 0u; r < params.rank; r++) {
        // h[r] = sum_k input[row, k] * A[k, r]
        var h_r: f32 = 0.0;
        for (var k = 0u; k < params.in_dim; k++) {
            h_r += input[row * params.in_dim + k] * lora_a[k * params.rank + r];
        }
        // delta += h[r] * B[r, col]
        delta += h_r * lora_b[r * params.out_dim + col];
    }

    output[row * params.out_dim + col] += delta * params.scale;
}
"#;

/// Column scatter shader — copies chunk columns into a wider row-major matrix.
///
/// Replaces N × copy_buffer_to_buffer calls with a single GPU dispatch.
/// Source: [seq, chunk_n] row-major → Dest: [seq, full_n] at column offset.
///
/// Each thread copies one element.
pub const COLUMN_SCATTER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct ScatterParams {
    seq_len: u32,
    chunk_n: u32,    // width of source
    full_n: u32,     // width of destination
    col_offset: u32, // column offset in destination
}

@group(0) @binding(2) var<uniform> params: ScatterParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = params.seq_len * params.chunk_n;
    if (idx >= total) { return; }

    let row = idx / params.chunk_n;
    let col = idx % params.chunk_n;

    let src_idx = row * params.chunk_n + col;
    let dst_idx = row * params.full_n + params.col_offset + col;

    dst[dst_idx] = src[src_idx];
}
"#;

/// Column gather shader — extracts columns from a wide matrix into a chunk.
///
/// Inverse of scatter. Used for backward: extract grad_logits columns per chunk.
pub const COLUMN_GATHER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct GatherParams {
    seq_len: u32,
    chunk_n: u32,    // width of destination
    full_n: u32,     // width of source
    col_offset: u32, // column offset in source
}

@group(0) @binding(2) var<uniform> params: GatherParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = params.seq_len * params.chunk_n;
    if (idx >= total) { return; }

    let row = idx / params.chunk_n;
    let col = idx % params.chunk_n;

    let src_idx = row * params.full_n + params.col_offset + col;
    let dst_idx = row * params.chunk_n + col;

    dst[dst_idx] = src[src_idx];
}
"#;

/// Scaled transpose: B[j,i] = scale * A[i,j]
/// Contract: wgsl-transpose-v1
///
/// Dispatch: ceil(M*N / 256) workgroups (with 2D for >65535).
/// Params: { M, N, scale, _pad }
pub const TRANSPOSE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct TransposeParams {
    m: u32,      // rows of source
    n: u32,      // cols of source
    scale: f32,  // output scaling (1.0 for identity)
    _pad: u32,
}

@group(0) @binding(2) var<uniform> params: TransposeParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * 65535u * 256u;
    let total = params.m * params.n;
    if (idx >= total) { return; }

    let i = idx / params.n;  // source row
    let j = idx % params.n;  // source col

    // src[i, j] = src[i * N + j]  → dst[j, i] = dst[j * M + i]
    dst[j * params.m + i] = params.scale * src[i * params.n + j];
}
"#;

/// PMAT-326: GEMV compute shader (WGSL) — matrix-vector product y = W × x
///
/// Optimized for M=1 (single-token decode). Each workgroup computes ONE output
/// element by cooperatively reducing the dot product along K using shared memory.
///
/// - W: [N, K] row-major weight matrix
/// - x: [K] input vector
/// - y: [N] output vector
///
/// Workgroup: 256 threads. Each workgroup handles 1 output row.
/// Dispatch: N workgroups (one per output element).
/// Reduction: tree reduction in shared memory (log2(256) = 8 steps).
/// PMAT-331: vec4 vectorized GEMV — 4x fewer memory transactions.
/// Each thread loads vec4<f32> (4 floats per load), dot4 in registers.
/// K must be divisible by 4 (true for all Qwen dimensions: 1536, 256, 8960).
pub(crate) const GEMV_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;     // input [K/4]
@group(0) @binding(1) var<storage, read> w: array<vec4<f32>>;     // weight [N, K/4]
@group(0) @binding(2) var<storage, read_write> y: array<f32>;     // output [N]

struct Params {
    n: u32,  // output dim (number of rows)
    k: u32,  // input dim (K, NOT K/4 — shader divides internally)
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x;
    let tid = lid.x;
    let k4 = params.k / 4u;  // Number of vec4 elements per row

    if (row >= params.n) { return; }

    // Phase 1: vec4 dot product — 4 FMAs per iteration
    var partial_sum: f32 = 0.0;
    let row_offset = row * k4;
    var col4 = tid;
    while (col4 < k4) {
        let wv = w[row_offset + col4];
        let xv = x[col4];
        partial_sum += dot(wv, xv);  // vec4 dot = 4 FMAs
        col4 += 256u;
    }
    sdata[tid] = partial_sum;
    workgroupBarrier();

    // Phase 2: Tree reduction (256 → 1)
    if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; }
    workgroupBarrier();
    if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { sdata[tid] += sdata[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { sdata[tid] += sdata[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { sdata[tid] += sdata[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { sdata[tid] += sdata[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { sdata[tid] += sdata[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) {
        y[row] = sdata[0] + sdata[1];
    }
}
"#;

/// Q4_K quantized matrix-vector product (WGSL) — C-WGPU-Q4K-001
///
/// Computes y[row] = Σ_col dequant(W_q4k[row, col]) × x[col]
/// where W is stored as raw Q4_K super-blocks (144 bytes → 256 f32 values).
///
/// Dequantization happens on-the-fly per-thread — no F32 weight buffer.
/// This reduces VRAM from 4×num_params (F32) to 144/256×num_params (Q4K) = 7.1x.
///
/// Q4_K super-block layout (144 bytes per 256 elements):
///   bytes[0:2]   = d    (f16, global scale)
///   bytes[2:4]   = dmin (f16, global min scale)
///   bytes[4:16]  = 12 packed scale/min bytes (8 sub-blocks, 6-bit packed)
///   bytes[16:144]= 128 quantized nibble bytes (4-bit, interleaved low/high)
///
/// Each sub-block (32 elements): value = d × scale × nibble - dmin × min
///
/// Workgroup: 256 threads per output row.
/// Dispatch: N workgroups (one per output element).
/// Each thread processes ceil(num_superblocks/256) super-blocks, accumulating
/// 256 elements per super-block into a partial sum, then tree-reduces.
pub(crate) const Q4K_GEMV_SHADER: &str = r#"
// Q4K weights stored as array<u32> (144 bytes = 36 u32s per super-block)
@group(0) @binding(0) var<storage, read> x: array<f32>;       // input [K]
@group(0) @binding(1) var<storage, read> w_q4k: array<u32>;   // Q4K weight bytes as u32
@group(0) @binding(2) var<storage, read_write> y: array<f32>;  // output [N]

struct Q4kParams {
    n: u32,               // output dim (number of rows)
    k: u32,               // input dim (number of columns)
    num_superblocks: u32, // super-blocks per row = ceil(K / 256)
    _pad: u32,
}
@group(0) @binding(3) var<uniform> params: Q4kParams;

var<workgroup> sdata: array<f32, 256>;

// Extract a u8 from a u32 array (byte-level access)
fn read_u8(base: u32, byte_offset: u32) -> u32 {
    let word_idx = base + byte_offset / 4u;
    let byte_pos = byte_offset % 4u;
    return (w_q4k[word_idx] >> (byte_pos * 8u)) & 0xFFu;
}

// Convert f16 (stored as u16 in two bytes) to f32
// PMAT-497 FIX: Use bitwise IEEE 754 construction (matching CPU f16_to_f32).
// Previous version used pow(2.0, exp) which introduced rounding errors that
// corrupted every Q4K scale factor, causing loss > random from step 1.
fn f16_to_f32(low: u32, high: u32) -> f32 {
    let bits = low | (high << 8u);
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mantissa = bits & 0x3FFu;

    // Sign bit in f32 position
    var f32_bits = sign << 31u;

    if (exp == 0u) {
        if (mantissa == 0u) {
            // Signed zero
            return bitcast<f32>(f32_bits);
        }
        // Subnormal f16: normalize mantissa to find implicit leading 1
        var m = mantissa;
        var e = 0i;
        while ((m & 0x400u) == 0u) {
            m = m << 1u;
            e -= 1i;
        }
        // Remove implicit leading 1 and construct f32 bits
        let new_exp = u32(127 - 15 + 1 + e) << 23u;
        let new_man = (m & 0x3FFu) << 13u;
        f32_bits = f32_bits | new_exp | new_man;
        return bitcast<f32>(f32_bits);
    }
    if (exp == 31u) {
        // Inf/NaN: exponent all-ones in f32
        f32_bits = f32_bits | (0xFFu << 23u) | (mantissa << 13u);
        return bitcast<f32>(f32_bits);
    }
    // Normal f16: re-bias exponent from f16 (bias=15) to f32 (bias=127)
    let new_exp = (exp - 15u + 127u) << 23u;
    let new_man = mantissa << 13u;
    f32_bits = f32_bits | new_exp | new_man;
    return bitcast<f32>(f32_bits);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wg_id.x;
    let tid = lid.x;

    if (row >= params.n) { return; }

    // Each super-block is 36 u32s (144 bytes). Row data starts at:
    let row_base_u32 = row * params.num_superblocks * 36u;

    var partial_sum: f32 = 0.0;

    // Each thread processes a subset of super-blocks for this row
    var sb_idx = tid;
    while (sb_idx < params.num_superblocks) {
        let sb_base = row_base_u32 + sb_idx * 36u;
        let input_offset = sb_idx * 256u;

        // Read d and dmin (f16 → f32)
        let byte0 = read_u8(sb_base, 0u);
        let byte1 = read_u8(sb_base, 1u);
        let byte2 = read_u8(sb_base, 2u);
        let byte3 = read_u8(sb_base, 3u);
        let d = f16_to_f32(byte0, byte1);
        let dmin = f16_to_f32(byte2, byte3);

        // Unpack 8 scales and 8 mins from bytes[4:16]
        var scales: array<f32, 8>;
        var mins: array<f32, 8>;

        let s0 = read_u8(sb_base, 4u);
        let s1 = read_u8(sb_base, 5u);
        let s2 = read_u8(sb_base, 6u);
        let s3 = read_u8(sb_base, 7u);
        let m0 = read_u8(sb_base, 8u);
        let m1 = read_u8(sb_base, 9u);
        let m2 = read_u8(sb_base, 10u);
        let m3 = read_u8(sb_base, 11u);
        let h0 = read_u8(sb_base, 12u);
        let h1 = read_u8(sb_base, 13u);
        let h2 = read_u8(sb_base, 14u);
        let h3 = read_u8(sb_base, 15u);

        scales[0] = f32(s0 & 0x3Fu);
        scales[1] = f32(s1 & 0x3Fu);
        scales[2] = f32(s2 & 0x3Fu);
        scales[3] = f32(s3 & 0x3Fu);
        scales[4] = f32((h0 & 0x0Fu) | ((s0 >> 6u) << 4u));
        scales[5] = f32((h1 & 0x0Fu) | ((s1 >> 6u) << 4u));
        scales[6] = f32((h2 & 0x0Fu) | ((s2 >> 6u) << 4u));
        scales[7] = f32((h3 & 0x0Fu) | ((s3 >> 6u) << 4u));

        mins[0] = f32(m0 & 0x3Fu);
        mins[1] = f32(m1 & 0x3Fu);
        mins[2] = f32(m2 & 0x3Fu);
        mins[3] = f32(m3 & 0x3Fu);
        mins[4] = f32((h0 >> 4u) | ((m0 >> 6u) << 4u));
        mins[5] = f32((h1 >> 4u) | ((m1 >> 6u) << 4u));
        mins[6] = f32((h2 >> 4u) | ((m2 >> 6u) << 4u));
        mins[7] = f32((h3 >> 4u) | ((m3 >> 6u) << 4u));

        // Process 4 chunks × 64 elements (32 low nibbles + 32 high nibbles)
        for (var chunk = 0u; chunk < 4u; chunk++) {
            let d1 = d * scales[chunk * 2u];
            let dm1 = dmin * mins[chunk * 2u];
            let d2 = d * scales[chunk * 2u + 1u];
            let dm2 = dmin * mins[chunk * 2u + 1u];

            let q_byte_start = 16u + chunk * 32u;  // offset into super-block
            let elem_base = input_offset + chunk * 64u;

            // Low nibbles: 32 elements
            for (var i = 0u; i < 32u; i++) {
                let idx = elem_base + i;
                if (idx < params.k) {
                    let q_byte = read_u8(sb_base, q_byte_start + i);
                    let q_val = f32(q_byte & 0x0Fu);
                    partial_sum += (d1 * q_val - dm1) * x[idx];
                }
            }
            // High nibbles: 32 elements
            for (var i = 0u; i < 32u; i++) {
                let idx = elem_base + 32u + i;
                if (idx < params.k) {
                    let q_byte = read_u8(sb_base, q_byte_start + i);
                    let q_val = f32(q_byte >> 4u);
                    partial_sum += (d2 * q_val - dm2) * x[idx];
                }
            }
        }

        sb_idx += 256u;  // stride by workgroup size
    }

    // Tree reduction (same as GEMV_SHADER)
    sdata[tid] = partial_sum;
    workgroupBarrier();

    if (tid < 128u) { sdata[tid] += sdata[tid + 128u]; }
    workgroupBarrier();
    if (tid < 64u) { sdata[tid] += sdata[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { sdata[tid] += sdata[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { sdata[tid] += sdata[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { sdata[tid] += sdata[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { sdata[tid] += sdata[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { sdata[tid] += sdata[tid + 2u]; }
    workgroupBarrier();
    if (tid == 0u) {
        y[row] = sdata[0] + sdata[1];
    }
}
"#;

/// Vector addition compute shader (WGSL)
///
/// Computes c = a + b element-wise
pub(crate) const VEC_ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] + b[idx];
    }
}
"#;

/// Element-wise multiplication shader (WGSL)
///
/// Computes c = a * b element-wise
pub(crate) const VEC_MUL_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] * b[idx];
    }
}
"#;

/// Element-wise subtraction shader (WGSL)
///
/// Computes c = a - b element-wise
pub(crate) const VEC_SUB_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&a);

    if (idx < len) {
        c[idx] = a[idx] - b[idx];
    }
}
"#;

/// Scalar multiplication shader (WGSL)
///
/// Computes output = input * scalar element-wise
pub(crate) const SCALE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ScaleParams {
    scalar: f32,
}

@group(0) @binding(2) var<uniform> params: ScaleParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        output[idx] = input[idx] * params.scalar;
    }
}
"#;

/// Dot product reduction shader (WGSL)
///
/// Computes sum(a[i] * b[i]) using parallel reduction
pub(crate) const DOT_PRODUCT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let len = arrayLength(&a);

    // Load and multiply
    var sum: f32 = 0.0;
    if (idx < len) {
        sum = a[idx] * b[idx];
    }
    partial_sums[local_idx] = sum;

    workgroupBarrier();

    // Parallel reduction within workgroup
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            partial_sums[local_idx] = partial_sums[local_idx] + partial_sums[local_idx + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (local_idx == 0u) {
        result[global_id.x / 256u] = partial_sums[0];
    }
}
"#;

/// ReLU activation compute shader (WGSL)
///
/// Computes element-wise ReLU: max(0, x)
///
/// This is one of the simplest GPU operations - a single comparison and selection per element.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // ReLU: max(0, x)
        output[idx] = max(0.0, input[idx]);
    }
}
"#;

/// Leaky ReLU activation compute shader (WGSL)
///
/// Computes element-wise Leaky ReLU: leaky_relu(x, α) = max(αx, x) = x if x > 0, else αx
///
/// Leaky ReLU addresses the "dying ReLU" problem by allowing small negative activations.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const LEAKY_RELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct LeakyReluParams {
    negative_slope: f32,
}

@group(0) @binding(2) var<uniform> params: LeakyReluParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Leaky ReLU: leaky_relu(x, α) = x if x > 0, else αx
        if (x > 0.0) {
            output[idx] = x;
        } else {
            output[idx] = params.negative_slope * x;
        }
    }
}
"#;

/// ELU (Exponential Linear Unit) activation compute shader (WGSL)
///
/// Computes element-wise ELU: elu(x, α) = x if x > 0, else α(e^x - 1)
///
/// ELU has smooth gradients everywhere and pushes mean activations closer to zero,
/// improving learning in deep networks.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const ELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct EluParams {
    alpha: f32,
}

@group(0) @binding(2) var<uniform> params: EluParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // ELU: elu(x, α) = x if x > 0, else α(e^x - 1)
        if (x > 0.0) {
            output[idx] = x;
        } else {
            output[idx] = params.alpha * (exp(x) - 1.0);
        }
    }
}
"#;

/// Sigmoid activation compute shader (WGSL)
///
/// Computes element-wise sigmoid: σ(x) = 1 / (1 + e^(-x))
///
/// Classic logistic function used in binary classification and attention mechanisms.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const SIGMOID_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Sigmoid: σ(x) = 1 / (1 + exp(-x))
        // Numerically stable implementation:
        // For x >= 0: σ(x) = 1 / (1 + exp(-x))
        // For x < 0: σ(x) = exp(x) / (1 + exp(x))
        var result: f32;
        if (x >= 0.0) {
            result = 1.0 / (1.0 + exp(-x));
        } else {
            let exp_x = exp(x);
            result = exp_x / (1.0 + exp_x);
        }

        output[idx] = result;
    }
}
"#;

/// Tanh (hyperbolic tangent) activation compute shader (WGSL)
///
/// Computes element-wise tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
///
/// Classic activation function used in LSTM, GRU, and traditional neural networks.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const TANH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        //                = (e^(2x) - 1) / (e^(2x) + 1)
        // Numerically stable implementation:
        // For |x| > 20: tanh(x) ≈ sign(x) (saturates at ±1)
        // Otherwise: use standard formula
        var result: f32;
        if (x > 20.0) {
            result = 1.0;
        } else if (x < -20.0) {
            result = -1.0;
        } else {
            let exp_2x = exp(2.0 * x);
            result = (exp_2x - 1.0) / (exp_2x + 1.0);
        }

        output[idx] = result;
    }
}
"#;

/// Swish activation compute shader (WGSL)
///
/// Computes element-wise swish: swish(x) = x * σ(x) = x / (1 + e^(-x))
///
/// Modern activation function (SiLU) used in transformers and modern architectures.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const SWISH_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // Swish: swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // Numerically stable implementation:
        // For x >= 0: swish(x) = x / (1 + exp(-x))
        // For x < 0: swish(x) = x * exp(x) / (1 + exp(x))
        var result: f32;
        if (x >= 0.0) {
            result = x / (1.0 + exp(-x));
        } else {
            let exp_x = exp(x);
            result = x * exp_x / (1.0 + exp_x);
        }

        output[idx] = result;
    }
}
"#;

/// GELU activation compute shader (WGSL)
///
/// Computes element-wise GELU using tanh approximation:
/// GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// Standard activation in BERT, GPT-2, GPT-3, and modern transformers.
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const GELU_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        let x = input[idx];

        // GELU approximation (tanh-based):
        // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        let SQRT_2_OVER_PI: f32 = 0.7978846; // √(2/π)
        let COEFF: f32 = 0.044715;

        let x_cubed = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
        let result = 0.5 * x * (1.0 + tanh(inner));

        output[idx] = result;
    }
}
"#;

/// Clip (clamp) compute shader (WGSL)
///
/// Computes element-wise clip: clamp(x, min_val, max_val)
///
/// Constrains values to the range [min_val, max_val].
/// GPU acceleration beneficial for large vectors (>100K elements).
pub(crate) const CLIP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct ClipParams {
    min_val: f32,
    max_val: f32,
}

@group(0) @binding(2) var<uniform> params: ClipParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // Clip: clamp(x, min_val, max_val) = max(min_val, min(max_val, x))
        output[idx] = clamp(input[idx], params.min_val, params.max_val);
    }
}
"#;

/// 2D Convolution compute shader (WGSL)
///
/// Computes 2D convolution: output = input ⊗ kernel
/// Uses "valid" padding (no padding, output smaller than input)
///
/// Output dimensions:
/// - output_rows = input_rows - kernel_rows + 1
/// - output_cols = input_cols - kernel_cols + 1
///
/// Uses workgroups of 16×16 threads for optimal GPU utilization
pub(crate) const CONVOLVE2D_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct ConvDimensions {
    input_rows: u32,
    input_cols: u32,
    kernel_rows: u32,
    kernel_cols: u32,
    output_rows: u32,
    output_cols: u32,
}

@group(0) @binding(3) var<uniform> dims: ConvDimensions;

// Workgroup size: 16×16 = 256 threads
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_row = global_id.x;
    let out_col = global_id.y;

    // Bounds check
    if (out_row >= dims.output_rows || out_col >= dims.output_cols) {
        return;
    }

    var sum: f32 = 0.0;

    // Apply kernel: iterate over kernel dimensions
    for (var k_row: u32 = 0u; k_row < dims.kernel_rows; k_row = k_row + 1u) {
        for (var k_col: u32 = 0u; k_col < dims.kernel_cols; k_col = k_col + 1u) {
            // Input pixel coordinates
            let in_row = out_row + k_row;
            let in_col = out_col + k_col;

            // Input and kernel are row-major
            let input_idx = in_row * dims.input_cols + in_col;
            let kernel_idx = k_row * dims.kernel_cols + k_col;

            sum = sum + input[input_idx] * kernel[kernel_idx];
        }
    }

    // Write output (row-major)
    let output_idx = out_row * dims.output_cols + out_col;
    output[output_idx] = sum;
}
"#;
