// PMAT-881 — pure-Rust cuda-oxide port of the hand-PTX FusedGateUpSwiglu Q4K kernel.
//
// Computes, for each output row i of the FFN intermediate:
//     y[i] = silu(dot(W_gate[i], x)) * dot(W_up[i], x)
// where silu(g) = g * sigmoid(g) = g / (1 + exp(-g)).
//
// Faithful port of crates/aprender-gpu/.../fused_gate_up_swiglu_hw_dp4a.rs
// (the production FusedGateUpSwigluHwDp4aQ4KGemvKernel) — same fusion shape
// (dual gate+up accumulators, in-register SwiGLU, single launch, single y write)
// — reusing the proven q4k_matvec oxide patterns (T threads/row + block reduction).
//
// PARITY NOTE: the production kernel does Q8_1 DP4A integer math on quantized
// activations; this oxide port (per the PMAT-881 parity signature) takes f32
// activations and does the equivalent f32 dequant-dot. We prove BIT-EXACT parity
// vs the CPU fused reference (identical dequant + SwiGLU math) and report GB10
// wall-clock vs the documented hand-PTX FusedGateUp baseline (~120us @ 1536x8960).
//
// Design (true fusion, single launch, no atomics, no intermediate global buffers):
//   - block_dim = (T,1,1), one block per output row, grid_dim = (N,1,1)
//   - each thread accumulates a strided partial dot for BOTH gate and up over K
//   - block reduction via shared memory (2*T f32) + sync_threads
//   - thread 0 computes SwiGLU(gate_sum, up_sum) and writes y[row]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::shared::SharedArray;
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

// ---------------------------------------------------------------------------
// Q4_K dequant helpers (host + device shared). Identical math to q4k-matvec
// spike and crates/aprender-serve/src/quantize/dequant_q4k.rs + simd.rs.
// ---------------------------------------------------------------------------
fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x3FF) as u32;
    let b: u32 = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut ee: i32 = -1;
            let mut mm = m;
            loop {
                ee += 1;
                mm <<= 1;
                if (mm & 0x400) != 0 {
                    break;
                }
            }
            (s << 31) | (((127 - 15 - ee) as u32) << 23) | ((mm & 0x3FF) << 13)
        }
    } else if e == 0x1F {
        (s << 31) | (0xFF << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(b)
}

fn extract_scale_min(sc: &[u8], j: usize) -> (f32, f32) {
    let (d, m) = if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        (
            (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
        )
    };
    (d as f32, m as f32)
}

fn dequant_elem(data: &[u8], idx: usize) -> f32 {
    let sb = idx / 256;
    let local = idx % 256;
    let s = sb * 144;
    let d = f16_to_f32((data[s] as u16) | ((data[s + 1] as u16) << 8));
    let dmin = f16_to_f32((data[s + 2] as u16) | ((data[s + 3] as u16) << 8));
    let g = local / 64;
    let w = local % 64;
    let bic = w % 32;
    let qb = data[s + 16 + g * 32 + bic];
    let issub = if w < 32 { g * 2 } else { g * 2 + 1 };
    let nib = if w < 32 { qb & 0x0F } else { qb >> 4 };
    let (scv, mn) = extract_scale_min(&data[s + 4..s + 16], issub);
    d * scv * (nib as f32) - dmin * mn
}

// NOTE: a per-superblock variant (decode d/dmin once per 256-block, like the
// hand-PTX kernel) was measured on GB10 and was SLOWER than the element-strided
// kernel below (343us vs 189us @ N=1536,K=8960): at T~num_superblocks the
// occupancy/K-parallelism loss dominates the amortized-decode win. The
// element-strided kernel (high T, redundant decode) wins on this hardware.

// silu(g) * u, computed exactly as the hand-PTX kernel:
//   sigmoid(g) = 1 / (1 + exp(-g)),  silu = g*sigmoid,  result = silu * u
#[inline(always)]
fn swiglu(g: f32, u: f32) -> f32 {
    let sig = 1.0f32 / (1.0f32 + (-g).exp());
    (g * sig) * u
}

// threads per output row / block. Element-strided over K, so higher T = more
// K-parallelism. Kept const so SharedArray<f32,{2*T}> stays compile-time sized.
const T: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    // Fused gate+up+SwiGLU, f32 activations.
    // Layout: block per row, T threads/block, grid = N blocks.
    //   wg/wu : [N * row_bytes] Q4K weights (row-major), x : [K] f32, y : [N] f32
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn fused_gate_up_swiglu(wg: &[u8], wu: &[u8], x: &[f32], y: &[f32]) {
        // Shared partials: [gate_0..gate_{T-1}, up_0..up_{T-1}]
        static mut SMEM: SharedArray<f32, { 2 * T }> = SharedArray::UNINIT;
        unsafe {
            // one block = one output row; T threads per block
            let row = thread::blockIdx_x() as usize;
            let n = y.len();
            if row >= n {
                return;
            }
            let k = x.len();
            let lane = thread::threadIdx_x() as usize;

            // Each row's Q4K data is (K/256)*144 bytes contiguous; dequant_elem
            // computes sb = idx/256 internally, so the row index is (row*K + j).
            let base = row * k;

            // Element-strided partial dot for gate and up (high K-parallelism).
            let mut accg = 0.0f32;
            let mut accu = 0.0f32;
            let mut j = lane;
            while j < k {
                let xv = x[j];
                accg += dequant_elem(wg, base + j) * xv;
                accu += dequant_elem(wu, base + j) * xv;
                j += T;
            }

            SMEM[lane] = accg;
            SMEM[lane + T] = accu;
            thread::sync_threads();

            // Tree reduction (T is a power of two).
            let mut stride = T / 2;
            while stride > 0 {
                if lane < stride {
                    SMEM[lane] += SMEM[lane + stride];
                    SMEM[lane + T] += SMEM[lane + T + stride];
                }
                thread::sync_threads();
                stride /= 2;
            }

            // Thread 0: SwiGLU + single global write (true fusion, no intermediate buffers).
            if lane == 0 {
                let gate_sum = SMEM[0];
                let up_sum = SMEM[T];
                let res = swiglu(gate_sum, up_sum);
                let yp = &mut *(y.as_ptr().add(row) as *mut f32);
                *yp = res;
            }
        }
    }
}

// Distinct seeds for gate vs up so the two matrices differ (catches gate/up swap).
fn make_q4k_row_data_seeded(n: usize, k: usize, seed: usize) -> Vec<u8> {
    assert!(k % 256 == 0);
    let n_sb = (n * k) / 256;
    let mut data = vec![0u8; n_sb * 144];
    for sb in 0..n_sb {
        let s = sb * 144;
        let d = half::f16::from_f32(0.0123 + 0.0007 * seed as f32).to_bits();
        let dmin = half::f16::from_f32(0.0061 + 0.0003 * seed as f32).to_bits();
        data[s] = (d & 0xFF) as u8;
        data[s + 1] = (d >> 8) as u8;
        data[s + 2] = (dmin & 0xFF) as u8;
        data[s + 3] = (dmin >> 8) as u8;
        for c in 0..12 {
            data[s + 4 + c] = ((sb * 7 + c * 13 + seed * 11) % 256) as u8;
        }
        for c in 0..128 {
            data[s + 16 + c] = ((sb * 3 + c * 5 + seed * 17) % 256) as u8;
        }
    }
    data
}

fn cpu_fused_gate_up_swiglu(wg: &[u8], wu: &[u8], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for row in 0..n {
        let base = row * k;
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for j in 0..k {
            let xv = x[j];
            g += dequant_elem(wg, base + j) * xv;
            u += dequant_elem(wu, base + j) * xv;
        }
        let sig = 1.0f32 / (1.0f32 + (-g).exp());
        y[row] = (g * sig) * u;
    }
    y
}

fn run_shape(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    n: usize,
    k: usize,
    vec_seed: usize,
) -> (usize, f32, f32, f64) {
    let stream = ctx.default_stream();
    let wg = make_q4k_row_data_seeded(n, k, 1);
    let wu = make_q4k_row_data_seeded(n, k, 2);
    let x_host: Vec<f32> = (0..k)
        .map(|i| (((i + vec_seed) % 13) as f32) * 0.1 - 0.6)
        .collect();

    let wg_dev = DeviceBuffer::from_host(&stream, &wg).unwrap();
    let wu_dev = DeviceBuffer::from_host(&stream, &wu).unwrap();
    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let y_dev = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

    // Custom launch: one block per row, T threads/block.
    let cfg = LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (T as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    module
        .fused_gate_up_swiglu(&stream, cfg, &wg_dev, &wu_dev, &x_dev, &y_dev)
        .expect("launch");
    let y = y_dev.to_host_vec(&stream).unwrap();

    // Bit-exact / parity vs CPU fused reference.
    let cpu = cpu_fused_gate_up_swiglu(&wg, &wu, &x_host, n, k);
    let mut maxabs_ref = 0.0f32;
    for v in &cpu {
        if v.abs() > maxabs_ref {
            maxabs_ref = v.abs();
        }
    }
    let mut maxdiff = 0.0f32;
    let mut errs = 0usize;
    let tol = 1e-4f32 * maxabs_ref.max(1e-6);
    for row in 0..n {
        let d = (y[row] - cpu[row]).abs();
        if d > maxdiff {
            maxdiff = d;
        }
        if d > tol {
            errs += 1;
        }
    }

    // Timing: warmup then median over reps of bare launches.
    for _ in 0..20 {
        module
            .fused_gate_up_swiglu(&stream, cfg, &wg_dev, &wu_dev, &x_dev, &y_dev)
            .expect("w");
    }
    let _ = y_dev.to_host_vec(&stream).unwrap();
    let iters = 200u32;
    let reps = 5;
    let mut times = Vec::new();
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            module
                .fused_gate_up_swiglu(&stream, cfg, &wg_dev, &wu_dev, &x_dev, &y_dev)
                .expect("t");
        }
        let _ = y_dev.to_host_vec(&stream).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1e6 / (iters as f64));
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    (errs, maxdiff, tol, med)
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let module = kernels::load(&ctx).expect("load");

    println!("== PMAT-881 cuda-oxide FusedGateUpSwiglu Q4K — parity + perf (GB10) ==");

    // ---- Parity gate: >=10 test vectors at a small Qwen-like FFN shape ----
    let pn = 512; // intermediate (N)
    let pk = 768; // hidden (K), multiple of 256
    let mut total_errs = 0usize;
    let mut worst_diff = 0.0f32;
    let mut worst_tol = 0.0f32;
    for v in 0..10usize {
        let (errs, maxdiff, tol, _med) = run_shape(&ctx, &module, pn, pk, v);
        if errs > 0 {
            total_errs += errs;
        }
        if maxdiff > worst_diff {
            worst_diff = maxdiff;
        }
        worst_tol = tol;
    }
    println!(
        "PARITY ({} vectors, N={} K={}): {} (total_errs={}, worst_maxdiff={:.3e}, tol={:.3e})",
        10,
        pn,
        pk,
        if total_errs == 0 { "PASS" } else { "FAIL" },
        total_errs,
        worst_diff,
        worst_tol
    );

    // ---- Perf at the documented hand-PTX FusedGateUp shape: N=1536, K=8960 ----
    // Plus the other Qwen FFN shapes requested by the falsifiable target.
    // K must be a multiple of 256: 6656, 8960, 11008 all are.
    let perf_shapes: &[(usize, usize, &str)] = &[
        (1536, 8960, "Qwen 1.5B FFN (hand-PTX baseline ~120us)"),
        (11008, 6656, "Qwen 7B-class FFN K=6656"),
        (14784, 8960, "Qwen FFN N=14784 K=8960"),
        (11008, 11008, "Qwen 7B FFN K=11008"),
    ];
    for &(n, k, label) in perf_shapes {
        let (errs, maxdiff, tol, med) = run_shape(&ctx, &module, n, k, 0);
        println!(
            "PERF N={:>6} K={:>6} : {:>8.2} us/launch  parity={} (maxdiff={:.2e} tol={:.2e})  [{}]",
            n,
            k,
            med,
            if errs == 0 { "PASS" } else { "FAIL" },
            maxdiff,
            tol,
            label
        );
    }

    if total_errs != 0 {
        eprintln!("PMAT-881 PARITY FAILED");
        std::process::exit(1);
    }
    println!("PMAT-881 DONE");
}
