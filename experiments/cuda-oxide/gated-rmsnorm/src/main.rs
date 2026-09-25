// #3522 OXIDE-001 O-1 — pure-Rust cuda-oxide port of the GDN gated RMSNorm.
//
// Target = hand-PTX `GatedRmsNormKernel`
// (crates/aprender-gpu/src/kernels/gdn/gated_rmsnorm.rs, entry `gdn_gated_rmsnorm`):
// one warp per head, head_dim and eps baked into the PTX, params
// (input, gate, weight, output).
//
//   for each head chunk of head_dim:
//       rms_scale = 1 / sqrt(sum(input^2) / head_dim + eps)
//       out[i]    = (input[i] * rms_scale * weight[i]) * silu(gate[i])
//
// `weight` is `[head_dim]`, shared across heads; `gate` is per head. CPU
// reference = `gated_rmsnorm` in crates/aprender-serve/src/gguf/inference/
// forward/forward_qwen35.rs, evaluated here in f64.
//
// SAFETY SHAPE (kernel-safety, #3522): both device kernels are safe Rust — no
// `unsafe` block, no raw pointer. Reads go through bounds-checked slice indexing;
// the one write goes through `DisjointSlice<f32, LinearTiles<TILE>>`, whose run
// is minted from the launch-checked thread index, so each lane owns TILE
// contiguous outputs and disjointness is a type fact, not a comment.
//
// That fixes the geometry: one warp per head, TILE = head_dim / 32 contiguous
// elements per lane. head_dim = 128 (Qwen3.5 head_v_dim, the only production
// shape) gives TILE = 4. The hand PTX strides lanes by 32 instead; both are one
// warp per head and read every element once, so the A/B is like for like.
//
// Two variants, differing only in silu's exponential:
//   (A) gated_rmsnorm_exp — `exp(-g)` (libdevice), the more precise
//   (B) gated_rmsnorm_ex2 — `exp2(-g * log2 e)`, mirroring the hand PTX's ex2.approx
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line
// prefixed `RECEIPT ` per variant; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_gated_rmsnorm/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, LinearTiles, cuda_module, kernel, launch_bounds, launch_contract, thread, warp,
};
use std::sync::Arc;

const HEAD_DIM: usize = 128;
const WARP: usize = 32;
const TILE: usize = HEAD_DIM / WARP;
const EPS: f32 = 1e-6;
const FULL: u32 = 0xFFFF_FFFF;

// F-OXIDE-ROPE-PARITY-001 form, as #3522's kernel-parity shape states it.
const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;
// kernel-timing shape: oxide_us / handptx_us <= 1.2.
const TIMING_RATIO_MAX: f64 = 1.2;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sum of squares of this lane's TILE, reduced across the warp and
    /// broadcast: the same shfl.down butterfly and lane-0 broadcast as the
    /// hand PTX.
    #[inline(always)]
    fn warp_sum_sq(input: &[f32], base: usize) -> f32 {
        let mut sq = 0.0f32;
        for k in 0..TILE {
            let v = input[base + k];
            sq += v * v;
        }
        sq += warp::shuffle_down_f32_sync(FULL, sq, 16);
        sq += warp::shuffle_down_f32_sync(FULL, sq, 8);
        sq += warp::shuffle_down_f32_sync(FULL, sq, 4);
        sq += warp::shuffle_down_f32_sync(FULL, sq, 2);
        sq += warp::shuffle_down_f32_sync(FULL, sq, 1);
        warp::shuffle_f32_sync(FULL, sq, 0)
    }

    /// (A) silu via `exp`.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn gated_rmsnorm_exp(
        input: &[f32],
        gate: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32, LinearTiles<TILE>>,
        eps: f32,
    ) {
        let t = thread::index_1d_u32(launch_context);
        let base = t.get() as usize * TILE;
        let lane_base = base % HEAD_DIM;
        let total = warp_sum_sq(input, base);
        let rms_scale = 1.0f32 / (total / HEAD_DIM as f32 + eps).sqrt();
        let Some(mut run) = out.thread_run32(t) else {
            return;
        };
        for k in 0..run.len() {
            let i = base + k as usize;
            let g = gate[i];
            let silu = g / (1.0f32 + (-g).exp());
            let v = input[i] * rms_scale * weight[lane_base + k as usize] * silu;
            if let Some(mut slot) = run.at(k) {
                slot.write(v);
            }
        }
    }

    /// (B) silu via `exp2(-g * log2 e)`, the hand PTX's form.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, coordinates = u32, block = (32, 1, 1))]
    pub fn gated_rmsnorm_ex2(
        input: &[f32],
        gate: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32, LinearTiles<TILE>>,
        eps: f32,
    ) {
        let t = thread::index_1d_u32(launch_context);
        let base = t.get() as usize * TILE;
        let lane_base = base % HEAD_DIM;
        let total = warp_sum_sq(input, base);
        let rms_scale = 1.0f32 / (total / HEAD_DIM as f32 + eps).sqrt();
        let Some(mut run) = out.thread_run32(t) else {
            return;
        };
        for k in 0..run.len() {
            let i = base + k as usize;
            let g = gate[i];
            let silu = g / (1.0f32 + ((-g) * std::f32::consts::LOG2_E).exp2());
            let v = input[i] * rms_scale * weight[lane_base + k as usize] * silu;
            if let Some(mut slot) = run.at(k) {
                slot.write(v);
            }
        }
    }
}

/// f64 evaluation of `gated_rmsnorm` (forward_qwen35.rs).
fn cpu_gated_rmsnorm(input: &[f32], gate: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (h, chunk) in input.chunks_exact(HEAD_DIM).enumerate() {
        let ss: f64 = chunk.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let rms = 1.0 / (ss / HEAD_DIM as f64 + f64::from(eps)).sqrt();
        for i in 0..HEAD_DIM {
            let j = h * HEAD_DIM + i;
            let g = f64::from(gate[j]);
            let silu = g / (1.0 + (-g).exp());
            out[j] = (f64::from(input[j]) * rms * f64::from(weight[i]) * silu) as f32;
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Deterministic inputs. Per-head input magnitudes differ (as in the aprender-gpu
/// device test), so a norm taken over the whole vector instead of per head
/// misses; the gate spans the silu knee on both sides.
fn make_inputs(heads: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let mut input = Vec::with_capacity(heads * HEAD_DIM);
    for h in 0..heads {
        let scale = 0.2 * (h as f32 + 1.0);
        for _ in 0..HEAD_DIM {
            input.push(next() * scale);
        }
    }
    let gate = (0..heads * HEAD_DIM).map(|_| next() * 4.0).collect();
    let weight = (0..HEAD_DIM).map(|_| 1.0 + 0.5 * next()).collect();
    (input, gate, weight)
}

struct Measured {
    cos: f64,
    maxdiff: f32,
    us: f64,
}

/// GPU-event median of 5 x 100 warm launches, in microseconds per launch.
fn time_us(stream: &Arc<cuda_core::CudaStream>, mut launch: impl FnMut()) -> f64 {
    for _ in 0..20 {
        launch();
    }
    stream.synchronize().expect("warmup sync");
    let flags = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let iters = 100;
    let mut times: Vec<f64> = (0..5)
        .map(|_| {
            let start = stream.record_event(flags).expect("event");
            for _ in 0..iters {
                launch();
            }
            let end = stream.record_event(flags).expect("event");
            f64::from(start.elapsed_ms(&end).expect("elapsed")) * 1000.0 / f64::from(iters)
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    heads: usize,
    ex2: bool,
    perf: bool,
) -> Measured {
    let stream = ctx.default_stream();
    let (input, gate, weight) = make_inputs(heads, 0x3522_0001 + heads as u64);
    let d_in = DeviceBuffer::from_host(&stream, &input).expect("input");
    let d_gate = DeviceBuffer::from_host(&stream, &gate).expect("gate");
    let d_w = DeviceBuffer::from_host(&stream, &weight).expect("weight");
    let mut d_out = DeviceBuffer::<f32>::zeroed(&stream, heads * HEAD_DIM).expect("out");

    let config = LaunchConfig1D::new(heads as u32, WARP as u32, 0);
    let launch = |d_out: &mut DeviceBuffer<f32>| {
        if ex2 {
            let p = module
                .prepare_gated_rmsnorm_ex2(config)
                .expect("prepare ex2");
            module
                .gated_rmsnorm_ex2(&stream, &p, &d_in, &d_gate, &d_w, d_out, EPS)
                .expect("launch ex2");
        } else {
            let p = module
                .prepare_gated_rmsnorm_exp(config)
                .expect("prepare exp");
            module
                .gated_rmsnorm_exp(&stream, &p, &d_in, &d_gate, &d_w, d_out, EPS)
                .expect("launch exp");
        }
    };
    launch(&mut d_out);
    let got = d_out.to_host_vec(&stream).expect("download");
    let want = cpu_gated_rmsnorm(&input, &gate, &weight, EPS);
    let us = if perf {
        time_us(&stream, || launch(&mut d_out))
    } else {
        0.0
    };
    Measured {
        cos: cosine(&got, &want),
        maxdiff: max_abs_diff(&got, &want),
        us,
    }
}

/// The hand PTX, loaded from the committed baseline and launched on the same data:
/// grid (heads), block (32), four pointer params.
fn run_handptx(ctx: &Arc<CudaContext>, ptx: &str, heads: usize) -> (Measured, u32) {
    let stream = ctx.default_stream();
    let (input, gate, weight) = make_inputs(heads, 0x3522_0001 + heads as u64);
    let module = ctx.load_module_from_ptx_src(ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_gated_rmsnorm")
        .expect("gdn_gated_rmsnorm");
    let regs = func.num_registers().expect("regs");
    let d_in = DeviceBuffer::from_host(&stream, &input).expect("input");
    let d_gate = DeviceBuffer::from_host(&stream, &gate).expect("gate");
    let d_w = DeviceBuffer::from_host(&stream, &weight).expect("weight");
    let d_out = DeviceBuffer::<f32>::zeroed(&stream, heads * HEAD_DIM).expect("out");
    let mut ptrs = [
        d_in.cu_deviceptr(),
        d_gate.cu_deviceptr(),
        d_w.cu_deviceptr(),
        d_out.cu_deviceptr(),
    ];
    let mut launch = || {
        let mut params: Vec<*mut std::ffi::c_void> =
            ptrs.iter_mut().map(|p| (p as *mut u64).cast()).collect();
        // SAFETY: four device pointers matching the entry's four .u64 params;
        // every buffer holds heads * HEAD_DIM f32 (weight HEAD_DIM), which is
        // what a (heads, 32) launch of this kernel touches.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &func,
                (heads as u32, 1, 1),
                (WARP as u32, 1, 1),
                0,
                &stream,
                &mut params,
            )
        }
        .expect("hand PTX launch");
    };
    launch();
    let got = d_out.to_host_vec(&stream).expect("download");
    let want = cpu_gated_rmsnorm(&input, &gate, &weight, EPS);
    let us = time_us(&stream, &mut launch);
    (
        Measured {
            cos: cosine(&got, &want),
            maxdiff: max_abs_diff(&got, &want),
            us,
        },
        regs,
    )
}

/// Register count of an oxide kernel, read back from the embedded module.
fn oxide_registers(ctx: &Arc<CudaContext>, name: &str) -> Option<u32> {
    let module = cuda_host::embedded::load_all_ptx_bundles_merged(ctx).ok()?;
    module.load_function(name).ok()?.num_registers().ok()
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let (major, minor) = ctx.compute_capability().expect("cc");
    let sm = format!("sm_{major}{minor}");
    // SAFETY: the embedded module is built from this crate's `kernels`.
    let module = unsafe { kernels::load(&ctx) }.expect("load oxide module");

    let ptx_path = format!("baseline-ptx/gdn_gated_rmsnorm_h{HEAD_DIM}.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_gated_rmsnorm_ptx_golden");
        std::process::exit(2);
    });

    println!("== #3522 O-1 cuda-oxide gated RMSNorm ({sm}) ==");
    println!("   head_dim={HEAD_DIM} eps={EPS:e} tile={TILE}/lane, 1 warp/head");

    // Parity sweep: Qwen3.5-0.8B (16 heads), larger GDN configs, and a 1-head edge.
    let mut all_ok = true;
    // A timing NO-GO is its own exit code (4), not a pass: the receipts still
    // print, so receipt.sh records `"pass":false` and then fails (#3522 quorum).
    let mut timing_all_ok = true;
    for ex2 in [false, true] {
        let name = if ex2 { "ex2 (B)" } else { "exp (A)" };
        for heads in [1usize, 16, 32, 48] {
            let r = run_oxide(&ctx, &module, heads, ex2, false);
            let ok = r.cos >= PARITY_COS && r.maxdiff < PARITY_MAXDIFF;
            all_ok &= ok;
            println!(
                "  parity {name} heads={heads:>2}: cos={:.7} maxdiff={:.3e} {}",
                r.cos,
                r.maxdiff,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    // Timing: the production shape (16 heads) plus 32 and 48.
    let regs_hand;
    {
        let (h, r) = run_handptx(&ctx, &ptx, 16);
        regs_hand = r;
        let ok = h.cos >= PARITY_COS && h.maxdiff < PARITY_MAXDIFF;
        println!(
            "  hand PTX heads=16: cos={:.7} maxdiff={:.3e} regs={r} {}",
            h.cos,
            h.maxdiff,
            if ok { "PASS" } else { "FAIL" }
        );
        all_ok &= ok;
    }
    println!("\n  heads | variant | oxide us | handPTX us | ratio | verdict");
    for ex2 in [false, true] {
        let (variant, entry) = if ex2 {
            ("ex2", "gated_rmsnorm_ex2")
        } else {
            ("exp", "gated_rmsnorm_exp")
        };
        let regs = oxide_registers(&ctx, entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, 0usize);
        let mut parity = (1.0f64, 0.0f32);
        for heads in [16usize, 32, 48] {
            let o = run_oxide(&ctx, &module, heads, ex2, true);
            let (h, _) = run_handptx(&ctx, &ptx, heads);
            let ratio = o.us / h.us;
            let ok = ratio <= TIMING_RATIO_MAX;
            println!(
                "  {heads:>5} | {variant:>7} | {:>8.3} | {:>10.3} | {ratio:.3} | {}",
                o.us,
                h.us,
                if ok { "GO" } else { "NO-GO" }
            );
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst = (o.us, h.us, heads);
            }
            parity = (parity.0.min(o.cos), parity.1.max(o.maxdiff));
        }
        let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
        timing_all_ok &= timing_ok;
        // One line per variant; receipt.sh adds host, sha, ptxas and writes the file.
        println!(
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_gated_rmsnorm\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_dim\":{HEAD_DIM},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"cos_floor\":{PARITY_COS},\"maxdiff_ceiling\":{PARITY_MAXDIFF:e},\"pass\":{}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_heads\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
            parity.0,
            parity.1,
            parity.0 >= PARITY_COS && parity.1 < PARITY_MAXDIFF,
            worst.0,
            worst.1,
            worst.2,
            regs.map_or("null".to_string(), |r| r.to_string()),
        );
    }

    if !all_ok {
        eprintln!("#3522 PARITY FAILED");
        std::process::exit(1);
    }
    if !timing_all_ok {
        eprintln!("#3522 TIMING NO-GO (oxide/hand > {TIMING_RATIO_MAX})");
        std::process::exit(4);
    }
    println!("#3522 O-1 DONE");
}
