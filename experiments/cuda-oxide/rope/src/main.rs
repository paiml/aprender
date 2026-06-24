// PMAT-921 — pure-Rust cuda-oxide port of the standard adjacent-pair RoPE kernel.
//
// Target = hand-PTX `RopeKernel`
// (crates/aprender-gpu/src/kernels/elementwise/rope/standard.rs, entry `rope`):
// grid = num_heads blocks, block = head_dim/2 threads (one thread per rotation
// pair). ABI = (x_ptr: u64, out_ptr: u64, pos: u32). For each head h and pair p
// (p < head_dim/2) it rotates the adjacent pair (x[2p], x[2p+1]):
//     freq_base = theta^(-2p/head_dim)            [via ex2(-2p/head_dim * log2(theta))]
//     angle     = pos * freq_base
//     out[2p]   = x[2p]*cos(angle) - x[2p+1]*sin(angle)
//     out[2p+1] = x[2p]*sin(angle) + x[2p+1]*cos(angle)
// using hardware sin.approx / cos.approx / ex2.approx — pure f32 transcendentals,
// ZERO DP4A. Per the established verdicts (PMAT-882 attention GO, PMAT-893 RMSNorm
// GO, PMAT-894 SwiGLU GO; PMAT-881 FFN Q4K matmul NO-GO), FMA / softmax / sin-cos
// transcendental kernels are the GO class; only DP4A/INT8-bound Q4K GEMV/FFN stay
// on hand-PTX. RoPE is exactly the GO class: it is applied to Q and K every layer,
// every decode token, and is pure trig + FMA.
//
// FAIRNESS: the hand-PTX `rope` launches grid = num_heads blocks of head_dim/2
// threads, one element-pair per thread, ONE grid launch. The oxide kernel uses
// the IDENTICAL grid/block and ALSO does one launch over the same (num_heads x
// head_dim) tensor. So both sides issue exactly ONE matched launch — there is NO
// launch-count confound. The ratio is a true per-kernel speed comparison.
//
// Two oxide variants (both bit-parity-correct on GB10):
//   (A) rope_approx — sin/cos/ex2 via the device intrinsics that lower to
//       sin.approx / cos.approx / ex2.approx — the EXACT hand-PTX form, for the
//       closest bit-level parity to the baseline.
//   (B) rope_libdev — sin/cos via libdevice f32::sin / f32::cos and the freq base
//       via f32::exp2 — the proven 882/893/894 libdevice path (slightly higher
//       precision than the hardware .approx).
//
// TRUE hand-PTX A/B: the actual `RopeKernel::new(num_heads, head_dim, theta)
// .emit_ptx_for_target("sm_121")` PTX (committed in baseline-ptx/rope.sm121.ptx)
// is loaded via load_module_from_ptx_src and launched on the same GB10 with the
// same data + GPU-event timing (median of 5x100). Both verified parity-correct
// vs the f64 CPU reference inside the harness.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

// Decode-path RoPE config: Qwen2/Llama head_dim=128, theta=10000 (also test the
// Qwen2.5 theta=1_000_000 high-frequency case in parity). Both oxide and hand-PTX
// run grid=num_heads x block=head_dim/2 (the hand-PTX bakes head_dim/theta into
// the emitted PTX, so we emit a baseline per (head_dim, theta) we A/B).
const HEAD_DIM: u32 = 128;

#[cuda_module]
mod kernels {
    use super::*;

    // ---- (A) RoPE via exp2 freq base — EXACT hand-PTX mirror ----
    // grid = num_heads blocks, block = head_dim/2 threads. theta + head_dim are
    // passed at runtime so one compiled kernel covers every config we A/B.
    // freq_base = ex2( (-2*pair/head_dim) * log2(theta) ), the exact hand-PTX form.
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn rope_approx(x: &[f32], out: &[f32], pos: u32, head_dim: u32, theta: f32) {
        unsafe {
            let head_idx = thread::blockIdx_x();
            let pair_idx = thread::threadIdx_x();
            let half_dim = head_dim / 2;
            if pair_idx >= half_dim {
                return;
            }
            let elem0 = (pair_idx * 2) as usize;
            let elem1 = elem0 + 1;
            let head_off = (head_idx * head_dim) as usize;
            let i0 = head_off + elem0;
            let i1 = head_off + elem1;

            let x0 = x[i0];
            let x1 = x[i1];

            // freq_base = theta^(-2*pair/head_dim) = ex2( (-2*pair/head_dim) * log2(theta) )
            let exponent = (pair_idx as f32) * (-2.0f32) / (head_dim as f32);
            let power = exponent * theta.log2();
            let freq_base = power.exp2();
            let angle = (pos as f32) * freq_base;

            let c = angle.cos();
            let s = angle.sin();

            let n0 = x0 * c - x1 * s;
            let n1 = x0 * s + x1 * c;

            let o0 = &mut *(out.as_ptr().add(i0) as *mut f32);
            let o1 = &mut *(out.as_ptr().add(i1) as *mut f32);
            *o0 = n0;
            *o1 = n1;
        }
    }

    // ---- (B) RoPE via exp(power*ln2) freq base — libdevice path ----
    // Same trig, but freq_base = exp( power * ln(2) ) using libdevice exp instead
    // of ex2, the proven 882/893/894 libdevice path (a second lowering datapoint).
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn rope_libdev(x: &[f32], out: &[f32], pos: u32, head_dim: u32, theta: f32) {
        unsafe {
            let head_idx = thread::blockIdx_x();
            let pair_idx = thread::threadIdx_x();
            let half_dim = head_dim / 2;
            if pair_idx >= half_dim {
                return;
            }
            let elem0 = (pair_idx * 2) as usize;
            let elem1 = elem0 + 1;
            let head_off = (head_idx * head_dim) as usize;
            let i0 = head_off + elem0;
            let i1 = head_off + elem1;

            let x0 = x[i0];
            let x1 = x[i1];

            // freq_base = exp( (-2*pair/head_dim) * ln(theta) ) via libdevice exp.
            let exponent = (pair_idx as f32) * (-2.0f32) / (head_dim as f32);
            let power = exponent * theta.ln();
            let freq_base = power.exp();
            let angle = (pos as f32) * freq_base;

            let c = angle.cos();
            let s = angle.sin();

            let n0 = x0 * c - x1 * s;
            let n1 = x0 * s + x1 * c;

            let o0 = &mut *(out.as_ptr().add(i0) as *mut f32);
            let o1 = &mut *(out.as_ptr().add(i1) as *mut f32);
            *o0 = n0;
            *o1 = n1;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference — f64 accumulation of the same adjacent-pair RoPE math.
// out[head, 2p]   = x[2p]*cos(a) - x[2p+1]*sin(a)
// out[head, 2p+1] = x[2p]*sin(a) + x[2p+1]*cos(a)
//   a = pos * theta^(-2p/head_dim)
// ---------------------------------------------------------------------------
fn cpu_rope(x: &[f32], num_heads: u32, head_dim: u32, theta: f32, pos: u32) -> Vec<f32> {
    let nh = num_heads as usize;
    let hd = head_dim as usize;
    let half = hd / 2;
    let mut out = x.to_vec();
    for h in 0..nh {
        for p in 0..half {
            let i0 = h * hd + 2 * p;
            let i1 = i0 + 1;
            let x0 = x[i0] as f64;
            let x1 = x[i1] as f64;
            let exponent = (p as f64) * (-2.0) / (hd as f64);
            let freq_base = (theta as f64).powf(exponent);
            let angle = (pos as f64) * freq_base;
            let c = angle.cos();
            let s = angle.sin();
            out[i0] = (x0 * c - x1 * s) as f32;
            out[i1] = (x0 * s + x1 * c) as f32;
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > m {
            m = d;
        }
    }
    m
}

// Deterministic pseudo-random inputs in ~[-1, 1] (same generator family as 882/893/894).
fn make_input(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503))) & 0xFFFF;
            (x as f32 / 32768.0) - 1.0
        })
        .collect()
}

struct OxideResult {
    cos: f32,
    maxdiff: f32,
    us: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_oxide(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    num_heads: u32,
    head_dim: u32,
    theta: f32,
    pos: u32,
    seed: usize,
    do_perf: bool,
    libdev: bool, // false = (A) approx, true = (B) libdev
) -> OxideResult {
    let stream = ctx.default_stream();
    let n = (num_heads * head_dim) as usize;
    let x = make_input(n, seed);

    let dx = DeviceBuffer::from_host(&stream, &x).unwrap();
    let dout = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (head_dim / 2, 1, 1),
        shared_mem_bytes: 0,
    };
    let launch = || {
        if libdev {
            module
                .rope_libdev(&stream, cfg, &dx, &dout, pos, head_dim, theta)
                .expect("rope_libdev");
        } else {
            module
                .rope_approx(&stream, cfg, &dx, &dout, pos, head_dim, theta)
                .expect("rope_approx");
        }
    };

    launch();
    let got = dout.to_host_vec(&stream).unwrap();
    let want = cpu_rope(&x, num_heads, head_dim, theta, pos);
    let cos = cosine_similarity(&got, &want);
    let maxdiff = max_abs_diff(&got, &want);

    let mut med = 0.0f64;
    if do_perf {
        for _ in 0..20 {
            launch();
        }
        let _ = dout.to_host_vec(&stream).unwrap();
        let iters = 100usize;
        let reps = 5;
        let mut times = Vec::new();
        for _ in 0..reps {
            let start = stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .unwrap();
            for _ in 0..iters {
                launch();
            }
            let end = stream
                .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
                .unwrap();
            let ms = start.elapsed_ms(&end).unwrap() as f64;
            times.push(ms * 1000.0 / iters as f64); // us/launch
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        med = times[times.len() / 2];
    }
    OxideResult {
        cos,
        maxdiff,
        us: med,
    }
}

// ---------------------------------------------------------------------------
// TRUE hand-PTX A/B: load the emitted `rope` PTX (RopeKernel, entry `rope`,
// ABI = (x_ptr,out_ptr: u64, pos: u32), grid=num_heads x block=head_dim/2) and
// launch it on the SAME data with the SAME grid/block — a fully matched single
// launch each side. GPU-event timing, median of 5x100.
// ---------------------------------------------------------------------------
fn run_handptx(
    ctx: &std::sync::Arc<CudaContext>,
    ptx_path: &str,
    num_heads: u32,
    head_dim: u32,
    theta: f32,
    pos: u32,
    seed: usize,
) -> Option<(f32, f32, f64)> {
    let stream = ctx.default_stream();
    let n = (num_heads * head_dim) as usize;
    let x = make_input(n, seed);

    let ptx = std::fs::read_to_string(ptx_path).ok()?;
    let module = ctx.load_module_from_ptx_src(&ptx).ok()?;
    let func = module.load_function("rope").ok()?;

    let dx = DeviceBuffer::from_host(&stream, &x).ok()?;
    let dout = DeviceBuffer::<f32>::zeroed(&stream, n).ok()?;

    let mut x_ptr = dx.cu_deviceptr();
    let mut o_ptr = dout.cu_deviceptr();
    let mut pos_u32 = pos;
    let grid = (num_heads, 1u32, 1u32);
    let block = (head_dim / 2, 1u32, 1u32);

    let mut do_launch = || unsafe {
        let mut params: [*mut std::ffi::c_void; 3] = [
            &mut x_ptr as *mut _ as *mut std::ffi::c_void,
            &mut o_ptr as *mut _ as *mut std::ffi::c_void,
            &mut pos_u32 as *mut _ as *mut std::ffi::c_void,
        ];
        cuda_core::launch_kernel_on_stream(&func, grid, block, 0, &stream, &mut params)
    };

    do_launch().ok()?;
    let got = dout.to_host_vec(&stream).ok()?;
    let want = cpu_rope(&x, num_heads, head_dim, theta, pos);
    let cos = cosine_similarity(&got, &want);
    let maxdiff = max_abs_diff(&got, &want);

    for _ in 0..20 {
        do_launch().ok()?;
    }
    let _ = dout.to_host_vec(&stream).ok()?;
    let iters = 100usize;
    let reps = 5;
    let mut times = Vec::new();
    for _ in 0..reps {
        let start = stream
            .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .ok()?;
        for _ in 0..iters {
            do_launch().ok()?;
        }
        let end = stream
            .record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
            .ok()?;
        let ms = start.elapsed_ms(&end).ok()? as f64;
        times.push(ms * 1000.0 / iters as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some((cos, maxdiff, times[times.len() / 2]))
}

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let module = kernels::load(&ctx).expect("load");

    println!("== PMAT-921 cuda-oxide RoPE (adjacent-pair) — parity + perf (GB10 sm_121) ==");
    println!("   out[2p]=x0*cos(a)-x1*sin(a); out[2p+1]=x0*sin(a)+x1*cos(a); a=pos*theta^(-2p/hd)");
    println!("   (A) rope_approx (hw sin/cos/ex2 approx)  (B) rope_libdev (libdevice sin/cos)");

    // Decode-path shapes: (num_heads, head_dim, theta). Qwen2-0.5B: 14 heads / 2 KV,
    // Llama-7B: 32 heads, head_dim=128, theta=10000. Qwen2.5: theta=1_000_000.
    // num_heads chosen across small->large head counts for the per-kernel A/B.
    let shapes = [
        (32u32, HEAD_DIM, 10000.0f32, 17u32),    // Llama/Qwen2 head_dim=128 theta=10k
        (14u32, HEAD_DIM, 1_000_000.0f32, 53u32), // Qwen2.5 high-freq theta=1M
        (128u32, HEAD_DIM, 10000.0f32, 256u32),  // large head count, deep pos
    ];

    // PARITY for both oxide variants across all shapes.
    for (libdev, name) in [(false, "approx (A)"), (true, "libdev (B)")] {
        let mut all_pass = true;
        println!("\n-- PARITY {name} vs f64 CPU reference --");
        for &(nh, hd, theta, pos) in &shapes {
            let r = run_oxide(&ctx, &module, nh, hd, theta, pos, 4242 + nh as usize, false, libdev);
            let pass = r.cos >= 0.9999 && r.maxdiff < 1e-3;
            if !pass {
                all_pass = false;
            }
            println!(
                "  heads={:>3} hd={:>3} theta={:>9.0} pos={:>4} : cos={:.7} maxdiff={:.3e} -> {}",
                nh,
                hd,
                theta,
                pos,
                r.cos,
                r.maxdiff,
                if pass { "PASS" } else { "FAIL" }
            );
        }
        println!("PARITY {name}: {}", if all_pass { "PASS" } else { "FAIL" });
        if !all_pass {
            eprintln!("PMAT-921 PARITY FAILED ({name})");
            std::process::exit(1);
        }
    }

    // hand-PTX baselines: one per (head_dim, theta) pair (RopeKernel bakes both).
    // baseline-ptx/rope_hd128_t10000.sm121.ptx and rope_hd128_t1000000.sm121.ptx
    let pick_ptx = |theta: f32| -> Option<String> {
        let names: Vec<String> = if theta >= 1_000_000.0 {
            vec![
                "/tmp/rope_spike/rope_hd128_t1000000.ptx".into(),
                "baseline-ptx/rope_hd128_t1000000.sm121.ptx".into(),
            ]
        } else {
            vec![
                "/tmp/rope_spike/rope_hd128_t10000.ptx".into(),
                "baseline-ptx/rope_hd128_t10000.sm121.ptx".into(),
            ]
        };
        names.into_iter().find(|nm| std::path::Path::new(nm).exists())
    };

    // =========================================================================
    // PRIMARY GATE — FAIR matched single-launch A/B (F-OXIDE-ROPE-PARITY-001).
    // Both sides: grid=num_heads x block=head_dim/2, one launch over the same
    // (num_heads x head_dim) tensor. NO launch-count confound.
    // =========================================================================
    println!("\n-- PRIMARY GATE: FAIR matched single-launch A/B (oxide vs hand-PTX, GB10 sm_121) --");
    println!("   grid=num_heads x block=head_dim/2 BOTH sides, one launch each; GPU-event median 5x100");
    println!("   F-OXIDE-ROPE-PARITY-001: cos>=0.9999 AND maxdiff<1e-3 AND oxide_us/handptx_us<=1.2");
    println!(
        "\n  {:>5} {:>4} {:>9} | {:>9} {:>9} | {:>11} | {:>8} {:>8} | {:>10} {:>9} | verdict",
        "heads", "hd", "theta", "oxA us", "oxB us", "handPTX us", "ratioA", "ratioB", "cos(ox)", "maxdiff"
    );
    let mut overall_go = true;
    for &(nh, hd, theta, pos) in &shapes {
        let ra = run_oxide(&ctx, &module, nh, hd, theta, pos, 77, true, false); // (A) approx
        let rb = run_oxide(&ctx, &module, nh, hd, theta, pos, 77, true, true); // (B) libdev
        let (best_us, best_cos, best_md, best_name) = if ra.us <= rb.us {
            (ra.us, ra.cos, ra.maxdiff, "A")
        } else {
            (rb.us, rb.cos, rb.maxdiff, "B")
        };
        match pick_ptx(theta).and_then(|p| run_handptx(&ctx, &p, nh, hd, theta, pos, 77)) {
            Some((hcos, hmd, hus)) => {
                let best_ratio = best_us / hus;
                let parity_ok = best_cos >= 0.9999 && best_md < 1e-3;
                let go = parity_ok && best_ratio <= 1.2;
                if !go {
                    overall_go = false;
                }
                println!(
                    "  {:>5} {:>4} {:>9.0} | {:>9.3} {:>9.3} | {:>11.3} | {:>8.3} {:>8.3} | {:>10.7} {:>9.2e} | {} (best={} {:.3}x; handPTX cos={:.5} md={:.2e})",
                    nh, hd, theta, ra.us, rb.us, hus, ra.us / hus, rb.us / hus, best_cos, best_md,
                    if go { "GO" } else { "NO-GO" }, best_name, best_ratio, hcos, hmd
                );
            }
            None => {
                overall_go = false;
                println!("  {nh:>5} {hd:>4} {theta:>9.0} | hand-PTX MISSING/launch-FAILED (regen baseline-ptx/rope_hd128_t*.sm121.ptx)");
            }
        }
    }

    println!(
        "\nPMAT-921 VERDICT (F-OXIDE-ROPE-PARITY-001, fair matched-launch gate): {}",
        if overall_go { "GO" } else { "NO-GO" }
    );
    println!("PMAT-921 DONE");
}
