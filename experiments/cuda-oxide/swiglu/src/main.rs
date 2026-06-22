// PMAT-894 — pure-Rust cuda-oxide port of the fused SwiGLU / SiLU ACTIVATION kernel.
//
// Target = hand-PTX `FusedSwigluKernel`
// (crates/aprender-gpu/src/kernels/elementwise/swiglu.rs, entry `fused_swiglu`):
// a FLAT 1-D grid kernel — gid = ctaid.x*ntid.x + tid.x, bounds-checked vs `n`,
// one element per thread. It computes
//     out[i] = silu(gate[i]) * up[i]
//     silu(x) = x * sigmoid(x)
//     sigmoid(x) = 1 / (1 + exp(-x))   [hand-PTX uses ex2.approx: 1/(1+exp2(-x*log2e))]
// over already-dequantized f32 gate/up vectors. This is the elementwise SwiGLU
// ACTIVATION (transcendental ex2 class), NOT the Q4K gate+up *matmul* (PMAT-881,
// DP4A-bound NO-GO). Per the PMAT-882 verdict, FMA/softmax/ex2 kernels are the GO
// class — same class as PMAT-882 softmax and PMAT-893 RMSNorm (both GO).
//
// This kernel is PURE elementwise (2 reads + 1 write per element, ~6 FLOP + 1
// transcendental), i.e. strongly DRAM-bandwidth-bound. The honest expectation is
// PARITY (a tie) that, by passing the <=1.2x gate, RETIRES the hand-PTX + the
// GH-480 Blackwell-JIT workaround — still a GO datapoint.
//
// FAIRNESS: the hand-PTX `fused_swiglu` processes the WHOLE length n in ONE grid
// launch with a multi-block grid (ceil(n/256) blocks of 256 threads). The oxide
// kernel uses the IDENTICAL grid/block and ALSO does one launch. So unlike the
// PMAT-893 RMSNorm single-row ABI, there is NO launch-count confound here: both
// sides issue exactly one matched launch. The ratio is a true per-kernel compare.
//
// Two oxide variants:
//   (A) swiglu_libdev — sigmoid via libdevice f32::exp (the proven 882/893 path;
//       slightly HIGHER precision than the hand-PTX ex2.approx).
//   (B) swiglu_ex2    — mirrors the hand-PTX EXACTLY: 1/(1+exp2(-gate*log2e))
//       via f32::exp2, for the closest bit-level parity to the baseline.
//
// TRUE hand-PTX A/B: the actual `FusedSwigluKernel::new(n).emit_ptx_for_target
// ("sm_121")` PTX (committed in baseline-ptx/fused_swiglu.sm121.ptx) is loaded
// via load_module_from_ptx_src and launched on the same GB10 with the same data
// + GPU-event timing (median of 5x100). Both verified parity-correct vs the f64
// CPU reference inside the harness. Negative gate values are deliberately in the
// input range to exercise the sigmoid on both sides of 0.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

// Threads per block — matches a typical elementwise launch; both oxide and the
// hand-PTX run ceil(n/TPB) blocks of TPB threads (the hand-PTX has no baked block
// size, so we choose the same grid for both = a fair matched launch).
const TPB: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    // ---- (A) SwiGLU via libdevice exp — out[i] = silu(gate[i]) * up[i] ----
    // Flat 1-D grid, faithful `fused_swiglu` analog. sigmoid via 1/(1+exp(-x)).
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn swiglu_libdev(gate: &[f32], up: &[f32], out: &[f32], n: u32) {
        unsafe {
            let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
            if gid >= n {
                return;
            }
            let i = gid as usize;
            let g = gate[i];
            let u = up[i];
            // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
            let sig = 1.0f32 / (1.0f32 + (-g).exp());
            let r = g * sig * u;
            let op = &mut *(out.as_ptr().add(i) as *mut f32);
            *op = r;
        }
    }

    // ---- (B) SwiGLU via exp2 — bit-closest to the hand-PTX ex2.approx path ----
    // sigmoid(g) = 1 / (1 + exp2(-g * log2e)), the exact form the hand-PTX emits.
    #[kernel]
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn swiglu_ex2(gate: &[f32], up: &[f32], out: &[f32], n: u32) {
        unsafe {
            let gid = thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x();
            if gid >= n {
                return;
            }
            let i = gid as usize;
            let g = gate[i];
            let u = up[i];
            // mirror hand-PTX: scaled = -g * LOG2_E ; exp_neg = exp2(scaled)
            let log2_e = std::f32::consts::LOG2_E;
            let scaled = (-g) * log2_e;
            let exp_neg = scaled.exp2();
            let sig = 1.0f32 / (1.0f32 + exp_neg);
            let r = g * sig * u;
            let op = &mut *(out.as_ptr().add(i) as *mut f32);
            *op = r;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference — f64 accumulation of the same SwiGLU activation math.
// out[i] = (g/(1+exp(-g))) * up[i]
// ---------------------------------------------------------------------------
fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; gate.len()];
    for i in 0..gate.len() {
        let g = gate[i] as f64;
        let u = up[i] as f64;
        let sig = 1.0f64 / (1.0f64 + (-g).exp());
        out[i] = (g * sig * u) as f32;
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

// Deterministic pseudo-random inputs (same generator family as PMAT-882/893).
// gate is widened to ~[-4, 4] so the sigmoid is exercised across its full range
// (deep negative => silu ~ 0, large positive => silu ~ x); up is ~[-2, 2].
fn make_inputs(n: usize, seed: usize) -> (Vec<f32>, Vec<f32>) {
    let g = |i: usize| -> f32 {
        let x = (i.wrapping_mul(2654435761).wrapping_add(seed.wrapping_mul(40503))) & 0xFFFF;
        (x as f32 / 32768.0) - 1.0 // ~[-1, 1)
    };
    let gate: Vec<f32> = (0..n).map(|i| 4.0 * g(i)).collect(); // ~[-4, 4)
    let up: Vec<f32> = (0..n).map(|i| 2.0 * g(i + 31337)).collect(); // ~[-2, 2)
    (gate, up)
}

struct OxideResult {
    cos: f32,
    maxdiff: f32,
    us: f64,
    n_neg: usize, // number of negative gate values actually exercised
}

#[allow(clippy::too_many_arguments)]
fn run_oxide(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    n: usize,
    seed: usize,
    do_perf: bool,
    ex2_variant: bool, // false = (A) libdev exp, true = (B) ex2
) -> OxideResult {
    let stream = ctx.default_stream();
    let (gate, up) = make_inputs(n, seed);
    let n_neg = gate.iter().filter(|&&x| x < 0.0).count();

    let dg = DeviceBuffer::from_host(&stream, &gate).unwrap();
    let du = DeviceBuffer::from_host(&stream, &up).unwrap();
    let dout = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

    let blocks = n.div_ceil(TPB);
    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (TPB as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let launch = || {
        if ex2_variant {
            module
                .swiglu_ex2(&stream, cfg, &dg, &du, &dout, n as u32)
                .expect("swiglu_ex2");
        } else {
            module
                .swiglu_libdev(&stream, cfg, &dg, &du, &dout, n as u32)
                .expect("swiglu_libdev");
        }
    };

    launch();
    let got = dout.to_host_vec(&stream).unwrap();
    let want = cpu_swiglu(&gate, &up);
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
        n_neg,
    }
}

// ---------------------------------------------------------------------------
// TRUE hand-PTX A/B: load the emitted `fused_swiglu` PTX (FusedSwigluKernel,
// ABI = (gate_ptr,up_ptr,output_ptr: u64, n: u32), flat 1-D grid) and launch it
// on the SAME data with the SAME grid/block (ceil(n/TPB) x TPB) — a fully matched
// single launch each side. GPU-event timing, median of 5x100.
// ---------------------------------------------------------------------------
fn run_handptx(
    ctx: &std::sync::Arc<CudaContext>,
    ptx_path: &str,
    n: usize,
    seed: usize,
) -> Option<(f32, f32, f64)> {
    let stream = ctx.default_stream();
    let (gate, up) = make_inputs(n, seed);

    let ptx = std::fs::read_to_string(ptx_path).ok()?;
    let module = ctx.load_module_from_ptx_src(&ptx).ok()?;
    let func = module.load_function("fused_swiglu").ok()?;

    let dg = DeviceBuffer::from_host(&stream, &gate).ok()?;
    let du = DeviceBuffer::from_host(&stream, &up).ok()?;
    let dout = DeviceBuffer::<f32>::zeroed(&stream, n).ok()?;

    let mut g_ptr = dg.cu_deviceptr();
    let mut u_ptr = du.cu_deviceptr();
    let mut o_ptr = dout.cu_deviceptr();
    let mut n_u32 = n as u32;
    let blocks = n.div_ceil(TPB);
    let grid = (blocks as u32, 1u32, 1u32);
    let block = (TPB as u32, 1u32, 1u32);

    let mut do_launch = || unsafe {
        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut g_ptr as *mut _ as *mut std::ffi::c_void,
            &mut u_ptr as *mut _ as *mut std::ffi::c_void,
            &mut o_ptr as *mut _ as *mut std::ffi::c_void,
            &mut n_u32 as *mut _ as *mut std::ffi::c_void,
        ];
        cuda_core::launch_kernel_on_stream(&func, grid, block, 0, &stream, &mut params)
    };

    do_launch().ok()?;
    let got = dout.to_host_vec(&stream).ok()?;
    let want = cpu_swiglu(&gate, &up);
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

    println!("== PMAT-894 cuda-oxide SwiGLU/SiLU activation — parity + perf (GB10 sm_121) ==");
    println!("   out[i] = silu(gate[i]) * up[i] ; (A) libdev exp  (B) ex2 (hand-PTX mirror)");
    println!("   FFN widths: gate in ~[-4,4] (negatives exercise sigmoid), up in ~[-2,2]");

    // FFN intermediate widths: Llama-7B-ish 11008, Mistral 14336, plus 4096.
    let ns = [4096usize, 11008, 14336];

    // PARITY for both oxide variants across all sizes.
    for (variant, name) in [(false, "libdev (A)"), (true, "ex2 (B)")] {
        let mut all_pass = true;
        println!("\n-- PARITY {name} vs f64 CPU reference --");
        for &n in &ns {
            let r = run_oxide(&ctx, &module, n, 4242 + n, false, variant);
            let pass = r.cos >= 0.9999 && r.maxdiff < 1e-4;
            if !pass {
                all_pass = false;
            }
            println!(
                "  n={:>6} (neg gate={:>5}) : cos={:.7} maxdiff={:.3e} -> {}",
                n,
                r.n_neg,
                r.cos,
                r.maxdiff,
                if pass { "PASS" } else { "FAIL" }
            );
        }
        println!("PARITY {name}: {}", if all_pass { "PASS" } else { "FAIL" });
        if !all_pass {
            eprintln!("PMAT-894 PARITY FAILED ({name})");
            std::process::exit(1);
        }
    }

    let find_ptx = |names: &[&str]| -> Option<String> {
        for nm in names {
            if std::path::Path::new(nm).exists() {
                return Some((*nm).to_string());
            }
        }
        None
    };
    let ptx = find_ptx(&[
        "/tmp/swiglu_spike/fused_swiglu.ptx",
        "baseline-ptx/fused_swiglu.sm121.ptx",
    ]);

    // =========================================================================
    // PRIMARY GATE — FAIR matched single-launch A/B (F-OXIDE-SWIGLU-PARITY-002).
    //
    // METHODOLOGY: the hand-PTX `fused_swiglu` is a flat 1-D grid kernel that
    // processes the WHOLE length n in ONE launch (ceil(n/TPB) blocks x TPB). The
    // oxide kernel uses the IDENTICAL grid/block and one launch. So both sides
    // issue exactly ONE matched launch over the same n — there is NO launch-count
    // confound (unlike the PMAT-893 RMSNorm single-row ABI). The ratio is a true
    // per-kernel speed comparison.
    // =========================================================================
    println!("\n-- PRIMARY GATE: FAIR matched single-launch A/B (oxide vs hand-PTX, GB10 sm_121) --");
    println!("   grid=ceil(n/{TPB}) x {TPB} BOTH sides, one launch each; GPU-event median of 5x100");
    println!("   F-OXIDE-SWIGLU-PARITY-002: cos>=0.9999 AND maxdiff<1e-4 AND oxide_us/handptx_us<=1.2");
    println!(
        "\n  {:>6} | {:>9} {:>9} | {:>11} | {:>8} {:>8} | {:>10} {:>9} | verdict",
        "n", "oxA us", "oxB us", "handPTX us", "ratioA", "ratioB", "cos(ox)", "maxdiff"
    );
    let mut overall_go = true;
    for &n in &ns {
        let ra = run_oxide(&ctx, &module, n, 77, true, false); // (A) libdev
        let rb = run_oxide(&ctx, &module, n, 77, true, true); // (B) ex2
        // best (fastest) parity-passing oxide variant is the GO candidate.
        let (best_us, best_cos, best_md, best_name) = if ra.us <= rb.us {
            (ra.us, ra.cos, ra.maxdiff, "A")
        } else {
            (rb.us, rb.cos, rb.maxdiff, "B")
        };
        match ptx.as_deref().and_then(|p| run_handptx(&ctx, p, n, 77)) {
            Some((hcos, hmd, hus)) => {
                let best_ratio = best_us / hus;
                let parity_ok = best_cos >= 0.9999 && best_md < 1e-4;
                let go = parity_ok && best_ratio <= 1.2;
                if !go {
                    overall_go = false;
                }
                println!(
                    "  {:>6} | {:>9.3} {:>9.3} | {:>11.3} | {:>8.3} {:>8.3} | {:>10.7} {:>9.2e} | {} (best={} {:.3}x; handPTX cos={:.5} md={:.2e})",
                    n, ra.us, rb.us, hus, ra.us / hus, rb.us / hus, best_cos, best_md,
                    if go { "GO" } else { "NO-GO" }, best_name, best_ratio, hcos, hmd
                );
            }
            None => {
                overall_go = false;
                println!("  {n:>6} | hand-PTX MISSING/launch-FAILED (regen baseline-ptx/fused_swiglu.sm121.ptx)");
            }
        }
    }

    println!(
        "\nPMAT-894 VERDICT (F-OXIDE-SWIGLU-PARITY-002, fair matched-launch gate): {}",
        if overall_go { "GO" } else { "NO-GO" }
    );
    println!("PMAT-894 DONE");
}
