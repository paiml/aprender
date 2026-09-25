// #3522 OXIDE-001 — pure-Rust cuda-oxide port of the row-batched GDN partial NEOX RoPE.
//
// Target = hand-PTX `PartialNeoxRopeRowsKernel`
// (crates/aprender-gpu/src/kernels/gdn/rows.rs, entry `gdn_partial_neox_rope_rows`):
// grid (heads, rows), block half = n_rot/2, one thread per (row, head, pair); head_dim,
// half and row_stride baked into the PTX; params (x, pos0, theta_scale), in place.
// It is `PartialNeoxRopeKernel` (row 9, ../partial-rope) with row t at position
// pos0 + t:
//
//   theta_j = (pos0 + t) * theta_scale^j        (j multiplications, the CPU's order)
//   (a, b) = (x[t*stride + h*head_dim + j], x[t*stride + h*head_dim + j + half])
//   a' = a*cos(theta_j) - b*sin(theta_j),  b' = a*sin(theta_j) + b*cos(theta_j)
//
// Dimensions n_rot..head_dim of each head are NOT rotated. Timed at the shipped
// device test's shape: Qwen3.5-9B's full-attention q, 16 heads x 256, n_rot 64, with
// the rows contiguous (row_stride = heads * head_dim).
//
// CPU reference = `apply_partial_neox_rope` (forward_qwen35.rs:191) per row, at that
// row's position, with theta built in f32 exactly as the CPU builds it, then sin/cos
// and the rotation in f64.
//
// SAFETY SHAPE (kernel-safety, #3522): both device kernels are safe Rust — no
// `unsafe` block, no raw pointer. Reads go through bounds-checked slice indexing.
// The write goes through `DisjointSlice<f32, RuntimeRowMajorTiles<8, 1>>` over the
// buffer viewed as rows of width HALF (bound on the host with `RowWidth`). A head is
// head_dim / HALF = 8 such rows, so tile-row g covers exactly head g (g = row*HEADS +
// h, rows contiguous): one column of one whole head per thread. The thread writes
// local rows 0 and 1 of its tile — the pair (j, j + half) — and leaves the six tail
// rows alone. The tile is disjoint from every other thread's, and `tile_2d32_rt`
// checks it stays inside the row and the slice.
//
// Unlike row 9's 2x1 pair tiles, a head-tall tile has no idle blocks: every tile-row
// is a head that needs rotating, so the grid is exactly (1, rows*heads).
//
// OUT OF PLACE, as in the other ports: a safe kernel cannot read `x` as `&[f32]` and
// write it through a `DisjointSlice` over the same memory. `out` starts as a copy of
// `x`; the harness checks the unrotated tails bit for bit.
//
// Two variants, differing only in sin/cos:
//   (A) partial_rope_rows_approx — the hand PTX's Cody-Waite reduction with a two-word
//       2*pi, then `sin.approx` / `cos.approx`
//   (B) partial_rope_rows_libm   — `f32::sin_cos`, libdevice's full-range sincosf
//
// Output: a human table on stdout plus one `apr-kernel-receipt/v1` JSON line
// prefixed `RECEIPT ` per variant; `receipt.sh` adds host facts and writes
// evidence/kernels/gdn_partial_rope_rows/<host>.json.

use cuda_core::{CudaContext, DeviceBuffer, IntoResult, LaunchConfig2D, sys};
use cuda_device::{
    DisjointSlice, LocalIndex32, RuntimeRowMajorTiles, cuda_module, float, kernel, launch_bounds,
    launch_contract, thread,
};
use cuda_host::RowWidth;
use std::sync::Arc;

const HEAD_DIM: usize = 256;
const N_ROT: usize = 64;
const HALF: usize = N_ROT / 2;
/// Heads per row; the rows are contiguous, so row_stride = HEADS * HEAD_DIM.
const HEADS: u32 = 16;
/// Rows of width HALF in one head: the tile height, `RuntimeRowMajorTiles<8, 1>`.
const TILE_ROWS: usize = HEAD_DIM / HALF;
const _: () = assert!(TILE_ROWS == 8);
/// High word of `2*pi`, exact in 8 significant bits (the hand kernel's constant).
const TWO_PI_HI: f32 = 6.281_25;
/// `2*pi - TWO_PI_HI`, rounded to f32.
const TWO_PI_LO: f32 = 0.001_935_307_2;

const PARITY_COS: f64 = 0.9999;
const PARITY_MAXDIFF: f32 = 1e-3;

const TIMING_RATIO_MAX: f64 = 1.2;

/// (rows, pos0, freq_base). Row 0 at position 0 is the identity; 20000+ drives theta
/// far outside [-pi, pi], where an unreduced `sin.approx` would fail. (11, 20000,
/// 1e7) is the shipped device test's case.
const PARITY_CASES: [(usize, u32, f32); 5] = [
    (1, 0, 10_000.0),
    (3, 1, 10_000.0),
    (11, 20_000, 10_000_000.0),
    (64, 17, 1_000_000.0),
    (512, 30_000, 1_000_000.0),
];

/// Prefill chunk sizes.
const TIMED_ROWS: [usize; 3] = [16, 128, 512];
const TIMED_POS: u32 = 1000;
const TIMED_FREQ_BASE: f32 = 1_000_000.0;

/// The kernel's `theta_scale` argument: the CPU's expression character for character.
fn theta_scale(freq_base: f32) -> f32 {
    freq_base.powf(-2.0 / N_ROT as f32)
}

#[cuda_module]
mod kernels {
    use super::*;

    /// theta_j = pos * theta_scale^j by j multiplications, in the CPU's order.
    #[inline(always)]
    fn theta_j(position: u32, theta_scale: f32, j: u32) -> f32 {
        let mut theta = position as f32;
        for _ in 0..j {
            theta *= theta_scale;
        }
        theta
    }

    /// Tile-row g is head g % HEADS of token row g / HEADS.
    #[inline(always)]
    fn position(pos0: u32, tile_row: u32) -> u32 {
        pos0 + tile_row / HEADS
    }

    /// The rotation, as the CPU's two expressions (mul/mul/sub, mul/mul/add; no fma),
    /// written to local rows 0 and 1 of this thread's head-tall tile.
    #[inline(always)]
    fn rotate(
        x: &[f32],
        out: &mut DisjointSlice<f32, RuntimeRowMajorTiles<8, 1>>,
        c: thread::ThreadCoord2D32,
        sin: f32,
        cos: f32,
    ) {
        let w = out.row_width() as usize;
        let ia = c.row() as usize * TILE_ROWS * w + c.col() as usize;
        let (a, b) = (x[ia], x[ia + w]);
        let rot_a = a * cos - b * sin;
        let rot_b = a * sin + b * cos;
        let Some(mut o) = out.tile_2d32_rt(c) else {
            return;
        };
        o.at(LocalIndex32::constant::<0>(), LocalIndex32::constant::<0>())
            .write(rot_a);
        o.at(LocalIndex32::constant::<1>(), LocalIndex32::constant::<0>())
            .write(rot_b);
    }

    /// (A) Cody-Waite reduction into [-pi, pi], then the hardware approximations.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 2, coordinates = u32, block = (32, 1, 1))]
    pub fn partial_rope_rows_approx(
        x: &[f32],
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<8, 1>>,
        pos0: u32,
        theta_scale: f32,
    ) {
        let c = thread::coord_2d_u32(launch_context);
        let theta = theta_j(position(pos0, c.row()), theta_scale, c.col());
        let n = (theta * (1.0 / core::f32::consts::TAU) + 0.5).floor();
        let reduced = n.mul_add(-TWO_PI_LO, n.mul_add(-TWO_PI_HI, theta));
        let sin = float::sin_approx_f32(reduced);
        let cos = float::cos_approx_f32(reduced);
        rotate(x, &mut out, c, sin, cos);
    }

    /// (B) libdevice sincosf: full-range, no reduction needed.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(32)]
    #[launch_contract(domain = 2, coordinates = u32, block = (32, 1, 1))]
    pub fn partial_rope_rows_libm(
        x: &[f32],
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<8, 1>>,
        pos0: u32,
        theta_scale: f32,
    ) {
        let c = thread::coord_2d_u32(launch_context);
        let (sin, cos) = theta_j(position(pos0, c.row()), theta_scale, c.col()).sin_cos();
        rotate(x, &mut out, c, sin, cos);
    }
}

/// `apply_partial_neox_rope` per row at position pos0 + row, with theta in f32 (the
/// CPU's), the rest in f64.
fn cpu_partial_rope_rows(x: &[f32], rows: usize, pos0: u32, freq_base: f32) -> Vec<f32> {
    let ts = theta_scale(freq_base);
    let mut out = x.to_vec();
    for g in 0..rows * HEADS as usize {
        let base = g * HEAD_DIM;
        let mut theta = (pos0 + (g / HEADS as usize) as u32) as f32;
        for j in 0..HALF {
            let (sin, cos) = f64::from(theta).sin_cos();
            let (a, b) = (f64::from(x[base + j]), f64::from(x[base + j + HALF]));
            out[base + j] = (a * cos - b * sin) as f32;
            out[base + j + HALF] = (a * sin + b * cos) as f32;
            theta *= ts;
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
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn make_inputs(n_heads: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n_heads * HEAD_DIM)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Every unrotated dimension (n_rot..head_dim of each head) is bit-identical to the input.
fn tail_intact(got: &[f32], x: &[f32]) -> bool {
    got.iter()
        .zip(x)
        .enumerate()
        .all(|(i, (g, v))| i % HEAD_DIM < N_ROT || g.to_bits() == v.to_bits())
}

struct Measured {
    cos: f64,
    maxdiff: f32,
    tail_ok: bool,
    /// Device time per launch, from a CUDA-graph replay (the gated number).
    us: f64,
    /// Eager per-launch time: recorded, never gated (see GDN-DECISIONS.md).
    eager_us: f64,
}

impl Measured {
    fn ok(&self) -> bool {
        self.cos >= PARITY_COS && self.maxdiff < PARITY_MAXDIFF && self.tail_ok
    }
}

/// GPU-event median of 5 x 100 warm eager launches, in microseconds per launch.
fn time_eager_us(stream: &Arc<cuda_core::CudaStream>, mut launch: impl FnMut()) -> f64 {
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

/// GPU-event median of 5 replays of a CUDA graph holding 100 captured launches,
/// in microseconds per launch. The graph takes host submission out of the number.
/// Needs a real (non-legacy) stream: capture refuses the null one.
fn time_graph_us(stream: &Arc<cuda_core::CudaStream>, mut launch: impl FnMut()) -> f64 {
    const NODES: u32 = 100;
    let s = stream.cu_stream();
    assert!(
        !s.is_null(),
        "graph capture needs a created stream, not the legacy default"
    );
    // SAFETY: `s` is a live stream owned by `stream`; the graph and its executable
    // are created, launched on `s` and destroyed inside this function, and the
    // captured launches reference buffers the caller keeps alive across the call.
    let exec = unsafe {
        sys::cuStreamBeginCapture_v2(
            s,
            sys::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .result()
        .expect("begin capture");
        for _ in 0..NODES {
            launch();
        }
        let mut graph = std::ptr::null_mut();
        sys::cuStreamEndCapture(s, &mut graph)
            .result()
            .expect("end capture");
        let mut exec = std::ptr::null_mut();
        sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0)
            .result()
            .expect("instantiate");
        sys::cuGraphDestroy(graph).result().expect("destroy graph");
        exec
    };
    // SAFETY: `exec` is the executable instantiated above, `s` its stream.
    let replay = || {
        unsafe { sys::cuGraphLaunch(exec, s) }
            .result()
            .expect("graph launch")
    };
    for _ in 0..3 {
        replay();
    }
    stream.synchronize().expect("warmup sync");
    let flags = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT);
    let mut times: Vec<f64> = (0..5)
        .map(|_| {
            let start = stream.record_event(flags).expect("event");
            replay();
            let end = stream.record_event(flags).expect("event");
            f64::from(start.elapsed_ms(&end).expect("elapsed")) * 1000.0 / f64::from(NODES)
        })
        .collect();
    // SAFETY: nothing else holds `exec`; its replays completed in elapsed_ms.
    unsafe { sys::cuGraphExecDestroy(exec) }
        .result()
        .expect("destroy exec");
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn measure(
    got: &[f32],
    x: &[f32],
    (rows, pos0, freq_base): (usize, u32, f32),
    us: f64,
    eager_us: f64,
) -> Measured {
    let want = cpu_partial_rope_rows(x, rows, pos0, freq_base);
    Measured {
        cos: cosine(got, &want),
        maxdiff: max_abs_diff(got, &want),
        tail_ok: tail_intact(got, x),
        us,
        eager_us,
    }
}

fn seed((rows, pos0, _): (usize, u32, f32)) -> u64 {
    0x3522_000D + (rows as u64) * 65_536 + u64::from(pos0)
}

fn run_oxide(
    ctx: &Arc<CudaContext>,
    module: &kernels::LoadedModule,
    case: (usize, u32, f32),
    libm: bool,
    perf: bool,
) -> Measured {
    let (rows, pos0, freq_base) = case;
    let ts = theta_scale(freq_base);
    let stream = ctx.new_stream().expect("stream");
    let x = make_inputs(rows * HEADS as usize, seed(case));
    let d_x = DeviceBuffer::from_host(&stream, &x).expect("x");
    // `out` starts as x: the unrotated tails the kernel must not touch are already right.
    let mut d_out = DeviceBuffer::from_host(&stream, &x).expect("out");

    let config = LaunchConfig2D::new((1, rows as u32 * HEADS), (HALF as u32, 1), 0);
    let launch = |d_out: &mut DeviceBuffer<f32>| {
        let out = RowWidth::new(d_out, HALF as u32);
        if libm {
            let p = module
                .prepare_partial_rope_rows_libm(config)
                .expect("prepare libm");
            module
                .partial_rope_rows_libm(&stream, &p, &d_x, out, pos0, ts)
                .expect("launch libm");
        } else {
            let p = module
                .prepare_partial_rope_rows_approx(config)
                .expect("prepare approx");
            module
                .partial_rope_rows_approx(&stream, &p, &d_x, out, pos0, ts)
                .expect("launch approx");
        }
    };
    launch(&mut d_out);
    let got = d_out.to_host_vec(&stream).expect("download");
    let (us, eager_us) = if perf {
        (
            time_graph_us(&stream, || launch(&mut d_out)),
            time_eager_us(&stream, || launch(&mut d_out)),
        )
    } else {
        (0.0, 0.0)
    };
    measure(&got, &x, case, us, eager_us)
}

/// The hand PTX, loaded from its committed baseline and launched on the same data:
/// grid (HEADS, rows), block HALF, params (x, pos0, theta_scale), in place.
fn run_handptx(ctx: &Arc<CudaContext>, sm: &str, case: (usize, u32, f32)) -> (Measured, u32) {
    let (rows, pos0, freq_base) = case;
    let ptx_path = format!("baseline-ptx/gdn_partial_neox_rope_rows.{sm}.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap_or_else(|e| {
        eprintln!("missing hand-PTX baseline {ptx_path}: {e}");
        eprintln!("regenerate: APR_BLESS_PTX=1 cargo test -p aprender-gpu --features cuda --lib gdn_partial_rope_rows_ptx_golden");
        std::process::exit(2);
    });
    let stream = ctx.new_stream().expect("stream");
    let x = make_inputs(rows * HEADS as usize, seed(case));
    let module = ctx.load_module_from_ptx_src(&ptx).expect("load hand PTX");
    let func = module
        .load_function("gdn_partial_neox_rope_rows")
        .expect("gdn_partial_neox_rope_rows");
    let regs = func.num_registers().expect("regs");
    let d_x = DeviceBuffer::from_host(&stream, &x).expect("x");
    let mut x_ptr = d_x.cu_deviceptr();
    let mut p0 = pos0;
    let mut ts = theta_scale(freq_base);
    let mut launch = || {
        let mut params: [*mut std::ffi::c_void; 3] = [
            (&raw mut x_ptr).cast(),
            (&raw mut p0).cast(),
            (&raw mut ts).cast(),
        ];
        // SAFETY: the three params match the entry's (.u64, .u32, .f32) in order;
        // the buffer holds rows * HEADS * HEAD_DIM f32 with row_stride HEADS *
        // HEAD_DIM, the stride and head width this PTX was baked with, and a
        // (HEADS, rows) x HALF launch touches only the rotated prefixes.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &func,
                (HEADS, rows as u32, 1),
                (HALF as u32, 1, 1),
                0,
                &stream,
                &mut params,
            )
        }
        .expect("hand PTX launch");
    };
    // Parity from the first launch; timing then rotates in place again, which
    // does the same work on already-rotated values.
    launch();
    let got = d_x.to_host_vec(&stream).expect("download");
    let us = time_graph_us(&stream, &mut launch);
    let eager_us = time_eager_us(&stream, &mut launch);
    (measure(&got, &x, case, us, eager_us), regs)
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

    println!("== #3522 cuda-oxide partial NEOX RoPE rows ({sm}) ==");
    println!(
        "   head_dim={HEAD_DIM} n_rot={N_ROT} heads={HEADS} 1 thread/(row, head, pair), timed at pos0={TIMED_POS} freq_base={TIMED_FREQ_BASE:e}"
    );

    let mut all_ok = true;
    // A timing NO-GO is its own exit code (4), not a pass: the receipts still
    // print, so receipt.sh records `"pass":false` and then fails.
    let mut timing_all_ok = true;
    for libm in [false, true] {
        let name = if libm { "libm (B)" } else { "approx (A)" };
        for case in PARITY_CASES {
            let r = run_oxide(&ctx, &module, case, libm, false);
            let ok = r.ok();
            all_ok &= ok;
            println!(
                "  parity {name} rows={} pos0={} freq_base={:e}: cos={:.7} maxdiff={:.3e} tail={} {}",
                case.0,
                case.1,
                case.2,
                r.cos,
                r.maxdiff,
                if r.tail_ok { "intact" } else { "CLOBBERED" },
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }

    let regs_hand;
    {
        let mut worst = (1.0f64, 0.0f32);
        let mut regs = 0;
        for case in PARITY_CASES {
            let (h, r) = run_handptx(&ctx, &sm, case);
            regs = r;
            all_ok &= h.ok();
            worst = (worst.0.min(h.cos), worst.1.max(h.maxdiff));
        }
        regs_hand = regs;
        println!(
            "  hand PTX over the parity cases: cos={:.7} maxdiff={:.3e} regs={regs}",
            worst.0, worst.1
        );
    }
    println!("\n   rows | variant | oxide us | handPTX us | ratio | verdict | eager oxide/hand");
    for libm in [false, true] {
        let (variant, entry) = if libm {
            ("libm", "partial_rope_rows_libm")
        } else {
            ("approx", "partial_rope_rows_approx")
        };
        let regs = oxide_registers(&ctx, entry);
        let mut worst_ratio = 0.0f64;
        let mut worst = (0.0f64, 0.0f64, 0usize);
        let mut worst_eager = 0.0f64;
        let mut parity = (1.0f64, 0.0f32, true);
        for n in TIMED_ROWS {
            let case = (n, TIMED_POS, TIMED_FREQ_BASE);
            // Three rounds, alternating which kernel is timed first; the round with
            // the median ratio is the row. A host clock step mid-run (yoga, see
            // GDN-DECISIONS.md, causal conv1d seq) then lands in at most one round.
            let mut rounds: Vec<(Measured, Measured)> = (0..3)
                .map(|round| {
                    if round % 2 == 0 {
                        let o = run_oxide(&ctx, &module, case, libm, true);
                        (o, run_handptx(&ctx, &sm, case).0)
                    } else {
                        let h = run_handptx(&ctx, &sm, case).0;
                        (run_oxide(&ctx, &module, case, libm, true), h)
                    }
                })
                .collect();
            rounds.sort_by(|a, b| (a.0.us / a.1.us).total_cmp(&(b.0.us / b.1.us)));
            let (o, h) = rounds.swap_remove(1);
            let ratio = o.us / h.us;
            let ok = ratio <= TIMING_RATIO_MAX;
            let eager_ratio = o.eager_us / h.eager_us;
            println!(
                "  {n:>5} | {variant:>7} | {:>8.3} | {:>10.3} | {ratio:.3} | {:>7} | {:.3}/{:.3} = {eager_ratio:.3}",
                o.us,
                h.us,
                if ok { "GO" } else { "NO-GO" },
                o.eager_us,
                h.eager_us,
            );
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst = (o.us, h.us, n);
            }
            worst_eager = worst_eager.max(eager_ratio);
            parity = (
                parity.0.min(o.cos),
                parity.1.max(o.maxdiff),
                parity.2 && o.tail_ok,
            );
        }
        let timing_ok = worst_ratio <= TIMING_RATIO_MAX;
        timing_all_ok &= timing_ok;
        let parity_ok = parity.0 >= PARITY_COS && parity.1 < PARITY_MAXDIFF && parity.2;
        // One line per variant; receipt.sh adds host, sha, ptxas and writes the file.
        println!(
            "RECEIPT {{\"schema\":\"apr-kernel-receipt/v1\",\"kernel\":\"gdn_partial_rope_rows\",\"variant\":\"{variant}\",\"entry\":\"{entry}\",\"authoring\":\"oxide\",\"cc\":\"{sm}\",\"head_dim\":{HEAD_DIM},\"n_rot\":{N_ROT},\"heads\":{HEADS},\"pos0\":{TIMED_POS},\"freq_base\":{TIMED_FREQ_BASE:e},\"parity\":{{\"cos_min\":{:.9},\"maxdiff_max\":{:.3e},\"tail_intact\":{},\"cos_floor\":{PARITY_COS},\"maxdiff_ceiling\":{PARITY_MAXDIFF:e},\"pass\":{parity_ok}}},\"timing\":{{\"oxide_us\":{:.3},\"handptx_us\":{:.3},\"ratio\":{worst_ratio:.4},\"worst_rows\":{},\"ratio_max\":{TIMING_RATIO_MAX},\"method\":\"cuda-graph-100\",\"eager_ratio_max\":{worst_eager:.4},\"pass\":{timing_ok}}},\"register_budget\":{{\"oxide\":{},\"handptx\":{regs_hand}}}}}",
            parity.0,
            parity.1,
            parity.2,
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
    println!("#3522 PARTIAL ROPE ROWS DONE");
}
