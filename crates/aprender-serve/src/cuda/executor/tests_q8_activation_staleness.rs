//! #4258: the DP4A GEMVs cache their Q8_1 activation behind `q8_activation_valid`. Before
//! #4258 nothing on the Qwen3.5 path cleared that flag, so every DP4A GEMV after the first
//! reused the FIRST quantized activation — across projections, layers and tokens.
//!
//! Each test runs a DP4A Q4_K GEMV whose input is NOT the activation last quantized, and
//! judges it against the float (Mwv) GEMV of the same input. A stale activation reproduces
//! the previous GEMV's output exactly; a fresh one lands within Q8_1 rounding of the float.

use super::CudaExecutor;
use crate::cuda::gpu_profile::Q4kVariant;
use serial_test::serial;
use trueno_gpu::driver::GpuBuffer;

const N: u32 = 32;
const K: u32 = 256;
/// Q8_1 activation rounding on these inputs is well under 1% relative L2.
const FRESH_TOL: f32 = 0.05;
/// The two inputs are built so their float outputs sit far apart; below this the test
/// could not tell a stale activation from a fresh one.
const DISTINCT_FLOOR: f32 = 0.2;

/// `N` rows of one Q4_K super-block each: d = 1.0, dmin = 0, every 6-bit scale 1,
/// pseudo-random nibbles.
fn q4k_weights() -> Vec<u8> {
    let mut seed = 0x2545_f491_u32;
    let mut out = Vec::with_capacity(N as usize * 144);
    for _ in 0..N {
        out.extend_from_slice(&[0x00, 0x3C, 0x00, 0x00]);
        out.extend_from_slice(&[0x01; 12]);
        for _ in 0..128 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            out.push((seed >> 16) as u8);
        }
    }
    out
}

fn input_a() -> Vec<f32> {
    (0..K).map(|i| (i as f32 * 0.37).sin()).collect()
}

fn input_b() -> Vec<f32> {
    (0..K)
        .map(|i| (i as f32 * 0.11).cos() * 2.0 - 0.5)
        .collect()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let num: f32 = got.iter().zip(want).map(|(g, w)| (g - w).powi(2)).sum();
    let den: f32 = want.iter().map(|w| w * w).sum();
    (num / den).sqrt()
}

struct Rig {
    ex: CudaExecutor,
    w: u64,
    out: GpuBuffer<f32>,
}

impl Rig {
    fn new(mut ex: CudaExecutor) -> Self {
        ex.init_workspace(K as usize, K as usize)
            .expect("init_workspace");
        ex.load_quantized_weights("q8_staleness_4258", &q4k_weights())
            .expect("load Q4_K");
        let w = ex
            .get_quantized_weight_ptr("q8_staleness_4258")
            .expect("weight ptr");
        let out = GpuBuffer::new(ex.context(), N as usize).expect("out");
        Self { ex, w, out }
    }

    fn buf(&self, host: &[f32]) -> GpuBuffer<f32> {
        GpuBuffer::from_host(self.ex.context(), host).expect("buffer")
    }

    fn gemv(&mut self, variant: Q4kVariant, input: &GpuBuffer<f32>) -> Vec<f32> {
        self.ex.gpu_profile.q4k = variant;
        self.ex
            .q4k_gemv_into(self.w, input, &self.out, N, K)
            .expect("gemv");
        self.ex.synchronize().expect("sync");
        let mut host = vec![0.0f32; N as usize];
        self.out.copy_to_host(&mut host).expect("copy out");
        host
    }

    /// The float GEMV of whatever `input` holds right now.
    fn float_ref(&mut self, input: &GpuBuffer<f32>) -> Vec<f32> {
        self.ex.synchronize().expect("sync");
        let mut host = vec![0.0f32; K as usize];
        input.copy_to_host(&mut host).expect("copy in");
        let fresh = self.buf(&host);
        self.gemv(Q4kVariant::Mwv, &fresh)
    }
}

const DP4A: [Q4kVariant; 2] = [Q4kVariant::HwDp4a, Q4kVariant::MwvDp4a];

/// Two DP4A GEMVs on two DIFFERENT buffers: the second must quantize its own input.
#[test]
#[serial]
fn a_dp4a_gemv_on_a_different_buffer_quantizes_that_buffer() {
    let ex = crate::cuda_executor_or_skip!(0);
    let mut rig = Rig::new(ex);
    let (a, b) = (rig.buf(&input_a()), rig.buf(&input_b()));
    let (ref_a, ref_b) = (rig.float_ref(&a), rig.float_ref(&b));
    assert!(rel_l2(&ref_a, &ref_b) > DISTINCT_FLOOR, "inputs too alike");

    for variant in DP4A {
        let got_b = rig.gemv(variant, &b);
        let got_a = rig.gemv(variant, &a);
        let (err_b, err_a) = (rel_l2(&got_b, &ref_b), rel_l2(&got_a, &ref_a));
        eprintln!("[4258] {variant:?}: first input {err_b:.3e}, second input {err_a:.3e}");
        assert!(
            err_b < FRESH_TOL,
            "{variant:?}: the first DP4A GEMV is off ({err_b})"
        );
        assert!(
            err_a < FRESH_TOL,
            "{variant:?}: a DP4A GEMV on a second buffer is {err_a} from the float GEMV of \
             that buffer — it reused the first buffer's Q8_1 activation (#4258)"
        );
    }
}

/// One buffer, rewritten in place by each Qwen3.5 decode primitive that feeds a GEMV: the
/// DP4A GEMV after the write must quantize the NEW contents.
#[test]
#[serial]
fn a_dp4a_gemv_after_a_qwen35_writer_rewrites_its_input_requantizes() {
    let ex = crate::cuda_executor_or_skip!(0);
    let mut rig = Rig::new(ex);
    let (a_host, b_host) = (input_a(), input_b());
    let (a, b) = (rig.buf(&a_host), rig.buf(&b_host));
    let ones = rig.buf(&vec![1.0f32; K as usize]);

    type Writer =
        fn(&mut CudaExecutor, &GpuBuffer<f32>, &GpuBuffer<f32>, &GpuBuffer<f32>, &GpuBuffer<f32>);
    let writers: [(&str, Writer); 5] = [
        ("rmsnorm_into", |ex, _a, b, one, x| {
            ex.rmsnorm_into(b, one, x, K, 1e-6).expect("rmsnorm_into");
        }),
        ("per_head_rmsnorm_into", |ex, _a, b, one, x| {
            // `one` is K ones; per-head gamma reads only its first 64.
            ex.per_head_rmsnorm_into(b, one, x, 64, K / 64, 1e-6)
                .expect("per_head_rmsnorm_into");
        }),
        ("residual_add_into", |ex, a, b, _one, x| {
            ex.residual_add_into(a, b, x, K).expect("residual_add_into");
        }),
        ("fused_swiglu_into", |ex, a, b, _one, x| {
            ex.fused_swiglu_into(b, a, x, K).expect("fused_swiglu_into");
        }),
        ("gdn_sigmoid_gate_into", |ex, _a, b, _one, x| {
            ex.gdn_sigmoid_gate_into(x, b, K)
                .expect("gdn_sigmoid_gate_into");
        }),
    ];

    for variant in DP4A {
        for (name, write) in writers {
            let x = rig.buf(&a_host);
            let before = rig.gemv(variant, &x);
            write(&mut rig.ex, &a, &b, &ones, &x);
            let want = rig.float_ref(&x);
            assert!(
                rel_l2(&before, &want) > DISTINCT_FLOOR,
                "{name}: write changed too little"
            );
            let got = rig.gemv(variant, &x);
            let err = rel_l2(&got, &want);
            eprintln!("[4258] {variant:?} after {name}: {err:.3e}");
            assert!(
                err < FRESH_TOL,
                "{variant:?}: the DP4A GEMV after `{name}` rewrote its input is {err} from the \
                 float GEMV of the new contents — it reused the stale Q8_1 activation (#4258)"
            );
        }
    }
}
