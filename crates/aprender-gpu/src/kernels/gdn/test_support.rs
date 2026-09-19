//! Shared fixtures for the Gated `DeltaNet` device tests (PMAT-3477).
//!
//! Deterministic inputs (a seeded LCG — no `rand` dependency), the real
//! Qwen3.5-0.8B shapes, and one launch helper so each kernel's test is only its
//! host reference plus its assertion.

#![cfg(test)]

/// `head_k_dim` / `head_v_dim` for Qwen3.5-0.8B (`qwen2.ssm.state_size`).
pub(crate) const HEAD_DIM: usize = 128;
/// `num_k_heads` for Qwen3.5-0.8B (`qwen2.ssm.group_count`).
pub(crate) const NUM_K_HEADS: usize = 16;
/// `num_v_heads` for Qwen3.5-0.8B (`qwen2.ssm.time_step_rank`).
pub(crate) const NUM_V_HEADS: usize = 16;
/// `conv_kernel` for Qwen3.5-0.8B (`qwen2.ssm.conv_kernel`).
pub(crate) const CONV_KERNEL: usize = 4;
/// `conv_dim` as `forward_deltanet` computes it: `k_dim * 2 + v_dim` = 6144.
pub(crate) const CONV_CHANNELS: usize = HEAD_DIM * NUM_K_HEADS * 2 + HEAD_DIM * NUM_V_HEADS;
/// `config.eps` of the 0.8B model.
pub(crate) const EPS: f32 = 1e-6;

/// A seeded LCG producing values in roughly `[-1, 1)`.
pub(crate) struct Lcg(u32);

impl Lcg {
    pub(crate) const fn new(seed: u32) -> Self {
        Self(seed)
    }

    /// Next value in `[-1, 1)`.
    pub(crate) fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        f32::from(((self.0 >> 16) & 0xFFFF) as u16) / 32768.0 - 1.0
    }

    /// Next value in `[-scale, scale)`.
    pub(crate) fn next_scaled(&mut self, scale: f32) -> f32 {
        self.next() * scale
    }

    /// `n` values in `[-scale, scale)`.
    pub(crate) fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_scaled(scale)).collect()
    }
}

#[cfg(feature = "cuda")]
mod device {
    use crate::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig};
    use crate::kernels::Kernel;

    /// Compile and launch `kernel` with raw device pointers as arguments.
    ///
    /// `sm_70` PTX is used deliberately: everything these kernels emit
    /// (`shfl.sync`, `ex2.approx`, `lg2.approx`) is Volta-era, so the same module
    /// JITs on every supported device.
    pub(crate) fn run_kernel<K: Kernel>(
        ctx: &CudaContext,
        stream: &CudaStream,
        kernel: &K,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &mut [u64],
    ) {
        let ptx = kernel.emit_ptx();
        let mut module = CudaModule::from_ptx(ctx, &ptx).expect("module from ptx");
        let config = LaunchConfig {
            grid,
            block,
            shared_mem: 0,
        };
        let mut raw: Vec<*mut std::ffi::c_void> = args
            .iter_mut()
            .map(|a| std::ptr::from_mut(a).cast())
            .collect();
        // SAFETY: every entry of `args` is a live device allocation sized by the
        // caller to the extent the kernel indexes, and grid/block are the kernel's
        // documented launch shape.
        unsafe {
            stream
                .launch_kernel(&mut module, kernel.name(), &config, &mut raw)
                .expect("kernel launch");
        }
        stream.synchronize().expect("stream synchronize");
    }
}

#[cfg(feature = "cuda")]
pub(crate) use device::run_kernel;

/// Assert two buffers agree, at `tol` **relative to the reference's own scale**.
///
/// The limit is `tol * max|want|`, not `tol` — an absolute tolerance is vacuous here.
/// The delta-rule outputs land near 5e-3 (`q`/`k` are unit-norm and the recurrence
/// scales by `D^-0.5`), so an absolute 1e-3 accepted a reference mutated by 1%:
/// measured, not assumed. A reference that is all ~zero fails outright, because then
/// nothing about the kernel has been checked.
pub(crate) fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length mismatch");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        scale > 1e-4,
        "{what}: the CPU reference is all ~zero (max |cpu| = {scale}); this comparison \
         would pass however wrong the kernel is"
    );
    let limit = tol * scale;
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst <= limit,
        "{what}: worst |gpu - cpu| = {worst} at index {worst_i} \
         (gpu {gpu}, cpu {cpu}), limit {limit} = {tol} * max|cpu| {scale}. \
         The CPU side here is a verbatim \
         port of aprender-serve's forward_qwen35.rs, which is the specification for \
         the Gated DeltaNet block (aprender#3090).",
        gpu = got[worst_i],
        cpu = want[worst_i],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gdn_lcg_is_deterministic_and_bounded() {
        let a = Lcg::new(7).vec(64, 1.0);
        let b = Lcg::new(7).vec(64, 1.0);
        assert_eq!(a, b, "the fixture must be reproducible");
        assert!(a.iter().all(|v| (-1.0..1.0).contains(v)), "{a:?}");
        // Not a constant stream — a fixture of one repeated value would make most
        // of these kernels' index arithmetic unfalsifiable.
        assert!(a.windows(2).any(|w| w[0] != w[1]));
    }

    #[test]
    fn gdn_shapes_match_qwen35_0_8b() {
        // group_count * state_size == inner_size, and conv_dim == 3 * inner_size.
        assert_eq!(NUM_K_HEADS * HEAD_DIM, 2048);
        assert_eq!(CONV_CHANNELS, 3 * 2048);
        assert_eq!(CONV_KERNEL, 4);
        assert_eq!(NUM_V_HEADS, 16);
    }
}
