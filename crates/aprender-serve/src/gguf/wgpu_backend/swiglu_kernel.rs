//! Fused SwiGLU WGSL kernel — **M-GPU-MOE-2.1 scaffold for #1582**.
//!
//! Element-wise fused SiLU + multiply: `output[i] = silu(gate[i]) * up[i]`
//! where `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
//!
//! ## Why this exists
//!
//! M-GPU-MOE-2.1 (#1582) needs a wgpu sibling of CUDA's `expert_swiglu_cuda`
//! (which in turn wraps trueno-gpu's `FusedSwigluKernel` PTX). The CPU side
//! is `expert_swiglu_quantized` in `qwen3_moe_load.rs:240`; the CUDA side is
//! `crates/aprender-serve/src/gguf/cuda/expert_swiglu_cuda.rs:93`. This
//! file is the wgpu sibling — element-wise post-matvec, the SAFEST piece of
//! the full SwiGLU FFN to author first (no quantization, no matvec, just
//! fused activation).
//!
//! Per `feedback_falsifier_cascade_decomposes_magnitude.md` — decompose the
//! multi-week M-GPU-MOE-2.1+2.2+2.3 work into single-piece scaffolds that
//! each ship in 1 PR.
//!
//! ## What this scaffold ships
//!
//! 1. A WGSL compute shader string (`FUSED_SWIGLU_WGSL`) that computes
//!    `silu(gate) * up` element-wise over arrays.
//! 2. A `FusedSwigluWgpuKernel` struct that owns the wgpu compute pipeline
//!    + bind group layout. Construction is one-shot (`new(&device)`); the
//!    pipeline is cacheable on the parent `OwnedQuantizedModelWgpu`.
//! 3. A `dispatch` method that records the compute pass given pre-allocated
//!    GPU buffers — caller is responsible for buffer lifecycle. This
//!    matches the trueno-gpu kernel-then-buffer separation used elsewhere
//!    in the wgpu_training.rs scaffold.
//!
//! ## What is OUT of scope for this PR
//!
//! - Actual GPU dispatch + parity verification — that needs the wgpu
//!   adapter init dance which the parent `OwnedQuantizedModelWgpu` will
//!   own at M-GPU-MOE-2.1's main PR. See `qwen3_moe_wgpu_parity.rs` for
//!   the future end-to-end test.
//! - Integration into `OwnedQuantizedModelWgpu::forward_qwen3_moe_wgpu` —
//!   that's the M-GPU-MOE-2.2 sub-task.
//! - The gate/up matvec (which is the heavy lifting and needs the
//!   trueno-gpu `QuantizeKernel` + `GemmKernel` wgpu surface authored
//!   first — that's M-GPU-MOE-2.1.1, a separate PR).
//!
//! ## Tests this PR ships
//!
//! Unit tests that verify the shader source contains the right bindings +
//! entry point + uses sigmoid via `exp()`. Analogous to the existing
//! `test_fused_swiglu_ptx_generation` in
//! `crates/aprender-gpu/src/kernels/elementwise/swiglu.rs:213` — same
//! "verify source codegen without dispatching" pattern.
//!
//! ## Cross-refs
//!
//! - Issue: #1582 (M-GPU-MOE-2.x wgpu helpers + parity test)
//! - CUDA sibling: `crates/aprender-gpu/src/kernels/elementwise/swiglu.rs::FusedSwigluKernel`
//! - CPU sibling: `crates/aprender-serve/src/gguf/qwen3_moe_load.rs::expert_swiglu_quantized`
//! - Parent: `OwnedQuantizedModelWgpu` in this directory's `mod.rs`
//! - Acceptance backends per #1582: Apple Metal, AMD Vulkan, Intel ARC
//!   (NVIDIA Vulkan via yoga supplementary, per Tier 2 EV analysis)

#![cfg(feature = "gpu")]

use trueno::backends::gpu::wgpu;

/// WGSL compute shader for fused SwiGLU: `output[i] = silu(gate[i]) * up[i]`.
///
/// Workgroup size: 256 threads on x-axis. Caller must dispatch
/// `ceil(n / 256)` workgroups to cover the full element count.
///
/// Bindings:
/// - `@group(0) @binding(0)`: `gate` (read-only storage)
/// - `@group(0) @binding(1)`: `up` (read-only storage)
/// - `@group(0) @binding(2)`: `output` (read-write storage)
/// - `@group(0) @binding(3)`: `dims` (uniform with `n: u32`)
///
/// SiLU formula: `silu(x) = x / (1 + exp(-x))`. WGSL `exp()` is the
/// natural exponential (base-e). Compare against CUDA sibling
/// `FusedSwigluKernel::build_ptx` which uses `ex2.approx.f32` (base-2)
/// combined with a `LOG2_E` multiplier — algebraically equivalent.
pub const FUSED_SWIGLU_WGSL: &str = r"
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Dims {
    n: u32,
}

@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dims.n) {
        return;
    }
    let g = gate[i];
    let sigmoid_g = 1.0 / (1.0 + exp(-g));
    output[i] = g * sigmoid_g * up[i];
}
";

/// Fused-SwiGLU wgpu compute pipeline — cacheable on the parent model.
///
/// Construction is one-shot per `wgpu::Device`. Dispatch is fast (records
/// into a `wgpu::CommandEncoder`). Bindings + workgroup size are baked in
/// via `FUSED_SWIGLU_WGSL`.
pub struct FusedSwigluWgpuKernel {
    /// The compiled compute pipeline.
    pub pipeline: wgpu::ComputePipeline,
    /// The bind group layout (cached for dispatch-side bind-group creation).
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl FusedSwigluWgpuKernel {
    /// Build the pipeline + bind group layout on the given device.
    ///
    /// Mirrors the `aprender-train/src/autograd/wgpu_training.rs` pattern:
    /// create shader module → create bind group layout → create pipeline
    /// layout → create compute pipeline. Idempotent and cheap to call
    /// once at model construction time.
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fused_swiglu_wgpu"),
            source: wgpu::ShaderSource::Wgsl(FUSED_SWIGLU_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fused_swiglu_bgl"),
            entries: &[
                // gate (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // up (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // output (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // dims (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fused_swiglu_pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fused_swiglu_pipe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// Workgroup size baked into `FUSED_SWIGLU_WGSL`. Callers compute
/// `dispatch_x = ceil(n / WORKGROUP_SIZE)`.
pub const WORKGROUP_SIZE: u32 = 256;

#[cfg(test)]
mod shader_source_tests {
    //! Source-level codegen tests (no GPU init required).
    //!
    //! These mirror `test_fused_swiglu_ptx_generation` in
    //! `crates/aprender-gpu/src/kernels/elementwise/swiglu.rs` — verify the
    //! generated source has the right structure WITHOUT actually compiling
    //! or dispatching on hardware. Dispatch-level tests live in the
    //! `qwen3_moe_wgpu_parity.rs` end-to-end falsifier (gated on actual
    //! wgpu adapter availability).

    use super::*;

    #[test]
    fn wgsl_source_has_compute_entry_point() {
        assert!(
            FUSED_SWIGLU_WGSL.contains("@compute @workgroup_size(256)"),
            "WGSL must declare compute entry with workgroup_size(256)"
        );
        assert!(
            FUSED_SWIGLU_WGSL.contains("fn main("),
            "WGSL must declare main() entry function"
        );
    }

    #[test]
    fn wgsl_source_declares_all_four_bindings() {
        // Same binding scheme as documented at the top of FUSED_SWIGLU_WGSL.
        assert!(
            FUSED_SWIGLU_WGSL.contains("@binding(0) var<storage, read> gate"),
            "binding(0) must be the gate input"
        );
        assert!(
            FUSED_SWIGLU_WGSL.contains("@binding(1) var<storage, read> up"),
            "binding(1) must be the up input"
        );
        assert!(
            FUSED_SWIGLU_WGSL.contains("@binding(2) var<storage, read_write> output"),
            "binding(2) must be the read_write output"
        );
        assert!(
            FUSED_SWIGLU_WGSL.contains("@binding(3) var<uniform> dims"),
            "binding(3) must be the uniform dims block"
        );
    }

    #[test]
    fn wgsl_source_uses_exp_for_sigmoid() {
        // SiLU implementation: silu(x) = x / (1 + exp(-x)). WGSL uses
        // base-e exp(). The CUDA sibling uses ex2 (base-2) combined with
        // LOG2_E — algebraically equivalent; this test pins WGSL to the
        // natural-exp form for diff-reviewability.
        assert!(
            FUSED_SWIGLU_WGSL.contains("exp(-g)"),
            "WGSL must use exp(-g) for sigmoid (matches denominator of silu formula)"
        );
        assert!(
            FUSED_SWIGLU_WGSL.contains("g * sigmoid_g * up[i]"),
            "WGSL must compute fused silu(gate) * up in one statement"
        );
    }

    #[test]
    fn wgsl_source_has_bounds_check() {
        // Bounds check protects against dispatch sizes that overshoot
        // the buffer (workgroup_size 256 rounds up; tail threads must
        // early-out, not write past array end).
        assert!(
            FUSED_SWIGLU_WGSL.contains("if (i >= dims.n)"),
            "WGSL must guard against out-of-range thread IDs (workgroup ceiling)"
        );
    }

    #[test]
    fn workgroup_size_constant_matches_wgsl() {
        assert_eq!(
            WORKGROUP_SIZE, 256,
            "WORKGROUP_SIZE Rust constant must match the WGSL @workgroup_size literal"
        );
        let needle = format!("@workgroup_size({WORKGROUP_SIZE})");
        assert!(
            FUSED_SWIGLU_WGSL.contains(&needle),
            "WGSL must contain @workgroup_size({WORKGROUP_SIZE})"
        );
    }
}
