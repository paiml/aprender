//! #4665: the Qwen3.5-MoE FFN on the device — the GPU twin of
//! `Qwen35Model::moe_combine_into` (`forward_qwen35_moe.rs`).
//!
//! A MoE layer's `ffn_gate`/`ffn_up`/`ffn_down` are its shared expert, which the dense
//! FFN code already runs into `ffn_down`. [`combine`] then does what the CPU combine
//! does, in the CPU's order:
//!
//! ```text
//! acc  = sigmoid(s·x) ⊙ shared_out
//! acc += p̂_e ⊙ down_e(silu(gate_e·x) ⊙ up_e·x)      for each routed e, in route order
//! ```
//!
//! and the caller adds `acc` to the residual stream. The router and the shared-expert
//! gate are ONE F32 GEMV (`[num_experts + 1] × hidden`, the gate as the last row), so
//! a layer has exactly one host round trip: the logits come down, the shared routing
//! rule [`route_top_k`] picks the experts, their weights go back up. That round trip
//! is why a MoE model cannot be captured as a CUDA graph (see
//! [`Qwen35CudaModel::has_moe`]).

use super::{gpu_err, CudaQuantWeight, Qwen35CudaModel};
use crate::cuda::{CudaExecutor, WeightQuantType};
use crate::error::{RealizarError, Result};
use crate::gguf::forward_qwen35::qwen35_moe::Qwen35MoeFfn;
use crate::gguf::qwen3_moe_load::route_top_k;
use crate::gguf::OwnedQuantizedTensor;
use trueno_gpu::driver::GpuBuffer;

/// One stacked expert projection, resident as ONE allocation: expert `e`'s `[n × k]`
/// matrix starts `e * stride` bytes past `base`.
#[derive(Debug, Clone, Copy)]
pub(super) struct CudaExpertStack {
    base: u64,
    stride: u64,
    qtype: WeightQuantType,
    n: u32,
    k: u32,
}

impl CudaExpertStack {
    fn expert(&self, e: usize) -> u64 {
        self.base + self.stride * e as u64
    }
}

/// One layer's routed experts and router, device-resident.
pub(super) struct CudaMoe {
    /// `[num_experts + 1] × hidden` F32: the router, then the shared-expert gate.
    router: CudaQuantWeight,
    gate: CudaExpertStack,
    up: CudaExpertStack,
    down: CudaExpertStack,
    num_experts: usize,
    top_k: usize,
}

impl CudaMoe {
    /// `expert_dim`, the routed SwiGLU's width.
    pub(super) fn expert_dim(&self) -> u32 {
        self.gate.n
    }
}

/// The scratch every MoE layer shares, sized for the widest layer.
pub(super) struct MoeScratch {
    /// `[num_experts + 1]` router logits, the last the shared-gate pre-activation.
    logits: GpuBuffer<f32>,
    logits_host: Vec<f32>,
    /// `[(top_k + 1) × hidden]`: row 0 the shared-gate scale, row `1 + slot` the routed
    /// weight of `slot`, each broadcast across `hidden`.
    weights: GpuBuffer<f32>,
    weights_host: Vec<f32>,
    e_gate: GpuBuffer<f32>,
    e_up: GpuBuffer<f32>,
    e_act: GpuBuffer<f32>,
    e_down: GpuBuffer<f32>,
    e_scaled: GpuBuffer<f32>,
    /// `[hidden]`, the layer's full FFN output — what the caller adds to the residual.
    pub(super) acc: GpuBuffer<f32>,
}

/// Bytes [`upload`] puts on the device for one layer — the capacity plan's term.
pub(crate) fn moe_upload_bytes(moe: &Qwen35MoeFfn) -> u64 {
    let stack = |v: &[OwnedQuantizedTensor]| v.iter().map(|t| t.data.len() as u64).sum::<u64>();
    4 * (moe.router.len() + moe.shared_gate.len()) as u64
        + stack(&moe.gate_exps)
        + stack(&moe.up_exps)
        + stack(&moe.down_exps)
}

fn refuse(reason: String) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: "qwen35moe_cuda_upload".to_string(),
        reason,
    }
}

/// Concatenate one projection's per-expert tensors and upload them as one stack.
fn upload_stack(
    executor: &mut CudaExecutor,
    name: &str,
    experts: &[OwnedQuantizedTensor],
) -> Result<CudaExpertStack> {
    let first = experts
        .first()
        .ok_or_else(|| refuse(format!("'{name}': no experts")))?;
    let qtype = WeightQuantType::from_ggml_type(first.qtype)
        .ok_or_else(|| refuse(super::no_gemv_kernel_reason(name, first.qtype)))?;
    let stride = first.data.len();
    let uniform = experts.iter().all(|t| {
        t.qtype == first.qtype
            && t.data.len() == stride
            && t.out_dim == first.out_dim
            && t.in_dim == first.in_dim
    });
    if !uniform || !qtype.matches_size(stride, first.out_dim, first.in_dim) {
        return Err(refuse(format!(
            "'{name}': the {} experts are not equal [{}][{}] {qtype:?} matrices of {stride} bytes",
            experts.len(),
            first.out_dim,
            first.in_dim
        )));
    }
    let mut bytes = Vec::with_capacity(stride * experts.len());
    for t in experts {
        bytes.extend_from_slice(&t.data);
    }
    executor
        .load_quantized_weights_with_type(name, &bytes, first.qtype)
        .map_err(|e| gpu_err("qwen35moe_cuda_upload", &e))?;
    let base = executor
        .get_quantized_weight_ptr(name)
        .map_err(|e| gpu_err("qwen35moe_cuda_upload", &e))?;
    Ok(CudaExpertStack {
        base,
        stride: stride as u64,
        qtype,
        n: u32::try_from(first.out_dim).unwrap_or(0),
        k: u32::try_from(first.in_dim).unwrap_or(0),
    })
}

/// Upload one layer's router (with the shared gate appended as its last row) and its
/// three expert stacks.
pub(super) fn upload(
    executor: &mut CudaExecutor,
    il: usize,
    moe: &Qwen35MoeFfn,
    hidden: usize,
) -> Result<CudaMoe> {
    let num_experts = moe.gate_exps.len();
    if moe.router.len() != num_experts * hidden || moe.shared_gate.len() != hidden {
        return Err(refuse(format!(
            "layer {il}: router {} values, shared gate {} — want {num_experts} × {hidden} and \
             {hidden}",
            moe.router.len(),
            moe.shared_gate.len()
        )));
    }
    let name = |s: &str| format!("qwen35.blk.{il}.{s}");
    let router_bytes: Vec<u8> = moe
        .router
        .iter()
        .chain(&moe.shared_gate)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let router_name = name("ffn_gate_inp+shexp.weight");
    executor
        .load_quantized_weights_with_type(
            &router_name,
            &router_bytes,
            crate::gguf::types::GGUF_TYPE_F32,
        )
        .map_err(|e| gpu_err("qwen35moe_cuda_upload", &e))?;
    let router_ptr = executor
        .get_quantized_weight_ptr(&router_name)
        .map_err(|e| gpu_err("qwen35moe_cuda_upload", &e))?;
    let gate = upload_stack(executor, &name("ffn_gate_exps.weight"), &moe.gate_exps)?;
    let up = upload_stack(executor, &name("ffn_up_exps.weight"), &moe.up_exps)?;
    let down = upload_stack(executor, &name("ffn_down_exps.weight"), &moe.down_exps)?;
    let h = u32::try_from(hidden).unwrap_or(0);
    if (gate.k, up.k, down.n) != (h, h, h) || gate.n != up.n || down.k != gate.n {
        return Err(refuse(format!(
            "layer {il}: expert shapes disagree — gate {}x{}, up {}x{}, down {}x{} (hidden \
             {hidden})",
            gate.n, gate.k, up.n, up.k, down.n, down.k
        )));
    }
    Ok(CudaMoe {
        router: CudaQuantWeight {
            ptr: router_ptr,
            qtype: WeightQuantType::F32,
            n: u32::try_from(num_experts + 1).unwrap_or(0),
            k: h,
        },
        gate,
        up,
        down,
        num_experts,
        top_k: moe.top_k,
    })
}

/// Allocate the scratch for the widest of `layers`; `None` for a dense model.
pub(super) fn build_scratch<'m>(
    executor: &CudaExecutor,
    layers: impl Iterator<Item = &'m CudaMoe>,
    hidden: usize,
) -> Result<Option<MoeScratch>> {
    let (mut experts, mut top_k, mut e_dim, mut any) = (0usize, 0usize, 0usize, false);
    for m in layers {
        any = true;
        experts = experts.max(m.num_experts);
        top_k = top_k.max(m.top_k);
        e_dim = e_dim.max(m.expert_dim() as usize);
    }
    if !any {
        return Ok(None);
    }
    let z = |len: usize| Qwen35CudaModel::zeros(executor, len);
    Ok(Some(MoeScratch {
        logits: z(experts + 1)?,
        logits_host: vec![0.0; experts + 1],
        weights: z((top_k + 1) * hidden)?,
        weights_host: vec![0.0; (top_k + 1) * hidden],
        e_gate: z(e_dim)?,
        e_up: z(e_dim)?,
        e_act: z(e_dim)?,
        e_down: z(hidden)?,
        e_scaled: z(hidden)?,
        acc: z(hidden)?,
    }))
}

/// `sigmoid(g)`, written as the CPU's `Qwen35MoeFfn::shared_scale` writes it.
fn sigmoid(g: f32) -> f32 {
    1.0 / (1.0 + (-g).exp())
}

/// The MoE half of one token's FFN, into `ms.acc`: the shared expert's output
/// `shared_out` scaled by its gate, plus every routed expert weighted by its
/// renormalized probability. `x` is the post-attention-normed token.
pub(super) fn combine(
    ex: &mut CudaExecutor,
    moe: &CudaMoe,
    ms: &mut MoeScratch,
    x: &GpuBuffer<f32>,
    shared_out: &GpuBuffer<f32>,
    hidden_dim: u32,
) -> std::result::Result<(), trueno_gpu::GpuError> {
    let hidden = hidden_dim as usize;
    let e_dim = moe.expert_dim();
    let r = &moe.router;
    ex.gemv_dispatch(r.qtype, r.ptr, x, &ms.logits, r.n, r.k)?;
    // The one host round trip. The sync also retires every earlier read of
    // `ms.weights`, so overwriting it below cannot race a previous layer.
    ex.sync_stream()?;
    let n_logits = moe.num_experts + 1;
    let logits = &mut ms.logits_host[..n_logits];
    if ms.logits.len() == n_logits {
        ms.logits.copy_to_host(logits)?;
    } else {
        let view = Qwen35CudaModel::view(&ms.logits, 0, n_logits as u32);
        let res = view.copy_to_host(logits);
        std::mem::forget(view);
        res?;
    }
    let routes = route_top_k(&logits[..moe.num_experts], moe.top_k);
    let scale = sigmoid(logits[moe.num_experts]);
    ms.weights_host[..hidden].fill(scale);
    for (slot, &(_, w)) in routes.iter().enumerate() {
        ms.weights_host[(slot + 1) * hidden..(slot + 2) * hidden].fill(w);
    }
    ms.weights.copy_from_host(&ms.weights_host)?;

    let row = |slot: usize| Qwen35CudaModel::view(&ms.weights, (slot * hidden) as u32, hidden_dim);
    let scale_row = row(0);
    let res = ex.elementwise_mul_into(shared_out, &scale_row, &ms.acc, hidden_dim);
    std::mem::forget(scale_row);
    res?;
    for (slot, &(e, _)) in routes.iter().enumerate() {
        let (g, u, dn) = (moe.gate, moe.up, moe.down);
        ex.gemv_dispatch(g.qtype, g.expert(e), x, &ms.e_gate, g.n, g.k)?;
        ex.gemv_dispatch(u.qtype, u.expert(e), x, &ms.e_up, u.n, u.k)?;
        ex.fused_swiglu_into(&ms.e_gate, &ms.e_up, &ms.e_act, e_dim)?;
        ex.gemv_dispatch(dn.qtype, dn.expert(e), &ms.e_act, &ms.e_down, dn.n, dn.k)?;
        let w_row = row(slot + 1);
        let res = ex.elementwise_mul_into(&ms.e_down, &w_row, &ms.e_scaled, hidden_dim);
        std::mem::forget(w_row);
        res?;
        ex.residual_add_into(&ms.acc, &ms.e_scaled, &ms.acc, hidden_dim)?;
    }
    Ok(())
}
