//! PMAT-286: Fused K+V KV-Cache Scatter Kernel
//!
//! Replaces 2 separate batched_kv_cache_scatter launches (K then V) with
//! 1 fused launch using `blockIdx.z` (z=0=K, z=1=V).
//!
//! Grid: (num_kv_heads, M, 2), Block: (head_dim, 1, 1)
//!
//! Saves 28 cuLaunchKernel calls per step (1 per layer × 28 layers).
//! At 17.5µs/launch, this saves ~0.49ms of CPU dispatch time per step.

/// Fused K+V KV-cache scatter kernel (PMAT-286)
///
/// Uses `blockIdx.z` to select K(z=0) or V(z=1) scatter target.
/// Both K and V share the same positions, stride, and kv_dim.
#[derive(Debug, Clone)]
pub struct FusedKvScatterKernel {
    /// Number of KV heads (grid X dimension)
    pub num_kv_heads: u32,
    /// Head dimension (block X dimension / bounds check)
    pub head_dim: u32,
    /// Maximum sequence length (for stride calculation)
    pub max_len: u32,
}

impl FusedKvScatterKernel {
    /// Create a new fused KV scatter kernel
    #[must_use]
    pub fn new(num_kv_heads: u32, head_dim: u32, max_len: u32) -> Self {
        Self { num_kv_heads, head_dim, max_len }
    }

    /// Kernel name for module caching
    #[must_use]
    pub fn name(&self) -> String {
        format!("fused_kv_scatter_{}_{}_{}", self.num_kv_heads, self.head_dim, self.max_len)
    }

    /// Emit PTX source for the target architecture
    #[must_use]
    pub fn emit_ptx(&self) -> String {
        let head_dim = self.head_dim;
        let max_len = self.max_len;
        let entry_name = self.name();

        format!(
            r#".version 7.0
.target sm_70
.address_size 64

// PMAT-286: Fused K+V scatter -- blockIdx.z selects K(0) or V(1)
.visible .entry {entry_name}(
    .param .u64 k_src_base,
    .param .u64 k_dst_base,
    .param .u64 v_src_base,
    .param .u64 v_dst_base,
    .param .u64 positions_ptr,
    .param .u32 stride_param,
    .param .u32 kv_dim_param
) {{
    .reg .u64 %rd<20>;
    .reg .u32 %r<16>;
    .reg .f32 %f<2>;
    .reg .pred %p, %p_kv;

    // Thread/block indices
    mov.u32 %r0, %tid.x;       // elem_idx
    mov.u32 %r1, %ctaid.x;     // head_idx
    mov.u32 %r2, %ctaid.y;     // seq_idx
    mov.u32 %r10, %ctaid.z;    // kv_sel (0=K, 1=V)

    // bounds check: elem_idx < head_dim
    setp.ge.u32 %p, %r0, {head_dim};
    @%p bra DONE;

    // Select src/dst based on kv_sel (branchless via selp.b64)
    setp.ne.u32 %p_kv, %r10, 0;
    ld.param.u64 %rd10, [k_src_base];
    ld.param.u64 %rd11, [v_src_base];
    selp.b64 %rd4, %rd11, %rd10, %p_kv;

    ld.param.u64 %rd12, [k_dst_base];
    ld.param.u64 %rd13, [v_dst_base];
    selp.b64 %rd7, %rd13, %rd12, %p_kv;

    // Load positions[seq_idx]
    ld.param.u64 %rd0, [positions_ptr];
    mul.wide.u32 %rd1, %r2, 4;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r3, [%rd2];            // pos = positions[seq_idx]

    // Source: src + (seq_idx * kv_dim + head_idx * head_dim + elem_idx) * 4
    ld.param.u32 %r4, [kv_dim_param];
    mul.lo.u32 %r5, %r2, %r4;             // seq_idx * kv_dim
    mul.lo.u32 %r6, %r1, {head_dim};      // head_idx * head_dim
    add.u32 %r5, %r5, %r6;
    add.u32 %r5, %r5, %r0;                // + elem_idx
    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd5, %rd4, %rd3;             // src_addr

    // Dest: dst + (seq_idx * stride + (head_idx * max_len + pos) * head_dim + elem_idx) * 4
    ld.param.u32 %r7, [stride_param];
    mul.lo.u32 %r8, %r2, %r7;             // seq_idx * stride
    mul.lo.u32 %r9, %r1, {max_len};       // head_idx * max_len
    add.u32 %r9, %r9, %r3;                // + pos
    mul.lo.u32 %r9, %r9, {head_dim};      // * head_dim
    add.u32 %r8, %r8, %r9;
    add.u32 %r8, %r8, %r0;                // + elem_idx
    mul.wide.u32 %rd6, %r8, 4;
    add.u64 %rd8, %rd7, %rd6;             // dst_addr

    // Copy: dst[...] = src[...]
    ld.global.f32 %f0, [%rd5];
    st.global.f32 [%rd8], %f0;

DONE:
    ret;
}}"#,
            entry_name = entry_name,
            head_dim = head_dim,
            max_len = max_len,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_kv_scatter_name() {
        let k = FusedKvScatterKernel::new(4, 64, 4096);
        assert_eq!(k.name(), "fused_kv_scatter_4_64_4096");
    }

    #[test]
    fn test_fused_kv_scatter_ptx_valid() {
        let k = FusedKvScatterKernel::new(4, 64, 4096);
        let ptx = k.emit_ptx();
        assert!(ptx.contains(".entry fused_kv_scatter_4_64_4096"));
        assert!(ptx.contains("selp.b64"));
        assert!(ptx.contains("ctaid.z"));
        assert!(ptx.contains("PMAT-286"));
    }
}
