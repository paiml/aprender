//! PMAT-291: Graph-based transformer layer decode.
//!
//! Alternative to `transformer_layer_batched()` that uses the tensor graph
//! executor. Dispatches 13 tensor-level operations instead of ~15 individual
//! kernel calls, providing a cleaner abstraction and foundation for CUDA
//! graph capture with fewer nodes.
//!
//! Activated by `GRAPH_DISPATCH=1` environment variable.

use trueno_gpu::driver::GpuBuffer;
use trueno_gpu::graph::execute_graph;
use trueno_gpu::GpuError;

use crate::cuda::executor::graph_builder::build_layer_graph;
use crate::cuda::executor::CudaExecutor;
use crate::cuda::types::ValidatedLayerWeights;

impl CudaExecutor {
    /// PMAT-291: Graph-based batched transformer layer decode.
    ///
    /// Equivalent to `transformer_layer_batched()` but routes through the
    /// tensor graph executor. Each layer is expressed as a 14-node compute
    /// graph (1 leaf + 13 ops) that dispatches the same kernels.
    ///
    /// The graph path skips the fused DP4A QKV optimization (uses individual
    /// projections) but gains a simpler dispatch loop suitable for CUDA graph
    /// capture with ~14 nodes per layer instead of ~15+ individual kernel calls.
    #[allow(clippy::too_many_arguments)]
    pub fn transformer_layer_batched_graph(
        &mut self,
        input: &GpuBuffer<f32>,
        layer_idx: usize,
        layer_weights: &ValidatedLayerWeights,
        m: u32,
        positions: &[u32],
        hidden_dim: u32,
        intermediate_dim: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        self.validate_batched_workspace(m, positions.len())?;

        let q_dim = (self.kv_num_heads * self.kv_head_dim) as u32;
        let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as u32;

        // Extract workspace buffer pointers
        let hidden_buf1_ptr = self
            .workspace
            .hidden_buf1
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: hidden_buf1".to_string()))?
            .as_ptr();
        let hidden_buf2_ptr = self
            .workspace
            .hidden_buf2
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: hidden_buf2".to_string()))?
            .as_ptr();
        let input_staging_ptr = self
            .workspace
            .input_staging
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: input_staging".to_string()))?
            .as_ptr();
        let q_buf_ptr = self
            .workspace
            .q_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: q_buf".to_string()))?
            .as_ptr();
        let k_buf_ptr = self
            .workspace
            .k_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: k_buf".to_string()))?
            .as_ptr();
        let v_buf_ptr = self
            .workspace
            .v_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: v_buf".to_string()))?
            .as_ptr();
        let attn_out_ptr = self
            .workspace
            .attn_out_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: attn_out".to_string()))?
            .as_ptr();
        let ffn_gate_ptr = self
            .workspace
            .ffn_gate_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: ffn_gate".to_string()))?
            .as_ptr();
        let ffn_up_ptr = self
            .workspace
            .ffn_up_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: ffn_up".to_string()))?
            .as_ptr();
        let ffn_act_ptr = self
            .workspace
            .ffn_act_buf
            .as_ref()
            .ok_or_else(|| GpuError::InvalidLaunchConfig("PMAT-291: ffn_act".to_string()))?
            .as_ptr();

        // Build the layer graph
        let (graph, _output_idx) = build_layer_graph(
            layer_weights,
            input.as_ptr(),
            hidden_dim,
            intermediate_dim,
            q_dim,
            kv_dim,
            m,
            epsilon,
            layer_idx,
            hidden_buf1_ptr,
            hidden_buf2_ptr,
            q_buf_ptr,
            k_buf_ptr,
            v_buf_ptr,
            attn_out_ptr,
            ffn_gate_ptr,
            ffn_up_ptr,
            ffn_act_ptr,
            input_staging_ptr,
        );

        // Set positions side-channel for dispatch_attention
        self.graph_dispatch_positions = positions.to_vec();

        // Execute the graph
        let _n_launches = execute_graph(&graph, self)?;

        // Clear positions
        self.graph_dispatch_positions.clear();

        // Output is in hidden_buf2 (same as transformer_layer_batched)
        Ok(())
    }

    /// Check if graph dispatch is enabled (default: ON, opt-out with GRAPH_DISPATCH=0).
    /// Cached after first check to avoid repeated env var lookups.
    pub(crate) fn use_graph_dispatch(&self) -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| std::env::var("GRAPH_DISPATCH").as_deref() != Ok("0"))
    }
}
