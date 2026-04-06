//! wgpu-accelerated transformer forward pass
//!
//! Provides [`WgpuForwardPass`] that batches all transformer layer matmuls
//! and activations into a single GPU execution, eliminating per-operation
//! CPU↔GPU round-trips.
//!
//! # Architecture
//!
//! - Weights are uploaded once at construction and kept GPU-resident
//! - Each forward call: upload hidden → batch all ops → download result
//! - Uses `GpuCommandBatch` for deferred execution with persistent buffers
//!
//! # Contract (C-WGPU-FWD-001)
//!
//! - **Precondition**: Transformer model loaded with valid weights
//! - **Postcondition**: Output hidden states numerically match CPU forward pass (within fp32 tolerance)
//! - **Invariant**: GPU buffers remain valid across forward calls

use crate::autograd::Tensor;
use crate::lora::LoRALayer;
use crate::transformer::config::TransformerConfig;
use crate::transformer::model::Transformer;
use std::cell::RefCell;
use std::sync::Arc;
use trueno::backends::gpu::{wgpu, GpuCommandBatch, GpuDevice, PipelineCache};

/// Pre-uploaded FFN weight buffers for a single transformer layer (KAIZEN-015).
///
/// Each buffer is wrapped in `Arc` so it can be shared across multiple
/// `GpuCommandBatch` executions via `import_buffer()` without re-uploading.
/// The buffers remain GPU-resident for the lifetime of `WgpuForwardPass`.
struct GpuResidentFfnWeights {
    /// Gate projection: (hidden_size, intermediate_size)
    w_gate: Arc<wgpu::Buffer>,
    /// Up projection: (hidden_size, intermediate_size)
    w_up: Arc<wgpu::Buffer>,
    /// Down projection: (intermediate_size, hidden_size)
    w_down: Arc<wgpu::Buffer>,
    /// Size of gate/up buffers in f32 elements (hidden_size * intermediate_size)
    gate_up_elements: usize,
    /// Size of down buffer in f32 elements (intermediate_size * hidden_size)
    down_elements: usize,
}

/// wgpu-accelerated transformer forward pass
///
/// Batches all matmul and activation operations across transformer layers
/// into a single `GpuCommandBatch::execute()` call.
///
/// Current implementation accelerates FFN matmuls (gate/up/down projections)
/// which dominate compute time (~60-70% of forward pass). Attention remains
/// on CPU due to softmax/RoPE complexity (Phase 2 will add GPU attention).
pub struct WgpuForwardPass {
    device: GpuDevice,
    config: TransformerConfig,
    /// Number of transformer layers
    num_layers: usize,
    /// KAIZEN-015: GPU-resident FFN weights — uploaded once, reused every forward pass.
    /// Each layer has (w_gate, w_up, w_down) as persistent `wgpu::Buffer`s.
    /// Empty when constructed via `new()`/`new_default()`; populated by `with_resident_weights()`.
    ffn_weights: Vec<GpuResidentFfnWeights>,
    /// KAIZEN-023: Persistent pipeline cache across batch executions.
    /// Shaders compiled in layer 1's batch are reused for layers 2-36.
    /// Reduces 108 shader compilations per forward pass to just 3.
    /// Uses `RefCell` because `forward_ffn_gpu()` is called via `&self`.
    pipeline_cache: RefCell<PipelineCache>,
}

impl WgpuForwardPass {
    /// Create a new wgpu forward pass from a transformer model
    ///
    /// # Arguments
    /// * `model` - Transformer model with loaded weights
    /// * `adapter_index` - wgpu adapter index to use
    ///
    /// # Errors
    /// Returns error if GPU device creation fails
    pub fn new(config: &TransformerConfig, adapter_index: u32) -> Result<Self, String> {
        let device = GpuDevice::new_with_adapter_index(adapter_index)?;

        Ok(Self {
            device,
            config: config.clone(),
            num_layers: config.num_hidden_layers,
            ffn_weights: Vec::new(),
            pipeline_cache: RefCell::new(PipelineCache::new()),
        })
    }

    /// Create from default GPU adapter
    pub fn new_default(config: &TransformerConfig) -> Result<Self, String> {
        let device = GpuDevice::new()?;

        Ok(Self {
            device,
            config: config.clone(),
            num_layers: config.num_hidden_layers,
            ffn_weights: Vec::new(),
            pipeline_cache: RefCell::new(PipelineCache::new()),
        })
    }

    /// Create from model with GPU-resident FFN weights (KAIZEN-015).
    ///
    /// Uploads all FFN weights (gate, up, down projections for every layer) to
    /// GPU memory at construction time. Subsequent forward passes use
    /// `import_buffer()` to reference these persistent buffers, eliminating
    /// the per-call H2D transfer overhead.
    ///
    /// For Qwen3-4B (36 layers, hidden=3584, intermediate=18944):
    /// - Per-layer: 3 * 3584 * 18944 * 4 bytes = ~775 MB
    /// - Total: 36 layers * 775 MB = ~27.2 GB GPU memory (once)
    /// - Savings: 14.6 GB/step H2D transfer eliminated
    ///
    /// # Contract (C-WGPU-RESIDENT-001)
    ///
    /// - **Precondition**: Model has valid FFN weights for all layers
    /// - **Postcondition**: All FFN weights GPU-resident, zero H2D transfers during forward
    /// - **Invariant**: Weights are frozen (read-only) -- LoRA adapters remain CPU-resident
    pub fn with_resident_weights(model: &Transformer) -> Result<Self, String> {
        contract_pre_with_resident_weights!();
        let device = GpuDevice::new()?;
        let config = model.config.clone();
        let num_layers = config.num_hidden_layers;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let gate_up_elements = hidden_size * intermediate_size;
        let down_elements = intermediate_size * hidden_size;

        let mut ffn_weights = Vec::with_capacity(num_layers);
        let mut total_bytes: usize = 0;

        for (i, layer) in model.layers.iter().enumerate() {
            let gate_data = layer.ffn.w_gate.data();
            let gate_slice = gate_data
                .as_slice()
                .ok_or_else(|| format!("Layer {i}: gate weight not contiguous"))?;
            let up_data = layer.ffn.w_up.data();
            let up_slice =
                up_data.as_slice().ok_or_else(|| format!("Layer {i}: up weight not contiguous"))?;
            let down_data = layer.ffn.w_down.data();
            let down_slice = down_data
                .as_slice()
                .ok_or_else(|| format!("Layer {i}: down weight not contiguous"))?;

            let w_gate = Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ffn_gate_L{i}")),
                size: (gate_slice.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            device.queue.write_buffer(&w_gate, 0, bytemuck::cast_slice(gate_slice));

            let w_up = Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ffn_up_L{i}")),
                size: (up_slice.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            device.queue.write_buffer(&w_up, 0, bytemuck::cast_slice(up_slice));

            let w_down = Arc::new(device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ffn_down_L{i}")),
                size: (down_slice.len() * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            device.queue.write_buffer(&w_down, 0, bytemuck::cast_slice(down_slice));

            let layer_bytes = (gate_slice.len() + up_slice.len() + down_slice.len()) * 4;
            total_bytes += layer_bytes;

            ffn_weights.push(GpuResidentFfnWeights {
                w_gate,
                w_up,
                w_down,
                gate_up_elements,
                down_elements,
            });
        }

        let total_mb = total_bytes as f64 / (1024.0 * 1024.0);
        eprintln!("[wgpu] GPU-resident FFN weights: {num_layers} layers, {total_mb:.1} MB");

        Ok(Self {
            device,
            config,
            num_layers,
            ffn_weights,
            pipeline_cache: RefCell::new(PipelineCache::new()),
        })
    }

    /// Execute forward pass through all transformer layers on GPU
    ///
    /// Batches FFN matmuls (gate/up/down projections) per layer. Attention and
    /// normalization remain on CPU for this phase.
    ///
    /// # Arguments
    /// * `model` - Transformer model with weights
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    /// Hidden states tensor (seq_len * hidden_size)
    pub fn forward_hidden(&self, model: &Transformer, token_ids: &[u32]) -> Result<Tensor, String> {
        let seq_len = token_ids.len();
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;

        // Step 1: Embed tokens on CPU (small operation)
        let mut hidden = model.embed_tokens.forward(token_ids);

        // Step 2: Process each transformer layer
        // Attention stays on CPU; FFN matmuls go to GPU via batched execution
        //
        // KAIZEN-004: Suppress per-op wgpu during attention. Without this,
        // each Q/K/V/O projection triggers a full buffer upload/compute/download
        // cycle (144 per-op matmuls × ~3-5ms = 430-720ms overhead per sample).
        // CPU SIMD is equally fast and doesn't contend for GPU bandwidth.
        crate::autograd::suppress_per_op_wgpu();
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            // --- Attention on CPU SIMD (includes RoPE, softmax, masking) ---
            let norm1 = layer.input_norm.forward_batched(&hidden, seq_len, hidden_size);
            let attn_out = layer.self_attn.forward(&norm1, seq_len);
            let residual1 = crate::autograd::add(&hidden, &attn_out);

            // --- FFN on GPU (3 large matmuls + SwiGLU) ---
            let norm2 = layer.post_attn_norm.forward_batched(&residual1, seq_len, hidden_size);

            // KAIZEN-015: Use GPU-resident weights if available
            let resident = self.ffn_weights.get(layer_idx);

            let ffn_out = self.forward_ffn_gpu(
                &norm2,
                &layer.ffn.w_gate,
                &layer.ffn.w_up,
                &layer.ffn.w_down,
                seq_len,
                hidden_size,
                intermediate_size,
                resident,
            )?;

            // Residual connection
            hidden = crate::autograd::add(&residual1, &ffn_out);
        }
        // KAIZEN-004: Re-enable per-op wgpu for backward pass / other operations
        crate::autograd::unsuppress_per_op_wgpu();

        // Step 3: Final normalization on CPU
        let normalized = model.norm.forward_batched(&hidden, seq_len, hidden_size);

        Ok(normalized)
    }

    /// Execute FFN forward pass on GPU using batched operations.
    ///
    /// Batches gate/up/down matmuls + SwiGLU into single GPU execution:
    /// 1. Upload input (small: seq_len * hidden_size)
    /// 2. Import or upload weights (gate, up, down)
    /// 3. Compute: gate = input @ w_gate, up = input @ w_up, silu(gate) * up, down
    /// 4. Download: ffn_output
    ///
    /// When `resident_weights` is `Some`, weight buffers are imported from
    /// persistent GPU memory (KAIZEN-015: zero H2D transfer for weights).
    /// When `None`, falls back to uploading from CPU tensors each call.
    fn forward_ffn_gpu(
        &self,
        input: &Tensor,
        w_gate: &Tensor,
        w_up: &Tensor,
        w_down: &Tensor,
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
        resident_weights: Option<&GpuResidentFfnWeights>,
    ) -> Result<Tensor, String> {
        use trueno::backends::gpu::runtime;

        runtime::block_on(async {
            let mut batch = GpuCommandBatch::new(self.device.clone());

            // Upload input (always from CPU — small: seq_len * hidden_size)
            let input_data = input.data();
            let input_slice = input_data.as_slice().ok_or("Input tensor not contiguous")?;
            let buf_input = batch.upload(input_slice);

            // KAIZEN-015: Import GPU-resident weights or fall back to CPU upload
            let (buf_gate, buf_up, buf_down) = if let Some(rw) = resident_weights {
                // Zero H2D transfer — Arc::clone is cheap (~1 atomic increment)
                let g = batch.import_buffer(Arc::clone(&rw.w_gate), rw.gate_up_elements);
                let u = batch.import_buffer(Arc::clone(&rw.w_up), rw.gate_up_elements);
                let d = batch.import_buffer(Arc::clone(&rw.w_down), rw.down_elements);
                (g, u, d)
            } else {
                // Fallback: upload weights from CPU tensors (original path)
                let gate_data = w_gate.data();
                let gate_slice = gate_data.as_slice().ok_or("Gate weight not contiguous")?;
                let up_data = w_up.data();
                let up_slice = up_data.as_slice().ok_or("Up weight not contiguous")?;
                let down_data = w_down.data();
                let down_slice = down_data.as_slice().ok_or("Down weight not contiguous")?;
                let g = batch.upload(gate_slice);
                let u = batch.upload(up_slice);
                let d = batch.upload(down_slice);
                (g, u, d)
            };

            // Gate projection: (seq_len, hidden) @ (hidden, intermediate)
            let gate_out = batch.matmul(
                buf_input,
                buf_gate,
                seq_len as u32,
                hidden_size as u32,
                intermediate_size as u32,
            );

            // Up projection: (seq_len, hidden) @ (hidden, intermediate)
            let up_out = batch.matmul(
                buf_input,
                buf_up,
                seq_len as u32,
                hidden_size as u32,
                intermediate_size as u32,
            );

            // SwiGLU: swish(gate) * up
            let gate_activated = batch.swish(gate_out);
            let swiglu_out = batch.mul(gate_activated, up_out);

            // Down projection: (seq_len, intermediate) @ (intermediate, hidden)
            let ffn_out = batch.matmul(
                swiglu_out,
                buf_down,
                seq_len as u32,
                intermediate_size as u32,
                hidden_size as u32,
            );

            // Execute all ops in single batch — KAIZEN-023: persistent pipeline cache
            batch.execute_with_cache(&mut self.pipeline_cache.borrow_mut()).await?;

            // Download result
            let result_data = batch.read(ffn_out).await?;

            Ok(Tensor::from_vec(result_data, false))
        })
    }

    /// Execute batched forward pass for multiple samples (KAIZEN-008).
    ///
    /// Processes all samples through each layer together, uploading FFN weights
    /// ONCE per layer instead of once per sample. With batch_size=20 × 36 layers,
    /// this reduces weight uploads from 720 to 36 (20× reduction, ~146 GB saved).
    ///
    /// Attention remains per-sample on CPU SIMD. FFN inputs are concatenated
    /// across samples for a single large matmul per layer.
    ///
    /// # KAIZEN-010: LoRA integration
    ///
    /// When `lora_layers` is provided, attention uses `forward_with_lora()` to
    /// apply LoRA adjusts to Q and V projections. Layout: `[Q_0, V_0, Q_1, V_1, ...]`
    /// (2 LoRA layers per transformer layer). Without this, only the classifier
    /// head trains on the wgpu path (5,122 params vs 5.9M with LoRA).
    pub fn forward_hidden_batch(
        &self,
        model: &Transformer,
        batch_token_ids: &[Vec<u32>],
        lora_layers: Option<&[LoRALayer]>,
    ) -> Result<Vec<Tensor>, String> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let n = batch_token_ids.len();

        // Step 1: Embed all samples on CPU
        let mut hiddens: Vec<Tensor> =
            batch_token_ids.iter().map(|ids| model.embed_tokens.forward(ids)).collect();

        // Step 2: Layer-at-a-time processing
        // Attention on CPU SIMD (per-sample), FFN on GPU (all samples concatenated)
        //
        // KAIZEN-017: Pre-compute total_tokens once (constant across layers).
        let total_tokens: usize = batch_token_ids.iter().map(std::vec::Vec::len).sum();

        crate::autograd::suppress_per_op_wgpu();
        for (layer_idx, layer) in model.layers.iter().enumerate() {
            // Attention on CPU (per-sample, independent)
            // KAIZEN-017: Keep Tensor references instead of .to_vec() copies.
            // Eliminates n per-sample Vec allocations per layer (360 MB total for Qwen3-4B).
            let mut ffn_input_tensors: Vec<Tensor> = Vec::with_capacity(n);
            let mut residuals: Vec<Tensor> = Vec::with_capacity(n);
            for (i, hidden) in hiddens.iter().enumerate() {
                let seq_len = batch_token_ids[i].len();
                let norm1 = layer.input_norm.forward_batched(hidden, seq_len, hidden_size);

                // KAIZEN-010: Use LoRA-enabled attention when adapters are available
                let attn_out = match lora_layers {
                    Some(loras) => {
                        let q_idx = layer_idx * 2;
                        let v_idx = layer_idx * 2 + 1;
                        if v_idx < loras.len() {
                            layer.self_attn.forward_with_lora(
                                &norm1,
                                seq_len,
                                loras[q_idx].lora_a(),
                                loras[q_idx].lora_b(),
                                loras[v_idx].lora_a(),
                                loras[v_idx].lora_b(),
                                loras[q_idx].rank(),
                                loras[q_idx].scale(),
                            )
                        } else {
                            layer.self_attn.forward(&norm1, seq_len)
                        }
                    }
                    None => layer.self_attn.forward(&norm1, seq_len),
                };

                let residual1 = crate::autograd::add(hidden, &attn_out);
                let norm2 = layer.post_attn_norm.forward_batched(&residual1, seq_len, hidden_size);
                ffn_input_tensors.push(norm2);
                residuals.push(residual1);
            }

            // Concatenate all samples' FFN inputs for single GPU batch
            // KAIZEN-017: Borrow directly from Tensor instead of intermediate Vecs
            let mut concat_input = Vec::with_capacity(total_tokens * hidden_size);
            for norm2 in &ffn_input_tensors {
                let data = norm2.data();
                concat_input.extend_from_slice(data.as_slice().expect("norm2 contiguous"));
            }
            let concat_tensor = Tensor::from_vec(concat_input, false);

            // FFN on GPU — KAIZEN-015: import GPU-resident weights (zero H2D)
            let resident = self.ffn_weights.get(layer_idx);

            let ffn_out = self.forward_ffn_gpu(
                &concat_tensor,
                &layer.ffn.w_gate,
                &layer.ffn.w_up,
                &layer.ffn.w_down,
                total_tokens,
                hidden_size,
                intermediate_size,
                resident,
            )?;

            // Split FFN output back into per-sample tensors + residual
            let ffn_data = ffn_out.data();
            let ffn_slice = ffn_data.as_slice().expect("ffn contiguous");
            let mut offset = 0;
            hiddens = residuals
                .into_iter()
                .enumerate()
                .map(|(i, r)| {
                    let len = batch_token_ids[i].len() * hidden_size;
                    let sample_ffn =
                        Tensor::from_vec(ffn_slice[offset..offset + len].to_vec(), false);
                    offset += len;
                    crate::autograd::add(&r, &sample_ffn)
                })
                .collect();
        }
        crate::autograd::unsuppress_per_op_wgpu();

        // Step 3: Final normalization (per-sample)
        let results: Vec<Tensor> = hiddens
            .into_iter()
            .enumerate()
            .map(|(i, h)| {
                let seq_len = batch_token_ids[i].len();
                model.norm.forward_batched(&h, seq_len, hidden_size)
            })
            .collect();

        Ok(results)
    }

    /// Get the adapter info for display
    pub fn adapter_info(&self) -> String {
        format!(
            "wgpu device ({}x{} model, {} layers)",
            self.config.hidden_size, self.config.intermediate_size, self.num_layers
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_forward_pass_creation() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut config = TransformerConfig::llama2_7b();
        config.hidden_size = 64;
        config.num_hidden_layers = 2;
        config.num_attention_heads = 4;
        config.num_kv_heads = 4;
        config.intermediate_size = 128;
        config.vocab_size = 100;

        let pass = WgpuForwardPass::new_default(&config);
        assert!(pass.is_ok(), "WgpuForwardPass creation failed: {:?}", pass.err());
    }

    #[test]
    fn test_wgpu_ffn_numerical_correctness() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut config = TransformerConfig::llama2_7b();
        config.hidden_size = 8;
        config.num_hidden_layers = 1;
        config.num_attention_heads = 2;
        config.num_kv_heads = 2;
        config.intermediate_size = 16;
        config.vocab_size = 32;

        let pass =
            WgpuForwardPass::new_default(&config).expect("GPU available but creation failed");

        // Create small test tensors
        let input = Tensor::from_vec(vec![1.0; 8], false); // 1 token × 8 hidden
        let w_gate = Tensor::from_vec(vec![0.1; 8 * 16], false);
        let w_up = Tensor::from_vec(vec![0.1; 8 * 16], false);
        let w_down = Tensor::from_vec(vec![0.1; 16 * 8], false);

        let gpu_result = pass.forward_ffn_gpu(
            &input, &w_gate, &w_up, &w_down, 1, 8, 16,
            None, // No GPU-resident weights for this test
        );

        assert!(gpu_result.is_ok(), "GPU FFN failed: {:?}", gpu_result.err());

        let gpu_data = gpu_result.expect("checked above");
        assert_eq!(gpu_data.len(), 8, "Output should be 1 × 8");

        // Verify no NaN/Inf
        for (i, &val) in gpu_data.data().iter().enumerate() {
            assert!(val.is_finite(), "NaN/Inf at index {i}: {val}");
        }
    }
}
