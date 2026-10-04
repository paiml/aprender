//! NF4 QLoRA backward pass: `backward_nf4_gpu_blocks`,
//! `backward_nf4_gpu_blocks_gpu_resident`, `backward_nf4_gpu_blocks_loop`.

#[allow(clippy::wildcard_imports)]
use super::*;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::GpuBuffer;

impl InstructPipeline {
    /// NF4 QLoRA backward pass through all GPU transformer blocks.
    ///
    /// Computes gradient flow through frozen NF4 weights and updates LoRA
    /// adapters. After each block backward, immediately runs the LoRA optimizer
    /// step (grad workspace is shared across layers).
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    pub(super) fn backward_nf4_gpu_blocks(
        &mut self,
        grad_final_hidden: &[f32],
        seq_len: usize,
    ) -> Option<()> {
        let hidden_size = self.model.config.hidden_size;

        // Upload gradient and run RMSNorm backward in a scope to release borrows
        // before calling the shared block-loop.
        {
            let trainer = self.cuda_trainer.as_ref()?;
            let training_state = self.gpu_training.as_mut()?;
            let stream = trainer.stream();

            // PMAT-420: Use trainer.upload (fresh alloc) instead of copy_from_host_at
            training_state.grad_upload_buf = trainer.upload(grad_final_hidden).ok()?;

            // PMAT-420: Re-allocate grad buffers at seq_len if forward re-sized layer_inputs
            let expected_len = seq_len * hidden_size;
            if training_state.grad_buf_a.len() != expected_len {
                training_state.grad_buf_a = trainer.zeros(expected_len).ok()?;
                training_state.grad_buf_b = trainer.zeros(expected_len).ok()?;
                training_state.output_scratch = trainer.zeros(expected_len).ok()?;
                training_state.grad_upload_buf = trainer.upload(grad_final_hidden).ok()?;
            }

            // RMSNorm backward on GPU
            crate::autograd::cuda_backward::rms_norm_backward(
                &training_state.blocks_output,
                &training_state.final_norm_weight,
                &training_state.grad_upload_buf,
                &mut training_state.grad_buf_a,
                &mut training_state.grad_final_norm_weight,
                seq_len as u32,
                hidden_size as u32,
                1e-5_f32,
                stream,
            )
            .ok()?;
        }

        self.backward_nf4_gpu_blocks_loop(seq_len)
    }

    /// GPU-resident backward: gradient already in grad_hidden_buf from GEMM (KAIZEN-065).
    ///
    /// Same as backward_nf4_gpu_blocks but reads gradient directly from
    /// grad_hidden_buf instead of uploading from CPU. Eliminates:
    /// - ~5MB D2H download (grad_hidden_buf -> CPU)
    /// - ~5MB H2D upload (CPU -> grad_upload_buf)
    /// - 1x stream.synchronize() GPU drain point
    /// - 1x Vec<f32> heap allocation (~5MB)
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    pub(super) fn backward_nf4_gpu_blocks_gpu_resident(&mut self, seq_len: usize) -> Option<()> {
        let hidden_size = self.model.config.hidden_size;

        // KAIZEN-065: grad_hidden_buf already contains the gradient from lm_head backward GEMM.
        {
            let trainer = self.cuda_trainer.as_ref()?;
            let training_state = self.gpu_training.as_mut()?;
            let stream = trainer.stream();

            crate::autograd::cuda_backward::rms_norm_backward(
                &training_state.blocks_output,
                &training_state.final_norm_weight,
                &training_state.grad_hidden_buf,
                &mut training_state.grad_buf_a,
                &mut training_state.grad_final_norm_weight,
                seq_len as u32,
                hidden_size as u32,
                1e-5_f32,
                stream,
            )
            .ok()?;
        }

        let result = self.backward_nf4_gpu_blocks_loop(seq_len);
        // entrenar#318: invalidate shared scratch causal mask after backward
        // to prevent gradient contamination on next forward (backward writes to shared scratch)
        if let Some(ref mut scratch) = self.shared_scratch {
            scratch.causal_mask_cached_seq_len = 0;
        }
        result
    }

    /// Shared backward loop for NF4 blocks -- called by both CPU-upload and
    /// GPU-resident backward paths after RMSNorm backward completes.
    ///
    /// PMAT-488: When CUDA_GRAPH=1, captures the entire backward loop into a
    /// CUDA graph on first call and replays it on subsequent calls. This
    /// eliminates 84.6% kernel launch overhead (6.5x throughput improvement).
    ///
    /// The graph captures: backward_nf4 + fused_clip + optimizer_step for all
    /// 28 layers. All operations are async GPU kernels with zero host-device
    /// sync (PMAT-477 fused clip).
    #[cfg(feature = "cuda")]
    #[allow(unsafe_code)]
    fn backward_nf4_gpu_blocks_loop(&mut self, seq_len: usize) -> Option<()> {
        // PMAT-488: Check for backward graph replay
        if self.replay_backward_graph(seq_len)? {
            return Some(());
        }

        let trainer = self.cuda_trainer.as_ref()?;
        let stream = trainer.stream();

        // Either graph not enabled, seq_len changed, or first capture needed.
        // R15a C3b: a clipped backward is captured only when the fused clip runs inside it.
        let use_graph = backward_graph_allowed(
            super::super::backward_graph::use_backward_graph(),
            self.config.gradient_clip_norm.is_some(),
            self.lora_fused_clip.is_some(),
        );

        let lr = self.optimizer.lr();
        let training_state = self.gpu_training.as_mut()?;
        let blocks = self.cuda_blocks.as_mut()?;
        let shared_scratch = self.shared_scratch.as_mut()?;
        let grad_lora = self.cuda_lora_grad_workspace.as_mut()?;
        let opt_states = self.cuda_lora_optimizer_states.as_mut()?;

        let num_layers = blocks.len();
        let grad_a_ptr: *mut GpuBuffer<f32> = std::ptr::from_mut(&mut training_state.grad_buf_a);
        let grad_b_ptr: *mut GpuBuffer<f32> = std::ptr::from_mut(&mut training_state.grad_buf_b);
        let mut grad_output_is_a = true;

        self.nf4_lora_step += 1;
        let step = self.nf4_lora_step;

        let output_scratch_ptr: *mut GpuBuffer<f32> =
            std::ptr::from_mut(&mut training_state.output_scratch);

        // PMAT-488: Begin graph capture if enabled
        if use_graph {
            begin_backward_capture(stream);
        }

        for layer_idx in (0..num_layers).rev() {
            // SAFETY: ping-pong double-buffering. The two raw pointers reference distinct, non-overlapping device buffers (the `_a`/`_b` scratch pair); the boolean flag picks one as `&` input and the other as `&mut` output, so the resulting references never alias the same allocation.
            let (grad_output, grad_input) = unsafe {
                if grad_output_is_a {
                    (&*grad_a_ptr, &mut *grad_b_ptr)
                } else {
                    (&*grad_b_ptr, &mut *grad_a_ptr)
                }
            };

            // PMAT-483: Per-layer backward profiling (only when NOT in graph capture)
            let layer_bwd_start = if !use_graph { Some(std::time::Instant::now()) } else { None };

            blocks[layer_idx]
                .backward_nf4(
                    &training_state.layer_inputs[layer_idx],
                    grad_output,
                    grad_input,
                    // SAFETY: dereferences a raw pointer to the output scratch buffer that outlives this call and is not aliased by any other live reference; the GPU kernel is the sole writer for the duration of the launch.
                    unsafe { &mut *output_scratch_ptr },
                    seq_len,
                    stream,
                    shared_scratch,
                    grad_lora,
                )
                .ok()?;

            // PMAT-477: Fused clip (zero D2H sync) — graph-capture compatible
            if let Some(max_norm) = self.config.gradient_clip_norm {
                if let Some(ref clip_state) = self.lora_fused_clip {
                    super::super::fused_lora_clip::clip_lora_gradients_fused(
                        grad_lora, max_norm, clip_state, stream,
                    );
                } else if !use_graph {
                    // Sync fallback only when NOT in graph capture (sync breaks capture)
                    grad_lora.clip_gradients(max_norm, stream);
                }
            }

            blocks[layer_idx]
                .lora_optimizer_step(
                    &mut opt_states[layer_idx],
                    step,
                    lr,
                    0.9,
                    0.999,
                    1e-8,
                    0.01,
                    stream,
                    grad_lora,
                )
                .ok()?;

            // PMAT-483: Record per-layer backward time (skip during graph capture)
            record_layer_bwd_us(
                &mut training_state.profiler_layer_bwd_us,
                layer_idx,
                layer_bwd_start,
            );

            grad_output_is_a = !grad_output_is_a;
        }

        // PMAT-488: End graph capture and cache
        if use_graph {
            end_backward_capture(
                stream,
                seq_len,
                num_layers,
                &mut training_state.backward_graph_state,
            );
        }

        stream.synchronize().ok()?;

        Some(())
    }

    /// PMAT-488: replays the backward graph captured at this `seq_len`, when there is one.
    ///
    /// Returns `Some(true)` when it replayed, leaving the loop nothing to run.
    #[cfg(feature = "cuda")]
    fn replay_backward_graph(&mut self, seq_len: usize) -> Option<bool> {
        let stream = self.cuda_trainer.as_ref()?.stream();
        let training_state = self.gpu_training.as_ref()?;
        let Some(state) = training_state.backward_graph_state.as_ref() else {
            return Some(false);
        };
        if !super::super::backward_graph::use_backward_graph() || state.cached_seq_len != seq_len {
            return Some(false);
        }
        // Replay cached backward graph — all 28 layers in one launch
        super::super::backward_graph::replay_backward(state, stream)?;
        // Still need to increment step counter for LR scheduling
        self.nf4_lora_step += 1;
        stream.synchronize().ok()?;
        Some(true)
    }
}

/// R15a C3b: whether the backward may run under graph capture.
///
/// The syncing clip cannot run inside a capture, so a backward with a clip set is captured only
/// when the fused clip state exists (FALSIFY-LORA_TARGET_SELECTION_V1_006).
#[cfg(feature = "cuda")]
fn backward_graph_allowed(graph_enabled: bool, clips: bool, fused_clip: bool) -> bool {
    graph_enabled && (!clips || fused_clip)
}

/// PMAT-488: starts capturing the backward into a graph; a failed start leaves it ungraphed.
#[cfg(feature = "cuda")]
fn begin_backward_capture(stream: &trueno_gpu::driver::CudaStream) {
    if let Err(e) = stream.begin_capture(trueno_gpu::driver::CaptureMode::ThreadLocal) {
        eprintln!("[CUDA] Backward graph capture begin failed: {e} — falling back to ungraphed");
    }
}

/// PMAT-488: ends the capture and caches the instantiated graph for `seq_len`.
#[cfg(feature = "cuda")]
fn end_backward_capture(
    stream: &trueno_gpu::driver::CudaStream,
    seq_len: usize,
    num_layers: usize,
    cache: &mut Option<super::super::backward_graph::BackwardGraphState>,
) {
    match stream.end_capture() {
        Ok(graph) => match graph.instantiate() {
            Ok(exec) => {
                eprintln!("[CUDA] Backward graph captured: seq_len={seq_len}, {num_layers} layers");
                *cache = Some(super::super::backward_graph::BackwardGraphState {
                    exec,
                    cached_seq_len: seq_len,
                });
            }
            Err(e) => eprintln!("[CUDA] Backward graph instantiate failed: {e}"),
        },
        Err(e) => eprintln!("[CUDA] Backward graph end_capture failed: {e}"),
    }
}

/// PMAT-483: records one layer's backward time, when it was timed (never under graph capture).
#[cfg(feature = "cuda")]
fn record_layer_bwd_us(
    layer_bwd_us: &mut [u64],
    layer_idx: usize,
    start: Option<std::time::Instant>,
) {
    if let (Some(start), Some(slot)) = (start, layer_bwd_us.get_mut(layer_idx)) {
        *slot = start.elapsed().as_micros() as u64;
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::backward_graph_allowed;

    /// FALSIFY-LORA_TARGET_SELECTION_V1_006: a backward with a clip set is captured only with the
    /// fused clip.
    #[test]
    fn falsify_lora_target_selection_v1_006_clipped_backward_needs_fused_clip_to_capture() {
        // (graphs on, clip set, fused clip present, captured)
        let cases = [
            (false, false, false, false),
            (false, false, true, false),
            (false, true, false, false),
            (false, true, true, false),
            (true, false, false, true),
            (true, false, true, true),
            (true, true, false, false),
            (true, true, true, true),
        ];
        for (graph, clips, fused, captured) in cases {
            assert_eq!(
                backward_graph_allowed(graph, clips, fused),
                captured,
                "graphs on {graph}, clip set {clips}, fused clip {fused}"
            );
        }
    }
}
