#[cfg(feature = "cuda")]
use super::{CudaBlockScratch, InstructGpuTrainingState, InstructPipeline, Transformer};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_training::CudaTrainer;
#[cfg(feature = "cuda")]
use crate::transformer::CudaBlock;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CaptureMode, GpuBuffer};

#[cfg(feature = "cuda")]
impl InstructPipeline {
    /// GPU-accelerated forward pass saving layer inputs for backward.
    #[allow(unsafe_code)]
    pub(super) fn forward_cuda_training(
        model: &Transformer,
        token_ids: &[u32],
        trainer: &CudaTrainer,
        cuda_blocks: &mut [CudaBlock],
        training_state: &mut InstructGpuTrainingState,
        shared_scratch: &mut Option<CudaBlockScratch>,
    ) -> Option<()> {
        let seq_len = token_ids.len();
        let hidden_size = model.config.hidden_size;
        let max_seq_len = shared_scratch
            .as_ref()
            .map_or(model.config.max_position_embeddings.min(512), |s| s.max_seq_len(hidden_size));
        let seq_len = if seq_len > max_seq_len { max_seq_len } else { seq_len };
        if seq_len == 0 {
            return None;
        }

        // Embed on CPU, upload to GPU
        let hidden = model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        // PMAT-420 / entrenar#316: Use seq_len-sized fresh buffers (like inference forward).
        training_state.fwd_scratch_a = trainer
            .upload(hidden_slice)
            .map_err(|e| eprintln!("[CUDA] embed upload failed: {e}"))
            .ok()?;
        training_state.fwd_scratch_b = trainer
            .zeros(seq_len * hidden_size)
            .map_err(|e| eprintln!("[CUDA] scratch_b alloc failed: {e}"))
            .ok()?;

        let scratch_a_ptr: *mut GpuBuffer<f32> =
            std::ptr::from_mut(&mut training_state.fwd_scratch_a);
        let scratch_b_ptr: *mut GpuBuffer<f32> =
            std::ptr::from_mut(&mut training_state.fwd_scratch_b);
        let mut input_is_a = true;

        let stream = trainer.stream();
        // entrenar#318: GPU-side scratch + training state zeroing (PMAT-453 NaN cascade fix).
        if let Some(ref mut scratch) = shared_scratch.as_mut() {
            scratch.zero_forward_buffers(stream);
        }
        for b in [
            &mut training_state.grad_buf_a,
            &mut training_state.grad_buf_b,
            &mut training_state.grad_hidden_buf,
            &mut training_state.output_scratch,
            &mut training_state.logits_buf,
        ] {
            b.zero_async(stream).ok();
        }

        // PMAT-464: CUDA graph capture/replay (CUDA_GRAPH=1)
        static USE_CUDA_GRAPH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_graph =
            *USE_CUDA_GRAPH.get_or_init(|| std::env::var("CUDA_GRAPH").as_deref() == Ok("1"));

        for (i, block_) in cuda_blocks.iter().enumerate() {
            let _ = block_;
            let expected_len = seq_len * hidden_size;
            if training_state.layer_inputs[i].len() != expected_len {
                training_state.layer_inputs[i] = trainer
                    .zeros(expected_len)
                    .map_err(|e| eprintln!("[CUDA] layer_input prealloc L{i}: {e}"))
                    .ok()?;
            }
        }

        if use_graph
            && training_state.graph_cached_seq_len == seq_len
            && training_state.forward_graph_exec.is_some()
        {
            // === GRAPH REPLAY ===
            let exec = training_state.forward_graph_exec.as_ref().unwrap();
            exec.launch(stream.raw())
                .map_err(|e| eprintln!("[CUDA] Graph replay failed: {e}"))
                .ok()?;
            for _ in 0..cuda_blocks.len() {
                input_is_a = !input_is_a;
            }
        } else {
            // === Standard or first-capture forward ===
            let capturing = use_graph && training_state.graph_cached_seq_len != seq_len;
            if capturing {
                // PMAT-063: Pre-allocate cuBLAS workspace before graph capture
                if training_state.cublas_workspace.is_none() {
                    training_state.cublas_workspace =
                        super::super::gpu_backward_fallback::preallocate_cublas_workspace(trainer);
                }
                stream
                    .begin_capture(CaptureMode::ThreadLocal)
                    .map_err(|e| eprintln!("[CUDA] Graph capture begin failed: {e}"))
                    .ok()?;
            }

            for (i, block) in cuda_blocks.iter_mut().enumerate() {
                let (gpu_input, gpu_output) = unsafe {
                    if input_is_a {
                        (&*scratch_a_ptr, &mut *scratch_b_ptr)
                    } else {
                        (&*scratch_b_ptr, &mut *scratch_a_ptr)
                    }
                };

                training_state.layer_inputs[i]
                    .copy_from_buffer(gpu_input)
                    .map_err(|e| eprintln!("[CUDA] layer_input copy L{i}: {e}"))
                    .ok()?;

                // PMAT-483: Per-layer forward profiling
                training_state.profiler_layer_start = Some(std::time::Instant::now());

                if let Err(e) =
                    block.forward(gpu_input, gpu_output, seq_len, stream, shared_scratch.as_mut())
                {
                    eprintln!(
                        "[CUDA] Layer {i} forward failed: {e} (seq_len={seq_len} in={} out={} hidden={hidden_size})",
                        gpu_input.len(), gpu_output.len(),
                    );
                    if capturing {
                        let _ = stream.end_capture();
                    }
                    return None;
                }

                // PMAT-483: Record per-layer forward time
                if let Some(start) = training_state.profiler_layer_start.take() {
                    training_state.profiler_layer_fwd_us[i] = start.elapsed().as_micros() as u64;
                }

                input_is_a = !input_is_a;
            }

            if capturing {
                match stream.end_capture() {
                    Ok(graph) => match graph.instantiate() {
                        Ok(exec) => {
                            eprintln!(
                                "[CUDA] Graph captured: {} layers, seq_len={seq_len}",
                                cuda_blocks.len()
                            );
                            training_state.forward_graph_exec = Some(exec);
                            training_state.graph_cached_seq_len = seq_len;
                        }
                        Err(e) => {
                            eprintln!(
                                "[CUDA] Graph instantiate failed: {e} — using non-graph path"
                            );
                        }
                    },
                    Err(e) => {
                        eprintln!("[CUDA] Graph end_capture failed: {e} — using non-graph path");
                    }
                }
            }
        }

        let final_output = unsafe {
            if input_is_a {
                &*scratch_a_ptr
            } else {
                &*scratch_b_ptr
            }
        };

        // Save blocks output for RMSNorm backward
        if training_state.blocks_output.len() != final_output.len() {
            training_state.blocks_output = trainer
                .zeros(final_output.len())
                .map_err(|e| eprintln!("[CUDA] blocks_output realloc failed: {e}"))
                .ok()?;
        }
        training_state
            .blocks_output
            .copy_from_buffer(final_output)
            .map_err(|e| eprintln!("[CUDA] blocks_output copy: {e}"))
            .ok()?;

        crate::autograd::cuda_backward::rms_norm_forward(
            final_output,
            &training_state.final_norm_weight,
            &mut training_state.lm_head_hidden_buf,
            seq_len as u32,
            hidden_size as u32,
            stream,
        )
        .map_err(|e| eprintln!("[CUDA] GPU RMSNorm forward failed: {e}"))
        .ok()?;

        Some(())
    }
    /// GPU-accelerated forward pass (inference-only, no layer input saving).
    pub(super) fn forward_cuda_inference(
        model: &Transformer,
        token_ids: &[u32],
        trainer: &CudaTrainer,
        cuda_blocks: &mut [CudaBlock],
        shared_scratch: &mut Option<CudaBlockScratch>,
    ) -> Option<Vec<f32>> {
        let seq_len = token_ids.len();
        let hidden_size = model.config.hidden_size;

        let hidden = model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        let mut gpu_input = trainer.upload(hidden_slice).ok()?;
        let mut gpu_output = trainer.zeros(seq_len * hidden_size).ok()?;

        let stream = trainer.stream();
        for (i, block) in cuda_blocks.iter_mut().enumerate() {
            if let Err(e) =
                block.forward(&gpu_input, &mut gpu_output, seq_len, stream, shared_scratch.as_mut())
            {
                eprintln!("[CUDA] Layer {i} forward failed: {e}");
                return None;
            }
            std::mem::swap(&mut gpu_input, &mut gpu_output);
        }

        if let Err(e) = stream.synchronize() {
            eprintln!("[CUDA] Stream sync failed: {e}");
            return None;
        }

        let result_data = trainer.download(&gpu_input).ok()?;
        if result_data.iter().any(|v| !v.is_finite()) {
            return None;
        }

        let result_tensor = crate::Tensor::from_vec(result_data, false);
        let normed = model.norm.forward_batched(&result_tensor, seq_len, hidden_size);
        let normed_data = normed.data();
        let normed_slice = normed_data.as_slice().expect("contiguous normed");
        Some(normed_slice.to_vec())
    }
    /// Forward pass dispatching to GPU. Returns logits as flat Vec<f32> [seq_len, vocab_size].
    /// lm_head GEMM runs on GPU: hidden[seq, hidden] @ embed_T[hidden, vocab] -> logits[seq, vocab]
    pub(super) fn forward_logits_gpu(&mut self, token_ids: &[u32]) -> Option<Vec<f32>> {
        let seq_len = token_ids.len();
        let vocab_size = self.model.config().vocab_size;
        let hidden_size = self.model.config().hidden_size;

        if self.gpu_training.is_some() {
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return None,
            };
            let mut training = self.gpu_training.take();
            let result = Self::forward_cuda_training(
                &self.model,
                token_ids,
                trainer,
                blocks,
                training.as_mut().expect("gpu_training was Some"),
                &mut self.shared_scratch,
            );
            self.gpu_training = training;
            result?;
        } else {
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return None,
            };
            let normed_hidden = Self::forward_cuda_inference(
                &self.model,
                token_ids,
                trainer,
                blocks,
                &mut self.shared_scratch,
            )?;
            let training = self.gpu_training.as_mut()?;
            training
                .lm_head_hidden_buf
                .copy_from_host_at(&normed_hidden, 0)
                .map_err(|e| eprintln!("[CUDA] lm_head forward: hidden upload failed: {e}"))
                .ok()?;
        }

        let trainer = self.cuda_trainer.as_ref()?;
        let training = self.gpu_training.as_mut()?;
        let stream = trainer.stream();

        eprintln!("[CUDA] lm_head BT: hidden_len={} embed_len={} logits_len={} seq={seq_len} h={hidden_size} v={vocab_size}",
            training.lm_head_hidden_buf.len(), training.embed_original.len(), training.logits_buf.len());
        if let Err(e) = crate::autograd::cuda_forward::gemm_forward_bt(
            &training.lm_head_hidden_buf,
            &training.embed_original,
            &mut training.logits_buf,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        ) {
            eprintln!("[CUDA] lm_head forward GEMM (BT) failed: {e}");
            return None;
        }

        if let Err(e) = stream.synchronize() {
            eprintln!("[CUDA] lm_head forward sync failed: {e}");
            return None;
        }

        let full_logits = trainer
            .download(&training.logits_buf)
            .map_err(|e| eprintln!("[CUDA] lm_head forward: logits download failed: {e}"))
            .ok()?;
        Some(full_logits[..seq_len * vocab_size].to_vec())
    }
    /// PMAT-420: Inference forward + save layer inputs for backward.
    /// Uses inference-style fresh buffers (no NaN) but saves layer inputs for GPU backward.
    pub(super) fn forward_inference_saving_inputs(
        &mut self,
        token_ids: &[u32],
    ) -> Option<Vec<f32>> {
        let seq_len = token_ids.len();
        let hidden_size = self.model.config().hidden_size;
        let vocab_size = self.model.config().vocab_size;

        let trainer = self.cuda_trainer.as_ref()?;
        let blocks = self.cuda_blocks.as_mut()?;
        let stream = trainer.stream();

        let hidden = self.model.embed_tokens.forward(token_ids);
        let hidden_data = hidden.data();
        let hidden_slice = hidden_data.as_slice().expect("contiguous hidden");

        let mut gpu_input = trainer.upload(hidden_slice).ok()?;
        let mut gpu_output = trainer.zeros(seq_len * hidden_size).ok()?;

        for (i, block) in blocks.iter_mut().enumerate() {
            if let Some(ref mut training) = self.gpu_training {
                if i < training.layer_inputs.len() {
                    if training.layer_inputs[i].len() != gpu_input.len() {
                        if let Ok(buf) = trainer.zeros(gpu_input.len()) {
                            training.layer_inputs[i] = buf;
                        }
                    }
                    training.layer_inputs[i]
                        .copy_from_buffer(&gpu_input)
                        .map_err(|e| eprintln!("[CUDA] layer_input copy L{i}: {e}"))
                        .ok();
                }
            }

            if let Err(e) = block.forward(
                &gpu_input,
                &mut gpu_output,
                seq_len,
                stream,
                self.shared_scratch.as_mut(),
            ) {
                eprintln!("[CUDA] Layer {i} forward failed: {e}");
                return None;
            }
            std::mem::swap(&mut gpu_input, &mut gpu_output);
        }

        stream.synchronize().ok()?;

        // Save blocks_output for RMSNorm backward
        if let Some(ref mut training) = self.gpu_training {
            if training.blocks_output.len() != gpu_input.len() {
                if let Ok(buf) = trainer.zeros(gpu_input.len()) {
                    training.blocks_output = buf;
                }
            }
            training
                .blocks_output
                .copy_from_buffer(&gpu_input)
                .map_err(|e| eprintln!("[CUDA] blocks_output copy: {e}"))
                .ok();
        }

        let result = trainer.download(&gpu_input).ok()?;
        if result.iter().any(|v| !v.is_finite()) {
            eprintln!("[CUDA] NaN in forward output — inference-style forward failed");
            return None;
        }

        // CPU RMSNorm
        let result_tensor = crate::autograd::Tensor::from_vec(result, false);
        let normed = self.model.norm.forward_batched(&result_tensor, seq_len, hidden_size);
        let normed_data = normed.data();
        let normed_slice = normed_data.as_slice().expect("contiguous normed");

        // Save normed hidden for lm_head backward
        if let Some(ref mut training) = self.gpu_training {
            if let Ok(buf) = trainer.upload(normed_slice) {
                training.lm_head_hidden_buf = buf;
            }
        }

        // CPU lm_head
        let lm_weight = self.model.lm_head.as_ref().unwrap_or(&self.model.embed_tokens.weight);
        let lm_data = lm_weight.data();
        let lm_slice = lm_data.as_slice().expect("contiguous lm_head");
        let logits = crate::autograd::ops::matmul::matmul_nt_compute(
            normed_slice,
            lm_slice,
            seq_len,
            hidden_size,
            vocab_size,
        );
        Some(logits)
    }
    /// GPU forward with logits staying GPU-resident (KAIZEN-064).
    /// After this call, `training.logits_buf` contains logits on GPU. Returns true on success.
    pub(super) fn forward_logits_gpu_resident(&mut self, token_ids: &[u32]) -> bool {
        let seq_len = token_ids.len();
        let vocab_size = self.model.config().vocab_size;
        let hidden_size = self.model.config().hidden_size;

        if self.gpu_training.is_some() {
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => {
                    eprintln!("[RES-FALSE] no trainer/blocks");
                    return false;
                }
            };
            let mut training = self.gpu_training.take();
            let result = Self::forward_cuda_training(
                &self.model,
                token_ids,
                trainer,
                blocks,
                training.as_mut().expect("gpu_training was Some"),
                &mut self.shared_scratch,
            );
            self.gpu_training = training;
            if result.is_none() {
                eprintln!("[RES-FALSE] forward_cuda_training returned None");
                return false;
            }
        } else {
            let (trainer, blocks) = match (&self.cuda_trainer, &mut self.cuda_blocks) {
                (Some(ref t), Some(ref mut b)) => (t, b),
                _ => return false,
            };
            let normed_hidden = match Self::forward_cuda_inference(
                &self.model,
                token_ids,
                trainer,
                blocks,
                &mut self.shared_scratch,
            ) {
                Some(h) => h,
                None => return false,
            };
            let training = match self.gpu_training.as_mut() {
                Some(t) => t,
                None => return false,
            };
            if training.lm_head_hidden_buf.copy_from_host_at(&normed_hidden, 0).is_err() {
                eprintln!("[CUDA] lm_head forward: hidden upload failed");
                return false;
            }
        }

        let (trainer, training) = match (&self.cuda_trainer, &mut self.gpu_training) {
            (Some(ref t), Some(ref mut tr)) => (t, tr),
            _ => {
                eprintln!("[RES-FALSE] no trainer/training");
                return false;
            }
        };

        let stream = trainer.stream();

        if crate::autograd::cuda_forward::gemm_forward_bt(
            &training.lm_head_hidden_buf,
            &training.embed_original,
            &mut training.logits_buf,
            seq_len as u32,
            hidden_size as u32,
            vocab_size as u32,
            stream,
        )
        .is_err()
        {
            eprintln!("[CUDA] lm_head forward GEMM (BT) failed");
            eprintln!("[RES-FALSE] BT GEMM failed");
            return false;
        }

        true
    }
}
