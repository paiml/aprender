
impl CudaExecutor {
    /// LM head dispatch + logits download (extracted from forward_all_layers_gpu_to_logits)
    ///
    /// Detects LM head quantization type via size-based heuristic, dispatches to
    /// the correct GEMV kernel, and downloads final logits.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_lm_head_and_download(
        &mut self,
        normed_hidden: &GpuBuffer<f32>,
        logits: &mut [f32],
        vocab_size: u32,
        hidden_dim: u32,
        debug_enabled: bool,
    ) -> Result<(), GpuError> {
        let profiling = self.profiler.is_enabled();
        let timer_lm_head = if profiling {
            self.start_brick_id(trueno::BrickId::LmHead)
        } else {
            None
        };

        // 6. LM head projection on GPU (no sync)
        // PAR-056: Tiled kernel selection based on K dimension
        let lm_head_name = "output.weight".to_string();

        // PAR-058: Detect LM head quantization type using size-based detection
        // ALB-098: Use pool-aware lookup (pool entries or individual cache)
        let (lm_head_ptr, lm_head_buf_size) = self.get_quantized_weight_ptr_and_size(&lm_head_name)?;
        let lm_head_qtype =
            WeightQuantType::from_size(lm_head_buf_size, vocab_size as usize, hidden_dim as usize)
                .unwrap_or_else(|| {
                    self.quantized_weight_types
                        .get(&lm_head_name)
                        .and_then(|&t| WeightQuantType::from_ggml_type(t))
                        .unwrap_or(WeightQuantType::Q4K)
                });

        // CORRECTNESS-002: Debug LM head weight buffer
        if debug_enabled {
            let lm_head_size = lm_head_buf_size;
            let super_blocks_per_row = (hidden_dim as usize + 255) / 256;
            let bytes_per_row = super_blocks_per_row * 210;
            let expected_size = vocab_size as usize * bytes_per_row;
            eprintln!(
                "[CORRECTNESS-002] LM head: ptr=0x{:x}, size={}, expected={}, qtype={:?}",
                lm_head_ptr, lm_head_size, expected_size, lm_head_qtype
            );
            eprintln!(
                "[CORRECTNESS-002] LM head dims: vocab_size={}, hidden_dim={}, sb_per_row={}, bytes_per_row={}",
                vocab_size, hidden_dim, super_blocks_per_row, bytes_per_row
            );
        }

        // Allocate logits buffer
        let logits_gpu = GpuBuffer::<f32>::new(&self.context, vocab_size as usize)?;

        // PMAT-027: Invalidate Q8 cache — LM head input is normed_hidden (different from layer GEMVs).
        self.q8_activation_valid = false;

        // PAR-058: Dispatch to correct kernel based on detected quantization type
        self.dispatch_lm_head_kernel(
            lm_head_qtype, lm_head_ptr,
            normed_hidden, &logits_gpu,
            &lm_head_name, vocab_size, hidden_dim,
            debug_enabled,
        )?;

        // PAR-PROFILE: Stop LM head timer
        if profiling {
            self.stop_brick_id(timer_lm_head, 1);
        }

        // 7. Final sync and download - sync point #2 (only required sync)
        self.stream.synchronize()?;
        logits_gpu.copy_to_host(logits)?;

        Ok(())
    }

    /// Dispatch LM head GEMV to the correct quantization kernel
    #[allow(clippy::too_many_arguments)]
    fn dispatch_lm_head_kernel(
        &mut self,
        lm_head_qtype: WeightQuantType,
        lm_head_ptr: u64,
        normed_hidden: &GpuBuffer<f32>,
        logits_gpu: &GpuBuffer<f32>,
        lm_head_name: &str,
        vocab_size: u32,
        hidden_dim: u32,
        debug_enabled: bool,
    ) -> Result<(), GpuError> {
        match lm_head_qtype {
            WeightQuantType::Q6K => {
                self.q6k_gemv_into(
                    lm_head_ptr, normed_hidden, logits_gpu, vocab_size, hidden_dim,
                )?;
                if debug_enabled {
                    self.debug_q6k_lm_head(logits_gpu, normed_hidden, lm_head_name, vocab_size, hidden_dim)?;
                }
            },
            WeightQuantType::Q5K => {
                self.q5k_gemv_into(lm_head_ptr, normed_hidden, logits_gpu, vocab_size, hidden_dim)?;
            },
            WeightQuantType::Q8_0 => {
                self.q8_0_gemv_into(lm_head_ptr, normed_hidden, logits_gpu, vocab_size, hidden_dim)?;
                if debug_enabled {
                    self.debug_q8_0_lm_head(logits_gpu, vocab_size)?;
                }
            },
            WeightQuantType::Q5_0 => {
                self.q5_0_gemv_into(lm_head_ptr, normed_hidden, logits_gpu, vocab_size, hidden_dim)?;
            },
            WeightQuantType::Q4_0 => {
                self.q4_0_gemv_into(lm_head_ptr, normed_hidden, logits_gpu, vocab_size, hidden_dim)?;
            },
            WeightQuantType::Q4_1 => {
                self.q4_1_gemv_into(lm_head_ptr, normed_hidden, logits_gpu, vocab_size, hidden_dim)?;
            },
            WeightQuantType::Q4K => {
                self.q4k_gemv_into(lm_head_ptr, normed_hidden, logits_gpu, vocab_size, hidden_dim)?;
            },
            WeightQuantType::F32 => {
                self.f32_gemv_into(lm_head_ptr, normed_hidden, logits_gpu, vocab_size, hidden_dim)?;
            },
        }
        Ok(())
    }

    /// Debug Q6K LM head logits (CORRECTNESS-003/004/007)
    #[allow(clippy::too_many_lines)]
    fn debug_q6k_lm_head(
        &mut self,
        logits_gpu: &GpuBuffer<f32>,
        normed_hidden: &GpuBuffer<f32>,
        lm_head_name: &str,
        vocab_size: u32,
        hidden_dim: u32,
    ) -> Result<(), GpuError> {
        self.stream.synchronize()?;
        // Download ALL logits for full analysis
        let mut all_logits = vec![0.0f32; vocab_size as usize];
        logits_gpu.copy_to_host(&mut all_logits)?;

        eprintln!(
            "[CORRECTNESS-003] Q6K LM head logits[0..20]: {:?}",
            all_logits.get(..20).unwrap_or(&[])
        );

        // Find global argmax
        let (global_max_idx, global_max_val) = all_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, v)| (i, *v))
            .expect("CUDA operation failed");
        eprintln!(
            "[CORRECTNESS-007] Global argmax: idx={}, val={:.4}",
            global_max_idx, global_max_val
        );

        // Check for outliers: tokens with logit > 10
        let outliers: Vec<(usize, f32)> = all_logits
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > 10.0)
            .map(|(i, v)| (i, *v))
            .collect();
        if !outliers.is_empty() {
            eprintln!(
                "[CORRECTNESS-007] Logits > 10.0 ({} tokens): {:?}",
                outliers.len(),
                &outliers[..10.min(outliers.len())]
            );
        }

        // Check expected tokens (15='0', 16='1', 17='2', 18='3', 19='4')
        eprintln!(
            "[CORRECTNESS-007] Digit logits: 0={:.4}, 1={:.4}, 2={:.4}, 3={:.4}, 4={:.4}",
            all_logits[15], all_logits[16], all_logits[17], all_logits[18], all_logits[19]
        );

        let logits_debug = all_logits[..20].to_vec();
        // Check for all-zeros or all-same values (sign of kernel issue)
        let first = logits_debug[0];
        let all_same = logits_debug.iter().all(|&x| (x - first).abs() < 0.001);
        if all_same {
            eprintln!(
                "[CORRECTNESS-003] WARNING: All logits are identical ({})",
                first
            );
        }
        // Check argmax
        let (max_idx, max_val) = logits_debug
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("CUDA operation failed");
        eprintln!(
            "[CORRECTNESS-003] Q6K argmax in first 20: idx={}, val={}",
            max_idx, max_val
        );

        // CORRECTNESS-004: Compare GPU vs CPU logits for same input
        let mut normed_host = vec![0.0f32; hidden_dim as usize];
        normed_hidden.copy_to_host(&mut normed_host)?;

        if let Some(lm_head_buf) = self.quantized_weight_cache.get(lm_head_name) {
            let mut weight_bytes = vec![0u8; lm_head_buf.size_bytes()];
            lm_head_buf.copy_to_host(&mut weight_bytes)?;

            // CPU dequant + matmul for first 20 vocab entries
            let super_blocks_per_row = (hidden_dim as usize + 255) / 256;
            let bytes_per_row = super_blocks_per_row * 210;
            let mut cpu_logits = vec![0.0f32; 20];

            for vocab_idx in 0..20 {
                let row_start = vocab_idx * bytes_per_row;
                if row_start + bytes_per_row <= weight_bytes.len() {
                    let row_data = &weight_bytes[row_start..row_start + bytes_per_row];
                    let mut dot_sum = 0.0f32;

                    for sb in 0..super_blocks_per_row {
                        let sb_offset = sb * 210;
                        if sb_offset + 210 > row_data.len() {
                            break;
                        }
                        let d_bytes = [row_data[sb_offset], row_data[sb_offset + 1]];
                        let d = half::f16::from_le_bytes(d_bytes).to_f32();
                        let ql = &row_data[sb_offset + 2..sb_offset + 2 + 128];
                        let qh = &row_data[sb_offset + 130..sb_offset + 130 + 64];
                        let scales = &row_data[sb_offset + 194..sb_offset + 194 + 16];

                        for i in 0..256 {
                            let hidden_idx = sb * 256 + i;
                            if hidden_idx >= hidden_dim as usize {
                                break;
                            }
                            let ql_idx = i / 2;
                            let ql_shift = (i % 2) * 4;
                            let ql_val = ((ql[ql_idx] >> ql_shift) & 0xF) as i8;
                            let qh_idx = i / 4;
                            let qh_shift = (i % 4) * 2;
                            let qh_val = ((qh[qh_idx] >> qh_shift) & 0x3) as i8;
                            let q_val = ql_val | (qh_val << 4);
                            let q_centered = q_val - 32;
                            let scale_idx = i / 16;
                            let scale = (scales[scale_idx] as i8) as f32;
                            let weight = d * scale * (q_centered as f32);
                            dot_sum += weight * normed_host[hidden_idx];
                        }
                    }
                    cpu_logits[vocab_idx] = dot_sum;
                }
            }

            eprintln!("[CORRECTNESS-004] CPU logits[0..20]: {:?}", cpu_logits);
            let max_diff = logits_debug
                .iter()
                .zip(cpu_logits.iter())
                .map(|(g, c)| (g - c).abs())
                .fold(0.0f32, f32::max);
            eprintln!("[CORRECTNESS-004] Max GPU-CPU diff: {:.6}", max_diff);
        }
        Ok(())
    }

    /// Debug Q8_0 LM head logits
    fn debug_q8_0_lm_head(
        &mut self,
        logits_gpu: &GpuBuffer<f32>,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        self.stream.synchronize()?;
        let mut all_logits = vec![0.0f32; vocab_size as usize];
        logits_gpu.copy_to_host(&mut all_logits)?;
        eprintln!(
            "[GQA-DEBUG] Q8_0 LM head logits[0..20]: {:?}",
            all_logits.get(..20).unwrap_or(&[])
        );
        let (argmax_idx, argmax_val) = all_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, v)| (i, *v))
            .expect("empty logits");
        eprintln!(
            "[GQA-DEBUG] Q8_0 argmax: idx={}, val={:.4}",
            argmax_idx, argmax_val
        );
        eprintln!(
            "[GQA-DEBUG] Token 19='4' logit = {:.4}",
            all_logits.get(19).unwrap_or(&0.0)
        );
        Ok(())
    }
}
