impl CudaExecutor {
    /// PAR-054: Graph-captured forward pass for decode (M=1)
    ///
    /// Uses CUDA graph capture to reduce kernel launch overhead from ~280 launches
    /// to 1 graph launch (~10µs vs ~5.6ms overhead).
    ///
    /// First decode token: captures the kernel sequence into a graph
    /// Subsequent tokens: replays the captured graph with updated position
    ///
    /// # Performance
    ///
    /// - Without graphs: ~280 kernel launches × ~20µs = ~5.6ms overhead/token
    /// - With graphs: 1 graph launch × ~10µs = ~0.01ms overhead/token
    /// - Expected speedup: ~500x reduction in launch overhead
    #[allow(clippy::too_many_arguments)]
    pub fn forward_all_layers_gpu_to_logits_graphed(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<(), GpuError> {
        // Fast path: replay existing graph or fall back to non-graphed
        if let Some(result) = self.graphed_fast_path(
            input, logits, position, num_layers, hidden_dim,
            intermediate_dim, vocab_size, epsilon,
        )? {
            return result;
        }

        // Prepare device buffers for graph capture
        self.prepare_graph_buffers(input, position, hidden_dim, vocab_size)?;

        // PAR-054-FIX: Pre-load all kernel modules BEFORE graph capture
        self.preload_modules_for_capture(num_layers, hidden_dim, intermediate_dim, vocab_size)?;

        // PAR-064-DEBUG: Skip graph capture if SKIP_CUDA_GRAPH=1
        let skip_graph = std::env::var("SKIP_CUDA_GRAPH")
            .map(|v| v == "1")
            .unwrap_or(false);
        if skip_graph {
            eprintln!("[PAR-064-DEBUG] SKIP_CUDA_GRAPH=1, using non-graphed path");
            return self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            );
        }

        // trueno#243: Skip stream capture (code 901 poisons context on driver 570.207).
        // Use manual graph construction via cuGraphAddKernelNode instead.
        eprintln!(
            "[trueno#243] Manual graph construction: pos={}, has_graph={}, capture_failed={}, token_count={}",
            position, self.decode_graph.is_some(), self.graph_capture_failed, self.decode_token_count
        );
        self.begin_graph_recording();
        self.is_capturing = true;
        let eager_result = self.forward_workspace_captured(
            num_layers, hidden_dim, intermediate_dim, vocab_size, epsilon,
        );
        self.is_capturing = false;

        if let Err(eager_err) = eager_result {
            self.graph_recording = false;
            self.graph_capture_failed = true;
            eprintln!("[trueno#243] Eager forward during recording failed: {:?}", eager_err);
            // Re-prepare buffers and fall back to non-recorded eager
            let _ = self.stream.synchronize();
            self.prepare_graph_buffers(input, position, hidden_dim, vocab_size)?;
            return self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            );
        }

        // Eager pass succeeded — build graph from recorded kernels
        match self.end_graph_recording() {
            Ok(n) if n > 0 => {
                // Manual graph built! First token computed by eager pass.
                self.decode_token_count += 1;
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }

                // realizr#198 A/B TEST: Replay graph at SAME position.
                // If graph == eager, logits should be identical.
                // The eager pass already wrote the correct KV + logits.
                // Graph replay re-runs all kernels with the same input/position.
                static AB_TEST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                if position > 0 && *AB_TEST.get_or_init(|| std::env::var("GRAPH_AB_TEST").as_deref() == Ok("1")) {
                    // Helper: save a buffer to CPU
                    fn save_buf(buf: &Option<GpuBuffer<f32>>, n: usize) -> Option<Vec<f32>> {
                        buf.as_ref().map(|b| {
                            let mut v = vec![0.0f32; n];
                            b.copy_to_host(&mut v).ok();
                            v
                        })
                    }
                    fn save_buf_u8(buf: &Option<GpuBuffer<u8>>, n: usize) -> Option<Vec<u8>> {
                        buf.as_ref().map(|b| {
                            let mut v = vec![0u8; n];
                            b.copy_to_host(&mut v).ok();
                            v
                        })
                    }
                    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
                        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
                    }
                    fn max_diff_u8(a: &[u8], b: &[u8]) -> u32 {
                        a.iter().zip(b.iter()).map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs()).max().unwrap_or(0)
                    }

                    let hd = hidden_dim as usize;
                    let q_dim = (self.kv_num_heads * self.kv_head_dim) as usize;
                    let kv_dim = (self.kv_num_kv_heads * self.kv_head_dim) as usize;
                    let inter_dim = self.workspace.ffn_gate_buf.as_ref().map_or(0, |b| b.len());

                    // Save ALL workspace buffers from eager pass
                    let eager_hb1 = save_buf(&self.workspace.hidden_buf1, hd);
                    let eager_hb2 = save_buf(&self.workspace.hidden_buf2, hd);
                    let eager_normed = save_buf(&self.workspace.normed_hidden_buf, hd);
                    let eager_input_staging = save_buf(&self.workspace.input_staging, hd);
                    let eager_q = save_buf(&self.workspace.q_buf, q_dim);
                    let eager_k = save_buf(&self.workspace.k_buf, kv_dim);
                    let eager_v = save_buf(&self.workspace.v_buf, kv_dim);
                    let eager_attn_out = save_buf(&self.workspace.attn_out_buf, q_dim);
                    let eager_ffn_gate = save_buf(&self.workspace.ffn_gate_buf, inter_dim);
                    let eager_ffn_up = save_buf(&self.workspace.ffn_up_buf, inter_dim);
                    let eager_ffn_act = save_buf(&self.workspace.ffn_act_buf, inter_dim);
                    let eager_q8 = save_buf_u8(&self.workspace.q8_activation_buf, self.workspace.q8_activation_buf.as_ref().map_or(0, |b| b.len()));
                    // Save graph_input_buf
                    let eager_input = if let Some(ref ib) = self.graph_input_buf {
                        let mut v = vec![0.0f32; hd];
                        ib.copy_to_host(&mut v).ok();
                        Some(v)
                    } else { None };

                    // Replay with same buffers (input/position already set)
                    if let Some(ref graph_exec) = self.decode_graph {
                        self.stream.launch_graph(graph_exec)?;
                        self.stream.synchronize()?;
                    }

                    // Compare ALL buffers after graph replay
                    eprintln!("[realizr#198-AB] === PER-BUFFER DIVERGENCE SCAN (pos={}) ===", position);

                    // graph_input_buf should be UNCHANGED (read-only by graph)
                    if let (Some(ref ei), Some(ref ib)) = (&eager_input, &self.graph_input_buf) {
                        let mut gi = vec![0.0f32; hd];
                        ib.copy_to_host(&mut gi)?;
                        let d = max_diff(ei, &gi);
                        eprintln!("[realizr#198-AB] graph_input_buf   diff={:.8} {}", d, if d == 0.0 { "OK" } else { "CHANGED!" });
                    }

                    macro_rules! compare_buf {
                        ($name:expr, $eager:expr, $ws_field:expr, $n:expr) => {
                            if let (Some(ref ev), Some(ref buf)) = (&$eager, &$ws_field) {
                                let mut gv = vec![0.0f32; $n];
                                buf.copy_to_host(&mut gv)?;
                                let d = max_diff(ev, &gv);
                                eprintln!(
                                    "[realizr#198-AB] {:20} diff={:.8} eager[0..3]={:.4?} graph[0..3]={:.4?}",
                                    $name, d, &ev[..3.min(ev.len())], &gv[..3.min(gv.len())]
                                );
                            }
                        };
                    }

                    compare_buf!("hidden_buf1", eager_hb1, self.workspace.hidden_buf1, hd);
                    compare_buf!("hidden_buf2", eager_hb2, self.workspace.hidden_buf2, hd);
                    compare_buf!("input_staging", eager_input_staging, self.workspace.input_staging, hd);
                    compare_buf!("normed_hidden", eager_normed, self.workspace.normed_hidden_buf, hd);
                    compare_buf!("q_buf", eager_q, self.workspace.q_buf, q_dim);
                    compare_buf!("k_buf", eager_k, self.workspace.k_buf, kv_dim);
                    compare_buf!("v_buf", eager_v, self.workspace.v_buf, kv_dim);
                    compare_buf!("attn_out_buf", eager_attn_out, self.workspace.attn_out_buf, q_dim);
                    compare_buf!("ffn_gate_buf", eager_ffn_gate, self.workspace.ffn_gate_buf, inter_dim);
                    compare_buf!("ffn_up_buf", eager_ffn_up, self.workspace.ffn_up_buf, inter_dim);
                    compare_buf!("ffn_act_buf", eager_ffn_act, self.workspace.ffn_act_buf, inter_dim);

                    // Q8 activation cache (u8)
                    if let (Some(ref eq8), Some(ref q8buf)) = (&eager_q8, &self.workspace.q8_activation_buf) {
                        let mut gq8 = vec![0u8; eq8.len()];
                        q8buf.copy_to_host(&mut gq8)?;
                        let d = max_diff_u8(eq8, &gq8);
                        eprintln!("[realizr#198-AB] {:20} diff={} (u8 max abs)", "q8_activation_buf", d);
                    }

                    // Compare logits
                    if let Some(ref lb) = self.workspace.logits_buf {
                        let mut graph_logits = vec![0.0f32; vocab_size as usize];
                        lb.copy_to_host(&mut graph_logits)?;
                        let ld = max_diff(logits, &graph_logits);
                        let eager_argmax = logits.iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
                        let graph_argmax = graph_logits.iter().enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0);
                        eprintln!(
                            "[realizr#198-AB] {:20} diff={:.6} eager_argmax={} graph_argmax={} match={}",
                            "logits", ld, eager_argmax, graph_argmax, eager_argmax == graph_argmax
                        );
                    }

                    eprintln!("[realizr#198-AB] === END SCAN ===");
                }

                Ok(())
            },
            Ok(_) => {
                // No kernels recorded — recording not wired to all ops yet.
                // First token was computed by eager pass.
                eprintln!("[trueno#243] 0 kernels recorded, using eager path for subsequent tokens");
                self.graph_capture_failed = true;
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
            Err(graph_err) => {
                self.graph_capture_failed = true;
                eprintln!("[trueno#243] Manual graph build failed: {:?}", graph_err);
                self.stream.synchronize()?;
                if let Some(ref logits_buf) = self.workspace.logits_buf {
                    logits_buf.copy_to_host(logits)?;
                }
                Ok(())
            },
        }
    }

    /// Check early-exit conditions: replay, disabled, failed, prerequisites.
    /// Returns Some(Ok(())) to replay/fallback, None to continue to capture.
    #[allow(clippy::too_many_arguments)]
    fn graphed_fast_path(
        &mut self,
        input: &[f32],
        logits: &mut [f32],
        position: u32,
        num_layers: usize,
        hidden_dim: u32,
        intermediate_dim: u32,
        vocab_size: u32,
        epsilon: f32,
    ) -> Result<Option<Result<(), GpuError>>, GpuError> {
        // realizr#201: Graph replay enabled by default for sm_89+ (Ada/Hopper).
        // Uses manual graph construction (trueno#243), bypasses stream capture bug.
        // Verified correct: A/B test ALL 13 buffers diff=0 (realizr#198).
        // +26% decode throughput (333 vs 264 tok/s on RTX 4090).
        // Opt-out: SKIP_CUDA_GRAPH=1. Legacy opt-in: CUDA_GRAPH_ENABLE=1.
        static GRAPH_ENV: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        let env_override = *GRAPH_ENV.get_or_init(|| {
            if std::env::var("CUDA_GRAPH_ENABLE").as_deref() == Ok("1") {
                return Some(true);
            }
            if std::env::var("SKIP_CUDA_GRAPH").as_deref() == Ok("1") {
                return Some(false);
            }
            None // Use CC-based default
        });
        let graph_enabled = env_override.unwrap_or(self.gpu_profile.cc >= 89);
        if !graph_enabled {
            return Ok(Some(self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            )));
        }

        // PAR-054: Replay existing graph (SKIP_CUDA_GRAPH handled above)
        if self.decode_graph.is_some() && self.decode_token_count > 0 {
            if self.decode_token_count <= 3 && verbose() {
                eprintln!(
                    "[PAR-054] Graph replay #{} (pos={})",
                    self.decode_token_count, position
                );
            }
            return Ok(Some(self.forward_graphed_replay(input, logits, position)));
        }

        // PAR-118: Skip if previous capture failed
        if self.graph_capture_failed {
            return Ok(Some(self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            )));
        }

        // Check prerequisites for graph capture
        let use_workspace = self.has_workspace()
            && self.has_indexed_weights()
            && self.indexed_layer_weights.len() == num_layers;

        if !use_workspace {
            eprintln!("[PAR-054] Workspace not ready, using non-graphed path");
            return Ok(Some(self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            )));
        }

        if self.lm_head_ptr == 0 {
            eprintln!("[PAR-054] lm_head_ptr not set, using non-graphed path");
            return Ok(Some(self.forward_all_layers_gpu_to_logits(
                input, logits, position, num_layers, hidden_dim,
                intermediate_dim, vocab_size, epsilon,
            )));
        }

        Ok(None) // Proceed to capture
    }

    /// Initialize or update device buffers needed for graph capture/replay.
    fn prepare_graph_buffers(
        &mut self,
        input: &[f32],
        position: u32,
        hidden_dim: u32,
        vocab_size: u32,
    ) -> Result<(), GpuError> {
        // Position buffer
        if self.position_buf.is_none() {
            self.position_buf = Some(GpuBuffer::from_host(&self.context, &[position])?);
        } else {
            self.position_buf.as_mut().expect("just checked").copy_from_host(&[position])?;
        }

        // Seq_len buffer (position + 1)
        let seq_len = position + 1;
        if self.seq_len_buf.is_none() {
            self.seq_len_buf = Some(GpuBuffer::from_host(&self.context, &[seq_len])?);
        } else {
            self.seq_len_buf.as_mut().expect("just checked").copy_from_host(&[seq_len])?;
        }

        // Input buffer
        let hidden_size = hidden_dim as usize;
        let needs_realloc = self.graph_input_buf.is_none()
            || self.graph_input_buf.as_ref().expect("just checked").len() != hidden_size;
        if needs_realloc {
            self.graph_input_buf = Some(GpuBuffer::from_host(&self.context, input)?);
        } else {
            self.graph_input_buf.as_mut().expect("just checked").copy_from_host(input)?;
        }

        // Pre-allocate output buffers if needed
        if self.workspace.normed_hidden_buf.is_none() {
            self.workspace.normed_hidden_buf = Some(GpuBuffer::new(&self.context, hidden_size)?);
        }
        // PMAT-088: Check size, not just existence — batched decode may have resized
        // logits_buf to M*vocab_size, causing D2H copy length mismatch.
        let needs_logits = self.workspace.logits_buf.as_ref().map_or(true, |b| {
            b.len() != vocab_size as usize
        });
        if needs_logits {
            self.workspace.logits_buf = Some(GpuBuffer::new(&self.context, vocab_size as usize)?);
        }

        Ok(())
    }
}
