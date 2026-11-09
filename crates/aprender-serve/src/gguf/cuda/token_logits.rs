//! #4971: a greedy decode step that also hands back the logits its argmax read.
//!
//! [`OwnedQuantizedModelCuda::forward_gpu_resident_to_token_id`] chooses on one
//! of three paths. The CPU-argmax path (LM head bias or `CORRECTNESS_MODE=1`)
//! and the eager/graph-capture path both download the logits and take a host
//! argmax; the graph-replay path takes a GPU argmax over the workspace logits
//! buffer. Asking for the logits takes the same path, with the same kernels and
//! the same argmax, and copies the logits the argmax read to the host after it.

use super::{OwnedQuantizedKVCache, OwnedQuantizedModelCuda, RealizarError, Result};

impl OwnedQuantizedModelCuda {
    /// [`Self::forward_gpu_resident_to_token_id`], and with `logits_out`
    /// (#4971) the logits the argmax read, copied to the host after it.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_gpu_resident_to_token_id`], and a failed logits copy.
    pub fn forward_gpu_resident_to_token_id_reading(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        logits_out: Option<&mut Vec<f32>>,
    ) -> Result<u32> {
        let Some(out) = logits_out else {
            return self.forward_gpu_resident_to_token_id(token_id, cache, position);
        };
        if self.model.lm_head_bias.is_some() || correctness_mode() {
            *out = self.forward_gpu_resident(token_id, cache, position)?;
        } else if self.executor.is_profiling_enabled() || !self.executor.has_decode_graph() {
            self.eager_step_logits(token_id, cache, position, out)?;
        } else {
            let token = self.forward_gpu_resident_to_token_id(token_id, cache, position)?;
            out.clear();
            out.resize(self.model.lm_head_weight.out_dim, 0.0);
            self.executor.download_workspace_logits(out).map_err(|e| {
                step_failed(format!("the logits copy after the device argmax: {e}"))
            })?;
            return Ok(token);
        }
        Ok(host_argmax(out))
    }

    /// The eager/graph-capture step of `forward_gpu_resident_to_token_id`,
    /// leaving its logits in `out`.
    fn eager_step_logits(
        &mut self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        out: &mut Vec<f32>,
    ) -> Result<()> {
        let hidden_dim = self.model.config.hidden_dim;
        let intermediate_dim = self.model.layers[0].ffn_up_weight.out_dim;
        let num_layers = self.model.layers.len();
        let vocab_size = self.model.lm_head_weight.out_dim;
        let eps = self.model.config.eps;

        self.model.embed_into(token_id, &mut self.embed_buf);
        out.clear();
        out.resize(vocab_size, 0.0);
        self.executor
            .forward_all_layers_gpu_to_logits_graphed(
                &self.embed_buf,
                out,
                position as u32,
                num_layers,
                hidden_dim as u32,
                intermediate_dim as u32,
                vocab_size as u32,
                eps,
            )
            .map_err(|e| {
                step_failed(format!(
                    "forward_all_layers_gpu_to_logits_graphed failed: {e}"
                ))
            })?;
        cache.advance();
        Ok(())
    }
}

/// `CORRECTNESS_MODE=1`, read as `forward_gpu_resident_to_token_id` reads it.
fn correctness_mode() -> bool {
    static CORRECTNESS_MODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CORRECTNESS_MODE.get_or_init(|| {
        std::env::var("CORRECTNESS_MODE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// The host argmax of `forward_gpu_resident_to_token_id` (ties go to the last
/// maximum, as they do there).
fn host_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx as u32)
}

fn step_failed(reason: String) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: "forward_gpu_resident_to_token_id_reading".to_string(),
        reason,
    }
}

#[cfg(test)]
mod tests_4971 {
    use super::host_argmax;

    #[test]
    fn host_argmax_breaks_ties_to_the_last_maximum_as_the_decode_step_does() {
        // (logits, expected): the decode step's host argmax keeps the LAST max.
        let table: &[(&[f32], u32)] = &[
            (&[0.0, 3.0, 1.0], 1),
            (&[2.0, 2.0, 1.0], 1),
            (&[1.0, 5.0, 5.0, 5.0], 3),
            (&[], 0),
        ];
        for &(logits, want) in table {
            assert_eq!(host_argmax(logits), want, "{logits:?}");
        }
    }
}
