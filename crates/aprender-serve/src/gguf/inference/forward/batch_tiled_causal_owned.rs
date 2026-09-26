impl OwnedQuantizedModel {
    /// Tiled causal attention
    ///
    /// IMP-111c: Flash Attention with causal masking.
    /// For position i, only attends to positions 0..=i.
    #[allow(clippy::too_many_arguments)]
    pub fn tiled_causal_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        tile_size: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            crate::gguf::ops::attend_row_online_tiled(
                &q[i * head_dim..(i + 1) * head_dim],
                k,
                v,
                head_dim,
                i + 1,
                scale,
                tile_size,
                &mut output[i * head_dim..(i + 1) * head_dim],
            );
        }
        Ok(output)
    }

    /// PMAT-395 step 3: Bidirectional attention for encoder
    ///
    /// Same as tiled_causal_attention but attends to ALL positions
    /// (no causal mask). Used by T5/Whisper encoder where each
    /// position can attend to every other position.
    #[allow(clippy::too_many_arguments)]
    pub fn tiled_bidirectional_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        scale: f32,
        tile_size: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; seq_len * head_dim];
        for i in 0..seq_len {
            crate::gguf::ops::attend_row_online_tiled(
                &q[i * head_dim..(i + 1) * head_dim],
                k,
                v,
                head_dim,
                seq_len,
                scale,
                tile_size,
                &mut output[i * head_dim..(i + 1) * head_dim],
            );
        }
        Ok(output)
    }
    /// PMAT-395 step 4: Cross-attention for encoder-decoder
    ///
    /// Q comes from decoder, K/V come from encoder output.
    /// No causal mask — decoder attends to all encoder positions.
    /// Used in T5 decoder layers between self-attention and FFN.
    #[allow(clippy::too_many_arguments)]
    pub fn tiled_cross_attention(
        &self,
        q: &[f32],     // [decoder_len, head_dim]
        enc_k: &[f32], // [encoder_len, head_dim]
        enc_v: &[f32], // [encoder_len, head_dim]
        decoder_len: usize,
        encoder_len: usize,
        head_dim: usize,
        scale: f32,
        tile_size: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; decoder_len * head_dim];
        for i in 0..decoder_len {
            crate::gguf::ops::attend_row_online_tiled(
                &q[i * head_dim..(i + 1) * head_dim],
                enc_k,
                enc_v,
                head_dim,
                encoder_len,
                scale,
                tile_size,
                &mut output[i * head_dim..(i + 1) * head_dim],
            );
        }
        Ok(output)
    }
}

include!("batched.rs");
include!("batch_size.rs");
include!("acceleration.rs");
include!("attention.rs");
