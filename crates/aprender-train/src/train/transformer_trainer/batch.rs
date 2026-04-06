//! Language model batch utilities

/// A batch of tokenized sequences for language model training.
///
/// # Memory Layout (ALB-100)
///
/// For causal LM, `target[i] = input[i+1]` — storing both wastes 50%.
/// `LMBatch` deduplicates by storing a single `tokens` buffer:
///
/// - **Shared layout** (`stride = seq_len + 1`): Used when all sequences are
///   the same length (the production pre-tokenized path). Input and target are
///   derived as overlapping slices with a 1-token offset.
///
/// - **Split layout** (`stride = 0`): Used when sequences have different
///   lengths (padding breaks the shift invariant). Stores `[input_ids...,
///   target_ids...]` concatenated, matching the legacy layout.
#[derive(Debug, Clone)]
pub struct LMBatch {
    /// Token storage. Layout depends on `stride`:
    /// - stride > 0 (shared): batch_size * stride tokens, input/target overlap
    /// - stride == 0 (split): batch_size * seq_len * 2 (input then target)
    tokens: Vec<u32>,
    /// Batch size
    pub batch_size: usize,
    /// Sequence length (tokens per input/target per batch item)
    pub seq_len: usize,
    /// Stride between batch items in shared layout, or 0 for split layout.
    stride: usize,
}

impl LMBatch {
    /// Create a new LM batch from token sequences.
    ///
    /// For causal LM, targets are inputs shifted by 1:
    /// ```text
    /// input:  [BOS, A, B, C, D]
    /// target: [A, B, C, D, EOS]
    /// ```
    ///
    /// When all sequences have the same length, uses shared layout (ALB-100):
    /// one buffer of `batch_size * (seq_len + 1)` tokens, saving ~50% memory.
    ///
    /// When sequences differ in length, falls back to split layout (padding
    /// breaks the overlap invariant at sequence boundaries).
    pub fn from_sequences(sequences: &[Vec<u32>], pad_id: u32, eos_id: u32) -> Self {
        if sequences.is_empty() {
            return Self { tokens: Vec::new(), batch_size: 0, seq_len: 0, stride: 0 };
        }

        let batch_size = sequences.len();
        let max_len = sequences.iter().map(Vec::len).max().unwrap_or(0);
        let seq_len = max_len.saturating_sub(1).max(1);

        // Check if all sequences have the same length — enables shared layout
        let uniform = sequences.iter().all(|s| s.len() == max_len);

        if uniform {
            // Shared layout: store raw tokens, input/target derived by offset
            let stride = seq_len + 1; // = max_len
            let mut tokens = Vec::with_capacity(batch_size * stride);

            for seq in sequences {
                // Copy the raw sequence (all max_len tokens)
                tokens.extend_from_slice(seq);
            }

            Self { tokens, batch_size, seq_len, stride }
        } else {
            // Split layout: separate input_ids then target_ids (legacy)
            let mut tokens = Vec::with_capacity(batch_size * seq_len * 2);

            // First half: input_ids
            for seq in sequences {
                for i in 0..seq_len {
                    if i < seq.len() - 1 {
                        tokens.push(seq[i]);
                    } else {
                        tokens.push(pad_id);
                    }
                }
            }

            // Second half: target_ids
            for seq in sequences {
                for i in 0..seq_len {
                    if i + 1 < seq.len() {
                        tokens.push(seq[i + 1]);
                    } else if i + 1 == seq.len() {
                        tokens.push(eos_id);
                    } else {
                        tokens.push(pad_id);
                    }
                }
            }

            Self { tokens, batch_size, seq_len, stride: 0 }
        }
    }

    /// Create a batch from a single sequence pair (for testing).
    ///
    /// Uses split layout since caller provides separate input/target vecs.
    pub fn single(input_ids: Vec<u32>, target_ids: Vec<u32>) -> Self {
        let seq_len = input_ids.len();
        let mut tokens = Vec::with_capacity(seq_len * 2);
        tokens.extend_from_slice(&input_ids);
        tokens.extend_from_slice(&target_ids);
        Self { tokens, batch_size: 1, seq_len, stride: 0 }
    }

    /// Get input IDs for a specific batch item.
    pub fn get_input(&self, batch_idx: usize) -> Option<&[u32]> {
        if batch_idx >= self.batch_size {
            return None;
        }
        if self.stride > 0 {
            // Shared layout: input is tokens[b*stride .. b*stride + seq_len]
            let start = batch_idx * self.stride;
            Some(&self.tokens[start..start + self.seq_len])
        } else {
            // Split layout: first half is input_ids
            let start = batch_idx * self.seq_len;
            Some(&self.tokens[start..start + self.seq_len])
        }
    }

    /// Get target IDs for a specific batch item.
    pub fn get_target(&self, batch_idx: usize) -> Option<&[u32]> {
        if batch_idx >= self.batch_size {
            return None;
        }
        if self.stride > 0 {
            // Shared layout: target is tokens[b*stride + 1 .. b*stride + 1 + seq_len]
            let start = batch_idx * self.stride + 1;
            Some(&self.tokens[start..start + self.seq_len])
        } else {
            // Split layout: second half is target_ids
            let offset = self.batch_size * self.seq_len;
            let start = offset + batch_idx * self.seq_len;
            Some(&self.tokens[start..start + self.seq_len])
        }
    }

    /// Total number of tokens in batch (input side).
    pub fn num_tokens(&self) -> usize {
        self.batch_size * self.seq_len
    }

    /// Returns true if this batch uses shared (deduplicated) token storage.
    #[cfg(test)]
    pub fn is_shared_layout(&self) -> bool {
        self.stride > 0
    }

    /// Returns true if the token buffer is non-empty.
    pub fn has_tokens(&self) -> bool {
        !self.tokens.is_empty()
    }
}
