
impl SlotManager {
    /// Create a new slot manager with the specified number of slots
    pub fn new(num_slots: usize, max_context_length: usize) -> Self {
        let slots = (0..num_slots).map(Slot::new).collect();
        Self {
            slots,
            max_context_length,
            next_request_id: 0,
        }
    }

    /// Get number of total slots
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Get number of idle slots
    pub fn num_idle_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.is_idle()).count()
    }

    /// Get number of active (non-idle) slots
    pub fn num_active_slots(&self) -> usize {
        self.slots.len() - self.num_idle_slots()
    }

    /// Find an idle slot
    pub fn find_idle_slot(&self) -> Option<usize> {
        self.slots.iter().position(Slot::is_idle)
    }

    /// Assign a request to an available slot
    ///
    /// Returns the slot ID if successful, None if no slots available.
    pub fn assign_request(
        &mut self,
        input_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> Option<(usize, u64)> {
        let slot_id = self.find_idle_slot()?;
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        self.slots[slot_id].assign(request_id, input_tokens, max_tokens);
        Some((slot_id, request_id))
    }

    /// Get a reference to a slot
    pub fn get_slot(&self, slot_id: usize) -> Option<&Slot> {
        self.slots.get(slot_id)
    }

    /// Get a mutable reference to a slot
    pub fn get_slot_mut(&mut self, slot_id: usize) -> Option<&mut Slot> {
        self.slots.get_mut(slot_id)
    }

    /// Get all active slots (non-idle)
    pub fn active_slots(&self) -> impl Iterator<Item = &Slot> {
        self.slots.iter().filter(|s| !s.is_idle())
    }

    /// Get all generating slots
    pub fn generating_slots(&self) -> impl Iterator<Item = &Slot> {
        self.slots.iter().filter(|s| s.is_generating())
    }

    /// Get slots ready for batch processing
    pub fn batch_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_generating())
            .map(|(i, _)| i)
            .collect()
    }

    /// Get server utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        if self.slots.is_empty() {
            0.0
        } else {
            self.num_active_slots() as f64 / self.slots.len() as f64
        }
    }

    /// Get aggregate tokens per second across all slots
    pub fn aggregate_tokens_per_second(&self) -> f64 {
        self.slots.iter().map(Slot::tokens_per_second).sum()
    }
}

// ============================================================================
// CONTINUOUS BATCHING (ubatch/sbatch per llama.cpp)
// ============================================================================
//
// llama.cpp's continuous batching system:
// - ubatch (micro-batch): Tokens processed in a single forward pass
// - sbatch (sequence batch): Multiple sequences grouped for batched inference
//
// This enables:
// - Dynamic batching: New sequences can join mid-inference
// - Efficient GPU utilization: Batch multiple decode steps together
// - Mixed prefill/decode: Process prefill and decode in same batch
// ============================================================================

/// Batch type for continuous batching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchType {
    /// Prefill batch (processing initial prompts)
    Prefill,
    /// Decode batch (generating tokens)
    Decode,
    /// Mixed prefill and decode
    Mixed,
}

impl Default for BatchType {
    fn default() -> Self {
        Self::Decode
    }
}

/// Token position within a batch
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BatchToken {
    /// Token ID
    pub token_id: u32,
    /// Sequence ID this token belongs to
    pub seq_idx: usize,
    /// Position within the sequence
    pub seq_pos: usize,
    /// Whether this is a prompt token (vs generated)
    pub is_prompt: bool,
}

impl BatchToken {
    /// Create a new batch token
    pub fn new(token_id: u32, seq_idx: usize, seq_pos: usize, is_prompt: bool) -> Self {
        Self {
            token_id,
            seq_idx,
            seq_pos,
            is_prompt,
        }
    }
}

/// Micro-batch (ubatch) - tokens for a single forward pass
///
/// Per llama.cpp: A ubatch contains tokens that will be processed together
/// in a single forward pass. Can contain tokens from multiple sequences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MicroBatch {
    /// Tokens in this micro-batch
    pub tokens: Vec<BatchToken>,
    /// Sequence indices included in this batch
    pub seq_indices: Vec<usize>,
    /// Batch type (prefill/decode/mixed)
    pub batch_type: BatchType,
    /// Maximum sequence length in batch
    pub max_seq_len: usize,
    /// Number of prompt tokens in batch
    pub n_prompt_tokens: usize,
    /// Number of decode tokens in batch
    pub n_decode_tokens: usize,
}

impl MicroBatch {
    /// Create a new empty micro-batch
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            seq_indices: Vec::new(),
            batch_type: BatchType::Decode,
            max_seq_len: 0,
            n_prompt_tokens: 0,
            n_decode_tokens: 0,
        }
    }

    /// Create a micro-batch with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            seq_indices: Vec::new(),
            batch_type: BatchType::Decode,
            max_seq_len: 0,
            n_prompt_tokens: 0,
            n_decode_tokens: 0,
        }
    }

    /// Add a token to the batch
    pub fn add_token(&mut self, token: BatchToken) {
        if token.is_prompt {
            self.n_prompt_tokens += 1;
        } else {
            self.n_decode_tokens += 1;
        }

        // Track sequence index
        if !self.seq_indices.contains(&token.seq_idx) {
            self.seq_indices.push(token.seq_idx);
        }

        // Update max sequence length
        self.max_seq_len = self.max_seq_len.max(token.seq_pos + 1);

        self.tokens.push(token);

        // Update batch type
        self.update_batch_type();
    }

    /// Update batch type based on token composition
    fn update_batch_type(&mut self) {
        self.batch_type = match (self.n_prompt_tokens > 0, self.n_decode_tokens > 0) {
            (true, false) => BatchType::Prefill,
            (true, true) => BatchType::Mixed,
            // Both (false, true) and (false, false) result in Decode
            (false, _) => BatchType::Decode,
        };
    }

    /// Total number of tokens
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Number of sequences in batch
    pub fn num_sequences(&self) -> usize {
        self.seq_indices.len()
    }

    /// Check if batch is pure prefill
    pub fn is_prefill(&self) -> bool {
        self.batch_type == BatchType::Prefill
    }

    /// Check if batch is pure decode
    pub fn is_decode(&self) -> bool {
        self.batch_type == BatchType::Decode
    }

    /// Check if batch is mixed
    pub fn is_mixed(&self) -> bool {
        self.batch_type == BatchType::Mixed
    }

    /// Get token IDs as a vector (for model input)
    pub fn token_ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|t| t.token_id).collect()
    }

    /// Get sequence positions (for position embeddings)
    pub fn positions(&self) -> Vec<usize> {
        self.tokens.iter().map(|t| t.seq_pos).collect()
    }

    /// Clear the batch
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.seq_indices.clear();
        self.batch_type = BatchType::Decode;
        self.max_seq_len = 0;
        self.n_prompt_tokens = 0;
        self.n_decode_tokens = 0;
    }
}

/// Sequence batch entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceBatchEntry {
    /// Sequence index
    pub seq_idx: usize,
    /// Slot ID
    pub slot_id: usize,
    /// Request ID
    pub request_id: u64,
    /// Current position in sequence
    pub position: usize,
    /// Tokens to process (for prefill)
    pub tokens: Vec<u32>,
    /// Is this sequence in prefill or decode mode
    pub is_prefill: bool,
}

impl SequenceBatchEntry {
    /// Create new sequence batch entry
    pub fn new(seq_idx: usize, slot_id: usize, request_id: u64) -> Self {
        Self {
            seq_idx,
            slot_id,
            request_id,
            position: 0,
            tokens: Vec::new(),
            is_prefill: true,
        }
    }

    /// Set tokens for prefill
    pub fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.tokens = tokens;
        self
    }

    /// Set position
    pub fn at_position(mut self, position: usize) -> Self {
        self.position = position;
        self
    }

    /// Mark as decode (not prefill)
    pub fn decoding(mut self) -> Self {
        self.is_prefill = false;
        self
    }
}

/// Sequence batch (sbatch) - multiple sequences for batched inference
///
/// Per llama.cpp: Groups sequences that will be processed together.
/// Manages the mapping from batch positions to individual sequences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SequenceBatch {
    /// Sequences in this batch
    pub sequences: Vec<SequenceBatchEntry>,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Current batch utilization
    pub utilization: f64,
}

impl SequenceBatch {
    /// Create a new sequence batch with max size
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            sequences: Vec::with_capacity(max_batch_size),
            max_batch_size,
            utilization: 0.0,
        }
    }

    /// Add a sequence to the batch
    pub fn add_sequence(&mut self, entry: SequenceBatchEntry) -> bool {
        if self.sequences.len() >= self.max_batch_size {
            return false;
        }
        self.sequences.push(entry);
        self.update_utilization();
        true
    }

    /// Remove a sequence by index
    pub fn remove_sequence(&mut self, seq_idx: usize) -> Option<SequenceBatchEntry> {
        let pos = self.sequences.iter().position(|s| s.seq_idx == seq_idx)?;
        let entry = self.sequences.remove(pos);
        self.update_utilization();
        Some(entry)
    }

    /// Update utilization metric
    fn update_utilization(&mut self) {
        self.utilization = if self.max_batch_size > 0 {
            self.sequences.len() as f64 / self.max_batch_size as f64
        } else {
            0.0
        };
    }

    /// Number of sequences in batch
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Check if batch is empty
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Check if batch is full
    pub fn is_full(&self) -> bool {
        self.sequences.len() >= self.max_batch_size
    }

    /// Get prefill sequences
    pub fn prefill_sequences(&self) -> impl Iterator<Item = &SequenceBatchEntry> {
        self.sequences.iter().filter(|s| s.is_prefill)
    }

    /// Get decode sequences
    pub fn decode_sequences(&self) -> impl Iterator<Item = &SequenceBatchEntry> {
        self.sequences.iter().filter(|s| !s.is_prefill)
    }

    /// Count prefill sequences
    pub fn num_prefill(&self) -> usize {
        self.sequences.iter().filter(|s| s.is_prefill).count()
    }

    /// Count decode sequences
    pub fn num_decode(&self) -> usize {
        self.sequences.iter().filter(|s| !s.is_prefill).count()
    }

    /// Clear the batch
    pub fn clear(&mut self) {
        self.sequences.clear();
        self.utilization = 0.0;
    }

    /// Get sequence by index
    pub fn get(&self, seq_idx: usize) -> Option<&SequenceBatchEntry> {
        self.sequences.iter().find(|s| s.seq_idx == seq_idx)
    }

    /// Get mutable sequence by index
    pub fn get_mut(&mut self, seq_idx: usize) -> Option<&mut SequenceBatchEntry> {
        self.sequences.iter_mut().find(|s| s.seq_idx == seq_idx)
    }
}

/// Batch configuration for continuous batching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum tokens per micro-batch
    pub max_ubatch_tokens: usize,
    /// Maximum sequences per sequence batch
    pub max_sbatch_sequences: usize,
    /// Prefer pure decode batches (vs mixed)
    pub prefer_pure_decode: bool,
    /// Maximum context length
    pub max_context_length: usize,
    /// Enable dynamic batching (add sequences mid-inference)
    pub dynamic_batching: bool,
}
