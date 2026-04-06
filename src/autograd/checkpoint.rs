//! Gradient checkpointing for memory-efficient training
//!
//! Gradient checkpointing trades compute for memory by recomputing intermediate
//! activations during the backward pass instead of storing them.
//!
//! ## How It Works
//!
//! 1. During forward pass, only inputs to checkpointed segments are saved
//! 2. During backward pass, the forward pass is recomputed to get activations
//! 3. Memory usage scales with O(sqrt(N)) instead of O(N) for N layers
//!
//! ## Example
//!
//! ```ignore
//! use entrenar::autograd::checkpoint::{checkpoint, CheckpointConfig};
//!
//! // Wrap a computation in a checkpoint
//! let output = checkpoint(|| {
//!     let h1 = layer1.forward(&input);
//!     let h2 = layer2.forward(&h1);
//!     layer3.forward(&h2)
//! }, &input);
//! ```

use crate::autograd::graph_opt::OpType;
use crate::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

/// Configuration for gradient checkpointing
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Whether checkpointing is enabled
    pub enabled: bool,
    /// Number of segments to divide the model into
    pub num_segments: usize,
    /// Whether to use selective checkpointing (only checkpoint attention)
    pub selective: bool,
}

impl CheckpointConfig {
    /// Create new config with checkpointing enabled
    pub fn enabled(num_segments: usize) -> Self {
        Self { enabled: true, num_segments, selective: false }
    }

    /// Create config with checkpointing disabled
    pub fn disabled() -> Self {
        Self { enabled: false, num_segments: 1, selective: false }
    }

    /// Enable selective checkpointing (only attention layers)
    pub fn with_selective(mut self) -> Self {
        self.selective = true;
        self
    }
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// A checkpointed computation segment
///
/// Stores the input tensor and a function to recompute the forward pass.
/// During backward, the forward pass is recomputed to recover activations.
pub struct CheckpointedSegment {
    /// Input tensor (saved for recomputation)
    input: Tensor,
    /// Output tensor (computed lazily or cached)
    output: RefCell<Option<Tensor>>,
    /// Whether this segment has been checkpointed
    is_checkpointed: bool,
}

impl CheckpointedSegment {
    /// Create a new checkpointed segment
    pub fn new(input: Tensor, is_checkpointed: bool) -> Self {
        Self { input, output: RefCell::new(None), is_checkpointed }
    }

    /// Get the input tensor
    pub fn input(&self) -> &Tensor {
        &self.input
    }

    /// Check if this segment is checkpointed
    pub fn is_checkpointed(&self) -> bool {
        self.is_checkpointed
    }

    /// Set the output (used during forward pass)
    pub fn set_output(&self, output: Tensor) {
        *self.output.borrow_mut() = Some(output);
    }

    /// Get the output (returns None if not computed yet)
    pub fn output(&self) -> Option<Tensor> {
        contract_pre_output!();
        self.output.borrow().clone()
    }

    /// Clear the output to free memory
    pub fn clear_output(&self) {
        *self.output.borrow_mut() = None;
    }
}

/// Checkpoint manager for coordinating checkpointed segments
pub struct CheckpointManager {
    /// Configuration
    config: CheckpointConfig,
    /// Segments in order
    segments: Vec<Rc<CheckpointedSegment>>,
    /// Current segment index during forward pass
    current_segment: RefCell<usize>,
    /// Memory saved (estimated bytes)
    memory_saved: RefCell<usize>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(config: CheckpointConfig) -> Self {
        Self {
            config,
            segments: Vec::new(),
            current_segment: RefCell::new(0),
            memory_saved: RefCell::new(0),
        }
    }

    /// Check if checkpointing is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the number of segments
    pub fn num_segments(&self) -> usize {
        self.config.num_segments
    }

    /// Register a new segment
    pub fn register_segment(&mut self, input: Tensor) -> Rc<CheckpointedSegment> {
        let idx = self.segments.len();
        let should_checkpoint = self.config.enabled && self.should_checkpoint_segment(idx);

        let segment = Rc::new(CheckpointedSegment::new(input, should_checkpoint));
        self.segments.push(segment.clone());

        // Track memory savings
        if should_checkpoint {
            // Estimate: we save the intermediate activations
            // For now, just track a placeholder value
            *self.memory_saved.borrow_mut() += 1;
        }

        segment
    }

    /// Determine if a segment should be checkpointed
    fn should_checkpoint_segment(&self, segment_idx: usize) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Checkpoint every N segments based on config
        let checkpoint_interval = self.segments.len().max(1) / self.config.num_segments.max(1);
        if checkpoint_interval == 0 {
            return true; // Checkpoint all if interval is 0
        }

        segment_idx.is_multiple_of(checkpoint_interval)
    }

    /// Get estimated memory saved (number of checkpointed segments)
    pub fn memory_saved_segments(&self) -> usize {
        *self.memory_saved.borrow()
    }

    /// Clear all segments (call after backward pass)
    pub fn clear(&mut self) {
        for segment in &self.segments {
            segment.clear_output();
        }
        self.segments.clear();
        *self.current_segment.borrow_mut() = 0;
    }

    /// Get total number of registered segments
    pub fn total_segments(&self) -> usize {
        self.segments.len()
    }
}

/// Run a computation with gradient checkpointing
///
/// The function `f` is executed during forward pass. During backward pass,
/// if checkpointing is enabled, `f` will be re-executed to recompute activations.
///
/// # Arguments
///
/// * `f` - Function that computes the forward pass
/// * `input` - Input tensor (saved for recomputation)
///
/// # Returns
///
/// The output tensor from the computation
pub fn checkpoint<F>(f: F, input: &Tensor) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    // Simply run the function - actual checkpointing happens in training loop
    f(input)
}

/// Run a computation with explicit checkpointing control
///
/// # Arguments
///
/// * `f` - Function that computes the forward pass
/// * `input` - Input tensor
/// * `should_checkpoint` - Whether to enable checkpointing for this segment
pub fn checkpoint_if<F>(f: F, input: &Tensor, should_checkpoint: bool) -> Tensor
where
    F: Fn(&Tensor) -> Tensor,
{
    if should_checkpoint {
        // In a full implementation, we would save `input` and `f` for recomputation
        // For now, just run the function
        f(input)
    } else {
        f(input)
    }
}

/// Estimate memory savings from checkpointing
///
/// # Arguments
///
/// * `num_layers` - Number of transformer layers
/// * `hidden_size` - Hidden dimension
/// * `seq_len` - Sequence length
/// * `batch_size` - Batch size
/// * `num_checkpoints` - Number of checkpoint segments
///
/// # Returns
///
/// Tuple of (memory_without_checkpoint, memory_with_checkpoint) in bytes
pub fn estimate_memory_savings(
    num_layers: usize,
    hidden_size: usize,
    seq_len: usize,
    batch_size: usize,
    num_checkpoints: usize,
) -> (usize, usize) {
    // Each activation: batch_size * seq_len * hidden_size * sizeof(f32)
    let activation_size = batch_size * seq_len * hidden_size * 4;

    // Without checkpointing: store all layer activations
    let memory_without = num_layers * activation_size;

    // With checkpointing: store only checkpoint boundaries + recompute cost
    // Memory scales as O(sqrt(N)) with optimal checkpointing
    let sqrt_layers = (num_layers as f64).sqrt().ceil() as usize;
    let memory_with = sqrt_layers.max(num_checkpoints) * activation_size;

    (memory_without, memory_with)
}

/// Calculate optimal number of checkpoints for given memory budget
///
/// Uses the formula: optimal_checkpoints = sqrt(num_layers)
pub fn optimal_checkpoints(num_layers: usize) -> usize {
    ((num_layers as f64).sqrt().ceil() as usize).max(1)
}

// ---------------------------------------------------------------------------
// Policy-based selective gradient checkpointing (GH-83)
// ---------------------------------------------------------------------------

/// Metadata about an operation, used by checkpoint policies to decide
/// whether to save or recompute its output activation.
#[derive(Debug, Clone)]
pub struct OperationInfo {
    /// The type of operation
    pub op_type: OpType,
    /// Output size in bytes (batch_size * elements * sizeof(f32))
    pub output_bytes: usize,
    /// Whether any input has a batch dimension (ndim > 2)
    pub has_batch_dim: bool,
    /// Layer index in the sequential model
    pub layer_index: usize,
}

impl OperationInfo {
    /// Create operation info for a given op type and output size
    pub fn new(op_type: OpType, output_bytes: usize) -> Self {
        Self { op_type, output_bytes, has_batch_dim: false, layer_index: 0 }
    }

    /// Set whether this operation has batch dimensions
    pub fn with_batch_dim(mut self, has_batch: bool) -> Self {
        self.has_batch_dim = has_batch;
        self
    }

    /// Set the layer index
    pub fn with_layer_index(mut self, index: usize) -> Self {
        self.layer_index = index;
        self
    }
}

/// Policy for deciding which activations to save vs recompute during
/// gradient checkpointing.
///
/// Implementations control the memory/compute tradeoff by returning `true`
/// from `should_save` for operations whose outputs should be cached.
pub trait CheckpointPolicy {
    /// Returns true if this operation's output should be saved (not recomputed)
    fn should_save(&self, op: &OperationInfo) -> bool;

    /// Estimated relative cost of recomputing this operation (default: 1.0)
    fn recompute_cost(&self, _op: &OperationInfo) -> f64 {
        1.0
    }
}

/// Save everything — maximum memory usage, no recomputation overhead.
pub struct SaveAll;

impl CheckpointPolicy for SaveAll {
    fn should_save(&self, _op: &OperationInfo) -> bool {
        true
    }
}

/// Save nothing — minimum memory usage, full recomputation during backward.
pub struct SaveNothing;

impl CheckpointPolicy for SaveNothing {
    fn should_save(&self, _op: &OperationInfo) -> bool {
        false
    }
}

/// Save only matrix multiplication results (most expensive to recompute).
pub struct SaveMatmuls;

impl CheckpointPolicy for SaveMatmuls {
    fn should_save(&self, op: &OperationInfo) -> bool {
        matches!(op.op_type, OpType::Matmul | OpType::Attention)
    }

    fn recompute_cost(&self, op: &OperationInfo) -> f64 {
        match op.op_type {
            OpType::Matmul => 100.0,
            OpType::Attention => 150.0,
            OpType::Add
            | OpType::Mul
            | OpType::Scale
            | OpType::Sum
            | OpType::Relu
            | OpType::Gelu
            | OpType::Softmax
            | OpType::LayerNorm
            | OpType::Constant => 1.0,
        }
    }
}

/// Save matmuls that do NOT have batch dimensions (common in transformers).
/// These are typically the most expensive weight-projection operations.
pub struct SaveUnbatchedMatmuls;

impl CheckpointPolicy for SaveUnbatchedMatmuls {
    fn should_save(&self, op: &OperationInfo) -> bool {
        matches!(op.op_type, OpType::Matmul | OpType::Attention) && !op.has_batch_dim
    }
}

/// Save activations at regular intervals (every N layers).
/// Uses the binomial checkpointing strategy: checkpoint sqrt(N) layers
/// for O(sqrt(N)) memory with O(1) extra forward passes.
pub struct BinomialCheckpointing {
    /// Total number of layers in the model
    pub num_layers: usize,
}

impl BinomialCheckpointing {
    /// Compute the indices that should be checkpointed
    pub fn checkpoint_indices(&self) -> Vec<usize> {
        let num_checkpoints = optimal_checkpoints(self.num_layers);
        let interval = self.num_layers / num_checkpoints.max(1);
        (0..self.num_layers).step_by(interval.max(1)).collect()
    }
}

impl CheckpointPolicy for BinomialCheckpointing {
    fn should_save(&self, op: &OperationInfo) -> bool {
        let indices = self.checkpoint_indices();
        indices.contains(&op.layer_index)
    }
}

/// Save activations up to a memory budget (in bytes).
pub struct MemoryBudget {
    /// Maximum total bytes for saved activations
    pub max_bytes: usize,
    /// Current bytes used (interior mutability for stateful tracking)
    used_bytes: RefCell<usize>,
}

impl MemoryBudget {
    /// Create a new memory budget policy
    pub fn new(max_bytes: usize) -> Self {
        Self { max_bytes, used_bytes: RefCell::new(0) }
    }

    /// Get the current bytes used
    pub fn used_bytes(&self) -> usize {
        *self.used_bytes.borrow()
    }

    /// Reset the used bytes counter
    pub fn reset(&self) {
        *self.used_bytes.borrow_mut() = 0;
    }
}

impl CheckpointPolicy for MemoryBudget {
    fn should_save(&self, op: &OperationInfo) -> bool {
        let current = *self.used_bytes.borrow();
        if current + op.output_bytes <= self.max_bytes {
            *self.used_bytes.borrow_mut() += op.output_bytes;
            true
        } else {
            false
        }
    }
}

/// Custom policy using a predicate function.
pub struct CustomPolicy<F: Fn(&OperationInfo) -> bool> {
    predicate: F,
}

impl<F: Fn(&OperationInfo) -> bool> CustomPolicy<F> {
    /// Create a custom policy from a predicate
    pub fn new(predicate: F) -> Self {
        Self { predicate }
    }
}

impl<F: Fn(&OperationInfo) -> bool> CheckpointPolicy for CustomPolicy<F> {
    fn should_save(&self, op: &OperationInfo) -> bool {
        (self.predicate)(op)
    }
}

/// Policy-based checkpoint manager that uses a `CheckpointPolicy` to
/// decide which activations to save vs recompute.
pub struct PolicyCheckpointManager {
    /// Activation storage (layer_index -> saved tensor)
    saved: Vec<Option<Tensor>>,
    /// Total bytes saved
    total_bytes_saved: usize,
    /// Number of layers
    num_layers: usize,
}

impl PolicyCheckpointManager {
    /// Create a new policy checkpoint manager
    pub fn new(num_layers: usize) -> Self {
        Self { saved: vec![None; num_layers], total_bytes_saved: 0, num_layers }
    }

    /// Record a forward activation, saving it if the policy says so
    pub fn record<P: CheckpointPolicy>(
        &mut self,
        layer_index: usize,
        activation: &Tensor,
        op_info: &OperationInfo,
        policy: &P,
    ) {
        if policy.should_save(op_info) && layer_index < self.num_layers {
            self.saved[layer_index] = Some(activation.clone());
            self.total_bytes_saved += op_info.output_bytes;
        }
    }

    /// Get a saved activation (returns None if it was not saved / needs recompute)
    pub fn get(&self, layer_index: usize) -> Option<&Tensor> {
        self.saved.get(layer_index).and_then(|s| s.as_ref())
    }

    /// Check if an activation is saved for a given layer
    pub fn is_saved(&self, layer_index: usize) -> bool {
        self.saved.get(layer_index).is_some_and(Option::is_some)
    }

    /// Get total bytes used by saved activations
    pub fn total_bytes(&self) -> usize {
        contract_pre_total_bytes!();
        self.total_bytes_saved
    }

    /// Get the number of saved activations
    pub fn num_saved(&self) -> usize {
        self.saved.iter().filter(|s| s.is_some()).count()
    }

    /// Clear all saved activations
    pub fn clear(&mut self) {
        self.saved.iter_mut().for_each(|s| *s = None);
        self.total_bytes_saved = 0;
    }

    /// Get the total number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// Estimate the memory/compute tradeoff for a given policy on a model.
///
/// Returns `(bytes_saved, bytes_used, recompute_overhead)` where:
/// - `bytes_saved` is the memory freed by not saving some activations
/// - `bytes_used` is the memory used by saved activations
/// - `recompute_overhead` is the estimated relative compute cost of recomputation
pub fn estimate_policy_tradeoff<P: CheckpointPolicy>(
    policy: &P,
    layer_infos: &[OperationInfo],
) -> (usize, usize, f64) {
    let mut bytes_saved = 0usize;
    let mut bytes_used = 0usize;
    let mut recompute_overhead = 0.0f64;

    for info in layer_infos {
        if policy.should_save(info) {
            bytes_used += info.output_bytes;
        } else {
            bytes_saved += info.output_bytes;
            recompute_overhead += policy.recompute_cost(info);
        }
    }

    (bytes_saved, bytes_used, recompute_overhead)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::scale;

    #[test]
    fn test_checkpoint_config_enabled() {
        let config = CheckpointConfig::enabled(4);
        assert!(config.enabled);
        assert_eq!(config.num_segments, 4);
        assert!(!config.selective);
    }

    #[test]
    fn test_checkpoint_config_disabled() {
        let config = CheckpointConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_checkpoint_config_selective() {
        let config = CheckpointConfig::enabled(4).with_selective();
        assert!(config.selective);
    }

    #[test]
    fn test_checkpointed_segment_new() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let segment = CheckpointedSegment::new(input, true);
        assert!(segment.is_checkpointed());
        assert!(segment.output().is_none());
    }

    #[test]
    fn test_checkpointed_segment_output() {
        let input = Tensor::from_vec(vec![1.0, 2.0], true);
        let segment = CheckpointedSegment::new(input, true);

        let output = Tensor::from_vec(vec![2.0, 4.0], true);
        segment.set_output(output.clone());

        assert!(segment.output().is_some());
        assert_eq!(segment.output().expect("operation should succeed").len(), 2);
    }

    #[test]
    fn test_checkpointed_segment_clear() {
        let input = Tensor::from_vec(vec![1.0], true);
        let segment = CheckpointedSegment::new(input, true);
        segment.set_output(Tensor::from_vec(vec![2.0], true));

        segment.clear_output();
        assert!(segment.output().is_none());
    }

    #[test]
    fn test_checkpoint_manager_new() {
        let config = CheckpointConfig::enabled(4);
        let manager = CheckpointManager::new(config);
        assert!(manager.is_enabled());
        assert_eq!(manager.num_segments(), 4);
    }

    #[test]
    fn test_checkpoint_manager_disabled() {
        let config = CheckpointConfig::disabled();
        let manager = CheckpointManager::new(config);
        assert!(!manager.is_enabled());
    }

    #[test]
    fn test_checkpoint_manager_register() {
        let config = CheckpointConfig::enabled(2);
        let mut manager = CheckpointManager::new(config);

        let input1 = Tensor::from_vec(vec![1.0], true);
        let input2 = Tensor::from_vec(vec![2.0], true);

        let seg1 = manager.register_segment(input1);
        let seg2 = manager.register_segment(input2);

        assert_eq!(manager.total_segments(), 2);
        assert_eq!(seg1.input().len(), 1);
        assert_eq!(seg2.input().len(), 1);
    }

    #[test]
    fn test_checkpoint_manager_clear() {
        let config = CheckpointConfig::enabled(2);
        let mut manager = CheckpointManager::new(config);

        manager.register_segment(Tensor::from_vec(vec![1.0], true));
        manager.register_segment(Tensor::from_vec(vec![2.0], true));

        manager.clear();
        assert_eq!(manager.total_segments(), 0);
    }

    #[test]
    fn test_checkpoint_function() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let output = checkpoint(|x| scale(x, 2.0), &input);
        assert_eq!(output.len(), 3);
        assert_eq!(output.data()[0], 2.0);
    }

    #[test]
    fn test_checkpoint_if_enabled() {
        let input = Tensor::from_vec(vec![1.0, 2.0], true);
        let output = checkpoint_if(|x| scale(x, 3.0), &input, true);
        assert_eq!(output.data()[0], 3.0);
    }

    #[test]
    fn test_checkpoint_if_disabled() {
        let input = Tensor::from_vec(vec![1.0, 2.0], true);
        let output = checkpoint_if(|x| scale(x, 3.0), &input, false);
        assert_eq!(output.data()[0], 3.0);
    }

    #[test]
    fn test_estimate_memory_savings() {
        let (without, with) = estimate_memory_savings(32, 4096, 512, 1, 6);

        // With checkpointing should use less memory
        assert!(with < without);

        // Sanity check: without checkpointing stores all layers
        // 32 layers * 512 seq * 4096 hidden * 4 bytes = 268,435,456 bytes
        assert_eq!(without, 32 * 512 * 4096 * 4);
    }

    #[test]
    fn test_optimal_checkpoints() {
        assert_eq!(optimal_checkpoints(1), 1);
        assert_eq!(optimal_checkpoints(4), 2);
        assert_eq!(optimal_checkpoints(16), 4);
        assert_eq!(optimal_checkpoints(32), 6);
        assert_eq!(optimal_checkpoints(64), 8);
    }

    #[test]
    fn test_memory_savings_formula() {
        // For 32 layers with optimal sqrt(32) ≈ 6 checkpoints
        let num_layers = 32;
        let checkpoints = optimal_checkpoints(num_layers);

        let (without, with) = estimate_memory_savings(num_layers, 1024, 128, 1, checkpoints);

        // Memory reduction factor should be approximately sqrt(N)
        let ratio = without as f64 / with as f64;
        assert!(ratio > 4.0); // Should be significant savings
    }

    #[test]
    fn test_checkpoint_preserves_computation() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);

        // Without checkpoint
        let direct = scale(&input, 2.5);

        // With checkpoint
        let checkpointed = checkpoint(|x| scale(x, 2.5), &input);

        // Results should be identical
        for i in 0..4 {
            assert_eq!(direct.data()[i], checkpointed.data()[i]);
        }
    }

    #[test]
    fn test_nested_checkpoints() {
        let input = Tensor::from_vec(vec![1.0, 2.0], true);

        let output = checkpoint(
            |x| {
                let h1 = scale(x, 2.0);
                checkpoint(|y| scale(y, 3.0), &h1)
            },
            &input,
        );

        // 1.0 * 2.0 * 3.0 = 6.0
        assert_eq!(output.data()[0], 6.0);
    }

    #[test]
    fn test_checkpoint_manager_memory_tracking() {
        let config = CheckpointConfig::enabled(2);
        let mut manager = CheckpointManager::new(config);

        for i in 0..4 {
            manager.register_segment(Tensor::from_vec(vec![i as f32], true));
        }

        // Should have tracked some memory savings
        assert!(manager.memory_saved_segments() > 0);
    }

    // --- Policy tests (GH-83) ---

    fn make_op(op_type: OpType, bytes: usize) -> OperationInfo {
        OperationInfo::new(op_type, bytes)
    }

    #[test]
    fn test_operation_info_builder() {
        let info =
            OperationInfo::new(OpType::Matmul, 1024).with_batch_dim(true).with_layer_index(5);
        assert_eq!(info.op_type, OpType::Matmul);
        assert_eq!(info.output_bytes, 1024);
        assert!(info.has_batch_dim);
        assert_eq!(info.layer_index, 5);
    }

    #[test]
    fn test_save_all_policy() {
        let policy = SaveAll;
        assert!(policy.should_save(&make_op(OpType::Add, 100)));
        assert!(policy.should_save(&make_op(OpType::Matmul, 10000)));
        assert!(policy.should_save(&make_op(OpType::Relu, 50)));
    }

    #[test]
    fn test_save_nothing_policy() {
        let policy = SaveNothing;
        assert!(!policy.should_save(&make_op(OpType::Add, 100)));
        assert!(!policy.should_save(&make_op(OpType::Matmul, 10000)));
        assert!(!policy.should_save(&make_op(OpType::Relu, 50)));
    }

    #[test]
    fn test_save_matmuls_policy() {
        let policy = SaveMatmuls;
        assert!(policy.should_save(&make_op(OpType::Matmul, 1000)));
        assert!(policy.should_save(&make_op(OpType::Attention, 2000)));
        assert!(!policy.should_save(&make_op(OpType::Add, 100)));
        assert!(!policy.should_save(&make_op(OpType::Relu, 50)));
        assert!(!policy.should_save(&make_op(OpType::Softmax, 100)));
    }

    #[test]
    fn test_save_matmuls_recompute_cost() {
        let policy = SaveMatmuls;
        assert!((policy.recompute_cost(&make_op(OpType::Matmul, 0)) - 100.0).abs() < f64::EPSILON);
        assert!(
            (policy.recompute_cost(&make_op(OpType::Attention, 0)) - 150.0).abs() < f64::EPSILON
        );
        assert!((policy.recompute_cost(&make_op(OpType::Add, 0)) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_save_unbatched_matmuls_policy() {
        let policy = SaveUnbatchedMatmuls;

        // No batch dim -> should save
        let unbatched = OperationInfo::new(OpType::Matmul, 1000).with_batch_dim(false);
        assert!(policy.should_save(&unbatched));

        // With batch dim -> should not save
        let batched = OperationInfo::new(OpType::Matmul, 1000).with_batch_dim(true);
        assert!(!policy.should_save(&batched));

        // Non-matmul -> should not save
        let add = OperationInfo::new(OpType::Add, 100).with_batch_dim(false);
        assert!(!policy.should_save(&add));
    }

    #[test]
    fn test_binomial_checkpointing_indices() {
        let policy = BinomialCheckpointing { num_layers: 16 };
        let indices = policy.checkpoint_indices();

        // sqrt(16) = 4 checkpoints, interval = 16/4 = 4
        assert_eq!(indices, vec![0, 4, 8, 12]);
    }

    #[test]
    fn test_binomial_checkpointing_policy() {
        let policy = BinomialCheckpointing { num_layers: 16 };

        let at_checkpoint = OperationInfo::new(OpType::Add, 100).with_layer_index(0);
        assert!(policy.should_save(&at_checkpoint));

        let not_at_checkpoint = OperationInfo::new(OpType::Add, 100).with_layer_index(1);
        assert!(!policy.should_save(&not_at_checkpoint));

        let at_checkpoint_4 = OperationInfo::new(OpType::Add, 100).with_layer_index(4);
        assert!(policy.should_save(&at_checkpoint_4));
    }

    #[test]
    fn test_memory_budget_policy() {
        let policy = MemoryBudget::new(500);

        // First op fits
        let op1 = make_op(OpType::Matmul, 200);
        assert!(policy.should_save(&op1));
        assert_eq!(policy.used_bytes(), 200);

        // Second op fits
        let op2 = make_op(OpType::Add, 200);
        assert!(policy.should_save(&op2));
        assert_eq!(policy.used_bytes(), 400);

        // Third op doesn't fit
        let op3 = make_op(OpType::Relu, 200);
        assert!(!policy.should_save(&op3));
        assert_eq!(policy.used_bytes(), 400);

        // Reset and try again
        policy.reset();
        assert_eq!(policy.used_bytes(), 0);
        assert!(policy.should_save(&op3));
    }

    #[test]
    fn test_custom_policy() {
        // Only save ops with output > 500 bytes
        let policy = CustomPolicy::new(|op: &OperationInfo| op.output_bytes > 500);

        assert!(!policy.should_save(&make_op(OpType::Add, 100)));
        assert!(policy.should_save(&make_op(OpType::Matmul, 1000)));
        assert!(!policy.should_save(&make_op(OpType::Relu, 500)));
        assert!(policy.should_save(&make_op(OpType::Softmax, 501)));
    }

    #[test]
    fn test_policy_checkpoint_manager_basic() {
        let mut manager = PolicyCheckpointManager::new(4);
        let policy = SaveAll;

        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let info = make_op(OpType::Matmul, 12);

        manager.record(0, &tensor, &info, &policy);
        assert!(manager.is_saved(0));
        assert!(!manager.is_saved(1));
        assert_eq!(manager.num_saved(), 1);
        assert_eq!(manager.total_bytes(), 12);

        // Retrieve saved activation
        let saved = manager.get(0).expect("key should exist");
        assert_eq!(saved.len(), 3);
    }

    #[test]
    fn test_policy_checkpoint_manager_selective() {
        let mut manager = PolicyCheckpointManager::new(4);
        let policy = SaveMatmuls;

        let t1 = Tensor::from_vec(vec![1.0], true);
        let t2 = Tensor::from_vec(vec![2.0], true);

        // Matmul -> saved
        manager.record(0, &t1, &make_op(OpType::Matmul, 4), &policy);
        // Add -> not saved
        manager.record(1, &t2, &make_op(OpType::Add, 4), &policy);

        assert!(manager.is_saved(0));
        assert!(!manager.is_saved(1));
        assert_eq!(manager.num_saved(), 1);
    }

    #[test]
    fn test_policy_checkpoint_manager_clear() {
        let mut manager = PolicyCheckpointManager::new(2);
        let policy = SaveAll;

        let t = Tensor::from_vec(vec![1.0], true);
        manager.record(0, &t, &make_op(OpType::Add, 4), &policy);

        manager.clear();
        assert_eq!(manager.num_saved(), 0);
        assert_eq!(manager.total_bytes(), 0);
        assert!(!manager.is_saved(0));
    }

    #[test]
    fn test_policy_checkpoint_manager_out_of_bounds() {
        let mut manager = PolicyCheckpointManager::new(2);
        let policy = SaveAll;

        let t = Tensor::from_vec(vec![1.0], true);
        // Layer index beyond capacity — should be a no-op
        manager.record(5, &t, &make_op(OpType::Add, 4), &policy);
        assert_eq!(manager.num_saved(), 0);
    }

    #[test]
    fn test_estimate_policy_tradeoff_save_all() {
        let policy = SaveAll;
        let infos = vec![
            make_op(OpType::Matmul, 1000),
            make_op(OpType::Add, 200),
            make_op(OpType::Relu, 200),
        ];

        let (saved, used, overhead) = estimate_policy_tradeoff(&policy, &infos);
        assert_eq!(saved, 0); // Everything saved
        assert_eq!(used, 1400);
        assert!((overhead - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_policy_tradeoff_save_nothing() {
        let policy = SaveNothing;
        let infos = vec![make_op(OpType::Matmul, 1000), make_op(OpType::Add, 200)];

        let (saved, used, overhead) = estimate_policy_tradeoff(&policy, &infos);
        assert_eq!(saved, 1200); // Nothing saved
        assert_eq!(used, 0);
        assert!(overhead > 0.0); // Must recompute everything
    }

    #[test]
    fn test_estimate_policy_tradeoff_save_matmuls() {
        let policy = SaveMatmuls;
        let infos = vec![
            make_op(OpType::Matmul, 1000),
            make_op(OpType::Add, 200),
            make_op(OpType::Relu, 200),
        ];

        let (saved, used, overhead) = estimate_policy_tradeoff(&policy, &infos);
        assert_eq!(used, 1000); // Only matmul saved
        assert_eq!(saved, 400); // Add + Relu not saved
        assert!(overhead > 0.0); // Recompute cost for add + relu
    }

    #[test]
    fn test_policy_checkpoint_manager_num_layers() {
        let manager = PolicyCheckpointManager::new(8);
        assert_eq!(manager.num_layers(), 8);
    }

    #[test]
    fn test_binomial_single_layer() {
        let policy = BinomialCheckpointing { num_layers: 1 };
        let indices = policy.checkpoint_indices();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_default_recompute_cost() {
        let policy = SaveAll;
        let info = make_op(OpType::Add, 100);
        assert!((policy.recompute_cost(&info) - 1.0).abs() < f64::EPSILON);
    }
}
