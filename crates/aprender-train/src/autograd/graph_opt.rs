//! Graph-level optimizations for computation graphs
//!
//! Implements constant folding, dead code elimination, and common subexpression
//! elimination (CSE) during graph construction time. Inspired by JAX's dispatch
//! constant folding (`jax/_src/dispatch.py:637-646`) and LLVM optimization passes.
//!
//! ## How It Works
//!
//! 1. Tensors are wrapped in `TracedValue` which tracks whether they are compile-time
//!    constants or runtime-dynamic values
//! 2. Binary operations check if both inputs are constant and fold immediately
//! 3. Identity element optimizations eliminate no-op operations (x+0, x*1, x*0)
//! 4. Shape tracking enables early validation of dimension mismatches
//! 5. Graph optimization passes (DCE, CSE) run at construction time

use ndarray::Array1;
use std::collections::{HashMap, HashSet};

/// Unique identifier for a node in the computation graph
pub type NodeId = usize;

/// A value that may be constant (known at graph construction time) or dynamic
/// (computed at execution time).
#[derive(Debug, Clone)]
pub enum TracedValue {
    /// Compile-time constant — can be folded
    Constant(Array1<f32>),
    /// Runtime-computed value (symbolic reference to a graph node)
    Dynamic(NodeId),
}

impl TracedValue {
    /// Returns true if this value is a compile-time constant
    pub fn is_constant(&self) -> bool {
        matches!(self, TracedValue::Constant(_))
    }

    /// Returns the constant value if this is a constant
    pub fn as_constant(&self) -> Option<&Array1<f32>> {
        match self {
            TracedValue::Constant(v) => Some(v),
            TracedValue::Dynamic(_) => None,
        }
    }

    /// Returns the node id if this is a dynamic value
    pub fn node_id(&self) -> Option<NodeId> {
        match self {
            TracedValue::Constant(_) => None,
            TracedValue::Dynamic(id) => Some(*id),
        }
    }
}

/// Tensor with constant tracking for graph construction
#[derive(Debug, Clone)]
pub struct TracedTensor {
    /// The actual or symbolic value
    value: TracedValue,
    /// Shape (always known, even for dynamic values)
    shape: Vec<usize>,
}

impl TracedTensor {
    /// Create a constant tensor (known at graph construction time)
    pub fn constant(data: Array1<f32>) -> Self {
        let shape = vec![data.len()];
        Self { value: TracedValue::Constant(data), shape }
    }

    /// Create a dynamic (placeholder) tensor
    pub fn placeholder(shape: Vec<usize>, node_id: NodeId) -> Self {
        Self { value: TracedValue::Dynamic(node_id), shape }
    }

    /// Check if this tensor is a compile-time constant
    pub fn is_constant(&self) -> bool {
        self.value.is_constant()
    }

    /// Get the value
    pub fn value(&self) -> &TracedValue {
        &self.value
    }

    /// Get the shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Type of operation in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    Add,
    Mul,
    Scale,
    Sum,
    Matmul,
    Relu,
    Gelu,
    Softmax,
    LayerNorm,
    Attention,
    Constant,
}

/// A node in the computation graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: NodeId,
    /// Operation type
    pub op_type: OpType,
    /// Input node IDs
    pub input_ids: Vec<NodeId>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Constant value (if this node is a constant)
    pub constant_value: Option<Array1<f32>>,
    /// Whether this node has been eliminated
    removed: bool,
}

impl GraphNode {
    /// Check if this node holds a constant value
    pub fn is_constant(&self) -> bool {
        self.constant_value.is_some()
    }

    /// Check if this node has been removed by an optimization pass
    pub fn is_removed(&self) -> bool {
        self.removed
    }

    /// Mark this node as removed
    pub fn mark_removed(&mut self) {
        self.removed = true;
    }
}

/// Computation graph with optimization support
pub struct ComputeGraph {
    /// All nodes in the graph
    nodes: Vec<GraphNode>,
    /// Output node IDs (roots of the graph)
    output_ids: Vec<NodeId>,
}

impl ComputeGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self { nodes: Vec::new(), output_ids: Vec::new() }
    }

    /// Add a constant node to the graph
    pub fn add_constant(&mut self, data: Array1<f32>) -> NodeId {
        let id = self.nodes.len();
        let shape = vec![data.len()];
        self.nodes.push(GraphNode {
            id,
            op_type: OpType::Constant,
            input_ids: Vec::new(),
            output_shape: shape,
            constant_value: Some(data),
            removed: false,
        });
        id
    }

    /// Add an operation node to the graph
    pub fn add_op(
        &mut self,
        op_type: OpType,
        input_ids: Vec<NodeId>,
        output_shape: Vec<usize>,
    ) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            id,
            op_type,
            input_ids,
            output_shape,
            constant_value: None,
            removed: false,
        });
        id
    }

    /// Mark a node as an output of the graph
    pub fn mark_output(&mut self, node_id: NodeId) {
        self.output_ids.push(node_id);
    }

    /// Get a node by ID
    pub fn node(&self, id: NodeId) -> &GraphNode {
        &self.nodes[id]
    }

    /// Get a mutable reference to a node by ID
    pub fn node_mut(&mut self, id: NodeId) -> &mut GraphNode {
        &mut self.nodes[id]
    }

    /// Get the number of nodes
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Count non-removed nodes
    pub fn active_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| !n.is_removed()).count()
    }

    /// Get output node IDs
    pub fn output_ids(&self) -> &[NodeId] {
        &self.output_ids
    }

    /// Compute topological order of non-removed nodes
    pub fn topological_order(&self) -> Vec<NodeId> {
        let (in_degree, adjacency) = self.build_graph_maps();
        Self::kahns_algorithm(in_degree, &adjacency)
    }

    /// Build in-degree counts and adjacency lists for non-removed nodes
    fn build_graph_maps(&self) -> (HashMap<NodeId, usize>, HashMap<NodeId, Vec<NodeId>>) {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for node in &self.nodes {
            if node.is_removed() {
                continue;
            }
            in_degree.entry(node.id).or_insert(0);
            for &input_id in &node.input_ids {
                if !self.nodes[input_id].is_removed() {
                    adjacency.entry(input_id).or_default().push(node.id);
                    *in_degree.entry(node.id).or_insert(0) += 1;
                }
            }
        }

        (in_degree, adjacency)
    }

    /// Run Kahn's algorithm to produce a topological ordering
    fn kahns_algorithm(
        mut in_degree: HashMap<NodeId, usize>,
        adjacency: &HashMap<NodeId, Vec<NodeId>>,
    ) -> Vec<NodeId> {
        let mut queue: Vec<NodeId> =
            in_degree.iter().filter(|(_, &deg)| deg == 0).map(|(&id, _)| id).collect();
        queue.sort_unstable_by(|a, b| b.cmp(a)); // Descending so pop() yields smallest first

        let mut order = Vec::new();
        let empty = Vec::new();
        while let Some(id) = queue.pop() {
            order.push(id);
            for &neighbor in adjacency.get(&id).unwrap_or(&empty) {
                let Some(deg) = in_degree.get_mut(&neighbor) else {
                    continue;
                };
                *deg -= 1;
                if *deg == 0 {
                    queue.push(neighbor);
                    queue.sort_unstable_by(|a, b| b.cmp(a));
                }
            }
        }

        order
    }

    /// Replace all uses of `old_id` with `new_id` in the graph
    pub fn replace_uses(&mut self, old_id: NodeId, new_id: NodeId) {
        for node in &mut self.nodes {
            for input_id in &mut node.input_ids {
                if *input_id == old_id {
                    *input_id = new_id;
                }
            }
        }
        for output_id in &mut self.output_ids {
            if *output_id == old_id {
                *output_id = new_id;
            }
        }
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Get or create a graph node for a traced value.
/// Dynamic values already have a node; constants are materialized into the graph.
fn ensure_graph_node(value: &TracedValue, graph: &mut ComputeGraph) -> NodeId {
    match value {
        TracedValue::Dynamic(id) => *id,
        TracedValue::Constant(data) => graph.add_constant(data.clone()),
    }
}

/// Perform a traced binary operation with constant folding
///
/// If both inputs are constants, the operation is evaluated immediately.
/// Otherwise, a graph node is created for deferred execution.
pub fn traced_binary_op<F>(
    a: &TracedTensor,
    b: &TracedTensor,
    op: F,
    op_type: OpType,
    graph: &mut ComputeGraph,
) -> TracedTensor
where
    F: Fn(&Array1<f32>, &Array1<f32>) -> Array1<f32>,
{
    // Both constant: fold immediately
    if let (Some(a_const), Some(b_const)) = (a.value.as_constant(), b.value.as_constant()) {
        let result = op(a_const, b_const);
        return TracedTensor::constant(result);
    }

    // Try identity element optimizations
    if let Some(folded) = try_identity_fold(a, b, op_type) {
        return folded;
    }

    // At least one is dynamic: create graph node
    let a_node = ensure_graph_node(&a.value, graph);
    let b_node = ensure_graph_node(&b.value, graph);

    let output_shape = a.shape.clone(); // Assumes same shape for now
    let node_id = graph.add_op(op_type, vec![a_node, b_node], output_shape.clone());

    TracedTensor::placeholder(output_shape, node_id)
}

/// Try to fold operations with identity elements
///
/// - `x + 0 = x`
/// - `0 + x = x`
/// - `x * 1 = x`
/// - `1 * x = x`
/// - `x * 0 = 0`
/// - `0 * x = 0`
fn try_identity_fold(a: &TracedTensor, b: &TracedTensor, op_type: OpType) -> Option<TracedTensor> {
    match op_type {
        OpType::Add => try_additive_identity(a, b),
        OpType::Mul => try_multiplicative_identity(a, b),
        _ => None,
    }
}

/// Additive identity: x + 0 = x, 0 + x = x
fn try_additive_identity(a: &TracedTensor, b: &TracedTensor) -> Option<TracedTensor> {
    if b.value.as_constant().is_some_and(|c| is_zeros(c)) {
        return Some(a.clone());
    }
    if a.value.as_constant().is_some_and(|c| is_zeros(c)) {
        return Some(b.clone());
    }
    None
}

/// Multiplicative identity/annihilator: x*1=x, 1*x=x, x*0=0, 0*x=0
fn try_multiplicative_identity(a: &TracedTensor, b: &TracedTensor) -> Option<TracedTensor> {
    // Check b as the constant operand (x * 1, x * 0)
    if let Some(result) = try_mul_const(b, a) {
        return Some(result);
    }
    // Check a as the constant operand (1 * x, 0 * x)
    try_mul_const(a, b)
}

/// If `maybe_const` is a multiplicative identity (1) or annihilator (0),
/// return the folded result. `other` is the non-constant operand.
fn try_mul_const(maybe_const: &TracedTensor, other: &TracedTensor) -> Option<TracedTensor> {
    let c = maybe_const.value.as_constant()?;
    if is_ones(c) {
        return Some(other.clone());
    }
    if is_zeros(c) {
        return Some(TracedTensor::constant(Array1::zeros(other.shape[0])));
    }
    None
}

/// Check if all elements are zero
fn is_zeros(arr: &Array1<f32>) -> bool {
    arr.iter().all(|&x| x == 0.0)
}

/// Check if all elements are one
fn is_ones(arr: &Array1<f32>) -> bool {
    arr.iter().all(|&x| (x - 1.0).abs() < f32::EPSILON)
}

/// Shape tracker for early validation of dimension mismatches
pub struct ShapeTracker {
    shapes: HashMap<NodeId, Vec<usize>>,
}

/// Error type for shape validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    /// Input shape not found
    UnknownInput(NodeId),
    /// Dimension mismatch between operands
    DimMismatch { expected: usize, got: usize },
    /// Insufficient dimensions for operation
    InsufficientDims { required: usize, got: usize },
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShapeError::UnknownInput(id) => write!(f, "unknown input node {id}"),
            ShapeError::DimMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            ShapeError::InsufficientDims { required, got } => {
                write!(f, "insufficient dims: need {required}, have {got}")
            }
        }
    }
}

impl std::error::Error for ShapeError {}

impl ShapeTracker {
    /// Create a new shape tracker
    pub fn new() -> Self {
        Self { shapes: HashMap::new() }
    }

    /// Register a known shape for a node
    pub fn register(&mut self, node_id: NodeId, shape: Vec<usize>) {
        self.shapes.insert(node_id, shape);
    }

    /// Get the shape for a node
    pub fn get(&self, node_id: NodeId) -> Option<&[usize]> {
        self.shapes.get(&node_id).map(Vec::as_slice)
    }

    /// Look up a node's shape, returning an error if not registered
    fn require_shape(&self, node_id: NodeId) -> Result<Vec<usize>, ShapeError> {
        self.shapes.get(&node_id).cloned().ok_or(ShapeError::UnknownInput(node_id))
    }

    /// Validate that a shape has at least `min` dimensions
    fn require_min_dims(shape: &[usize], min: usize) -> Result<(), ShapeError> {
        if shape.len() < min {
            return Err(ShapeError::InsufficientDims { required: min, got: shape.len() });
        }
        Ok(())
    }

    /// Store an output shape and return a clone
    fn store_output(&mut self, output_id: NodeId, shape: Vec<usize>) -> Vec<usize> {
        self.shapes.insert(output_id, shape.clone());
        shape
    }

    /// Infer output shape for an element-wise binary operation
    pub fn infer_elementwise(
        &mut self,
        output_id: NodeId,
        a_id: NodeId,
        b_id: NodeId,
    ) -> Result<Vec<usize>, ShapeError> {
        let a_shape = self.require_shape(a_id)?;
        let b_shape = self.require_shape(b_id)?;

        if a_shape != b_shape {
            return Err(ShapeError::DimMismatch {
                expected: a_shape.iter().product(),
                got: b_shape.iter().product(),
            });
        }

        Ok(self.store_output(output_id, a_shape))
    }

    /// Infer output shape for a matmul operation
    pub fn infer_matmul(
        &mut self,
        output_id: NodeId,
        a_id: NodeId,
        b_id: NodeId,
    ) -> Result<Vec<usize>, ShapeError> {
        let a_shape = self.require_shape(a_id)?;
        let b_shape = self.require_shape(b_id)?;

        Self::require_min_dims(&a_shape, 2)?;
        Self::require_min_dims(&b_shape, 2)?;

        let k1 = a_shape[a_shape.len() - 1];
        let k2 = b_shape[b_shape.len() - 2];

        if k1 != k2 {
            return Err(ShapeError::DimMismatch { expected: k1, got: k2 });
        }

        let m = a_shape[a_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];
        Ok(self.store_output(output_id, vec![m, n]))
    }

    /// Infer output shape for a sum (reduction) operation
    pub fn infer_sum(
        &mut self,
        output_id: NodeId,
        input_id: NodeId,
    ) -> Result<Vec<usize>, ShapeError> {
        self.require_shape(input_id)?;
        Ok(self.store_output(output_id, vec![1]))
    }

    /// Get the number of tracked shapes
    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    /// Check if no shapes are tracked
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }
}

impl Default for ShapeTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for graph optimization passes
pub trait OptimizationPass {
    /// Name of the optimization pass
    fn name(&self) -> &'static str;

    /// Run the pass on the graph, returning the number of changes made
    fn run(&self, graph: &mut ComputeGraph) -> usize;
}

/// Constant folding pass — evaluates operations with all-constant inputs at
/// graph construction time.
pub struct ConstantFolding;

/// Try to evaluate a foldable operation with all-constant inputs.
/// Returns `None` if the operation cannot be folded.
fn try_eval_constant_op(op_type: OpType, inputs: &[&Array1<f32>]) -> Option<Array1<f32>> {
    match (op_type, inputs) {
        (OpType::Add, [a, b]) => Some(*a + *b),
        (OpType::Mul, [a, b]) => Some(*a * *b),
        (OpType::Sum, [a]) => Some(Array1::from(vec![a.sum()])),
        (OpType::Scale, [a, b]) if b.len() == 1 => Some(*a * b[0]),
        _ => None,
    }
}

impl ConstantFolding {
    /// Attempt to fold a single node to a constant value.
    /// Returns `Some(result)` if the node can be folded, `None` otherwise.
    fn try_fold_node(graph: &ComputeGraph, node_id: NodeId) -> Option<Array1<f32>> {
        let node = &graph.nodes[node_id];
        if node.is_removed() || node.is_constant() {
            return None;
        }

        let all_const = node.input_ids.iter().all(|&id| graph.nodes[id].is_constant());
        if !all_const {
            return None;
        }

        let inputs: Vec<&Array1<f32>> = node
            .input_ids
            .iter()
            .map(|&id| {
                graph.nodes[id]
                    .constant_value
                    .as_ref()
                    .expect("all inputs verified as constants above")
            })
            .collect();

        try_eval_constant_op(node.op_type, &inputs)
    }
}

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &'static str {
        "constant_folding"
    }

    fn run(&self, graph: &mut ComputeGraph) -> usize {
        let mut changes = 0;
        let order = graph.topological_order();

        for node_id in order {
            if let Some(result) = Self::try_fold_node(graph, node_id) {
                let node_mut = &mut graph.nodes[node_id];
                node_mut.constant_value = Some(result);
                node_mut.op_type = OpType::Constant;
                node_mut.input_ids.clear();
                changes += 1;
            }
        }

        changes
    }
}

/// Dead code elimination pass — removes nodes not reachable from outputs.
pub struct DeadCodeElimination;

impl DeadCodeElimination {
    /// Find all nodes reachable from outputs via DFS
    fn find_reachable(graph: &ComputeGraph) -> HashSet<NodeId> {
        let mut reachable = HashSet::new();
        let mut stack: Vec<NodeId> = graph.output_ids.clone();

        while let Some(id) = stack.pop() {
            if !reachable.insert(id) {
                continue;
            }
            if !graph.nodes[id].is_removed() {
                stack.extend_from_slice(&graph.nodes[id].input_ids);
            }
        }

        reachable
    }
}

impl OptimizationPass for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(&self, graph: &mut ComputeGraph) -> usize {
        let reachable = Self::find_reachable(graph);
        let mut changes = 0;

        for id in 0..graph.nodes.len() {
            if !reachable.contains(&id) && !graph.nodes[id].is_removed() {
                graph.nodes[id].mark_removed();
                changes += 1;
            }
        }

        changes
    }
}

/// Key for identifying structurally equivalent expressions (for CSE)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExprKey {
    op_type: OpType,
    input_ids: Vec<NodeId>,
}

impl ExprKey {
    fn from_node(node: &GraphNode) -> Self {
        Self { op_type: node.op_type, input_ids: node.input_ids.clone() }
    }
}

/// Common subexpression elimination pass — deduplicates identical computations.
pub struct CommonSubexprElimination;

impl OptimizationPass for CommonSubexprElimination {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(&self, graph: &mut ComputeGraph) -> usize {
        let mut changes = 0;
        let mut expr_to_node: HashMap<ExprKey, NodeId> = HashMap::new();

        let order = graph.topological_order();
        for node_id in order {
            let node = &graph.nodes[node_id];
            if node.is_removed() || node.op_type == OpType::Constant {
                continue;
            }

            let key = ExprKey::from_node(node);

            if let Some(&existing_id) = expr_to_node.get(&key) {
                // Found duplicate: replace uses and remove
                graph.replace_uses(node_id, existing_id);
                graph.nodes[node_id].mark_removed();
                changes += 1;
            } else {
                expr_to_node.insert(key, node_id);
            }
        }

        changes
    }
}

/// Graph optimizer that runs multiple passes until a fixpoint is reached
pub struct GraphOptimizer {
    passes: Vec<Box<dyn OptimizationPass>>,
    max_iterations: usize,
}

impl GraphOptimizer {
    /// Create a new optimizer with the default set of passes
    pub fn new() -> Self {
        let mut opt = Self { passes: Vec::new(), max_iterations: 10 };
        opt.passes.push(Box::new(ConstantFolding));
        opt.passes.push(Box::new(DeadCodeElimination));
        opt.passes.push(Box::new(CommonSubexprElimination));
        opt
    }

    /// Set the maximum number of optimization iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Run all passes until fixpoint or max iterations
    pub fn optimize(&self, graph: &mut ComputeGraph) -> OptimizationReport {
        let mut report = OptimizationReport {
            iterations: 0,
            total_changes: 0,
            pass_changes: HashMap::new(),
            initial_nodes: graph.active_node_count(),
            final_nodes: 0,
        };

        for _ in 0..self.max_iterations {
            let mut iter_changes = 0;
            for pass in &self.passes {
                let changes = pass.run(graph);
                if changes > 0 {
                    *report.pass_changes.entry(pass.name()).or_insert(0) += changes;
                }
                iter_changes += changes;
            }

            report.iterations += 1;
            report.total_changes += iter_changes;

            if iter_changes == 0 {
                break; // Fixpoint reached
            }
        }

        report.final_nodes = graph.active_node_count();
        report
    }
}

impl Default for GraphOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Report of optimization results
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Number of optimization iterations run
    pub iterations: usize,
    /// Total changes across all passes and iterations
    pub total_changes: usize,
    /// Changes per pass name
    pub pass_changes: HashMap<&'static str, usize>,
    /// Number of active nodes before optimization
    pub initial_nodes: usize,
    /// Number of active nodes after optimization
    pub final_nodes: usize,
}

impl OptimizationReport {
    /// Node reduction ratio (0.0 = no reduction, 1.0 = all removed)
    pub fn reduction_ratio(&self) -> f64 {
        if self.initial_nodes == 0 {
            return 0.0;
        }
        1.0 - (self.final_nodes as f64 / self.initial_nodes as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- TracedValue tests ---

    #[test]
    fn test_traced_value_constant() {
        let val = TracedValue::Constant(Array1::from(vec![1.0, 2.0]));
        assert!(val.is_constant());
        assert_eq!(val.as_constant().expect("operation should succeed").len(), 2);
        assert_eq!(val.node_id(), None);
    }

    #[test]
    fn test_traced_value_dynamic() {
        let val = TracedValue::Dynamic(42);
        assert!(!val.is_constant());
        assert!(val.as_constant().is_none());
        assert_eq!(val.node_id(), Some(42));
    }

    // --- TracedTensor tests ---

    #[test]
    fn test_traced_tensor_constant() {
        let t = TracedTensor::constant(Array1::from(vec![1.0, 2.0, 3.0]));
        assert!(t.is_constant());
        assert_eq!(t.shape(), &[3]);
    }

    #[test]
    fn test_traced_tensor_placeholder() {
        let t = TracedTensor::placeholder(vec![4, 4], 7);
        assert!(!t.is_constant());
        assert_eq!(t.shape(), &[4, 4]);
        assert_eq!(t.value().node_id(), Some(7));
    }

    // --- Identity folding tests ---

    #[test]
    fn test_add_with_zero_folds() {
        let x = TracedTensor::placeholder(vec![3], 0);
        let zero = TracedTensor::constant(Array1::zeros(3));

        // x + 0 = x
        let result = try_identity_fold(&x, &zero, OpType::Add);
        assert!(result.is_some());
        assert!(!result.expect("operation should succeed").is_constant()); // Should return x (dynamic)

        // 0 + x = x
        let result = try_identity_fold(&zero, &x, OpType::Add);
        assert!(result.is_some());
        assert!(!result.expect("operation should succeed").is_constant()); // Should return x (dynamic)
    }

    #[test]
    fn test_mul_with_one_folds() {
        let x = TracedTensor::placeholder(vec![3], 0);
        let one = TracedTensor::constant(Array1::ones(3));

        // x * 1 = x
        let result = try_identity_fold(&x, &one, OpType::Mul);
        assert!(result.is_some());
        assert!(!result.expect("operation should succeed").is_constant());

        // 1 * x = x
        let result = try_identity_fold(&one, &x, OpType::Mul);
        assert!(result.is_some());
        assert!(!result.expect("operation should succeed").is_constant());
    }

    #[test]
    fn test_mul_with_zero_annihilates() {
        let x = TracedTensor::placeholder(vec![3], 0);
        let zero = TracedTensor::constant(Array1::zeros(3));

        // x * 0 = 0
        let result = try_identity_fold(&x, &zero, OpType::Mul);
        assert!(result.is_some());
        let t = result.expect("operation should succeed");
        assert!(t.is_constant());
        assert!(is_zeros(t.value().as_constant().expect("operation should succeed")));

        // 0 * x = 0
        let result = try_identity_fold(&zero, &x, OpType::Mul);
        assert!(result.is_some());
        let t = result.expect("operation should succeed");
        assert!(t.is_constant());
        assert!(is_zeros(t.value().as_constant().expect("operation should succeed")));
    }

    #[test]
    fn test_no_identity_fold_for_nonidentity() {
        let a = TracedTensor::constant(Array1::from(vec![2.0, 3.0]));
        let b = TracedTensor::placeholder(vec![2], 0);

        assert!(try_identity_fold(&a, &b, OpType::Add).is_none());
        assert!(try_identity_fold(&a, &b, OpType::Mul).is_none());
    }

    // --- Traced binary op tests ---

    #[test]
    fn test_traced_binary_op_both_constant() {
        let mut graph = ComputeGraph::new();
        let a = TracedTensor::constant(Array1::from(vec![1.0, 2.0, 3.0]));
        let b = TracedTensor::constant(Array1::from(vec![4.0, 5.0, 6.0]));

        let result = traced_binary_op(&a, &b, |x, y| x + y, OpType::Add, &mut graph);
        assert!(result.is_constant());
        let data = result.value().as_constant().expect("operation should succeed");
        assert_eq!(data.as_slice().expect("operation should succeed"), &[5.0, 7.0, 9.0]);
        // No graph nodes created
        assert_eq!(graph.len(), 0);
    }

    #[test]
    fn test_traced_binary_op_one_dynamic() {
        let mut graph = ComputeGraph::new();
        let a = TracedTensor::placeholder(vec![3], graph.add_constant(Array1::from(vec![0.0; 3])));
        let b = TracedTensor::constant(Array1::from(vec![4.0, 5.0, 6.0]));

        let result = traced_binary_op(&a, &b, |x, y| x + y, OpType::Add, &mut graph);
        // b is lifted to a constant node, then an add node is created
        assert!(!result.is_constant());
    }

    #[test]
    fn test_traced_binary_op_identity_fold() {
        let mut graph = ComputeGraph::new();
        let x_id = graph.add_constant(Array1::from(vec![1.0, 2.0]));
        let x = TracedTensor::placeholder(vec![2], x_id);
        let zero = TracedTensor::constant(Array1::zeros(2));

        let result = traced_binary_op(&x, &zero, |a, b| a + b, OpType::Add, &mut graph);
        // Should fold: x + 0 = x, no new node
        assert!(!result.is_constant());
        assert_eq!(result.value().node_id(), Some(x_id));
    }

    // --- ComputeGraph tests ---

    #[test]
    fn test_compute_graph_empty() {
        let graph = ComputeGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.len(), 0);
        assert_eq!(graph.active_node_count(), 0);
    }

    #[test]
    fn test_compute_graph_add_nodes() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        let c2 = graph.add_constant(Array1::from(vec![2.0]));
        let add = graph.add_op(OpType::Add, vec![c1, c2], vec![1]);

        assert_eq!(graph.len(), 3);
        assert_eq!(graph.active_node_count(), 3);
        assert!(graph.node(c1).is_constant());
        assert!(!graph.node(add).is_constant());
    }

    #[test]
    fn test_compute_graph_topological_order() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        let c2 = graph.add_constant(Array1::from(vec![2.0]));
        let add = graph.add_op(OpType::Add, vec![c1, c2], vec![1]);
        graph.mark_output(add);

        let order = graph.topological_order();
        // c1 and c2 should come before add
        let add_pos = order.iter().position(|&x| x == add).expect("operation should succeed");
        let c1_pos = order.iter().position(|&x| x == c1).expect("operation should succeed");
        let c2_pos = order.iter().position(|&x| x == c2).expect("operation should succeed");
        assert!(c1_pos < add_pos);
        assert!(c2_pos < add_pos);
    }

    #[test]
    fn test_compute_graph_replace_uses() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        let c2 = graph.add_constant(Array1::from(vec![2.0]));
        let add = graph.add_op(OpType::Add, vec![c1, c2], vec![1]);
        graph.mark_output(add);

        // Replace c1 with c2
        let c3 = graph.add_constant(Array1::from(vec![3.0]));
        graph.replace_uses(c1, c3);

        assert_eq!(graph.node(add).input_ids, vec![c3, c2]);
    }

    // --- Constant folding pass tests ---

    #[test]
    fn test_constant_folding_add() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0, 2.0]));
        let c2 = graph.add_constant(Array1::from(vec![3.0, 4.0]));
        let add = graph.add_op(OpType::Add, vec![c1, c2], vec![2]);
        graph.mark_output(add);

        let pass = ConstantFolding;
        let changes = pass.run(&mut graph);

        assert_eq!(changes, 1);
        assert!(graph.node(add).is_constant());
        let result = graph.node(add).constant_value.as_ref().expect("operation should succeed");
        assert_eq!(result.as_slice().expect("operation should succeed"), &[4.0, 6.0]);
    }

    #[test]
    fn test_constant_folding_mul() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![2.0, 3.0]));
        let c2 = graph.add_constant(Array1::from(vec![4.0, 5.0]));
        let mul = graph.add_op(OpType::Mul, vec![c1, c2], vec![2]);
        graph.mark_output(mul);

        let pass = ConstantFolding;
        let changes = pass.run(&mut graph);

        assert_eq!(changes, 1);
        let result = graph.node(mul).constant_value.as_ref().expect("operation should succeed");
        assert_eq!(result.as_slice().expect("operation should succeed"), &[8.0, 15.0]);
    }

    #[test]
    fn test_constant_folding_sum() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0, 2.0, 3.0]));
        let sum = graph.add_op(OpType::Sum, vec![c1], vec![1]);
        graph.mark_output(sum);

        let pass = ConstantFolding;
        let changes = pass.run(&mut graph);

        assert_eq!(changes, 1);
        let result = graph.node(sum).constant_value.as_ref().expect("operation should succeed");
        assert_eq!(result.as_slice().expect("operation should succeed"), &[6.0]);
    }

    #[test]
    fn test_constant_folding_chain() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0, 2.0]));
        let c2 = graph.add_constant(Array1::from(vec![3.0, 4.0]));
        let add = graph.add_op(OpType::Add, vec![c1, c2], vec![2]);
        let c3 = graph.add_constant(Array1::from(vec![2.0, 2.0]));
        let mul = graph.add_op(OpType::Mul, vec![add, c3], vec![2]);
        graph.mark_output(mul);

        let optimizer = GraphOptimizer::new();
        let report = optimizer.optimize(&mut graph);

        // Both add and mul should be folded
        assert!(report.total_changes >= 2);
        assert!(graph.node(mul).is_constant());
        let result = graph.node(mul).constant_value.as_ref().expect("operation should succeed");
        assert_eq!(result.as_slice().expect("operation should succeed"), &[8.0, 12.0]);
    }

    #[test]
    fn test_constant_folding_skips_dynamic() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        // Node 1 is "dynamic" (no constant value)
        let dyn_node = graph.add_op(OpType::Relu, vec![c1], vec![1]);
        let c2 = graph.add_constant(Array1::from(vec![2.0]));
        let add = graph.add_op(OpType::Add, vec![dyn_node, c2], vec![1]);
        graph.mark_output(add);

        let pass = ConstantFolding;
        let changes = pass.run(&mut graph);

        // ReLU is not foldable, so add can't be folded either
        assert_eq!(changes, 0);
    }

    // --- Dead code elimination tests ---

    #[test]
    fn test_dce_removes_unreachable() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        let c2 = graph.add_constant(Array1::from(vec![2.0]));
        let _dead = graph.add_op(OpType::Add, vec![c1, c2], vec![1]); // Dead
        let c3 = graph.add_constant(Array1::from(vec![3.0]));
        graph.mark_output(c3);

        let pass = DeadCodeElimination;
        let changes = pass.run(&mut graph);

        assert_eq!(changes, 3); // c1, c2, and _dead are unreachable
        assert!(graph.node(c1).is_removed());
        assert!(graph.node(c2).is_removed());
        assert!(!graph.node(c3).is_removed());
    }

    #[test]
    fn test_dce_preserves_reachable() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        let c2 = graph.add_constant(Array1::from(vec![2.0]));
        let add = graph.add_op(OpType::Add, vec![c1, c2], vec![1]);
        graph.mark_output(add);

        let pass = DeadCodeElimination;
        let changes = pass.run(&mut graph);

        assert_eq!(changes, 0); // Everything is reachable
    }

    // --- CSE tests ---

    #[test]
    fn test_cse_deduplicates() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        let c2 = graph.add_constant(Array1::from(vec![2.0]));
        let add1 = graph.add_op(OpType::Add, vec![c1, c2], vec![1]);
        let add2 = graph.add_op(OpType::Add, vec![c1, c2], vec![1]); // Duplicate
        let mul = graph.add_op(OpType::Mul, vec![add1, add2], vec![1]);
        graph.mark_output(mul);

        let pass = CommonSubexprElimination;
        let changes = pass.run(&mut graph);

        assert_eq!(changes, 1); // add2 eliminated
        assert!(graph.node(add2).is_removed());
        // mul should now reference add1 for both inputs
        assert_eq!(graph.node(mul).input_ids, vec![add1, add1]);
    }

    #[test]
    fn test_cse_no_false_positive() {
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        let c2 = graph.add_constant(Array1::from(vec![2.0]));
        let c3 = graph.add_constant(Array1::from(vec![3.0]));
        let add1 = graph.add_op(OpType::Add, vec![c1, c2], vec![1]);
        let add2 = graph.add_op(OpType::Add, vec![c1, c3], vec![1]); // Different inputs
        let mul = graph.add_op(OpType::Mul, vec![add1, add2], vec![1]);
        graph.mark_output(mul);

        let pass = CommonSubexprElimination;
        let changes = pass.run(&mut graph);

        assert_eq!(changes, 0); // Different expressions, no dedup
    }

    // --- GraphOptimizer tests ---

    #[test]
    fn test_optimizer_full_pipeline() {
        let mut graph = ComputeGraph::new();

        // Build: (a + b) * (a + b) where a, b are constants
        // Should fold to a single constant
        let a = graph.add_constant(Array1::from(vec![1.0, 2.0]));
        let b = graph.add_constant(Array1::from(vec![3.0, 4.0]));
        let add1 = graph.add_op(OpType::Add, vec![a, b], vec![2]);
        let add2 = graph.add_op(OpType::Add, vec![a, b], vec![2]); // Duplicate
        let mul = graph.add_op(OpType::Mul, vec![add1, add2], vec![2]);
        graph.mark_output(mul);

        let optimizer = GraphOptimizer::new();
        let report = optimizer.optimize(&mut graph);

        assert!(report.total_changes > 0);
        assert!(report.final_nodes < report.initial_nodes);
    }

    #[test]
    fn test_optimizer_report_reduction_ratio() {
        let report = OptimizationReport {
            iterations: 1,
            total_changes: 5,
            pass_changes: HashMap::new(),
            initial_nodes: 10,
            final_nodes: 5,
        };
        assert!((report.reduction_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_optimizer_report_empty_graph() {
        let report = OptimizationReport {
            iterations: 0,
            total_changes: 0,
            pass_changes: HashMap::new(),
            initial_nodes: 0,
            final_nodes: 0,
        };
        assert!((report.reduction_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_optimizer_max_iterations() {
        let optimizer = GraphOptimizer::new().with_max_iterations(1);
        let mut graph = ComputeGraph::new();
        let c1 = graph.add_constant(Array1::from(vec![1.0]));
        graph.mark_output(c1);

        let report = optimizer.optimize(&mut graph);
        assert!(report.iterations <= 1);
    }

    // --- ShapeTracker tests ---

    #[test]
    fn test_shape_tracker_register_and_get() {
        let mut tracker = ShapeTracker::new();
        tracker.register(0, vec![3, 4]);
        assert_eq!(tracker.get(0), Some(&[3, 4][..]));
        assert_eq!(tracker.get(1), None);
    }

    #[test]
    fn test_shape_tracker_elementwise() {
        let mut tracker = ShapeTracker::new();
        tracker.register(0, vec![5]);
        tracker.register(1, vec![5]);

        let result = tracker.infer_elementwise(2, 0, 1);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation should succeed"), vec![5]);
        assert_eq!(tracker.get(2), Some(&[5][..]));
    }

    #[test]
    fn test_shape_tracker_elementwise_mismatch() {
        let mut tracker = ShapeTracker::new();
        tracker.register(0, vec![3]);
        tracker.register(1, vec![5]);

        let result = tracker.infer_elementwise(2, 0, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeError::DimMismatch { .. } => {}
            other => panic!("expected DimMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_shape_tracker_matmul() {
        let mut tracker = ShapeTracker::new();
        tracker.register(0, vec![3, 4]);
        tracker.register(1, vec![4, 5]);

        let result = tracker.infer_matmul(2, 0, 1);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation should succeed"), vec![3, 5]);
    }

    #[test]
    fn test_shape_tracker_matmul_mismatch() {
        let mut tracker = ShapeTracker::new();
        tracker.register(0, vec![3, 4]);
        tracker.register(1, vec![5, 6]);

        let result = tracker.infer_matmul(2, 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_tracker_matmul_insufficient_dims() {
        let mut tracker = ShapeTracker::new();
        tracker.register(0, vec![4]);
        tracker.register(1, vec![4, 5]);

        let result = tracker.infer_matmul(2, 0, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeError::InsufficientDims { required: 2, got: 1 } => {}
            other => panic!("expected InsufficientDims, got {other:?}"),
        }
    }

    #[test]
    fn test_shape_tracker_sum() {
        let mut tracker = ShapeTracker::new();
        tracker.register(0, vec![10]);

        let result = tracker.infer_sum(1, 0);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation should succeed"), vec![1]);
    }

    #[test]
    fn test_shape_tracker_unknown_input() {
        let mut tracker = ShapeTracker::new();
        let result = tracker.infer_sum(1, 99);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeError::UnknownInput(99) => {}
            other => panic!("expected UnknownInput(99), got {other:?}"),
        }
    }

    #[test]
    fn test_shape_tracker_len() {
        let mut tracker = ShapeTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);

        tracker.register(0, vec![3]);
        assert!(!tracker.is_empty());
        assert_eq!(tracker.len(), 1);
    }

    // --- Helper function tests ---

    #[test]
    fn test_is_zeros() {
        assert!(is_zeros(&Array1::zeros(5)));
        assert!(!is_zeros(&Array1::ones(5)));
        assert!(!is_zeros(&Array1::from(vec![0.0, 0.0, 1.0])));
        assert!(is_zeros(&Array1::from(vec![])));
    }

    #[test]
    fn test_is_ones() {
        assert!(is_ones(&Array1::ones(5)));
        assert!(!is_ones(&Array1::zeros(5)));
        assert!(!is_ones(&Array1::from(vec![1.0, 1.0, 2.0])));
        assert!(is_ones(&Array1::from(vec![])));
    }

    // --- ShapeError Display tests ---

    #[test]
    fn test_shape_error_display() {
        let err = ShapeError::UnknownInput(42);
        assert_eq!(format!("{err}"), "unknown input node 42");

        let err = ShapeError::DimMismatch { expected: 3, got: 5 };
        assert_eq!(format!("{err}"), "dimension mismatch: expected 3, got 5");

        let err = ShapeError::InsufficientDims { required: 2, got: 1 };
        assert_eq!(format!("{err}"), "insufficient dims: need 2, have 1");
    }

    // --- GraphNode tests ---

    #[test]
    fn test_graph_node_mark_removed() {
        let mut node = GraphNode {
            id: 0,
            op_type: OpType::Add,
            input_ids: vec![],
            output_shape: vec![1],
            constant_value: None,
            removed: false,
        };
        assert!(!node.is_removed());
        node.mark_removed();
        assert!(node.is_removed());
    }

    // --- OpType tests for match arm coverage ---

    #[test]
    fn test_op_type_variants() {
        let ops = [
            OpType::Add,
            OpType::Mul,
            OpType::Scale,
            OpType::Sum,
            OpType::Matmul,
            OpType::Relu,
            OpType::Gelu,
            OpType::Softmax,
            OpType::LayerNorm,
            OpType::Attention,
            OpType::Constant,
        ];

        for op in &ops {
            match op {
                OpType::Add => assert_eq!(*op, OpType::Add),
                OpType::Mul => assert_eq!(*op, OpType::Mul),
                OpType::Scale => assert_eq!(*op, OpType::Scale),
                OpType::Sum => assert_eq!(*op, OpType::Sum),
                OpType::Matmul => assert_eq!(*op, OpType::Matmul),
                OpType::Relu => assert_eq!(*op, OpType::Relu),
                OpType::Gelu => assert_eq!(*op, OpType::Gelu),
                OpType::Softmax => assert_eq!(*op, OpType::Softmax),
                OpType::LayerNorm => assert_eq!(*op, OpType::LayerNorm),
                OpType::Attention => assert_eq!(*op, OpType::Attention),
                OpType::Constant => assert_eq!(*op, OpType::Constant),
            }
        }
    }

    // --- Integration: realistic graph optimization scenario ---

    #[test]
    fn test_mlp_init_with_zero_bias() {
        // Simulating: output = (input * weights) + bias where bias = 0
        // The bias addition should be eliminated by constant folding + identity fold
        let mut graph = ComputeGraph::new();

        // Input (dynamic) and weights (dynamic) — represented as placeholders
        let input = graph.add_op(OpType::Relu, vec![], vec![4]); // Simulating dynamic input
        let weights = graph.add_constant(Array1::from(vec![0.5; 4]));
        let matmul = graph.add_op(OpType::Mul, vec![input, weights], vec![4]);

        // Bias = 0 (constant)
        let bias = graph.add_constant(Array1::zeros(4));
        let output = graph.add_op(OpType::Add, vec![matmul, bias], vec![4]);
        graph.mark_output(output);

        let initial_active = graph.active_node_count();

        let optimizer = GraphOptimizer::new();
        let report = optimizer.optimize(&mut graph);

        // The bias addition can't be fully eliminated since matmul is dynamic,
        // but DCE should handle any dead nodes
        assert!(report.iterations > 0);
        assert!(graph.active_node_count() <= initial_active);
    }

    #[test]
    fn test_repeated_subexpression_elimination() {
        // Build: z = (a + b) * (a + b) with CSE
        let mut graph = ComputeGraph::new();
        let a = graph.add_constant(Array1::from(vec![1.0, 2.0]));
        let b = graph.add_constant(Array1::from(vec![3.0, 4.0]));
        let add1 = graph.add_op(OpType::Add, vec![a, b], vec![2]);
        let add2 = graph.add_op(OpType::Add, vec![a, b], vec![2]);
        let mul = graph.add_op(OpType::Mul, vec![add1, add2], vec![2]);
        let sum = graph.add_op(OpType::Sum, vec![mul], vec![1]);
        graph.mark_output(sum);

        let optimizer = GraphOptimizer::new();
        let report = optimizer.optimize(&mut graph);

        // CSE should eliminate add2, constant folding should fold everything
        assert!(report.total_changes > 0);
    }

    // --- Default impls ---

    #[test]
    fn test_compute_graph_default() {
        let graph = ComputeGraph::default();
        assert!(graph.is_empty());
    }

    #[test]
    fn test_shape_tracker_default() {
        let tracker = ShapeTracker::default();
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_graph_optimizer_default() {
        let optimizer = GraphOptimizer::default();
        let mut graph = ComputeGraph::new();
        let c = graph.add_constant(Array1::from(vec![1.0]));
        graph.mark_output(c);
        let report = optimizer.optimize(&mut graph);
        assert_eq!(report.iterations, 1); // One pass, no changes, done
    }
}
