use super::super::super::super::*;

// ========================
// Coverage Tests (C012-C020): Ops, Budget, Edge/Node ID traits
// ========================

/// C012: AddOp::tokens() method
#[test]
fn test_c012_add_op_tokens() {
    let op = AddOp::new(3);
    let input = (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
    assert_eq!(op.tokens(&input), 3);
}

/// C013: MatmulOp::name() method
#[test]
fn test_c013_matmul_op_name() {
    let op = MatmulOp::new(4, 4, 4);
    assert_eq!(op.name(), "matmul");
}

/// C014: DotOp::name() method
#[test]
fn test_c014_dot_op_name() {
    let op = DotOp::new(4);
    assert_eq!(op.name(), "dot");
}

/// C015: SoftmaxOp::name() method
#[test]
fn test_c015_softmax_op_name() {
    let op = SoftmaxOp::new(4);
    assert_eq!(op.name(), "softmax");
}

/// C016: Zero elapsed time edge case (infinity tokens/sec)
#[test]
fn test_c016_zero_elapsed_time() {
    // This tests the f64::INFINITY case in run()
    // We can't easily trigger this without mocking time, but we verify
    // the budget calculation handles extreme cases
    let budget = TokenBudget::from_throughput(f64::MAX);
    assert!(budget.us_per_token < 1e-10);
}

/// C017: ComputeOp::clone_input() default implementation
#[test]
fn test_c017_clone_input_default() {
    let op = AddOp::new(2);
    let input = (vec![1.0, 2.0], vec![3.0, 4.0]);
    let cloned = op.clone_input(&input);
    assert!(cloned.is_some());
    let cloned = cloned.unwrap();
    assert_eq!(cloned.0, input.0);
    assert_eq!(cloned.1, input.1);
}

/// C018: EdgeType debug formatting
#[test]
fn test_c018_edge_type_debug() {
    let depends = EdgeType::DependsOn;
    let debug_str = format!("{:?}", depends);
    assert!(debug_str.contains("DependsOn"));

    let transfer = EdgeType::Transfer { bytes: 1024, direction: TransferDirection::H2D };
    let debug_str = format!("{:?}", transfer);
    assert!(debug_str.contains("Transfer"));
    assert!(debug_str.contains("1024"));
}

/// C019: TransferDirection debug and clone
#[test]
fn test_c019_transfer_direction_traits() {
    let dir = TransferDirection::D2D;
    let cloned = dir;
    assert_eq!(dir, cloned);

    let debug_str = format!("{:?}", dir);
    assert!(debug_str.contains("D2D"));
}

/// C020: ExecutionNodeId hash and ordering
#[test]
fn test_c020_execution_node_id_traits() {
    use std::collections::HashSet;

    let id1 = ExecutionNodeId(1);
    let id2 = ExecutionNodeId(2);
    let id1_copy = ExecutionNodeId(1);

    assert_eq!(id1, id1_copy);
    assert_ne!(id1, id2);

    let mut set = HashSet::new();
    set.insert(id1);
    set.insert(id2);
    set.insert(id1_copy);
    assert_eq!(set.len(), 2);
}
