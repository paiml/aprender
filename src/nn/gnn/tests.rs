use super::*;
/// Helper function to create deterministic test data.
fn create_test_tensor(shape: &[usize], seed: u32) -> Tensor {
    let len: usize = shape.iter().product();
    let data: Vec<f32> = (0..len)
        .map(|i| ((i as f32 + seed as f32) * 0.1).sin())
        .collect();
    Tensor::new(&data, shape)
}

// ==================== AdjacencyMatrix Tests ====================

#[test]
fn test_adjacency_matrix_creation() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 0]], 3);
    assert_eq!(adj.num_nodes(), 3);
    assert_eq!(adj.num_edges(), 3);
}

#[test]
fn test_adjacency_matrix_from_coo() {
    let adj = AdjacencyMatrix::from_coo(vec![0, 1, 2], vec![1, 2, 0], 3);
    assert_eq!(adj.num_nodes(), 3);
    assert_eq!(adj.num_edges(), 3);
}

#[test]
fn test_adjacency_matrix_add_self_loops() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);
    assert!(!adj.has_self_loops());

    let adj_with_loops = adj.add_self_loops();
    assert!(adj_with_loops.has_self_loops());
    assert_eq!(adj_with_loops.num_edges(), 5); // 2 original + 3 self-loops
}

#[test]
fn test_adjacency_matrix_degrees() {
    // Graph: 0 -> 1 -> 2
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let in_deg = adj.in_degrees();
    assert_eq!(in_deg, vec![0.0, 1.0, 1.0]); // 0 has no incoming, 1 and 2 have 1 each

    let out_deg = adj.out_degrees();
    assert_eq!(out_deg, vec![1.0, 1.0, 0.0]); // 0 and 1 have 1 outgoing, 2 has none
}

#[test]
fn test_adjacency_matrix_neighbors() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [0, 2], [1, 2]], 3);
    let neighbors = adj.neighbors(0);
    assert_eq!(neighbors, vec![1, 2]);
}

#[test]
fn test_adjacency_matrix_to_dense() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);
    let dense = adj.to_dense();

    assert_eq!(dense.n_rows(), 3);
    assert_eq!(dense.n_cols(), 3);
    // Check edge (0,1) exists
    assert!((dense.get(0, 1) - 1.0).abs() < 0.01);
    // Check edge (1,2) exists
    assert!((dense.get(1, 2) - 1.0).abs() < 0.01);
}

#[test]
fn test_adjacency_matrix_with_weights() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3).with_weights(vec![0.5, 2.0]);
    let dense = adj.to_dense();

    assert!((dense.get(0, 1) - 0.5).abs() < 0.01);
    assert!((dense.get(1, 2) - 2.0).abs() < 0.01);
}

// ==================== GCNConv Tests ====================

#[test]
fn test_gcn_creation() {
    let gcn = GCNConv::new(64, 32);
    assert_eq!(gcn.in_features(), 64);
    assert_eq!(gcn.out_features(), 32);
}

#[test]
fn test_gcn_without_bias() {
    let gcn = GCNConv::new(64, 32).without_bias();
    assert!(gcn.bias().is_none());
}

#[test]
fn test_gcn_forward_shape() {
    let gcn = GCNConv::new(8, 4);
    let x = create_test_tensor(&[5, 8], 1); // 5 nodes, 8 features
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

    let out = gcn.forward(&x, &adj);
    assert_eq!(out.shape(), &[5, 4]);
}

#[test]
fn test_gcn_forward_values() {
    let gcn = GCNConv::new(4, 2);

    // Simple graph: 0 <-> 1 <-> 2
    let x = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0, // Node 0
            0.0, 1.0, 0.0, 0.0, // Node 1
            0.0, 0.0, 1.0, 0.0, // Node 2
        ],
        &[3, 4],
    );
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 0], [1, 2], [2, 1]], 3);

    let out = gcn.forward(&x, &adj);

    // Output should be non-zero (features are propagated)
    let out_data = out.data();
    let sum: f32 = out_data.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0, "Output should have non-zero values");
}

#[test]
fn test_gcn_normalized_aggregation() {
    let gcn = GCNConv::new(2, 2);

    // Complete graph K3 (fully connected)
    let x = Tensor::new(
        &[
            1.0, 1.0, // Node 0
            1.0, 1.0, // Node 1
            1.0, 1.0, // Node 2
        ],
        &[3, 2],
    );
    let adj =
        AdjacencyMatrix::from_edge_index(&[[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]], 3);

    let out = gcn.forward(&x, &adj);

    // All nodes should have similar output (symmetric graph, same features)
    let out_data = out.data();
    let diff_01 = (out_data[0] - out_data[2]).abs() + (out_data[1] - out_data[3]).abs();
    let diff_12 = (out_data[2] - out_data[4]).abs() + (out_data[3] - out_data[5]).abs();

    assert!(diff_01 < 0.1, "Symmetric nodes should have similar outputs");
    assert!(diff_12 < 0.1, "Symmetric nodes should have similar outputs");
}

#[test]
fn test_gcn_without_self_loops() {
    let gcn = GCNConv::new(4, 2).without_self_loops();
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = gcn.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

#[test]
fn test_gcn_without_normalize() {
    let gcn = GCNConv::new(4, 2).without_normalize();
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = gcn.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

// ==================== SAGEConv Tests ====================

#[test]
fn test_sage_creation() {
    let sage = SAGEConv::new(64, 32);
    assert_eq!(sage.in_features(), 64);
    assert_eq!(sage.out_features(), 32);
    assert_eq!(sage.aggregation(), SAGEAggregation::Mean);
}

#[test]
fn test_sage_with_aggregation() {
    let sage_max = SAGEConv::new(64, 32).with_aggregation(SAGEAggregation::Max);
    assert_eq!(sage_max.aggregation(), SAGEAggregation::Max);

    let sage_sum = SAGEConv::new(64, 32).with_aggregation(SAGEAggregation::Sum);
    assert_eq!(sage_sum.aggregation(), SAGEAggregation::Sum);
}

#[test]
fn test_sage_forward_shape() {
    let sage = SAGEConv::new(8, 4);
    let x = create_test_tensor(&[5, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[5, 4]);
}

#[test]
fn test_sage_mean_aggregation() {
    let sage = SAGEConv::new(4, 2).with_aggregation(SAGEAggregation::Mean);
    let x = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0, // Node 0
            0.0, 1.0, 0.0, 0.0, // Node 1
            0.0, 0.0, 1.0, 0.0, // Node 2
        ],
        &[3, 4],
    );
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

#[test]
fn test_sage_max_aggregation() {
    let sage = SAGEConv::new(4, 2).with_aggregation(SAGEAggregation::Max);
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // Node 0
            5.0, 6.0, 7.0, 8.0, // Node 1
        ],
        &[2, 4],
    );
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 0]], 2);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[2, 2]);
}

#[test]
fn test_sage_sum_aggregation() {
    let sage = SAGEConv::new(4, 2).with_aggregation(SAGEAggregation::Sum);
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

#[test]
fn test_sage_with_normalize() {
    let sage = SAGEConv::new(4, 2).with_normalize();
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = sage.forward(&x, &adj);

    // Check that outputs are normalized (L2 norm ≈ 1)
    let out_data = out.data();
    for node in 0..3 {
        let norm: f32 = (0..2)
            .map(|f| out_data[node * 2 + f].powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01 || norm < 0.01,
            "Normalized output should have unit norm, got {}",
            norm
        );
    }
}

#[test]
fn test_sage_without_root() {
    let sage = SAGEConv::new(4, 2).without_root();
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

// ==================== GATConv Tests ====================

#[test]
fn test_gat_creation() {
    let gat = GATConv::new(64, 32, 4);
    assert_eq!(gat.in_features(), 64);
    assert_eq!(gat.out_features(), 32);
    assert_eq!(gat.num_heads(), 4);
    assert_eq!(gat.total_out_features(), 128); // 32 * 4
}

#[test]
fn test_gat_without_concat() {
    let gat = GATConv::new(64, 32, 4).without_concat();
    assert_eq!(gat.total_out_features(), 32); // Averaged, not concatenated
}

#[test]
fn test_gat_forward_shape_concat() {
    let gat = GATConv::new(8, 4, 2); // 2 heads, 4 features each
    let x = create_test_tensor(&[5, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[5, 8]); // 4 * 2 = 8
}

#[test]
fn test_gat_forward_shape_avg() {
    let gat = GATConv::new(8, 4, 2).without_concat();
    let x = create_test_tensor(&[5, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[5, 4]); // Averaged heads
}

#[test]
fn test_gat_attention_different_neighbors() {
    let gat = GATConv::new(4, 2, 1);

    // Graph where node 2 has two different neighbors
    let x = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0, // Node 0 (distinct feature)
            0.0, 1.0, 0.0, 0.0, // Node 1 (distinct feature)
            0.0, 0.0, 0.0, 0.0, // Node 2 (target)
        ],
        &[3, 4],
    );
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 2], [1, 2]], 3);

    let out = gat.forward(&x, &adj);

    // Node 2 should have output that is a weighted combination
    let out_data = out.data();
    let node2_out = &out_data[4..6];
    let has_nonzero = node2_out.iter().any(|&x| x.abs() > 1e-6);
    assert!(
        has_nonzero,
        "Node 2 should have non-zero output from attention"
    );
}

#[test]
fn test_gat_with_negative_slope() {
    let gat = GATConv::new(8, 4, 2).with_negative_slope(0.1);
    let x = create_test_tensor(&[3, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_gat_without_self_loops() {
    let gat = GATConv::new(8, 4, 2).without_self_loops();
    let x = create_test_tensor(&[3, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_gat_without_bias() {
    let gat = GATConv::new(8, 4, 2).without_bias();
    let x = create_test_tensor(&[3, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_gat_single_head() {
    let gat = GATConv::new(8, 4, 1);
    let x = create_test_tensor(&[5, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3]], 5);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[5, 4]);
}

// ==================== Integration Tests ====================

#[test]
fn test_gnn_stack() {
    // Test stacking multiple GNN layers
    let gcn1 = GCNConv::new(8, 16);
    let gcn2 = GCNConv::new(16, 4);

    let x = create_test_tensor(&[5, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]], 5);

    let h1 = gcn1.forward(&x, &adj);
    assert_eq!(h1.shape(), &[5, 16]);

    let h2 = gcn2.forward(&h1, &adj);
    assert_eq!(h2.shape(), &[5, 4]);
}

#[test]
fn test_gnn_heterogeneous_layers() {
    // Test mixing different GNN layers
    let gcn = GCNConv::new(8, 16);
    let gat = GATConv::new(16, 8, 2).without_concat();
    let sage = SAGEConv::new(8, 4);

    let x = create_test_tensor(&[5, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 4]], 5);

    let h1 = gcn.forward(&x, &adj);
    assert_eq!(h1.shape(), &[5, 16]);

    let h2 = gat.forward(&h1, &adj);
    assert_eq!(h2.shape(), &[5, 8]);

    let h3 = sage.forward(&h2, &adj);
    assert_eq!(h3.shape(), &[5, 4]);
}

#[test]
fn test_gnn_empty_graph() {
    let gcn = GCNConv::new(4, 2);
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[], 3); // No edges

    let out = gcn.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

#[test]
fn test_gnn_single_node() {
    let gcn = GCNConv::new(4, 2);
    let x = create_test_tensor(&[1, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[], 1);

    let out = gcn.forward(&x, &adj);
    assert_eq!(out.shape(), &[1, 2]);
}

#[test]
fn test_gnn_disconnected_graph() {
    // Graph with two disconnected components
    let sage = SAGEConv::new(4, 2);
    let x = create_test_tensor(&[4, 4], 1);
    // Component 1: 0-1, Component 2: 2-3
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 0], [2, 3], [3, 2]], 4);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[4, 2]);
}

include!("tests_part_02.rs");
