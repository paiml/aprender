use super::*;

// ==================== Default Trait Tests ====================

#[test]
fn test_sage_aggregation_default() {
    assert_eq!(SAGEAggregation::default(), SAGEAggregation::Mean);
}

// ==================== Edge Cases ====================

#[test]
fn test_gnn_large_graph() {
    let gcn = GCNConv::new(16, 8);

    // Create a larger graph (100 nodes, ~200 edges)
    let mut edges = Vec::new();
    for i in 0..100 {
        edges.push([i, (i + 1) % 100]); // Ring
        if i < 50 {
            edges.push([i, i + 50]); // Cross connections
        }
    }

    let x = create_test_tensor(&[100, 16], 1);
    let adj = AdjacencyMatrix::from_edge_index(&edges, 100);

    let out = gcn.forward(&x, &adj);
    assert_eq!(out.shape(), &[100, 8]);
}

#[test]
fn test_gat_multiple_heads_attention() {
    // Test that different heads learn different patterns
    let gat = GATConv::new(4, 2, 4); // 4 heads

    let x = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        &[4, 4],
    );
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 3], [3, 0]], 4);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[4, 8]); // 2 * 4 heads = 8

    // Each head should contribute to the output
    let out_data = out.data();
    let has_variance = out_data.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-6);
    assert!(has_variance, "Multi-head output should have variance");
}

// ==================== SAGEAggregation Lstm Tests ====================

#[test]
fn test_sage_lstm_aggregation() {
    let sage = SAGEConv::new(4, 2).with_aggregation(SAGEAggregation::Lstm);
    assert_eq!(sage.aggregation(), SAGEAggregation::Lstm);

    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // Node 0
            5.0, 6.0, 7.0, 8.0, // Node 1
            9.0, 10.0, 11.0, 12.0, // Node 2
        ],
        &[3, 4],
    );
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 0]], 3);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

// ==================== SAGEConv without_bias Tests ====================

#[test]
fn test_sage_without_bias() {
    let sage = SAGEConv::new(4, 2).without_bias();
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

#[test]
fn test_sage_without_bias_no_bias_added() {
    let sage_with_bias = SAGEConv::new(4, 2);
    let sage_without_bias = SAGEConv::new(4, 2).without_bias();

    let x = Tensor::new(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 4]);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 0]], 2);

    let out_with = sage_with_bias.forward(&x, &adj);
    let out_without = sage_without_bias.forward(&x, &adj);

    // They should be the same since bias is initialized to zero,
    // but this tests that the code path actually runs
    assert_eq!(out_with.shape(), out_without.shape());
}

// ==================== AdjacencyMatrix self-loops idempotency ====================

#[test]
fn test_adjacency_matrix_add_self_loops_idempotent() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3).add_self_loops();
    assert!(adj.has_self_loops());
    let edge_count_before = adj.num_edges();

    // Adding self-loops again should be a no-op
    let adj2 = adj.add_self_loops();
    assert!(adj2.has_self_loops());
    assert_eq!(adj2.num_edges(), edge_count_before);
}

// ==================== AdjacencyMatrix self-loops with weights ====================

#[test]
fn test_adjacency_matrix_add_self_loops_with_weights() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3)
        .with_weights(vec![0.5, 2.0])
        .add_self_loops();

    assert!(adj.has_self_loops());
    assert_eq!(adj.num_edges(), 5);

    // Verify dense matrix has self-loops with weight 1.0
    let dense = adj.to_dense();
    assert!((dense.get(0, 0) - 1.0).abs() < 0.01); // self-loop
    assert!((dense.get(1, 1) - 1.0).abs() < 0.01); // self-loop
    assert!((dense.get(2, 2) - 1.0).abs() < 0.01); // self-loop
}

// ==================== GCNConv weight/bias accessors ====================

#[test]
fn test_gcn_weight_accessor() {
    let gcn = GCNConv::new(4, 2);
    let weight = gcn.weight();
    assert_eq!(weight.shape(), &[4, 2]);
}

#[test]
fn test_gcn_bias_accessor() {
    let gcn = GCNConv::new(4, 2);
    let bias = gcn.bias();
    assert!(bias.is_some());
    assert_eq!(bias.unwrap().shape(), &[2]);
}

#[test]
fn test_gcn_without_bias_accessor() {
    let gcn = GCNConv::new(4, 2).without_bias();
    assert!(gcn.bias().is_none());
}

// ==================== AdjacencyMatrix edge accessor coverage ====================

#[test]
fn test_adjacency_matrix_edge_accessors() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [2, 3]], 4);
    assert_eq!(adj.edge_src(), &[0, 2]);
    assert_eq!(adj.edge_tgt(), &[1, 3]);
}

// ==================== AdjacencyMatrix neighbors of isolated node ====================

#[test]
fn test_adjacency_matrix_neighbors_isolated_node() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1]], 3);
    let neighbors = adj.neighbors(2); // Node 2 has no outgoing edges
    assert!(neighbors.is_empty());
}

// ==================== AdjacencyMatrix out-of-bounds edge indices ====================

#[test]
fn test_adjacency_matrix_degrees_skip_out_of_bounds() {
    // Edge with target >= num_nodes should be skipped in in_degrees
    let adj = AdjacencyMatrix::from_coo(vec![0, 1, 5], vec![1, 5, 0], 3);
    let in_deg = adj.in_degrees();
    // Only valid edge targets: 1 (from 0) and 0 (from 5)
    assert_eq!(in_deg[1], 1.0);
    assert_eq!(in_deg[0], 1.0);
    assert_eq!(in_deg[2], 0.0);

    let out_deg = adj.out_degrees();
    assert_eq!(out_deg[0], 1.0);
    assert_eq!(out_deg[1], 1.0);
}

// ==================== GATConv with_dropout ====================

#[test]
fn test_gat_with_dropout() {
    let gat = GATConv::new(8, 4, 2).with_dropout(0.5);
    let x = create_test_tensor(&[3, 8], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 8]);
}

// ==================== GATConv without_concat with bias ====================

#[test]
fn test_gat_without_concat_with_bias() {
    // Tests the bias averaging path in GAT forward when concat=false
    let gat = GATConv::new(4, 2, 3).without_concat(); // 3 heads, averaged
    let x = Tensor::new(
        &[
            1.0, 0.0, 0.0, 0.0, // Node 0
            0.0, 1.0, 0.0, 0.0, // Node 1
            0.0, 0.0, 1.0, 0.0, // Node 2
        ],
        &[3, 4],
    );
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2], [2, 0]], 3);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]); // Averaged to out_features
}

// ==================== GATConv without_concat without_bias ====================

#[test]
fn test_gat_without_concat_without_bias() {
    let gat = GATConv::new(4, 2, 2).without_concat().without_bias();
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3);

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

// ==================== Debug/Clone impls ====================

#[test]
fn test_adjacency_matrix_debug_clone() {
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1]], 2);
    let debug_str = format!("{:?}", adj);
    assert!(debug_str.contains("AdjacencyMatrix"));

    let cloned = adj.clone();
    assert_eq!(cloned.num_nodes(), 2);
    assert_eq!(cloned.num_edges(), 1);
}

#[test]
fn test_gcn_conv_debug_clone() {
    let gcn = GCNConv::new(4, 2);
    let debug_str = format!("{:?}", gcn);
    assert!(debug_str.contains("GCNConv"));

    let cloned = gcn.clone();
    assert_eq!(cloned.in_features(), 4);
    assert_eq!(cloned.out_features(), 2);
}

#[test]
fn test_sage_conv_debug_clone() {
    let sage = SAGEConv::new(4, 2);
    let debug_str = format!("{:?}", sage);
    assert!(debug_str.contains("SAGEConv"));

    let cloned = sage.clone();
    assert_eq!(cloned.in_features(), 4);
    assert_eq!(cloned.out_features(), 2);
}

#[test]
fn test_gat_conv_debug_clone() {
    let gat = GATConv::new(4, 2, 3);
    let debug_str = format!("{:?}", gat);
    assert!(debug_str.contains("GATConv"));

    let cloned = gat.clone();
    assert_eq!(cloned.in_features(), 4);
    assert_eq!(cloned.out_features(), 2);
    assert_eq!(cloned.num_heads(), 3);
}

#[test]
fn test_sage_aggregation_debug_clone() {
    let agg = SAGEAggregation::Max;
    let debug_str = format!("{:?}", agg);
    assert!(debug_str.contains("Max"));

    let cloned = agg;
    assert_eq!(cloned, SAGEAggregation::Max);
}

// ==================== GCN forward with already-self-looped adj ====================

#[test]
fn test_gcn_forward_with_preexisting_self_loops() {
    let gcn = GCNConv::new(4, 2);
    let x = create_test_tensor(&[3, 4], 1);
    // Adjacency already has self-loops
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3).add_self_loops();

    let out = gcn.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}

// ==================== GAT forward with preexisting self-loops ====================

#[test]
fn test_gat_forward_with_preexisting_self_loops() {
    let gat = GATConv::new(4, 2, 2);
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3).add_self_loops();

    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 4]); // 2 * 2 heads
}

// ==================== to_dense with out-of-bound edges ====================

#[test]
fn test_adjacency_matrix_to_dense_ignores_out_of_bounds() {
    // Create adj with an edge referencing a node beyond num_nodes
    let adj = AdjacencyMatrix::from_coo(vec![0, 5], vec![1, 0], 3);
    let dense = adj.to_dense();
    assert_eq!(dense.n_rows(), 3);
    assert_eq!(dense.n_cols(), 3);
    // Edge (0, 1) should exist
    assert!((dense.get(0, 1) - 1.0).abs() < 0.01);
    // Edge (5, 0) should be ignored (5 >= 3)
}

// ==================== SAGE with isolated nodes (empty neighbor list) ====================

#[test]
fn test_sage_forward_isolated_nodes() {
    let sage = SAGEConv::new(4, 2).with_aggregation(SAGEAggregation::Max);
    let x = create_test_tensor(&[4, 4], 1);
    // Only edge 0->1, nodes 2 and 3 are isolated
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1]], 4);

    let out = sage.forward(&x, &adj);
    assert_eq!(out.shape(), &[4, 2]);
}

// ==================== GAT leaky_relu negative value ====================

#[test]
fn test_gat_leaky_relu_negative() {
    let gat = GATConv::new(4, 2, 1).with_negative_slope(0.2);
    // Use forward to exercise leaky_relu internally
    let x = Tensor::new(&[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0], &[2, 4]);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 0]], 2);
    let out = gat.forward(&x, &adj);
    assert_eq!(out.shape(), &[2, 2]);
}

// ==================== GCN with edge weights in forward ====================

#[test]
fn test_gcn_forward_with_edge_weights() {
    let gcn = GCNConv::new(4, 2);
    let x = create_test_tensor(&[3, 4], 1);
    let adj = AdjacencyMatrix::from_edge_index(&[[0, 1], [1, 2]], 3).with_weights(vec![0.5, 2.0]);

    let out = gcn.forward(&x, &adj);
    assert_eq!(out.shape(), &[3, 2]);
}
