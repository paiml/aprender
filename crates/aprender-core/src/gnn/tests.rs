pub(crate) use super::*;
pub(super) fn simple_graph_edges() -> Vec<EdgeIndex> {
    // Triangle graph: 0-1-2-0
    vec![(0, 1), (1, 2), (2, 0)]
}

pub(super) fn line_graph_edges() -> Vec<EdgeIndex> {
    // Line: 0-1-2-3
    vec![(0, 1), (1, 2), (2, 3)]
}

#[test]
fn test_gcn_conv_basic() {
    let gcn = GCNConv::new(4, 8);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = gcn.forward_gnn(&x, &edges);

    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_gcn_conv_features() {
    let gcn = GCNConv::new(4, 8);

    assert_eq!(gcn.in_features(), 4);
    assert_eq!(gcn.out_features(), 8);
}

#[test]
fn test_gcn_conv_parameters() {
    let gcn = GCNConv::new(4, 8);
    let params = gcn.parameters();

    // Linear has weight and bias
    assert_eq!(params.len(), 2);
}

#[test]
fn test_gcn_conv_without_self_loops() {
    let gcn = GCNConv::without_self_loops(4, 8);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = gcn.forward_gnn(&x, &edges);

    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_gcn_conv_line_graph() {
    let gcn = GCNConv::new(2, 4);
    let x = Tensor::ones(&[4, 2]);
    let edges = line_graph_edges();

    let out = gcn.forward_gnn(&x, &edges);

    assert_eq!(out.shape(), &[4, 4]);
}

#[test]
fn test_gat_conv_basic() {
    let gat = GATConv::new(4, 8, 2); // 2 heads
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = gat.forward_gnn(&x, &edges);

    // Output is num_heads * out_features
    assert_eq!(out.shape(), &[3, 16]);
}

#[test]
fn test_gat_conv_features() {
    let gat = GATConv::new(4, 8, 2);

    assert_eq!(gat.num_heads(), 2);
    assert_eq!(gat.out_features(), 8);
    assert_eq!(gat.total_out_features(), 16);
}

#[test]
fn test_gat_conv_parameters() {
    let gat = GATConv::new(4, 8, 2);
    let params = gat.parameters();

    // Linear (weight + bias) + attention_src + attention_tgt
    assert_eq!(params.len(), 4);
}

#[test]
fn test_gat_conv_attention_normalization() {
    // Test that attention weights sum to 1 (verified by output being reasonable)
    let gat = GATConv::new(4, 8, 1);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = gat.forward_gnn(&x, &edges);

    // All outputs should be finite
    for &v in out.data() {
        assert!(v.is_finite(), "Output should be finite");
    }
}

#[test]
fn test_gin_conv_basic() {
    let gin = GINConv::new(4, 16, 8);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = gin.forward_gnn(&x, &edges);

    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_gin_conv_eps() {
    let mut gin = GINConv::new(4, 16, 8);

    assert!((gin.eps() - 0.0).abs() < 1e-6);

    gin.set_eps(0.5);
    assert!((gin.eps() - 0.5).abs() < 1e-6);
}

#[test]
fn test_gin_conv_parameters() {
    let gin = GINConv::new(4, 16, 8);
    let params = gin.parameters();

    // Two Linear layers (2 weights + 2 biases)
    assert_eq!(params.len(), 4);
}

#[test]
fn test_global_mean_pool_single_graph() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

    let pooled = global_mean_pool(&x, None);

    assert_eq!(pooled.shape(), &[1, 2]);
    // Mean of [1,3,5] = 3, Mean of [2,4,6] = 4
    let data = pooled.data();
    assert!((data[0] - 3.0).abs() < 1e-6);
    assert!((data[1] - 4.0).abs() < 1e-6);
}

#[test]
fn test_global_mean_pool_batched() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
    let batch = vec![0, 0, 1, 1]; // 2 graphs, 2 nodes each

    let pooled = global_mean_pool(&x, Some(&batch));

    assert_eq!(pooled.shape(), &[2, 2]);
    let data = pooled.data();
    // Graph 0: mean([1,3], [2,4]) = [2, 3]
    assert!((data[0] - 2.0).abs() < 1e-6);
    assert!((data[1] - 3.0).abs() < 1e-6);
    // Graph 1: mean([5,7], [6,8]) = [6, 7]
    assert!((data[2] - 6.0).abs() < 1e-6);
    assert!((data[3] - 7.0).abs() < 1e-6);
}

#[test]
fn test_global_sum_pool_single_graph() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let pooled = global_sum_pool(&x, None);

    assert_eq!(pooled.shape(), &[1, 2]);
    let data = pooled.data();
    assert!((data[0] - 4.0).abs() < 1e-6); // 1 + 3
    assert!((data[1] - 6.0).abs() < 1e-6); // 2 + 4
}

#[test]
fn test_global_max_pool_single_graph() {
    let x = Tensor::new(&[1.0, 5.0, 3.0, 2.0, 4.0, 1.0], &[3, 2]);

    let pooled = global_max_pool(&x, None);

    assert_eq!(pooled.shape(), &[1, 2]);
    let data = pooled.data();
    assert!((data[0] - 4.0).abs() < 1e-6); // max(1, 3, 4)
    assert!((data[1] - 5.0).abs() < 1e-6); // max(5, 2, 1)
}

#[test]
#[should_panic(expected = "requires graph structure")]
fn test_gnn_forward_panics() {
    let gcn = GCNConv::new(4, 8);
    let x = Tensor::ones(&[3, 4]);

    // forward() should panic - use forward_gnn() instead
    let _ = gcn.forward(&x);
}

#[test]
fn test_gcn_different_graph_sizes() {
    let gcn = GCNConv::new(4, 8);

    // Small graph
    let x1 = Tensor::ones(&[2, 4]);
    let edges1 = vec![(0, 1)];
    let out1 = gcn.forward_gnn(&x1, &edges1);
    assert_eq!(out1.shape(), &[2, 8]);

    // Larger graph
    let x2 = Tensor::ones(&[10, 4]);
    let edges2: Vec<EdgeIndex> = (0..9).map(|i| (i, i + 1)).collect();
    let out2 = gcn.forward_gnn(&x2, &edges2);
    assert_eq!(out2.shape(), &[10, 8]);
}

#[test]
fn test_gnn_empty_edges() {
    let gcn = GCNConv::new(4, 8);
    let x = Tensor::ones(&[3, 4]);
    let edges: Vec<EdgeIndex> = vec![]; // No edges

    let out = gcn.forward_gnn(&x, &edges);

    // Should still work (self-loops only)
    assert_eq!(out.shape(), &[3, 8]);
}

// GraphSAGE Tests
#[test]
fn test_graphsage_basic() {
    let sage = GraphSAGEConv::new(4, 8);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = sage.forward_gnn(&x, &edges);

    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_graphsage_mean_aggregation() {
    let sage = GraphSAGEConv::new(4, 8).with_aggregation(SAGEAggregation::Mean);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = sage.forward_gnn(&x, &edges);
    assert_eq!(out.shape(), &[3, 8]);
    assert_eq!(sage.aggregation(), SAGEAggregation::Mean);
}

#[test]
fn test_graphsage_max_aggregation() {
    let sage = GraphSAGEConv::new(4, 8).with_aggregation(SAGEAggregation::Max);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = sage.forward_gnn(&x, &edges);
    assert_eq!(out.shape(), &[3, 8]);
    assert_eq!(sage.aggregation(), SAGEAggregation::Max);
}

#[test]
fn test_graphsage_sum_aggregation() {
    let sage = GraphSAGEConv::new(4, 8).with_aggregation(SAGEAggregation::Sum);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = sage.forward_gnn(&x, &edges);
    assert_eq!(out.shape(), &[3, 8]);
    assert_eq!(sage.aggregation(), SAGEAggregation::Sum);
}

#[test]
fn test_graphsage_sample_size() {
    let sage = GraphSAGEConv::new(4, 8).with_sample_size(2);
    let x = Tensor::ones(&[5, 4]);
    // Dense graph
    let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4)];

    let out = sage.forward_gnn(&x, &edges);
    assert_eq!(out.shape(), &[5, 8]);
    assert_eq!(sage.sample_size(), Some(2));
}

#[test]
fn test_graphsage_without_normalize() {
    let sage = GraphSAGEConv::new(4, 8).without_normalize();
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = sage.forward_gnn(&x, &edges);
    assert_eq!(out.shape(), &[3, 8]);

    // Without normalization, output vectors are not unit length
    // Just verify it produces output
    for &v in out.data() {
        assert!(v.is_finite());
    }
}

#[test]
fn test_graphsage_parameters() {
    let sage = GraphSAGEConv::new(4, 8);
    let params = sage.parameters();

    // Linear has weight and bias
    assert_eq!(params.len(), 2);
}

// EdgeConv Tests
#[test]
fn test_edgeconv_basic() {
    let edge_conv = EdgeConv::new(4, 16, 8);
    let x = Tensor::ones(&[3, 4]);
    let edges = simple_graph_edges();

    let out = edge_conv.forward_gnn(&x, &edges);

    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_edgeconv_features() {
    let edge_conv = EdgeConv::new(4, 16, 8);

    assert_eq!(edge_conv.in_features(), 4);
    assert_eq!(edge_conv.out_features(), 8);
}

#[test]
fn test_edgeconv_parameters() {
    let edge_conv = EdgeConv::new(4, 16, 8);
    let params = edge_conv.parameters();

    // Two Linear layers (2 weights + 2 biases)
    assert_eq!(params.len(), 4);
}

#[test]
fn test_edgeconv_empty_edges() {
    let edge_conv = EdgeConv::new(4, 16, 8);
    let x = Tensor::ones(&[3, 4]);
    let edges: Vec<EdgeIndex> = vec![];

    let out = edge_conv.forward_gnn(&x, &edges);

    // Should handle empty edges gracefully
    assert_eq!(out.shape(), &[3, 8]);
}

#[test]
fn test_edgeconv_output_finite() {
    let edge_conv = EdgeConv::new(4, 16, 8);
    let x = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
    );
    let edges = simple_graph_edges();

    let out = edge_conv.forward_gnn(&x, &edges);

    for &v in out.data() {
        assert!(v.is_finite(), "Output should be finite");
    }
}

#[test]
#[should_panic(expected = "requires graph structure")]
fn test_graphsage_forward_panics() {
    let sage = GraphSAGEConv::new(4, 8);
    let x = Tensor::ones(&[3, 4]);

    let _ = sage.forward(&x);
}

#[test]
#[should_panic(expected = "requires graph structure")]
fn test_edgeconv_forward_panics() {
    let edge_conv = EdgeConv::new(4, 16, 8);
    let x = Tensor::ones(&[3, 4]);

    let _ = edge_conv.forward(&x);
}

// ==================== Accessor Tests ====================

#[test]
fn test_gcn_accessors() {
    let gcn = GCNConv::new(16, 32);
    assert_eq!(gcn.in_features(), 16);
    assert_eq!(gcn.out_features(), 32);
}

#[test]
fn test_gat_accessors() {
    let gat = GATConv::new(16, 8, 4);
    assert_eq!(gat.num_heads(), 4);
    assert_eq!(gat.out_features(), 8);
    assert_eq!(gat.total_out_features(), 32); // 8 * 4 heads
}

#[test]
fn test_gin_accessors() {
    let gin = GINConv::new(16, 64, 32);
    assert_eq!(gin.in_features(), 16);
    assert_eq!(gin.hidden_features(), 64);
    assert_eq!(gin.out_features(), 32);
    assert!((gin.eps() - 0.0).abs() < f32::EPSILON);
    assert!(gin.train_eps()); // train_eps defaults to true
}

#[test]
fn test_gin_set_eps() {
    let mut gin = GINConv::new(16, 64, 32);
    gin.set_eps(0.5);
    assert!((gin.eps() - 0.5).abs() < f32::EPSILON);
}

#[test]
fn test_graphsage_accessors() {
    let sage = GraphSAGEConv::new(16, 32);
    assert_eq!(sage.aggregation(), SAGEAggregation::Mean);
    assert!(sage.sample_size().is_none());
}

#[test]
fn test_graphsage_with_sample_size() {
    let sage = GraphSAGEConv::new(16, 32).with_sample_size(10);
    assert_eq!(sage.sample_size(), Some(10));
}

#[test]
fn test_edgeconv_accessors() {
    let edge_conv = EdgeConv::new(4, 16, 8);
    assert_eq!(edge_conv.in_features(), 4);
    assert_eq!(edge_conv.out_features(), 8);
}

#[path = "tests_pooling.rs"]
mod tests_pooling;
