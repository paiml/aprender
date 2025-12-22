# Case Study: Graph Neural Networks for Node Classification

This case study demonstrates Aprender's GNN module for learning on graph-structured data.

## Overview

The `gnn` module provides:
- **GCNConv**: Graph Convolutional Network layer
- **GATConv**: Graph Attention Network layer
- **GNNModule trait**: Interface for graph-aware layers

## Basic GCN Usage

```rust
use aprender::gnn::{GCNConv, GNNModule, EdgeIndex};
use aprender::autograd::Tensor;

fn main() {
    // Create GCN layer: 16 input features → 32 output features
    let gcn = GCNConv::new(16, 32);

    // Node features: 4 nodes, 16 features each
    let x = Tensor::ones(&[4, 16]);

    // Graph structure: a simple cycle 0 → 1 → 2 → 3 → 0
    let edge_index: Vec<EdgeIndex> = vec![
        (0, 1), (1, 0),  // Edge 0-1 (bidirectional)
        (1, 2), (2, 1),  // Edge 1-2
        (2, 3), (3, 2),  // Edge 2-3
        (3, 0), (0, 3),  // Edge 3-0
    ];

    // Forward pass
    let out = gcn.forward_gnn(&x, &edge_index);

    assert_eq!(out.shape(), &[4, 32]);
    println!("Output shape: {:?}", out.shape());
}
```

## Multi-Layer GCN

```rust
use aprender::gnn::{GCNConv, GNNModule, EdgeIndex};
use aprender::autograd::Tensor;
use aprender::nn::Module;

struct GCN {
    conv1: GCNConv,
    conv2: GCNConv,
}

impl GCN {
    fn new(in_features: usize, hidden: usize, out_features: usize) -> Self {
        Self {
            conv1: GCNConv::new(in_features, hidden),
            conv2: GCNConv::new(hidden, out_features),
        }
    }

    fn forward(&self, x: &Tensor, edge_index: &[EdgeIndex]) -> Tensor {
        // Layer 1: Input → Hidden with ReLU
        let h = self.conv1.forward_gnn(x, edge_index).relu();

        // Layer 2: Hidden → Output (no activation for logits)
        self.conv2.forward_gnn(&h, edge_index)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params
    }
}
```

## Node Classification Task

```rust
use aprender::gnn::{GCNConv, GNNModule, EdgeIndex};
use aprender::autograd::{Tensor, clear_graph, no_grad};
use aprender::nn::Module;

fn train_node_classifier() {
    // Karate Club graph (simplified)
    // 34 nodes, 2 classes (communities)
    let num_nodes = 34;
    let num_features = 34;  // One-hot encoding
    let num_classes = 2;

    // Create model
    let mut model = GCN::new(num_features, 16, num_classes);

    // Node features: identity matrix (each node is unique)
    let x = Tensor::eye(num_nodes);

    // Graph edges (simplified subset)
    let edge_index: Vec<EdgeIndex> = vec![
        (0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0),
        (1, 2), (2, 1), (2, 3), (3, 2),
        // ... more edges
    ];

    // Labels for some nodes (semi-supervised)
    let labeled_nodes = vec![0, 33];  // First and last node
    let labels = vec![0, 1];  // Different communities

    let lr = 0.01;
    let epochs = 200;

    for epoch in 0..epochs {
        // Forward pass
        let logits = model.forward(&x, &edge_index);

        // Compute loss only on labeled nodes
        let mut loss_val = 0.0;
        for (&node, &label) in labeled_nodes.iter().zip(labels.iter()) {
            let node_logits = logits.select(0, node);
            let probs = node_logits.softmax();

            // Cross-entropy loss
            let log_prob = probs.log();
            loss_val -= log_prob.data()[label];
        }

        let loss = Tensor::from_slice(&[loss_val as f32]);
        loss.backward();

        // Update parameters
        no_grad(|| {
            for param in model.parameters() {
                if let Some(grad) = param.grad() {
                    let update = grad.mul(&Tensor::from_slice(&[lr]));
                    // param = param - lr * grad
                }
            }
        });

        clear_graph();

        if epoch % 50 == 0 {
            println!("Epoch {}: loss = {:.4}", epoch, loss_val);
        }
    }

    // Inference
    no_grad(|| {
        let logits = model.forward(&x, &edge_index);
        let predictions = logits.argmax(1);
        println!("Predictions: {:?}", predictions.data());
    });
}
```

## Graph Attention Network

```rust
use aprender::gnn::{GATConv, GNNModule, EdgeIndex};
use aprender::autograd::Tensor;

fn main() {
    // GAT with 4 attention heads
    let gat = GATConv::new(16, 8, 4);  // 16 in → 8*4=32 out

    let x = Tensor::ones(&[4, 16]);
    let edge_index: Vec<EdgeIndex> = vec![
        (0, 1), (1, 2), (2, 3), (3, 0),
    ];

    let out = gat.forward_gnn(&x, &edge_index);
    println!("GAT output: {:?}", out.shape());  // [4, 32]

    // Access attention weights for interpretability
    let attention = gat.get_attention_weights();
    println!("Attention on edge (0,1): {:.3}", attention[&(0, 1)]);
}
```

## Building Graph from Data

```rust
use aprender::gnn::EdgeIndex;

/// Build edge index from adjacency list
fn adjacency_list_to_edges(adj: &[Vec<usize>]) -> Vec<EdgeIndex> {
    let mut edges = Vec::new();
    for (src, neighbors) in adj.iter().enumerate() {
        for &tgt in neighbors {
            edges.push((src, tgt));
        }
    }
    edges
}

/// Build edge index from adjacency matrix
fn adjacency_matrix_to_edges(adj: &[Vec<f32>]) -> Vec<EdgeIndex> {
    let mut edges = Vec::new();
    for (i, row) in adj.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val > 0.0 {
                edges.push((i, j));
            }
        }
    }
    edges
}

fn main() {
    // From adjacency list
    let adj_list = vec![
        vec![1, 2],     // Node 0 connects to 1, 2
        vec![0, 2],     // Node 1 connects to 0, 2
        vec![0, 1, 3],  // Node 2 connects to 0, 1, 3
        vec![2],        // Node 3 connects to 2
    ];
    let edges = adjacency_list_to_edges(&adj_list);
    println!("Edges: {:?}", edges);
}
```

## Handling Self-Loops

```rust
use aprender::gnn::{GCNConv, GNNModule, EdgeIndex};
use aprender::autograd::Tensor;

fn main() {
    // GCN with self-loops (default)
    let gcn_with_loops = GCNConv::new(16, 32);

    // GCN without self-loops
    let gcn_no_loops = GCNConv::without_self_loops(16, 32);

    let x = Tensor::ones(&[4, 16]);
    let edges: Vec<EdgeIndex> = vec![(0, 1), (1, 2), (2, 3)];

    // With self-loops: nodes aggregate their own features
    let out1 = gcn_with_loops.forward_gnn(&x, &edges);

    // Without: only neighbor features (isolated nodes get zero)
    let out2 = gcn_no_loops.forward_gnn(&x, &edges);

    println!("With self-loops: node features preserved");
    println!("Without: isolated nodes may lose information");
}
```

## Graph Batching

Process multiple graphs as a single disconnected graph:

```rust
use aprender::gnn::EdgeIndex;

struct BatchedGraph {
    x: Vec<f32>,           // Concatenated node features
    edge_index: Vec<EdgeIndex>,
    batch: Vec<usize>,     // Graph ID for each node
}

fn batch_graphs(graphs: &[(Vec<f32>, Vec<EdgeIndex>, usize)]) -> BatchedGraph {
    let mut x = Vec::new();
    let mut edge_index = Vec::new();
    let mut batch = Vec::new();

    let mut node_offset = 0;

    for (graph_id, (features, edges, num_nodes)) in graphs.iter().enumerate() {
        // Add node features
        x.extend(features);

        // Add edges with offset
        for &(src, tgt) in edges {
            edge_index.push((src + node_offset, tgt + node_offset));
        }

        // Record which graph each node belongs to
        for _ in 0..*num_nodes {
            batch.push(graph_id);
        }

        node_offset += num_nodes;
    }

    BatchedGraph { x, edge_index, batch }
}
```

## Graph-Level Prediction

```rust
use aprender::gnn::{GCNConv, GNNModule, EdgeIndex};
use aprender::autograd::Tensor;
use aprender::nn::{Linear, Module};

struct GraphClassifier {
    conv1: GCNConv,
    conv2: GCNConv,
    fc: Linear,
}

impl GraphClassifier {
    fn new(in_features: usize, hidden: usize, num_classes: usize) -> Self {
        Self {
            conv1: GCNConv::new(in_features, hidden),
            conv2: GCNConv::new(hidden, hidden),
            fc: Linear::new(hidden, num_classes),
        }
    }

    fn forward(&self, x: &Tensor, edge_index: &[EdgeIndex], batch: &[usize]) -> Tensor {
        // Node-level embeddings
        let h = self.conv1.forward_gnn(x, edge_index).relu();
        let h = self.conv2.forward_gnn(&h, edge_index).relu();

        // Global mean pooling per graph
        let graph_embeddings = global_mean_pool(&h, batch);

        // Graph-level prediction
        self.fc.forward(&graph_embeddings)
    }
}

fn global_mean_pool(h: &Tensor, batch: &[usize]) -> Tensor {
    let num_graphs = batch.iter().max().map(|&m| m + 1).unwrap_or(0);
    let hidden_dim = h.shape()[1];

    let mut pooled = vec![0.0f32; num_graphs * hidden_dim];
    let mut counts = vec![0usize; num_graphs];

    let h_data = h.data();
    for (node_idx, &graph_id) in batch.iter().enumerate() {
        counts[graph_id] += 1;
        for f in 0..hidden_dim {
            pooled[graph_id * hidden_dim + f] += h_data[node_idx * hidden_dim + f];
        }
    }

    // Average
    for graph_id in 0..num_graphs {
        if counts[graph_id] > 0 {
            for f in 0..hidden_dim {
                pooled[graph_id * hidden_dim + f] /= counts[graph_id] as f32;
            }
        }
    }

    Tensor::new(&pooled, &[num_graphs, hidden_dim])
}
```

## Feature Initialization

```rust
use aprender::autograd::Tensor;

/// One-hot encoding for node IDs
fn one_hot_features(num_nodes: usize) -> Tensor {
    Tensor::eye(num_nodes)
}

/// Degree-based features
fn degree_features(edge_index: &[(usize, usize)], num_nodes: usize) -> Tensor {
    let mut degrees = vec![0.0f32; num_nodes];
    for &(src, _) in edge_index {
        degrees[src] += 1.0;
    }

    // Normalize
    let max_deg = degrees.iter().cloned().fold(1.0, f32::max);
    for d in &mut degrees {
        *d /= max_deg;
    }

    Tensor::new(&degrees, &[num_nodes, 1])
}

/// Random features (for structure-only learning)
fn random_features(num_nodes: usize, dim: usize) -> Tensor {
    let data: Vec<f32> = (0..num_nodes * dim)
        .map(|_| rand::random::<f32>())
        .collect();
    Tensor::new(&data, &[num_nodes, dim])
}
```

## Running Examples

```bash
# Basic GCN
cargo run --example gnn_basic

# Node classification
cargo run --example gnn_node_classification

# Graph classification
cargo run --example gnn_graph_classification
```

## References

- Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.
- Velickovic et al. (2018). "Graph Attention Networks." ICLR.
- Hamilton et al. (2017). "Inductive Representation Learning on Large Graphs." NeurIPS.
