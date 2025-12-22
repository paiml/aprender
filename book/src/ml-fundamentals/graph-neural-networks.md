# Graph Neural Networks Theory

Graph Neural Networks (GNNs) extend deep learning to graph-structured data, enabling learning on social networks, molecules, knowledge graphs, and more.

## Why Graphs?

Many real-world systems are naturally graphs:

| Domain | Nodes | Edges |
|--------|-------|-------|
| Social Networks | Users | Friendships |
| Molecules | Atoms | Bonds |
| Knowledge Graphs | Entities | Relations |
| Citation Networks | Papers | Citations |
| Traffic | Intersections | Roads |

Traditional neural networks require fixed-size inputs. GNNs handle:
- Variable number of nodes
- Variable node connectivity
- Permutation invariance (node ordering doesn't matter)

## Message Passing Framework

Most GNNs follow the **message passing** paradigm:

```
For each layer:
  1. AGGREGATE: Collect messages from neighbors
  2. UPDATE: Transform aggregated messages
  3. COMBINE: Merge with node's own features
```

Mathematically:

```
h_v^(l+1) = UPDATE(h_v^(l), AGGREGATE({h_u^(l) : u ∈ N(v)}))
```

Where:
- h_v^(l) = node v's representation at layer l
- N(v) = neighbors of node v

## Graph Convolutional Network (GCN)

Kipf & Welling (2017) introduced GCN with symmetric normalization:

```
H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
```

Where:
- Ã = A + I (adjacency with self-loops)
- D̃ = degree matrix of Ã
- W = learnable weight matrix
- σ = activation function

**Per-node formulation:**

```
h_i' = σ(Σⱼ (1/√(dᵢdⱼ)) · W · hⱼ)
```

The normalization 1/√(dᵢdⱼ) prevents feature explosion in high-degree nodes.

## Graph Attention Network (GAT)

Velickovic et al. (2018) introduced attention to learn edge importance:

```
α_ij = softmax_j(LeakyReLU(aᵀ[Wh_i || Wh_j]))
h_i' = σ(Σⱼ α_ij · W · hⱼ)
```

**Multi-head attention:**

```
h_i' = ||ₖ₌₁ᴷ σ(Σⱼ α_ij^k · W^k · hⱼ)
```

Where || denotes concatenation across K attention heads.

**Advantages over GCN:**
- Learns which neighbors are important
- Handles heterogeneous graphs better
- More expressive aggregation

## GraphSAGE

Hamilton et al. (2017) introduced sampling and aggregation:

```
h_N(v) = AGGREGATE({h_u : u ∈ Sample(N(v), k)})
h_v' = σ(W · [h_v || h_N(v)])
```

**Aggregation functions:**
- Mean: h_N(v) = mean({h_u})
- Max-pooling: h_N(v) = max({σ(W_pool · h_u)})
- LSTM: h_N(v) = LSTM({h_u}) (permutation variant)

**Key innovation:** Samples fixed-size neighborhood for scalability.

## Comparison of GNN Architectures

| Architecture | Aggregation | Normalization | Attention |
|--------------|-------------|---------------|-----------|
| GCN | Sum | Symmetric | No |
| GAT | Weighted sum | None | Yes |
| GraphSAGE | Mean/Max/LSTM | None | No |
| GIN | Sum + MLP | None | No |

## Expressive Power

**Weisfeiler-Lehman Test:**
GNNs are at most as powerful as the 1-WL graph isomorphism test.

```
Two nodes get same embedding if and only if
they have the same 1-WL color after k iterations.
```

**Graph Isomorphism Network (GIN):**
Xu et al. (2019) designed maximally expressive GNN:

```
h_v' = MLP((1 + ε) · h_v + Σⱼ h_j)
```

This achieves theoretical maximum expressiveness under the WL framework.

## Over-Smoothing Problem

**Issue:** Deep GNNs make all node embeddings converge:

```
Layer 1: h_v distinct
Layer 2: h_v similar to neighbors
Layer 3: h_v similar to 2-hop neighbors
...
Layer k: All h_v nearly identical
```

**Solutions:**
1. **Skip connections:** h' = h + GNN(h)
2. **Jumping Knowledge:** Concat all layer outputs
3. **DropEdge:** Randomly remove edges during training
4. **PairNorm:** Normalize to maintain separation

## Node Classification

**Task:** Predict labels for nodes given partial labels.

```
Input: Graph G, node features X, labels Y_L for subset L
Output: Labels for unlabeled nodes
```

**Architecture:**

```
X → GCN → ReLU → Dropout → GCN → Softmax → Ŷ
```

**Loss:** Cross-entropy on labeled nodes:

```
L = -Σᵢ∈L Σc y_ic · log(ŷ_ic)
```

## Graph Classification

**Task:** Predict label for entire graphs.

```
Input: Set of graphs {G_1, G_2, ...} with labels
Output: Graph-level classifier
```

**Readout (pooling):**

```
h_G = READOUT({h_v : v ∈ G})
```

Common readouts:
- Mean: h_G = mean(h_v)
- Sum: h_G = Σ h_v
- Set2Set: Attention-based
- DiffPool: Hierarchical clustering

## Link Prediction

**Task:** Predict missing edges.

```
Input: Graph with some edges removed
Output: Score for each potential edge
```

**Scoring function:**

```
score(u, v) = h_u · h_v  (dot product)
score(u, v) = MLP([h_u || h_v])  (neural)
```

## Heterogeneous Graphs

Graphs with multiple node/edge types:

```
RGCN: h_v' = σ(Σᵣ Σⱼ (1/|N_r(v)|) · Wᵣ · hⱼ)
```

Where r indexes relation types.

## Temporal Graphs

Graphs evolving over time:

```
h_v^(t+1) = GNN(h_v^(t), Graph^(t))
```

Combine GNN with sequence models (LSTM, Transformer).

## Computational Considerations

### Mini-batching

Sampling strategies for large graphs:
1. **Node sampling:** Random subset of nodes
2. **Layer sampling:** Sample neighbors per layer (GraphSAGE)
3. **Subgraph sampling:** Extract connected subgraphs

### Sparse Operations

Use sparse matrix operations for efficiency:

```
# Dense: O(N²) memory
H' = A @ H @ W

# Sparse: O(E) memory
H' = sparse_mm(A, H) @ W
```

## Implementation Notes

### Edge Index Format

COO (Coordinate) format:

```
edge_index = [(0, 1), (1, 2), (2, 0), ...]
             source   target
```

### Self-Loops

Adding self-loops (A + I):
- Ensures node's own features contribute
- Prevents information loss in disconnected nodes
- Required for GCN normalization

## References

- Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.
- Velickovic, P., et al. (2018). "Graph Attention Networks." ICLR.
- Hamilton, W. L., et al. (2017). "Inductive Representation Learning on Large Graphs." NeurIPS.
- Xu, K., et al. (2019). "How Powerful are Graph Neural Networks?" ICLR.
