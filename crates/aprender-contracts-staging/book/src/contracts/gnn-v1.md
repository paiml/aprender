# gnn-v1

**Version:** 1.0.0

Graph Neural Network layers and pooling operations

## References

- Kipf & Welling (2017) Semi-Supervised Classification with Graph Convolutional Networks
- Gilmer et al. (2017) Neural Message Passing for Quantum Chemistry

## Equations

### gcn_aggregate

$$
H^{l+1} = sigma(D_hat^{-1/2} * A_hat * D_hat^{-1/2} * H^{l} * W^{l})
$$

**Domain:** $A_hat = A + I adjacency with self-loops, D_hat = degree matrix of A_hat, H in R^{n x d_in}, W in R^{d_in x d_out}$

**Codomain:** $H' in R^{n x d_out}$

**Invariants:**

- $Output has same number of nodes as input (n preserved)$
- $Output feature dimension equals weight matrix output dimension$
- $Self-loops ensure every node receives its own features$

### global_max_pool

$$
r_j = max_{i in V} h_{ij} for each feature dimension j
$$

**Domain:** $h_i in R^d for i in V, |V| >= 1$

**Codomain:** $r in R^d (graph-level embedding)$

**Invariants:**

- $Output dimension equals node feature dimension$
- $Output is bounded by maximum node feature value per dimension$
- $r_j >= h_{ij} for all i (max is an upper bound)$

### global_mean_pool

$$
r = (1/|V|) * sum_{i in V} h_i
$$

**Domain:** $h_i in R^d for i in V, |V| >= 1$

**Codomain:** $r in R^d (graph-level embedding)$

**Invariants:**

- $Output dimension equals node feature dimension$
- $Output is finite when all node features are finite$
- $Output is bounded: min(h) <= r_j <= max(h) for each dimension j$

### message_passing

$$
h_i^{l+1} = U(h_i^{l}, aggregate_{j in N(i)} M(h_i, h_j))
$$

**Domain:** $h_i in R^d node features, N(i) neighborhood of node i, M message function, U update function$

**Codomain:** $h_i' in R^{d'} updated node features$

**Invariants:**

- $Output has same number of nodes as input$
- $Each node is updated based only on its neighborhood (locality)$
- $Permutation equivariant with respect to node ordering$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | GCN preserves node count | `output.shape[0] == input.shape[0] for GCN forward` |
| 2 | invariant | Message passing preserves node count | `propagate(x, adj).shape[0] == x.shape[0]` |
| 3 | bound | Global mean pool output is finite | `forall j: r_j.is_finite() when all h_ij are finite` |
| 4 | bound | Global max pool bounded by node features | $forall j: r_j <= max_{i in V}(h_{ij})$ |
| 5 | invariant | Pooling output dimension matches feature dimension | `pool(H).shape[1] == H.shape[1]` |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-GNN-001 | GCN node count preservation | GCN output has same number of nodes as input | GCN aggregation drops or duplicates nodes |
| FALSIFY-GNN-002 | Message passing node count preservation | Propagate preserves node count | Message passing aggregation changes node set |
| FALSIFY-GNN-003 | Global mean pool finiteness | Pooled output is finite for finite inputs | Division by zero (empty graph) or overflow in summation |
| FALSIFY-GNN-004 | Global max pool upper bound | Each pooled feature <= max of corresponding node features | Max pool introduces values exceeding node feature range |
| FALSIFY-GNN-005 | Pooling dimension preservation | Pool output has same feature dimension as input | Pooling operation changes feature dimension |
| FALSIFY-GNN-006 | GCN output finiteness | GCN output is finite for finite inputs and weights | Numerical instability in degree normalization or matmul |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-GNN-001 | GNN-INV-001 | 8 | stub_float |
| KANI-GNN-002 | GNN-BND-001 | 8 | stub_float |
| KANI-GNN-003 | GNN-BND-002 | 8 | stub_float |
| KANI-GNN_V1-004 | GCN preserves node count | 8 | exhaustive |
| KANI-GNN_V1-005 | Message passing preserves node count | 8 | exhaustive |
| KANI-GNN_V1-006 | Global mean pool output is finite | 8 | exhaustive |
| KANI-GNN_V1-007 | Global max pool bounded by node features | 8 | stub_float |
| KANI-GNN_V1-008 | Pooling output dimension matches feature dimension | 8 | exhaustive |

## QA Gate

**GNN Contract** (F-GNN-001)

Graph neural network layer and pooling correctness quality gate

**Checks:** gcn_node_count, message_passing_node_count, mean_pool_finiteness, max_pool_bound, pooling_dimension, gcn_finiteness

**Pass criteria:** All 6 falsification tests pass + Kani harnesses verify

