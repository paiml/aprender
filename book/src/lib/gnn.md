<!-- PCU: lib-gnn | contract: contracts/apr-page-lib-gnn-v1.yaml -->

# Module: `aprender::gnn`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/gnn/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/gnn/mod.rs) or directory.

## Example

```rust
use aprender::gnn::{GCNConv, GATConv, GINConv};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::gnn` is the differentiable graph-neural-network layer kit:
Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), Graph
Isomorphism Networks (GIN), and GraphSAGE, plus the readout / pooling
functions (`global_mean_pool`, `global_sum_pool`, `global_max_pool`) that
aggregate node-level features into graph-level embeddings. The `GNNModule`
trait extends `nn::Module` with `forward(x, edge_index)` so GNN layers can
slot into the standard `Sequential` container.

## Key types

| Type | Description |
|------|-------------|
| `GNNModule` | Trait extending `Module`. `forward(x, edge_index)`. |
| `GCNConv` | Kipf & Welling graph-convolutional layer (D⁻¹/² A D⁻¹/² X W). |
| `GATConv` | Velickovic et al. graph-attention layer with multi-head attention. |
| `GINConv` | Xu et al. graph-isomorphism layer with learnable epsilon. |
| `GraphSAGEConv` | Hamilton et al. inductive aggregation layer. |
| `EdgeIndex` | Type alias `(usize, usize)` for an edge tuple. |
| `global_mean_pool`, `global_sum_pool`, `global_max_pool` | Readout aggregations. |

The same `GCNConv` / `GATConv` types are also re-exported by `nn::gnn` for
convenience inside neural-network module trees.

## Usage patterns

### Pattern 1: Configure a single GCN layer

```rust
use aprender::gnn::GCNConv;

// 16-D node features → 8-D hidden representation.
let gcn = GCNConv::new(16, 8);
// In a real training loop you'd call gcn.forward(&x, &edge_index)
// and compose with non-linearities + readouts.
let _ = gcn;
```

### Pattern 2: Build a GIN block

```rust
use aprender::gnn::GINConv;

let gin = GINConv::new(64, 128, 64);
assert_eq!(gin.in_features(), 64);
assert_eq!(gin.hidden_features(), 128);
assert_eq!(gin.out_features(), 64);
assert!(!gin.train_eps(), "default GIN keeps epsilon fixed");
```

## See also

- [`graph`](graph.md) — classical graph algorithms and CSR data structure
- [`nn`](nn.md) — `Module` and `Sequential` host these layers
- [`autograd`](autograd.md) — tensor + backward pass driven by GNN forwards
- [`models`](models.md) — full GNN architectures combining layers + readouts

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
