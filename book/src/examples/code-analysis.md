# Code Analysis with Code2Vec and MPNN

This chapter demonstrates aprender's code analysis capabilities using Code2Vec embeddings and Message Passing Neural Networks (MPNN).

## Overview

The `aprender::code` module provides tools for:
- **AST Representation**: Lightweight AST node types for code structures
- **Path Extraction**: Code2Vec-style paths between terminal nodes
- **Code Embeddings**: Dense vector representations of code
- **Graph Neural Networks**: MPNN for type/lifetime propagation

## Use Cases

| Application | Description |
|-------------|-------------|
| Code Similarity | Find similar functions across codebases |
| Function Naming | Predict meaningful function names |
| Type Inference | Propagate types through data flow |
| Bug Detection | Identify anomalous code patterns |

## Quick Start

```rust
use aprender::code::{
    AstNode, AstNodeType, Code2VecEncoder, PathExtractor,
    CodeGraph, CodeGraphNode, CodeGraphEdge, CodeEdgeType, CodeMPNN,
};

// Build an AST
let mut func = AstNode::new(AstNodeType::Function, "add");
func.add_child(AstNode::new(AstNodeType::Parameter, "x"));
func.add_child(AstNode::new(AstNodeType::Parameter, "y"));
func.add_child(AstNode::new(AstNodeType::Return, "result"));

// Extract Code2Vec paths
let extractor = PathExtractor::new(8);
let paths = extractor.extract(&func);

// Generate embedding
let encoder = Code2VecEncoder::new(128);
let embedding = encoder.aggregate_paths(&paths);
println!("Embedding dimension: {}", embedding.dim());
```

## AST Representation

The module provides 24 AST node types covering common code constructs:

### Node Types

| Category | Types |
|----------|-------|
| Definitions | `Function`, `Struct`, `Enum`, `Trait`, `Impl`, `Module` |
| Statements | `Variable`, `Assignment`, `Return`, `Conditional`, `Loop`, `Match` |
| Expressions | `BinaryOp`, `UnaryOp`, `Call`, `Literal`, `Index`, `FieldAccess` |
| Types | `TypeAnnotation`, `Generic`, `Parameter` |
| Other | `Block`, `MatchArm`, `Import` |

### Token Types

| Type | Description |
|------|-------------|
| `Identifier` | Variable/function names |
| `Number` | Numeric literals |
| `String` | String literals |
| `TypeName` | Type names |
| `Operator` | Operators (+, -, *, /) |
| `Keyword` | Language keywords |

## Code2Vec Path Extraction

Paths connect terminal nodes (leaves) through their lowest common ancestor:

```
fn add(x, y) -> x + y

Paths extracted:
  x → Param ↑ Func ↓ Param → y
  x → Param ↑ Func ↓ Return ↓ BinaryOp → result
  ...
```

### Path Extractor Configuration

```rust
let extractor = PathExtractor::new(8)  // Max path length
    .with_max_paths(200);              // Max paths per method

let paths = extractor.extract(&ast);
let contexts = extractor.extract_with_context(&ast);  // With position info
```

## Code Embeddings

The `Code2VecEncoder` generates dense vector representations:

```rust
let encoder = Code2VecEncoder::new(128)  // Embedding dimension
    .with_seed(42);                      // Reproducible

// Single path embedding
let path_emb = encoder.encode_path(&path);

// Aggregate all paths with attention
let code_emb = encoder.aggregate_paths(&paths);

// Access attention weights for interpretability
if let Some(weights) = code_emb.attention_weights() {
    println!("Most attended path weight: {:.3}", weights[0]);
}
```

### Code Similarity

```rust
let emb1 = encoder.aggregate_paths(&paths1);
let emb2 = encoder.aggregate_paths(&paths2);

let similarity = emb1.cosine_similarity(&emb2);
println!("Similarity: {:.4}", similarity);
```

## Code Graph Neural Networks

For more complex analysis, use MPNN on code graphs:

### Edge Types

| Edge Type | Description |
|-----------|-------------|
| `ControlFlow` | CFG edges |
| `DataFlow` | Def-use chains |
| `AstChild` | AST parent-child |
| `TypeAnnotation` | Type relationships |
| `Ownership` | Borrow/ownership |
| `Call` | Function calls |
| `Return` | Return edges |

### Building a Code Graph

```rust
use aprender::code::{
    CodeGraph, CodeGraphNode, CodeGraphEdge, CodeEdgeType,
};

let mut graph = CodeGraph::new();

// Add nodes with features
graph.add_node(CodeGraphNode::new(0, vec![1.0, 0.0, 0.0], "variable"));
graph.add_node(CodeGraphNode::new(1, vec![0.0, 1.0, 0.0], "variable"));
graph.add_node(CodeGraphNode::new(2, vec![0.0, 0.0, 1.0], "function"));

// Add typed edges
graph.add_edge(CodeGraphEdge::new(0, 2, CodeEdgeType::DataFlow));
graph.add_edge(CodeGraphEdge::new(1, 2, CodeEdgeType::DataFlow));
```

### MPNN Forward Pass

```rust
use aprender::code::{CodeMPNN, pooling};

// Create MPNN with layer dimensions
let mpnn = CodeMPNN::new(&[3, 16, 8, 4]);  // 3 -> 16 -> 8 -> 4

// Forward pass
let node_embeddings = mpnn.forward(&graph);

// Graph-level embedding via pooling
let graph_emb = pooling::mean_pool(&node_embeddings);
// Also available: max_pool, sum_pool
```

## Complete Example

```rust
use aprender::code::{
    pooling, AstNode, AstNodeType, Code2VecEncoder, CodeEdgeType,
    CodeGraph, CodeGraphEdge, CodeGraphNode, CodeMPNN, PathExtractor,
};

fn main() {
    // 1. Build AST for: fn add(x, y) -> x + y
    let mut func = AstNode::new(AstNodeType::Function, "add");
    func.add_child(AstNode::new(AstNodeType::Parameter, "x"));
    func.add_child(AstNode::new(AstNodeType::Parameter, "y"));

    let mut body = AstNode::new(AstNodeType::Block, "body");
    let mut op = AstNode::new(AstNodeType::BinaryOp, "+");
    op.add_child(AstNode::new(AstNodeType::Variable, "x"));
    op.add_child(AstNode::new(AstNodeType::Variable, "y"));

    let mut ret = AstNode::new(AstNodeType::Return, "return");
    ret.add_child(op);
    body.add_child(ret);
    func.add_child(body);

    // 2. Extract paths and generate embedding
    let extractor = PathExtractor::new(8);
    let paths = extractor.extract(&func);
    println!("Extracted {} paths", paths.len());

    let encoder = Code2VecEncoder::new(64);
    let embedding = encoder.aggregate_paths(&paths);
    println!("Function embedding: {} dimensions", embedding.dim());

    // 3. Build code graph for MPNN
    let mut graph = CodeGraph::new();
    graph.add_node(CodeGraphNode::new(0, vec![1.0, 0.0], "param_x"));
    graph.add_node(CodeGraphNode::new(1, vec![0.0, 1.0], "param_y"));
    graph.add_node(CodeGraphNode::new(2, vec![0.5, 0.5], "add_op"));

    graph.add_edge(CodeGraphEdge::new(0, 2, CodeEdgeType::DataFlow));
    graph.add_edge(CodeGraphEdge::new(1, 2, CodeEdgeType::DataFlow));

    // 4. Run MPNN
    let mpnn = CodeMPNN::new(&[2, 8, 4]);
    let node_embs = mpnn.forward(&graph);
    let graph_emb = pooling::mean_pool(&node_embs);

    println!("Graph embedding: {:?}", &graph_emb[..4]);
}
```

## Running the Example

```bash
cargo run --example code_analysis
```

Output:
```
=== Code Analysis with Code2Vec and MPNN ===

1. Building AST for a simple function
   Function: fn add(x: i32, y: i32) -> i32 { x + y }

   AST Structure:
   Func: add
     Param: x
       Type: i32
     Param: y
       Type: i32
     Type: i32
     Block: body
       Ret: return
         BinOp: +
           Var: x
           Var: y

2. Extracting Code2Vec Paths
   Found 10 paths between terminal nodes

3. Generating Code Embeddings
   Function embedding dim: 64
   Attention weights (first 3): [0.111, 0.115, 0.086]

4. Computing Code Similarity
   add() vs sum():      0.3964 (similar structure)
   add() vs multiply(): -0.5212 (different operation)
...
```

## References

- Alon et al. (2019), "code2vec: Learning distributed representations of code"
- Allamanis et al. (2018), "A survey of machine learning for big code"
- Gilmer et al. (2017), "Neural Message Passing for Quantum Chemistry"

## See Also

- [Graph Algorithms](./graph-algorithms-comprehensive.md) - General graph analysis
<!-- GNN Module API reference not yet written -->
- [Text Processing](./text-preprocessing.md) - NLP for code comments
