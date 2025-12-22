# Case Study: TensorLogic Neuro-Symbolic Reasoning

This case study demonstrates TensorLogic, a neuro-symbolic reasoning system that combines neural network learning with logical inference using tensor operations.

## Overview

TensorLogic enables:
- **Differentiable Logic**: Logical operations that support gradient-based learning
- **Knowledge Graph Inference**: Forward and backward chaining over knowledge bases
- **Weighted Logic Programming**: Probabilistic inference with uncertainty quantification
- **Neural-Symbolic Integration**: Combining learned representations with symbolic reasoning

## Example: Family Tree Reasoning

```rust
use aprender::logic::{
    KnowledgeBase, LogicalTensor, TensorLogicEngine,
    logical_join, logical_project, logical_select, logical_aggregate,
};

fn main() {
    // Create a knowledge base with family relationships
    let mut kb = KnowledgeBase::new();

    // Add facts: parent(X, Y) means X is parent of Y
    // Alice -> Bob -> Charlie -> David
    kb.add_fact("parent", vec!["Alice", "Bob"]);
    kb.add_fact("parent", vec!["Bob", "Charlie"]);
    kb.add_fact("parent", vec!["Charlie", "David"]);

    // Create TensorLogic engine
    let engine = TensorLogicEngine::new();

    // Convert to logical tensors (4x4 binary matrices)
    let parent = engine.relation_to_tensor(&kb, "parent");

    // Compute grandparent = parent . parent (matrix multiplication)
    let grandparent = logical_join(&parent, &parent);

    // Query: Who is grandparent of Charlie?
    let result = logical_select(&grandparent, "Charlie");
    println!("Grandparent of Charlie: {:?}", result);
    // Output: Alice

    // Compute great-grandparent
    let great_grandparent = logical_join(&grandparent, &parent);
    println!("Great-grandparent of David: {:?}",
             logical_select(&great_grandparent, "David"));
    // Output: Alice
}
```

## Logical Tensor Operations

### Join (Composition)
```rust
// grandparent(X, Z) = parent(X, Y) AND parent(Y, Z)
let grandparent = logical_join(&parent, &parent);
```

### Project (Existential Quantification)
```rust
// has_child(X) = EXISTS Y: parent(X, Y)
let has_child = logical_project(&parent, 1);
```

### Select (Query)
```rust
// Find all Y where parent(Alice, Y)
let alice_children = logical_select(&parent, "Alice");
```

### Aggregate
```rust
// Count children for each person
let child_counts = logical_aggregate(&parent, AggregateOp::Count, 1);
```

## Weighted Logic Programming

TensorLogic supports probabilistic inference with uncertainty:

```rust
use aprender::logic::{WeightedFact, InferenceEngine};

// Create weighted knowledge base
let mut wkb = WeightedKnowledgeBase::new();

// Add facts with confidence weights
wkb.add_weighted_fact("parent", vec!["Alice", "Bob"], 1.0);
wkb.add_weighted_fact("parent", vec!["Bob", "Charlie"], 0.9);
wkb.add_weighted_fact("parent", vec!["Charlie", "David"], 0.8);

// Probabilistic inference
let engine = InferenceEngine::new();
let grandparent_probs = engine.infer_weighted(&wkb, "grandparent");

// P(Alice is grandparent of Charlie) = 1.0 * 0.9 = 0.9
println!("P(grandparent(Alice, Charlie)): {}",
         grandparent_probs.get("Alice", "Charlie"));
```

## Forward and Backward Chaining

### Forward Chaining
Derive all possible conclusions from known facts:

```rust
let engine = TensorLogicEngine::new();

// Rules
let rules = vec![
    Rule::new("grandparent", vec!["parent", "parent"]),
    Rule::new("ancestor", vec!["parent"]),
    Rule::new("ancestor", vec!["parent", "ancestor"]),
];

// Forward chain to derive all facts
let derived = engine.forward_chain(&kb, &rules, max_iterations: 10);
println!("Derived {} new facts", derived.len());
```

### Backward Chaining
Query-driven inference with goal-directed search:

```rust
// Query: Is Alice an ancestor of David?
let query = Query::new("ancestor", vec!["Alice", "David"]);
let result = engine.backward_chain(&kb, &rules, &query);

match result {
    ProofResult::Proved(proof) => {
        println!("Proved! Proof tree:");
        proof.display();
    }
    ProofResult::Failed => println!("Cannot prove"),
}
```

## Differentiable Logic Layers

For neural-symbolic integration:

```rust
use aprender::logic::DifferentiableLogic;
use aprender::nn::{NeuralNetwork, Layer};

// Create a neural network with logic layer
let mut model = NeuralNetwork::new();
model.add(Layer::dense(64, 32));
model.add(Layer::logic(LogicOp::And));  // Differentiable AND
model.add(Layer::dense(32, 10));

// Train with backpropagation through logic
model.fit(&x_train, &y_train, epochs: 100);
```

## Use Cases

1. **Knowledge Graph Completion**: Infer missing links in knowledge graphs
2. **Question Answering**: Multi-hop reasoning over structured data
3. **Program Synthesis**: Generate programs from input-output examples
4. **Explainable AI**: Provide logical explanations for neural predictions

## Running the Example

```bash
cargo run -p aprender@0.20.1 --example logic_family_tree
```

## Test Coverage

TensorLogic is verified with 20 specification points (K1-K20):
- K1-K5: Core tensor operations
- K6-K10: Knowledge graph inference
- K11-K15: Weighted logic programming
- K16-K20: Differentiable logic and SIMD acceleration

All tests pass with comprehensive property-based testing.

## References

- DeepProbLog: Neural Probabilistic Logic Programming
- TensorLog: A Differentiable Deductive Database
- Logic Tensor Networks: Integrating Learning and Reasoning
