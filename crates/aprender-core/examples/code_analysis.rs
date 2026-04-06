#![allow(clippy::disallowed_methods)]
//! Code Analysis with Code2Vec Embeddings and MPNN
//!
//! This example demonstrates aprender's code analysis capabilities:
//! - AST representation for code structures
//! - Code2Vec path extraction between terminal nodes
//! - Embedding generation for code similarity
//! - Message Passing Neural Networks for code graphs
//!
//! # Use Cases
//! - Code similarity detection
//! - Function name prediction
//! - Type inference
//! - Bug detection
//!
//! # References
//! - Alon et al. (2019), "code2vec: Learning distributed representations of code"

use aprender::code::{
    pooling, AstNode, AstNodeType, Code2VecEncoder, CodeEdgeType, CodeGraph, CodeGraphEdge,
    CodeGraphNode, CodeMPNN, PathExtractor,
};

fn main() {
    println!("=== Code Analysis with Code2Vec and MPNN ===\n");

    // Part 1: AST Representation
    println!("1. Building AST for a simple function");
    println!("   Function: fn add(x: i32, y: i32) -> i32 {{ x + y }}");
    println!();

    let ast = build_add_function_ast();
    println!("   AST Structure:");
    print_ast(&ast, 3);
    println!();

    // Part 2: Path Extraction (Code2Vec style)
    println!("2. Extracting Code2Vec Paths");
    let extractor = PathExtractor::new(8);
    let paths = extractor.extract(&ast);
    println!("   Found {} paths between terminal nodes:", paths.len());
    for (i, path) in paths.iter().take(5).enumerate() {
        println!(
            "   Path {}: {} -> {:?} -> {}",
            i + 1,
            path.source().value(),
            path.path_nodes()
                .iter()
                .map(|n| format!("{n}"))
                .collect::<Vec<_>>()
                .join(" → "),
            path.target().value()
        );
    }
    println!();

    // Part 3: Code Embeddings
    println!("3. Generating Code Embeddings");
    let encoder = Code2VecEncoder::new(64).with_seed(42);

    // Encode individual paths
    let path_embedding = encoder.encode_path(&paths[0]);
    println!(
        "   Single path embedding dim: {} (first 5: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}])",
        path_embedding.len(),
        path_embedding[0],
        path_embedding[1],
        path_embedding[2],
        path_embedding[3],
        path_embedding[4]
    );

    // Aggregate all paths into a single function embedding
    let function_embedding = encoder.aggregate_paths(&paths);
    println!("   Function embedding dim: {}", function_embedding.dim());
    if let Some(weights) = function_embedding.attention_weights() {
        println!(
            "   Attention weights (first 3): [{:.3}, {:.3}, {:.3}]",
            weights[0], weights[1], weights[2]
        );
    }
    println!();

    // Part 4: Code Similarity
    println!("4. Computing Code Similarity");

    // Build a similar function: fn sum(a, b) -> result
    let similar_ast = build_similar_function_ast();
    let similar_paths = extractor.extract(&similar_ast);
    let similar_embedding = encoder.aggregate_paths(&similar_paths);

    // Build a different function: fn multiply(x, y) -> product
    let different_ast = build_multiply_function_ast();
    let different_paths = extractor.extract(&different_ast);
    let different_embedding = encoder.aggregate_paths(&different_paths);

    let sim_similar = function_embedding.cosine_similarity(&similar_embedding);
    let sim_different = function_embedding.cosine_similarity(&different_embedding);

    println!(
        "   add() vs sum():      {:.4} (similar structure)",
        sim_similar
    );
    println!(
        "   add() vs multiply(): {:.4} (different operation)",
        sim_different
    );
    println!();

    // Part 5: Code Graph with MPNN
    println!("5. Building Code Graph for MPNN");
    let graph = build_code_graph();
    println!(
        "   Graph: {} nodes, {} edges",
        graph.num_nodes(),
        graph.num_edges()
    );
    println!();

    // Part 6: MPNN Forward Pass
    println!("6. Running MPNN for Type Propagation");
    let mpnn = CodeMPNN::new(&[3, 8, 4]); // 3 input features -> 8 hidden -> 4 output
    let node_embeddings = mpnn.forward(&graph);

    println!("   Node embeddings after message passing:");
    for (i, emb) in node_embeddings.iter().enumerate() {
        let node_type = graph.node(i).map(|n| n.node_type.as_str()).unwrap_or("?");
        println!(
            "   Node {} ({}): [{:.3}, {:.3}, {:.3}, {:.3}]",
            i, node_type, emb[0], emb[1], emb[2], emb[3]
        );
    }
    println!();

    // Part 7: Graph-level Embedding via Pooling
    println!("7. Graph-level Embedding via Pooling");
    let mean_pooled = pooling::mean_pool(&node_embeddings);
    let max_pooled = pooling::max_pool(&node_embeddings);
    let sum_pooled = pooling::sum_pool(&node_embeddings);

    println!(
        "   Mean pooling: [{:.3}, {:.3}, {:.3}, {:.3}]",
        mean_pooled[0], mean_pooled[1], mean_pooled[2], mean_pooled[3]
    );
    println!(
        "   Max pooling:  [{:.3}, {:.3}, {:.3}, {:.3}]",
        max_pooled[0], max_pooled[1], max_pooled[2], max_pooled[3]
    );
    println!(
        "   Sum pooling:  [{:.3}, {:.3}, {:.3}, {:.3}]",
        sum_pooled[0], sum_pooled[1], sum_pooled[2], sum_pooled[3]
    );
    println!();

    println!("=== Code Analysis Complete ===");
}

/// Build AST for: fn add(x: i32, y: i32) -> i32 { x + y }
fn build_add_function_ast() -> AstNode {
    let mut func = AstNode::new(AstNodeType::Function, "add");

    // Parameters
    let mut param_x = AstNode::new(AstNodeType::Parameter, "x");
    param_x.add_child(AstNode::new(AstNodeType::TypeAnnotation, "i32"));
    func.add_child(param_x);

    let mut param_y = AstNode::new(AstNodeType::Parameter, "y");
    param_y.add_child(AstNode::new(AstNodeType::TypeAnnotation, "i32"));
    func.add_child(param_y);

    // Return type
    func.add_child(AstNode::new(AstNodeType::TypeAnnotation, "i32"));

    // Body: x + y
    let mut body = AstNode::new(AstNodeType::Block, "body");
    let mut binary_op = AstNode::new(AstNodeType::BinaryOp, "+");
    binary_op.add_child(AstNode::new(AstNodeType::Variable, "x"));
    binary_op.add_child(AstNode::new(AstNodeType::Variable, "y"));

    let mut ret = AstNode::new(AstNodeType::Return, "return");
    ret.add_child(binary_op);
    body.add_child(ret);
    func.add_child(body);

    func
}

/// Build AST for: fn sum(a, b) -> result (similar to add)
fn build_similar_function_ast() -> AstNode {
    let mut func = AstNode::new(AstNodeType::Function, "sum");

    func.add_child(AstNode::new(AstNodeType::Parameter, "a"));
    func.add_child(AstNode::new(AstNodeType::Parameter, "b"));

    let mut body = AstNode::new(AstNodeType::Block, "body");
    let mut binary_op = AstNode::new(AstNodeType::BinaryOp, "+");
    binary_op.add_child(AstNode::new(AstNodeType::Variable, "a"));
    binary_op.add_child(AstNode::new(AstNodeType::Variable, "b"));

    let mut ret = AstNode::new(AstNodeType::Return, "result");
    ret.add_child(binary_op);
    body.add_child(ret);
    func.add_child(body);

    func
}

/// Build AST for: fn multiply(x, y) -> product (different operation)
fn build_multiply_function_ast() -> AstNode {
    let mut func = AstNode::new(AstNodeType::Function, "multiply");

    func.add_child(AstNode::new(AstNodeType::Parameter, "x"));
    func.add_child(AstNode::new(AstNodeType::Parameter, "y"));

    let mut body = AstNode::new(AstNodeType::Block, "body");
    let mut binary_op = AstNode::new(AstNodeType::BinaryOp, "*");
    binary_op.add_child(AstNode::new(AstNodeType::Variable, "x"));
    binary_op.add_child(AstNode::new(AstNodeType::Variable, "y"));

    let mut ret = AstNode::new(AstNodeType::Return, "product");
    ret.add_child(binary_op);
    body.add_child(ret);
    func.add_child(body);

    func
}

/// Build a code graph for MPNN demonstration
fn build_code_graph() -> CodeGraph {
    let mut graph = CodeGraph::new();

    // Nodes: x (variable), y (variable), add (function), result (variable)
    graph.add_node(CodeGraphNode::new(0, vec![1.0, 0.0, 0.0], "variable")); // x
    graph.add_node(CodeGraphNode::new(1, vec![0.0, 1.0, 0.0], "variable")); // y
    graph.add_node(CodeGraphNode::new(2, vec![0.0, 0.0, 1.0], "function")); // add
    graph.add_node(CodeGraphNode::new(3, vec![0.5, 0.5, 0.0], "variable")); // result

    // Edges: data flow from x,y to add, then add to result
    graph.add_edge(CodeGraphEdge::new(0, 2, CodeEdgeType::DataFlow)); // x -> add
    graph.add_edge(CodeGraphEdge::new(1, 2, CodeEdgeType::DataFlow)); // y -> add
    graph.add_edge(CodeGraphEdge::new(2, 3, CodeEdgeType::DataFlow)); // add -> result

    graph
}

/// Pretty print AST with indentation
fn print_ast(node: &AstNode, indent: usize) {
    let prefix = " ".repeat(indent);
    println!("{}{}: {}", prefix, node.node_type(), node.value());
    for child in node.children() {
        print_ast(child, indent + 2);
    }
}
