
impl Default for GNNErrorEncoder {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Program-feedback graph structure.
///
/// Per Yasunaga & Liang (2020), this graph connects symbols
/// in source code with diagnostic feedback for GNN reasoning.
#[derive(Debug, Clone)]
pub struct ProgramFeedbackGraph {
    /// Node features
    pub node_features: Vec<Vec<f32>>,
    /// Node types
    pub node_types: Vec<NodeType>,
    /// Edges (source, target)
    pub edges: Vec<(usize, usize)>,
    /// Edge types
    pub edge_types: Vec<EdgeType>,
}

impl ProgramFeedbackGraph {
    /// Create a new empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            node_features: Vec::new(),
            node_types: Vec::new(),
            edges: Vec::new(),
            edge_types: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node_type: NodeType, features: Vec<f32>) -> usize {
        let idx = self.node_features.len();
        self.node_features.push(features);
        self.node_types.push(node_type);
        idx
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, source: usize, target: usize, edge_type: EdgeType) {
        self.edges.push((source, target));
        self.edge_types.push(edge_type);
    }

    /// Get the number of nodes.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.node_features.len()
    }

    /// Get the number of edges.
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

impl Default for ProgramFeedbackGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Node type in program-feedback graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    /// AST node (variable, type, expression, etc.)
    Ast,
    /// Compiler diagnostic
    Diagnostic,
    /// Expected type in type error
    ExpectedType,
    /// Found type in type error
    FoundType,
    /// Compiler suggestion
    Suggestion,
}

/// Edge type in program-feedback graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeType {
    /// AST parent-child relationship
    AstChild,
    /// Data flow edge
    DataFlow,
    /// Control flow edge
    ControlFlow,
    /// Diagnostic refers to code location
    DiagnosticRefers,
    /// Type expectation
    Expects,
    /// Type found
    Found,
}

#[cfg(test)]
mod tests;
