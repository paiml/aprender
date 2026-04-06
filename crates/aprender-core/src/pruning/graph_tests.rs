
#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================================
    // FALSIFICATION: Graph construction
    // ==========================================================================
    #[test]
    fn test_graph_new() {
        let graph = DependencyGraph::new();
        assert_eq!(
            graph.num_nodes(),
            0,
            "GRA-01 FALSIFIED: New graph should be empty"
        );
        assert_eq!(graph.num_edges(), 0);
    }

    #[test]
    fn test_graph_add_node() {
        let mut graph = DependencyGraph::new();
        let node = GraphNode::new("layer0", "Linear1", NodeType::Linear);
        graph.add_node(node);

        assert_eq!(graph.num_nodes(), 1, "GRA-02 FALSIFIED: Should have 1 node");
        assert!(graph.get_node("layer0").is_some());
    }

    #[test]
    fn test_graph_add_edge() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear));

        let edge = GraphEdge::new("a", "b", DependencyType::Sequential);
        graph.add_edge(edge).unwrap();

        assert_eq!(graph.num_edges(), 1, "GRA-03 FALSIFIED: Should have 1 edge");
    }

    #[test]
    fn test_graph_add_edge_invalid_source() {
        let mut graph = DependencyGraph::new();
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear));

        let edge = GraphEdge::new("nonexistent", "b", DependencyType::Sequential);
        let result = graph.add_edge(edge);

        assert!(
            result.is_err(),
            "GRA-04 FALSIFIED: Should error on invalid source"
        );
    }

    #[test]
    fn test_graph_add_edge_invalid_target() {
        let mut graph = DependencyGraph::new();
        graph.add_node(GraphNode::new("a", "A", NodeType::Linear));

        let edge = GraphEdge::new("a", "nonexistent", DependencyType::Sequential);
        let result = graph.add_edge(edge);

        assert!(
            result.is_err(),
            "GRA-05 FALSIFIED: Should error on invalid target"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Linear chain helper
    // ==========================================================================
    #[test]
    fn test_linear_chain() {
        let graph = DependencyGraph::linear_chain(
            &[(768, 512), (512, 256), (256, 128)],
            &["fc1", "fc2", "fc3"],
        );

        assert_eq!(
            graph.num_nodes(),
            3,
            "GRA-06 FALSIFIED: Should have 3 nodes"
        );
        assert_eq!(
            graph.num_edges(),
            2,
            "GRA-06 FALSIFIED: Should have 2 edges"
        );

        let node = graph.get_node("layer_0").unwrap();
        assert_eq!(node.input_dim, 768);
        assert_eq!(node.output_dim, 512);
    }

    // ==========================================================================
    // FALSIFICATION: Dependency traversal
    // ==========================================================================
    #[test]
    fn test_downstream_dependents() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear));
        graph.add_node(GraphNode::new("c", "C", NodeType::Linear));

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();
        graph
            .add_edge(GraphEdge::new("b", "c", DependencyType::Sequential))
            .unwrap();

        let deps = graph.downstream_dependents("a");

        assert!(
            deps.contains("b"),
            "GRA-07 FALSIFIED: B should be downstream of A"
        );
        assert!(
            deps.contains("c"),
            "GRA-07 FALSIFIED: C should be downstream of A"
        );
        assert!(
            !deps.contains("a"),
            "GRA-07 FALSIFIED: A should not be its own dependent"
        );
    }

    #[test]
    fn test_upstream_dependents() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear));
        graph.add_node(GraphNode::new("c", "C", NodeType::Linear));

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();
        graph
            .add_edge(GraphEdge::new("b", "c", DependencyType::Sequential))
            .unwrap();

        let deps = graph.upstream_dependents("c");

        assert!(
            deps.contains("a"),
            "GRA-08 FALSIFIED: A should be upstream of C"
        );
        assert!(
            deps.contains("b"),
            "GRA-08 FALSIFIED: B should be upstream of C"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Prunable nodes
    // ==========================================================================
    #[test]
    fn test_prunable_nodes() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear).with_prunable(true));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear).with_prunable(false));
        graph.add_node(GraphNode::new("c", "C", NodeType::Linear).with_prunable(true));

        let prunable = graph.prunable_nodes();

        assert_eq!(
            prunable.len(),
            2,
            "GRA-09 FALSIFIED: Should have 2 prunable nodes"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Graph validation
    // ==========================================================================
    #[test]
    fn test_validate_dimension_mismatch() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear).with_dims(10, 20));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear).with_dims(30, 40)); // Mismatch!

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();

        let result = graph.validate();
        assert!(
            result.is_err(),
            "GRA-10 FALSIFIED: Should detect dimension mismatch"
        );
    }

    #[test]
    fn test_validate_dimension_match() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear).with_dims(10, 20));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear).with_dims(20, 40));

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();

        let result = graph.validate();
        assert!(
            result.is_ok(),
            "GRA-11 FALSIFIED: Matching dimensions should pass"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Pruning plan
    // ==========================================================================
    #[test]
    fn test_pruning_plan_new() {
        let plan = PruningPlan::new();
        assert_eq!(plan.total_channels_removed(), 0);
        assert_eq!(plan.total_layers_removed(), 0);
        assert!(!plan.is_validated());
    }

    #[test]
    fn test_pruning_plan_add_channels() {
        let mut plan = PruningPlan::new();
        plan.remove_channels("layer0", vec![0, 5, 10]);
        plan.remove_channels("layer1", vec![1, 2]);

        assert_eq!(plan.total_channels_removed(), 5);
        assert_eq!(plan.channels_to_remove("layer0"), Some(&vec![0, 5, 10]));
    }

    #[test]
    fn test_pruning_plan_remove_layer() {
        let mut plan = PruningPlan::new();
        plan.remove_layer("layer5");
        plan.remove_layer("layer10");

        assert_eq!(plan.total_layers_removed(), 2);
        assert!(plan.is_layer_removed("layer5"));
        assert!(!plan.is_layer_removed("layer0"));
    }

    #[test]
    fn test_pruning_plan_validate() {
        let graph = DependencyGraph::linear_chain(&[(100, 50), (50, 25)], &["fc1", "fc2"]);

        let mut plan = PruningPlan::new();
        plan.remove_channels("layer_0", vec![0, 10, 20]);

        let result = plan.validate(&graph);
        assert!(result.is_ok(), "GRA-12 FALSIFIED: Valid plan should pass");
        assert!(plan.is_validated());
    }

    #[test]
    fn test_pruning_plan_validate_invalid_layer() {
        let graph = DependencyGraph::linear_chain(&[(100, 50)], &["fc1"]);

        let mut plan = PruningPlan::new();
        plan.remove_channels("nonexistent", vec![0]);

        let result = plan.validate(&graph);
        assert!(
            result.is_err(),
            "GRA-13 FALSIFIED: Should error on invalid layer"
        );
    }

    #[test]
    fn test_pruning_plan_validate_invalid_channel() {
        let graph = DependencyGraph::linear_chain(&[(100, 50)], &["fc1"]);

        let mut plan = PruningPlan::new();
        plan.remove_channels("layer_0", vec![100]); // Out of bounds (output_dim = 50)

        let result = plan.validate(&graph);
        assert!(
            result.is_err(),
            "GRA-14 FALSIFIED: Should error on invalid channel index"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Channel propagation
    // ==========================================================================
    #[test]
    fn test_propagate_channel_pruning() {
        let mut graph = DependencyGraph::new();

        graph.add_node(GraphNode::new("a", "A", NodeType::Linear).with_dims(100, 50));
        graph.add_node(GraphNode::new("b", "B", NodeType::Linear).with_dims(50, 25));

        graph
            .add_edge(GraphEdge::new("a", "b", DependencyType::Sequential))
            .unwrap();

        let propagation = propagate_channel_pruning(&graph, "a", &[5, 10, 15]);

        assert!(
            propagation.contains_key("b"),
            "GRA-15 FALSIFIED: B should be affected by pruning A"
        );
        assert_eq!(propagation.get("b"), Some(&vec![5, 10, 15]));
    }

    // ==========================================================================
    // FALSIFICATION: GraphNode builder
    // ==========================================================================
    #[test]
    fn test_graph_node_builder() {
        let node = GraphNode::new("test", "Test Layer", NodeType::Linear)
            .with_dims(100, 200)
            .with_prunable(false);

        assert_eq!(node.id, "test");
        assert_eq!(node.name, "Test Layer");
        assert_eq!(node.input_dim, 100);
        assert_eq!(node.output_dim, 200);
        assert!(!node.prunable);
    }

    // ==========================================================================
    // FALSIFICATION: Edge types
    // ==========================================================================
    #[test]
    fn test_dependency_types() {
        let sequential = DependencyType::Sequential;
        let skip = DependencyType::Skip;

        assert_ne!(
            sequential, skip,
            "GRA-16 FALSIFIED: Types should be distinct"
        );
    }

    #[test]
    fn test_edge_with_dim_index() {
        let edge = GraphEdge::new("a", "b", DependencyType::Concat).with_dim_index(2);
        assert_eq!(edge.dim_index, 2);
    }

    // ==========================================================================
    // FALSIFICATION: Clone and Debug
    // ==========================================================================
    #[test]
    fn test_graph_clone() {
        let mut orig = DependencyGraph::new();
        orig.add_node(GraphNode::new("a", "A", NodeType::Linear));

        let cloned = orig.clone();
        assert_eq!(orig.num_nodes(), cloned.num_nodes());
    }

    #[test]
    fn test_graph_debug() {
        let graph = DependencyGraph::new();
        let debug = format!("{:?}", graph);
        assert!(debug.contains("DependencyGraph"));
    }

    #[test]
    fn test_node_debug() {
        let node = GraphNode::new("test", "Test", NodeType::Linear);
        let debug = format!("{:?}", node);
        assert!(debug.contains("GraphNode"));
    }

    #[test]
    fn test_edge_debug() {
        let edge = GraphEdge::new("a", "b", DependencyType::Sequential);
        let debug = format!("{:?}", edge);
        assert!(debug.contains("GraphEdge"));
    }

    #[test]
    fn test_plan_debug() {
        let plan = PruningPlan::new();
        let debug = format!("{:?}", plan);
        assert!(debug.contains("PruningPlan"));
    }

    // ==========================================================================
    // FALSIFICATION: Default implementations
    // ==========================================================================
    #[test]
    fn test_graph_default() {
        let graph = DependencyGraph::default();
        assert_eq!(graph.num_nodes(), 0);
    }

    #[test]
    fn test_plan_default() {
        let plan = PruningPlan::default();
        assert_eq!(plan.total_channels_removed(), 0);
    }
}
