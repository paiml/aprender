//! Stack Graph TUI Visualization
//!
//! Visualizes the PAIML Sovereign AI Stack dependency graph using the Graph TUI module.
//!
//! ## Features
//!
//! - **Dependency Visualization**: Shows crate relationships
//! - **PageRank**: Identifies most important crates
//! - **Community Detection**: Groups related crates
//! - **Multiple Layouts**: Hierarchical, Radial, Concentric
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example stack_graph_tui --features native
//! ```
//!
//! ## Toyota Way Principles
//!
//! - **Mieruka**: Visual management of stack dependencies
//! - **Genchi Genbutsu**: Real dependency data from workspace

#[cfg(feature = "native")]
use batuta::tui::{
    Edge, Graph, GraphAnalytics, GraphAnalyticsExt, GraphRenderer, LayoutAlgorithm, LayoutConfig,
    LayoutEngine, Node, NodeStatus, RenderMode,
};

/// PAIML Stack layer assignment for nodes
#[cfg(feature = "native")]
fn get_layer(crate_name: &str) -> (&'static str, NodeStatus) {
    match crate_name {
        // Core/Primitives layer - foundation
        "trueno" => ("Primitives", NodeStatus::Healthy),
        "trueno-viz" | "trueno-db" | "trueno-graph" | "trueno-rag" => {
            ("Primitives", NodeStatus::Healthy)
        }

        // ML/Algorithms layer
        "aprender" | "aprender-shell" | "aprender-tsp" => ("ML Algorithms", NodeStatus::Healthy),

        // Transpilers layer
        "realizar" | "depyler" | "renacer" => ("Transpilers", NodeStatus::Info),

        // Data layer
        "alimentar" | "pacha" => ("Data", NodeStatus::Healthy),

        // Training layer
        "entrenar" => ("Training", NodeStatus::Warning),

        // Orchestration layer
        "batuta" | "repartir" => ("Orchestration", NodeStatus::Healthy),

        // Quality/Testing layer
        "certeza" | "pmat" => ("Quality", NodeStatus::Healthy),

        // Presentation layer
        "presentar" => ("Presentation", NodeStatus::Healthy),
        "profesor" => ("Presentation", NodeStatus::Info), // WASM LMS
        "jugar" => ("Presentation", NodeStatus::Info),    // WASM Game Engine

        // Utilities
        "ruchy" | "decy" => ("Utilities", NodeStatus::Neutral),

        // Books/Docs
        _ if crate_name.contains("cookbook") || crate_name.contains("book") => {
            ("Documentation", NodeStatus::Info)
        }

        _ => ("Other", NodeStatus::Neutral),
    }
}

#[cfg(feature = "native")]
fn main() {
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("     PAIML Sovereign AI Stack - Graph TUI Visualization");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Build the stack dependency graph
    let mut graph = build_stack_graph();

    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 1: Stack Statistics                                              │");
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    println!("  Stack Components: {}", graph.node_count());
    println!("  Dependencies:     {}", graph.edge_count());
    println!();

    // Apply PageRank to find most important crates
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 2: PageRank Analysis (Most Important Crates)                     │");
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    graph.compute_pagerank(0.85, 30);
    let ranks = graph.pagerank_scores();

    let mut ranked: Vec<_> = ranks.iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Top 10 by PageRank (dependency importance):\n");
    println!("  {:<20} {:<12} {:<15}", "Crate", "PageRank", "Layer");
    println!("  {}", "─".repeat(50));

    for (name, score) in ranked.iter().take(10) {
        let (layer, _) = get_layer(name);
        println!("  {:<20} {:<12.4} {:<15}", name, score, layer);
    }

    // Apply community detection
    println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 3: Community Detection (Related Crate Groups)                    │");
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    let num_communities = graph.detect_communities();
    println!("  Detected {} communities\n", num_communities);

    // Centrality analysis
    println!("┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 4: Centrality Analysis                                           │");
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    let degree = GraphAnalytics::degree_centrality(&graph);
    let betweenness = GraphAnalytics::betweenness_centrality(&graph);

    println!("  Top 5 by Degree Centrality (most connections):\n");
    let mut by_degree: Vec<_> = degree.iter().collect();
    by_degree.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (name, score) in by_degree.iter().take(5) {
        println!("    {:<20} {:.4}", name, score);
    }

    println!("\n  Top 5 by Betweenness Centrality (bridges):\n");
    let mut by_between: Vec<_> = betweenness.iter().collect();
    by_between.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (name, score) in by_between.iter().take(5) {
        println!("    {:<20} {:.4}", name, score);
    }

    // Path analysis
    println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 5: Path Analysis                                                 │");
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    if let Some(path) = GraphAnalytics::shortest_path(&graph, "batuta", "trueno") {
        println!("  Shortest path from batuta to trueno:");
        println!("    {}", path.join(" → "));
    }

    if let Some(path) = GraphAnalytics::shortest_path(&graph, "presentar", "trueno") {
        println!("\n  Shortest path from presentar to trueno:");
        println!("    {}", path.join(" → "));
    }

    // Visualizations
    println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Phase 6: Graph Visualizations                                          │");
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    // Filter to core crates for cleaner visualization
    let core_graph = graph.filter_top_n(15);

    // Hierarchical layout - best for dependency graphs
    println!("  Hierarchical Layout (Dependency Flow):\n");
    render_graph(&core_graph, LayoutAlgorithm::Hierarchical, 70, 16);

    // Radial layout - shows central crates
    println!("\n  Radial Layout (Core at Center):\n");
    render_graph(&core_graph, LayoutAlgorithm::Radial, 70, 16);

    // Concentric layout - by importance
    println!("\n  Concentric Layout (Importance Rings):\n");
    render_graph(&core_graph, LayoutAlgorithm::Concentric, 70, 16);

    // Full stack circular
    println!("\n  Circular Layout (Full Stack):\n");
    render_graph(&graph, LayoutAlgorithm::Circular, 70, 18);

    // Legend
    println!("\n┌─────────────────────────────────────────────────────────────────────────┐");
    println!("│ Legend                                                                  │");
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    println!(
        "  {} Healthy (Published)      {} Warning (Needs Update)",
        NodeStatus::Healthy.shape().unicode(),
        NodeStatus::Warning.shape().unicode()
    );
    println!(
        "  {} Info (Transpiler)        {} Neutral (Utility)",
        NodeStatus::Info.shape().unicode(),
        NodeStatus::Neutral.shape().unicode()
    );
    println!("  ····  Dependency edge");

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("     Stack Visualization Complete!");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
}

#[cfg(feature = "native")]
fn build_stack_graph() -> Graph<&'static str, &'static str> {
    let mut graph: Graph<&str, &str> = Graph::new();

    // Define the PAIML stack crates with their layers
    let crates = [
        "trueno",
        "trueno-viz",
        "trueno-graph",
        "aprender",
        "realizar",
        "renacer",
        "alimentar",
        "entrenar",
        "batuta",
        "certeza",
        "presentar",
        "pacha",
        "repartir",
        "depyler",
        "pmat",
        "profesor", // WASM-native LMS (quizzes, labs, simulations)
        "jugar",    // WASM-native game engine
    ];

    // Add nodes with status based on layer
    for crate_name in &crates {
        let (layer, status) = get_layer(crate_name);
        let importance = match layer {
            "Primitives" => 1.0,
            "ML Algorithms" => 0.9,
            "Transpilers" => 0.8,
            "Data" => 0.7,
            "Training" => 0.75,
            "Orchestration" => 0.85,
            "Quality" => 0.6,
            "Presentation" => 0.65,
            _ => 0.5,
        };

        graph.add_node(
            Node::new(*crate_name, layer)
                .with_status(status)
                .with_label(*crate_name)
                .with_importance(importance),
        );
    }

    // Define dependencies (from -> to means "from depends on to")
    let dependencies = [
        // trueno is the foundation - no dependencies on other PAIML crates
        // trueno-viz depends on trueno
        ("trueno-viz", "trueno", "viz"),
        ("trueno-graph", "trueno", "graph-ext"),
        // aprender depends on trueno
        ("aprender", "trueno", "tensor-ops"),
        // realizar depends on aprender
        ("realizar", "aprender", "ml-models"),
        ("realizar", "trueno", "core"),
        // renacer (Python AST)
        ("renacer", "trueno", "core"),
        // alimentar (data) depends on trueno
        ("alimentar", "trueno", "core"),
        // entrenar depends on aprender
        ("entrenar", "aprender", "training"),
        ("entrenar", "trueno", "core"),
        // batuta orchestrates
        ("batuta", "aprender", "ml"),
        ("batuta", "trueno", "core"),
        ("batuta", "realizaar", "transpile"),
        ("batuta", "certeza", "quality"),
        // certeza for quality
        ("certeza", "trueno", "core"),
        // presentar for visualization
        ("presentar", "trueno-viz", "viz"),
        ("presentar", "trueno", "core"),
        // pacha data
        ("pacha", "alimentar", "data"),
        ("pacha", "trueno", "core"),
        // repartir distribution
        ("repartir", "trueno", "core"),
        ("repartir", "batuta", "orch"),
        // depyler transpiler
        ("depyler", "renacer", "ast"),
        ("depyler", "trueno", "core"),
        // pmat depends on certeza
        ("pmat", "certeza", "quality"),
        // profesor (WASM LMS) dependencies
        ("profesor", "trueno", "simd"),
        ("profesor", "aprender", "adaptive"),
        ("profesor", "alimentar", "content"),
        ("profesor", "presentar", "ui"),
        // jugar (WASM Game Engine) dependencies
        ("jugar", "trueno", "physics"),
        ("jugar", "trueno-viz", "render"),
        ("jugar", "aprender", "ai-agents"),
        ("jugar", "presentar", "platform"),
    ];

    // Add edges (only if both nodes exist)
    for (from, to, label) in &dependencies {
        if graph.get_node(from).is_some() && graph.get_node(to).is_some() {
            graph.add_edge(Edge::new(*from, *to, *label).with_label(*label));
        }
    }

    graph
}

#[cfg(feature = "native")]
fn render_graph<N, E>(graph: &Graph<N, E>, algorithm: LayoutAlgorithm, width: usize, height: usize)
where
    N: Clone,
    E: Clone,
{
    let mut g = graph.clone();
    let config = LayoutConfig {
        algorithm,
        width: width as f32,
        height: height as f32,
        iterations: 100,
        ..Default::default()
    };
    LayoutEngine::compute(&mut g, &config);

    let renderer = GraphRenderer::new().with_mode(RenderMode::Unicode);
    let output = renderer.render(&g, width, height);

    println!("{}", output.to_string_colored());
}

#[cfg(not(feature = "native"))]
fn main() {
    println!("This example requires the 'native' feature.");
    println!("Run with: cargo run --example stack_graph_tui --features native");
}
