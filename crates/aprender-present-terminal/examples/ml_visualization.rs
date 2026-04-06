//! ML Visualization Example
//!
//! Demonstrates the SIMD/WGPU-first ML visualization widgets:
//! - `ViolinPlot`: Distribution visualization with KDE
//! - `RocPrCurve`: ROC and Precision-Recall curves
//! - `LossCurve`: Training loss with EMA smoothing
//! - `ForceGraph`: Network/graph visualization
//! - Treemap: Hierarchical data visualization
//!
//! Run with: cargo run --example `ml_visualization`

use presentar_core::{Color, Rect, Widget};
use presentar_terminal::{
    CellBuffer, CurveData, CurveMode, DirectTerminalCanvas, EmaConfig, ForceGraph, ForceParams,
    GraphEdge, GraphNode, LossCurve, RocPrCurve, Treemap, TreemapNode, ViolinData,
    ViolinOrientation, ViolinPlot,
};

fn main() {
    println!("ML Visualization Widgets Demo\n");
    println!("==============================\n");

    // Demo ViolinPlot
    demo_violin_plot();

    // Demo RocPrCurve
    demo_roc_pr_curve();

    // Demo LossCurve
    demo_loss_curve();

    // Demo ForceGraph
    demo_force_graph();

    // Demo Treemap
    demo_treemap();
}

fn demo_violin_plot() {
    println!("1. ViolinPlot - Distribution Visualization");
    println!("   ----------------------------------------");

    // Create sample distributions
    let normal_data: Vec<f64> = (0..100)
        .map(|i| {
            let x = (f64::from(i) - 50.0) / 15.0;
            (10.0 * (-x * x / 2.0).exp()).mul_add(1.0 + (f64::from(i) * 0.1).sin(), 50.0)
        })
        .collect();

    let bimodal_data: Vec<f64> = (0..100)
        .map(|i| {
            if i < 50 {
                5.0f64.mul_add((f64::from(i) * 0.2).sin(), 30.0)
            } else {
                5.0f64.mul_add((f64::from(i) * 0.2).cos(), 70.0)
            }
        })
        .collect();

    let mut plot = ViolinPlot::new(vec![
        ViolinData::new("Normal", normal_data).with_color(Color::new(0.3, 0.7, 1.0, 1.0)),
        ViolinData::new("Bimodal", bimodal_data).with_color(Color::new(1.0, 0.5, 0.3, 1.0)),
    ])
    .with_orientation(ViolinOrientation::Vertical)
    .with_median(true)
    .with_kde_points(50);

    // Render
    let mut buffer = CellBuffer::new(60, 20);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);
    plot.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
    plot.paint(&mut canvas);

    println!("   Created violin plot with 2 distributions");
    println!("   - Normal: centered at 50");
    println!("   - Bimodal: peaks at 30 and 70\n");
}

fn demo_roc_pr_curve() {
    println!("2. RocPrCurve - Model Evaluation");
    println!("   ------------------------------");

    // Create sample predictions for a good model
    let y_true: Vec<f64> = (0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();
    let y_score_good: Vec<f64> = (0..100)
        .map(|i| {
            if i < 50 {
                0.2f64.mul_add(f64::from(i) / 50.0, 0.2)
            } else {
                0.3f64.mul_add(f64::from(i - 50) / 50.0, 0.6)
            }
        })
        .collect();

    // Create sample predictions for a random model
    let y_score_random: Vec<f64> = (0..100).map(|i| (f64::from(i) * 0.37) % 1.0).collect();

    let mut curve = RocPrCurve::new(vec![
        CurveData::new("Good Model", y_true.clone(), y_score_good)
            .with_color(Color::new(0.3, 0.8, 0.3, 1.0)),
        CurveData::new("Random", y_true, y_score_random).with_color(Color::new(0.8, 0.3, 0.3, 1.0)),
    ])
    .with_mode(CurveMode::Both)
    .with_auc(true)
    .with_baseline(true);

    // Render
    let mut buffer = CellBuffer::new(80, 20);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);
    curve.layout(Rect::new(0.0, 0.0, 80.0, 20.0));
    curve.paint(&mut canvas);

    println!("   Created ROC/PR curves for 2 models");
    println!("   - Good Model: AUC should be high (~0.8+)");
    println!("   - Random: AUC should be ~0.5\n");
}

fn demo_loss_curve() {
    println!("3. LossCurve - Training Visualization");
    println!("   -----------------------------------");

    // Create noisy training loss that decreases over time
    let train_loss: Vec<f64> = (0..100)
        .map(|i| {
            let base = 2.0 * (f64::from(-i) / 30.0).exp();
            let noise = 0.1 * ((f64::from(i) * 0.5).sin() + (f64::from(i) * 1.3).cos());
            (base + noise).max(0.01)
        })
        .collect();

    // Validation loss with more noise and slight overfitting at end
    let val_loss: Vec<f64> = (0..100)
        .map(|i| {
            let base = 2.2 * (f64::from(-i) / 35.0).exp();
            let noise = 0.15 * ((f64::from(i) * 0.7).sin() + (f64::from(i) * 1.1).cos());
            let overfit = if i > 70 {
                0.05 * f64::from(i - 70) / 30.0
            } else {
                0.0
            };
            (base + noise + overfit).max(0.02)
        })
        .collect();

    let mut curve = LossCurve::new()
        .with_ema(EmaConfig { alpha: 0.1 })
        .with_log_scale(true)
        .with_raw_visible(true)
        .add_series("Train", train_loss, Color::new(0.3, 0.7, 1.0, 1.0))
        .add_series("Val", val_loss, Color::new(1.0, 0.5, 0.3, 1.0));

    // Render
    let mut buffer = CellBuffer::new(60, 20);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);
    curve.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
    curve.paint(&mut canvas);

    println!("   Created loss curve with EMA smoothing");
    println!("   - Training loss: exponential decay with noise");
    println!("   - Validation loss: slight overfitting after epoch 70\n");
}

fn demo_force_graph() {
    println!("4. ForceGraph - Network Visualization");
    println!("   -----------------------------------");

    // Create a small network graph
    let nodes = vec![
        GraphNode::new("A")
            .with_label("Hub")
            .with_position(0.5, 0.5)
            .with_size(2.0),
        GraphNode::new("B")
            .with_label("Node1")
            .with_position(0.3, 0.3),
        GraphNode::new("C")
            .with_label("Node2")
            .with_position(0.7, 0.3),
        GraphNode::new("D")
            .with_label("Node3")
            .with_position(0.3, 0.7),
        GraphNode::new("E")
            .with_label("Node4")
            .with_position(0.7, 0.7),
    ];

    let edges = vec![
        GraphEdge::new(0, 1), // A-B
        GraphEdge::new(0, 2), // A-C
        GraphEdge::new(0, 3), // A-D
        GraphEdge::new(0, 4), // A-E
        GraphEdge::new(1, 2), // B-C
        GraphEdge::new(3, 4), // D-E
    ];

    let mut graph = ForceGraph::new(nodes, edges)
        .with_params(ForceParams {
            repulsion: 300.0,
            spring_strength: 0.1,
            spring_length: 0.3,
            damping: 0.9,
            gravity: 0.1,
        })
        .with_iterations(20)
        .with_labels(true);

    // Render
    let mut buffer = CellBuffer::new(60, 30);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);
    graph.layout(Rect::new(0.0, 0.0, 60.0, 30.0));
    graph.paint(&mut canvas);

    println!("   Created force-directed graph with 5 nodes");
    println!("   - Hub node in center connected to all");
    println!("   - Additional edges: B-C, D-E\n");
}

fn demo_treemap() {
    println!("5. Treemap - Hierarchical Visualization");
    println!("   -------------------------------------");

    // Create a file size treemap
    let root = TreemapNode::branch(
        "Project",
        vec![
            TreemapNode::branch(
                "src",
                vec![
                    TreemapNode::leaf_colored("main.rs", 100.0, Color::new(0.3, 0.7, 1.0, 1.0)),
                    TreemapNode::leaf_colored("lib.rs", 500.0, Color::new(0.4, 0.7, 0.9, 1.0)),
                    TreemapNode::branch(
                        "widgets",
                        vec![
                            TreemapNode::leaf_colored(
                                "button.rs",
                                200.0,
                                Color::new(0.5, 0.7, 0.8, 1.0),
                            ),
                            TreemapNode::leaf_colored(
                                "chart.rs",
                                800.0,
                                Color::new(0.5, 0.8, 0.7, 1.0),
                            ),
                            TreemapNode::leaf_colored(
                                "table.rs",
                                400.0,
                                Color::new(0.6, 0.7, 0.7, 1.0),
                            ),
                        ],
                    ),
                ],
            ),
            TreemapNode::branch(
                "tests",
                vec![
                    TreemapNode::leaf_colored(
                        "test_main.rs",
                        150.0,
                        Color::new(0.7, 0.5, 0.3, 1.0),
                    ),
                    TreemapNode::leaf_colored(
                        "test_widgets.rs",
                        300.0,
                        Color::new(0.8, 0.5, 0.3, 1.0),
                    ),
                ],
            ),
            TreemapNode::leaf_colored("Cargo.toml", 50.0, Color::new(0.6, 0.6, 0.6, 1.0)),
        ],
    );

    let mut treemap = Treemap::new().with_root(root);

    // Render
    let mut buffer = CellBuffer::new(60, 20);
    let mut canvas = DirectTerminalCanvas::new(&mut buffer);
    treemap.layout(Rect::new(0.0, 0.0, 60.0, 20.0));
    treemap.paint(&mut canvas);

    println!("   Created treemap for project structure");
    println!("   - Sizes represent file line counts");
    println!("   - chart.rs is largest (800 lines)\n");

    println!("==============================");
    println!("Demo complete!");
}
