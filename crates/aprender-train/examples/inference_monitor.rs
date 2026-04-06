//! Example: Real-Time Inference Monitoring with Decision Traces
//!
//! Demonstrates the inference monitoring system for APR format models.
//! Features:
//! - Decision path tracing (LinearPath, TreePath, etc.)
//! - Multiple collector strategies (Ring, Stream, HashChain)
//! - Safety Andon integration for alerts
//! - Provenance graphs for incident reconstruction
//!
//! Toyota Way: 現地現物 (Genchi Genbutsu) - Every decision traceable to ground truth

use entrenar::monitor::inference::{
    collector::{HashChainCollector, RingCollector, StreamCollector, StreamFormat, TraceCollector},
    counterfactual::Counterfactual,
    path::{LeafInfo, LinearPath, TreePath, TreeSplit},
    provenance::{
        CausalRelation, IncidentReconstructor, ProvenanceEdge, ProvenanceGraph, ProvenanceNode,
    },
    safety_andon::{SafetyAndon, SafetyIntegrityLevel},
    serialization::{PathType, TraceFormat, TraceSerializer},
    trace::DecisionTrace,
    Explainable, InferenceMonitor,
};
use std::io::Cursor;

// =============================================================================
// Mock Linear Model (implements Explainable)
// =============================================================================

/// Simple linear model for demonstration
struct LinearModel {
    weights: Vec<f32>,
    intercept: f32,
}

impl LinearModel {
    fn new(weights: Vec<f32>, intercept: f32) -> Self {
        Self { weights, intercept }
    }
}

impl Explainable for LinearModel {
    type Path = LinearPath;

    fn predict_explained(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<Self::Path>) {
        let features_per_sample = x.len() / n_samples;
        let mut outputs = Vec::with_capacity(n_samples);
        let mut paths = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start = i * features_per_sample;
            let sample = &x[start..start + features_per_sample];

            let contributions: Vec<f32> =
                self.weights.iter().zip(sample).map(|(w, xi)| w * xi).collect();
            let logit: f32 = contributions.iter().sum::<f32>() + self.intercept;
            let prediction = 1.0 / (1.0 + (-logit).exp()); // sigmoid

            let path = LinearPath::new(contributions, self.intercept, logit, prediction)
                .with_probability(prediction);

            outputs.push(prediction);
            paths.push(path);
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        self.predict_explained(sample, 1).1.into_iter().next().expect("operation should succeed")
    }
}

// =============================================================================
// Mock Decision Tree Model
// =============================================================================

/// Simple decision tree for demonstration
struct DecisionTreeModel;

impl DecisionTreeModel {
    fn new() -> Self {
        Self
    }
}

impl Explainable for DecisionTreeModel {
    type Path = TreePath;

    fn predict_explained(&self, x: &[f32], n_samples: usize) -> (Vec<f32>, Vec<Self::Path>) {
        let features_per_sample = x.len() / n_samples;
        let mut outputs = Vec::with_capacity(n_samples);
        let mut paths = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let start = i * features_per_sample;
            let sample = &x[start..start + features_per_sample];
            let went_left = sample[0] <= 0.5;

            let splits =
                vec![TreeSplit { feature_idx: 0, threshold: 0.5, went_left, n_samples: 100 }];

            let (prediction, class_dist) =
                if went_left { (0.2, vec![0.8, 0.2]) } else { (0.9, vec![0.1, 0.9]) };

            let leaf = LeafInfo {
                prediction,
                n_samples: if went_left { 40 } else { 60 },
                class_distribution: Some(class_dist),
            };

            let path = TreePath::new(splits, leaf);
            outputs.push(prediction);
            paths.push(path);
        }

        (outputs, paths)
    }

    fn explain_one(&self, sample: &[f32]) -> Self::Path {
        self.predict_explained(sample, 1).1.into_iter().next().expect("operation should succeed")
    }
}

// =============================================================================
// Extracted demo functions (one per Part) to reduce cognitive complexity
// =============================================================================

/// Part 1: Linear Model with Ring Collector (Real-Time Gaming Use Case)
fn demo_linear_ring_collector() {
    let model = LinearModel::new(vec![2.0, -1.5, 0.8], 0.5);
    let collector: RingCollector<LinearPath, 64> = RingCollector::new();
    let mut monitor = InferenceMonitor::new(model, collector);

    // Simulate AI decisions in a game
    let game_inputs = vec![
        vec![0.8, 0.3, 0.5], // Player visible, close, has ammo
        vec![0.2, 0.9, 0.1], // Player hidden, far, low ammo
        vec![0.6, 0.5, 0.7], // Moderate visibility
    ];

    println!("Game AI decisions:");
    for (i, input) in game_inputs.iter().enumerate() {
        let outputs = monitor.predict(input, 1);
        let traces = monitor.collector().recent(1);
        let trace = traces[0];

        let decision = if outputs[0] > 0.5 { "ATTACK" } else { "RETREAT" };
        println!("  Frame {i}: input={input:?}");
        println!("           output={:.3} -> {}", outputs[0], decision);
        println!("           confidence={:.1}%", trace.confidence() * 100.0);
        println!("           latency={}ns", trace.latency_ns);
        println!();
    }

    // Show explanation for last decision
    let last_traces = monitor.collector().recent(1);
    let last_trace = last_traces[0];
    println!("Decision explanation:\n{}", last_trace.explain());
}

/// Part 2: Decision Tree with Stream Collector (Persistent Logging)
fn demo_tree_stream_collector() {
    let tree_model = DecisionTreeModel::new();
    let buffer = Cursor::new(Vec::new());
    let stream_collector: StreamCollector<TreePath, _> =
        StreamCollector::new(buffer, StreamFormat::JsonLines);
    let mut tree_monitor = InferenceMonitor::new(tree_model, stream_collector);

    let test_inputs = vec![
        vec![0.3], // Goes left
        vec![0.7], // Goes right
        vec![0.5], // Boundary case
    ];

    println!("Decision tree predictions:");
    for input in &test_inputs {
        let outputs = tree_monitor.predict(input, 1);
        println!("  Input: {input:?}");
        println!("  Prediction: {:.3}", outputs[0]);
        println!();
    }

    println!("Total traces logged: {}", tree_monitor.collector().len());
}

/// Part 3: Hash Chain Collector (Safety-Critical AV Use Case)
fn demo_hash_chain_collector() {
    let av_model = LinearModel::new(vec![1.0, -0.5, 0.3, 0.8], 0.1);
    let hash_collector: HashChainCollector<LinearPath> = HashChainCollector::new();
    let mut av_monitor = InferenceMonitor::new(av_model, hash_collector);

    // Simulate AV perception decisions
    let av_inputs = vec![
        vec![0.9, 0.2, 0.8, 0.1], // Clear road
        vec![0.7, 0.5, 0.6, 0.3], // Pedestrian nearby
        vec![0.3, 0.8, 0.2, 0.9], // Obstacle detected
    ];

    println!("AV perception decisions (hash-chained):");
    for (i, input) in av_inputs.iter().enumerate() {
        let outputs = av_monitor.predict(input, 1);
        let action = match outputs[0] {
            x if x > 0.7 => "PROCEED",
            x if x > 0.4 => "SLOW_DOWN",
            _ => "STOP",
        };

        println!("  T-{}.000s: {} (output={:.3})", 3 - i, action, outputs[0]);
    }

    // Verify chain integrity
    let verification = av_monitor.collector().verify_chain();
    println!("\n  Chain verification: {}", if verification.valid { "VALID" } else { "BROKEN" });
    println!("  Entries verified: {}", verification.entries_verified);
    println!(
        "  Latest hash: {:02x}{:02x}{:02x}{:02x}...",
        av_monitor.collector().latest_hash()[0],
        av_monitor.collector().latest_hash()[1],
        av_monitor.collector().latest_hash()[2],
        av_monitor.collector().latest_hash()[3]
    );
}

/// Part 4: Safety Andon Integration
fn demo_safety_andon() {
    let mut andon = SafetyAndon::new(SafetyIntegrityLevel::SIL2)
        .with_min_confidence(0.7)
        .with_low_confidence_threshold(3);

    println!("Safety Integrity Level: {:?}", andon.sil());
    println!("Min confidence threshold: {:.1}%", andon.sil().min_confidence() * 100.0);
    println!("Max latency: {}ms\n", andon.sil().max_latency_ns() / 1_000_000);

    // Create traces with varying confidence
    let test_traces = vec![
        create_test_trace(0.9, 1_000_000), // High confidence, fast
        create_test_trace(0.5, 5_000_000), // Low confidence
        create_test_trace(0.4, 5_000_000), // Low confidence
        create_test_trace(0.3, 5_000_000), // Low confidence (triggers alert)
    ];

    println!("Checking traces against safety rules:");
    for (i, trace) in test_traces.iter().enumerate() {
        andon.check_trace(trace, 50_000_000);
        println!(
            "  Trace {}: confidence={:.1}%, alerts={}",
            i,
            trace.confidence() * 100.0,
            andon.history().len()
        );
    }

    println!("\nAndon alerts triggered: {}", andon.history().len());
    for alert in andon.history() {
        println!("  [{:?}] {}", alert.level, alert.message);
    }
}

/// Part 5: Counterfactual Explanations
fn demo_counterfactual() {
    let feature_names =
        vec!["income".to_string(), "debt_ratio".to_string(), "credit_score".to_string()];
    let counterfactual = Counterfactual::new(
        vec![0.3, 0.8, 0.2], // Original: loan denied
        0,                   // Denied class
        0.3,                 // Low confidence
        vec![0.6, 0.5, 0.4], // Counterfactual: would be approved
        1,                   // Approved class
        0.8,                 // High confidence
    )
    .with_feature_names(&feature_names);

    println!("Original decision: DENIED");
    println!("Counterfactual (what would flip decision):\n");
    println!("{}", counterfactual.explain());
    println!("\nSparsity (L1): {:.3}", counterfactual.sparsity);
    println!("Distance (L2): {:.3}", counterfactual.distance);
}

/// Part 6: Provenance Graph for Incident Reconstruction
fn demo_provenance_graph() {
    let mut graph = ProvenanceGraph::new();

    // Build incident provenance: Camera -> Preprocess -> Detect -> Action
    let camera = graph.add_node(ProvenanceNode::Input {
        source: "front_camera".to_string(),
        timestamp_ns: 1000,
        hash: 0xdeadbeef,
    });

    let preprocess = graph.add_node(ProvenanceNode::Transform {
        operation: "normalize".to_string(),
        input_refs: vec![camera],
    });

    let detection = graph.add_node(ProvenanceNode::Inference {
        model_id: "object_detector".to_string(),
        model_version: "v2.1".to_string(),
        confidence: 0.3, // Low confidence - anomaly!
        output: 0.0,
    });

    let action = graph.add_node(ProvenanceNode::Action {
        action_type: "emergency_brake".to_string(),
        confidence: 0.85,
        alternatives: vec![("swerve".to_string(), 0.12)],
    });

    // Connect the graph
    graph.add_edge(ProvenanceEdge {
        from: camera,
        to: preprocess,
        relation: CausalRelation::DataFlow,
        timestamp_ns: 1100,
    });
    graph.add_edge(ProvenanceEdge {
        from: preprocess,
        to: detection,
        relation: CausalRelation::Triggered,
        timestamp_ns: 1200,
    });
    graph.add_edge(ProvenanceEdge {
        from: detection,
        to: action,
        relation: CausalRelation::Caused,
        timestamp_ns: 1300,
    });

    println!("Provenance graph constructed:");
    println!("  Nodes: {}", graph.node_count());
    println!("  Edges: {}", graph.edge_count());

    // Reconstruct incident path
    let reconstructor = IncidentReconstructor::new(&graph);
    let attack_path = reconstructor.reconstruct_path(action, 10);

    println!("\nIncident path reconstruction (root cause analysis):");
    for (id, node) in &attack_path.nodes {
        println!("  [{}] {}", id, node.type_name());
    }

    // Identify anomalies
    let anomalies = reconstructor.identify_anomalies(&attack_path, 0.7);
    println!("\nAnomalies detected: {}", anomalies.len());
    for anomaly in &anomalies {
        println!(
            "  Node {}: {} (severity: {:.1}%)",
            anomaly.node_id,
            anomaly.description,
            anomaly.severity * 100.0
        );
    }
}

/// Part 7: Serialization (Binary and JSON)
fn demo_serialization() {
    let path = LinearPath::new(vec![0.5, -0.3, 0.2], 0.1, 0.5, 0.62).with_probability(0.62);
    let trace = DecisionTrace::new(1_000_000_000, 42, 0xdeadbeef, path, 0.62, 500);

    // Binary serialization
    let binary_serializer = TraceSerializer::new(TraceFormat::Binary);
    let binary_bytes =
        binary_serializer.serialize(&trace, PathType::Linear).expect("operation should succeed");
    println!("Binary (APRT format):");
    println!("  Size: {} bytes", binary_bytes.len());
    println!(
        "  Magic: {:02X} {:02X} {:02X} {:02X} (\"APRT\")",
        binary_bytes[0], binary_bytes[1], binary_bytes[2], binary_bytes[3]
    );

    // JSON serialization
    let json_serializer = TraceSerializer::new(TraceFormat::Json);
    let json_bytes =
        json_serializer.serialize(&trace, PathType::Linear).expect("operation should succeed");
    println!("\nJSON format:");
    println!("  Size: {} bytes", json_bytes.len());
    println!("  Preview: {}...", String::from_utf8_lossy(&json_bytes[..80.min(json_bytes.len())]));

    // Round-trip verification
    let restored: DecisionTrace<LinearPath> =
        binary_serializer.deserialize(&binary_bytes).expect("operation should succeed");
    println!("\nRound-trip verification:");
    println!("  Original sequence: {}", trace.sequence);
    println!("  Restored sequence: {}", restored.sequence);
    println!("  Match: {}", trace.sequence == restored.sequence);
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     Real-Time Inference Monitoring Example                      ║");
    println!("║     Toyota Way: 現地現物 (Genchi Genbutsu)                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 1: Linear Model with RingCollector (Video Games)");
    println!("Target: <100ns per trace, bounded memory");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    demo_linear_ring_collector();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 2: Decision Tree with StreamCollector (Persistent Logging)");
    println!("Target: Write-through to file/network");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    demo_tree_stream_collector();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 3: HashChainCollector (Autonomous Vehicle)");
    println!("Target: Tamper-evident audit trail with SHA-256");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    demo_hash_chain_collector();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 4: Safety Andon (自働化 Jidoka)");
    println!("Target: Automatic quality checks with alerts");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    demo_safety_andon();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 5: Counterfactual Explanations");
    println!("Target: 'What would need to change to flip the decision?'");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    demo_counterfactual();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 6: Provenance Graph (Incident Reconstruction)");
    println!("Target: Reconstruct causal chain for forensic analysis");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    demo_provenance_graph();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Part 7: Trace Serialization (APRT Binary + JSON)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    demo_serialization();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Example Complete!                            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

/// Helper to create test traces
fn create_test_trace(probability: f32, latency_ns: u64) -> DecisionTrace<LinearPath> {
    let path = LinearPath::new(vec![0.5], 0.0, 0.0, probability).with_probability(probability);
    DecisionTrace::new(1_000_000, 0, 0, path, probability, latency_ns)
}
