# Stack Visualization, Diagnostics, and Reporting Specification v1.0

**Status**: Draft
**Version**: 1.0.0
**Last Updated**: 2025-12-07
**Author**: PAIML Engineering
**Classification**: Technical Specification

---

## Executive Summary

Batuta Stack Diagnostics is a unified ML-driven system for visualizing, diagnosing, and reporting on the health of the Sovereign AI Stack. It synthesizes **dependency graph analysis** (trueno-graph), **quality signals** (pmat/certeza), **runtime traces** (renacer), and **performance metrics** (trueno) to provide actionable insights about key dependencies, errors, and improvement opportunities across all 20+ stack components.

### Design Philosophy: Toyota Production System

This specification applies Toyota Way principles to stack observability:

- **Mieruka (Visual Control)**: Rich ASCII visualizations make stack health immediately visible
- **Jidoka (Autonomation with Human Intelligence)**: ML models surface anomalies; humans approve remediation
- **Genchi Genbutsu (Go and See)**: Evidence-based diagnosis from actual dependency graphs, not assumptions
- **Andon (Stop-the-Line)**: Automatic alerts when critical dependencies degrade
- **Yokoten (Horizontal Deployment)**: Share insights across stack components via knowledge graph

### Scientific Foundation

This specification synthesizes methods from peer-reviewed publications spanning graph analytics, anomaly detection, dependency analysis, and software visualization (see [References](#references)).

---

## 1. Problem Statement

### 1.1 Current State: Fragmented Stack Visibility

Sovereign AI Stack operators face fragmented visibility across 20+ components:

| Signal Source | What It Provides | Integration Status |
|---------------|------------------|-------------------|
| `cargo tree` | Dependency listing | Text dump, no analysis |
| `pmat demo-score` | Project quality score | Per-project, no cross-stack |
| `certeza` | Quality gate enforcement | Per-project, no aggregation |
| `renacer` | Syscall traces | Per-execution, no correlation |
| `trueno-graph` | Graph analytics | Raw graphs, no interpretation |
| `cargo audit` | Security advisories | Per-project, no stack view |
| CI/CD logs | Build/test status | Scattered, no synthesis |

**Problem**: No unified system synthesizes these signals into a coherent stack health view with ML-driven insights.

### 1.2 Target State: Unified Stack Intelligence

Batuta Diagnostics provides:

| Capability | Description | Implementation |
|------------|-------------|----------------|
| **Dependency Graph** | Interactive visualization of all stack dependencies | trueno-graph + ASCII renderer |
| **Health Dashboard** | Aggregate quality scores across all components | pmat + certeza synthesis |
| **Anomaly Detection** | ML-driven identification of unusual patterns | aprender clustering + isolation forest |
| **Error Correlation** | Link errors across components to root causes | renacer traces + graph analysis |
| **Upgrade Advisor** | Recommend dependency upgrades with impact analysis | PageRank + breaking change detection |
| **Performance Insights** | Identify bottlenecks across stack boundaries | trueno profiling + trace correlation |

---

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BATUTA STACK DIAGNOSTICS SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      DATA COLLECTION LAYER                          │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ DEPENDENCY SIGNALS                                           │   │   │
│  │  │ • Cargo.toml parsing (direct dependencies)                   │   │   │
│  │  │ • Cargo.lock resolution (transitive closure)                 │   │   │
│  │  │ • Feature flags (conditional compilation)                    │   │   │
│  │  │ • Version constraints (semver ranges)                        │   │   │
│  │  │ • Workspace relationships (multi-crate)                      │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ QUALITY SIGNALS                                              │   │   │
│  │  │ • pmat demo-score (110-point normalized to 100)              │   │   │
│  │  │ • certeza gates (coverage, mutation, complexity)             │   │   │
│  │  │ • TDG grades (A++ to F per file)                             │   │   │
│  │  │ • SATD markers (TODO, FIXME, HACK counts)                    │   │   │
│  │  │ • Dead code percentage                                       │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ RUNTIME SIGNALS                                              │   │   │
│  │  │ • renacer syscall traces (I/O, memory, network)              │   │   │
│  │  │ • Build times (incremental, full)                            │   │   │
│  │  │ • Test execution times                                       │   │   │
│  │  │ • Binary sizes                                               │   │   │
│  │  │ • Memory footprints                                          │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    STACK KNOWLEDGE GRAPH                            │   │
│  │  StackGraph {                                                       │   │
│  │    nodes: Vec<ComponentNode>,    // 20+ stack components           │   │
│  │    edges: Vec<DependencyEdge>,   // Direct + transitive deps       │   │
│  │    metrics: HashMap<NodeId, ComponentMetrics>,                      │   │
│  │    history: Vec<SnapshotDelta>,  // Change over time               │   │
│  │  }                                                                  │   │
│  │                                                                     │   │
│  │  ComponentNode {                                                    │   │
│  │    id: NodeId,                                                      │   │
│  │    name: String,                 // e.g., "trueno", "aprender"     │   │
│  │    version: Version,                                                │   │
│  │    layer: StackLayer,            // Compute, ML, Transpiler, ...   │   │
│  │    health: HealthStatus,         // Green, Yellow, Red             │   │
│  │  }                                                                  │   │
│  │                                                                     │   │
│  │  DependencyEdge {                                                   │   │
│  │    from: NodeId,                                                    │   │
│  │    to: NodeId,                                                      │   │
│  │    kind: EdgeKind,               // Direct, Dev, Build, Optional   │   │
│  │    version_req: VersionReq,                                         │   │
│  │    features: Vec<String>,                                           │   │
│  │  }                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ML ANALYTICS ENGINE                            │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐     │   │
│  │  │ Graph Metrics  │  │ Anomaly        │  │ Trend Prediction   │     │   │
│  │  │ (trueno-graph) │  │ Detection (ap) │  │ (aprender)         │     │   │
│  │  │                │  │                │  │                    │     │   │
│  │  │ • PageRank     │  │ • Isolation    │  │ • Quality trends   │     │   │
│  │  │ • Betweenness  │  │   Forest       │  │ • Upgrade impact   │     │   │
│  │  │ • Clustering   │  │ • K-means      │  │ • Risk scoring     │     │   │
│  │  │ • Communities  │  │ • DBSCAN       │  │ • Forecasting      │     │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘     │   │
│  │                              │                                       │   │
│  │                              ▼                                       │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ INSIGHT SYNTHESIS                                            │   │   │
│  │  │ • Critical path identification                               │   │   │
│  │  │ • Bottleneck detection                                       │   │   │
│  │  │ • Upgrade recommendations                                    │   │   │
│  │  │ • Risk assessment                                            │   │   │
│  │  │ • Improvement prioritization                                 │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      VISUALIZATION LAYER                            │   │
│  │                                                                      │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐     │   │
│  │   │ ASCII Graph  │  │ Health       │  │ Trend Charts         │     │   │
│  │   │ Renderer     │  │ Dashboard    │  │ (Spark-lines)        │     │   │
│  │   │              │  │              │  │                      │     │   │
│  │   │ • Box-drawing│  │ • Status     │  │ • Quality over time  │     │   │
│  │   │ • owo-colors │  │   indicators │  │ • Dependency growth  │     │   │
│  │   │ • Interactive│  │ • Grade bars │  │ • Build time trends  │     │   │
│  │   └──────────────┘  └──────────────┘  └──────────────────────┘     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Integration Matrix

| Component | Source | Role in Diagnostics |
|-----------|--------|---------------------|
| **trueno-graph** | PAIML | PageRank, community detection, BFS/DFS traversal |
| **aprender** | PAIML | RandomForest for risk scoring, k-NN for similar issues |
| **trueno** | PAIML | SIMD-accelerated vector operations for embeddings |
| **renacer** | PAIML | Syscall tracing for runtime behavior analysis |
| **pmat** | PAIML | Quality scoring, TDG analysis, demo-score |
| **certeza** | PAIML | Quality gate enforcement, mutation testing |
| **batuta** | PAIML | Orchestration, knowledge graph, version management |

---

## 3. Stack Layer Taxonomy

### 3.1 Layer Definitions

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackLayer {
    /// Core compute primitives (SIMD, GPU, WASM)
    Compute,

    /// Machine learning (training, inference, algorithms)
    MachineLearning,

    /// Language transpilers (Python, Bash, TypeScript)
    Transpiler,

    /// Data management (databases, graphs, RAG)
    DataLayer,

    /// Quality and testing (coverage, mutation, TDG)
    Quality,

    /// Orchestration and tooling
    Orchestration,
}

impl StackLayer {
    pub fn components(&self) -> &[&str] {
        match self {
            StackLayer::Compute => &["trueno"],
            StackLayer::MachineLearning => &["aprender", "realizar", "entrenar"],
            StackLayer::Transpiler => &["depyler", "bashrs", "decy"],
            StackLayer::DataLayer => &["trueno-db", "trueno-graph", "trueno-rag"],
            StackLayer::Quality => &["pmat", "certeza", "verificar"],
            StackLayer::Orchestration => &["batuta", "renacer", "repartir"],
        }
    }
}
```

### 3.2 Layer Dependency Rules

The layer hierarchy defines **cross-layer impact propagation**. When a lower layer (e.g., Compute) experiences issues, all dependent upper layers are affected. Batuta tracks these cascading effects:

- **Downward Dependency**: Upper layers depend on lower layers (ML depends on Data depends on Compute)
- **Impact Amplification**: A Compute-layer bug may cause failures in 15+ downstream components
- **Root Cause Attribution**: Graph traversal identifies the originating layer for cross-cutting issues

See [Section 4: Dependency Graph Analysis](#4-dependency-graph-analysis) for algorithms that trace these cross-layer relationships.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         STACK LAYER HIERARCHY                             │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 0: COMPUTE PRIMITIVES                                        │  │
│  │ ┌──────────┐                                                       │  │
│  │ │  trueno  │  SIMD/GPU/WASM compute                               │  │
│  │ └────┬─────┘                                                       │  │
│  └──────┼─────────────────────────────────────────────────────────────┘  │
│         │ depends_on                                                     │
│         ▼                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 1: DATA LAYER                                                │  │
│  │ ┌──────────┐  ┌──────────────┐  ┌────────────┐                     │  │
│  │ │trueno-db │  │ trueno-graph │  │ trueno-rag │                     │  │
│  │ └────┬─────┘  └──────┬───────┘  └─────┬──────┘                     │  │
│  └──────┼───────────────┼────────────────┼────────────────────────────┘  │
│         │               │                │                               │
│         ▼               ▼                ▼                               │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 2: MACHINE LEARNING                                          │  │
│  │ ┌──────────┐  ┌──────────┐  ┌──────────┐                           │  │
│  │ │ aprender │  │ realizar │  │ entrenar │                           │  │
│  │ └────┬─────┘  └────┬─────┘  └────┬─────┘                           │  │
│  └──────┼─────────────┼─────────────┼─────────────────────────────────┘  │
│         │             │             │                                    │
│         ▼             ▼             ▼                                    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 3: TRANSPILERS                                               │  │
│  │ ┌──────────┐  ┌──────────┐  ┌──────────┐                           │  │
│  │ │ depyler  │  │  bashrs  │  │   decy   │                           │  │
│  │ └────┬─────┘  └────┬─────┘  └────┬─────┘                           │  │
│  └──────┼─────────────┼─────────────┼─────────────────────────────────┘  │
│         │             │             │                                    │
│         ▼             ▼             ▼                                    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ LAYER 4: QUALITY & ORCHESTRATION                                   │  │
│  │ ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  │
│  │ │   pmat   │  │ certeza  │  │  batuta  │  │ renacer  │             │  │
│  │ └──────────┘  └──────────┘  └──────────┘  └──────────┘             │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Dependency Graph Analysis

### 4.1 Graph Construction

```rust
/// Build the stack dependency graph from all component Cargo.tomls
pub async fn build_stack_graph(stack_root: &Path) -> Result<StackGraph> {
    let mut graph = StackGraph::new();

    // Discover all stack components
    let components = discover_components(stack_root).await?;

    for component in &components {
        // Add node
        let node = ComponentNode {
            id: NodeId::new(&component.name),
            name: component.name.clone(),
            version: component.version.clone(),
            layer: classify_layer(&component.name),
            health: HealthStatus::Unknown,
        };
        graph.add_node(node);

        // Parse Cargo.toml for dependencies
        let cargo_toml = component.path.join("Cargo.toml");
        let manifest = parse_cargo_toml(&cargo_toml)?;

        for (dep_name, dep_info) in manifest.dependencies {
            if is_stack_component(&dep_name) {
                let edge = DependencyEdge {
                    from: NodeId::new(&component.name),
                    to: NodeId::new(&dep_name),
                    kind: EdgeKind::Direct,
                    version_req: dep_info.version,
                    features: dep_info.features,
                };
                graph.add_edge(edge);
            }
        }

        // Add dev-dependencies
        for (dep_name, dep_info) in manifest.dev_dependencies {
            if is_stack_component(&dep_name) {
                let edge = DependencyEdge {
                    from: NodeId::new(&component.name),
                    to: NodeId::new(&dep_name),
                    kind: EdgeKind::Dev,
                    version_req: dep_info.version,
                    features: dep_info.features,
                };
                graph.add_edge(edge);
            }
        }
    }

    // Compute transitive closure
    graph.compute_transitive_closure();

    Ok(graph)
}
```

### 4.2 Graph Metrics

| Metric | Algorithm | Insight |
|--------|-----------|---------|
| **PageRank** | Power iteration | Identifies most critical components |
| **Betweenness Centrality** | Brandes algorithm | Finds bottleneck components |
| **Clustering Coefficient** | Triangle counting | Measures component cohesion |
| **Community Detection** | Louvain algorithm | Identifies natural groupings |
| **Dependency Depth** | BFS from roots | Measures build complexity |

```rust
/// Compute all graph metrics for stack analysis
pub fn compute_graph_metrics(graph: &StackGraph) -> GraphMetrics {
    GraphMetrics {
        pagerank: graph.pagerank(0.85, 100),
        betweenness: graph.betweenness_centrality(),
        clustering: graph.clustering_coefficient(),
        communities: graph.louvain_communities(),
        depth_map: graph.compute_depth_from_roots(),

        // Aggregate metrics
        total_nodes: graph.node_count(),
        total_edges: graph.edge_count(),
        density: graph.density(),
        avg_degree: graph.average_degree(),
        max_depth: graph.max_depth(),
    }
}
```

### 4.3 Critical Path Analysis

```rust
/// Identify the critical path through the stack
/// (longest dependency chain affecting build time)
pub fn critical_path(graph: &StackGraph) -> Vec<NodeId> {
    // Use DAG longest path algorithm
    let topo_order = graph.topological_sort()?;

    let mut dist = HashMap::new();
    let mut pred = HashMap::new();

    for node in &topo_order {
        dist.insert(*node, 0);
    }

    for node in &topo_order {
        for edge in graph.outgoing_edges(*node) {
            let new_dist = dist[node] + edge.weight();
            if new_dist > dist[&edge.to] {
                dist.insert(edge.to, new_dist);
                pred.insert(edge.to, *node);
            }
        }
    }

    // Reconstruct path from sink to source
    let sink = dist.iter().max_by_key(|(_, d)| *d).map(|(n, _)| *n)?;
    let mut path = vec![sink];
    let mut current = sink;

    while let Some(&prev) = pred.get(&current) {
        path.push(prev);
        current = prev;
    }

    path.reverse();
    path
}
```

---

## 5. ML-Driven Insights

### 5.1 Anomaly Detection

```rust
/// Detect anomalous components using Isolation Forest
pub struct AnomalyDetector {
    forest: IsolationForest,
    feature_extractor: FeatureExtractor,
}

impl AnomalyDetector {
    pub fn detect_anomalies(&self, graph: &StackGraph) -> Vec<Anomaly> {
        let features = self.feature_extractor.extract_all(graph);

        features.iter()
            .filter_map(|(node_id, feature_vec)| {
                let score = self.forest.anomaly_score(feature_vec);
                if score > ANOMALY_THRESHOLD {
                    Some(Anomaly {
                        node_id: *node_id,
                        score,
                        category: self.classify_anomaly(feature_vec),
                        evidence: self.explain_anomaly(node_id, feature_vec),
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct FeatureVector {
    // Quality features
    pub demo_score: f32,
    pub coverage: f32,
    pub mutation_score: f32,
    pub complexity_avg: f32,
    pub satd_count: u32,
    pub dead_code_pct: f32,

    // Graph features
    pub pagerank: f32,
    pub betweenness: f32,
    pub in_degree: u32,
    pub out_degree: u32,
    pub depth: u32,

    // Runtime features
    pub build_time_secs: f32,
    pub test_time_secs: f32,
    pub binary_size_kb: u32,

    // Historical features
    pub churn_rate: f32,
    pub defect_rate: f32,
    pub version_age_days: u32,
}
```

### 5.2 Upgrade Impact Prediction

```rust
/// Predict impact of upgrading a dependency
pub struct UpgradeAdvisor {
    model: RandomForest,
    knowledge_base: StackKnowledgeGraph,
}

impl UpgradeAdvisor {
    pub fn analyze_upgrade(
        &self,
        component: &str,
        from_version: &Version,
        to_version: &Version,
    ) -> UpgradeAnalysis {
        // Compute version delta features
        let delta = VersionDelta::compute(from_version, to_version);

        // Query historical upgrade outcomes
        let similar_upgrades = self.knowledge_base
            .query_similar_upgrades(component, &delta);

        // Predict risk using Random Forest
        let features = self.extract_upgrade_features(&delta, &similar_upgrades);
        let risk_score = self.model.predict(&features);

        // Identify affected downstream components
        let affected = self.knowledge_base
            .graph
            .reverse_dependencies(component);

        UpgradeAnalysis {
            component: component.to_string(),
            from: from_version.clone(),
            to: to_version.clone(),
            risk_score,
            risk_category: RiskCategory::from_score(risk_score),
            breaking_changes: delta.breaking_changes(),
            affected_components: affected,
            recommendations: self.generate_recommendations(&delta, risk_score),
            similar_outcomes: similar_upgrades,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RiskCategory {
    Low,      // < 0.3: Safe to auto-upgrade
    Medium,   // 0.3-0.7: Review recommended
    High,     // > 0.7: Manual testing required
    Critical, // Breaking changes detected
}
```

### 5.3 Quality Trend Forecasting

Error forecasting predicts **future error volume spikes** based on historical patterns, enabling proactive capacity planning and maintenance scheduling. Specifically:

- **Seasonal Patterns**: Detects weekly/monthly cycles (e.g., higher error rates during release windows)
- **Threshold Prediction**: Forecasts when error counts will exceed alert thresholds
- **Trend Extrapolation**: Projects quality degradation to schedule preventive refactoring

```rust
/// Forecast quality metrics using time series analysis
pub struct QualityForecaster {
    // Simple exponential smoothing for each metric
    smoothing_alpha: f32,
    history: VecDeque<StackSnapshot>,
}

impl QualityForecaster {
    pub fn forecast(&self, horizon_days: u32) -> Vec<QualityForecast> {
        self.knowledge_base
            .graph
            .nodes()
            .map(|node| {
                let history = self.get_metric_history(node.id);

                QualityForecast {
                    component: node.name.clone(),
                    demo_score: self.exponential_smooth(&history.demo_scores, horizon_days),
                    coverage: self.exponential_smooth(&history.coverages, horizon_days),
                    trend: self.compute_trend(&history),
                    confidence: self.compute_confidence(&history),
                }
            })
            .collect()
    }

    fn compute_trend(&self, history: &MetricHistory) -> Trend {
        let slope = linear_regression_slope(&history.demo_scores);

        if slope > IMPROVING_THRESHOLD {
            Trend::Improving
        } else if slope < DEGRADING_THRESHOLD {
            Trend::Degrading
        } else {
            Trend::Stable
        }
    }
}
```

---

## 6. Error Correlation and Root Cause Analysis

### 6.1 Cross-Component Error Linking

```rust
/// Link errors across component boundaries
pub struct ErrorCorrelator {
    graph: StackGraph,
    traces: Vec<RenacerTrace>,
}

impl ErrorCorrelator {
    pub fn correlate_error(&self, error: &StackError) -> ErrorCorrelation {
        // 1. Identify the failing component
        let failing_component = self.identify_component(&error.location);

        // 2. Find upstream dependencies that might be root cause
        let upstream = self.graph.ancestors(failing_component);

        // 3. Check for recent changes in upstream components
        let recent_changes = upstream.iter()
            .filter_map(|c| self.get_recent_changes(c))
            .collect();

        // 4. Correlate with renacer traces if available
        let trace_evidence = self.find_trace_evidence(&error.timestamp);

        // 5. Apply five-whys analysis
        let root_cause = self.five_whys_analysis(
            error,
            &upstream,
            &recent_changes,
            &trace_evidence,
        );

        ErrorCorrelation {
            error: error.clone(),
            failing_component,
            upstream_candidates: upstream,
            recent_changes,
            trace_evidence,
            root_cause,
            confidence: self.compute_confidence(&root_cause),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RootCause {
    pub component: String,
    pub category: ErrorCategory,
    pub description: String,
    pub evidence: Vec<Evidence>,
    pub suggested_fix: Option<SuggestedFix>,
    pub prevention: String,
}
```

### 6.2 Error Categories

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    // Dependency errors
    VersionMismatch,
    FeatureMissing,
    TransitiveDependency,

    // API errors
    BreakingChange,
    DeprecatedUsage,
    TypeMismatch,

    // Build errors
    CompilationFailure,
    LinkError,
    ResourceExhaustion,

    // Runtime errors
    PanicUnwind,
    AssertionFailure,
    Timeout,

    // Quality errors
    CoverageRegression,
    MutationEscape,
    ComplexityExceeded,
}
```

### 6.3 Fishbone (Ishikawa) Root Cause Diagram

Batuta generates Fishbone diagrams to visualize root cause categories. The 6 M's adapted for software:

```
                        ┌─────────────────────────────────────────────────────────┐
                        │               FISHBONE ROOT CAUSE ANALYSIS              │
                        └─────────────────────────────────────────────────────────┘

        Method                     Machine                    Material
           │                          │                          │
           │  Incorrect algorithm     │  GPU driver mismatch     │  Corrupt config
           │  Race condition          │  Memory exhaustion       │  Invalid input
           │  Timeout logic           │  CPU throttling          │  Schema mismatch
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                                      ▼
              ┌───────────────────────────────────────────────────┐
              │                 STACK FAILURE                      │
              │          (e.g., trueno-graph BFS timeout)         │
              └───────────────────────────────────────────────────┘
                                      ▲
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           │  Human misconfiguration  │  Upstream breaking API   │  New deploy
           │  Operator error          │  Transitive dep update   │  CI skip
           │  Missing documentation   │  Feature flag conflict   │  Rollback
           │                          │                          │
        People                    Environment               Measurement
        (Human Factors)           (Dependencies)            (Process)
```

**People Category** explicitly captures:
- **Human Misconfiguration**: Incorrect `.toml` settings, environment variables
- **Operator Error**: Wrong command flags, skipped validation steps
- **Documentation Gap**: Undocumented requirements leading to incorrect usage

---

## 7. Visualization and Reporting

### 7.1 ASCII Graph Renderer

```rust
/// Render stack graph as ASCII art
pub struct AsciiGraphRenderer {
    width: usize,
    use_colors: bool,
}

impl AsciiGraphRenderer {
    pub fn render(&self, graph: &StackGraph, metrics: &GraphMetrics) -> String {
        let mut output = String::new();

        // Header
        output.push_str(&self.render_header());

        // Layer-by-layer rendering
        for layer in StackLayer::all() {
            output.push_str(&self.render_layer(graph, layer, metrics));
        }

        // Legend
        output.push_str(&self.render_legend());

        output
    }

    fn render_layer(
        &self,
        graph: &StackGraph,
        layer: StackLayer,
        metrics: &GraphMetrics,
    ) -> String {
        let nodes = graph.nodes_in_layer(layer);
        let mut layer_str = format!("\n{:═<60}\n", format!(" {:?} ", layer));

        for node in nodes {
            let health_icon = match node.health {
                HealthStatus::Green => "●",
                HealthStatus::Yellow => "◐",
                HealthStatus::Red => "○",
                HealthStatus::Unknown => "◌",
            };

            let score = metrics.pagerank.get(&node.id).unwrap_or(&0.0);
            let bar = self.render_bar(*score * 100.0, 20);

            layer_str.push_str(&format!(
                "  {} {:<15} {} {:.1}\n",
                health_icon, node.name, bar, score * 100.0
            ));
        }

        layer_str
    }

    fn render_bar(&self, value: f32, width: usize) -> String {
        let filled = (value / 100.0 * width as f32) as usize;
        let empty = width - filled;
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }
}
```

### 7.2 Rich Health Dashboard

The ASCII dashboard implements **Mieruka (Visual Control)** with dynamic, real-time updates:

- **Refresh Interval**: Configurable polling (default: 60s) redraws the entire dashboard
- **Watch Mode**: `batuta diagnose --watch` streams updates using ANSI cursor repositioning
- **Differential Updates**: Only changed cells redraw to minimize terminal flicker
- **Interactive Navigation**: Arrow keys navigate between components; Enter drills down to detail views
- **Color Coding**: owo-colors applies semantic highlighting (green=healthy, yellow=warning, red=critical)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  SOVEREIGN AI STACK HEALTH DASHBOARD                     │
│                  Timestamp: 2025-12-07 14:30:00                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ╔════════════════════════════════════════════════════════════════════╗ │
│  ║  ANDON STATUS: 🟢 GREEN (All systems healthy)                      ║ │
│  ╚════════════════════════════════════════════════════════════════════╝ │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│  STACK SUMMARY                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Total Components:    20                                                │
│  Healthy:             17 (85%)                                          │
│  Warnings:             3 (15%)                                          │
│  Critical:             0 (0%)                                           │
│  Average Demo Score:  84.7/100                                          │
│  Average Coverage:    92.3%                                             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ QUALITY BY LAYER                                                 │   │
│  │                                                                  │   │
│  │ Compute          ██████████████████░░ 91.2 (A-)                 │   │
│  │ Data Layer       █████████████████░░░ 87.4 (A-)                 │   │
│  │ ML               ████████████████░░░░ 84.6 (B+)                 │   │
│  │ Transpiler       ███████████████░░░░░ 79.8 (B)                  │   │
│  │ Quality          ██████████████████░░ 90.1 (A-)                 │   │
│  │ Orchestration    █████████████████░░░ 85.2 (A-)                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│  CRITICAL PATH (longest dependency chain)                               │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  trueno → trueno-graph → aprender → depyler → pmat → batuta            │
│  └─────────────────────────────────────────────────────────────────┘    │
│  Depth: 6 | Build Impact: High | Test Impact: Medium                    │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│  PAGERANK TOP 5 (most critical components)                              │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Rank │ Component      │ PageRank │ Health │ Demo Score │ Coverage     │
│  ─────┼────────────────┼──────────┼────────┼────────────┼─────────────│
│  #1   │ trueno         │ 0.1842   │ 🟢     │ 91.2       │ 100.0%      │
│  #2   │ aprender       │ 0.1234   │ 🟢     │ 85.3       │ 95.2%       │
│  #3   │ trueno-graph   │ 0.0921   │ 🟡     │ 78.4       │ 89.1%       │
│  #4   │ pmat           │ 0.0876   │ 🟢     │ 88.2       │ 94.7%       │
│  #5   │ depyler        │ 0.0654   │ 🟡     │ 76.9       │ 87.3%       │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│  COMMUNITY DETECTION (Louvain, modularity=0.72)                         │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Community 1 (Core): trueno, trueno-db, trueno-graph, trueno-rag       │
│  Community 2 (ML):   aprender, realizar, entrenar                       │
│  Community 3 (Lang): depyler, bashrs, decy                              │
│  Community 4 (QA):   pmat, certeza, verificar                           │
│  Community 5 (Ops):  batuta, renacer, repartir                          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│  ANOMALIES DETECTED (Isolation Forest, threshold=0.65)                  │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  ⚠️  trueno-graph: Coverage dropped 5.2% since last week                │
│      └─ Evidence: lcov.info shows missing tests in gpu/ module          │
│      └─ Recommendation: Add tests for GPU BFS implementation            │
│                                                                         │
│  ⚠️  depyler: Build time increased 40% (45s → 63s)                      │
│      └─ Evidence: New macro expansion in ast_transform.rs               │
│      └─ Recommendation: Consider incremental compilation cache          │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│  UPGRADE RECOMMENDATIONS (sorted by impact × confidence)                │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  1. [LOW RISK] trueno 0.7.0 → 0.7.1 (patch release)                    │
│     └─ Affected: 8 downstream components                                │
│     └─ Impact: Performance fix for matrix multiply                      │
│                                                                         │
│  2. [MEDIUM RISK] serde 1.0.210 → 1.0.215 (5 patch versions)           │
│     └─ Affected: 18 downstream components                               │
│     └─ Impact: Security fix CVE-2025-XXXX                               │
│                                                                         │
│  ════════════════════════════════════════════════════════════════════   │
│  TREND FORECAST (next 7 days)                                           │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                         │
│  Demo Score:  84.7 → 85.2 (↑ improving)                                 │
│  Coverage:    92.3 → 92.8 (↑ improving)                                 │
│  Build Time:  2.4m → 2.3m (↑ improving)                                 │
│                                                                         │
│  Trend Spark-lines (last 14 days):                                      │
│  Demo Score: ▁▂▃▃▄▄▅▅▆▆▆▇▇█                                            │
│  Coverage:   ▃▃▄▄▄▅▅▅▆▆▆▆▇▇                                            │
│  Complexity: ▇▇▆▆▆▅▅▅▄▄▄▃▃▂                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Dependency Graph Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   STACK DEPENDENCY GRAPH                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                           ┌─────────┐                                   │
│                           │ trueno  │ (0.7.0)                           │
│                           │   🟢    │                                   │
│                           └────┬────┘                                   │
│                                │                                        │
│           ┌────────────────────┼────────────────────┐                   │
│           │                    │                    │                   │
│           ▼                    ▼                    ▼                   │
│     ┌──────────┐        ┌────────────┐       ┌────────────┐            │
│     │trueno-db │        │trueno-graph│       │ trueno-rag │            │
│     │    🟢    │        │    🟡      │       │    🟢      │            │
│     └────┬─────┘        └─────┬──────┘       └─────┬──────┘            │
│          │                    │                    │                   │
│          └──────────┬─────────┴──────────┬─────────┘                   │
│                     │                    │                             │
│                     ▼                    ▼                             │
│              ┌──────────┐         ┌──────────┐                         │
│              │ aprender │         │ realizar │                         │
│              │    🟢    │         │    🟢    │                         │
│              └────┬─────┘         └────┬─────┘                         │
│                   │                    │                               │
│          ┌────────┴────────┐           │                               │
│          ▼                 ▼           ▼                               │
│    ┌──────────┐      ┌──────────┐ ┌──────────┐                         │
│    │ depyler  │      │  bashrs  │ │   decy   │                         │
│    │    🟡    │      │    🟢    │ │    🟢    │                         │
│    └────┬─────┘      └────┬─────┘ └────┬─────┘                         │
│         │                 │            │                               │
│         └─────────────────┴────────────┘                               │
│                           │                                            │
│                           ▼                                            │
│                    ┌──────────┐                                        │
│                    │   pmat   │                                        │
│                    │    🟢    │                                        │
│                    └────┬─────┘                                        │
│                         │                                              │
│                         ▼                                              │
│                    ┌──────────┐                                        │
│                    │  batuta  │                                        │
│                    │    🟢    │                                        │
│                    └──────────┘                                        │
│                                                                         │
│  Legend: 🟢 Healthy  🟡 Warning  🔴 Critical  ─▶ Direct Dependency      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. CLI Interface

### 8.1 Commands

```bash
# Full stack health dashboard
batuta diagnose --format rich

# Dependency graph visualization
batuta graph --format ascii --layers all

# Single component analysis
batuta diagnose --component trueno

# Anomaly detection
batuta diagnose --anomalies --threshold 0.6

# Upgrade impact analysis
batuta upgrade-check trueno 0.7.0 0.8.0

# Error correlation
batuta correlate-error --file error.log

# Quality trend forecast
batuta forecast --horizon 7d

# Export to JSON
batuta diagnose --format json --output stack-health.json

# Watch mode (continuous monitoring)
batuta diagnose --watch --interval 60s
```

### 8.2 Configuration File

```toml
# .batuta-diagnostics.toml

[stack]
root_path = "~/src"
components = [
    "trueno", "trueno-db", "trueno-graph", "trueno-rag",
    "aprender", "realizar", "entrenar",
    "depyler", "bashrs", "decy",
    "pmat", "certeza", "verificar",
    "batuta", "renacer", "repartir",
]

[thresholds]
# Health status thresholds
green_min_score = 85.0
yellow_min_score = 70.0
# Below 70 is red

# Anomaly detection
anomaly_threshold = 0.65

# Coverage requirements
min_coverage = 95.0
min_mutation_score = 80.0

[graph]
# PageRank parameters
damping_factor = 0.85
max_iterations = 100

# Louvain parameters
resolution = 1.0

[ml]
# Isolation Forest
n_estimators = 100
contamination = 0.1

# Random Forest
n_trees = 50
max_depth = 10

[reporting]
format = "rich"
include_sparklines = true
include_recommendations = true
max_anomalies = 10

[alerts]
enabled = true
slack_webhook = "${SLACK_WEBHOOK}"
email = ["team@example.com"]
```

---

## 9. Integration with Stack Components

### 9.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INTEGRATION FLOW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                        │
│  │ Cargo.toml  │ ──parse──▶ ┌──────────────┐                           │
│  │ Cargo.lock  │            │   Dependency │                           │
│  └─────────────┘            │     Graph    │                           │
│                             └──────┬───────┘                           │
│  ┌─────────────┐                   │                                   │
│  │    pmat     │ ──score──▶        │                                   │
│  │ demo-score  │            ┌──────▼───────┐                           │
│  └─────────────┘            │    Stack     │──analyze──▶ ┌───────────┐ │
│                             │   Knowledge  │             │   ASCII   │ │
│  ┌─────────────┐            │    Graph     │             │  Report   │ │
│  │  certeza    │ ──gates──▶ └──────┬───────┘             └───────────┘ │
│  │   checks    │                   │                                   │
│  └─────────────┘                   │                                   │
│                             ┌──────▼───────┐                           │
│  ┌─────────────┐            │     ML       │                           │
│  │  renacer    │ ──trace──▶ │   Analytics  │──predict──▶ ┌───────────┐ │
│  │   traces    │            │    Engine    │             │  Insights │ │
│  └─────────────┘            └──────────────┘             └───────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Component APIs

| Component | API | Purpose |
|-----------|-----|---------|
| **trueno-graph** | `CsrGraph`, `pagerank()`, `bfs()` | Graph construction and metrics |
| **aprender** | `RandomForest`, `KMeans`, `IsolationForest` | ML-driven analysis |
| **pmat** | `demo_score()`, `tdg_analyze()` | Quality scoring |
| **certeza** | `check_coverage()`, `mutation_score()` | Quality gates |
| **renacer** | `Trace::parse()`, `syscall_stats()` | Runtime behavior |

---

## 10. Implementation Phases

### Phase 1: Foundation (2 weeks)
- [ ] Stack graph construction from Cargo manifests
- [ ] Basic ASCII graph renderer
- [ ] Health status aggregation from pmat scores
- [ ] CLI skeleton with `batuta diagnose`

### Phase 2: Graph Analytics (2 weeks)
- [ ] trueno-graph integration for PageRank/centrality
- [ ] Community detection with Louvain
- [ ] Critical path analysis
- [ ] Dependency depth computation

### Phase 3: ML Insights (3 weeks)
- [ ] Feature extraction pipeline
- [ ] Isolation Forest anomaly detection
- [ ] Upgrade impact prediction (Random Forest)
- [ ] Quality trend forecasting

### Phase 4: Error Correlation (2 weeks)
- [ ] Cross-component error linking
- [ ] renacer trace integration
- [ ] Five-whys root cause analysis
- [ ] Error category taxonomy

### Phase 5: Rich Reporting (2 weeks)
- [ ] Full ASCII dashboard with owo-colors
- [ ] Spark-line trend charts
- [ ] JSON/Markdown export
- [ ] Watch mode for continuous monitoring

### Phase 6: Alerts & Automation (1 week)
- [ ] Slack/email alert integration
- [ ] Threshold-based notifications
- [ ] Scheduled health checks

### Phase 7: Advanced Integrations (Future)

Specific external system integrations for enterprise adoption:

- **Incident Management**:
  - PagerDuty: Auto-create incidents from red-status components
  - Jira: Generate tickets for anomalies with root cause details
  - Linear: Create issues linked to affected code paths

- **Cloud Provider APIs**:
  - AWS CloudWatch: Export metrics for dashboard embedding
  - GCP Monitoring: Push health signals to Stackdriver
  - Datadog: Custom metrics and traces integration

- **CI/CD Systems**:
  - GitHub Actions: Quality gate checks on PRs
  - GitLab CI: Pipeline stage for stack health verification
  - Buildkite: Pre-merge dependency impact analysis

- **Communication**:
  - Slack: Rich block-kit formatted alerts with action buttons
  - Discord: Webhook integration for team channels
  - Microsoft Teams: Adaptive card notifications

---

## 11. References

### Graph Analytics

1. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). "The PageRank Citation Ranking: Bringing Order to the Web." *Stanford InfoLab Technical Report*.

2. Blondel, V.D., Guillaume, J.L., Lambiotte, R., & Lefebvre, E. (2008). "Fast Unfolding of Communities in Large Networks." *Journal of Statistical Mechanics*, P10008.

3. Brandes, U. (2001). "A Faster Algorithm for Betweenness Centrality." *Journal of Mathematical Sociology*, 25(2), pp. 163-177.

### Anomaly Detection

4. Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). "Isolation Forest." *ICDM 2008*, pp. 413-422.

5. Chandola, V., Banerjee, A., & Kumar, V. (2009). "Anomaly Detection: A Survey." *ACM Computing Surveys*, 41(3), Article 15.

### Dependency Analysis

6. Bavota, G., Canfora, G., Di Penta, M., Oliveto, R., & Panichella, S. (2013). "The Evolution of Project Inter-dependencies in a Software Ecosystem." *ICSM 2013*, pp. 280-289.

7. Kikas, R., Gousios, G., Dumas, M., & Pfahl, D. (2017). "Structure and Evolution of Package Dependency Networks." *MSR 2017*, pp. 102-112.

### Software Visualization

8. Ball, T., & Eick, S.G. (1996). "Software Visualization in the Large." *IEEE Computer*, 29(4), pp. 33-43.

9. Caserta, P., & Zendra, O. (2011). "Visualization of the Static Aspects of Software: A Survey." *IEEE TVCG*, 17(7), pp. 913-933.

### Time Series & Forecasting

10. Hyndman, R.J., & Athanasopoulos, G. (2018). "Forecasting: Principles and Practice." *OTexts*. [Exponential smoothing]

---

## Appendix A: Metric Definitions

| Metric | Definition | Range | Good |
|--------|------------|-------|------|
| **Demo Score** | Normalized 110-point quality score | 0-100 | ≥85 |
| **Coverage** | Test line coverage percentage | 0-100% | ≥95% |
| **Mutation Score** | Percentage of killed mutants | 0-100% | ≥80% |
| **PageRank** | Graph centrality (relative importance) | 0-1 | Context-dependent |
| **Betweenness** | Fraction of shortest paths through node | 0-1 | Lower is better |
| **Clustering Coeff** | Neighbor interconnectedness | 0-1 | Higher is cohesive |
| **Anomaly Score** | Isolation Forest isolation depth | 0-1 | <0.65 is normal |

---

## Appendix B: Health Status Colors

| Status | Condition | Andon Meaning |
|--------|-----------|---------------|
| 🟢 **Green** | Demo score ≥85, all gates pass | Normal operation |
| 🟡 **Yellow** | Demo score 70-84 or minor issues | Attention needed |
| 🔴 **Red** | Demo score <70 or critical failure | Stop-the-line |
| ⚪ **Unknown** | Not yet analyzed | Pending assessment |

---

## Appendix C: Toyota Way Mapping

| Toyota Principle | Diagnostics Implementation |
|------------------|----------------------------|
| **Mieruka (Visual Control)** | ASCII dashboards make health visible at a glance |
| **Jidoka** | ML anomaly detection surfaces issues automatically |
| **Genchi Genbutsu** | Graph analysis based on actual dependency data |
| **Andon** | Red/Yellow/Green status with stop-the-line alerts |
| **Yokoten** | Cross-component insight sharing via knowledge graph |
| **Kaizen** | Trend forecasting enables continuous improvement |
| **Muda** | Identifies wasted effort from unnecessary dependencies |
| **Heijunka** | Upgrade recommendations level out maintenance work |

---

*Document generated by PAIML Engineering. For questions, contact the batuta maintainers.*
