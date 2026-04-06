# Real-Time Audit Log & Explainability for APR Format Models

> **Version:** 1.0.0
> **Status:** Draft Specification
> **Author:** PAIML Team
> **Toyota Way Principle:** 現地現物 (Genchi Genbutsu) - Go and see for yourself;
decisions must be traceable to ground truth

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Design Philosophy](#3-design-philosophy)
4. [Architecture Overview](#4-architecture-overview)
5. [Trueno Backend Integration](#5-trueno-backend-integration)
6. [APR Model Explainability](#6-apr-model-explainability)
7. [Real-Time Decision Tracing](#7-real-time-decision-tracing)
8. [Deep Audit Logging](#8-deep-audit-logging)
9. [Safety-Critical Applications](#9-safety-critical-applications)
10. [API Specification](#10-api-specification)
11. [Data Schema](#11-data-schema)
12. [Performance Budgets](#12-performance-budgets)
13. [Toyota Way Integration](#13-toyota-way-integration)
14. [Academic Foundation](#14-academic-foundation)
15. [Implementation Roadmap](#15-implementation-roadmap)
16. [Acceptance Criteria](#16-acceptance-criteria)

---

## 1. Executive Summary

This specification defines a unified system for **real-time explainability** and **deep audit logging** of machine
learning decisions made by APR format models (aprender). The system operates across all trueno compute backends
(Scalar, SIMD, GPU, WASM) with deterministic, reproducible decision traces.

### Key Capabilities

| Capability | Use Case | Latency Budget | Storage |
|------------|----------|----------------|---------|
| **Real-Time Trace** | Video games, drones | <1ms | Ring buffer (64 entries) |
| **Deep Audit Log** | Self-driving cars, medical | <10ms | Append-only with hash chain |
| **Forensic Replay** | Incident investigation | Offline | Full reconstruction |

### Target Applications

1. **Video Games (jugar)**: AI opponent decisions visible in debug overlay
2. **Autonomous Vehicles**: Every steering/braking decision logged with mathematical proof
3. **Drones**: Real-time obstacle avoidance with decision trace
4. **Medical Diagnosis**: Explainable predictions with feature attribution
5. **Financial Trading**: Regulatory-compliant decision audit

---

## 2. Problem Statement

### The Accountability Gap

Modern ML systems make millions of decisions per second, yet most operate as black boxes:

```
Input → [BLACK BOX] → Output → ??? → Accident
                                ↓
                         "Why did it decide that?"
```

**Cloudflare Incident (2025-11-18)**: A single `unwrap()` panic in production caused a 3+ hour global outage. The root
cause took hours to identify because decision paths were not logged.

**Tesla Autopilot Investigations**: NHTSA investigations require reconstruction of every decision made in the 30 seconds
before an incident—data that often doesn't exist in sufficient detail.

### Current State in PAIML Stack

```
aprender::predict() → f32 → lost
                       ↓
              No trace of HOW or WHY
```

### Required State

```
aprender::predict_explained() → (f32, DecisionTrace)
                                        ↓
                               Auditable, reproducible, debuggable
```

---

## 3. Design Philosophy

### Toyota Way Principles Applied

| Principle | Japanese | Application |
|-----------|----------|-------------|
| **Genchi Genbutsu** | 現地現物 | Every decision traceable to input data |
| **Jidoka** | 自働化 | Automatic quality checks on predictions |
| **Andon** | 行燈 | Immediate alerts on anomalous decisions |
| **Mieruka** | 見える化 | Visual decision path for debugging |
| **Poka-Yoke** | ポカヨケ | Type system prevents untraceable decisions |
| **Kaizen** | 改善 | Continuous improvement via decision analysis |
| **Heijunka** | 平準化 | Consistent latency across backends |

### Core Design Constraints

1. **Zero-Cost When Disabled**: Feature-gated, compiles out completely
2. **Deterministic Replay**: Same input + same model = same trace
3. **Backend Agnostic**: Works on Scalar, SIMD, GPU, WASM
4. **No Heap Allocation in Hot Path**: Stack-allocated traces for real-time
5. **Tamper-Evident**: Hash chain for safety-critical logs

---

## 4. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ENTRENAR INFERENCE MONITOR                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   aprender  │───▶│  Explainer  │───▶│  Collector  │───▶│   Storage   │  │
│  │   Model     │    │   Trait     │    │ (strategy)  │    │  (backend)  │  │
│  │  (.apr)     │    │             │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                  │                  │                  │          │
│         │           ┌──────┴──────┐    ┌─────┴─────┐     ┌──────┴──────┐   │
│         │           │ LinearPath  │    │   Ring    │     │   Memory    │   │
│         │           │ TreePath    │    │  Stream   │     │   File      │   │
│         │           │ ForestPath  │    │ HashChain │     │   Network   │   │
│         │           │ KNNPath     │    └───────────┘     └─────────────┘   │
│         │           │ NeuralPath  │                                        │
│         │           └─────────────┘                                        │
│         │                                                                   │
│  ┌──────┴──────────────────────────────────────────────────────────────┐   │
│  │                         TRUENO BACKENDS                              │   │
│  ├──────────────┬──────────────┬──────────────┬─────────────────────────┤   │
│  │   Scalar     │    SIMD      │     GPU      │         WASM            │   │
│  │  (fallback)  │ (AVX2/NEON)  │   (wgpu)     │     (SIMD128)           │   │
│  │              │              │              │                         │   │
│  │  <1ms trace  │  <1ms trace  │  <5ms trace  │     <1ms trace          │   │
│  │  per sample  │  per sample  │  per batch   │     per sample          │   │
│  └──────────────┴──────────────┴──────────────┴─────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         ANDON SYSTEM                                  │   │
│  │  • Confidence < threshold → Alert                                    │   │
│  │  • Decision drift detected → Alert                                   │   │
│  │  • Latency budget exceeded → Alert                                   │   │
│  │  • Anomalous input detected → Alert                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Model receives input** → trueno tensor (any backend)
2. **Explainer intercepts** → captures pre-decision state
3. **Model computes** → using optimal backend (SIMD/GPU/WASM)
4. **Explainer captures** → decision path, intermediate values
5. **Collector records** → strategy-dependent (ring/stream/chain)
6. **Andon checks** → fires alerts if thresholds breached
7. **Storage persists** → format-dependent (memory/file/network)

---

## 5. Trueno Backend Integration

### Backend-Specific Trace Capture

Each trueno backend has different characteristics for trace capture:

| Backend | Trace Capture | Overhead | Best For |
|---------|--------------|----------|----------|
| **Scalar** | Per-operation | <0.1% | Debugging, small models |
| **SIMD** | Per-vector-op | <0.5% | Real-time games, drones |
| **GPU** | Per-batch | <2% | Large batch inference |
| **WASM** | Per-operation | <1% | Browser games, edge |

### SIMD-Accelerated Trace Aggregation

Decision traces themselves use trueno SIMD for aggregation:

```rust
/// Aggregate trace statistics using SIMD
pub fn aggregate_traces(traces: &[DecisionTrace]) -> TraceStatistics {
    // Feature contributions across traces
    let contributions: Vec<f32> = traces
        .iter()
        .flat_map(|t| t.feature_contributions())
        .collect();

    // SIMD-accelerated statistics
    let contrib_vec = trueno::Vector::from_slice(&contributions);

    TraceStatistics {
        mean_contribution: contrib_vec.mean().unwrap_or(0.0),
        max_contribution: contrib_vec.max().unwrap_or(0.0),
        contribution_variance: contrib_vec.variance().unwrap_or(0.0),
        // ...
    }
}
```

### GPU Batch Trace Pattern

For GPU inference, traces are captured per-batch to avoid PCIe round-trips:

```rust
/// GPU-optimized batch trace
pub struct BatchTrace {
    /// Batch of input hashes (computed on GPU)
    pub input_hashes: Vec<u64>,
    /// Per-sample decision paths (sparse representation)
    pub decision_paths: Vec<CompactPath>,
    /// Batch-level statistics (computed on GPU)
    pub batch_stats: BatchStatistics,
    /// Single timestamp for entire batch
    pub timestamp_ns: u64,
}
```

### WASM Real-Time Pattern

For browser/edge deployment, traces use fixed-size stack allocation:

```rust
/// Stack-allocated trace for WASM (no heap allocation)
#[repr(C)]
pub struct WasmTrace {
    pub timestamp_ns: u64,
    pub input_hash: u64,
    pub output: f32,
    pub confidence: f32,
    /// Fixed-size path (max 16 decision points)
    pub path: [DecisionPoint; 16],
    pub path_len: u8,
}

// Compile-time size verification
const _: () = assert!(core::mem::size_of::<WasmTrace>() <= 256);
```

---

## 6. APR Model Explainability

### The Explainable Trait (aprender)

Each APR model type implements a minimal explainability trait:

```rust
// aprender/src/explain.rs

/// Trait for models that can explain their decisions
pub trait Explainable {
    /// Model-specific decision path type
    type Path: DecisionPath;

    /// Predict with full decision trace
    fn predict_explained(&self, x: &Matrix<f32>) -> (Vector<f32>, Vec<Self::Path>);

    /// Single-sample explanation (for streaming)
    fn explain_one(&self, sample: &[f32]) -> Self::Path;
}

/// Common interface for all decision paths
pub trait DecisionPath: Clone + Send + Sync {
    /// Human-readable explanation
    fn explain(&self) -> String;

    /// Feature importance scores
    fn feature_contributions(&self) -> &[f32];

    /// Confidence in this decision (0.0 - 1.0)
    fn confidence(&self) -> f32;

    /// Compact binary representation
    fn to_bytes(&self) -> Vec<u8>;
}
```

### Model-Specific Paths

#### Linear Models

```rust
/// Decision path for linear regression/logistic regression
#[derive(Clone, Debug)]
pub struct LinearPath {
    /// Per-feature contributions: coefficient[i] * input[i]
    pub contributions: Vec<f32>,
    /// Bias term contribution
    pub intercept: f32,
    /// Raw prediction before activation
    pub logit: f32,
    /// Final prediction
    pub prediction: f32,
    /// For classification: probability
    pub probability: Option<f32>,
}

impl LinearPath {
    pub fn top_features(&self, k: usize) -> Vec<(usize, f32)> {
        // Return k features with highest absolute contribution
    }
}
```

**Example Explanation**:
```
Prediction: 0.87 (class=1, confidence=87%)
Top contributing features:
  - feature[3] (income): +0.42 (coefficient=0.12, value=3.5)
  - feature[7] (age): +0.28 (coefficient=0.04, value=7.0)
  - feature[1] (education): -0.15 (coefficient=-0.05, value=3.0)
Intercept contribution: +0.32
```

#### Decision Trees

```rust
/// Decision path for tree-based models
#[derive(Clone, Debug)]
pub struct TreePath {
    /// Sequence of splits taken
    pub splits: Vec<TreeSplit>,
    /// Leaf node statistics
    pub leaf: LeafInfo,
    /// Gini impurity at each node
    pub gini_path: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct TreeSplit {
    /// Feature index used for split
    pub feature_idx: usize,
    /// Threshold value
    pub threshold: f32,
    /// Direction taken (true = left, false = right)
    pub went_left: bool,
    /// Samples in node before split
    pub n_samples: usize,
}

#[derive(Clone, Debug)]
pub struct LeafInfo {
    /// Predicted class or value
    pub prediction: f32,
    /// Samples in training that reached this leaf
    pub n_samples: usize,
    /// Class distribution (for classification)
    pub class_distribution: Option<Vec<f32>>,
}
```

**Example Explanation**:
```
Decision Path (depth=4):
  Node 0: age <= 35.0? YES (went left, n=1000)
  Node 1: income <= 50000.0? NO (went right, n=600)
  Node 3: credit_score <= 700.0? YES (went left, n=250)
  Node 7: LEAF → class=1 (approve), confidence=0.92
          Training samples: 230/250 were class=1
```

#### Random Forests

```rust
/// Decision path for ensemble models
#[derive(Clone, Debug)]
pub struct ForestPath {
    /// Individual tree paths
    pub tree_paths: Vec<TreePath>,
    /// Per-tree predictions
    pub tree_predictions: Vec<f32>,
    /// Aggregated prediction
    pub ensemble_prediction: f32,
    /// Agreement ratio among trees
    pub tree_agreement: f32,
    /// Feature importance (averaged across trees)
    pub feature_importance: Vec<f32>,
}
```

#### K-Nearest Neighbors

```rust
/// Decision path for KNN
#[derive(Clone, Debug)]
pub struct KNNPath {
    /// Indices of k nearest neighbors
    pub neighbor_indices: Vec<usize>,
    /// Distances to neighbors
    pub distances: Vec<f32>,
    /// Labels of neighbors
    pub neighbor_labels: Vec<usize>,
    /// Vote distribution
    pub votes: Vec<(usize, usize)>,  // (class, count)
    /// Weighted vote (if distance-weighted)
    pub weighted_votes: Option<Vec<f32>>,
}
```

#### Neural Networks

```rust
/// Decision path for neural networks (gradient-based)
#[derive(Clone, Debug)]
pub struct NeuralPath {
    /// Input gradient (saliency)
    pub input_gradient: Vec<f32>,
    /// Layer activations (optional, feature-gated)
    pub activations: Option<Vec<Vec<f32>>>,
    /// Attention weights (for transformers)
    pub attention_weights: Option<Vec<Vec<f32>>>,
    /// Integrated gradients attribution
    pub integrated_gradients: Option<Vec<f32>>,
}
```

### Counterfactual Explanations

Beyond explaining *what is*, counterfactuals answer *what could have been*—the minimal change to flip the decision:

```rust
/// Counterfactual explanation for a decision
#[derive(Clone, Debug)]
pub struct Counterfactual {
    /// Original input that produced the decision
    pub original_input: Vec<f32>,
    /// Original decision/class
    pub original_decision: usize,
    /// Modified input that would flip the decision
    pub counterfactual_input: Vec<f32>,
    /// The alternative decision
    pub alternative_decision: usize,
    /// Which features changed and by how much
    pub changes: Vec<FeatureChange>,
    /// L1 distance (sparsity of changes)
    pub sparsity: f32,
    /// L2 distance (magnitude of changes)
    pub distance: f32,
}

#[derive(Clone, Debug)]
pub struct FeatureChange {
    pub feature_idx: usize,
    pub feature_name: Option<String>,
    pub original_value: f32,
    pub counterfactual_value: f32,
    pub delta: f32,
}

impl Counterfactual {
    /// Generate natural language explanation
    pub fn explain(&self) -> String {
        // "The loan would have been APPROVED if:
        //  - income increased from $45,000 to $52,000 (+$7,000)
        //  - debt_ratio decreased from 0.42 to 0.35 (-0.07)"
    }
}
```

**Use Case**: When an AV's perception model misclassifies a cat as "unknown object," the counterfactual reveals:
"Classification would have been 'animal' if pixel region [x1,y1,x2,y2] brightness increased by 12%"—indicating a
lighting sensitivity issue.

### The Gray Box Architecture

To resolve the tension between deep learning performance and regulatory transparency, we adopt a **Gray Box**
architecture that wraps high-performance neural networks in mathematically rigorous explainability layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GRAY BOX ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    BLACK BOX (High Performance)                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │   │
│  │  │   Trueno    │  │  LogBERT    │  │   CNN/RNN   │             │   │
│  │  │    SIMD     │  │ Transformer │  │   Layers    │             │   │
│  │  │   Backend   │  │             │  │             │             │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                 TRANSPARENCY LAYER (Audit-Ready)                 │   │
│  │                                                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │   │
│  │  │   SHAP   │  │   LIME   │  │ Counter- │  │ Decision │       │   │
│  │  │  Values  │  │  Local   │  │ factuals │  │  Trace   │       │   │
│  │  │          │  │  Approx  │  │          │  │          │       │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │   │
│  │                                                                  │   │
│  │  Output: JSON artifact with decision + explanation + provenance  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  INTEGRITY LAYER (Tamper-Proof)                  │   │
│  │                                                                  │   │
│  │  Hash Chain → Merkle Root → Blockchain Anchor (LogStamping)     │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

This architecture satisfies:
- **GDPR Article 22**: Right to explanation for automated decisions
- **EU AI Act**: Risk-based audit requirements for high-risk AI
- **ISO 26262 ASIL-D**: Traceability of safety-critical automotive functions
- **FDA SaMD**: Software as Medical Device decision logging

---

## 7. Real-Time Decision Tracing

### Use Case: Video Games (jugar)

Games require sub-millisecond decision tracing without frame drops:

```rust
/// Real-time trace collector for games
pub struct GameTraceCollector {
    /// Ring buffer (fixed size, no allocation)
    buffer: RingBuffer<DecisionTrace, 64>,
    /// Debug overlay enabled
    debug_visible: bool,
    /// Last N decisions for overlay
    overlay_traces: ArrayVec<TraceOverlay, 8>,
}

impl GameTraceCollector {
    /// Record a decision (O(1), no allocation)
    #[inline(always)]
    pub fn record(&mut self, trace: DecisionTrace) {
        self.buffer.push(trace);

        if self.debug_visible {
            self.overlay_traces.push(TraceOverlay::from(&trace));
        }
    }

    /// Get traces for debug overlay
    pub fn overlay_data(&self) -> &[TraceOverlay] {
        &self.overlay_traces
    }
}
```

### Debug Overlay Example (jugar-ai)

```
┌─────────────────────────────────────────┐
│ AI Decision: CHASE_PLAYER               │
├─────────────────────────────────────────┤
│ Confidence: 87%                         │
│                                         │
│ Decision Path:                          │
│  ├─ player_visible: YES                 │
│  ├─ player_distance: 45.2 < 100.0       │
│  ├─ health: 0.75 > 0.3                  │
│  └─ ammo: 12 > 0                        │
│                                         │
│ Alternatives Considered:                │
│  • TAKE_COVER (13%)                     │
│  • RELOAD (0%)                          │
│                                         │
│ Frame Time: 0.3ms                       │
└─────────────────────────────────────────┘
```

### Use Case: Drones

Drones require deterministic, reproducible decision traces:

```rust
/// Drone decision trace with sensor fusion
pub struct DroneTrace {
    /// Base decision trace
    pub base: DecisionTrace,
    /// Sensor readings at decision time
    pub sensors: SensorSnapshot,
    /// Control output
    pub control: ControlOutput,
    /// Safety margins
    pub safety: SafetyMargins,
}

#[derive(Clone, Debug)]
pub struct SensorSnapshot {
    pub timestamp_ns: u64,
    pub imu: ImuReading,
    pub gps: GpsReading,
    pub lidar_points: u32,  // Count only, not full point cloud
    pub camera_frame_id: u64,
}

#[derive(Clone, Debug)]
pub struct ControlOutput {
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub throttle: f32,
}

#[derive(Clone, Debug)]
pub struct SafetyMargins {
    /// Distance to nearest obstacle
    pub min_obstacle_distance: f32,
    /// Remaining battery percentage
    pub battery_remaining: f32,
    /// Signal strength
    pub signal_strength: f32,
}
```

---

## 8. Deep Audit Logging

### Use Case: Self-Driving Cars

Autonomous vehicles require complete, tamper-evident audit logs for incident investigation:

```rust
/// Hash-chained audit log for safety-critical systems
pub struct SafetyAuditLog {
    /// Append-only storage backend
    storage: Box<dyn AuditStorage>,
    /// Hash of previous entry (chain integrity)
    prev_hash: [u8; 32],
    /// Sequence number (monotonic)
    sequence: u64,
    /// Signing key (optional HSM)
    signer: Option<Box<dyn Signer>>,
}

impl SafetyAuditLog {
    /// Record a decision with cryptographic integrity
    pub fn record(&mut self, trace: DecisionTrace) -> Result<AuditEntry> {
        let entry = AuditEntry {
            sequence: self.sequence,
            timestamp_ns: monotonic_ns(),
            prev_hash: self.prev_hash,
            trace,
            // SHA-256 of (sequence || prev_hash || trace)
            hash: [0u8; 32],  // Computed below
            signature: None,
        };

        // Compute hash
        let hash = self.compute_hash(&entry);
        let mut entry = entry;
        entry.hash = hash;

        // Sign if signer available
        if let Some(signer) = &self.signer {
            entry.signature = Some(signer.sign(&entry.hash)?);
        }

        // Persist
        self.storage.append(&entry)?;

        // Update chain state
        self.prev_hash = hash;
        self.sequence += 1;

        Ok(entry)
    }

    /// Verify chain integrity
    pub fn verify_chain(&self) -> Result<ChainVerification> {
        // Verify each entry's hash chains to previous
        // Returns first broken link if any
    }
}
```

### Audit Entry Schema

```rust
/// Tamper-evident audit entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Monotonic sequence number
    pub sequence: u64,
    /// Nanosecond timestamp (monotonic clock)
    pub timestamp_ns: u64,
    /// Hash of previous entry (forms chain)
    pub prev_hash: [u8; 32],
    /// The decision trace
    pub trace: DecisionTrace,
    /// SHA-256(sequence || prev_hash || trace)
    pub hash: [u8; 32],
    /// Optional cryptographic signature
    pub signature: Option<Signature>,
}
```

### Incident Investigation: "The Cat Scenario"

When a self-driving car runs over a cat, investigators need:

1. **30-second pre-incident window**: All decisions with full traces
2. **Sensor fusion state**: What the car "saw"
3. **Decision alternatives**: What other actions were considered
4. **Why the chosen action**: Mathematical proof of decision

```rust
/// Incident investigation query
pub struct IncidentQuery {
    /// Timestamp of incident (from vehicle logs)
    pub incident_time_ns: u64,
    /// Window before incident to analyze
    pub lookback_seconds: u32,
    /// Window after incident
    pub lookahead_seconds: u32,
}

/// Incident investigation result
pub struct IncidentReport {
    /// All decisions in window
    pub decisions: Vec<AuditEntry>,
    /// Chain verification result
    pub chain_integrity: ChainVerification,
    /// Reconstructed decision sequence
    pub decision_timeline: Vec<DecisionTimeline>,
    /// Anomalies detected
    pub anomalies: Vec<Anomaly>,
}

#[derive(Debug)]
pub struct DecisionTimeline {
    /// Time relative to incident (negative = before)
    pub relative_ms: i64,
    /// Decision made
    pub decision: String,
    /// Confidence
    pub confidence: f32,
    /// Alternatives and their scores
    pub alternatives: Vec<(String, f32)>,
    /// Why this decision won
    pub rationale: String,
}
```

### Example Incident Report

```
═══════════════════════════════════════════════════════════════════════════════
                         INCIDENT INVESTIGATION REPORT
                         Timestamp: 2025-03-15T14:32:47.123456789Z
                         Vehicle: AV-2847-XK
                         Location: 37.7749° N, 122.4194° W
═══════════════════════════════════════════════════════════════════════════════

CHAIN INTEGRITY: ✓ VERIFIED (847 entries, no breaks)

DECISION TIMELINE (T-0 = incident):
───────────────────────────────────────────────────────────────────────────────

T-2.340s │ DECISION: MAINTAIN_SPEED
         │ Confidence: 94%
         │ Detected Objects: pedestrian (left, 15m), parked_car (right, 8m)
         │ Alternatives: SLOW_DOWN (4%), CHANGE_LANE (2%)
         │ Rationale: Clear path ahead, objects outside trajectory

T-1.890s │ DECISION: MAINTAIN_SPEED
         │ Confidence: 91%
         │ Detected Objects: pedestrian (left, 12m), parked_car (right, 6m)
         │ Alternatives: SLOW_DOWN (7%), CHANGE_LANE (2%)
         │ Rationale: Clear path ahead, objects outside trajectory

T-0.847s │ DECISION: MAINTAIN_SPEED ⚠️ ANOMALY
         │ Confidence: 78% (DEGRADED)
         │ Detected Objects: pedestrian (left, 8m), parked_car (right, 4m)
         │ Alternatives: SLOW_DOWN (18%), EMERGENCY_BRAKE (4%)
         │ Rationale: Path predicted clear
         │
         │ ⚠️ ANOMALY: Small object detected at (2.3m, 0.1m)
         │    Classification: UNKNOWN (confidence: 23%)
         │    Size estimate: 0.3m x 0.2m
         │    Motion vector: crossing trajectory at 2.1 m/s

T-0.412s │ DECISION: SLOW_DOWN
         │ Confidence: 65%
         │ Detected Objects: UNKNOWN_SMALL (center, 1.8m)
         │ Alternatives: EMERGENCY_BRAKE (31%), SWERVE_LEFT (4%)
         │ Rationale: Unknown object in path, reducing speed

T-0.156s │ DECISION: EMERGENCY_BRAKE
         │ Confidence: 89%
         │ Detected Objects: ANIMAL (center, 0.9m) - NOW CLASSIFIED
         │ Alternatives: SWERVE_LEFT (8%), SWERVE_RIGHT (3%)
         │ Rationale: Animal detected in collision path
         │
         │ Physics: Braking distance at 35 km/h = 8.2m
         │          Distance to object: 0.9m
         │          TIME TO COLLISION: 0.093s
         │          COLLISION UNAVOIDABLE

T-0.000s │ IMPACT
         │ Speed at impact: 31 km/h (reduced from 35 km/h)
         │ Braking force applied: 0.8g for 0.156s

───────────────────────────────────────────────────────────────────────────────

ANALYSIS:
  • Object first appeared at T-0.847s as UNKNOWN (23% confidence)
  • Correctly reclassified as ANIMAL at T-0.156s
  • Classification delay: 691ms
  • Emergency brake initiated upon animal classification
  • Collision unavoidable due to late classification

ROOT CAUSE:
  • Small, fast-moving object (cat) initially misclassified
  • Training data underrepresented small animals in motion
  • RECOMMENDATION: Augment training data with small animal crossing scenarios

═══════════════════════════════════════════════════════════════════════════════
```

---

## 9. Safety-Critical Applications

### Compliance Requirements

| Standard | Requirement | How Satisfied |
|----------|-------------|---------------|
| ISO 26262 (Automotive) | Traceability of safety functions | Hash-chained audit log |
| DO-178C (Aviation) | Deterministic behavior | Reproducible traces |
| IEC 62304 (Medical) | Risk management records | Full decision history |
| FDA 21 CFR Part 11 | Electronic records integrity | Cryptographic signatures |

### Safety Integrity Levels

```rust
/// Safety Integrity Level configuration
#[derive(Clone, Copy, Debug)]
pub enum SafetyIntegrityLevel {
    /// SIL 0: No safety requirements (games, entertainment)
    /// - Ring buffer traces
    /// - Best-effort logging
    QM,

    /// SIL 1: Low safety requirements
    /// - Persistent traces
    /// - Hash verification
    SIL1,

    /// SIL 2: Medium safety requirements
    /// - Hash chain
    /// - Redundant storage
    SIL2,

    /// SIL 3: High safety requirements (automotive ASIL C)
    /// - Hash chain with signatures
    /// - Triple redundant storage
    /// - Hardware security module
    SIL3,

    /// SIL 4: Highest safety requirements (automotive ASIL D)
    /// - All SIL 3 requirements
    /// - Formal verification of trace system
    /// - Independent safety monitor
    SIL4,
}
```

### Andon Integration

The Andon system (from existing entrenar monitor) triggers alerts:

```rust
/// Safety-critical Andon rules
pub struct SafetyAndon {
    /// Alert if confidence below threshold
    pub min_confidence: f32,
    /// Alert if decision latency exceeds budget
    pub max_latency_ms: f32,
    /// Alert if unknown object classification
    pub alert_on_unknown: bool,
    /// Alert if safety margin violated
    pub min_safety_margin: f32,
    /// Stop-the-line conditions
    pub emergency_conditions: Vec<EmergencyCondition>,
}

#[derive(Clone, Debug)]
pub enum EmergencyCondition {
    /// Immediate stop if triggered
    CollisionImminent { time_to_collision_ms: f32 },
    /// Sensor failure
    SensorDegraded { sensor: String, quality: f32 },
    /// Chain integrity failure
    AuditChainBroken,
    /// Decision system failure
    DecisionTimeout { max_ms: f32 },
}
```

---

## 9.5. Provenance Graph Integration

Beyond sequential decision traces, safety-critical applications benefit from **Provenance Graphs** that capture the
causal relationships between system entities. This approach, pioneered by frameworks like Flash and Unicorn, enables
reconstruction of complex multi-step decision chains.

### Provenance Graph for ML Decisions

```rust
/// Provenance graph node types for ML inference
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProvenanceNode {
    /// Raw sensor/input data
    Input {
        source: String,
        timestamp_ns: u64,
        hash: u64,
    },
    /// Preprocessing transformation
    Transform {
        operation: String,
        input_refs: Vec<NodeId>,
    },
    /// Model inference
    Inference {
        model_id: String,
        model_version: String,
        decision_trace: DecisionTrace,
    },
    /// Post-processing or fusion
    Fusion {
        strategy: String,
        input_refs: Vec<NodeId>,
    },
    /// Final action/output
    Action {
        action_type: String,
        confidence: f32,
        alternatives: Vec<(String, f32)>,
    },
}

/// Edge in provenance graph (directed, acyclic)
#[derive(Clone, Debug)]
pub struct ProvenanceEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub relation: CausalRelation,
    pub timestamp_ns: u64,
}

#[derive(Clone, Copy, Debug)]
pub enum CausalRelation {
    /// Data flowed from source to sink
    DataFlow,
    /// Inference triggered by input
    Triggered,
    /// Decision influenced by
    Influenced,
    /// Action caused by decision
    Caused,
}
```

### Attack Path Reconstruction Pattern

For security-critical applications (e.g., detecting adversarial inputs to AV perception):

```rust
/// Reconstruct decision lineage for incident analysis
pub struct IncidentReconstructor {
    graph: ProvenanceGraph,
}

impl IncidentReconstructor {
    /// Trace backwards from incident to root causes
    pub fn reconstruct_attack_path(
        &self,
        incident_node: NodeId,
        max_depth: usize,
    ) -> AttackPath {
        // BFS backwards through causal edges
        // Similar to Flash's temporal-causal enforcement
    }

    /// Find anomalous nodes in the path
    pub fn identify_anomalies(&self, path: &AttackPath) -> Vec<Anomaly> {
        // Compare against baseline provenance patterns
        // Flag nodes where decision confidence dropped or
        // input hash doesn't match expected distribution
    }
}

/// Reconstructed attack/incident path
pub struct AttackPath {
    /// Nodes in causal order (root cause → incident)
    pub nodes: Vec<ProvenanceNode>,
    /// Edges connecting the path
    pub edges: Vec<ProvenanceEdge>,
    /// Time span of the incident
    pub duration_ns: u64,
    /// Identified anomaly points
    pub anomaly_indices: Vec<usize>,
}
```

### Streaming Graph Anomaly Detection

For real-time detection (inspired by StreamSpot):

```rust
/// Streaming provenance graph monitor
pub struct StreamingProvenanceMonitor {
    /// Graph shingle (local substructure) sketches
    sketches: HashMap<ShingleType, CountMinSketch>,
    /// Baseline distribution of shingles
    baseline: BaselineDistribution,
    /// Anomaly threshold
    threshold: f32,
}

impl StreamingProvenanceMonitor {
    /// Process new edge in O(1) time, constant memory
    pub fn observe_edge(&mut self, edge: &ProvenanceEdge) -> Option<Alert> {
        let shingle = self.extract_shingle(edge);
        self.sketches.get_mut(&shingle.type_).map(|s| s.increment(&shingle));

        // Check deviation from baseline
        let deviation = self.compute_deviation(&shingle);
        if deviation > self.threshold {
            Some(Alert::ProvenanceAnomaly {
                shingle,
                deviation,
                edge: edge.clone(),
            })
        } else {
            None
        }
    }
}
```

---

## 10. API Specification

### Core Traits (aprender)

```rust
// aprender/src/explain.rs

/// Marker trait for explainable models
pub trait Explainable {
    type Path: DecisionPath;

    fn predict_explained(&self, x: &Matrix<f32>) -> (Vector<f32>, Vec<Self::Path>);
    fn explain_one(&self, sample: &[f32]) -> Self::Path;
}

/// Common decision path interface
pub trait DecisionPath: Clone + Send + Sync + 'static {
    fn explain(&self) -> String;
    fn feature_contributions(&self) -> &[f32];
    fn confidence(&self) -> f32;
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Result<Self> where Self: Sized;
}
```

### Collector Strategies (entrenar)

```rust
// entrenar/src/monitor/inference/collector.rs

/// Strategy for collecting decision traces
pub trait TraceCollector: Send + Sync {
    fn record<P: DecisionPath>(&mut self, trace: DecisionTrace<P>);
    fn flush(&mut self) -> Result<()>;
    fn len(&self) -> usize;
}

/// Ring buffer collector (real-time, bounded memory)
pub struct RingCollector<P: DecisionPath, const N: usize> {
    buffer: [MaybeUninit<DecisionTrace<P>>; N],
    head: usize,
    count: usize,
}

/// Streaming collector (unbounded, write-through)
pub struct StreamCollector<P: DecisionPath, W: Write> {
    writer: W,
    format: TraceFormat,
    buffer: Vec<DecisionTrace<P>>,
    flush_threshold: usize,
}

/// Hash-chain collector (safety-critical)
pub struct HashChainCollector<P: DecisionPath, S: AuditStorage> {
    storage: S,
    prev_hash: [u8; 32],
    sequence: u64,
    signer: Option<Box<dyn Signer>>,
}
```

### Inference Monitor (entrenar)

```rust
// entrenar/src/monitor/inference/mod.rs

/// High-level inference monitor
pub struct InferenceMonitor<M, C>
where
    M: Explainable,
    C: TraceCollector,
{
    model: M,
    collector: C,
    andon: Option<AndonSystem>,
    latency_budget_ns: u64,
}

impl<M, C> InferenceMonitor<M, C>
where
    M: Explainable,
    C: TraceCollector,
{
    /// Predict with automatic tracing
    pub fn predict(&mut self, x: &Matrix<f32>) -> Vector<f32> {
        let start = Instant::now();

        let (output, paths) = self.model.predict_explained(x);

        let elapsed_ns = start.elapsed().as_nanos() as u64;

        for (i, path) in paths.into_iter().enumerate() {
            let trace = DecisionTrace {
                timestamp_ns: monotonic_ns(),
                sequence: self.collector.len() as u64,
                input_hash: hash_row(x, i),
                path,
                output: output[i],
                latency_ns: elapsed_ns,
            };

            self.collector.record(trace);

            // Andon check
            if let Some(andon) = &self.andon {
                andon.check(&trace);
            }
        }

        output
    }
}
```

---

## 11. Data Schema

### DecisionTrace (Core)

```rust
/// Universal decision trace structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionTrace<P: DecisionPath> {
    /// Monotonic nanosecond timestamp
    pub timestamp_ns: u64,
    /// Sequence number within session
    pub sequence: u64,
    /// FNV-1a hash of input features
    pub input_hash: u64,
    /// Model-specific decision path
    pub path: P,
    /// Final output value
    pub output: f32,
    /// Inference latency in nanoseconds
    pub latency_ns: u64,
}
```

### Binary Format (Compact)

For high-throughput logging, a compact binary format:

```
┌────────────────────────────────────────────────────────────┐
│ DecisionTrace Binary Format (v1)                           │
├────────────────────────────────────────────────────────────┤
│ Offset │ Size │ Field                                      │
├────────┼──────┼────────────────────────────────────────────┤
│ 0      │ 4    │ Magic: 0x41505254 ("APRT")                 │
│ 4      │ 1    │ Version: 0x01                              │
│ 5      │ 1    │ Path type (0=Linear, 1=Tree, 2=Forest...)  │
│ 6      │ 2    │ Reserved (alignment)                       │
│ 8      │ 8    │ timestamp_ns (u64 LE)                      │
│ 16     │ 8    │ sequence (u64 LE)                          │
│ 24     │ 8    │ input_hash (u64 LE)                        │
│ 32     │ 4    │ output (f32 LE)                            │
│ 36     │ 4    │ latency_ns (u32 LE, microsecond precision) │
│ 40     │ 4    │ path_length (u32 LE)                       │
│ 44     │ var  │ path_data (model-specific)                 │
└────────┴──────┴────────────────────────────────────────────┘
```

### JSON Format (Human-Readable)

```json
{
  "version": "1.0",
  "trace": {
    "timestamp_ns": 1710512347123456789,
    "sequence": 42,
    "input_hash": "0xdeadbeef12345678",
    "output": 0.87,
    "latency_ns": 234567,
    "path": {
      "type": "LinearPath",
      "contributions": [0.42, 0.28, -0.15, 0.08, 0.12],
      "intercept": 0.32,
      "prediction": 0.87,
      "top_features": [
        {"index": 0, "name": "income", "contribution": 0.42},
        {"index": 1, "name": "age", "contribution": 0.28}
      ]
    }
  }
}
```

---

## 12. Performance Budgets

### Latency Targets by Application

| Application | Trace Overhead | Total Latency | Memory |
|-------------|---------------|---------------|--------|
| Video Game (60 FPS) | <0.5ms | <16ms | 64KB ring |
| Drone (100 Hz control) | <1ms | <10ms | 256KB ring |
| Self-Driving (30 Hz) | <5ms | <33ms | Unbounded |
| Medical Diagnosis | <100ms | <1s | Unbounded |

### Benchmark Targets

```rust
/// Performance benchmarks for trace system
#[cfg(test)]
mod benchmarks {
    use criterion::{criterion_group, Criterion};

    fn bench_linear_trace(c: &mut Criterion) {
        // Target: <100ns per trace (ring buffer)
        c.bench_function("linear_trace_record", |b| {
            let mut collector = RingCollector::<LinearPath, 64>::new();
            let trace = DecisionTrace::mock_linear();

            b.iter(|| collector.record(trace.clone()))
        });
    }

    fn bench_tree_trace(c: &mut Criterion) {
        // Target: <500ns per trace (depth-10 tree)
        c.bench_function("tree_trace_record", |b| {
            let mut collector = RingCollector::<TreePath, 64>::new();
            let trace = DecisionTrace::mock_tree(10);

            b.iter(|| collector.record(trace.clone()))
        });
    }

    fn bench_hash_chain(c: &mut Criterion) {
        // Target: <10µs per entry (including SHA-256)
        c.bench_function("hash_chain_record", |b| {
            let mut collector = HashChainCollector::new_memory();
            let trace = DecisionTrace::mock_linear();

            b.iter(|| collector.record(trace.clone()))
        });
    }
}
```

---

## 13. Toyota Way Integration

### Visual Management (見える化 Mieruka)

Decision traces enable visual debugging:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DECISION STREAM MONITOR                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Decisions/sec: ████████████████████░░░░░░░░░░ 847/1000             │
│  Avg Latency:   ██████░░░░░░░░░░░░░░░░░░░░░░░░ 2.3ms               │
│  Confidence:    █████████████████████████████░ 94%                 │
│                                                                     │
│  Recent Decisions:                                                  │
│  ──────────────────────────────────────────────────────────────    │
│  14:32:47.123 │ APPROVE │ conf=0.94 │ income=+0.42, age=+0.28      │
│  14:32:47.156 │ DENY    │ conf=0.87 │ debt=-0.65, history=-0.23    │
│  14:32:47.189 │ APPROVE │ conf=0.91 │ income=+0.38, years=+0.31    │
│  14:32:47.222 │ APPROVE │ conf=0.78 │ income=+0.25, age=+0.22  ⚠️ │
│                                                     ↑ low conf     │
│  Anomalies: 1 (low confidence)                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Stop-the-Line (Andon 行燈)

Automatic alerting on decision anomalies:

```rust
/// Andon rules for inference monitoring
pub struct InferenceAndon {
    /// Alert levels
    pub yellow_threshold: f32,  // Warning
    pub red_threshold: f32,     // Stop-the-line

    /// Conditions
    pub rules: Vec<AndonRule>,
}

pub enum AndonRule {
    /// Confidence below threshold
    LowConfidence { threshold: f32, level: AlertLevel },

    /// Latency exceeded
    HighLatency { max_ms: f32, level: AlertLevel },

    /// Decision drift detected
    DriftDetected { detector: DriftDetector, level: AlertLevel },

    /// Unknown input pattern
    OutOfDistribution { threshold: f32, level: AlertLevel },

    /// Consecutive same decision (stuck)
    RepetitiveDecision { count: usize, level: AlertLevel },
}
```

### Continuous Improvement (改善 Kaizen)

Decision traces enable model improvement:

```rust
/// Analyze decision traces for improvement opportunities
pub struct KaizenAnalyzer {
    /// Identify low-confidence patterns
    pub fn find_uncertainty_clusters(&self, traces: &[DecisionTrace])
        -> Vec<UncertaintyCluster>;

    /// Identify feature importance drift
    pub fn detect_feature_drift(&self, traces: &[DecisionTrace])
        -> Vec<FeatureDrift>;

    /// Identify decision boundary changes
    pub fn analyze_decision_boundaries(&self, traces: &[DecisionTrace])
        -> BoundaryAnalysis;

    /// Generate retraining recommendations
    pub fn recommend_retraining(&self, analysis: &KaizenAnalysis)
        -> Vec<RetrainingRecommendation>;
}
```

---

## 14. Academic Foundation

This specification builds on peer-reviewed research in explainable AI, audit logging, log anomaly detection, provenance
analysis, and safety-critical systems. The citations are organized by domain to facilitate deeper exploration.

### Explainability Foundations (XAI)

1. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016). "Why Should I Trust You?" Explaining the Predictions of Any
   Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*,
   1135-1144. https://doi.org/10.1145/2939672.2939778
   - *Foundation for LIME (Local Interpretable Model-agnostic Explanations). Post-hoc explanation via local
     perturbation.*

2. **Lundberg, S. M., & Lee, S. I.** (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural
   Information Processing Systems*, 30, 4765-4774.
   - *SHAP values: Game-theoretic feature attribution with consistency guarantees.*

3. **Molnar, C.** (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable* (2nd ed.).
   <https://christophm.github.io/interpretable-ml-book/>
   - *Comprehensive survey of explainability methods including counterfactuals.*

4. **Arrieta, A. B., et al.** (2020). Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and
   challenges toward responsible AI. *Information Fusion*, 58, 82-115. https://doi.org/10.1016/j.inffus.2019.12.012
   - *Taxonomy of XAI approaches for regulatory compliance (GDPR, EU AI Act).*

5. **Guidotti, R., et al.** (2018). A Survey of Methods for Explaining Black Box Models. *ACM Computing Surveys*, 51(5),
   1-42. https://doi.org/10.1145/3236009
   - *Systematic review: model-agnostic vs model-specific explanation methods.*

### Decision Trees & Ensemble Explainability

6. **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5-32. https://doi.org/10.1023/A:1010933404324
   - *Original Random Forest paper with permutation-based feature importance.*

7. **Friedman, J. H.** (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*,
   29(5), 1189-1232.
   - *Gradient boosting with partial dependence plots for feature effects.*

8. **Palczewska, A., et al.** (2014). Interpreting Random Forest Classification Models Using a Feature Contribution
   Method. *Integration of Reusable Systems*, 193-218. https://doi.org/10.1007/978-3-319-04717-1_9
   - *Per-prediction feature contribution decomposition for ensemble models.*

### Neural Network Interpretability

9. **Simonyan, K., Vedaldi, A., & Zisserman, A.** (2014). Deep Inside Convolutional Networks: Visualising Image
   Classification Models and Saliency Maps. *ICLR Workshop*.
   - *Gradient-based saliency maps for input attribution.*

10. **Sundararajan, M., Taly, A., & Yan, Q.** (2017). Axiomatic Attribution for Deep Networks. *Proceedings of the 34th
    International Conference on Machine Learning*, 70, 3319-3328.
    - *Integrated Gradients: Axiomatically-grounded attribution satisfying sensitivity and implementation invariance.*

11. **Selvaraju, R. R., et al.** (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization. *Proceedings of the IEEE International Conference on Computer Vision*, 618-626.
    - *Gradient-weighted Class Activation Mapping for CNN interpretability.*

### Log Anomaly Detection & Sequence Modeling

12. **Du, M., Li, F., Zheng, G., & Srikumar, V.** (2017). DeepLog: Anomaly Detection and Diagnosis from System Logs
    through Deep Learning. *Proceedings of the ACM SIGSAC Conference on Computer and Communications Security*,
    1285-1298.
    - *Foundational LSTM-based log anomaly detection treating logs as natural language sequences.*

13. **Meng, W., et al.** (2019). LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in
    Unstructured Logs. *IJCAI*, 4739-4745.
    - *Template2Vec semantic embeddings for unified sequential and quantitative anomaly detection.*

14. **Guo, H., et al.** (2021). LogBERT: Log Anomaly Detection via BERT. *International Joint Conference on Neural
    Networks (IJCNN)*.
    - *Transformer-based self-supervised log analysis with masked log key prediction.*

15. **Yang, L., et al.** (2021). PLELog: Semi-supervised Log Anomaly Detection via Probabilistic Label Estimation.
    *ICSE*.
    - *Attention-based GRU with probabilistic label estimation for label-scarce environments.*

### Log Parsing & Semantic Representation

16. **He, P., et al.** (2017). Drain: An Online Log Parsing Approach with Fixed Depth Tree. *ICWS*.
    - *Streaming log parser using fixed-depth tree structure for real-time template extraction.*

17. **Nedelkoski, S., Bogatinovski, J., Acker, A., Cardoso, J., & Kao, O.** (2020). Self-Supervised Log Parsing. *ECML
    PKDD*.
    - *NuLog: Masked language modeling for joint parsing and representation learning.*

18. **Meng, W., et al.** (2020). Log2Vec: A Heterogeneous Graph Embedding Based Approach for Detecting Cyber Threats
    within Enterprise. *CCS*.
    - *Semantic-aware log embeddings handling out-of-vocabulary tokens.*

### Provenance Analysis & Attack Path Reconstruction

19. **Hassan, W. U., et al.** (2020). Tactical Provenance Analysis for Endpoint Detection and Response Systems. *IEEE
    S&P*.
    - *Flash: Provenance graph representation learning with temporal-causal enforcement for APT detection.*

20. **Han, X., et al.** (2020). Unicorn: Runtime Provenance-Based Detector for Advanced Persistent Threats. *NDSS*.
    - *Graph sketching for modeling long-running system execution and detecting slow APT campaigns.*

21. **Manzoor, E., Milajerdi, S. M., & Akoglu, L.** (2016). Fast Memory-efficient Anomaly Detection in Streaming
    Heterogeneous Graphs. *KDD*.
    - *StreamSpot: Constant-memory streaming graph anomaly detection via graph shingles.*

22. **Liu, F., et al.** (2018). ProTracer: Towards Practical Provenance Tracing by Alternating Between Logging and
    Tainting. *NDSS*.
    - *Alternating logging/tainting to reduce provenance graph storage overhead while maintaining forensic fidelity.*

### Safety-Critical Systems & Automotive

23. **Koopman, P., & Wagner, M.** (2016). Challenges in Autonomous Vehicle Testing and Validation. *SAE International
    Journal of Transportation Safety*, 4(1), 15-24. https://doi.org/10.4271/2016-01-0128
    - *AV testing requirements: edge cases, simulation, and operational design domains.*

24. **Shalev-Shwartz, S., Shammah, S., & Shashua, A.** (2017). On a Formal Model of Safe and Scalable Self-driving Cars.
    *arXiv preprint arXiv:1708.06374*.
    - *Responsibility-Sensitive Safety (RSS): Mathematical framework for blame-free AV decisions.*

25. **ISO 26262:2018** Road vehicles — Functional safety. *International Organization for Standardization*.
    - *ASIL levels (A-D) for automotive functional safety with traceability requirements.*

### Audit Logging, Integrity & Blockchain

26. **Buneman, P., Khanna, S., & Tan, W. C.** (2001). Why and Where: A Characterization of Data Provenance.
    *International Conference on Database Theory*, 316-330. https://doi.org/10.1007/3-540-44503-X_20
    - *Foundational theory distinguishing why-provenance from where-provenance.*

27. **Herschel, M., Diestelkämper, R., & Ben Lahmar, H.** (2017). A Survey on Provenance: What for? What form? What
    from? *The VLDB Journal*, 26(6), 881-906. https://doi.org/10.1007/s00778-017-0486-1
    - *Comprehensive provenance taxonomy: annotation, inversion, and lazy vs eager capture.*

28. **Nakamoto, S.** (2008). Bitcoin: A Peer-to-Peer Electronic Cash System.
    - *Hash chain and Merkle tree foundations for tamper-evident audit logs.*

29. **Sutton, A., & Samavi, R.** (2017). Blockchain Enabled Privacy Audit Logs. *International Semantic Web Conference*.
    - *LogStamping: Blockchain-anchored log integrity with IPFS storage.*

### Real-Time Systems & Scheduling

30. **Liu, C. L., & Layland, J. W.** (1973). Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment.
    *Journal of the ACM*, 20(1), 46-61. https://doi.org/10.1145/321738.321743
    - *Rate-monotonic scheduling theory for real-time latency guarantees.*

31. **Kopetz, H.** (2011). *Real-Time Systems: Design Principles for Distributed Embedded Applications* (2nd ed.).
    Springer.
    - *Determinism, fault tolerance, and temporal firewalls in safety-critical systems.*

### SIMD, GPU & Performance Optimization

32. **Fog, A.** (2021). *Optimizing Software in C++: An Optimization Guide for Windows, Linux and Mac Platforms*.
    <https://www.agner.org/optimize/>
    - *SIMD optimization: AVX2/AVX-512 vectorization, cache-aware algorithms.*

33. **Intel Corporation.** (2023). *Intel® 64 and IA-32 Architectures Optimization Reference Manual*.
    - *Microarchitecture-aware optimization for low-latency trace capture.*

### Drift Detection & Concept Shift

34. **Gama, J., et al.** (2014). A Survey on Concept Drift Adaptation. *ACM Computing Surveys*, 46(4), 1-37.
    <https://doi.org/10.1145/2523813>
    - *Drift detection methods: DDM, EDDM, ADWIN for evolving data streams.*

35. **Baena-García, M., et al.** (2006). Early Drift Detection Method. *Fourth International Workshop on Knowledge
    Discovery from Data Streams*.
    - *EDDM: Statistical approach for early warning of distribution shift.*

### Agentic AI & LLM Integration

36. **Chen, Q., et al.** (2024). Audit-LLM: A Multi-Agent Framework for AI-driven Log Analysis. *arXiv preprint*.
    - *Multi-agent LLM architecture (Decomposer, Tool Builder, Executor) for autonomous investigation.*

37. **Jiang, Z., et al.** (2023). LogSummary: Unsupervised Log Summarization for System Event Comprehension. *ASE*.
    - *Natural language summarization of log streams to combat alert fatigue.*

### Audit-Ready & Regulatory Frameworks

38. **Antunes, N., et al.** (2023). EARL: Explainable and Audit-Ready Logging for Clinical AI. *Journal of Biomedical
    Informatics*.
    - *Unified architecture: prediction + XAI explanation + provenance in cryptographically-secured audit trail.*

39. **European Commission.** (2024). EU Artificial Intelligence Act. *Official Journal of the European Union*.
    - *Legal requirements for AI explainability, risk classification, and audit trails.*

### Toyota Production System & Quality

40. **Liker, J. K.** (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*.
    McGraw-Hill.
    - *Genchi Genbutsu (go and see), Jidoka (automation with human oversight), Andon (visual alerting), Kaizen
      (continuous improvement).*

---

## 15. Implementation Roadmap

### Phase 1: Core Traits (Week 1-2)

**Deliverables:**
- [ ] `aprender::explain::Explainable` trait
- [ ] `aprender::explain::DecisionPath` trait
- [ ] `LinearPath`, `TreePath` implementations
- [ ] Unit tests with 95%+ coverage

**Ticket:** ENT-080

### Phase 2: Collectors (Week 3-4)

**Deliverables:**
- [ ] `RingCollector<P, N>` (stack-allocated)
- [ ] `StreamCollector<P, W>` (write-through)
- [ ] Binary and JSON serialization
- [ ] Benchmark suite (<100ns per trace)

**Ticket:** ENT-081

### Phase 3: Safety Collector (Week 5-6)

**Deliverables:**
- [ ] `HashChainCollector` with SHA-256
- [ ] Optional signature support
- [ ] Chain verification API
- [ ] Incident query API

**Ticket:** ENT-082

### Phase 4: InferenceMonitor (Week 7-8)

**Deliverables:**
- [ ] `InferenceMonitor<M, C>` wrapper
- [ ] Andon integration
- [ ] Latency tracking
- [ ] Integration with existing `entrenar::monitor`

**Ticket:** ENT-083

### Phase 5: Application Integration (Week 9-10)

**Deliverables:**
- [ ] jugar-ai integration example
- [ ] Safety-critical example (mock AV)
- [ ] Documentation and tutorials
- [ ] Performance validation

**Ticket:** ENT-084

---

## 16. Acceptance Criteria

### Functional Requirements

- [ ] All APR model types implement `Explainable`
- [ ] Ring collector has zero heap allocation in hot path
- [ ] Hash chain collector produces verifiable chains
- [ ] Traces are reproducible (same input → same trace)
- [ ] All trueno backends produce identical traces

### Performance Requirements

- [ ] Ring collector: <100ns per trace
- [ ] Stream collector: <1µs per trace
- [ ] Hash chain collector: <10µs per trace
- [ ] Memory overhead: <1KB per model instance

### Quality Requirements

- [ ] Test coverage: >95%
- [ ] Mutation score: >85%
- [ ] Zero `unwrap()` calls in production code
- [ ] All APIs documented with examples
- [ ] Property tests for serialization round-trips

### Safety Requirements

- [ ] Hash chain integrity verified on load
- [ ] Signature verification when enabled
- [ ] Andon alerts fire within 1ms of condition
- [ ] No data loss on process termination (flush guarantees)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **APR** | Aprender model format (.apr files) |
| **Andon** | Toyota visual management system for alerting |
| **Decision Path** | Model-specific explanation of how a decision was made |
| **Decision Trace** | Complete record of a single prediction with path |
| **Genchi Genbutsu** | "Go and see" - decisions traceable to ground truth |
| **Hash Chain** | Linked list where each entry contains hash of previous |
| **Jidoka** | Automation with human oversight |
| **Kaizen** | Continuous improvement |
| **Mieruka** | Visual management |
| **Poka-Yoke** | Error-proofing |
| **Ring Buffer** | Fixed-size circular buffer, overwrites oldest |
| **SIL** | Safety Integrity Level (IEC 61508) |
| **SHAP** | SHapley Additive exPlanations |
| **Trueno** | PAIML SIMD/GPU compute library |

---

## Appendix B: Example Configurations

### Video Game (jugar)

```toml
[inference_monitor]
collector = "ring"
ring_size = 64
debug_overlay = true
andon_enabled = false

[performance]
max_latency_ms = 1.0
trace_sampling_rate = 1.0  # 100% of decisions
```

### Autonomous Vehicle

```toml
[inference_monitor]
collector = "hash_chain"
storage = "file"
storage_path = "/var/log/av/decisions"
signature_enabled = true
hsm_device = "/dev/hsm0"

[safety]
integrity_level = "SIL3"
min_confidence = 0.7
max_latency_ms = 10.0
redundant_storage = true

[andon]
enabled = true
yellow_confidence = 0.8
red_confidence = 0.6
alert_endpoint = "http://safety-monitor:8080/alert"
```

### Medical Diagnosis

```toml
[inference_monitor]
collector = "stream"
storage = "database"
connection = "postgres://..."
retention_days = 2555  # 7 years (regulatory requirement)

[compliance]
standard = "IEC_62304"
audit_trail = true
user_attribution = true

[explainability]
include_feature_names = true
include_shap_values = true
confidence_intervals = true
```

---

*Document generated in accordance with Toyota Way principles. All decisions in this specification are traceable to
requirements and academic foundations.*
