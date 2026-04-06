# Sovereign AI Architecture Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: sovereign-ai-spec, stack-visualization-diagnostics-reporting

---

## 1. Formal Architecture

### Core Invariants

- **Memory safety**: `unsafe` isolated to backend implementations
- **Lifetime hierarchy**: `'tensor: 'db: 'model` enforced at compile time
- **GPU dispatch**: 5x PCIe rule (`compute > 5 * transfer`) per Gregg & Hazelwood (2011)
- **WASM performance**: 50-70% native baseline, 80-90% with SIMD128 (Haas et al., 2017)
- **FFI trade-off**: Safe Rust transpilation incurs 20-40% overhead vs raw `unsafe`

### Layer Definitions

```rust
// Layer 1: Compute (Trueno)
pub struct Tensor<'a, T> {
    data: &'a [T],
    backend: Backend,     // SIMD/GPU dispatch
}

// Layer 2: Persistence (Trueno-DB)
pub struct VectorDB<'mmap> {
    buffer: Mmap,
    index: HNSWIndex<'mmap>,
    compute: Arc<TruenoBackend>,  // Send + Sync
}

// Layer 3: ML (Aprender + Realizar)
pub struct Model<'w> {
    weights: &'w [Tensor<'w, f32>],
    config: ModelConfig,
}

// Layer 4: Observability (Renacer)
pub struct Tracer {
    events: Vec<SyscallEvent>,
    stats: SIMDStats,
}

// Layer 5: Migration (Decy, Depyler)
pub trait Transpiler {
    fn transpile(&self, src: &str) -> Result<String, TranspileError>;
}

// Layer 6: Agency (MCP Toolkit)
pub trait MCPServer: Send + Sync {
    async fn handle_tool_call(&self, tool: &str, args: Value) -> Result<Value>;
}

// Layer 7: Orchestration (Batuta)
pub struct Pipeline {
    stages: Vec<Stage>,
    validation: ValidationStrategy,
}
```

### Lifetime Constraints

Cross-layer invariant enforced at compile time:

```rust
fn inference_pipeline<'m, 'db, 't>(
    model: &Model<'m>,
    db: &VectorDB<'db>,
    query: Tensor<'t, f32>,
) -> Result<Output>
where
    't: 'db,   // Query tensor outlives DB access
    'db: 'm,   // DB outlives model borrows
{ }
```

### Memory Model

**Backend dispatch:**

```rust
pub enum Backend {
    Scalar,
    SIMD { features: CpuFeatures },
    GPU {
        context: Arc<GpuContext>,
        sync: SyncMode,  // Explicit/Implicit
    },
}
```

**PCIe transfer model:**
- Host -> Device: `wgpu::Buffer::write()` with staging buffer
- Device -> Host: `wgpu::Buffer::map_async()` with callback
- Overhead: 14-55ms empirical (Trueno benchmarks)
- Rule: dispatch to GPU only when `compute_time > 5 * transfer_time`

### Zero-Copy Semantics

Trueno-DB mmap safety:

```rust
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
    _lifetime: PhantomData<&'static ()>,  // 'static for GPU
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}  // Requires external synchronization
```

---

## 2. Component Specifications (Empirical)

### Trueno (Compute)

| Metric | Value |
|--------|-------|
| Tests | 759 |
| Coverage | 100% |
| TDG | 96.1/100 |

**SIMD benchmark results:**
- Dot product: 182% faster (AVX2 vs scalar)
- Matmul 512x512: 7x faster (GPU vs scalar)
- Element-wise add: 3-10% faster (memory-bound)

### Trueno-DB (Vector Database)

| Metric | Value |
|--------|-------|
| Property tests | 1,100 |
| Coverage | 95.58% |

HNSW index: O(log n) search with beam search, `ef_construction=200`, `m=16`.

### Aprender (ML)

| Metric | Value |
|--------|-------|
| Tests | 184 |
| Coverage | ~97% |
| TDG | 93.3/100 |

Algorithms: Linear/logistic regression (Cholesky), RandomForest (Rayon parallel), KMeans, PCA, DBSCAN.

---

## 3. Stack Visualization and Diagnostics

### Problem

Sovereign AI Stack operators face fragmented visibility across 20+ components:

| Signal Source | Provides | Gap |
|---------------|----------|-----|
| `cargo tree` | Dependency listing | No analysis |
| `pmat demo-score` | Project quality | Per-project only |
| `certeza` | Quality gates | Per-project only |
| `renacer` | Syscall traces | Per-execution only |
| `cargo audit` | Security advisories | Per-project only |

### Solution: Unified Stack Intelligence

| Capability | Description | Implementation |
|------------|-------------|----------------|
| Dependency Graph | Interactive visualization | trueno-graph + ASCII renderer |
| Health Dashboard | Aggregate quality scores | pmat + certeza synthesis |
| Anomaly Detection | ML-driven pattern identification | aprender clustering + isolation forest |
| Error Correlation | Link errors to root causes | renacer traces + graph analysis |
| Upgrade Advisor | Impact analysis for upgrades | PageRank + breaking change detection |
| Performance Insights | Cross-stack bottleneck identification | trueno profiling + trace correlation |

### Diagnostics Architecture

```
Signal Sources                     Synthesis                    Output
--------------                     ---------                    ------
cargo tree        --+
pmat/certeza      --+--> trueno-graph --> ML Anomaly --> Health Dashboard
renacer traces    --+    (knowledge      Detection      Upgrade Advisor
cargo audit       --+     graph)          (aprender)    Error Reports
CI/CD logs        --+                                   Performance Insights
```

### Design Philosophy (Toyota Production System)

| Principle | Application |
|-----------|-------------|
| **Mieruka** (Visual Control) | Rich ASCII visualizations make health immediately visible |
| **Jidoka** (Autonomation) | ML surfaces anomalies; humans approve remediation |
| **Genchi Genbutsu** (Go and see) | Evidence-based from actual dependency graphs |
| **Andon** (Stop-the-line) | Automatic alerts on critical dependency degradation |
| **Yokoten** (Horizontal deployment) | Share insights across stack via knowledge graph |

### Anomaly Detection Pipeline

```
1. Collect signals (dependency graph, quality scores, traces)
2. Feature extraction (graph metrics, trend analysis)
3. Anomaly detection (Isolation Forest via aprender)
4. Root cause analysis (graph traversal via trueno-graph)
5. Remediation suggestions (ranked by impact)
6. Human approval (Jidoka principle)
```

### CLI Commands

```bash
# Dependency graph visualization
batuta stack graph                    # ASCII dependency graph
batuta stack graph --format dot       # GraphViz DOT output
batuta stack graph --focus trueno     # Focus on one component

# Health dashboard
batuta stack dashboard                # TUI health overview
batuta stack dashboard --json         # Machine-readable

# Diagnostics
batuta stack diagnose                 # Full diagnostic run
batuta stack diagnose --component realizar  # Single component
batuta stack diagnose --anomalies     # ML anomaly detection only

# Upgrade advisor
batuta stack upgrade-check            # Check for available upgrades
batuta stack upgrade-check --impact   # With breaking change analysis
```

---

## 4. Data Sovereignty Architecture

### Privacy Tier Enforcement

| Tier | Network Access | Data Storage | Inference | Audit |
|------|---------------|-------------|-----------|-------|
| **Sovereign** | Local only | On-premise | realizar local | Full syscall trace |
| **Private** | VPC/dedicated | Encrypted at rest | Dedicated endpoints | API-level logging |
| **Standard** | Any | Provider-managed | Any backend | Basic logging |

### Compliance Considerations

| Regulation | Stack Support |
|-----------|---------------|
| EU AI Act | Sovereign tier + privacy tiers + audit trail |
| GDPR | Data residency enforcement + PII detection |
| HIPAA | Encryption at rest + audit logging |
| SOC 2 | Renacer trace correlation + quality gates |

### Supply Chain Security

- All PAIML crates published to crates.io with Ed25519 signatures (pacha)
- `cargo audit` integrated into stack health checks
- Dependency graph analysis for transitive vulnerability detection
- WASM compilation eliminates native code supply chain risks for edge deployment
