# External Integrations Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: data-platforms-integration-spec-query, data-visualization-integration-query, content-creation-tooling-spec, manzana-apple-hardware-spec

---

## 1. Data Platform Integrations

### Platform Landscape

| Platform | Primary Use Case | Data Sovereignty | PAIML Integration |
|----------|-----------------|------------------|-------------------|
| **Databricks** | Unified Analytics | Configurable (VPC) | Delta Lake <-> Alimentar |
| **Snowflake** | Cloud Data Warehouse | Multi-cloud | Iceberg <-> Alimentar |
| **AWS** | Infrastructure + ML | Region-locked | S3/SageMaker <-> Stack |
| **HuggingFace** | Model Hub | Public/Enterprise | Hub <-> Pacha |

### CLI Interface

```bash
# View data platform ecosystem
batuta data tree
batuta data tree --integration
batuta data tree --platform databricks
batuta data tree --platform snowflake

# Export for tooling
batuta data tree --format json > platforms.json
```

### v1.1.0 Enhancements

| Feature | Toyota Principle | Description |
|---------|-----------------|-------------|
| Cost Andon Cord | Andon | Pre-flight cost estimation before data operations |
| Resumable Sync | Kaizen | Stateful checkpointing for interrupted transfers |
| Schema Drift Detection | Jidoka | Automatic detection of upstream schema changes |
| Adaptive Throttling | Heijunka | Rate limiting based on platform load |
| OS-Level Egress Filtering | Poka-Yoke | Prevent accidental data leakage at network level |
| Federation (Virtual Catalogs) | Genchi Genbutsu | Query across platforms without data movement |
| Information Flow Control | Jidoka | Data provenance tracking across platforms |

### Sovereign Tier Behavior

When `PrivacyTier::Sovereign` is active:
- All non-VPC endpoints blocked
- Data egress requires explicit opt-in
- Platform connections audit-logged
- Network-level egress filtering enforced

---

## 2. Data Visualization

### Framework Replacement Matrix

| Python Framework | PAIML Replacement | Migration Path |
|------------------|-------------------|----------------|
| **Gradio** | Presentar | Depyler transpilation |
| **Streamlit** | Presentar | Depyler transpilation |
| **Panel** | Trueno-Viz | Depyler transpilation |
| **Dash** | Presentar + Trueno-Viz | Depyler transpilation |
| **Matplotlib** | Trueno-Viz | Direct API mapping |
| **Plotly** | Trueno-Viz | Direct API mapping |
| **Bokeh** | Trueno-Viz | Direct API mapping |

**Core principle:** Python visualization frameworks are replaced by sovereign Rust alternatives. No Python runtime permitted in production.

### CLI Interface

```bash
batuta viz tree                     # Full ecosystem view
batuta viz tree --integration       # PAIML replacement mapping
batuta viz tree --framework gradio  # Filter by framework
batuta viz tree --format json       # JSON export
```

### Presentar Capabilities

| Feature | Description | Target |
|---------|-------------|--------|
| WASM-first | Browser-native rendering | WebGPU/Canvas2D |
| Terminal fallback | ASCII visualization | Any terminal |
| Interactive demos | Course content demos | mdBook integration |
| Chart types | Line, bar, scatter, heatmap | Trueno-Viz backend |

---

## 3. Content Creation Tooling

### Overview

Content creation system that operates as a **prompt emission engine** -- generates optimized prompts for conversational AI assistants, not AI-generated content directly.

### Content Type Taxonomy

| Type | Code | Output Format | Target Length |
|------|------|---------------|---------------|
| High-Level Outline | HLO | YAML/Markdown | 50-200 lines |
| Detailed Outline | DLO | YAML/Markdown | 200-1000 lines |
| Book Chapter | BCH | Markdown (mdBook) | 2000-8000 words |
| Blog Post | BLP | Markdown + TOML | 500-3000 words |
| Presentar Demo | PDM | HTML + YAML config | N/A |

### Content Hierarchy

```
High-Level Outline (HLO)
    +-- Detailed Outline (DLO)
            +-- Book Chapter (BCH)
            +-- Blog Post (BLP)
            +-- Presentar Demo (PDM)
```

### Quality Gates per Content Type

| Gate | HLO | DLO | BCH | BLP | PDM |
|------|-----|-----|-----|-----|-----|
| YAML valid | Yes | Yes | -- | -- | Yes |
| Markdown valid | Yes | Yes | Yes | Yes | -- |
| Frontmatter present | -- | -- | Yes | Yes | -- |
| Code examples compile | -- | -- | Yes | Yes | Yes |
| Word count in range | -- | -- | Yes | Yes | -- |

### Toyota Way Integration

| Principle | Application |
|-----------|-------------|
| Genchi Genbutsu | Prompts require source material review |
| Jidoka | Validation schemas embedded in prompts |
| Poka-Yoke | Structural constraints in templates |
| Heijunka | Consistent content sizing targets |
| Kanban | Content type progression tracking |

---

## 4. Apple Hardware Integration (Manzana)

### Overview

`manzana` (Spanish: "apple") provides safe, pure Rust interfaces to Apple hardware subsystems for sovereign, on-premise ML workloads on macOS.

| Field | Value |
|-------|-------|
| Crate | `manzana` |
| Version | 0.1.0 |
| Tests | 174 passing |
| Status | Ready for release |

### Supported Hardware

| Accelerator | Module | Mac Pro | Apple Silicon | Intel Mac |
|-------------|--------|---------|---------------|-----------|
| Afterburner FPGA | `afterburner` | Yes | -- | -- |
| Neural Engine | `neural_engine` | -- | Yes | -- |
| Metal GPU | `metal` | Yes | Yes | Yes |
| Secure Enclave | `secure_enclave` | T2 | Yes | T2 |
| Unified Memory | `unified_memory` | -- | Yes | -- |

### Stack Integration

```
batuta orchestration
  +-- realizar (inference) --+
  +-- repartir (scheduling) -+-- manzana (Apple HW) --+-- trueno (compute)
  +-- entrenar (training) ---+
```

### Use Cases

| Accelerator | Use Case | Capability |
|-------------|----------|-----------|
| **Afterburner FPGA** | ProRes video decode for ML training data | 23x 4K or 6x 8K streams |
| **Neural Engine** | CoreML model inference | 15.8+ TOPS, zero-copy with UMA |
| **Metal GPU** | General-purpose GPU compute | wgpu backend via Metal API |
| **Secure Enclave** | Model signing, key storage | Ed25519, AES-256 |
| **Unified Memory** | Zero-copy CPU<->GPU | Eliminates PCIe transfer overhead |

### Sovereign AI Considerations

| Concern | Manzana Approach |
|---------|-----------------|
| Data privacy | Secure Enclave for key storage; on-device inference only |
| Computation residency | All processing on local Apple hardware |
| Hardware attestation | Secure Enclave provides hardware-rooted trust |
| Performance | UMA eliminates data copy overhead vs discrete GPU |
