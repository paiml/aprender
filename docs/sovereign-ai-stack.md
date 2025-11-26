# Paiml Sovereign AI Stack

**Pure Rust ML/AI Infrastructure - No US Cloud Dependency Required**

aprender is part of the **Paiml Sovereign AI Stack**, a complete pure Rust ecosystem for machine learning and AI that can run fully on-premises, in sovereign clouds, or air-gapped environments.

## Stack Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SOVEREIGN AI STACK                                  │
│                     Pure Rust • WASM • GPU/SIMD                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        APPLICATION LAYER                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  ruchy          │ Python-like scripting → Rust transpiler           │   │
│  │  depyler        │ Python → Rust compiler                            │   │
│  │  decy           │ C → Rust transpiler                               │   │
│  │  batuta         │ Orchestration & workflow management               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          ML/AI LAYER                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  realizar       │ Model serving, MLOps, LLMOps                      │   │
│  │  entrenar       │ Training & optimization                           │   │
│  │  ★ aprender ★   │ ML/DL algorithms (YOU ARE HERE)                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          DATA LAYER                                 │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  alimentar      │ Data loading, HuggingFace import, distribution    │   │
│  │  trueno-db      │ GPU-first columnar database (Arrow/Parquet)       │   │
│  │  trueno-graph   │ Graph database for code analysis                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         COMPUTE LAYER                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  trueno         │ SIMD/GPU/WASM tensor operations (foundation)      │   │
│  │  repartir       │ Distributed computing primitives                  │   │
│  │  trueno-viz     │ WASM visualization for browser                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        QUALITY LAYER                                │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │  pmat           │ Code quality analysis & scaffolding               │   │
│  │  certeza        │ Provability & testing framework                   │   │
│  │  renacer        │ Deep inspection & profiling                       │   │
│  │  verificar      │ Grammar → AST generation                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Pipeline Flow

```
┌──────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Sources    │    │  alimentar  │    │   Storage   │    │     ML      │
│              │    │             │    │             │    │             │
│ • HF Hub     │───▶│ • Import    │───▶│ • trueno-db │───▶│ • aprender  │
│ • Local      │    │ • Transform │    │ • trueno-   │    │ • entrenar  │
│ • S3/MinIO   │    │ • Stream    │    │   graph     │    │             │
│ • HTTP       │    │             │    │             │    │             │
└──────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                 │
                                                                 ▼
                                              ┌─────────────────────────────┐
                                              │          Serving            │
                                              │                             │
                                              │ • realizar (MLOps)          │
                                              │ • trueno-viz (browser)      │
                                              │ • WASM edge deployment      │
                                              └─────────────────────────────┘
```

## Component Details

### Compute Foundation

| Component | Description | crates.io |
|-----------|-------------|-----------|
| [trueno](https://github.com/paiml/trueno) | SIMD/GPU/WASM tensor operations | [![](https://img.shields.io/crates/v/trueno.svg)](https://crates.io/crates/trueno) |
| [repartir](https://github.com/paiml/repartir) | Distributed computing primitives | - |
| [trueno-viz](https://github.com/paiml/trueno-viz) | Browser visualization (WASM) | - |

### Data Layer

| Component | Description | crates.io |
|-----------|-------------|-----------|
| [alimentar](https://github.com/paiml/alimentar) | Data loading & distribution | - |
| [trueno-db](https://github.com/paiml/trueno-db) | GPU columnar database (Arrow 53) | [![](https://img.shields.io/crates/v/trueno-db.svg)](https://crates.io/crates/trueno-db) |
| [trueno-graph](https://github.com/paiml/trueno-graph) | Graph database | [![](https://img.shields.io/crates/v/trueno-graph.svg)](https://crates.io/crates/trueno-graph) |

### ML/AI Layer

| Component | Description | crates.io |
|-----------|-------------|-----------|
| [aprender](https://github.com/paiml/aprender) | ML/DL algorithms | [![](https://img.shields.io/crates/v/aprender.svg)](https://crates.io/crates/aprender) |
| [entrenar](https://github.com/paiml/entrenar) | Training & optimization | - |
| [realizar](https://github.com/paiml/realizar) | Model serving, MLOps | - |

### Language & Tooling

| Component | Description | crates.io |
|-----------|-------------|-----------|
| [ruchy](https://github.com/paiml/ruchy) | Python-like → Rust transpiler | - |
| [depyler](https://github.com/paiml/depyler) | Python → Rust compiler | - |
| [decy](https://github.com/paiml/decy) | C → Rust transpiler | - |
| [batuta](https://github.com/paiml/batuta) | Orchestration | - |

### Quality & Verification

| Component | Description | crates.io |
|-----------|-------------|-----------|
| [pmat](https://github.com/paiml/pmat) | Code quality & scaffolding | - |
| [certeza](https://github.com/paiml/certeza) | Provability framework | - |
| [renacer](https://github.com/paiml/renacer) | Deep inspection | - |
| [verificar](https://github.com/paiml/verificar) | Grammar → AST | - |

## Sovereign Deployment

The stack supports fully sovereign deployments:

### Storage Options (No US Cloud Required)

| Provider | Region | S3-Compatible Endpoint |
|----------|--------|------------------------|
| **Local** | On-premises | `file://` |
| **MinIO** | Self-hosted | `http://minio.local:9000` |
| **Ceph** | Self-hosted | Custom endpoint |
| **Scaleway** | EU (France) | `s3.fr-par.scw.cloud` |
| **OVH** | EU (France) | `s3.gra.cloud.ovh.net` |
| **Wasabi** | EU/APAC | Regional endpoints |
| **Cloudflare R2** | Global | `*.r2.cloudflarestorage.com` |

### Air-Gapped Deployment

```bash
# All components work fully offline
alimentar registry init ./local-datasets
trueno-db create ./local-db
aprender train --data ./local-datasets/train.parquet
```

### WASM Browser Deployment

```rust
// Run ML in browser - no server required
use aprender::prelude::*;
use trueno_viz::Chart;

let model = LinearRegression::load_wasm(bytes)?;
let predictions = model.predict(&input);
Chart::line(predictions).render()?;
```

## Integration Example

```rust
use alimentar::{ArrowDataset, DataLoader};
use aprender::prelude::*;
use trueno_db::Database;

// 1. Load data (alimentar)
let dataset = ArrowDataset::open("./train.parquet")?;
let loader = DataLoader::new(dataset).batch_size(32);

// 2. Store in database (trueno-db)
let db = Database::new()?;
for batch in loader.iter() {
    db.insert_batch("training", batch)?;
}

// 3. Train model (aprender)
let x = db.query("SELECT features FROM training")?.to_matrix()?;
let y = db.query("SELECT target FROM training")?.to_vector()?;

let mut model = RandomForestClassifier::new(100);
model.fit(&x, &y)?;

// 4. Serve (realizar)
realizar::serve(model, "0.0.0.0:8080")?;
```

## Why Sovereign?

1. **Data Residency** - Keep data in your jurisdiction (GDPR, CCPA, etc.)
2. **No Vendor Lock-in** - S3-compatible, runs anywhere
3. **Air-Gap Ready** - Works fully offline
4. **WASM Portable** - Run in browser, edge, or embedded
5. **Auditability** - Pure Rust, no hidden dependencies
6. **Performance** - GPU/SIMD acceleration without cloud latency

## Resources

- [Sovereign AI Stack Book](https://github.com/paiml/sovereign-ai-stack-book)
- [trueno Documentation](https://docs.rs/trueno)
- [aprender Documentation](https://docs.rs/aprender)
- [trueno-db Documentation](https://docs.rs/trueno-db)
