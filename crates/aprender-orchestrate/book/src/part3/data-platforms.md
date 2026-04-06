# Data Platforms Integration

Batuta provides a unified interface for integrating with enterprise data platforms while maintaining sovereignty over your ML infrastructure. The `batuta data` command visualizes the ecosystem and shows how PAIML stack components map to commercial alternatives.

## Toyota Way Principles

The data platforms integration embodies key Lean principles:

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Direct platform API queries - go to the source |
| **Poka-Yoke** | OS-level egress filtering for sovereignty enforcement |
| **Heijunka** | Adaptive throttling for shared resources |
| **Jidoka** | Schema drift detection stops the line |
| **Muda** | Federation over migration (zero-copy where possible) |
| **Andon** | Cost estimation before query execution |

## Supported Platforms

### Databricks

```
DATABRICKS
├── Unity Catalog
│   └── Schemas, Tables, Views
├── Delta Lake
│   └── Parquet storage, Transaction log, Time travel
├── MLflow
│   └── Experiment tracking, Model registry, Model serving
└── Spark
    └── DataFrames, Structured Streaming, MLlib
```

**PAIML Mappings:**
- Delta Lake → Alimentar (.ald format) - Alternative
- Unity Catalog → Pacha Registry - Alternative
- MLflow → Entrenar experiment tracking - Alternative
- Spark DataFrames → Trueno tensors - Alternative

### Snowflake

```
SNOWFLAKE
├── Virtual Warehouse
│   └── Compute clusters, Result cache, Auto-scaling
├── Iceberg Tables
│   └── Open format, Schema evolution, Partition pruning
├── Snowpark
│   └── Python UDFs, Java/Scala UDFs, ML functions
└── Data Sharing
    └── Secure shares, Reader accounts, Marketplace
```

**PAIML Mappings:**
- Iceberg Tables → Alimentar (.ald) - Compatible (open format)
- Snowpark Python → Depyler transpilation - Transpiles
- Snowpark ML → Aprender - Alternative

### AWS

```
AWS
├── Storage
│   ├── S3 (Objects, Versioning, Lifecycle)
│   ├── Glue Catalog (Databases, Tables, Crawlers)
│   └── Lake Formation
├── Compute
│   ├── EMR, Lambda, ECS/EKS
├── ML
│   ├── SageMaker (Training, Endpoints, Pipelines)
│   ├── Bedrock (Foundation models, Fine-tuning, Agents)
│   └── Comprehend
└── Analytics
    └── Athena, Redshift, QuickSight
```

**PAIML Mappings:**
- S3 → Alimentar sync - Compatible
- Glue Catalog → Pacha Registry - Alternative
- SageMaker Training → Entrenar - Alternative
- Bedrock → Realizar + serve module - Alternative
- Lambda Python → Depyler transpilation - Transpiles

### HuggingFace

```
HUGGINGFACE
├── Hub
│   └── Models, Datasets, Spaces, Organizations
├── Transformers
│   └── Models, Tokenizers, Pipelines
├── Datasets
│   └── Streaming, Arrow format, Processing
└── Inference API
    └── Serverless, Dedicated, TEI/TGI
```

**PAIML Mappings:**
- Hub → Pacha Registry - Alternative
- Transformers → Realizar (via GGUF) - Compatible
- Datasets Arrow → Alimentar (.ald) - Compatible
- GGUF models → Realizar inference - Uses

## CLI Usage

### View All Platforms

```bash
batuta data tree
```

### Filter by Platform

```bash
batuta data tree --platform databricks
batuta data tree --platform snowflake
batuta data tree --platform aws
batuta data tree --platform huggingface
```

### View PAIML Integration Mappings

```bash
batuta data tree --integration
```

Output shows all 31 integration points:

```
PAIML ↔ DATA PLATFORMS INTEGRATION
==================================

STORAGE & CATALOGS
├── [ALT] Alimentar (.ald) ←→ Delta Lake
├── [CMP] Alimentar (.ald) ←→ Iceberg Tables
├── [CMP] Alimentar (sync) ←→ S3
├── [ALT] Pacha Registry ←→ Unity Catalog
├── [ALT] Pacha Registry ←→ Glue Catalog
├── [ALT] Pacha Registry ←→ HuggingFace Hub

COMPUTE & PROCESSING
├── [ALT] Trueno ←→ Spark DataFrames
├── [ALT] Trueno ←→ Snowpark
├── [ALT] Trueno ←→ EMR
├── [TRN] Depyler → Rust ←→ Snowpark Python
├── [TRN] Depyler → Rust ←→ Lambda Python
├── [ALT] Trueno-Graph ←→ Neptune/GraphQL

ML TRAINING
├── [ALT] Aprender ←→ MLlib
├── [ALT] Aprender ←→ Snowpark ML
├── [ALT] Entrenar ←→ SageMaker Training
├── [ALT] Entrenar ←→ MLflow Tracking
├── [ALT] Entrenar ←→ SageMaker Experiments
├── [USE] Entrenar ←→ W&B

MODEL SERVING
├── [ALT] Realizar ←→ MLflow Serving
├── [ALT] Realizar ←→ SageMaker Endpoints
├── [ALT] Realizar + serve ←→ Bedrock
├── [USE] Realizar ←→ GGUF models
├── [CMP] Realizar (via GGUF) ←→ HF Transformers

ORCHESTRATION
├── [ORC] Batuta ←→ Databricks Workflows
├── [ORC] Batuta ←→ Snowflake Tasks
├── [ORC] Batuta ←→ Step Functions
├── [ORC] Batuta ←→ Airflow/Prefect

Legend: [CMP]=Compatible [ALT]=Alternative [USE]=Uses
        [TRN]=Transpiles [ORC]=Orchestrates
```

### JSON Output

```bash
batuta data tree --format json
batuta data tree --platform aws --format json
batuta data tree --integration --format json
```

## Integration Types

| Code | Type | Description |
|------|------|-------------|
| CMP | Compatible | Works directly with PAIML component |
| ALT | Alternative | PAIML provides sovereign alternative |
| USE | Uses | PAIML component consumes this format |
| TRN | Transpiles | Depyler converts code to Rust |
| ORC | Orchestrates | Batuta can coordinate workflows |

## Data Sovereignty Tiers

The integration supports four sovereignty levels:

```rust
pub enum DataSovereigntyTier {
    /// All data stays on-premises, no external calls
    FullySovereign,
    /// Private cloud (AWS GovCloud, Azure Gov)
    HybridSovereign,
    /// Standard private cloud deployment
    PrivateCloud,
    /// Standard commercial cloud
    Standard,
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BATUTA ORCHESTRATOR                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────────────┐   │
│  │Databricks│  │Snowflake │  │   AWS   │  │ HuggingFace │   │
│  │ Adapter │  │ Adapter  │  │ Adapter │  │   Adapter   │   │
│  └────┬────┘  └────┬─────┘  └────┬────┘  └──────┬──────┘   │
│       │            │             │              │           │
│       └────────────┴──────┬──────┴──────────────┘           │
│                           │                                  │
│                    ┌──────▼──────┐                          │
│                    │  Unified    │                          │
│                    │  Data API   │                          │
│                    └──────┬──────┘                          │
│                           │                                  │
│    ┌──────────────────────┼──────────────────────┐         │
│    │                      │                      │          │
│    ▼                      ▼                      ▼          │
│ ┌──────┐            ┌──────────┐           ┌─────────┐     │
│ │Alimentar│          │  Pacha   │           │ Entrenar│     │
│ │(.ald)  │          │ Registry │           │Tracking │     │
│ └────────┘          └──────────┘           └─────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Kaizen Recommendations

Based on Toyota Way analysis, future enhancements include:

1. **Cost Andon Cord** - Pre-flight cost estimation before expensive queries
2. **Resumable Sync** - Stateful checkpointing for long-running transfers
3. **Schema Drift Detection** - Jidoka-style automatic stops on upstream changes
4. **Adaptive Throttling** - Heijunka-based rate limiting for shared warehouses
5. **Federation Architecture** - Virtual catalogs to eliminate migration waste
6. **Information Flow Control** - Taint tracking for data provenance

## See Also

- [Oracle Mode](./oracle-mode.md) - Query the stack for recommendations
- [HuggingFace Integration](../part6/cli-hf.md) - Detailed HF Hub operations
- [Alimentar Specification](https://github.com/paiml/alimentar) - Data format details
