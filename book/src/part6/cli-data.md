# `batuta data`

Data platforms integration commands for visualizing and querying the enterprise data ecosystem.

## Synopsis

```bash
batuta data <COMMAND> [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| `tree` | Display data platforms ecosystem tree |

## Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-d, --debug` | Enable debug output |
| `-h, --help` | Print help |

---

## `batuta data tree`

Display hierarchical visualization of data platforms and their components, or show PAIML stack integration mappings.

### Usage

```bash
batuta data tree [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--platform <NAME>` | Filter by platform (databricks, snowflake, aws, huggingface) | All platforms |
| `--integration` | Show PAIML integration mappings instead of platform tree | false |
| `--format <FORMAT>` | Output format (ascii, json) | ascii |

### Examples

#### View All Platforms

```bash
$ batuta data tree

DATA PLATFORMS ECOSYSTEM
========================

DATABRICKS
├── Unity Catalog
│   └── Unity Catalog
│       ├── Schemas
│       ├── Tables
│       └── Views
├── Delta Lake
│   └── Delta Lake
│       ├── Parquet storage
│       ├── Transaction log
│       └── Time travel
...
```

#### Filter by Platform

```bash
$ batuta data tree --platform snowflake

SNOWFLAKE
├── Virtual Warehouse
│   └── Virtual Warehouse
│       ├── Compute clusters
│       ├── Result cache
│       └── Auto-scaling
├── Iceberg Tables
│   └── Iceberg Tables
│       ├── Open format
│       ├── Schema evolution
│       └── Partition pruning
├── Snowpark
│   └── Snowpark
│       ├── Python UDFs
│       ├── Java/Scala UDFs
│       └── ML functions
└── Data Sharing
    └── Data Sharing
        ├── Secure shares
        ├── Reader accounts
        └── Marketplace
```

#### View Integration Mappings

```bash
$ batuta data tree --integration

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

Summary: 3 compatible, 16 alternatives, 2 uses, 2 transpiles, 4 orchestrates
         Total: 27 integration points
```

#### JSON Output

```bash
$ batuta data tree --platform databricks --format json

{
  "platform": "Databricks",
  "categories": [
    {
      "name": "Unity Catalog",
      "components": [
        {
          "name": "Unity Catalog",
          "description": "Unified governance for data and AI",
          "sub_components": ["Schemas", "Tables", "Views"]
        }
      ]
    },
    ...
  ]
}
```

```bash
$ batuta data tree --integration --format json

[
  {
    "platform_component": "Delta Lake",
    "paiml_component": "Alimentar (.ald)",
    "integration_type": "Alternative",
    "category": "STORAGE & CATALOGS"
  },
  ...
]
```

### Integration Type Legend

| Code | Type | Meaning |
|------|------|---------|
| `CMP` | Compatible | Direct interoperability with PAIML component |
| `ALT` | Alternative | PAIML provides a sovereign replacement |
| `USE` | Uses | PAIML component consumes this as input |
| `TRN` | Transpiles | Depyler converts source code to Rust |
| `ORC` | Orchestrates | Batuta can coordinate external workflows |

### Supported Platforms

| Platform | Description |
|----------|-------------|
| `databricks` | Unity Catalog, Delta Lake, MLflow, Spark |
| `snowflake` | Virtual Warehouse, Iceberg, Snowpark, Data Sharing |
| `aws` | S3, Glue, SageMaker, Bedrock, EMR, Lambda |
| `huggingface` | Hub, Transformers, Datasets, Inference API |

## See Also

- [`batuta hf`](./cli-hf.md) - HuggingFace Hub operations
- [`batuta stack`](./cli-stack.md) - PAIML stack management
- [`batuta oracle`](./cli-oracle.md) - Intelligent query interface
- [Data Platforms Integration](../part3/data-platforms.md) - Detailed documentation
