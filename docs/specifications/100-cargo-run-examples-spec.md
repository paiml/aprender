# Alimentar: 100 Cargo Run Examples Specification

**Project**: alimentar v0.2.1
**Date**: 2025-11-30
**Methodology**: Toyota Production System (TPS)
**Standard**: ISO/IEC 25010:2011 Software Quality Model

---

## Table of Contents

1. [Philosophy & QA Checklist](#1-philosophy--qa-checklist)
2. [Section A: Basic Loading (1-10)](#section-a-basic-loading-1-10)
3. [Section B: DataLoader & Batching (11-20)](#section-b-dataloader--batching-11-20)
4. [Section C: Streaming & Memory (21-30)](#section-c-streaming--memory-21-30)
5. [Section D: Transforms Pipeline (31-45)](#section-d-transforms-pipeline-31-45)
6. [Section E: Quality & Validation (46-55)](#section-e-quality--validation-46-55)
7. [Section F: Drift Detection (56-65)](#section-f-drift-detection-56-65)
8. [Section G: Federated & Splitting (66-75)](#section-g-federated--splitting-66-75)
9. [Section H: HuggingFace Hub (76-85)](#section-h-huggingface-hub-76-85)
10. [Section I: CLI & REPL (86-95)](#section-i-cli--repl-86-95)
11. [Section J: Edge Cases & WASM (96-100)](#section-j-edge-cases--wasm-96-100)
12. [Appendix: Peer-Reviewed Citations](#appendix-peer-reviewed-citations)

---

## 1. Philosophy & QA Checklist

### Toyota Way Principles Applied

| Principle | Application to Data Loading |
|-----------|----------------------------|
| **Jidoka** | Stop on corrupt data, NaN detection, schema validation |
| **Poka-Yoke** | Type-safe APIs, compile-time checks, Result types |
| **Heijunka** | Leveled batch processing, streaming for memory stability |
| **Kaizen** | Continuous quality metrics, drift monitoring |
| **Genchi Genbutsu** | Direct data inspection, quality reports |

### 15-Point QA Checklist (Per Example)

**Data Integrity**
1. [ ] Schema matches expected structure
2. [ ] No silent data truncation
3. [ ] Null handling explicit and correct
4. [ ] SHA256 hash verification (if applicable)
5. [ ] Arrow RecordBatch valid

**Memory & Performance**
6. [ ] Memory usage bounded (no leaks)
7. [ ] Throughput meets baseline (>10k rows/sec)
8. [ ] No unnecessary copies (zero-copy verified)
9. [ ] Streaming maintains constant memory
10. [ ] WASM compatible (where applicable)

**Quality & Safety**
11. [ ] Error handling complete (no panics)
12. [ ] Clippy clean (no warnings)
13. [ ] Example compiles with `--release`
14. [ ] Documentation accurate
15. [ ] Deterministic output (fixed seed)

---

## Section A: Basic Loading (1-10)

**1. CSV Loading**
```bash
cargo run --example basic_loading
```
*QA Focus*: CSV parsing, header detection, type inference. *Validation*: [ ]

**2. JSON Loading**
```bash
cargo run --example basic_loading -- --format json
```
*QA Focus*: Nested JSON flattening, array handling. *Validation*: [ ]

**3. Parquet Loading**
```bash
cargo run --example basic_loading -- --format parquet
```
*QA Focus*: Column pruning, predicate pushdown. *Validation*: [ ]

**4. Schema Inference**
```bash
cargo run --example basic_loading -- --infer-schema
```
*QA Focus*: Type detection accuracy (int vs float vs string). *Validation*: [ ]

**5. Explicit Schema**
```bash
cargo run --example basic_loading -- --schema schema.json
```
*QA Focus*: Schema enforcement, type coercion errors. *Validation*: [ ]

**6. Multi-File Loading**
```bash
cargo run --example basic_loading -- --glob "data/*.parquet"
```
*QA Focus*: Schema unification across files. *Validation*: [ ]

**7. Memory-Mapped Loading**
```bash
cargo run --example basic_loading --features mmap -- --mmap
```
*QA Focus*: Virtual memory usage, page fault efficiency. *Validation*: [ ]

**8. Compressed Input (ZSTD)**
```bash
cargo run --example basic_loading -- --input data.parquet.zst
```
*QA Focus*: Decompression throughput >100MB/s. *Validation*: [ ]

**9. Compressed Input (LZ4)**
```bash
cargo run --example basic_loading -- --input data.parquet.lz4
```
*QA Focus*: Low-latency decompression. *Validation*: [ ]

**10. Large File (>1GB)**
```bash
cargo run --example basic_loading --release -- --input large.parquet
```
*QA Focus*: Memory stays <2GB for 1GB file. *Validation*: [ ]

---

## Section B: DataLoader & Batching (11-20)

**11. Basic Batching**
```bash
cargo run --example dataloader_batching
```
*QA Focus*: Batch size respected, row count correct. *Validation*: [ ]

**12. Shuffle with Seed**
```bash
cargo run --example dataloader_batching --features shuffle -- --seed 42
```
*QA Focus*: Deterministic ordering with same seed. *Validation*: [ ]

**13. Drop Last Batch**
```bash
cargo run --example dataloader_batching -- --drop-last
```
*QA Focus*: Incomplete batches discarded correctly. *Validation*: [ ]

**14. Parallel DataLoader**
```bash
cargo run --example dataloader_batching -- --workers 4
```
*QA Focus*: No race conditions, correct batch ordering. *Validation*: [ ]

**15. Prefetch Tuning**
```bash
cargo run --example dataloader_batching -- --prefetch 8
```
*QA Focus*: I/O latency hidden, memory bounded. *Validation*: [ ]

**16. Weighted Sampling**
```bash
cargo run --example dataloader_batching --features shuffle -- --weighted
```
*QA Focus*: Class imbalance correction verified. *Validation*: [ ]

**17. Stratified Batching**
```bash
cargo run --example dataloader_batching -- --stratify label
```
*QA Focus*: Each batch has proportional class distribution. *Validation*: [ ]

**18. Infinite Iterator**
```bash
cargo run --example dataloader_batching -- --epochs infinite --max-steps 1000
```
*QA Focus*: Epoch boundary handling, no memory growth. *Validation*: [ ]

**19. Custom Collate Function**
```bash
cargo run --example dataloader_batching -- --collate pad
```
*QA Focus*: Variable-length sequences padded correctly. *Validation*: [ ]

**20. Batch Size Benchmark**
```bash
cargo run --example dataloader_batching --release -- --benchmark
```
*QA Focus*: Throughput scales with batch size. *Validation*: [ ]

---

## Section C: Streaming & Memory (21-30)

**21. Basic Streaming**
```bash
cargo run --example streaming_large
```
*QA Focus*: Constant memory regardless of dataset size. *Validation*: [ ]

**22. Chained Sources**
```bash
cargo run --example streaming_large -- --chain file1.parquet file2.parquet
```
*QA Focus*: Seamless iteration across sources. *Validation*: [ ]

**23. Memory Source**
```bash
cargo run --example streaming_large -- --memory
```
*QA Focus*: In-memory batches stream correctly. *Validation*: [ ]

**24. Parquet Streaming**
```bash
cargo run --example streaming_large -- --parquet data/
```
*QA Focus*: Row group streaming, no full file load. *Validation*: [ ]

**25. Buffer Size Tuning**
```bash
cargo run --example streaming_large -- --buffer-size 16
```
*QA Focus*: Memory vs latency tradeoff documented. *Validation*: [ ]

**26. Async Prefetch**
```bash
cargo run --example streaming_large --features tokio-runtime -- --async
```
*QA Focus*: Background I/O, main thread unblocked. *Validation*: [ ]

**27. Backpressure Handling**
```bash
cargo run --example streaming_large -- --slow-consumer
```
*QA Focus*: Producer waits, no unbounded buffering. *Validation*: [ ]

**28. Iterator Reset**
```bash
cargo run --example streaming_large -- --reset-test
```
*QA Focus*: Multiple iterations produce same data. *Validation*: [ ]

**29. Memory Profile (Valgrind)**
```bash
valgrind --leak-check=full cargo run --example streaming_large --release
```
*QA Focus*: Zero memory leaks reported. *Validation*: [ ]

**30. 10GB Dataset Test**
```bash
cargo run --example streaming_large --release -- --generate-10gb
```
*QA Focus*: Completes without OOM, <500MB RSS. *Validation*: [ ]

---

## Section D: Transforms Pipeline (31-45)

**31. Column Selection**
```bash
cargo run --example transforms_pipeline -- --select id,name,value
```
*QA Focus*: Only specified columns in output. *Validation*: [ ]

**32. Column Drop**
```bash
cargo run --example transforms_pipeline -- --drop temp_col
```
*QA Focus*: Column removed, schema updated. *Validation*: [ ]

**33. Column Rename**
```bash
cargo run --example transforms_pipeline -- --rename old_name:new_name
```
*QA Focus*: Rename propagates through pipeline. *Validation*: [ ]

**34. Row Filter (Numeric)**
```bash
cargo run --example transforms_pipeline -- --filter "value > 100"
```
*QA Focus*: Predicate pushdown verified. *Validation*: [ ]

**35. Row Filter (String)**
```bash
cargo run --example transforms_pipeline -- --filter "region = 'US'"
```
*QA Focus*: String comparison correct. *Validation*: [ ]

**36. Null Fill (Mean)**
```bash
cargo run --example transforms_pipeline -- --fill-null mean
```
*QA Focus*: Mean computed correctly, nulls replaced. *Validation*: [ ]

**37. Null Fill (Constant)**
```bash
cargo run --example transforms_pipeline -- --fill-null 0.0
```
*QA Focus*: All nulls become specified value. *Validation*: [ ]

**38. Normalize (MinMax)**
```bash
cargo run --example transforms_pipeline -- --normalize minmax
```
*QA Focus*: Values in [0,1] range. *Validation*: [ ]

**39. Normalize (ZScore)**
```bash
cargo run --example transforms_pipeline -- --normalize zscore
```
*QA Focus*: Mean≈0, Std≈1 verified. *Validation*: [ ]

**40. Sort Ascending**
```bash
cargo run --example transforms_pipeline -- --sort value:asc
```
*QA Focus*: Order verified, stable sort. *Validation*: [ ]

**41. Sort Descending**
```bash
cargo run --example transforms_pipeline -- --sort value:desc
```
*QA Focus*: Reverse order correct. *Validation*: [ ]

**42. Take/Limit**
```bash
cargo run --example transforms_pipeline -- --take 100
```
*QA Focus*: Exactly N rows returned. *Validation*: [ ]

**43. Skip/Offset**
```bash
cargo run --example transforms_pipeline -- --skip 50 --take 100
```
*QA Focus*: Pagination correct. *Validation*: [ ]

**44. Unique/Dedup**
```bash
cargo run --example transforms_pipeline -- --unique id
```
*QA Focus*: Duplicates removed, first kept. *Validation*: [ ]

**45. Transform Chain**
```bash
cargo run --example transforms_pipeline -- --chain "select,filter,normalize,sort"
```
*QA Focus*: All transforms compose correctly. *Validation*: [ ]

---

## Section E: Quality & Validation (46-55)

**46. Basic Quality Check**
```bash
cargo run --example quality_check
```
*QA Focus*: Report generated, score calculated. *Validation*: [ ]

**47. Missing Value Detection**
```bash
cargo run --example quality_check -- --check missing
```
*QA Focus*: Null percentage per column accurate. *Validation*: [ ]

**48. Duplicate Detection**
```bash
cargo run --example quality_check -- --check duplicates
```
*QA Focus*: Duplicate rows identified. *Validation*: [ ]

**49. Type Validation**
```bash
cargo run --example quality_check -- --check types
```
*QA Focus*: Type mismatches flagged. *Validation*: [ ]

**50. Range Checking**
```bash
cargo run --example quality_check -- --check range --min 0 --max 100
```
*QA Focus*: Out-of-range values reported. *Validation*: [ ]

**51. Cardinality Check**
```bash
cargo run --example quality_check -- --check cardinality
```
*QA Focus*: Unique value counts correct. *Validation*: [ ]

**52. Constant Column Detection**
```bash
cargo run --example quality_check -- --check constant
```
*QA Focus*: Single-value columns identified. *Validation*: [ ]

**53. Quality Score Calculation**
```bash
cargo run --example quality_check -- --score
```
*QA Focus*: 100-point weighted score computed. *Validation*: [ ]

**54. Quality Profile (Strict)**
```bash
cargo run --example quality_check -- --profile strict
```
*QA Focus*: Strict thresholds applied. *Validation*: [ ]

**55. Quality Report Export**
```bash
cargo run --example quality_check -- --export report.json
```
*QA Focus*: JSON schema valid. *Validation*: [ ]

---

## Section F: Drift Detection (56-65)

**56. Basic Drift Detection**
```bash
cargo run --example drift_detection
```
*QA Focus*: Drift report generated. *Validation*: [ ]

**57. KS Test (Numeric)**
```bash
cargo run --example drift_detection -- --test ks
```
*QA Focus*: Kolmogorov-Smirnov statistic correct. *Validation*: [ ]

**58. Chi-Square Test (Categorical)**
```bash
cargo run --example drift_detection -- --test chi2
```
*QA Focus*: Chi-square statistic correct. *Validation*: [ ]

**59. PSI (Population Stability Index)**
```bash
cargo run --example drift_detection -- --test psi
```
*QA Focus*: PSI buckets computed correctly. *Validation*: [ ]

**60. Drift Severity Classification**
```bash
cargo run --example drift_detection -- --severity
```
*QA Focus*: None/Low/Medium/High correctly assigned. *Validation*: [ ]

**61. Column-Level Drift**
```bash
cargo run --example drift_detection -- --columns age,income
```
*QA Focus*: Per-column drift scores. *Validation*: [ ]

**62. Drift Threshold Alert**
```bash
cargo run --example drift_detection -- --threshold 0.1
```
*QA Focus*: Alert triggered when exceeded. *Validation*: [ ]

**63. Drift Sketch (HyperLogLog)**
```bash
cargo run --example drift_detection -- --sketch hll
```
*QA Focus*: Approximate cardinality accurate. *Validation*: [ ]

**64. Drift Sketch (CountMin)**
```bash
cargo run --example drift_detection -- --sketch countmin
```
*QA Focus*: Frequency estimation accurate. *Validation*: [ ]

**65. Drift Report Export**
```bash
cargo run --example drift_detection -- --export drift.json
```
*QA Focus*: JSON schema valid. *Validation*: [ ]

---

## Section G: Federated & Splitting (66-75)

**66. Basic Train/Test Split**
```bash
cargo run --example federated_split -- --ratio 0.8
```
*QA Focus*: 80/20 split exact. *Validation*: [ ]

**67. Stratified Split**
```bash
cargo run --example federated_split -- --stratify label
```
*QA Focus*: Class proportions preserved. *Validation*: [ ]

**68. K-Fold Cross Validation**
```bash
cargo run --example federated_split -- --kfold 5
```
*QA Focus*: 5 non-overlapping folds. *Validation*: [ ]

**69. Leave-One-Out**
```bash
cargo run --example federated_split -- --loo
```
*QA Focus*: N-1/1 splits generated. *Validation*: [ ]

**70. Node Manifest Generation**
```bash
cargo run --example federated_split -- --manifest
```
*QA Focus*: No raw data in manifest. *Validation*: [ ]

**71. Federated Coordinator**
```bash
cargo run --example federated_split -- --coordinate
```
*QA Focus*: Split plan distributed correctly. *Validation*: [ ]

**72. IID Strategy**
```bash
cargo run --example federated_split -- --strategy iid
```
*QA Focus*: Random assignment verified. *Validation*: [ ]

**73. Non-IID Strategy**
```bash
cargo run --example federated_split -- --strategy non-iid
```
*QA Focus*: Heterogeneous distribution. *Validation*: [ ]

**74. Dirichlet Partition**
```bash
cargo run --example federated_split -- --strategy dirichlet --alpha 0.5
```
*QA Focus*: Concentration parameter respected. *Validation*: [ ]

**75. Multi-Node Simulation**
```bash
cargo run --example federated_split -- --nodes 10
```
*QA Focus*: All nodes receive data. *Validation*: [ ]

---

## Section H: HuggingFace Hub (76-85)

**76. Dataset Download**
```bash
cargo run --example hub_publishing --features hf-hub -- --download squad
```
*QA Focus*: Cache populated, checksum valid. *Validation*: [ ]

**77. Dataset Card Validation**
```bash
cargo run --example hub_publishing --features hf-hub -- --validate-card
```
*QA Focus*: Required fields present. *Validation*: [ ]

**78. Quality Score for Hub**
```bash
cargo run --example hub_publishing --features hf-hub -- --quality-score
```
*QA Focus*: 100-point score computed. *Validation*: [ ]

**79. README Generation**
```bash
cargo run --example hub_publishing --features hf-hub -- --generate-readme
```
*QA Focus*: Markdown valid, stats accurate. *Validation*: [ ]

**80. Native Hub Upload**
```bash
cargo run --example hub_publishing --features hf-hub -- --upload test-dataset
```
*QA Focus*: Upload succeeds (dry-run OK). *Validation*: [ ]

**81. Revision/Branch Support**
```bash
cargo run --example hub_publishing --features hf-hub -- --revision main
```
*QA Focus*: Branch selection works. *Validation*: [ ]

**82. Private Dataset Upload**
```bash
cargo run --example hub_publishing --features hf-hub -- --private
```
*QA Focus*: Visibility set correctly. *Validation*: [ ]

**83. Hub Cache Management**
```bash
cargo run --example hub_publishing --features hf-hub -- --cache-info
```
*QA Focus*: Cache size and entries reported. *Validation*: [ ]

**84. Offline Mode**
```bash
cargo run --example hub_publishing --features hf-hub -- --offline
```
*QA Focus*: Uses cached data only. *Validation*: [ ]

**85. Hub Token Auth**
```bash
HF_TOKEN=xxx cargo run --example hub_publishing --features hf-hub -- --auth-test
```
*QA Focus*: Token validated securely. *Validation*: [ ]

---

## Section I: CLI & REPL (86-95)

**86. CLI Help**
```bash
cargo run --features cli -- --help
```
*QA Focus*: All subcommands documented. *Validation*: [ ]

**87. CLI Info Command**
```bash
cargo run --features cli -- info data.parquet
```
*QA Focus*: Schema and stats displayed. *Validation*: [ ]

**88. CLI Head Command**
```bash
cargo run --features cli -- head -n 10 data.parquet
```
*QA Focus*: First N rows correct. *Validation*: [ ]

**89. CLI Convert Command**
```bash
cargo run --features cli -- convert input.csv output.parquet
```
*QA Focus*: Format conversion lossless. *Validation*: [ ]

**90. CLI Quality Command**
```bash
cargo run --features cli -- quality data.parquet
```
*QA Focus*: Quality report printed. *Validation*: [ ]

**91. REPL Session Start**
```bash
cargo run --example repl_session --features repl
```
*QA Focus*: Prompt appears, commands work. *Validation*: [ ]

**92. REPL Tab Completion**
```bash
cargo run --example repl_completer --features repl
```
*QA Focus*: Completions accurate. *Validation*: [ ]

**93. REPL Command Parsing**
```bash
cargo run --example repl_commands --features repl
```
*QA Focus*: All commands parse correctly. *Validation*: [ ]

**94. REPL History**
```bash
cargo run --example repl_session --features repl -- --history
```
*QA Focus*: History persists across sessions. *Validation*: [ ]

**95. CLI Batch Script**
```bash
cargo run --example cli_batch_commands --features cli
```
*QA Focus*: Script executes all commands. *Validation*: [ ]

---

## Section J: Edge Cases & WASM (96-100)

**96. WASM Build**
```bash
cargo build --target wasm32-unknown-unknown --no-default-features --features wasm
```
*QA Focus*: Compiles without errors, <500KB. *Validation*: [ ]

**97. Empty Dataset Handling**
```bash
cargo run --example basic_loading -- --input empty.parquet
```
*QA Focus*: Graceful error, no panic. *Validation*: [ ]

**98. Malformed Input (Jidoka)**
```bash
cargo run --example basic_loading -- --input corrupt.parquet
```
*QA Focus*: Error reported, system halts cleanly. *Validation*: [ ]

**99. S3 Backend Integration**
```bash
cargo run --example registry_publish --features s3 -- --backend s3://bucket/path
```
*QA Focus*: MinIO/S3 compatible (docker-compose). *Validation*: [ ]

**100. Golden Run (All Features)**
```bash
cargo test --all-features && cargo clippy -- -D warnings && cargo fmt --check
```
*QA Focus*: **ALL 15 QA CHECKS PASS.** Peer review required. *Validation*: [ ]

---

## Appendix: Peer-Reviewed Citations

| # | Citation | Application |
|---|----------|-------------|
| [1] | Liker, J.K. (2004). *The Toyota Way*. McGraw-Hill. | Jidoka, Poka-Yoke principles |
| [2] | Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly. | Streaming, backpressure |
| [3] | Abadi, D. et al. (2013). "The Design of the Borealis Stream Processing Engine." *CIDR*. | Data flow architecture |
| [4] | Zaharia, M. et al. (2016). "Apache Spark: A Unified Engine." *CACM* 59(11). | DataLoader patterns |
| [5] | Polyzotis, N. et al. (2019). "Data Validation for ML." *MLSys*. | Quality checking |
| [6] | Sculley, D. et al. (2015). "Hidden Technical Debt in ML Systems." *NeurIPS*. | Drift detection |
| [7] | McMahan, H.B. et al. (2017). "Federated Learning." *AISTATS*. | Federated splitting |
| [8] | Wolf, T. et al. (2020). "HuggingFace's Transformers." *EMNLP*. | Hub integration |
| [9] | Armbrust, M. et al. (2015). "Spark SQL." *SIGMOD*. | Arrow/Parquet patterns |
| [10] | Klabnik, S. & Nichols, C. (2023). *The Rust Programming Language*. No Starch. | Safety guarantees |

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| QA Lead | | | |
| Tech Lead | | | |

**Final Score**: ___/100 | **Grade**: ___ | **Ship Decision**: [ ] GO / [ ] NO-GO

---

*Generated: 2025-11-30 | alimentar v0.2.1 | Toyota Way QA Process*
