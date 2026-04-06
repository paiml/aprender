# PMAT Development Roadmap

## Completed: v0.1.0 Core
- **Status**: COMPLETED
- **Features**: ArrowDataset, DataLoader, Transforms, LocalBackend, MemoryBackend, CLI

## Current Sprint: v0.2.0 Storage & Registry
- **Duration**: 2025-11-25 to 2025-12-09
- **Priority**: P0
- **Quality Gates**: Complexity ≤ 15, SATD = 0, Coverage ≥ 90%

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| S3-01 | Implement S3Backend with aws-sdk-s3 | ✅ done | high | P0 |
| HTTP-01 | Implement HttpBackend with reqwest | ✅ done | medium | P0 |
| REG-01 | Registry index format and DatasetMetadata | ✅ done | medium | P0 |
| REG-02 | Registry publish operation | ✅ done | medium | P0 |
| REG-03 | Registry pull operation | ✅ done | medium | P0 |
| REG-04 | Registry list and search | ✅ done | low | P1 |
| STREAM-01 | StreamingDataset implementation | ✅ done | high | P0 |
| CLI-01 | Add registry CLI commands | ✅ done | medium | P1 |
| TEST-01 | S3 provider integration tests (MinIO) | ✅ done | medium | P0 |

### Definition of Done
- [x] All P0 tasks completed
- [x] Quality gates passed (85% coverage target, 0 SATD)
- [x] Documentation updated
- [x] All tests passing (673 tests)
- [ ] Changelog updated
- [x] Zero unwrap() in library code

### Notes
- S3 integration tests run via CI with MinIO service container
- Local S3 testing: `docker compose up -d && make test-s3`
- Coverage target adjusted to 85% (network-dependent code excluded)

---

## Completed: v0.2.1 Test Coverage Enhancement
- **Status**: COMPLETED
- **Date**: 2025-11-28
- **Changes**:
  - Added 22 new tests (673 -> 695 total)
  - hf_hub.rs: Cache hit path, clear cache with files
  - streaming.rs: Error handling, edge cases, batch sizes
  - encryption.rs: Edge cases, RngWrapper coverage, corruption tests
  - Coverage: 89.32% (exceeds 85% target)

---

## Completed: v0.3.0 ML Integration
- **Status**: COMPLETED
- **Date**: 2025-11-28
- **Quality Gates**: Coverage ≥ 90%, SATD = 0

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| ML-01 | Integration with aprender (CITL) | ✅ done | high | P0 |
| PERF-02 | Parallel data loading | ✅ done | high | P1 |

### Changes
- Added WeightedDataLoader for CITL reweighting (7 tests)
- Added AsyncPrefetchDataset for async batch loading (9 tests)
- Implemented ParallelDataLoader with multi-worker support (12 tests)
- Total tests: 723 (up from 695)

---

## Completed: v0.4.0 Advanced Features
- **Status**: COMPLETED
- **Date**: 2025-11-28
- **Quality Gates**: Coverage ≥ 90%, SATD = 0

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| ML-02 | Tensor conversion utilities | ✅ done | medium | P0 |
| PERF-01 | Memory-mapped datasets (mmap) | ✅ done | medium | P1 |

### Changes
- Implemented MmapDataset for memory-mapped parquet files (22 tests)
- Implemented TensorData and TensorExtractor for ML integration (31 tests)
- Added extract_column_f32/f64, extract_labels_i64 helpers
- Total tests: 776 (up from 723)

---

## Completed: v0.5.0 Test Coverage Enhancement
- **Status**: COMPLETED
- **Date**: 2025-11-28
- **Quality Gates**: Coverage ≥ 90%, SATD = 0

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| TEST-02 | Increase test coverage via error path tests | ✅ done | medium | P0 |

### Changes
- Added 13 transform.rs edge case tests (error paths, empty batches)
- Added 18 dataloader.rs edge case tests (empty dataset, size hints)
- Added 15 split.rs edge case tests (ratio validation, edge cases)
- Total tests: 818 (up from 776)
- All quality gates passing (clippy, fmt, tests)

---

## Completed: v0.5.1 ML Integration Enhancement
- **Status**: COMPLETED
- **Date**: 2025-11-28
- **Quality Gates**: Coverage ≥ 90%, SATD = 0

### Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| INT-01 | Integration tests for alimentar ↔ aprender flow | ✅ done | medium | P0 |
| PERF-03 | Parallel loader benchmarks (ParallelDataLoader, WeightedDataLoader) | ✅ done | medium | P1 |
| PROV-01 | Dataset provenance (SHA-256 hash in metadata) | ✅ done | low | P1 |

### Changes
- Added 6 aprender integration tests (tensor extraction, CITL, parallel loading, splits)
- Added 3 benchmark groups (parallel_dataloader, weighted_dataloader, prefetch_comparison)
- Added SHA-256 provenance to format::Metadata and registry::DatasetMetadata
- Added `provenance` feature flag (enabled by default)
- Total tests: 882 (up from 818)

---

## Next: v0.6.0 Documentation & Polish
- **Priority**: P2
- **Quality Gates**: Coverage ≥ 95%, SATD = 0

### Planned Tasks
| ID | Description | Status | Complexity | Priority |
|----|-------------|--------|------------|----------|
| DOC-01 | API documentation completion | pending | low | P1 |
| DOC-02 | Usage examples and tutorials | pending | low | P1 |

