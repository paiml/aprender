# Glossary

This glossary defines key terms used throughout Trueno-DB documentation and codebase.

## Technical Terms

### Apache Arrow
A cross-language development platform for in-memory columnar data. Trueno-DB uses Arrow's columnar format for zero-copy GPU transfers and efficient SIMD operations. Arrow provides `RecordBatch`, `Array` types, and schema definitions.

### AVX / AVX-512 / AVX2
Advanced Vector Extensions (AVX) are SIMD instruction set extensions for x86 processors. AVX-512 provides 512-bit wide vector operations, AVX2 provides 256-bit, and original AVX provides 128-bit. Trueno (the SIMD library) auto-detects and uses the best available instruction set.

### Backend
The compute engine that executes queries. Trueno-DB supports three backends:
- **GPU Backend**: Executes compute shaders on graphics hardware via wgpu
- **SIMD Backend**: Uses CPU vector instructions via the trueno crate
- **Scalar Backend**: Fallback single-threaded CPU execution

### Cargo
Rust's package manager and build system. Used for dependency management, compilation, testing, and benchmarking.

### Cost-Based Selection
Algorithm for choosing between GPU and SIMD backends based on estimated execution cost. Uses the **5x rule**: GPU is selected when compute time exceeds 5x the PCIe transfer time.

### GPU (Graphics Processing Unit)
Specialized hardware for parallel computation. Trueno-DB uses GPUs for massively parallel aggregations (SUM, COUNT, MIN, MAX) achieving 50-100x speedup over CPU for large datasets (100M+ rows).

### JIT (Just-In-Time Compilation)
Runtime code generation technique. Trueno-DB's JIT compiler generates optimized WGSL compute shaders for **kernel fusion** (combining multiple operations like filter+sum into a single GPU pass).

### Kernel / Compute Kernel
A function executed on the GPU. Written in WGSL (WebGPU Shading Language). Examples: `sum_kernel`, `min_max_kernel`, `fused_filter_sum_kernel`.

### Kernel Fusion
Optimization technique that combines multiple operations into a single GPU kernel to eliminate intermediate memory writes. Example: `filter(x > 100) + sum(x)` becomes a single fused kernel instead of two separate passes.

### Morsel
A fixed-size chunk of data (default: 128 MB) processed as a unit. Enables **out-of-core execution** by preventing GPU VRAM exhaustion. Based on morsel-driven parallelism from HyPer/Umbra research.

### OLAP vs OLTP
- **OLAP** (Online Analytical Processing): Read-heavy workloads with complex aggregations. Trueno-DB is optimized for OLAP.
- **OLTP** (Online Transaction Processing): Write-heavy workloads with individual row updates. Not supported by Trueno-DB (append-only pattern enforced).

### Parquet
Apache Parquet is a columnar storage file format. Trueno-DB can load Parquet files directly into Arrow format for zero-copy processing.

### PCIe (Peripheral Component Interconnect Express)
High-speed bus connecting CPU to GPU. Data transfer over PCIe is the primary bottleneck for GPU databases. Trueno-DB's cost model accounts for PCIe bandwidth (typically 32 GB/s for Gen4 x16).

### Property-Based Testing
Testing approach that validates properties (invariants) rather than specific examples. Trueno-DB uses `proptest` to generate thousands of test cases automatically. Example: "top_k always returns ≤ k rows" for any input.

### SIMD (Single Instruction, Multiple Data)
CPU instruction set for parallel data processing. Processes multiple values in a single instruction (e.g., add 8 floats simultaneously). Trueno-DB uses SIMD via the `trueno` crate for 10-20x speedup over scalar code.

### TDG (Technical Debt Grading)
Code quality metric scoring files from 0-100 (grades F to A+). Trueno-DB maintains ≥85/100 (B+ minimum). Analyzes complexity, documentation, error handling, and test coverage.

### Top-K Selection
Algorithm for finding the K largest/smallest values without full sorting. Uses heap-based selection with O(N log K) complexity vs O(N log N) for full sort. Achieves 28.75x speedup for K=10, N=1M.

### Trueno
The SIMD compute library powering Trueno-DB's CPU fallback. Provides auto-vectorized implementations of aggregations with AVX-512/AVX2/SSE2 auto-detection. Separate crate: `trueno` on crates.io.

### VRAM (Video RAM)
GPU memory. Faster than system RAM but limited in size (typically 8-24 GB). Trueno-DB uses **morsel-driven iteration** to prevent VRAM exhaustion by processing data in chunks.

### WASM / WebAssembly
Binary instruction format for web browsers. Trueno-DB targets WASM for client-side analytics dashboards. Phase 4 roadmap includes WebGPU + SIMD128 support.

### WebGPU
Modern cross-platform GPU API for the web. Provides GPU access from browsers via JavaScript/WASM. Trueno-DB's GPU backend uses wgpu which abstracts Vulkan/Metal/DX12/WebGPU.

### WGSL (WebGPU Shading Language)
Shader language for WebGPU. Similar to GLSL/HLSL. Trueno-DB's compute kernels are written in WGSL for portability across all platforms.

### wgpu
Rust library implementing WebGPU API. Provides safe, cross-platform GPU access. Trueno-DB uses wgpu for GPU backend initialization, buffer management, and shader execution.

## Toyota Way Principles

Trueno-DB's development methodology is guided by Toyota Production System principles:

### Genchi Genbutsu (Go and See)
Going to the source to verify facts. In Trueno-DB:
- **Syscall tracing** (via renacer) verifies zero-copy claims
- **Benchmarks** validate all performance claims
- **Property tests** exhaustively verify correctness
- **Red team audits** challenge assumptions

Example: PCIe 5x rule is validated empirically with actual GPU hardware, not just theory.

### Jidoka (Built-in Quality)
Building quality into the process, not inspecting it in later. In Trueno-DB:
- **EXTREME TDD**: Write tests before implementation (RED → GREEN → REFACTOR)
- **Pre-commit hooks**: Quality gates run automatically on every commit
- **Mutation testing**: Validates test suite effectiveness
- **Property-based testing**: Generates thousands of test cases automatically

Example: 156 tests, 95.24% coverage, zero critical defects.

### Kaizen (Continuous Improvement)
Incremental, ongoing improvement. In Trueno-DB:
- **Algorithmic optimization**: O(N log K) Top-K vs O(N log N) full sort
- **Code quality**: TDG score improved from 81.8 to 96.3 (A+)
- **Documentation**: Iteratively filling gaps (roadmap, glossary, etc.)
- **Performance**: Regular benchmarking identifies regressions

Example: Eliminated 43 `.unwrap()` calls to prevent production panics.

### Muda (Waste Elimination)
Removing activities that don't add value. In Trueno-DB:
- **Feature-gating**: Optional `gpu` feature saves 3.8 MB and 45s compile time
- **Kernel fusion**: Eliminates intermediate memory writes
- **Zero-copy**: Arrow → GPU transfers avoid data duplication
- **Removing redundant code**: Deleted stub benchmarks

Example: JIT-compiled fused kernels eliminate wasted memory bandwidth.

### Poka-Yoke (Mistake Proofing)
Designing systems to prevent errors. In Trueno-DB:
- **OLAP-only API**: `append_batch()` enforced, `update_row()` deprecated
- **Type safety**: Rust's type system prevents null pointer errors
- **Schema validation**: Prevents incompatible batch appends
- **Bounded queues**: Prevent GPU memory exhaustion

Example: Attempting `update_row()` returns an error explaining OLAP vs OLTP.

## Acronyms

| Acronym | Full Form | Description |
|---------|-----------|-------------|
| API | Application Programming Interface | How users interact with Trueno-DB |
| CI/CD | Continuous Integration/Continuous Deployment | Automated testing and deployment pipeline |
| CPU | Central Processing Unit | General-purpose processor |
| DX12 | DirectX 12 | Microsoft's graphics API for Windows |
| FLOPs | Floating-Point Operations | Measure of computational work |
| GB/s | Gigabytes per second | Data transfer rate unit |
| GFLOP/s | Giga Floating-Point Operations per second | Compute throughput unit |
| GPU | Graphics Processing Unit | Specialized parallel processor |
| JIT | Just-In-Time | Runtime code compilation |
| LRU | Least Recently Used | Cache eviction policy |
| MVP | Minimum Viable Product | Initial feature-complete release |
| OLAP | Online Analytical Processing | Read-heavy analytics workloads |
| OLTP | Online Transaction Processing | Write-heavy transactional workloads |
| PCIe | Peripheral Component Interconnect Express | CPU-GPU interconnect |
| SATD | Self-Admitted Technical Debt | TODOs, FIXMEs in code |
| SIMD | Single Instruction, Multiple Data | Parallel CPU instructions |
| SQL | Structured Query Language | Declarative database query language |
| SSE2 | Streaming SIMD Extensions 2 | Intel SIMD instruction set (128-bit) |
| TDD | Test-Driven Development | Write tests before code |
| TDG | Technical Debt Grading | Code quality metric (0-100 score) |
| VRAM | Video RAM | GPU memory |
| WASM | WebAssembly | Binary format for web browsers |
| WGSL | WebGPU Shading Language | Shader programming language |

## Related Resources

- **[Roadmap](../dev/roadmap.md)**: Project development phases and milestones
- **[Contributing](../dev/contributing.md)**: How to contribute to Trueno-DB
- **[API Documentation](./api-docs.md)**: Complete API reference
- **[Examples](../dev/examples.md)**: Usage examples and tutorials
- **[Research Papers](../academic/research-papers.md)**: Academic foundations

## See Also

For implementation details, see:
- **Architecture**: [System Overview](../architecture/system-overview.md)
- **Performance**: [Benchmarking Guide](../performance/benchmarking.md)
- **Quality**: [TDG Score](../quality/tdg-score.md), [Code Coverage](../quality/code-coverage.md)
- **Toyota Way**: [Jidoka](../toyota/jidoka.md), [Kaizen](../toyota/kaizen.md)
