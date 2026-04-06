# References

External resources, academic papers, and related projects that inform Trueno-DB's design and implementation.

## Academic Papers

Foundational research papers cited in Trueno-DB's architecture and implementation.

### Database Systems

**MonetDB/X100: Vectorized Query Execution**
- **Authors**: Peter Boncz, Marcin Zukowski, Niels Nes
- **Conference**: CIDR 2005 (Conference on Innovative Data Systems Research)
- **Summary**: Introduced vectorized query execution with columnar data processing
- **Relevance**: Foundation for Trueno-DB's SIMD backend and chunked processing
- **URL**: [cidrdb.org/cidr2005/papers/P19.pdf](http://cidrdb.org/cidr2005/papers/P19.pdf)

**Morsel-Driven Parallelism: NUMA-Aware Query Evaluation**
- **Authors**: Viktor Leis, Peter Boncz, Alfons Kemper, Thomas Neumann
- **Conference**: SIGMOD 2014
- **Summary**: Morsel-based parallelism for NUMA-aware execution (foundation for out-of-core GPU processing)
- **Relevance**: Trueno-DB's 128 MB morsel size prevents VRAM exhaustion
- **DOI**: 10.1145/2588555.2610507

**Volcano Optimizer: Cost-Based Query Optimization**
- **Authors**: Goetz Graefe
- **Journal**: IEEE Data Engineering Bulletin, 1993
- **Summary**: Cascading query optimizer with cost-based plan selection
- **Relevance**: Trueno-DB's backend dispatcher uses cost-based selection (5x rule)
- **URL**: [sites.computer.org/debull/93dec-cd.pdf](http://sites.computer.org/debull/93dec-cd.pdf)

### GPU Databases

**HeavyDB (formerly MapD): GPU Database Patterns**
- **Authors**: Todd Mostak
- **Conference**: SIGMOD 2017
- **Summary**: GPU-accelerated database with kernel fusion and parallel aggregations
- **Relevance**: Trueno-DB's JIT compiler and fused filter+sum kernels
- **DOI**: 10.1145/3035918.3056100

**Kernel Fusion for GPU Databases**
- **Authors**: Mark Harris (NVIDIA)
- **Year**: 2007
- **Summary**: Parallel reduction algorithms and kernel fusion techniques
- **Relevance**: Trueno-DB's 2-stage reduction in MIN/MAX kernels
- **URL**: [developer.nvidia.com/gpugems/gpugems3](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

### Columnar Storage

**Apache Arrow: Columnar In-Memory Format**
- **Authors**: Wes McKinney, et al.
- **Conference**: VLDB 2020
- **Summary**: Language-agnostic columnar memory format for zero-copy data interchange
- **Relevance**: Trueno-DB's storage layer and GPU transfer format
- **URL**: [arrow.apache.org](https://arrow.apache.org/)
- **Paper**: [vldb.org/pvldb/vol13/p2872-sindhu.pdf](https://www.vldb.org/pvldb/vol13/p2872-sindhu.pdf)

## Related Projects

### Analytical Databases

**DuckDB**
- **Description**: In-process analytical database with vectorized execution
- **Language**: C++
- **Relevance**: Competitive benchmark baseline for OLAP queries
- **URL**: [duckdb.org](https://duckdb.org/)
- **GitHub**: [github.com/duckdb/duckdb](https://github.com/duckdb/duckdb)

**Polars**
- **Description**: Lightning-fast DataFrame library with query optimization
- **Language**: Rust
- **Relevance**: Rust ecosystem leader in columnar analytics
- **URL**: [pola.rs](https://pola.rs/)
- **GitHub**: [github.com/pola-rs/polars](https://github.com/pola-rs/polars)

**SQLite**
- **Description**: Self-contained SQL database engine (row-oriented)
- **Language**: C
- **Relevance**: OLTP baseline for performance comparisons
- **URL**: [sqlite.org](https://sqlite.org/)
- **Architecture**: [sqlite.org/arch.html](https://www.sqlite.org/arch.html)

### GPU Computing

**wgpu**
- **Description**: Safe Rust implementation of WebGPU API
- **Language**: Rust
- **Relevance**: Trueno-DB's GPU backend (Vulkan/Metal/DX12/WebGPU abstraction)
- **URL**: [wgpu.rs](https://wgpu.rs/)
- **GitHub**: [github.com/gfx-rs/wgpu](https://github.com/gfx-rs/wgpu)

**WebGPU Specification**
- **Description**: Modern cross-platform GPU API standard
- **Organization**: W3C GPU for the Web Community Group
- **Relevance**: Enables browser deployment (Phase 4)
- **URL**: [gpuweb.github.io/gpuweb](https://gpuweb.github.io/gpuweb/)

**WGSL Specification**
- **Description**: WebGPU Shading Language (shader programming)
- **Organization**: W3C
- **Relevance**: Language for Trueno-DB's compute kernels and JIT compiler
- **URL**: [w3.org/TR/WGSL](https://www.w3.org/TR/WGSL/)

### SIMD Libraries

**trueno**
- **Description**: SIMD-accelerated aggregations for Rust (AVX-512/AVX2/SSE2)
- **Language**: Rust
- **Relevance**: Trueno-DB's CPU fallback backend
- **Version**: 0.6.0 (dependency in Cargo.toml)
- **GitHub**: [github.com/paiml/trueno](https://github.com/paiml/trueno)

**Intel Intrinsics Guide**
- **Description**: Complete reference for x86 SIMD instructions
- **Relevance**: Understanding AVX-512/AVX2/SSE2 performance characteristics
- **Search**: "Intel Intrinsics Guide" (official documentation)

## Rust Ecosystem

### Core Language

**The Rust Programming Language (Book)**
- **Authors**: Steve Klabnik, Carol Nichols
- **URL**: [doc.rust-lang.org/book](https://doc.rust-lang.org/book/)
- **Relevance**: Foundation for understanding Trueno-DB's implementation

**Rust Performance Book**
- **URL**: [nnethercote.github.io/perf-book](https://nnethercote.github.io/perf-book/)
- **Relevance**: Optimization techniques applied in Trueno-DB

### Testing & Benchmarking

**Criterion.rs**
- **Description**: Statistical benchmarking framework
- **Relevance**: All Trueno-DB performance claims validated with Criterion
- **URL**: [github.com/bheisler/criterion.rs](https://github.com/bheisler/criterion.rs)

**proptest**
- **Description**: Property-based testing framework
- **Relevance**: Trueno-DB's 11 property tests (1,100 scenarios)
- **URL**: [github.com/proptest-rs/proptest](https://github.com/proptest-rs/proptest)

**cargo-llvm-cov**
- **Description**: Code coverage tool using LLVM instrumentation
- **Relevance**: Trueno-DB's 95.24% coverage measurement
- **URL**: [github.com/taiki-e/cargo-llvm-cov](https://github.com/taiki-e/cargo-llvm-cov)

### Development Tools

**Renacer**
- **Description**: Modern syscall tracer with DWARF support (Rust `strace` replacement)
- **Relevance**: Validates zero-copy operations and GPU memory mapping
- **URL**: [github.com/paiml/renacer](https://github.com/paiml/renacer)

**mdBook**
- **Description**: Rust documentation book generator
- **Relevance**: Powers Trueno-DB's documentation site
- **URL**: [rust-lang.github.io/mdBook](https://rust-lang.github.io/mdBook/)

## Toyota Way & Quality

### Toyota Production System

**The Toyota Way: 14 Management Principles**
- **Author**: Jeffrey Liker
- **ISBN**: 978-0071392310
- **Relevance**: Trueno-DB's development methodology (Jidoka, Kaizen, Genchi Genbutsu)

**Toyota Production System: Beyond Large-Scale Production**
- **Author**: Taiichi Ohno
- **ISBN**: 978-0915299140
- **Relevance**: Foundation for EXTREME TDD and quality-first development

### Software Quality

**Mutation Testing: A Comprehensive Survey**
- **Authors**: Yue Jia, Mark Harman
- **Journal**: IEEE Transactions on Software Engineering, 2011
- **Relevance**: Trueno-DB's mutation testing for test suite validation
- **URL**: [ieeexplore.ieee.org/document/5487526](https://ieeexplore.ieee.org/document/5487526)

**Technical Debt Grading (TDG)**
- **Concept**: Code quality scoring from 0-100 (F to A+)
- **Relevance**: Trueno-DB maintains 96.3/100 (A+) across all modules
- **Factors**: Complexity, documentation, error handling, test coverage

## Community & Support

### Trueno-DB Resources

**Official Repository**
- **URL**: [github.com/paiml/trueno-db](https://github.com/paiml/trueno-db)
- **License**: MIT
- **Issues**: Report bugs and request features

**Documentation Book**
- **URL**: [paiml.github.io/trueno-db](https://paiml.github.io/trueno-db/)
- **Content**: Architecture, API docs, examples, Toyota Way methodology

**Crates.io Package**
- **Status**: In development (v0.2.1)
- **Planned**: Publication after Phase 3 completion

### Learning Resources

**GPU Programming Guides**
- **NVIDIA CUDA Handbook**: Understanding parallel reduction patterns
- **WebGPU Samples**: [webgpu.github.io/webgpu-samples](https://webgpu.github.io/webgpu-samples/)

**Database Internals (Book)**
- **Author**: Alex Petrov
- **ISBN**: 978-1492040347
- **Relevance**: Understanding database storage and query execution

**Designing Data-Intensive Applications**
- **Author**: Martin Kleppmann
- **ISBN**: 978-1449373320
- **Relevance**: System design principles for analytics databases

## Standards & Specifications

**SQL:2023 Standard**
- **Organization**: ISO/IEC
- **Relevance**: SQL query syntax support in Trueno-DB
- **URL**: [iso.org/standard/76583.html](https://www.iso.org/standard/76583.html)

**IEEE 754 Floating-Point Arithmetic**
- **Organization**: IEEE
- **Relevance**: Floating-point aggregation correctness (backend equivalence testing)
- **URL**: [ieeexplore.ieee.org/document/8766229](https://ieeexplore.ieee.org/document/8766229)

**Semantic Versioning 2.0.0**
- **URL**: [semver.org](https://semver.org/)
- **Relevance**: Trueno-DB's versioning scheme (currently v0.2.1)

## See Also

- **[Glossary](./glossary.md)** - Technical terminology and concepts
- **[API Documentation](./api-docs.md)** - Complete API reference
- **[License](./license.md)** - MIT License and dependency licenses
- **[Roadmap](../dev/roadmap.md)** - Development phases and milestones
- **[Contributing](../dev/contributing.md)** - How to contribute to Trueno-DB

## Citation

If you use Trueno-DB in academic research, please cite:

```bibtex
@software{trueno_db,
  title = {Trueno-DB: GPU-First Embedded Analytics Database},
  author = {{Pragmatic AI Labs}},
  year = {2025},
  url = {https://github.com/paiml/trueno-db},
  version = {0.2.1},
  license = {MIT}
}
```

## Feedback

Found a broken link or want to suggest a resource? Open an issue:
- **GitHub Issues**: [paiml/trueno-db/issues](https://github.com/paiml/trueno-db/issues)
- **Email**: info@paiml.com
