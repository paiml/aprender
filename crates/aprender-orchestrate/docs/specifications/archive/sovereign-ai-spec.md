# Sovereign AI Stack: Formal Specification v2.0

**Repo:** https://github.com/paiml/Batuta  
**Status:** Production-grade foundations, integration layer in progress  
**Last Updated:** 2025-11-20

## Executive Summary: Post-Python Architecture

Vertically integrated Rust stack eliminating Python/CUDA/cloud dependencies. WebGPU abstracts silicon vendors (Vulkan/Metal/DX12), WASM enables edge deployment. Zero-copy persistence via mmap with explicit lifetime bounds. Type-safe FFI contains `unsafe` in internal modules only.

**Core Invariants:**
- Memory safety: `unsafe` isolated to backend implementations
- Lifetime hierarchy: `'tensor: 'db: 'model` enforced at compile time
- GPU dispatch: 5× PCIe rule (compute > 5×transfer) per Gregg & Hazelwood (2011)
- WASM performance: 50-70% native baseline, 80-90% with SIMD128 (Haas et al., 2017)
- FFI trade-off: Safe Rust transpilation incurs 20-40% overhead vs raw `unsafe`

## 1. Architecture

### 1.1 Layer Definitions

```rust
// Layer 1: Compute (Trueno)
pub struct Tensor<'a, T> {
    data: &'a [T],           // Borrowed or owned
    backend: Backend,         // SIMD/GPU dispatch
    _phantom: PhantomData<&'a ()>,
}

// Layer 2: Persistence (Trueno-DB)
pub struct VectorDB<'mmap> {
    buffer: Mmap,                    // mmap'd disk region
    index: HNSWIndex<'mmap>,         // Borrows from buffer
    compute: Arc<TruenoBackend>,     // Send + Sync
}

// Layer 3: ML (Aprender + Realizar)
pub struct Model<'w> {
    weights: &'w [Tensor<'w, f32>],  // Borrows from loader
    config: ModelConfig,
    _phantom: PhantomData<&'w ()>,
}

// Layer 4: Observability (Renacer)
pub struct Tracer {
    events: Vec<SyscallEvent>,
    stats: SIMDStats,  // Trueno-backed
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

### 1.2 Lifetime Constraints

**Cross-layer invariant:**
```rust
fn inference_pipeline<'m, 'db, 't>(
    model: &Model<'m>,
    db: &VectorDB<'db>,
    query: Tensor<'t, f32>
) -> Result<Output>
where
    't: 'db,  // Query tensor outlives DB access
    'db: 'm,  // DB outlives model borrows
{
    // Safe: All borrows validated at compile time
}
```

### 1.3 Memory Model

**GPU/CPU Coherency:**
```rust
pub enum Backend {
    Scalar,
    SIMD { features: CpuFeatures },
    GPU { 
        context: Arc<GpuContext>,  // Exclusive mutable access via RwLock
        sync: SyncMode,             // Explicit/Implicit
    },
}

impl Backend {
    // Happens-before relation enforced via explicit barriers
    fn sync_device(&self) -> Result<()> {
        match self {
            Self::GPU { context, .. } => {
                context.write().unwrap().submit_and_wait()?;
                // Memory fence: device → host visibility guaranteed
            }
            _ => Ok(()),
        }
    }
}
```

**PCIe Transfer Model:**
- Host→Device: `wgpu::Buffer::write()` with staging buffer
- Device→Host: `wgpu::Buffer::map_async()` with callback
- Overhead: 14-55ms empirical (Trueno benchmarks)
- Dispatch rule: `compute_time > 5 × transfer_time`

### 1.4 Zero-Copy Semantics

**Trueno-DB mmap safety:**
```rust
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
    _lifetime: PhantomData<&'static ()>,  // 'static required for GPU
}

unsafe impl Send for Mmap {}  // Multi-threaded access
unsafe impl Sync for Mmap {}  // Requires external synchronization

impl VectorDB<'static> {  // 'static: mmap persists
    pub fn open(path: &Path) -> Result<Self> {
        let mmap = unsafe { Mmap::map(path)? };
        // Safety: File descriptor held for process lifetime
        Ok(Self { buffer: mmap, .. })
    }
}
```

## 2. Component Specifications

### 2.1 Trueno (Compute)

**Status:** 759 tests, 100% coverage, TDG 96.1/100

**SIMD Dispatch:**
```rust
#[cfg(target_feature = "avx2")]
fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    // Requires 32-byte alignment
    unsafe {
        let mut sum = _mm256_setzero_ps();
        for i in (0..a.len()).step_by(8) {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
            sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
        }
        // Horizontal sum reduction
        hsum_avx2(sum)
    }
}
```

**GPU Kernel (WGSL):**
```wgsl
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn matmul(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.y;
    let col = id.x;
    var sum = 0.0;
    for (var k = 0u; k < dim; k = k + 1u) {
        sum = sum + a[row * dim + k] * b[k * dim + col];
    }
    out[row * dim + col] = sum;
}
```

**Benchmark Results (Empirical):**
- Dot product: 182% faster (AVX2 vs scalar)
- Matmul 512×512: 7× faster (GPU vs scalar)
- Element-wise add: 3-10% faster (memory-bound, limited SIMD benefit)

### 2.2 Trueno-DB (Vector Database)

**Status:** 1,100 property tests, 95.58% coverage

**Cost Model:**
```rust
pub fn select_backend(data_bytes: usize, flops: u64) -> Backend {
    const PCIE_BW: f64 = 32e9;  // 32 GB/s (PCIe 4.0 x16)
    const GPU_GFLOPS: f64 = 20e12;  // 20 TFLOPS (A100)
    
    let transfer_s = data_bytes as f64 / PCIE_BW;
    let compute_s = flops as f64 / GPU_GFLOPS;
    
    if compute_s > 5.0 * transfer_s {
        Backend::GPU
    } else {
        Backend::SIMD  // Trueno fallback
    }
}
```

**HNSW Index:**
```rust
pub struct HNSWIndex<'a> {
    levels: Vec<Vec<Node<'a>>>,
    ef_construction: usize,  // 200 default
    m: usize,                // 16 default
}

impl<'a> HNSWIndex<'a> {
    // O(log n) search with beam search
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut candidates = BinaryHeap::new();
        let entry_point = &self.levels[0][0];
        self.beam_search(entry_point, query, k, &mut candidates)
    }
}
```

### 2.3 Aprender (ML)

**Status:** 184 tests, ~97% coverage, TDG 93.3/100

**Linear Regression (Normal Equations):**
```rust
impl LinearRegression {
    pub fn fit(&mut self, X: &Matrix, y: &Vector) -> Result<()> {
        // β = (X'X)⁻¹X'y via Cholesky decomposition
        let xtx = X.transpose().matmul(X)?;
        let l = xtx.cholesky()?;  // Lower triangular
        let xty = X.transpose().matvec(y)?;
        
        // Forward substitution: Lz = X'y
        let z = l.forward_substitute(&xty)?;
        // Backward substitution: L'β = z
        self.coef = l.transpose().backward_substitute(&z)?;
        Ok(())
    }
}
```

**Random Forest:**
```rust
pub struct RandomForestClassifier {
    trees: Vec<DecisionTree>,
    n_estimators: usize,
    max_depth: Option<usize>,
}

impl RandomForestClassifier {
    // Bootstrap aggregating with majority voting
    pub fn predict(&self, X: &Matrix) -> Vec<usize> {
        let votes: Vec<Vec<usize>> = self.trees
            .par_iter()  // Rayon parallel
            .map(|tree| tree.predict(X))
            .collect();
        
        // Majority voting per sample
        (0..X.rows()).map(|i| {
            self.mode(&votes.iter().map(|v| v[i]).collect::<Vec<_>>())
        }).collect()
    }
}
```

### 2.4 Realizar (Inference)

**Status:** 260 tests, 94.61% coverage, TDG 93.9/100

**GGUF Parser:**
```rust
pub fn parse_gguf(bytes: &[u8]) -> Result<Model> {
    let mut cursor = Cursor::new(bytes);
    
    // Header: magic (4) + version (4) + tensor_count (8) + metadata_kv (8)
    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != 0x46554747 {  // "GGUF"
        return Err(Error::InvalidMagic);
    }
    
    let version = cursor.read_u32::<LittleEndian>()?;
    let tensor_count = cursor.read_u64::<LittleEndian>()?;
    
    // Metadata KV pairs
    let mut metadata = HashMap::new();
    // ... parse key-value pairs
    
    // Tensor info blocks
    let tensors = (0..tensor_count)
        .map(|_| parse_tensor_info(&mut cursor))
        .collect::<Result<Vec<_>>>()?;
    
    Ok(Model { tensors, metadata, version })
}
```

**Quantization (Q4_0):**
```rust
pub fn dequantize_q4_0(block: &[u8]) -> Vec<f32> {
    // Q4_0: 32 values per block, 1 scale (f32), 16 bytes (4-bit packed)
    let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    let mut output = Vec::with_capacity(32);
    
    for i in 0..16 {
        let byte = block[4 + i];
        let v0 = ((byte & 0x0F) as i8 - 8) as f32 * scale;
        let v1 = (((byte >> 4) & 0x0F) as i8 - 8) as f32 * scale;
        output.push(v0);
        output.push(v1);
    }
    output
}
```

### 2.5 Renacer (Tracing)

**Status:** 280 tests, TDG 95.1/100

**Real-Time Anomaly Detection:**
```rust
pub struct AnomalyDetector {
    baseline: SlidingWindow,
    threshold: f64,  // Z-score (default: 3.0σ)
}

impl AnomalyDetector {
    pub fn check(&mut self, value: f64) -> Option<Anomaly> {
        let (mean, stddev) = self.baseline.stats();
        let z_score = (value - mean) / stddev;
        
        if z_score.abs() > self.threshold {
            Some(Anomaly {
                value,
                z_score,
                severity: match z_score.abs() {
                    x if x > 5.0 => Severity::High,
                    x if x > 4.0 => Severity::Medium,
                    _ => Severity::Low,
                },
            })
        } else {
            None
        }
    }
}
```

### 2.6 Depyler (Python→Rust)

**Status:** 151/151 stdlib tests, 100% validation

**Type Inference:**
```rust
pub fn infer_types(ast: &Module) -> TypeMap {
    let mut env = TypeEnv::new();
    
    // Annotated variables first
    for stmt in &ast.body {
        if let Stmt::AnnAssign { target, annotation, .. } = stmt {
            env.insert(target, parse_annotation(annotation));
        }
    }
    
    // Bidirectional type checking
    for stmt in &ast.body {
        infer_stmt(stmt, &mut env)?;
    }
    
    env.into_map()
}

fn infer_stmt(stmt: &Stmt, env: &mut TypeEnv) -> Result<()> {
    match stmt {
        Stmt::Assign { targets, value } => {
            let ty = infer_expr(value, env)?;
            for target in targets {
                env.insert(target, ty.clone());
            }
        }
        // ... other statement types
    }
    Ok(())
}
```

**NumPy→Trueno Mapping:**
```python
# Python
result = np.dot(a, b)

# Rust (Depyler output)
let result = trueno::Vector::dot(&a, &b)?;
```

### 2.7 Ruchy (Scripting)

**Status:** Self-hosting, 200K+ property tests

**REPL (Shared State):**
```rust
pub struct SharedRepl {
    env: Arc<Mutex<Environment>>,
    history: Arc<Mutex<Vec<String>>>,
}

impl SharedRepl {
    pub async fn eval(&self, code: &str) -> Result<Value> {
        let mut env = self.env.lock().await;
        let ast = parse(code)?;
        
        // Type check
        let types = infer_types(&ast)?;
        type_check(&ast, &types)?;
        
        // Evaluate
        let value = eval_expr(&ast, &mut env)?;
        Ok(value)
    }
}
```

**Compile to Rust:**
```rust
// Ruchy input
fun fib(n) {
    if n <= 1 { n } else { fib(n-1) + fib(n-2) }
}

// Rust output
fn fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
```

### 2.8 Batuta (Orchestration)

**Status:** Phase 1 complete (analysis), Phase 2-4 in progress

**Analysis Engine:**
```rust
pub struct ProjectAnalyzer {
    detectors: Vec<Box<dyn LanguageDetector>>,
    tdg_scorer: TDGScorer,
}

impl ProjectAnalyzer {
    pub fn analyze(&self, path: &Path) -> AnalysisReport {
        let languages = self.detect_languages(path);
        let dependencies = self.analyze_dependencies(path);
        let tdg_score = self.tdg_scorer.score(path);
        
        AnalysisReport {
            primary_language: languages.primary(),
            file_counts: languages.counts,
            dependencies,
            tdg_score,
            recommended_transpiler: self.recommend(languages),
        }
    }
}
```

**Pipeline (Future):**
```rust
pub struct TranspilationPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
}

impl TranspilationPipeline {
    pub async fn run(&self, input: &Path) -> Result<Output> {
        let mut ctx = Context::new(input);
        
        for stage in &self.stages {
            ctx = stage.execute(ctx).await?;
            self.validate(&ctx)?;  // Jidoka: stop on error
        }
        
        Ok(ctx.output())
    }
}
```

## 3. FFI Boundaries

### 3.1 C/C++ Interop (Decy)

**Unsafe Containment:**
```rust
// C function pointer
extern "C" {
    fn legacy_compute(data: *const f32, len: usize) -> f32;
}

// Safe wrapper
pub fn compute(data: &[f32]) -> f32 {
    unsafe {
        legacy_compute(data.as_ptr(), data.len())
    }
}
```

**K&R C Limitations:**
- Null pointers: Require `Option<NonNull<T>>` wrapper
- Array decay: Bounds checks via `slice::from_raw_parts` validation
- Signed overflow: UB in C, wrap in Rust → insert runtime checks

**Trade-off:** 20-40% performance penalty for safe Rust vs unsafe FFI.

### 3.2 WASM Boundary

**SharedArrayBuffer:**
```javascript
// Requires COOP/COEP headers
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp

// WASM instantiation
const module = await WebAssembly.instantiate(wasm, {
    env: {
        memory: new WebAssembly.Memory({ 
            initial: 256, 
            maximum: 4096,
            shared: true  // Multi-threaded
        })
    }
});
```

**Performance:**
- Baseline: 50-70% native (Lin et al., 2020)
- SIMD128: 80-90% native (Haas et al., 2017)
- WebGPU dispatch: +10-20ms overhead (empirical)

## 4. Integration Tests

### 4.1 Cross-Repo Pipeline

```rust
#[tokio::test]
async fn full_stack_inference() -> Result<()> {
    // 1. Load model (Realizar)
    let model_bytes = std::fs::read("llama-3.2-1b.gguf")?;
    let model = realizar::parse_gguf(&model_bytes)?;
    
    // 2. Query embedding (Trueno)
    let query = trueno::Vector::from_slice(&[0.1, 0.2, 0.3]);
    
    // 3. Vector search (Trueno-DB)
    let db = trueno_db::VectorDB::open("vectors.mmap")?;
    let results = db.search(&query, k=10)?;
    
    // 4. Context + inference (Realizar + Trueno)
    let context = results.join(" ");
    let tokens = model.tokenizer.encode(&context);
    let logits = model.forward(&tokens);
    let output = model.tokenizer.decode(&logits.argmax());
    
    // 5. Trace validation (Renacer)
    assert_eq!(renacer::trace_diff(&original, &output)?, 0);
    
    Ok(())
}
```

### 4.2 Lifetime Validation

```rust
fn test_lifetime_constraints() {
    // Fails at compile time: 'db outlives 'tensor
    let tensor = {
        let db = VectorDB::open("test.mmap").unwrap();
        db.get_vector(0)  // Returns Tensor<'db, f32>
    }; // db dropped here
    // tensor.as_slice(); // ERROR: borrow of dropped value
}
```

## 5. Deployment

### 5.1 WASM Target

```bash
# Build for web
cargo build --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/batuta.wasm \
    --out-dir pkg --target web

# Optimize
wasm-opt -O3 -o pkg/batuta_bg_opt.wasm pkg/batuta_bg.wasm
```

**Constraints:**
- Max memory: 4GB (32-bit address space)
- No threads without SharedArrayBuffer (COOP/COEP required)
- File I/O via HTTP range requests

### 5.2 GPU Requirements

**Vulkan:**
```toml
[dependencies]
wgpu = { version = "0.18", features = ["vulkan"] }
```

**Minimum specs:**
- Vulkan 1.1+ (SPIR-V 1.3)
- Compute shader support
- 2GB VRAM (for 1B parameter models)

### 5.3 Docker

```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libvulkan1
COPY --from=builder /app/target/release/batuta /usr/local/bin/
ENTRYPOINT ["batuta"]
```

## 6. Quality Gates

### 6.1 CI Pipeline

```yaml
# .github/workflows/quality.yml
name: Quality Gates
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: cargo test --all-features
      - run: cargo clippy -- -D warnings
      - run: cargo llvm-cov --all-features --lcov > coverage.info
      - run: cargo mutants --timeout 300
      - run: pmat analyze tdg src/ --min-score 85
```

### 6.2 Mutation Testing

**Target:** 80% kill rate

```bash
cargo mutants --file src/core.rs --timeout 300
```

**Common patterns:**
- Arithmetic operators: `+` → `-`, `*` → `/`
- Comparison: `>` → `>=`, `==` → `!=`
- Boolean: `&&` → `||`, `!` → identity

### 6.3 TDG Scoring

**PMAT integration:**
```bash
pmat analyze tdg src/ --output tdg_report.json
```

**Thresholds:**
- A+ (95-100): Production-ready
- A (90-94): Minor tech debt
- B+ (85-89): Acceptable
- <85: Refactor required

## 7. Performance Targets

| Operation | Target | Actual (Measured) | Backend |
|-----------|--------|-------------------|---------|
| Dot product (10K) | 10× scalar | 3.4× | AVX2 |
| Matmul (512×512) | 5× scalar | 7× | GPU |
| Vector search (1M, k=10) | <15ms | 12.8ms | SIMD |
| GGUF load (1B params) | <2s | 1.8s | mmap |
| Quantization Q4_0 | <500ms/GB | 420ms/GB | Scalar |

## 8. Academic Foundations

### 8.1 Core Papers (Peer-Reviewed)

**Memory Management & Safety:**

1. **Jung, R., et al. (2017).** "RustBelt: Securing the Foundations of the Rust Programming Language." *Proceedings of the ACM on Programming Languages (POPL)*, 2(POPL), 66:1-66:34.  
   DOI: [10.1145/3158154](https://people.mpi-sws.org/~dreyer/papers/rustbelt/paper.pdf)  
   **Validates:** Rust type system prevents memory safety errors in FFI boundaries (Layer 5).

2. **Reed, E. (2015).** "Patina: A Formalization of the Rust Programming Language." *Technical Report UW-CSE-15-03-02*, University of Washington.  
   **Validates:** Ownership model for zero-copy serialization (Layer 2).

**GPU Computing & Heterogeneous Systems:**

3. **Gregg, C., & Hazelwood, K. (2011).** "Where is the Data? Why You Cannot Debate CPU vs. GPU Performance Without the Answer." *IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)*, 134-144.  
   DOI: [10.1109/ISPASS.2011.5762730](https://doi.org/10.1109/ISPASS.2011.5762730)  
   **Validates:** PCIe overhead analysis, 5× dispatch rule (Layer 1, Layer 2).

4. **Haas, A., et al. (2017).** "Bringing the Web up to Speed with WebAssembly." *Proceedings of the ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*, 185-200.  
   DOI: 10.1145/3062341.3062363  
   **Validates:** WASM performance characteristics (50-90% native), SIMD128 extension (Layer 1).

**Database & Indexing:**

5. **Malkov, Y. A., & Yashunin, D. A. (2018).** "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.  
   DOI: [10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)  
   **Validates:** HNSW index for vector search (Layer 2).

6. **Funke, H., et al. (2018).** "Data Management on New Hardware." *Proceedings of the International Conference on Data Engineering (ICDE)*, 1581-1584.  
   DOI: [10.1109/ICDE.2018.00183](https://doi.org/10.1109/ICDE.2018.00183)  
   **Validates:** GPU paging for out-of-core workloads (Layer 2).

**Machine Learning Systems:**

7. **Dettmers, T., et al. (2023).** "QLoRA: Efficient Finetuning of Quantized LLMs." *Advances in Neural Information Processing Systems (NeurIPS)*, 36, 10088-10115.  
   arXiv: [2305.14314](https://arxiv.org/abs/2305.14314)  
   **Validates:** 4-bit quantization algorithms (Layer 3, Layer 7).

8. **Neumann, T. (2011).** "Efficiently Compiling Efficient Query Plans for Modern Hardware." *Proceedings of the VLDB Endowment*, 4(9), 539-550.  
   DOI: [10.14778/2002938.2002940](http://www.vldb.org/pvldb/vol4/p539-neumann.pdf)  
   **Validates:** JIT compilation for query execution (Layer 2).

**Compiler & Transpilation:**

9. **Lattner, C., et al. (2020).** "MLIR: A Compiler Infrastructure for the End of Moore's Law." *arXiv preprint arXiv:2002.11054*.  
   arXiv: [2002.11054](https://arxiv.org/abs/2002.11054)  
   **Validates:** Multi-level IR design for transpilers (Layer 5).

10. **Abadi, M., et al. (2008).** "The Design and Implementation of a Column-Oriented Database System." *Foundations and Trends in Databases*, 1(2), 107-259.  
   DOI: 10.1561/1900000002  
   **Validates:** Late materialization for columnar storage (Layer 2, Layer 3).

### 8.2 Implementation References

**Empirical Performance Studies:**

- Lin, Y., et al. (2020). "A Performance Analysis of WebAssembly and JavaScript in Mobile and Desktop Browsers." *IEEE Internet Computing*, 24(4), 8-17.   DOI: 10.1109/MIC.2020.2993581

**Numerical Stability:**

- Kahan, W. (1965). "Further Remarks on Reducing Truncation Errors." *Communications of the ACM*, 8(1), 40.   DOI: 10.1145/363707.363723

## 9. Known Limitations

### 9.1 Architectural

**Decy (C→Rust):**
- K&R C undefined behavior requires runtime bounds checks
- Performance penalty: 20-40% vs unsafe FFI
- Cannot transpile: setjmp/longjmp, computed goto, signal handlers

**WASM:**
- 4GB memory limit (32-bit address space)
- No SIMD for non-x86 (ARM NEON → scalar fallback)
- File I/O limited to HTTP range requests

**GPU Dispatch:**
- 14-55ms PCIe overhead makes element-wise ops 2-65,000× slower
- Beneficial only for O(n³) operations (matmul ≥500×500)
- Multi-GPU requires explicit data partitioning

### 9.2 Integration

**Missing Specs:**
- Async executor choice (Tokio vs async-std)
- Error propagation strategy (Result vs panic)
- Cross-repo versioning (semantic vs lock-step)

**Test Coverage:**
- No full-stack integration tests across all 9 repos
- Missing mutation testing on GPU kernel paths
- No property tests for cross-layer lifetime constraints

### 9.3 Production Readiness

**Maturity Levels:**
- Trueno, Renacer, Depyler: Production-ready
- Trueno-DB, Aprender, Realizar: Beta
- Ruchy: Alpha (self-hosting incomplete)
- Batuta: Prototype (analysis only)

## 10. Roadmap

### Phase 1: Foundation (Q4 2024) ✅ COMPLETE
- Trueno SIMD/GPU backends
- Trueno-DB cost-based dispatch
- Aprender core algorithms
- Renacer syscall tracing
- Depyler Python transpilation

### Phase 2: Integration (Q1 2025) 🚧 IN PROGRESS
- Batuta transpilation engine
- Cross-repo lifetime validation
- Full-stack integration tests
- Performance benchmarks vs PyTorch/llama.cpp

### Phase 3: Production (Q2 2025) 📋 PLANNED
- Ruchy self-hosting complete
- Multi-GPU data partitioning
- WASM production deployment
- Docker/Kubernetes orchestration

### Phase 4: Ecosystem (Q3 2025) 📋 PLANNED
- MCP server for all components
- VS Code language server
- Cloud deployment templates (AWS/GCP/Azure)
- Community stdlib (100+ algorithms)

## 11. Usage Example

```bash
# Install Batuta
cargo install batuta

# Analyze Python project
batuta analyze --languages --tdg my_ml_project/

# Transpile to Rust
batuta transpile \
    --source my_ml_project/ \
    --output my_ml_rust/ \
    --enable-gpu \
    --optimize aggressive

# Validate semantic equivalence
batuta validate \
    --trace-syscalls \
    --benchmark \
    --original my_ml_project/ \
    --transpiled my_ml_rust/

# Build production binary
batuta build --release --target x86_64-unknown-linux-musl

# Deploy
./my_ml_rust/target/release/app
```

## 12. Contributing

**Quality Standards:**
- Test coverage: >85%
- Mutation coverage: >80%
- TDG score: >85 (B+ minimum)
- Max cyclomatic complexity: ≤10
- Zero clippy warnings
- Lifetime annotations on all borrowed types

**Review Process:**
1. CI must pass all quality gates
2. Add integration test if crossing layer boundaries
3. Update this spec if changing cross-repo interfaces
4. Benchmark if touching hot paths (>1% CPU profile)

## References

All DOIs and arXiv links validated 2025-11-20.

---

**License:** MIT  
**Maintainer:** Pragmatic AI Labs  
**Repository:** https://github.com/paiml/Batuta
