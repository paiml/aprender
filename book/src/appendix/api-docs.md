# API Documentation

Complete reference for Trueno-DB's public API. For detailed Rust documentation, run `cargo doc --open`.

## Quick Start

```rust
use trueno_db::{Database, Backend};

// Create database with GPU backend
let db = Database::builder()
    .backend(Backend::Gpu)
    .build()?;

// Load Parquet data
let storage = db.load_table("events", "data/events.parquet").await?;

// Execute SQL query
let result = db.query("SELECT category, SUM(value) FROM events GROUP BY category")
    .execute()
    .await?;
```

## Core Types

### Database

Main entry point for Trueno-DB operations.

```rust
pub struct Database {
    // Internal fields
}

impl Database {
    /// Create a new database builder
    pub fn builder() -> DatabaseBuilder;
}
```

**Example:**
```rust
let db = Database::builder()
    .backend(Backend::Gpu)
    .cache_size_mb(512)
    .build()?;
```

### DatabaseBuilder

Builder pattern for configuring `Database` instances.

```rust
pub struct DatabaseBuilder {
    // Configuration fields
}

impl DatabaseBuilder {
    /// Set the backend (GPU, SIMD, or Scalar)
    pub fn backend(self, backend: Backend) -> Self;

    /// Set morsel size for out-of-core execution (default: 128 MB)
    pub fn morsel_size_mb(self, size: usize) -> Self;

    /// Set query cache size (default: 100 queries)
    pub fn cache_size_mb(self, size: usize) -> Self;

    /// Build the database instance
    pub fn build(self) -> Result<Database>;
}
```

**Example:**
```rust
let db = Database::builder()
    .backend(Backend::Gpu)
    .morsel_size_mb(256)  // 256 MB morsels
    .cache_size_mb(1024)  // 1 GB cache
    .build()?;
```

### Backend

Compute backend selection.

```rust
pub enum Backend {
    /// GPU backend via wgpu (50-100x speedup for large datasets)
    Gpu,

    /// SIMD backend via trueno (10-20x speedup)
    Trueno(trueno::Backend),

    /// Scalar fallback (single-threaded CPU)
    Scalar,
}
```

**Backend Auto-Selection:**

The `BackendDispatcher` automatically selects backends based on:
- **Data size**: Must be >10 MB for GPU consideration
- **Arithmetic intensity**: Compute FLOPs must exceed 5x PCIe transfer cost

```rust
use trueno_db::backend::BackendDispatcher;

// Manual backend selection
let backend = BackendDispatcher::select(
    total_bytes,      // Data size in bytes
    estimated_flops,  // Estimated floating-point operations
);
```

**Example:**
```rust
// Explicit GPU backend
let db = Database::builder()
    .backend(Backend::Gpu)
    .build()?;

// SIMD with auto-detect (AVX-512 > AVX2 > SSE2)
let db = Database::builder()
    .backend(Backend::Trueno(trueno::Backend::Auto))
    .build()?;

// Scalar fallback (testing only)
let db = Database::builder()
    .backend(Backend::Scalar)
    .build()?;
```

## Storage API

### StorageEngine

Manages Apache Arrow columnar data storage.

```rust
pub struct StorageEngine {
    // Internal Arrow batches
}

impl StorageEngine {
    /// Load data from Parquet file
    pub fn load_parquet<P: AsRef<Path>>(path: P) -> Result<Self>;

    /// Get all batches (zero-copy reference)
    pub fn batches(&self) -> &[RecordBatch];

    /// Create morsel iterator for out-of-core execution
    pub fn morsels(&self) -> MorselIterator<'_>;

    /// Append new batch (OLAP pattern)
    pub fn append_batch(&mut self, batch: RecordBatch) -> Result<()>;
}
```

**Example:**
```rust
use trueno_db::storage::StorageEngine;

// Load from Parquet
let storage = StorageEngine::load_parquet("data/events.parquet")?;

// Iterate in 128 MB chunks (prevents VRAM OOM)
for morsel in storage.morsels() {
    process_morsel(morsel)?;
}

// Append new data (OLAP pattern)
let new_batch = create_batch()?;
storage.append_batch(new_batch)?;
```

### MorselIterator

Iterator for chunked data processing (prevents GPU VRAM exhaustion).

```rust
pub struct MorselIterator<'a> {
    // Internal state
}

impl Iterator for MorselIterator<'_> {
    type Item = RecordBatch;

    fn next(&mut self) -> Option<Self::Item>;
}
```

**Default morsel size:** 128 MB (optimized for most GPUs)

### GpuTransferQueue

Bounded queue for asynchronous GPU transfers.

```rust
pub struct GpuTransferQueue {
    // Internal channel
}

impl GpuTransferQueue {
    /// Create new transfer queue (max 2 in-flight batches)
    pub fn new() -> Self;

    /// Get sender for enqueuing batches
    pub fn sender(&self) -> tokio::sync::mpsc::Sender<RecordBatch>;
}
```

**Backpressure:** Queue blocks when 2 batches are already in flight.

## Top-K Selection

### TopKSelection Trait

High-performance heap-based Top-K selection (O(N log K) complexity).

```rust
pub trait TopKSelection {
    /// Select top K rows by column
    fn top_k(&self, column_index: usize, k: usize, order: SortOrder) -> Result<RecordBatch>;
}

impl TopKSelection for RecordBatch {
    // Implemented for Int32, Int64, Float32, Float64
}
```

**Supported types:**
- `Int32Array`
- `Int64Array`
- `Float32Array`
- `Float64Array`

**Example:**
```rust
use trueno_db::topk::{TopKSelection, SortOrder};

// Top 10 highest scores (descending)
let top10 = batch.top_k(2, 10, SortOrder::Descending)?;

// Bottom 5 lowest scores (ascending)
let bottom5 = batch.top_k(2, 5, SortOrder::Ascending)?;
```

### SortOrder

```rust
pub enum SortOrder {
    Ascending,   // Smallest first
    Descending,  // Largest first
}
```

**Performance:** 28.75x faster than full sort for K=10, N=1M rows.

## Query API

### QueryEngine

SQL query parser and planner.

```rust
pub struct QueryEngine {
    // Internal parser state
}

impl QueryEngine {
    /// Create new query engine
    pub fn new() -> Self;

    /// Parse SQL into QueryPlan
    pub fn parse(&self, sql: &str) -> Result<QueryPlan>;
}
```

**Supported SQL:**
- `SELECT` with column projections
- `WHERE` predicates (=, <, >, <=, >=, !=)
- `GROUP BY` with aggregations
- `ORDER BY` (ascending/descending)
- `LIMIT` clause

**Not supported:**
- `JOIN` operations (Phase 3)
- `HAVING` clause
- Subqueries
- Window functions (Phase 3)

**Example:**
```rust
use trueno_db::query::QueryEngine;

let engine = QueryEngine::new();

let plan = engine.parse(
    "SELECT category, SUM(value) as total
     FROM events
     WHERE timestamp > '2025-11-01'
     GROUP BY category
     ORDER BY total DESC
     LIMIT 10"
)?;
```

### QueryPlan

Parsed SQL query representation.

```rust
pub struct QueryPlan {
    pub columns: Vec<String>,
    pub where_clause: Option<FilterExpression>,
    pub group_by: Vec<String>,
    pub aggregates: Vec<AggregateFunction>,
    pub order_by: Option<(String, OrderDirection)>,
    pub limit: Option<usize>,
}
```

### AggregateFunction

```rust
pub enum AggregateFunction {
    Sum(String),    // Column name
    Avg(String),
    Count(String),
    Min(String),
    Max(String),
}
```

### OrderDirection

```rust
pub enum OrderDirection {
    Asc,
    Desc,
}
```

## GPU API

### GpuEngine

Low-level GPU compute engine (advanced users).

```rust
pub struct GpuEngine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuEngine {
    /// Initialize GPU engine
    pub async fn new() -> Result<Self>;

    /// Execute MIN aggregation (Int32 only)
    pub async fn min_i32(&self, data: &[i32]) -> Result<i32>;

    /// Execute MAX aggregation (Int32 only)
    pub async fn max_i32(&self, data: &[i32]) -> Result<i32>;

    /// Execute SUM aggregation (Int32 only)
    pub async fn sum_i32(&self, data: &[i32]) -> Result<i64>;

    /// Execute COUNT aggregation
    pub async fn count(&self, len: usize) -> Result<usize>;

    /// Execute fused filter+sum kernel (JIT-compiled)
    pub async fn fused_filter_sum(
        &self,
        data: &[i32],
        threshold: i32,
        operator: &str,  // "gt", "lt", "eq", "gte", "lte", "ne"
    ) -> Result<i64>;
}
```

**Kernel Fusion Example:**
```rust
// Instead of: filter(x > 100) → sum(x)  (2 passes)
let result = gpu.fused_filter_sum(data, 100, "gt").await?;  // 1 pass
```

### JitCompiler

Runtime WGSL shader generation for kernel fusion.

```rust
pub struct JitCompiler {
    // Internal shader cache
}

impl JitCompiler {
    /// Create new JIT compiler
    pub fn new() -> Self;

    /// Generate fused filter+sum shader
    pub fn generate_fused_filter_sum(
        &self,
        filter_threshold: i32,
        filter_op: &str,
    ) -> String;  // WGSL source

    /// Compile and cache shader
    pub async fn compile_fused_filter_sum(
        &self,
        device: &wgpu::Device,
        threshold: i32,
        op: &str,
    ) -> Result<Arc<wgpu::ShaderModule>>;
}
```

**Performance:** 1.5-2x speedup vs separate filter+sum kernels.

## Multi-GPU API

### MultiGpuManager

Manages multiple GPU devices for distributed execution.

```rust
pub struct MultiGpuManager {
    // GPU device pool
}

impl MultiGpuManager {
    /// Initialize multi-GPU manager
    pub fn new() -> Result<Self>;

    /// Get number of available GPUs
    pub fn device_count(&self) -> usize;

    /// Get device information
    pub fn devices(&self) -> &[GpuDeviceInfo];
}
```

### PartitionStrategy

Data partitioning strategies for multi-GPU distribution.

```rust
pub enum PartitionStrategy {
    /// Range-based partitioning (sorted data)
    Range,

    /// Hash-based partitioning (aggregations)
    Hash,

    /// Chunk-based partitioning (scans)
    Chunk,
}
```

**Example:**
```rust
use trueno_db::gpu::multigpu::{partition_data, PartitionStrategy};

let partitions = partition_data(
    data,
    4,  // Number of GPUs
    PartitionStrategy::Hash,
)?;

// Execute on each GPU in parallel
for partition in partitions {
    process_on_gpu(partition).await?;
}
```

## Error Handling

### Error Enum

```rust
pub enum Error {
    Storage(String),
    Io(std::io::Error),
    GpuInitFailed(String),
    VramExhausted { required: usize, available: usize },
    ParseError(String),
    InvalidInput(String),
    QueueClosed,
    BackendMismatch { expected: Backend, got: Backend },
    Other(String),
}
```

**Example:**
```rust
use trueno_db::Error;

match engine.execute(query).await {
    Ok(result) => process(result),
    Err(Error::VramExhausted { required, available }) => {
        eprintln!("GPU OOM: need {required} bytes, have {available}");
        // Retry with smaller morsel size
    }
    Err(e) => eprintln!("Query failed: {e}"),
}
```

## Feature Flags

Trueno-DB uses Cargo features to minimize binary size:

```toml
[dependencies]
trueno-db = { version = "0.2", features = ["gpu"] }
```

### Available Features

| Feature | Description | Binary Size | Compile Time |
|---------|-------------|-------------|--------------|
| `simd` (default) | SIMD-only backend | -0.4 MB | 18s |
| `gpu` | GPU backend (wgpu) | +3.8 MB | 63s |
| `distributed` | Multi-node execution (Phase 3) | TBD | TBD |
| `wasm` | WebAssembly build (Phase 4) | TBD | TBD |

**Example (SIMD-only):**
```toml
[dependencies]
trueno-db = { version = "0.2", default-features = false, features = ["simd"] }
```

**Example (GPU + SIMD):**
```toml
[dependencies]
trueno-db = { version = "0.2", features = ["gpu"] }
```

## Performance Tips

1. **Use morsels for large datasets:**
   ```rust
   for morsel in storage.morsels() {
       process(morsel)?;  // Prevents VRAM OOM
   }
   ```

2. **Enable GPU for 100K+ rows:**
   ```rust
   let db = Database::builder()
       .backend(Backend::Gpu)  // 50-100x speedup
       .build()?;
   ```

3. **Use Top-K instead of full sort:**
   ```rust
   // Fast: O(N log K)
   let top10 = batch.top_k(col, 10, SortOrder::Descending)?;

   // Slow: O(N log N)
   let sorted = batch.sort(col)?;
   ```

4. **Leverage kernel fusion:**
   ```rust
   // Fused (1 GPU pass)
   let result = gpu.fused_filter_sum(data, 100, "gt").await?;

   // Separate (2 GPU passes)
   let filtered = gpu.filter(data, 100, "gt").await?;
   let sum = gpu.sum(&filtered).await?;
   ```

## See Also

- **[Examples](../dev/examples.md)** - Complete usage examples
- **[Getting Started](../dev/getting-started.md)** - Quickstart guide
- **[Glossary](./glossary.md)** - Technical terminology
- **[Architecture](../architecture/system-overview.md)** - System design
- **Rust API Docs**: Run `cargo doc --open` for detailed documentation
