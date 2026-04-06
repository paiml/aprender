# SQL Query Interface

The SQL Query Interface provides a DuckDB-like API for executing OLAP analytics queries on Arrow data using GPU/SIMD acceleration.

## Overview

**Status**: Phase 1 Complete (v0.3.0)
**Coverage**: 92.64% test coverage
**Performance**: 2.78x faster aggregations (SIMD), 5-28x faster Top-K

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ SQL Query   │────▶│ QueryEngine  │────▶│ QueryPlan   │
│ (String)    │     │ (Parser)     │     │ (AST)       │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │ Query       │
                                         │ Executor    │
                                         └─────────────┘
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          ▼                     ▼                     ▼
                    ┌──────────┐         ┌──────────┐         ┌──────────┐
                    │ Filter   │         │ Aggregate│         │ Top-K    │
                    │ (WHERE)  │         │ (SUM/AVG)│         │ (ORDER)  │
                    └──────────┘         └──────────┘         └──────────┘
                          │                     │                     │
                          └─────────────────────┴─────────────────────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │ RecordBatch │
                                         │ (Results)   │
                                         └─────────────┘
```

## Supported Features

### Phase 1 (Current)

✅ **SELECT** - Column projection
✅ **WHERE** - Filtering with comparison operators (`>`, `>=`, `<`, `<=`, `=`, `!=`/`<>`)
✅ **Aggregations** - `SUM`, `AVG`, `COUNT`, `MIN`, `MAX`
✅ **ORDER BY** - Ascending/descending sort with Top-K optimization
✅ **LIMIT** - Result set limiting

### Phase 2 (Future)

❌ **GROUP BY** - Aggregations with grouping (planned)
❌ **JOIN** - Inner/outer joins (planned)
❌ **Subqueries** - Nested queries (planned)
❌ **Window Functions** - OVER clause (planned)

## Quick Start

```rust
use trueno_db::query::{QueryEngine, QueryExecutor};
use trueno_db::storage::StorageEngine;

// Load data
let storage = StorageEngine::load_parquet("data/events.parquet")?;

// Initialize query engine
let engine = QueryEngine::new();
let executor = QueryExecutor::new();

// Parse and execute query
let plan = engine.parse("SELECT SUM(value) FROM events WHERE value > 100")?;
let result = executor.execute(&plan, &storage)?;
```

## Examples

### Example 1: Simple Aggregation

```rust
let sql = "SELECT COUNT(*), SUM(revenue), AVG(revenue) FROM sales";
let plan = engine.parse(sql)?;
let result = executor.execute(&plan, &storage)?;

// Extract results
let count = result.column(0).as_any().downcast_ref::<Int64Array>()?.value(0);
let sum = result.column(1).as_any().downcast_ref::<Float64Array>()?.value(0);
let avg = result.column(2).as_any().downcast_ref::<Float64Array>()?.value(0);
```

### Example 2: Filtering

```rust
let sql = "SELECT user_id, score FROM leaderboard WHERE score > 1000";
let plan = engine.parse(sql)?;
let result = executor.execute(&plan, &storage)?;
```

### Example 3: Top-K Query

Uses O(N log K) heap-based selection instead of O(N log N) full sort:

```rust
let sql = "SELECT player_id, score FROM rankings ORDER BY score DESC LIMIT 10";
let plan = engine.parse(sql)?;
let result = executor.execute(&plan, &storage)?;
```

**Performance**: 5-28x faster than sorting all rows for K << N.

### Example 4: Filter + Aggregation

```rust
let sql = "SELECT MIN(price), MAX(price), AVG(price)
           FROM products
           WHERE category = 'Electronics'";
let plan = engine.parse(sql)?;
let result = executor.execute(&plan, &storage)?;
```

## Performance Characteristics

### Aggregations

- **SIMD Acceleration**: 2.78x faster than scalar baseline
- **Backend Selection**: Automatic GPU dispatch for large datasets (>10MB, >5x transfer cost)
- **Data Types**: Int32, Int64, Float32, Float64

### Top-K Selection

- **Algorithm**: Heap-based selection (O(N log K))
- **Speedup**: 5-28x vs full sort for K << N
- **Example**: Top-100 from 1M rows = 28x faster

### Filtering

- **Zero-Copy**: Uses Arrow compute functions
- **SIMD**: Vectorized comparison operations
- **Selectivity**: Best for <20% result sets

## Implementation Details

### Query Parsing

Uses `sqlparser-rs` for SQL parsing:

```rust
pub struct QueryPlan {
    pub columns: Vec<String>,                // SELECT columns
    pub table: String,                       // FROM table
    pub filter: Option<String>,              // WHERE clause
    pub aggregations: Vec<Aggregation>,      // Aggregate functions
    pub group_by: Vec<String>,               // GROUP BY (Phase 2)
    pub order_by: Vec<(String, OrderDirection)>, // ORDER BY
    pub limit: Option<usize>,                // LIMIT
}
```

### Query Execution

1. **Parse**: SQL → QueryPlan
2. **Combine Batches**: Merge storage batches
3. **Filter**: Apply WHERE predicate
4. **Aggregate**: Execute SUM/AVG/COUNT/MIN/MAX
5. **Project**: Select columns
6. **Sort**: Apply Top-K if ORDER BY + LIMIT
7. **Return**: Arrow RecordBatch

### Backend Selection

```rust
// Cost-based dispatch
if data_size > 10MB && compute_time > 5x_transfer_time {
    use_gpu();
} else {
    use_simd();
}
```

## Error Handling

```rust
match executor.execute(&plan, &storage) {
    Ok(result) => /* process results */,
    Err(Error::InvalidInput(msg)) => /* column not found, etc. */,
    Err(Error::ParseError(msg)) => /* SQL syntax error */,
    Err(e) => /* other errors */,
}
```

## Testing

- **156 tests** total (18 SQL-specific)
- **Property-based tests** with proptest
- **Backend equivalence**: GPU == SIMD == Scalar
- **92.64% code coverage**

## Benchmarks

Run SQL query benchmarks:

```bash
cargo bench --bench sql_query_benchmarks
```

Key benchmark groups:
- `sql_sum_aggregation`: SUM performance
- `sql_top_k_order_by_limit`: Top-K optimization
- `sql_filter_aggregate`: Combined operations
- `scalar_baseline_sum`: Baseline comparison

## Limitations (Phase 1)

1. **No GROUP BY**: Only simple aggregations (all rows)
2. **No JOINs**: Single table queries only
3. **Simple WHERE**: Single predicate only (`col op value`)
4. **No type coercion**: Explicit type matching required

## Future Enhancements (Phase 2)

1. **GROUP BY** with hash aggregation
2. **Hash JOIN** (inner/outer)
3. **Complex WHERE** (AND/OR/NOT)
4. **Window functions** (OVER clause)
5. **GPU aggregation** for large datasets

## See Also

- [JIT WGSL Compiler](./jit-compiler.md)
- [Kernel Fusion](./kernel-fusion.md)
- [Examples](../../dev/examples.md)

## References

- **sqlparser-rs**: SQL parser library
- **Apache Arrow**: Columnar format
- **DuckDB**: API inspiration
- **MonetDB/X100**: Vectorized execution model
