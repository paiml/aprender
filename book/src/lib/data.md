<!-- PCU: lib-data | contract: contracts/apr-page-lib-data-v1.yaml -->

# Module: `aprender::data`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/data/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/data/mod.rs) or directory.

## Example

```rust
use aprender::data::{DataFrame, ColumnStats};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::data` provides a minimal columnar `DataFrame` for ML pipelines —
named `Vector<f32>` columns, schema-aware selection, row / matrix
extraction, and a `describe` summary that returns `ColumnStats` per column.
The companion submodules cover PII redaction (`data::pii`), schema evolution
(`data::evolve`), and quality filtering (`data::quality_filter`) — the
basics needed to clean and validate datasets before training.

## Key types

| Type | Description |
|------|-------------|
| `DataFrame` | Named-column container of `Vector<f32>`s. Methods: `column`, `select`, `row`, `to_matrix`, `iter_columns`, `add_column`, `drop_column`, `describe`. |
| `ColumnStats` | Summary statistics per column (mean, std, min, max, etc.). |
| `data::pii` | PII detection / redaction utilities. |
| `data::evolve` | Schema evolution between dataset versions. |
| `data::quality_filter` | Quality-based row filtering (NaN, Inf, range checks). |

## Usage patterns

### Pattern 1: Build a DataFrame and convert to Matrix

```rust
use aprender::data::DataFrame;
use aprender::primitives::Vector;

let df = DataFrame::new(vec![
    ("age".to_string(), Vector::from_slice(&[25.0, 30.0, 35.0, 40.0])),
    ("income".to_string(), Vector::from_slice(&[40_000.0, 55_000.0, 65_000.0, 80_000.0])),
    ("score".to_string(), Vector::from_slice(&[0.7, 0.8, 0.85, 0.9])),
]).expect("valid DataFrame");

assert_eq!(df.shape(), (4, 3));
let names = df.column_names();
println!("columns: {:?}", names);

// Pull a `Matrix<f32>` ready for an Estimator.
let x = df.to_matrix();
assert_eq!(x.shape(), (4, 3));
```

### Pattern 2: Select columns and summarise

```rust
use aprender::data::DataFrame;
use aprender::primitives::Vector;

let df = DataFrame::new(vec![
    ("a".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 4.0])),
    ("b".to_string(), Vector::from_slice(&[5.0, 6.0, 7.0, 8.0])),
    ("c".to_string(), Vector::from_slice(&[9.0, 10.0, 11.0, 12.0])),
]).expect("valid DataFrame");

let subset = df.select(&["a", "c"]).expect("select existing cols");
assert_eq!(subset.n_cols(), 2);

for stats in df.describe() {
    println!("column stats: {:?}", stats);
}
```

## See also

- [`primitives`](primitives.md) — `Matrix` and `Vector` are the underlying storage
- [`preprocessing`](preprocessing.md) — apply `StandardScaler` / encoders to a `DataFrame`
- [`loading`](loading.md) — read CSV / Parquet / JSON into a `DataFrame`
- [`mining`](mining.md) — pattern mining on transactional / categorical data

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
