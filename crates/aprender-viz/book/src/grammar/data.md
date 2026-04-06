# Data Layer

The data layer provides the foundation for all visualizations. Trueno-viz
uses a `DataFrame` abstraction for tabular data.

## Creating DataFrames

### From Vectors

```rust
use trueno_viz::grammar::DataFrame;

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0, 4.0, 5.0])
    .column("y", &[2.1, 3.9, 6.2, 7.8, 10.1])
    .column("label", &["A", "B", "C", "D", "E"]);

// Verify column count
assert_eq!(df.columns().len(), 3);
assert_eq!(df.len(), 5);
```

**Test Reference**: `src/grammar/data.rs::test_dataframe_columns`

### From Iterators

```rust
use trueno_viz::grammar::DataFrame;

let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
let y: Vec<f32> = x.iter().map(|x| x * x).collect();

let df = DataFrame::new()
    .column("x", &x)
    .column("y", &y);

assert_eq!(df.len(), 100);
```

### Empty DataFrame

```rust
use trueno_viz::grammar::DataFrame;

let df = DataFrame::new();
assert!(df.is_empty());
assert_eq!(df.len(), 0);
```

**Test Reference**: `src/grammar/data.rs::test_dataframe_empty`

## Accessing Data

### Column Access

```rust
use trueno_viz::grammar::DataFrame;

let df = DataFrame::new()
    .column("temp", &[20.0, 22.5, 25.0, 23.0]);

// Get column by name
if let Some(col) = df.get("temp") {
    assert_eq!(col.len(), 4);
}

// Check column existence
assert!(df.columns().contains(&"temp".to_string()));
```

### Row Iteration

```rust
use trueno_viz::grammar::DataFrame;

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0])
    .column("y", &[4.0, 5.0, 6.0]);

// Iterate through data
for i in 0..df.len() {
    let x = df.get("x").unwrap()[i];
    let y = df.get("y").unwrap()[i];
    println!("Point: ({}, {})", x, y);
}
```

## Column Types

The DataFrame supports multiple column types:

```rust
use trueno_viz::grammar::{DataFrame, Column};

// Numeric column
let numeric = Column::Numeric(vec![1.0, 2.0, 3.0]);

// Categorical column
let categorical = Column::Categorical(vec![
    "low".to_string(),
    "medium".to_string(),
    "high".to_string(),
]);
```

## Data Transformations

### Filtering

```rust
use trueno_viz::grammar::DataFrame;

let df = DataFrame::new()
    .column("x", &[1.0, 2.0, 3.0, 4.0, 5.0])
    .column("y", &[10.0, 20.0, 30.0, 40.0, 50.0]);

// Filter where x > 2
let filtered = df.filter(|row| row.get("x").unwrap() > &2.0);
```

### Sorting

```rust
use trueno_viz::grammar::DataFrame;

let df = DataFrame::new()
    .column("name", &["Charlie", "Alice", "Bob"])
    .column("score", &[85.0, 92.0, 78.0]);

// Sort by score descending
let sorted = df.sort_by("score", false);
```

## Integration with Aprender

When the `ml` feature is enabled:

```rust
#[cfg(feature = "ml")]
use aprender::DataFrame as AprenderDF;

#[cfg(feature = "ml")]
let aprender_df: AprenderDF = load_data();

#[cfg(feature = "ml")]
let viz_df = DataFrame::from_aprender(&aprender_df);
```

## Memory Layout

DataFrames use column-major storage for SIMD efficiency:

```text
┌─────────────────────────────────────────┐
│            DataFrame                     │
├─────────────────────────────────────────┤
│  Column "x": [1.0, 2.0, 3.0, 4.0, 5.0] │  ← Contiguous f32
│  Column "y": [2.1, 3.9, 6.2, 7.8, 10.1]│  ← Contiguous f32
│  Column "label": ["A", "B", "C", ...]  │  ← String vec
└─────────────────────────────────────────┘
```

This layout enables SIMD operations on numeric columns:

```rust
// SIMD acceleration happens automatically
let col = df.get("x").unwrap();
let sum: f32 = col.iter().sum();  // Uses SIMD internally
```

## Complete Example

```rust
use trueno_viz::grammar::{DataFrame, GGPlot, Aes, Geom};

fn main() {
    // Create dataset
    let df = DataFrame::new()
        .column("height", &[160.0, 165.0, 170.0, 175.0, 180.0])
        .column("weight", &[55.0, 62.0, 70.0, 78.0, 85.0])
        .column("gender", &["F", "F", "M", "M", "M"]);

    // Create visualization
    let plot = GGPlot::new(df)
        .aes(Aes::new().x("height").y("weight").color("gender"))
        .geom(Geom::point())
        .title("Height vs Weight by Gender");

    // Render
    let _ = plot.render();
}
```

## Next Chapter

Continue to [Aesthetic Mappings](./aes.md) to learn how data columns
map to visual properties.
