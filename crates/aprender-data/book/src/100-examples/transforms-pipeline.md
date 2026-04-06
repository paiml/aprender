# Transforms Pipeline (Examples 31-45)

This section covers data transformation operations.

## Examples 31-33: Column Operations

```rust
use alimentar::{Select, Drop as DropTransform, Rename, Transform};

// Select columns
let select = Select::new(vec!["id".to_string(), "value".to_string()]);
let result = select.apply(batch)?;

// Drop columns
let drop = DropTransform::new(vec!["name".to_string()]);
let result = drop.apply(batch)?;

// Rename columns
let rename = Rename::new(vec![("old_name".into(), "new_name".into())]);
let result = rename.apply(batch)?;
```

## Examples 34-35: Row Filtering

```rust
use alimentar::{Filter, Transform};

// Numeric filter
let filter = Filter::new("value > 100");
let result = filter.apply(batch)?;

// String filter
let filter = Filter::new("name LIKE 'item_%'");
let result = filter.apply(batch)?;
```

## Examples 36-37: Null Fill Strategies

```rust
use alimentar::{FillNull, FillStrategy, Transform};

// Fill with mean
let fill = FillNull::new("score", FillStrategy::Mean);
let result = fill.apply(batch)?;

// Fill with constant
let fill = FillNull::new("score", FillStrategy::Constant(0.0));
let result = fill.apply(batch)?;
```

## Examples 38-39: Normalization

```rust
use alimentar::{Normalize, NormStrategy, Transform};

// MinMax normalization [0, 1]
let norm = Normalize::new("value", NormStrategy::MinMax);
let result = norm.apply(batch)?;

// Z-score normalization (mean=0, std=1)
let norm = Normalize::new("value", NormStrategy::ZScore);
let result = norm.apply(batch)?;
```

## Examples 40-41: Sorting

```rust
use alimentar::{Sort, Transform};

// Sort ascending
let sort = Sort::new("value", true);
let result = sort.apply(batch)?;

// Sort descending
let sort = Sort::new("value", false);
let result = sort.apply(batch)?;
```

## Examples 42-44: Take, Skip, Unique

```rust
use alimentar::{Take, Skip, Unique, Transform};

// Take first N rows
let take = Take::new(100);
let result = take.apply(batch)?;

// Skip first N rows
let skip = Skip::new(10);
let result = skip.apply(batch)?;

// Remove duplicates
let unique = Unique::new(vec!["id".to_string()]);
let result = unique.apply(batch)?;
```

## Example 45: Transform Chain

```rust
use alimentar::{TransformChain, Select, Filter, Normalize, Transform};

let chain = TransformChain::new()
    .add(Select::new(vec!["id".into(), "value".into()]))
    .add(Filter::new("value > 0"))
    .add(Normalize::new("value", NormStrategy::MinMax));

let result = chain.apply(batch)?;
```

## Key Concepts

- **Immutable transforms**: Each transform returns new batch
- **Composability**: Chain transforms together
- **Type safety**: Schema validation at each step
- **Zero-copy where possible**: Arrow slice semantics
