# Trueno-DB Queries

Trueno-viz integrates with trueno-db for visualizing query results directly.

## Enabling Integration

```toml
[dependencies]
trueno-viz = { version = "0.1", features = ["db"] }
trueno-db = "0.1"
```

## Direct Query Visualization

```rust
use trueno_db::Database;
use trueno_viz::interop::db as viz;

let db = Database::open("sales.db")?;

// Visualize query results directly
let plot = viz::query_scatter(&db,
    "SELECT revenue, profit FROM sales WHERE year = 2024")
    .xlabel("Revenue")
    .ylabel("Profit")
    .build();

plot.render_to_file("revenue_profit.png")?;
```

## Aggregation Charts

```rust
let plot = viz::query_bar(&db,
    "SELECT category, SUM(sales) as total FROM products GROUP BY category")
    .title("Sales by Category")
    .build();
```

## Time Series from DB

```rust
let plot = viz::query_line(&db,
    "SELECT date, value FROM metrics ORDER BY date")
    .title("Daily Metrics")
    .build();
```

## Histogram from Column

```rust
let plot = viz::query_histogram(&db,
    "SELECT age FROM customers",
    20)  // Number of bins
    .title("Customer Age Distribution")
    .build();
```

## Complete Example

```rust
use trueno_db::Database;
use trueno_viz::prelude::*;
use trueno_viz::interop::db as viz;

fn main() -> Result<()> {
    let db = Database::open("analytics.db")?;

    // Sales over time
    let time_series = viz::query_line(&db,
        "SELECT month, revenue FROM monthly_sales ORDER BY month")
        .color(Rgba::new(66, 133, 244, 255))
        .title("Monthly Revenue")
        .build();
    time_series.render_to_file("monthly_revenue.png")?;

    // Category breakdown
    let categories = viz::query_bar(&db,
        "SELECT category, COUNT(*) as count FROM products GROUP BY category")
        .title("Products by Category")
        .build();
    categories.render_to_file("category_breakdown.png")?;

    Ok(())
}
```

## Next Chapter

Continue to [Test-Driven Visualization](../tdd/test-driven-viz.md) for quality practices.
