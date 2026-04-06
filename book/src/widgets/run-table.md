# RunTable

RunTable displays a sortable table of experiment runs with their status, duration, and metrics. Perfect for experiment tracking dashboards.

## Basic Usage

```rust
use trueno_viz::prelude::*;

let runs = vec![
    RunRow::new("run-001", RunStatus::Completed)
        .with_duration(3600.0)
        .with_metric("loss", 0.05)
        .with_metric("accuracy", 0.95),
    RunRow::new("run-002", RunStatus::Running)
        .with_duration(1800.0)
        .with_metric("loss", 0.15),
    RunRow::new("run-003", RunStatus::Pending),
];

let table = RunTable::from_runs(runs);
println!("{}", table.render());
```

Output:
```
| ID | Status | Duration | accuracy | loss |
|----|---------|---------|----------|------|
| run-001 | ✅ Completed | 1.0h | 0.9500 | 0.0500 |
| run-002 | ▶ Running | 30.0m | - | 0.1500 |
| run-003 | ⏳ Pending | - | - | - |
```

## Run Status

| Status | Indicator | Terminal? | Active? |
|--------|-----------|-----------|---------|
| `Pending` | ⏳ | No | Yes |
| `Running` | ▶ | No | Yes |
| `Completed` | ✅ | Yes | No |
| `Failed` | ❌ | Yes | No |

```rust
use trueno_viz::prelude::*;

let status = RunStatus::Running;
println!("{} {}", status.indicator(), status); // ▶ Running
assert!(!status.is_terminal());
assert!(status.is_active());
```

## Sorting

Tables can be sorted by any column:

```rust
use trueno_viz::prelude::*;

let mut table = RunTable::from_runs(runs);

// Sort by ID (default, ascending)
// Already sorted on creation

// Toggle to descending
table.sort_by(SortColumn::Id);

// Sort by duration
table.sort_by(SortColumn::Duration);

// Sort by a metric
table.sort_by_metric("loss");
```

## Status Counts

```rust
use trueno_viz::prelude::*;

let table = RunTable::from_runs(runs);
let counts = table.status_counts();

println!("Running: {}", counts.get(&RunStatus::Running).unwrap_or(&0));
println!("Completed: {}", counts.get(&RunStatus::Completed).unwrap_or(&0));
```

## Filtering

```rust
use trueno_viz::prelude::*;

let table = RunTable::from_runs(runs);

// Get only failed runs
let failed: Vec<&RunRow> = table.filter_by_status(RunStatus::Failed);
```

## Duration Display

Duration is automatically formatted:

| Duration (seconds) | Display |
|-------------------|---------|
| 30.0 | `30.0s` |
| 120.0 | `2.0m` |
| 7200.0 | `2.0h` |
| None | `-` |

## API Reference

### RunRow

| Method | Description |
|--------|-------------|
| `new(id, status)` | Create a new run row |
| `with_duration(secs)` | Set duration in seconds |
| `with_metric(name, value)` | Add a metric |
| `metric(name)` | Get metric value |
| `duration_display()` | Format duration as string |

### RunTable

| Method | Description |
|--------|-------------|
| `new()` | Create empty table |
| `from_runs(vec)` | Create from vector of RunRow |
| `add_run(row)` | Add a run |
| `runs()` | Get runs slice |
| `len()` | Get count |
| `is_empty()` | Check if empty |
| `sort_by(column)` | Sort by column (toggles direction) |
| `sort_by_metric(name)` | Sort by metric name |
| `status_counts()` | Get HashMap of status counts |
| `filter_by_status(status)` | Filter runs by status |
| `render()` | Render as markdown table string |

### SortColumn

| Variant | Description |
|---------|-------------|
| `Id` | Sort by run ID |
| `Status` | Sort by status |
| `Duration` | Sort by duration |
| `Metric(idx)` | Sort by metric column index |

### SortDirection

| Variant | Description |
|---------|-------------|
| `Ascending` | A-Z, 0-9 |
| `Descending` | Z-A, 9-0 |
