# alimentar view

Interactive TUI viewer for exploring datasets in the terminal.

## Synopsis

```bash
alimentar view [OPTIONS] <PATH>
```

## Description

The `view` command launches an interactive terminal-based viewer for exploring datasets. It supports Parquet, Arrow IPC, CSV, and JSON formats.

The viewer automatically selects between two modes based on dataset size:
- **InMemory Mode**: For datasets < 100,000 rows. All data loaded upfront for fast random access.
- **Streaming Mode**: For datasets >= 100,000 rows. Lazy batch loading for memory efficiency.

## Arguments

| Argument | Description |
|----------|-------------|
| `<PATH>` | Path to dataset file (Parquet, Arrow IPC, CSV, or JSON) |

## Options

| Option | Description |
|--------|-------------|
| `--search <QUERY>` | Initial search query - jumps to first matching row |
| `-h, --help` | Print help information |

## Keyboard Controls

### Navigation

| Key | Action |
|-----|--------|
| `↑` / `k` | Scroll up one row |
| `↓` / `j` | Scroll down one row |
| `PgUp` | Scroll up one page |
| `PgDn` / `Space` | Scroll down one page |
| `Home` / `g` | Jump to first row |
| `End` / `G` | Jump to last row |

### Search

| Key | Action |
|-----|--------|
| `/` | Open search prompt |
| `Enter` | Execute search (in search mode) |
| `Esc` | Cancel search (in search mode) |

### Exit

| Key | Action |
|-----|--------|
| `q` | Quit viewer |
| `Esc` | Quit viewer (when not in search mode) |
| `Ctrl+C` | Force quit |

## Examples

### Basic Usage

```bash
# View a Parquet file
alimentar view data.parquet

# View a CSV file
alimentar view data.csv

# View an Arrow IPC file
alimentar view data.arrow

# View a JSON file
alimentar view data.json
```

### Search on Open

```bash
# Open viewer and jump to first row containing "error"
alimentar view logs.parquet --search "error"

# Search for a specific ID
alimentar view users.csv --search "user_12345"
```

### Workflow Integration

```bash
# Quick inspection workflow
alimentar info data.parquet      # Check schema and stats
alimentar head data.parquet -n 5 # Preview first rows
alimentar view data.parquet      # Interactive exploration

# Quality check then explore
alimentar quality check data.csv && alimentar view data.csv
```

## Display

The viewer displays:

1. **Title Bar**: Filename, row count, and adapter mode (InMemory/Streaming)
2. **Data Table**: Scrollable table with column headers
3. **Status Bar**: Current row range, total rows, and available commands

### Column Rendering

- Strings are displayed as-is with proper Unicode width calculation
- Numbers are formatted with appropriate precision
- Null values are displayed as `NULL`
- Long values are truncated with `...` to fit column width

## Programmatic Usage

The TUI components can also be used programmatically in your Rust code:

```rust
use alimentar::tui::{DatasetAdapter, DatasetViewer};
use alimentar::ArrowDataset;

// Load dataset
let dataset = ArrowDataset::from_parquet("data.parquet")?;
let adapter = DatasetAdapter::from_dataset(&dataset)?;

// Create viewer with custom dimensions
let mut viewer = DatasetViewer::with_dimensions(adapter, 80, 24);

// Navigate programmatically
viewer.scroll_down();
viewer.page_down();
viewer.home();

// Search
if let Some(row) = viewer.search("query") {
    println!("Found at row {}", row);
}

// Render to strings
for line in viewer.render_lines() {
    println!("{}", line);
}
```

## See Also

- [alimentar info](./info.md) - Display dataset information
- [alimentar head](./head.md) - Display first N rows
- [alimentar schema](./schema.md) - Display dataset schema
