# CLI & REPL (Examples 86-95)

This section covers the command-line interface and REPL.

## Examples 86-87: CLI Help and Info

```bash
# Show help
alimentar --help

# Dataset info
alimentar info data.parquet
# Output:
# Format: Parquet
# Rows: 1000
# Columns: 3 (id: Int32, name: Utf8, value: Float64)
# Size: 45.2 KB
```

## Examples 88-89: Head and Convert

```bash
# Show first N rows
alimentar head data.parquet --rows 10

# Format conversion
alimentar convert input.csv output.parquet
alimentar convert data.parquet data.json
```

## Example 90: Quality Command

```bash
# Quality report
alimentar quality data.parquet

# JSON output
alimentar quality data.parquet --format json

# Quality score only
alimentar quality score data.parquet
# Output: Quality Score: 0.92 (A)
```

## Examples 91-92: REPL Session and Completion

```rust
use alimentar::repl::{ReplSession, Completer};

// Start REPL session
let mut session = ReplSession::new();
session.run()?;

// Programmatic usage
session.execute("load data.parquet")?;
session.execute("head 10")?;

// Tab completion
let completer = Completer::new();
let suggestions = completer.complete("loa", 3);
// Returns: ["load"]
```

## Examples 93-94: REPL Commands and History

```bash
# REPL commands
alimentar repl

> load data.parquet
Loaded: 1000 rows, 3 columns

> head 5
+----+--------+-------+
| id | name   | value |
+----+--------+-------+
| 1  | item_1 | 0.1   |
| 2  | item_2 | 0.2   |
...

> schema
id: Int32 (not null)
name: Utf8 (not null)
value: Float64 (not null)

> quality
Quality Score: 0.95 (A)

> history
1: load data.parquet
2: head 5
3: schema
4: quality

> quit
```

## Example 95: CLI Batch Script

```bash
# Batch execution from script
cat commands.txt
load data.parquet
quality
convert data.parquet output.json

# Execute batch
alimentar batch commands.txt

# Or via stdin
cat commands.txt | alimentar batch -
```

## REPL Commands Reference

| Command | Description |
|---------|-------------|
| `load <file>` | Load dataset |
| `head [n]` | Show first n rows (default 10) |
| `tail [n]` | Show last n rows |
| `schema` | Show schema |
| `info` | Show dataset info |
| `quality` | Run quality check |
| `drift <file>` | Compare with another dataset |
| `convert <file>` | Save to different format |
| `filter <expr>` | Filter rows |
| `select <cols>` | Select columns |
| `history` | Show command history |
| `help` | Show help |
| `quit` | Exit REPL |

## Key Concepts

- **Subcommands**: info, head, convert, quality, etc.
- **REPL**: Interactive exploration
- **Completion**: Tab completion for commands
- **Batch**: Non-interactive script execution
