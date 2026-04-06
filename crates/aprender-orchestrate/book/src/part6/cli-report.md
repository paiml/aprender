# `batuta report`

Generate a migration report summarizing the transpilation pipeline results.

## Synopsis

```bash
batuta report [OPTIONS]
```

## Description

The report command generates a comprehensive migration report covering all 5 pipeline phases. It includes analysis results, transpilation statistics, optimization recommendations, validation results, and build status.

## Options

| Option | Description |
|--------|-------------|
| `--output <PATH>` | Output file path (default: `migration_report.html`) |
| `--format <FORMAT>` | Report format: `html` (default), `markdown`, `json`, `text` |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Output Formats

| Format | Description |
|--------|-------------|
| `html` | Rich HTML report with charts and styling |
| `markdown` | Markdown for GitHub/GitLab integration |
| `json` | Machine-readable JSON for CI/CD pipelines |
| `text` | Plain text for terminal viewing |

## Examples

### HTML Report (Default)

```bash
$ batuta report

📊 Generating migration report...
Report saved to: migration_report.html
```

### Markdown for GitHub

```bash
$ batuta report --format markdown --output MIGRATION.md
```

### JSON for CI/CD

```bash
$ batuta report --format json --output report.json
```

## See Also

- [`batuta status`](./cli-status.md) - Quick status check
- [`batuta build`](./cli-build.md) - Preceding build phase

---

**Previous:** [`batuta build`](./cli-build.md)
**Next:** [`batuta status`](./cli-status.md)
