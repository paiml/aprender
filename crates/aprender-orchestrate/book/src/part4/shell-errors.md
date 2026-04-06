# Error Handling: Shell to Rust

Bash error handling relies on exit codes, `set -e`, and `trap`. Bashrs converts
these patterns into Rust's `Result` type, providing typed errors with context
at every failure point.

## set -e to Result Propagation

**Bash**

```bash
set -e
mkdir -p /tmp/build
cp -r src/ /tmp/build/
cargo build --release
```

With `set -e`, any command that returns a non-zero exit code terminates the
script. The equivalent in Rust is the `?` operator on `Result`:

**Rust**

```rust
fn build() -> Result<()> {
    fs::create_dir_all("/tmp/build")?;
    copy_dir("src/", "/tmp/build/")?;
    let status = Command::new("cargo")
        .args(["build", "--release"])
        .status()
        .context("Failed to start cargo build")?;
    if !status.success() {
        anyhow::bail!("cargo build exited with {status}");
    }
    Ok(())
}
```

Unlike `set -e`, each `?` propagation carries context about which operation
failed. Bash's `set -e` provides no indication of *which* command failed when
the script exits silently.

## Exit Codes to Typed Errors

**Bash**

```bash
validate_config() {
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Config file not found" >&2
        return 1
    fi
    if ! jq empty "$CONFIG_FILE" 2>/dev/null; then
        echo "Invalid JSON in config" >&2
        return 2
    fi
    return 0
}
```

**Rust**

```rust
#[derive(Debug, thiserror::Error)]
enum ConfigError {
    #[error("Config file not found: {path}")]
    NotFound { path: PathBuf },

    #[error("Invalid JSON in config: {source}")]
    InvalidJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

fn validate_config(path: &Path) -> Result<Config, ConfigError> {
    let content = fs::read_to_string(path)
        .map_err(|_| ConfigError::NotFound { path: path.into() })?;
    let config: Config = serde_json::from_str(&content)
        .map_err(|e| ConfigError::InvalidJson {
            path: path.into(),
            source: e,
        })?;
    Ok(config)
}
```

Numeric exit codes (1, 2) become named enum variants with structured data.
Callers can match on the error type and take specific recovery actions rather
than checking magic numbers.

## Trap Handlers to Drop

**Bash**

```bash
TMPDIR=$(mktemp -d)
trap "rm -rf ${TMPDIR}" EXIT

# Work with temporary files...
cp important.dat "${TMPDIR}/work.dat"
process "${TMPDIR}/work.dat"
```

**Rust**

```rust
use tempfile::TempDir;

fn process_with_temp() -> Result<()> {
    let tmpdir = TempDir::new()?;
    // tmpdir is automatically deleted when it goes out of scope

    let work_path = tmpdir.path().join("work.dat");
    fs::copy("important.dat", &work_path)?;
    process(&work_path)?;

    Ok(())
    // TempDir::drop() removes the directory here
}
```

Bash `trap ... EXIT` is a cleanup hook that runs when the script exits.
Rust's `Drop` trait serves the same purpose but is scoped to the owning
variable. The `tempfile` crate provides `TempDir` which deletes itself on
drop, even if the function returns early due to an error.

## Pipefail to Checked Pipelines

**Bash**

```bash
set -o pipefail
curl -s "$URL" | jq '.data' | process_data
```

Without `pipefail`, only the exit code of the last command in a pipeline is
checked. With it, any failure in the chain is caught. In Rust, each step is
checked individually:

**Rust**

```rust
fn fetch_and_process(url: &str) -> Result<()> {
    let response = Command::new("curl")
        .args(["-s", url])
        .output()
        .context("curl failed")?;
    if !response.status.success() {
        anyhow::bail!("curl returned {}", response.status);
    }

    let parsed: Value = serde_json::from_slice(&response.stdout)
        .context("Failed to parse JSON response")?;
    let data = parsed.get("data")
        .context("Missing 'data' field in response")?;

    process_data(data)?;
    Ok(())
}
```

## Key Takeaways

- `set -e` maps to `Result` with `?` propagation, but each step includes
  context about what failed.
- Numeric exit codes become typed error enums with structured diagnostic data.
- `trap ... EXIT` cleanup maps to Rust's `Drop` trait, which runs even on
  early returns.
- `set -o pipefail` becomes explicit status checks on each pipeline stage.
- Rust errors compose: a function can wrap lower-level errors with
  `.context()` to build a full failure trace.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
