# CLI Design: Shell to Rust

Bashrs converts shell argument parsing patterns (`getopts`, `getopt`, manual
`$1`/`$2` handling) into structured `clap` derive macros with type safety,
validation, and auto-generated help text.

## Positional Arguments

**Bash**

```bash
#!/bin/bash
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input> <output>" >&2
    exit 1
fi
INPUT="$1"
OUTPUT="$2"
```

**Rust (clap)**

```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "convert", about = "Convert input file to output format")]
struct Args {
    /// Input file path
    input: PathBuf,

    /// Output file path
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    convert(&args.input, &args.output)?;
    Ok(())
}
```

Clap generates usage text, `--help`, and error messages automatically. Missing
arguments produce clear diagnostics instead of the generic Bash error.

## Flags and Options

**Bash (getopts)**

```bash
VERBOSE=false
DRY_RUN=false
WORKERS=4

while getopts "vdw:" opt; do
    case $opt in
        v) VERBOSE=true ;;
        d) DRY_RUN=true ;;
        w) WORKERS=$OPTARG ;;
        *) echo "Usage: $0 [-v] [-d] [-w workers]" >&2; exit 1 ;;
    esac
done
```

**Rust (clap)**

```rust
#[derive(Parser)]
#[command(name = "deploy")]
struct Args {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Perform a dry run without making changes
    #[arg(short, long)]
    dry_run: bool,

    /// Number of parallel workers
    #[arg(short, long, default_value_t = 4)]
    workers: u32,
}
```

The `workers` field is typed as `u32`. Clap rejects non-numeric input at parse
time, while Bash would silently assign a string to `$WORKERS` and fail later
in arithmetic.

## Subcommands

**Bash**

```bash
case "$1" in
    build)  shift; do_build "$@" ;;
    test)   shift; do_test "$@" ;;
    deploy) shift; do_deploy "$@" ;;
    *)      echo "Unknown command: $1" >&2; exit 1 ;;
esac
```

**Rust (clap)**

```rust
#[derive(Parser)]
#[command(name = "app")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build the project
    Build {
        /// Build in release mode
        #[arg(long)]
        release: bool,
    },
    /// Run tests
    Test {
        /// Test filter pattern
        filter: Option<String>,
    },
    /// Deploy to production
    Deploy {
        /// Target environment
        #[arg(long, default_value = "staging")]
        env: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Build { release } => do_build(release),
        Commands::Test { filter } => do_test(filter),
        Commands::Deploy { env } => do_deploy(&env),
    }
}
```

Each subcommand becomes an enum variant with its own typed fields. The compiler
ensures all variants are handled in the `match` expression.

## Shell Completion Generation

Clap can generate shell completion scripts for Bash, Zsh, Fish, and PowerShell:

```rust
use clap_complete::{generate, Shell};

fn print_completions(shell: Shell, cmd: &mut clap::Command) {
    generate(shell, cmd, "app", &mut std::io::stdout());
}
```

```bash
# Generate and install completions
app --generate-completions bash > /etc/bash_completion.d/app
app --generate-completions zsh > ~/.zsh/completions/_app
```

This gives the converted CLI better tab-completion than the original Bash
script, which would require manually writing a completion function.

## Environment Variable Integration

Bashrs promotes environment variables to first-class `clap` arguments:

```rust
#[derive(Parser)]
struct Config {
    /// API endpoint
    #[arg(long, env = "API_URL")]
    api_url: String,

    /// Authentication token
    #[arg(long, env = "API_TOKEN")]
    api_token: String,

    /// Log level
    #[arg(long, env = "LOG_LEVEL", default_value = "info")]
    log_level: String,
}
```

Users can set values via flags (`--api-url https://...`) or environment
variables (`API_URL=https://...`). The `--help` output documents both options.

## Key Takeaways

- Positional arguments and flags move from string parsing to typed structs with
  compile-time validation.
- `getopts`/`getopt` case statements become `clap` derive macros with
  auto-generated help and error messages.
- Subcommands map to Rust enums, ensuring exhaustive handling.
- Shell completion is generated automatically for Bash, Zsh, Fish, and
  PowerShell.
- Environment variables integrate directly into the argument parser with
  `env` attributes.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
