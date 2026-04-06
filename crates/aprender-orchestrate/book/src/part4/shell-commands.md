# Command Parsing: Shell to Rust

Bashrs converts shell command invocations, pipe chains, and environment variable
access into typed Rust equivalents using `std::process::Command` and iterator
chains.

## Simple Commands

**Bash**

```bash
docker build -t myapp:latest .
```

**Rust**

```rust
use std::process::Command;

let status = Command::new("docker")
    .args(["build", "-t", "myapp:latest", "."])
    .status()?;
```

Each shell command becomes a `Command::new` call. Arguments are passed as a
slice, avoiding shell injection vulnerabilities that arise from string
interpolation in Bash.

## Pipe Chains

**Bash**

```bash
cat access.log | grep "ERROR" | awk '{print $4}' | sort | uniq -c | sort -rn
```

**Rust (process pipes)**

```rust
use std::process::{Command, Stdio};

let grep = Command::new("grep")
    .arg("ERROR")
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .spawn()?;

let awk = Command::new("awk")
    .arg("{print $4}")
    .stdin(grep.stdout.unwrap())
    .stdout(Stdio::piped())
    .spawn()?;
```

For pipelines that process text, bashrs can also convert to pure Rust iterator
chains, eliminating external process overhead:

**Rust (iterator chain)**

```rust
use std::fs;

let content = fs::read_to_string("access.log")?;
let mut counts: HashMap<String, usize> = HashMap::new();

for line in content.lines().filter(|l| l.contains("ERROR")) {
    if let Some(field) = line.split_whitespace().nth(3) {
        *counts.entry(field.to_string()).or_default() += 1;
    }
}

let mut sorted: Vec<_> = counts.into_iter().collect();
sorted.sort_by(|a, b| b.1.cmp(&a.1));
```

The iterator version is typically faster because it avoids spawning four
separate processes and piping data through the kernel.

## Environment Variables

**Bash**

```bash
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
CONNECTION="postgresql://${DB_HOST}:${DB_PORT}/mydb"
```

**Rust**

```rust
use std::env;

let db_host = env::var("DB_HOST").unwrap_or_else(|_| "localhost".into());
let db_port = env::var("DB_PORT").unwrap_or_else(|_| "5432".into());
let connection = format!("postgresql://{db_host}:{db_port}/mydb");
```

For CLI tools, bashrs promotes environment variables to typed `clap` arguments
with `env` attributes, providing both flag and env-var access:

```rust
#[derive(clap::Parser)]
struct Config {
    #[arg(long, env = "DB_HOST", default_value = "localhost")]
    db_host: String,

    #[arg(long, env = "DB_PORT", default_value_t = 5432)]
    db_port: u16,  // Typed as integer, not string
}
```

## Command Substitution

**Bash**

```bash
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "On branch: ${CURRENT_BRANCH}"
```

**Rust**

```rust
let output = Command::new("git")
    .args(["rev-parse", "--abbrev-ref", "HEAD"])
    .output()?;

let current_branch = String::from_utf8(output.stdout)?
    .trim()
    .to_string();
println!("On branch: {current_branch}");
```

`Command::output()` captures both stdout and stderr. The output is explicit
bytes that must be decoded, catching encoding issues that Bash would silently
pass through.

## Conditional Execution

**Bash**

```bash
command -v docker >/dev/null 2>&1 || { echo "docker not found"; exit 1; }
```

**Rust**

```rust
use which::which;

if which("docker").is_err() {
    eprintln!("docker not found");
    std::process::exit(1);
}
```

The `which` crate provides cross-platform command detection, replacing the
Bash-specific `command -v` builtin.

## Key Takeaways

- Shell commands become `Command::new` with typed argument slices, eliminating
  injection risks.
- Pipe chains can remain as process pipes or convert to iterator chains for
  better performance.
- Environment variables with defaults map to `clap` arguments with `env`
  attributes and typed parsing.
- Command substitution uses `Command::output()` with explicit encoding.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
