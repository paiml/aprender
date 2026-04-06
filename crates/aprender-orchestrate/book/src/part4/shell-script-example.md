# Example 3: Shell Script Conversion

This walkthrough demonstrates converting a Bash build-and-deploy script into a
typed Rust CLI using `bashrs`, the Shell-to-Rust transpiler.

## Scenario

A DevOps team maintains `deploy.sh`, a 400-line Bash script that builds a
Docker image, runs integration tests, pushes to a registry, and deploys to
Kubernetes. The script has grown organically and suffers from silent failures,
unclear error messages, and environment-specific bugs. The goal is a portable
Rust CLI with proper error handling and typed configuration.

## Source Script (simplified)

```bash
#!/bin/bash
set -euo pipefail

REGISTRY="${DOCKER_REGISTRY:-ghcr.io/team}"
TAG="${GIT_SHA:-$(git rev-parse --short HEAD)}"
IMAGE="${REGISTRY}/app:${TAG}"

echo "Building ${IMAGE}..."
docker build -t "${IMAGE}" .

echo "Running tests..."
docker run --rm "${IMAGE}" /app/run_tests.sh
if [ $? -ne 0 ]; then
    echo "Tests failed!" >&2
    exit 1
fi

echo "Pushing ${IMAGE}..."
docker push "${IMAGE}"

echo "Deploying to cluster..."
kubectl set image deployment/app app="${IMAGE}" --record
kubectl rollout status deployment/app --timeout=300s
```

## Step 1 -- Analyze

```bash
batuta analyze --languages --tdg ./scripts
```

```
Languages detected: Shell (100%)
Commands used: docker, kubectl, git, echo
Environment variables: DOCKER_REGISTRY, GIT_SHA
Error handling: set -e (global), 1 explicit check
TDG Score: D (45/100) — weak error handling, unquoted variables
```

## Step 2 -- Transpile

```bash
batuta transpile ./scripts/deploy.sh --tool bashrs --output ./deploy_cli
```

Bashrs converts the script into a Rust CLI project with:

- `clap` derive macros for argument parsing (see [CLI Design](./shell-cli.md))
- `std::process::Command` for external process execution
  (see [Command Parsing](./shell-commands.md))
- `Result`-based error propagation replacing `set -e`
  (see [Error Handling](./shell-errors.md))

## Step 3 -- Optimize

```bash
batuta optimize ./deploy_cli
```

For shell-to-Rust conversions, the optimizer focuses on replacing sequential
pipe chains with parallel execution where data dependencies allow, and
replacing temporary files with in-memory buffers.

## Step 4 -- Validate

```bash
batuta validate ./deploy_cli --reference ./scripts/deploy.sh
```

Validation confirms that the Rust CLI produces identical stdout/stderr output
and exit codes for a set of test scenarios, including success, test failure,
push failure, and deployment timeout.

## Generated Rust CLI (simplified)

```rust
use anyhow::{Context, Result};
use clap::Parser;
use std::process::Command;

#[derive(Parser)]
#[command(name = "deploy")]
struct Args {
    /// Docker registry (default: ghcr.io/team)
    #[arg(long, env = "DOCKER_REGISTRY", default_value = "ghcr.io/team")]
    registry: String,

    /// Git SHA for image tag
    #[arg(long, env = "GIT_SHA")]
    tag: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tag = args.tag.unwrap_or_else(|| git_short_sha().unwrap());
    let image = format!("{}/app:{}", args.registry, tag);

    build_image(&image)?;
    run_tests(&image)?;
    push_image(&image)?;
    deploy(&image)?;

    Ok(())
}

fn build_image(image: &str) -> Result<()> {
    println!("Building {image}...");
    let status = Command::new("docker")
        .args(["build", "-t", image, "."])
        .status()
        .context("Failed to run docker build")?;
    if !status.success() {
        anyhow::bail!("docker build failed with {status}");
    }
    Ok(())
}
```

## Result

| Metric            | Bash          | Rust CLI       |
|-------------------|---------------|----------------|
| Error handling    | `set -e` only | Typed `Result` |
| Configuration     | Env vars      | Typed args     |
| Portability       | Linux + Bash  | Any OS         |
| Shell completion  | None          | Auto-generated |
| Binary            | Interpreted   | 2.1 MB static  |

## Key Takeaways

- Bashrs converts shell commands to `std::process::Command` calls with proper
  error checking on every invocation.
- Environment variables become typed `clap` arguments with defaults and
  validation.
- `set -e` semantics are replaced by `Result` propagation with contextual
  error messages at each step.
- The following sub-chapters detail command parsing, error handling, and CLI
  design patterns.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
