# Command Overview

Batuta provides a unified CLI for the entire transpilation-to-deployment pipeline, plus ML model serving, stack orchestration, and intelligent query interfaces.

## Pipeline Commands (5-Phase Workflow)

| Command | Phase | Description |
|---------|-------|-------------|
| [`batuta init`](./cli-init.md) | Setup | Initialize project with `batuta.toml` |
| [`batuta analyze`](./cli-analyze.md) | 1 | Analyze source codebase (languages, deps, TDG) |
| [`batuta transpile`](./cli-transpile.md) | 2 | Transpile source code to Rust |
| [`batuta optimize`](./cli-optimize.md) | 3 | MoE backend selection + Cargo profile tuning |
| [`batuta validate`](./cli-validate.md) | 4 | Verify semantic equivalence |
| [`batuta build`](./cli-build.md) | 5 | Build final binary (release, cross-compile, WASM) |

## Workflow Management

| Command | Description |
|---------|-------------|
| [`batuta status`](./cli-status.md) | Show current workflow phase and progress |
| [`batuta reset`](./cli-reset.md) | Reset workflow state to start over |
| [`batuta report`](./cli-report.md) | Generate migration report (HTML/Markdown/JSON) |

## Intelligence & Query

| Command | Description |
|---------|-------------|
| [`batuta oracle`](./cli-oracle.md) | Knowledge graph queries, RAG search, PMAT code search |
| [`batuta bug-hunter`](./cli-bug-hunter.md) | Popperian falsification-driven defect discovery |
| [`batuta falsify`](./cli-falsify.md) | Run Sovereign AI Assurance Protocol checklist |

## Agent Runtime

| Command | Description |
|---------|-------------|
| [`batuta agent`](./cli-agent.md) | Autonomous agent runtime (`--features agents`) |
| [`batuta playbook`](./cli-playbook.md) | Deterministic YAML pipelines with BLAKE3 caching |

## ML Model Ecosystem

| Command | Description |
|---------|-------------|
| [`batuta serve`](./cli-serve.md) | Serve models via Realizar (OpenAI-compatible API) |
| [`batuta deploy`](./cli-deploy.md) | Deploy to Docker, Lambda, K8s, Fly.io, Cloudflare |
| [`batuta mcp`](./cli-mcp.md) | MCP server for AI tool integration |
| [`batuta hf`](./cli-hf.md) | HuggingFace Hub integration |

## Stack & Data

| Command | Description |
|---------|-------------|
| [`batuta stack`](./cli-stack.md) | PAIML Stack dependency orchestration |
| [`batuta data`](./cli-data.md) | Data platform integration |
| [`batuta viz`](./cli-viz.md) | Visualization frameworks |
| [`batuta content`](./cli-content.md) | Content creation tooling |

## Global Options

All commands support these flags:

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-d, --debug` | Enable debug output |
| `--strict` | Enforce strict drift checking |
| `--allow-drift` | Allow drift warnings without blocking |
| `-h, --help` | Print help |
| `-V, --version` | Print version |

---

**Navigate:** [Table of Contents](../SUMMARY.md)
