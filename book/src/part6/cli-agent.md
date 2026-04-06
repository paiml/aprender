# `batuta agent`

Sovereign agent runtime using the perceive-reason-act pattern.

## Synopsis

```bash
batuta agent run --manifest <MANIFEST> --prompt <PROMPT> [--max-iterations <N>] [--daemon]
batuta agent chat --manifest <MANIFEST>
batuta agent validate --manifest <MANIFEST>
batuta agent status --manifest <MANIFEST>
batuta agent sign --manifest <MANIFEST> [--signer <ID>] [--output <PATH>]
batuta agent verify-sig --manifest <MANIFEST> --pubkey <PATH> [--signature <PATH>]
batuta agent contracts
```

## Subcommands

### `run`

Execute a single agent invocation with the given prompt.

```bash
batuta agent run --manifest agent.toml --prompt "Summarize the codebase"
```

**Options:**

| Flag | Description |
|------|-------------|
| `--manifest <PATH>` | Path to agent manifest TOML file |
| `--prompt <TEXT>` | Prompt to send to the agent |
| `--max-iterations <N>` | Override max iterations from manifest |
| `--daemon` | Run as a long-lived service (for forjar deployments) |

### `chat`

Start an interactive chat session with the agent. Type `quit` or `exit` to end.

```bash
batuta agent chat --manifest agent.toml
```

The chat loop runs `run_agent_loop()` for each user message, maintaining
persistent memory across turns (recalled via BM25 when using TruenoMemory).

### `validate`

Validate an agent manifest without running it.

```bash
batuta agent validate --manifest agent.toml
```

### `status`

Display agent manifest summary, resource quotas, model config, and capabilities.

```bash
batuta agent status --manifest agent.toml
```

Reports validation errors (if any), manifest metadata, resource limits
(max iterations, tool calls, cost budget), model configuration, and
the list of granted capabilities.

### `sign`

Cryptographically sign an agent manifest using Ed25519 via pacha+BLAKE3.

```bash
batuta agent sign --manifest agent.toml --signer "admin@paiml.com"
batuta agent sign --manifest agent.toml --output agent.toml.sig
```

The manifest is normalized to canonical TOML before hashing to ensure
deterministic signatures regardless of whitespace or key ordering.

### `verify-sig`

Verify an Ed25519 signature on an agent manifest.

```bash
batuta agent verify-sig --manifest agent.toml --pubkey key.pub
batuta agent verify-sig --manifest agent.toml --pubkey key.pub --signature agent.toml.sig
```

### `contracts`

Display the design-by-contract invariants from `contracts/agent-loop-v1.yaml`.

```bash
batuta agent contracts
```

Shows all invariants (INV-001 through INV-007), their test bindings,
and verification targets (coverage, mutation, complexity thresholds).

## Agent Manifest

The agent manifest is a TOML file that configures the runtime:

```toml
name = "code-reviewer"
version = "0.1.0"
description = "Reviews code for quality issues"

[model]
model_path = "/models/llama3-8b.gguf"
max_tokens = 4096
temperature = 0.3
system_prompt = "You are a code review assistant."

[resources]
max_iterations = 20
max_tool_calls = 50
max_cost_usd = 0.0  # 0 = unlimited (sovereign)

capabilities = ["Rag", "Memory"]
privacy = "Sovereign"
```

## Architecture

The agent uses a perceive-reason-act loop (Toyota Way: Jidoka):

```
┌─────────────────────────────────────┐
│         Perceive (Memory Recall)    │
│  Recall relevant memories, augment  │
│  system prompt with context         │
├─────────────────────────────────────┤
│    Context Management [F-003]       │
│  Pre-subtract system+tool tokens,   │
│  truncate messages via SlidingWindow│
├─────────────────────────────────────┤
│         Reason (LLM Completion)     │
│  Send truncated conversation to     │
│  LlmDriver with retry+backoff      │
├─────────────────────────────────────┤
│         Act (Tool Execution)        │
│  Execute tools with capability      │
│  checks (Poka-Yoke), store results  │
├─────────────────────────────────────┤
│         Guard (Jidoka)              │
│  Check iteration limits, ping-pong  │
│  detection, cost budget             │
└─────────────────────────────────────┘
```

## Context Management

The agent integrates `serve::context::ContextManager` for token-aware
truncation before each LLM call. This prevents context overflow errors
and ensures long conversations degrade gracefully.

**Budget calculation:**

```
effective_window = driver.context_window()
                 - estimate_tokens(system_prompt)
                 - estimate_tokens(tool_definitions)
                 - output_reserve (max_tokens)
```

The system prompt and tool schemas are pre-subtracted from the window.
Only conversation messages are passed to the `SlidingWindow` truncation
strategy, which keeps the most recent messages when the budget is exceeded.

**Error modes:**

- If messages fit: no truncation, zero overhead
- If messages overflow: oldest messages dropped (SlidingWindow)
- If overflow after truncation: `AgentError::ContextOverflow`

## Retry with Exponential Backoff

Driver calls use automatic retry for transient errors:

| Error Type | Retryable | Backoff |
|------------|-----------|---------|
| `RateLimited` | Yes | 1s, 2s, 4s |
| `Overloaded` | Yes | 1s, 2s, 4s |
| `Network` | Yes | 1s, 2s, 4s |
| `ModelNotFound` | No | Immediate fail |
| `InferenceFailed` | No | Immediate fail |

Maximum 3 retry attempts with exponential backoff (base 1s).

## Safety Features

- **LoopGuard**: Prevents runaway loops (max iterations, tool call limits)
- **Ping-pong detection**: FxHash-based detection of oscillatory tool calls
- **Capability filtering**: Tools only accessible if manifest grants capability
- **Cost circuit breaker**: Stops execution when cost budget exceeded
- **Context truncation**: Automatic SlidingWindow truncation for long conversations
- **Consecutive MaxTokens**: Circuit-breaks after 5 consecutive truncated responses
- **Privacy tier**: Sovereign (local-only), Private, or Standard

## Daemon Mode

The `--daemon` flag runs the agent as a long-lived service process,
suitable for forjar deployments:

```bash
batuta agent run \
  --manifest /etc/batuta/agent.toml \
  --prompt "Monitor system health" \
  --daemon
```

Daemon mode:
- Runs the agent loop as a background service
- Responds to SIGTERM/SIGINT for graceful shutdown
- Designed for systemd integration via forjar provisioning

## Examples

```bash
# Validate a manifest
batuta agent validate --manifest examples/agent.toml

# Run with a prompt
batuta agent run \
  --manifest examples/agent.toml \
  --prompt "What are the main modules in this project?"

# Override iteration limit
batuta agent run \
  --manifest examples/agent.toml \
  --prompt "Find all TODO comments" \
  --max-iterations 5

# Run as daemon (forjar)
batuta agent run \
  --manifest examples/agent.toml \
  --prompt "Monitor logs" \
  --daemon
```

## Driver Backends

| Driver | Privacy Tier | Feature | Description |
|--------|-------------|---------|-------------|
| `RealizarDriver` | Sovereign | `inference` | Local GGUF/APR inference via realizar |
| `MockDriver` | Sovereign | `agents` | Deterministic responses for testing |
| `RemoteDriver` | Standard | `native` | HTTP to Anthropic/OpenAI APIs |
| `RoutingDriver` | Configurable | `native` | Local-first with remote fallback |

### RoutingDriver

The `RoutingDriver` wraps a primary (typically local/sovereign) and fallback
(typically remote/cloud) driver. Three strategies:

| Strategy | Behavior |
|----------|----------|
| `PrimaryWithFallback` | Try primary; on retryable error, spillover to fallback |
| `PrimaryOnly` | Primary only, no fallback |
| `FallbackOnly` | Fallback only, skip primary |

Privacy tier inherits the most permissive of the two drivers — if the
fallback is `Standard`, data *may* leave the machine on spillover.

### RemoteDriver

Supports both Anthropic Messages API and OpenAI Chat Completions API:

| Provider | Endpoint | Tool Format |
|----------|----------|-------------|
| Anthropic | `/v1/messages` | `tool_use` content blocks |
| OpenAI | `/v1/chat/completions` | `function` tool_calls |

Error mapping: HTTP 429 → RateLimited, 529/503 → Overloaded, other → Network.

## Builtin Tools

| Tool | Capability | Feature | Description |
|------|-----------|---------|-------------|
| `MemoryTool` | `Memory` | `agents` | Read/write agent persistent state |
| `RagTool` | `Rag` | `rag` | Search indexed documentation via BM25+vector |
| `ShellTool` | `Shell` | `agents` | Sandboxed subprocess execution with allowlisting |
| `ComputeTool` | `Compute` | `agents` | Parallel task execution via JoinSet |
| `BrowserTool` | `Browser` | `agents-browser` | Headless Chromium automation |

### ShellTool

Executes shell commands with capability-based allowlisting (Poka-Yoke):

- Only allowlisted commands are executable
- Working directory is restricted
- Output truncated to 8192 bytes to prevent context overflow
- Configurable timeout (default: 30 seconds)

### ComputeTool

Parallel task execution for compute-intensive workflows:

- Single task execution (`run` action)
- Parallel execution (`parallel` action) via tokio JoinSet
- Max concurrent tasks configurable (default: 4)
- Output truncated to 16KB per task
- Configurable timeout (default: 5 minutes)

### BrowserTool Actions

| Action | Input | Description |
|--------|-------|-------------|
| `navigate` | `{ "url": "..." }` | Navigate to URL (Sovereign: localhost only) |
| `screenshot` | `{}` | Take page screenshot (base64 PNG) |
| `evaluate` | `{ "expression": "..." }` | Evaluate JavaScript |
| `eval_wasm` | `{ "expression": "..." }` | Evaluate WASM expression |
| `click` | `{ "selector": "..." }` | Click CSS selector |
| `wait_wasm` | `{}` | Wait for WASM runtime readiness |
| `console` | `{}` | Get console messages |

## Programmatic Usage

### Basic Usage

```rust
use batuta::agent::manifest::AgentManifest;
use batuta::agent::driver::mock::MockDriver;
use batuta::agent::memory::InMemorySubstrate;
use batuta::agent::runtime::run_agent_loop;
use batuta::agent::tool::ToolRegistry;

let manifest = AgentManifest::default();
let driver = MockDriver::single_response("Hello!");
let registry = ToolRegistry::default();
let memory = InMemorySubstrate::new();

let result = run_agent_loop(
    &manifest,
    "Say hello",
    &driver,
    &registry,
    &memory,
    None,  // Optional stream event channel
).await?;

println!("Response: {}", result.text);
```

### Using AgentBuilder

```rust
use batuta::agent::AgentBuilder;
use batuta::agent::manifest::AgentManifest;
use batuta::agent::driver::mock::MockDriver;

let manifest = AgentManifest::default();
let driver = MockDriver::single_response("Built!");

let result = AgentBuilder::new(&manifest)
    .driver(&driver)
    .run("Hello builder")
    .await?;

println!("{}", result.text);  // "Built!"
```

### With Stream Events

```rust
use tokio::sync::mpsc;
use batuta::agent::AgentBuilder;
use batuta::agent::driver::StreamEvent;

let (tx, mut rx) = mpsc::channel(64);

let result = AgentBuilder::new(&manifest)
    .driver(&driver)
    .stream(tx)
    .run("Hello")
    .await?;

while let Ok(event) = rx.try_recv() {
    match event {
        StreamEvent::PhaseChange { phase } => {
            println!("Phase: {phase}");
        }
        StreamEvent::TextDelta { text } => {
            print!("{text}");
        }
        _ => {}
    }
}
```

## Quality Gates

The agent module passes all PMAT quality gates:

- **Zero** SATD comments (QA-001)
- **All** source files ≤500 lines (QA-002)
- **95%+** line coverage (QA-003)
- **Zero** cognitive complexity violations (QA-005)
- **16/16** design-by-contract invariants verified
- **27/27** integration demo scenarios passing

Run quality verification:

```bash
# Contract invariants
cargo run --example agent_contracts --features agents

# Full integration demos
cargo run --example agent_demo --features agents
```

## See Also

- [Architecture Overview](../part9/architecture-overview.md)
- [Toyota Way Principles](../part1/toyota-way.md)
- [Jidoka: Built-in Quality](../part1/jidoka.md)
