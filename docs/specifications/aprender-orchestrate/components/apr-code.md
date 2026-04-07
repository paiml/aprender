# apr code Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Runtime: [agent-and-playbook.md](agent-and-playbook.md) (batuta agent engine)
> Inference: realizar (local GGUF/APR — Sovereign only)
> UX: [presentar-probar-integration.md](presentar-probar-integration.md) (TUI + testing)

---

## 1. Overview

`apr code` is the **agentic coding assistant** for the Sovereign AI Stack — the equivalent of Claude Code, but sovereign-first. It is a top-level subcommand of the `apr` CLI (the unified entrypoint for all stack operations) that provides an interactive, tool-using AI assistant for software engineering tasks.

**Key differentiator:** `apr code` runs **exclusively on local models** via realizar. All inference stays on your machine. No API keys, no cloud, no data egress — ever. This is the Sovereign AI Stack's core promise: your code never leaves your hardware.

### Design Principles

| Principle | Application |
|-----------|-------------|
| **Sovereign-only** | All inference via realizar on local hardware. No remote APIs. |
| **Minimal binary footprint** | Two Rust binaries: `apr` (CLI + inference via `apr serve`) and `batuta` (agent runtime). No npm, no Python, no Docker. `apr serve` auto-launched as subprocess; falls back to embedded RealizarDriver if `apr` not on PATH. |
| **Stack-native** | Deep integration with all 20+ PAIML crates (not just shell wrappers) |
| **Always offline** | No `--offline` flag needed — apr code is always sovereign, always local. Zero network by design. |
| **Provably correct** | UX contracts verified by probar, behavior contracts by provable-contracts |

### What apr code IS

- An interactive AI coding assistant in the terminal — **100% local, zero cloud**
- A tool-using agent that can read/write files, run commands, search code
- A sovereign alternative to Claude Code / Cursor / Codex
- Primary entrypoint: `apr code` (via `apr-cli` in the aprender workspace)
- Engine: `batuta code` (batuta agent runtime underneath)

### What apr code is NOT

- Not a model trainer (use `apr train` / `entrenar`)
- Not a model inspector (use `apr inspect` / `apr validate`)
- Not a deployment tool (use `apr serve` / `batuta playbook`)
- Not a transpiler (use `batuta transpile` / `depyler` / `decy`)

---

## 2. Architecture

```
User
  |
  v
apr code ("Fix the auth bug")       <-- apr-cli subcommand (aprender)
  |
  v
+----------------------------------------------+
|            apr-code Runtime                   |
|  (thin shim in apr-cli, delegates to batuta)  |
+------+--------------------+------------------+
       |                    |
       v                    v
+------+------+    +--------+---------+     +------------------+
| batuta      |    | AprServeDriver   |---->| apr serve run    |
| agent       |    | (PRIMARY, HTTP)  |     | (CUDA/GPU, full  |
| runtime     |    | auto-launch +    |     |  APR+GGUF, fast) |
| (perceive-  |    | localhost:port   |     +------------------+
|  reason-    |    +------------------+
|  act loop)  |    | RealizarDriver   |  <-- fallback (no apr binary)
+------+------+    | (embedded, CPU)  |
       |           +------------------+
       v
+------+-----------------------------+
|   presentar-terminal TUI           |
|   (streaming, tools, cost, etc.)   |
+------------------------------------+
       |
       v (uses stack tools natively)
+------+-----------------------------+
| pmat_query | rag     | shell   |   |
| file_read  | file_write | grep  |   |
| file_edit  | glob    | memory  |   |
+------------------------------------+
```

**PMAT-160: Inference via `apr serve` (first-class).** `batuta code` auto-launches `apr serve run <model>` as a subprocess on a random port, connects via OpenAI-compatible HTTP API. This gives full CUDA/GPU acceleration, APR+GGUF support, and avoids feature flag issues. If `apr` is not on PATH, falls back to embedded RealizarDriver (CPU-only).

### Crate Boundaries

| Crate | Role in apr code |
|-------|-----------------|
| **apr-cli** (aprender) | Defines `Code` subcommand variant (PMAT-182); thin dispatch to `batuta::agent::code::cmd_code()`. Feature-gated behind `code` (default). |
| **batuta** | Agent runtime, tool execution, session management, context compaction. Public entry: `agent/code.rs` |
| **presentar-terminal** | TUI rendering (6-panel adaptive layout) |
| **realizar** | Local LLM inference (Sovereign tier) |
| **trueno-rag** | Codebase indexing, semantic search |
| **renacer** | Syscall tracing for sandbox enforcement |
| **pmat** | Code quality queries via dedicated `PmatQueryTool` (PMAT-163) |
| **jugar-probar** | UX testing, state machine validation, **LLM load testing** (TTFT/TPOT/P99 SLO enforcement for `apr serve`) |
| **cgp** (trueno) | GPU/SIMD profiling, roofline analysis for inference kernels (Q4K/Q6K matvec, attention) |
| **provable-contracts** | Compile-time contract enforcement |

### Why apr, Not batuta?

`apr` is already the stack's universal CLI with 48+ subcommands for model operations (`apr run`, `apr inspect`, `apr serve`, `apr chat`). Users already know `apr`. Adding `apr code` makes the coding assistant discoverable alongside existing workflows:

```bash
apr run model.apr           # Run inference
apr chat model.apr          # Interactive chat
apr code                    # Agentic coding assistant  <-- NEW
apr serve model.apr         # Start server
apr inspect model.apr       # View model metadata
```

The alternative (`batuta agent run`) requires users to know about batuta as a separate tool. `apr code` is the user-facing surface; `batuta` is the engine.

---

## 3. User Experience

### 3.1 Launch

```bash
# Default: auto-detect from ~/.apr/models/ (prefers APR over GGUF)
apr code

# Explicit model — Qwen3 1.7B is the default (best tool-use at 1.2GB)
apr code --model ~/.apr/models/qwen3-1.7b-q4k.apr

# Larger model for complex tasks
apr code --model ~/.apr/models/qwen3-8b-q4k.apr

# With project context
apr code --project ./my-rust-project

# Non-interactive (single prompt)
apr code -p "Fix the auth bug in src/auth.rs"

# Resume previous session
apr code --resume
```

**Note:** `apr code` is the primary user-facing entrypoint (via `apr-cli` in the aprender workspace). `batuta code` is the engine underneath — same functionality, accessed when building batuta directly.

### 3.2 Interactive Session

```
$ apr code

  apr code 0.7.3 (Sovereign tier)
  Model: Qwen3-1.7B-Q4_K_M.gguf (GGUF)
  tip: Convert to APR for faster loading: apr convert --to-apr <model>.gguf

  Type a message, /help for commands, /quit to exit.

> Fix the authentication bug in src/auth.rs

  [Perceive] Recalling project context...
  [Reason]   Reading src/auth.rs...
  [Act]      file_read src/auth.rs
  [Act]      pmat query "authentication" --include-source
  [Reason]   Found the issue: token expiry check uses <= instead of <
  [Act]      file_edit src/auth.rs (line 47)
  [Act]      shell: cargo test auth
  [Remember] Fixed: changed <= to < in token expiry comparison

  Done. 1 file modified, 12 tests pass.
  Cost: $0.00 (local inference)  |  Context: 23%  |  Turn 1/50

>
```

### 3.3 Multi-Turn Conversation

Each REPL session maintains a persistent `Vec<Message>` history across turns. When the user submits a prompt:

1. `run_agent_turn()` receives the full history + new query
2. The agent loop processes tool calls within the turn (inner loop)
3. On completion, all new messages (user prompt, tool calls, tool results, assistant response) are committed to history
4. Next turn sees full context from all prior turns

**Context management:**
- `/context` shows token usage (`~N / window tokens (X%)`) with warning at 80%+
- `/compact` strips tool call/result details from older turns, preserving user queries and assistant summaries
- `/clear` resets history and screen
- **Auto-compaction at 80%** context window — triggers automatically after each turn (spec §7.3, PMAT-133)
- Automatic truncation via `ContextManager` (sliding window) keeps messages within the model's context window
- Project instructions (CLAUDE.md) scaled to 25% of context budget, skipped for <4K models (PMAT-142)

**Implementation:** `run_agent_turn()` in `src/agent/runtime.rs`, `compact_history()` in `src/agent/repl_display.rs` (PMAT-115).

### 3.4 Slash Commands

**Implemented (10 commands):**

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/test` | Run `cargo test --lib` |
| `/quality` | Run clippy + test quality gate |
| `/context` | Show token usage (N / window tokens, %) |
| `/compact` | Manually trigger context compaction |
| `/session` | Show current session info (ID, turns, messages) |
| `/sessions` | List recent sessions with resume instructions |
| `/cost` | Show session cost breakdown |
| `/clear` | Clear conversation history and screen |
| `/quit` | Exit apr code |

**Stub (not yet functional):**

| Command | Description | Status |
|---------|-------------|--------|
| `/model` | Switch model mid-session | Wired in enum, prints "not yet implemented" |
| `/sandbox` | Show sandbox/capability policy | Not implemented |

### 3.5 CLAUDE.md / APR.md Support

`apr code` reads project-level configuration files (like Claude Code reads CLAUDE.md):

**Discovery order (implemented):**
1. `APR.md` in project root (preferred, stack-native)
2. `CLAUDE.md` in project root (compatible, for projects also using Claude Code)

**Planned (not yet implemented):**
3. `~/.apr/APR.md` (global user-level)
4. `~/.apr/projects/{project-hash}/APR.md` (per-project user overrides)

**APR.md format:**

```markdown
# APR.md

## Project Instructions

This project uses the Sovereign AI Stack. Always use `pmat query` for code search.

## Build Commands

- `cargo build` — debug build
- `make test` — run tests
- `make lint` — clippy

## Coding Standards

- 95% test coverage required
- Zero clippy warnings
- All functions must have doc comments

## Privacy

- Tier: Sovereign (no external API calls)
- Allowed tools: file_read, file_write, shell, pmat_query
- Blocked tools: web_fetch, web_search
```

---

## 4. Tool System

### 4.1 Builtin Tools

**Phase 1 (implemented, PMAT-103 through PMAT-106):**

| Tool | Capability | Implementation | Status |
|------|-----------|----------------|--------|
| **file_read** | `FileRead` | `agent/tool/file.rs` — line range, 128KB limit, numbered output | Done |
| **file_write** | `FileWrite` | `agent/tool/file.rs` — create/overwrite, parent dir creation | Done |
| **file_edit** | `FileWrite` | `agent/tool/file.rs` — unique string replacement | Done |
| **shell** | `Shell` | `agent/tool/shell.rs` — allowlist, injection blocking, timeout | Done (pre-existing) |
| **glob** | `FileRead` | `agent/tool/search.rs` — pattern match, mtime sort, 200 cap | Done |
| **grep** | `FileRead` | `agent/tool/search.rs` — substring match, file glob filter, binary skip | Done |
| **memory** | `Memory` | `agent/tool/memory.rs` — remember/recall via InMemorySubstrate | Done (pre-existing) |

**Phase 4a (implemented, PMAT-163):**

| Tool | Capability | Implementation | Status |
|------|-----------|----------------|--------|
| **pmat_query** | `FileRead` | `agent/tool/pmat_query.rs` — semantic/regex/literal search, TDG grade filter, complexity filter, fault patterns, exclude-tests | Done (PMAT-163) |
| **rag** | `Rag` | `agent/tool/rag.rs` — trueno-rag search, starts empty, populated via `batuta oracle --rag-index` | Done (PMAT-153) |

**Available via shell fallback (dedicated tools planned):**

| Tool | Current Access | Planned Enhancement |
|------|---------------|-------------------|
| **cargo** | `shell: cargo test` / `/test` slash command | Dedicated tool with parsed results |
| **git** | `shell: git status` | libgit2 integration |
| **oracle** | `shell: batuta oracle "..."` | Direct oracle API |

**Future tools:**

| Tool | Dependency |
|------|-----------|
| **renacer_trace** | renacer integration |
| **apr_inspect** | apr-cli integration |

### 4.2 Tool-in-Prompt Architecture (Local Models)

Unlike API-based drivers (Anthropic/OpenAI) which accept tool definitions as structured parameters, local models via RealizarDriver need explicit tool definitions in the prompt text. The `build_enriched_system()` function in `chat_template.rs` (PMAT-121) appends:

1. **Tool definitions** — name, description, and compact JSON Schema for each tool
2. **Format instructions** — teaches the model to emit `<tool_call>` blocks
3. **Parsing contract** — `parse_tool_calls()` in `realizar.rs` extracts these blocks

This means the system prompt grows proportionally to the number of tools (estimated ~50 tokens per tool). With 9 tools, this adds an estimated ~450 tokens to context. Actual token count depends on the tokenizer used.

### 4.3 Stack-Native vs Shell Fallback

Where possible, tools use native Rust APIs instead of shelling out:

| Operation | Current (Phases 1-4a) | Future Enhancement |
|-----------|----------------------|-------------------|
| Code search | **`pmat_query` tool** (semantic/regex/literal, TDG grades — PMAT-163) | — |
| Build/test | `shell: cargo test` + `/test` slash command | Dedicated cargo tool |
| Git | `shell: git status` | libgit2 integration |
| Model ops | AprServeDriver (CUDA/GPU) → RealizarDriver fallback (CPU) | — |
| File ops | `file_read`/`file_write`/`file_edit` (native Rust I/O) | — |
| File search | `glob` tool (native glob crate) + `grep` tool (substring) | — |
| RAG | `rag` tool (trueno-rag, empty-index start — PMAT-153) | Auto-index on first run |

File tools use native Rust I/O. Everything else is available via `shell` tool subprocess. Slash commands (`/test`, `/quality`) provide one-keystroke shortcuts for common operations.

### 4.3 Tool Permission Model

Four active layers, two planned:

| Layer | Mechanism | Enforcement | Status |
|-------|-----------|-------------|--------|
| **Capability** | Manifest declares allowed tools per `Capability` enum | Application-level | **Done** — `capability_matches()` in runtime.rs |
| **Allowlist** | ShellTool validates command prefix against allowlist | Application-level | **Done** — injection blocking in shell.rs |
| **Path restriction** | FileRead/FileWrite tools check `allowed_paths` via `check_prefix()` | Application-level | **Done** — symlink traversal blocked |
| **Privacy tier** | Sovereign blocks network egress in agent loop | Application-level | **Done** — runtime.rs |
| **Hook** | Pre/post hooks intercept destructive actions | Application-level | **Planned** (Phase 5) |
| **Sandbox** | Landlock/Seatbelt restricts file/network access | Kernel-level | **Planned** (Phase 5) |

---

## 5. Sovereign Inference

### 5.1 Model Discovery

`apr code` auto-discovers local models when no `--model` flag is provided:

```
Search order:
1. ~/.apr/models/           (apr model cache — download via `apr pull`)
2. ~/.cache/huggingface/    (HF cache, converted on first use)
3. ./models/                (project-local models)
```

**Format preference:** Within each directory, `.apr` files are preferred over `.gguf` (APR is the stack's native format — faster loading, row-major layout, LZ4/ZSTD compression). Files sorted by modification time (newest first).

**PMAT-150 (Jidoka validation at discovery):** APR files are validated at discovery time — if an APR file lacks an embedded tokenizer (required for inference), it is deprioritized so valid GGUF files are tried first. This prevents the user from hitting a dead-end error when the only APR model was converted without tokenizer data. The CLI warns when GGUF fallback occurs due to an invalid APR file and suggests `apr convert` to fix it.

**Implementation:** `ModelConfig::discover_model()` in `src/agent/manifest.rs` (PMAT-116, PMAT-150). Validation via `is_valid_model_file()` in `src/agent/driver/realizar.rs`.

**Chat template auto-detection:** The prompt format is selected based on model filename:
- Qwen, DeepSeek, Yi → ChatML (`<|im_start|>role`)
- Llama → Llama 3.x format (`<|start_header_id|>role<|end_header_id|>`)
- Unknown → ChatML (most widely supported by instruct models)

### 5.2 Model Requirements

Minimum model capabilities for agentic coding:

| Capability | Minimum | Recommended |
|-----------|---------|-------------|
| Parameters | 1.7B+ (Qwen3 — 0.960 tool score) | 7B+ |
| Context window | 4K tokens | 8K+ tokens |
| Tool use | Function calling support | Native tool_use |
| Code generation | Instruction-following | Instruction-following + tool_use |
| Format | APR v2, GGUF, SafeTensors | APR v2 (fastest, preferred) |

**Default model: Qwen3 1.7B (APR/GGUF Q4K).** PMAT-179: Replaced Qwen2.5-Coder 1.5B after dogfood proved it cannot do tool-use. Qwen3 1.7B scores **0.960** on the [tool-calling benchmark](https://github.com/MikeVeerman/tool-calling-benchmark) — the highest of any sub-4B model. Native `<tool_call>` format support, thinking mode, 32K context. ~1.2GB at Q4K.

**Warning**: Models under 1B (TinyLlama, Qwen3 0.6B) may struggle with complex multi-tool tasks. Qwen3 0.6B scores 0.880 and works for simple queries but lacks judgment for chained operations. Qwen2.5-Coder 1.5B **cannot do tool-use at all** (PMAT-178 dogfood confirmed).

### 5.3 Recommended Models

| Model | Size | Tool Score | Format | Speed | Notes |
|-------|------|-----------|--------|-------|-------|
| **Qwen3 1.7B** | **1.2GB** | **0.960** | **APR/GGUF Q4K** | **Fast** | **Default — best tool-use for size (PMAT-179)** |
| Qwen3 0.6B | 397MB | 0.880 | APR/GGUF Q4K | Very fast | Ultra-constrained devices |
| xLAM-2 1B (Salesforce) | 986MB | 0.789 BFCL | GGUF Q4K | Fast | Function-calling specialist |
| Qwen3 8B | 5GB | — | APR/GGUF Q4K | Fast | Complex tasks, reasoning |
| Qwen3 32B | 20GB | — | APR/GGUF Q4K | Medium | Best quality |

**Tool Score**: From the [tool-calling benchmark](https://github.com/MikeVeerman/tool-calling-benchmark) (0-1 scale, higher = better). Qwen3 1.7B is the champion at 0.960.

All models run locally via realizar. Prefer APR format over GGUF (native, row-major, faster loading). Download with `apr pull <model>`.

### 5.4 Always Sovereign (Enforced)

`apr code` is **always sovereign** — there is no online mode. `build_default_manifest()` unconditionally returns `PrivacyTier::Sovereign` regardless of any parameter (PMAT-117):
- Zero network syscalls (verifiable by renacer)
- All inference via realizar (local GGUF/APR)
- All search via local grep/glob (pmat query via shell)
- All tools restricted to local filesystem
- No API keys required or accepted
- The `--offline` flag is a no-op (always offline by design)

---

## 6. Session Management

Inherits session persistence from agent-and-playbook.md section 11, with `apr code`-specific additions:

### 6.1 Session Storage

```
~/.apr/sessions/
  +-- {session_id}/
  |     +-- manifest.json        # Session metadata
  |     +-- messages.jsonl       # Conversation (append-only)
  |     +-- memory.json          # Session-local memory
  |     +-- cost.json            # Cost tracking
  |     +-- project_snapshot.json # Git state at session start
  |     +-- tools_log.jsonl      # All tool calls + results (audit)
```

### 6.2 Project Context

On session start, `apr code` captures:
- Git branch, HEAD commit, dirty files
- Project language (Rust/Python/etc.)
- APR.md / CLAUDE.md instructions
- File count, structure summary
- pmat quality baseline (if available)

### 6.3 Auto-Resume

If `apr code` detects a recent session (<24h) for the same project directory, it offers to resume:

```
$ apr code
  Found previous session (2h ago, 12 turns, $0.05)
  Resume? [Y/n]
```

---

## 7. Context Management

### 7.1 Project Indexing

On first run, `apr code` builds a trueno-rag index of the project:

```
Indexing /home/user/project...
  [P0] APR.md, CLAUDE.md, README.md
  [P1] Cargo.toml, Makefile, .github/
  [P2] src/**/*.rs (142 files)
  [P3] tests/**/*.rs (38 files)
  Index: 1,247 chunks, 892KB
  Time: 1.2s
```

Priority levels match batuta oracle's RAG indexing (P0 = project config, P1 = build config, P2 = source, P3 = tests).

### 7.2 Smart Context Selection

Before each LLM call, `apr code` selects relevant context:

1. **Always included**: System prompt + APR.md instructions
2. **RAG retrieved**: Top-K chunks relevant to current query (trueno-rag)
3. **Active files**: Files recently read/written in this session
4. **Tool results**: Recent tool outputs (subject to compaction)
5. **Git diff**: Uncommitted changes (if relevant)

### 7.3 Compaction

Two-phase compaction from agent-and-playbook.md section 7:
- Micro-compact at 60%: strip tool result bodies
- Auto-compact at 80%: summarize history, re-fetch from RAG

---

## 8. Integration with apr CLI

### 8.1 Subcommand Definition

In `apr-cli/src/commands_enum.rs`:

```rust
/// Sovereign AI coding assistant — all inference local via realizar.
Code {
    /// Path to local GGUF/APR model file (prefers .apr format)
    #[arg(long)]
    model: Option<PathBuf>,

    /// Project directory (loads APR.md/CLAUDE.md from this path)
    #[arg(long, default_value = ".")]
    project: PathBuf,

    /// Resume previous session (by ID, or most recent for cwd)
    #[arg(long)]
    resume: Option<Option<String>>,

    /// Agent manifest (advanced — overrides defaults)
    #[arg(long)]
    manifest: Option<PathBuf>,

    /// Initial prompt (non-interactive mode)
    #[arg(trailing_var_arg = true)]
    prompt: Vec<String>,

    /// Print mode (non-interactive, single response)
    #[arg(long, short)]
    print: bool,

    /// Max turns before stopping
    #[arg(long, default_value = "50")]
    max_turns: u32,
},
```

**Note:** No `--offline` flag — `apr code` is always offline. No `--tier` flag — always Sovereign. No `--budget` flag — local inference is free.

### 8.2 Dispatch

In `apr-cli/src/dispatch.rs` (PMAT-162):

```rust
#[cfg(feature = "code")]
Commands::Code { model, project, resume, prompt, print, max_turns, manifest } => {
    batuta::agent::code::cmd_code(
        model.clone(), project.clone(), resume.clone(),
        prompt.clone(), *print, *max_turns, manifest.clone(),
    ).map_err(|e| CliError::Aprender(e.to_string()))
}
```

### 8.3 Dependency

In `apr-cli/Cargo.toml`:

```toml
[dependencies]
batuta = { version = "0.7", path = "../../../batuta", optional = true }

[features]
code = ["dep:batuta"]  # apr code requires batuta
```

This keeps `apr` lightweight for model-only operations while enabling `apr code` when the full agent stack is needed.

---

## 9. Non-Interactive Mode

For CI/CD, scripts, and automation:

```bash
# Single prompt, print response, exit
apr code -p "Add error handling to src/parser.rs"

# Pipe input
echo "Fix the failing test" | apr code -p

# With specific model
apr code --model qwen3-1.7b-q4k.gguf -p "Explain src/main.rs"

# JSON output for tooling
apr code -p --json "List all functions in src/"
```

### 9.1 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success (task completed) |
| 1 | Agent error (tool failure, LLM error) |
| 2 | Budget exhausted |
| 3 | Max turns reached |
| 4 | Sandbox violation |
| 5 | No model available |

---

## 10. Configuration

### 10.1 Global Config (Planned — not yet implemented)

Config file reading is not yet wired into `cmd_code()`. The planned format:

```toml
# ~/.apr/config.toml

[code]
default_model = "~/.apr/models/qwen3-1.7b-q4k.gguf"
max_turns = 50
theme = "tokyo-night"
auto_resume = true

[code.tools]
# Tool allowlist (empty = all allowed)
allowed = []
# Tool blocklist
blocked = []
```

### 10.2 Environment Variables

| Variable | Purpose |
|----------|---------|
| `APR_CODE_MODEL` | Override default model path |
| `APR_CODE_THEME` | Override TUI theme |

---

## 11. Comparison with Claude Code

| Feature | Claude Code | apr code |
|---------|------------|----------|
| **Runtime** | Anthropic cloud (requires internet) | Local only via realizar (zero network) |
| **Cost** | $3-15 per million tokens | **Free** — local inference |
| **Privacy** | Data sent to Anthropic servers | **Sovereign** — code never leaves your machine |
| **Default model** | Claude (cloud) | **Qwen3 1.7B** (GGUF/APR, local, 0.960 tool score) |
| **Model formats** | Claude only | APR (preferred), GGUF, SafeTensors |
| **Binary** | Node.js + npm | Two Rust binaries: `apr` (CLI/inference) + `batuta` (agent) |
| **Tools** | ~15 builtin | 9 tools (incl. pmat_query, rag) + 10 slash commands + shell fallback |
| **Sessions** | Cloud-synced | JSONL at `~/.apr/sessions/` with `--resume` |
| **Context mgmt** | Automatic | Auto-compact at 80%, `/context` token tracking |
| **Project config** | CLAUDE.md | APR.md (preferred) + CLAUDE.md (compatible) |
| **Sandboxing** | Landlock/Seatbelt | Capability + allowlist + path restrict |
| **Quality** | No formal verification | provable-contracts (apr_model_validity, etc.) |

**The trade-off is clear:** Claude Code has frontier model quality but requires cloud. `apr code` has local-only inference (smaller models) but guarantees sovereignty. They serve different threat models.

---

## 12. Implementation Phases

| Phase | Scope | Status | Refs |
|-------|-------|--------|------|
| **1** | MVP: `batuta code` subcommand, REPL with slash commands, 7 tools (file_read/write/edit, glob, grep, shell, memory), MockDriver dry-run, `-p` non-interactive mode | **DONE** | PMAT-103 through 107 |
| **1b** | Real model: RealizarDriver with local GGUF via `--model` flag | **DONE** — model loads, agent loop initializes (7 tools, 4 caps). CPU inference slow on debug build. | PMAT-114 |
| **2a** | Multi-turn conversation history, model discovery (APR-preferred), always-Sovereign enforcement, chat template auto-detection (ChatML/Llama3/Generic) | **DONE** | PMAT-115 through 117 |
| **2b** | Tool definitions injected into prompt for local models, enriched system prompt with `<tool_call>` format, APR.md/CLAUDE.md project instruction loading, session persistence (JSONL at `~/.apr/sessions/`) | **DONE** | PMAT-121 through 124 |
| **2c** | `--resume` and `--project` CLI flags, `/session` and `/sessions` slash commands, session resume wired end-to-end, integration tests (session roundtrip, tool injection, multi-turn) | **DONE** | PMAT-129 through 131 |
| **3a** | Auto-compaction at 80% context window (spec §7.3), token usage tracking in `/context`, `/test` and `/quality` shortcut commands | **DONE** | PMAT-133 through 135 |
| **3b** | `agents` in default features (binary ships with `code` subcommand), APR format awareness (GGUF→APR conversion tip), improved no-model UX | **DONE** | PMAT-136 through 138 |
| **3c** | Spec v2.4.0 reconciliation, project context enrichment (git, language, build system) | **DONE** | PMAT-139, PMAT-140 |
| **3d** | `inference` in default features (RealizarDriver available), context-aware prompt budgeting (scales CLAUDE.md to model context window, skips for <4K models) | **DONE** | PMAT-141, PMAT-142 |
| **3e** | **Contract: `apr_model_validity`** — APR files validated at load boundary (Jidoka). Missing tokenizer caught before REPL, not at inference. GGUF magic validated. 5 falsification tests. | **DONE** | PMAT-144, PMAT-145 |
| **3f** | Output sanitization — strip echoed system prompt from small model responses, strip leaked chat template markers, model size warning for <2K context | **DONE** | PMAT-146, PMAT-147 |
| **3g** | **Jidoka model discovery** — validate APR tokenizer at discovery time (not just at load). Invalid APR deprioritized behind valid GGUF. Tightened header scan to reject `vocab_size`-only metadata. UX warning on GGUF fallback. | **DONE** | PMAT-150 |
| **3h** | **Exit codes 2/3/4** — map `CircuitBreak`→2 (budget), `MaxIterationsReached`→3 (max turns), `CapabilityDenied`→4 (sandbox) in non-interactive mode. Constants in `exit_code` module. | **DONE** | PMAT-152 |
| **3i** | **Wire RagTool** — register `RagTool` in `build_code_tools()` with empty-index oracle. Adds `Capability::Rag` to manifest. 8 tools total. Index populated via `batuta oracle --rag-index`. | **DONE** | PMAT-153 |
| **3j** | **AprServeDriver** — auto-launch `apr serve run` as subprocess, connect via OpenAI HTTP API. Full CUDA/GPU via apr-cli. Conditional `no_gpu` for APR (wgpu bug). Lenient tool_call parser (unclosed XML + markdown code blocks). | **DONE** | PMAT-158, PMAT-160 |
| **3k** | **AprServeDriver cleanup** — remove debug `[PMAT-160]` eprintln, conditional `--gpu` flag (GGUF gets GPU, APR skips due to wgpu shader bug). Clean user-facing output. | **DONE** | PMAT-164 |
| **4a** | **Dedicated `pmat_query` tool** — replaces `shell: pmat query "..."` fallback with structured `PmatQueryTool`. Supports semantic, regex, literal search with TDG grade/complexity/fault filters. 9 tools total (10 with RAG). 7 tests. | **DONE** | PMAT-163 |
| **4c** | **Auto-resume prompt** — interactive [Y/n] when recent session (<24h) exists for cwd. 24h age filter in `find_recent_for_cwd`. Age display from chrono timestamps. | **DONE** | PMAT-165 |
| **4d** | **Graceful shutdown** — AprServeDriver Drop sends SIGTERM, waits 2s, then SIGKILL. Clean model unload. | **DONE** | PMAT-166 |
| **4e** | **System prompt optimized for small models** — explicit 9-tool table with example inputs (file_read, pmat_query, shell, etc.). Replaces "tools listed below" with concrete patterns for 1.5B-7B. | **DONE** | PMAT-168 |
| **4f** | **`/cost` local inference UX** — shows "free (local)" instead of misleading $0.00. Token counts always shown. | **DONE** | PMAT-169 |
| **4g** | **max_tokens raised to 1024** — 512 cap truncated file edits mid-output. 1024 accommodates tool calls + code blocks. | **DONE** | PMAT-170 |
| **4h** | **apr serve stderr capture** — on startup failure or subprocess death, reads stderr, shows last lines + debug command. Detects crash during `wait_for_ready` via `try_wait`. | **DONE** | PMAT-171 |
| **4b** | Stack-native tools: git integration (libgit2), auto-index on first run | Planned | |
| **4i** | **Stuck-loop detection + `-p` mode hardening** — `run_agent_turn` detects 4+ identical tool calls and breaks. `-p` mode caps iterations at 10 (not 50). Prevents silent budget exhaustion on small model loops. | **DONE** | PMAT-172 |
| **4j** | **Tool format alignment** — system prompt `## Tools` section stripped correctly by AprServeDriver (was only stripping `## Available Tools`). HTTP compact list now uses `<tool_call>` format matching the parser. Eliminates conflicting format instructions. | **DONE** | PMAT-173 |
| **4k** | **Piped stdin guard** — `offer_auto_resume()` skips when stdin is not a TTY (`IsTerminal` check). Prevents piped input from being consumed by the resume prompt. | **DONE** | PMAT-174 |
| **4l** | **Shell wildcard mode** — injection filter (`;`, `|`, `&&`, `` ` ``) skipped in wildcard mode (`allowed_commands: ["*"]`). Coding tasks can now use pipes, chains, subshells. Restricted mode still blocks injection. | **DONE** | PMAT-175 |
| **4m** | **Preserve tool table for small models** — AprServeDriver now only strips verbose `## Available Tools` (JSON schemas). Keeps compact `## Tools` table (names + examples) from CODE_SYSTEM_PROMPT. 1.5B models now see tool descriptions over HTTP. | **DONE** | PMAT-176 |
| **4n** | **Tool-use nudge** — `run_agent_loop_with_nudge()` retries once when model returns EndTurn without tool calls. Used by `-p` mode. Nudge says "Use a tool. Emit a `<tool_call>` block." Generic `run_agent_loop` unchanged. | **DONE** | PMAT-177 |
| **4o** | **Default model → Qwen3 1.7B** — replaced Qwen2.5-Coder 1.5B (can't do tool-use) with Qwen3 1.7B (0.960 tool-calling score, [benchmark](https://github.com/MikeVeerman/tool-calling-benchmark)). Updated spec recommended models, launch examples, error messages, falsification §14.2. Qwen3 has native `<tool_call>` format matching our parser. | **DONE** | PMAT-179 |
| **4p** | **GGUF GPU disabled + Qwen3 thinking blocks stripped** — removed default `--gpu` for all GGUF (Qwen3 produces garbage with CUDA). Added `strip_thinking_blocks()` to remove `<think>...</think>` and bare `</think>` from responses. **BLOCKER:** `apr serve` doesn't support `enable_thinking=false` — Qwen3 GGUF loops on `</think>` tokens. Needs realizar fix (PMAT-181). | **DONE** (batuta side) | PMAT-180, PMAT-181 |
| **4q** | **Qwen3NoThinkTemplate in realizar** — ChatML variant that pre-fills empty `<think>\n</think>` block so Qwen3 skips thinking mode. Fixed incomplete `ChatMLTemplate` trait impl. Cached GGUF architecture in `AppState::with_quantized_model_and_vocab()` so `detect_format_from_name("qwen3")` auto-selects `Qwen3NoThinkTemplate`. | **DONE** | PMAT-181 |
| **4r** | **Model discovery: mtime-first** — `discover_model()` now sorts valid > newest > APR (was valid > APR > newest). Prevents broken-for-tool-use APR from shadowing better GGUF. Added model name logging in `-p` mode. | **DONE** | PMAT-185 |
| **4s** | **-p mode empty-response diagnostic** — `run_single_prompt()` detects empty text after thinking-block strip. Prints warning: "Model may be in thinking mode — rebuild apr for Qwen3NoThinkTemplate fix." Root fix: publish realizar with Qwen3NoThinkTemplate + architecture cache (PMAT-181). JSON parse error was bash test bug, not realizar. | **DONE** | PMAT-190 |
| **4t** | **Provable contracts for apr-cli components** — `http-api-v1` (5 FALSIFY tests), `session-v1` (4 FALSIFY tests), `tokenizer-v1` (5 FALSIFY tests). Total: 46+ enforcement tests across batuta + realizar. | **DONE** | PMAT-189 |
| **4u** | **Contract enforcement gap closure** — 15 new FALSIFY tests closing 9 equation gaps. Serve: format detection (APR/GGUF magic bytes), graceful shutdown (ready flag drain), degraded health (latency), Prometheus format, mmap threshold. HTTP: ChatCompletionRequest/Response schema, tool definitions roundtrip, SSE event format. Chat: KV-cache token growth, JSONL session roundtrip, multi-turn ordering. CLI: idempotent inspection, tokenize/data dispatch. Batuta: sovereignty guarantee, capability match, APR format preference, tool_call format, context window, session dir. runtime_helpers.rs extracted to keep runtime.rs under 500 lines. | **DONE** | PMAT-190 |
| **4v** | **Tokenizer + data pipeline FALSIFY enforcement** — 12 new tests in apr-cli: BPE roundtrip (ASCII, multi-word), vocab size bounds, deterministic encoding, empty input handling, merge order preservation, byte coverage (trained chars), token ID bounds, thread safety, nonexistent path rejection, training vocab consistency, WordPiece roundtrip. **Findings:** BPE training non-deterministic (HashMap iteration order), byte encoder scoped to training corpus (not byte-level). | **DONE** | PMAT-191, PMAT-192 |
| **4w** | **Popperian falsification of spec claims** — 10 FALSIFY-SPEC tests attempting to break: sovereignty override, shell wildcard mode, file path unrestriction, max iterations=50, temperature=0.0, search dirs completeness, project instruction loading, all 9 tools in prompt, exit code distinctness, zero cost. All 10 pass — no spec claim disproved. **Dogfood findings:** model discovery correctly picks Qwen3 GGUF (mtime-first), AprServeDriver launches in 1.5s, CPU inference slow (CUDA not enabled, PMAT-159). **Total: 82+ FALSIFY tests across batuta + apr-cli (6241 batuta tests).** | **DONE** | PMAT-185 |
| **4x** | **`apr serve loadtest` + `apr serve bench` wired into apr-cli** — `Loadtest` and `Bench` variants added to `ServeCommands` in `serve_commands.rs`. Dispatch to `jugar-probar` LLM module via `serve_loadtest.rs`. Feature-gated behind `llm-loadtest` (default with `inference`). `jugar-probar` upgraded 0.4→1.0 with `llm` feature. 8 FALSIFY-LT tests: SLO TTFT/TPOT enforcement, prompt loading (default + JSONL), empty rejection, baseline roundtrip. Fixed pre-existing `TuiFrame::from_buffer` → `from_lines` API break from probar 1.0. `apr serve loadtest --help` verified. | **DONE** | PMAT-196 |
| **4y** | **Model discovery integration tests** — 9 FALSIFY-DISC-10x tests with real temp files: valid/invalid APR with magic bytes, valid GGUF, mtime-beats-format, APR tiebreak at same mtime, Jidoka deprioritization (newer invalid APR behind older valid GGUF), four-way mixed sort, nonexistent path, format detection. Uses `filetime` for controlled mtimes. | **DONE** | PMAT-184 |
| **4z** | **Finetune provable contract `apr-finetune-v1`** — YAML contract with 5 equations (rank_bounds_safety, vram_feasibility, alpha_rank_ratio, merge_tensor_shape, checkpoint_metadata_roundtrip). 8 FALSIFY-FT tests in apr-cli: rank zero degenerate, default alpha, merge shape preserved, QLoRA < LoRA memory, memory monotonic with rank, merge identity at alpha=0, method variants, OptimalConfig fields. PMAT-182 verified DONE (full Code wiring in apr-cli). **Total: 6250 batuta tests, 105+ FALSIFY tests.** | **DONE** | PMAT-193, PMAT-182 |
| **5** | Hooks, Landlock/Seatbelt OS sandbox enforcement | Planned | |
| **6** | **`batuta` library API** — `cmd_code()` in `agent/code.rs` as public library entrypoint (PMAT-162). `trueno-explain` made optional behind `cuda` feature (PMAT-167). | **DONE** | PMAT-162, PMAT-167 |
| **6b** | **`apr-cli` wiring** — `Code` variant added to `commands_enum.rs` behind `code` feature flag (default). Dispatch calls `batuta::agent::code::cmd_code()`. `batuta` dep added with `agents`+`agents-inference`+`rag` features. `apr code --help` verified end-to-end. Also fixed: realizar `ChatMLTemplate` missing trait methods, apr-cli BrickStats type inference, trueno SyncMode version mismatch. | **DONE** | PMAT-182 |
| **6c** | **`batuta code -p` end-to-end working** — Two fixes: (1) COMPACT_SYSTEM_PROMPT for -p mode — minimal "Answer the question" framing instead of full 9-tool table + CLAUDE.md that overwhelmed Qwen3 1.7B causing `</think>` loops. (2) Context window 4096→32768 — the default 4096 was consumed entirely by 9 tool JSON schemas (~4000 tokens), causing `truncate_messages` to drop the user query. Now `batuta code -p "What is 2+2?"` → `"2 + 2 = 4."` and `batuta code -p "Write a Rust function..."` → complete working code. Also removed nudge from -p mode (forced unnecessary tool calls on simple questions). | **DONE** | PMAT-197 |
| **6d** | **Prompt scaling by model size** — `estimate_model_params_from_name()` parses parameter count from filename (e.g., `Qwen3-1.7B` → 1.7). `scale_prompt_for_model()` selects prompt tier: <2B → COMPACT (direct answer), 2-7B → MID (tool names, no examples), 7B+ → FULL (table with examples). Applied after `discover_and_set_model()` in `cmd_code()` for both `-p` and interactive modes. 8 tests (5 param parsing, 3 prompt selection). Interactive REPL startup verified: banner, session, APR conversion tip all display correctly. **6258 batuta tests.** | **DONE** | PMAT-198 |
| **7** | Probar testing, Brick UX contracts, visual regression baselines | Planned | |

---

## 13. Provable Contracts

### 13.1 Agent Contract (`apr-code-v1.yaml`)

See `../provable-contracts/contracts/batuta/apr-code-v1.yaml`. Key equations:

| Equation | Property |
|----------|----------|
| `sovereignty_guarantee` | Zero network syscalls — all inference local via realizar (renacer verifiable) | FALSIFY-CODE-001 |
| `tool_safety` | Every tool call passes capability + allowlist + path restriction checks | FALSIFY-CODE-002 |
| `session_integrity` | resume(persist(session)) reproduces identical state | FALSIFY-CHAT-007 |
| `apr_md_compliance` | Agent respects all APR.md instructions (blocked tools, coding standards) | — |
| `no_model_error` | If no local model found, clear error + download instructions + exit code 5 (never silent failure) | FALSIFY-DISC-003 |
| `apr_model_validity` | **APR files validated at load boundary AND discovery time (Jidoka)** (PMAT-144, PMAT-150) | FALSIFY-DISC-002 |
| `apr_format_preferred` | APR format preferred over GGUF at same mtime (stack-native) | FALSIFY-CODE-003 |
| `tool_call_format` | System prompt teaches `<tool_call>` XML format for local model parsing | FALSIFY-CODE-004 |
| `single_binary` | Works with only `apr` binary — no npm, Python, Docker | — |
| `startup_latency` | Cold start < 2s on NVMe with 1000-file project | — |

### 13.2 Chat Template Contract (`chat-template-v1.yaml`) — PMAT-187

See `../provable-contracts/contracts/realizar/chat-template-v1.yaml`. Created after 3 dogfood bugs shipped without contract enforcement.

| Equation | Property | Falsification Test |
|----------|----------|-------------------|
| `trait_completeness` | All `ChatTemplateEngine` impls have all 5 required methods | FALSIFY-CT-005 |
| `architecture_aware_selection` | Qwen3 models MUST get `Qwen3NoThink`, never `ChatML` | FALSIFY-CT-001 |
| `appstate_architecture_cache` | GGUF `AppState` constructors MUST cache architecture | FALSIFY-CT-002 |
| `thinking_block_suppression` | `strip_thinking_blocks()` removes all `<think>` tags | FALSIFY-CT-003 |
| `format_conversation_determinism` | Same messages always produce same prompt | FALSIFY-CT-004 |

**Bug found by contract:** `Qwen3NoThinkTemplate::format()` returned `TemplateFormat::ChatML` instead of `Qwen3NoThink`. The `falsify_ct_create_template_roundtrip` test caught this immediately — proves the contract's value.

**Test locations:**
- FALSIFY-CT-001/004/005/006: `realizar/src/chat_template_contract_tests.rs`
- FALSIFY-CT-002: `realizar/src/api/tests/chat_template_contract.rs`
- FALSIFY-CT-003: `batuta/src/agent/driver/apr_serve.rs`

### 13.3 CLI Dispatch Contract (`cli-dispatch-v1.yaml`) — updated PMAT-192

See `../provable-contracts/contracts/aprender/cli-dispatch-v1.yaml`. Updated to include `code` subcommand and feature-gated dispatch.

| Equation | Property | Falsification Test |
|----------|----------|-------------------|
| `dispatch_completeness` | Every Commands variant has a dispatch arm | FALSIFY-CLI-001, CLI-001b |
| `feature_gated_dispatch` | `Code` requires `code` feature, dispatches to batuta | FALSIFY-CLI-005, CLI-006, CLI-006b, CLI-006c |
| `exit_code_semantics` | Distinct errors → distinct non-zero codes | FALSIFY-CLI-002, CLI-002b |
| `output_format_fidelity` | `--help` produces parseable output | FALSIFY-CLI-003 |

**Test locations (apr-cli):**
- FALSIFY-CLI-001/001b/002/002b/003/006/006b/006c: `apr-cli/tests/falsification_chat_http_cli.rs` (PMAT-192)
- FALSIFY-CLI-005/005b/005c: `apr-cli/tests/includes/cli_contract_dispatch.rs` (PMAT-188)

### 13.4 Serve Contract (`apr-serve-v1.yaml`) — updated PMAT-191

See `../provable-contracts/contracts/aprender/apr-serve-v1.yaml`. Updated with chat template dispatch, format detection, and 15 enforcement tests.

| Equation | Property | Falsification Test |
|----------|----------|-------------------|
| `server_lifecycle` | Init→Binding→Loading→Ready→Draining→Stopped | FALSIFY-SRV-001, SRV-001b |
| `request_routing` | Error responses are valid JSON with envelope | FALSIFY-SRV-003, SRV-003b, SRV-003c |
| `concurrent_inference_isolation` | Metrics thread-safe, no cross-request contamination | FALSIFY-SRV-004 |
| `chat_template_dispatch` | Architecture-based template selection (Qwen3→NoThink) | FALSIFY-SRV-005 |
| `health_response` | HealthResponse has all required fields, status lowercase | FALSIFY-SRV-006, SRV-007 |
| `format_detection` | GGUF vs APR detected by magic bytes, not extension | — |

**Test locations (apr-cli):**
- FALSIFY-SRV-001 through SRV-007 + FALSIFY-HTTP-001 through HTTP-004b: `apr-cli/src/commands/serve/tests_contract_enforcement.rs` (PMAT-191)

### 13.4b Chat Session Contract (`apr-chat-session-v1.yaml`) — PMAT-192

See `../provable-contracts/contracts/aprender/apr-chat-session-v1.yaml`.

| Equation | Property | Falsification Test |
|----------|----------|-------------------|
| `chat_template_application` | Template idempotent — no double-wrapping of ChatML markers | FALSIFY-CHAT-001 |
| `session_state_machine` | Message serde roundtrip preserves role + content | FALSIFY-CHAT-003, CHAT-003b |
| `kv_cache_management` | History is append-only — previous messages unchanged | FALSIFY-CHAT-005 |

**Test locations (apr-cli):**
- FALSIFY-CHAT-001/003/003b/005: `apr-cli/tests/falsification_chat_http_cli.rs` (PMAT-192)

### 13.5 Model Discovery Contract (`apr-model-discovery-v1.yaml`) — PMAT-188

See `../provable-contracts/contracts/batuta/apr-model-discovery-v1.yaml`.

| Equation | Property | Falsification Test |
|----------|----------|-------------------|
| `search_order` | ~/.apr/models/ → ~/.cache/huggingface/ → ./models/ | FALSIFY-DISC-004 |
| `sort_priority` | valid > mtime > APR (mtime beats format) | FALSIFY-DISC-001 |
| `jidoka_validation` | Invalid APR deprioritized behind valid GGUF | FALSIFY-DISC-002 |
| `architecture_extraction` | GGUF architecture cached in AppState | FALSIFY-DISC-005 |
| `no_model_ux` | No model → exit 5 + download instructions | FALSIFY-DISC-003 |

### 13.6 HTTP API Contract (`http-api-v1.yaml`) — PMAT-189

See `../provable-contracts/contracts/batuta/http-api-v1.yaml`. Governs the OpenAI-compatible HTTP endpoint used by AprServeDriver.

| Equation | Property | Falsification Test |
|----------|----------|-------------------|
| `body_schema_compliance` | Request body matches OpenAI chat/completions schema | FALSIFY-HTTP-001 |
| `max_tokens_cap` | max_tokens capped at 1024 (AprServeDriver hardcap) | FALSIFY-HTTP-002 |
| `tool_format_fidelity` | Tool definitions roundtrip correctly through HTTP body | FALSIFY-HTTP-003 |
| `thinking_block_strip` | `<think>...</think>` and bare `</think>` stripped from HTTP responses | FALSIFY-HTTP-004 |
| `response_schema` | Response has `choices[0].message.content` path | FALSIFY-HTTP-005 |

### 13.7 Session Contract (`session-v1.yaml`) — PMAT-189

See `../provable-contracts/contracts/batuta/session-v1.yaml`. Governs JSONL session persistence.

| Equation | Property | Falsification Test |
|----------|----------|-------------------|
| `jsonl_roundtrip` | Messages survive persist→resume without loss | FALSIFY-SESSION-001, -002 |
| `manifest_serde` | Manifest fields (id, turns, cwd) roundtrip via JSON | FALSIFY-SESSION-003 |
| `age_filter` | `find_recent_for_cwd` returns only sessions < 24h old | FALSIFY-SESSION-004 |
| `append_only` | Messages are only appended, never mutated or deleted | — |

### 13.8 Tokenizer Contract (`tokenizer-v1.yaml`) — PMAT-189

See `../provable-contracts/contracts/batuta/tokenizer-v1.yaml`. Governs tokenizer correctness for context window management.

| Equation | Property | Falsification Test |
|----------|----------|-------------------|
| `deterministic_encode` | Same input always produces same token count | FALSIFY-TOK-001 |
| `empty_input` | Empty string produces zero tokens | FALSIFY-TOK-002 |
| `vocab_size_bound` | All token IDs < vocab_size | FALSIFY-TOK-003 |
| `thread_safety` | Concurrent tokenization produces consistent results | FALSIFY-TOK-004 |
| `roundtrip` | encode→decode preserves content (modulo normalization) | FALSIFY-TOK-005 |

---

## 14. Falsification

### 14.1 Falsifiable Claims

| Claim | Test | What Failure Means |
|-------|------|-------------------|
| **Zero network calls** | renacer trace of full session; assert zero connect/sendto syscalls | Sovereignty broken — data egress possible |
| **Local inference meets minimum quality** | Benchmark tool-use accuracy on 50 coding tasks with Qwen3 1.7B (default) and 8B | Local models can't do tool-use — apr code is a chat-only toy |
| **Single binary, no external deps** | Build `apr code` on clean machine; run without npm/Python/Docker | Dependency leaked — packaging broken |
| **APR.md instructions are followed** | Block `web_fetch` in APR.md; attempt web fetch; assert blocked | Config file ignored — trust broken |
| **Session resume is lossless** | Save session at turn 20, resume, compare next 5 turns (temp=0) | Resume corrupts context — user loses work |
| **No model = clear error** | Run `apr code` with no models installed; assert helpful error message with download instructions | User confused, no path to fix |
| **Startup time < 2s** | Cold start with 1000-file project on NVMe | Too slow — users won't wait |
| **pmat query outperforms grep for code tasks** | 50 code search tasks: compare pmat query vs grep for result relevance | Stack-native tools don't justify complexity |

### 14.1.1 Dogfood Findings (2026-04-03)

| Finding | Resolution | Ref |
|---------|-----------|-----|
| **P0: APR Q4K converter missing tokenizer** | `save_model_tensors_q4k()` in aprender never called `insert_tokenizer_metadata()`. All other APR creation paths embedded it. Root cause: Q4K fallback path (from SafeTensors) built metadata without tokenizer. Fix: pass `gguf_tokenizer` and call `insert_f32_tokenizer_metadata()`. | PMAT-154 |
| **P0: AprTransformer reads Q4K bytes as F32** | `try_load_llama_style()` in realizar loaded quantized APR files through the F32-only `AprTransformer` path, producing garbage output. Fix: `has_quantized_tensors_apr()` check skips `AprTransformer` for Q4K/Q6K APR files, routing to `OwnedQuantizedModel` (keeps weights quantized). | PMAT-156 |
| **Conditional no_gpu: APR only** | wgpu shader panicked on `-inf` literal for APR. Initially set `no_gpu: true` globally — but this forced GGUF to CPU too (minutes per response, unusable). Fixed: `no_gpu` only for `.apr` extension. GGUF gets GPU when available. | PMAT-156, PMAT-158 |
| **Tool call parser: unclosed tags + markdown blocks** | Qwen2.5-Coder 1.5B emits tool calls as: (a) `<tool_call>` without `</tool_call>`, or (b) ` ```json ` markdown blocks. Parser now handles all three formats: XML with closing tag, XML without, and markdown code blocks. Only JSON with `"name"` field is treated as tool call. | PMAT-158 |
| **GGUF CPU-only is unusable** | Without `cuda` feature, GGUF inference dequantizes to F32 on CPU — takes minutes for a single response. Batuta doesn't enable `realizar/cuda`. **Open issue:** need either `cuda` feature flag or `apr serve` HTTP backend for interactive use. | PMAT-158 |
| **APR on crates.io still broken** | `realizar 0.8.3` on crates.io lacks PMAT-156 fix (`has_quantized_tensors_apr`). APR Q4K inference only works with local path dep. Needs `realizar 0.8.4` publish. | PMAT-157 |
| **Validation markers didn't match real APR format** | Batuta's `validate_apr_header` checked for `"\"merges\""` (literal `"merges"` with quotes) and `tokenizer_vocab`. Real APR uses `tokenizer.merges` and `tokenizer.vocabulary`. Fixed markers. | PMAT-154 |
| **Missing contract: APR tokenizer at write time** | `model-format-conversion-v1.yaml` had no equation for tokenizer embedding. Added `apr_tokenizer_embedding` equation + FALSIFY-CONV-007 test. | PMAT-154 |
| **APR tokenizer validation false positive** | `validate_apr_header` matched `"vocab_size"` as containing `"vocab"`. Tightened to require `tokenizer.merges`, `tokenizer.vocabulary`, or `tokenizer.ggml`. | PMAT-150 |
| **APR-preferred discovery hits dead end** | `discover_model()` selected broken APR over valid GGUF. Added Jidoka validation at discovery: invalid APR deprioritized so GGUF wins. | PMAT-150 |
| **No UX for GGUF fallback** | When APR is skipped, user saw no explanation. Added warning with `apr convert` instructions. | PMAT-150 |
| **`validate_model_file` read entire file** | Used `std::fs::read` on 1.1GB APR file to get 64KB header. Fixed to `File::take(65536)`. | PMAT-150 |
| **TinyLlama echoes system prompt** | Chat model (not coding model) regurgitates instructions. Output sanitization catches some markers but model lacks tool-use ability entirely. Not a bug — expected with non-coding models. | — |
| **AprServeDriver always passed `--gpu`** | Even for APR files where wgpu shader panics on `-inf`. Fixed: conditional GPU flag — GGUF gets `--gpu`, APR skips it. | PMAT-164 |
| **Debug `[PMAT-160]` output in production** | `AprServeDriver::launch()` and `wait_for_ready()` emitted debug-prefixed eprintln. Cleaned to user-facing messages. | PMAT-164 |
| **`cmd_code` in binary crate, not library** | `cli/code.rs` was `mod cli` in `main.rs` — inaccessible to apr-cli. Refactored to `agent/code.rs` in library crate. Binary `cli/code.rs` is now a thin wrapper. | PMAT-162 |
| **Shell fallback for pmat query** | Agent used `shell: pmat query "..."` instead of a dedicated tool. Structured `PmatQueryTool` now provides semantic/regex/literal search with grade, complexity, fault filters. | PMAT-163 |
| **Pre-existing clippy: `unwrap()` in tool call parser** | `parse_tool_calls()` used `parsed["name"].as_str().unwrap()` after `is_some()` check. Refactored to `if let Some(name)` pattern. | PMAT-164 |
| **No auto-resume prompt** | Spec §6.3 requires interactive [Y/n] when recent session exists. Code had `None => None` — silently started fresh. Added `offer_auto_resume()` with chrono age display. | PMAT-165 |
| **No session age filter** | `find_recent_for_cwd` returned sessions of any age. 3-week-old sessions would be offered. Added 24h filter via `find_recent_for_cwd_within(max_age)`. | PMAT-165 |
| **AprServeDriver SIGKILL-only shutdown** | Drop impl called `kill()` (SIGKILL) immediately with no graceful shutdown window. Fixed: SIGTERM via `kill -TERM` → 2s wait with `try_wait` polling → SIGKILL fallback. | PMAT-166 |
| **trueno-explain blocks apr-cli build** | `trueno-explain 0.2.2` uses `trueno_gpu::ptx` unconditionally but ptx is gated behind `cuda` feature in trueno-gpu. Made trueno-explain optional, gated behind `cuda` feature in apr-cli. PTX commands (`apr ptx`, `apr ptx-map`) now require `--features cuda`. | PMAT-167 |
| **System prompt says "tools listed below" but never lists them** | Small models (1.5B-7B) need explicit tool enumeration. Replaced with 9-tool table with example inputs. Each tool gets name + use case + concrete JSON example. | PMAT-168 |
| **`/cost` shows $0.00 / $inf for local inference** | Misleading dollar amounts when inference is free. Now shows "free (local)" when cost < $0.0001, always shows token counts. | PMAT-169 |
| **max_tokens=512 truncates file edits** | AprServeDriver capped HTTP responses at 512 tokens. Long file edits and multi-tool responses got cut off. Raised to 1024 with comment explaining rationale. | PMAT-170 |
| **apr serve crash shows generic error** | On startup failure (CUDA OOM, model incompatible), user saw "did not become ready within 30s". Now captures subprocess stderr, detects early exit via `try_wait`, shows last 10 lines + debug command. | PMAT-171 |
| **`-p` mode exhausts 50 iterations silently** | `batuta code -p "What files..."` ran 50 agent iterations with no output. Model stuck in tool-call loop (1.5B model repeating same call). Three fixes: (1) `-p` caps iterations at 10, (2) stuck-loop detector breaks on 4+ identical tool calls, (3) `map_error_to_exit_code` for clean error reporting. | PMAT-172 |
| **Tool format mismatch: `<tool_call>` vs raw JSON** | System prompt teaches `<tool_call>` blocks but AprServeDriver appended conflicting "respond with JSON object". Strip logic only matched `"## Available Tools"` but prompt uses `"## Tools"`. Fix: multi-pattern strip + aligned `<tool_call>` format in compact list. | PMAT-173 |
| **Auto-resume consumes piped stdin** | `offer_auto_resume()` calls `stdin().read_line()` which steals piped input intended for the prompt. Fix: `IsTerminal` check — skip resume prompt when stdin is not a TTY. | PMAT-174 |
| **Shell injection filter blocks pipes in wildcard mode** | `has_injection()` blocked `|`, `&&`, backticks even with `allowed_commands: ["*"]`. Coding tasks like `cargo test \| tail` fail. Fix: skip injection filter in wildcard mode (agent has full shell access by design). Restricted allowlists still filter. | PMAT-175 |
| **AprServeDriver strips compact tool table** | `build_openai_body()` stripped `## Tools` (compact table from CODE_SYSTEM_PROMPT) along with `## Available Tools` (verbose enriched schemas). 1.5B model never saw tool descriptions over HTTP — just bare names. Fix: only strip `## Available Tools`. | PMAT-176 |
| **Model ignores tools, outputs "Hello, World!"** | **Misdiagnosed** — the "Hello, World!" was from `src/main.rs` stub, not the model. The real CLI had uncommitted local replacement. Restored from git. PMAT-176/177 fixes are still correct (tool table was being stripped, nudge is useful) but the dogfood evidence was invalid. | PMAT-177, PMAT-178 |
| **main.rs replaced with stub** | Uncommitted local modification replaced the full CLI (`src/main.rs`) with `println!("Hello, World!")`. All `-p` dogfood since PMAT-172 tested the stub. Real dogfood with restored CLI shows: model loads, produces output (hallucinated code, no tool use). 1.5B APR model quality is the bottleneck, not infrastructure. | PMAT-178 |
| **Qwen3 GGUF produces garbage with --gpu** | `Qwen3-1.7B-Q4_K_M.gguf` via `apr serve` with `--gpu` outputs mojibake. Works correctly with `--no-gpu` (CPU) via `apr run`. Fix: removed default `--gpu` from AprServeDriver entirely. CPU inference is correct for all formats. | PMAT-180 |
| **Qwen3 loops on `</think>` tokens** | Without `enable_thinking=false`, Qwen3 enters thinking mode and emits only `</think>` tokens indefinitely — 1024 tokens of closing tags. `apr serve` doesn't support thinking mode control. Added `strip_thinking_blocks()` in AprServeDriver to clean output. **BLOCKER for Qwen3 via apr serve** — needs realizar fix (PMAT-181). `apr run --chat --no-gpu` works because it generates `<think>` block first then answers. | PMAT-180, PMAT-181 |
| **apr-cli had NO Code variant (spec claimed DONE)** | PMAT-162 marked Phase 6 as DONE but apr-cli had zero `Code` wiring — no command enum variant, no dispatch, no batuta dependency. Only the batuta library side was complete. Root cause: Phase 6 only implemented the batuta public API (`cmd_code()`), the apr-cli side was never done. Fix: PMAT-182 adds `Code` variant to `commands_enum.rs`, dispatch to `batuta::agent::code::cmd_code()`, `batuta` dep behind `code` feature (default). `apr code --help` verified end-to-end. | PMAT-182 |
| **realizar `ChatMLTemplate` missing trait methods** | `Qwen3NoThinkTemplate` was added mid-edit (PMAT-181) but left `ChatMLTemplate` impl incomplete — missing `special_tokens()`, `format()`, `supports_system_prompt()`. Also had duplicate `special_tokens()` in `Qwen3NoThinkTemplate`. Fix: restore missing methods, remove duplicate. | PMAT-181, PMAT-182 |
| **apr-cli BrickStats type mismatch** | `trueno::BrickStats` re-export path changed between trueno 0.16 (crates.io) and 0.17 (local). Fix: use `Vec<_>` type inference instead of explicit type annotation. | PMAT-182 |
| **apr-cli trueno SyncMode version conflict** | `set_profiler_sync_mode(trueno::SyncMode::Immediate)` fails because apr-cli's trueno (0.16, crates.io) differs from realizar's trueno (0.17, local path). Temporary fix: commented out. Proper fix: re-export SyncMode from realizar or unify trueno versions. | PMAT-182 |
| **Model discovery: APR preference shadows better GGUF** | `discover_model()` sorted valid > APR > mtime, so Qwen2.5-Coder APR (valid but can't do tool-use) was selected over newer Qwen3 GGUF. Dogfood: `batuta code -p "What is 2+2?"` loaded wrong model, got `obj['whisper.apr']` gibberish. Fix: sort valid > mtime > APR — newest model wins, APR is tiebreaker only. | PMAT-185 |
| **Qwen3 thinking blocks: root cause in AppState** | `with_quantized_model_and_vocab()` set `cached_architecture: None`. CPU GGUF path called `model_architecture()` → returned `None` (or late fallback) → `format_chat_messages` used Raw template → Qwen3 entered thinking mode → `</think>` tokens leaked through. Fix: extract `quantized_model.config.architecture` and cache it. Now `detect_format_from_name("qwen3")` → `Qwen3NoThinkTemplate` with pre-filled empty thinking block. | PMAT-181 |
| **No model name in -p mode output** | `batuta code -p` showed only "Launched apr serve on port..." with no indication of which model was discovered. Added `Model: {name} (auto-discovered)` eprintln in `discover_and_set_model()`. | PMAT-185 |
| **Qwen3 1.7B GGUF tool-use confirmed working** | Direct HTTP test of `apr serve` with Qwen3 1.7B GGUF: model correctly emits `<tool_call>{"name":"shell","input":{"command":"ls"}}</tool_call>` when prompted with tool definitions. Simple questions answered correctly ("4" for "What is 2+2?"). 0.960 tool score validated in practice. | PMAT-185 |
| **`batuta code -p` retry exhaustion on CPU** | (2026-04-05 dogfood) `batuta code -p "What is 2+2?"` with Qwen3 1.7B GGUF on CPU: model discovered, `apr serve` launched and health-checked in 1.5s, but `/v1/chat/completions` failed with 4 retry attempts (exponential backoff 1s→2s→4s→8s). Root cause: CPU inference of 1.7B Q4K too slow — HTTP request times out before inference completes. The health endpoint passes (trivial) but completion endpoint hangs. This confirms PMAT-159: CUDA feature must be enabled for usable `-p` mode. Workaround: `apr run --chat --no-gpu` works (blocking I/O, no timeout). | PMAT-159 |
| **CUDA enabled but model quality blocks enriched prompt** | (2026-04-05 dogfood) Added `realizar/cuda` feature to batuta Cargo.toml (local path dep). Build succeeds, 6250 tests pass. However: (1) Qwen3 GGUF with `--gpu` still produces garbage — CUDA Q4K kernels corrupt output (PMAT-180 confirmed still valid). (2) CPU inference works for simple prompts (`"What is 2+2?"` → `"4"`) but the full CODE_SYSTEM_PROMPT (9 tool definitions, ~500 tokens) causes Qwen3 1.7B to loop on `</think>` tokens (100+ close tags). Root cause: model quality limitation — 1.7B model degrades with large system prompts. (3) Installed local `apr` from source (build `1f0962d5` with `Qwen3NoThinkTemplate`) — template is applied (architecture detected as `qwen3`, `format_chat_messages` uses `Qwen3NoThink`), but doesn't prevent thinking loops with enriched prompts. **Resolved by PMAT-197.** | PMAT-159 |
| **CUDA FIXED: GPU inference working with serial prefill** | (2026-04-05) `BATCHED_PREFILL=0` env var disables FP8 batched prefill, forcing serial Q4K/Q6K GEMV. This produces **correct output** on GPU: `batuta code -p "What is 2+2?"` → `"4"`, `batuta code -p "Write Rust function..."` → complete working code. AprServeDriver now launches `apr serve` with `--gpu` and `BATCHED_PREFILL=0`. Startup 2.5-3.0s (GPU kernel compilation). Root fix for FP8 path is a separate realizar task. | PMAT-181 |
| **CUDA five-whys: DEEP — three code paths, FP8 prefill is the real issue** | (2026-04-05 deep investigation across realizar) **Level 1:** CPU coherent, GPU semi-coherent-wrong from token 1. GQA detected correctly. Q/K/V match at layer 0. FP8 cache NOT the cause. **Level 2:** Qwen3 Q4_K_M GGUF uses MIXED quant types: Q4_K (dtype 12) for Q/K/O_proj/gate, Q6_K (dtype 14) for V/ffn_down/embeddings. `fused_matmul_cuda_with_key` only dispatched Q4_K to GPU, Q6_K fell back to CPU → precision divergence. **Level 3:** Added Q6_K GPU dispatch to `matmul.rs` — same wrong output. The fix was correct but **the chat handler doesn't use this code path**. **Level 4 (root cause):** `apr serve --gpu` routes through `try_cuda_backend()` → `generate_gpu_resident_streaming()` → `run_prefill()` which uses **FP8 matmul** (batched prefill, line 275+ in generate_2.rs), not Q4K/Q6K GEMV. The FP8 weight cache requantizes Q4K→FP8 at warmup (197 matrices, 1640MB). This FP8 requantization path may mishandle Q6_K weights or Qwen3-specific tensor layout. **Level 5:** The `forward_gpu_resident()` in `uses.rs:117` and the batched prefill in `generate_2.rs:275` bypass `fused_matmul_cuda_with_key` entirely — they use FP8 hgemm kernels. The Q6K→FP8 conversion at cache warmup is the likely corruption source. **Fix required in realizar FP8 prefill path for Q6K tensors.** | PMAT-181 |
| **Three root causes for -p failure (RESOLVED)** | (2026-04-05 dogfood, PMAT-197) Deep investigation found three independent bugs: (1) **Context window 4096→32768**: the default 4096 was entirely consumed by 9 tool JSON schemas (~4000 tokens). `truncate_messages` dropped the user query — the model never saw "What is 2+2?". Debug showed body with only system message, no user message. (2) **COMPACT_SYSTEM_PROMPT for -p mode**: full CODE_SYSTEM_PROMPT (tool table + project context + CLAUDE.md ~2600 bytes) overwhelms Qwen3 1.7B causing `</think>` loops. Compact prompt "Answer the question. Be direct." lets model answer. (3) **No-nudge for -p**: `run_agent_loop_with_nudge` retried with "Use a tool!" when model returned text answer — forced stuck tool-call loops on simple questions. Fix: use `run_agent_loop` for -p. After all three fixes: `batuta code -p "What is 2+2?"` → `"2 + 2 = 4."` and `batuta code -p "Write a Rust function..."` → complete working code. | PMAT-197 |
| **No provable-contract for chat templates** | Three separate template bugs shipped (missing trait methods, wrong template for Qwen3, uncached architecture, wrong format() return value) with ZERO contract enforcement. All caught by manual dogfood. Created `chat-template-v1.yaml` with 6 equations and 10 falsification tests. **Contract immediately found a 4th bug:** `Qwen3NoThinkTemplate::format()` returned `ChatML` instead of `Qwen3NoThink`. | PMAT-187 |
| **`-p` mode blank output with thinking models** | `batuta code -p "What is 2+2?"` printed nothing — exit 0 but empty stdout. Root cause: model response is ALL thinking (`</think>\n\n`) → `strip_thinking_blocks()` reduces to empty string → `println!("{}", "")` prints blank line. Confirmed: installed `apr` (0.4.11, crates.io) doesn't have `Qwen3NoThinkTemplate` → Qwen3 enters thinking mode → entire response is thinking tokens. Fix: added diagnostic message when text is empty ("Model may be in thinking mode"). **Root fix:** publish realizar with `Qwen3NoThinkTemplate` + cached architecture (PMAT-181). | PMAT-190 |
| **JSON parse error was test script bug, not realizarr** | Earlier dogfood reported "Invalid control character at column 171" from `apr serve`. Investigation showed: (1) `serde_json::to_string()` correctly escapes `\n` in JSON, (2) bash `echo "$VAR"` was unescaping `\n` to literal newline before piping to python. Proper test with `python3 -c "json.loads(sys.stdin.read())"` parses correctly. **No JSON serialization bug exists.** | PMAT-190 |

### 14.2 What Would Disprove This Specification

1. **~~Qwen2.5-Coder 1.5B fails >60% of coding tasks.~~** **DISPROVED (PMAT-178/179).** Dogfood confirmed: Qwen2.5-Coder 1.5B cannot do tool-use at all — outputs hallucinated code, never emits `<tool_call>` blocks. Default switched to **Qwen3 1.7B** (0.960 tool-calling score). Remaining risk: Qwen3 1.7B may still fail on complex multi-step coding tasks requiring 5+ tool calls in sequence. (Check: benchmark with Qwen3 1.7B on file edit + test + fix workflow)

2. **presentar-terminal TUI adds >50ms input latency.** If the 6-panel TUI slows down interactive typing, users will disable it. The TUI must be zero-cost when no streaming is active. (Check: measure keystroke-to-echo latency with TUI on vs off)

3. **batuta dependency makes apr binary >50MB.** If the `code` feature flag bloats `apr` from ~15MB to >50MB, it should be a separate binary (`apr-code`) instead of a subcommand. (Check: measure binary size with and without `code` feature)

4. **trueno-rag indexing takes >10s for medium projects.** If initial indexing blocks the user for >10s on a 500-file Rust project, indexing must be async/incremental. (Check: benchmark on batuta, realizar, trueno repos)

5. **~~CPU inference too slow for HTTP-backed `-p` mode.~~** **RESOLVED (PMAT-197).** Root cause was NOT speed — it was three bugs: (1) context window 4096 consumed by tool schemas, user query dropped by `truncate_messages`, (2) full CODE_SYSTEM_PROMPT overwhelmed 1.7B model causing `</think>` loops, (3) `run_agent_loop_with_nudge` forced tool calls on simple questions. Fix: compact prompt + 32K window + no-nudge. CPU inference completes in ~2s for simple prompts — fast enough for `-p` mode.

---

## 15. Dogfood Status (2026-04-05, updated after PMAT-197)

### What Works

| Feature | Status | Evidence |
|---------|--------|----------|
| **`batuta code -p "What is 2+2?"`** | **Working** | `→ "2 + 2 = 4."` (Qwen3 1.7B, CPU, PMAT-197) |
| **`batuta code -p "Write Rust function..."`** | **Working** | Returns complete code with explanation (PMAT-197) |
| `batuta code --help` | Working | All flags: --model, --project, --resume, -p |
| Model discovery | Working | Auto-finds `Qwen3-1.7B-Q4_K_M.gguf` via mtime-first sort |
| APR preference (tiebreaker) | Working | FALSIFY-CODE-003, FALSIFY-DISC-001, FALSIFY-DISC-105 |
| Jidoka validation | Working | FALSIFY-DISC-106: invalid APR behind valid GGUF (real files) |
| AprServeDriver launch | Working | `apr serve` starts on random port, health in 1.5s |
| Session directory | Working | `~/.apr/sessions/` (FALSIFY-CODE-006) |
| Sovereignty guarantee | Working | FALSIFY-SPEC-001: hardcoded `PrivacyTier::Sovereign` |
| 9 tools registered | Working | FALSIFY-CODE-002, FALSIFY-SPEC-008 |
| `apr code` in apr-cli | Working | FALSIFY-CLI-006, PMAT-182 verified complete |
| `apr serve loadtest` | Working | Wired into ServeCommands, 8 FALSIFY-LT tests (PMAT-196) |

### What Doesn't Work (Remaining Blockers)

| Feature | Status | Blocker |
|---------|--------|---------|
| CUDA Q4K inference | Broken | PMAT-181: `--gpu` produces garbage for Qwen3 GGUF |
| APR Q4K via crates.io | Blocked | PMAT-157: realizar 0.8.4 not published |
| Interactive REPL with enriched prompt | Untested | Full CODE_SYSTEM_PROMPT may still cause issues for 1.7B |

### What Was Fixed (This Dogfood Cycle)

| Blocker | Root Cause | Fix |
|---------|-----------|-----|
| ~~`-p` mode empty/loop~~ | Context window 4096 consumed by tool schemas; user query dropped | 32K context window (PMAT-197) |
| ~~`-p` mode `</think>` loops~~ | Full system prompt overwhelms 1.7B model | COMPACT_SYSTEM_PROMPT for -p (PMAT-197) |
| ~~`-p` mode stuck tool loops~~ | Nudge forces tool calls on simple questions | No-nudge `run_agent_loop` for -p (PMAT-197) |
| ~~CUDA not enabled~~ | Missing feature flag | `realizar/cuda` in Cargo.toml (PMAT-159) |
| ~~apr-cli Code wiring~~ | Was marked inprogress but actually complete | Verified: dispatch + dep + feature all present (PMAT-182) |

### Contract Enforcement Coverage

| Contract | Equations | Enforcement Tests | Coverage |
|----------|-----------|-------------------|----------|
| apr-code-v1 | 10 | 6 FALSIFY-CODE + 10 FALSIFY-SPEC | Good |
| apr-model-discovery-v1 | 5 | 4 FALSIFY-DISC + 9 FALSIFY-DISC-10x (real files) | Good |
| chat-template-v1 | 5 | 10 FALSIFY-CT-BATUTA | Good |
| http-api-v1 | 5 | 10 FALSIFY-HTTP | Good |
| session-v1 | 4 | 4 FALSIFY-SESSION | Good |
| tokenizer-v1 | 8 | 10 FALSIFY-TOK | Good |
| apr-serve-v1 | 6 | 15 FALSIFY-SRV | Good |
| apr-serve-loadtest-v1 | 5 | 8 FALSIFY-LT | Good |
| apr-chat-session-v1 | 3 | 8 FALSIFY-CHAT | Good |
| apr-finetune-v1 | 5 | 8 FALSIFY-FT | Good |
| model-format-conversion-v1 | 7 | 9 FALSIFY-CONV | Good |
| apr-model-lifecycle-v1 | 5 | 8 FALSIFY-LIFE | Good |
| cli-dispatch-v1 | 4 | 10 FALSIFY-CLI | Good |
| **Total** | **72** | **129** | |

### Next Steps (Priority Order)

**Dependency chain:**
```
1. Fix CUDA Q4K ──┬── 3. Convert Qwen3 to APR
                  ├── 5. cgp roofline profiling
                  └── 6. probar load testing baselines

2. Publish realizar 0.8.4 ── (removes all local path deps)

4. Interactive REPL dogfood ── 7. Prompt scaling by model size

8. OS-native sandboxing (independent)
```

1. **Fix CUDA Q4K dequantization for Qwen3** (PMAT-181) — `--gpu` produces garbage. Root-cause in `realizar/src/gguf/inference/` fused matvec kernel. Likely layout or stride mismatch. Once fixed, inference goes from ~5 tok/s (CPU) to ~50+ tok/s (GPU). **Unblocks steps 3, 5, 6.**

2. **Publish realizar 0.8.4** (PMAT-157) — clean-room build with `Qwen3NoThinkTemplate`, `has_quantized_tensors_apr()`, architecture caching. Removes local path deps. Check `provable-contracts` version pins (`version = "0.2"` alongside path). **Run `make clean-room-p1` in `../infra/machines/clean-room/`.**

3. **Convert Qwen3 1.7B to APR** — `apr convert --to-apr Qwen3-1.7B-Q4_K_M.gguf` with embedded tokenizer. APR is stack-native: row-major, LZ4/ZSTD, faster load. Discovery will prefer it. Blocked on step 1 (CUDA) for validation.

4. **Interactive REPL dogfood** — `batuta code` (no `-p`) with full CODE_SYSTEM_PROMPT. May need prompt scaling for 1.7B. Test: multi-turn file edit + test + fix workflow.

5. **cgp roofline profiling** (PMAT-194) — profile Q4K/Q6K matvec, attention, softmax after CUDA fix. `cgp roofline --kernel fused_q4k_matvec --backend cuda`. Must exceed 50% roofline.

6. **probar LLM load testing** (PMAT-195) — `apr serve loadtest --concurrency 4 --duration 30s`. Establish SLO baselines: TTFT P99 < 1s, TPOT P99 < 50ms (GPU).

7. **Prompt scaling by model size** — detect model parameters at discovery. <2B → COMPACT_SYSTEM_PROMPT. 2-7B → tool table without examples. 7B+ → full prompt. Automatic — user doesn't choose.

8. **OS-native sandboxing** (Phase 5) — Landlock (Linux) / Seatbelt (macOS). Lower priority: 3 layers already active (capability + allowlist + path restriction).
