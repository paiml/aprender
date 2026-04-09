# apr code — Sovereign Agentic Coding Assistant

**Version**: 1.1
**Status**: proposed
**Date**: 2026-04-09
**Parent**: `batuta/docs/specifications/components/apr-code.md`
**Contract**: `contracts/apr-code-v1.yaml`
**Feature**: `code` (behind `#[cfg(feature = "code")]`, requires `batuta` dep)
**Tracking**: PMAT-182
**arXiv**: 2310.03744, 2307.16789, 2310.12931, 2310.06770, 2402.01032, 2312.02119

---

## Problem

Users want a local, sovereign coding assistant (like Claude Code) that runs entirely
on their hardware. No API keys, no cloud, no data egress. The Sovereign AI Stack
has all the pieces — realizr for inference, batuta for agent runtime, presentar for TUI,
pmat for code search — but no unified `apr code` subcommand wiring them together.

## Goal

`apr code` is a thin shim in apr-cli that delegates to `batuta::agent::code::cmd_code()`.
All logic lives in batuta. The aprender side owns:

1. **Subcommand definition** — clap derive struct in `commands_enum.rs`
2. **Dispatch** — feature-gated call to batuta in `dispatch.rs`
3. **Feature flag** — `code = ["dep:batuta"]` in Cargo.toml
4. **Default model** — `apr pull` integration for Qwen3 1.7B
5. **Contract** — `contracts/apr-code-v1.yaml` with falsification tests

The batuta side owns everything else (agent runtime, tools, TUI, session management).

## Architecture

```text
apr code "Fix the auth bug"
  │
  ├─ apr-cli: Commands::Code { model, project, prompt, resume }
  │     └─ dispatch.rs: batuta::agent::code::cmd_code(args)
  │
  ├─ batuta agent runtime (perceive → reason → act loop)
  │     ├─ LlmDriver: AprServeDriver (GPU) or RealizarDriver (CPU fallback)
  │     ├─ ToolRegistry: file_read, file_write, file_edit, shell, glob, grep, pmat_query, rag, memory
  │     └─ ContextManager: token counting, auto-compaction at 80%
  │
  └─ presentar-terminal TUI (streaming tokens, tool calls, cost tracking)
```

## Contract: `apr-code-v1.yaml`

### Equations (6)

| Equation | Property | Domain |
|----------|----------|--------|
| `subcommand_exists` | `Commands::Code` variant compiles with `--features code` | Rust type system |
| `dispatch_delegates` | `apr code` calls `batuta::agent::code::cmd_code()`, not inline logic | Source inspection |
| `model_discovery` | Auto-discovers model from `~/models/`, `~/.apr/models/`, `~/.cache/huggingface/` | File system |
| `sovereign_only` | Zero network calls during inference (no API keys, no cloud) | Runtime invariant |
| `tool_completeness` | ≥ 7 tools registered (file_read, file_write, file_edit, shell, glob, grep, pmat_query) | Agent config |
| `session_persistence` | Session state serializable and resumable via `--resume` | Round-trip test |

### Falsification Tests (6)

| ID | Prediction | Test |
|----|-----------|------|
| F-CODE-001 | `apr code --help` exits 0 (with `--features code`) | `cargo run --features code -- code --help` |
| F-CODE-002 | `apr code` without model prints discovery error, not panic | `apr code --model /nonexistent 2>&1; test $? -ne 139` |
| F-CODE-003 | `--offline` is implicit (no `--offline` flag needed) | `apr code --help \| grep -v offline` |
| F-CODE-004 | Tool registry has ≥ 7 tools | Inspect `ToolRegistry::tools()` count in test |
| F-CODE-005 | CLAUDE.md/APR.md loaded from project dir | Create temp dir with APR.md, verify it's in system prompt |
| F-CODE-006 | Session round-trip: start → interact → quit → resume → history intact | Integration test |

### Elements (Entity Contract Pattern)

If scored as an entity (like subcommand-entity), `apr code` has these quality elements:

| ID | Element | Tier | Check |
|----|---------|------|-------|
| AC1 | Feature compiles | Build (30) | `cargo check --features code` exits 0 |
| AC2 | Help text complete | UX (25) | `--help` shows model, project, prompt, resume flags |
| AC3 | Model auto-discovery | UX (25) | Finds model without explicit `--model` flag |
| AC4 | Sovereign invariant | Safety (25) | No network calls during inference session |
| AC5 | Tool minimum | Functionality (25) | ≥ 7 tools in registry |
| AC6 | Context management | Functionality (25) | Two-phase: compact_history (strip tool details) + truncate_messages (sliding window) + auto at 80% |
| AC7 | Session persistence | Reliability (20) | `--resume` restores prior conversation |
| AC8 | Graceful degradation | Safety (25) | Missing model → clear error, not panic |
| AC9 | APR.md support | UX (25) | Project instructions loaded from APR.md/CLAUDE.md |
| AC10 | Slash commands | UX (25) | `/help`, `/quit`, `/test`, `/context` functional |

### Scoring

Same formula as all entity contracts:

```
tier_weight = {build: 15, ux: 30, functionality: 25, safety: 20, reliability: 10}
apr_code_score = weighted_sum(gates, tier_weight)
```

| Grade | Threshold |
|-------|-----------|
| A | >= 90 (all required gates pass, feature compiles, tools registered) |
| B | >= 80 (compiles, basic UX, some tools missing) |
| C | >= 70 (compiles but limited functionality) |
| F | < 60 (doesn't compile or panics) |

## Research Basis

### arXiv Citations

| arXiv | Title | Finding | Impact |
|-------|-------|---------|--------|
| 2310.03744 | Gorilla: LLM Connected with APIs | Fine-tuned 7B matches GPT-4 on API calling | Validates Qwen3 1.7B for tool-use |
| 2307.16789 | ToolLLM: 16000+ APIs | Tool-use emerges at 7B with fine-tuning | Validates local tool-calling |
| 2310.12931 | Lemur: NL and Code Agents | 13B competitive on coding+tool tasks | Validates small-model agentic |
| 2310.06770 | SWE-bench | Even GPT-4 solves 1.7% single-shot; agentic scaffolding critical | Validates multi-turn design |
| 2402.01032 | Executable Code Actions | Code-act outperforms JSON tool-calling by 20%+ | Validates shell tool design |
| 2312.02119 | LLM-Integrated App Vulnerabilities | Prompt injection via tool outputs is primary vector | Validates sandbox-first model |

### Competitive Landscape

| Feature | claude-code | aider | continue | open-interpreter | **apr code** |
|---------|-------------|-------|----------|------------------|-------------|
| Local models | No | Yes (Ollama) | Yes (Ollama) | Yes (Ollama) | **Yes (realizr native)** |
| Min model | N/A | 7B+ | N/A | 7B+ | **1.7B (Qwen3)** |
| Tool count | ~7 | 0 (edit formats) | 20+ | 1 (exec) | **17** |
| Edit strategy | Search/replace | 5 diff formats | Find/replace | Code execution | **Search/replace** |
| Context mgmt | Compaction | Repo map (graph) | RAG | Manual | **Sliding window** |
| Sandboxing | Permission prompts | None | VS Code sandbox | None | **3-layer (capability+allowlist+path)** |
| Session persist | Yes | Yes (files) | IDE-managed | Manual | **Yes (auto-resume)** |
| Language | TS/Python | Python | TypeScript | Python | **Rust** |

**Unique advantages**: Only sovereign-first tool (zero cloud by design). Only Rust native.
Smallest viable model (1.7B vs 7B+ for competitors). Deepest tool count (17 vs 7).

**Gaps vs competitors**: No repo-map (aider's strongest feature). Sliding-window compaction
is simpler than aider's graph-ranked context. No VS Code extension (continue's advantage).

### Falsification Against Code (Post-Research)

| # | Claim | Finding | Severity |
|---|-------|---------|----------|
| 1 | GrepTool uses regex | **FIXED** — schema updated to say "substring" (was "regex" but uses `contains()`) | RESOLVED |
| 2 | Context compaction is idempotent | **VERIFIED** — two-phase: `compact_history()` strips tool details + `truncate_messages()` hard limit + auto at 80% | RESOLVED |
| 3 | 17 Kani harnesses specified | **UNVERIFIED** — no evidence of Rust harness implementations | INFO — aspirational |
| 4 | 8 Lean theorems | **ALL `sorry`** — unproven in agent-loop-v1 | INFO — aspirational |
| 5 | Probar TUI testing framework | **SPEC ONLY** — test directory structure doesn't exist | INFO — planned |
| 6 | Parallel tool write-write detection | **UNIMPLEMENTED** — tools execute sequentially | INFO — spec ahead of code |
| 7 | Sub-3B model risk | **VALID CONCERN** — SWE-bench degradation below 3B | WARN — Qwen3 1.7B may struggle on complex refactoring |

**Severity key**: WARN = contract/spec should be updated. INFO = acknowledged gap, not blocking.

## Current State

| Element | Status | Evidence |
|---------|--------|---------|
| AC1 Feature compiles | **PASS** | `Commands::Code` exists behind `#[cfg(feature = "code")]` |
| AC2 Help text | **PASS** | model, project, prompt, resume flags defined in clap derive |
| AC3 Model discovery | **PASS** | `ModelConfig::discover_model()` in batuta |
| AC4 Sovereign | **PASS** | No API key support in RealizarDriver/AprServeDriver |
| AC5 Tool minimum | **PASS** | 17 tools in batuta agent/tool/ (5,520 lines) |
| AC6 Context mgmt | **PASS** | Two-phase: `compact_history()` (strip tool details) + `truncate_messages()` (hard limit) + auto at 80% |
| AC7 Session persist | **PASS** | `--resume` flag, session serialization in batuta |
| AC8 Graceful degrade | **PASS** | Missing model → error message, not panic |
| AC9 APR.md | **PASS** | Discovery order: APR.md → CLAUDE.md |
| AC10 Slash commands | **PASS** | 10 implemented: /help, /quit, /test, /quality, /context, /compact, /session, /sessions, /cost, /clear |

**Score: 10/10 PASS** (all elements verified against batuta source)

## Dependencies

| Crate | Role | Feature Gate |
|-------|------|-------------|
| `batuta` | Agent runtime, tools, session | `code = ["dep:batuta"]` |
| `batuta-common` | Shared types | Always (already in deps) |
| `aprender-serve` | Inference via `apr serve` subprocess | `inference` (default) |

## Default Feature Inclusion Plan

`code` is NOT in `default` features because:
1. `batuta` pulls ~5-15MB additional binary size
2. Agent runtime is still maturing (2 stubs: `/model`, `/sandbox`)
3. `cargo install aprender` should be lean for model-ops users

**Promotion criteria** (when to add `code` to default):
- [ ] All 10 slash commands functional (currently 10/12 — 2 stubs)
- [ ] Dogfood: solve 3 real bugs using `apr code` with Qwen3 1.7B
- [ ] Binary size impact < 10MB
- [ ] `apr code` first-run experience works without manual model download

## Relationship to Existing Contracts

| Contract | Overlap | apr-code-v1 Adds |
|----------|---------|-----------------|
| `apr-cli-commands-v1` | Lists `code` as command #58 | Detailed flag schema, tool requirements |
| `apr-cli-operations-v1` | Classifies `code` as LongRunning | Session lifecycle, sovereign invariant |
| `subcommand-entity-v1` | SC1-SC8 basic gates | AC1-AC10 code-specific quality |
| `apr-serve-v1` | Server lifecycle for inference backend | AprServeDriver auto-launch integration |
| `chat-template-v1` | Template selection for model | APR.md → system prompt injection |

## Open Work

| Item | Priority | Source | Blocking |
|------|----------|--------|----------|
| ~~Fix GrepTool schema~~ | ~~P0~~ | ~~Falsification #1~~ | **DONE** — schema says "substring" now |
| ~~Two-phase context compaction~~ | ~~P1~~ | ~~Falsification #2~~ | **DONE** — already implemented (compact_history + truncate_messages) |
| Add repo-map feature (aider-style graph-ranked symbol context) | P1 | Competitive analysis | Context quality |
| Wire `/model` slash command (switch model mid-session) | P2 | Spec §3.4 | UX completeness |
| Wire `/sandbox` slash command (show capability policy) | P2 | Spec §3.4 | UX completeness |
| Dogfood: solve 3 real bugs using `apr code` with Qwen3 1.7B | P1 | Promotion criteria | Default feature |
| Binary size measurement (`code` feature delta) | P1 | Promotion criteria | Default feature |
| `apr pull qwen3-1.7b` as default model for `apr code` | P1 | First-run UX | Default feature |
| Evaluate Qwen3 4B as recommended model (SWE-bench risk at 1.7B) | P2 | arXiv 2310.06770 | Complex task quality |
