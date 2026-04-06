# ADR-002: Ollama-style CLI Location - batuta as Orchestration Layer

## Status
Accepted

## Context

The Sovereign AI Stack needs an ollama-style CLI for interactive model serving with commands like `run`, `pull`, `list`, `serve`, and `chat`. Three options were evaluated:

| Option | Location | Pros | Cons |
|--------|----------|------|------|
| A | `realizar` CLI | Direct inference access, simple deps | No pacha registry, limited orchestration |
| B | `batuta` CLI | Full stack access, pacha integration, existing infra | Heavier deps |
| C | New `pacha-cli` | Clean separation | Yet another crate to maintain |

## Decision

**Option B: `batuta` CLI** is the canonical location for ollama-style commands.

### Rationale

1. **batuta already integrates the full stack**: It has pacha (registry), realizar (inference), and aprender (ML) as dependencies. Adding model serving commands is a natural extension of its orchestration role.

2. **Pacha URI scheme is critical**: The `pacha://models/llama3.2:3b` URI scheme for model resolution requires pacha integration, which batuta already has but realizar does not.

3. **Existing infrastructure**: batuta already has `batuta serve` module infrastructure (circuit breakers, failover, privacy tiers), CLI argument parsing (clap), and TUI support.

4. **Consistent user experience**: Users already use `batuta stack`, `batuta oracle`, `batuta falsify`. Adding `batuta run` and `batuta serve` keeps one CLI entry point.

5. **realizar stays focused**: realizar is a library crate for inference primitives. Adding CLI and registry concerns would violate separation of concerns.

## Command Structure

```
batuta run <model>              # Interactive inference (like `ollama run`)
batuta serve <model>            # Start HTTP API server
batuta pull <model>             # Download from registry (delegates to pacha)
batuta list                     # List local models (delegates to pacha)
batuta chat <model>             # Interactive chat session
```

### Delegation Pattern

```
User → batuta CLI → pacha (model resolution) → realizar (inference)
```

- `batuta run/chat`: Resolves model via pacha, loads via realizar, runs interactive loop
- `batuta serve`: Resolves model, starts HTTP server with realizars inference engine
- `batuta pull/list`: Pure pacha delegation for registry operations

## Consequences

### Positive
- Single CLI entry point for all stack operations
- Full access to pacha registry and model resolution
- Leverage existing serve module (circuit breakers, failover)
- Feature-gated via `inference` feature flag to keep default binary lean

### Negative
- batuta binary is heavier when `inference` feature is enabled
- Users who only want inference must install batuta (not just realizar)

### Mitigations
- `inference` feature is opt-in, not default
- `batuta` can be installed without inference: `cargo install batuta --no-default-features --features native`
- Future: slim `batuta-run` binary that only includes inference path

## References
- GH-14: RFC discussion
- GH-12: Serve/deploy commands specification
- Ollama CLI: https://ollama.ai
