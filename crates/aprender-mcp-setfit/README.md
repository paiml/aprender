# aprender-mcp-setfit

A **thin, single-model MCP server**: one SetFit classifier behind one `classify`
tool, built on [pmcp](https://github.com/paiml/rust-mcp-sdk) and deployable to
pmcp.run. This crate is the template for wrapping an aprender model as a
business-process-curated MCP connector — deliberately NOT a general-purpose ML
server (that's `crates/aprender-mcp`, the developer toolchain surface).

## Run locally (stdio)

```bash
# Train an artifact first — recipe in
# crates/aprender-mcp/tests/e2e_setfit_predict.rs
cargo run -p aprender-mcp-setfit -- --model models/setfit-abortion-s17x8.apr
```

Register in an MCP client (Claude Desktop / Claude Code / Cursor) as a stdio
server with those same arguments. The server advertises exactly one tool:

| Tool | Arguments | Returns |
|------|-----------|---------|
| `classify` | `texts: [string]` (≤256), `include_logits?: bool` | One result per text, in order: `label`, `probabilities`, `margin`, `token_count`, `truncated` — the same `ClassifyResponse` envelope as `apr predict --json` and `POST /v1/classify`. |

## Architecture

- Inference is **in-process**: `aprender-core`'s `VerifiedSetFitModel::classify`,
  the one implementation every surface routes to (Phase 4 D-09, OPS-03). The
  model loads once at startup through the artifact verification ladder and
  stays warm. No subprocess, no second inference path.
- Bounds are the contract's (`contracts/setfit-apr-v1.yaml` item 11):
  `max_batch_texts = 256` (enforced by core), `max_request_body_bytes = 1 MiB`
  (enforced here — this server is a reading surface).
- Unknown argument keys are refused (`deny_unknown_fields`), end to end.

## Tests

```bash
cargo test -p aprender-mcp-setfit                  # unit + visible-SKIP E2E
APR_MCP_E2E_SETFIT_MODEL=$PWD/models/<model>.apr \
  cargo test -p aprender-mcp-setfit                # armed E2E over live stdio
```

## Deployment (pmcp.run)

The pmcp.run deployment runs a Lambda loopback wrapper (streamable-http,
stateless) that embeds the model bytes in the binary via `include_bytes!` and
loads them with `load_model_from_bytes`. See the deploy crate/config added in
the deployment phase.
