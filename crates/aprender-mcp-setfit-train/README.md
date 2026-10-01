# aprender-mcp-setfit-train

Part of the [Aprender monorepo](https://github.com/paiml/aprender).

A **thin, single-algorithm MCP training server**: SetFit few-shot training
exposed as an async **MCP Task** (spec 2025-11-25), built on
[pmcp](https://github.com/paiml/rust-mcp-sdk). The training sibling of
`aprender-mcp-setfit` (predict): same thin philosophy, different lifecycle —
training and prediction are separate MCP endpoints because they have different
users at different times.

## Architecture

- **Training goes through the CLI door.** The server supervises a pinned
  `apr setfit train` child process — the CLI adapter is training's one
  validated door (attested ingest, file-first config, atomic artifact write),
  and its ingest sequence is deliberately private to apr-cli so no second
  reader can drift from it. The predict sibling is in-process because
  predict's one door is a public library API; each operation goes through its
  own door. The same child-process job shape runs unchanged in a Lambda
  container or as a SageMaker container entrypoint later.
- **Async as MCP Tasks, on chess-mcp's model.** The handler MINTS the task
  itself (`mint_for_request`) so it holds the canonical id before dispatching
  anything, arms a handoff that pmcp's create gate consults instead of minting
  a second id, stashes the run's inputs in the task's envelope, dispatches, and
  answers `working`. Whoever finishes the work performs a guarded terminal
  write. A dispatch that fails compensates immediately, so a task never wedges
  in `working`. See `src/task_store.rs` for the full rationale.
- **One `TaskStore`, one identity.** The store implements pmcp core's
  `TaskStore` directly rather than depending on `pmcp-tasks` (whose trait is a
  different one of the same name). The task id IS the run id, so `tasks/get`
  and `train_status` cannot describe different things.
- **A `TaskBackend` seam.** `InMemoryTaskBackend` backs stdio and every test;
  a DynamoDB backend drops in behind the same seam for the serverless
  deployment without touching the store, the tools or the server wiring.
- **One job at a time.** A second submit while one trains is refused, naming
  the running task. This is a RESOURCE policy about this process's CPU, so
  per-process state is the correct scope — unlike task state, which must be
  durable and shared. It carries no correctness weight for the pairing.
- **Fail on the request first.** Submit runs the CLI's `--dry-run` pre-flight
  synchronously, so a bad config/selection is a tool error in seconds, not a
  failed job minutes later.

## Deployment shape (not yet built)

pmcp.run fronts Lambda with API Gateway HTTP API, whose 30-second integration
timeout cannot be raised — and Lambda freezes the execution environment when a
handler returns, so a spawned waiter stops mid-training. Both are why the work
must be dispatched out of band rather than run inside the request. The plan
follows chess-mcp: the cargo-pmcp-managed `deploy/` stack keeps the MCP Lambda,
and a sibling CDK app holds the DynamoDB table, the artifact bucket, the Step
Functions state machine and the finalizer Lambda that performs the terminal
write. Steps 1, 2 and 5 of the lifecycle above are already the deployed shape;
only the dispatch and the writer change.

## Tools

| Tool | Arguments | Returns |
|------|-----------|---------|
| `train` | `config: object` — the twelve-knob SetFit training config, passed verbatim to `apr setfit train --config` (bounded at 1 MiB) | A task-shaped `working` value carrying `task_id`. Task-augmented callers poll `tasks/get` and read `tasks/result`; plain callers poll `train_status` with the same id. |
| `train_status` | `task_id: string` | `{schema_version, task_id, phase, artifact_path, report, error}` — `report` is the trainer's own `--json` report (artifact sha256, provenance, resolved config). The same payload the terminal task result carries. |

## Run it

**Streamable HTTP is the default** — this is a remote MCP server. Stdio is the
local-development opt-in, for a client that spawns the binary directly.

```bash
. scripts/apr_bin.sh || exit 1   # never a bare `apr`, never a hardcoded path

# Remote (default): binds 127.0.0.1:8080; the bound address is printed.
cargo run -p aprender-mcp-setfit-train -- \
  --apr-bin "$APR" \
  --data data/tweet-eval-stance \
  --selection data/tweet-eval-stance/selection-manifest.json \
  --model-dir ~/.cache/aprender/minilm-l6-v2-1110a243 \
  --output-dir /tmp/setfit-train-out \
  --addr 0.0.0.0:8080          # accept remote clients; :0 lets the OS pick

# Local: stdio, for Claude Desktop / Claude Code / Cursor.
cargo run -p aprender-mcp-setfit-train -- ... --stdio
```

Every flag also reads `APRENDER_SETFIT_TRAIN_<FLAG>`, including
`APRENDER_SETFIT_TRAIN_ADDR`, so a container configures the server without argv.
The operator provisions the dataset, selection, encoder checkout and output
directory; the client varies only the training config. Measured envelope for the
8-shot reference recipe: **127 s wall, 4.0 GB peak RSS** (M-series CPU).

## Tests

```bash
cargo test -p aprender-mcp-setfit-train -- --nocapture      # unit + E2E; --nocapture is
                                                            # what makes the SKIP line
                                                            # VISIBLE (libtest swallows a
                                                            # passing test's stdout)
# Armed: a REAL training run as an MCP task, once per transport.
APR_MCP_E2E_SETFIT_TRAIN_BIN="$APR" \
  cargo test -p aprender-mcp-setfit-train --test e2e_http   # streamable HTTP (the deployed
                                                            # transport): sessions, Accept
                                                            # negotiation, a real socket
APR_MCP_E2E_SETFIT_TRAIN_BIN="$APR" \
  cargo test -p aprender-mcp-setfit-train --test e2e_stdio  # stdio (what a local client
                                                            # spawns)
```
