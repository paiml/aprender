<!-- PCU: tools-mcp-server | contract: contracts/apr-page-tools-mcp-server-v1.yaml -->

# aprender-mcp — Model Context Protocol Server

`aprender-mcp` is a Model Context Protocol (MCP) server that exposes the `apr`
CLI as MCP tools over JSON-RPC 2.0 stdio transport. MCP clients — Claude Code,
Cursor, Cline, Aider, Continue — connect to it via `.mcp.json` and invoke
`apr.run`, `apr.qa`, `apr.trace`, etc. on local models. The server speaks MCP
protocol `2024-11-05` and is launched via the `apr mcp` subcommand.

Authoritative spec: [`docs/specifications/apr-mcp-server-spec.md`](https://github.com/paiml/aprender/blob/main/docs/specifications/apr-mcp-server-spec.md).
Crate README: [`crates/aprender-mcp/README.md`](https://github.com/paiml/aprender/blob/main/crates/aprender-mcp/README.md).

## Status

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Skeleton: `initialize` + `tools/list` + `apr.version` | Shipped |
| M2 | 7 Phase-1 subprocess wrappers + dispatcher hardening | Shipped |
| M3 | `apr.finetune` synchronous wrapper, `notifications/cancelled` → SIGTERM→SIGKILL, build.rs schema codegen, opt-in `notifications/progress` for `apr.finetune` | Shipped |
| M4 | Claude Code dogfood session, contract promoted DRAFT → ENFORCED | Pending |
| M5 | Port dispatcher to `pmcp` v2.3; add SSE / WebSocket transports | Planned |

M3 ships `notifications/cancelled` handling (FALSIFY-MCP-006), the 8th
Phase-1 tool `apr.finetune`, full build-time schema code generation
(FALSIFY-MCP-008) for every tool, and opt-in per-line
`notifications/progress` for `apr.finetune` when the client supplies
`params._meta.progressToken` (FALSIFY-MCP-PROGRESS-001). Per-step
structured progress for `apr.finetune` and progress notifications for
`apr.run` remain follow-up slices (the CLI needs an event-channel
prereq and an `apr run --stream` flag).

## Installation

`aprender-mcp` ships as part of the main `aprender` crate; no separate install
step is required:

```bash
cargo install aprender
apr --version
apr mcp --help
```

The server is invoked as the `apr mcp` subcommand. To smoke-test stdio
framing manually (press Ctrl-D to exit):

```bash
apr mcp
```

## Client configuration

The `.mcp.json` file lives at the root of the project directory opened in the
client. Claude Code, Cursor, and Cline all look there; none search parent
directories.

### `apr` resolved from PATH

```json
{
  "mcpServers": {
    "aprender": {
      "command": "apr",
      "args": ["mcp"]
    }
  }
}
```

### Absolute path (GUI-launched clients)

Clients launched from macOS Dock / Windows Start menu do not inherit the
shell `PATH`. Use the absolute-path variant plus any env vars you need:

```json
{
  "mcpServers": {
    "aprender": {
      "command": "/home/you/.cargo/bin/apr",
      "args": ["mcp"],
      "env": {
        "APR_MODEL_DIR": "/home/you/.cache/apr/models"
      }
    }
  }
}
```

Both snippets work as-is for Claude Code, Cursor, and Cline — the
`mcpServers` schema is shared across those clients.

## Tool catalog

Nine tools are registered: `apr.version` (in-process) plus 8 subprocess
wrappers. Each wrapper spawns `apr <subcommand> --json` and returns stdout
verbatim as a single text content block. Non-zero exit is mapped to
`isError: true` with stderr attached.

Every tool's `inputSchema` is generated at build time from
`contracts/apr-mcp-tool-schemas-v1.yaml` — see *Schema codegen* below.

### apr.version

In-process tool. Returns the server version and protocol version. No
arguments.

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"apr.version","arguments":{}}}
```

Response payload:

```json
{"server":"aprender-mcp","version":"0.31.0","protocol_version":"2024-11-05"}
```

### apr.validate

Wraps `apr validate <model_path> --json`. Validates model integrity and
quality gates.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | string | yes | Path to `.apr`, `.gguf`, or `.safetensors` file |

```json
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"apr.validate","arguments":{"model_path":"./qwen2.5-0.5b-instruct-q4km.gguf"}}}
```

Returns `apr validate --json` stdout verbatim.

### apr.tensors

Wraps `apr tensors <model_path> --json [--stats] [--filter <pat>]`. Lists
tensor names, shapes, and (optionally) summary statistics.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | string | yes | Path to the model file |
| `stats` | boolean | no | Include mean/std/min/max per tensor |
| `filter` | string | no | Substring filter on tensor name |

```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"apr.tensors","arguments":{"model_path":"./model.apr","stats":true,"filter":"attn"}}}
```

### apr.bench

Wraps `apr bench <model_path> --json [--iterations N] [--max-tokens N] [--prompt X]`.
Reports throughput and latency percentiles.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | string | yes | Path to the model file |
| `iterations` | integer | no | Measurement iterations (default 5) |
| `max_tokens` | integer | no | Tokens generated per iteration (default 32) |
| `prompt` | string | no | Test prompt (default is model-specific) |

```json
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"apr.bench","arguments":{"model_path":"./model.gguf","iterations":10,"max_tokens":128}}}
```

### apr.qa

Wraps `apr qa <model_path> --json [--assert-tps N] [--max-tokens N] [--iterations N]`.
Runs the 8-gate falsifiable QA checklist.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | string | yes | Path to the model file |
| `assert_tps` | number | no | Minimum throughput gate in tok/s |
| `max_tokens` | integer | no | Tokens per iteration (default 32) |
| `iterations` | integer | no | Benchmark iterations (default 10) |

```json
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"apr.qa","arguments":{"model_path":"./model.gguf","assert_tps":100}}}
```

### apr.trace

Wraps `apr trace <model_path> --json [--layer <pat>] [--reference <path>]`.
Layer-by-layer tensor trace; supports diffing against a reference model.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | string | yes | Path to the model file |
| `layer` | string | no | Substring filter on layer name |
| `reference` | string | no | Reference model to diff against |

```json
{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"apr.trace","arguments":{"model_path":"./model.apr","layer":"layer_0","reference":"./ref.gguf"}}}
```

### apr.run

Wraps `apr run <model_path> --json [--prompt X] [--max-tokens N] [--temperature T] [--top-p P]`.
Synchronous inference; the entire generation completes before the tool
returns. Cancellation via `notifications/cancelled` is wired (see
*Cancellation* below).

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | string | yes | Path to file or `hf://org/repo` |
| `prompt` | string | no | Text prompt to generate from |
| `max_tokens` | integer | no | Maximum tokens (default 32) |
| `temperature` | number | no | Sampling temperature; `0.0` is greedy argmax |
| `top_p` | number | no | Top-p nucleus sampling threshold |

```json
{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"apr.run","arguments":{"model_path":"./qwen2.5-0.5b-instruct-q4km.gguf","prompt":"1+1=","max_tokens":16}}}
```

### apr.serve

Wraps `apr serve <model_path> --port <port>`. Fire-and-forget: the tool
spawns the daemon, captures its pid, and returns `{pid, url, note}`. The
caller is responsible for killing the pid out-of-band.

M3 shipped `notifications/cancelled` for `apr.run` only — `apr.serve` is
still fire-and-forget because it returns `{pid, url}` synchronously and
leaves the daemon detached. A lifecycle-tracked registry (cancel token →
SIGTERM the captured pid with 30s grace → SIGKILL) is a post-M3
follow-up, targeted at M5 alongside the pmcp dispatcher port.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `model_path` | string | yes | Path to file or `hf://org/repo` |
| `port` | integer | no | TCP port (default 8080) |

```json
{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"apr.serve","arguments":{"model_path":"./model.gguf","port":8080}}}
```

Response payload:

```json
{"pid":12345,"url":"http://localhost:8080","note":"fire-and-forget: kill pid via OS to stop"}
```

### apr.finetune

Wraps `apr finetune <base_model> --json [--data <path>] [--rank <N>] [--epochs <N>] [--method <m>] [--output <path>]`.
Synchronous: blocks until training completes, then returns the final JSON
payload from the CLI. Per-step `notifications/progress` streaming is a
follow-up M3 slice.

The MCP argument names (`base_model`, `dataset`, `lora_rank`) differ from
the underlying CLI flags (positional base-model path, `--data`, `--rank`);
the wrapper maps them at dispatch time.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `base_model` | string | yes | Base model path or `hf://org/repo` |
| `dataset` | string | no | JSONL training-data path (→ `--data`) |
| `lora_rank` | integer | no | LoRA rank (→ `--rank`); omit for auto |
| `epochs` | integer | no | Training epochs (default 3) |
| `method` | string | no | `auto`, `full`, `lora`, or `qlora` (default `auto`) |
| `output` | string | no | Output adapter/checkpoint path |

```json
{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"apr.finetune","arguments":{"base_model":"./base.gguf","dataset":"./train.jsonl","lora_rank":8,"epochs":3}}}
```

## Cancellation

M3 wires `notifications/cancelled` end-to-end. Each `tools/call` is handled
on a dedicated worker thread registered in an in-flight table keyed by
JSON-RPC request id. A `notifications/cancelled` whose `params.requestId`
matches a registered id signals that worker's cancel channel.

The worker's subprocess poll loop (see `crates/aprender-mcp/src/tools/subprocess.rs`)
checks the cancel channel between `try_wait` probes. On signal it sends
`SIGTERM` to the spawned `apr` subprocess, waits up to
`CANCEL_GRACE_MS` (30 s, per spec), then escalates to `SIGKILL` if the
child has not exited. Captured partial stdout is returned in the
`ToolCallResult` with `isError: true` and a message prefixed `Cancelled:`.

Example cancel notification (targets the in-flight `apr.run` id):

```json
{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":7,"reason":"user aborted"}}
```

Notes:

- Notifications have no `id` and MUST NOT receive a response.
- Cancelling an id that is not currently in-flight is a silent no-op.
- On non-Unix targets `SIGTERM` is unavailable; the implementation falls
  back to `child.kill()` (equivalent to `SIGKILL`).
- `apr.serve` is fire-and-forget and is not cancellable through this
  path; the caller must kill the returned `pid` directly.

## Schema codegen

Every tool's `inputSchema` is emitted at build time by
`crates/aprender-mcp/build.rs` from
[`contracts/apr-mcp-tool-schemas-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-mcp-tool-schemas-v1.yaml),
the single source of truth for MCP tool argument shape. Each tool's
`*_tool_definition()` parses the generated constant
`crate::schemas::APR_<TOOL>_SCHEMA` into an `InputSchema`. There are no
hand-maintained schemas in the tools source.

FALSIFY-MCP-008 asserts byte-identity (after JSON canonicalization)
between each live `tools/list` schema and the YAML contract entry, so a
schema drift between the two breaks CI.

To change a tool's schema: edit the YAML, rebuild. The Rust source does
not need editing.

## Falsification gates

| Gate | Assertion | Status |
|------|-----------|--------|
| FALSIFY-MCP-001 | `initialize` responds within 500 ms (CI threshold: 50 ms) with `{"protocolVersion":"2024-11-05", ...}` | ACTIVE |
| FALSIFY-MCP-002 | `tools/list` returns every registered tool with a valid object-typed JSON Schema Draft 7 | ACTIVE |
| FALSIFY-MCP-003 | `tools/call apr.run` on `qwen2.5-0.5b-instruct-q4km.gguf` with prompt `"1+1="` decodes `"2"` as first token within 5 s | Deferred to M4 |
| FALSIFY-MCP-004 | `tools/call apr.qa` returns 8 gates byte-identical to `apr qa --json` CLI output | Deferred to M4 |
| FALSIFY-MCP-005 | Malformed request (`"jsonrpc": "1.0"`) returns JSON-RPC error `-32600`, server stays alive | ACTIVE |
| FALSIFY-MCP-006 | `notifications/cancelled` during a long-running tool call stops the subprocess within the grace window and returns a partial result | ACTIVE |
| FALSIFY-MCP-007 | `initialize` with `protocolVersion != "2024-11-05"` returns `-32602`, does not attempt `tools/list` | ACTIVE |
| FALSIFY-MCP-008 | `tools/list` schema is byte-identical (after canonicalization) to `contracts/apr-mcp-tool-schemas-v1.yaml` | ACTIVE |

Additional invariant enforced by the dispatcher:

| Gate | Assertion | Status |
|------|-----------|--------|
| FALSIFY-MCP-VALIDATE-001 | Tool argument validation failure surfaces as `isError: true`, not as a JSON-RPC error | ACTIVE |

The full definitions live in
[`docs/specifications/apr-mcp-server-spec.md#falsification-conditions-for-apr-mcp-server-v1yaml`](https://github.com/paiml/aprender/blob/main/docs/specifications/apr-mcp-server-spec.md#falsification-conditions-for-apr-mcp-server-v1yaml).

## Troubleshooting

**`apr: command not found` from an MCP client.** The client was launched
from a GUI (macOS Dock, Windows Start menu) and did not inherit the shell
`PATH`. Use the absolute-path `.mcp.json` variant above, or symlink
`/usr/local/bin/apr` to `~/.cargo/bin/apr`.

**`.mcp.json` not picked up.** The file must live at the repository root of
the workspace opened in the client. None of the supported clients search
parent directories.

**`protocolVersion mismatch` / `-32602 Invalid Params`.** The client
requested a protocol version other than `2024-11-05`. Upgrade the client
or pin it to a release that speaks `2024-11-05`. FALSIFY-MCP-007 enforces
this — there is no compatibility shim.

**In-flight cancel seems to do nothing.** Check `params.requestId`: it
must match the JSON-RPC `id` of the `tools/call` exactly (string vs
integer matters). Cancelling an unknown id is a silent no-op by design.
