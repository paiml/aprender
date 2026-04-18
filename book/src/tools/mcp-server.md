# aprender-mcp — Model Context Protocol Server

`aprender-mcp` is a Model Context Protocol (MCP) server that exposes the `apr`
CLI as MCP tools over JSON-RPC 2.0 stdio transport. It lets MCP clients —
Claude Code, Cursor, Cline, Aider, Continue — invoke `apr.validate`,
`apr.tensors`, `apr.bench`, `apr.qa`, `apr.trace`, and `apr.version` on local
models. The server speaks MCP protocol `2024-11-05` and is launched via the
`apr mcp` subcommand.

Spec: [`docs/specifications/apr-mcp-server-spec.md`](https://github.com/paiml/aprender/blob/main/docs/specifications/apr-mcp-server-spec.md).
Crate README: [`crates/aprender-mcp/README.md`](https://github.com/paiml/aprender/blob/main/crates/aprender-mcp/README.md).

## Quick start

Install aprender and confirm the `apr mcp` subcommand is present.

```bash
cargo install aprender
apr --version
apr mcp --help
```

Run the server directly to smoke-test stdio framing (press Ctrl-D to exit):

```bash
apr mcp
```

Wire it into an MCP client with an `.mcp.json` file (see next section), then
ask the client to list tools — you should see `apr.version`, `apr.validate`,
`apr.tensors`, `apr.bench`, `apr.qa`, and `apr.trace`.

## Client configuration

The `.mcp.json` file lives at the root of your project (Claude Code, Cursor,
Cline all look there). Two variants are supported: `apr` resolved from
`PATH`, or an absolute binary path.

### `apr` on PATH

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

### Absolute path

Useful when the client doesn't inherit your shell `PATH` (common on macOS
GUI launches):

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
`mcpServers` schema is shared across those three clients.

## Tool catalog

Each tool is a thin subprocess wrapper around `apr <subcommand> --json`. The
descriptions below match what MCP clients see in `tools/list`.

### apr.version

*Return the aprender-mcp server version. Takes no arguments.*

- Wraps: none (in-process)
- Arguments: (none)
- Returns: `{server, version, protocol_version}`

```text
apr.version()
```

### apr.validate

*Validate a model file's integrity and quality. Wraps `apr validate <model> --json`.*

- Wraps: `apr validate <model_path> --json`
- Arguments: `model_path` (string, required)

```text
apr.validate(model_path="./qwen2.5-0.5b-instruct-q4km.gguf")
```

### apr.tensors

*List tensors in a model with shapes and dtypes. Wraps `apr tensors <model> --json`.*

- Wraps: `apr tensors <model_path> --json [--stats] [--filter <pat>]`
- Arguments:
  - `model_path` (string, required)
  - `stats` (boolean) — include mean/std/min/max
  - `filter` (string) — substring match on tensor name

```text
apr.tensors(model_path="./model.apr", stats=true, filter="attn")
```

### apr.bench

*Benchmark model throughput and latency. Wraps `apr bench <model> --json`.*

- Wraps: `apr bench <model_path> --json [--iterations N] [--max-tokens N] [--prompt X]`
- Arguments:
  - `model_path` (string, required)
  - `iterations` (integer, default 5)
  - `max_tokens` (integer, default 32)
  - `prompt` (string, default model-specific)

```text
apr.bench(model_path="./model.gguf", iterations=10, max_tokens=128)
```

### apr.qa

*Run the 8-gate falsifiable QA checklist on a model. Wraps `apr qa <model> --json`.*

- Wraps: `apr qa <model_path> --json [--assert-tps N] [--max-tokens N] [--iterations N]`
- Arguments:
  - `model_path` (string, required)
  - `assert_tps` (number) — minimum throughput gate in tok/s
  - `max_tokens` (integer, default 32)
  - `iterations` (integer, default 10)

```text
apr.qa(model_path="./model.gguf", assert_tps=100)
```

### apr.trace

*Layer-by-layer tensor trace with per-layer stats. Wraps `apr trace <model> --json`.*

- Wraps: `apr trace <model_path> --json [--layer <pat>] [--reference <path>]`
- Arguments:
  - `model_path` (string, required)
  - `layer` (string) — substring filter on layer name
  - `reference` (string) — reference model to diff against

```text
apr.trace(model_path="./model.apr", layer="layer_0", reference="./ref.gguf")
```

## Falsifiers

The currently enforced gates (subset of the full 1..8 in the spec):

| Gate | What it asserts |
|------|-----------------|
| FALSIFY-MCP-001 | `initialize` round-trip under 50ms (spec budget 500ms) |
| FALSIFY-MCP-002 | every registered tool exposes a valid object-typed schema |
| FALSIFY-MCP-005 | `jsonrpc != "2.0"` is rejected with `-32600 Invalid Request` |
| FALSIFY-MCP-007 | `initialize.params.protocolVersion` mismatch returns `-32602 Invalid Params` |
| FALSIFY-MCP-VALIDATE-001 | tool argument validation surfaces as `isError:true`, not as a JSON-RPC error |

The full 1..8 list lives in [`docs/specifications/apr-mcp-server-spec.md#falsification-conditions-for-apr-mcp-server-v1yaml`](https://github.com/paiml/aprender/blob/main/docs/specifications/apr-mcp-server-spec.md#falsification-conditions-for-apr-mcp-server-v1yaml).
FALSIFY-MCP-003/-004/-006/-008 land with M3 (streaming) and M4 (end-to-end).

## Roadmap

Per the spec milestones: **M3** adds `notifications/progress` streaming for
long-running tools (`apr.run`, `apr.finetune`) plus cancellation wired
through `notifications/cancelled` → SIGTERM/SIGKILL of the spawned `apr`
subprocess. **M4** adds a Claude Code dogfood session using only `apr.*`
tools and promotes `contracts/apr-mcp-server-v1.yaml` from DRAFT to
ENFORCED. See `docs/specifications/apr-mcp-server-spec.md` sections *M3*
and *M4* for the full acceptance criteria.

## Troubleshooting

**`apr: command not found` from an MCP client.** The client launched from a
GUI (macOS Dock, Windows Start menu) does not inherit your shell `PATH`.
Use the absolute-path variant of `.mcp.json` above, or symlink
`/usr/local/bin/apr` to `~/.cargo/bin/apr`.

**`.mcp.json` not picked up.** The file must live at the repository root of
the workspace you opened in the client. Claude Code, Cursor, and Cline do
not search parent directories. `ls -l .mcp.json` at the project root; if it
is absent or in a subdirectory the server will not start.

**`protocolVersion mismatch` or `-32602 Invalid Params`.** The client
requested a protocol version other than `2024-11-05`. Upgrade the client,
or pin it to a release that speaks `2024-11-05`. FALSIFY-MCP-007 enforces
this — no partial compatibility shim is offered.
