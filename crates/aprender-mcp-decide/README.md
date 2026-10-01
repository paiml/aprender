# aprender-mcp-decide

A **thin, single-model MCP server** for decision models: one verified
`decide-apr-v1` artifact (a Laya ModernBERT decision model) behind ONE `classify`
tool, built on [pmcp](https://github.com/paiml/rust-mcp-sdk). The tool is bound to
the artifact's own task — its question and its ordered labels come from the
artifact, never from the caller (Phase 8 D-09). It copies the
`aprender-mcp-setfit` template: transport only, in-process, bounds owned by the
transport and named from a contract.

Part of the [aprender](https://github.com/paiml/aprender) monorepo.

## Run locally (stdio)

```bash
cargo run -p aprender-mcp-decide -- --model models/decide/<model>.apr
APRENDER_DECIDE_MODEL=models/decide/<model>.apr cargo run -p aprender-mcp-decide
```

Register it in an MCP client (Claude Desktop, Claude Code, Cursor) as a stdio
server with the same arguments. Everything human-readable goes to stderr; stdout
belongs to the protocol. This runner reads a LOCAL path only — the pmcp.run
deployment is `aprender-mcp-decide-lambda`, which fetches the artifact from S3.

The server advertises exactly one tool:

| Tool | Arguments | Returns |
|------|-----------|---------|
| `classify` | `texts: [string]` — 1..=8 texts, each at most 16384 UTF-8 bytes, at most 800 built model tokens over the request (the 10,240 MB Lambda tier of `contracts/decide-tool-boundary-v1.yaml` 7.0.0; the superseded 3,008 MB tier allowed 2 texts / 120 tokens). Nothing else (`deny_unknown_fields`). | `model` {`artifact_sha256`, `recipe_id`, `method`, `base`}, `labels` (task order), and `results`, one per text in input order: `label`, `probabilities` (calibrated, one per label in `labels` order — an array, never a map), `tokens`, `truncated`. |

The tool description is built from the artifact: its question, its labels in
order, the bounds, and a truncation sentence derived from the artifact's window
and the tier's token budget (plan 08-28, A-derive). Each element of `texts` is ONE
complete document. A truncated text builds a row of the model's full window, so:

- when that window fits `classify_max_total_tokens`, the description says long texts
  are truncated by the model and flagged `truncated: true` (D-12). Laya-en's 512-token
  window fits the contracted 800-token budget, so this is what it serves;
- when it does not — the same window at the superseded 3,008 MB tier's 120-token
  budget — every text long enough to be truncated is refused by the budget first, so
  the description says a text whose built row exceeds the budget is refused and to
  send a shorter excerpt.

## Bounds (`contracts/decide-tool-boundary-v1.yaml`)

- **On the async handler path** (`precheck`, which takes no model and so cannot
  tokenize): the text count, then each text's UTF-8 byte length.
- **Inside one admitted blocking section** (`classify_blocking`): tokenize once,
  check the sum of built-row tokens against the budget, then score.
- **Admission, per process (the library door):** among concurrent callers of
  `ClassifyService::call`, at most `classify_max_in_flight` (1) computation runs and
  at most `classify_max_pending` (4) calls are admitted; the next is refused at once.
  A slot is released only when its blocking work ends, so a disconnected caller
  cannot free CPU that is still being spent.
- **Through the shipped transports, dispatch is serial.** pmcp 2.19.3 runs one tool
  call at a time: on stdio a single worker drains an unbounded queue, so calls a
  client pipelines wait inside pmcp (bounded only by that client); over streamable
  HTTP (the Lambda crate) the server sits behind a mutex held across the tool call.
  So at most one classify runs per process, and the `classify_max_pending` refusal is
  not reachable from a transport. `pipelined_calls_are_serialized_not_refused`
  (`tests/e2e_stdio.rs`) pins this: `classify_max_pending + 1` calls written before
  any reply is read are all classified, in request order.

Every bound refusal (count, argument shape, byte length, token budget, admission)
is `pmcp::Error::tool_rejected`, which pmcp 2.19.3 sends as a **successful**
`tools/call` result with `isError: true` and the refusal message as its one text
content — never a JSON-RPC error — so a client reads it as a tool answer it can act
on (plan 08-28, B-iserror). For example, nine texts at the 10,240 MB tier:

```json
{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"classify: 9 texts exceeds classify_max_texts 8 (contracts/decide-tool-boundary-v1.yaml); split the batch"}],"isError":true}}
```

Model and internal failures (a tokenizer refusal, a forward failure) stay JSON-RPC
-32603. Every refusal names the contract key and the observed value, and never
echoes the caller's text. `ClassifyLimits::CONTRACTED` is asserted equal to the
contract by a unit test.

## Tests

```bash
cargo test -p aprender-mcp-decide --lib               # bounds, admission, shape, contract mirror
cargo test -p aprender-mcp-decide --test e2e_stdio    # live stdio on the packed tiny fixture
APR_MCP_E2E_DECIDE_MODEL=$PWD/models/decide/<model>.apr \
  cargo test -p aprender-mcp-decide --test e2e_stdio  # plus the real-model leg
```
