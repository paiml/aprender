# aprender-mcp-decide-lambda

The AWS Lambda custom-runtime wrapper around `crates/aprender-mcp-decide` (Phase 8 D-15).
It adds no tool, no bound and no inference path: the server crate owns the one `classify`
tool, its bounds (`contracts/decide-tool-boundary-v1.yaml`) and the decide-apr-v1 load
doors, and this crate is a transport shim in front of them.

The Custom Runtime API requires the executable to be named `bootstrap`, so that is the
binary this crate produces. Each invocation is proxied over loopback to an in-process
pmcp streamable-HTTP server configured `stateless()`. That is the only mode that survives
serverless: API Gateway routes successive requests to different containers, so a cold
container can receive a `tools/call` whose `initialize` landed on another one. A loopback
test proves the stateless server answers exactly that request.

## Where the weights come from (D-18)

| Variable | Meaning |
|---|---|
| `APRENDER_DECIDE_S3_URI` | `s3://bucket/key` of the served `.apr` (the Lambda source) |
| `APRENDER_DECIDE_SHA256` | the artifact's sha256, 64 lowercase hex characters. Required with the S3 URI |
| `APRENDER_DECIDE_MODEL` | a local `.apr` path (local runs), optionally pinned by `APRENDER_DECIDE_SHA256` |
| `PMCP_SERVER_ID` | server name (default `aprender-decide`) |

Setting both sources is refused, as is setting neither. The S3 layout is content-addressed:
`s3://<bucket>/decide/<server>/<sha256>.apr`, so the key and the pin name the same bytes.

**The weights are never baked into the package.** Spikes 021 and 026 measured baked weights
7-18x worse on cold start than an S3 fetch at 10,240 MB. At cold start the object is
fetched by 16-way 64 MiB ranged GETs, with 5 attempts per part (8 s each) and a 10.739 s
overall deadline, **straight into one pre-sized in-memory buffer**. That deadline is the 30 s
gateway cap minus the post-download work measured on the 10,240 MB tier (sha256 + load ladder
+ the largest classify, 15.261 s) and the contract's 4 s margin. The download is also capped at
the invocation's remaining time minus that 19.261 s reserve. Lambda's scratch disk
is 512 MB and cargo-pmcp cannot raise it, so a 0.85 GB artifact cannot be staged there.
A length over the decide-apr-v1 cap is refused before the buffer is allocated.

The buffer's sha256 must equal the pin **before the load ladder parses a byte**. The pin is
both the tamper guard and the identity every `classify` response reports as
`model.artifact_sha256`.

## Lazy load and cold-start evidence

The model loads on the first MCP request (a `POST`) through a tokio `OnceCell`, so
concurrent first calls share one load. No other method loads: `GET` answers the health
body, `OPTIONS` the CORS preflight, and anything else is a 405. The health body says
`ok: true` only when the model-source config parses; a missing or malformed
`APRENDER_DECIDE_*` config answers 503 with `ok: false` and `config_error`. It does not
load in the init phase, because default Lambda caps init near 10 s. A failed load returns
HTTP 503 with only the failure kind (the detail goes to the log), and the next request
retries. If the in-process MCP server task ever ends, the process exits so Lambda
replaces the environment.

The request that performs the load logs
`decide.load performed_load=true probe_id=<id> load_ms=<n> download_ms=<n> build_ms=<n> graviton=<gen>`.
Every other request logs `performed_load=false`. Every proxied response carries
`x-decide-load: cold;load_ms=<n>` or `warm`. The probe id comes from the optional
`x-decide-probe-id` request header, accepted only as at most 64 `[A-Za-z0-9-]` characters.
Request text is never logged.

## Probe

`examples/probe.rs` checks a live (or local) endpoint against a local copy of the
artifact. It checks for exactly one `classify` tool, the labels in task order and the
served sha256. With `--maximal concentrated|distributed --cold-first` it sends the maximal
legal request as the first and only POST, for the accepted-region test.

```bash
cargo run -p aprender-mcp-decide-lambda --example probe -- \
    --url https://<endpoint>/ --apr model.apr [--expect-sha256 <hex>] \
    [--maximal concentrated|distributed] [--cold-first] [--probe-id <id>]
```

A bearer token is read from `--bearer` or `APRENDER_DECIDE_PROBE_TOKEN`, and it is never
printed.

## Deployed

Live on pmcp.run as **`aprender-mcp-decide`** since 2026-09-27, on the 10,240 MB tier since
2026-09-28 (plan 08-30). The outcome is `deployed-passed`. The record is
`.planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/08-LIVE-REDEPLOY-EVIDENCE.json`.

| | |
|---|---|
| Endpoint | `https://aprender-mcp-decide.us-east.true-mcp.com/mcp` (POST only) |
| Auth | Off, by the owner's decision (the cost risk was accepted). No token or header is needed |
| Tier | 10,240 MB, arm64 |
| Task | Abortion stance (TweetEval abortion target, "legalization of abortion"), labels `none`, `against`, `favor` |
| Model | `laya-stance-64.apr`, sha256 `24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a` |

### Connecting an MCP client

POST JSON-RPC to the endpoint. The server is stateless:
- a bare `tools/call` works with no `initialize`;
- a full client (`initialize` → `notifications/initialized` → `tools/list` → `tools/call`) works too.

`GET /mcp` is answered by the pmcp.run edge with 405, not by this server. `GET /health` is the
platform's answer, and it does not reflect whether MCP calls succeed.

```
POST https://aprender-mcp-decide.us-east.true-mcp.com/mcp
content-type: application/json
accept: application/json, text/event-stream

{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"classify","arguments":{"texts":["Every life deserves protection from conception. #prolife"]}}}
```

The tool result text is JSON:
- `model`: `artifact_sha256`, `recipe_id`, `method`, `base`;
- `labels`;
- one entry in `results` per text: `label`, `probabilities` in label order, `tokens`, `truncated`.

That tweet returned `against` (none 0.072, against 0.835, favor 0.094). An off-topic text reads `none`.

### Identity check

Every response carries `model.artifact_sha256`. It must equal
`24a44d7e050166c9b64e2716f2bcb3ce91747f7a3b927d03d6eeae5f89b6275a`, the pin the function
verifies before it parses a byte. `examples/probe.rs --expect-sha256 <that>` checks it, together
with the single `classify` tool and the label order.

### Limits on the 10,240 MB tier

The door refuses anything over these limits (`contracts/decide-tool-boundary-v1.yaml`):
- **8 texts** or fewer per call.
- **800 built tokens** or fewer per call.
- **Cold call: 22.6-27.0 s** at the client, over 8 proven-cold samples of the two maximal
  requests (2 texts or 8 texts, 800 built tokens each). The first call after idle loads 846 MB
  from S3 (8.9-9.1 s of it). The worst sample left 2.99 s under the 30 s gateway cap.
- **Warm call: 0.87-0.98 s** (10 calls).

### Claim scope

The served model passed laya-finetune-gate-v1. Its claim (`eval_set.claim`) is:

> The gate now certifies margin and calibration on held-out data drawn like the tenant's shots.
> It does NOT certify robustness to a shifted input population.

It is in-distribution calibration, not shift robustness. On the SemEval-2016 test split, the reported
shift probe measured ece_post 0.1896. That would fail the gate's 0.10 clause if it were one.

## Build

```bash
cargo test -p aprender-mcp-decide-lambda --lib
ulimit -n 65536
cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.34 \
    -p aprender-mcp-decide-lambda --bin bootstrap
```

The deploy config template is `.pmcp/deploy.toml.template`. The generated `deploy.toml`
is gitignored, because the bucket name carries the AWS account id.

Part of the [aprender](https://github.com/paiml/aprender) monorepo.
