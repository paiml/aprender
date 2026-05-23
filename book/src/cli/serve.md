<!-- PCU: cli-serve | contract: contracts/apr-page-cli-serve-v1.yaml -->

# apr serve

Inference server (plan/run)

**Category**: Inference

## Synopsis

```text
apr serve [OPTIONS]
```

## Example

```bash
apr serve run qwen2.5-coder-1.5b --port 8080
```

## What this does

`apr serve` exposes a model as an OpenAI-compatible HTTP API with `/v1/chat/completions`,
`/v1/completions`, Prometheus metrics, and Server-Sent Events streaming. Use it
when an IDE or agent client expects an Ollama / OpenAI endpoint. The `plan`
subcommand answers "does this model fit in my VRAM, and what's the roofline
ceiling?" without actually loading weights.

## Key flags

| Flag | What it does | Example |
|------|-------------|---------|
| `--port P` | Listen port (default 8080) | `--port 11434` |
| `--host H` | Bind host (default 127.0.0.1) | `--host 0.0.0.0` |
| `--batch` | Enable batched GPU prefill (2x+ throughput) | `--batch` |
| `--gpu` / `--no-gpu` | Force backend selection | `--gpu` |
| `--no-metrics` | Disable `/metrics` endpoint | `--no-metrics` |
| `--otlp-endpoint URL` | Export distributed traces | `--otlp-endpoint http://jaeger:4317` |

## Common workflows

**Drop-in Ollama replacement on the standard port.**

```bash
apr serve run qwen2.5-coder-7b.apr --port 11434 --host 0.0.0.0 --batch
curl http://localhost:11434/v1/chat/completions \
    -d '{"model":"qwen2.5-coder-7b","messages":[{"role":"user","content":"hi"}]}'
```

**Capacity-check before deploying a larger model.**

```bash
apr serve plan qwen3-coder-30b-a3b.apr --batch-size 4
# Reports: VRAM budget, KV cache size, roofline tok/s ceiling, contract status
```

## Troubleshooting

- **`address already in use` on 8080** — another process owns the port. List with
  `lsof -i :8080` or pick a new one via `--port 8081`.
- **Throughput stuck ~20 tok/s on a CUDA host** — `--batch` is off by default for
  determinism. Add it for production; expect ~8x prefill speedup on Qwen2.5.
- **Prometheus scrape returns 404** — `/metrics` is gated behind `--no-metrics`
  being absent; double-check the flag didn't sneak in via your systemd unit.

## See also

- Source: [`crates/apr-cli/src/commands/serve.rs`](https://github.com/paiml/aprender/blob/main/crates/apr-cli/src/commands/serve.rs)
- Contract: [`contracts/apr-page-cli-serve-v1.yaml`](https://github.com/paiml/aprender/blob/main/contracts/apr-page-cli-serve-v1.yaml)

