# aprender-mcp-chronos-lambda

The AWS Lambda custom-runtime wrapper around `crates/aprender-mcp-chronos`. It
adds no tool, no bound and no inference path: the server crate owns the
`forecast` surface and the model loading, and this crate is a transport shim in
front of it. The Custom Runtime API requires the executable to be named
`bootstrap` exactly, so that is the binary this crate produces. Each invocation
is proxied over loopback to an in-process pmcp streamable-HTTP server configured
`stateless()` — the only mode that survives serverless, because API Gateway
routes successive requests to different containers and sessions/SSE do not
carry across them. The model resolves once per container and stays warm.

## Build

Chronos-Bolt weights (`models/chronos-bolt-tiny/f16`):

- **At build time** — `aprender-mcp-chronos` stages `model.safetensors` + `config.json` into
  `OUT_DIR` so `include_bytes!` compiles them into the binary (~24 MB total). This is the
  self-contained deployment build with zero runtime downloads and ~52ms cold starts.
- **At run time** — when weights were not present during build, empty markers are compiled in
  and `resolve_model()` falls back to reading `CHRONOS_MODEL_DIR` as a path.

```bash
# Build release binary (automatically embeds models/chronos-bolt-tiny/f16 if present)
cargo build --release -p aprender-mcp-chronos-lambda

# Deploy via cargo-pmcp to pmcp.run
cargo pmcp deploy --manifest-path crates/aprender-mcp-chronos-lambda
```

Part of the [aprender](https://github.com/paiml/aprender) monorepo.
