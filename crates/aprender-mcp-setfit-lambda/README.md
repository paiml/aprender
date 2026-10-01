# aprender-mcp-setfit-lambda

The AWS Lambda custom-runtime wrapper around `crates/aprender-mcp-setfit`. It
adds no tool, no bound and no inference path: the server crate owns the
`classify` surface and the model loading, and this crate is a transport shim in
front of it. The Custom Runtime API requires the executable to be named
`bootstrap` exactly, so that is the binary this crate produces. Each invocation
is proxied over loopback to an in-process pmcp streamable-HTTP server configured
`stateless()` — the only mode that survives serverless, because API Gateway
routes successive requests to different containers and sessions/SSE do not
carry across them. The model resolves once per container and stays warm.

## Build

`APRENDER_SETFIT_MODEL` is read twice, at two different times:

- **At build time** — `build.rs` copies the artifact it names into `OUT_DIR` so
  `include_bytes!` compiles it into the binary. This is the self-contained
  deployment build.
- **At run time** — when the variable was unset during the build, an empty
  embed marker is compiled in instead and `resolve_model()` falls back to
  reading the same variable as a path to the artifact.

```bash
# Self-contained deployment binary with the model embedded
APRENDER_SETFIT_MODEL=$PWD/models/<model>.apr \
  cargo build --release -p aprender-mcp-setfit-lambda

# Plain build (no artifact available, e.g. CI) — model supplied at run time
cargo build -p aprender-mcp-setfit-lambda
```

Both doors run the same artifact verification ladder. Neither trusts an
artifact because of where it came from.

Part of the [aprender](https://github.com/paiml/aprender) monorepo.
