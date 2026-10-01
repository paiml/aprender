# API Coverage — Phase 6

No external API integration: Phase 6 ports Prophet, NeuralProphet and Chronos-Bolt natively into pure Rust and serves them through this project's own thin pmcp servers.

The ports live in `crates/aprender-forecast` (Prophet 1.4.0, NeuralProphet 0.9.0, Chronos-Bolt). There is no third-party API, SDK or hosted service whose surface could be partially covered.

The `api_coverage_gate` detector fired on a single signal: the noun `api` inside a source comment in an existing Phase 6 plan (`<!-- Core API the np.rs port calls ... -->`). That comment names an INTERNAL Rust API — the `aprender` autograd and tensor surface the `np.rs` port calls — not an external one, so the signal was a false positive rather than an uncovered integration.

The only network fetch anywhere in the phase is `just fetch-chronos-tiny`, which downloads the pinned `amazon/chronos-bolt-tiny` weights once and verifies them by sha256. That is a build-time artifact download with a pinned digest, not a runtime API integration: nothing in `crates/aprender-forecast`, `crates/aprender-mcp-forecast` or `crates/aprender-mcp-chronos` opens a socket to a third party while serving a request, and every `forecast` call is stateless and local (D-01).

This file therefore carries no coverage table. The seal gate accepts a reasoned declaration with no rows, and fabricating rows for integrations that do not exist would assert coverage of capabilities the phase never built.
