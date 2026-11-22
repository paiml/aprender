# crates.io Trusted Publishing (OIDC) Setup for aprender

This document specifies the crates.io Trusted Publishing configuration required so
`.github/workflows/release.yml` can publish all `77` crates without a stored
`CARGO_REGISTRY_TOKEN` secret (when `token_auth=false`, the default).

> **Generated list:** The crate table below is derived directly from `TIERS[]` in
> `scripts/cascade-publish.sh` by `bash scripts/release/oidc_ready.sh --print-doc`.
> Do not edit the crate list by hand; run `bash scripts/release/oidc_ready.sh` to verify
> workflow readiness and doc sync.

## Required GitHub Actions Configuration (`.github/workflows/release.yml`)

- **Workflow permissions:** `id-token: write`
- **Job environment (`jobs.release`):** `infra-read`
- **Auth step:** `rust-lang/crates-io-auth-action` gated by `!inputs.dry_run && !inputs.token_auth`
- **Fallback preserved:** `token_auth=true` continues to use `secrets.CARGO_REGISTRY_TOKEN`

## Exact crates.io Trusted Publishing Settings (Per Crate)

Configure a GitHub Actions Trusted Publisher on `https://crates.io/crates/<crate>/settings`
for **every** crate below with these exact values:

- **Repository owner:** `paiml`
- **Repository name:** `aprender`
- **Workflow filename:** `release.yml`
- **Environment name:** `infra-read` (must match `jobs.release.environment`; leaving it blank causes crates.io OIDC exchange to fail)

## Crate List (77 crates from `TIERS[]`)

| Tier | Crate | Owner | Repository | Workflow | Environment |
| --- | --- | --- | --- | --- | --- |
| 1 | `aprender-legacy-codec` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-update` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `apr-format` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-build-sha` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-contracts-macros` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-quant` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-gemm-codegen` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-sparse` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-solve` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-rand` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-fft` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-image` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-tensor` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 1 | `aprender-cupti` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 2 | `aprender-contracts` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 2 | `aprender-core` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 2 | `aprender-profile-core` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 2 | `aprender-graph` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 2 | `aprender-contrastive-data` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 3 | `aprender-profile` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 4 | `aprender-gpu` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 5 | `aprender-cuda-edge` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 5 | `aprender-cgp` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 6 | `aprender-compute` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 7 | `aprender-cbtop` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 7 | `aprender-ptx-debug` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 7 | `aprender-explain` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-common` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-train-common` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-train` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-train-lora` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-train-distill` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-serve` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-mcp` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-data` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 8 | `aprender-orchestrate` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 9 | `aprender-present-core` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 9 | `aprender-present-layout` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 9 | `aprender-present-yaml` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 9 | `aprender-present-widgets` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 9 | `aprender-present-terminal` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 10 | `apr-cli` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 11 | `aprender` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 12 | `aprender-tsp` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 12 | `aprender-monte-carlo` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 12 | `aprender-shell` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-test-derive` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-test-lib` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-test-js-gen` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-test-cli` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-test-showcase` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-zram-core` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-zram-adaptive` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-zram-generator` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-zram-cli` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-zram` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-present-test-macros` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-present-test` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-present-lib` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-present-cli` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-db` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-rag` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-viz` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-registry` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-distribute` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-simulate` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-verify` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-verify-ml` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-train-shell` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-train-inspect` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-train-bench` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-train-wasm` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-contracts-cli` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 13 | `aprender-rag-cli` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 14 | `provable-contracts` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 14 | `provable-contracts-macros` | `paiml` | `aprender` | `release.yml` | `infra-read` |
| 14 | `provable-contracts-cli` | `paiml` | `aprender` | `release.yml` | `infra-read` |
