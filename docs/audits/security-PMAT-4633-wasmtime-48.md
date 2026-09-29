# PMAT-4633: wasmtime 47.0.4 → 48.0.3 (RUSTSEC-2026-0315, RUSTSEC-2026-0316)

Review input for #4632. Reviewers get the `Cargo.toml` diff, the generated lock-delta
table below, the advisory text, and the before/after checks. They do not get the raw
`Cargo.lock`. The lockfile is checked by machine: `cargo audit`, `cargo deny check advisories`
and `cargo metadata --locked`, each run on the base and on this head.

- base: `2817c6d97b` (origin/main)
- head: `1a68b112ba` (the only commit that touches `Cargo.toml` / `Cargo.lock`)
- No advisory is ignored. The diff touches neither `deny.toml` nor `.cargo/audit.toml`.

## Machine checks

| check | base 2817c6d97b | head |
|---|---|---|
| `cargo audit` | rc=1: RUSTSEC-2026-0315 and RUSTSEC-2026-0316 on wasmtime 47.0.4 | rc=0, 0 vulnerabilities |
| `cargo deny check advisories` | rc=0 | rc=0 |
| `cargo deny --all-features check advisories` | rc=0 | rc=0 |
| `cargo metadata --locked` | n/a | rc=0 (the lock resolves unchanged) |
| `cargo +1.98.0 test --locked -p aprender-test-lib --features runtime` (gx10, aarch64) | n/a | rc=0: compiles wasmtime v48.0.3; lib 6744 passed, 0 failed, of which 45 are `runtime::` tests; no source change needed |

`cargo audit` read advisory-db commit `f23b7682` (2026-09-29T09:32:15+02:00, 1277 advisories).
`cargo deny` passes on both sides, so it does not separate them: wasmtime is reached only
through the optional `runtime` feature of `aprender-test-lib`. Only `cargo audit`
tells before from after, and it goes from rc=1 to rc=0.

wasmtime 48 is a semver-major bump, and CI never enables the `runtime` feature, so CI does not
compile it. The runtime-feature build and test above is the evidence that the new API
still fits `aprender-test-lib`. It needs Rust 1.95 or newer, so it ran on 1.98.0.

## Advisories

### RUSTSEC-2026-0316 — Dynamic record lifting can allocate beyond the hostcall fuel limit

- package: wasmtime 47.0.4; date 2026-09-24; url https://github.com/bytecodealliance/wasmtime/pull/14415
- patched: >=36.0.16, <37.0.0, >=48.0.3, <49.0.0, >=49.0.1; aliases: GHSA-jqpg-j7w6-42pr

This is an entry in the RustSec database for the Wasmtime security advisory
located at
https://github.com/bytecodealliance/wasmtime/security/advisories/GHSA-jqpg-j7w6-42pr
For more information see the GitHub-hosted security advisory.

### RUSTSEC-2026-0315 — `call_ref` and exception `catch` can drop some fuel accounting, leading to exponential fuel amplification

- package: wasmtime 47.0.4; date 2026-09-24; url https://github.com/bytecodealliance/wasmtime/pull/14407
- patched: >=48.0.3, <49.0.0, >=49.0.1; aliases: GHSA-m63x-6p34-q65x

This is an entry in the RustSec database for the Wasmtime security advisory
located at
https://github.com/bytecodealliance/wasmtime/security/advisories/GHSA-m63x-6p34-q65x
For more information see the GitHub-hosted security advisory.

## Lock delta (generated from `Cargo.lock`, base → head)

The table is produced by parsing both lockfiles with `tomllib`. It lists every package whose set
of versions or whose checksum changed. The checksum column is the first 16 hex characters of the
registry sha256 for the new version.

| crate | old | new | checksum (new, sha256 prefix) |
|---|---|---|---|
| cranelift-assembler-x64 | 0.134.4 | 0.135.3 | 9d620e7c8e86c48f |
| cranelift-assembler-x64-meta | 0.134.4 | 0.135.3 | 75015effc364f66d |
| cranelift-bforest | 0.134.4 | 0.135.3 | e18dd511b085013a |
| cranelift-bitset | 0.134.4 | 0.135.3 | a92f504aa9105998 |
| cranelift-codegen | 0.134.4 | 0.135.3 | f26f0f8551262c68 |
| cranelift-codegen-meta | 0.134.4 | 0.135.3 | 6672eac6d00c944a |
| cranelift-codegen-shared | 0.134.4 | 0.135.3 | 2f2bd3116e2cdef4 |
| cranelift-control | 0.134.4 | 0.135.3 | 9706989f98e3f7f1 |
| cranelift-entity | 0.134.4 | 0.135.3 | 26094b0d1072871e |
| cranelift-frontend | 0.134.4 | 0.135.3 | 247fa35f45fd4477 |
| cranelift-isle | 0.134.4 | 0.135.3 | ad1e07f7c111564a |
| cranelift-native | 0.134.4 | 0.135.3 | 4d0ee388c00a256f |
| cranelift-srcgen | 0.134.4 | 0.135.3 | 1f739117ccbeec70 |
| pulley-interpreter | 47.0.4 | 48.0.3 | 76169f68135d6e5b |
| pulley-macros | 47.0.4 | 48.0.3 | be2b53c01b72bd94 |
| regalloc2 | 0.15.1 | 0.15.2 | 757712e8e61590d6 |
| wasm-compose | 0.252.0 | 0.254.0 | 8fd717357bff09b5 |
| wasm-encoder | 0.252.0 | 0.254.0, 0.259.0 | 09480d646178e5fd, b1d0246511d901aa |
| wasm-metadata | — | 0.254.0 | b01df5f3b4ca7881 |
| wasmparser | 0.252.0 | 0.254.0, 0.259.0 | d5769a29f799fbab, 0f7c12eac7bb5878 |
| wasmprinter | 0.252.0 | 0.254.0 | 64e3ba11e024f504 |
| wasmtime | 47.0.4 | 48.0.3 | 8ec755a941ae13c7 |
| wasmtime-environ | 47.0.4 | 48.0.3 | 9dca6a4a583fe560 |
| wasmtime-internal-cache | 47.0.4 | 48.0.3 | fde154a8c7afe761 |
| wasmtime-internal-component-macro | 47.0.4 | 48.0.3 | 0363b11ef205e551 |
| wasmtime-internal-component-util | 47.0.4 | 48.0.3 | 0b9bfdaa8f3e07e7 |
| wasmtime-internal-core | 47.0.4 | 48.0.3 | 72250d72e27bea7d |
| wasmtime-internal-cranelift | 47.0.4 | 48.0.3 | 52abc6a4ac7ab272 |
| wasmtime-internal-fiber | 47.0.4 | 48.0.3 | 818599f226d134c0 |
| wasmtime-internal-jit-debug | 47.0.4 | 48.0.3 | 0a0c01126eeaa5d4 |
| wasmtime-internal-jit-icache-coherence | 47.0.4 | 48.0.3 | 4d42e1fc31e9e23c |
| wasmtime-internal-unwinder | 47.0.4 | 48.0.3 | 71f967bac5a11867 |
| wasmtime-internal-versioned-export-macros | 47.0.4 | 48.0.3 | 65b0174e50f6f882 |
| wasmtime-internal-wit-bindgen | 47.0.4 | 48.0.3 | 62a5137cc642ddf5 |
| wast | 252.0.0 | 259.0.0 | c69beba8d9da07af |
| wat | 1.252.0 | 1.259.0 | c6eec44b0c80391b |
| wit-component | — | 0.254.0 | b0e65bb94c369b3c |
| wit-parser | 0.252.0 | 0.254.0 | 1655131e4f7d3f0c |
