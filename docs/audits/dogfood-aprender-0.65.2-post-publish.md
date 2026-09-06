# Dogfood aprender 0.65.2 — post-publish receipt (single host, lambda)

PMAT-246. Registry-published `aprender` 0.65.2, installed fresh from crates.io into a
throwaway root (never `~/.cargo/bin`, never a workspace build) and probed by absolute
path. This is a single-host smoke receipt, not a re-run of the fleet-wide 0.65.2
dogfood already recorded at `evidence/dogfood/0.65.2/{lambda,intel,gx10,mini}.json`
and `evidence/dogfood/0.65.2/VERDICT.md` (decided_by noah, 2026-09-05: **NO-GO on
measured parity evidence, publish kept**). This host (`noah-Lambda-Vector`) is that
matrix's `lambda` row; where a probe here touches the same ground as that receipt the
result is cross-checked against it below rather than re-litigated.

## Identity

| field | value |
|---|---|
| crate | `aprender` |
| version | `0.65.2` |
| registry check | `cargo search aprender --limit 1` → `aprender = "0.65.2"` (exit 0); `curl https://crates.io/api/v1/crates/aprender` → `.crate.max_version = "0.65.2"` |
| install command | `cargo install aprender --version 0.65.2 --locked --root /tmp/aprender-0.65.2-dogfood` |
| install exit | `0` |
| install wall time | 85s (`Finished release profile [optimized] target(s) in 1m 24s`) |
| binary produced | `/tmp/aprender-0.65.2-dogfood/bin/apr` (one binary; no others) |
| binary sha256 | `77f01d6a6e04690e50433894ebfe57e4bfcd31aca20acb144815f1e29e0f3258` |
| `apr --version` output | `apr 0.65.2 (v0.65.2+no-git)` (exit 0) |
| host | `noah-Lambda-Vector`, Linux 6.8.0-90-generic x86_64, AMD Ryzen Threadripper 7960X, NVIDIA RTX 4090 |
| rustc / cargo | `rustc 1.98.0 (88d9e12ae 2026-08-18)` / `cargo 1.98.0 (797e8a9bc 2026-08-05)` |
| repo HEAD (this worktree, unrelated to the installed artifact) | `027ed889dfc6665a461a5baeee4f53b872395e2f` |
| date | 2026-09-06 |

**Cross-check against the fleet receipt for this same host**
(`evidence/dogfood/0.65.2/lambda.json`, dated 2026-09-04, installed to the default
`~/.cargo/bin` with `--force`): `binary_sha256` there is
`77f01d6a6e04690e50433894ebfe57e4bfcd31aca20acb144815f1e29e0f3258` — **byte-identical**
to the binary this run just built from the same source (`--version` string also
identical). The registry artifact is reproducible on this host across two independent
installs two days apart.

## Probe table

| probe | command | expected | observed | exit | verdict |
|---|---|---|---|---|---|
| version banner | `apr --version` | `apr 0.65.2 (...)`, exit 0 | `apr 0.65.2 (v0.65.2+no-git)` | 0 | PASS |
| help surface count | `apr --help` (parse `Commands:` block) | matches prior fleet receipt (111 advertised) | 111 subcommands listed | 0 | PASS |
| help answers per subcommand | `apr <cmd> --help` for all 111 | every advertised subcommand answers `--help`, except `help` itself | 110/111 answered `--help`; only `help` did not (it IS the help) | n/a | PASS — matches `lambda.json`'s recorded 111/110 exactly |
| error handling: inspect missing file | `apr inspect /tmp/does-not-exist.gguf` | non-zero exit, error on stderr | `error: File not found: ...`, exit 3 | 3 | PASS |
| error handling: explain missing file (historic P0 #2386: printed to stdout, exit 0) | `apr explain /tmp/does-not-exist.gguf` | non-zero exit, error on stderr not stdout | stdout empty, stderr has the error, exit 3 | 3 | PASS — the 0.63.0 regression (#2386/#2368) does not reproduce in 0.65.2 |
| registry: list cached models | `apr list` | prints a model table, exit 0 | prints 34 orphaned models, "0 tracked", exit 0 | 0 | PASS (exits/renders correctly); **observational note**: every cached model still reads `(orphan)` / "0 tracked", the same shape as historic #2408 ("registry manifest never written"), but this run did not control for *how* those 34 files were cached, so it is reported, not scored, as a defect recurrence |
| `--offline` compliance (historic P0 #2379: `--offline` silently did network I/O) | `apr pull --offline hf://Qwen/Qwen2.5-0.5B-Instruct-nonexistent-test-marker` | refuses network access, non-zero exit, no HTTP traffic | `error: Network error: Cannot fetch ... in --offline mode. Network access is disabled by --offline ...`, exit 10 | 10 | PASS — `--offline` is honored in 0.65.2 |
| inference smoke, default backend | `apr run <local 0.5B q4_k_m gguf> "What is 2+2? Answer with just the number." --max-tokens 8` | correct short completion | `Backend: wgpu (Vulkan)`, "Preparing GPU weights: dequantizing 24 layers to F32", output `4`, 22.98s wall, RSS 1.83GB | 0 | WARN — functionally correct, but see banner check below: the default install is not CPU-only, and 8 tokens from a 0.5B q4_k_m model taking ~23s on an idle RTX 4090 host is consistent with the fleet receipt's documented decode/prefill regressions (PMAT-962/963), not a new finding |
| inference smoke, `--no-gpu` | `apr run <same model> <same prompt> --max-tokens 8 --no-gpu` | correct short completion, CPU path, no wgpu banner | output `4`, 22.09s wall, RSS 1.62GB, no `Backend: wgpu` line | 0 | PASS (correct) / WARN (near-identical wall time to the "GPU" path suggests load/dequantize dominates, not decode) |
| MCP stdio initialize | `printf '<initialize JSON-RPC>\n' \| apr mcp` | valid `initialize` response, correct `serverInfo.version` | `{"jsonrpc":"2.0","id":1,"result":{...,"serverInfo":{"name":"aprender-mcp","version":"0.65.2"}}}` | 0 | PASS |
| bare-`apr`-on-PATH shadow (historic P0 #2384) | `which apr` / `apr --version` (ambient shell, not the installed binary) | resolves the version actually intended for use | `which apr` → `/home/noah/.local/bin/apr`, reporting `apr 0.64.0 (v0.64.0+no-git)`, a binary dated 2026-08-24; `~/.cargo/bin/apr` (sha256-identical to this run's install) is shadowed because `.local/bin` precedes `.cargo/bin` on `$PATH` | n/a | FAIL (host-environment finding, not a crate defect) — any subprocess-backed tool that shells to a bare `apr` on this host silently runs 0.64.0, reproducing the #2384 defect *class* even though the underlying CLI resolution bug it named was fixed; this is a stale leftover binary in `.local/bin`, not something the crate can fix |

Probes not run (time-boxed; NotRun, counted against GO per protocol): `apr serve` HTTP
transport routes, `apr code`, `apr chat`, `apr rerank`, `apr rosetta *`, `apr train` /
`apr finetune`, `apr cbtop`, `apr qa`, cross-transport (CLI/HTTP/MCP) invariance check,
any parity/perf lane beyond the two `apr run` timings above.

## Banner / README checks

| claim | location | checked against | result |
|---|---|---|---|
| `cargo install aprender  # CPU ONLY - no GPU backend is compiled in` | `README.md:23` | `apr run` with the plain `cargo install aprender --version 0.65.2 --locked` build (no `--features` requested) prints `Backend: wgpu (Vulkan)` and dequantizes to F32 for a GPU path | **FAIL** — the default install is not CPU-only. Root cause traced in-tree: `crates/apr-cli/Cargo.toml` `default` enables `inference`, which pulls in `realizar` (package `aprender-serve`) as `{ workspace = true, optional = true }` with no `default-features = false`; `crates/aprender-serve/Cargo.toml` declares `default = ["server", "cli", "gpu"]` and `gpu = ["trueno/gpu"]  # Enable GPU acceleration via Trueno (wgpu)`. So the plain, no-flags `cargo install aprender` compiles the wgpu/Vulkan backend in and `apr run` uses it by default. This is the same shape as the historic #2378 finding (0.63.0 ledger) and still reproduces in 0.65.2 |
| `apr mcp` server identifies itself correctly | `serverInfo.version` in the `initialize` response | `apr mcp` stdio probe above | PASS — reports `0.65.2` |
| CLI `--help` surface matches what subcommands answer | top-level `--help` vs. per-subcommand `--help` | full 111-subcommand sweep above | PASS — 110/111, unchanged from the fleet receipt |

## Verdict

**NO-GO.**

Applying the protocol's letter: any red gate is NO-GO, and this run has a red gate —
the README/banner claim `cargo install aprender # CPU ONLY - no GPU backend is
compiled in` (`README.md:23`) does not hold for the artifact installed straight off
crates.io. This is a live, reproduced defect on 0.65.2, not a carried-over historic
one, with a traced root cause (`aprender-serve`'s default `gpu` feature is inherited
through `apr-cli`'s `inference` feature with no `default-features = false` override).

Failing probes:
- Banner check: `README.md:23` CPU-only claim — FAIL (see above)
- `bare-apr-on-PATH shadow` — FAIL, but scored as a host-environment finding, not a
  crate defect (a stale 0.64.0 binary left in `/home/noah/.local/bin`, ahead of
  `~/.cargo/bin` on `$PATH`); recorded because it reproduces the #2384 defect *class*
  in practice on this box, not because the 0.65.2 crate is at fault
- Several probes NotRun (listed above) — NotRun counts against GO per protocol,
  independent of the banner failure

This is consistent with, not contradictory to, the standing determination in
`evidence/dogfood/0.65.2/VERDICT.md`: that document already recorded **post-publish
dogfood NO-GO on measured parity evidence** for 0.65.2 across all four fleet hosts
(decided_by noah, 2026-09-05, decision: publish stays, no yank, no 0.65.3, these
receipts become the 0.66 baseline). This receipt does not reopen that decision; it
adds one more concretely reproduced, previously-undocumented-for-0.65.2 defect (the
banner/default-backend mismatch) found by an independent single-host smoke pass, and
confirms bit-for-bit binary reproducibility of the published artifact on this host.

## Gaps

- This receipt covers one host (`lambda`) with a narrow, time-boxed probe list (CLI
  help surface, four error-handling/compliance probes, two inference smokes, one MCP
  probe, one PATH-hygiene observation). It is not a repeat of the ~700-invocation
  0.63.0 audit (`docs/audits/dogfood-0.63.0-ledger.md`) and does not attempt to
  reproduce or re-verify every P0/P1 row in that ledger against 0.65.2 — only the
  four called out above (`#2386`/`#2368` explain-exit-code, `#2379`/`#2416`
  `--offline`, `#2378` default-backend, `#2384` bare-path-resolution) were
  deliberately re-checked because they were historically P0.
- `apr list`'s "0 tracked + 34 orphans" output is reported as an observation, not
  scored as a recurrence of historic #2408: this run did not control for how those 34
  cache entries were produced (some may predate `0.65.2` entirely), so attributing the
  orphan status to the current binary would be an unfounded claim.
- No new HTTP-transport, cross-transport-invariance, `apr code`/`apr chat`/`apr
  rerank`/`apr rosetta`/train-family, or accelerated (`--features cuda`) probes were
  run here — the existing fleet receipt (`evidence/dogfood/0.65.2/lambda.json`) already
  carries a CUDA-feature install and parity lanes for this host and is the citation of
  record for those surfaces; duplicating that multi-hour parity harness was out of
  scope for this single-file, single-worker ticket.
- The `README.md:23` finding is scoped to a doc/messaging defect (an incorrect user
  promise about the default install), not a functional regression — `apr run` still
  produces a correct completion on both the default and `--no-gpu` paths in this
  probe; the defect is that the default path is silently *not* what the README says
  it is.
- The bare-`apr`-on-PATH shadow is this host's own leftover state (a manually
  installed 0.64.0 binary in `~/.local/bin` predating this ticket) and is not
  something a code change to the `aprender` crate can fix; it is recorded because a
  subprocess-backed tool (MCP, `apr code`) invoking a bare `apr` on this exact host
  would silently run the wrong version today.
