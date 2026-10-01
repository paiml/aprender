# Phase 8: Laya Decision Model: Local Fine-Tune and Thin MCP Server - Research

**Researched:** 2026-09-25
**Domain:** ModernBERT encoder port (Rust, trueno GEMM), Python fine-tune/calibration harness (uv + torch), APR v2 artifact schema, thin pmcp MCP server on pmcp.run default Lambda with S3-loaded weights
**Confidence:** HIGH for the in-repo architecture, the spike-proven numerics and the deploy traps (each probed this session); MEDIUM for the gate thresholds and the `classify` bound values, which are derived from spike measurements rather than measured on pmcp.run

## Summary

Spikes 024–026 have already de-risked the numerics. A full fine-tune of Laya beats SetFit on stance (0.538 / 0.608 F_avg at 16 / 64 shots). A roughly 400-line Rust port on `trueno::blis::gemm_blis` matched torch fp32 to 3.8e-6 on probabilities with ids 14/14. Default Lambda at 10,240 MB cold-starts Laya from S3 in 11.3–12.0 s and decides a tweet in 0.75 s on Graviton2. The in-tree trueno already carries the NEON 8×6 microkernel (commit `9d67b7247`), so the spike-016 worktree is **not** needed. What remains is productisation, and this session found five traps that decide how the plan must be shaped. Each was probed, not assumed:

1. **`cargo pmcp deploy` would ship the WRONG server.** cargo-pmcp resolves the package to build by `<deploy-root>/{server_name}-lambda`, and if that path does not exist it falls back to the *first* `*-lambda` package with a `bootstrap` bin in `cargo metadata` order. That order is alphabetical (probed: `sorted? True`). The current order is `aprender-mcp-chronos-lambda`, `aprender-mcp-setfit-lambda`, `aprender-setfit-train-lambda`. A new `aprender-mcp-decide-lambda` sorts after chronos, so the chronos-style deploy command (`--manifest-path crates/aprender-mcp-decide-lambda`) would silently build and ship the Chronos binary under the decide server's name. The fix belongs upstream in cargo-pmcp, or the plan needs a workaround and a post-deploy identity check.
2. **`serde_json/preserve_order` is feature-unified in by pmcp** (pmcp 2.19.3 enables it unconditionally), but it is absent when a library crate is tested on its own. Criteria order is the label index (D-05), so any path that goes through `serde_json::Value`/`Map` permutes labels silently in one build shape and not the other. Probed: a `MapAccess` visitor reading from the raw bytes keeps document order under both features, while `from_value` over a `Value` does not.
3. **Lambda `/tmp` is 512 MB and cargo-pmcp cannot raise it.** `[server]` exposes only `memory_mb` and `timeout_seconds`. The ~0.85 GB artifact must be downloaded straight into memory, not to `/tmp` as spike 026 did (spike 026 set a 4 GB ephemeral volume by hand).
4. **pmcp.run sits behind the API Gateway HTTP API's 30 s integration timeout, which cannot be raised.** A cold start (~12 s), plus a 512-token row at 5.6 s on Graviton2, means a count-only list bound is either useless or unsafe. The `classify` maximum needs a **token budget** priced for a cold start on Graviton2.
5. **The fine-tuned checkpoint is served in F16 (D-17), so every Python-side number must be computed on the F16-rounded reload.** That covers the gate, the calibration and the parity probes. Scoring the fp32 in-memory model would compare a model that is never served.

**Primary recommendation:** Lift the spike-025 port as-is into `aprender-core::models::modernbert` (encoder only, generic prefix-aware loader) and `aprender-decide::laya` (head, scorer, builder, temperature). Write the .apr with apr-format's `AprV2Writer` from a Rust packer (an `examples/` target, so no new bin). Make Python own only training, calibration and the gate, over a pinned `uv` project. Have the packer re-score every eval row in Rust and refuse the pack if any |Δp| > 1e-5 against the Python F16-reload probabilities. Serve through `aprender-mcp-decide` / `aprender-mcp-decide-lambda`, with in-memory S3 ranged GETs and a pinned sha256. Gate the live deploy on the cargo-pmcp package-resolution fix plus a post-deploy identity probe.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Local training harness
- **D-01:** Training runs in **Python, on Laya's own code** — the spike-024 full fine-tune (`ft_laya.py` recipe:
  rows built by Laya's `Agent._encode_state`, so training rows are byte-identical to inference rows). Rust owns
  inference; the handoff is probability parity to Python fp32. A Rust trainer is not planned.
- **D-02:** Invocation is **`just laya-train` recipes over an in-tree, pinned `uv` project** (lockfile pinning torch,
  transformers, and Laya at `4066d5d` / weights `convaiinnovations/laya` @ `55cf4c4e`). No new `apr` subcommand — the
  CLI registry (`contracts/apr-cli-commands-v1.yaml`) is untouched.
- **D-03:** Device is **auto-selected MPS → CUDA → CPU**, and the report records the device *actually used* (from
  torch, not an env var) — CLAUDE.md Verification Discipline #2. CPU is allowed but is flagged in the report.
- **D-04:** Base checkpoint is the **English root only** in this phase. The base is a **declared field** in the
  artifact manifest (not an implicit assumption), so adding multilingual (mmBERT) and `typed-decisions` later is
  additive. Recipe values are fixed to the spike's: AdamW, encoder lr 2.5e-5, head lr 1e-4, cosine to 1e-6, clip
  1.0, batch 8, loss CE + (−proper_reward, w 0.75); epochs 12 at ≤16 shots/class, a declared value in 4–12 at 64.
  The recipe is written into the artifact before any score is read.

#### Dataset in, quality gate out
- **D-05:** Input is **`task.json` + `train.jsonl`**. `task.json` = `{type: "choice", instructions, criteria}` with
  criteria **ordered** (order is the label index — `serde_json` `preserve_order`, per spike 026). `train.jsonl` rows
  are `{text, label}`. The served tool reads the same `task.json` from the artifact.
  — **Reversibility:** costly — the schema is shared by trainer, artifact manifest and server; changing it touches all three plus fixtures.
- **D-06:** **`eval.jsonl` is required** and held out for the gate. Temperature calibration is refit on a **seeded
  held-out slice of the train shots** (spike 024 step 2 — every fine-tuned run is over-confident, ECE 0.17–0.38).
  Eval and calibration data never overlap.
- **D-07:** **Fail-closed gate.** The report must show (a) the fine-tuned model beats zero-shot Laya on `eval.jsonl`
  and (b) post-calibration ECE is under a **declared** ceiling. The deploy recipe **refuses** an artifact without a
  passing report. Thresholds are declared in a contract before any run is read (Phase 5 claims-gate philosophy).
- **D-08:** **One declared seed by default; `--seeds N` produces a variance report** (mean ± sd). The report states
  "single seed" plainly when N = 1 (spike 024 saw 0.545–0.679 across seeds at 64 shots). The shipped model is always
  the declared seed — **never the best seed on eval**.

#### MCP tool surface
- **D-09:** The predict server exposes a **task-bound `classify`** tool: the question and labels come from the
  artifact's `task.json`; the caller sends only text. No generic `decide`/`/v1/systemone` surface — a fine-tuned
  model answers one question. One trained model per deployed server (thin-server rule).
  — **Reversibility:** costly — this is the published tool contract agents integrate against.
- **D-10:** `classify` accepts a **list of texts** with a **contract-owned maximum** (a tool-boundary contract in
  the shape of `contracts/forecast-tool-boundary-v1.yaml`). Per-row forwards are acceptable; batched GEMM is an
  optimisation, not a requirement.
- **D-11:** Each result returns **`label` + calibrated `probabilities` over every label in order**; the response
  carries **model identity** (artifact content hash + recipe id) so a caller can prove which model answered.
- **D-12:** Text past the 512-token window is **truncated exactly as Laya's builder does**, with **`truncated: true`**
  on that result (parity with Python; the spike-025 fixture has a 512-token truncation row).

#### Crate home, artifact and deploy
- **D-13:** **ModernBERT lives in aprender-core at `crates/aprender-core/src/models/modernbert/`**, beside
  `models/bert/` and shaped like it (config, embeddings, layer, encoder, load-from-.apr). It is a reusable encoder,
  not Laya-specific. aprender-core's own BERT (`models/bert/`, post-norm HF BERT + WordPiece) is the wrong
  architecture on every block and is NOT reused — the port is spike 025's semantics table.
- **D-14:** The decision layer is a new **method-neutral crate `aprender-decide`**: a decision-method seam (artifact
  manifest, `task.json`, the classify contract) with **Laya as the only implementation** (head, scorer, request
  builder, temperature buckets on top of core's ModernBERT). Kev/Jev are deferred. The name is deliberately not
  `llm-`: Laya is an encoder.
  — **Reversibility:** one-way — a crate name becomes permanent once published to crates.io; confirm before the first publish.
- **D-15:** Servers follow the setfit/chronos pairs: **`aprender-mcp-decide`** (stdio, pmcp) and
  **`aprender-mcp-decide-lambda`** (bootstrap). Both join the thin-server `[[bin]]` list in
  `crates/aprender-core/tests/monorepo_invariants.rs` (FALSIFY-MONO-011) — no baseline change is expected.
- **D-16:** CLAUDE.md's realizar-first table gains a **third documented exception row** (decision models), argued
  like SetFit D-09 and Forecast D-07: a 421M encoder with no KV cache and no LLM kernels, whose only parity-proven
  implementation lives with the code; serving a second port through realizar would violate OPS-03.
- **D-17:** The served artifact is **.apr** (converted from Laya's safetensors F16 + tokenizer + configs), stored
  **F16 and widened to f32 at load** — keeps the 0.84 GB artifact and spike 026's ~12 s cold start. Parity is
  proven end to end, **torch → .apr → Rust**, against the spike-025 fixture (probs ≤ 1e-5; spike got 3.8e-6, ids 14/14).
  The .apr also carries (or its manifest references) `task.json`, recipe, calibrated temperatures, gate report and
  sha256 of every input file. An APR schema contract is written in the shape of `contracts/setfit-apr-v1.yaml`.
  — **Reversibility:** costly — the APR schema for ModernBERT/decision artifacts becomes the on-disk format every later method and deployed model reads.
- **D-18:** **Live pmcp.run deploy of one trained model** on default Lambda at 10,240 MB, weights fetched from S3
  at cold start (never baked into the image — spike 026 / `aws-mcp-model-hosting.md`), verified by a real
  `classify` call. The live deploy itself is a **human checkpoint** (outward-facing).
- **D-19:** The demo model is **TweetEval stance** at 16 or 64 shots/class — spike baselines exist to sanity-check
  the gate (full FT F_avg 0.538 ± 0.017 @16, 0.608 ± 0.050 @64; SetFit 0.512 / 0.561).

### Claude's Discretion
- Exact file layout of the uv project and just recipe names.
- How the Python output becomes .apr (a Python-side export vs extending `apr import`/the converter) — pick whichever
  keeps one converter implementation and a clean parity chain; research should compare.
- S3 layout, IAM scoping and the cold-start loader, following the SetFit/Chronos Lambda crates.
- The declared ECE ceiling and zero-shot margin values (must be declared in the contract before the demo run).
- The `classify` list maximum (priced against the Lambda envelope, not guessed — see Phase 7's accepted-region lesson).

### Deferred Ideas (OUT OF SCOPE)
- **Multilingual Laya base (mmBERT-base 322M, 256k vocab, different RoPE)** — future phase, when a use case arrives.
- **`typed-decisions` base as a fine-tune starting point** — future phase; its full fine-tune was never measured.
- **Kev / Jev as `aprender-decide` methods** — future phase; Kev's Rust path lives only on unpushed `spike/016-upstream-sync`.
- **A Laya/decide training MCP server** (the `aprender-mcp-setfit-train` shape) — not requested; this phase trains locally.
- **Rust trainer for ModernBERT** — not planned; Python stays the training oracle.
- **Batched-row GEMM for `classify`** — optimisation after the tracer works.

#### Reviewed Todos (not folded)
- `aprender-train-gpu-ledger-tests-red.md` — matched on generic keywords only; unrelated.
- `qwen35-9b-hybrid-forward-unimplemented.md` — Kev/Qwen territory; relevant only when Kev joins `aprender-decide`.
- `workspace-test-gate-blocked-by-renacer-validate.md` — CI gate issue, unrelated to this phase's scope (may still affect its CI runs).
</user_constraints>

<phase_requirements>
## Phase Requirements

No REQ-IDs are mapped (ROADMAP: "Requirements: TBD"). Coverage below is derived from the CONTEXT decisions and the ROADMAP goal. The planner should use these D-NN rows as its requirement keys.

| ID | Description | Research Support |
|----|-------------|------------------|
| D-01/D-02 | Python fine-tune on Laya's code via `just laya-train` over a pinned uv project | §Standard Stack (Python), §Pattern 5, Pitfall 5, §Environment |
| D-03 | Device auto-select, report records the device actually used | §Pattern 5 (`next(model.parameters()).device`), Pitfall 11 |
| D-04 | English root; base is a declared manifest field; recipe fixed and written first | §Pattern 4 (manifest `base`, `recipe`), verified pins in §Standard Stack |
| D-05 | `task.json` + `train.jsonl`, ordered criteria | Pitfall 2 (preserve_order), §Code Examples (order-preserving visitor, probed) |
| D-06 | Required `eval.jsonl`; seeded calibration slice; no overlap | §Pattern 5 calibration, Pitfall 6 (clamp [0.5, 5]) |
| D-07 | Fail-closed gate, thresholds declared in a contract first | §Pattern 6, recommended values in §Assumptions A6/A7 |
| D-08 | One declared seed; `--seeds N` variance report | §Pattern 5 |
| D-09/D-10/D-11/D-12 | Task-bound `classify`, contract-owned list max, label + ordered probs + identity, truncation flag | §Pattern 7, Pitfall 4 (token budget), §Code Examples (response shape) |
| D-13 | `models/modernbert/` in aprender-core, reusable | §Pattern 1, §Recommended Project Structure |
| D-14 | `aprender-decide` method-neutral seam, Laya only | §Pattern 2 |
| D-15 | `aprender-mcp-decide` + `-lambda`, FALSIFY-MONO-011 register | §Pattern 7, Pitfall 7, Pitfall 1 |
| D-16 | CLAUDE.md third realizar-first exception row | §Pattern 8, Pitfall 9 (FALSIFY-DOCS-CLAUDE-001) |
| D-17 | .apr F16 widened at load; torch → .apr → Rust parity ≤ 1e-5; schema contract | §Pattern 3/4, Pitfall 5, §Validation Architecture |
| D-18 | Live pmcp.run deploy, S3 at cold start, human checkpoint | §Pattern 9, Pitfalls 1, 3, 4 |
| D-19 | TweetEval stance demo, 16 or 64 shots | §Pattern 5 (in-repo `data/tweet-eval-stance/`, `benchmarks/tweeteval-stance/selections/`) |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Realizar-first table.** All inference/serving goes through `realizar` unless an exception is argued in the table. D-16 adds a third row, argued like SetFit (Phase 4 D-09) and Forecasting (Phase 6 D-07). It must also be worded so the SafeTensors carve-out is not widened: after this phase the served path reads .apr, not safetensors.
- **Contracts use `pv`, never bash/yq/python re-implementations.** In this tree `pv` is `cargo run --release -p aprender-contracts-cli --bin pv --` (Makefile:1878 `PV_BIN`); it is **not on PATH**. `pv diff` takes two file paths, not a git revision.
- **Code search uses `pmat query`**, not grep, for semantic lookups (the planner's tasks should say so).
- **Workspace lints:** `unsafe_code = "forbid"`, clippy all + pedantic warn, and `unwrap()` is banned (`.clippy.toml` disallowed-methods). Use `expect()` / `ok_or_else(..)?`. The `serde_json::json!` / schemars derive exception is taken with a file-level `#![allow(clippy::disallowed_methods)]`, as in `aprender-mcp-setfit`.
- **Coverage co-evolves with contracts:** every tested function gets a `#[contract]` annotation and falsification conditions in the YAML. Note that `#[contract]` is metadata-only in this repo (see setfit-apr-v1 "ANNOTATION HONESTY"); the enforcement is the named test.
- **Verification discipline:** never read `$?` through a pipe (and the Bash tool runs **zsh**, so use `bash -c`/script files for pipelines and loops). Prove the mechanism engaged (the device actually used, the Graviton part from `/proc/cpuinfo`). Pin the binary. Guard regexes ship case tables. Watch for shadowed artifacts.
- **Git:** `main` is protected; work goes on a branch through a PR. **Check in before** modifying `.github/workflows/*.yml`, crates.io publish, or destructive ops. Compute spend over 1 hour on non-lambda-vector hosts needs a check-in.
- **Publishing safety (CB-510):** `.gitignore` patterns are root-anchored (`/models/`). Weights are never committed. Re-run the include-guard scripts after any ignore change; `check_package_includes.sh` is the one that actually works on macOS (memory note).
- **Scripts:** use `justfile` for deploy and dev recipes (user global rule; the justfile's own header says it is the deployment surface and the Makefile holds the quality gates). Shell scripts pass `bashrs lint`; sourced libraries stay option-neutral.
- **Tests:** a new `tests/*.rs` target is dark until it is added to the explicit `--test` line in `.github/workflows/ci.yml:471`. `cargo build --examples --workspace --keep-going` runs in CI on that same line, so an `examples/` packer is at least compiled in CI.
- **Tensor layout:** everything written to .apr is row-major (`contracts/tensor-layout-v1.yaml`). ModernBERT weights are `[out, in]` row-major and are used as the GEMM A operand with no transpose (the spike-020 layout).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Fine-tune, calibration, gate report | Back-office Python (uv project, Laya's own code) | — | D-01: training stays in Python; rows must be built by Laya's `Agent._encode_state` |
| F16 checkpoint → .apr conversion + Rust re-score | Rust library (`aprender-decide::artifact`, run via an example) | apr-format `AprV2Writer` | One container writer (apr-format); the parity check needs the Rust forward |
| ModernBERT encoder forward | Rust library (`aprender-core::models::modernbert`) | trueno `gemm_blis` | D-13: a reusable encoder; GEMM-bound work lives in trueno |
| Laya head, scorer, builder, temperature, task binding | Rust library (`aprender-decide::laya`) | tokenizers 0.23.1 | D-14: the decision layer is method-specific, not an encoder concern |
| Tool boundary (bounds, refusals, response shape) | MCP transport (`aprender-mcp-decide`) | contract YAML | Thin-server rule: the transport owns bounds; the library door re-checks nothing (forecast precedent) |
| Weight fetch at cold start, loopback proxy | Lambda transport (`aprender-mcp-decide-lambda`) | S3, API Gateway (pmcp.run) | D-18; S3 and Lambda concerns must not leak into the library |
| Artifact storage, IAM | AWS (S3 bucket + role policy) | CDK in `deploy-extensions/` | Follows the SetFit training stack precedent; pmcp.run owns the function role |

## Standard Stack

### Core (Rust) — every crate is already in `Cargo.lock`; no new registry dependency
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `aprender-compute` (trueno) `blis::gemm_blis` | workspace (in-tree, NEON 8×6 kernel at `9d67b7247`) | Every dense product | [VERIFIED: crates/aprender-compute/src/blis/compute.rs:838 `pub fn gemm_blis(`; microkernels/neon.rs has `microkernel_8x6_neon`, dispatched at compute.rs:21-22,131] The only aprender dependency spike 025 needed |
| `apr-format` v2 (`AprV2Writer`, `AprV2ReaderRef`, `TensorDType::F16`) | workspace | The .apr container | [VERIFIED: crates/apr-format/src/v2/tensor_index_impl.rs:159 `F16 = 1,`; writer.rs:42 `pub fn add_tensor(`] One container implementation |
| `aprender-core::format::AprV2DequantExt::get_tensor_as_f32` | workspace | F16 → f32 widening at load | [VERIFIED: crates/aprender-core/src/format/dequant_ext.rs:85-90, uses `apr_format::f16_to_f32`, which is `f16::from_bits(bits).to_f32()` at apr-format/src/f16.rs:30-31] The same `half` conversion the spike used |
| `batuta_common::math::erfc_precise` | workspace | Exact GELU (erf form) | [VERIFIED: aprender-core/src/autograd/ops/activation.rs:205 `batuta_common::math::erfc_precise(-xd / std::f64::consts::SQRT_2)`] The house erf. Use it instead of adding `libm` (the spike used `libm::erf`; re-prove parity) |
| `tokenizers` | 0.23.1 (workspace pin) | ModernBERT BPE `tokenizer.json` | [VERIFIED: Cargo.toml:242 `tokenizers = { version = "0.23.1", default-features = false, features = ["fancy-regex"] }`; Cargo.lock has 0.23.1] The version spike 025 reproduced ids 14/14 with |
| `pmcp` | 2.19.3 (locked) | MCP server, stateless streamable HTTP | [VERIFIED: Cargo.lock:9964-9965] Template crates use it |
| `serde_json` | 1.0.150 | Parsing (see Pitfall 2) | — |
| `sha2` | 0.10.9 | Artifact/input hashes | Already used by setfit |
| `aws-sdk-s3` / `aws-config` | 1.137.0 / 1.8.18 (locked) | Cold-start weight fetch (Lambda crate only) | [VERIFIED: Cargo.lock:2792-2793, 2649] Do **not** `cargo update` it: >1.137 needs rustc 1.94.1 [CITED: aws-mcp-model-hosting.md Constraints], and the pin is `channel = "1.93.0"` (rust-toolchain.toml:2) |
| `lambda_http` 0.13.0, `reqwest` 0.12.28, `once_cell` | locked | Loopback bootstrap | Copied from `aprender-mcp-chronos-lambda` |

### Core (Python, back-office only)
| Package | Version | Purpose | Notes |
|---------|---------|---------|-------|
| `laya` | git `https://github.com/NandhaKishorM/laya` @ `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0` | Training rows, `proper_reward`, `Agent.predict`, `clamp_temperature` | [VERIFIED: `rtk proxy git rev-parse HEAD` in `.planning/spikes/024-*/vendor/laya` → that SHA] Not on PyPI; install as a git dependency |
| weights `convaiinnovations/laya` | revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | English root | [VERIFIED: HF cache LFS blob name = sha256 `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c` for `model.safetensors`, 803.6 MiB] Pass `expected_sha256` to `Agent(...)` (supported by the `Agent.__init__` signature, agent.py:246) |
| `transformers` | 5.17.0 | ModernBERT (SDPA) | The version spike 025's semantics table was read against [CITED: laya-rust-inference.md §2]; exists on PyPI [VERIFIED: pip index] |
| `torch` | 2.14.0 | Training / oracle | [ASSUMED] the spike's resolved version (newest cached wheel); pin in `uv.lock` at Wave 0 and record it |
| `tokenizers`, `safetensors`, `huggingface-hub`, `numpy` | resolved by the lock | — | Pinned via `uv.lock` |
| Python | 3.13.7 (`.python-version`) | — | Precedent `scripts/setfit_fixtures/.python-version` = `3.13.7` |

### Alternatives Considered (conversion path — Claude's discretion)
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Rust packer on `AprV2Writer` (recommended) | Python-side .apr writer | A **second** implementation of the APR v2 container in Python. It violates one-implementation and cannot run the Rust re-score, so it is rejected |
| Rust packer | `apr import --arch modernbert` (extend `Architecture` enum + converter) | `Architecture` (converter_types.rs:113) feeds tensor-expectation, name-map and config-inference match arms. That is a large blast radius, and it still carries no task / recipe / gate / probes, so a second step would be needed anyway. Defer until a plain ModernBERT (no decision head) needs importing |
| Custom deploy root | Upstream cargo-pmcp fix | See Pitfall 1 / Open Question 1 |

**Installation (Python project, suggested `scripts/laya_train/`):**
```bash
cd scripts/laya_train && uv lock && uv sync --frozen   # lock is committed; CI never runs it
```

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | Verdict | Disposition |
|---------|----------|-----|-----------|-------------|---------|-------------|
| pmcp | crates.io | multi-year | (seam: OK) | github.com/paiml/rust-mcp-sdk | OK | Approved (already locked) |
| tokenizers | crates.io | multi-year | OK | github.com/huggingface/tokenizers | OK | Approved (locked) |
| aws-sdk-s3 / aws-config | crates.io | multi-year | OK | github.com/awslabs/aws-sdk-rust | OK | Approved (locked) |
| lambda_http, sha2, half | crates.io | multi-year | OK | — | OK | Approved (locked) |
| torch | PyPI | seam flag `too-new` (newest release) | seam: unknown | github.com/pytorch/pytorch | SUS | Flagged — human-verify before `uv lock` |
| transformers | PyPI | `too-new` | unknown | github.com/huggingface/transformers | SUS | Flagged — human-verify |
| tokenizers (py) | PyPI | `too-new` | unknown | github.com/huggingface/tokenizers | SUS | Flagged — human-verify |
| safetensors | PyPI | — | unknown | github.com/huggingface/safetensors | SUS | Flagged — human-verify |
| huggingface-hub | PyPI | `too-new` | unknown | github.com/huggingface/huggingface_hub | SUS | Flagged — human-verify |
| numpy | PyPI | `too-new` | unknown | seam reports `no-repository` | SUS | Flagged — human-verify |
| laya (git) | not a registry package | — | — | github.com/NandhaKishorM/laya | n/a | Pin the commit SHA; a human confirms the remote |

The seam flagged every PyPI package SUS on the signals `too-new` / `unknown-downloads` (the seam could not read PyPI download counts). They are canonical packages, but the rule is the rule. **The planner inserts one `checkpoint:human-verify` before the first `uv lock`**, and that checkpoint confirms the names, the versions and the `uv.lock` hashes. No package was removed (`SLOP`: none).

## Architecture Patterns

### System Architecture Diagram

```
 BACK OFFICE (laptop, Python via `just laya-train`)                    SERVING (Rust)
 ─────────────────────────────────────────────────                     ─────────────────────────────
 task.json + train.jsonl + eval.jsonl
        │  validate (ordered criteria, labels ∈ criteria, no eval/train overlap)
        ▼
 seeded split: train shots ──► fit slice │ calibration slice (stratified, seed)
        │
        ▼
 Laya Agent(root@55cf4c4e, sha256-pinned) ─► full FT (D-04 recipe, device = torch-reported)
        │
        ▼
 save F16 checkpoint dir ──► RELOAD in torch fp32 from F16  ◄── everything below uses the reload
        │                         │
        │               ┌─────────┼──────────────────┐
        │               ▼         ▼                  ▼
        │      zero-shot Laya   fit T (bucket) on   eval.jsonl probs (post-calibration)
        │      on eval.jsonl    calibration slice   + probes (N short rows)
        │               └─────────┬──────────────────┘
        │                         ▼
        │                 gate-report.json (declared thresholds from contract) ── FAIL ─► stop
        ▼                         │ PASS
 checkpoint dir + rl_agent_config (calibrated T) + task.json + recipe.json + gate-report.json
        │
        ▼  `just laya-pack` (cargo run --example, Rust)
 aprender-decide::artifact::pack ──► AprV2Writer: F16 tensors (raw bytes) + tokenizer.blob + task.blob
        │                              + one custom key "decide" (manifest)
        ▼
 reload .apr in Rust ─► re-score every eval row + probes ─► |Δp| ≤ 1e-5 & argmax equal ─ else refuse
        │
        ▼
 laya-stance.apr (~0.85 GB, sha256 H)  ──► `aws s3 cp` ──► s3://<bucket>/decide/<server>/<H>.apr
                                                                     │
 MCP client ─► pmcp.run API GW (30 s cap) ─► Lambda `bootstrap` (10,240 MB)
                                                  │ first call: ranged GETs into RAM ─► sha256 == H (env pin)
                                                  │            ─► load ladder ─► probe replay ─► ready
                                                  ▼
                                   loopback pmcp StreamableHttpServer (stateless)
                                                  │ classify{texts}
                                                  ▼
                        precheck (count, bytes, token budget) ─► aprender_decide::Decider::classify
                                                  ▼
                        {model:{artifact_sha256:H, recipe_id, base, method:"laya"}, labels:[..],
                         results:[{label, probabilities:[..in order], truncated, tokens}]}
```

### Recommended Project Structure
```
crates/aprender-core/src/models/modernbert/   # D-13: config.rs, embeddings.rs, layer.rs (attn+mlp), encoder.rs,
                                              #        load.rs (prefix-aware, from AprV2ReaderRef), gemm.rs (Linear on gemm_blis)
crates/aprender-decide/                       # D-14: lib only (NO [[bin]] — FALSIFY-MONO-011)
  src/lib.rs          # DecisionMethod seam, Decider, Decision, ModelIdentity
  src/task.rs         # task.json parse (order-preserving visitor), label set
  src/artifact.rs     # decide-apr-v1: manifest, writer, bounded read, load ladder, probes
  src/laya/{mod,head,scorer,builder,temperature}.rs
  examples/pack_laya.rs   # the packer + Rust re-score (run by `just laya-pack`)
  tests/fixtures/     # tiny synthetic ModernBERT+Laya fixture (CI), spike-025 fixture copy (gated)
crates/aprender-mcp-decide/                   # D-15 stdio server (publish = false)
crates/aprender-mcp-decide-lambda/            # D-15 bootstrap + S3 loader (publish = false), .pmcp/ (see Pitfall 1)
contracts/decide-apr-v1.yaml                  # artifact schema + load ladder (setfit-apr-v1 shape)
contracts/laya-parity-v1.yaml                 # torch → .apr → Rust tolerances (chronos-bolt-parity shape)
contracts/decide-tool-boundary-v1.yaml        # classify bounds + gate thresholds (forecast-tool-boundary shape)
scripts/laya_train/                           # uv project: pyproject.toml, uv.lock, .python-version, train.py, gate.py, fixtures.py
deploy-extensions/lib/decide-weights-stack.ts # S3 bucket + read policy output (optional; see Pattern 9)
```

### Pattern 1: ModernBERT in aprender-core as a lift of spike 025 (plain slices, no autograd)
**What:** Move the encoder half of `sources/025-laya-rust-forward-parity/src/laya.rs` (`Linear` on `gemm_blis`, `layer_norm` with f64 accumulation, rotate-half `rope`, windowed `attention`, GeGLU MLP) into `models/modernbert/`. Keep the numerics byte-for-byte (the parity was first-run and must not be re-debugged). Apply four generalisations:
- Per-layer attention type comes from `config.layer_types` when present (the checkpoint has it; `encoder/config.json` lists `"full_attention"` / `"sliding_attention"` for all 28 layers). Fall back to `i % global_attn_every_n_layers == 0`, and assert the two agree when both exist.
- The window is `local_attention / 2` inclusive (`|i−j| ≤ 64`), proven by mutation. Keep the `WINDOW` mutation as a test.
- RoPE thetas come from `rope_parameters.full_attention.rope_theta` (160000.0) and `sliding_attention.rope_theta` (10000.0) [VERIFIED: cached `encoder/config.json`].
- `load.rs` takes a **tensor-name prefix** (`"encoder."` for Laya). A plain HF ModernBERT then loads with a different prefix, which is what makes it reusable (D-13).

**Feature gating:** rayon is optional in core (`default = ["parallel"]`, `parallel = ["rayon"]`, aprender-core/Cargo.toml:226-227 [VERIFIED]). Gate the module `#[cfg(feature = "parallel")]` so `cargo check -p aprender-core --no-default-features` stays green, with no new feature and no tokenizer dependency in core. The tokenizer and builder live in aprender-decide.
**Do not** route through `autograd::Tensor` or `models/bert`. Both are the wrong architecture and slower (D-13).

### Pattern 2: `aprender-decide`, one seam designed for a known second implementation
**What:** A `DecisionMethod` trait with one impl (`Laya`):
```rust
// Sketch — names are proposals; the trait must not grow Kev-speculative methods.
pub trait DecisionMethod: Send + Sync {
    fn task(&self) -> &Task;                       // parsed task.json (ordered labels)
    fn identity(&self) -> &ModelIdentity;          // artifact_sha256, recipe_id, base, method
    fn classify(&self, texts: &[String]) -> Result<Vec<Decision>, DecideError>;
}
pub struct Decision { pub label_index: usize, pub probabilities: Vec<f32>, pub truncated: bool, pub tokens: usize }
```
The manifest carries `method: "laya"` and `base: {family:"laya", checkpoint:"en-root", repo, revision, sha256}` (D-04 declared base). Kev later adds a variant and a tensor set, not a new artifact shape. `Decider::load(bytes)` dispatches on `manifest.method`, and an unknown method is a typed refusal.

### Pattern 3: .apr layout (`decide-apr-v1`), following the setfit-apr-v1 rules
- `model_type = "decide"` (typed metadata tag). Exactly **one** custom metadata key `"decide"` holds the manifest. `AprV2Metadata.custom` is a `HashMap` with `#[serde(flatten)]` (header_impl.rs:297-299 [VERIFIED]), so several keys serialize in an unspecified order (setfit-apr-v1 "EXACTLY ONE CUSTOM METADATA KEY").
- **Tensors:** all checkpoint tensors under their verbatim Laya names, written with `add_tensor(name, TensorDType::F16, shape, raw_le_bytes)`. Raw bytes are copied from safetensors, never through `add_f16_tensor(f32)`, so the weights are byte-identical to what torch saw. The F32 `temperature` buffer stays F32. `act_head.*` is stored (bijection with the checkpoint) and is unused by `classify`.
- `tokenizer.blob`: U8 tensor holding the byte-identical `tokenizer.json` (3.4 MB), plus its sha256 in the manifest (checked before any tensor is installed, as setfit rung 5 does).
- `decide.task_json` blob: U8 tensor holding the raw `task.json` bytes. **The ordered label list also goes in the manifest as a JSON array**, never as an object (Pitfall 2).
- Manifest fields: `schema:"decide-apr-v1"`, `schema_version`, `method`, `base`, `encoder_config` (from `encoder/config.json`), `agent_config` (head_layers, max_len 512, head_max_len 192), `task_sha256`, `labels:[...]`, `recipe` (+`recipe_id` = sha256 of canonical recipe JSON), `calibration {bucket, t_fitted, t_applied, clamp_hit, slice_ids_sha256}`, `gate_report` (full), `inputs_sha256 {task.json, train.jsonl, eval.jsonl, base model.safetensors, tokenizer.json}`, `device_used`, `probes[]` (input text, expected probs as f32 hex bit patterns, as setfit does).
- No timestamp is written, so two packs of the same inputs are byte-identical (FALSIFY-APR-002 analogue).
- **Size bound:** the setfit cap is `max_artifact_bytes: 1073741824` (setfit-apr-v1.yaml:624 [VERIFIED]). A Laya artifact is ~843 MB weights + 3.4 MB tokenizer + index, about 0.85 GB, so 1 GiB gives only ~1.26× headroom. Derive the cap in `decide-apr-v1` from the tensor set, as setfit does (e.g. 1.25 GiB). Do not inherit the setfit number silently.
- **Integrity:** the APR v2 header CRC covers **only the 64-byte header** (header_impl.rs:104-111 `compute_checksum` over bytes 0..40 and 44..64 [VERIFIED]). Payload integrity therefore has to come from the whole-file sha256 (the D-11 identity) plus the tokenizer/task blob hashes.

### Pattern 4: Load ladder (from setfit-apr-v1 `load_validation_ladder`, adapted)
`bounded read → length cap → header+CRC+row-major → model_type=="decide" + one-key manifest parse (deny_unknown_fields) + schema_version → structural (architecture-derived tensor set from encoder_config, per-entry size == product(shape)·width, tokenizer/task blob sha256) → non-finite scan → rebuild (widen F16) → probe replay → Verified decider` (private constructor; a trybuild compile-fail test proves no second minting path).
- Use `AprV2ReaderRef::from_bytes` (zero-copy). `AprV2Reader::from_bytes` does `data.to_vec()` (reader_impl.rs:169 area [VERIFIED]) and would hold a second 0.85 GB copy.
- **Probe count is priced against cold start:** keep probes to ~2 short rows (≤ 48 tokens, ≈ 0.4 s each on Graviton2, extrapolated from 39 tok = 126 ms on M4 × 3.3). Full-eval parity runs at **pack time**, not at every load.

### Pattern 5: The Python harness (`scripts/laya_train/`, driven by `just laya-train`)
- **Inputs:** `task.json`, `train.jsonl` rows `{text, label}`, `eval.jsonl` (required). Label = the criterion **name**, validated against the criteria keys (see Open Question 3). Refuse eval/train text overlap (hash the normalized text).
- **Demo data is in-repo:** `data/tweet-eval-stance/{train,validation,test}.jsonl` rows `{"id","input","label","label_text","source_split"}`, and attested shot selections at `benchmarks/tweeteval-stance/selections/s{k}-seed{seed}/selection-manifest.json` [VERIFIED: directory listing]. A small converter writes `task.json` with criteria in order `none, against, favor` (label index 0/1/2 per spike tasks.py) and `eval.jsonl` from `test.jsonl` (280 rows).
- **Recipe:** copy `ft_laya.py` exactly (AdamW param groups `encoder.*` 2.5e-5 / rest 1e-4, wd 0.01, cosine to 1e-6 over `epochs·ceil(n/8)` steps, clip 1.0, batch 8, `loss = ce + (−proper_reward(softmax(z/T), onehot, qt, mask, w_sph=0.75, w_rps=1.0)).mean() + 0·act.sum()`, `T` = the initial bucket temperature). Write `recipe.json` and its `recipe_id` **before** training starts.
- **Device:** request MPS → CUDA → CPU explicitly, then record `str(next(agent.model.parameters()).device)` and `torch.__version__` after the model moves. Laya's own fallback prints a warning and silently uses CPU (agent.py:343-354 [VERIFIED]), so the requested device is not evidence.
- **Save:** state_dict → **F16** safetensors (keep the F32 `temperature` buffer as F32), plus the copied `tokenizer/`, `encoder/`, and an `rl_agent_config.json` with the calibrated `temperature_by_options[bucket]`. **Then reload the dir with `Agent(path, device="cpu")` + `.float()`** and do the gate, calibration scoring, eval probabilities and probes on that reload.
- **Calibration:** grid or golden-section search of T on the calibration slice NLL, **bounded to [0.5, 5.0]**. Record `t_fitted`, `t_applied` and `clamp_hit` (Pitfall 6). The slice is seeded and stratified; declare its size (e.g. 25 %, at least 2 per class).
- **Seeds:** default one declared seed. `--seeds N` trains N and reports mean ± sd, but the **shipped** artifact is always the declared seed (D-08); the report prints "single seed" when N = 1.
- **Cost:** 26–316 s per stance run on M4 MPS [CITED: spike 024 results]. Memory: 421M params with AdamW is about 7 GB of fp32 state plus activations [ASSUMED arithmetic]. CPU training is untested and slow, which is why D-03 flags it.

### Pattern 6: The gate (D-07), with thresholds declared first
Put `gate_min_macro_f1_margin` and `gate_max_ece` in `contracts/decide-tool-boundary-v1.yaml` `constants:` (or a separate gate contract), committed **before** the demo run. `gate.py` reads them from the YAML and the pack/deploy recipes re-read them. Pass requires (a) `macro_f1(ft) − macro_f1(zero_shot) ≥ margin` on `eval.jsonl` and (b) post-calibration top-label ECE (15 equal-width bins, the `fewshot015.metrics` definition) `≤ ceiling`. Zero-shot uses the same `task.json` (same criteria text) on the root checkpoint. Recommended values are in the Assumptions Log (A6, A7). The deploy recipe refuses when `gate_report.pass != true` or the report's thresholds differ from the contract's.

### Pattern 7: Thin servers (copy `aprender-mcp-setfit` + `aprender-mcp-chronos-lambda`)
- `aprender-mcp-decide` (lib + stdio bin, `publish = false`) re-exports the model type. `build_server(model: Arc<Verified>, name, version)` registers exactly one tool, `classify`. `ClassifyArgs { texts: Vec<String> }` is `#[serde(deny_unknown_fields)]` (a unit test pins `additionalProperties:false` in the advertised schema). `precheck` enforces the contract bounds with `pmcp::Error::validation` naming the bound, and `spawn_blocking` runs the forward.
- The tool description is built **from the artifact** (instructions + ordered labels) so an agent sees the question it is asking, plus the bounds and the "one complete document per element" guidance copied from SetFit.
- `aprender-mcp-decide-lambda` (`[[bin]] name = "bootstrap"`, `publish = false`, lib with `server_config() = StreamableHttpServerConfig::stateless()`). It reuses the chronos loopback handler (GET health, OPTIONS CORS, proxy). **Lazy load on first call** (the spike default, 11.3–12.0 s). Do not use EAGER init: default Lambda's init phase is capped at about 10 s [ASSUMED], below the ~10.4 s download + build.
- Register both package names in `deployment_unit_bins` (monorepo_invariants.rs:312-319 [VERIFIED: the list currently ends `"aprender-mcp-chronos-lambda",  // Chronos-Bolt AWS Lambda custom runtime (bootstrap)`]). The gate also asserts `publish = false` for every entry. `aprender-decide` must have **no** bin target: the packer is an `examples/` target, which is cargo kind `example`, not `bin`.

### Pattern 8: CLAUDE.md third exception row (D-16)
Add a table row `Decision-model classification (Laya / ModernBERT encoder; Kev, Jev later)` | **Primary** (`crates/aprender-decide` on `aprender-core::models::modernbert`; thin servers `crates/aprender-mcp-decide`, `crates/aprender-mcp-decide-lambda` are transport only) | Never | Compute (`gemm_blis`)`. The argument paragraph follows SetFit/Forecast:
- realizar-first is a performance argument about LLM kernels;
- Laya is a 421M bidirectional encoder with no generation, no KV cache and no LLM kernels;
- its only parity-proven implementation is the spike-025 port (3.8e-6 vs torch fp32, `contracts/laya-parity-v1.yaml`);
- serving a second port through realizar would violate OPS-03 and re-open the parity fixtures.

State that the artifact is .apr (so the SafeTensors carve-out is not widened, and safetensors is read only by the back-office packer). Every path cited must exist at that commit (Pitfall 9).

### Pattern 9: S3 + IAM + deploy (Claude's discretion)
- **Layout:** content-addressed `s3://<bucket>/decide/<server-name>/<sha256>.apr` (immutable; a new model is a new key). The deploy.toml `[environment]` sets `APRENDER_DECIDE_S3_URI` and `APRENDER_DECIDE_SHA256`. The loader refuses a mismatch before parsing, which is also the D-11 identity.
- **Loader:** 16 parallel 64 MB ranged GETs with per-part retry ×5 (spike 021/026), each writing into its disjoint `&mut [u8]` slice of a **pre-sized in-memory buffer**. No `/tmp` (Pitfall 3). Record a load timeline (download MB/s, sha256 ms, build ms, probe ms, RSS) and the Graviton part in the first response's forensics, or log it.
- **IAM:** `s3:GetObject` on `arn:aws:s3:::<bucket>/decide/<server-name>/*` only, plus `s3:ListBucket` scoped to that prefix so a missing key reads as 404, not 403 (the setfit stack's own lesson, setfit-training-stack.ts:278-286). Attach it after `cargo pmcp deploy` with `aws iam put-role-policy` on the discovered role (the `pmcp-train-grant` recipe pattern, justfile ~228-270). Re-run after any pmcp.run redeploy.
- **Bucket:** a small CDK stack in `deploy-extensions/` (tagged; RETAIN in prod) that outputs the policy document, reusing the SetFit training stack's shape. Alternative: add a prefix to that stack's existing artifact bucket.
- `deploy.toml` embeds the account id through the bucket name, so generate it from a tracked template and keep it gitignored (the `pmcp-train-config` precedent).
- `just laya-deploy` refuses unless (1) the gate report passes against contract thresholds, (2) the local .apr sha256 equals `APRENDER_DECIDE_SHA256`, and (3) after deploy a real `classify` returns `model.artifact_sha256 == H` and the `tools/list` description contains the artifact's labels (the identity probe that catches Pitfall 1). The live deploy is a `checkpoint:human-verify` (D-18).

### Anti-Patterns to Avoid
- **Parsing `task.json` into `serde_json::Value` and iterating the object.** The label order then depends on a feature flag (Pitfall 2).
- **Returning probabilities as a JSON object keyed by label.** Same order hazard. Use a top-level `labels: [...]` plus per-result `probabilities: [f32; K]`.
- **Scoring the gate on the in-memory fp32 model.** It is not what ships (Pitfall 5).
- **Shipping the best seed** (D-08), or head-only adapters (`bias` / `head_ft`) as the adaptation (spike 024: they move Laya ≤ 0.05).
- **Downloading weights to `/tmp` on pmcp.run**, or baking them into the zip (spike 021: 7–18× worse).
- **Reusing aprender-core `models/bert` or `setfit/encoder.rs`** for ModernBERT (wrong on every block).
- **Hardcoding head attention as `d/64` heads × 64** (the spike did). Laya uses `nhead = max(1, d // 64)` (common.py:165 [VERIFIED]); hd = d / nhead, which matters for the tiny CI fixture.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| .apr container I/O | a Python APR writer or a second Rust container writer | `apr_format::v2::AprV2Writer` / `AprV2ReaderRef` | One implementation; alignment, index and CRC are handled |
| F16 → f32 | custom bit-twiddling | `AprV2DequantExt::get_tensor_as_f32` → `half` | Already bit-proven against the legacy path (apr-format/src/f16.rs tests) |
| erf / GELU | `libm` dependency or an approximation | `batuta_common::math::erfc_precise` (as `gelu_exact`) | One erf in the house. Approximations miss the 1e-5 bar |
| GEMM | a naive triple loop | `trueno::blis::gemm_blis` with the spike-020 banding | 61 GFLOP/s/core NEON; the port is GEMM-bound |
| BPE tokenization | a Rust BPE | `tokenizers` 0.23.1 `Tokenizer::from_bytes` | ids 14/14 with Python |
| Laya's builder, temperature buckets, `render_options` | a re-derivation from the paper | port `laya.rs::Builder`, `temperature`, spike-026 `render_options` verbatim | Proven against Python; `choice` is the only type served here, but keep the refusals |
| Training loop / proper scoring rule | a re-implementation | Laya's `Agent._encode_state`, `proper_reward`, `clamp_temperature` | D-01; rows byte-identical to inference |
| MCP transport, stateless HTTP | an axum handler | pmcp `Server::builder().tool_typed_with_description` + `StreamableHttpServerConfig::stateless()` | Template precedent; the 4 MiB body cap is already enforced |
| Contract validation | yq/python checks | `pv validate` / `pv audit` via `$(PV_BIN)`, and `test_support::constant_u64`-style readers | CLAUDE.md mandate |

**Key insight:** Every numerically sensitive piece already has a proven implementation (spike code or house code). The risk in this phase is **drift at the seams**: feature unification, F16 rounding, deploy package resolution, the timeout envelope. It is not the math.

## Common Pitfalls

### Pitfall 1: cargo-pmcp deploys the alphabetically-first `*-lambda` crate (would ship Chronos)
**What goes wrong:** `cargo pmcp deploy --manifest-path crates/aprender-mcp-decide-lambda` builds `aprender-mcp-chronos-lambda` and deploys it under the decide server's name. It reports healthy (the chronos binary passes `cargo pmcp test check/conformance`).
**Why it happens:** `find_lambda_package_dir(&config.server.name)` checks `<project_root>/{server_name}-lambda` first, then takes the **first** workspace package with a `bootstrap` bin whose name ends in `-lambda` (cargo-pmcp src/deployment/builder.rs:333-334, 382 [VERIFIED]). `cargo metadata` lists packages **sorted** (probed from `crates/aprender-mcp-chronos-lambda`: `sorted? True`; bootstrap packages in order `aprender-mcp-chronos-lambda`, `aprender-mcp-setfit-lambda`, `aprender-setfit-train-lambda`). The chronos deploy works only because chronos sorts first. The same trap already burned the setfit training deploy (crates/.pmcp/deploy.toml.template comment). `[server] binary` is not read by the Lambda path.
**How to avoid:** Preferred fix: a small upstream cargo-pmcp change (resolve the package whose manifest dir == project_root first, or honour `[server] binary`/a package key), released and pinned before the deploy task. Checkpoint, because it is an external repo (paiml/rust-mcp-sdk). Fallback: a dedicated deploy root where `<root>/{server_name}-lambda` resolves by construction (see Open Question 1). **Either way**, the deploy recipe must assert identity afterwards: the compile log names `aprender-mcp-decide-lambda`, and a live `classify` returns the expected `artifact_sha256`.
**Warning signs:** The deploy log compiles `aprender-mcp-chronos-lambda`; `tools/list` shows `forecast`.

### Pitfall 2: Criteria order flips with `serde_json/preserve_order` feature unification
**What goes wrong:** Labels are silently permuted, e.g. `["against","favor","none"]` instead of `["none","against","favor"]`, so every probability is attributed to the wrong label.
**Why it happens:** pmcp 2.19.3 enables `serde_json` `"preserve_order"` unconditionally (pmcp-2.19.3/Cargo.toml:1347 [VERIFIED]). So any binary linking pmcp gets IndexMap, while `cargo test -p aprender-decide` alone gets BTreeMap. aprender-forecast hit exactly this (`crates/aprender-forecast/Cargo.toml:37-51`, the dev-dependency that turns `preserve_order` on in tests). setfit-apr-v1 assumes BTreeMap ordering for byte determinism (setfit-apr-v1.yaml:168).
**How to avoid:** Parse `task.json` **from the raw bytes** with a `MapAccess` visitor into `Vec<(String, Option<String>)>`. Store labels as an **array** in the manifest and response. Test under both backings (a dev-dependency feature flip, as aprender-forecast does). Probe result this session: visitor via `from_str` gave `["none","against","favor"]` with and without the feature, while `from_value` over a `Value` gave `["against","favor","none"]` without it.
**Warning signs:** The label order in `tools/list` differs from `task.json`; a test passes under `-p aprender-decide` and fails under `--workspace`.

### Pitfall 3: `/tmp` is 512 MB on pmcp.run and cannot be raised through cargo-pmcp
**What goes wrong:** The first call dies writing 0.85 GB to `/tmp` (ENOSPC), or passes locally and fails only on Lambda.
**Why it happens:** Spike 026 wrote to `/tmp/laya` with a hand-set 4 GB ephemeral volume (spike 021 README: "`/tmp` (4 GB ephemeral)"). cargo-pmcp's `[server]` supports only `memory_mb` and `timeout_seconds`; its own comment says adding ephemeral storage "is a new array entry PLUS a signature change" (cargo-pmcp src/deployment/config.rs:389 [VERIFIED]).
**How to avoid:** Download into a pre-sized `Vec<u8>` (10 GB RAM; peak ≈ 0.85 GB bytes + 1.7 GB f32 model ≈ 2.6 GB, versus 3.4–3.6 GB measured in the spike). Parse zero-copy, then drop the bytes.

### Pitfall 4: The API Gateway 30 s cap makes a count-only list bound wrong
**What goes wrong:** A legal request (inside the advertised list max) returns 504 on a cold start, or the bound is set so low it is useless.
**Why it happens:** pmcp.run fronts the function with an API Gateway HTTP API whose 30 s integration timeout "cannot be raised" (crates/.pmcp/deploy.toml.template:51 [CITED in-repo]). Measured on Graviton2: cold 11.3–12.0 s; a 94-token tweet 0.75 s; a 512-token row 5.6 s (spike 026). Graviton3 is ~1.5× faster, and which one you get is a lottery.
**How to avoid:** Two contract bounds, both checked in `precheck` after tokenization (the server holds the tokenizer): `max_texts` and `max_total_tokens`, the latter priced for a **cold start on Graviton2** with a declared margin. Add a per-text byte bound before tokenization so a 1 MiB single text cannot buy tokenizer CPU (pmcp already caps the body: `DEFAULT_MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;`, `DEFAULT_MAX_TOOL_ARGS_BYTES: usize = 1024 * 1024;`, pmcp-2.19.3/src/server/limits.rs:46,49 [VERIFIED]). Ship an **accepted-region test**: the maximal legal request completes under the cap on the live endpoint, cold (the Phase 7 lesson). Recommended values are in A8.
**Warning signs:** 504 on first call; the retry succeeds (the environment finished loading after the gateway gave up).

### Pitfall 5: Parity or gate measured against a model that is never served
**What goes wrong:** Rust-vs-torch |Δp| ~1e-3 on the fine-tuned model (looks like a port bug), or gate numbers that do not describe the served model.
**Why it happens:** Training leaves fp32 weights, while D-17 serves F16. The spike's 3.8e-6 was torch fp32 **on F16-loaded weights** (`agent.model.float()` after loading F16, oracle.py) versus Rust widening the same F16.
**How to avoid:** Save F16, reload, and compute everything from the reload. The pack step's Rust re-score of all eval rows then proves the chain end to end.

### Pitfall 6: Calibrated temperature silently clamped to 5
**What goes wrong:** The fitted T for a memorised model exceeds 5. Laya's loader (and the ported Rust `temperature`) clamps it, so the served ECE differs from the one the harness reported.
**Why it happens:** `TEMP_MIN = 0.5`, `TEMP_MAX = 5.0` (laya/common.py:376-377 [VERIFIED]), applied by `clamp_temperature` at every load.
**How to avoid:** Fit within [0.5, 5.0], record `clamp_hit`, and compute the gate ECE with the **applied** T. If the clamp is hit and ECE still fails, the gate fails. That is the fail-closed design.

### Pitfall 7: A bin in `aprender-decide` breaks FALSIFY-MONO-011
**What goes wrong:** The packer as `src/bin/*.rs` makes `aprender-decide` ship a bin and not be in either register, so `monorepo_invariants` goes red. Adding it to the register would also force `publish = false` on a crate D-14 intends to publish.
**How to avoid:** Use an `examples/pack_laya.rs`, run by `cargo run -p aprender-decide --example pack_laya --release`. CI still compiles it (`cargo build --examples --workspace --keep-going`, ci.yml:471).

### Pitfall 8: Feature-gated tests compiled out of CI
**What goes wrong:** Module tests never run in `workspace-test`.
**Why it happens:** CI documents that feature-gated setfit code "COMPILES IT OUT" of `--workspace --lib` (ci.yml:291-296).
**How to avoid:** Gate modernbert only on the **default** `parallel` feature. Keep `aprender-decide` ungated. Verify with `cargo nextest list --workspace --lib | grep modernbert` in Wave 0. Any `tests/*.rs` target needs the ci.yml:471 line (a CI edit, so check in).

### Pitfall 9: CLAUDE.md path gate and README counters
**What goes wrong:** FALSIFY-DOCS-CLAUDE-001 fails because the CLAUDE.md row cites a path that does not yet exist. FALSIFY-README-005/007 fail because the README crate count (from `cargo metadata`) and the contract count moved.
**How to avoid:** Land the CLAUDE.md edit in the same PR as, or after, the cited files. Update README's `| Workspace crates | **N** workspace crates |` row (+3 crates) and `**N** provable contracts` (+ new YAMLs) from the commands, not by hand. Each new crate needs a `README.md` containing `paiml/aprender` (FALSIFY-README-CRATE-001/002). Add the new contracts to Makefile `CONTRACTS` (Makefile:1880+) and register their equations in `contracts/aprender/binding.yaml` (`pending` → `implemented`).

### Pitfall 10: Tokenizer special-token injection is inherited, not fixed
User text containing `[SEP]`/`[CLS]` is tokenized to special ids by `tokenizers` (Laya only replaces `[MASK]`). Parity requires reproducing it, and the spike fixture's injection-probe row pins it. Marker positions come from the builder, not from scanning ids, so fake options cannot be injected. Document this as an accepted, parity-bound behaviour; do not "fix" it in Rust alone.

### Pitfall 11: `rtk` hook rewrites tool output
`git log` through the hook printed a **different short hash** (`46f88cd`) than `rtk proxy git rev-parse HEAD` (`4066d5d…`) for the same Laya checkout, and `cargo test` output is filtered (memory note). Use `rtk proxy` for any hash, status or count that ends up in a contract or a report.

## Code Examples

### Order-preserving criteria (probed this session: holds with and without `preserve_order`)
```rust
// Parse task.json FROM BYTES; never via serde_json::Value/Map.
struct OrderedCriteria(Vec<(String, Option<String>)>);
impl<'de> serde::Deserialize<'de> for OrderedCriteria {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = OrderedCriteria;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { f.write_str("criteria object") }
            fn visit_map<A: serde::de::MapAccess<'de>>(self, mut m: A) -> Result<Self::Value, A::Error> {
                let mut v = Vec::new();
                while let Some((k, d)) = m.next_entry::<String, Option<String>>()? {
                    if v.iter().any(|(kk, _)| kk == &k) { return Err(serde::de::Error::custom(format!("duplicate criterion {k:?}"))); }
                    v.push((k, d.filter(|s| !s.is_empty())));
                }
                Ok(OrderedCriteria(v))
            }
        }
        d.deserialize_map(V)
    }
}
// Task::from_slice(bytes) = serde_json::from_slice::<TaskDoc>(bytes) with #[serde(deny_unknown_fields)];
// refuse type != "choice", fewer than 2 criteria, options beyond head_max_len (spike-026 refusals).
```

### Option rendering for `choice` (spike 026 `render_options`, verbatim semantics)
```rust
// option text = "name: description" when a description is present, else "name"; the label key = name.
let opts: Vec<String> = crit.iter().map(|(k, d)| d.as_ref().map_or_else(|| k.clone(), |d| format!("{k}: {d}"))).collect();
```

### Response shape (D-11; arrays, not maps)
```json
{"model": {"artifact_sha256": "<H>", "recipe_id": "<sha256>", "method": "laya", "base": "laya-en-root@55cf4c4e"},
 "labels": ["none", "against", "favor"],
 "results": [{"label": "against", "probabilities": [0.12, 0.81, 0.07], "tokens": 94, "truncated": false}]}
```

### In-memory ranged download (replaces spike 026's `/tmp` `write_all_at`)
```rust
// Pre-size, split into disjoint 64 MB slices, fill each from its own ranged GET with retry x5.
let mut buf = vec![0u8; len];
let parts: Vec<(usize, &mut [u8])> = buf.chunks_mut(PART).enumerate().map(|(i, c)| (i * PART, c)).collect();
// Each future: GET bytes={off}-{off+c.len()-1}, check body.len()==c.len(), c.copy_from_slice(&body).
// Then: sha256(&buf) == APRENDER_DECIDE_SHA256 before any parse.
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Weights bundled in the deploy package (SetFit, Chronos-tiny) | S3 fetch at cold start for ~1 GB models | spike 021/026 (2026-09-24/25) | Needs IAM grant + in-memory loader |
| Spike trueno from the spike-016 worktree | In-tree `aprender-compute` NEON 8×6 kernel | `9d67b7247` | No worktree dependency |
| Spike 026 generic `decide` tool | Task-bound `classify` (D-09) | CONTEXT 2026-09-25 | Simpler boundary; one question per server |
| Safetensors served directly (spike 025/026) | .apr with manifest, probes, identity (D-17) | this phase | Adds pack step + load ladder |

**Deprecated/outdated:** the spike's `libm::erf` (use `erfc_precise`), hardcoded `d/64` head split, `i % 3` global-layer rule without `layer_types`, `/tmp` staging, `expect("gemm_blis")` panics in library code (return typed errors).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | torch 2.14.0 is what spike 024 ran (newest cached wheel) | Standard Stack | The lock pins a different torch than the spike; results may shift slightly. Re-run the gate anyway |
| A2 | pmcp.run in the target account accepts `memory_mb = 10240` (spike 026 used a 10 GB function; the account's Lambda memory quota is not confirmed for pmcp.run-created functions) | Pattern 9 | Deploy fails, or falls back to a smaller size (4 GB was 2.4× slower) |
| A3 | Default-Lambda init phase ~10 s cap makes EAGER_LOAD unsafe | Pattern 7 | If wrong, EAGER could shave first-call latency; lazy is still correct |
| A4 | sha256 of 0.85 GB on Graviton2 costs < 1 s (ARMv8 SHA extensions via `sha2`) | Pattern 9 | Adds seconds to cold start; measure in the timeline |
| A5 | Resolver-2 unification compiles modernbert tests in `--workspace --lib` when only `parallel` gates it | Pitfall 8 | Tests dark in CI. Wave 0 verifies with `nextest list` |
| A6 | Gate margin: `macro_f1(ft) − macro_f1(zs) ≥ 0.05` | Pattern 6 | Too strict fails a good model; too loose passes a useless one. Spike: zs macro-F1 0.354, FT r2@16 0.494–0.513 |
| A7 | Gate ECE ceiling 0.10 (15-bin top-label) after calibration, T ∈ [0.5, 5] | Pattern 6 | Unmeasured post-calibration ECE; the demo may fail the gate (spike FT ECE 0.12–0.43 before calibration; zs 0.255) |
| A8 | `classify` bounds: `max_texts = 8`, `max_total_tokens = 1024`, `max_text_bytes = 16384`. Derivation: (30 s − 12 s cold − ~1 s probes − 4 s margin) / (5.6 s / 512 tok ≈ 11 ms/tok on G2) ≈ 1180 → 1024 | Pitfall 4 | Must be proven by the live accepted-region test; pmcp.run's own proxy overhead is unmeasured |
| A9 | Calibration slice 25 % stratified (≥ 2 per class) of the shots | Pattern 5 | Fewer training shots than the spike's cells (12 vs 16/class at "16 shots"); expect slightly lower F1 than the spike baselines |
| A10 | pmcp.run functions run in the ze-kasher-dev account (the `pmcp-train-grant` recipe finds them with that profile), so a put-role-policy grant is possible | Pattern 9 | If pmcp.run owns the account, S3 must be granted differently (bucket policy on the pmcp.run role ARN) |
| A11 | The house `erfc_precise` matches `libm::erf` closely enough to keep probs ≤ 1e-5 | Standard Stack | If not, parity breaks; the CI tiny fixture catches it on day one |

## Open Questions

1. **How is the cargo-pmcp package-resolution trap closed?**
   - What we know: the fallback picks the alphabetically-first bootstrap `*-lambda` package (verified); `[server] binary` is ignored.
   - What's unclear: whether an upstream fix (paiml/rust-mcp-sdk) can ship and be pinned inside this phase.
   - Recommendation: plan an upstream PR as a checkpointed task, and pin `cargo pmcp --version` in the recipe. Fallback deploy root: nest the lambda crate so `<root>/{server}-lambda` resolves, which constrains the server name, so it is inferior. Never deploy without the post-deploy identity probe.
2. **Where do pmcp.run functions live (account) and can their role be granted S3?** (A10) Confirm at the human deploy checkpoint.
3. **Is `train.jsonl` `label` the criterion name or the index?** D-05 does not say. Name is safer (an index silently re-means under a criteria reorder). Recommend name, refuse unknown names, and confirm with the user.
4. **16 or 64 shots for the demo?** 16 trains in ~80 s at r2; 64 in ~315 s with higher variance (0.545–0.679). Recommend 16 shots/class at 12 epochs, the declared seed 13 (the spike's first seed), and `--seeds 3` for the variance report. The user may prefer 64 for the headline number.
5. **Should the stdio server accept a local .apr path only, or also S3?** Recommend a path only (the stdio template), with S3 confined to the Lambda crate.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| rustc (pinned) | all Rust | ✓ | 1.93.0 | — |
| just | recipes | ✓ | 1.46.0 | — |
| uv / Python | training harness | ✓ | uv 0.9.5 / Python 3.13.7 | — |
| torch MPS | D-03 fast path | ✓ | local torch 2.12.1 reports `mps True` | CUDA (untested) / CPU (flagged) |
| HF cache: laya @ 55cf4c4e | train, parity | ✓ | model.safetensors sha256 891102d3… | download with pinned revision + `expected_sha256` |
| spike-025 fixture | gated parity test | ✓ | `.planning/spikes/025-laya-rust-forward-parity/fixtures/laya-en_fixture.json` (keys: model, max_len 512, head_max_len 192, cls 50281, sep 50282, mask 50284, ladder, records) | ladder `.bin` regenerates via oracle |
| cargo-zigbuild + zig | arm64 bootstrap | ✓ | 0.23.4 / 0.15.2 | — |
| cargo-pmcp | pmcp.run deploy | ✓ | 0.24.3 (latest on crates.io) | — (see Pitfall 1) |
| aws-cli + profile | S3 upload, IAM grant | ✓ | 2.29.0, identity resolves (user/Guy) | boto3 |
| node/npx (CDK) | bucket stack | ✓ | node 24.19.0 | aws-cli bucket creation |
| cargo-nextest | CI-equivalent runs | ✓ | 0.9.102 | cargo test |
| `pv` | contract validation | ✗ on PATH | — | `cargo run --release -p aprender-contracts-cli --bin pv --` (Makefile `PV_BIN`) |
| pmat | code search / gates | ✓ | 3.15.0 (CLAUDE.md sample said 3.30.0 — re-derive) | — |

**Missing dependencies with no fallback:** none.
**Missing with fallback:** `pv` (use the cargo-run form).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust libtest / cargo-nextest 0.9.102; trybuild for the private-constructor proof; pytest-free Python (the harness self-checks and exits non-zero) |
| Config file | none new; the CI profile is `--profile ci` |
| Quick run command | `cargo test -p aprender-decide --lib && cargo test -p aprender-core --lib models::modernbert` |
| Full suite command | `cargo nextest run --profile ci --workspace --lib --exclude aprender-gpu --exclude aprender-cuda-edge --exclude aprender-compute` + `cargo test -p aprender-core --test monorepo_invariants --test readme_contract` |

### Phase Requirements → Test Map
| Req | Behavior | Test Type | Automated Command | File Exists? |
|-----|----------|-----------|-------------------|-------------|
| D-13 | Tiny-config ModernBERT+Laya matches a transformers/Laya fp32 fixture (ids, every layer, logits, probs ≤ 1e-5) | unit (CI) | `cargo test -p aprender-decide --lib laya::parity::tiny` | ❌ Wave 0: `scripts/laya_train/fixtures.py` generates `crates/aprender-decide/tests/fixtures/laya_tiny.json` (d=128, 3 layers, local_attention 8, seq ≥ 20 so the window bites) |
| D-13 | Window mutation: `|i−j| ≤ w−1` or `w+1` breaks the first local layer only | unit (CI) | `cargo test -p aprender-core --lib models::modernbert::window_mutation` | ❌ |
| D-13 | `layer_types` vs `i % n` agreement; prefix-aware loader refuses a missing tensor by name | unit | `cargo test -p aprender-core --lib models::modernbert::load` | ❌ |
| D-17 | Full-model parity vs spike-025 fixture: ids 14/14, probs ≤ 1e-5, argmax 14/14 | integration (env-gated `LAYA_MODEL_DIR`, SKIP + return, never `#[ignore]`) | `LAYA_MODEL_DIR=... cargo test -p aprender-decide --release --test laya_parity` | ❌ |
| D-17 | .apr byte determinism across processes; closure `pack(unpack(x)) == x` | unit | `cargo test -p aprender-decide --lib artifact::determinism` | ❌ |
| D-17 | Load ladder: each rung refuses its induced negative (oversize, CRC, wrong tag, unknown key, missing tensor, tokenizer hash, NaN, probe mismatch) | unit | `cargo test -p aprender-decide --lib artifact::ladder` | ❌ |
| D-17 | Private constructor unreachable | trybuild | `cargo test -p aprender-decide --test ui` | ❌ |
| D-05 | Criteria order survives both `preserve_order` backings; duplicate/unknown/short refusals | unit (+ dev-dep feature flip) | `cargo test -p aprender-decide --lib task` | ❌ |
| D-10/D-12 | Bounds refuse at N+1 and accept the maximal legal request; truncation flag on the 512-token row | unit | `cargo test -p aprender-mcp-decide --lib` | ❌ |
| D-09/D-11 | stdio E2E: `tools/list` has one `classify` with labels from the artifact; call returns identity + ordered probs | integration (env-gated `APR_MCP_E2E_DECIDE_MODEL`) | `cargo test -p aprender-mcp-decide --test e2e_stdio` | ❌ |
| D-15 | Both server packages in `deployment_unit_bins`, `publish = false` | existing gate | `cargo test -p aprender-core --test monorepo_invariants` | ✅ (edit list) |
| D-16 | CLAUDE.md cited paths exist; README counts | existing gate | `cargo test -p aprender-core --test readme_contract` | ✅ |
| D-07/D-08 | Gate refuses a failing report; deploy refuses a mismatched threshold/sha | recipe self-test | `just laya-gate-selftest` (runs the gate on a fabricated failing report, expects rc≠0) | ❌ |
| D-01..D-04 | Training run records the device used, recipe before scores, F16 reload | manual + report assertions | `just laya-train data/decide/tweet-stance-16` | ❌ (manual, laptop GPU minutes) |
| D-18 | Live cold call completes with identity == H; maximal legal request under 30 s cold | manual checkpoint | `just laya-deploy-verify` | ❌ (human checkpoint) |

Contracts: `$(PV_BIN) validate contracts/decide-apr-v1.yaml contracts/laya-parity-v1.yaml contracts/decide-tool-boundary-v1.yaml` and `$(PV_BIN) audit ... --binding contracts/aprender/binding.yaml`. Tolerances are read from YAML at test time, never as literals (chronos-bolt-parity D-15 pattern).

Recommended parity tolerances to declare **before** running:
- ids / marker positions: EXACT.
- Probabilities: ≤ 1e-5 abs (D-17).
- Logits: ≤ 1e-4 abs.
- Final-norm output: ≤ 1e-4 abs.
- Embeddings: ≤ 1e-5.
- Per-layer ladder: ≤ 1e-3 × ref rms (relative). Layer 27 was 5.7e-2 abs on rms 128, i.e. 4.4e-4 rel.
- argmax: EXACT.

### Sampling Rate
- **Per task commit:** the quick run command.
- **Per wave merge:** full suite + `make contract-validate` (with the new contracts added to `CONTRACTS`).
- **Phase gate:** full suite green, the gated full-model parity run locally with its output pasted, `just laya-pack` re-score passing, the live deploy checkpoint.

### Wave 0 Gaps
- [ ] `scripts/laya_train/` uv project (pyproject, `.python-version` 3.13.7, `uv.lock`), after the human-verify of the SUS PyPI packages.
- [ ] Tiny fixture generator + committed `laya_tiny.json` (+ tokenizer: the real 3.4 MB `tokenizer.json` or a trimmed test vocab; check the size budget).
- [ ] Copy of the spike-025 fixture into `crates/aprender-decide/tests/fixtures/` (936 KB) for the gated test.
- [ ] Three contract YAMLs with thresholds, bounds and tolerances, committed before any run is read.
- [ ] `nextest list` proof that the modernbert/decide tests are compiled into the CI lib run (A5).
- [ ] Decision on Pitfall 1 (the upstream cargo-pmcp fix) before the deploy wave.

## Security Domain

Security enforcement is enabled (ASVS L1, block on high).

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V1 Architecture | yes | Thin server, transport-owned bounds, library door; one model per server |
| V2 Authentication | yes (endpoint) | pmcp.run `[auth]` (OAuth/DCR). The chronos precedent ships `enabled = false`. For a 10 GB Lambda an open endpoint is a cost-amplification vector, so recommend enabling auth or recording an explicit risk acceptance at the deploy checkpoint |
| V3 Session | no | Stateless (`StreamableHttpServerConfig::stateless()`) |
| V4 Access Control | yes (cloud) | IAM `s3:GetObject` on one prefix; no write; grant discovered-role-scoped |
| V5 Input Validation | yes | `deny_unknown_fields`; `max_texts`, `max_total_tokens`, `max_text_bytes`; pmcp 4 MiB body / 1 MiB args caps; refusals as `pmcp::Error::validation` naming the bound |
| V6 Cryptography | yes (integrity only) | sha256 (`sha2`) of artifact and inputs; no custom crypto |
| V7 Errors/Logging | yes | Never log input texts (tweets can be personal data); log timings and counts only; typed errors, no panics in the library (`expect` only on invariants) |
| V10 Malicious code / supply chain | yes | Pinned uv.lock hashes, pinned Laya git SHA, `expected_sha256` on the HF download, no `cargo update` of aws-sdk |
| V12 Files/Resources | yes | Bounded read before allocation; artifact size cap; zero-copy parse; sha256 pin before parse |
| V14 Configuration | yes | deploy.toml generated from a tracked template and gitignored (account id); no secrets in repo; weights never committed (`/models/`) |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Tampered artifact in S3 | Tampering | `APRENDER_DECIDE_SHA256` pinned in deploy config, checked before parse; content-addressed keys |
| Oversized/hostile artifact | DoS | Bounded read, declared-length refusal, header/index structural checks before allocation |
| Cost/DoS via large batches or long texts | DoS | Token budget + count + byte bounds priced to the 30 s envelope; auth on the endpoint |
| Special-token injection (`[SEP]`, `[MASK]`) | Tampering | `[MASK]` replaced (Laya parity); markers from the builder; the injection probe row pins the behaviour |
| Wrong binary deployed under the server name | Spoofing | Post-deploy identity probe (`artifact_sha256`, labels in the description); cargo-pmcp resolution fix |
| Label permutation via map ordering | Tampering (integrity of output) | Ordered arrays only; dual-backing tests |
| Unverified model served | Spoofing | Private constructor behind the ladder + trybuild |
| PII in logs | Info disclosure | No text logging; CloudWatch retention 30 days (chronos deploy.toml default) |

## Sources

### Primary (HIGH confidence)
- In-repo, read this session: `crates/aprender-mcp-setfit/src/{lib,main}.rs`, `crates/aprender-mcp-chronos-lambda/{Cargo.toml,src/*,.pmcp/deploy.toml}`, `contracts/setfit-apr-v1.yaml` (§storage map, size bounds, bounded read, load ladder), `contracts/forecast-tool-boundary-v1.yaml` (constants mirror), `contracts/chronos-bolt-parity-v1.yaml`, `crates/aprender-core/tests/{monorepo_invariants,readme_contract}.rs`, `.github/workflows/ci.yml:289-471`, `crates/apr-format/src/v2/*`, `crates/aprender-core/src/format/dequant_ext.rs`, `crates/aprender-core/Cargo.toml`, `justfile`, `crates/.pmcp/deploy.toml.template`, `Makefile:1878-2200`.
- Spike sources: `.claude/skills/spike-findings-aprender/{SKILL.md, references/laya-decision-model.md, laya-rust-inference.md, aws-mcp-model-hosting.md, sources/024/025/026}`.
- Laya @ 4066d5d: `laya/agent.py` (load path, device fallback, `_fix_tokenizer_config`), `laya/common.py` (`build_model`, `DecisionModel`, `temp_bucket`, `clamp_temperature`).
- cargo-pmcp source (`~/Development/mcp/sdk/rust-mcp-sdk/cargo-pmcp`, 0.24.3): `deployment/builder.rs:333-400`, `deployment/naming.rs:68-93`, `deployment/config.rs:389`, `targets/pmcp_run/deploy.rs:1190-1260`.
- pmcp 2.19.3 registry source: `Cargo.toml:1343-1348`, `src/server/limits.rs:35-49`.
- Probes run: serde_json order probe (scratchpad crate, both features); `cargo metadata` package order; HF cache sha256/blob names; PyPI version index.

### Secondary (MEDIUM confidence)
- Spike-measured latencies and cold starts (single-account, single-day measurements).

### Tertiary (LOW confidence)
- The Lambda init-phase cap, sha256 throughput on Graviton2, and the torch version the spike resolved (Assumptions A1, A3, A4).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH. Every Rust crate is locked and in-repo; the Python pins are verified except torch.
- Architecture: HIGH. It follows three in-tree precedents (setfit artifact/ladder, chronos lambda, forecast boundary).
- Pitfalls: HIGH for 1–3, 5–9 and 11 (probed or source-read); MEDIUM for 4 (the envelope comes from the spike, not from pmcp.run).
- Gate/bound values: MEDIUM-LOW. They are declared defaults that need the demo run and the live accepted-region test.

**Research date:** 2026-09-25
**Valid until:** 2026-10-25 (cargo-pmcp and pmcp move fast; re-check Pitfall 1 against the current cargo-pmcp before the deploy wave)
