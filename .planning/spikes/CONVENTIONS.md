# Spike Conventions

Patterns and stack choices established across spike sessions. New spikes follow these unless the question requires otherwise.

## Stack
- **Rust, standalone crate per spike** under `.planning/spikes/NNN-name/` with an empty `[workspace]` table and
  `aprender = { path = "../../../crates/aprender-core", package = "aprender-core", default-features = false }`.
  Copy the workspace `Cargo.lock` in so versions match. Build with `CARGO_TARGET_DIR=../../../target` so
  all spikes share one compiled `aprender-core` (23 s the first time, cached after). `.planning/spikes/.gitignore` ignores `target/`.
- **Python oracles via `uv run --with …`**, never a project venv: `prophet==1.4.0` (`--with prophet --with pandas`),
  `neuralprophet==0.9.0` (`--with neuralprophet --with "pandas<3" --with "numpy<2.3" --with "torch<2.6"`).
  `chronos-forecasting==2.3.1` (`--with chronos-forecasting --with pandas --with safetensors`; pulls torch + transformers).
  Oracle scripts live in the spike's `tools/`, their outputs in `fixtures/*.json`, committed. Model weights are NOT
  committed: `models/` is gitignored; the oracle downloads them and the README says where to copy from.

## Structure
- `src/main.rs` driver prints Markdown (tables) to stdout → saved as `RUN-OUTPUT.md`; `results.json` for numbers;
  `report*.html` self-contained SVG pages (no JS libraries) for what the user should *see*.
- Multi-fixture drivers loop over `fixtures/*.json`; the first fixture gets the deep probes.
- MCP spikes: `src/lib.rs` (tool + `http_app`), `src/main.rs` (`--stdio` default, `--http PORT`, `--bench`),
  `static/index.html` (a real MCP client), `tests/e2e.rs` (in-process streamable-HTTP).
- The `rtk` hook filters `cargo test` / `cargo clippy` output; run the `target/release/deps/<test>-*` binary directly
  for `println!` lines, and `rtk proxy cargo clippy …` when the raw warning list matters.
- Zero-shot model spikes: `models/<name>` is a symlink to the HF snapshot or a sibling spike's weights (gitignored);
  `tools/oracle.py` dumps a ladder (scaling → features → embeddings → hidden rows → quantiles, model AND pipeline
  outputs, timings) plus edge probes into one `fixtures/*_fixture.json`; the driver prints one ladder table.
- **Wrapped findings live in `.claude/skills/spike-findings-aprender/`** (`/gsd-spike --wrap-up`, 2026-09-05
  for 001–010, 2026-09-20 for 011–014, 2026-09-23 for 015–020, 2026-09-25 for 021–026): one reference per feature area plus `sources/NNN-*/` (README, Cargo.toml,
  build.rs, src/, tools/, tests/, static/, RUN-OUTPUT, results, baseline.json, PR.md). Fixtures, `models/`,
  `report*.html`, `*.log` and any oversized chart-feeding `results.json` (011's is 256 KB) are NOT copied — cite
  `.planning/spikes/NNN-*/` from the reference instead. SKILL.md groups Requirements **by idea key** once more
  than one idea has been wrapped. Re-run the wrap-up after new spikes; the skill's `processed_spikes` list is
  the filter.

## Patterns
- **Parity ladder before optimiser claims:** data prep (0 diff) → objective at the oracle's MAP (1e-12) → finite-difference
  gradient at a *perturbed* point → fit → forecast diff vs oracle → **control**: how much does the oracle disagree with
  itself (Newton vs L-BFGS, another seed / lr)? A diff inside that band is parity.
- **Datasets are sorted by `ds` and de-duplicated at load, in BOTH the Rust driver and the Python oracle** (`wp_log_R.csv` is not chronological; spike 006's cross-check caught the mismatch).
- **Accuracy claims use rolling origins and MASE** (period 7 daily / 12 monthly), with naive + seasonal-naive rows and a horizon slice; a single split flatters every model.
- **Never select by test error**: learning rates and epochs are selected by *train* loss; test MAE is only reported.
- **Prophet fit config:** exact L1, objective ÷ T, non-finite guard, `LbfgsF64::new(2000, 1e-7, 20)`, restart from
  the stall point until improvement < 1e-6 relative (≤ 8 rounds), cache `value_and_grad` by x, wall-clock budget.
  `Stalled` is the success status.
- **NeuralProphet training:** `clear_graph()` after every step, Huber from ops with a constant mask (core `SmoothL1Loss`
  is detached), lr sweep {0.01, 0.03, 0.1} lag-free / {0.03, 0.1} with lags, ~4× NP's auto epochs for linear AR.
- Dates are `i64` days since epoch via civil-date arithmetic (`days_from_civil`/`civil_from_days`); no `chrono`.
- **Model ports:** dump a ladder of intermediate tensors from the oracle (scaling, embeddings, hidden states) and
  compare bottom-up before comparing outputs; add edge probes (NaN, short, constant, huge scale) as a second fixture.
- **Hot loops:** write dot products with 8 independent accumulators (LLVM will not vectorise a strict-order float
  reduction). Since spike 008 `blis::gemm_blis` has a NEON 8×6 microkernel on aarch64 (65–77 GFLOP/s, 0.7× faer):
  route every multi-row product through it (projections, feed-forward, patch embeddings, attention scores/context);
  keep `dot8` for single rows. The rayon `blis::gemm` (feature `parallel`) splits M only in 128-row blocks — expect
  ≤ 1.4× on transformer shapes below ~500 tokens.
- **Kernel or core changes are parity work:** measure before/after with the *maintainer's own instrument*
  (`benches/gemm_comparison.rs` for GEMM) on the same machine, run the control on untouched `upstream/main` in a
  git worktree (`git worktree add ../aprender-<topic> -b <branch> upstream/main`), diff raw clippy output between the
  trees, and ship a `contracts/*.yaml` with falsification tests. A flaky test is reported with its solo-run
  failure rate on both trees, never attributed to the change.
- **Model ports keep only the transposed `[in, out]` weights** (the plain-loop A/B copies double memory: Chronos-2
  peaked at 1.45 GB). `safetensors.rs` from spike 007 decodes F32/F16/BF16 from a byte slice, so embedded and
  on-disk weights share one loader; write f16 copies with `tools/to_f16.py` and *measure* the accuracy cost per
  model (0.1 % of scale for Bolt-tiny, 0.3–0.6 % for Chronos-2, 3 % on five-point series).
- **Embedded weights:** `build.rs` stages `model.safetensors` + `config.json` from a build-time env var
  (`CHRONOS_EMBED_DIR`) into `OUT_DIR` for `include_bytes!`, empty markers when unset, and the runtime falls back
  to the same-named `*_MODEL_DIR` path — the `aprender-mcp-setfit-lambda` pattern.
- **Concurrency probes carry a throughput row.** pmcp's streamable-HTTP router holds one `Arc<Mutex<Server>>`
  across each tool call, so a single router serialises fits; a pool of K routers behind a round-robin `fallback`
  handler (spike 010) restores parallelism with bit-identical outputs.
- **Tool boundary refuses, never defaults** (004, 007): `#[serde(deny_unknown_fields)]`, `MIN_POINTS`/`MAX_POINTS`/
  `MAX_HORIZON` consts, `ds` strictly ascending, unique, real calendar dates (`parse_date` round-trips through
  `civil_from_days`), `ds`/`y` same length, unknown `freq` refused, constant `y` refused (Prophet diverges), `y`
  nullable only for zero-shot models. Refusals are `pmcp::Error::validation`; each one has an e2e case.
- **Zero-shot horizon gating** (006 → 007): accept the model's native horizon by default; beyond it require an
  explicit `allow_long_horizon: true` and put a `warning` in the response that cites the measured degradation.
- **Determinism under load** (010): a `seed` argument (default 42) drives every RNG per request; the release check
  is the JSON signature of `ds/yhat/bands/trend/components` for a request run alone vs under 8–16 concurrent
  requests — must be identical, and the wall-time row must show parallelism (else the transport is serialising).

- **Compose with the shipped crate; never fork it into the spike** (011, 013, 014). Build the
  prototype against the crate's public API by path dependency, so "does this need restructuring?"
  is answered by whether the spike *compiles* rather than by reading. `aprender-forecast` exposes
  `Design` (all fields `pub`), `columns`, `feature_row`, `holiday_day_sets`, `make_design`,
  `Model`, `fit_prophet`, `NpModel::forward`, `rows_for`, `weighted_huber`, `one_cycle_lr` and
  `train_cost` — enough to splice new design columns or add a `Linear(E,1)` block beside the model
  and hand both parameter sets to one `AdamW`.
- **Check a change request's premises against its own pinned commit BEFORE decomposing** (011).
  `git show <tag>:path`, not HEAD — the consumer is on the tag. Four premises were checked and
  three were refuted (a claimed silent-ignore that the door already refuses; a "no fixture covers
  this" that a committed fixture does cover; an entry point that does not exist), which re-priced
  the request before a line was written.
- **Every identity or invariance claim ships a falsification probe in the same table** (012, 013,
  014): a 1-ULP mutation the signature must detect, a different seed that must differ, a garbage
  fill value that must change something somewhere. A green "bit-identical" column with no probe
  beside it is indistinguishable from a broken harness.
- **Compare f64 by `to_bits()`, never by a formatted decimal** (012, 014); normalise `-0.0` to
  `0.0` and let every other bit pattern be significant. A printed comparison silently accepts any
  change below the print precision.
- **Two optimisers disagreeing on a parameter is not a defect until the OBJECTIVE says so** (011).
  Evaluate the model's own objective at both parameter vectors and report the slack against the
  contract bar; then explain the disagreement (collinearity with the trend or Fourier basis) rather
  than tuning it away.
- **Sweep a cost or bound claim to the DOOR'S OWN CEILING, not to a typical value** (013). Events
  cost 1.3x at 6 columns and 7.6x at `MAX_HOLIDAY_COLUMNS` (1000); only the second number decides
  whether the budget promise survives. Report the shape (linear in E) as transferable and the
  constant as machine-specific.

**Prophet 1.4.0 oracle facts** (011), so they are not rediscovered:
- `make_all_seasonality_features(df)` returns `(features, prior_scales, component_cols, modes)`.
  `s_a` / `s_m` are DERIVED in `fit()` from `component_cols`, not returned.
- Column order is **seasonalities -> holidays (sorted by the generated `{name}_delim_{±off}`
  string, so `+0, +1, +2, -1`) -> extra regressors in INSERTION order**. Regressors append; they
  never reorder an existing column, which is what makes no-argument invariance free.
- `add_regressor` standardisation: `mu` = mean over HISTORY rows only; `std` = pandas
  `Series.std()`, i.e. **ddof = 1**, not numpy's default 0; `standardize="auto"` leaves a column
  whose unique values are exactly `{0, 1}` at `mu=0, std=1`.
- The fixture CSVs under `crates/aprender-forecast/tests/fixtures/` **quote their fields**
  (`"2007-12-10"`); strip quotes before parsing or the date validator refuses the row.

## Tools & Libraries
- `pmcp = { version = "2.19", features = ["streamable-http", "schema-generation"] }`, `schemars = "1.0"`, `axum = "0.8"`,
  `reqwest = "0.12"` (dev) — all already in the workspace lock.
- `half = "2.7"` (F16/BF16 decode), `tower = "0.5"` (router pool `oneshot`), `trueno` feature `parallel` only to
  measure; `uv run --python 3.12 --with huggingface_hub` for weight downloads.
- Avoid: `aprender::nn::loss::SmoothL1Loss` (no gradient); `FISTA` for f64 problems (f32, fixed step); relying on
  `Instant::now()` deltas being non-zero on Apple Silicon (41.67 ns ticks).

## Decision-model spikes (idea `llm-decision-classifier`, 015–020)

- **Python oracles that ship their own lockfile run inside their vendored checkout**, not via `uv run --with`:
  Kev (`jaredpalmer/kev` @ `7405b72`) lives at `015-kev-vs-setfit-few-shot/vendor/kev` (gitignored) and every
  Kev-side tool is `cd vendor/kev && uv run python ../../tools/<x>.py`. Its torch fp32 CPU path is the parity oracle;
  its MLX bf16 path is only for fast feature extraction.
- **Compare to SetFit on PAIRED rows**: replay the committed `benchmarks/tweeteval-stance/selections/sK-seedS/`
  manifests (`train:N` = HF row N, verified 587/587 + 280/280) and score with the benchmark's own metric (F_avg).
  A task `apr setfit` cannot ingest uses Python `setfit` with `train-config.json`'s knobs, flagged as a stand-in.
- **Every few-shot recipe is declared before a test score is read**; when the author's recipe fits a different data
  regime (Kev's README recipe = 12 optimizer steps at 16 shots) declare a second one up front and report both.
- **Upstream-dependent Rust spikes build against a local merge worktree**: `../aprender-016-upstream-sync` on branch
  `spike/016-upstream-sync` (never pushed), spike crates path-depend on its crates and share
  `CARGO_TARGET_DIR=<repo>/target`. Every change made there is also saved as `<spike>/upstream-*.patch`.
- **Model handoff chain** (017): PEFT `merge_and_unload()` in fp32 → `Qwen3_5ForCausalLM.save_pretrained` →
  llama.cpp `convert_hf_to_gguf.py --outtype f32` (shallow clone in `vendor/llama.cpp`, run with
  `uv run --python 3.12 --with ./gguf-py --with "transformers>=5" --with torch …`) → upstream `Qwen35Model`.
  Oracle fixtures carry token ids, readout offsets and hidden states, so the tokenizer is a separate rung.
- **When a sampling profile says "idle", time the phases**: `sample` blamed thread caps (61 % `__psynch_cvwait`);
  thread-local phase timers showed a single-threaded loop was the real limit (020). Serial work shows up as other
  threads waiting, not as its own hot symbol.
- **Lambda proxy** (019): fresh process per run, `RAYON_NUM_THREADS=6`, an RSS-per-step timeline via `ps -o rss=`,
  peak from `/usr/bin/time -l`, steady state as p50/p95 over ≥ 5 rounds. State the two things it cannot measure
  (Graviton vs local CPU, cold container-image reads) as estimates with their arithmetic.
- **Two deployment targets for AWS (user direction, 2026-09-23):** small Rust MCP servers (SetFit, Chronos-Bolt,
  forecast) stay on default Lambda, where they scale to zero. Qwen-sized models target **Lambda Managed Instances**:
  32 GB / 16 vCPU, arm64 / Graviton4, no per-request cold start, concurrent requests in one process via
  `run_concurrent`. A deployment spike states which one it proxies. The 019 proxy (6 threads, 10 GB) is the
  default-Lambda envelope; an LMI proxy is `RAYON_NUM_THREADS` = the function's vCPU with a 32 GB cap.
- **Separate documented facts from projections in deployment findings**: cite the AWS doc URL and the date read for
  every platform limit, and mark every latency on hardware we have not run as *est.* with its arithmetic.
- **zsh traps hit here**: `"$r:crates/…"` applies the `:c` modifier (write `"${r}:crates/…"`); unquoted `$VAR` holding
  a list is ONE word (`${=VAR}` or a `while read` loop); `/bin/bash` is 3.2 (use `#!/opt/homebrew/bin/bash` for
  associative arrays).

## Cloud deployment spikes (idea `mcp-model-hosting-aws`, 021–023)

- **One crate, every host.** A model-server spike builds one library (the pmcp tool + load timeline) with two bins:
  `bootstrap` (Lambda/LMI loopback shim, the `aprender-mcp-chronos-lambda` pattern) and `server` (container: load,
  THEN bind :8080). The difference between hosts is then the host, not the code.
- **Every response carries its own forensics**: load timeline (ms since process start + RSS per step), host facts
  (`/proc/cpuinfo` CPU part → Graviton generation, `available_parallelism`, rayon threads, cgroup/memory), parity
  vs the oracle, `first_call_in_process`. Client-side wall time is measured around the whole call; Lambda's REPORT
  tail (`LogType=Tail`, default Lambda only) adds Init Duration and Max Memory Used.
- **Cross-build for Graviton on the laptop**: `cargo zigbuild --release --target aarch64-unknown-linux-gnu.2.34`
  (glibc floor 2.34 for `provided.al2023`; check with `strings bin | grep -oE 'GLIBC_2\.[0-9]+'`). AWS SDK crates
  newer than the pinned rustc: `CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback cargo update`.
- **Never move multi-GB artifacts over the laptop uplink twice** (it is ~3.7 MB/s here; default `aws s3 cp`
  dropped connections — use `max_concurrent_requests = 2`, `multipart_chunksize = 16MB` in a TEMP
  `AWS_CONFIG_FILE`, never the user's config). Upload weights to S3 once; build images **inside AWS**
  (CodeBuild ARM_CONTAINER, privileged, source zip from S3; `--provenance=false` locally for Lambda images).
- **Weights live in S3 and are pulled with parallel ranged GETs + per-part retry** (16 x 64 MB), never baked into
  an image: measured 7–18x worse on Lambda and 4x worse on Fargate.
- **AWS resources**: us-east-1, tag `spike=<range>`, least-privilege roles scoped to the spike bucket/repo, SG
  ingress from the one client IP (`checkip.amazonaws.com`), no public Lambda URL (invoke with an API-GW v2 event),
  and a `tools/teardown.sh`. Anything that bills while idle (LMI) is torn down in-session; LMI hosts are only
  visible with `DescribeInstances(IncludeManagedResources=True)`.
- **Vary one input before naming a cap**: the Lambda S3 ceiling was believed only after 16 x 64 MB and 32 x 32 MB
  both landed at 72–80 MB/s; and a changed row index invalidates a latency comparison (021 run 2 did both — rerun).

## Alternative-base spikes (024–026: Laya vs Kev)

- **A new base model is compared on the previous spike's exact rows, seeds and metric code**: copy the helpers
  VERBATIM into `tools/fewshot015.py`-style modules (015's script ran at import time, so it could not be imported),
  replay the committed SetFit selections, and fill the same table columns.
- **Capture frozen features with forward hooks on the model's OWN inference path** (`Agent.predict`), then prove the
  hooks with a recompute check (024: probabilities from hooked logits match the API to its 4-dp rounding).
- **When a frozen-feature adapter does not move (std 0 across seeds), diagnose on TRAIN data before reporting**:
  train accuracy after fitting at lr 1e-4/1e-3/1e-2. If the head cannot fit the shots, the recipe is structurally
  wrong for that architecture (Laya's shared marker scorer), and a new adaptation must be pre-declared before its
  first test score — as 015's LoRA r2 and 024's full-FT r1/r2 were.
- **torch CPU timings on Apple Silicon are AMX (Accelerate), not a CPU baseline**: ~1 TFLOP/s on 6 threads. Compare a
  Rust port against Rust, and project to Graviton from a Rust number (Graviton2 ≈ 3.2–3.3x slower than an M4 at 6
  threads, measured twice: Kev 021, Laya 026).
- **Every numeric identity ships a mutation that must break it** (025: local window 63/65 must break layer 1 only).
- **Serverless sizing**: Lambda bills GB-seconds and scales vCPU with memory, so for a CPU-bound model the
  largest function (10,240 MB) costs the same per decision and is fastest (026: 4 GB = same GB-s, 2.4x slower).
