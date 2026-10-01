# Phase 8: Laya Decision Model: Local Fine-Tune and Thin MCP Server - Context

**Gathered:** 2026-09-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Productise spikes 024–026 as the first member of a **generalized decision-model family**:

1. A user fine-tunes Laya (ModernBERT-large + 2-layer head + shared marker scorer) **locally** on their own
   labelled shots, calibrates it, and gets a gate report.
2. The checkpoint is converted to **.apr** and served in Rust through a thin, task-bound `classify` MCP server.
3. One trained model (TweetEval stance) is **deployed live on pmcp.run** (default Lambda) and verified by a real call.

In-tree, linted, contracted and CI-tested. ModernBERT becomes a reusable aprender-core model, and the decision
layer is a method-neutral crate that Kev/Jev can join later — but only Laya is implemented in this phase.

Not in this phase: a training MCP server, Kev/Jev ports, multilingual or `typed-decisions` bases, a Rust trainer.

</domain>

<decisions>
## Implementation Decisions

### Local training harness
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

### Dataset in, quality gate out
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
  — **Amended 2026-09-27 (user, option 1 from spikes 027/028):** the gate's eval set is in-distribution held-out data defined by rule, and the SemEval test split is a reported shift probe, never a gate clause (A2). The gate certifies calibration on data like the shots, NOT robustness to shifted input; this set was chosen after the test split failed the gate twice (spike 027 measured the SemEval train-to-test shift as the cause). Thresholds unchanged (ECE <= 0.10, margin >= 0.05, T in [0.5, 5.0]). See laya-finetune-gate-v1 1.4.0 (published 2.0.0) `eval_set`.
- **D-08:** **One declared seed by default; `--seeds N` produces a variance report** (mean ± sd). The report states
  "single seed" plainly when N = 1 (spike 024 saw 0.545–0.679 across seeds at 64 shots). The shipped model is always
  the declared seed — **never the best seed on eval**.
  — **Amended 2026-09-27 (user, option 1 from spikes 027/028):** production runs evaluate the gate over seeds 13/17/23 and ship the median-ECE seed (rank floor(ece_post x 10000), ties to the smaller seed); the gate passes only if that seed passes both clauses, and all three runs are reported. This supersedes "the shipped model is always the declared seed" for production runs (a legacy run without `seed_selection` keeps the declared-seed rule and is never deploy-eligible). The median is selected WITH eval labels and the contract says so: median, not best (A3). See laya-finetune-gate-v1 1.4.0 (published 2.0.0) `seed_policy`.

### MCP tool surface
- **D-09:** The predict server exposes a **task-bound `classify`** tool: the question and labels come from the
  artifact's `task.json`; the caller sends only text. No generic `decide`/`/v1/systemone` surface — a fine-tuned
  model answers one question. One trained model per deployed server (thin-server rule).
  — **Reversibility:** costly — this is the published tool contract agents integrate against.
  — **Amended 2026-09-28 (user, gap round 08-28):** truncation sentence A-derive; refusal wire shape B-iserror; takes effect at the next deploy (plan 08-30).
  A-derive: the served description's truncation sentence is derived from the loaded artifact's
  `manifest().agent.max_len` against the tier's `classify_max_total_tokens`. When a full-window row fits the
  budget the sentence stays "Long texts are truncated by the model itself to its window, and each such result
  reports `truncated: true`."; otherwise it says a text whose built row exceeds the budget is refused and to
  send a shorter excerpt (WR-03: at 3 008 MB the budget is 120 and Laya-en's window is 512, so truncation is
  unreachable). The `The labels, in this order: [` segment stays byte-identical (the Lambda probe parses it).
  B-iserror: bound refusals (count, shape/unknown key, byte length, token budget, admission `Busy`, the
  build-time served-task fit) are `pmcp::Error::tool_rejected`, which pmcp 2.19.3's tool dispatch sends as a
  successful `CallToolResult { isError: true, content: [text] }` instead of JSON-RPC -32603; model and internal
  failures stay -32603. Neither change reaches the live Lambda (RUNNING, D-ITEM-08-18-A) until plan 08-30's
  redeploy decision.
- **D-10:** `classify` accepts a **list of texts** with a **contract-owned maximum** (a tool-boundary contract in
  the shape of `contracts/forecast-tool-boundary-v1.yaml`). Per-row forwards are acceptable; batched GEMM is an
  optimisation, not a requirement.
- **D-11:** Each result returns **`label` + calibrated `probabilities` over every label in order**; the response
  carries **model identity** (artifact content hash + recipe id) so a caller can prove which model answered.
- **D-12:** Text past the 512-token window is **truncated exactly as Laya's builder does**, with **`truncated: true`**
  on that result (parity with Python; the spike-025 fixture has a 512-token truncation row).

### Crate home, artifact and deploy
- **D-13:** **ModernBERT lives in aprender-core at `crates/aprender-core/src/models/modernbert/`**, beside
  `models/bert/` and shaped like it (config, embeddings, layer, encoder, load-from-.apr). It is a reusable encoder,
  not Laya-specific. aprender-core's own BERT (`models/bert/`, post-norm HF BERT + WordPiece) is the wrong
  architecture on every block and is NOT reused — the port is spike 025's semantics table.
- **D-14:** The decision layer is a new **method-neutral crate `aprender-decide`**: a decision-method seam (artifact
  manifest, `task.json`, the classify contract) with **Laya as the only implementation** (head, scorer, request
  builder, temperature buckets on top of core's ModernBERT). Kev/Jev are deferred. The name is deliberately not
  `llm-`: Laya is an encoder.
  — **Reversibility:** one-way — a crate name becomes permanent once published to crates.io; confirm before the first publish.
  — **Amended 2026-09-29 (user, plan 08-31 Task 2 decision `publish-false`):** aprender-decide is `publish = false` until the crate name is confirmed; it stays in-tree and CI-tested and is not added to the release cascade's TIERS. `scripts/check_cascade_covers_all_crates.sh` no longer lists it (its remaining offender, aprender-contrastive-data, is not Phase 8's and is deferred). Reversible: drop the key and add the crate to TIERS after aprender-core when the name is confirmed. Nothing was published.
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
  — **Amended 2026-09-27 (user, option 1 from spikes 027/028):** the pack/verify re-score bar is max(1e-5, 4 x the per-checkpoint, per-set torch-fp32-vs-float64 noise), recomputed in Rust from a hash-bound float64 record (rescore-noise.json; no record = the 1e-5 floor; a bound above 1.0e-3 refused); argmax stays exact. The 1e-5 literal stays for fixture rows (the spike-025 fixture and the tiny fixtures); final_norm_abs / logits_abs are scoped fixture-only with values unchanged; x86_64 is recorded as unmeasured (A1; spike 028 showed the fixed 1e-5 bar refused the only gate-passing checkpoint for torch's own fp32 rounding). See laya-parity-v1 A1 (2.0.0).
- **D-18:** **Live pmcp.run deploy of one trained model** on default Lambda at 10,240 MB, weights fetched from S3
  at cold start (never baked into the image — spike 026 / `aws-mcp-model-hosting.md`), verified by a real
  `classify` call. The live deploy itself is a **human checkpoint** (outward-facing).
  — **Amended 2026-09-28 (user, plan 08-30 tier decision `restore-10240-tier`):** "Please use the RAM memory that is needed, as we had to down grade the token count due to the memory size. The 8000 MB is only a test I did for the option to above 3008 MB." The 3,008 MB tier of plans 08-17/08-18 (the account's former Lambda memory ceiling) was a forced downgrade and is superseded; the deploy returns to this decision's 10,240 MB, and decide-tool-boundary-v1 7.0.0 re-derives every bound for it (8 texts / 800 built tokens; v1.0.0's 1024 is not restored because it omitted the sha256 pin and rounded the per-token price down). The 8000 MB console setting was the owner's test, not a tier.
  — **Amended 2026-09-28 (user, plan 08-30 Task 2 decision `redeploy-and-measure`):** redeploy the hardened bootstrap with the same artifact (sha256 24a44d7e…) at 10,240 MB and cold-sample both maximal shapes; the S3 download deadline is derived from those samples, and if the derived deadline lands below the largest measured download the plan stops and asks again rather than deploying it ("Stop and ask again").
  — **Amended 2026-09-28 (user, plan 08-30 Task 3 deadline decision `keep-800-apply-10739`):** the 10,240 MB tier was measured live (4 proven-cold samples, 22,647-27,015 ms, all under 30 s). decide-tool-boundary-v1 keeps 8 texts / 800 built tokens on its acceptance rule (every recorded cold sample < 30 s) rather than the per-term-max re-derivation, which the samples put at 648 tokens; the contract (8.0.0) shows the priced and measured terms side by side, including the 420-500 ms build + probe-replay overshoot and the 2,751 ms gateway spike. The Lambda S3 download deadline becomes 10,739 ms (30,000 - (15,261 measured sha + build + classify + 4,000 margin)), and the handler also bounds the download by the invocation's remaining time (V4-b). The stop-guard's reference set is corrected to this tier's own downloads (max 9,065 ms): the 3,008 MB downloads (13,494-17,831 ms) came from a superseded tier. Redeploy and re-measure (2 cold samples per shape) before accepted_region_cold is flipped to implemented.
- **D-19:** The demo model is **TweetEval stance** at 16 or 64 shots/class — spike baselines exist to sanity-check
  the gate (full FT F_avg 0.538 ± 0.017 @16, 0.608 ± 0.050 @64; SetFit 0.512 / 0.561).
  — **Amended 2026-09-27 (user, option 1 from spikes 027/028):** the demo cell moves from 16 to 64 shots/class (s64-seed13, early stopping with at most 12 epochs, three seeds under the median rule), gated on the 459-row in-distribution held-out set ([111, 291, 57]) with the 280 test rows as the reported shift probe; one declared run, a gate fail halts for a human. The s16 runs stay as the recorded 1.2.0 outcome (gate_fail) and as the verifier's two fail-closed vectors. See laya-finetune-gate-v1 `demo_s64`.

### Claude's Discretion
- Exact file layout of the uv project and just recipe names.
- How the Python output becomes .apr (a Python-side export vs extending `apr import`/the converter) — pick whichever
  keeps one converter implementation and a clean parity chain; research should compare.
- S3 layout, IAM scoping and the cold-start loader, following the SetFit/Chronos Lambda crates.
- The declared ECE ceiling and zero-shot margin values (must be declared in the contract before the demo run).
- The `classify` list maximum (priced against the Lambda envelope, not guessed — see Phase 7's accepted-region lesson).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Spike evidence (Laya)
- `.claude/skills/spike-findings-aprender/SKILL.md` — findings index; load via `Skill("spike-findings-aprender")`
- `.claude/skills/spike-findings-aprender/references/laya-decision-model.md` — fine-tune recipe, calibration need, what to avoid (head-only adapters)
- `.claude/skills/spike-findings-aprender/references/laya-rust-inference.md` — ModernBERT/head/scorer/builder semantics table, local window |i−j| ≤ 64, parity ladder, productisation checklist
- `.claude/skills/spike-findings-aprender/references/aws-mcp-model-hosting.md` — default Lambda for ≤1 GB models, S3 cold start, never bake weights
- `.claude/skills/spike-findings-aprender/sources/024-laya-vs-kev-few-shot/tools/ft_laya.py` — the training recipe to productise
- `.claude/skills/spike-findings-aprender/sources/025-laya-rust-forward-parity/` — Rust port (`src/laya.rs`), oracle (`tools/oracle.py`)
- `.claude/skills/spike-findings-aprender/sources/026-laya-mcp-default-lambda/src/lib.rs` — `render_options`, request path, S3 loader
- `.planning/spikes/025-laya-rust-forward-parity/fixtures/laya-en_fixture.json` — 14-row parity fixture (ladder `.bin` regenerates via the oracle)

### Architecture rules and precedents
- `CLAUDE.md` §"CRITICAL: Realizar-First Architecture" — the table gaining a third exception row (D-16); SetFit and Forecast rows are the argument template
- `contracts/setfit-apr-v1.yaml` — APR schema/load-rule contract precedent (D-17)
- `contracts/forecast-tool-boundary-v1.yaml` — tool-boundary contract precedent (D-10)
- `contracts/chronos-bolt-parity-v1.yaml` — parity contract precedent (D-17)
- `crates/aprender-core/tests/monorepo_invariants.rs:313-319` — thin-server `[[bin]]` list (D-15)

### Existing code to mirror
- `crates/aprender-core/src/models/bert/` — shape for `models/modernbert/` (D-13)
- `crates/aprender-mcp-setfit/` — thin pmcp predict server template
- `crates/aprender-mcp-chronos-lambda/`, `crates/aprender-mcp-setfit-lambda/` — Lambda bootstrap + deploy config pattern
- `crates/aprender-core/src/format/converter/` — import/convert path the .apr conversion must fit (D-17)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `trueno::blis::gemm_blis` + the spike-020 layout (`Cᵀ = W · Xᵀ`, weight `[out,in]` as A, rows banded over rayon) — the only aprender dependency spike 025 needed.
- `crates/aprender-core/src/format/converter/f16_convert.rs` — F16 handling for the F16-in-.apr decision (D-17).
- `crates/aprender-mcp-setfit/` — in-process predict via a public library door; the Laya server does the same through `aprender-decide`.
- Spike 026's `render_options` / refusal rules (unknown types, <2 options, options beyond `head_max_len` → refused, not defaulted).

### Established Patterns
- Thin server per model; transport-only servers; bounds owned by a contract and not re-checked by the library door.
- Lambda handler binary is named `bootstrap`; cargo-pmcp discovers deployables by that name and has no `--package`
  flag — a new Lambda crate needs `--manifest-path` (SetFit training precedent).
- Workspace lints: `unsafe_code = forbid`, no `unwrap()`, pedantic clippy; coverage must co-evolve with contracts (`#[contract]` + falsification tests).
- New `tests/*.rs` targets are dark until added to the explicit `--test` line in `.github/workflows/ci.yml` — editing CI workflows needs a check-in per CLAUDE.md.

### Integration Points
- `crates/aprender-core/src/models/mod.rs` — register `modernbert`.
- Root `Cargo.toml` workspace members — add `aprender-decide`, `aprender-mcp-decide`, `aprender-mcp-decide-lambda`.
- `crates/aprender-core/tests/readme_contract.rs` — new crates need READMEs and the monorepo link; README crate/contract counts will move.
- The deploy recipe reads the gate report and refuses on fail (D-07).

</code_context>

<specifics>
## Specific Ideas

- The user wants the design **generalized**: ModernBERT supported "in a similar way that other BERT models are
  supported", and the decision crate able to host "other methods that will pop up soon with the success of Jev, Kev,
  and Laya". The trait/seam should be real enough that Kev can join without reshaping the artifact or tool contract —
  but it must not be speculative abstraction; one implementation, designed for a known second.
- Future bases (multilingual mmBERT, `typed-decisions`) are expected "when we get use cases for it".

</specifics>

<deferred>
## Deferred Ideas

- **Multilingual Laya base (mmBERT-base 322M, 256k vocab, different RoPE)** — future phase, when a use case arrives.
- **`typed-decisions` base as a fine-tune starting point** — future phase; its full fine-tune was never measured.
- **Kev / Jev as `aprender-decide` methods** — future phase; Kev's Rust path lives only on unpushed `spike/016-upstream-sync`.
- **A Laya/decide training MCP server** (the `aprender-mcp-setfit-train` shape) — not requested; this phase trains locally.
- **Rust trainer for ModernBERT** — not planned; Python stays the training oracle.
- **Batched-row GEMM for `classify`** — optimisation after the tracer works.

### Reviewed Todos (not folded)
- `aprender-train-gpu-ledger-tests-red.md` — matched on generic keywords only; unrelated.
- `qwen35-9b-hybrid-forward-unimplemented.md` — Kev/Qwen territory; relevant only when Kev joins `aprender-decide`.
- `workspace-test-gate-blocked-by-renacer-validate.md` — CI gate issue, unrelated to this phase's scope (may still affect its CI runs).

</deferred>

---

*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Context gathered: 2026-09-25*
