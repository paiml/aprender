# APR-EMBED-001 — EmbeddingGemma 2 on every `apr` verb: embed, serve, model ops, fine-tune, prune, distill

Status: DRAFT v0.1 for quorum (one agy · one claude · one apr) · 2026-10-07 · owner: Noah Gift · train 0.72 `[O 2026-10-07]`
Drop at `docs/specifications/APR-EMBED-001-embeddinggemma-2-support.md`. Run: `Implement docs/specifications/APR-EMBED-001-embeddinggemma-2-support.md autonomously.`
Marks: `[V]` measured 2026-10-07 on `origin/main` @ `409800713` by the probe beside it (grep and a YAML parse in a read-only clone; EG-0 re-binds code seams with `pmat query`) · `[C]` computed, arithmetic shown · `[O]` operator statement · `[A]` taken from a spec, or an estimate · `[X]` third party, never a claim · `[U]` unmeasured; the row that measures it is named.

---

## ELI5

**Goal.** `apr` runs Google's EmbeddingGemma 2 from the Unsloth GGUF. Every verb either works with a receipt or says no by name.

**How it works.**
- A model gets a *kind*: generative or embedding. The kind is a type. It is read from the file.
- Every verb in the registry chooses, per kind: `receipt`, `refuse`, or `no-model`.
- Then the training verbs: fine-tune, prune, distill. They are judged by retrieval on held-out data.

**What is true today.**
- Nothing loads it. The architecture string has zero hits in the code.
- The pinned llama.cpp oracle is three weeks older than the model.
- If the file merely loaded, the embeddings route would return a wrong vector with HTTP 200.

**What stops the line.** A wrong vector that looks right · a dropped image · float16 math · a pin change without the operator's yes.

**Done when.** 114 of 114 verbs classified · 6 of 6 files at parity on both required GPU hosts · 0 silent-wrong cells · one fine-tune that beats its base on held-out data, with controls.

| Question | Answer |
|---|---|
| Does `apr` load `gemma-embedding2` today? | No `[V]` |
| Does Unsloth prune or distill this model? | No `[X]` — neither word is in its README or its embedding guide |
| Does the Unsloth GGUF hold the vision and audio encoders? | No `[C]` — 558 MB at 16 bits is the text tower |
| Can upstream llama.cpp serve image or audio embeddings? | No `[X]` — its issue 30082 was closed as not planned |
| Is 0.72 "Train What You Serve" on `main`? | No `[V]` — `docs/roadmaps/epics.yaml` maps 0.72 to Agent Ready |

## Purpose and terms posture

- The goal is an open-source Rust ML framework. Models are not sold. Nothing here is built to compete with any provider.
- No hosted-model output is ever a training target. Training data here is gold labels and public pair data with a recorded licence.
- Claude Code is used only by its paying account owner. No shared credentials. No free accounts.
- Public wording never positions a model as a replacement for a commercial service.
- EmbeddingGemma 2 is Apache-2.0 and its card binds deployments to the Gemma Prohibited Use Policy `[X]`. Any weights derived here (fine-tuned, pruned, distilled, re-quantised) keep that licence and policy reference in their stamp. **This spec publishes no crate and no weights.**

---

## §0 Operating assumptions

1. **"0.7.2" is train 0.72.** Tag `v0.70.2` exists (`89e261cda`, 2026-10-07T12:33Z) `[V]`.
2. **Guide coverage.** `sovereign-ai-stack-claude-project-guide.md` (snapshot 2026-07-02) does not cover EmbeddingGemma, Unsloth, model kinds, or the 0.7x trains. Only the publish gate and the workflow rules below are guide-grounded.
3. **Two artifacts.** The GGUF files are the inference artifact. The safetensors checkpoint is the training source. Training verbs take `.apr`.
4. **Kind is a type.** `ModelKind::{Generative, Embedding}`. Verb dispatch is an exhaustive `match`, so a new kind cannot compile until every verb decides.
5. **Refusal is a shipping behaviour, never a success condition.** `refuse` cells are counted on their own line and never as green.
6. **Measure, then build.** EG-0 lands with no engine code.
7. **One declaration per fact.** The registry and the capability contract are extended. No parallel table is minted.
8. **Cross-repo work is filed, not done.** Oracle build and competitor leg are forjar resources in paiml/infra. Cells that need them are `NotRun{NoDeclaredExecutor}` until they land.
9. **The operator has no pre-steps.** Anything only Noah can do is a §7 STOP.
10. **The two attachments of 2026-10-07** (Jev-Omni, LLM2Vec) are read as adjacent scope. They are Appendix B, not 0.72 rows.

---

## §1 Ground truth (baseline, frozen; EG-0 re-derives it; never quote as current)

### 1.1 The tree

| # | Fact | Mark | Source / probe |
|---|---|---|---|
| G1 | The verb registry holds **114** commands in 13 categories. **42** take a model, **72** do not | `[V]` | parse `commands:` in `contracts/apr-cli-commands-v1.yaml`, count `requires_model` |
| G2 | `gemma-embedding` in any spelling: **0** hits in `crates/` and `contracts/`. `gemma4`: **0** hits. Architecture literals: `gemma` 119, `gemma2` 51, `gemma3` 24, `gemma3n` 5 | `[V]` | zero hits: `grep -rIilE` over both trees; literals: `grep -rhoE '"gemma[a-z0-9_-]*"' crates --include='*.rs' \| sort \| uniq -c` |
| G3 | The contract gate treats any name starting with `gemma` as the Gemma family and fails loud unless it is exactly v1 or v2. So today's expected behaviour is a loud refusal `[C]`. The gate's own doc states that alternating local and global attention is **not implemented** | `[V]` | `is_gemma_family`, `is_gemma1_supported`, `is_gemma2_supported` in `crates/aprender-serve/src/contract_gate.rs`; `is_gemma1`, `is_gemma2` in `crates/aprender-serve/src/gguf/config.rs` |
| G4 | `apr embed` is a BERT bi-encoder: WordPiece, `.apr` only, layer sizes typed as flags, forward in `aprender::models::bert` and not in realizar | `[V]` | `crates/apr-cli/src/commands/embed.rs` |
| G5 | `/v1/embeddings` mean-pools the final hidden state of whatever model is loaded, L2-normalises it, and returns the trunk width. The contract pins "dim == model hidden size" | `[V]` | `crates/aprender-serve/src/api/realize_handlers_embed_completion.rs`; EMBEDDINGS-MODEL-BACKED in `contracts/apr-serve-openai-compat-v1.yaml` |
| G6 | The capability registry derives `CausalMask` for **every** architecture (`op_derivation.always`) and maps a gated MLP to the `SwiGLU` op. `GeluMlp`, `LayerNorm`, `AttnFinalSoftcap`, `PostAttnFfnNorm` are GPU-unsupported. `BF16`, `F16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K` are GPU-supported | `[V]` | `contracts/apr-model-capability-v1.yaml` (`ops`, `op_derivation`, `quant_types`) |
| G7 | An LLM-as-embedder path through realizar already exists, with query and passage prefixes. The other real embedder is fastembed, an optional external dependency on ONNX Runtime | `[V]` | `crates/aprender-rag/src/embed/nemotron.rs`; `FastEmbedder` in `crates/aprender-rag/src/embed/mod.rs`; `crates/aprender-rag/Cargo.toml` |
| G8 | `finetune` (auto, full, LoRA, QLoRA), `prune` (magnitude, structured, depth) and `distill` (teacher to student, YAML config) all take `.apr` | `[V]` | module docs of `crates/apr-cli/src/commands/finetune.rs`, `prune.rs`, `distill.rs` |
| G9 | Contrastive seams exist: `info_nce_loss`, `InfoNCELoss`, `TripletLoss`; `ContrastiveTask`; `SimCSE`; a pair-data crate. Masked next-token prediction, Matryoshka and multiple-negatives ranking: **0** hits each | `[V]` | `crates/aprender-core/src/loss/loss.rs`; `crates/aprender-core/src/nn/self_supervised.rs`; `crates/aprender-core/src/nn/self_supervised_byol_simcse.rs`; `crates/aprender-contrastive-data` |
| G10 | Oracle pin: `build_commit = "d1d3c3396"`, `pinned_on = "2026-09-15"`, `pin_expiry = "2026-12-01"`. Upstream support for this model merged 2026-10-06 `[X]`. So the pinned oracle cannot load it `[C]` | `[V]` | `scripts/llama_pin.toml`; resolver `scripts/llama_bin.sh` |
| G11 | Parity floor `min_cosine: 0.98` has a measured basis on autoregressive models over at least 64 positions. A model listed without its own `min_cosine` is fail-closed | `[V]` | `evidence/parity/thresholds.yaml` |
| G12 | Ladder: required hosts lambda (sm_89) and gx10 (sm_121). `cells.verbs` is `run, chat, serve, code`. All 8 rungs are Qwen. The inventory sweep measures every matching file a host holds. A Q4_K rung must be required and must claim cuda | `[V]` | `contracts/model-capability-ladder-v1.yaml` (`hosts`, `cells`, `inventory`, the 3712 comment) |
| G13 | Train map: 0.72 is E5 "Agent Ready" (issue 4000, budget 5, 9 rows) and its plan puts new model support out of scope. 0.74 is "Any Model". 0.75 is "CRUX Fine-Tune/Distill". "Train What You Serve": **0** hits | `[V]` | `docs/roadmaps/epics.yaml`; `docs/specifications/EPIC-0.72-agent-ready-plan.md` §5 |
| G14 | A tag carries named scope only through the release-critical table | `[V]` | `docs/specifications/APR-RELEASE-001-train-and-build-kaizen.md` §1.5 |
| G15 | Debt baselines this spec must not raise: shared attention and FFN blocks are called by 0 of 8 production forward files; 42 files match on quant type | `[A]` 2026-09-23 | `docs/specifications/EPIC-0.74-any-model-plan.md` §1 and §6 |
| G16 | No fleet host declares Unsloth. `apr` concedes in-loop QLoRA throughput on GPU | `[A]` 2026-09-23 | `docs/specifications/EPIC-0.75-crux-finetune-distill-plan.md` §1 |
| G17 | Capacity sample: 8 to 20 merged PRs per train | `[A]` 2026-09-10 | `docs/specifications/06x-release-schedule.md` §1.2 |

### 1.2 The model — all `[X]` until EG-0 measures the files

| # | Fact | Source |
|---|---|---|
| M1 | Architecture `gemma-embedding2`. 24 layers, width 512, FFN hidden 2,048, 4 heads, KV heads 2 (local) and 1 (global), local to global 5:1, sliding window 1,024, vocabulary 262,144, gated FFN with GELU, mean pooling, projection 512 to 768, context 8,192 | model card |
| M2 | The text encoder is bidirectional | Unsloth PR 12865 |
| M3 | Unsloth files: UD-Q4_K_XL 176 MB, UD-Q5_K_XL 210 MB, UD-Q6_K_XL 249 MB, Q8_0 310 MB, BF16 558 MB, F16 558 MB. Listed size 0.3B parameters. 558 MB ÷ 2 bytes ≈ 279 M parameters: the text tower alone `[C]` | Unsloth GGUF page |
| M4 | Output dimensions 768, 512, 256, 128. A truncated vector must be re-normalised | model card |
| M5 | Seven task prompts (`SearchQuery`, `QuestionAnswering`, `FactChecking`, `CodeRetrieval`, `Classification`, `Clustering`, `SentenceSimilarity`) and the document form `title: {title or none} \| text: {content}`. Omitting the prompt lowers quality without an error | model card |
| M6 | Float16 activations return NaN or silently degraded vectors. bfloat16 and float32 are safe. Unsloth pools in float32 for this reason | model card; Unsloth PR 12865 |
| M7 | Upstream llama.cpp: text embeddings work. `/embedding` with `image_data` answers 200 and ignores the image; three different images gave identical vectors | llama.cpp PR 30054, issue 30082 |
| M8 | What Unsloth does with it: text serving through llama.cpp (`--embeddings --pooling mean`); LoRA, QLoRA and full fine-tuning (rank 16, alpha 32, targets q/k/v/o/gate/up/down, multiple-negatives ranking loss, bf16 on, fp16 off); save adapters, save merged, push. Vision and audio layers are excluded from LoRA by default | Unsloth model guide, embedding guide, PR 12865 |
| M9 | `[U]` → EG-0: per-tensor quant types inside the three UD files · exact names of five of the six files · metadata keys (causal flag, pooling type, window pattern, per-layer KV heads) · whether the 512 to 768 projection is in the main GGUF · whether the trunk uses post-norms, QK-norm or RoPE · whether F16 weights degrade on the oracle itself | — |

**Consequence of G5 + M1 `[C]`.** The trunk width (512) is also a legal truncated dimension. A build that loads this file and skips the projection returns a 512-long, unit-norm, plausible vector. A length check cannot see it. Only a value check can.

---

## §2 Invariants (each has a falsifier in §9)

| # | Invariant |
|---|---|
| **I-1** | **Publishing: clean-room CI on the tag is the hard gate, named first.** No workflow runs `cargo publish`. Never `--allow-dirty`. No EG row publishes a crate or weights; `apr publish` is exercised with `--dry-run` only |
| **I-2** | **Totality.** Every registry command declares one verdict per model kind: `receipt`, `refuse`, or `no-model`. There is no fourth state |
| **I-3** | **No silent wrong answer.** A `receipt` cell that exits 0 has passed its check. A `refuse` cell exits non-zero and names the verb to use. A permanent refusal is `Refused{NotThisKind}`; an owed one is `Refused{NotYet, removed_by: <train>}` |
| **I-4** | **The output is the projected vector.** The default response is 768 long. A truncated response equals the leading dimensions of the 768 vector, re-normalised. It is never the pooled trunk state |
| **I-5** | **Finite or refuse.** No non-finite value leaves `apr`. Activations for this architecture never run in float16; a float16 kernel dispatch refuses by name |
| **I-6** | **Batch and order invariance.** A text embeds the same alone, in a batch, and at any batch position |
| **I-7** | **Media is refused, never dropped.** An image, audio or video input to a file with no such encoder gets a 4xx that names the missing encoder |
| **I-8** | **Parity is against an independent implementation on the same file.** The oracle is llama.cpp at the declared pin. The threshold is set by the thresholds file's own method: known-good and known-bad pairs, n = 5, each required host. GPU against CPU the same way |
| **I-9** | **Per-layer facts come from the file.** Attention type, KV-head count, pooling and the causal flag are read from metadata and tensor shapes. Never from layer-index arithmetic. Never from an architecture-level constant |
| **I-10** | **No new debt.** The new forward path calls the shared attention and FFN blocks. No quant-type `match` is added outside the type table |
| **I-11** | **Trained weights are judged by retrieval, with controls.** A fine-tune must beat its base beyond the 3-seed band. A shuffled-pair run must not |
| **I-12** | **Receipts carry identity.** `apr` version and sha, feature set, model sha256, quant, backend, host, oracle build, task prompt, dimension. A receipt whose oracle build differs from the pin on `main` at the cut is STALE |

---

## §3 Rows (EV order; one ticket per session; one parent issue per PR)

Selector: each session takes the **first row whose `done_when` fails at `origin/main`**, does that row only, and stops.
Until a `receipt` cell is green it refuses by name with `removed_by`. The exit bar counts greens only.

| Row | Title | Verbs it receipts | Deliver | done_when | K̂ `[A]` |
|---|---|---|---|---|---|
| **EG-0** | Measure: probe, oracle, contracts, corpus. **No engine code** | — | (a) Probe receipt: all 42 model-taking verbs and the 8 model-adjacent ones on the UD-Q4_K_XL file, exit code and first line each. (b) `apr inspect` and `apr tensors` on 6 of 6 files: resolves M9. (c) Oracle: the pinned build refuses the file; a build at the upstream merge of PR 30054 or later writes golden vectors and token ids for the corpus. (d) Registry gains a per-command, per-kind verdict; the totality guard runs in `ci / gate`. (e) Capability contract gains the new ops and makes `CausalMask` kind-conditional, source and mirror together. (f) Forward contract (local and global attention, mean pool, projection, normalise, truncation) passes `pv validate` with no anonymous obligation. (g) Pre-registered corpus with a sha256 manifest: texts × 7 prompts and the document form; lengths 8, 128, 1,024, 2,000 and 8,000 tokens; a batch-invariance set; a retrieval micro-set; a held-out fine-tune set | all on `main`; the guard has a first green and its mutation RED in the required check; every M9 item has a value and its command | 80 |
| **EG-1** | Load: GGUF, tokenizer, kind | `pull` `list` `rm` `tokenize` `capability` `inspect` `debug` `validate` `lint` `tensors` `diff` `hex` `tree` `flow` | Loader admits the architecture as `ModelKind::Embedding` by a distinct path; the `gemma` prefix match is not widened. Per-layer KV heads and attention type read from the file (I-9). SentencePiece vocabulary of 262,144. Model-family contract for the variant | token ids equal the oracle's on 100% of the corpus; the 14 verbs green on 6 of 6 files | 100 |
| **EG-2** | Embed on CPU | `embed` `trace` | Forward in realizar on the shared blocks (I-10): local and global attention, gated GELU FFN, mean pool over real tokens, projection, L2 normalise, truncation. `apr embed` takes a GGUF and reads sizes from the file; `--task`, `--title`, `--dim`. The BERT path is unchanged | parity (I-8) on 6 of 6 files × 7 prompts × 4 dimensions × 5 lengths; I-4 and I-6 green; BERT `apr embed` tests still green | 150 |
| **EG-3** | Serve | `serve` | Kind-aware routes: `/v1/embeddings`, `/api/embeddings`, `/realize/embed` use the EG-2 path; generative routes answer 4xx naming the kind. The request's `dimensions` field maps to truncation. Media refused (I-7). The contract obligation becomes "dim == hidden size for the generative kind; dim == declared output for the embedding kind" | HTTP and CLI vectors equal for the same input; I-7 green; the amended contract passes `pv validate` | 80 |
| **EG-4** | CUDA on sm_89 and sm_121 | `parity` `profile` `ptx-map` | GPU forward for every op EG-0 found, with the GELU gate (not the SiLU one). Float16 poka-yoke (I-5). Long-input cells at 2,000 and 8,000 tokens | GPU-against-CPU parity on both required hosts, 6 of 6 files; I-5 green; a line that reads `falling back to CPU` is RED | 120 |
| **EG-5** | Refusals, `apr qa`, the ladder | `qa` and the five refusals: `run` `chat` `code` `rerank` `showcase` | Named refusals (I-3). `apr qa` gains a golden-embedding gate for this kind. The ladder contract gains kind-aware cells (`embedding: [embed, serve]`); the inventory sweep cannot judge an embedding file on generative verbs. One rung per file with its sha256 | `scripts/check_model_ladder.sh` green on both hosts with the new rungs; 5 of 5 refusals by name | 80 |
| **EG-6** | Model ops round trip | `import` `convert` `export` `rosetta` `stamp` `compile` `quantize` `merge` `shard` `unshard` `encrypt` `decrypt` `publish` (dry run) | gguf ↔ safetensors ↔ `.apr`. Each output is re-embedded and compared with its source by I-8. An exported GGUF must load in the oracle. `quantize` from BF16 to Q8_0 and Q4_K is compared with Unsloth's own files, report only | 13 verbs green; round-trip parity on the corpus; the oracle loads every exported GGUF | 120 |
| **EG-7** | Fine-tune | `finetune` `tune` `diagnose` | Backward through the encoder. In-batch-negative contrastive loss on pairs and triplets. A batch sampler with no duplicates. bf16 or f32 only. Method order: LoRA, then QLoRA, then full; each is its own cell. Adapter and merged outputs. FILE ONLY: an Unsloth leg in paiml/infra, pinned in forjar | I-11 green over 3 seeds; the merged model passes EG-2 and EG-6 checks; no non-finite loss | 180 |
| **EG-8** | Prune, then distill | `prune` `distill` | Depth pruning in whole local-to-global periods of 6 layers, unless the target format carries a per-layer pattern (I-9). Magnitude and structured pruning report curves. `apr distill` from the original to the pruned student on an embedding loss | a quality-retention curve published whichever way it falls; no threshold until 3 records; every output finite and loadable | 150 |
| **EG-9** | Evaluate, measure, dogfood | `bench` `eval` `check` `qualify` `compare-hf` `rag` | Throughput is report only: texts and tokens per second at batch 1, 8 and 32, both hosts, beside the oracle server on the same file, under the exclusive GPU lock. An `Embedder` for this model beside `NemotronEmbedder`. `apr rag index` and `apr rag query` on one real corpus. FILE ONLY: the semantic leg of the infra RAG eval | one retrieval receipt beside the `FastEmbedder` baseline; APR-OBS timing rows present for `embed` and `serve` | 120 |

**Cell arithmetic `[C]`.** Receipt-owed: 14 + 2 + 1 + 3 + 1 + 13 + 3 + 2 + 6 = **45**. Refusals: **5**. No-model: **64**. Total 114.
**Capacity `[C]`.** E5 already holds 9 rows. These 10 make 19, the top of the G17 band, before any rerun.

---

## §4 Execution (per row)

| Phase | Do | Stop if |
|---|---|---|
| P0 | Ticket first: the cop mints it with `pmat work add`. Per-session worktree, never the primary checkout. Re-bind every `[U]` and `[A]` the row leans on with `pmat query`, at `origin/main`; state the tree in the receipt | a premise is false at HEAD: record `premise-falsified`, skip the row, report |
| P1 | Contract and falsifier first: one provable contract per row. The §9 plant for the row's invariants is RED before any engine line | a falsifier cannot be made RED: the gate is vacuous |
| P2 | Implement. Workspace crates only: trueno is `aprender-compute`, realizar is `aprender-serve`, entrenar is `aprender-train`. Never a crates.io `trueno` | — |
| P3 | Mutation RED then GREEN, observed **in the required check**, pasted in the PR body | green only locally |
| P4 | Quorum: one agy, one claude, one apr. Two non-passing rounds go to the human review queue | — |
| P5 | The agent arms its own merge through the merge queue. One PR from this spec in CI at a time. Receipt under `docs/audits/` | — |

GPU development runs on lambda-labs or on yoga through existing make targets. sm_121 cells run on gx10. Hosts are provisioned only by `forjar apply -f machines/<host>/forjar.yaml` or a make target in paiml/infra. No ad-hoc SSH.

---

## §5 Definitions of done (the exit bar)

| # | Bar | Target | Probe |
|---|---|---|---|
| D1 | Registry commands with a verdict for the embedding kind | 114 of 114; unclassified 0 | the totality test beside `crates/apr-cli/tests/cli_commands.rs` |
| D2 | `receipt` cells green, per required host and claimed backend, at the cut sha | every cell the tag claims; unclaimed cells refuse with `removed_by` | `scripts/check_model_ladder.sh` |
| D3 | Refusals by name | 5 of 5, on their own line | probe receipt |
| D4 | Silent-wrong cells | **0** | I-3 plants |
| D5 | Files at parity | 6 of 6 Unsloth, 2 of 2 upstream as control | parity receipts |
| D6 | Parity threshold | a `models:` entry with basis, n and command | `evidence/parity/thresholds.yaml` |
| D7 | Default dimension 768 · non-finite values emitted · float16 dispatches · media inputs answered 200 · batch cells below threshold | 100% · 0 · 0 · 0 · 0 | I-4 to I-7 plants |
| D8 | Private attention loops added · quant `match` sites added | 0 · 0 | the two census commands in `docs/specifications/EPIC-0.74-any-model-plan.md` §6 |
| D9 | Fine-tune positive control · shuffled control · non-finite losses | 1 · 1 · 0 | EG-7 receipt |
| D10 | Dogfood | 1 corpus, 1 retrieval receipt | EG-9 receipt |
| D11 | Flow | ≤ 1 PR of this spec in CI; no PR older than 24 h | `gh pr list` |

Throughput, quality retention and competitor comparisons have **no threshold**. They are recorded, then ratcheted after 3 records, each with its committed command.

---

## §6 Budget and routing

- Per row: K = ⌈1.1 · K̂⌉, andon at 0.8 · K. At andon: WIP commit, draft PR, `PARTIAL(andon)`, list what is unplanned.
- Must-carry set EG-0..EG-6: K̂ 730, K 803, andon 642. All ten rows: K̂ 1,180, K 1,298, andon 1,038. All `[A]`.
- Opus 5.5 takes cross-crate rows (EG-0, EG-2, EG-4, EG-7, EG-8). Sonnet 5.5 takes single-module rows. Fable is banned.
- Owner: the 0.72 look-ahead worker. It never competes with the current train for runners; the cop grants the CI slot.
- Backends beyond cpu and cuda follow whatever the ladder contract certifies at the cut. This spec adds none.

---

## §7 STOP conditions (stop and report; never work around)

| Stop | When |
|---|---|
| `pin-change` | a gate needs an oracle build newer than `build_commit` in `scripts/llama_pin.toml`. A pin move is a gate change and needs the operator's yes (Appendix C, D-3). EG-0 may measure against a provisional build; no row goes green in the release gate before the pin on `main` equals it |
| `epic-unbound` | no entry in `docs/roadmaps/epics.yaml` and no §1.5 row names this scope for 0.72 (D-1) |
| `projection-absent` | the 512 to 768 projection is not in the main GGUF and no declared sidecar carries it |
| `oracle-suspect` | the oracle disagrees with the card's stated behaviour, or degrades on its own F16 file. Diagnose the harness first; record, do not chase |
| `tokenizer-mismatch` | token ids differ from the oracle's on any corpus item |
| `op-unsupported-gpu` | EG-0 finds an op the capability contract marks GPU-unsupported. The CUDA cell refuses by name; it never falls back |
| `quorum-split` | after one amendment round |
| always | any step that needs a waiver, a threshold with no basis, new Python in the tree, a publish of a crate or of weights, a force-push, a re-tag, a ruleset or secret change, or a hand install on a host |

Every stop names the other work that continues.

---

## §8 Final report schema

```
APR-EMBED-001 | row=<EG-n> | verdict=<MERGED|STOP(reason)|NOOP> | main=<sha> | PR=<url>
verbs:   classified <n>/114 | receipt <green>/<owed> | refuse <k>/5 by name | silent-wrong <0>
files:   <n>/6 unsloth + <n>/2 upstream | lambda=<g|r> gx10=<g|r> | backends <list>
parity:  oracle=<build> min_cos=<v> threshold=<t> basis=<path> | gpu-vs-cpu min_cos=<v> | known-bad max_cos=<v>
invar:   dim768=<n>/<N> nonfinite=<0> f16=<0> media200=<0> batch_below=<0>
train:   method=<lora|qlora|full> band=<v> gain=<v> shuffled=<in-band|VIOLATION> | curve=<path>
debt:    private_attn=+<0> quant_match=+<0> lines=<n>
unbound: [U] resolved <k>/<n> | filed <infra tickets>
next:    <row id>
```

---

## §9 Falsifiers (one per invariant; each plant is observed RED before the row may merge)

| Invariant | Plant that must go RED | Control that must stay GREEN |
|---|---|---|
| I-1 | an EG diff that adds `cargo publish`, or `apr publish` without `--dry-run` | the publish preflight is byte-unchanged by every EG PR |
| I-2 | add a command to the registry with no verdict for one kind; add a kind with one verb undecided | the landed registry prints 114 classified. A scan of 0 commands is RED |
| I-3 | make `chat` on the embedding file exit 0 | `chat` on a Qwen rung stays green; `embed` on the file stays green |
| I-4 | return the pooled trunk state when 512 is requested; drop the re-normalisation after truncation | default is 768; the 256 response equals the leading 256 of the 768 response, re-normalised |
| I-5 | force float16 activations on the 2,000-token input | bf16 and f32 stay finite at 8,000 tokens |
| I-6 | mean-pool over padding positions | single against batched, and batch order permuted, stay above threshold; max \|Δ\| is recorded |
| I-7 | send three different images to the embeddings route: any 200 is RED | a text input returns its vector |
| I-8 | four known-bad pairs: a causal mask · the SiLU gate in place of GELU · the local-to-global pattern shifted by one layer · the document form in place of the query prompt. Each must fall below the threshold. One that does not means the threshold is too loose: raise it until it separates | `apr` against the oracle on the same file is above the threshold; the oracle against itself is recorded as the noise floor |
| I-9 | force 2 KV heads on a global layer; depth-prune one layer out of a period and export | the unpruned file loads; a whole-period prune round-trips |
| I-10 | a private attention loop in the new forward file; a quant-type `match` outside the type table | shared-block calls in the new file are above 0 |
| I-11 | train on shuffled pairs; leak held-out pairs into training | the clean run gains beyond the 3-seed band; the corpus guard reads the manifest |
| I-12 | a receipt with no model sha256; a receipt measured on an oracle build that is not the pin | a complete receipt is accepted |

---

## Appendix A — verb × kind matrix for `kind = embedding` (proposal `[A]`; EG-0's quorum rules it)

| Verdict | Count | Verbs |
|---|---|---|
| `receipt`, takes a model | 37 | `embed` `serve` · `inspect` `debug` `validate` `lint` `tensors` `trace` `diff` `hex` `tree` `flow` · `export` `convert` `stamp` `compile` `merge` `quantize` `rosetta` `shard` `encrypt` `decrypt` · `publish` · `finetune` `prune` `distill` `tune` · `bench` `eval` `check` `qa` `qualify` `compare-hf` `parity` · `profile` `ptx-map` · `diagnose` |
| `receipt`, model-adjacent | 8 | `pull` `import` `list` `rm` `unshard` `tokenize` `capability` `rag` |
| `refuse`, permanent | 5 | `run` `chat` `code` (no language-model head) · `rerank` (cross-encoder verb) · `showcase` (a Qwen demo). Each names `apr embed` |
| `no-model` | 64 | every other registry command |

37 + 8 + 5 + 64 = 114. The 42 that take a model are the 37 and the 5.

## Appendix B — adjacent tracks (not 0.72 rows)

**Track J — Jev-Omni, a decision classifier on Gemma 4 12B `[X]`.** Open weights, Apache-2.0, safetensors BF16, tag `gemma4_unified`, a Python loader, CUDA required. It answers yes/no, choice and score questions with one probability per option.

| Row | What | When |
|---|---|---|
| J-1 | APR-DECIDE-001 holds that the vendor model has no artifact sha, so no comparison is stateable `[A]` (Project copy). An open-weight model with the same interface has one. Add a CRUX decision contract row naming it as a competitor; its card numbers stay `[X]`; the comparator stays `NotRun{NoDeclaredExecutor}` until a forjar-declared leg exists | with APR-DECIDE-001 DEC-0; contract only |
| J-2 | `apr` runs it. Needs the Gemma 4 family (0 hits, G2), a third kind (`Classifier`), a head loader and the media towers. 12 B parameters × 2 bytes ≈ 24 GB `[C]`: too tight for a 24 GB card, fits gx10 | not before 0.74 (D-5) |

**Track L — LLM2Vec, a recipe that turns a decoder into an encoder `[X]`.** Three steps: bidirectional attention, masked next-token prediction, contrastive training.

| Row | What | When |
|---|---|---|
| L-1 | EG-7 builds the embedding objective (pooling, contrastive loss, bidirectional backward) with the backbone as a parameter. L-1 adds only the masked next-token objective (0 hits, G9) and the mask switch | one row under E8, 0.75 |
| L-2 | Qwen 3.5 is a hybrid of recurrent and attention layers `[A]` (APR-DECIDE-001, Project copy). A mask cannot make a recurrent layer bidirectional. First target: a dense attention rung the ladder already certifies. Qwen 3.5 gets a measured attention-layers-only arm, published whichever way it falls | after L-1 |

## Appendix C — operator decisions (the default is what this spec assumes)

| # | Decision | Default |
|---|---|---|
| D-1 | Where the 0.72 scope lives. `main` maps 0.72 to Agent Ready and excludes new model support (G13) | a new "Embeddings" entry on train 0.72 in `docs/roadmaps/epics.yaml`, budget 6 `[A]`, plus a 0.72 line in the §1.5 table (G14) naming EG-0..EG-6 |
| D-2 | The cut line | EG-0..EG-6 are must-carry: 34 receipt cells and 5 refusals. EG-7..EG-9 ship in the first train where their bar is green; until then their verbs refuse with `removed_by` |
| D-3 | The oracle pin (a gate change) | one pin, moved once at the 0.72 open to the upstream merge of PR 30054 or later. A bridge receipt measures the existing parity protocol at the old and the new pin back to back on one host, so ratios stay linkable. No second pin |
| D-4 | Images, audio, video | refuse by name in 0.72. `removed_by` is set when a GGUF that holds the encoders and an oracle that serves them both exist |
| D-5 | Tracks J and L; the architecture that 0.74's "real addition" measures | J-1 now, contract only. J-2 and L wait. Propose the Gemma 4 family as that addition: this model, Jev-Omni and the Gemma 4 chat files all need it |

## Appendix D — five whys for the dominant risk (non-normative)

1. Why would `apr` return a wrong vector with HTTP 200? The route pools the hidden state of whatever model is loaded (G5).
2. Why? The route does not know what kind of model it holds.
3. Why? No model-kind type exists. "Model" means causal language model in every verb.
4. Why was that never caught? All 8 ladder rungs are Qwen causal models (G12), and the capability registry derives a causal mask for every architecture (G6).
5. **Terminal mechanism:** two contracts state one kind's truth as universal — "dim == hidden size" and "always CausalMask" — so the wrong answer would also pass the contract. Countermeasure: I-2, I-4, and EG-0 items (d) and (e).

## Appendix E — sources `[X]`

- Unsloth GGUF: https://huggingface.co/unsloth/embeddinggemma-2-GGUF · upstream GGUF: https://huggingface.co/ggml-org/embeddinggemma-2-GGUF
- Unsloth model guide: https://unsloth.ai/docs/models/embeddinggemma-2 · embedding fine-tuning guide: https://unsloth.ai/docs/basics/embedding-finetuning
- Unsloth PR 12865: https://github.com/unslothai/unsloth/pull/12865 · Unsloth README: https://github.com/unslothai/unsloth/
- llama.cpp PR 30054 (merged 2026-10-06; merge commit read as `4fbc76dec` — resolve it, do not copy it): https://github.com/ggml-org/llama.cpp/pull/30054
- llama.cpp issue 30082: https://github.com/ggml-org/llama.cpp/issues/30082
- Google developer guide (2026-10-06): https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/
- Jev-Omni card: https://huggingface.co/akhilaaa3/Jev-Omni · LLM2Vec: https://github.com/McGill-NLP/llm2vec
