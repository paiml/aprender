# APR-LAYA-TRAIN-001: a Rust trainer for the Laya decision model

**Spec id:** `APR-LAYA-TRAIN-001` · **Version:** 1.0 (2026-10-10) · **Owner:** the dedicated worker `laya-train`; the release cop owns its PR place and rows LT-00 and LT-10's judge step
**Drop at:** `docs/specifications/APR-LAYA-TRAIN-001-rust-trainer.md`, by a docs-only PR (row LT-00). The cop pushes the branch at once and the worker reads the spec there.
**Launch:** the cop spawns `laya-train` with Appendix A as its standing prompt.
**Rides:** epic #4002 (E8, train 0.75) as checklist rows `[A]`. A row merges when it is green; it does not wait for the train.
**Needs:** the PR that carries #4941 onto `main`. That carry is another worker's row and is not part of this spec. §5 names the rows that do not need it.
**Related:** `APR-EMBED-001` (row EG-7 needs the same encoder block), `APR-DECIDE-001`, `EPIC-0.75-crux-finetune-distill-plan.md`, `APR-LOOKAHEAD-002`, `FLOW-001`.
**Review:** three fresh-context lanes read the first draft: evidence against the three trees, the judge's acceptance path, and authority and execution. They returned 44 findings, 14 of which made the draft false, unsafe or unworkable. All are applied, two in a different form (Appendix B). The rewritten text was not re-read by a lane. The cop's quorum has not reviewed this file.

**Marks:** `[V]` verified by the author from public git or a public page at the sha or time named · `[C]` carried from a contract, a commit message, a review lane, a Project copy or an earlier document, not re-checked by the author · `[O]` operator statement · `[P]` proposal, not in force · `[A]` assumption · `[U]` unverified, measured in row LT-0 or LT-1.

---

## ELI5

| Question | Answer |
|---|---|
| What is asked? | Train the Laya decision model in Rust. No Python in this repo. |
| What do we start from? | PR #4941: a Rust encoder, a Rust judge and four contracts. The training half is Python and stays in the contributor's fork. |
| What is the work? | Seven equations of the fine-tune contract have no Rust enforcer. Each gets one. That is the trainer. |
| Who judges a trained model? | The verifier #4941 brings, unchanged. The trainer never judges its own run. |
| Must it match torch bit for bit? | No. It must pass the same gate. The Python run's own seed spread is a comparison band. |
| When? | Rows merge when green. The one full-size run is due on train 0.75 `[A]`. |
| What stops it? | §8. Four stops go to the operator; the rest go to the cop. |

**How it works**
- Measure first: a proxy encoder layer, forward and backward, before any model code.
- Use an autograd engine that is already on `main`. No third engine.
- Every new operation has a finite-difference gradient check and a permanent mutant test.
- The trainer writes the run directory the judge already reads.
- One full-size run, declared on `main` before it starts.

| | Yes / No |
|---|---|
| Does this spec add Python? | No |
| Does it move a threshold, a tolerance, a seed list or a recipe value? | No |
| Does it edit the judge? | No |
| Does it add an `apr` verb, a serve route or a published crate? | No |
| May the worker open a draft PR? | No. A draft runs the full CI |
| Does any row wait for the contributor? | No |

## Purpose and terms posture

- The goal is an open-source Rust ML framework. Models are not sold and are not built to compete with any provider.
- No hosted-model output is ever a training target.
- Claude Code is used only by its paying account owner: no shared credentials, no free accounts.
- Public wording never positions a model as a replacement for a commercial service.
- Laya's code and weights are Apache-2.0 `[V]`. Every ported formula cites the upstream file and commit. The new crate's README credits the upstream author and the contributor.
- The Python trainer's recorded run is an engineering comparator at pinned versions. It is not a claim against any product.

---

## §0 Operating assumptions

1. The worker `laya-train` runs this spec and nothing else. It is dedicated and is not one of the three look-ahead slots, as `apr-embed` is not `[C]`. Look-ahead rules hold for it: it never stops, it throttles, and it yields runners, queue places and review lanes to a patch release, the current train, the #4941 carry and slots 1 to 3, in that order.
2. The worker never mints a ticket, never writes a label or a milestone, and never touches a release PR, a human's PR or another worker's PR. A row rides epic #4002 as a checklist row and uses the epic's number as its ticket. The cop edits the epic body and may place the rows on another epic.
3. Every fact in §1 is a baseline. Row LT-0 re-measures it on `main`. Where this spec and a file on `main` disagree, the file wins: quote the line.
4. A `[P]` line is not in force until an operator block names it. §3 gives the one proposal a default that holds without an answer, so no row waits for it.
5. Flow per row: branch, PR, `ci / gate`; one contract or binding and one planted falsifier per feature; five-whys end in a mechanism; `pmat query` over grep; fold means move, never copy.
6. Branches. `laya-train/wip` is the tip of all work and is pushed every iteration. A *staged branch* `laya-train/<row>` holds one row, PR-ready, cut from `main` or from the staged row before it. At most three are staged `[A]`; beyond three the work stays on `wip`. A staged branch is never rebased; it takes one merge of `main` when it no longer merges cleanly. The worker holds at most one open PR, and the cop grants the place.
7. Unattended, a question goes to the cop, which decides or sends it to the review quorum (one agy lane, one Claude lane, one apr lane). Only the stops marked *operator* in §8 wait for the operator.
8. Model routing follows the fleet's routing spec `[C]`. Fable is not used `[O]`.

**Terms.** *Judge*: `aprender_decide::verify`, through its two public doors `pack_for_serving` and `verify_path`. *Run directory*: the files of the contract's `run_dir_layout`. *Inference forward*: the forward-only path #4941 brings. *Training forward*: the same function built on an autograd engine. *Engine*: an autograd implementation on `main`; *core* is `crates/aprender-core/src/autograd` with `nn/`, *train* is `crates/aprender-train/src/autograd`. *Comparator*: the Python trainer at its pins, and its one recorded run. *Tiny fixture*: `modernbert_tiny` and `laya_tiny`. *Full size*: the declared base, 421M parameters `[C]`. *Gate*: equation `gate_pass` of `laya-finetune-gate-v1`. *The seven*: the equations of G8. *Carry*: the cop's procedure that lands a contributor's commit from an in-repo branch with authorship kept and the contributor told on his PR. *LAB*: a check that prints and cannot block. *Release pass*: from a release's freeze until it is live.

---

## §1 Ground truth (baseline 2026-10-10, 08:10Z to 09:10Z; never quote as current)

Three trees were read: `paiml/aprender` `main` at `6e7ea0133`; PR #4941's head `e7730a3`; PR #4634's head `dad7f6a`. Upstream Laya was read at `4066d5d`. Nothing was built or run. The judge's two files, `verify.rs` and `pack.rs`, are byte-identical on `e7730a3` and `dad7f6a` `[C]`.

| # | Fact | Mark | Source |
|---|---|---|---|
| G1 | `main` is `6e7ea0133`, committed 2026-10-10 05:01Z. | [V] | `git rev-parse`, `git log -1` |
| G2 | #4941 is a draft from a fork with one commit, `e7730a3`. It adds `contracts/{laya-parity-v1,decide-apr-v1,laya-finetune-gate-v1}.yaml`, edits `calibration-v1.yaml`, and adds `crates/aprender-core/src/models/modernbert/` (7 files), `crates/aprender-decide/` (`publish = false`), the fixtures `modernbert_tiny` and `laya_tiny`, and `scripts/laya_train/numeric_cases.json`. It carries no `.py` under `scripts/laya_train/` and nothing under `benchmarks/tweeteval-stance/`. | [V] | `git ls-tree -r e7730a3` |
| G3 | G4 to G7 and G11 to G15 were read on `dad7f6a`, not on `main`. `main` holds none of the Laya files yet. | [V] | this spec's method |
| G4 | The Python trainer is `scripts/laya_train/`: 9 `.py` files, 4,717 lines. It pins torch 2.14.0, transformers 5.17.0, numpy 2.5.3 and upstream `laya` at `4066d5d`. Its lock resolves for macOS arm64 only. | [V] | `wc -l`; `pyproject.toml` |
| G5 | The training loop (`train.py:303-400`). Two AdamW groups: parameters named `encoder.*` at 2.5e-5, the rest at 1.0e-4; weight decay 0.01 on every parameter, including embeddings, norms, biases and `act_head`. Betas and epsilon are torch's defaults; the recipe does not pin them. Cosine annealing over `epochs × ceil(n_fit / batch)` steps to 1.0e-6. Batch 8; the last partial batch is kept. Per batch: zero the gradients, backward, clip the global norm at 1.0 over all parameters, optimizer step, scheduler step. The fit rows are shuffled each epoch from a generator seeded once per seed. Train mode; dropout exists in the head only. Loss: `cross_entropy(z, y) + mean(-proper_reward(softmax(z / T_loss), onehot)) + 0.0 × act.sum()`, with `w_sph` 0.75, `w_rps` 1.0, and `T_loss` the base's clamped temperature for the question's bucket. | [V] loop and loss; details [C] | the file; contract `recipe`; review lane |
| G6 | `proper_reward` (upstream `laya/common.py:278`): a log score with `q` clamped at 1e-12 and the log floored at −9.21, plus `w_sph` times a spherical score `(target · q) / max(‖q‖, 1e-9)`. A ranked-probability term applies only to questions of type `score`. | [V] | the file at `4066d5d` |
| G7 | The model (upstream `laya/common.py:149`): encoder, plus a type embedding, then two `nn.TransformerEncoderLayer(d, d/64 heads, 4d, dropout 0.1, norm_first)` with a padding mask, a gather at the marker positions, and a scorer (LayerNorm, Linear, GELU, Linear to 1). An `act_head` exists; the loss gives it a zero gradient, not none. The Rust head (`laya/head.rs:1`) states a ReLU feed-forward and a bias on every projection and norm. | [V] | the files |
| G8 | `laya-finetune-gate-v1` on `dad7f6a` is version 5.0.0 with 13 equations. Seven bind to the recipe `just laya-train-selftest`: `calibration_fit_bounded`, `declared_seed_ships`, `eval_set_in_distribution`, `recipe_before_scores`, `device_recorded`, `f16_reload_scoring`, `early_stopping_train_side`. Six bind to Rust. On `e7730a3` the binding file holds 6 rows for this contract and none names that recipe: the seven are unbound there, and #4941's body reports seven `BIND-001`. Ten `test_harness` lines and one `replayed_by` entry in the gate and parity contracts name that recipe or a `.py` file `[C]`. | [V]; the PR body and the count [C] | `contracts/aprender/binding.yaml`; the contracts |
| G9 | Declared values: margin at least 0.05; ECE at most 0.10 in 15 bins; temperature in [0.5, 5.0]; calibration slice 0.25 with at least 2 rows per class; seeds 13, 17, 23, the median-ECE seed ships; early stopping with patience 3, `min_delta` 0.001, first candidate epoch 1; 12 epochs at 16 shots per class or fewer, else a declared count in [4, 12]. | [V] | the contract |
| G10 | The judge's readers refuse unknown fields: 18 structs in `pack.rs`, `Recipe` and `GateReport` among them, carry `#[serde(deny_unknown_fields)]`. `GateReport.torch_version` is a required string; the contract lists it as report-only: "it enters no verdict". `device_is_cpu == (device_used == "cpu")` is checked. `rescore_noise_sha256` is optional; absent, both re-scores are held to 1.0e-5. | [V] | `pack.rs:57`, `:317-358`; `verify.rs:1805`, `:2229`; contract `run_field_bindings` |
| G11 | The judge's doors are public and take a policy: `pack_for_serving(run_dir, data_dir, base_dir, out, &VerifyPolicy)` and `verify_path(apr, run_dir, data_dir, base_dir, &VerifyPolicy)`. `VerifyPolicy` has public fields, the base pins among them. Both doors call `check_variant` first. `pack::load_checkpoint_for_scoring(checkpoint_dir, task)` is public and returns the scorer the judge uses. | [V] signatures; the call order [C] | `verify.rs:96`, `:124`, `:3397`, `:3429`; `pack.rs:802` |
| G12 | The judge re-scores the packed bytes and the declared base with the inference forward and recomputes every gate metric in f64 with exactly-rounded sums. The packed weights are a byte copy of the F16 checkpoint. The forward's banding reads the thread count, and its bits move with the GEMM kernel and the host's math library. | [C] | `verify.rs`; `pack.rs:826-843`; `modernbert/gemm.rs:56` |
| G13 | Calibration split: Python draws with `numpy.random.RandomState(seed).permutation` over text groups per class. The judge checks the rule, not the draw. Temperature fit: a bisection on `1/T`, 200 iterations, f64. Early stopping: best-anchored. | [V] | `data.py:283`; `verify.rs:1707-1790`; `gate.py:65`, `:94` |
| G14 | The comparator's one recorded run: TweetEval `stance_abortion`, 64 shots per class, 459 eval rows with class counts [111, 291, 57], a 280-row shift probe; device `mps:0`; shipped seed 17. Per seed: margin 0.1439, 0.2217, 0.2478; ECE 0.0850, 0.0443, 0.0372; all three pass. Zero-shot macro-F1 0.4092. Fine-tuned re-score max abs 1.13e-5. Per-seed macro-F1 0.5530, 0.6308, 0.6570 and `wall_seconds` 684 are in the fork's evidence file only; what that clock covers is `[U]`. | [V] | contract `demo_s64.outcome_record`; `08-GATE-RUN-EVIDENCE.json` on `dad7f6a` |
| G15 | The contract records that MPS training is not bitwise reproducible: one replicate turned a FAIL into a PASS and ECE moved by up to 0.048. It scopes `logits_abs` and `final_norm_abs` FIXTURE-ONLY: torch's own fp32 logits sat up to 2.8e-3 from a float64 forward on real fine-tuned rows. | [V] | `seed_policy.honesty`; `laya-parity-v1:145-150` |
| G16 | Two autograd engines are on `main`. **Core:** shaped tensors; `matmul` (two-dimensional, GEMM-backed through `trueno` above 64 `[C]`), `softmax`, `gelu`, `gelu_exact`, `relu`, `embedding_gather` with a scatter-add backward `[C]`, `additive_attention_mask`, `log`, `exp`, `sqrt`, `div`; `nn::MultiHeadAttention::forward_self(x, attn_mask)` over `[batch, len, dim]`; a pre-norm `nn::TransformerEncoderLayer` with dropout and a tanh GELU; `Dropout::with_seed`; `Adam`, `AdamW` (one rate; it skips parameters without a gradient `[C]`), `CosineAnnealingLR`; a differentiable MiniLM encoder, `setfit/encoder.rs`, built on it. **Train:** flat f32 tensors and one sequence per forward; `matmul`, `layer_norm`, `gelu`, `softmax`, `attention` (full or causal); a private rotate-half `apply_rope` with a backward; `AdamW` (one rate); `CosineAnnealingLR`; `clip_grad_norm`; CUDA and wgpu backward paths; a post-norm BERT `EncoderModel`; `ClassifyTrainer`, which saves a head and LoRA adapters. | [V] names and signatures; no behaviour | `crates/aprender-core/src/{autograd,nn,setfit}`; `crates/aprender-train/src/{autograd,optim,transformer,finetune}` |
| G17 | Absent on both engines: a differentiable slice, chunk or concatenation, and a clamp. Absent in core: gradient clipping, and a rotate-half RoPE with a backward (core's `RotaryPositionEmbedding` is interleaved and returns a tensor without a gradient function). | [C] | review lane, with `attention_helpers.rs:202`, `:311`; `optim/clip.rs:23` |
| G18 | Encoders on `main`: `aprender-core/src/models/bert/`, `aprender-core/src/setfit/encoder.rs` and `aprender-train/src/transformer/encoder.rs`. #4941 adds a fourth, forward only; its module comment says it is not built on `models::bert` or `autograd`. | [V] | the trees; `modernbert/mod.rs:12` |
| G19 | Among workspace crates `aprender-decide` depends on `aprender-core` and `provable-contracts-macros` only. `aprender-train` depends on `aprender-core`. | [V] | both `Cargo.toml` |
| G20 | `aprender-decide`'s lib test `safetensors_is_read_only_by_pack` scans every `.rs` under its `src/` and fails on any file but `src/pack.rs` that names the safetensors crate (`src/test_support.rs` is allowed). `pack.rs`'s record structs derive `Deserialize` only `[C]`. | [V] | `src/lib.rs:342` on `e7730a3` |
| G21 | `docs/roadmaps/epics.yaml`: E8 "CRUX Fine-Tune/Distill", issue 4002, train 0.75, budget 3, absorbs none. The 0.75 plan passes a cell on quality when it is at least the best competitor's minus a band, "where the band comes from each engine's own seed-to-seed spread (3 seeds), never chosen by hand". It also has a cost clause, and its cells are Qwen 3.5 with LoRA, QLoRA and distillation. | [V] | the two files on `main` |
| G22 | `main` holds 170 `.py` files. | [V] | `git ls-tree -r --name-only origin/main \| grep -c '\.py$'` |
| G23 | Upstream Laya code is Apache-2.0. The weights `convaiinnovations/laya` are tagged `apache-2.0`. The licence of the TweetEval files is `[U]`. | [V]; [U] | the two pages, 10-10 |
| G24 | The operator on #4634: "Any training code has to fit in .71 to .75" and "binary always stays pure Rust" (09-29). | [V] | PR page |
| G25 | Open PRs 12 against a cap of 10. Open issues 461 against a cap of 100. | [V] counts; caps [C] | repository pages 08:00Z; `FLOW-001` |
| G26 | Inference takes 226 ms at 85 tokens on 6 threads of an Apple M4-class machine. No Rust training step has been timed. | [C]; [U] | spike 025 |
| G27 | `main` has no path under `benchmarks/tweeteval-stance/` and its CLI contract names no `tweet-eval-stance` command. The demo's selection manifest and its benchmark contract are on #4634 only. | [V] | `git ls-tree`; `contracts/apr-cli-commands-v1.yaml` |

**What these facts change**

| # | Finding | Consequence |
|---|---|---|
| F1 | G8, G11, G12: the judge exists, is Rust and is public. | The trainer is accepted by the judge's verdict. No judge is written or edited. |
| F2 | G15: the comparator does not reproduce itself to the bit. | Parity with torch is not a bit test. It is the gate, plus the band of G21. |
| F3 | G10: `torch_version` is a required string that enters no verdict. | A Rust run writes `none` there and records its engine in `rl_agent_config.json` `training`, which the judge does not read. No reader changes. |
| F4 | G11, G12: a trainer that scores through `load_checkpoint_for_scoring` on its saved F16 bytes computes what the judge re-scores. | On one host, binary and thread count the re-score matches to the bit `[C]`, and the float64 noise record stays absent. Across hosts it may not; that check is LAB. |
| F5 | G16, G19: core already trains an encoder, holds masks, batches and seeded dropout, and adds no dependency edge. Its speed at full size is unknown. | LT-1 measures both engines before any model code. |
| F6 | G7, G16: the head needs a ReLU feed-forward; core's layer uses a tanh GELU. | The head is a variant of core's layer, not a reuse of it. |
| F7 | G13: the judge checks the split's rule, not its draw. | Rust declares its own generator. No numpy generator is ported. |
| F8 | G20: a second safetensors user under `aprender-decide/src/` fails a required test. | The trainer is its own unpublished crate. Cargo then enforces the fence of H2. |
| F9 | G2, G27: the demo's data rule and manifest are not in #4941. | The cop asks for that slice at LT-00. LT-8 tests the rule on a synthetic fixture. |

---

## §2 Hard rules

| # | Rule |
|---|---|
| H1 | **No Python.** No row adds a `.py`, `pyproject.toml` or `uv.lock`, or a line in a Makefile, a `justfile`, a script or a workflow that runs `python`, `uv` or `pip`. No test of this spec needs a run directory that Python must write. Frozen data already on `main` with its provenance may be read. |
| H2 | **The judge is not edited.** No row changes a file under `crates/aprender-decide/` or `crates/aprender-core/src/models/modernbert/`, their fixtures, or `scripts/laya_train/numeric_cases.json`. The trainer uses the judge's public API only. A needed item that is not public is stop 3. |
| H3 | **Nothing declared moves.** No threshold, tolerance, seed list, recipe value, bin count, calibration bound or eval rule of the four contracts changes. A row may add a binding, rebind an enforcer from a Python recipe to a Rust test, reword prose that names Python as a source, and add an equation with its enforcer. |
| H4 | **Measure first.** No model code merges before LT-1's receipt is on `main`. No full-size run starts before LT-9's declaration is on `main`. |
| H5 | **No third engine.** The trainer uses the engine LT-1 chooses. A missing operation is added to that engine with its backward. |
| H6 | **One trainable ModernBERT block.** It is written once, in the chosen engine's crate, with no Laya name in it. Before writing it the worker runs `pmat query` for an existing one and sends the block's signature to the `apr-embed` worker through the cop. It does not wait for a reply. |
| H7 | **Every reported score comes from the judge's scorer on the saved bytes.** Scores are taken through `load_checkpoint_for_scoring` after the fitted temperature is written. The training forward computes gradients, the early-stopping monitor and the check of H8. |
| H8 | **Forward agreement is measured in every run.** After training, the training forward in eval mode and the inference forward are compared on the first `min(5, n)` calibration rows `[A]`, on F16-reloaded weights. On the tiny fixture a logit difference above `logits_abs` is `REFUSED forward-agreement`, exit 2, with no gate report. At full size the number is recorded and is LAB (G15). |
| H9 | **Same inputs, same bytes.** On the tiny fixture, the same seed, host, binary and thread count give a byte-identical `model.safetensors`. At full size the replicate is LAB. |
| H10 | **One declared run.** The full-size cell is written to `main` before the run. One run. Its outcome is recorded whatever it is. A run cut short by a release pass or a host failure is `preempted` and restarts under the same declaration. A second run after an outcome needs a new declaration that names a merged fix. |
| H11 | **No new public surface.** No `apr` verb, no CLI-contract edit, no serve route, no published crate, no change to any `publish` setting, no required check. One unpublished crate is added: `crates/aprender-decide-train`. |
| H12 | **Large or licensed bytes stay out of git.** Base weights, run directories and tweet text never enter the repo. A host gets them through forjar-declared resources, anonymously, with no token. |
| H13 | **A contract lands with its enforcer.** An equation, its binding and its falsifier arrive in one PR. |
| H14 | **A falsifier is a permanent test.** Each plant is a test that applies the mutant and asserts the refusal. It runs in the required check on every PR. No PR is pushed red on purpose. |
| H15 | Never: `--allow-dirty`; a force-push; a tag delete or re-tag; a publish; a waiver; provisioning outside `forjar apply` or a make target; ad-hoc SSH; a receipt for a run that did not happen. |

---

## §3 Operator statements and one open proposal

**Stated by the operator, 2026-10-10 `[O]`:** "if we never want python in our project, ever" · "why not use his pull request as a spec to enhance A" · "lets build into a comprehensive spec and have a dedicated \"look ahead\" worker that will build this".

**Read from that, by the author `[A]`:** the trainer is Rust in this repo; the Python trainer stays in the contributor's fork; one dedicated worker outside the three slots builds it; #4002 is its epic until the cop places it elsewhere.

| # | Question | Options | Why the first |
|---|---|---|---|
| P1 | Beyond the judge's verdict, must the one declared run land inside the comparator's band? | **BAND**: the shipped seed's macro-F1 is at least the comparator's shipped macro-F1 minus the comparator's own three-seed range. At G14's values: 0.63082 − (0.65700 − 0.55305) = 0.52687. · **VERDICT**: the judge's verdict alone; the band line prints as LAB. | It is the quality clause of the 0.75 plan (G21) applied to one cell. The plan's cost clause is not applied: the comparator ran on a GPU-class device and this run is on a CPU; cost is recorded. Three seeds make a coarse band, and the receipt says so. |

**Default, in force without an answer:** VERDICT. LT-9 declares it and LT-10 prints the band line. An answer of BAND before LT-9 merges changes the declaration.

---

## §4 Rows, in order of expected value

One row is one PR. A row is **done** when `git cat-file -e origin/main:evidence/laya-train/<row>/receipt.json` exits 0. A receipt names the row, the tests that enforce it, the command, the host and the commit it ran on. LT-00 is done when its two files are on `main`.

### Phase 0: read and measure

| Row | Work |
|---|---|
| **LT-00** | The cop's row. Push branch `docs/apr-laya-train-001` at once with this file and with Appendix A as `docs/prompts/laya-train-worker.md`; open the docs-only PR when a place is free. Add the checklist rows to #4002. Post the operator's comment on #4634, which asks the contributor for the stance-benchmark slice. File one `infra` issue for the resources of H12: the four pinned base files and the dataset files. |
| **LT-0** | First reads, from commands, in one report. (a) Re-measure G1, G2, G8, G10, G11, G16 to G20, G22 and G27 on `main`. (b) Say whether the carry of #4941 is merged, which cited files differ from `dad7f6a`, and whether the carry kept the seven in the contract or moved them out. (c) The operation table: every operation the training forward of G5 to G7 needs, and for each engine *present with backward*, *present without* or *absent*, with file and line. (d) The judge table: Appendix C re-read from the code, the public items the trainer will call, and which of the seven the judge re-checks. A needed item that is not public is stop 3, asked now. (e) How a command is run on a CPU pool host today: a make target or a dispatch workflow, named. If none fits, LT-1's PR adds one workflow that starts on `workflow_dispatch` only. (f) The licence of the dataset files. (g) The rule checker: `scripts/check_laya_train_rules.sh`, bash with awk, printing one line per rule of §9 that is a command. The cop runs it on each PR of this spec before the queue is armed. (h) Three questions, sent to the cop with their defaults: the surface (default: the example stays, no verb); the GPU path (default: CPU only in this spec); whether E8's recipe-schema row takes this contract's recipe block as its first instance (default: no coupling). |
| **LT-1** | Engine measurement and choice. A **proxy layer** built only from operations present on both engines today: the attention and feed-forward matrix products of one encoder layer at the shapes of the base's `encoder/config.json`, batch 8, length 128 `[A]`, random weights. On one CPU pool host, outside a release pass: forward wall, backward wall and peak memory on each engine, and the inference forward of one real layer for scale. Print the projected wall of the declared cell for each engine. **Choice rule, fixed before measuring:** core, unless its projected wall is more than twice train's `[A]`. The receipt is provisional and satisfies H4; LT-3 re-measures with the real block. A second host is LAB. |

### Phase 1: the pieces (tiny fixture only; each fits PR CI)

| Row | Work | Binds to | Mutant tests (H14) |
|---|---|---|---|
| **LT-2** | Every operation LT-0 marked *absent* or *present without*, added to the chosen engine with its backward. Expected for core (G17): slice or chunk, clamp, gradient clipping, a rotate-half RoPE with a backward, the additive mask for the local window and for padding. | one new contract, name proposed `modernbert-train-v1` | Per operation, a central finite difference in f64 disagrees with a wrong-sign backward. |
| **LT-3** | The trainable encoder block and encoder (H6): pre-norm, bias-free norms, identity attention norm on layer 0, fused `Wqkv`, rotate-half RoPE with the global and the local theta, the inclusive window `\|i − j\| ≤ local_attention / 2`, the gated feed-forward `gelu_exact(input) × gate`, the final norm. It loads the tensor names of `expected_modernbert_tensor_names`. It re-measures LT-1 with the real block. | `train_forward_matches_inference`, `every_parameter_receives_gradient` (same contract); tolerances are `laya-parity-v1`'s `per_layer_rel_rms` and `final_norm_abs`, by reference | The window mutants, half-window minus one and plus one, each fail. The test first asserts a fixture row longer than the window exists. One detached edge leaves a parameter with a zero gradient. |
| **LT-4** | The trainable head: type embedding, two pre-norm layers with a ReLU feed-forward, biases, a padding mask and seeded dropout 0.1, the marker gather, the scorer. `act_head` and `temperature` are loaded and carried. Eval mode applies no dropout. | `train_logits_match_inference` (new contract, name proposed `laya-train-v1`): against the inference forward on every `laya_tiny` row, within `logits_abs` | Dropout left on in eval mode. ReLU swapped for GELU. |
| **LT-5** | The loss of G5 and G6 for `choice` questions, on raw logits, with `T_loss` from the base. A `score` question is refused by name, `REFUSED qtype-not-supported`. | `finetune_loss` (`laya-train-v1`). Expected values are closed forms written in the contract with their derivation: a uniform `q` over K gives a log score of `ln(1/K)` and a spherical score of `1/√K`; a one-hot `q` on the target gives 0 and 1. | A flipped sign on the reward term. `w_sph` applied to the log term. |
| **LT-6** | The optimizer of G5: two groups split on the name prefix `encoder.`; decoupled weight decay on every parameter, including one whose gradient is zero; one cosine schedule per group over the planned steps, never re-planned when training stops early; the clip before the step; torch's default betas and epsilon `[C]`, recorded in the run's provenance. The PR states each difference from torch's update rule that it finds, with the line. | `optimizer_update` (`laya-train-v1`). Closed forms: a zero-gradient parameter after n steps equals its start times `Π(1 − lr_t × 0.01)`; the first step moves a parameter by its group's rate regardless of the gradient's size, up to epsilon. | Both groups at one rate. Clip after the step. A schedule re-planned at an early stop. Decay skipped on a zero-gradient parameter. |

### Phase 2: the loop and the seven

| Row | Work |
|---|---|
| **LT-7** | The loop, in `crates/aprender-decide-train` (`publish = false`), with one example binary as its entry. In order: read the data directory with the judge's public reader; split by text group with a declared generator; write `recipe.json` and print its id before any score; per seed train, monitor the calibration NLL at the bounded fitted temperature in eval mode, restore the best epoch, save F16 with an F32 `temperature`, reload, fit the temperature by the bisection of G13, write it into `rl_agent_config.json`, reload, score eval through H7; score the base once; choose the median-ECE seed; delete the other checkpoints; write `probes.json`, the shift probe when `shift.jsonl` is present, `variance-report.json` and `gate-report.json`. A non-empty output directory is refused. Appendix C lists what the judge binds. `torch_version` is `none`; `device_used` is `cpu`; the engine, its version, the build sha, the generator, the thread count and the optimizer constants go under `training` in `rl_agent_config.json`. If the carry moved the seven out of the contract, their text is restored from #4634 unchanged, with the contributor's `Co-authored-by` line. |
| | **Binds:** six of the seven, each to a function of the new crate, each with a mutant test: a score written before `recipe.json`; a temperature applied outside its bounds; the last epoch restored instead of the best; a score taken from unsaved fp32 weights; a device string that disagrees with its flag; a declared seed that does not ship on a single-seed run. Where LT-0's judge table shows the judge re-checks an equation, the same plant is also given to the judge's door and must be refused there. The contract's own stopping case `m = [1.0, 0.9993, 0.9988]` keeps epoch 3. The fit agrees with a 1e-4 grid search written in the test. |
| | **Tiny acceptance:** a three-seed tiny run, declared `production` on the tiny base, is given to `pack_for_serving` and then `verify_path` under a `VerifyPolicy` whose base pins are the tiny fixture's. The verdict is acceptance or `GateFailed`; any other refusal is red. The same run declared `synthetic-fixture` is refused by both doors as `SyntheticNotDeployable`. H8 and H9 run here. |
| | **Rebinds:** every `test`, `test_harness`, `replayed_by` and binding in the four contracts that names a `.py` file or a `just` recipe now names a Rust test, except those of `eval_set_in_distribution`. A generator may be named only as provenance, with a commit and a sha256. `pv audit` on `laya-finetune-gate-v1` prints at most one `BIND-001`. |
| **LT-8** | The demo data rule in Rust: `eval_set_in_distribution` as a pure function, and a preparation example that reads the dataset files from a forjar-declared path and writes a data directory outside the repo. **Binds:** `eval_set_in_distribution`. Its tests use a synthetic fixture with planted shots, exclusions, group members and duplicates, and no tweet text. Mutants: one shot left in the pool; a duplicate kept. Then `pv audit` prints 0 `BIND-001`. The real counts, 459 rows as [111, 291, 57], are a preflight of LT-10. The manifest and the benchmark contract arrive by the contributor's slice or by a carry; the worker does not copy them. |

### Phase 3: full size

| Row | Work |
|---|---|
| **LT-9** | The declaration (H10), as `evidence/laya-train/lt-9/declaration.json` and not as a contract edit: the data directory's hashes, seeds 13, 17, 23, the epoch limit and stopping rule of `demo_s64`, the host, the thread count, the binary sha, wall, memory and disk caps set at twice LT-3's projection `[A]`, the pass rule of §3, the seed-13 replicate, and each named difference from `demo_s64`: engine, device, generator, no noise record. |
| **LT-10** | The one run, through the path LT-0 named, never in PR CI and never during a release pass. Preflight: the four base hashes, the data counts, the caps. Then the cop, not the worker, starts the judge on the same host, binary and thread count: `pack_for_serving`, then `verify_path`. Seed 13 is trained a second time and the two `model.safetensors` hashes are compared (LAB). A verify on a second host is LAB. The outcome is written once: `gate_pass`, `gate_fail`, `pack_refused` or `aborted`, with wall by phase, peak memory, the per-seed rows, the judge's verdict, the band line, H8's number and both hashes. |
| **LT-11** | LAB, never blocks. If the contributor supplies one gradient fixture for the tiny model as JSON, with its generator's commit named, a test compares it and its tolerance is set from the first record. Otherwise the line reads `not_supplied`. |

---

## §5 Phases

| Phase | Rows | Needs |
|---|---|---|
| 0 | LT-00, LT-0, LT-1 | nothing. LT-0 and LT-1 do not wait for LT-00's merge. LT-1 needs a pool host outside a release pass. |
| 1a | LT-2 | LT-1's receipt |
| 1b | LT-3, LT-4, LT-5, LT-6 | LT-2; the carry of #4941 on `main` |
| 2 | LT-7, LT-8 | LT-3 to LT-6. LT-8's code needs nothing but the carry. |
| 3 | LT-9, LT-10 | LT-7, LT-8; the `infra` resources; the manifest on `main` |
| any | LT-11 | LT-4 |

While a phase waits, the worker prepares the next row on `laya-train/wip`. It is never idle.

---

## §6 Definitions of done

- **A row:** its receipt is on `main` (§4), and its mutant tests run in the required check.
- **The seven:** `pv audit` prints 0 `BIND-001` for `laya-finetune-gate-v1`, and no enforcer in the four contracts is Python.
- **This spec:** LT-10's outcome is on `main`, whatever it is; the seven are done; the rule checker prints 0 for Python added.
- **The public sentence "aprender trains Laya":** only on `gate_pass` with the judge's acceptance of the exact bytes.
- A criterion is never waived. It moves only by an operator answer.

---

## §7 Budget and routing

| Rows | K̂ (min) `[A]` | Note |
|---|---|---|
| LT-0, LT-1 | 90, 120 | LT-1's host time is extra |
| LT-2 to LT-6 | 300, 300, 180, 120, 150 | LT-2 grows with the operation table |
| LT-7, LT-8 | 420, 150 | |
| LT-9, LT-10 | 60, 240 | LT-10's run wall is extra and unmeasured |

**K̂ = 2,130 · K = 1.65 × K̂ = 3,515 · andon at 0.8 K = 2,812** `[A]`; the multipliers are the fleet's convention `[C]`. The cop re-sizes each row at ticketing.

- **Hosts.** PR CI runs the tiny fixture only. LT-1 and LT-10 run on one CPU pool host named in the receipt. This spec's host jobs use no GPU host and no clean-room runner.
- **Throttle.** The worker throttles with slot 3: at 70% of the 5-hour budget it drops to a heartbeat and `laya-train/wip` `[C]`.
- **Andon.** At a row's andon the worker stops that row and writes the five-whys. The cop splits or re-plans it.
- **PR age.** A PR of this spec open for 24 hours is an andon (`FLOW-001`).

---

## §8 Stop conditions

One numbered question, options in words. The worker keeps working on other rows.

| # | Stop | Answered by |
|---|---|---|
| 1 | A row cannot be done without Python in this repo or in a CI step. | operator |
| 2 | A declared value would have to move (H3). | operator |
| 3 | The judge would have to be edited, or the trainer needs an item of it that is not public (H2). | operator |
| 4 | Anything H11 forbids, or a second new crate. | operator |
| 5 | LT-1 or LT-3 projects more than 24 hours `[A]` for the declared cell on both engines. The options name the GPU path and a smaller declared cell. | cop |
| 6 | LT-10 ends `gate_fail` or `pack_refused`. The options are a new declaration after a named, merged fix, or closing the spec on the recorded result. | cop, reported to the operator |
| 7 | The carry of #4941 is closed unmerged, or its merged tree lacks a file a row binds to. | cop |
| 8 | A resource of H12 cannot be declared through forjar, or the dataset's licence does not allow this use. | cop |
| 9 | A PR fails two review quorums with a change between the rounds. | cop |

**Not stops. The worker decides, acts and reports.**
- A missing ticket: the row rides #4002's number and the cop settles it.
- No PR place: the work waits staged, or on `wip` beyond three.
- A red CI run: fix it on a new commit.
- An absent operation: LT-2 adds it.
- A generated file another PR also wrote: regenerate once on top of `main`.
- A silent contributor or a silent `apr-embed`: proceed.

**Never, with or without a question:** H15; a write to #4941, #4634 or any contributor branch; a host job of this spec on a GPU host or the clean-room pool; a host job during a release pass.

---

## §9 Falsifiers

Rows marked *cmd* are lines of the rule checker (LT-0 g).

| Rule | Falsifier | Anti-vacuity arm |
|---|---|---|
| H1 *cmd* | Added files matching `*.py`, `pyproject.toml`, `uv.lock`, or added lines matching `python\|uv run\|pip ` in a Makefile, `justfile`, script or workflow, in `git diff origin/main...<head>`: red. | A planted `x.py` and a planted `python3` line are each named. |
| H2 *cmd* | `git diff --name-only origin/main...<head>` lists a path of H2: red. | A planted edit of `verify.rs` is named. |
| H3 | `pv diff` on each of the four contracts, base against head, reports a changed constant, tolerance or formula: red. | A planted `gate_max_ece: 0.11` is reported; an added binding is not. |
| H4 *cmd* | A PR of this spec that touches `crates/` while `evidence/laya-train/lt-1/receipt.json` is absent on `main`: red. | The path is printed with its presence. |
| H5 *cmd* | The new crate's `Cargo.toml` and sources name no autograd engine but the chosen one; a second `Tensor` type with a `backward` in the diff: red. | The chosen engine's import is found. |
| H6 | `pmat query` for the gated feed-forward with a backward returns more than one definition in the chosen engine's crate: red. | It returns one. |
| H7 | A mutant that scores from the unsaved fp32 weights is refused by the judge's door on the tiny run. | The clean tiny run is re-scored with max abs 0 on the same host. |
| H8 | A mutant that scales one head weight by 1.001 in the training forward only prints `REFUSED forward-agreement`. | The clean tiny run prints a difference at or below `logits_abs`; the two forwards are separate functions. |
| H9 | Two tiny runs with one seed differ in `model.safetensors` sha256: red. | A second seed gives a different sha256. |
| H10 *cmd* | An outcome whose run began before its declaration's merge commit, or two outcomes for one declaration: red. | The declaration's merge commit is printed. |
| H11 *cmd* | The diff lists `contracts/apr-cli-commands-v1.yaml`, a `publish` change, a second new `crates/*/Cargo.toml`, or a workflow with a trigger other than `workflow_dispatch`: red. | The one new crate is listed. |
| H12 *cmd* | An added file above 1 MiB `[A]`: red. | The check prints the largest added file. |
| H13 | `pv audit` on a PR's head reports an equation added by that PR without a binding: red. | It prints the bound count, above 0. |
| H14 | A falsifier of §4 with no test that applies its mutant: named by the row's receipt check. | The receipt lists each mutant test by name. |
| H15 *cmd* | `--allow-dirty`, a publish command or `ssh ` added to a script or workflow: red. | A planted line is named. |
| The seven | LT-7's and LT-8's mutant tests. | Each prints the refusal it expects. |
| P1 | Under BAND, a planted shipped macro-F1 of 0.50 fails. | The comparator's three values carry their source sha in the declaration. |

**What would prove this spec wrong**

- LT-0 finds the carry changed the judge or the contract so that G8, G10 or G11 no longer holds: the rows are re-read against it.
- The public API cannot build a tiny-base policy, or cannot score a checkpoint: stop 3, and F1 is weaker than stated.
- LT-1 shows neither engine can hold one full-size layer's backward in memory on a pool host: Phase 3 needs a GPU path.
- The training forward cannot be brought within `logits_abs` of the inference forward on the tiny fixture: the two differ by design, and H8's bar is asked, never moved.
- The declared run fails the gate on all three seeds while the comparator passed on all three: the recipe does not transfer, and the difference is in LT-6's stated deviations or in the generator.
- The re-score on the training host is not bit-equal: F4 is wrong, and a float64 noise record becomes a row.

---

## §10 Report

To the cop, five lines or fewer, at the end of every iteration. Nightly, one line:

```
LAYA-TRAIN <date> | row=<id> <state> | PR=<n|none> staged=<k>/3 | bind001=<n>/7 | py_added=<n> | ops_absent=<n> | engine=<core|train|undecided> | fwd_agree_max=<x|unmeasured> | step_s=<x|unmeasured> | asked=<ids> | next=<row>
```

At the end, the scorecard: rows merged; minutes against K̂; engine and why; the seven bound; Python added; the declared run's outcome, wall and peak memory; the band line; questions asked and when.

---

## Appendix A: worker prompt (LT-00 writes it to `docs/prompts/laya-train-worker.md`)

```
You are `laya-train`, the dedicated worker for
docs/specifications/APR-LAYA-TRAIN-001-rust-trainer.md. Until that file is on
main, read it on branch docs/apr-laya-train-001.
You are not one of the three look-ahead slots. You build a Rust trainer for the
Laya decision model. You never stop; under pressure you throttle.

Loop (live state on origin/main wins over memory):
 1. Read the spec, the checklist on the epic the cop named, your open PR and your
    staged branches. Write a heartbeat.
 2. During a release pass, or when throttled: work on laya-train/wip only.
    No PR, no host job.
 3. Take the first row of spec §4 that is not done and whose needs in §5 are met.
    Done means: git cat-file -e origin/main:evidence/laya-train/<row>/receipt.json
    exits 0. One row per iteration. If no row is ready, prepare the next one on
    laya-train/wip.
 4. Work in your own worktree. Write the mutant test first and see it fail on the
    unfinished code; then make the feature pass. Every mutant test stays in the
    suite. One contract or binding per feature. pmat query over grep.
 5. One row is one PR, from a staged branch laya-train/<row> (at most 3 staged;
    beyond that stay on wip). Never a draft PR. Never push a PR red on purpose.
    The cop grants your one PR place and runs the rule checker before you arm
    the merge queue. You merge your own PR. A PR open 24 hours is an andon.
 6. Push laya-train/wip at the end of every iteration. A staged branch takes a
    merge of main, never a rebase.
 7. Ask a question the moment you find it: one numbered question, options in
    words, with your default, to the cop. Then take another row.
 8. Report to the cop in five lines or fewer.

Never:
 - add Python, or a line that runs python, uv or pip, or a test that needs a
   Python-written run directory;
 - change any file under crates/aprender-decide/ or
   crates/aprender-core/src/models/modernbert/, their fixtures, or
   scripts/laya_train/numeric_cases.json; use the judge's public API only;
 - move a threshold, tolerance, seed list or recipe value in any contract;
 - write a third autograd engine or a second trainable ModernBERT block;
 - add an `apr` verb, a CLI-contract edit, a serve route, a published crate, a
   second new crate, a required check, or change a `publish` setting;
 - merge model code before LT-1's receipt is on main, or start the full-size
   run before LT-9's declaration is on main; run it once; the cop starts the
   judge, not you;
 - mint a ticket, write a label or a milestone, or touch a release PR, a
   human's PR or branch (#4941, #4634), or another worker's PR; copy commits
   between open PRs;
 - force-push, delete or archive a ref, delete or move a tag, publish, use
   --allow-dirty, ask for a waiver, or use ad-hoc SSH;
 - put weights, run directories or tweet text in git, or download with a token;
 - start a host job on a GPU host or the clean-room pool, or any host job
   during a release pass; use the Fable model.

Stops: spec §8. A stop is one question; every other row keeps moving.
```

## Appendix B: not checked, and what the review changed

**Not checked**
- Nothing was compiled or run. Every `[V]` is a file, a line, a count or a page.
- G4 to G7 and G11 to G15 were read on #4634's head, not on `main`.
- Facts marked `[C]` from a review lane were not re-read by the author, except G10's report-only line, G11's signatures, G15's scope, G18's fourth encoder and G20's test.
- Labels, the epic's body, runner health, live PR counts after 08:00Z and the private `infra` repository were not readable.
- `APR-LOOKAHEAD-002` and `APR-071` were read as Project copies; what is in force is the cop's to say.
- The K̂ figures are the author's, with no measured row behind them.

**What the three lanes changed**
- The judge is not edited at all. The draft proposed a reader change for provenance; the judge lane showed `torch_version` enters no verdict and that the change would reach the load ladder, the field-binding table and every built server.
- The trainer moved from a module of `aprender-decide` to its own unpublished crate (G20).
- The tiny acceptance now uses the judge's two public doors under a tiny-base policy. The draft's wording could not be met: both doors stop at `check_variant`.
- H8's bar is fixture-only; at full size it is recorded. The draft's bar could have aborted the one declared run (G15).
- "To the bit" is scoped to one host, binary and thread count (G12). The cop starts the judge.
- The checkpoint carries `act_head` and `temperature`; the order is save, reload, fit, write, reload, score; the shift probe is written (Appendix C).
- The declaration and the outcome live under `evidence/`, not in the contract. A preempted run restarts.
- LT-1 measures a proxy layer, so it no longer waits on LT-2. LT-5 and LT-6 moved behind the carry.
- LT-8 tests the data rule on a synthetic fixture; the real counts are a preflight.
- Each row has one done test. Falsifiers are permanent mutant tests. A rule checker lands in LT-0.
- Stops name who answers. The worker prompt gained the nevers the draft lacked.
- **Taken in a different form:** the lanes asked for a second author for the loss and optimizer reference values; this spec uses closed forms written in the contract instead. They asked that the seven bind to judge-fed tests only; this spec binds each to a trainer function and also feeds the plant to the judge wherever the judge re-checks it.

## Appendix C: what the judge binds in a run directory `[C]`

LT-0 re-reads this from the code; the code wins.

- **Checkpoint tensors:** a bijection with the artifact's expected set: `encoder.*`, the head layers, `type_emb.weight`, `scorer.{0,1,3}.*`, `act_head.{0,2}.{weight,bias}` and `temperature`. F16 everywhere except `temperature`, which is F32.
- **`checkpoint/rl_agent_config.json`:** the fitted temperature under `temperature_by_options[bucket]`; the report's `t_applied` is bit-equal to it; `training.stopping` holds the early-stopping record.
- **Copied from the base, byte for byte:** `checkpoint/tokenizer/tokenizer.json` and `checkpoint/encoder/config.json`.
- **`recipe.json`:** exactly the reader's keys; values equal to the contract's recipe; seed 13; `seed_selection` and `early_stopping` equal to the contract; `shots_per_class` is the largest class count of `train.jsonl`; `base` carries no `laya_git`.
- **`gate-report.json`:** schema `laya-gate-report-v1`; thresholds equal to the contract; `seeds.label` is `median-ECE seed of 3 seeds`; per-seed rows in seed order, each with its rank key, its probability file's hash and its model hash; `device_used` exactly `cpu`; `f_avg` non-null for the stance task; the input hashes.
- **Probability files:** f32 values written exactly; the hash of each row's exact text; `seeds/seed-<s>/eval-probs.json` for every seed; the top-level `eval-probs.json` a byte copy of the shipped seed's; `zero-shot-probs.json`.
- **`probes.json`:** from the judge's own classifier on the contract's probe task.
- **With `shift.jsonl` present:** `shift-probs.json`, `shift-zero-shot-probs.json`, the report's `shift_probe` and `inputs_sha256.shift_jsonl`.
- **Not read by the judge:** `tokenizer_config.json`, `variance-report.json`, `training` other than as above, `rescore-noise.json`.
- **The base:** four pinned files: `model.safetensors`, `encoder/config.json`, `rl_agent_config.json`, `tokenizer.json`.

## Sources

- `paiml/aprender` `main` `6e7ea0133`: `docs/roadmaps/epics.yaml`; `docs/specifications/EPIC-0.75-crux-finetune-distill-plan.md`; `crates/aprender-core/src/{autograd,nn,setfit}`; `crates/aprender-train/src/{autograd,optim,transformer,finetune}`; `contracts/apr-cli-commands-v1.yaml`.
- PR #4941 head `e7730a3` and PR #4634 head `dad7f6a` (`guyernest/aprender`): `contracts/laya-finetune-gate-v1.yaml`, `contracts/laya-parity-v1.yaml`, `contracts/aprender/binding.yaml`, `crates/aprender-decide/src/{lib,verify,pack,artifact}.rs`, `crates/aprender-decide/src/laya/`, `crates/aprender-core/src/models/modernbert/`, `scripts/laya_train/`, `.planning/phases/08-…/08-GATE-RUN-EVIDENCE.json`.
- `NandhaKishorM/laya` at `4066d5d`: `laya/common.py`, `LICENSE`. `huggingface.co/convaiinnovations/laya`.
