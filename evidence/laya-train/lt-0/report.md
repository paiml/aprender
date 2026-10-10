# LT-0: first reads (APR-LAYA-TRAIN-001)

Measured 2026-10-10, 10:50Z to 11:00Z, on `noah-Lambda-Vector` (48 threads, 125 GiB), by the
worker `laya-train` (claude-opus-5-5). Nothing was built or run except the rule checker's
self-test. `main` = `e39f91a65e` (committed 2026-10-10T08:04:52Z). PR heads were read from
`refs/pull/4941/head` = `e7730a3a4c` and `refs/pull/4634/head` = `dad7f6a5e0`, read-only.

## (a) Baseline facts re-measured on `main`

| # | Spec said | Measured | Command |
|---|---|---|---|
| G1 | `6e7ea0133` | `e39f91a65e`, 2026-10-10T08:04:52Z. `main` moved; the spec's facts below still hold | `git rev-parse origin/main`; `git log -1 --format=%cI` |
| G2 | #4941 is a draft from a fork, one commit `e7730a3` | still OPEN, draft, head `e7730a3a4c`, not merged | `gh pr view 4941 --json state,isDraft,headRefOid,mergedAt` |
| G8 | gate contract 5.0.0, 13 equations, 6 binding rows, the seven unbound | holds on `e7730a3`: 5.0.0, 13 equations; `contracts/aprender/binding.yaml` has 6 rows (gate_pass, ece_top_label, split_disjointness, seed_selection_median, shift_probe_reported, synthetic_not_deployable), none of the seven. 54 lines of the gate contract and 6 of the parity contract name the `just laya-train-selftest` recipe or a `.py` file | `git show refs/laya/pr4941:<path>` |
| G10 | 18 `deny_unknown_fields` in `pack.rs`; `torch_version` required; `device_is_cpu` check | holds: 18; `torch_version` is a required `String` with no value check (`pack.rs:340`); `verify.rs:1805` | same |
| G11 | doors public, take a policy | holds: `verify.rs:3397`, `:3429`; `VerifyPolicy` all 18 fields `pub`, no `Default`, one constructor `from_contract_views(&GateContractView, &ParityContractView)` (`:378`); `pack::load_checkpoint_for_scoring` `pack.rs:802`, and `scorer_from_parts` `:819` | same |
| G16, G17 | core and train engines; absent ops | holds, with detail in (c) | file reads on `main` |
| G18 | three encoders on `main`, modernbert absent | holds: `models/bert/` (7 files), `setfit/encoder.rs`, `aprender-train/src/transformer/encoder.rs`; `models/modernbert` absent | `git ls-tree` |
| G19 | `aprender-train` depends on `aprender-core` | holds: `crates/aprender-train/Cargo.toml:85`, a path dependency | file |
| G20 | `safetensors_is_read_only_by_pack` | holds: `lib.rs:341-398` scans every `.rs` under `src/` for `safetensors::` and `SafeTensors`; only `src/pack.rs` and `src/test_support.rs` allowed; `tests/` and `examples/` are not scanned | file |
| G22 | 170 `.py` | 170 | `git ls-tree -r --name-only origin/main \| grep -c '\.py$'` |
| G27 | no `benchmarks/tweeteval-stance/`, no CLI command | holds: 0 paths, 0 matches | `git ls-tree`; `grep -c tweet-eval-stance contracts/apr-cli-commands-v1.yaml` |

## (b) The carry of #4941

Not merged; no carry PR exists (`gh pr list --search 4941 --state all` returns #4941 only).
`verify.rs`, `pack.rs` and `models/modernbert/` are identical on `e7730a3` and `dad7f6a`.
The two heads differ in the contracts only: `laya-finetune-gate-v1.yaml` gains one line
(`- tweet-eval-stance-benchmark-v1`, a reference) on `dad7f6a`, and `laya-parity-v1.yaml`
differs by 8 lines. The seven stay in the contract on both heads. Rows LT-3 on wait (§5).

## (c) Operation table

P+B present with backward · P−B present without backward · A absent. `core/` =
`crates/aprender-core/src/`, `train/` = `crates/aprender-train/src/`. Each backward was
found where its grad struct is attached to the output.

| Op | core | train |
|---|---|---|
| matmul 2D | P+B `core/autograd/ops/activation.rs:274`, `MatmulBackward` `gradient.rs:110` | P+B `train/autograd/ops/matmul.rs:453` |
| batched matmul | P+B, 4D only, `pub(crate)`: `nn/transformer/positional_encoding.rs:326`, `BatchedMatmul4dBackward` `grad_fn.rs:1553` | A |
| bias add (broadcast) | P+B `broadcast_add` `activation.rs:376` (`nn::functional::linear`'s `broadcast_add_1d` drops the gradient; use `nn::Linear`) | P−B `train/transformer/attention.rs:16`, `feedforward.rs:217` |
| elementwise mul | P+B `autograd/ops/mod.rs:86` | P+B `train/autograd/ops/basic.rs:54` |
| layer_norm (bias / no bias) | P+B `nn/functional.rs:369`, `LayerNormBackward` `grad_fn.rs:858` | P−B batched `train/transformer/norm.rs:247`; the autograd one normalises a whole 1-D tensor |
| softmax | P+B 2D `activation.rs:238`; last-dim N-D `positional_encoding.rs:499` (`pub(super)`) | P+B 1-D `train/autograd/ops/activations.rs:174` |
| log / exp / sqrt / div | P+B `ops/mod.rs:227`, `:205`, `:267`, `:117` | A (scalar `scale` only) |
| gelu exact | P+B `activation.rs:183` | A |
| gelu tanh | P+B `activation.rs:121` | P+B `activations.rs:52` |
| relu | P+B `activation.rs:9` | P+B `activations.rs:10` |
| embedding gather | P+B `autograd/ops/embedding.rs:42`, scatter-add `grad_fn.rs:1450` | P−B `train/transformer/embedding.rs:63` |
| slice / narrow | **A** | A |
| chunk / split (fused `Wqkv`) | **A** | A |
| concat | **A** | A |
| clamp | **A** | A |
| max | **A** | A |
| additive mask, padding | P+B `autograd/ops/masking.rs:47` + `add_mask` `positional_encoding.rs:480` | A (causal hard-coded) |
| additive mask, local window | **A**: a constant `[S,S]` mask can be built and combined with padding | A |
| MHA self, batched, masked | P+B `nn/transformer/mod.rs:390` (chain of backwards through `grad_fn.rs:1496-1647`) | P+B unbatched, no padding mask, `train/transformer/attention.rs:471` |
| rotate-half RoPE, two thetas | **P−B, wrong layout**: `nn/transformer/attention_helpers.rs:270` is interleaved and returns a tensor with no grad | P+B rotate-half, private, one theta: `train/transformer/attention.rs:79`, `RopeBackward` `:148` |
| seeded dropout | P+B `nn/dropout/mod.rs:77` | A |
| reshape / transpose | P+B `activation.rs:428`, `:328`, `positional_encoding.rs:221`, `:548` | P+B (flat tensors) |
| marker gather | **A**; `embedding_gather` over `[B·S,H]` with positions as ids is a P+B workaround | P−B CLS only, `train/transformer/encoder.rs:173` |
| sum / mean | P+B full reductions `ops/mod.rs:320`, `:342` | P+B sum only |
| cross_entropy | P+B `nn/loss.rs:259` | P+B, stops at the logits, `train/train/loss/cross_entropy.rs:38` |
| spherical score norm | P+B `autograd/ops/normalize.rs:60`, `similarity.rs:59` | A |
| grad clip, global norm | **A on tensors**; `metrics/grad_norm.rs:72` works on `&mut [f64]` | present `train/optim/clip.rs:23` |
| AdamW, two groups | **A**: one rate `nn/optim/mod.rs:391`; two instances are the workaround. `Optimizer::step` for AdamW does not update parameters; `step_with_params` (`rm_sprop.rs:87`) does | A, one rate `train/optim/adamw.rs:16` |
| AdamW decay on a parameter with no gradient | **skips** it: `rm_sprop.rs:46-48`. G5 decays every parameter, so LT-6 must give each parameter a gradient (zeros) or change the step | skips: `adamw.rs:120` |
| cosine schedule with floor | present `nn/scheduler/mod.rs:188`, `with_min_lr` `:215`; not clamped past `t_max` | present, clamps |
| F16 save | present `serialization/safetensors.rs:382` (round-to-nearest-even `:284`) | A (scalar converter only) |

Counts: core absent 9 (slice, chunk, concat, clamp, max, local-window mask, marker gather,
tensor grad clipping, AdamW groups); core present-without 1 (RoPE, wrong layout). Train
absent 16, present-without 5. Core's `TransformerEncoderLayer` is pre-norm with a tanh GELU
and biased projections (`nn/transformer/mod.rs:589-613`, `:243-246`), as F6 says.
Provisional reading, for LT-1 to decide: core needs fewer additions; train's encoder parts
(norm, feed-forward, embedding, bias add) have no backward.

## (d) The judge table

Every item the trainer calls is public. All modules are `pub` (`lib.rs:45-50`).

| Item the trainer calls | Where (on `e7730a3`) |
|---|---|
| `verify::pack_for_serving`, `verify::verify_path` | `verify.rs:3397`, `:3429` |
| `verify::VerifyPolicy::from_contract_views` + overwrite of the pub field `base` with `BasePins` (tiny pins); the views are `pub` + `Deserialize`, the caller parses YAML | `verify.rs:96-141`, `:146-163`, `:378`; the crate's own tests do this, `verify/tests.rs:58-77` |
| `pack::load_checkpoint_for_scoring`, `pack::scorer_from_parts` | `pack.rs:802`, `:819` |
| probabilities: `DecisionMethod::classify` on `Laya`; `Laya::classify_for_task`; raw logits `Laya::forward_row` | `lib.rs:159`, `:182`; `laya/mod.rs:478`, `:410` |
| `laya::temperature::{bucket_key, clamp_temperature, softmax_t, TEMP_MIN, TEMP_MAX}` | `laya/temperature.rs:12-64` |
| data reader `verify::read_data_dir` → `DataDir { task, train, eval, shift }` | `verify.rs:1254`, `:1196` |
| `verify::normalize_text`, `normalized_sha256` | `verify.rs:1638`, `:1645` |
| `artifact::*` manifest types (Serialize + Deserialize) | `artifact.rs:129-251` |

Not public, and not needed (no stop 3):
- No function derives the calibration slice; `check_split` only validates it. The trainer
  writes its own split with its own declared generator (F7).
- `pack::Recipe` and `pack::GateReport` derive `Deserialize` only. The trainer writes the
  JSON with its own serde structs and proves each file by reading it back through the
  judge's door, so a key drift is refused by `deny_unknown_fields`.
- The tiny fixture path is `pub(crate)` under `cfg(test)`; the trainer names
  `crates/aprender-decide/tests/fixtures/laya_tiny` as a path. The fixture itself is
  `synthetic-fixture` (its `recipe.json` has epochs 0, seed 20260925), so both doors refuse
  it as `SyntheticNotDeployable`, as the spec's LT-7 tiny acceptance already expects; a
  `production` tiny run must be written by the trainer.

Appendix C re-read, differences from the spec only:
- `t_applied` is compared bit for bit with the **clamped** `temperature_by_options[bucket]`
  (`artifact.rs:1617-1677`, `temperature.rs:44-53`); the report also needs
  `t_applied == clamp(t_fitted, 0.5, 5.0)` and `clamp_hit` consistent (`verify.rs:1804-1846`).
- Of the files "copied byte for byte from the base", only `tokenizer.json` is compared
  (`pack.rs:728-732`, `verify.rs:1569-1575`). The checkpoint's `encoder/config.json` is not
  compared with the base; the base dir's four files are pinned (`verify.rs:1565-1568`).
- Probability rows: `text_sha256` is the hash of the **raw** eval text, not the normalised
  one; values finite, in [0,1], sum to 1 within 1e-5 (`verify.rs:1852-1957`).
- `seeds.label` is checked only under `seed_selection` (`verify.rs:2630`).

Which of the seven the judge re-checks:

| Equation | Judge re-checks? |
|---|---|
| calibration_fit_bounded | partly: bound and clamp relations (`verify.rs:1804-1846`); it cannot re-fit `t_fitted` |
| declared_seed_ships | partly: `recipe.seed == 13` (`:1405`), `seeds.declared == recipe.seed` (`:2601`) |
| eval_set_in_distribution | no |
| recipe_before_scores | no (only `recipe_id == sha256(recipe.json)`, `pack.rs:678`) |
| device_recorded | yes, `verify.rs:1805-1815` |
| f16_reload_scoring | no directly; implied by the re-score of the packed bytes (`:1984`) |
| early_stopping_train_side | no (only the recipe block, `:1426-1444`) |

So LT-7 feeds the judge the plants of `device_recorded`, and the partial ones of
`calibration_fit_bounded` and `declared_seed_ships`.

## (e) How a command runs on a CPU pool host today

Nothing fits. No document defines "CPU pool host" or names one. Every dispatchable
self-hosted workflow runs on a `clean-room` runner, `intel,perf-solo` (the clean-room-16
box) or a GPU runner; no make target dispatches remote work (the nearest pattern is
`nightly-train-run`, a local `systemctl --user` unit). Per the spec, LT-1's PR adds one
workflow that starts on `workflow_dispatch` only, and its runner label is question Q2.

## (f) Licence of the dataset files

TweetEval on Hugging Face (`cardiffnlp/tweet_eval`): licence tag "unknown"; stance subsets
"Undefined"; Twitter terms apply. The original SemEval-2016 Task 6 data (NRC Canada,
saifmohammad.com/WebPages/StanceDataset.htm): "available free for research purposes",
cite the paper; commercial use by request; "Do not redistribute". Reading: research and
benchmark use by this open-source trainer is allowed; tweet text, labels and derived splits
never enter git (H12 already says so); prefer the NRC download over the third-party mirror.
This is not stop 8 by the worker's reading; it is Q4 for the cop.

## (g) The rule checker

`scripts/check_laya_train_rules.sh` prints one line per *cmd* rule of §9 (H1, H2, H4, H5,
H10, H11, H12, H15) for `<base>...<head>`, each with its anti-vacuity arm, and exits 1 on a
RED. `--self-test` plants 35 cases in a throwaway repository (each rule's mutant must be
named RED, each clean case PASS) and runs in `workspace-test` through
`ci/explicit-test-commands.d/795-laya-train-rules-self-test.cmd`. Two readings of the spec,
stated so they can be overruled:
- H4: a head that adds LT-1's own receipt may touch `crates/`. One path is exempt by the Q5
  ruling (handoff/quorum-a3-laya-q5-lt1-order.md): `crates/aprender-train/examples/laya_engine_probe.rs`,
  the proxy probe, which must merge before its receipt can exist; the arm prints the exempt count.
- H15: the needles are plain substrings, so every form of an ssh call is a hit. Exactly one
  path is skipped (C314, handoff/c314-a3-laya-lt0-lt1a.md): `scripts/check_laya_train_rules.sh`,
  which must name the needles to plant them; the arm prints the skipped path.
- H5: the chosen engine is read from `"engine"` in LT-1's receipt on the base.

## (h) Questions to the cop

- Q1 surface: default, the example binary stays and no `apr` verb is added.
- Q2 host for LT-1 and LT-10: which runner label or host is "a CPU pool host"? Default: none
  is guessed; LT-1 is prepared on `laya-train/wip` and its PR adds one `workflow_dispatch`
  workflow whose `runs-on` the cop names.
- Q3 E8's recipe-schema row: default, no coupling.
- Q4 dataset licence: default, research use, NRC source, no redistribution, proceed.
