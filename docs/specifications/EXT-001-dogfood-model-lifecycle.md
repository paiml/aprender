# EXT-001: Dogfood model lifecycle — experiment tracking, continuous improvement, and the model release train (v1)

**Spec id:** `EXT-001` · **Rows:** `EXT-00..EXT-31`. The aprender traffic cop mints one `pmat` ticket per row, and the single-minter rule applies.
**Target repo:** `paiml/aprender`, path `docs/specifications/EXT-001-dogfood-model-lifecycle.md`. This file supersedes the v0 draft at `~/infra/docs/specifications/EXT-001-experiment-tracking.md`, which was never committed; delete that draft, do not merge it.
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/EXT-001-dogfood-model-lifecycle.md autonomously.`
**Cross-repo (filed, not done):**
- `paiml/infra`: ledger columns (infra#1057), generated `docs/MODELS.md`, and forjar declarations for publish credentials and eval cells.
- `paiml/paiml-implement` and arbiter: lane recusal.

**Related specs:**
- REX-001 (review-lane experiment: cells, sealed corpus, H4/H5 promotion)
- ARB-APR-001 (apr lane pin and weights sha)
- ARB-SELF-001 (dogfood outcome ledger)
- APR-PERF-GATE-001 (perf receipts, comparator rule, clean-room Mode A artifact)
- APR-RELEASE-001 (train doctrine; published tags are immutable)
- `30-day-plan.md` rules 11 and 13

**Status:** spec, not implemented. Dated 2026-09-25, v1.1; folded into the repo on 2026-10-03 as v1.2. Rewritten from v0 after the 2026-09-25 review; findings F1–F9 of that review are folded in.
- v1.1 adds CRUX competitor gates (§10, rows EXT-24..31) and the 0.71 epic (§11).
- v1.1 also corrects EXT-16: no comparator ratio may appear on a published card (APR-PERF-GATE-001 §3.2).
- v1.2 moves the spec to the path above and applies the EXT-24 bind census, bound at `316dee2cd4` with no build (`la-71/ext-24-crux-bind`, `docs/lookahead/0.71-ext-24-crux-bind.md`):
  - T30 is re-derived and marked `[V]`. `gap_effect` is a per-PR review output, not a contract field. `metadata.category` is not in the contract schema at all. `competitor` is a closed registry.
  - EXT-24's enum question is answered: there is no enum (§10.1).
  - EXT-25 is not docs-only (§10.1).
  - §11.1's label exception rests on a premise that is false at `316dee2cd4`. This opens look-ahead ruling R-71-5.

**Provenance marks:** `[V]` verified at the cited sha/time/host · `[C]` computed or cited from a committed spec · `[A]` asserted · `[U]` unverified/unmeasured · `[X]` third-party.

---

## §0 Operating assumptions

1. **There are two products and one pipeline.**
   - The **engine** (`apr`, version line `X.Y.Z`) improves in *speed* with each aprender release.
   - The **model** (the paiml Qwen 3.5 4B line, its own version line `vX.Y.Z[-rc.N]`) improves in *quality* with each model release.
   - The two version lines are independent. Receipts join them: every model receipt names the engine that produced it, and every engine speed receipt names the model it ran.

2. **Publishing is proof.**
   - A model on Hugging Face is a public claim: "this stack built this artifact, and it performs as stated."
   - Every number on a model card, release note or post is **rendered from a receipt**, never typed by hand.
   - Anything that cannot be receipted is not claimed.

3. **The loop is sub-threshold by construction.** Design basis: Zhang, Yuan & Zhang, arXiv:2607.04277 `[X]`.
   - The improvement loop is `M ∘ V ∘ f_T`: the candidate generator (M), the evaluator (V), and a bounded evaluation run (f_T). In Yampolskiy's taxonomy it is **Level 2 (weak RSI)**, and deliberately so.
   - **Theorem A3** (improvement is undecidable in general): there is no total improvement predicate. V is therefore a *bounded, pre-registered, sealed* evaluation with a statistical decision rule. V is never the candidate's own judgment.
   - **Corollary A1** (no internal hard invariants): anything inside the loop's write perimeter is rewritable. So **every gate, eval corpus, promotion rule and publish credential sits outside that perimeter** (§3.2).
   - **Model collapse** (Shumailov et al. 2024 `[X]`): self-generated data is admissible only when paired with an externally resolved label (§3.3).

4. **Producer is never the gate.**
   - Candidates are built by a **released, tarball-sha-verified `apr`**, and gates run a released `apr`.
   - Parity is judged against an independent implementation: llama.cpp `d1d3c3396` `[X]`.
   - The apr lane is recused from any PR that touches this pipeline's gates (I-12).

5. **Genchi genbutsu.** EXT-00 re-derives every `[U]` and `[A]` row of §1 at HEAD, stamping host, tree and time. A row whose premise is false is recorded `premise-falsified` and skipped or re-planned, never forced.

6. **A refusal is a result.** `finetune` and `distill` on `qwen35` refuse until 0.71 `[C]`.
   - Weight-changing rows are `NotRun{Refused(removed_by=0.71)}` until that refusal is lifted.
   - The pipeline is proven end-to-end first with a **packaging release** (EXT-18), which needs no training.

7. **Sovereignty.** The system of record is ours: pacha on each host, the JSONL export on the RAID, and the infra ledger. Hugging Face and GHCR are **distribution mirrors**, verified by sha256. Losing either of them loses no data.

8. **Cross-repo work is filed, not done.** Rows that depend on a filed issue are `NotRun{NoDeclaredExecutor}` until it lands.

---

## §1 Ground truth (re-derive in EXT-00)

| # | Fact | Mark | Source / probe |
|---|---|---|---|
| T1 | pacha `~/.pacha/registry.db`: models=81, runs=0, lineage=0 | `[A]` (v0 sample, host unstamped) | `sqlite3 ~/.pacha/registry.db "select (select count(*) from models),(select count(*) from runs),(select count(*) from lineage);"` |
| T2 | entrenar `~/.entrenar/experiments.db`: 3,940 experiments, 3,607 named `.tmp%`, 238 `running`, 0 artifacts | `[A]` | `sqlite3 ~/.entrenar/experiments.db "select count(*), sum(name like '.tmp%') from experiments; select status,count(*) from runs group by 1;"` |
| T3 | 48 loose run dirs under `/mnt/nvme-raid0/runs/` | `[A]` | `ls -1d /mnt/nvme-raid0/runs/*/ \| wc -l` (stamp the host) |
| T4 | 0.68.2 with Qwen3.5-4B on the 4090: TTFT 7.7× and e2e 4.9× slower than llama.cpp `d1d3c3396`; 14 s load dominates | `[C]` | APR-DECIDE-001 §1 |
| T5 | `qwen35` finetune/distill refuse until 0.71; batched `m>1` refuses until 0.70 | `[C]` | 0.69 explicit-refusal list |
| T6 | Parity oracle: llama.cpp pin `scripts/llama_pin.toml build_commit` = `d1d3c3396`; `ds.yaml` min_cosine 0.98 | `[C]` | `scripts/llama_pin.toml`, `ds.yaml` |
| T7 | Release themes: 0.70 = verbs that are fast (TTFT within 2× llama.cpp, `apr serve` resident); 0.71 = wider models + training | `[C]` | release schedule spec |
| T8 | Train departures: 0.70 on 09-26, 0.71 on 09-29 (72 h takt `[A]`) | `[C]` | `30-day-plan.md` §6 |
| T9 | REX-001: 4B review lane is a shadow lane on gx10 (provisional); sealed corpus plus contamination contract; REX-08 rules the cells | `[C]` | `docs/specifications/review-experiment-protocol.md` |
| T10 | Decode ceiling for 4B Q4_K_M at ≈2.5 GB/token: 4090 ≈ 400 tok/s, GB10 ≈ 110 tok/s | `[C]` theoretical | REX-001 G10 |
| T11 | Sizes: 4B BF16 ≈ 8 GB (4e9 × 2 B); Q4_K_M ≈ 2.5 GB | `[C]` | arithmetic |
| T12 | **GitHub Models is not a weight store.** Its org "custom models" feature is bring-your-own API key, supports only OpenAI and AzureAI providers, and is in public preview | `[X]` | docs.github.com, GitHub Models BYOK page |
| T13 | GHCR stores OCI artifacts with a 10 GB per-layer limit and a 10 min upload timeout | `[X]` (community thread, 2023) → `[U]` | re-verify with a real push in EXT-17 |
| T14 | GitHub Release assets: 2 GiB per-file limit | `[U]` | re-verify; if true, Release assets carry manifest, card and receipts only, never weights |
| T15 | HF model repos support branches and tags as revisions, reachable over the HTTP API | `[U]` | EXT-00 binds the exact endpoints |
| T16 | An `apr` verb that uploads to HF exists | `[U]` | `pmat query "hugging face upload"` scoped to `crates/apr-cli`; `apr --help` |
| T17 | `apr` convert/quantize supports `qwen35` at the pinned release | `[U]` | `apr quantize --help`; one run receipt |
| T18 | `apr` exports GGUF | `[U]` | `pmat query "gguf export"` |
| T19 | The Qwen 3.5 4B upstream license permits redistribution of derivatives, with its stated attribution terms | `[U]` | read the upstream model card at the pinned revision |
| T20 | `paiml/albor-370m-v1` exists on HF | `[A]` | HF API |
| T21 | A Rust OCI client crate exists (`ocipkg`) | `[X]` | docs.rs; admission is decided in EXT-17 |
| T22 | Competitor quants of the same base on HF: `unsloth/Qwen3.5-4B-GGUF` (BF16 8.42 GB, IQ4/Q3/Q4 variants, "UD" dynamic quants, an imatrix file; License apache-2.0; `main` has 37 commits) and `bartowski/Qwen_Qwen3.5-4B-GGUF` (Q4_K_M, Q8_0, …) | `[X]` | huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main · huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF |
| T23 | Ollama ships `qwen3.5:4b`: Q4_K_M, 3.4 GB, arch `qwen35`, 4.66B params, Apache-2.0, default sampling (temp 1, top_k 20, top_p 0.95, presence_penalty 1.5); vision and tools tagged | `[X]` | ollama.com/library/qwen3.5:4b |
| T24 | Qwen 3.5 is a vision-language family (image-text-to-text). The upstream `Qwen/Qwen3.5-4B` license is Apache-2.0 per derivative cards linking the upstream LICENSE | `[X]` → T19 still binds on the upstream card itself | huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-abliterated (license_link) |
| T25 | Unsloth advertises local fine-tuning of Qwen3.5 | `[X]` | unsloth/Qwen3.5-4B-GGUF card |
| T26 | MLflow 3 makes `LoggedModel` first-class, with lineage across models, runs, traces, prompts and evaluation metrics. Model Stages have been deprecated since MLflow 2.9 in favour of aliases (e.g. `@champion`). Latest release line is 3.14 (Aug 2026) | `[X]` | mlflow.org/docs/latest/ml/model-registry · github.com/mlflow/mlflow/releases |
| T27 | Conceded segments (no speed war): raw training throughput vs Unsloth's Triton kernels, datacenter concurrent serving (vLLM class), Apple-silicon decode vs Ollama-MLX | `[C]` (pre-2026-09-25 review doc; re-rule if contested) | `fable-architectural-review.md` §7c |
| T28 | Rules: a number in README/book/docs is legal only when it cites an `evidence/` receipt; **a comparator ratio is illegal regardless of citation**; an `[X]` figure is illegal regardless of citation | `[C]` | APR-PERF-GATE-001 §3.2 |
| T29 | The 0.70 train excludes `label:crux-competitor`; 0.71 excludes `label:gate-framework-new` and carries ≤1 must-carry; 0.71 departs 09-29 | `[C]` | `30-day-plan.md` §6 |
| T30 | CRUX contracts: `contracts/crux-*.yaml` (318 at `316dee2cd4`, ids `crux-<LETTER>-<NN>`), matched on `category` + surface. **`competitor`** is a closed registry: `CRUX_COMPETITORS`, 15 entries, rule CRUX-002 (`crates/aprender-contracts/src/schema/validator.rs`). **`metadata.category`** is not in the contract schema. `Metadata` (`crates/aprender-contracts/src/schema/types.rs`) has no `category` field and is not `deny_unknown_fields`, so the parser drops it. The master registry's `CruxStory` accepts it unchecked ("no rule constrains"). The labels also collide: the 316 contracts that set `category` use 21 labels over 15 letters (A, B, F and K have 2 names each; E has 3). **`gap_effect`** (closes, widens or none) is a **per-PR review output** (`.claude/skills/pr-review/SKILL.md` §3.C, with the comparator block in §3.C.1), not a contract field. No schema type has `gap_effect` or `comparator`, `metric` exists only on the `Beat` block (BEAT-001..007), and 0 contracts carry `gap_effect`. A `gap_effect` written into a contract today is silently dropped. `crux_coverage: none` is today a *finding*, not a gate | `[V]` `316dee2cd4`, 2026-10-03 (v1.2; was `[C]` / `[U]`) | PR-REVIEW-SKILL-002 §3.C; APR-DECIDE-001 App. A |
| T31 | Qwen3.6 is released upstream (a MAJOR-line watch signal, not a gate) | `[X]` | ollama.com/library |

---

## §2 Goals

| Id | Goal | Done when (falsifiable) |
|---|---|---|
| **G1** | **One system of record.** Every `apr finetune\|train\|distill\|merge\|quantize\|prune` records a run by default. | 6/6 verbs produce `runs` +1 with params, dataset manifest sha, base sha, output sha, engine identity, host, GPU and wall time (case table). |
| **G2** | **Lineage.** Every model resolves to its root. | `apr registry lineage <model> --json` returns a chain ending at an HF revision or an imported GGUF sha, for 100% of registered training outputs. |
| **G3** | **Engine speed improves continuously on the blessed model.** | Every aprender tag produces a speed receipt per ruled cell for the current blessed model (100% coverage). Ratios vs llama.cpp are a shrink-only ratchet after 3 records. |
| **G4** | **Model quality improves continuously, judged externally.** | Every promotion carries a sealed-suite comparison against the incumbent under the pre-registered rule (§3.6). There are 0 promotions without one. |
| **G5** | **The model release train mirrors the engine train.** | The rc → tag → published → (yanked) lifecycle exists; 100% of published revisions pass gates M-CR and M0–M6 before publish and M7 after. |
| **G6** | **Public proof.** | Each model release exists on HF as an immutable tag. Every number on its card maps to a receipt id (0 unmapped). The card carries the engine that built it and the independent parity result. |
| **G7** | **Clean store.** | Real-home writes by tests = 0 (canary). Orphan `running` rows = 0. Dangling metrics pointers = 0. Orphan models = 0. |
| **G8** | **Fleet ledger.** | The infra#1057 ledger and `docs/MODELS.md` are generated from the export with 0 hand edits. |

### §2.1 Quantified targets (Toyota framing)

| Lever | Metric | Baseline | Target | Probe |
|---|---|---|---|---|
| Poka-yoke | test writes to real `$HOME/.entrenar` or `$HOME/.pacha` | 3,607 leaked rows `[A]` | **0**, canary sha unchanged; the mutation turns it RED | FALSIFY-EXT-001 |
| Jidoka | `running` rows whose `(host,boot_id,pid,start)` is not live | 238 `[A]` | **0** after gc; **0** live runs reaped | `apr runs gc --dry-run` |
| Jidoka | models from training verbs without an inbound `produced` edge; dangling metrics pointers | n/a | **0 / 0** | `apr runs fsck` (exit ≠0 on any) |
| Genchi genbutsu | training verbs recording by default | 0/6 | **6/6** | case table, FALSIFY-EXT-005 |
| Kaizen | loose run dirs accounted for | 0/48 `[A]` | **48/48** imported or skipped with a reason; **0** fabricated fields | `apr runs import --dry-run` |
| Jidoka | published revisions without a full gate receipt set | n/a | **0** | `model-gate-receipt-v1` join |
| Jidoka | post-publish sha mismatch (HF or GHCR vs manifest) | n/a | **0**; any mismatch → yank in the same session | EXT-15 |
| Andon | card numbers without a receipt id | n/a | **0** (lint) | EXT-16 |
| Kaizen | TTFT and e2e ratio vs llama.cpp, 4B Q4_K_M, `apr serve` resident, per ruled cell | 7.7× / 4.9× on 4090 at 0.68.2 `[C]` | 0.70 theme: TTFT ≤ 2× `[C]`; then shrink-only after 3 records | APR-PERF-GATE-001 receipts |
| Kaizen | sealed-suite recall/precision of the review lane | REX-06 measurement `[U]` | non-regression at every promotion; ratchet on the incumbent | §3.6 rule |
| Heijunka | ext PRs in CI at once | n/a | **≤1**; counts toward the aprender 10-PR cap | `gh pr list` |
| Andon | tracking overhead in tok/s | unmeasured | `[U]`: EXT-04 first measures the tok/s coefficient of variation on albor-50m (n=10), then sets the bound as `basis=` that receipt | instrument first |
| Kaizen | promotable-candidate → HF tag wall clock | unmeasured | `[U]`: instrument, then shrink-only ratchet after 3 records | release receipts |

---

## §3 Architecture

### §3.1 Stores and keys

- **pacha is the system of record** for models, datasets, runs, lineage and evals (D-1).
- **entrenar SQLite is the per-step metrics writer only.**
  - There is one metrics DB per host, on the RAID (`/mnt/nvme-raid0/cache`).
  - Per-checkpoint `.entrenar` DBs are import-only.
  - Writes use WAL mode and `busy_timeout`.
- **Keys:**
  - Run ids are ULIDs.
  - The join key for models is `sha256(file)`.
  - The join key for datasets is the canonical manifest sha (§3.3).
  - Host-local autoincrement ids never leave the host.
- **Pointer join:** `runs.run_json` holds `{metrics_db, run_id}` plus summary metrics. I-6 (referential integrity) is checked by `apr runs fsck`.
- **Export:** a nightly JSONL export per host goes to the RAID. Re-exporting is byte-identical (I-7). The fleet merge happens on content keys, never on host ids.

### §3.2 The improvement loop and its write perimeter

```
            ┌──────────────── WRITE PERIMETER (the loop may change these) ────────────────┐
            │ recipes (prompt, decoding config) · non-sealed datasets · candidate weights │
            │ checkpoints · run records · candidate model cards (drafts)                  │
            └───────────────┬─────────────────────────────────────────────▲────────────────┘
          M (candidates)    │                                             │ promote = new version
                            ▼                                             │
   f_T: bounded eval run on a released apr, on ruled cells ──► V: sealed suites + pre-registered rule
            ┌──────────────── OUTSIDE (read-only to the loop; changes need a PR + quorum) ────┐
            │ sealed eval manifest (hashes) · promotion contract · base-owned gate workflow   │
            │ llama.cpp parity pin · human HRQ labels · merge/revert outcomes · publish creds │
            └──────────────────────────────────────────────────────────────────────────────────┘
```

- **M, the candidate generators, by availability:**
  - prompt/decoding recipes: now;
  - quantization variants: now, if T17 holds;
  - QLoRA fine-tune and distill: 0.71+;
  - merge: 0.71+.
- **V is never the candidate.** No verdict produced by a candidate enters its own promotion evidence (I-9).
- The training process runs **without** read access to publish credentials and **without** write access to the sealed manifest or the promotion contract. A test asserts both (FALSIFY-EXT-011).

### §3.3 Data provenance and collapse controls

- **Dataset manifest:** the canonical hash is sha256 over the sorted `(relpath, bytes, sha256)` list. Permuting the file order leaves the hash unchanged (a property test).
- **Every row carries an origin:** `origin ∈ {human, upstream, external_model, self_generated}`.
- **Admissibility of self-generated rows.** A `self_generated` row is admissible only when it carries an **externally resolved label**:
  - a human HRQ verdict;
  - a PR merged or reverted within the observation window;
  - a test pass/fail;
  - or a llama.cpp-verified output.

  Unlabeled self-generated rows are refused at dataset registration.
- **Synthetic fraction:** the per-dataset share of `self_generated` rows is recorded and rendered on the card. There is no threshold: it is `[U]`, instrumented first.
- **Contamination check:** `training manifest ∩ sealed eval item hashes = ∅` (I-11). The check reuses REX-001's `review-corpus-contamination-v1`.

### §3.4 Version lines and the release manifest

- **Model line:** working name `paiml/qwen3.5-4b-apr`, provisional per D-3.
  - `MAJOR`: base model changes (for example, a different upstream model).
  - `MINOR`: tensors change (fine-tune, distill, merge, new quant variant) or the shipped recipe changes.
  - `PATCH`: tensor-identical changes (card, metadata, packaging fixes).
  - `-rc.N`: release candidate.
- **`model-release-v1.json`** is committed, published, and signed by its receipt chain. It contains:

```
{ line, version, channel: rc|released|yanked,
  files: [{name, format, quant, bytes, sha256}],
  base: {hf_id, revision, sha256},
  lineage: [run ULIDs], datasets: [manifest sha],
  engine: {apr_version, crate_tarball_sha256},
  gates: {M-CR, M0..M6: receipt ids}, license: {spdx_or_name, upstream_notice_sha256},
  recipe: {prompt_sha256, decoding} }
```

### §3.5 Channels

| Channel | Where it lives | Mutability |
|---|---|---|
| `dev` | pacha stage `development`; never leaves the fleet | free |
| `rc` | HF branch `rc/vX.Y.Z-rc.N`; GHCR tag `vX.Y.Z-rc.N` (if D-2) | immutable once pushed; a fix becomes `rc.N+1` |
| `released` | HF tag `vX.Y.Z`; HF `main` points to the newest released version; GHCR `vX.Y.Z` + `latest` | **immutable forever** (I-8) |
| `yanked` | tag kept; card banner rendered; `yanked.json` records the reason and the receipt id | never deleted |

### §3.6 Model gates (clean-room first)

| Gate | Check | Phase | Blocks |
|---|---|---|---|
| **M-CR clean-room** | In a fresh container, install `apr` from the clean-room Mode A artifact of the pinned release. Fetch the rc from HF **by revision**, verify every sha256 against the manifest, then run the M1 parity and M3 smoke on the fetched bytes. | rc → release | **yes (hard, first)** |
| M0 identity | Every file hashed; the producing engine is a released tarball-sha-verified `apr`; the lineage chain resolves (I-1, I-2, I-5) | rc | yes |
| M1 parity | cosine ≥ 0.98 vs llama.cpp `d1d3c3396` on the exact bytes (`ds.yaml`, `[C]`); greedy-token agreement is report-only | rc | yes |
| M2 quality | Sealed suites, paired candidate-vs-incumbent: McNemar exact for binary items, seeded bootstrap CI otherwise, Holm across suites. α is pre-registered in the contract (basis: REX-001 §3). **Promote iff** no suite is significantly worse **and** (≥1 suite is significantly better **or** the release class is PATCH/packaging with tensor-identical files **or** it is the first release, which reports absolute levels and claims no improvement). Underpowered by the sample-size rule → no promotion (S-10). | rc | yes |
| M3 smoke + parse | `apr run` and `apr serve` answer the fixed probe set; review-lane verdict parse rate reported | rc | yes (smoke); parse rate reported |
| M4 contamination | I-11 | rc | yes |
| M5 license | Upstream license and NOTICE present and matching T19 | rc | yes |
| M6 card truth | Every number on the rendered card maps to a receipt id; comparative claims carry a comparator block; `[X]` figures are labelled as third-party | rc | yes |
| M7 post-publish | Re-download from HF (and GHCR) at the tag and compare shas to the manifest. A mismatch triggers a yank and stops the line. | after publish | yank trigger |

### §3.7 The engine speed loop

- On every aprender tag, `perf_gate` runs with the released binary on each REX-08-ruled cell against the **current blessed model** (the newest `released` model version).
- Receipts go to the speed ledger (EXT-19).
- **The model card does not churn per engine release.** It carries the speed table measured at model-release time and links to the live ledger in `docs/MODELS.md`.

### §3.8 Publishing mechanics

**Clean-room (M-CR) runs first.** Publishing happens only on the driver host, from a detached checkout of the promoted aprender tag. It is **automatic when every gate is green**, under the standing operator directive of 2026-09-20; the operator is never asked.

- **Tokens:**
  - the HF token and GHCR token live only on the driver host;
  - they are never in GitHub secrets;
  - no workflow ever pushes;
  - no training process can read them (R-7).
- **The publisher is Rust.** No `huggingface_hub`, no `hf` CLI, no Python (R-1).
- **Uploads are idempotent and resumable:** a re-run after an interruption produces the same revision or no-ops.

### §3.9 Dogfood linkage

- The blessed model *is* the review lane's model (REX-001) and the arbiter `decide` lane's model (ARB-APR-001).
- Promotion to `released` does **not** change any lane's pin. Lane pins move only through forjar declarations, by their own specs, using the published sha256.
- The apr lane is **recused** from PRs touching `contracts/dogfood-model-lifecycle-v1.yaml`, the sealed corpus manifest, the gate code, or the publisher (I-12). This is a filed issue in paiml-implement and arbiter.

---

## §4 Invariants (contract obligations; each has a named id)

- **I-1 No orphan model.** Every `models` row created by a training verb has ≥1 inbound `produced` edge from a `runs` row.
- **I-2 Hash truth.** `models.content_hash` = sha256 of the file at registration. A re-hash mismatch is RED.
- **I-3 Provenance honesty.** `provenance ∈ {recorded, backfilled, imported}`. `backfilled` rows never feed a recorded claim, a card, or a ledger quality column without a label.
- **I-4 Test isolation.** No test process opens the real `$HOME/.entrenar` or `$HOME/.pacha`.
- **I-5 Engine identity.** A run records `(apr_version, git_sha | crates_io_tarball_sha)`. A build that is dirty or unidentifiable is refused at `start_run`, and the refusal names `--no-track`.
- **I-6 Referential integrity.** Every `metrics_db` pointer resolves and its `run_id` exists.
- **I-7 Export determinism.** Re-exporting an unchanged host is byte-identical.
- **I-8 Release immutability.** A published tag's file set never changes. Fixes are new versions, and a yank never deletes.
- **I-9 External judge.** Promotion evidence contains no verdict produced by the candidate. The sealed manifest sha is fixed in the contract.
- **I-10 Card truth.** Every card number maps to a receipt id.
- **I-11 No contamination.** Training manifest ∩ sealed eval hashes = ∅.
- **I-12 Recusal.** The apr lane casts no vote on PRs touching the paths in §3.9.
- **I-13 License carry.** Every published revision contains the upstream license and NOTICE.
- **I-14 Self-generated data needs an external label** (§3.3).

---

## §5 Rows (EV-ordered)

`K̂` is in minutes `[A]`, recalibrated after the first three rows. Each row gets one ticket, one PR on branch `ext/<row>`, TDD, and mutation RED→GREEN in the PR body.

### Phase A — ground truth and store hygiene (milestone 0.70.x; independent of D-1)

| Row | Work | Done when (all must hold) | K̂ |
|---|---|---|---|
| **EXT-00** bind | Re-derive §1 T1–T3 and T13–T21 at HEAD, stamping host, tree and time. Emit `ext-bind-receipt.json`. | Every `[U]`/`[A]` row becomes `[V]` or `premise-falsified`, with the probe output recorded. T19 negative → S-5. T16 negative → EXT-14 re-scoped to "implement". | 45 |
| **EXT-01** contract | `contracts/dogfood-model-lifecycle-v1.yaml`: I-1..I-14 as named obligations, and FALSIFY-EXT-001..024 named and mapped to test paths (stubs allowed, but `#[ignore]` stubs count as 0). | `pv validate` reports `24 evaluated`, 0 anonymous obligations; each stub is listed as owed by its row. | 60 |
| **EXT-02** isolation | nextest profile env sets `HOME`, `ENTRENAR_HOME` and `APR_HOME` to a tempdir. Writers are found with `pmat query` by serialisation site. CI seeds a canary DB at the real-home path. | After the workspace `cargo nextest`, the canary sha is unchanged (CI and dev box). Removing the env override turns FALSIFY-EXT-001 RED. | 90 |
| **EXT-03** reaper | Liveness = `(host, boot_id, pid, proc_start_time)`. Add a heartbeat column if absent at HEAD (premise recorded). Add `apr runs gc`: `--dry-run` is the default, `--yes` is required, a backup with its sha256 is written first. | Dry-run lists `.tmp*` plus orphans (union counted, no double count). After running: 0 non-live `running` rows. The planted live pid and the planted reused pid are both left alone (FALSIFY-EXT-002). | 90 |

### Phase B — record (milestone 0.71; needs D-1)

| Row | Work | Done when | K̂ |
|---|---|---|---|
| **EXT-04** backend | `PachaBackend` for `TrackingBackend`; ULID run ids; WAL; `apr runs fsck`. **First step:** measure the tok/s coefficient of variation on albor-50m (n=10) and receipt it. | Round-trip property test; ≥2 concurrent writers with no lost rows; a planted dangling pointer makes `fsck` exit ≠0 (FALSIFY-EXT-003). | 180 |
| **EXT-05** train verbs | `finetune` and `train` record by default (`--no-track` opts out); I-5 identity. | Dev test on a pacha tmp home: `runs` +1, `lineage` **+3** (base, dataset, produced), `models` +1 with base pre-registered, output sha matches. The overhead delta is within the EXT-04 noise band, with `basis=` that receipt (FALSIFY-EXT-004). | 120 |
| **EXT-06** other verbs | `distill`, `merge` (N parents), `quantize`, `prune` with edge types `distilled_from`, `merged_from`, `quantized_from`, `pruned_from`. | Case table: the expected edge type and parent count per verb (FALSIFY-EXT-005). | 120 |
| **EXT-07** lineage verb | `apr registry lineage <model\|sha> [--json]`. | A 3-hop fixture gives 3 nodes and 2 edges; a cycle fixture is refused (FALSIFY-EXT-006). | 60 |
| **EXT-08** datasets | Canonical dataset manifest; `origin` per row; I-11 and I-14 checks at registration. | The permutation property test holds. A planted sealed-item leak is refused. A planted unlabeled `self_generated` row is refused (FALSIFY-EXT-012, -013). | 120 |
| **EXT-09** eval attach | New `evals` table: `model_sha, suite, suite_manifest_sha, score, n, engine identity, host, ts`. `apr eval` and `apr qa --json` write it. | `apr runs show` displays the output model's evals. An unregistered file warns and still runs. | 90 |
| **EXT-10** backfill | `apr runs import <dir>`; `provenance=backfilled`; unknown fields are NULL; df is checked first. | 48/48 imported or skipped with a reason; 0 fabricated fields (FALSIFY-EXT-008). | 90 |

### Phase C — the model release train (milestone 0.71; needs Phase B rows EXT-04..06)

| Row | Work | Done when | K̂ |
|---|---|---|---|
| **EXT-11** packager | `apr model pack` builds a release dir plus `model-release-v1.json` from pacha lineage. The output is deterministic. | Two runs give byte-identical output (FALSIFY-EXT-014). The manifest validates against its schema. | 150 |
| **EXT-12** gates | `apr model gate` runs M0–M6 and writes `model-gate-receipt-v1`. The M2 statistics live in one module, tested against hand-computed fixtures. | Each gate has a planted failure that turns it RED. M2 refuses to promote an underpowered comparison (FALSIFY-EXT-015). | 180 |
| **EXT-13** model clean-room | M-CR as a clean-room job: fresh container, clean-room Mode A `apr`, fetch by HF revision, sha verify, parity + smoke. | A planted one-byte corruption in the rc fails M-CR. `HEAD == tag` is asserted for the engine (FALSIFY-EXT-016). | 120 |
| **EXT-14** HF publisher | Rust HF client: create the rc branch, upload (resumable), create the tag on promote, move `main`. Idempotent. Token read from the driver-host path only. | Re-running a completed publish is a no-op. An interrupted upload resumes to the same revision. No token appears in env or logs (grep of the receipt shows 0 hits). Python dependency needed → S-6. | 240 |
| **EXT-15** confirm + yank | M7 after every publish; `apr model yank <version> --reason` renders the banner and writes `yanked.json`. | A planted mismatch triggers the yank path and a STOP. A yanked tag is still fetchable (FALSIFY-EXT-017). | 60 |
| **EXT-16** card renderer | The card is rendered from receipts: what, lineage, engine, parity, sealed-suite scores, speed table, synthetic fraction, license, known refusals.<br>Competitor arms are shown only as **absolute values we measured ourselves**, each with its comparator block, side by side. There is **no ratio, no "N×" and no `[X]` figure** (T28).<br>A lint rejects any number without a receipt id, and any ratio token next to a competitor name. | A planted hand-typed number fails the lint. A planted "2.1× llama.cpp" fails the lint (FALSIFY-EXT-018, -024). | 90 |
| **EXT-17** GHCR mirror (if D-2 = yes) | Rust OCI push of the release files plus the manifest as an artifact; T13 re-verified with a real push. | M7 covers GHCR; the per-layer limit is receipted from the actual push. | 120 |
| **EXT-18** first public release | `v0.1.0-rc.1` → `v0.1.0`: a packaging-only release of the Qwen3.5-4B quant the review lane runs, produced by a released `apr` (T17), with parity against llama.cpp. No improvement claim. | The HF tag `v0.1.0` exists; M-CR and M0–M7 are all green; the card renders with 0 unmapped numbers; the pacha lineage resolves to the upstream HF revision. | 120 |

### Phase D — continuous loops (start after EXT-18)

| Row | Work | Done when | K̂ |
|---|---|---|---|
| **EXT-19** speed ledger | On every aprender tag, speed receipts for the blessed model on the REX-08 cells, using the EXT-28 arms. Ratios are computed only inside the internal ledger (the ratchet input); no ratio appears in any published surface (T28). | 100% of tags since activation have a row per cell, or `NotRun{reason}`. The ratchet arms after 3 records. | 90 |
| **EXT-20** recipe candidates | Prompt/decoding candidates evaluated on the sealed REX suites; a promotion is a MINOR release (recipe shipped in the model repo). | ≥1 candidate evaluated end-to-end under §3.6. The decision (promote/reject) is receipted either way. | 150 |
| **EXT-21** weight candidates | QLoRA and distill candidates built from the review ledger, using external labels only (I-14). | `NotRun{Refused(removed_by=0.71)}` until the refusal lifts. After that, the first candidate is receipted through M2, whichever way it goes. | 240 |
| **EXT-22** fleet export | Nightly JSONL export on the RAID. File infra issues for `docs/MODELS.md` (generated) and the infra#1057 dogfood-model columns. Filed: infra#1179 (https://github.com/paiml/infra/issues/1179), infra#1180 (https://github.com/paiml/infra/issues/1180). | Export idempotence (I-7). The issue URLs are recorded with checkable acceptance criteria. A deleted export row turns the ledger test RED once infra lands (FALSIFY-EXT-010). | 90 |
| **EXT-23** recusal | File the paiml-implement and arbiter issue for I-12. | The issue URL is recorded; the acceptance criteria name the paths in §3.9 and a planted-PR fixture. | 30 |

---

## §6 Hard rules

- **R-1 Rust only.** Shell must be bashrs-clean. No Python, including `huggingface_hub` and the `hf` CLI.
- **R-2 No destructive DB operation** without a backup (with its sha256 recorded) and `--yes`. `gc` defaults to `--dry-run`.
- **R-3 No third-party tracker SaaS.** The only network writes allowed are through EXT-14 and EXT-17 to HF and GHCR, from the driver host.
- **R-4 Fleet measurements and all gates use a released, tarball-sha-verified `apr`,** never HEAD. The one exception is EXT-05's own dev test.
- **R-5 One ticket per session, one PR per row, ≤1 ext PR in CI.** Standard quorum; auto-merge only. The operator never merges by hand.
- **R-6 The metrics DB and the export live on the RAID.** `df` is checked before backfill and before packing.
- **R-7 Credentials.** The HF and GHCR tokens live on the driver host only. They are never in GitHub secrets, never in a workflow, and never in the environment of any training verb (tested).
- **R-8 Published revisions are immutable.** A fix is a new version; a yank never deletes.
- **R-9 The lambda-labs 4090 is excluded** from experiment and eval jobs (REX-001 R-9). When `train-active` is set, these jobs yield on clean-room and CUDA pools.
- **R-10 Every published claim has a receipt id.** A comparative claim needs a comparator block. `[X]` figures appear only when labelled as third-party.
- **R-11 Every measurement states its tree, host and per-session worktree.**
- **R-12 Routing.** Worker `ext` on Opus 5.5 at medium effort. Sonnet 5 for single-module rows (EXT-03, 07, 16, 23). Fable is never used.
- **R-13 No invented thresholds.** Every threshold carries `basis=<receipt>` or `[U]`.
- **R-1a Comparator carve-out.** Competitor arms (llama.cpp, Ollama, PEFT/TRL, Unsloth, MLflow) run as **pinned black boxes** in comparator containers:
  - image digest pinned; lockfile with hashes;
  - network denied after the fetch step;
  - output captured into a comparator block.

  Our code, harness and gates stay Rust. No competitor ever runs in a gate's decision path except as a measured arm.
- **R-14 A new surface ships with its contract.** Every new CLI verb or flag this spec adds ships its `contracts/crux-*.yaml` in the same PR. After EXT-25, `crux_coverage: none` on an EXT surface turns `ci / gate` RED.
- **R-15 Gate on the delta of the gap, not the gap.** A competitor gap blocks only when it **widens** versus our previous release on the same cell/corpus/arm pins. An absolute gap is reported with `gap_effect`, never used as a floor (the first record just establishes the baseline).
- **R-16 Conceded segments get no speed gate** (T27). Unsloth training speed, vLLM-class concurrency and MLX decode are **correctness-only** or excluded, and no claims are made about them.
- **R-17 Like-for-like only.** Quant arms compare the same quant type (Q4_K_M vs Q4_K_M). Dynamic or imatrix variants (e.g. Unsloth "UD") are a separate, labelled arm. Speed arms run the text-only path (Qwen 3.5 is vision-language, T24), m=1, resident.

---

## §7 STOP conditions

- **S-1 D-1 not ruled:** Phase A proceeds; Phase B and later wait.
- **S-2 Isolation needs a CI workflow change:** surface it and ask (workflow changes are a check-in item).
- **S-3 Backfill would need guessing:** leave the field NULL. If more than half of a dir is unknowable, skip it with a reason.
- **S-4 Tracking overhead exceeds the EXT-04 noise band:** batch the writes before continuing.
- **S-5 The license (T19) forbids redistribution of derivatives,** or requires terms we cannot meet: stop all publish rows (EXT-14..18). Tracking continues.
- **S-6 The HF upload needs a non-Rust dependency,** or a protocol that cannot be implemented within 2× K̂: STOP-report. A GHCR mirror alone never counts as "published to HF".
- **S-7 M7 finds a post-publish mismatch:** yank and stop the model train.
- **S-8 A contamination hit** makes the candidate inadmissible. Stop that candidate and report.
- **S-9 A gate would need to live inside the write perimeter** (§3.2): stop and report.
- **S-10 M2 is underpowered by the sample-size rule:** no promotion; report the required n.
- **S-11 Anything needing SSH outside forjar,** a second ext PR in CI, an invented threshold, or a hand-typed card number.

---

## §8 Decisions

**Ruled at review on 2026-09-25 (launching this spec ratifies them):**
- **D-1:** pacha is the system of record; entrenar holds per-step metrics only.
- **D-Q2:** the `.tmp*` test rows are deleted after a backup with its sha256 is recorded.
- **D-Q3:** dogfood models go in a separate, generated `docs/MODELS.md` in infra.
- **D-Q4:** new worker `ext`, on Opus.
- **D-Q5:** Phase A goes to 0.70.x; Phases B and C go to 0.71; Phase D is continuous after EXT-18.

**Open. Answer before Phase C (EXT-11) starts; until then, the recommended option is the default:**

| Id | Question | Recommendation |
|---|---|---|
| **D-2** | GitHub mirror? GitHub Models cannot host our weights (T12). | **GHCR OCI mirror** (EXT-17). Release assets carry only the manifest, card and receipts (T14). |
| **D-3** | HF naming: one repo per model line with version tags, or a version in the repo name (`albor-370m-v1` style)? | **One repo per line, versions as tags** (`paiml/qwen3.5-4b-apr`, provisional name). Migrate albor at its next release. |
| **D-4** | Formats to publish: apr-native only, or also GGUF? | **Both, if T18 holds.** GGUF lets third parties verify with llama.cpp, and independent verification is the strongest proof. If T18 fails, file the export verb for 0.72. |
| **D-5** | Are rc branches public? | **Public.** The proof includes the process: rc → gates → tag. |
| **D-6** | Quant variants in v0.1.0? | **Exactly the one the review lane runs** (one variant = one proof). Add variants as MINOR releases. |
| **D-7** | Promotion semantics: a single global stage (`development→production`), or aliases? MLflow deprecated stages for aliases (T26). | **Aliases.** `@blessed` = the newest released version; lanes keep pinning by sha through forjar. `apr registry promote` sets an alias and never mutates a version. |
| **D-8** | 0.71 must-carry (≤1, T29): which EXT row, if any? | **EXT-25** (CRUX contracts + coverage gate on EXT surfaces). It is cheap, defines every later gate, and has no training dependency. If 0.71 already carries one must-carry, the whole epic rides as non-blocking scope. |
| **D-9** | Quant competitor set for EXT-27 | Q4_K_M from unsloth, bartowski and Ollama (like-for-like), plus Unsloth UD-Q4 as a labelled separate arm. BF16 from upstream is the reference. |

---

## §9 Falsifiers and report

| Id | Falsifier (each ships with the mutation that turns it RED) | Row |
|---|---|---|
| 001 | canary sha unchanged after the workspace test run; removing the env override → RED | EXT-02 |
| 002 | reaper leaves the planted live pid and the planted reused-pid run alone | EXT-03 |
| 003 | PachaBackend round-trip; planted dangling pointer → `fsck` ≠0 | EXT-04 |
| 004 | train-verb delta on a tmp pacha home (runs +1, lineage +3, models +1) | EXT-05 |
| 005 | edge type and parent count per verb (case table) | EXT-06 |
| 006 | lineage cycle refused | EXT-07 |
| 007 | promote without an eval row refused | EXT-12 |
| 008 | backfill never fabricates (NULL where unknown) | EXT-10 |
| 009 | dirty or unidentifiable engine refused at `start_run`; the message names `--no-track` | EXT-05 |
| 010 | deleted export row → ledger test RED | EXT-22 |
| 011 | a training process cannot read the publish token or write the sealed manifest | EXT-04 |
| 012 | planted sealed-item leak refused at dataset registration | EXT-08 |
| 013 | planted unlabeled `self_generated` row refused | EXT-08 |
| 014 | two `apr model pack` runs are byte-identical | EXT-11 |
| 015 | each gate M0–M6 turns RED on its plant; underpowered M2 refuses promotion | EXT-12 |
| 016 | one-byte rc corruption fails M-CR | EXT-13 |
| 017 | planted post-publish mismatch → yank + STOP; a yanked tag stays fetchable | EXT-15 |
| 018 | a hand-typed card number fails the lint | EXT-16 |
| 019 | an EXT surface with no CRUX contract turns `ci / gate` RED; deleting one contract → RED | EXT-25 |
| 020 | a comparator arm missing any field of `{command, version, env_sha256, artifact_sha256, log_path}` blocks | EXT-26 |
| 021 | artifact-quality gate: a planted degraded quant (one tensor re-quantized to Q2_K) widens the gap → RED | EXT-27 |
| 022 | speed ratchet: a planted sleep in the decode loop widens the gap vs the llama.cpp arm → RED at release phase | EXT-28 |
| 023 | stock-baseline gate: a candidate planted to regress below upstream stock on one sealed suite → promotion refused | EXT-30 |
| 024 | the card lint rejects a ratio token next to a competitor name | EXT-16 |

**Final report (one per session, terse):**

```
EXT-001 | row=<id> | verdict=<MERGED|STOP(reason)|NOOP|NotRun(reason)> | main=<sha> | PR=<url>
store:   pacha models/runs/lineage/evals=<n/n/n/n> host=<h> tree=<HEAD|origin/main> | fsck=<0|n dangling>
hygiene: canary=<unchanged|CHANGED!> | running_nonlive=<n> | tmp_rows=<n>
model:   line=<line> version=<v> channel=<rc|released|yanked> | M-CR=<g/r> M0..M6=<g/r ×7> M7=<g/r|n/a>
proof:   HF=<revision|none> GHCR=<digest|none> | card_unmapped_numbers=<n>
speed:   apr=<v> cell=<id> ttft_ratio=<x> e2e_ratio=<x> vs llama.cpp d1d3c3396 | receipt=<path>
quality: suites=<k> vs incumbent=<v> | verdict=<promote|reject|underpowered(n_req)> | receipt=<path>
found:   aprender/infra issues filed=<ids|none>
next:    <row id>
```

---

## §10 CRUX competitors and gates

Each **surface** this spec creates or depends on gets a CRUX contract: category + surface + competitor + comparator + metric + `gap_effect`. The rules that govern them are R-14 through R-17. EXT-24 pins every competitor arm; for runnable arms, the pin includes the HF revision or image digest and sha256.

**v1.2 (T30):** at `316dee2cd4` a contract can hold only one field from that list: `competitor`, and only for the 15 registered names. `category` is dropped at parse, and `comparator`, `metric` and `gap_effect` have no schema, so a contract written to this table today loses them silently. The bind census proposes splitting EXT-24 so that the schema exists before EXT-25 writes contracts: EXT-24a registers the missing competitors, EXT-24b adds a contract block that reuses the §3.C `gap_effect` vocabulary and the §3.C.1 comparator block, and EXT-24c settles the colliding letters (look-ahead ruling R-71-4). These remain proposals until the cop mints them as rows. Before EXT-24b adds a block, it checks the existing `Beat` block (`metric`, `incumbent`, `direction`; BEAT-001..007) so that it does not duplicate it (PR-REVIEW §3.A).

| # | Category · surface | Competitors (arms) | Comparator | Metric (same tool for every arm) | Gate | Row |
|---|---|---|---|---|---|---|
| **C1** | `model-artifact` · the released quant file | `unsloth/Qwen3.5-4B-GGUF` Q4_K_M; `bartowski/Qwen_Qwen3.5-4B-GGUF` Q4_K_M; Ollama `qwen3.5:4b` (Q4_K_M blob); Unsloth UD-Q4 (labelled separate arm); **reference:** upstream BF16 | hermetic: pinned revision + sha256, measured with llama.cpp `d1d3c3396` | KL divergence vs BF16 logits and top-1 agreement, on a committed text corpus; also file bytes | **blocks when our gap widens** vs our previous release (R-15); the first release sets the baseline | EXT-27 |
| **C2** | `inference` · `apr serve` / `apr run` on the blessed model | llama.cpp `d1d3c3396` (existing oracle); Ollama (pinned version); mistral.rs (Rust peer; `Refused{…}` recorded if it cannot load `qwen35`) | hermetic | TTFT, ITL, e2e, load time, peak RSS/VRAM; m=1, resident, text-only, per REX-08 cell | **Reuse the existing CRUX inference contracts**; add the model binding only (no duplicates, PR-REVIEW §3.A). Ratio vs llama.cpp is a shrink-only ratchet at release phase; 0.70 theme target TTFT ≤ 2× `[C]` | EXT-28 |
| **C3** | `training` · `apr finetune --qlora` / `distill` on `qwen35` | HF PEFT + TRL (reference implementation); Unsloth (speed is **conceded**, R-16) | hermetic container, lockfile with hashes | **Correctness:** step-0 loss and gradient cosine vs PEFT on identical data, seed and LoRA config; loss trajectory inside the reference's own seed band (band measured from k reference seeds, `[U]` until measured); final sealed-suite eval of the adapter. Speed is report-only | correctness blocks once the band exists; before that, report-only | EXT-29 |
| **C4** | `quantize` · `apr quantize` | llama.cpp `llama-quantize` (+ `llama-imatrix`) | hermetic | Output measured via C1; byte-determinism across runs and hosts (ours is required by I-7 and FALSIFY-014; theirs is recorded) | covered by C1 + FALSIFY-014 | EXT-27 |
| **C5** | `tracking` · `apr runs`, `apr registry lineage`, `apr registry promote` | MLflow 3 (LoggedModel lineage, aliases); DVC (content-addressed data/model versions); W&B (`comparator: unrunnable-hermetically`: SaaS, no artifact sha, so "N× W&B" can never be stated) | behavioural fixtures; MLflow and DVC pinned in containers | Behaviours, each a fixture: (a) one command answers "which run, dataset sha and base produced model X"; (b) dataset identity is content-addressed; (c) promotion is an alias, never a mutation (D-7); (d) works offline with no server; (e) write overhead per metric row (report-only) | coverage only (R-14); `gap_effect` recorded | EXT-31 |
| **C6** | `distribution` · the HF model line + card + release process | Qwen official, unsloth, bartowski, Ollama library | behavioural fixtures on pinned revisions | (a) re-download by version tag is sha-stable; (b) tags are immutable (competitors publish by committing to `main`, T22); (c) every card number is receipt-backed; (d) a build receipt / lineage is published | coverage only; our (a)–(d) are already blocking via M-CR, M6, M7 and I-8 | EXT-31 |
| **C7** | `decision` · the review-lane task on the sealed REX suites | **upstream stock Qwen3.5-4B via llama.cpp** (the baseline every MINOR must not lose to); Qwen3.5-9B control; Haiku 4.5 and the agy lane (API arms: model id + version recorded, no artifact sha, report-only); TypeSafe Jev (existing `crux-decision` contract, unrunnable) | stock and 9B: hermetic; API arms: non-hermetic, report-only | the same §3.6 statistics as M2 | **blocks promotion:** a candidate must be non-inferior to upstream stock on every sealed suite (Holm) in addition to the M2 incumbent rule | EXT-30 |

**Watch signals (not gates):** a new upstream family (Qwen3.6, T31) triggers a MAJOR-line evaluation issue, filed and never auto-adopted. A new competitor quant revision triggers re-pinning in EXT-24; the gap baseline is re-established and labelled `rebased`.

### §10.1 Rows (after EXT-18 unless noted)

| Row | Work | Done when | K̂ |
|---|---|---|---|
| **EXT-24** CRUX bind (**runs before EXT-11**) | Enumerate the existing `contracts/crux-*.yaml` by category using `pmat query` over the loader, not grep. Resolve whether `category` is a closed enum and who owns it (**answered in v1.2, T30:** it is not an enum and is not parsed at all, so there is no enum edit to own; what remains open is the colliding letters, ruling R-71-4). Pin every arm in the table above (HF revision + sha256, Ollama manifest digest, container digests, lockfiles). Emit `crux-bind-receipt.json` with a coverage matrix (EXT surface × contract). | Every arm is pinned or `Refused/NotRun{reason}`; the matrix lists each EXT surface; the existing inference contracts reused by C2 are named by path. Enum owned elsewhere → file an issue, do not fork (APR-DECIDE §11); moot since v1.2, because there is no enum (T30). | 60 |
| **EXT-25** contracts + coverage gate (**D-8 must-carry candidate**) | Write one `contracts/crux-<category>-<surface>-v1.yaml` per new surface (C1, C3–C7; C2 extends the existing ones), with honest initial `gap_effect`, once the schema can hold it (T30; until then the parser drops it). **Not docs-only (v1.2):** `category` needs no enum edit (T30), but `competitor` is closed (CRUX-002). The bind census found 7 arm owners in §10 that are not in `CRUX_COMPETITORS`: unsloth, bartowski, mistral_rs, mlflow, dvc, wandb and qwen. Registering them is a `validator.rs` change, and it lands first. The `crux-<category>-<surface>` file name waits on R-71-4, because contract ids use one letter and the letters collide. Add a `ci / gate` check: an EXT surface with no contract → RED. File the infra issue for the fleet-wide version (APR-DECIDE CX-J-2) if it has not landed. | FALSIFY-019 RED→GREEN observed in the required check; `pv validate` counts the new obligations; 0 anonymous obligations. | 120 |
| **EXT-26** comparator harness | Rust driver plus pinned comparator containers (R-1a). Every arm run writes a comparator block `{command, version, env_sha256, artifact_sha256, log_path}` into its receipt. The existing `perf_gate` / `AbRecord` is reused if present at HEAD (`[U]`, bind with `pmat query "AbRecord"`). | FALSIFY-020; two identical runs give identical comparator blocks except timestamps. | 180 |
| **EXT-27** artifact-quality arms (C1, C4) | KL divergence vs BF16 and top-1 agreement for our quant and the D-9 arms, on the committed corpus, with the llama.cpp pin. Gap computed per arm and ratchet armed (R-15). Wired into the M1 stage as `M1b`. | First record committed as baseline; FALSIFY-021; the card renders absolute values per arm with no ratio. | 150 |
| **EXT-28** speed arms (C2) | Bind the blessed model to the existing inference contracts; add the Ollama and mistral.rs arms (pinned) next to llama.cpp; release phase per APR-PERF-GATE-001 §4.1 (merge phase stays wall-clock-free). | 100% of ruled cells have a receipt or `NotRun`; FALSIFY-022; ratchet arms after 3 records. | 120 |
| **EXT-29** training correctness arms (C3) | PEFT/TRL reference arm; step-0 loss and gradient cosine; k-seed reference band. | `NotRun{Refused(removed_by=0.71)}` until the refusal lifts. After that: band receipted, then the first candidate is measured against it. | 240 |
| **EXT-30** stock-baseline gate (C7) | Extend M2 with the upstream-stock non-inferiority arm; API arms report-only. | FALSIFY-023; the first release records the stock baseline (no improvement claim). | 90 |
| **EXT-31** behavioural contracts (C5, C6) | Fixtures for C5 (a)–(e) and C6 (a)–(d). Competitor behaviour is recorded from pinned versions and docs as `[X]`, and `gap_effect` is computed. | Each fixture has a planted failure (e.g. a mutated promote that rewrites a version → RED); contract `gap_effect` values are rendered into `docs/MODELS.md` (infra, filed). This needs the schema first (T30). | 90 |

### §10.2 STOP additions

- **S-12** A competitor arm cannot be pinned hermetically (no digest, no revision): mark it `comparator: unrunnable-hermetically`. It is report-only, and no claim may ever cite it.
- **S-13** A blocking arm would require a conceded-segment speed race (R-16): drop the arm and report.
- **S-14** Arms are not like-for-like (different quant type, vision path included, m>1): refuse the comparison; do not normalise it.

---

## §11 0.71 release epic

**The traffic cop executes this section. It is the sole minter, and none of this is done by hand.**

1. **Epic.**
   - Issue title: `EXT-001: dogfood model lifecycle — tracking, model release train, CRUX gates`.
   - Milestone: `0.71`. Labels: `epic`, `crux-competitor`, `dogfood`.
   - **Not** `gate-framework-new`: the CRUX gates use the existing `contracts/crux-*.yaml` schema, and the model gates are product verbs.
     - **v1.2:** this premise is false at `316dee2cd4`. The existing schema cannot hold `comparator`, `metric` or `gap_effect` (T30), so the CRUX gates need a schema extension (EXT-24b).
     - Whether that extension needs the `gate-framework-new` label, which 0.71 excludes (T29), is look-ahead ruling R-71-5. It is still open.
   - Body: this spec's path at `origin/main`, the §2.1 target table, and the D-* table with current answers.
2. **Children.** One `pmat work add` ticket per row, with one fragment per ticket at `docs/roadmaps/entries/<TICKET>.yaml` (never a direct `roadmap.yaml` edit). Each child is linked to the epic with `keeps-open`, never with a closing keyword.

| Train | Rows | Note |
|---|---|---|
| **0.70.x** | EXT-00..EXT-03 | Hygiene, with no `crux-competitor` label (0.70 excludes it, T29). |
| **0.71** (epic milestone) | EXT-24, **EXT-25 (must-carry per D-8)**, EXT-04..EXT-10, EXT-26, EXT-11..EXT-13 | Order: EXT-24 → EXT-25 → Phase B → EXT-26 → packager/gates/clean-room. |
| **0.71 → rolls forward** | EXT-14..EXT-18, EXT-27, EXT-28, EXT-30, EXT-31 | Scope, not dates, moves (train rule 7): anything unmerged at the 09-29 cut rolls to 0.72 automatically. |
| **gated on the 0.71 refusal lift** | EXT-21, EXT-29 | `NotRun` until `qwen35` finetune/distill ships. |
| **continuous** | EXT-19, EXT-20, EXT-22, EXT-23 | After EXT-18. |

3. **Exit criteria for the epic** (the epic closes only when all of these hold):
   - `v0.1.0` is tagged on HF with M-CR, M0–M7 and M1b green;
   - all C1–C7 contracts exist with `crux_coverage` complete for the EXT surfaces;
   - the ratchets for C1 and C2 are armed (3 records each);
   - FALSIFY-EXT-001..024 are all mapped to CI-wired tests with their mutations observed RED in the required check.

---

## Appendix — mapping from the design basis (arXiv:2607.04277 `[X]`) to mechanism

| Paper result | What it forbids or requires here | Mechanism |
|---|---|---|
| Theorem A3: improvement is undecidable | No universal "is it better" check | Bounded sealed suites + a pre-registered statistical rule (M2); underpowered → no promotion |
| Corollary A1: no internal hard invariants | Gates inside the loop are soft | §3.2 perimeter; I-9, I-12, R-7; base-owned gate workflow |
| §2.4 Huang et al.: self-correction fails without external feedback | The model cannot grade itself | External labels only (I-14); llama.cpp parity oracle; human HRQ |
| §2.4 model collapse | Recursive self-training degrades | Origin per row; synthetic fraction rendered; unlabeled self-output refused |
| §6.3.1 the gate sits outside the write perimeter | Human/institutional boundary | Recusal (I-12); operator decisions D-*; quorum on perimeter paths |
