# research-design-pr-agent-qwen-3.5.md — PRA-001: Capture-First Trace Corpus for Distilling a Local Qwen 3.5 PR Reviewer

Every quorum is currently throwing away its training pairs. The fix is to capture every lane's exact input bytes, full raw output and findings text at dispatch, under a content-addressed and provenance-tagged `agent-trace-v1` contract, before any training happens. Silver and gold labels stay separable on every row, and a measured lane-independence guard (error-kappa) controls whether silver distillation is allowed at all.

## TL;DR
- **Capture before training.** `review-ledger-v1` stores findings as a count, so each of the 50–200 quorums/day [A] loses its (input, output, findings) triples. PRA-001 moves capture to dispatch inside the quorum rail (paiml-implement #436). Each row stores the exact input bytes, the full raw reply, decoding params, top-k=20 [A] logits for local lanes, provider, exact model id, access channel and terms reference. Rows go into content-addressed zstd blobs on almacen, with a nightly second copy on separate media.
- **Silver is allowed but tagged and fenced.** The literature supports a conservative design. Top-k logit caches give biased teacher estimates (Anshumann et al., ACL 2025: top-K caching gives "biased estimates of teacher probability distribution"), and so does sequence-level KD alone. Code-review corpora carry heavy label noise: Liu et al. found only 64% of the original CodeReviewer comments were valid. They also carry near-duplicate/temporal leakage: Lee et al. (ACL 2022) found train-test overlap in over 4% of standard validation sets, and Allamanis found code-duplication inflates metrics by up to 100%. LLM errors are strongly correlated across providers: Kim et al. (ICML 2025) found model pairs agree about 60% of the time when both are wrong on HELM, against 1/3 by chance. So every row carries `label_tier`, `provider`, `model_id` and `terms_ref`. Splits are grouped by repo#PR and ordered by time, and outcome labels mature only after the 14-day [A] revert window.
- **The dissent lane has to stay independent.** Pairwise Cohen's κ on errors against gold is measured before and after any silver training. If κ rises above the pre-registered baseline + margin [A], the silver-trained candidate is refused as the Qwen shadow lane; it may still be promoted elsewhere via PRM-001 §5.4. Anthropic's and Google's current terms both contain competing-model/ML-development restrictions (summarised in §2.10 and Appendix A.7, not legal advice). The operator has ruled S-6, so the design's job is to make any provider's rows filterable later.

---

## Header

| Field | Value |
|---|---|
| Spec id | **PRA-001** (PR-Review Agent trace corpus) |
| Title | Research design: collecting PR-review agent data to fine-tune and distill a local Qwen 3.5 reviewer |
| Author | Noah Gift, Pragmatic AI Labs (paiml), Sovereign AI Stack |
| Date | 2026-09-25 |
| Target repo / path | `paiml/aprender` → `docs/specifications/research-design-pr-agent-qwen-3.5.md` |
| Runner | the aprender traffic cop |
| Launch line | `Implement docs/specifications/research-design-pr-agent-qwen-3.5.md autonomously.` |
| Related specs | REX-001 / PRM-001 "Prometheus" (H1–H6, champion/challenger §5.4, sealed-test rotation after 20 [A] promotion decisions); ARB-SELF-001 (dogfood loop, outcome ledger, escape join); ARB-APR-001 (arbiter on apr); FLOW-001 (fleet flow rules); EXT-001 (continuous HF release of the dogfooded model) |
| Contracts introduced | `agent-trace-v1`, `review-ledger-v2`, `trace-blob-store-v1`, `trace-backup-v1`, `transcript-retention-v1`, `trace-admission-secret-v1`, `trace-dedup-v1`, `trace-split-guard-v1`, `trace-outcome-join-v1`, `sparse-logits-v1`, `lane-independence-v1`, `trace-datacard-v1` |
| Contracts reused | `review-corpus-contamination-v1` (sealed test, corpus `review-corpus-v1`) |

**Provenance marks (apply to every number in this document)**

| Mark | Meaning | Obligation |
|---|---|---|
| **[V]** | Verified by a measurement command in this repo/fleet | The command is quoted beside the number |
| **[C]** | Computed from other marked numbers | The arithmetic is shown; it inherits the weakest input mark |
| **[A]** | Asserted (operator decision or design choice) | Changing it is a spec change, not a tuning knob |
| **[U]** | Unmeasured estimate | Must be replaced by [V] at the first scheduled probe (§3), or the row goes to andon |
| **[X]** | Third-party figure (paper, vendor, docs) | Cited in Appendix A; never used as a gate threshold without local re-measurement |

Rule: there are no invented thresholds. A gate threshold is either [V] (a measured baseline plus a declared ratchet) or [A] (an explicit policy choice such as 0 or 100%).

---

## §0 Operating assumptions

1. **Capture-first.** Anything not captured at dispatch is lost forever. Capture comes before curation, curation before training, and training before promotion. No sampling and no TTL [A].
2. **Silver and gold are separable per row, forever.** `label_tier ∈ {silver, gold, pending, quarantined}` together with `provider` and `model_id` is enough to filter any row out of any pool without re-deriving anything.
3. **The sealed test is inviolable.** The sealed test (105 items, 70 defects [A], corpus `review-corpus-v1`) is never in any training pool. Zero sealed hashes in any pool [A]. Rotation follows PRM-001 only.
4. **The shadow lane is zero-weight.** The Qwen row is mandatory in every receipt, but it never blocks, breaks a tie or escalates. PRA-001 changes what is *recorded*, not how votes are counted.
5. **The producer is never the gate.** The process that writes a trace never admits it to the training pool. Admission (§4) is a separate binary with its own contract.
6. **The operator has no pre-steps.** Anything only the operator can do (credentials, media purchase, ToS decisions, sealed-corpus changes) is a §8 STOP, not a hidden prerequisite.
7. **Pure Rust + bashrs-clean shell. No Python anywhere.** That includes training-data tooling, scanners and dedup. Provisioning uses only `forjar apply` or make targets, with released and sha-verified `apr` tags. Weights are pinned by sha256.
8. **Legal posture.** Provider terms are recorded as row metadata (§2.10). This document is **not legal advice**. The operator's S-6 ruling governs; PRA-001 only guarantees that rows can be filtered later.

---

## §1 Ground truth (frozen baseline, 2026-09-25)

Every operator-supplied fact is [A] until ticket T0 converts it to [V] by running the listed command. If a measured value disagrees with the [A] value, that is a §8 STOP (baseline drift).

| # | Fact | Mark | Source of truth | Verify command (T0) |
|---|---|---|---|---|
| G1 | Quorums/day across paiml repos: 50–200 | [A] | quorum receipts | `make trace-yield WINDOW=7d` |
| G2 | Counted lanes: Claude Sonnet 5, Claude Haiku 4.5, agy (Antigravity / Gemini 3.1 Pro); shadow: Qwen3.5-4B-Q4_K_M via `apr serve` | [A] | paiml-implement rail config | `pmat quorum lanes --json` |
| G3 | Qwen row states: Verdict \| Refused \| NotRun{NoExecutor\|Busy\|Timeout\|ContextOverflow\|TrainActive} | [A] | receipt schema | receipt schema test |
| G4 | Dispatch ladder: gx10-cuda → lambda-cuda (shadow-only, dispatch-time, GPU-lock free, never resident) → intel-wgpu → mini-metal → intel-cpu → NotRun{NoExecutor}; row records the backend that actually ran | [A] | rail dispatcher | `make trace-served-by-audit` |
| G5 | `review-ledger-v1` keeps {family, model, verdict, findings: count} only; text/inputs/outputs discarded | [A] | `crates/aprender-review-experiment/src/ledger.rs` | `grep -n 'findings' crates/aprender-review-experiment/src/ledger.rs` |
| G6 | S-6: Claude and agy outputs MAY train the local model (silver) | [A] | operator ruling S-6 | n/a (ruling) |
| G7 | Gold = sealed corpus + outcome joins (merged / reverted ≤ 14 days / `regression`-labelled escape) | [A] | ARB-SELF-001 outcome ledger | `make trace-outcome-join --dry-run` |
| G8 | Sealed test: 105 items, 70 defects, corpus `review-corpus-v1` | [A] | `review-corpus-contamination-v1` | `pv check review-corpus-contamination-v1` |
| G9 | Storage: almacen UNAS Pro, 3×16 TB RAID 5, 32 TB usable, 10G | [A] | forjar inventory | `forjar inventory almacen --json` |
| G10 | Raw growth ≤ ~0.45 TB/yr | [U] | estimate | `make trace-bytes WINDOW=7d` (replace in week 1) |
| G11 | A real AWS-key literal has already landed in a diff | [A] | incident record | `make trace-secret-scan-backfill` |
| G12 | `apr finetune/distill/merge` refuse qwen35 until aprender 0.71; `apr quantize` available; `--json-schema` planned 0.70 | [A] | aprender release notes | `apr --version && apr finetune --model-arch qwen35 --dry-run` (expect refusal) |
| G13 | Qwen3.5-27B teacher on gx10 can write top-k logits | [A] | apr capability | `make teacher-logit-smoke` |
| G14 | Claude Code stores transcripts in plaintext under `~/.claude/projects/` "for 30 days by default" (Claude Code docs); Desktop/Cowork sessions exempt by default | [X] | Claude Code docs (App. A.8) | `make transcript-retention-audit` |

---

### §1.1 Amendment (2026-09-25, T0): verify commands that exist

T0 (`evidence/pra-001/t0-baseline.yaml`) found that none of the §1 verify commands exist as written: the seven `make` targets, `pmat quorum`, `pv check` and `apr --model-arch`. Cop ruling (2026-09-25): a missing instrument is not a contradiction. G1, G7, G10, G11 and G13 are **[U]-pending-instrument**, each tied to the ticket that builds it. G2 and G4 are paiml-implement#436 scope (T1, R-9). This table replaces the §1 "Verify command" column. A `make` target named in §1 lands with its ticket and replaces the interim command here.

| # | Verify command (real, today) | Until then |
|---|---|---|
| G1 | lands with T1: `trace::weekly` row count over the index | [U]-pending-instrument (T1) |
| G2 | `jq '.quorum' ~/.claude/skills/paiml-implement/config.json` | STOP S-BASE → #436 scope |
| G3 | `cargo test -p aprender-review-experiment --lib every_wire_form_round_trips` | — |
| G4 | the #436 dispatcher's row `served_by` audit; no dispatcher exists yet | STOP S-BASE → #436 R-9 |
| G5 | `grep -n 'findings' crates/aprender-review-experiment/src/ledger.rs` | — |
| G6 | n/a (ruling) | — |
| G7 | lands with T10 (`trace-outcome-join-v1`) | [U]-pending-instrument (T10) |
| G8 | `pv validate contracts/review-corpus-contamination-v1.yaml && cargo test -p aprender-review-experiment --lib contamination` | — |
| G9 | `df -B1` on the almacen mount, on a host that mounts it | — |
| G10 | lands with T1: `trace::weekly` `bytes_raw`/`bytes_zstd` over week 1 | [U]-pending-instrument (T1) |
| G11 | lands with T6 (secret-scan backfill over the index) | [U]-pending-instrument (T6) |
| G12 | `. scripts/apr_bin.sh && "$APR" --version && "$APR" finetune <qwen35.gguf> --plan` (expect a refusal; `apr` has no `--model-arch`) | STOP S-BASE: `--plan` exits 0 with no refusal on 0.69.3 |
| G13 | lands with T16 (27B teacher on gx10, infra#1132) | [U]-pending-instrument (T16) |
| G14 | per host: `jq .cleanupPeriodDays ~/.claude/settings.json` and `find ~/.claude/projects -mtime +30 \| wc -l` | — |

---

## §2 Design

### §2.1 Capture points

| Stream | Where | When | Why here |
|---|---|---|---|
| **S1: lane traces** | Inside the quorum rail (paiml-implement, issue #436), at the dispatch boundary of each lane call | Synchronously, before the verdict is parsed and before the receipt is written | This is the only point where the exact input bytes, the raw reply and the actual backend co-exist. Parsing is lossy and receipts are summaries. |
| **S2: session transcripts** | Claude Code `~/.claude/projects/**/*.jsonl` (including `subagents/`), agy logs | Harvested by a forjar-declared timer on each workstation into almacen; never edited in place | For later agentic (multi-turn, tool-use) distillation. Retention is set by config, not by hope (§2.8). |

Mechanism for S1: the rail wraps every lane call in `TraceSpan::open(lane, round)`. The span hashes the request bytes *as serialised to the wire*, not a reconstruction. It streams the response into a blob writer while the lane client reads it. It closes with the parsed verdict and findings. If the blob write fails, the receipt row is written with `trace_status = CaptureFailed` and andon fires. The quorum itself is never blocked (jidoka on the capture line, not on the review line).

### §2.2 `agent-trace-v1` row schema (one row per lane per round)

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema` | const `"agent-trace-v1"` | ✔ | versioned; readers reject unknown majors |
| `trace_id` | ULID | ✔ | time-sortable |
| `quorum_id` | string | ✔ | joins to receipt |
| `round` | u16 | ✔ | re-review rounds within a quorum |
| `lane` | enum {sonnet, haiku, agy, qwen-shadow, teacher-27b, …} | ✔ | |
| `counted` | bool | ✔ | false for qwen-shadow and teacher |
| `provider` | enum {anthropic, google, local} | ✔ | filter key for terms (§2.10) |
| `model_id` | string (exact, e.g. dated snapshot id or weights sha256) | ✔ | never a family alias |
| `access_channel` | enum {anthropic-api, claude-code-cli, antigravity, gemini-api, vertex, apr-serve} | ✔ | terms differ by channel (App. A.7) |
| `terms_ref` | {url, effective_date, fetched_at} | ✔ | snapshot of governing terms at capture |
| `repo`, `pr`, `head_sha`, `base_sha` | string/u64 | ✔ | |
| `diff_sha256` | hex | ✔ | sha of the exact diff bytes sent |
| `input_sha` | hex | ✔ | sha of the full wire request (system prompt + prompt version + diff + tools) |
| `input_parts` | [{role, blob_sha, bytes}] | ✔ | enables dedup of shared system prompts/diffs across lanes |
| `prompt_version` | semver | ✔ | |
| `output_sha` | hex | ✔ | full raw reply, including any returned reasoning; never truncated |
| `decoding` | {temperature, top_p, top_k, max_tokens, seed?, json_schema_sha?} | ✔ | provider-reported where available, otherwise as sent |
| `tokens` | {input, output, reasoning?, cached?} | ✔ | provider-reported; `null` is not the same as 0 |
| `latency_ms` | {ttft?, total} | ✔ | |
| `served_by` | string (cell that actually ran, e.g. `gx10-cpu`) | ✔ | records parity-guard fallback |
| `lane_state` | enum Verdict \| Refused \| NotRun{…} | ✔ | NotRun rows carry no output blob but are still rows |
| `verdict` | enum {approve, request_changes, comment, abstain} | if Verdict | |
| `findings` | [{file, line_start, line_end, severity, category, text}] | if Verdict | **text, not count** |
| `parse_status` | enum {ok, repaired, failed} | ✔ | failed parses keep raw output |
| `logits_sha` | hex \| null | local lanes | `sparse-logits-v1` blob (§2.7) |
| `label_tier` | enum {silver, gold, pending, quarantined} | ✔ | set by admission, never by producer |
| `outcome` | enum {merged, reverted_le14d, regression_escape, closed_unmerged, pending} | ✔ | via `trace-outcome-join-v1` |
| `split_guard` | {group_key: "repo#pr", split: train\|val\|test\|sealed, assigned_at, dedup_cluster} | ✔ | §2.9 |
| `secret_scan` | {scanner_versions, hits, status} | ✔ | §4 |
| `trace_status` | enum {ok, capture_failed, partial} | ✔ | |
| `producer` | {binary, version, git_sha} | ✔ | provenance of the writer |

### §2.3 Blob layout on almacen

```
/almacen/traces/
  blobs/sha256/ab/cd/<sha256>.zst        # content-addressed, zstd, immutable, write-once
  index/agent-trace-v1/YYYY/MM/DD.jsonl  # uncompressed JSONL index, append-only
  transcripts/claude-code/<host>/<sha256>.zst
  transcripts/agy/<host>/<sha256>.zst
  quarantine/sha256/…                    # secret-hit blobs, mode 0400, excluded from pool
  manifests/YYYY-MM-DD.sha256            # daily Merkle-style manifest of new blobs
  datacard/                              # Croissant + RAI metadata (§2.11)
```

- The address is the sha256 of the **uncompressed** bytes. The compression level is recorded in the index, never implied by the path. This allows recompression, or later dictionary training, without re-addressing.
- Blobs are write-once (`O_EXCL`), and existing addresses are skipped, which gives free dedup of shared system prompts and diffs across the 4 lanes. The zstd level defaults to 19 [A] for archival blobs and 3 [A] for hot index rotation; replace both with the measured ratio from §3 M8.
- The index is uncompressed so that `grep`/`jq`-class Rust tools and the mining index (§2.9) can read it without decompression.

### §2.4 Second copy (RAID 5 is not a backup)

- Nightly `trace-backup-v1`: every blob named in `manifests/<date>.sha256` is copied to separate media on a separate host. This is 3-2-1 in shape: primary almacen, second media, and a third offsite copy as a §8 STOP pending operator media.
- Weekly restore drill: sample N=100 [A] random blobs from the second copy, decompress them, re-hash them, and compare. Target mismatch = 0 [A].
- Rationale (computed, worst case): a degraded 3×16 TB RAID 5 rebuild reads the two surviving 16 TB members, 32 TB ≈ 2.56×10¹⁴ bits [C]. At a consumer URE spec of 1 per 10¹⁴ bits [X], the expected UREs are ≈ 2.56 [C] and P(≥1) ≈ 1 − e^−2.56 ≈ 0.92 [C]. At an enterprise spec of 1 per 10¹⁵ [X], P ≈ 0.23 [C]. Spec rates are warranty floors, and field rates are lower [X]. But "kept forever" plus a single-parity array is not a durability story. The second copy is mandatory.

### §2.5 Index schema (JSONL, one object per row)

This is the full `agent-trace-v1` row minus blob contents, plus the mining columns:

| Column | Definition |
|---|---|
| `agreed` | all counted lanes returned the same verdict in this round |
| `lane_disagreement` | bitmask of lanes that differ from the counted majority (Qwen included, as a measurement only) |
| `outcome` | as §2.2 |
| `label_tier` | as §2.2 |
| `unique_diff` | true if this `diff_sha256` is not in any earlier row and not in a near-dup cluster already seen (§2.9) |
| `dedup_cluster` | MinHash-LSH cluster id over diff shingles |
| `outcome_matured_at` | timestamp when the 14-day [A] revert window closed; null means pending |

### §2.6 Retention

Keep forever. No sampling, no TTL [A]. A row is never deleted. The only operation is to *quarantine* it (move the blob, flip `label_tier`). Deletion requests (for example, a leaked secret that must be purged from all copies) are a §8 STOP handled by the operator. The index records the tombstone.

### §2.7 Logit storage format (`sparse-logits-v1`)

The design choice is to store top-k **plus the residual mass**, and to record the sampling policy. Future distillation objectives (forward KL, reverse KL, on-policy) need more than the top-k ids.

Per output token t:

```
struct TokenLogits {
  token_id_sampled: u32,
  k: u8,                    // 20 [A]
  ids: [u32; k],            // u32 because Qwen vocab > 2^16 [A: verify via `apr inspect`]
  logprobs: [f16; k],       // log-softmax at temperature 1.0, pre-sampling-filters
  logsumexp_full: f32,      // normaliser over full vocab
  topk_mass: f16,           // Σ exp(logprobs) = retained mass M ≤ 1
}
```

- Header per blob: `{model_sha256, tokenizer_sha256, apr_tag, backend(served_by), temperature_of_record, k, n_tokens}`.
- Encoding: columnar (all ids, then all logprobs), then zstd. Size ≈ 20 × (4+2) + 8 ≈ 128 B/token raw [C]. That is ≈ 256 KB [C] for a 2,000-token [U] review.
- Why residual mass: naive top-k caching gives biased estimates of the teacher distribution and hurts calibration [X, App. A.1]. A fused forward-KL objective over a sparse top-K teacher explicitly uses retained mass M ≤ 1 [X]. Storing `topk_mass` and `logsumexp_full` keeps the "tail bucket" correction available.
- Option reserved: `sparse-logits-v1` has a `mode` field {topk, rs-sample}. It allows importance-sampled (RS-KD-style) token sets to be written by the 27B teacher later without a schema change [X, App. A.1].
- Frontier lanes (Claude, agy) provide no full logits. Their rows carry `logits_sha = null`; sequence-level KD only.

### §2.8 Session transcripts (S2)

- `transcript-retention-v1`: `forjar apply` writes Claude Code **managed settings** fleet-wide with `cleanupPeriodDays` set to a large value (36500 [A]). The value must be ≥1 [X]; 0 fails validation in current versions [X], and in an earlier version it silently disabled persistence [X]. Managed settings take precedence over user settings [X].
- The harvester copies `projects/**/*.jsonl` **and** `projects/**/subagents/*.jsonl` to almacen hourly [A]. These are content-addressed, so re-harvesting an appended file stores a new blob, and the index links versions by `session_id`.
- Transcripts are plaintext and include tool outputs, file contents and anything a command printed [X]. They are the highest secret-risk stream, so every transcript blob passes §4 G-SEC before it is eligible for any pool.
- Falsifier: plant a canary transcript older than 31 days [A] on a test host. Run a Claude Code session start. Assert the canary survives.

### §2.9 Split policy (`trace-split-guard-v1`)

1. **Group key = `repo#pr`.** Every row, round and re-review of a PR goes to one split [A]. Rows are never split individually. Re-reviews are near-duplicates.
2. **Time-ordered.** Splits are assigned by PR *open time* with forward-chaining cutoffs: train < val < test in time [A]. There is an embargo gap ≥ 14 days [A] between train cutoff and val start, so that outcome labels in val cannot leak through revert chains.
3. **Cross-PR near-dup closure.** Cherry-picks, backports and re-opened PRs produce near-identical diffs under different keys. Diff shingles are MinHash-LSH clustered. A union-find closure assigns every cluster to the split of its **earliest** member [A]. Target cross-split cluster count = 0 [A].
4. **Sealed firewall.** Any row whose `diff_sha256` or cluster matches `review-corpus-v1` sealed items gets `split = sealed` and is never in a pool (G-CON, §4).
5. **Outcome maturity.** A row is eligible for gold only after `outcome_matured_at` is set. Pending is never treated as negative [A].

Mining at training time: oversample `lane_disagreement ≠ 0` and `outcome ∈ {reverted_le14d, regression_escape}`. The oversampling factor is chosen per run and logged; it is not a spec constant.

### §2.10 Provider-terms provenance (tagging requirement, not advice)

Every row carries `provider`, `model_id`, `access_channel` and `terms_ref`. The governing terms differ by channel (App. A.7). The pool builder takes a `--exclude-provider` / `--exclude-channel` filter as a first-class argument. Rebuilding any pool without a given source is a single command, and it is falsified in §9 (P-TERMS). **Not legal advice; S-6 governs.**

### §2.11 Dataset documentation (`trace-datacard-v1`)

- Machine-readable: a Croissant JSON-LD file (dataset metadata, resources, record sets, semantics) with the Croissant-RAI extension fields: collection process, labelling (silver/gold provenance), limitations, intended use, PII/secret handling. It is generated by a Rust binary from the index, never hand-edited.
- Human-readable: a Datasheet-style card (motivation, composition, collection, preprocessing, uses, distribution, maintenance) under `datacard/`. It is regenerated per snapshot.
- Publication to HF (EXT-001) is gold-and-local-only by default [A]. Rows with `provider ∈ {anthropic, google}` are excluded from any public release unless the operator overrides this at a §8 STOP.

---

## §3 Research questions and pre-registered measurements

All measurements run from make targets delivered by the tickets in §7. The first run sets the [V] baseline, and later gates ratchet from it.

| ID | Question | Metric (definition) | Command | Baseline mark now |
|---|---|---|---|---|
| M1 | Yield | admitted rows/week by `label_tier` × lane | `make trace-yield WINDOW=7d` | [U] |
| M2 | Capture completeness | rows with all ✔ fields ÷ lane calls dispatched | `make trace-completeness` | target 100% [A] |
| M3 | Disagreement rate | fraction of rounds with `lane_disagreement ≠ 0` among counted lanes | `make trace-disagreement` | [U] |
| M4 | Error overlap | Cohen's κ on binary error indicators vs gold, per lane pair (incl. Qwen); CAPA as secondary | `make lane-kappa SPLIT=val` | [U] |
| M5 | Near-dup rate | fraction of rows in a cluster of size > 1; cross-split clusters | `make trace-dedup-report` | cross-split = 0 [A]; rate [U] |
| M6 | Bytes/round | raw and compressed bytes per lane-round, split by blob kind (input, output, logits) | `make trace-bytes WINDOW=7d` | [U] (G10 ≤ ~0.45 TB/yr) |
| M7 | Secret hit rate | hits ÷ rows scanned, by stream (S1 diff, S1 output, S2 transcript) | `make trace-secret-report` | [U] |
| M8 | Compression ratio | raw ÷ zstd bytes by blob kind and level | `make trace-zstd-bench` | [U]; literature suggests 6–7× for JSONL at level 3 [X] |
| M9 | Outcome label yield | gold rows matured/week; revert and escape counts | `make trace-outcome-join` | [U] |
| M10 | Shadow availability | Qwen `lane_state` distribution by `served_by` | `make trace-served-by-audit` | [U] |
| M11 | Token lengths | p50/p95 input/output/reasoning tokens by lane | `make trace-tokens` | [U] |
| M12 | Silver noise | fraction of silver findings contradicted by gold outcome (on matured rows) | `make silver-audit` | [U] |

Pre-registered hypotheses (they extend REX-001 H1–H6 and do not replace them):
- **PRA-H1:** Silver-trained Qwen increases κ_err(Qwen, counted lane) by more than the pre-registered margin [A] relative to base Qwen.
- **PRA-H2:** Gold-only fine-tuning does not increase κ_err beyond the margin [A].
- **PRA-H3:** Disagreement-oversampled training improves sealed-test defect recall over uniform sampling at an equal token budget.
- **PRA-H4:** Grouped time-ordered splits lower val score relative to random row splits. If they do, the size of the drop is the leakage measure, and it is reported.

---

## §4 Admission gates to the training pool

Admission is a separate binary (`trace-admit`). It is never the producer. A row enters the pool only if **all** gates pass. Failure is recorded; there is no silent drop.

| Gate | Rule | Target | Falsifier (planted RED → GREEN) |
|---|---|---|---|
| **G-SEC** secret scan | Union of two independent scanners over input blobs, output blobs and transcripts: (a) an in-process Rust entropy+pattern scanner (ripsecrets-class); (b) the gitleaks public ruleset ported **as data** (TOML rules → Rust `regex` set), with no Go/Python runtime. Any hit sends the row to `quarantined`, blobs to `quarantine/`. **No in-place redaction** [A]. | quarantined-row leakage into pool = 0 [A] | Plant the known AWS-key-literal diff plus 20 [A] synthetic canaries (AWS, GitHub PAT, private key, JWT, high-entropy generic) in a fixture PR. RED: the pool contains the canary. GREEN: all canaries quarantined, pool count 0. |
| **G-SEC-FN** scanner recall | Measured recall on the canary set plus a held-out labelled fixture | 100% on canaries [A]; real-world recall [U] | Remove one rule; the canary for it must go RED. |
| **G-CON** contamination | 0 sealed-test hashes (exact `diff_sha256`) and 0 sealed near-dup clusters in any pool (`review-corpus-contamination-v1` extended to MinHash clusters) | 0 [A] | Insert one sealed item with whitespace/rename perturbation into a candidate pool. RED on exact-hash-only; GREEN once cluster check lands. |
| **G-ID** identity completeness | `provider`, `model_id` (exact), `access_channel`, `terms_ref`, `input_sha`, `output_sha`, `served_by`, `label_tier`, `split_guard` all non-null | 100% [A] | Row with `model_id = "claude-sonnet"` (alias) must be refused. |
| **G-DUP** dedup and split | Row's cluster is not in a different split; exact duplicates of admitted rows are linked, not re-admitted | cross-split = 0 [A] | Backport PR with identical diff under new PR number assigned to val while original in train: RED → GREEN after union-find. |
| **G-MAT** outcome maturity | `label_tier = gold` requires `outcome_matured_at` ≠ null | 0 immature gold rows [A] | Gold row 3 days old must be refused. |
| **G-HASH** blob integrity | Re-hash of every referenced blob equals its address | mismatch = 0 [A] | Flip one byte in a test blob. |

---

## §5 How the data feeds the improvement loop

| Phase | When | What | Data used |
|---|---|---|---|
| **B1: prompt / few-shot** | now (apr 0.69+) | Mine disagreement and escape rows into retrieval few-shot exemplars for the Qwen prompt, then version the prompt (`prompt_version`). After 0.70, use `--json-schema` constrained decoding to drive `parse_status = failed` toward 0. | gold first; silver exemplars only when tagged in the prompt manifest |
| **B2a: sequence-level distill / SFT** | apr 0.71 | `apr finetune` on (input, output) pairs. Silver and gold are pooled per run manifest, and every manifest lists its tiers and providers. | admitted pool |
| **B2b: logit distill from 27B teacher** | apr 0.71 | The 27B teacher re-scores diffs, writing `sparse-logits-v1`. The student is trained with a forward-KL objective that uses the tail-mass correction, with reverse-KL / on-policy (GKD-style) as challengers. On-policy needs the student's own samples, which the Qwen shadow rows already are. | local rows only (teacher and student both local) |
| **Promotion** | per PRM-001 §5.4 | Champion/challenger on the sealed test. Rotation after 20 [A] promotion decisions. | sealed (evaluation only) |

**Preserving dissent independence (`lane-independence-v1`).**
1. **Measure before training.** κ_err for every (Qwen, counted-lane) pair on matured gold val rows goes into the scoreboard as the [V] baseline.
2. **Two heads, two pools.** The *dissent* checkpoint that fills the shadow lane is trained on **gold only**. A *general* checkpoint may use silver. Only the dissent checkpoint may occupy the Qwen shadow lane.
3. **Gate.** A candidate for the shadow lane is refused if κ_err(candidate, any counted lane) > baseline + δ, with δ [A] pre-registered in the run manifest before training. Refusal is an andon, not a warning.
4. **Diversity option (research, not default).** Down-weight silver examples where all three counted lanes agree and the outcome is still pending. These are exactly the examples that teach "agree with the voters", with no evidence behind them.

---

## §6 Hard rules (each falsified in §9)

| # | Rule | Falsifier id |
|---|---|---|
| R1 | Every lane call produces exactly one `agent-trace-v1` row, NotRun included | F-R1 |
| R2 | `output_sha` covers the full raw reply; truncation anywhere is a defect | F-R2 |
| R3 | Findings are stored as text; a count-only findings field fails schema validation | F-R3 |
| R4 | The producer never sets `label_tier` other than `pending` | F-R4 |
| R5 | No row with a secret hit ever enters a pool; no in-place redaction | F-R5 |
| R6 | 0 sealed hashes or sealed clusters in any pool | F-R6 |
| R7 | Splits are by `repo#pr` group, time-ordered, cluster-closed | F-R7 |
| R8 | Every row has exact `model_id`, `provider`, `access_channel`, `terms_ref` | F-R8 |
| R9 | Blobs are write-once and re-hash equal to their address | F-R9 |
| R10 | Nightly second copy exists and the weekly restore drill mismatch is 0 | F-R10 |
| R11 | Transcript retention is declared fleet-wide by forjar; no host runs the default 30 | F-R11 |
| R12 | Qwen shadow lane weight stays 0 regardless of trace content | F-R12 |
| R13 | A shadow-lane candidate failing the κ_err gate is not deployed | F-R13 |
| R14 | No Python in any PRA-001 artifact; shell is bashrs-clean | F-R14 |
| R15 | `apr` is a released, sha-verified tag; weights are pinned by sha256; `served_by` reflects the actual backend | F-R15 |

---

## §7 EV-ordered tickets

Each ticket: `pmat work add`, feature branch, PR, protected main, `ci / gate`, TDD, one `pv` contract with its falsifier planted RED then GREEN. **K̂** = estimated minutes [A]; **K** = measured minutes (recorded at close); **andon** if K > 2·K̂ [A] or any falsifier fails to go RED first.

| EV rank | Ticket | Contract | Done when | K̂ [A] |
|---|---|---|---|---|
| 1 | T1 Capture at dispatch in quorum rail (paiml-implement #436) | `agent-trace-v1` | F-R1, F-R2, F-R3 GREEN; 7 days of 100% completeness (M2) | 240 |
| 2 | T2 Ledger v2: findings text + trace links; v1 readers keep working | `review-ledger-v2` | v1 fixtures parse; v2 row carries findings text; migration test | 120 |
| 3 | T3 Content-addressed blob store on almacen | `trace-blob-store-v1` | F-R9 GREEN; O_EXCL write-once test; forjar-declared mount | 120 |
| 4 | T4 Transcript retention + harvester | `transcript-retention-v1` | F-R11 GREEN on every host in inventory; subagents/ harvested | 90 |
| 5 | T5 Nightly second copy + weekly restore drill | `trace-backup-v1` | F-R10 GREEN; first drill mismatch = 0 | 120 |
| 6 | T6 Secret admission gate (Rust scanner + ported ruleset) + backfill | `trace-admission-secret-v1` | F-R5 GREEN; AWS-key literal quarantined in backfill | 180 |
| 7 | T0 Baseline freeze: run §1 verify commands, convert [A]/[U]→[V] | (report only) | §1 table all [V] or STOP raised | 60 |
| 8 | T7 Contamination gate extended to clusters | `review-corpus-contamination-v1` (extend) | F-R6 GREEN incl. perturbed sealed item | 90 |
| 9 | T8 MinHash-LSH dedup index (pure Rust) | `trace-dedup-v1` | M5 report produced; cluster ids in index | 150 |
| 10 | T9 Group/time split assigner with union-find closure | `trace-split-guard-v1` | F-R7 GREEN; cross-split clusters = 0 | 120 |
| 11 | T10 Outcome join (merged/revert ≤14d/regression escape) | `trace-outcome-join-v1` | G-MAT falsifier GREEN; M9 produced | 150 |
| 12 | T11 Qwen top-k logits capture in `apr serve` rows | `sparse-logits-v1` | header + residual mass round-trip test; M6 logits bytes measured | 180 |
| 13 | T12 Terms/provider tagging + exclusion filter | `agent-trace-v1` (terms fields) | P-TERMS GREEN (pool rebuild excluding a provider) | 60 |
| 14 | T13 Lane κ_err probe + scoreboard | `lane-independence-v1` | M4 baseline [V] recorded for every pair | 120 |
| 15 | T14 Datacard: Croissant + RAI + datasheet generator | `trace-datacard-v1` | generated file passes Croissant validation; regenerated per snapshot | 90 |
| 16 | T15 B1 few-shot miner from disagreement/escape rows | (uses `agent-trace-v1`) | prompt_version bump with manifest of exemplar tiers | 120 |
| 17 | T16 27B teacher re-score job (gx10) | `sparse-logits-v1` (mode topk) | teacher blobs for all admitted local rows of week N | 180 |

EV rationale: T1–T5 stop irreversible loss, which is ongoing and compounds daily. T6–T10 make the data *admissible*. T11–T17 make it *useful*. Nothing training-related (B2) is ticketed until aprender 0.71 ships; that is a §8 STOP.

---

## §8 STOP conditions (exhaustive)

The runner halts, writes the §9 report with `status: STOPPED`, and waits for the operator when:

1. **S-BASE:** any §1 verify command disagrees with its [A] value (baseline drift).
2. **S-CRED:** a step needs credentials, API keys, or account settings the runner does not hold.
3. **S-MEDIA:** the second-copy media or an offsite target does not exist in forjar inventory.
4. **S-CAP:** almacen free space < 20% [A], or the M6 projection exceeds 50% of usable capacity within 5 years [A].
5. **S-SECRET-PURGE:** a secret hit requires purging from *all* copies (rotation is the operator's job; the runner only quarantines).
6. **S-SEALED:** any change to `review-corpus-v1`, sealed rotation outside PRM-001, or any sealed hash found in a pool.
7. **S-TERMS:** any change in provider terms detected by `terms_ref` refresh, any publication including non-local rows, or any request to change S-6.
8. **S-APR:** a required verb is missing from the released `apr` tag (for example B2 before 0.71), or a tag's sha fails verification. HEAD is never a fallback.
9. **S-PY:** a dependency would introduce Python, or a non-forjar install would be needed.
10. **S-KAPPA:** the κ_err gate refuses a shadow-lane candidate (the operator decides whether δ was right; the runner never widens δ).
11. **S-WEIGHT:** any change would give the Qwen lane non-zero weight, tie-break or escalation power.
12. **S-ANDON:** K > 2·K̂ [A] on any ticket, or a falsifier that cannot be made RED first.
13. **S-PII:** a transcript contains third-party personal data beyond author identities (scrubbing policy is an operator decision).
14. **S-LOSS:** any blob mismatch in the restore drill, or `capture_failed` rows > 0 for 24 h [A].

---

## §9 Final report schema and scoreboard

```yaml
report: PRA-001
version: 1
generated_at: <RFC3339>
status: COMPLETE | STOPPED | IN_PROGRESS
stop: {code: <S-*>|null, evidence: <path>}
apr_tag: <vX.Y.Z>  # sha-verified
baseline:           # §1 after T0
  - {id: G1, value: <n>, mark: V, command: "make trace-yield WINDOW=7d"}
tickets:
  - {id: T1, contract: agent-trace-v1, pr: <url>, khat_min: 240, k_min: <n>, andon: false,
     falsifier: {id: F-R1, red_commit: <sha>, green_commit: <sha>}}
measurements:       # §3
  - {id: M4, pair: [qwen-shadow, sonnet], kappa_err: <x>, n: <n>, split: val, mark: V}
gates:              # §4
  - {id: G-SEC, pool_leaks: 0, canaries_quarantined: <n>/<n>}
falsifiers:         # §6
  - {id: F-R5, state: GREEN, evidence: <path>}
storage:
  bytes_raw_per_week: <n>
  bytes_zstd_per_week: <n>
  projected_tb_per_year: <x>   # replaces G10 [U]
  restore_drill_mismatch: 0
```

**Scoreboard (targets and probes only)**

| Probe | Target | Mark |
|---|---|---|
| Capture completeness (M2) | 100% | [A] |
| Rows lost to `capture_failed` | 0 | [A] |
| Secret-hit rows in any pool | 0 | [A] |
| Sealed hashes/clusters in any pool | 0 | [A] |
| Cross-split near-dup clusters | 0 | [A] |
| Rows missing identity fields | 0 | [A] |
| Immature gold rows | 0 | [A] |
| Restore-drill mismatches | 0 | [A] |
| Hosts with default transcript retention | 0 | [A] |
| Qwen lane weight | 0 | [A] |
| κ_err(shadow candidate, counted lane) | ≤ baseline + δ | [V] baseline, [A] δ |
| Bytes/year | ≤ measured M6 projection; ratchet | [U]→[V] |
| Yield/week by tier | ratchet up from week-1 baseline | [U]→[V] |
| P-TERMS: pool rebuild excluding one provider | succeeds, 0 rows of that provider | [A] |

Falsifier catalogue (F-R1…F-R15): each is a `pv` contract test that is committed RED (failing on a planted defect) before the implementing commit turns it GREEN. Examples: F-R1 kills a lane mid-call and asserts a NotRun row. F-R2 sends a 1 MB reply and asserts the byte-exact `output_sha`. F-R12 injects a Qwen verdict that disagrees with all counted lanes and asserts the quorum outcome is unchanged. F-R14 runs a repo-wide scan for `*.py`, `python`, `pip` in PRA-001 paths and expects 0 hits.

---

## Appendix A: Research findings and how each changed the design

**A.1 Distillation (sequence vs logit, sparse top-k).**
- Anshumann et al., *Sparse Logit Sampling* (ACL 2025 oral, arXiv 2503.16870), prove that caching top-K probabilities gives "biased estimates of teacher probability distribution to the student, resulting in suboptimal performance and calibration". Their importance-sampling RS-KD gives unbiased estimates while using only 0.01% of precomputed teacher logits. A secondary summary reports parity with full KD using about 12 tokens/step and ≈3.6 TB per 100B tokens versus ≈90 TB for top-300 [X, secondary]. 
- A 2026 preprint (arXiv 2608.03796) frames sparse top-K KD as forward KL against a teacher with retained mass M ≤ 1. 
- MiniLLM (Gu et al., ICLR 2024) argues reverse KL suits generative students. GKD (Agarwal et al., ICLR 2024) trains on student-generated sequences with teacher feedback. 
- **Design changes:** store `topk_mass` + `logsumexp_full`, reserve `mode: rs-sample`, and keep Qwen's own outputs as on-policy samples to be re-scored by the local 27B teacher. Frontier lanes are sequence-level only, because no logits are available.

**A.2 Code-review datasets and label quality.**
- CodeReviewer (Li et al., ESEC/FSE 2022) built a multilingual corpus (nine languages) of real diffs and review comments with three tasks: quality estimation, comment generation, refinement. Its schema is diff-in / label-or-comment-out, with human reviewer comments as labels. 
- *Too Noisy To Learn* (Liu, Lin & Thongtanunam, arXiv 2502.02757) shows substantial vague or non-actionable noise remains after heuristic and SVM cleaning. Only 64% of the original CodeReviewer comments were valid, and LLM cleaning raised that "to up to 85%", with 66–85% precision in detecting valid comments. 
- **Design change:** findings text alone is not a label. Gold comes from outcomes and the sealed corpus, silver is tagged, and M12 measures silver noise against matured outcomes.

**A.3 Outcome signals as labels.**
- SZZ-derived bug-inducing labels are notoriously noisy. Herbold et al. (arXiv 1911.08938) report that "about one quarter of the links to defects detected by SZZ is wrong", and that for every correctly labelled bug issue there are 0.74 mislabelled ones. Afric et al. report 14.3% mislabelled even after refinement (via ReDef, arXiv 2509.09192). 
- **Design change:** gold outcomes are restricted to direct, observable events (merge, revert ≤14 days [A], explicit `regression` label). No blame-tracing heuristics in v1. Outcomes mature before use (G-MAT), and pending is never negative.

**A.4 Near-duplicates and leakage.**
- Allamanis (Onward! 2019) showed code duplication inflates reported metrics by up to 100%. 
- Lee et al. (ACL 2022) found near-duplicates via MinHash (e.g. 3.04% of C4) and train-test overlap affecting over 4% of validation sets. 
- **Design change:** group-by-PR and time-ordered splits, MinHash-LSH cluster closure across PRs, contamination extended from exact hash to cluster (G-CON), and PRA-H4 to quantify leakage.

**A.5 Ensemble diversity and error correlation.**
- Kim et al. (*Correlated Errors in LLMs*, ICML 2025) evaluated 350+ LLMs. On HELM, model pairs agree about 60% of the time when both are wrong, against 1/3 for uniformly random wrong answers. Shared provider and architecture raise correlation, and larger, more accurate models are highly correlated even across providers. 
- Goel et al. (*Great Models Think Alike*, ICML 2025) introduce CAPA and show judges favour similar models. They find weak-to-strong gains are larger when supervisor and student are functionally different. 
- **Design change:** κ_err is measured before and after training (M4, CAPA as secondary). The dissent checkpoint is gold-only. A pre-registered δ gates the shadow lane (§5).

**A.6 Secret detection.**
- Basak et al. (arXiv 2307.00714) measured tools on SecretBench (15,084 true secrets among 97,479 reported). Gitleaks had the best recall (88%) at 46% precision and the top F1 (60%), and TruffleHog reached 52% recall. GitHub's scanner had the best precision (75%) but only 6% recall. "No current tool has the coveted high precision and high recall." 
- On a different benchmark (AssetHarvester, arXiv 2403.19072), Gitleaks recall fell to 0.02 and TruffleHog reached 0.40. Recall is dataset-dependent. 
- ripsecrets is a Rust, local-only scanner that claims to be ≥95× faster [X, vendor claim]. 
- **Design change:** union of two independent rule sources, implemented in Rust (the gitleaks ruleset ported as data). Quarantine, not redaction. Recall is measured locally with planted canaries rather than trusting published numbers.

**A.7 Provider terms (factual summary; not legal advice; S-6 governs).**
- **Anthropic Commercial Terms of Service** (effective June 17, 2025), §D.4: "Customer may not and must not attempt to (a) access the Services to build a competing product or service, including to train competing AI models or resell the Services except as expressly approved by Anthropic…". §B also states that the customer owns its Outputs and that Anthropic may not train on Customer Content. 
- **Gemini API Additional Terms** (effective March 23, 2026): "You may not use the Services to develop models that compete with the Services (e.g., Gemini API or Google AI Studio)." 
- **Google Antigravity Additional Terms** (undated page) incorporate the Google Terms of Service (effective July 30, 2026). Those forbid "using AI-generated content from our services to develop machine learning models or related AI technology." Antigravity §8 binds users who select an Anthropic model to Anthropic's commercial terms. 
- Google Cloud service terms exempt Gemini Enterprise Agent Platform (formerly Vertex AI) when no Google pre-trained model is used. 
- Consumer terms for subscription-based Claude Code were not verified here. 
- **Design change:** `access_channel` and `terms_ref` are mandatory. Pools can be rebuilt excluding any provider or channel (P-TERMS). Public releases default to local/gold rows. Terms changes are S-TERMS.

**A.8 Claude Code transcripts.**
- Official docs: transcripts under `~/.claude/projects/` are plaintext, include everything that passed through tools, and are deleted after `cleanupPeriodDays`. The default is 30 and the minimum is 1; 0 fails validation. Managed settings govern. 
- A February 2026 issue (#23710) reported that 0 had silently disabled persistence in an earlier version. Another issue (#58154) reports that `subagents/` transcripts are not swept. 
- **Design change:** forjar-managed settings with a large explicit value, harvesting that includes `subagents/`, a survival canary, and a secret scan on every transcript.

**A.9 Dataset documentation.**
- Croissant (Akhtar et al., NeurIPS 2024 D&B) defines four layers (metadata, resources, structure, semantics). Croissant-RAI builds on Datasheets for Datasets and Data Cards. 
- NeurIPS 2026's Evaluations & Datasets track now requires RAI metadata in the Croissant file. Hugging Face auto-generates core Croissant. 
- **Design change:** `trace-datacard-v1` generates Croissant + RAI + a datasheet from the index on each snapshot.

**A.10 Storage sizing.**
- A consumer drive URE spec of 1 per 10¹⁴ bits (≈12.5 TB read) is a worst-case warranty floor. Field studies show lower rates [X]. 
- zstd at level 3 is reported at 6–7× on JSONL [X, blog], and dictionaries help small similar files [X]. 
- **Design change:** compression level is recorded per blob and not implied, M8 measures our own ratio, the second copy plus restore drill is mandatory, and G10 stays [U] until week 1.

## Caveats
- Some literature figures come from secondary summaries (the RS-KD storage numbers, and the Afric et al. 14.3% figure, which comes via ReDef) and are marked [X, secondary] where applicable. None is used as a gate threshold.
- Secret-scanner recall differs by an order of magnitude between benchmarks. Local canary recall is necessary, not sufficient, and a zero-hit scan is not proof of absence.
- Provider terms change: Anthropic updates take effect 30 days after posting. `terms_ref` must be refreshed on a schedule, and nothing here is legal advice.
- κ_err needs matured gold rows. Until M9 produces enough of them, the independence gate cannot be evaluated, and B2 shadow-lane deployment must wait (S-KAPPA).