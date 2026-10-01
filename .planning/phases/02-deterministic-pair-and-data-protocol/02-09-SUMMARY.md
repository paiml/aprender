---
phase: 02-deterministic-pair-and-data-protocol
plan: 09
subsystem: apr-cli
tags: [cli, adapter, attested-ingest, atomic-write, strict-replay, contracted-seed, budget-binding, docs, d05, d09, d14, data-03, data-04, data-05]
requires:
  - "02-05 (SelectionConfig/FewShotSelector::select, SelectionManifest::{from_selection,to_file_bytes,from_bytes}, Selection::replay)"
  - "02-06 (PreparedDataset::<Canonical>::from_attested_bytes, attestation_bytes_from_manifest, schema_version 2)"
  - "02-07 (PairConfig/PairSampler/PairReplayRecord/pair_manifest_hash/dump_pairs/parse_pair_dump/validate_pair_records)"
provides:
  - "apr data select — attested ingest, contracted-seed pre-flight, atomically written crate-owned selection manifest"
  - "apr data pairs — strict replay, one-place budget resolution, tuple-committing hash, streaming --dump"
  - "crates/apr-cli/src/commands/data_contrastive.rs — the whole user surface, 36 tests"
  - "atomic_write_with — temp-in-destination + fsync + rename, no-clobber, temp cleanup, with a test-only pre-rename failure seam"
  - "data_tweeteval::fixtures — the shared synthetic canonical source-tree writer, now text-taggable"
  - "docs/examples/tweet-eval-stance.md — the prepare -> select -> pairs workflow, every command executed"
affects:
  - "Phase 3 (the trainer consumes selection-manifest.json and the pair replay tuple; both are now user-producible)"
  - "Phase 5 (the selection lock reads payload.access_ledger out of the manifest this command writes)"
tech-stack:
  added: []
  patterns:
    - "read the SPLIT ROLE SET out of the attestation rather than hardcoding a canonical triple, so a wrong-profile directory is refused by PROFILE instead of by a missing filename"
    - "pre-flight the REQUEST before touching the filesystem, so a bad flag is never reported as a problem with the data"
    - "derive a reported mode from a recorded value instead of storing it beside it, so the two cannot disagree"
    - "one writer for every artifact, taking a fill closure rather than a byte slice, so a streaming dump inherits the write-safety proofs without buffering"
    - "reseal a forged manifest with the crate's own digest primitive, because a parser that verifies before returning makes un-resealed tampering test the parser instead of the ladder"
    - "map a crate error to the CLI by appending the FLAG to change — the crate cannot know flag names, and a constraint without a knob is diagnosable but not actionable"
key-files:
  created:
    - crates/apr-cli/src/commands/data_contrastive.rs
  modified:
    - crates/apr-cli/src/data_commands.rs
    - crates/apr-cli/src/dispatch_analysis.rs
    - crates/apr-cli/src/commands/mod.rs
    - crates/apr-cli/src/commands/data_tweeteval.rs
    - docs/examples/tweet-eval-stance.md
decisions:
  - "The seed MODE is derived from the recorded root_seed, not stored as a manifest field — SelectionPayload is crate-owned and deny_unknown_fields, and a parallel field could disagree with the seed printed beside it"
  - "cli.offline is deliberately NOT threaded into either command: neither opens a socket, so an offline switch would advertise a capability that does not exist"
  - "atomic_write is generalized to atomic_write_with(target, force, fill) so --dump streams instead of buffering; the byte form is a thin wrapper, so one rename site and one set of write-safety proofs still cover both artifacts"
  - "The plan's Task 3 rejection list is unreachable from the CLI without resealing the envelope digest; the tests reseal with the crate's public hash::exact_hash rather than dropping the rungs or naming Sha256 in this module"
  - "Task 3 had no executable RED — the tests could not compile until the interface existed — so its gates are falsified by three induced mutations instead, which is stated rather than papered over"
metrics:
  duration: ~2h45m (including a 20-minute ENOSPC stop-and-report)
  tasks: 3
  files: 6
  completed: 2026-08-09
requirements: [DATA-03, DATA-04, DATA-05]
---

# Phase 2 Plan 09: The User Surface Summary

Every Phase 2 ROADMAP criterion is phrased "A user can…". This plan is the one that makes
them literally true, so the evidence below is **commands a user runs against the live
pinned TweetEval revision**, not unit tests calling internal functions. `apr data select`
and `apr data pairs` are thin filesystem adapters: 0 occurrences of `Sha256`, `swap(`,
`unrank` or `json!` in the module's non-comment lines, exactly one `fs::rename` site, zero
`File::create`, zero `unwrap()`.

**36 new tests** (19 select + 17 pairs). apr-cli lib **6639 → 6675**; cross-crate baseline
**14,254 → 14,290**. Both deltas are exactly 36 — no pre-existing test changed state.

## Commits

| # | Task | Commit | Result |
|---|------|--------|--------|
| 1 | clap variants + dispatch arms + module surface | `f5e8ea056` | `cargo check` green on a task that dispatches to a module it creates |
| 2 RED | failing tests for `run_select` | `04f4b7f98` | **0 passed / 19 failed** |
| 2 GREEN | attested ingest, seed policy, the shared atomic writer | `8540eba88` | 19/19; apr-cli lib 6658 |
| 3 | `run_pairs`, strict replay, streaming dump | `05d484aaf` | 36/36 |
| 3 docs | the prepare → select → pairs workflow | `2ba6781f6` | every command executed |
| — | plan completion | this commit | SUMMARY, STATE, ROADMAP |

No branch, no push, no PR — the policy 02-01 set. `git diff --diff-filter=D 37cc105ad..HEAD`
is **empty**; the three `crates/aprender-train/src/prune/snapshots/*.snap.new` were deleted
by the `aprender-train` run and restored with `git checkout --` before this commit.

---

## The ROADMAP criteria, discharged by real commands

Binary pinned per CLAUDE.md Step 0 — `. scripts/apr_bin.sh` resolved
`target/debug/apr` reporting **`apr 0.63.0 (05d484aaf)`** against `HEAD = 05d484aaf`. The
stale binary that existed before the rebuild reported `(37cc105ad)`, the commit *before*
this plan, and contained neither subcommand: exactly the trap Step 0 exists for.

### 1. Acquire the pinned source and produce canonical 587/66/280

`"$APR" data tweet-eval-stance --output <DIR>` — a real network fetch, `rc=0`:

```
  Revision: 4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66 (downloaded)
  Test: test.jsonl (280 samples)
  Train: train.jsonl (587 samples)
  Validation: validation.jsonl (66 samples)
  Cross-split duplicates: 1 group(s), 1 training row(s) excluded from the selection pool
```

`wc -l` on the written files: 587 / 66 / 280. The excluded row is `train:70`, matching
02-06's live golden, and reduced pools are 158 / 319 / 109.

### 2. Select 8/16/32/64 per class for every contracted seed, replayably

All ten seeds at 8 shots, each run twice:

```
  seed 13  1d2dbbb37edea7ab…  (replays)      seed 37  b1bcee2f238239fd…  (replays)
  seed 17  09d382a717c818e4…  (replays)      seed 41  1e213047520cdab0…  (replays)
  seed 23  41eeb5ad63074a4d…  (replays)      seed 43  74db8edefa475a88…  (replays)
  seed 29  4eaaa13e70273d9f…  (replays)      seed 47  2a8798d4038cebf2…  (replays)
  seed 31  a4115b5c1f03590f…  (replays)      seed 53  49688740ff683ea5…  (replays)
  distinct hashes over the ten seeds: 10 of 10
```

Ten seeds, ten *distinct* selections, each reproducing its own hash — "deterministic"
cannot be satisfied here by a constant. 64 shots was exercised too (pools 158/319/109 all
exceed 64).

### 3. Replay pairs whose endpoints are distinct selected training IDs

`"$APR" data pairs --selection … --data …`:

```
  Pair manifest hash: 5f4d4bb6cb9a3e49e4ad30869e1eb0b27679603658e9cd54a98c8bb29e74b524
  Budget: 384 pair(s) per epoch, hard cap 1048576
  Emitted kinds: both
  Counts: 192 positive, 192 negative
  Singleton policy: negatives_only (v1), 0 affected class(es)
```

Two runs, byte-identical hash. **The 256-pair dump was audited against the splits rather
than trusted**, and every clause of the criterion holds: every endpoint is a selected
*training* id; no endpoint is a validation or test id; no self-pair; 162 distinct unordered
identities over 256 draws (repeats are oversampling, deviation clauses 1–2); every `target`
agrees with class identity; `train:70` never appears.

### 4. A per-epoch budget with `O(examples + budget)` behaviour

| shots | examples | `--budget 256` → emitted | dump lines | default budget |
|---|---|---|---|---|
| 8 | 24 | 256 (128/128) | 256 | 384 |
| 64 | 192 | 256 (128/128) | 256 | 24576 |

An 8× growth in examples under a fixed budget changes neither the emitted count nor the
storage. The control beside it — the *default* budget does grow, 384 → 24576 — is what
stops "the budget held" from being satisfied by a command that ignores the layout. The
retained-state half of the invariant is proven in-crate by 02-07/02-08; this is its
user-visible face, plus a `--dump` that streams rather than buffers (see Deviation 3).

### 5. Fail-closed isolation

Four directories that look fine and are not, each refused with `exit=5`:

```
# SetFit compatibility layout  (refused on BOTH commands, not just select)
contrastive data: dataset profile mismatch: expected "canonical", got "compatibility"
# two preparations mixed, validation.jsonl swapped
contrastive data: validation split hash mismatch: expected 1ac2119f…, got c7f5921a…
# edited attestation
contrastive data: dataset fingerprint mismatch: expected 00000000…, got 082ad5b0…
# stale directory
benchmark-manifest.json declares schema_version 1, but this build supports [2]. There is no
migration … Re-prepare the benchmark with `apr data tweet-eval-stance --output <DIR> --force`.
```

---

## Error paths are part of the user surface

Eight failure modes run end to end. **Every one exits 5 (`ValidationFailed`), none panics,
and none dumps a raw `Debug`:**

| Input | Message |
|---|---|
| `--shots 9` | `--shots 9 is not a contracted few-shot size; expected one of {8, 16, 32, 64}` |
| `--seed 42` | names all ten contracted seeds, `--any-seed`, and why the seed is recorded |
| missing dir | `…/benchmark-manifest.json not found. Prepare one with \`apr data tweet-eval-stance --output <DIR>\`…` |
| existing manifest | `Refusing to replace existing file … (pass --force to replace it)` |
| `--budget 500 --hard-cap 100` | `requested pair budget 500 exceeds hard_cap 100 — raise --hard-cap or lower --budget…` |
| `--budget 0` | `pair budget must be greater than zero — pass --budget <N> with N >= 1…` |
| `--hard-cap 0` | `pair hard_cap must be greater than zero — pass --hard-cap <N> with N >= 1…` |
| missing selection | `…not found. Write one with \`apr data select --data <DIR> --shots <N> --seed <SEED>\`.` |

The zero-budget and zero-cap messages differ, because one is a defect in the request and
the other in the configuration. The `--shots` and `--seed` checks run **before** the
filesystem is touched, asserted by pointing them at a directory that does not exist: if the
order were wrong the test would see a missing-manifest error instead.

---

## `--seed` is provably required, not vacuously

The plan's original check ("help output does not contain `default_value`") could never
fail — clap renders `[default: X]`, never that literal. The replacement can fail, and was
measured:

```
Usage: apr data select [OPTIONS] --data <DIR> --shots <N> --seed <SEED>
```

`--seed <SEED>` appears **unbracketed** in the required-args usage line (`grep -cE
"^Usage:.*\[--seed"` = 0), and `[default:` appears **nowhere** in the select help at all
(count 0, not merely absent from the `--seed` line). `grep -c '"42"'` over
`data_commands.rs` non-comment lines is 0.

---

## The write-safety seam, and why the proofs are not taken on trust

`crates/apr-cli/src/commands/data_contrastive.rs` carries a `#[cfg(test)] thread_local`
fault-injection seam, `FAIL_BEFORE_RENAME`, read by `fill_and_sync` **after** `sync_all`
and **before** `rename` — the one window in which a partial artifact could exist. It is a
`thread_local` because cargo runs each test on its own thread, so two tests cannot see each
other's injection. Three of this plan's write-safety tests depend on it, and a future reader
should know it is the mechanism rather than assume the tests are self-evidently sound.

**Five mutations were applied, RUN, observed and reverted.** Each predicted a specific RED
set and produced exactly it:

| Mutation | Predicted | Observed |
|---|---|---|
| delete the temp-file cleanup | the two induced-failure tests RED | `rc=101`, 17 passed / 2 failed; the message named the leftover `.selection-manifest.json.tmp.85207.1`, so the failure is diagnosable rather than merely red |
| move the injection window AFTER the `rename` | the same two RED | `rc=101`, 17 / 2 — the window is where it is claimed to be |
| placeholder `pairs_outcome` body | every pairs test RED | `rc=101`, **19 passed / 17 failed** — the select half untouched |
| silently clamp an over-cap budget | the over-cap test RED | `rc=101`, 35 / 1, exactly `a_budget_above_the_hard_cap_…` |
| `--dump` written directly instead of through the shared helper | the two dump write-safety tests RED | `rc=101`, 34 / 2 — the dump genuinely inherits the helper |

Reverted each time; 36/36 green afterwards.

---

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 — Bug] The plan's Task 3 rejection list is unreachable from the CLI**

- **Found during:** Task 3, writing the replay rejections.
- **Issue:** the plan requires named tests for `EndpointNotInSelection`, `RowHashMismatch`
  and `SelectionReplayMismatch` raised from a *manifest*. `SelectionManifest::from_bytes`
  verifies the envelope digest **before returning**, so any hand-edited manifest is
  `SemanticHashMismatch` at parse and never reaches `Selection::replay` at all. Plan 02-05
  hit the same wall inside the crate and solved it by constructing `SelectionManifest`
  *values*; the CLI cannot, because it only has bytes.
- **Fix:** the tests **reseal** the envelope after mutating the payload, using the crate's
  own public `hash::exact_hash` (`semantic_hash` is SHA-256 over
  `payload.to_canonical_bytes()`). All three rungs are now CLI-reachable, including the
  twelfth-rung case that only recomputation catches: one selected row replaced by another
  row from the same class's pool carrying that row's *real* hashes, counts and ordering
  intact, envelope resealed. A control asserts the untouched manifest replays first, so
  "the forgery is refused" is not satisfied by a broken fixture. The reseal helper itself
  asserts the forgery parses cleanly, so each test targets the rung it names.
- **Why not just name `Sha256`:** the acceptance criterion requires 0 occurrences in this
  file, and its intent — the CLI implements no hashing — is honoured exactly by calling the
  crate's primitive rather than by evading the grep.
- **Commit:** `05d484aaf`

**2. [Rule 1 — Bug] A documented command whose output is 64/65 vacuous `ok` lines**

- **Found during:** Task 3, checking that every command in the docs actually runs.
- **Issue:** the file's last section told a reader to run
  `cargo test -p apr-cli pinned_upstream_satisfies_the_dataset_contract -- --ignored`.
  Measured: it prints **65** `test result: ok` lines, 64 of which matched nothing. A reader
  grepping for `ok` is satisfied by a suite that ran no test. Same defect class 02-06 fixed
  in the tweet-eval contract; it survived in the docs.
- **Fix:** `--lib` added, taking it to one line with a real passed count (`1 passed; …;
  6688 filtered out`), plus a sentence saying why `--lib` is load-bearing and that the
  passed count, never the word `ok`, is the thing to read. The command was run in both
  forms to obtain those two numbers — and the live network test **passed**.
- **Commit:** `2ba6781f6`

**3. [Rule 2 — Missing critical functionality] `--dump` would have buffered the whole stream**

- **Found during:** Task 3, wiring the dump into Task 2's writer.
- **Issue:** Task 2's helper takes `&[u8]`, so the dump would have been materialized into a
  `Vec` before writing — up to ~60 MB at the default hard cap. Reintroducing `O(budget)`
  memory in the one command whose job is to demonstrate its absence is the sort of thing
  nobody would look for.
- **Fix:** `atomic_write_with(target, force, fill)` takes a closure over the open temp file;
  `dump_pairs` streams straight into it. `atomic_write(target, bytes, force)` remains as a
  thin wrapper, so there is still exactly ONE `rename` site and Task 2's three proofs cover
  both artifacts unchanged. Falsification C above shows the dump genuinely goes through it.
- **Commit:** `05d484aaf`

### Interface decisions the plan did not anticipate

**4. The seed mode is derived, not stored.** The plan asks that "the manifest records which
mode was used". `SelectionPayload` is crate-owned and `deny_unknown_fields`; adding a field
would need a schema bump that invalidates 02-05's four committed payload goldens. It is also
the worse design: the manifest already records `root_seed`, and the mode is a *total
function* of it, so a stored `seed_mode` could only ever introduce a way to disagree with the
seed printed beside it. The CLI reports the mode in both output forms
(`"seed_mode": "uncontracted"` under `--json`) and the doc states the derivation.

**5. `cli.offline` is not threaded through.** The plan's `key_links` says the dispatch arms
pass `cli.json`/`cli.offline`. `cli.json` is passed (via the existing `let json = cli.json`
binding at the top of `dispatch_data_command`). `offline` is not: neither command opens a
socket — the crate cannot (`make contrastive-data-boundary` enforces it) and the adapter
only reads local files — so accepting a network switch in order to ignore it would advertise
a capability that does not exist. The reason is a comment at the dispatch site. Note that
`--offline` still *renders* in both help texts because it is a global flag; that is
pre-existing behaviour on every subcommand, not something this plan introduced.

**6. `pub(crate) mod data_contrastive;`, not `pub mod`.** Matches the neighbouring
`pub(crate) mod data_tweeteval;` and every other command module.

### Files modified beyond `files_modified`

`crates/apr-cli/src/commands/data_tweeteval.rs` is not in the plan's `files_modified`, and
was changed twice, both anticipated by 02-06's own handoff ("`attestation_bytes_from_manifest`
is the read path — make it `pub(crate)` when a second caller appears"):

1. **Four items became `pub(crate)`:** `attestation_bytes_from_manifest`, `MANIFEST_FILE`,
   `BENCHMARK_SEEDS`, `FEW_SHOT_SIZES`. This is what keeps ONE reader of
   `benchmark-manifest.json` and ONE list of seeds and shot sizes — the same arrays written
   into every manifest's `few_shot` section — rather than a second copy in the new module.
2. **The synthetic fixture writer moved** into a `#[cfg(test)] pub(crate) mod fixtures` and
   gained a text `tag` parameter, so a genuinely *mixed* directory (two preparations, one
   split swapped) is constructible. The default tag preserves the historical bytes; the
   16 pre-existing `data_tweeteval` tests pass unchanged, 2 ignored, exactly as 02-06 left
   them.

---

## The TDD claim, stated precisely

**Task 2 had a real RED**: `cargo test -p apr-cli --lib data_contrastive` reported
**0 passed / 19 failed** against the placeholder body, committed at `04f4b7f98` before any
implementation existed.

**Task 3 did not.** Its tests reference `PairsOutcome`, `pairs_outcome` and
`pairs_report_json`, so they could not *compile* until the interface existed — the same
produced-before-consumed constraint the plan's own checker found in Task 1. Rather than
manufacture a RED after the fact and describe it as one, the Task 3 gates are falsified by
mutations A–C above; falsification A (the placeholder body) is the RED that would have been
observed, run afterwards and reported as what it is. Three mutations that each turn exactly
the predicted tests red are stronger evidence than a RED, which only ever shows "not yet
implemented".

One further honesty note: after the 36/36 run, `cargo fmt` reflowed four lines and the disk
filled before the suite could be re-run on the committed bytes. That gap was reported rather
than glossed, and closed on resume — **36 passed / 0 failed on the exact committed bytes**.

---

## The ENOSPC interruption

Execution stopped mid-plan with `ld: write() failed, errno=28`, at **546 MiB free** with
`target/debug/incremental` at 25 GB. Per the standing instruction nothing was deleted, no
`cargo clean` was run, and the run was reported as a checkpoint with the measured numbers and
the exact remedy. The coordinator freed 21 GB; every build after that point used
`CARGO_INCREMENTAL=0` and the cache did not regrow (15 GB still free at the end). Recorded
here because it is the phase's second ENOSPC and both had the same cause.

---

## Verification

Statuses captured directly (`cmd > log 2>&1; rc=$?`), never through a pipe (CLAUDE.md
rule 1). `rtk proxy` for every porcelain and grep result. Counts read from raw logs, since
`rtk` rewrites `cargo test` output into a summary line.

| Gate | rc | Result |
|---|---|---|
| `cargo test -p apr-cli --lib data_contrastive` | 0 | **36 passed**, 0 failed (on the committed bytes) |
| `cargo test -p apr-cli --lib` | 0 | **6675 passed**, 14 ignored, 0 failed |
| **cross-crate baseline** (`--lib -- --skip gpu::`) | 0 | **14,290 passed**, 0 failed (6675 + 212 + 7403) |
| `cargo check -p apr-cli -p aprender-train -p aprender-contrastive-data --all-targets` | 0 | clean |
| `cargo fmt --check` (workspace) | 0 | clean |
| `cargo clippy -p apr-cli --no-deps --lib -- -D warnings` | 0 | clean |
| `cargo clippy -p aprender-contrastive-data --all-targets --no-deps -- -D warnings` | 0 | clean |
| `make contrastive-data-boundary` | 0 | deps a subset of the allowlist; no fs/net/path under `src/` |
| `make contract-audit-phase2` | 0 | every equation bound, 9/9 obligations covered |
| `cargo package --no-verify -p aprender-contrastive-data` | 0 | 59 files, 599.8 KiB |
| `scripts/check_apr_bin_pinned.sh` | 0 | 28 CI-surface files, every invocation pinned |
| `cargo test -p apr-cli --lib pinned_upstream… -- --ignored` | 0 | **1 passed** (live network) |
| Phase 1 setfit lib gate | 0 | 162 passed — unchanged |
| `cargo clippy -p apr-cli --no-deps --all-targets` | **101** | **pre-existing** — see below |
| `cargo package --no-verify -p apr-cli` | **101** | **KNOWN-RED, expected** — see below |
| `make tier2` | **2** | **RED — pre-existing D-ITEM-02** |

**The baseline reconciles exactly.** 14,254 → 14,290 is +36, precisely this plan's 36 new
lib tests (apr-cli 6639 → 6675). `aprender-contrastive-data` 212 and `aprender-train` 7403
are both unchanged.

### The three reds, each attributed rather than waved past

- **`cargo clippy -p apr-cli --all-targets`** — 0 of the errors name `data_contrastive` or
  `data_tweeteval`. Offenders: `tests/falsification_crux_a_25.rs` (31),
  `falsification_crux_f_07.rs` (7), `falsification_crux_k_08.rs` (4) — all disallowed
  `unwrap()` in integration tests — and `src/commands/nf4_classifier.rs` (2 `unused_mut`,
  the out-of-scope observation standing since 02-05). Control:
  `git diff --name-only 37cc105ad..HEAD` lists six files and **none of those four is among
  them**. The `--lib` form, which is what 02-06 gated on, is green.
- **`cargo package -p apr-cli`** — unchanged from 02-08: `no matching package named
  aprender-contrastive-data found`. Any form, verifying or not. Expected until the
  human-approved publish cascade lands `aprender-contrastive-data` first. Nothing was
  published.
- **`make tier2`** — 24 clippy errors, attributed by file: **`aprender-compute` 19,
  `aprender-zram-core` 3, zero anywhere else**; `grep` for any Phase 2 crate across the
  error context returns **0**. D-ITEM-02, identical to 02-03/05/06/07/08. CI is all
  `[self-hosted, X64, Linux]` and never lints these aarch64 arms.

### The pinning guard's scope was measured, not read

`scripts/check_apr_bin_pinned.sh` is green (28 files). Its scope was established with a
two-sided case table rather than by reading the regex (CLAUDE.md rule 7):

| Probe | Predicted | Observed |
|---|---|---|
| bare `apr data select …` added to `docs/examples/tweet-eval-stance.md` | silent — docs are out of scope by design | rc=0, unchanged |
| bare `@apr qa model.apr` added as a Makefile recipe | flagged | `rc=1`, `BARE-APR Makefile:1282` |

Both probes reverted. So the guard does **not** cover documentation, and nobody should later
assume it does. All 9 `apr` invocations added to the doc are pinned anyway; the pre-existing
`finetune`/`eval` examples were left alone because this plan did not execute them and does
not vouch for them.

### Measurement notes

- Every doc transcript came from a real run; the only edit is the output directory,
  shortened to `data/tweet-eval-stance`. That the hashes are path-independent was
  *verified*, not assumed: seed 13 written to two different `-o` directories produced the
  same `1d2dbbb3…`.
- **`bashrs` is NOT installed** (CLAUDE.md mandates it over shellcheck). This plan added no
  shell logic — no Makefile or script change — so nothing was owed, and shellcheck was not
  substituted.
- `cargo test` takes ONE positional; every command quoted here carries at most one and each
  was executed rather than transcribed.
- The pair-dump audit was computed from the artifacts (`json.load` over the manifest, the
  dump and all three splits), not asserted from the command's own output.

---

## Threat Model Dispositions

| Threat | Disposition |
|---|---|
| T-02-29 forged selection manifest replayed into pairs | **mitigated** — `Selection::replay` is the only route back (the CLI cannot construct one; `assemble` is crate-private), and three resealed forgeries exercise the membership, row-hash and recomputation rungs from the CLI |
| T-02-30 selection over a compatibility directory | **mitigated** — only the canonical constructor is ever called; demonstrated on a real setfit directory, refused on BOTH commands |
| T-02-48 mixed / stale / tampered directory | **mitigated** — four real directories refused by split digest, schema version, fingerprint and profile, with the messages quoted above |
| T-02-31 partial or clobbered writes | **mitigated** — one writer, temp-in-destination + `sync_all` + `rename`, no-clobber, temp cleanup; two mutations prove the cleanup and the injection window, a third proves `--dump` inherits them |
| T-02-49 unbounded pair budget from the command line | **mitigated** — over-cap is `BudgetExceedsHardCap` naming both numbers plus the flag to change; the silent-clamp mutation turns exactly that test red |
| T-02-50 selection on an uncontracted seed without a record | **mitigated** — `--seed` required and validated; `--any-seed` explicit; the seed is in the manifest and the mode is a total function of it |
| T-02-32 ambiguous human-only output | **mitigated** — `--json` on both commands, carrying ordered ids with labels, both hashes, exclusions, reduced pools, the persisted ledger, and the full replay tuple with all three deviation clauses verbatim |

## Threat Flags

None. No new network endpoint, auth path, or trust-boundary schema. The one new file-access
pattern — reading a benchmark directory the user names and writing one artifact into it — is
inside the directory the user supplied and goes through the audited writer.

## Known Stubs

None. Both commands are fully implemented; no placeholder body, feature gate or `#[ignore]`
remains from this plan.

---

## Notes for later plans

- **Phase 3** — `selection-manifest.json` is the handoff artifact. Read it with
  `SelectionManifest::from_bytes` and reconstitute only through `Selection::replay`; the pair
  stream is `PairConfig → PairSampler::new → PairReplayRecord::from_sampler`, offset-resumable
  via `iter_from`.
- **Phase 5** — the selection lock has a real artifact: `payload.access_ledger` with
  `ledger_hash` beside it, written by the command a user actually runs.
- **Anyone touching `data_contrastive.rs`** — the `FAIL_BEFORE_RENAME` seam
  (`#[cfg(test)] thread_local`, read by `fill_and_sync` between `sync_all` and `rename`) is
  the mechanism behind three write-safety proofs. If you refactor the writer, re-run
  mutations 1, 2 and 5 above rather than trusting that the tests still mean what they say.
- **Anyone adding a `#[contract]` site to apr-cli** — `make contract-audit-phase2` is
  blocking in tier3 since 02-08. This plan added no equation and no annotation, and the gate
  is green.
- **Out-of-scope observations, unchanged since 02-05:** `nf4_classifier.rs` still emits two
  `unused_mut` warnings, and `crates/apr-cli/tests/falsification_crux_*.rs` carry disallowed
  `unwrap()` calls that make `clippy --all-targets` red for the package. Neither was touched.

## Self-Check: PASSED

| Item | Status |
|---|---|
| `crates/apr-cli/src/commands/data_contrastive.rs` | FOUND — contains `from_attested_bytes` (2), `Selection::replay` (2), `atomic_write_with` (3) |
| `crates/apr-cli/src/data_commands.rs` | FOUND — contains `Select`, `Pairs`; `grep -c '"42"'` = 0 |
| `crates/apr-cli/src/dispatch_analysis.rs` | FOUND — contains `data_contrastive` (2 arms) |
| `crates/apr-cli/src/commands/mod.rs` | FOUND — contains `data_contrastive` |
| `crates/apr-cli/src/commands/data_tweeteval.rs` | FOUND — contains `pub(crate) mod fixtures` |
| `docs/examples/tweet-eval-stance.md` | FOUND — contains `apr data select`, `apr data pairs` |
| commits `f5e8ea056` `04f4b7f98` `8540eba88` `05d484aaf` `2ba6781f6` | FOUND (5/5) |
| Three `.snap.new` files under `crates/aprender-train/src/prune/snapshots/` | intact; `git diff --diff-filter=D 37cc105ad..HEAD` empty |
| `Sha256` / `swap(` / `unrank` / `json!` in non-comment lines | 0 / 0 / 0 / 0 (control: `BTreeMap` = 3) |
| `fs::rename` sites / `File::create` / `unwrap()` | 1 / 0 / 0 |
