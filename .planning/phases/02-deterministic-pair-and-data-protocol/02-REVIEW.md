---
phase: 02-deterministic-pair-and-data-protocol
reviewed: 2026-08-09T00:00:00Z
depth: standard
files_reviewed: 34
files_reviewed_list:
  - crates/apr-cli/src/commands/data_contrastive.rs
  - crates/apr-cli/src/commands/data_tweeteval.rs
  - crates/apr-cli/src/commands/eval/eval_mod_tests.rs
  - crates/apr-cli/src/commands/eval/mod.rs
  - crates/apr-cli/src/commands/mod.rs
  - crates/apr-cli/src/data_commands.rs
  - crates/apr-cli/src/dispatch_analysis.rs
  - crates/apr-cli/src/extended_commands.rs
  - crates/aprender-contrastive-data/src/attestation.rs
  - crates/aprender-contrastive-data/src/buckets.rs
  - crates/aprender-contrastive-data/src/dedup.rs
  - crates/aprender-contrastive-data/src/error.rs
  - crates/aprender-contrastive-data/src/hash.rs
  - crates/aprender-contrastive-data/src/ledger.rs
  - crates/aprender-contrastive-data/src/lib.rs
  - crates/aprender-contrastive-data/src/manifest.rs
  - crates/aprender-contrastive-data/src/pairs.rs
  - crates/aprender-contrastive-data/src/prepared.rs
  - crates/aprender-contrastive-data/src/rng.rs
  - crates/aprender-contrastive-data/src/schema.rs
  - crates/aprender-contrastive-data/src/select.rs
  - crates/aprender-contrastive-data/src/split.rs
  - crates/aprender-contrastive-data/tests/common/mod.rs
  - crates/aprender-contrastive-data/tests/goldens_regenerate.rs
  - crates/aprender-contrastive-data/tests/negative_leaky.rs
  - crates/aprender-contrastive-data/tests/negative_materializing.rs
  - crates/aprender-contrastive-data/tests/pair_counts.rs
  - crates/aprender-contrastive-data/tests/reference_fixtures.rs
  - crates/aprender-contrastive-data/tests/ui.rs
  - crates/aprender-train/src/eval/classification/basic_tests.rs
  - crates/aprender-train/src/eval/classification/metrics.rs
  - crates/aprender-train/src/eval/classification/mod.rs
  - scripts/setfit_fixtures/generate_fixtures.py
  - Makefile
findings:
  critical: 3
  warning: 14
  info: 0
  total: 17
status: issues_found
---

# Phase 2: Code Review Report

**Reviewed:** 2026-08-09
**Depth:** standard
**Files Reviewed:** 34
**Status:** issues_found

## Summary

The library core (`aprender-contrastive-data/src/`) is the strongest part of this
submission. I actively looked for the failure modes the domain brief named and did **not**
find them: there is no `HashMap`/`HashSet` anywhere in `src/` (union-find is a `Vec`,
every keyed structure is a `BTreeMap`/`BTreeSet`); no clock, no filesystem, no thread
scheduling reaches a hashed artifact; the Philox path is genuinely stateless and ordinal-
pure; `positive_capacity`/`negative_capacity`/`default_epoch_budget`/`unique_capacity_check`
all use checked arithmetic and the running-prefix form of `negative_capacity` really does
overflow later than the `(S²−Σn²)/2` form; the split-hash comparison in `verified_split`
really does precede the parse; and `PairSampler` retains nothing proportional to the
budget. The dedup coalescing is correct (one forest, both edge kinds) and the
`shares_a_key` / `class_of_global` edge cases hold up under empty and repeated-offset
layouts.

The defects are concentrated at the **edges** the core deliberately pushed outward: the
`apr-cli` filesystem adapters, the replay ladder's *unchecked* payload fields, and the
gating machinery that is supposed to make all of this evidence rather than assertion.

Three findings block:

1. `apr data select` / `apr data pairs` build filesystem paths out of **unvalidated role
   strings taken from an untrusted attestation**, before the crate ever gets to reject them.
2. `apr data tweet-eval-stance --force` **destroys a pre-existing valid benchmark directory**
   on any later failure — it truncates in place and then unlinks on rollback, while the same
   phase's sibling module has a correct `atomic_write_with` that it does not use.
3. The three gate lines this phase wired into `tier2`/`tier3` **cannot fail** on any host
   with GNU Make ≥ 3.82, which I reproduced. The Makefile comments assert the opposite
   ("BLOCKING", "wired into tier3").

The recurring shape of the warnings is worth naming: several artifacts are *written* with a
field and then never *checked* on the way back in (`normalization_version`, `label_names`,
`ledger_hash`, `access_ledger`), and one contract equation is bound to a function no
production path calls. Both patterns produce artifacts that look like evidence and are not.

## Critical Issues

### CR-01: Untrusted attestation role names are used to build filesystem paths (path traversal)

**File:** `crates/apr-cli/src/commands/data_contrastive.rs:221-227, 256-262`

**Issue:**
`read_attested_canonical` reads the benchmark manifest, extracts the `dataset_attestation`
section, deserializes it, and then loops over **the attestation's own split-role keys** to
build the paths it reads:

```rust
for role in attestation.splits.keys() {
    buffers.insert(role.clone(), read_required(&data_dir.join(role_file(role)))?);
}
```

with `role_file(role)` = `format!("{role}.jsonl")` for anything that is not
`compatibility_test`. `role` is an arbitrary JSON object key from a file inside the
directory the user pointed `--data` at — precisely the "bytes that arrive from object
storage are untrusted" threat model the crate's own module docs state.

Two concrete escapes, both standard `std::path::Path::join` semantics:

- `"role": "../../../../etc/shadow"` → reads `<data>/../../../../etc/shadow.jsonl`.
- `"role": "/etc/shadow"` → `Path::join` with an absolute component **replaces the base
  entirely**, so this reads `/etc/shadow.jsonl`.

The role-set validation that would refuse these lives in
`attestation::preflight` (`attestation.rs:233-240`), which runs inside
`from_attested_bytes` — i.e. *after* every one of these reads has already completed.
Consequences that survive the later rejection:

- **File-existence oracle.** `read_required` returns a distinct
  `"{path} not found. Prepare one with ..."` message for `NotFound` vs. falling through to
  the crate's `ConflictingSourceRole`, so the error text discriminates existence of any
  attacker-chosen `*.jsonl` path.
- **Unbounded read / hang.** The whole file is `fs::read`-ed into memory before any
  validation, so a role naming a large file or a FIFO exhausts memory or blocks forever.
- The attestation map is a `BTreeMap`, so an attacker can supply arbitrarily many roles and
  drive arbitrarily many such reads in one invocation.

Nothing in the test module covers a role containing a path separator (I grepped: the only
role literals in the tests are `train`/`validation`/`test`/`compatibility_test`).

Note the correct pattern already exists two files over:
`data_tweeteval::verify_prepared_directory` iterates `role_files(profile)` — a hardcoded
list — and is not vulnerable.

**Fix:** Iterate the profile's own role list, and treat any attestation role outside it as
an error *before* touching the filesystem.

```rust
use aprender_contrastive_data::attestation::AttestedProfile;

fn read_attested_canonical(
    data_dir: &Path,
    ledger: &mut AccessLedger,
) -> Result<PreparedDataset<Canonical>> {
    let manifest_bytes = read_required(&data_dir.join(data_tweeteval::MANIFEST_FILE))?;
    let attestation_bytes = data_tweeteval::attestation_bytes_from_manifest(&manifest_bytes)?;
    let attestation =
        DatasetAttestation::from_bytes(&attestation_bytes).map_err(|e| dataset_error(&e))?;

    // Refuse a role this profile does not emit BEFORE it can name a path. The crate
    // re-checks in `preflight`; this copy exists only so the check precedes the read.
    for role in attestation.splits.keys() {
        if !Canonical::ROLES.contains(&role.as_str()) {
            return Err(CliError::ValidationFailed(format!(
                "{} names split role {role:?}, which the canonical profile does not emit; \
                 expected one of {:?}",
                data_tweeteval::MANIFEST_FILE,
                Canonical::ROLES
            )));
        }
    }

    let mut buffers = BTreeMap::new();
    for role in Canonical::ROLES {
        if attestation.splits.contains_key(*role) {
            buffers.insert(
                (*role).to_string(),
                read_required(&data_dir.join(role_file(role)))?,
            );
        }
    }
    PreparedDataset::<Canonical>::from_attested_bytes(&attestation_bytes, &buffers, ledger)
        .map_err(|e| dataset_error(&e))
}
```

Add a regression test whose manifest carries `"../escape"` and `"/etc/passwd"` roles and
asserts the command fails without reading outside `--data`.

---

### CR-02: `--force` re-prepare truncates in place and then deletes the pre-existing benchmark on rollback

**File:** `crates/apr-cli/src/commands/data_tweeteval.rs:738-804` (`write_outputs`),
`813-825` (`write_file`), `807-811` (`remove_all`)

**Issue:**
With `--force`, `write_file` opens each split with `create(true).truncate(true)` and writes
in place (`820`). Every successfully written path is pushed onto `written`. If any *later*
step fails — the manifest encode/write (`788-796`) or, much more likely, the read-back
`verify_prepared_directory` (`799-802`) — the handler calls `remove_all(&written)`, which
**unlinks** those paths (`807-811`).

Those paths are not necessarily new files. On a `--force` re-prepare of an existing, valid
benchmark directory they are the user's previous `train.jsonl` / `validation.jsonl` /
`test.jsonl` / `benchmark-manifest.json`, whose contents this run already destroyed by
truncation. So the "rollback" leaves the directory strictly worse than it found it: the
prior good dataset is gone and nothing replaced it. That is data destruction on an error
path, not a rollback.

The Setfit branch makes it worse: `778-786` deletes a stale `validation.jsonl` under
`--force`, and that deletion is never undone by `remove_all` either.

`verify_prepared_directory` is not a hypothetical failure — it is the whole attested
boundary (schema version, per-split digest, class counts, exclusion digest, fingerprint),
and it is deliberately placed last.

This is especially avoidable because the *same phase* built the correct writer next door:
`data_contrastive::atomic_write_with` (`data_contrastive.rs:157-179`) writes a temp file in
the destination directory, `sync_all`s it, renames, and on failure removes only the temp —
leaving any pre-existing target untouched. `data_tweeteval.rs` routes through none of it.

**Fix:** Stage every artifact and rename only after `verify_prepared_directory` passes;
never unlink a path this run did not create.

```rust
// Sketch: write all splits + manifest to `<output_dir>/.apr-stage.<pid>/`, run
// verify_prepared_directory against the STAGING dir, and only then rename each file
// into place. On any failure remove the staging dir and nothing else.
//
// Minimum viable fix if staging is too large a change: record which of `known_files`
// existed BEFORE the run, and have `remove_all` skip those — a pre-existing file must
// never be unlinked by a rollback.
```

Either way, hoist the shared helper (`atomic_write_with` and a `staged_dir` sibling) into a
module both commands use, so "every artifact goes through ONE writer" is true across the
phase rather than within one file.

---

### CR-03: The three gates this phase wired into tier2/tier3 cannot fail under GNU Make ≥ 3.82

**File:** `Makefile:20` (`.ONESHELL:`), `Makefile:243`, `Makefile:294`, `Makefile:303`

**Issue:**
The Makefile sets `.ONESHELL:` (line 20) and `SHELL := /bin/bash` (line 11) but never sets
`.SHELLFLAGS`. GNU Make's default is `-c`, with **no `-e`**. Under `.ONESHELL`, the entire
recipe is handed to one `bash -c`, so make sees only the **last** command's exit status;
every intermediate failure is discarded.

The three lines this phase added are all mid-recipe:

| line | command | last line of its recipe |
|---|---|---|
| 243 | `@cargo test -p aprender-contrastive-data` | 252 `@echo "Tier 2: PASSED"` |
| 294 | `@$(MAKE) contract-audit-phase2` | 312 `@echo "Tier 3: PASSED"` |
| 303 | `@$(MAKE) contrastive-data-boundary` | 312 `@echo "Tier 3: PASSED"` |

So on any host with GNU Make ≥ 3.82, a failing `cargo test -p aprender-contrastive-data`
prints its failure and then `Tier 2: PASSED`, exit 0.

**Reproduced, not inferred.** Minimal replica with the same two directives:

```make
SHELL := /bin/bash
.ONESHELL:
t2:
	@echo "step A"
	@false
	@echo "step B"
	@echo "tier2 done"
```

- `make 3.81` (the macOS default on this box; `.ONESHELL` predates its support and is
  ignored) → `exit=2`, stops after `step A`.
- `gmake 4.4.1` → `exit=0`, prints `step A`, `step B`, `tier2 done`.

That version split is the trap: the gates were verified on a host where `.ONESHELL` is
inert, and are non-blocking everywhere `.ONESHELL` actually works — which includes every
Linux CI image.

This is the failure the project's own Verification Discipline names ("a guard that cannot
fail is theater", rule 5), and the Makefile comments assert the opposite in three places:
`"This target is BLOCKING and is wired into tier3"` (1136-block), `"the BLOCKING coverage
gate"` (tier3 comment), `"a gate outside the tiers is a target that stops being run"`.
The standalone-run evidence recorded in those comments is real but proves only that the
targets pass in isolation; it cannot prove they *block* when wired.

Note the two Phase-2 targets themselves are internally sound — `contrastive-data-boundary`
and `contract-audit-phase2` both terminate with explicit `exit 1`, which under `.ONESHELL`
does exit the single shell. The defect is purely in how their status is consumed by the
enclosing recipe. `.ONESHELL:` is pre-existing, so the fix is one line at the top and it
repairs every tier at once.

**Fix:**

```make
SHELL := /bin/bash
# .ONESHELL runs each recipe as ONE bash -c. Without -e, make can only see the LAST
# command's status, so every mid-recipe gate is silently non-blocking. -o pipefail
# additionally stops a piped gate from reporting the pipe's status (CLAUDE.md rule 1).
.SHELLFLAGS := -e -u -o pipefail -c
.ONESHELL:
```

Then **re-mutate in the new scope** (Verification Discipline rule 4): break
`aprender-contrastive-data`'s test suite on purpose, run `gmake tier2`, and confirm a
non-zero exit — the existing standalone proof does not transfer. Expect the `-e -u` flip to
surface latent failures in other recipes; that is the point.

## Warnings

### WR-01: `Selection::replay` never validates `payload.normalization_version`

**File:** `crates/aprender-contrastive-data/src/select.rs:511-526` (`check_versions`),
`manifest.rs:121-122`

**Issue:** `SelectionPayload.normalization_version` is written at `select.rs:186` and never
read back. `check_versions` checks only `schema_version` and `algorithm_version`;
`check_provenance` does not look at it. So a manifest declaring
`"normalization_version": "nfc-trim-ws-v99"` replays cleanly and yields a `Selection` whose
public payload advertises a normalization pipeline this build does not implement.

The typed error already exists and its own doc explains exactly why this matters
(`error.rs:366-381`: *"Silently accepting a foreign tag would let an exclusion record
computed under different collapsing rules be replayed as if it had been computed under
these ones"*), and `attestation::preflight` (`attestation.rs:218-223`) does perform this
check. Only the selection-replay ladder omits it. Partial mitigation: the *exclusion
record's* embedded version is covered transitively by the exclusion-hash comparison — but
the payload's own top-level field, which is what a consumer reads, is not.

**Fix:** Add to `check_versions`:

```rust
if payload.normalization_version != CONTENT_NORMALIZATION_VERSION {
    return Err(ContrastiveDataError::UnsupportedNormalizationVersion {
        got: payload.normalization_version.clone(),
        supported: CONTENT_NORMALIZATION_VERSION,
    });
}
```

### WR-02: `Selection::replay` never compares `payload.label_names` with the dataset's

**File:** `crates/aprender-contrastive-data/src/select.rs:529-562` (`check_provenance`),
`615-633` (`check_class_balance`)

**Issue:** `payload.label_names` is used only for its `len()` (`select.rs:620`). Its
*contents* are never compared against `dataset.label_names()`. The dataset fingerprint does
absorb the dataset's label map (`hash.rs:169-172`), and `check_provenance` compares that
fingerprint — but that pins the *dataset's* names, not the payload's copy of them. A
manifest whose `label_names` are permuted or renamed with the same arity
(`["favor","against","none"]` vs `["none","against","favor"]`) survives every rung including
the step-5 recomputation, because `SelectedExample` carries a numeric `label` and no name.

The result is a replayed `Selection` whose `payload().label_names` mislabels every class for
any downstream consumer that reads them — and this crate's own `apr data select --json`
report (`data_contrastive.rs:414-417`) does read them.

**Fix:** In `check_provenance`:

```rust
if payload.label_names != dataset.label_names() {
    return Err(ContrastiveDataError::SelectionReplayMismatch {
        field: "label_names".to_string(),
    });
}
```

### WR-03: `Selection::replay` accepts `access_ledger` and `ledger_hash` verbatim, re-deriving neither

**File:** `crates/aprender-contrastive-data/src/select.rs:486-491`, `496-508`
(`digest_from_hex`)

**Issue:** Replay ends with

```rust
let ledger_hash = digest_from_hex(&payload.ledger_hash).ok_or_else(...)?;
Self::assemble(recorded, payload.clone(), ledger_hash)
```

`digest_from_hex` only checks that the string is 64 hex characters. Nothing verifies that
`payload.ledger_hash == hash(LedgerWire { records: payload.access_ledger })`, and nothing
verifies `payload.access_ledger` against anything at all. Both fields sit inside the hashed
payload, so a hand-edited manifest that is **resealed** — which `manifest_replay_tests`'s
`reseal()` helper demonstrates is trivial, and which step 5 exists precisely because such
forgeries are internally consistent — passes with an arbitrary access log and an arbitrary
digest beside it.

Step 5 recomputes `ordered_examples` and nothing else, so it does not cover this. The result
is that `Selection::ledger_hash()` after a replay returns an attacker-chosen value, and the
ledger the module doc calls *"the artifact a later phase's selection lock reads"*
(`ledger.rs:80-86`) is not evidence once it has round-tripped through a file.

Two smaller defects in the same helper: `char::to_digit(16)` accepts `A`-`F`, so
`digest_from_hex` contradicts its own doc comment ("lowercase hex digest") and will accept a
non-canonical rendering; and the parsed digest is never compared back to its source string.

**Fix:** Recompute rather than parse.

```rust
// The ledger inside the payload must be self-consistent, or the record it constitutes
// proves nothing. Rebuild it and take its own hash instead of trusting the hex field.
let recorded_ledger = AccessLedger::from_records(payload.access_ledger.clone());
let ledger_hash = recorded_ledger.ledger_hash();
if hex(&ledger_hash) != payload.ledger_hash {
    return Err(ContrastiveDataError::SelectionReplayMismatch {
        field: "ledger_hash".to_string(),
    });
}
```

(`AccessLedger` will need a `pub(crate) fn from_records(Vec<AccessRecord>) -> Self`; it must
stay `pub(crate)` so the append-only public API is unchanged.) Consider also asserting that
the recorded ledger's last entry has `purpose == "select"` and the expected
`dataset_fingerprint`, which is the property the future selection lock actually needs.

### WR-04: the `default_epoch_budget` contract binding names a function no production path calls

**File:** `crates/aprender-contrastive-data/src/pairs.rs:211-223`
(`effective_default_budget`), `383-418` (`resolve_budget`, esp. 407-408);
`contracts/aprender/binding.yaml:1002-1007`

**Issue:** `binding.yaml` binds equation `default_epoch_budget` to
`aprender_contrastive_data::pairs::effective_default_budget`, and
`contract-audit-phase2` — the phase's new BLOCKING gate — certifies that binding. But
`effective_default_budget` has no non-test caller: I grepped the whole `crates/` tree and
its only references are `tests/pair_counts.rs`, `src/pairs.rs`'s own unit tests, and a doc
link. The path the sampler actually takes is `PairLayout::from_class_sizes` →
`resolve_budget`, which re-implements the clamp inline:

```rust
let closed_form = default_epoch_budget(class_sizes)?;
let resolved = closed_form.min(hard_cap);
```

So there are two copies of the contracted equation and the audited one is not the one that
runs. `pair_counts.rs` does exercise `resolve_budget` against the committed fixtures, which
is why the two agree today — but a change to `resolve_budget`'s clamp would leave the
bound function, its tests and the binding audit all green while the emitted stream moved.
This is the same "two ladders that agree today are two ladders that will disagree
eventually" hazard the crate calls out for the split constructors and the assembly points,
applied inconsistently here.

**Fix:** Make `resolve_budget` call the bound function, so there is one implementation:

```rust
None => {
    let closed_form = default_epoch_budget(class_sizes)?;
    let resolved = effective_default_budget(class_sizes, hard_cap)?;
    if resolved == 0 { return Err(ContrastiveDataError::ZeroBudget); }
    Ok((resolved, closed_form > hard_cap))
}
```

### WR-05: `PairReplayRecord::to_config` sets `hard_cap = budget`, letting an untrusted record bypass `DEFAULT_HARD_CAP`

**File:** `crates/aprender-contrastive-data/src/manifest.rs:400-447`, esp. 444-445

**Issue:** `PairReplayRecord` derives `Deserialize` and is documented as the record a
consumer "carries" and replays from — i.e. untrusted input. `to_config` deliberately sets
`hard_cap: Some(self.budget.max(1))`, so whatever budget the record declares is
self-authorising. A record with `"budget": 18446744073709551615` yields a `PairConfig` that
`resolve_budget` accepts, and every consumer of the resulting sampler
(`pair_manifest_hash` at `manifest.rs:523`, `dump_pairs` at `manifest.rs:556`,
`count_kinds` at `data_contrastive.rs:600`) then loops that many times. The module doc for
`DEFAULT_HARD_CAP` states the cap is *"a denial-of-service ceiling for adversarial inputs"*
(`pairs.rs:34-41`); on this path it is not applied at all.

The current CLI does not deserialize a `PairReplayRecord` from disk, so this is latent
rather than live — but the type is public, `to_config` is the documented replay entry point,
and it carries the `pair_manifest_replay` contract annotation.

**Fix:** Bound the replayed budget against the build's own ceiling and report the refusal
rather than honouring it:

```rust
if self.budget > DEFAULT_HARD_CAP {
    return Err(ContrastiveDataError::BudgetExceedsHardCap {
        budget: self.budget,
        hard_cap: DEFAULT_HARD_CAP,
    });
}
```

### WR-06: `parse_pair_dump` silently skips blank lines and misreports the offending index

**File:** `crates/aprender-contrastive-data/src/pairs.rs:1096-1117`

**Issue:** Two problems in one iterator chain:

```rust
for (index, line) in text.lines().filter(|line| !line.trim().is_empty()).enumerate()
```

1. **Blank lines are dropped.** `schema::jsonl_lines` (`schema.rs:92-105`) deliberately does
   the opposite and documents why: *"silently skipping blanks would let a truncated upload
   look like a shorter dataset."* The same reasoning applies to a pair dump, which is the
   audit artifact. Two parsers in one crate with opposite blank-line policies is a drift the
   docs already argue against.
2. **`index` is post-filter.** `enumerate()` is applied *after* `filter`, so the
   `MalformedRow { index }` reported for a bad record counts non-blank lines, not lines.
   With any blank line earlier in the file the reported index does not identify the line a
   reader must open — which is exactly what `error.rs`'s house style ("name the split, the
   index") exists to prevent.

`text.lines()` also strips `\r` from CRLF input, so a `\r`-terminated JSON line silently
parses where the JSONL path would reject it.

**Fix:** Enumerate before filtering, and reject blanks rather than skipping them:

```rust
for (index, line) in text.lines().enumerate() {
    if line.trim().is_empty() {
        return Err(ContrastiveDataError::MalformedRow {
            split: PAIR_DUMP_SPLIT.to_string(),
            index,
            reason: "blank line: a pair dump has exactly one record per line".to_string(),
        });
    }
    ...
}
```
(If a trailing newline must be tolerated, mirror `schema::jsonl_lines` exactly rather than
inventing a third rule.)

### WR-07: unchecked arithmetic inside `PairLayout`, contradicting the module's own claim

**File:** `crates/aprender-contrastive-data/src/pairs.rs:613`, `625`, `766`

**Issue:** The module header states *"Every capacity function returns `Result<u64, …>` and
uses checked arithmetic at every step"* (`pairs.rs:18-23`), and `positive_capacity` /
`negative_capacity` / `default_epoch_budget` / `unique_capacity_check` honour that. Three
sites in the same file do not:

```rust
pos_running += n * n.saturating_sub(1) / 2;   // 613  — unchecked * and +
running += n;                                  // 625  — unchecked +
let capacity = nonzero(n * n.saturating_sub(1) / 2, ...)?;  // 766 — unchecked *
```

These are currently *safe*, and I traced why: `positive_capacity(class_sizes)?` runs first
(`592`) and `checked_mul`s the identical product, and `total_examples` is accumulated with
`checked_add` (`599-601`). But the safety is an ordering coincidence in a sibling call, not
a local property — the neighbouring negative-weight lines (`618-623`) do use `checked_mul`/
`checked_add` for exactly the same class of quantity, so the file is internally inconsistent
about it. Reordering or extracting `from_class_sizes` would silently reintroduce a wrapped,
plausible-looking capacity, which is the failure the module doc calls out by name.

**Fix:** Use the checked forms and route the failures through the existing `overflow()`
helper, matching lines 618-623:

```rust
let pos_weight = n
    .checked_mul(n.saturating_sub(1))
    .ok_or_else(|| overflow("pair_layout/positive_class_weight"))?
    / 2;
pos_running = pos_running
    .checked_add(pos_weight)
    .ok_or_else(|| overflow("pair_layout/positive_weight_total"))?;
```

### WR-08: `triangular_prefix` has an unguarded precondition that underflows `u128` at `n = 0`

**File:** `crates/aprender-contrastive-data/src/pairs.rs:862-887`

**Issue:**

```rust
fn triangular_prefix(n: u64, i: u64) -> u64 {
    let doubled = u128::from(i) * (2 * u128::from(n) - 1 - u128::from(i));
    ...
}
```

At `n = 0` this evaluates `0u128 - 1`: a panic under `debug_assertions` and a wrap to
`u128::MAX` in release. `triangular_unrank`'s doc states the precondition (`n >= 2`) but
neither function asserts it, and `triangular_unrank(0, r)` reaches
`triangular_prefix(0, 0)` because `hi = n.saturating_sub(2)` collapses the search loop.

I traced the live call path and it is currently unreachable: `positive_draw` (`759-769`)
computes `nonzero(n * (n-1) / 2, …)?` before unranking, which short-circuits for `n < 2`.
So this is a latent hazard, not a live bug — but the crate elsewhere guards exactly this
kind of caller obligation with `debug_assert!` (`hash.rs:103-107`, `hash.rs:158-164`), and
the release-mode behaviour here is a silently wrong pair identity rather than a crash.

**Fix:**

```rust
fn triangular_prefix(n: u64, i: u64) -> u64 {
    debug_assert!(n >= 1, "triangular_prefix requires n >= 1; got {n}");
    debug_assert!(i < n, "triangular_prefix requires i < n; got i={i}, n={n}");
    ...
}

fn triangular_unrank(n: u64, rank: u64) -> (u64, u64) {
    debug_assert!(n >= 2, "triangular_unrank requires n >= 2; got {n}");
    ...
}
```

### WR-09: 63 banned `.unwrap()` calls in `data_tweeteval.rs`

**File:** `crates/apr-cli/src/commands/data_tweeteval.rs:908-1544` (test module)

**Issue:** `.clippy.toml` disallows `Option::unwrap` and `Result::unwrap` project-wide
("GH-41: unwrap() elimination complete — enforcement enabled"), and CLAUDE.md restates it as
a hard rule. This file, added by this phase, contains 63 `.unwrap()` calls. The sibling file
added by the same phase, `data_contrastive.rs`, contains **zero** and 129 `.expect(...)` —
so compliance is clearly achievable within the phase's own conventions and this file is the
outlier.

All 63 are inside `#[cfg(test)] mod tests`, which is why they currently escape:
`make tier1`/`tier2` run `cargo clippy -- -D warnings` without `--all-targets`
(`Makefile:127, 188, 259`), so the `cfg(test)` module is never compiled for lint. The rule
in `.clippy.toml` carries no test exemption.

**Fix:** Convert to `.expect("…")` with the message the sibling file already models
(`fs::read(&path).expect("benchmark manifest is readable")`), and consider adding
`--all-targets` to at least the tier2 clippy line so the ban is actually enforced where it
is written.

### WR-10: the `contrastive-data-boundary` symbol ban is bypassable by a braced `use`

**File:** `Makefile:422-425`

**Issue:** The source-surface half of the D-04 gate matches on literal strings:

```make
{ grep -nE 'std::fs|std::net|std::path' "$$srcfile" || true; \
  grep -nwE 'Path|PathBuf' "$$srcfile" || true; } \
| grep -vE '^[0-9]+:[[:space:]]*//' \
```

Must-not-pass cases the pattern misses:

| input | matched? |
|---|---|
| `use std::fs;` | yes |
| `use std::fs::File;` | yes |
| `use std::{fs, io};` then `fs::read(p)` | **no** — the literal `std::fs` never appears |
| `use std::{path::PathBuf};` | **no** for the first grep; caught by `-nwE 'PathBuf'` |
| `use std::{net::TcpStream};` | **no** — neither pattern matches |
| `/* uses Path */` block comment | flagged (false positive; only `//` lines are dropped) |

`use std::{fs, net};` therefore admits both banned capabilities. This is the exact class of
defect CLAUDE.md rule 7 records ("Guard regexes ship a case table … the `apr`-invocation
patterns were wrong five times; every one was caught by a must-match/must-not-match table,
none by review"), and the target's own comment block enumerates four induced failures — none
of which is a braced `use`.

**Fix:** Widen the pattern to cover brace grouping and commit the case table beside it:

```make
{ grep -nE 'std::(\{[^}]*\b)?(fs|net|path)\b' "$$srcfile" || true; \
  grep -nwE 'Path|PathBuf|File|OpenOptions|read_to_string|TcpStream|TcpListener' "$$srcfile" || true; } \
```

Then re-run the mutation table — including `use std::{fs, net};` — and record the result in
the comment block, since extending a guard's scope requires re-mutating in the new scope.

### WR-11: the crate's integration tests are not reachable from CI

**File:** `.github/workflows/ci.yml:272, 317`; `Makefile:243`

**Issue:** Five of this phase's eight test targets live under `tests/` and none of them runs
in CI:

- The workspace step is `cargo nextest run --profile ci --workspace --lib …` (`ci.yml:272`)
  — **`--lib` only**, so no integration targets and no doctests.
- The "Integration tests" step (`ci.yml:317`) is an explicit chain of 18
  `cargo test -p X --test Y` invocations; `aprender-contrastive-data` appears in none of
  them.

So `tests/ui.rs` (the trybuild compile-fail proofs that discharge
`OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE` / DATA-06), `tests/negative_materializing.rs` (the
`OBLIG-CPP-CAPACITY-INVARIANT` K = N boundedness gate),
`tests/negative_leaky.rs` (`OBLIG-CPP-LEAKY-RED`), `tests/pair_counts.rs` and
`tests/reference_fixtures.rs` execute only via `make tier2` — which per **CR-03** does not
block on any GNU Make 4 host. The obligations these files discharge are the phase's headline
claims, and the *evidence* for them is not in the required status checks.

**Fix:** Add `cargo test -p aprender-contrastive-data` (all targets) to the `ci.yml`
integration chain, in addition to fixing `.SHELLFLAGS`. Both are needed: the Makefile fix
makes `make tier2` honest for developers, the CI line makes the gate reachable by
`ci / gate` + `workspace-test`, which are the checks branch protection actually enforces.

### WR-12: `apr data pairs --dump` validates the no-clobber precondition only after two full stream sweeps

**File:** `crates/apr-cli/src/commands/data_contrastive.rs:625-666`

**Issue:** `pairs_outcome` runs, in order: `pair_manifest_hash` (`649`, one full
`0..budget` sweep), `count_kinds` (`650`, a second full sweep), and only then
`atomic_write_with(path, force, …)` (`653`), whose first act is the
`!force && target.exists()` refusal (`161-166`).

So `apr data pairs --dump existing.jsonl` without `--force` generates the entire pair stream
twice — up to `DEFAULT_HARD_CAP` = 1,048,576 pairs, each costing 2-3 SHA-256 key derivations
— before reporting a request defect it could have detected instantly. That directly
contradicts the module's own stated discipline for `run_select`: *"Fail-closed on the
REQUEST first. A bad shot count or an off-protocol seed is not a problem with the data, and
reporting it as one sends the user to the wrong place"* (`466-467`).

**Fix:** Hoist the existence check ahead of the sweeps.

```rust
// Refuse a request defect before doing any work, matching run_select's discipline.
if let Some(path) = dump {
    if !force && path.exists() {
        return Err(CliError::ValidationFailed(format!(
            "Refusing to replace existing file {} (pass --force to replace it)",
            path.display()
        )));
    }
}
```
(Keep the check inside `atomic_write_with` too — it is the one that must be authoritative;
this is a fast-path pre-flight, not a replacement.)

### WR-13: `common::manifest_entries` silently drops malformed manifest lines

**File:** `crates/aprender-contrastive-data/tests/common/mod.rs:147-160`

**Issue:**

```rust
text.lines().filter_map(|line| {
    let mut parts = line.split_whitespace();
    match (parts.next(), parts.next()) {
        (Some(digest), Some(name)) => Some((digest.to_string(), name.to_string())),
        _ => None,
    }
}).collect()
```

Any line that does not yield two whitespace-separated tokens is discarded without comment.
A truncated or corrupted `manifest.sha256` line therefore removes that file from the
integrity check rather than failing it — which is the "the manifest claims coverage it does
not have" failure the sibling verifier in `manifest.rs:900-911` explicitly guards against by
using `.expect("each manifest line is `<hex>  <name>`")`. Two readers of the same file
format with opposite strictness.

The `manifest_entries().is_empty()` guard at `mod.rs:186-188` only catches total loss, and
`reference_fixtures.rs:40-44` pins the expected count at 12 — which does catch this case
today, but only because the count is hardcoded. Make the reader strict so the guard does not
depend on a second assertion happening to exist.

**Fix:** Mirror the strict form:

```rust
.map(|line| {
    let (digest, name) = line
        .split_once("  ")
        .unwrap_or_else(|| panic!("malformed manifest line: {line:?}"));
    (digest.to_string(), name.to_string())
})
```
(with the blank/comment filter applied first, as `manifest.rs:902-903` does).

### WR-14: `f1_average_for_classes` accepts duplicate class indices and silently double-counts

**File:** `crates/aprender-train/src/eval/classification/metrics.rs:6-14`

**Issue:**

```rust
pub fn f1_average_for_classes(f1: &[f64], classes: &[usize]) -> Option<f64> {
    if classes.is_empty() || classes.iter().any(|&class| class >= f1.len()) {
        return None;
    }
    let total = classes.iter().map(|&class| f1[class]).sum::<f64>();
    Some(total / classes.len() as f64)
}
```

`f1_avg_for_classes(&[1, 1])` returns `f1[1]` rather than `None`. The doc says the function
"prevent[s] a silently mislabelled benchmark score", and a duplicated index is precisely a
caller error that produces a plausible-looking but wrong score — e.g. `&[1, 1, 2]` weights
`against` twice and reports something that is not the official TweetEval F_avg. The existing
guard already establishes that this function is meant to reject malformed selections;
duplicates are the one malformed case it lets through.

`F_AVG_CLASSES` is a constant `[1, 2]`, so no current caller trips this — but the function
is `pub` and re-exported from `aprender_train::eval::classification`.

**Fix:**

```rust
if classes.is_empty() || classes.iter().any(|&class| class >= f1.len()) {
    return None;
}
// A repeated index would weight one class twice and report a score that is not the
// average it claims to be.
let mut seen = classes.to_vec();
seen.sort_unstable();
seen.dedup();
if seen.len() != classes.len() {
    return None;
}
```
with a matching case in `basic_tests.rs::test_f1_average_for_explicit_classes_rejects_invalid_input`.

---

_Reviewed: 2026-08-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
