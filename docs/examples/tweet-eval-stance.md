# TweetEval abortion stance benchmark

This benchmark is aprender's reference social-media classification dataset for
comparing lightweight sentence-encoder classifiers, SetFit-style contrastive
fine-tuning, and larger LoRA classifiers.

The default command downloads a pinned revision of canonical TweetEval and
converts it to aprender JSONL. The original tweet text is not stored in this
repository.

```bash
apr data tweet-eval-stance --output data/tweet-eval-stance
```

The generated canonical layout is:

```text
data/tweet-eval-stance/
├── train.jsonl               # 587 samples
├── validation.jsonl          # 66 samples
├── test.jsonl                # 280 samples
└── benchmark-manifest.json   # labels, hashes, provenance, protocol
```

Each JSONL row can be loaded directly by the classification fine-tuning
pipeline:

```json
{"id":"train:0","input":"...","label":1,"label_text":"against","source_split":"train"}
```

The fixed label mapping is `0=none`, `1=against`, and `2=favor`.

## Offline or mirrored preparation

Supply a directory containing the canonical `train_text.txt`,
`train_labels.txt`, `val_text.txt`, `val_labels.txt`, `test_text.txt`, and
`test_labels.txt` files:

```bash
apr --offline data tweet-eval-stance \
  --source /datasets/tweeteval/datasets/stance/abortion \
  --output data/tweet-eval-stance
```

The command verifies the exact canonical split and class counts before writing
anything. Existing benchmark outputs are not replaced unless `--force` is
provided.

When files come from `--source`, the `--revision` recorded in the manifest is
your assertion, not a verified fact — the command cannot prove a local
directory came from that commit. The manifest records this explicitly as
`source.revision_verified: false`; it is `true` only for runs that downloaded
the pinned revision themselves.

## SetFit compatibility layout

For comparison with the two-split SetFit Hugging Face wrapper:

```bash
apr data tweet-eval-stance \
  --profile setfit \
  --output data/tweet-eval-stance-setfit
```

This emits 587 training rows and a 346-row test file made by concatenating the
canonical validation and test sets. It is a compatibility mode, not the
recommended model-selection protocol.

## 9B classifier baseline

Use an actual pretrained model artifact rather than `--model-size` alone:

```bash
apr finetune qwen-9b.apr \
  --task classify \
  --num-classes 3 \
  --data data/tweet-eval-stance/train.jsonl \
  --output checkpoints/tweet-eval-qwen-9b
```

Select checkpoints using `validation.jsonl`, then evaluate the fixed test set:

```bash
apr eval checkpoints/tweet-eval-qwen-9b \
  --task classify \
  --dataset tweet-eval-stance \
  --num-classes 3 \
  --model-size 9B \
  --data data/tweet-eval-stance/test.jsonl
```

The benchmark's primary metric is the official stance score:

```text
F_avg = (F1_against + F1_favor) / 2
```

The evaluation report also includes three-class macro-F1, per-class metrics,
MCC, calibration, confidence intervals, and confusion diagnostics. Accuracy is
supplemental because the test split is dominated by the `against` class.

For few-shot comparisons, the manifest fixes balanced 8, 16, 32, and 64
examples per class and ten deterministic seeds. All classifier variants must
use the same sampled IDs and must not use the canonical test split for tuning.
The next two sections are how you produce those sampled IDs.

## Few-shot selection and contrastive pairs

`apr data select` turns a prepared benchmark directory into a replayable
few-shot selection; `apr data pairs` turns that selection into a bounded
contrastive pair stream. Both are thin filesystem adapters over the
`aprender-contrastive-data` crate: they read bytes, call the crate, and write
the bytes the crate hands back.

### Running these from a checkout

Every transcript below was produced by the pinned binary, never a bare `apr`:

```bash
. scripts/apr_bin.sh || exit 1   # exports $APR, proves it was built from HEAD
```

A bare `apr` runs whatever `PATH` resolves, which on this project's dev box has
meant a 26-day-old build validating a gate merged the day before. If you
installed with `cargo install aprender`, substitute `apr` for `"$APR"`.

The only edit made to the transcripts is the output directory, shortened to
`data/tweet-eval-stance`. Nothing else is elided: the hashes are a function of
the pinned dataset content and the seed, not of where you put the files, so
they reproduce for anyone running the same commands at the same revision.

### Select

```bash
"$APR" data select --data data/tweet-eval-stance --shots 8 --seed 13
```

```text
=== Few-Shot Selection ===

  Data: data/tweet-eval-stance
  Profile: canonical
  Shots per class: 8
  Root seed: 13 (contracted)
  Selected: 24 example(s) across 3 class(es)
  Semantic hash: 1d2dbbb37edea7aba85b251860ee9c92734bd3b09fa5f4d571bed62355fb7ad0
  Ledger hash: a44f6d821f4e276ed1d803c129223688abf48774fe31e887cdc0cf52c7476116
  Excluded: 1 training row(s) removed from the pool before selection
  Manifest: data/tweet-eval-stance/selection-manifest.json
```

That writes `selection-manifest.json` beside the splits (use `-o/--output` to
put it elsewhere). The manifest is the artifact every later stage consumes. Its
payload carries the ordered selected examples **with their labels** and both
content hashes, the dataset and validation fingerprints, the exclusion record,
the resolved seed and shot count, and the persisted access ledger. Add `--json`
to get the same content on stdout:

```bash
"$APR" --json data select --data data/tweet-eval-stance --shots 8 --seed 13 --force
```

```json
{
  "command": "data-select",
  "profile": "canonical",
  "root_seed": 13,
  "seed_mode": "contracted",
  "shots_per_class": 8,
  "selected": 24,
  "semantic_hash": "1d2dbbb37edea7aba85b251860ee9c92734bd3b09fa5f4d571bed62355fb7ad0",
  "ledger_hash": "a44f6d821f4e276ed1d803c129223688abf48774fe31e887cdc0cf52c7476116",
  "ordered_examples": [
    {
      "id": "train:281",
      "label": 0,
      "exact_hash": "989279f86b18288af807f3d3e52670edee916380c4ff210535c25c57f463abbc",
      "normalized_hash": "3fba44b4ac096d08d416223d397e093575ccef78f2326cbacab49ef4323d61c8"
    }
  ]
}
```

(`ordered_examples` is truncated to one row here; the real output has all 24,
plus `exclusions` and `access_ledger`.) On the pinned revision the exclusion
record reports `excluded_train_ids: ["train:70"]` and reduced pools of 158, 319
and 109 — `train:70` duplicates `validation:3`, so it is removed from the
selection pool and can never be selected or paired.

### The seed policy

`--seed` is **required and has no default.** It must be one of the ten
contracted benchmark seeds — 13, 17, 23, 29, 31, 37, 41, 43, 47, 53 — and 42 is
deliberately not among them, which is exactly why there is no default to fall
back on. All ten produce ten distinct selections, and each replays to the same
semantic hash every time.

```bash
"$APR" data select --data data/tweet-eval-stance --shots 8 --seed 42
```

```text
error: Validation failed: --seed 42 is not one of the ten contracted benchmark seeds
[13, 17, 23, 29, 31, 37, 41, 43, 47, 53]. Pass --any-seed to draw an experimental
selection anyway; the seed itself is recorded in the manifest, so a reader can
always tell which of the two you did
```

`--any-seed` is the documented escape hatch for experimentation. It costs you
comparability: a selection drawn on an uncontracted seed is not a benchmark
cell, and `--json` reports `"seed_mode": "uncontracted"` so nobody can mistake
it for one. The mode is derived from the recorded `root_seed` rather than
stored beside it, so it cannot disagree with the seed it describes.

`--shots` is likewise restricted to the contracted sizes:

```bash
"$APR" data select --data data/tweet-eval-stance --shots 9 --seed 13
```

```text
error: Validation failed: --shots 9 is not a contracted few-shot size; expected one of {8, 16, 32, 64}
```

Both checks run before anything is read, so a bad request is never reported as
a problem with your data.

### Pairs

```bash
"$APR" data pairs \
  --selection data/tweet-eval-stance/selection-manifest.json \
  --data data/tweet-eval-stance
```

```text
=== Contrastive Pair Stream ===

  Selection: data/tweet-eval-stance/selection-manifest.json
  Data: data/tweet-eval-stance
  Selection hash: 1d2dbbb37edea7aba85b251860ee9c92734bd3b09fa5f4d571bed62355fb7ad0
  Pair manifest hash: 5f4d4bb6cb9a3e49e4ad30869e1eb0b27679603658e9cd54a98c8bb29e74b524
  Root seed: 13
  Strategy: oversampling (v1)
  Budget: 384 pair(s) per epoch, hard cap 1048576
  Emitted kinds: both
  Counts: 192 positive, 192 negative
  Singleton policy: negatives_only (v1), 0 affected class(es)
  Degenerate policy: v1
```

There is deliberately **no `--seed` on `apr data pairs`.** The root seed is part
of the selection manifest; re-supplying it here would create two sources of
truth for one replay tuple. The manifest is replayed strictly — profile,
fingerprints, exclusions, membership, both per-row hashes, class balance,
ordering, versions and the envelope digest are all revalidated, and the ordered
list is then recomputed from the seed. A hand-edited manifest that is
internally consistent in every other way still fails that last step.

### Pairs are replayed, not stored

Only the ~200-byte replay tuple and the hash above are persisted. Forty
benchmark cells times ten seeds times many epochs of pair bytes is gigabytes of
derivable data, so the pairs are regenerated on demand. `--dump` is the
explicit audit and fixture path:

```bash
"$APR" data pairs \
  --selection data/tweet-eval-stance/selection-manifest.json \
  --data data/tweet-eval-stance \
  --budget 256 --dump data/audit.jsonl
```

```text
{"lo":"train:318","hi":"train:5","target":1.0}
{"lo":"train:5","hi":"train:427","target":0.0}
```

One JSON line per pair, in stream order. Auditing that file against the
selection confirms what the protocol promises: every endpoint is a selected
*training* id, no endpoint is a validation or test id, no pair is a self-pair,
`train:70` never appears, and each `target` agrees with class identity.

### Budget and hard cap

Omitting `--budget` gives the contracted default: the closed-form oversampling
count, clamped by `--hard-cap`. The default grows with the example count (384
pairs at 8 shots, 24576 at 64), while an explicit `--budget` does not — 8 shots
and 64 shots under `--budget 256` both emit exactly 256 pairs, 128 of each
kind, and the dump is 256 lines either way. That is the `O(examples + budget)`
guarantee at the user surface.

A `--budget` above `--hard-cap` is an **error, not a silent clamp**:

```bash
"$APR" data pairs --selection data/tweet-eval-stance/selection-manifest.json \
  --data data/tweet-eval-stance --budget 500 --hard-cap 100
```

```text
error: Validation failed: contrastive data: requested pair budget 500 exceeds hard_cap 100
— raise --hard-cap or lower --budget. The cap BINDS an explicit budget rather than
clamping it, so that a run can never quietly emit fewer pairs than asked for
```

A clamp would let a run quietly emit fewer pairs than it was asked for and
still report success. `--budget 0` and `--hard-cap 0` are separately typed
errors: a zero budget is a defect in the request, a zero cap is a defect in the
configuration.

### Everything is written atomically

Both commands write through one helper: a temp file in the destination
directory, `write_all`, `fsync`, then `rename`. Files are never clobbered
without `--force`, and an interrupted write leaves neither a partial artifact
nor a stray temp file.

```bash
"$APR" data select --data data/tweet-eval-stance --shots 8 --seed 17
```

```text
error: Validation failed: Refusing to replace existing file
data/tweet-eval-stance/selection-manifest.json (pass --force to replace it)
```

### Split isolation is fail-closed

Selection reads canonical splits only, through the crate's attested boundary:
the label map, per-class counts, per-split digests, exclusion digest and
dataset fingerprint in `benchmark-manifest.json` are all re-derived and
compared before any row is exposed. Four directories that look fine and are
not:

```text
# a SetFit compatibility directory (validation and test merged)
error: Validation failed: contrastive data: dataset profile mismatch: expected "canonical", got "compatibility"

# a directory assembled from two different preparations
error: Validation failed: contrastive data: validation split hash mismatch: expected 1ac2119f…, got c7f5921a…

# an edited attestation
error: Validation failed: contrastive data: dataset fingerprint mismatch: expected 00000000…, got 082ad5b0…

# a stale directory
error: Validation failed: benchmark-manifest.json declares schema_version 1, but this
build supports [2]. There is no migration: a version-1 manifest predates the cross-split
exclusion record, so any value synthesized for it would be an unattested guess.
Re-prepare the benchmark with `apr data tweet-eval-stance --output <DIR> --force`.
```

The compatibility layout is refused on both commands, not just on `select`.

### Schema version policy

`benchmark-manifest.json` must be `schema_version: 2`. There is **no migration
from version 1** and there will not be one: a version-1 manifest predates the
cross-split exclusion record, so any value synthesized for it would be an
unattested guess wearing an attestation's clothes. Re-prepare with `--force`.

## Test suite

The normal test suite uses authored fixtures and makes no network requests. To
check the pinned upstream files explicitly, run the ignored integration test:

```bash
cargo test -p apr-cli --lib pinned_upstream_satisfies_the_dataset_contract -- --ignored
```

```text
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 6688 filtered out
```

`--lib` is load-bearing. Without it the same filter runs every target in the
package and prints **65** `test result: ok` lines, 64 of which matched nothing —
so a reader grepping for `ok` would be satisfied by a suite that ran no test at
all. Read the passed count, never the word `ok`.
