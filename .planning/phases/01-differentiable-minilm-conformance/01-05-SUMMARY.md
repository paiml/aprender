---
phase: 01-differentiable-minilm-conformance
plan: 05
subsystem: setfit
tags: [setfit, minilm, tokenizer, import, enc-01, enc-02, d-05, d-06, d-08, a-01, features, supply-chain]
requires:
  - fixture:setfit-slice-apr
  - fixture:setfit-tokenizer-cases
  - fixture:setfit-manifest-sha256
  - contract:setfit-encoder-conformance-v1
provides:
  - feature:setfit
  - feature:conformance-fixtures
  - type:SetFitError
  - type:SentenceBatch
  - type:MiniLmTokenizer
  - type:MiniLmImport
  - type:VocabRemap
  - type:SliceConfig
  - type:ModelDims
  - amendment:A-01-read-tensor-pub-crate
affects:
  - Cargo.toml
  - crates/aprender-core/Cargo.toml
  - crates/aprender-core/src/lib.rs
  - crates/aprender-core/src/models/bert/load.rs
  - crates/aprender-core/src/setfit/
tech-stack:
  added:
    - "tokenizers 0.23.1 (workspace pin, default-features = false, features = [\"fancy-regex\"])"
  patterns:
    - "structural seal: pub(crate) constructors + pub(crate) fields with read-only accessors, so a mismatched pair is not constructible rather than merely detected"
    - "config pin validates every behavior-affecting field and tolerates unmodelled metadata (no deny_unknown_fields)"
    - "pinned artifacts embedded verbatim in tests and digest-checked against the frozen upstream manifest"
    - "mutation matrix: one field per test, each asserting the offending field's own name appears in the typed error"
key-files:
  created:
    - crates/aprender-core/src/setfit/mod.rs
    - crates/aprender-core/src/setfit/error.rs
    - crates/aprender-core/src/setfit/tokenizer.rs
    - crates/aprender-core/src/setfit/tokenizer_tests.rs
    - crates/aprender-core/src/setfit/import.rs
    - crates/aprender-core/src/setfit/import_tests.rs
  modified:
    - Cargo.toml
    - crates/aprender-core/Cargo.toml
    - crates/aprender-core/src/lib.rs
    - crates/aprender-core/src/models/bert/load.rs
    - Cargo.lock
decisions:
  - "truncation facts need a SECOND pass: measured, the plan's `ids.len() + sum(overflowing.len())` reports 512 for the frozen 482-token case because each overflow chunk gets its own [CLS]/[SEP] and is then padded"
  - "`sha2 = [\"dep:sha2\"]` declared explicitly — writing `dep:sha2` anywhere stops cargo synthesising the implicit feature that format-encryption and hf-hub-integration referenced by bare name"
  - "sentence_bert_config.json validated WHEN PRESENT: it is outside 01-04's frozen upstream file set, so requiring it would reject the D-10 checkout; absence cannot change behaviour because the 256 bound lives in this crate"
  - "`open()` also exercised end-to-end against the real 86.7 MB checkout, because every other pin test proves a rejection and would leave acceptance an inference"
  - "`#![allow(dead_code)]` scoped to import.rs: the D-08 seal makes every constructor pub(crate) and its in-crate callers land in 01-06/01-07"
metrics:
  duration: ~2h
  completed: 2026-08-08
  tasks: 3
  commits: 5
  tests_added: 64
requirements: [ENC-01, ENC-02]
---

# Phase 1 Plan 05: SetFit Feature, Tokenizer Boundary and Pinned Import Summary

The `setfit` feature and module: a dependency-closed `tokenizers` 0.23.1 pin with no C/C++
toolchain, exact ENC-02 tokenizer parity over the whole frozen corpus of record, and an
ENC-01 pin that rejects twenty-three distinct behavior-affecting mutations by name while
still accepting the real pinned model — proven against the real 86.7 MB checkout, not
inferred from rejections.

## What Was Built

| File | Role |
|------|------|
| `setfit/mod.rs` | Module docs, the visibility rule, type-only re-exports |
| `setfit/error.rs` | `SetFitError`, 13 variants incl. `Op(OpError)` and `From` impls |
| `setfit/tokenizer.rs` | `MiniLmTokenizer` + read-only `SentenceBatch` (ENC-02) |
| `setfit/import.rs` | `MiniLmImport`, `VocabRemap`, `SliceConfig`, `ModelDims` (ENC-01) |

Plus the workspace `tokenizers` pin, the `setfit` / `conformance-fixtures` features, and the
sanctioned A-01 one-line visibility change.

## The Three Things That Could Have Been Faked, and Were Not

**1. The dependency pin is proven off the C toolchain, not declared off it.**
`tokenizers`' default feature set is `[progressbar, onig, esaxx_fast]`, which pulls the
`onig` C library and the `esaxx-rs/cpp` C++ build script. `cargo tree -f "{p} {f}"` reports
`tokenizers v0.23.1 fancy-regex` — exactly one feature — and `esaxx-rs v0.1.10` with **no
features** (so its `cpp` build path is off). `onig` is absent from the tree entirely.
`cargo tree -p aprender-core --no-default-features -e normal` contains no `tokenizers` node
at all, closing D-06.

**2. The pinned artifacts in the tests are the pinned bytes.**
`config.json`, `modules.json` and `1_Pooling/config.json` are embedded verbatim in
`import_tests.rs` and every one of their sha256 digests is asserted against
`tests/fixtures/setfit/upstream_manifest.json`. `PINNED_REVISION` and
`PINNED_TOKENIZER_SHA256` are asserted against the same file. So the pin in this plan cannot
drift from the artifact set 01-04 froze, and a typo in the embedded config fails loudly
instead of quietly weakening all twenty-three mutations derived from it.

**3. Acceptance is measured, not inferred.**
Every other pin test proves a *rejection*, or stops at the missing-weights boundary. That
leaves "the pin accepts the pinned model" as an assumption — the exact shape of a gate that
looks strong and admits nothing. `open()` is therefore also driven end to end against the
real 86.7 MB `full_model.apr` in `~/.cache/aprender/minilm-l6-v2-1110a243`: 37 tensor reads
plus a full finiteness scan over 22M parameters. It **ran** — the suite printed no `SKIP`
line, and pointing `APRENDER_MINILM_DIR` at a nonexistent path *does* print one, so the pass
is not a silent skip (CLAUDE.md verification rule 2).

## D-08 Delivered Structurally, at Both Layers

| Layer | Mechanism | Verified by |
|-------|-----------|-------------|
| Constructor | `from_bytes`, `open`, `open_slice_fixture`, both `from_json_bytes` are `pub(crate)` | source grep returns no bare `pub fn` on any of them in production files |
| Data | every `SentenceBatch` field is `pub(crate)` with read-only accessors | `grep -nE '^[^/]*\bpub (input_ids\|…)\b'` returns nothing |
| Defense in depth | batch stamps its producing tokenizer's sha256 | `sentence_batch_carries_the_producing_tokenizer_identity` |

The seal regexes were proven by a must-match / must-not-match case table rather than by
re-reading them (CLAUDE.md rule 7). Must-match: `pub input_ids:`, `pub seq:`,
`pub tokenizer_sha256:`, `pub fn from_bytes`. Must-not-match: the `pub(crate)` forms, the
`pub fn input_ids(&self)` accessors, and doc/plain comments that mention the old form.
All eleven cases behaved as specified.

`#[non_exhaustive]` was deliberately not added: it is redundant once fields are `pub(crate)`
and is a no-op inside the defining crate.

## ENC-02: Tokenizer Parity

12 tests. The parity test iterates **every** case in `tokenizer_cases.json` (12 today) with
exact integer equality on `input_ids`, `token_type_ids`, `attention_mask`, `batch`, `seq`
and both truncation facts. No case list is hardcoded, so a fixture text added later widens
this gate automatically. No tolerance literal exists in the file — tokenizer parity has no
epsilon.

Canonical ids stay canonical: a dedicated test asserts the emitted ids include values above
97 (the slice vocabulary size), so a future change that silently remapped the batch to slice
rows would fail here rather than in wave 6.

## The Truncation-Fact Strategy Was Decided by Measurement

The plan preferred a single pass: `original_len = ids.len() + sum(overflowing.len())`, with
an explicit instruction to make the alternative honest rather than hide a second pass.
Probed against the frozen `truncation_long` case (482 real tokens) with the pinned
tokenizers 0.23.1:

```
ids=256  overflow_chunks=1  chunk0 len=256   naive_sum=512   fixture_expected=482
untruncated pass len=482
```

The overflow chunk receives its own `[CLS]`/`[SEP]` from post-processing and is then padded
to the batch-longest width, so the naive sum is off by exactly the specials-and-padding of
each chunk. Reporting 512 would have made ENC-02's `original_len` quietly wrong precisely on
the inputs it exists to describe.

**Resolution, declared not hidden:** the batch is tokenized once; rows that were *actually*
cut are re-tokenized once more without truncation. The extra pass is empty for every
non-truncating input. The module docs state ENC-02's guarantee as "one user-facing call; two
internal passes when a row is truncated", not "one tokenizer pass".

## ENC-01: The Rejection Matrix

25 `import_pin_rejects_*` tests, of which **23 are behaviour mutations** (two are I/O shape:
missing / unparseable `config.json`). Each differs from the pin in exactly one respect and
each asserts the offending field's own name appears in the typed error — a test that only
checked `is_err()` would pass just as happily if the loader rejected for an unrelated reason.

| Group | Fields |
|-------|--------|
| Dimensions | `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, `type_vocab_size` |
| Numerics | `layer_norm_eps`, `pad_token_id`, `hidden_dropout_prob`, `attention_probs_dropout_prob` |
| Semantics | `hidden_act` (`gelu_new`, `gelu_pytorch_tanh`, `relu` — three separate tests), `position_embedding_type`, `architectures`, `model_type` |
| Sentence-transformers | `pooling_mode_mean_tokens`, `pooling_mode_max_tokens`, the trailing `Normalize` module, `max_seq_length` |
| Provenance | one-byte-flipped `tokenizer.json`, slice dimensions handed to the PUBLIC pin path |

Two acceptance tests hold the other side of the line: the real config with its six
unmodelled fields is accepted, and a brand-new unknown field is accepted. `deny_unknown_fields`
is absent, and a source assertion enforces that — written comment-safe, because the module
docs deliberately *discuss* the attribute and a naive `contains` flagged the explanation of
why it is absent.

`layer_norm_eps` is compared at `f32`, on purpose: `1e-12f32` and `1e-12f64` are different
numbers, so comparing the parsed `f64` against `BertConfig::minilm_l6()` in `f64` would have
rejected the pinned config's own value.

The slice constructor bypasses **only** equality with the pinned dimensions. It still
enforces the exact-erf activation, the source revision, the tokenizer digest, head geometry
(`hidden == heads * head_dim`), non-zero dims, remap validity, tensor presence, tensor shape
and finiteness — 16 `import_slice_*` tests, including a clean-synthetic-APR control so the
three corruption tests are known to fail for their stated reason rather than because the
synthetic builder is unreadable.

## Verification

Every exit code below was captured with `cmd > file 2>&1; rc=$?` — never through a pipe and
never from a trailing `echo` (CLAUDE.md rule 1).

| Check | Result |
|-------|--------|
| `cargo check -p aprender-core --no-default-features` | 0 |
| `cargo check -p aprender-core --features setfit` | 0 |
| `cargo check -p aprender-core --features conformance-fixtures` | 0 |
| `cargo check -p aprender-core --features setfit,conformance-fixtures,format-encryption,hf-hub-integration` | 0 (proves the sha2 refactor) |
| `cargo check --workspace --exclude aprender-profile` (D5) | 0 |
| `cargo tree --no-default-features -e normal \| tokenizers` | absent (D-06) |
| `cargo tree --features setfit -f "{p} {f}"` | `tokenizers v0.23.1 fancy-regex`; `esaxx-rs` no features; no `onig` |
| `cargo test -p aprender-core --lib --features conformance-fixtures tokenizer_parity` | 0 — 4 passed |
| `… sentence_batch` | 0 — 8 passed |
| `… import_pin` | 0 — 32 passed |
| `… import_slice` | 0 — 16 passed |
| full lib, default features | 0 — 14063 passed |
| full lib, `--features conformance-fixtures` | 0 — 14127 passed (+64) |
| `cargo clippy -p aprender-core --lib --tests --features conformance-fixtures` (D2 form) | 0 findings under `src/setfit/` |
| `rustfmt --check` on all six setfit files | 0 |
| `cargo package -p aprender-core --list` | all 6 setfit sources + all 16 fixtures present |
| `git check-ignore -v` on all 6 new sources | 1 (none ignored) |
| `git diff --numstat` on `src/models/bert/` | `2 1` — the A-01 line plus its comment, nothing else |
| real-checkout `open()` (86.7 MB, 37 tensors) | 0 — ran, no SKIP; skip branch proven reachable |

`scripts/check_include_files.sh` was **not** relied on (D1: vacuous on macOS). These files
use ordinary `mod` / `#[path]` declarations rather than `include!()`, so the packaging and
tracking were verified directly by `cargo package --list`, `git ls-files` and
`git check-ignore`.

## A-01 Discharged

`crates/aprender-core/src/models/bert/load.rs` line 114: `fn read_tensor(` →
`pub(crate) fn read_tensor(`, plus a one-line comment recording the amendment. The whole
diff against the plan base for `crates/aprender-core/src/models/bert/` is **2 insertions,
1 deletion**. No facade was built and the read logic was not duplicated — `import.rs` calls
`read_tensor` directly, so the checked-read semantics (presence, dtype path, element count)
cannot drift from the loader they came from.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `dep:sha2` broke two unrelated features at manifest-parse time**

- **Found during:** Task 1, on the very first `cargo check --features setfit`.
- **Issue:** cargo synthesises an implicit feature per optional dependency **only while no
  feature uses `dep:` for it**. `format-encryption` and `hf-hub-integration` referenced
  `sha2` by bare name. Writing `setfit = ["dep:tokenizers", "dep:sha2"]` therefore made the
  whole workspace fail to load:
  `feature 'format-encryption' includes 'sha2', but 'sha2' is an optional dependency without
  an implicit feature`.
- **Fix:** declared `sha2 = ["dep:sha2"]` explicitly. This is the minimum-surface repair —
  it keeps the published feature surface byte-identical (an external consumer that enabled
  `aprender-core/sha2` still can) while letting `setfit` use the unambiguous `dep:` form.
  Rewriting the two existing features to `dep:sha2` would have silently removed a public
  feature name instead.
- **Files modified:** `crates/aprender-core/Cargo.toml`
- **Commit:** `6e555f93c`
- **Generalisation logged as D23:** every other optional dependency in that manifest is
  still referenced by bare name, so the next plan reaching for `dep:` meets the same wall.

**2. [Rule 1 - Bug] `original_len` reported the padded width for non-truncated rows**

- **Found during:** Task 2 GREEN, caught by the parity gate.
- **Issue:** `original_len` was read from `Encoding::get_ids().len()`, which with
  `BatchLongest` padding is the *padded* width. The frozen `mixed_length_pair` case reported
  20 for its 5-token first row.
- **Fix:** for rows that were not truncated, `original_len` is the attention-mask population
  count — the true token count including specials. Truncated rows use the untruncated pass.
- **Files modified:** `crates/aprender-core/src/setfit/tokenizer.rs`
- **Commit:** `254a7c966`

### Deliberate Departures

**3. Truncation facts need two internal passes (the plan's sanctioned fallback)**

Covered in full above. The plan explicitly authorised this path *provided* the finding was
measured and the guarantee renamed rather than the pass hidden. Both were done: the probe
output is quoted in the commit message and the module docs, and the docs say "one
user-facing call", not "one tokenizer pass".

**4. `sentence_bert_config.json` is validated when present, not required**

01-04's `upstream_manifest.json` pins five upstream files and `sentence_bert_config.json` is
not among them, so `fetch_full_weights.py` cannot produce it. Requiring it would reject the
very checkout the D-10 gated suite materialises — the "the loader is broken at wave 7"
failure that script's own docstring warns about. Its absence cannot change behaviour: the
256-token bound is applied from this crate's `MAX_SEQUENCE_LENGTH` constant and is never
read from the checkout. A **present-but-different** value is a real disagreement and is
rejected, proven by `import_pin_rejects_a_mutated_sentence_transformer_max_seq_length`.
Logged as D20 with the fixture-regeneration fix direction.

**5. Two extra tests beyond the plan's list, and one extra `#[allow]`**

`import_pin_opens_the_real_pinned_checkout_when_materialised` (the acceptance measurement
above) and `import_slice_accepts_a_synthetic_but_well_formed_apr` (the control for the three
corruption tests) are additions, both Rule 2: without them two claims in this plan would
have been inferences. `#![allow(dead_code)]` is scoped to `import.rs` with its reason
recorded in the source — the D-08 seal makes every constructor `pub(crate)` and its in-crate
callers (01-06's encoder, 01-07's `SetFitMiniLm`) do not exist yet, so a library-only build
reports ~15 unreachable-item findings that describe no defect. Widening visibility to silence
them would break the seal. Measured before and after: the same clippy invocation reported 12
`src/setfit/` findings before the allow and 0 after.

**6. `SetFitError` Display for two variants now names the config key**

`UnsupportedActivation` and `UnsupportedArchitecture` originally printed only the offending
value. `"gelu_new"` without `hidden_act` is not actionable, and the ENC-01 acceptance
criterion requires the mutated field's name in the error. Both Display impls now name the key
they read. `UnsupportedPooling` carries the key inside its `got` string for the same reason.

## Known Stubs

None. Every function in the module is implemented; no placeholder, hardcoded-empty value, or
"coming soon" path exists. The two RED commits contained deliberate stubs and both were
replaced by their GREEN commit in the same task.

## TDD Gate Compliance

Tasks 2 and 3 are `tdd="true"` and both show the full RED → GREEN sequence in git:

| Gate | Task 2 | Task 3 |
|------|--------|--------|
| RED (`test(...)`) | `78377b372` — 7 of 8 `sentence_batch` and 4 of 4 `tokenizer_parity` fail | `8c8e4e17f` — 42 of 47 fail |
| GREEN (`feat(...)`) | `254a7c966` — 12 pass | `be1042327` — 48 pass |

RED honesty was checked rather than assumed. In Task 3's first RED run, three remap tests
passed *vacuously* against a blanket-rejecting stub; they were strengthened to name the
offending direction and arity so they now fail for the right reason. The tests that do pass
in RED are data and source assertions — digest agreement with the frozen manifest, agreement
between the embedded config and `BertConfig::minilm_l6()`, and the D-08 seal — none of which
assert unimplemented behaviour.

No REFACTOR commit was needed; neither GREEN implementation required cleanup that changed
behaviour.

## Threat Flags

None new. The plan's register is discharged:

- **T-1-09** — 23 behaviour mutations rejected by name; per-tensor presence / element-count /
  finiteness checks before use; unknown metadata tolerated so the gate cannot fail on the
  correct artifact (proven by two acceptance tests).
- **T-1-10** — revision recorded as an immutable sha constant, asserted equal to the frozen
  manifest. No branch-name resolution exists anywhere in the module.
- **T-1-11** — empty input, oversize input, malformed tokenizer bytes, unparseable config and
  missing files all return typed errors. No `assert!`, no `unwrap()`, no panic on any
  reachable path.
- **T-1-21** — structural seal at the constructor layer AND the data layer, with the sha256
  stamp retained as defense in depth.
- **T-1-22** — `VocabRemap` validated for arity, range and mutual inverse in **both**
  directions at construction; `to_slice_row` returns `VocabOutOfSlice`, never a zero-fill.
- **T-1-SC** — `tokenizers 0.23.1` confirmed on crates.io before use; `default-features = false`
  proven to keep `onig` out of the tree and `esaxx-rs/cpp` off.

## For the Next Plans

- **01-06** consumes `SentenceBatch` (fields are `pub(crate)`, so read them directly or via
  accessors), `MiniLmImport::{dims, layer_norm_eps, reader, tensor_prefix, vocab_remap}`, and
  `VocabRemap::to_slice_row`. `SetFitError::Op(OpError)` with `From<OpError>` exists so
  `forward_tokens`/`encode` can compose the ungated ops with `?`.
- **01-06 should delete `#![allow(dead_code)]` from `import.rs`** once the encoder calls
  `MiniLmImport` — that is the condition the allow was written against.
- **01-07**'s `SetFitMiniLm::{from_pretrained_dir, from_slice_fixture}` are the public
  entry points; they must build the tokenizer and the import from the same source. Every
  constructor they need is `pub(crate)`, which is sufficient because they are in-crate.
- The full-pin path reads `full_model.apr`, falling back to `model.apr`.

## For the Orchestrator

- `STATE.md`, `ROADMAP.md` and `REQUIREMENTS.md` were **not** touched (worktree mode).
- ENC-01 and ENC-02 are implemented and proven here, but they are gated end-to-end by 01-08's
  conformance suite; the orchestrator owns whether to mark them complete.
- New deferred items appended as **D20-D23**. Known items D1, D2, D5, D8 and D9 were used as
  documented and not re-logged.

## Self-Check: PASSED

All 6 created files verified present on disk and tracked by `git ls-files`. All 5 commits
verified as commit objects (`git cat-file -t`) and ancestors of HEAD
(`git merge-base --is-ancestor`) — not by grepping reformatted `git log` output, which is the
failure 01-04 recorded. Working tree clean apart from this SUMMARY and `deferred-items.md`.
