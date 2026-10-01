# Phase 1 — Deferred Items

Out-of-scope discoveries logged during plan execution. **Not fixed** — they are
pre-existing and unrelated to the changes that surfaced them.

Plans 01-01 and 01-02 executed in parallel and independently surfaced D1 and D2.
Two agents reaching the same finding from different code paths raises confidence
that these are real and reproducible, not artifacts of one agent's environment.

Numbering note: plans 01-04 and 01-09 also ran in parallel and both allocated
"D7"/"D8" without knowledge of each other. 01-09's two items were renumbered to
D10/D11 by the orchestrator at merge time; their content is unchanged. IDs here
are unique across the phase — cite them as such.

## From plan 01-01 (2026-08-08)

### D1. `scripts/check_include_files.sh` is a no-op on macOS

*Independently confirmed by plan 01-02.*

The script uses `grep -P` / `grep -oP` (PCRE), which BSD grep rejects:

```
grep: invalid option -- P
OK: All 0 include!() files are tracked by git
```

It then reports success over **zero** files, exiting 0. CB-510 exists because a
gitignore pattern silently hid `include!()` sources from git and crates.io; on
macOS this guard cannot detect that recurrence — it is theater there. It
presumably works on the CI Linux runner, so the drift is platform-split rather
than total. CLAUDE.md documents the repo as having 562 `include!()` files; the
guard sees 0 of them on darwin.

Discovered while adding four new `include!()` files under
`crates/aprender-core/src/autograd/ops/`. Those were verified by hand instead
(`git check-ignore -v` exits 1 for each; `git ls-files` lists all four).

Fix direction: `grep -Eo` with a POSIX-ERE equivalent, or `rg` if it is an
accepted dependency. Whichever is chosen, re-run the must-match / must-not-match
case table on macOS **and** Linux (CLAUDE.md verification rule 7) — the pattern
is exactly the kind that has been wrong five times before.

### D2. `cargo clippy -p aprender-core -- -D warnings` fails on `aprender-compute`

*Independently confirmed by plan 01-02.*

`aprender-compute` is a workspace path dependency, so command-line `-D warnings`
applies to it too. It carries ~20 pre-existing findings (unreachable
expressions, unused imports/constants, unused variables, dead functions in
cfg-gated NEON/AVX paths inactive on this target), so the invocation named in
the 01-01 and 01-02 plans' `<verification>` blocks exits 101 regardless of the
state of `aprender-core`.

`aprender-core` itself is clean: `cargo clippy -p aprender-core --lib --tests`
exits 0 and reports zero findings in the touched files of either plan.

Fix direction: clean `aprender-compute`, or scope the phase gate to
`--lib --tests` on the crate under change. Do NOT paper over it with a
workspace-level allow — that would hide future regressions in compute kernels.

### D3. `generated_contracts.rs` had drifted badly from `contracts/`

Re-running the sanctioned regeneration command
(`pv codegen contracts/ -o crates/aprender-core/src/generated_contracts.rs`)
produced a ~31k-line diff **before** this plan's contract was accounted for:
2415 macros at `HEAD` versus 2903 after regeneration, with 18 macros dropped
entirely. So `pv codegen` had not been re-run after a long run of contract
edits.

No consumer was broken — all 18 dropped macros were verified to have zero call
sites, `cargo check -p aprender-core` exits 0, and the full lib suite is green
(13,972 passing before this plan's tests were added). But nothing in the repo
detects the drift.

Fix direction: a tier3/CI check that regenerates into a temp file and fails on
any diff, so "the generated file matches the contracts" becomes falsifiable
rather than assumed.

### D4. Equation-name collision: `mse_loss`

`loss-functions-v1` and `setfit-encoder-conformance-v1` both declare an equation
named `mse_loss`. `pv codegen` derives macro names from the equation name alone,
so both emit `contract_pre_mse_loss!` / `contract_inv_mse_loss!` into one
`#[macro_use]` module and the later definition shadows the earlier.

Contained for now: the 01-01 contract's preconditions and invariants were chosen
so the emitted macros are **byte-identical** to the `loss-functions-v1` ones,
making the shadowing a semantic no-op (verified at
`generated_contracts.rs:20334` and `:29825`). A YAML comment on the equation
records the constraint.

This is a latent trap for any future contract, not a defect in either contract.

Fix direction: have `pv codegen` either namespace macros by contract stem or
reject duplicate equation names across contracts outright. The current behaviour
lets one contract silently redefine another's assertions.

## From plan 01-02 (2026-08-08)

### D5. `cargo check --workspace` fails on darwin (intentional platform gate)

`crates/aprender-profile` hard-stops via
`#[cfg(not(target_os = "linux"))] compile_error!("renacer requires Linux (ptrace syscall tracing)")`.

This is an intentional platform gate, not a defect — but it means the
`cargo check --workspace` command named in plan verification blocks can never
pass on macOS. Verified instead with
`cargo check --workspace --exclude aprender-profile` (exit 0, all other 77
crates clean).

Fix direction: phase verification blocks targeting macOS developers should name
the `--exclude aprender-profile` form, or the repo should provide a
`just check` recipe that applies the exclusion per-platform.

### D6. Pre-existing warnings in `aprender-core` test builds

- `f16_first_u16` never used — `serialization/safetensors_tests_core.rs:571`
- unused `#[must_use]` return — `models/bert/embeddings.rs:128`

Unrelated files, not caused by either plan's changes.

## From plan 01-04 (2026-08-08)

### D7. `pv codegen` output is unformatted, so any drift check against the committed file is 61k lines of noise

Plan 01-04 Task 3 required a codegen drift check. Running
`pv codegen contracts/ -o <tmp>/generated_contracts.rs` and diffing against the
committed `crates/aprender-core/src/generated_contracts.rs` reports **61,263
changed lines** — which reads like massive semantic drift and is entirely
formatting. `pv codegen` emits unformatted Rust
(`debug_assert!(x, "msg")` on one line); the committed file has been rustfmt'd
(same call wrapped across four lines).

Proven cosmetic, not assumed: both sides define exactly **3415** macros with
identical name sets, and the whitespace-normalized digests are IDENTICAL
(`0474ab4a26e1c764a9f4abce9585ea27a4206627c6d895c3935ad05c24134d04`).

Why it matters: this is a live trap for **plan 01-08 Task 1**, which is slated to
"regenerate `generated_contracts.rs` as its own commit" if drift is detected. A
naive `diff` will ALWAYS report drift, and acting on it would commit a 61k-line
pure-reformatting churn that hides any real future change. It also means the
repo currently has no usable way to detect genuine codegen drift.

Fix direction: either have `pv codegen` run rustfmt on its output, or add a
`make check-codegen-drift` that normalizes formatting (e.g. `rustfmt` the temp
file, or compare macro-name sets + whitespace-normalized digests) before
comparing. Until then, compare with
`tr -d ' \n\t' < a | shasum -a 256` on both sides.

### D8. `apr convert` cannot read SafeTensors, contradicting the documented example

`CLAUDE.md` documents `apr convert model.safetensors --quantize int8 -o model-int8.apr`,
but `apr convert --help` states the input is a "Path to .apr model file" and the
call fails with `error: Validation failed: At least one of --quantize or
--compress must be specified`. It is an APR→APR quantize/compress optimizer, not
an importer. The working path for SafeTensors→APR is
`apr import <file> -o <out> --arch bert` (used by `slice_model.py`, 37 tensors
validated against `tensor-layout-v1.yaml`).

Secondary inconsistency: `apr convert` accepts `-f/--force`, `apr import` has no
force flag at all, so regeneration must `unlink` the output first.

Fix direction: correct the `CLAUDE.md` example to use `apr import`, and either
add `--force` to `apr import` or document the asymmetry.

### D9. `apr import` misreports BERT's `layer_norm_eps` as `rms_norm_eps` and grades a valid model F

Importing the sliced MiniLM (a faithful index-slice of real pinned weights)
emits:

```
Warning: rms_norm_eps 0.000000000001 below minimum 1e-10 (model-metadata-bounds-v1)
Score 4/100  Grade F
```

Two separate issues. (1) BERT uses **LayerNorm**, not RMSNorm, and its
`layer_norm_eps` is `1e-12` — the pinned upstream `config.json` says so. The
bound in `model-metadata-bounds-v1` (`min 1e-10`) therefore excludes a legitimate
and extremely common value, and the message names the wrong parameter. (2) A
structurally valid model that passes tensor-layout contract validation scores
4/100 / grade F, so the score carries no signal for this artifact class.

Not fixed here: the import succeeds (exit 0), the APR is correct (37 f32 tensors,
HF dotted names preserved), and touching `model-metadata-bounds-v1` is outside
this plan's single-contract scope.
## From plan 01-09 (2026-08-08)

### D10. `cargo test -p batuta-common` silently tests a **crates.io** crate

`batuta-common` is not a package in this workspace. It is a dependency *alias*:

```toml
# Cargo.toml:288
batuta-common = { path = "crates/aprender-common", version = "0.63.0", package = "aprender-common" }
```

The in-tree package is named `aprender-common` (with `[lib] name = "batuta_common"`).
A real, unrelated `batuta-common` crate also exists on crates.io, so:

```
$ cargo pkgid -p batuta-common
registry+https://github.com/rust-lang/crates.io-index#batuta-common@0.1.0
```

`cargo test -p batuta-common --lib erf` therefore **exits 0 having compiled and
tested the registry crate**, not the local source. It was observed reporting
"5 passed" while the four tests just added to `crates/aprender-common/src/math.rs`
were never built. The correct invocation is `-p aprender-common`.

This is the CLAUDE.md rule 8 class (a shadowed artifact is worse than a missing
one): the run is green, the exit code is 0, and it proves nothing about the code
under change. Anyone verifying a change to `crates/aprender-common/` by package
alias will get a false green.

Fix direction: either rename the local package to `batuta-common` (it already
owns that lib name), or drop the alias and depend on `aprender-common` directly
so no registry package can shadow the `-p` selector. Out of scope here because
the alias is load-bearing across many crates' `use batuta_common::` paths.

### D11. `Tensor::gelu` verified CORRECT — recorded to stop a future false alarm

Not a defect; logged because the investigation cost real time and the wrong
conclusion was very nearly recorded as one.

`Tensor::gelu(1.0)` returns `0.8411920`, which looks wrong next to the exact GELU
`0.8413447` and invites the reading "the tanh implementation has a 1.5e-4 bug".
It does not. `0.8411920` is exactly what the tanh approximation evaluates to in
f64 — verified against an independent reference at x = 1, -1 and 2 (matching to
seven digits each). The 1.5e-4 gap is the *algorithmic* difference between the
tanh and erf forms, which is the entire premise of amendment A-03, not an
implementation error.

Anyone comparing the two activations point-by-point will meet this again.

## From plan 01-03 (2026-08-08)

### D12. Contract prose for `cosine_similarity_rows` clamps the PRODUCT; the implementation clamps each FACTOR

`contracts/setfit-encoder-conformance-v1.yaml:241` states the formula as

```
out[b] = <a[b], c[b]> / max(||a[b]||_2 * ||c[b]||_2, eps)
```

while plan 01-03's `<interfaces>` block specifies, and 01-03 implements,

```
out[b] = <a[b], b[b]> / (max(||a[b]||_2, eps) * max(||b[b]||_2, eps))
```

The plan wins for execution (its derivative specification, its acceptance
criteria, and its four-branch FD coverage all presuppose per-factor clamping),
and per-factor clamping is what `torch.nn.functional.cosine_similarity`
implements. But the two forms are not textually reconciled, and the contract is
the phase gate.

**Impact is confined to the degenerate branch.** Wherever both norms exceed
`eps` — the entire non-degenerate domain, and everything the encoder will ever
see with real weights — `max(n_a, eps) * max(n_b, eps) == n_a * n_b ==
max(n_a * n_b, eps)`, so the two definitions agree exactly. They differ only
when at least one row is degenerate, and the invariant `|out| <= 1` holds under
both.

Not fixed here: `contracts/` is deliberately untouched by this plan (01-04 owns
the contract's tolerance commit and 01-01 authored the formula), and editing it
from a parallel worktree risks a merge conflict with the agent that owns it.

Fix direction: 01-04 or 01-08 should reword the YAML formula to the per-factor
form and note in the equation's invariants that the two coincide above the
clamp. Do NOT change the implementation to match the current prose — that would
break the per-input branch independence the FD tests prove.

### D13. `cargo test -p aprender-core --lib mse_loss` is not scoped to the new op

The plan's `<verification>` block names `cargo test -p aprender-core --lib
mse_loss`. That filter is a substring match and picks up **22** tests, only 11
of which are 01-03's; the other 11 are pre-existing `mse_loss` tests elsewhere
in the crate (`nn/loss.rs` and friends). At the RED gate the command reported
`11 passed; 11 failed` — i.e. a naive reader could see 11 green tests and
conclude something about the new op that was in fact entirely stubbed.

The unambiguous form is `cargo test -p aprender-core --lib
tests_similarity_backward` (the module path), which matches exactly the 28 tests
this plan added. Both forms are recorded in the 01-03 SUMMARY.

Same class as D10: a green exit code that proves less than it appears to.

Fix direction: phase verification blocks should filter by test-module path
rather than by op name whenever the op name is a common word already used
elsewhere in the crate.

### D14. `cargo test -- --nocapture` output is swallowed in this environment

Measuring anything from a test's `println!` does not work here. A test invoked
as `cargo test ... -- --ignored --nocapture` exits 0 and reports
`1 passed`, but **none of the printed lines reach the log** — the `rtk` CLI
proxy that wraps `cargo` in this environment filters test stdout as noise.

This is a measurement hazard rather than a repo defect, but it is the exact
class CLAUDE.md rule 1 warns about: the run is green, the exit code is right,
and the evidence you asked for is silently absent. It cost one full
build-and-run cycle before the cause was identified.

Workaround used by 01-03: have the probe write to a file with
`std::fs::File` + `writeln!` and read the file afterwards, instead of relying
on stdout capture.

Fix direction: none needed in-tree. Recorded so the next agent that needs a
numeric measurement out of a test reaches for a file immediately.
## From plan 01-05 (2026-08-08)

Numbered from **D20** to leave room for the plan running in parallel with this
one.

### D20. `sentence_bert_config.json` is outside the frozen upstream file set, so the `max_seq_length` pin is skippable on a real checkout

`upstream_manifest.json` (01-04) records digests for five files —
`1_Pooling/config.json`, `config.json`, `model.safetensors`, `modules.json`,
`tokenizer.json` — and `fetch_full_weights.py` copies exactly those. The
upstream repo also publishes `sentence_bert_config.json`
(`{"max_seq_length": 256, "do_lower_case": false}`), which is where the
sentence-transformers sequence bound actually lives.

ENC-01's mutation matrix requires that a wrong `max_seq_length` be rejected. It
is — but because the D-10 checkout cannot contain the file, `MiniLmImport::open`
validates it **only when present**. A checkout that simply omits it passes.

That is safe today, and the reason is worth recording: the 256-token bound is
applied by `MiniLmTokenizer` from this crate's own `MAX_SEQUENCE_LENGTH`
constant, never read from the checkout, so deleting the file cannot change what
the model does. It is nonetheless a check that an artifact can opt out of by
omission, which is a weaker property than the other twenty-two pin fields have.

Fix direction: add `sentence_bert_config.json` to `UPSTREAM_FILES` in
`scripts/setfit_fixtures/slice_model.py`, regenerate `upstream_manifest.json`
with its digest, and then make the file **required** in `open()`. That is a
fixture-regeneration task (it changes a frozen manifest), which is why it was
not done inside a plan whose scope is Rust.

### D21. Two `tokenizers` versions now coexist in `Cargo.lock`

`aprender-bench-tokenizer` pins `tokenizers = "0.22"` with
`features = ["progress"]`; `aprender-core` now pins the workspace
`tokenizers 0.23.1` with `default-features = false, features = ["fancy-regex"]`.
Cargo resolves both, so `Cargo.lock` carries `tokenizers 0.22.2` and
`tokenizers 0.23.1`.

Not unified here on purpose: the bench crate's whole stated purpose is a
head-to-head against **HF v0.22** (its own description and its results table say
so), and it enables `progress`, which pulls `indicatif`. Silently bumping it
would change what the published benchmark numbers mean.

Fix direction: decide whether the benchmark is pinned to 0.22 as a historical
baseline (in which case add a comment saying so, because right now it reads like
drift) or should track the workspace pin (in which case re-run and re-publish
the numbers).

### D22. `cargo check -p aprender-core --all-features` cannot pass on macOS

`--all-features` enables `audio-alsa`, which pulls `alsa-sys`, whose build
script needs the Linux ALSA development headers:

```
error: failed to run custom build command for `alsa-sys v0.3.1`
```

Proven independent of this plan's changes: `cargo check -p aprender-core
--no-default-features --features audio-alsa` — which touches no setfit code at
all — fails identically.

This is the same class as D5 but a different command and a different crate, so
it is logged separately: D5 is about `cargo check --workspace` and
`aprender-profile`. Any plan whose `<verification>` block names
`-p aprender-core --all-features` will fail on a macOS developer box for reasons
unrelated to that plan.

Fix direction: name a platform-appropriate feature union in verification blocks,
or add a `just check-features` recipe that excludes the OS-specific audio
backends per platform.

### D23. Every optional dependency in `aprender-core/Cargo.toml` is referenced by IMPLICIT feature name, and one `dep:` anywhere breaks them all

Cargo synthesises an implicit feature per optional dependency **only while no
feature uses the `dep:` prefix for it**. `aprender-core` referenced `sha2` by
bare name from `format-encryption` and `hf-hub-integration`. The moment this plan
wrote `setfit = ["dep:tokenizers", "dep:sha2"]`, the manifest stopped parsing:

```
feature `format-encryption` includes `sha2`, but `sha2` is an optional
dependency without an implicit feature. Use `dep:sha2` to enable the dependency.
```

Fixed here by declaring `sha2 = ["dep:sha2"]` explicitly, which keeps the
published feature surface byte-identical while allowing the unambiguous form.

The trap is not specific to `sha2`. Every other optional dependency in that
manifest is still referenced by bare name — `lz4_flex`, `zstd`, `half`,
`ed25519-dalek`, `aes-gcm`, `argon2`, `x25519-dalek`, `hkdf`, `hf-hub`, `dirs`,
`ureq`, `safetensors`, `wasm-bindgen`, `js-sys`, `alimentar`,
`trueno-zram-core`, `hf-xet` — so the next plan that reaches for `dep:` on any
of them meets the same wall. It is also an accidental public API surface: each
of those names is an enableable feature of the published crate.

Fix direction: convert the whole `[features]` block to explicit `dep:` form in
one pass, declaring an explicit forwarding feature for any bare name that must
remain enableable. Out of scope here because it touches every feature in the
crate and this plan's blast radius should stay at `setfit`.

## From plan 01-06 (2026-08-08)

Numbered from **D30** as the orchestrator directed.

### D30. This plan's own `<verification>` filter is not scoped to this plan — D13 recurring

`cargo test -p aprender-core --lib --features conformance-fixtures encoder_`
is the command 01-06's verification block names. Measured, it runs **149**
tests. Only **42** of them belong to this plan; the other 107 are pre-existing
tests whose *path* contains `encoder_` — `transformer_encoder_...`,
`bert_encoder_...` and friends. A reader seeing "149 passed" learns almost
nothing about the encoder.

This is exactly D13, one plan later, in a verification block written after D13
was logged. The unambiguous form is the module path:

```
cargo test -p aprender-core --lib --features conformance-fixtures \
  setfit::encoder::encoder_tests      # 42, exactly this plan's
cargo test -p aprender-core --lib mha_seeded_dropout_   # 10, exactly this plan's
```

Both forms are recorded in the 01-06 SUMMARY; every count reported there was
taken with the module-path form.

Fix direction: the phase's plan template should require a module-path filter,
not a name-prefix filter, whenever the prefix is a word the crate already uses.
`grep -c` the existing test corpus for a candidate prefix before writing it into
a verification block — that check costs one command and would have caught this
and D13.

### D31. Two freshly constructed `nn` modules do NOT have the same weights

`Linear::new` (and therefore `MultiHeadAttention::new`,
`TransformerEncoderLayer::new`, …) draws **random** initial weights. Any test
that builds two modules and compares their outputs is measuring the
initialiser, not whatever it meant to measure.

This cost a cycle here and was caught only because the failure message printed
the two values: the first draft of
`mha_seeded_dropout_same_seed_gives_bitwise_identical_output` compared two
identically-seeded `MultiHeadAttention`s and failed with
`-0.1428478 vs -0.3084763` — a difference far larger than any dropout mask
could explain. Worse, the *sibling* test
(`..._different_seeds_give_different_output`) PASSED, for the wrong reason,
and the RED measurement taken against the pair was invalid until both were
rewritten and RED was re-measured.

The general principle (CLAUDE.md rule 2: prove the mechanism engaged) was
known; the specific instance still went wrong, because "same seed" reads like
it controls everything about the module when it controls only the dropout.

Workaround used: a `deterministic_mha` helper that installs fixed weights via
`q_proj_mut().set_weight(...)` before comparing. `crates/aprender-core/tests/
batched_graph_spike.rs` does the same thing for the same reason.

Fix direction: none needed in-tree — this is correct behaviour for an
initialiser. Recorded so the next agent comparing two `nn` modules reaches for
explicit weights immediately rather than after a confusing failure.

### D32. The D-08 seal makes a whole construction path look dead, and the `#[allow]` cannot be narrowed

01-05 recorded `#![allow(dead_code)]` in `setfit/import.rs` with the removal
condition "stops being needed the moment 01-06 wires the encoder to
`MiniLmImport`". Measured after wiring: the surface fell from ~15 findings to
exactly **three** — `VocabRemap::from_json_bytes`, `SliceConfig::from_json_bytes`
and `validate_pooling` — so the allow is still load-bearing and was NOT removed.
The comment in `import.rs` now records the measurement.

`setfit/encoder.rs` needed the same allow for the same reason, and a *targeted*
version was tried first and measured to be whack-a-mole: annotating
`install_projection` and `EMBEDDINGS_DROPOUT_SITE` simply moved the finding to
`site_seed`. The cause is structural — `from_import` is `pub(crate)` under the
seal and has no non-test caller, so dead-code analysis walks the entire
construction path it reaches.

Fix direction: 01-07 is the plan that can delete both allows, because
`SetFitMiniLm` is the first non-test caller of either constructor. It should
delete them and re-run clippy rather than assume they became unnecessary.

### D33. `Module::train`/`eval` recurse on `Sequential` but not on `MultiHeadAttention`

01-02 established `set_training` as the propagation channel (D-17) and left
`train`/`eval` leaf-local — but `Sequential::eval()` already recursed via
`child.eval()` (01-02 recorded this) while `MultiHeadAttention::eval()` sets
only its own flag. So the crate has two conventions and no way to tell which a
given composite follows.

That is a live footgun for any module whose behaviour depends on mode:
`model.eval()` returning a model with dropout still active produces stochastic
"inference" with no error anywhere. `BertSentenceEncoder` therefore routes both
spellings through `set_training`, and a test asserts it — but that is one
module opting out of an inconsistency, not a fix.

Fix direction: pick one convention crate-wide. The safe one is "`train`/`eval`
delegate to `set_training`, which is the only method a composite overrides";
it makes the wrong thing impossible to write rather than merely documented.
That is a change across every `impl Module`, hence deferred.

## From plan 01-07 (2026-08-08)

Numbered from **D40** as the orchestrator directed. D1-D14, D20-D23 and D30-D33
are untouched. D1, D2, D5, D13, D14 and D30 were used as documented and are not
re-logged. **D32 is CLOSED by this plan** — see the 01-07 SUMMARY.

### D40. The plan's D-08 declaration `grep`, run as written, returns 8 matches on the pre-existing tree

Plan 01-07 Task 2 specifies

```
grep -rnE --include='*.rs' '^[^/]*\bpub fn (from_bytes|open|open_slice_fixture|from_import)\b' \
  crates/aprender-core/src/setfit/
```

and asserts it "must return no match". Measured on 2026-08-08 at the plan's base
commit, before this plan changed anything relevant, it returns **8** matches:

```
setfit/encoder_tests.rs:57    !src.contains("pub fn from_import("),
setfit/encoder_tests.rs:58    "D-08 seal broken: a bare `pub fn from_import(` exists in setfit/encoder.rs"
setfit/tokenizer_tests.rs:354 !src.contains("pub fn from_bytes"),
setfit/tokenizer_tests.rs:355 "D-08 seal broken: a bare `pub fn from_bytes` exists"
setfit/import_tests.rs:631    !src.contains("pub fn open("),
setfit/import_tests.rs:632    "D-08 seal broken: a bare `pub fn open(` exists in import.rs"
setfit/import_tests.rs:639    !src.contains("pub fn open_slice_fixture("),
setfit/import_tests.rs:640    "D-08 seal broken: a bare `pub fn open_slice_fixture(` exists"
```

Every one is a **string literal inside 01-05's and 01-06's own seal assertions**.
None is a declaration. The `^[^/]*` prefix excludes `//` comments but has no
notion of a string literal, and the plan's nine-row case table contains no
string-literal row, so the table could not have caught it either.

Two further wrinkles the same root cause produces:

1. A test file that ships a must-MATCH case table under `src/setfit/` trips its
   own gate. Observed here on the first run; worked around by assembling the
   table rows at runtime from a `PUB_FN` constant so the source text stays clean.
2. Tightening the pattern to exclude string literals is not straightforward: the
   obvious "no `\"` before the match" rule rejects the plan's own row 3
   (`#[cfg(feature = "conformance-fixtures")] pub fn open_slice_fixture(`), which
   must MATCH.

Resolved here by scoping the scan to non-test sources
(`--exclude='*_tests.rs'`), which returns 0. The justification is not
convenience: a `#[cfg(test)]` module is not compiled into the library at all, so
it cannot reopen the seal for an out-of-crate consumer — and the compile probe,
which is the primary evidence, is immune to the whole question.

The plan's separate warning about crate-wide widening was RE-MEASURED and is
exactly right: `crates/aprender-core/src/` yields 22 lines, of which 14 are the
legitimate pre-existing declarations it lists (apr/mmap/bundle/onnx/gguf/hnsw
readers) and 8 are the string literals above.

Fix direction: phase plans that prescribe a guard regex should run it against
the CURRENT tree before writing "must return no match" into an acceptance
criterion, and case tables for source-scanning regexes should include a
string-literal row. One command would have caught this — the same lesson D30 and
D13 record for test filters, now for guard patterns.

### D41. `pub(crate)` METHODS are rejected with E0624, not E0603

Plan 01-07 requires the D-08 seal to be demonstrated red by an out-of-crate
compile probe whose log "contains `E0603`". It does not. All four sealed
constructors are **associated functions on public types**, and rustc's
diagnostic for those is:

```
error[E0624]: associated function `open` is private
   --> crates/aprender-core/tests/zz_seal_probe.rs:13:45
    |
 13 |     let _ = aprender::setfit::MiniLmImport::open(Path::new("/nonexistent"));
    |                                             ^^^^ private associated function
    |
   ::: crates/aprender-core/src/setfit/import.rs:398:5
    |
398 |     pub(crate) fn open(dir: &Path) -> Result<Self, SetFitError> {
    |     ----------------------------------------------------------- private associated function defined here
```

E0603 is `... is private` for an item reached through a **module path** (a
private module, a private free function, a private `use`). It would be the right
code if the seal had been implemented by making the *module* private, which it
was not.

This matters because the probe was to be measured by
`grep -c E0603 /tmp/seal_probe.log`. Against a perfectly sealed crate that
returns **0**, which reads exactly like "the seal is not holding" — a false
NEGATIVE on the phase's structural gate, in the same class as CLAUDE.md rule 1's
"the run is green and proves nothing", only inverted.

Measured here: the probe exits non-zero with **four E0624 errors, zero E0603**.

Fix direction: 01-08 and any later plan that re-runs this probe should assert on
`E0624` (or, more robustly, on `is private` plus a non-zero exit), and the
acceptance criterion in 01-07-PLAN.md should be corrected if it is ever re-run.

### D42. `requires_grad(false)` does NOT prevent a parameter from receiving a gradient

Recorded because it inverts the intuition an optimizer author brings from torch,
and because it was found by measurement rather than by reading.

Freeze mutation D (this plan) made `trainable_parameters_mut()` ignore the freeze
policy while leaving `apply_freeze`'s `requires_grad_(false)` in place. If the
flag were sufficient, nothing would have changed. Instead
`encoder.layer.1.attention.self.query.weight` **moved across the optimizer
step**: the `Linear` weight still received a gradient, because the ops that
consume it register the edge based on their INPUT requiring grad and then produce
gradients for both operands regardless of the weight's own flag.

The two halves are therefore not redundant. Exclusion from
`trainable_parameters_mut()` is the load-bearing mechanism; the flag protects
only those parameters whose consuming op checks it (notably `embedding_gather`,
which is why `embeddings.word_embeddings.weight` stops receiving gradient
entirely once frozen).

Consequence for **01-08**: build the AdamW parameter set from
`SetFitMiniLm::trainable_parameters_mut()`. Constructing it from
`encoder().named_parameters_mut()` and relying on `requires_grad` to skip the
frozen ones will silently train frozen weights, and the fixture parity gates
cannot see it.

Fix direction: none required in-tree — this is a coherent design given that the
optimizer owns its parameter set. Recorded so nobody re-derives it from a
confusing ENC-04 result.

### D43. `mod.rs` is now 500+ lines and mixes module wiring with a public type

`crates/aprender-core/src/setfit/mod.rs` was 57 lines of module declarations and
re-exports. This plan added `FreezeGroup` and `SetFitMiniLm` to it, because the
plan's `<must_haves>` artifact list names `mod.rs` as the file that must contain
`pub struct SetFitMiniLm`. The result is a `mod.rs` that is both the module's
table of contents and the home of its largest type, which is the opposite of the
convention every other file in `src/setfit/` follows.

Not fixed here: moving the type to `setfit/model.rs` and re-exporting it would
change the artifact path the plan's must-haves assert, and a plan checker reading
`contains: "pub struct SetFitMiniLm"` against `mod.rs` would then fail.

Fix direction: a follow-up that moves `FreezeGroup` + `SetFitMiniLm` into
`setfit/model.rs` with `pub use model::{FreezeGroup, SetFitMiniLm};` in `mod.rs`.
The public API is unchanged by that move, so it is a pure refactor — but it must
be paired with an update to 01-07's must-haves or it will read as a regression.

## From plan 01-08 (2026-08-08)

Numbered from **D50** as the orchestrator directed. D1-D14, D20-D23, D30-D33 and
D40-D43 are untouched. D1, D2, D5, D7, D10, D12, D13, D14, D22, D30, D41 and D42
were used as documented and are not re-logged. **D12 is CLOSED by this plan** —
the contract was reworded to the per-factor clamp and `pv validate` re-run clean.

### D50. `make tier2` and `make tier3` are BOTH RED at this phase's base commit

Measured on 2026-08-08, statuses captured directly (`cmd > file 2>&1; rc=$?`):

| target | rc | dies at | cause |
|---|---|---|---|
| `make tier2` | 101 | recipe line 2, `cargo clippy -- -D warnings` | pre-existing findings in `aprender-compute` (unused imports in `q4k/gemv/mod.rs`, `blis/packing.rs`, `vector/ops/rounding.rs`) and an unreachable expression in `aprender-present-terminal/src/compute_block.rs:93` |
| `make tier3` | 101 | recipe line 1, `cargo test --all` | 12 x `E0063` in `crates/aprender-serve/tests/driver_cpu.rs` (missing `post_attn_norm_weight`, `post_ffw_norm_weight`, `query_pre_attn_scalar`) plus `aprender-profile`'s intentional macOS `E0601` |

Neither is attributable to this phase:
`git diff <base>..HEAD -- crates/aprender-present-terminal/ crates/aprender-serve/`
is **empty**. The tier2 cause is D2 recurring at tier scope (D2 recorded the same
`-D warnings`-over-path-dependencies problem for `cargo clippy -p aprender-core`);
the tier3 cause includes D5 recurring (`aprender-profile` is Linux-only).

**Consequence for 01-08's acceptance criteria.** GNU make on this box is 3.81, so
`.ONESHELL:` is ignored and each recipe line gets its own shell — a failing line
stops the recipe. The Phase 1 lines this plan added therefore never execute under
a plain `make tier2` / `make tier3`, and the criteria "make tier2 green" and
"`make tier3` output shows the setfit contract validated" cannot be met on this
machine no matter what this phase does. Evidence was provided instead by running
both tiers with ONLY those two pre-existing-red steps disabled (rc=0 for both;
tier3's output then shows `pv validate contracts/setfit-encoder-conformance-v1.yaml`
-> "0 error(s), 0 warning(s)" and "Tier 3: PASSED"). The shipped Makefile is
unmodified in those two lines.

Fix direction: two separate tickets, neither in this phase's blast radius —
(a) clean `aprender-compute` + `aprender-present-terminal` or scope tier2's clippy
to the crates under change, and (b) repair `aprender-serve/tests/driver_cpu.rs`
against the current `OwnedQuantizedLayer` / `GGUFConfig` shapes, plus exclude
`aprender-profile` from `cargo test --all` on non-Linux.

### D51. `cargo mutants` cannot see any op composed with `include!()`

`crates/aprender-core/src/autograd/ops/mod.rs` pulls in seven files with
`include!("activation.rs")`, `include!("embedding.rs")`, `include!("masking.rs")`,
`include!("pooling.rs")`, `include!("normalize.rs")`, `include!("similarity.rs")`
and `include!("op_error.rs")`. cargo-mutants parses sources with `syn` and follows
`mod` declarations; it does **not** expand `include!`. Measured:

```
cargo mutants --list --package aprender-core -f '**/autograd/ops/*.rs'
  -> 39 mutants, EVERY ONE in mod.rs
```

Zero mutants are generated for `embedding_gather`, `masked_mean_pool`,
`l2_normalize_rows`, `cosine_similarity_rows`, `mse_loss`,
`additive_attention_mask` or `apply_additive_mask`. **D-25's stated scope
("autograd ops") is unreachable by file glob**, and a run that names it will
report a confident green over code it never mutated — the "the run is green and
proves nothing" class, with the emptiness hidden behind a plausible mutant count
from a neighbouring file.

01-01 and 01-03 chose `include!()` deliberately (the ops are one logical module
split for reviewability), and 01-01 already recorded that `include!()` files are
invisible to `check_include_files.sh` on macOS (D1). This is the same composition
choice costing a different tool.

Fix direction: convert the seven `include!`s to `#[path = "..."] mod` declarations
— the crate already uses that form for its test modules and the public paths would
be unchanged if each is `pub(crate) use`d — or drive mutation from whole-crate
scope with an explicit exclusion list instead of a file glob.

### D52. `cargo mutants -f <literal path>` silently lists ZERO mutants

The plan's command reads
`-f 'crates/aprender-core/src/setfit/encoder.rs'`. Run verbatim:

```
cargo mutants --list -f 'crates/aprender-core/src/autograd/ops/*.rs' \
  -f 'crates/aprender-core/src/setfit/encoder.rs' \
  -f 'crates/aprender-core/src/nn/transformer/positional_encoding.rs'
-> rc=0, ZERO mutants listed
```

cargo-mutants 25.3.1 matches `--file` globs against the path with `**` semantics;
a repo-root-relative literal matches nothing. The correct form is
`-f '**/setfit/encoder.rs'`, which lists **84**.

The failure mode is the dangerous direction: `rc=0` with an empty list, so
`cargo mutants` would report "0 mutants tested, 0 missed" and a reader would
record a perfect mutation score over nothing at all.

Fix direction: any plan that prescribes a cargo-mutants glob should `--list` it
first and assert a non-zero count before the run — one command, and it is the same
lesson D40 records for guard regexes and D13/D30 record for test filters.

### D53. `provable-contracts = "0.3"` in `aprender-core`'s dev-deps is the CRATES.IO crate

`crates/aprender-core/Cargo.toml` carries `provable-contracts = "0.3"` under
`[dev-dependencies]` while the workspace is at 0.63.0 and ships the same code
in-tree as `crates/aprender-contracts` (whose `[lib] name` is also
`provable_contracts`). So `use provable_contracts::…` in any aprender-core test
resolves to the **registry** crate, not to the schema this repository owns.

This is D10's class exactly — a green run against code that is not under change —
and `crates/aprender-core/tests/contract_traits.rs:10` already does
`use provable_contracts::traits::{…}`.

01-08 needed the in-tree parser for its tolerance generator and could not simply
write `use provable_contracts::`: both packages declare the same lib name, so the
in-tree one is only reachable under a different dependency key. It was added as
`setfit-contract-schema = { path = "../aprender-contracts", package = "aprender-contracts" }`
with the reason recorded in the manifest.

Fix direction: repoint the dev-dep at the path (`provable-contracts = { path =
"../aprender-contracts", package = "aprender-contracts" }`) and re-run
`contract_traits.rs`, which may well have been asserting against a two-years-stale
API. Out of scope here because it changes what an existing test tests.

### D54. Surviving mutants in `setfit/encoder.rs`, outside the graph-recording blocks

01-08's D-25 breadth pass over `crates/aprender-core/src/setfit/encoder.rs`
surfaced these MISSED mutants. **None is in a graph-recording block**, so none
violates 01-08's acceptance criterion (Pass A over `add_mask`, `forward_layers`,
`forward_tokens`, `forward_tokens_per_layer` and `encode` was 7 caught /
1 unviable / **0 missed**). They are real coverage gaps in 01-06's encoder tests
and are recorded rather than silently absorbed.

**M1. `encoder.rs:524:14: replace > with >= in BertSentenceEncoder::validate`**

```rust
let max = self.max_seq();
if s > max { return Err(SetFitError::OversizeInput { len: s, max }); }
```

With `>=` a batch of **exactly** `max_seq()` tokens (64 on the slice, 256 on the
full pin) is rejected. No test feeds that boundary: 01-06's `max_seq` test drives
a length *above* the bound to trigger rejection, and every corpus text is 9-20
tokens. So the boundary itself is unpinned in both directions.

This is not academic — 256 is the real production bound, and rejecting a
legitimately 256-token input would look like a tokenizer bug, not a validation
bug.

Fix direction: one test in `src/setfit/encoder_tests.rs` that builds a batch of
exactly `encoder.max_seq()` positions and asserts it is ACCEPTED, paired with the
existing `max_seq + 1` rejection. Not added by 01-08 because
`src/setfit/encoder_tests.rs` is 01-06's file and 01-08's acceptance criteria
require `git diff --stat crates/aprender-core/src/` to be empty; reaching into a
wave-4 plan's implementation is the B5 defect class the plan explicitly forbids.

**M2. `encoder.rs:119:19: replace ^ with | in site_seed`**

The per-dropout-site seed derivation mixes with XOR. Replacing it with OR still
produces *a* deterministic seed, so every reproducibility test still passes — but
OR is not a mixing function: it saturates toward all-ones and can collapse
distinct sites onto the same stream.

01-06's seeded-dropout tests assert same-seed/different-seed behaviour at the
`MultiHeadAttention` level, and 01-08's fixtures are all eval-mode (dropout
inert), so nothing observes the *distinctness* of the seven site streams.

Fix direction: a test asserting the seven values `site_seed` produces for one
root seed are pairwise distinct, and that two different root seeds produce
disjoint sets. Same file-ownership constraint as M1.

**M3. `encoder.rs:658:9: replace <impl Module for BertSentenceEncoder>::parameters_mut -> Vec<&mut Tensor> with vec![]`**

The sibling `named_parameters_mut` mutation IS caught (line 735), because
`SetFitMiniLm::trainable_parameters_mut()` — and therefore every optimizer in the
phase — goes through the NAMED accessor. Nothing in the phase consumes the
positional one, so emptying it is invisible.

`nn/module.rs` states the invariant explicitly: *"`named_parameters()` has the
same length as `Module::parameters`, and element `i` of each refers to the same
tensor."* An empty `parameters_mut()` breaks that silently, and any future caller
that reaches for the positional accessor (a generic optimizer, a
`Sequential`-style composite, a checkpoint writer) gets nothing and reports
success.

Fix direction: one arity/identity test per `impl Module` —
`parameters_mut().len() == named_parameters_mut().len()` and pairwise
`Tensor::id()` equality — ideally as a shared helper so it is cheap to apply to
every implementor rather than to `BertSentenceEncoder` alone.

**Pass B was STOPPED at its declared wall-clock budget** after 12 of the file's
**84** mutants (9 caught, 3 missed), so this list is not exhaustive. A complete
run of `setfit/encoder.rs` at the measured ~2.5 min/mutant needs ~3.5 hours. See
the 01-08 SUMMARY for the full budget accounting.

## From phase verification (2026-08-08)

### D55. The ENC-04 optimizer-step obligation cannot fail — two independent causes [CLOSED]

Found by code review (CR-01) and extended by the phase verifier's mutation M4.
Confirmed independently by the orchestrator. **Closed 2026-08-08 — see the
resolution at the end of this entry. ENC-04 is now fully satisfied.** The
diagnosis below is kept verbatim as the record of what was wrong.

**Cause 1 — the tolerance is larger than the effect it measures.**
`scripts/setfit_fixtures/generate_fixtures.py:557` assigns `grad_delta` (the
GRADIENT family's f32/f64 delta) to the optimizer family. No f64 optimizer step
is ever run, and both families share `FAMILY_REDUCTION_WIDTH = 1024`, so the
`gradients` and `optimizer_step` entries in `tolerances_measured.json` are
byte-identical:

```
gradients      => {"max_abs_f32_f64_delta": 3.6317159e-07, "recommended_tolerance": 3.05175781e-05}
optimizer_step => {"max_abs_f32_f64_delta": 3.6317159e-07, "recommended_tolerance": 3.05175781e-05}
```

Max per-element AdamW step-1 displacement is `2.017474e-05` against a
`3.052e-05` tolerance — 1.526x. Verifier mutation **M1** (delete `weight_decay`)
survives: 25 passed, exit 0. Mutation **M2** (halve `lr`) passes the post-step
parity obligation but the enclosing test still fails at `loss_after`
(1.688e-3 vs 7.63e-6) — so the OBLIGATION is vacuous while that one test is not.

The `activation` family already carries a guard asserting its tolerance sits far
below the effect it must separate (`generate_fixtures.py:320-324`).
`optimizer_step` has no such guard. The pattern existed and was not applied
uniformly.

**Cause 2 — betas are unconstrainable by any single-step fixture (structural).**
Verifier mutation **M4**: hardcoding `b1 = b2 = 0.5` survives BOTH the
conformance suite (25 passed) and the pre-existing `--lib adamw` suite
(26 passed). At step 1 with bias correction, `m_hat = (1-b1)g/(1-b1) = g` and
`v_hat = (1-b2)g^2/(1-b2) = g^2`, so the update is `lr*g/(|g|+eps)` for every
choice of betas. **No tolerance edit can fix this** — it requires a multi-step
optimizer fixture.

**Already applied:** the false "Detected here: ... a wrong AdamW hyperparameter
or decay coupling" claim in `tests/setfit_conformance.rs` has been corrected to
state the shortfall, so Phase 3 does not inherit a gate that advertises
protection it does not provide.

**RESOLVED — option 1 (fix the generator), 2026-08-08.**

Contract `setfit-encoder-conformance-v1` **1.0.0 → 2.0.0** (`pv diff` classifies
both changes as MAJOR; both are strictenings).

*Cause 1 closed.* `generate_fixtures.py` now runs a real f64 AdamW step through
the shared `adamw_trajectory()` helper and records the optimizer family's OWN
delta. `FAMILY_REDUCTION_WIDTH["optimizer_step"]` drops 1024 → 1 (a post-step
parameter is not a 1024-wide reduction: at step 1 the update saturates to
`lr*sign(g)`, so gradient reduction noise does not reach it), and the family
passes an explicit `scale` = the parameter magnitude it compares at. A new
`assert_separation()` — the generalized form of the `activation` guard at
`:320-324` — fails generation unless a tolerance sits 10x below the effect it
gates, and is applied to both optimizer families.

```
optimizer_step  3.05175781e-05 -> 1.89172753e-06   (16x tighter; 10.6x below the displacement)
```

*Cause 2 closed.* New obligation `OBLIG-ENC-04-MULTISTEP-TRAJECTORY-PARITY`
replays `MULTISTEP_N = 20` steps and compares the LOSS TRAJECTORY
(`optimizer_multistep.json`, tolerance 7.62939453e-06). A loss trajectory rather
than a second parameter dump because a max-abs parameter comparison is limited
by its noisiest single element — measured, the betas mutation stays within 2.6x
of the f32/f64 noise floor at every step count up to 50 — while the loss
contracts the model into one number and accumulates divergence coherently. The
generator runs the betas mutation itself and asserts the separation, so the
margin is measured on every regeneration rather than asserted in a comment.

**Re-mutation evidence (all run against the merged tree, exit codes captured
directly, not through a pipe):**

| Mutation | conformance suite | `--lib adamw` |
|---|---|---|
| M4 betas `(0.5, 0.5)` | **exit 101** — trajectory off 4.0513277e-4 vs 7.63e-6 tol | exit 0, 26 passed |
| M5 update scale halved | **exit 101** — post-step off 1.001358e-5 vs 1.89e-6 tol | — |
| M1 decay deleted | exit 0, 26 passed | **exit 101** — `falsify_aw_001` |

M5 is the direct measure of what the tolerance fix bought: 1.001358e-5 sat
comfortably inside the old 3.05e-5 tolerance and is now rejected.

**M1 remains structurally undetectable here, and this is now stated rather than
implied.** At `lr = 2e-5, wd = 0.01` the decay term is `lr*wd*|p| <= 1.747567e-07`
— 3.36x the f32 ulp at the parameters it acts on — so no f32 reference
comparison separates it at any step count. `adamw-kernel-v1`'s
`falsify_aw_001_decoupled_weight_decay` owns it and was confirmed RED under M1.

*Also hardened:* `assert_encoder_updates` clause (f) moved from `delta > 0` to a
`[0.9, 1.1] x lr` band for non-exempt tensors (measured band across the fixture:
`[1.00052, 1.00739] x lr`), gated by a new `GateInput::step_lr` so only a
genuine first step asserts it. It carries its own negative test —
`gradient_gate_clause_f_rejects_a_step_of_the_wrong_magnitude` — because the
post-step parity assertion runs first and would otherwise mask a live mutation.

Regeneration was byte-deterministic: only `tolerances_measured.json`,
`manifest.sha256`, and the new `optimizer_multistep.json` changed. Every other
fixture is unchanged, which is the D-13 evidence that this was a surgical edit
and not a re-baseline.

Suite: 25 -> 27 tests, exit 0. Full `-p aprender-core --lib`: 14120 passed, exit 0.

**Blast radius:** Phase 3 (TRN-03) is now unblocked.

### D56. `make tier2` / `make tier3` are RED at the phase base (inherited)

Two acceptance criteria of plan 01-08 are UNMET. Verified inherited, not a phase
regression: `make tier2` fails identically at phase base `e6dce92a0`, in
`aprender-zram-core` (3 errors) and `aprender-present-terminal` (1 error), and
this phase changed **0 files** in those crates or in `aprender-compute`.

Consequence established by the verifier: `make tier2` exits 2 at line 188 and
the log contains **no "Phase 1 SetFit" marker** — the gates wired at lines
210-213 never execute. The tier3 contract path is green in isolation
(`make contract-validate` exit 0), and the suite runs green via direct
`cargo test -p aprender-core --test setfit_conformance --features setfit,conformance-fixtures`.

So the Phase 1 gate exists and passes, but cannot currently be reached through
the tier targets end-to-end. Fixing the two inherited crates is out of scope for
this phase and belongs to whoever owns them.
