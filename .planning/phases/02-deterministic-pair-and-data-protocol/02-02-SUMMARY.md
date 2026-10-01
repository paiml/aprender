---
phase: 02-deterministic-pair-and-data-protocol
plan: 02
subsystem: contrastive-data
tags: [new-crate, d04-bytes-boundary, contract, pv, makefile-gates, error-taxonomy, philox, publish-cascade]
requires:
  - "02-01 (branch gsd/phase-2-contract-gate, $(CONTRACTS) wiring precedent, D-06 baseline)"
provides:
  - "crates/aprender-contrastive-data — publishable workspace member, compiles, packages"
  - "ContrastiveDataError — 34 typed variants, #[non_exhaustive], the phase's whole failure surface"
  - "the COMPLETE 13-module pub mod skeleton, so plans 02-03..02-07 never edit lib.rs"
  - "root [workspace.dependencies] entries for aprender-rand, unicode-normalization, trybuild, aprender-contrastive-data"
  - "contracts/contrastive-pair-protocol-v1.yaml v1.0.0 — pv-green, 24 equations / 15 obligations / 20 falsification tests / 2 kani declarations / gate F-CONTRASTIVE-001"
  - "make contrastive-data-boundary — D-04 enforced by two POSITIVE checks, all four failure modes observed"
  - "tier2 line for the crate; tier3 reaches both the new contract and the boundary gate"
  - "binding-audit mechanism probed: cross-crate module_path ACCEPTED (02-08 unblocked), plus two recorded traps"
affects:
  - "plans 02-03..02-07 (they fill the module stubs and bind to these equations; lib.rs is off their edit path)"
  - "plan 02-08 (owns contracts/aprender/binding.yaml; the probe answers its blocking question and names two traps)"
  - "plan 02-09 (CLI surface consumes the crate through apr-cli's new dependency)"
  - "the release cascade — aprender-contrastive-data MUST publish BEFORE apr-cli"
tech-stack:
  added: []
  patterns:
    - "positive dependency allowlist compared against the resolved cargo tree closure, instead of a deny-list"
    - "src/-wide fs/net/path symbol ban with NO cfg(test) exemption, comments filtered from RESULTS so line numbers stay true"
    - "gate failure modes induced, observed and reverted before the gate is trusted"
    - "module skeleton declared once so parallel waves never contend on lib.rs"
key-files:
  created:
    - crates/aprender-contrastive-data/Cargo.toml
    - crates/aprender-contrastive-data/allowed-deps.txt
    - crates/aprender-contrastive-data/src/lib.rs
    - crates/aprender-contrastive-data/src/error.rs
    - crates/aprender-contrastive-data/src/{schema,hash,split,prepared,attestation,dedup,ledger,rng,buckets,select,pairs,manifest}.rs
    - contracts/contrastive-pair-protocol-v1.yaml
    - .planning/phases/02-deterministic-pair-and-data-protocol/deferred-items.md
  modified:
    - Cargo.toml
    - Cargo.lock
    - crates/apr-cli/Cargo.toml
    - Makefile
decisions:
  - "cargo package -p apr-cli is red from wave 2 to the publish cascade — AND SO IS --no-verify; --no-verify skips the build, not the manifest resolution"
  - "Both CB-510 guard scripts pass VACUOUSLY on macOS: they use GNU grep -P, so they report 0 include!() files where the true count is 1768"
  - "The binding registry accepts module_path: aprender_contrastive_data::* under target_crate: aprender — 02-08 proceeds as written"
  - "A binding entry whose contract: field carries a ../ prefix parses cleanly and binds NOTHING — a silent no-op"
  - "cargo test accepts exactly ONE positional filter; every contract command form was verified runnable before being written down"
metrics:
  duration: ~35m
  tasks: 4
  files: 19
  completed: 2026-08-08
---

# Phase 2 Plan 02: Contrastive-Data Crate, Phase Contract, and the D-04 Boundary Gate Summary

Created the `aprender-contrastive-data` crate as a compiling, packaging, publishable workspace
member with the phase's entire typed error surface and a module skeleton that keeps later waves
parallel; authored `contrastive-pair-protocol-v1.yaml` pv-green on first commit; and turned D-04
from an aspiration into two gates whose every failure mode was induced and watched rather than
assumed.

Four commits on `gsd/phase-2-contract-gate` (no new branches, no PRs, per the policy 02-01 set):

| # | Task | Commit | What |
|---|------|--------|------|
| 1 | Scaffold + workspace wiring + error surface | `06cda1009` | 19 files, +813 |
| 2 | Contract part 1 — equations, obligations | `9c1812768` | +880 |
| 2b | Contract part 2 — falsification, Kani, qa_gate | `e97ffe7ca` | +423, **0 deletions** |
| 3 | Makefile: $(CONTRACTS), tier2, boundary gate | `c104221f4` | +118 −1 |

## What Was Built

### Task 1 — the crate

**The three missing workspace dependency entries were genuinely missing.** Verified before
adding: root `[workspace.dependencies]` had no `aprender-rand` (a workspace *member* since the
monorepo consolidation, at `members` line 27, but with no dependency entry, so every consumer had
to re-declare the path), no `unicode-normalization` (declared per-crate at `"0.1"` by
`aprender-train` and `apr-cli`), and no `trybuild` (per-crate at `"1"` by
`aprender-contracts-macros` and `aprender-test-derive`). All three now exist at the versions the
existing consumers already pin, so hoisting changes nobody's resolution. Without them every
`{ workspace = true }` in waves 3–6 would fail to resolve.

**`ContrastiveDataError` — 34 variants, `#[non_exhaustive]`.** DATA-02's eleven ingest classes,
four selection/manifest classes, five attestation classes, eight pair classes, and six
version/arithmetic/plumbing classes. Messages carry split, index, id, expected and got in the
`data_tweeteval.rs` style. The module doc frames the enum explicitly as an **exhaustive initial
design, not a freeze**: a later plan may add a variant provided it records the addition in its
SUMMARY and extends `OBLIG-CPP-ERROR-TAXONOMY`.

**The module skeleton is the parallelism.** Thirteen `pub mod` declarations and one crate-root
re-export land here, once. Every later plan edits only its own module file, so waves 3–5 never
contend on `lib.rs`. Each stub carries a `//!` doc naming both what it will hold and the plan that
fills it, and — where it matters — *why* the design is what it is, so the reasoning is in the file
the implementer opens rather than in a planning document they may not read.

**`allowed-deps.txt` is a reviewed artifact, not a dump.** 28 packages, each traced to its parent
in the resolved tree and grouped by provenance in the header (direct / proc-macro toolchain /
serde runtime / RustCrypto / unicode). `libc` is called out specifically: it arrives only through
`cpufeatures`' SHA-2 CPU-feature detection, and the src/-wide symbol ban is what makes "and we
don't use it for files or sockets" checkable rather than merely stated.

### Task 2 + 2b — the contract

`pv status`: **v1.0.0, 24 equations, 15 proof obligations, 20 falsification tests, 2 Kani
harnesses, gate F-CONTRASTIVE-001.** `pv validate` exits 0 with 0 errors and 0 warnings.

Task 2b's diff against Task 2 is **423 insertions and zero deletions** — the equations and
obligations were not touched when the evidence sections were added.

The load-bearing content, in one line each:

- **`rng_key_derivation` freezes the bytes.** `DOMAIN_TAG = b"apr-contrastive-v1\x00"`, seed
  serialized little-endian, key truncation little-endian in both lanes, lane 0 is the low half of
  an assembled `u64`. Two implementations that "both use Philox" can still disagree on every
  sampled id if any of these is left implicit.
- **`rng_domain_strings`** freezes the table at exactly six strings and pins how `{label}` renders
  (base-10, no padding, no locale), so cross-platform formatting cannot drift the selection.
- **`bounded_draw` takes `NonZeroU64`**, making a zero bound unrepresentable rather than
  misbehaving. Modulo and `next_f32` index draws are forbidden, with the reason attached.
- **`pair_stream` contracts the O(K) negative sampler** — per-class weights `n_c * (S - n_c)`, a
  first-endpoint draw, then a block-skipping map over the class-offset array. The O(K²) class-pair
  enumeration appears only as the **rejected** alternative, named so it is not reinvented.
- **`budget_resolution` makes the hard cap bind explicit budgets.** `budget > hard_cap` fails
  loudly. A silent clamp would keep the DoS ceiling while discarding a number the user typed, so
  the run succeeds and produces a *different dataset than the one requested* — the worst available
  reproducibility outcome, because nothing is red and the manifest looks fine.
- **`selection_canonical_payload` is non-circular** — the hashed payload excludes its own digest,
  which lives in the outer envelope with the volatile metadata.
- **`pair_manifest_hash` commits the replay tuple**, `record_bytes || 0x1E || streamed pairs`, so
  two different seeds cannot collide to one attestation; pairs are fed to the hasher as emitted
  and never collected, which would have reintroduced O(budget) memory in the one place nobody
  would look.
- **`cross_split_exclusion` coalesces by connected component** over the union of exact- and
  normalized-hash edges, so a row that is both kinds of duplicate decrements its pool once.

All four required predictions are present with their numbers: the `[4,1]` divergence (measured
reference epoch length **22** vs contracted budget **12**, asserted as a *divergence* so it cannot
be "fixed" by regenerating one side), the clamp-engages case, the explicit-over-cap
`BudgetExceedsHardCap` case, and the K≈N adversarial case (512 singleton classes, fixed budget 64).

### Task 3 — the gates

`$(CONTRACTS)` gains the contract; `make contract-validate` now reports **44/44 valid** (43
before) and reaches the new entry. tier2 gains `cargo test -p aprender-contrastive-data` with a
measured comment (2 s / 2 s / 2 s, three warm runs). tier3 gains `contrastive-data-boundary`,
which ran standalone green at 1 s before being wired.

## The Gate Mutation Proofs (all induced, observed, reverted)

A gate that has only ever been observed passing is not evidence. Each mutation was applied, run
with its status captured directly, and reverted; the crate directory is byte-clean against HEAD
afterwards.

| Mutation | Result | Output |
|---|---|---|
| `tempfile = { workspace = true }` added to `[dependencies]` | rc=2 | prints `tempfile` **and its six transitive deps** — `bitflags`, `errno`, `fastrand`, `getrandom`, `once_cell`, `rustix` |
| `use std::path::PathBuf;` appended to `src/schema.rs` | rc=2 | `crates/aprender-contrastive-data/src/schema.rs:9:use std::path::PathBuf;` |
| the same, **inside `#[cfg(test)] mod tests`** | rc=2 | names `schema.rs:11` and `schema.rs:15` — no exemption, by design |
| `allowed-deps.txt` renamed away (vacuity proof) | rc=2 | `allowed-deps.txt is MISSING … this gate would report PASS while checking nothing` |

The first row is the positive allowlist earning its keep. A deny-list containing the string
`tempfile` would have said nothing whatsoever about the six-package subtree it drags in; the
allowlist names all seven because none of them was reviewed.

## Deviations from Plan

### Auto-fixed issues

**1. [Rule 1 — Bug] The plan's own Kani-backing test command cannot run**

- **Found during:** Task 2b.
- **Issue:** the plan specified `cargo test -p aprender-contrastive-data pairs canonical_pair_ordering`
  as the runnable evidence for KANI-CPP-001. Measured against this crate: exit 1,
  `error: unexpected argument 'canonical_pair_ordering' found`. `cargo test` accepts exactly one
  positional TESTNAME. Shipping it would have repeated, inside a contract, the exact defect plan
  02-01 fixed — a document naming an invocation that cannot run.
- **Fix:** single-filter form `cargo test -p aprender-contrastive-data canonical_pair_ordering`
  (verified rc=0), same for `capacity_no_overflow`, and a comment block above
  `falsification_tests:` recording the measurement and why the `--` form is the wrong tool (libtest
  treats multiple filters as a *union*, so it cannot name one test).
- **Commit:** `e97ffe7ca`

**2. [Rule 1 — Bug] The plan's own Task 2 gate contradicted the contract text it asked for**

- **Found during:** Task 2.
- **Issue:** the plan asked the contract to explain that `rand-philox-v1.yaml` is a dangling
  reference, while its automated gate requires `grep -c "rand-philox-v1"` to return **0**. The
  first draft failed its own gate.
- **Fix:** the metadata now makes the same point without the literal token — "the crate-level doc
  comment of aprender-rand names a Philox contract file under a `# Contract:` heading, but no file
  of that name exists anywhere in this repository". Meaning preserved, gate satisfied, and no
  dangling name is introduced into a file that other tooling greps.
- **Commit:** `9c1812768`

**3. [Rule 3 — Blocking] `cargo package` refuses a dirty tree**

- **Found during:** Task 1 verification, which must run before the commit that would clean the tree.
- **Fix:** ran `--allow-dirty` pre-commit to obtain the file list, then re-ran **without** it after
  the commit as the real clean-tree measurement (rc=0, 19 files). Both statuses recorded.
- **Commit:** `06cda1009`

### Scope additions

**4. `.planning/phases/…/deferred-items.md` created** to log the CB-510 vacuity finding below,
per the executor scope boundary (log out-of-scope discoveries, do not fix them).

## KNOWN-RED, EXPECTED, NOT A REGRESSION — and WIDER than the plan predicted

**`cargo package -p apr-cli` is red from this wave until the human-approved publish cascade.**
That was expected. What was **not** predicted by this plan, by plan 02-08, or by STATE.md's
caveat is that **`--no-verify` does not help**:

```
$ cargo package --no-verify -p apr-cli
error: failed to prepare local package for uploading
Caused by:
  no matching package named `aprender-contrastive-data` found
  location searched: crates.io index
```

`--no-verify` skips the *build* of the packaged crate; it does not skip **manifest resolution**,
and resolution is what rewrites the path dependency into a registry dependency that does not exist
yet. So the plan's acceptance criterion "both `cargo package --no-verify` runs exit 0" is
**falsified by measurement** for `apr-cli`.

**Control (CLAUDE.md rule 6 — vary the input before naming a cause):** with the
`aprender-contrastive-data = { workspace = true }` line temporarily removed from
`crates/apr-cli/Cargo.toml`, the identical command exits **0** and packages **581 files**. The
line was then restored and byte-compared against its backup (identical). The cause is
unambiguous and attributable.

**What IS green and must stay green:** `cargo package --no-verify -p aprender-contrastive-data`
(rc=0, 19 files, clean tree, post-commit).

**Consequence for downstream:** plan 02-08 and `/gsd:verify-work` must read **any**
`cargo package -p apr-cli` — verifying or not — as this expected state, not as a phase
regression. **Exit condition unchanged:** publish `aprender-contrastive-data` **before** `apr-cli`
in the cascade. That is a human-approved release action; CLAUDE.md forbids self-serving it, and
nothing was published here.

## Repo-Wide Gap Surfaced, Not Fixed

**Both CB-510 packaging guards are VACUOUS on macOS.** `scripts/check_include_files.sh:18` and
`scripts/check_package_includes.sh:25` use `grep -oP`. BSD grep has no `-P`; it exits 2 with
`grep: invalid option -- P`, the trailing `|| true` swallows it, and both scripts print
`OK: All 0 include!() files …` and **exit 0**. Measured with GNU grep (`ggrep`, present at
`/opt/homebrew/bin/ggrep`) the true count over `crates/` and `src/` is **1768**.

So on every macOS developer machine the two guards that exist to prevent the CB-510 publish break
report PASS while inspecting nothing. CI runs on Linux where `-P` works, so this is a local
false-green rather than a shipped hole — but a developer running `make tier3` before pushing is
being told something untrue. Logged as D-ITEM-01 in `deferred-items.md`; the fix is a repo-wide
shell-portability change needing its own ticket and its own must-match/must-not-match case table.

**Compensating evidence taken instead, for this crate specifically:** it contains **zero**
`include!()` macros (checked directly); all 16 new files appear in
`git ls-files --others --exclude-standard`; `git check-ignore` exits 1 for every one of them
(none ignored); and the produced `.crate` contains all 14 `src/*.rs` files plus
`allowed-deps.txt`. The CB-510 property this plan needed is established by direct evidence, not by
the guards.

## Binding-Audit Probe — the cross-crate question is ANSWERED

Baseline recorded (`/tmp/cpp-audit.log`): the mechanism reaches this contract and reports all
**24 equations BIND-001 unbound**, `Bound equations: 0`. That is the correct state before wave 3.

Then the throwaway probe the plan asked for — one entry with
`module_path: aprender_contrastive_data::hash` under `target_crate: aprender`, run, observed,
reverted:

> `Total equations: 24` / **`Bound equations: 1`** / `Not implemented: 1`, BIND-001 count
> **24 → 23**, and the remaining finding for that equation downgrades to
> `[WARN] BIND-004: Equation 'normalized_content_hash' … is pending implementation`.
> **No complaint about the crate, the namespace, or `target_crate`.**

**Verdict: ACCEPTED. Plan 02-08 proceeds as written**; neither candidate remedy is needed.
`git diff --stat contracts/aprender/binding.yaml` is empty and porcelain is clean — 02-08 remains
that file's single owner.

**Two traps found on the way, which 02-08 needs and would otherwise hit mid-wave:**

1. **The `contract:` field must be the BARE filename.** The first probe used
   `../contrastive-pair-protocol-v1.yaml` — the form several *existing* entries use for other
   contracts. It parses cleanly, raises nothing, and **binds nothing**: `Bound equations` stayed
   `0` and BIND-001 stayed `24`. A silent no-op inside what 02-08 makes a *blocking* gate is worse
   than an error, because the gate would then be green or red for a reason unrelated to the work.
2. **`status:` accepts only `implemented | partial | not_implemented | pending`.** `planned` is a
   hard parse error (`unknown variant 'planned'`). Use `pending` for equations landing in a later
   wave — it yields a BIND-004 *warning* rather than a BIND-001 *error*, which is exactly the
   gradation 02-08 wants.

A third, smaller note: append points matter. The first attempt appended at end-of-file and landed
inside `critical_path:` (which follows `bindings:` at line 854), producing
`invalid type: map, expected a string`. That was a defect in the probe, not a finding about the
registry.

**`make contract-audit` stays red, and it is not this plan's doing.** Control: the previous last
`$(CONTRACTS)` entry, `tweet-eval-stance-benchmark-v1.yaml`, audits red on its own (rc=1,
`official_f_avg` unbound), so the target was already failing before this change. Phase 1's setfit
contract adds 10 more. No tier calls `contract-audit`.

## Provability Honesty

`cargo-kani` is not installed and there is not one `#[kani::proof]` harness anywhere under
`crates/`. Both declared harnesses say so **in their own `property` prose** — "DECLARED AND NOT
EXECUTED HERE" — and each names the identically bounded (bound 4) proptest that is the actual
evidence, which is in turn a first-class falsification test (`FALSIFY-CPP-019`, `-020`) with its
own runnable, verified command. The `qa_gate` states that a harness whose named proptest is not
green is a FAILED gate, not a pending one. Nothing in the contract can be read as a claim of
machine-checked proof.

## Measurement Notes (CLAUDE.md Verification Discipline)

- **`make` output is abridged by the rtk hook.** The unproxied `make contract-validate` showed
  **16** "Contract is valid." lines for 44 contracts; `rtk proxy make` showed **44/44**. `rc` was
  0 in both cases — only the evidence was truncated, and a silently abridged log is a poor thing
  to cite. Every count in this SUMMARY was taken through `rtk proxy`.
- **`git status --porcelain` prints `ok` on a clean path** under the hook, inverting emptiness
  assertions. Every cleanliness check here ran through `rtk proxy git status --porcelain`,
  including the four post-mutation reverts.
- **Statuses were captured directly**, never read through a pipe: `cmd > /tmp/x.log 2>&1; rc=$?`.
  Several Bash invocations in this session reported a non-zero *outer* exit purely because a
  trailing `grep -c` found no matches; the `rc` values quoted here are the captured ones.

## Evidence

| Check | Result |
|---|---|
| `cargo check -p aprender-contrastive-data` | 0 |
| `cargo check -p apr-cli` | 0 |
| `cargo check -p apr-cli -p aprender-train -p aprender-contrastive-data --all-targets` | 0 |
| `cargo test -p apr-cli -p aprender-train -p aprender-contrastive-data --lib -- --skip gpu::` | **0 — 14035 passed, 0 failed** (matches the pre-plan baseline exactly) |
| `cargo test -p aprender-contrastive-data` | 0 — determinism doctest green |
| `cargo clippy -p aprender-contrastive-data --no-deps --all-targets -- -D warnings` | 0 |
| `cargo clippy -p apr-cli --no-deps -- -D warnings` | 0 |
| `cargo fmt -p aprender-contrastive-data -p apr-cli --check` | 0 |
| `cargo metadata --format-version 1` | 0; crate present |
| `cargo package --no-verify -p aprender-contrastive-data` (clean tree, post-commit) | 0 — 19 files, 55.1 KiB |
| `cargo package --no-verify -p apr-cli` | **1 — expected, control-verified (581 files / rc=0 without the dep)** |
| `scripts/check_include_files.sh` / `check_package_includes.sh` | 0 / 0 — **but vacuous on this host, see D-ITEM-01** |
| `git check-ignore` over all 16 new files | exit 1 each (none ignored) |
| `pv validate contracts/contrastive-pair-protocol-v1.yaml` after Task 2 | 1 — exactly 3 PROVABILITY-001 + 1 SCHEMA-013, **zero schema errors** |
| `pv validate contracts/contrastive-pair-protocol-v1.yaml` after Task 2b | **0 — 0 errors, 0 warnings** |
| `pv status` | v1.0.0; 24 equations, 15 obligations, 20 falsification tests, 2 kani, F-CONTRASTIVE-001 |
| `make contract-validate` | 0 — **44/44** valid, new contract reached |
| `make contrastive-data-boundary` standalone | 0, 1 s wall |
| the four gate mutations | rc=2 each, correct offender named, all reverted, tree byte-clean |
| `pv audit <contract> --binding` baseline | 1 — 24 BIND-001, mechanism confirmed reaching the contract |
| cross-crate `module_path` probe | **ACCEPTED** — bound 0→1, BIND-001 24→23, no namespace complaint; reverted |
| `git diff --stat contracts/aprender/binding.yaml` | empty |
| `make -n` on contrastive-data-boundary / tier2 / tier3 | 0 / 0 / 0 |
| tier2 line wall time (3 warm runs) | 2 s / 2 s / 2 s, rc=0 each |

## Threat Model Dispositions

| Threat | Disposition |
|---|---|
| T-02-04 supply chain via the dependency closure | **mitigated** — positive allowlist of 28 reviewed packages vs the resolved `-e normal` closure; failure observed (tempfile + 6 transitives named) |
| T-02-05 bytes-boundary bypass via fs/path APIs | **mitigated** — src/-wide symbol ban, no cfg(test) exemption; both failure modes observed; `OBLIG-CPP-BYTES-BOUNDARY` names the check |
| T-02-06 unreachable or vacuous contract gates | **mitigated** — contract in `$(CONTRACTS)` (44/44 reached), boundary target inside tier3 with D-26 evidence discipline, vacuity proof performed before the gate was trusted |
| T-02-35 binding audit assumed to work | **mitigated** — probed, baseline recorded, cross-crate path proven accepted, two traps documented for 02-08 |
| T-02-36 unbounded pair budget accepted at the API | **mitigated** — `budget_resolution` contracted non-bypassable; `ZeroBudget`, `ZeroHardCap`, `BudgetExceedsHardCap`, `BudgetExceedsCapacity`, `NoPairCapacity`, `OrdinalOutOfRange` all ship in the error surface |
| T-02-SC package tampering | **n/a** — zero packages installed; publishing remains a human-approved action and none was performed |

## Tools Used

`pv` for every contract operation (validate, status, audit) — no bash/yq/python workaround.
`rtk proxy` for all raw git/make/grep output. `cargo tree`, `cargo package`, `cargo metadata` for
the packaging and closure facts. Targeted `Read` on the files the plan cited by line number.
`pmat query` was **not** used: this plan created files rather than locating existing ones, and
every analog was pre-resolved by 02-PATTERNS.md with exact paths and line numbers.
`bashrs` is **not installed on this host**, so the new Makefile shell was reviewed by hand against
its rules instead (no `$?` through a pipe, every `$$var` quoted, no `ls` iteration, `while read`
fed by redirect not a pipe) and `make -n` used to prove it parses.

## Notes for the Next Plans

- **Do not edit `crates/aprender-contrastive-data/src/lib.rs`.** Every module is already declared.
  Fill your own stub; the doc comment in it names what it owns.
- **Consumers use module paths** (`aprender_contrastive_data::split::Split`), not crate-root
  re-exports. `ContrastiveDataError` is the one exception.
- **Adding an error variant is allowed** — record it in your SUMMARY and extend
  `OBLIG-CPP-ERROR-TAXONOMY`.
- **Every new dependency, including transitive, must be added to `allowed-deps.txt` deliberately**
  or tier3 fails. That is the intended friction.
- **Nothing under `src/` may name `std::fs`, `std::net`, `std::path`, `Path` or `PathBuf`** — not
  even in `#[cfg(test)]`. Filesystem-needing tests go in `tests/` or in `apr-cli`.
- **02-08:** bare filename in `contract:`, `pending` (not `planned`) in `status:`, and insert
  before `critical_path:` — not at end of file.

## Self-Check: PASSED

- `crates/aprender-contrastive-data/Cargo.toml` — FOUND, contains `aprender-rand`
- `crates/aprender-contrastive-data/allowed-deps.txt` — FOUND, contains `aprender-rand`, 28 entries, sorted, exactly equals the resolved closure
- `crates/aprender-contrastive-data/src/error.rs` — FOUND, contains `BudgetExceedsHardCap`; all 34 named variants present
- `crates/aprender-contrastive-data/src/lib.rs` — FOUND, contains `pub mod prepared`; 13 `pub mod` lines; `rand-philox-v1` count 0
- all 13 module files under `src/` — FOUND
- `contracts/contrastive-pair-protocol-v1.yaml` — FOUND, contains `proof_obligations`; pv-valid
- `Makefile` — FOUND, contains `contrastive-data-boundary`, the `$(CONTRACTS)` entry, the tier2 line and the tier3 invocation
- `.planning/phases/02-deterministic-pair-and-data-protocol/deferred-items.md` — FOUND
- commit `06cda1009` — FOUND
- commit `9c1812768` — FOUND
- commit `e97ffe7ca` — FOUND
- commit `c104221f4` — FOUND
