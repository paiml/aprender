# Phase 2 — Deferred Items

Out-of-scope discoveries surfaced during execution. Logged, not fixed (executor Scope
Boundary: only issues directly caused by the current task's changes are auto-fixed).

## D-ITEM-01 — Both CB-510 packaging guards are VACUOUS on macOS/BSD

**Found:** plan 02-02, Task 1 (2026-08-08). **Pre-existing**, not caused by this phase.

`scripts/check_include_files.sh:18` and `scripts/check_package_includes.sh:25` both use
GNU PCRE grep:

```sh
included=$(echo "$content" | grep -oP 'include!\(\s*"([^"]+)"\s*\)' | sed ... || true)
```

BSD `grep` (the default on Darwin) has no `-P`. It exits 2 with `grep: invalid option -- P`,
the `|| true` swallows the failure, `$included` is empty, and both scripts print

```
OK: All 0 include!() files are tracked by git
OK: All 0 include!() files are included in cargo package
```

and exit **0**. The true count, measured with GNU grep (`ggrep`, present at
`/opt/homebrew/bin/ggrep`) over `crates/` and `src/`, is **1768** `include!("...")`
occurrences. So on every macOS developer machine these two guards report PASS while
inspecting nothing — the exact "a guard that passes vacuously" failure class CLAUDE.md
Verification Discipline rules 1 and 5 describe, and the guards in question are the ones
that exist to prevent the CB-510 publish break.

CI runs on Linux, where `grep -P` works, so the guards are presumably real there. That
makes this a *local* false-green rather than a shipped hole — but a developer who runs
`make tier3` before pushing is being told something untrue.

**Why not fixed here:** the scripts are untouched by this plan, the fix is a repo-wide
shell-portability change (`grep -oE` with a POSIX-ERE rewrite, or a `grep -P`-capability
probe that hard-fails instead of degrading), and it needs its own bashrs lint pass plus a
must-match/must-not-match case table per CLAUDE.md rule 7. Worth its own PMAT ticket.

**Compensating measurement taken for plan 02-02:** `aprender-contrastive-data` contains
**zero** `include!()` macros (verified directly), and all 16 of its new files are visible
to git and to `cargo package` (verified via `git ls-files --others --exclude-standard`,
`git check-ignore` on each file, and the 19-entry `.crate` listing). The CB-510 property
this plan needed is therefore established by direct evidence, not by the vacuous guards.

## D-ITEM-02 — `make tier2`'s clippy step is RED on arm64 and unlintable in CI

**Found:** plan 02-03, final verification (2026-08-09). **Pre-existing**, not caused by
this phase.

`cargo clippy -- -D warnings` (tier2 step 2) fails with **25 errors across 5 crates**:

```
crates/aprender-compute      (19)  backends/q4k, backends/q6k, blis/*, brick/*, hardware, vector/ops
crates/aprender-zram-core     (3)  src/lz4/neon.rs:246,303,318 — ptr_as_ptr
crates/aprender-core          (1)  src/demo/reliable/performance.rs:126
crates/aprender-present-terminal (1) src/compute_block.rs:93
crates/aprender-serve         (1)  src/quantize/simd_backend.rs:47
```

Zero are in `aprender-contrastive-data`. Every location is inside **architecture-gated
SIMD code**: the lints are `unused_imports` / `dead_code` / `unreachable_expression` /
`unused_variables` on the x86 side (`MR_512V2`, `NR_512V2`, `matmul_q4k_f32_parallel`,
`pack_b_block_nr16`, …) plus `ptr_as_ptr` on the NEON side. Host is `arm64`; on arm64 the
x86 dispatch arms become dead and the NEON arms become live, so a different set of code
is linted than on x86_64.

**Proof it is independent of this plan** (not inferred — measured):
- `cargo clippy -p aprender-zram-core -- -D warnings` → **rc=101, 4 errors**, and
  `cargo tree -p aprender-zram-core` contains **zero** references to
  `aprender-contrastive-data`. The failure reproduces with this phase's crate entirely
  absent from the dependency graph.
- `git log d50a0d818^..HEAD -- crates/aprender-zram-core crates/aprender-compute` is
  **empty**: no commit in this plan touched any failing crate. `src/lz4/neon.rs` last
  changed in `47d63b434`.

**Why CI never catches it:** every `runs-on` in `.github/workflows/ci.yml` is
`[self-hosted, X64, Linux, clean-room]`. The aarch64-live arms are therefore **never
clippy-linted by CI at all**, so `ci / gate` is green while every arm64 developer's
`make tier2` is red. This is CLAUDE.md Verification Discipline rule 5 — the guard does not
scan the surface where the decision is made — with the twist that the *unscanned surface*
is an entire CPU architecture.

**Why not fixed here:** 25 errors across 5 crates untouched by this plan, all in SIMD
dispatch that needs per-arch review (several are genuinely dead x86 helpers that may want
`#[cfg]` gating rather than an allow). Needs its own PMAT ticket, plus an arm64 CI lane —
fixing the lints without adding the lane just means they silently return.

**Compensating measurement for plan 02-03:** every OTHER tier2 step was run individually
and passed, so the only red is the pre-existing one above:

| tier2 step | rc | result |
|---|---|---|
| `cargo test --lib` (root facade) | 0 | 0 tests — see D-ITEM-03 |
| `cargo clippy -- -D warnings` | **2** | **25 pre-existing arm64 errors** |
| setfit lib gate | 0 | 162 passed |
| setfit conformance gate | 0 | 27 passed, 1 ignored |
| `cargo test -p aprender-contrastive-data` | 0 | 72 passed (67 lib + 5 doc) |

Additionally `cargo clippy -p aprender-contrastive-data --all-targets -- -D warnings`
is **rc=0**, so this plan's own code is clippy-clean at the strict setting.

## D-ITEM-03 — tier2's headline `cargo test --lib` runs ZERO tests

**Found:** plan 02-03, final verification (2026-08-09). **Pre-existing.**

`make tier2` step 1 is `PROPTEST_CASES=5 QUICKCHECK_TESTS=5 cargo test --lib`. Run from
the workspace root, that selects **only the root facade package** (`[lib] name =
"aprender"`, `path = "src/lib.rs"`, Cargo.toml:561) — not the workspace. Measured output:

```
Running unittests src/lib.rs (target/debug/deps/aprender-f0487d1dd47661b1)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

The line that reads like "run the unit tests before committing" executes **zero** of the
repo's 25,300+ tests. The real coverage in tier2 comes only from the three explicitly
`-p`-scoped gate lines appended below it. Anyone reading the Makefile — or trusting a
green tier2 — would reasonably believe the workspace lib suite had run.

`--workspace` is the obvious fix but is NOT a drive-by: it turns a sub-second step into a
multi-minute one and would change the pre-commit tier's cost profile, which the existing
tier2 comments show was deliberately measured. Needs a decision (widen tier2, or move the
workspace suite to tier3 and rename this step honestly), so it gets its own ticket.

## D-ITEM-04 — `make contract-audit` reports 132 unbound equations and exits 0

**Found:** plan 02-08, Task 3 (2026-08-09). **Pre-existing**, not caused by this phase.

The repo-wide binding-coverage target (`Makefile`, `contract-audit`) iterates all 44
contracts in `$(CONTRACTS)` and its loop body is:

```make
	@for contract in $(CONTRACTS); do \
		echo ""; \
		$(PV_BIN) audit "$$contract" --binding $(BINDING); \
	done
```

The audit's status is never read — no `|| exit 1`, no captured `$$?` — so the target's exit
code is that of the trailing `echo`. Measured directly
(`make contract-audit > /tmp/ca-repo.log 2>&1; rc=$?`, never through a pipe):

```
rc=0
132 [ERROR] BIND-001 lines, across 38 of the 44 contracts
```

Attributed by contract, the ten largest: `setfit-encoder-conformance-v1` 10,
`hybrid-layer-dispatch-v1` 6, then `model-config-algebra-v1`, `tensor-shape-flow-v1`,
`gated-delta-net-v1`, `tensor-inventory-v1`, `performance-grading-v1`, `lora-algebra-v1`,
`q4k-q6k-superblock-v1`, `sampling-algorithms-v1`, `qwen35-shapes-v1` and
`kv-cache-sizing-v1` at 5 each. **Neither Phase 2 contract appears** — both are fully bound
as of this plan.

So the target that exists to prove equations are implemented prints 132 failures and
reports success. No tier calls it, which limits the damage, but `make contract-check`
(`contract-validate contract-test contract-audit`) does — and it too would report PASS.

**Why not fixed here:** the one-line fix (read the status) immediately turns
`contract-check` red on 132 pre-existing unbound equations spread over 38 contracts this
phase does not own, and binding them is a repo-wide archaeology exercise per kernel. Plan
02-08 therefore added the SCOPED `contract-audit-phase2` — blocking, wired into tier3, its
failure mode induced and observed — and left the broad target alone. The honest repo-wide
fix is: make `contract-audit` read its status, and either bind the 132 or record each with
`status: pending` (a BIND-004 warning rather than a BIND-001 error). Own PMAT ticket.
