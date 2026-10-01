# Deferred items — Phase 06

Out-of-scope discoveries surfaced while executing this phase. Per the executor scope
boundary these were **observed and logged, not fixed**: they are pre-existing, unrelated to
the task that surfaced them, and fixing them inside an unrelated commit would hide who
broke what.

## From plan 06-01 (2026-09-05)

### D-ITEM-06-01-a — `readme_contract` was already red on three counts before Phase 6

`cargo test -p aprender-core --test readme_contract` fails 4/15 on `440d7009a` (the
pre-plan commit). Plan 06-01 fixed the ONE it moved (`FALSIFY-README-005`, the workspace
crate count, which went 82→85 because this plan adds two members) and left the other
three, all of which predate Phase 6 and are untouched by it:

| Gate | Finding | Why not fixed here |
|---|---|---|
| `FALSIFY-README-007` | `README.md` claims **1778** provable contracts; `find contracts -name '*.yaml' \| wc -l` reports **1786** | This plan adds no contract; the drift is someone else's and dates from before it |
| `FALSIFY-README-CRATE-001` | `aprender-mcp-setfit-lambda` and `aprender-contrastive-data` have no `README.md` | Neither crate is touched by Phase 6 |
| `FALSIFY-README-CRATE-002` | `crates/aprender-mcp-setfit/README.md` lacks the `paiml/aprender` monorepo link | Phase 6's own two READMEs both carry it; 06-RESEARCH F7 already records this one as known |

Both Phase 6 crates satisfy all four gates. Net effect of 06-01 on this test: 4 failures →
3 failures.

### D-ITEM-06-01-b — `cargo clippy -p <crate> -- -D warnings` cannot pass for ANY crate in this tree

The plan's `<verify>` block runs
`cargo clippy -p aprender-forecast -p aprender-mcp-forecast --all-targets -- -D warnings`.
On this toolchain (1.93.0) the `-D warnings` flag reaches **path dependencies**, so the
command fails with 18 errors inside `crates/aprender-compute` — unused imports, unreachable
expressions, `dead_code`, unused variables.

Measured control, so this is not a guess about my own code: the same command against the
untouched, pre-existing `aprender-mcp-setfit` fails identically (`rc=101`, same 18
`aprender-compute` findings), as does `-p aprender-serve` (failing inside
`aprender-present-terminal`). The condition is workspace-wide and predates Phase 6.

06-01 therefore gated its crates with `--no-deps`, which scopes the lint to the primary
packages — the thing the criterion was trying to measure. Engagement was PROVEN, not
assumed: a `clippy::needless_bool` mutation inserted into `aprender-forecast/src/lib.rs`
turned the `--no-deps` command red (`rc=101`, `needless_bool` cited), and removing it
turned it green again.

Not fixed here because the 18 findings are in a crate this phase does not touch, and
because clearing them is a real piece of work with its own blast radius
(`aprender-compute` is depended on by most of the workspace). Worth a dedicated ticket:
either clean the crate or record why those lints are allowed there.

### D-ITEM-06-02-a — `aprender-core`'s own lib has a pre-existing `unreachable_code` error under `-D warnings`

Found while checking that plan 06-02's one-file test edit would not fail a lint gate.
`cargo clippy -p aprender-core --test monorepo_invariants --no-deps -- -D warnings` exits
101 on:

```
error: unreachable expression
   --> crates/aprender-core/src/demo/reliable/performance.rs:126:5
124 |         return "NEON".to_string();
126 |     "Scalar".to_string()
```

`--no-deps` does NOT rescue this one, because the failing code is in the primary package's
own lib, which the test target links against.

Measured control rather than assumed: the identical command against the **untouched**
sibling target `--test readme_contract` fails with the same single error (`rc=101`), and
06-02 modified no file under `crates/aprender-core/src/`. The finding predates this plan.

Not fixed here: it is a `cfg`-shaped early return in a demo module that no Phase 6 task
touches, and the fix (an `#[allow]`, a `cfg` restructure, or deleting the dead branch) is a
judgement call belonging to whoever owns `src/demo/`. Note that the 06-01 wording of
D-ITEM-06-01-b — "cannot pass for ANY crate in this tree" — has a second cause: not just
`-D warnings` reaching path dependencies, but `aprender-core`'s own lib.

---

## D-ITEM-06-03-a — every `cargo test` invocation in this workspace pays a ~12 s forced rebuild

**Found during:** plan 06-03 Task 2, measuring the warm parity-ladder wall (RESEARCH Pitfall 9).

**Measured, not inferred.** Three consecutive `cargo test -p aprender-forecast --lib
prophet::parity` invocations with `CARGO_INCREMENTAL=0`, no source edit between them:

| Invocation | cargo wall | `Finished ... in` | test execution |
|---|---|---|---|
| warm run 1 | 14 s | 12.21 s | 1.73 s |
| warm run 2 | 14 s | 12.20 s | 1.78 s |
| the same 32 tests, running `target/debug/deps/aprender_forecast-*` DIRECTLY | — | — | 1.77–1.79 s |

Every run recompiles `aprender-compute`, `aprender-core` and `aprender-forecast` even though
nothing changed. Both `crates/aprender-compute/build.rs` and `crates/aprender-core/build.rs`
exist; one or both is missing a `cargo:rerun-if-changed` (or emits an always-changing key), so
cargo can never call the unit fresh.

**Why it matters beyond this plan.** 06-01 Task 2's profile decision was taken on a
per-invocation wall (`13 s`) and projected as `7 x 13 = 91 s` for the ladder. Re-measured this
session, that 13 s proxy is **11.73 s of forced rebuild + 1.64 s of test** — so the projection
multiplied a CONSTANT per-invocation build cost seven times. The real ladder is ONE invocation:
1.73 s of tests, 14 s wall. The `[profile.dev.package.aprender-forecast] opt-level = 3` decision
is still correct (the fits genuinely need it), but the "~91 s against a 60 s target" concern
06-01 left for this plan is **closed as a measurement artefact**, not as an optimisation win.

**Not fixed here:** the build scripts belong to `aprender-compute` and `aprender-core`, which
this plan does not touch, and diagnosing which key is unstable is its own task. Owner: 06-09.

## From plan 06-06 (2026-09-06)

### D-ITEM-06-01-a is now CLOSED (not deferred any further)

`cargo test -p aprender-core --test readme_contract` is **15 passed / 0 failed** as of
`14eceff67`. The three failures 06-01 logged were closed one at a time by the plan that
moved each count:

| Gate | Closed by |
|---|---|
| `FALSIFY-README-CRATE-001` / `-002` | closed between 06-01 and 06-06 (both crate READMEs and the monorepo link now present) |
| `FALSIFY-README-007` (contract count) | 06-06 — adding `contracts/forecast-tool-boundary-v1.yaml` moved `find contracts -name '*.yaml'` to 1790, so this plan's edit turned it red and this plan corrected the row |

Nothing here is deferred; the row is left as the audit trail for the phase.

### D-ITEM-06-06-a — `.pv/contracts.idx`, `.pv/contracts.idx.mtime`, `.pv/lint-previous.json` are dirty in the working tree

These three TRACKED files are `pv`'s local index cache and are rewritten by any `pv`
invocation, including a read-only `pv validate`. They were **already modified before this
plan's first command** (visible in the pre-plan `git status`), so the drift predates
06-06; running `pv` here only advanced it further.

NOT committed by this plan: a cache blob is not a reviewable artifact, and committing one
inside a contract commit would make every future `pv` run produce a spurious diff for the
next author. The real question — should `.pv/` be tracked at all, or `.gitignore`d like
other tool caches — belongs to whoever owns `pv`, not to a forecasting plan.

### D-ITEM-06-07-a — the router pool is deferred as an advanced feature; its gate is disarmed, not passing

**Owner decision (2026-09-06):** the pmcp router pool (`pooled_app`, `--pool K`) is an
advanced feature held for a later stage. It stays in the tree — it is measured, correct and
opt-out (`pool <= 1` returns the plain `http_app`) — but nothing gates its *engagement*.

What is deferred, precisely:

- `just forecast-pool-ratio` **does not exist**. `contracts/forecast-tool-boundary-v1.yaml`
  cited it as the test for `FALSIFY-BOUNDARY-011`, so that test could never run. It is now
  marked `status: deferred` and removed from the contract's `guarantee` block, because a
  gate that cannot evaluate must not be counted as green.
- `pool_equality` asserts **determinism, not engagement**. Both concurrency tests pass
  bit-identically (8/8, 16/16, `max|Δ yhat| = 0.0`) even with the pool collapsed to a single
  router — a stateless router is trivially deterministic. So no test in CI would notice the
  pool silently reverting to one instance.

What is NOT deferred and still holds:

- `FALSIFY-BOUNDARY-012` — the `POOL SPEEDUP:` line and its `arch`/`profile`/`workers`/`cpus`
  provenance fields are printed and format-checked by the unit test today. Arming the gate
  later needs the recipe only, not a re-derivation of the format.
- The measured evidence stands and is reproducible: `POOL=1 -> 1.002x`, `POOL=8 -> 2.070x`,
  same host, same eight requests. The `1.002x` control is what makes the other number
  evidence rather than an assertion, and it reproduces spike-010's finding that pmcp's
  router mutex — not CPU — serialises concurrent fits.

To arm it later: add the `forecast-pool-ratio` recipe (release build, aarch64, three runs,
parse the single `POOL SPEEDUP:` line, assert best ratio >= 2.0), flip
`FALSIFY-BOUNDARY-011` off `status: deferred`, and restore it to the `guarantee` block.
The upstream fix — pmcp not holding `Arc<Mutex<Server>>` across the whole tool future —
would delete the pool entirely; that is the better long-term resolution.

---

## From plan 06-08 (host-gated evidence recipes) — 2026-09-06

### 1. `FALSIFY-BOUNDARY-011` can now be re-armed (was blocked on this plan)

The note above says the gate was deferred because "`just forecast-pool-ratio` **does not
exist**". It exists as of commit `adc8a560a` and PASSES on aarch64 release: three attempts at
5.150 / 5.162 / 5.146, best 5.162x against a 2.0 bar (06-EVIDENCE.md §3). The remaining work
is contract-side only — flip `FALSIFY-BOUNDARY-011` off `status: deferred` in
`contracts/forecast-tool-boundary-v1.yaml` and restore it to the `guarantee` block. Not done
here because that file is outside this plan's `files_modified` and already carries
uncommitted edits from an earlier review pass; 06-09 is the natural home.

Caveat to carry with it: the bar is **host-gated**. `forecast-pool-ratio` measures a release
build on aarch64; the CI runner is X64 and builds debug. The contract should say which host
the gate is claimed on, or CI will be asked to assert a number nobody measured there.

### 2. PRE-EXISTING: `cargo clippy -- -D warnings` is red workspace-wide on aarch64 macOS

Not caused by this plan; recorded because plan 06-08 Task 1's verify tripped over it and the
finding would otherwise be lost.

`cargo clippy -p aprender-forecast --all-targets -- -D warnings` exits 101 with **18 errors,
all in `crates/aprender-compute/`** and none in the crate being linted. Confirmed by control:
the same command with the new example removed from the tree produces the identical 18-finding
set, and `cargo clippy -p aprender-forecast --example mase_rolling_origin -- -D warnings`
reports the same 18, again none in `aprender-forecast`. (Mechanism: trailing `-- -D warnings`
becomes `CLIPPY_ARGS`, which clippy applies to every locally-compiled crate, not just the
selected package.)

The findings:

- `unreachable expression` x3 — `blis/backend_selection.rs:127`, `brick/simd_config/mod.rs:88`,
  `hardware/mod.rs:375`. Each follows an unconditional `return ComputeBackend::Neon` /
  `SimdWidth::Neon128` that is **cfg-gated to aarch64**, so the trailing scalar fallback is
  live code on x86_64 and dead here. **This class is arch-conditional and therefore invisible
  to CI**, which is X64 — the same shape as CLAUDE.md #2370's "findings accumulate where no
  gate looks".
- `dead_code` x9 — `pack_a_block_generic`, `pack_b_block_generic`, `pack_b_block_nr16`,
  `matmul_q4k_f32_parallel`, `compute_chunk_q4k_scalar`, `compute_chunk_scalar`,
  `extract_q6k_values`, `PREFETCH_DISTANCE`, `NT_STORE_THRESHOLD_BYTES`, `GEMV_TILE_THRESHOLD`.
- `unused_imports` x3 — `q4k/gemv/mod.rs:14`, `blis/packing.rs:419`, `vector/ops/rounding.rs:9`.
- `unused_variables` x2 — `brick/quant_ops/mod.rs:219,318` (`backend`).

Consequence worth stating plainly: **`make tier1` / `make tier2` cannot pass on an aarch64
macOS dev box today.** Whoever picks this up should fix `aprender-compute`, not add
`#[allow]`s, and should re-run on BOTH arches — the point of the finding is that one arch's
green says nothing about the other's.

### 3. PRE-EXISTING: substantial uncommitted phase-06 work in the tree

At plan 06-08's start the working tree carried 14 modified source files this plan did not
create — `crates/aprender-forecast/src/{forecast,np,prophet,safetensors,test_support,types,
chronos}.rs`, `build.rs`, `crates/aprender-mcp-forecast/src/{lib,main}.rs`,
`contracts/forecast-tool-boundary-v1.yaml`,
`crates/aprender-core/tests/monorepo_invariants.rs`. They look like review-hardening from a
pass over 06-06/06-07 (a `MAX_SPAN_DAYS` bound, cross-model option refusals, a `MAX_POOL`
ceiling) that was never committed: `git show HEAD:crates/aprender-forecast/src/forecast.rs`
has no `MAX_SPAN_DAYS`.

06-08 did not commit them (out of scope) but every measurement in 06-EVIDENCE.md ran against
them, which is why that file records the delta's sha256 beside the commit hash. **Someone
should decide whether that work lands or reverts before the phase closes** — right now the
branch's committed state and its tested state are different things.

### 4. OPEN ITEM for 06-09 — D-18 clause 2 in CI, gated on one x86_64 measurement

Decision at plan 06-08 Task 3's blocking-human checkpoint: **`measure-x86-first`**. Neither
hunk of `06-ci-chronos-step.patch` is applied in this phase by default. The patch is
preserved, verified to apply cleanly against ci.yml as of `adc8a560a`, and `.github/` is
untouched in the working tree and in every commit of plan 06-08.

**Why not `apply-now`.** Hunk (a) has two prerequisites nobody has provisioned, and a step
that dies at the weight-hash check before running a single test is a red run that proves
nothing:

1. a runner-local weights mount (`/srv/models` in the diff is a PLACEHOLDER), and
2. a way for `uv run --with huggingface_hub --with safetensors --with numpy` to resolve
   inside the network-less `sovereign-ci:stable` clean-room — a warm uv-cache mount
   (`/srv/uv-cache` is proposed) or those three packages baked into the image.

**The gating work, which is this phase's own open item and not new scope.** Run
`just chronos-gate` ONCE on an x86_64 Linux host with the weights (lambda-vector qualifies
and is pre-authorized compute per CLAUDE.md). Read two things off it: the printed
`f32_quantile_bar` line, and the measured max|delta| from
`peyton_ladder_matches_oracle_f32`. Then tighten `quantiles_abs_f32_nonaarch64` in
`contracts/chronos-bolt-parity-v1.yaml` from the PROVISIONAL-UNMEASURED `5.0e-6` to
measurement + margin, as a `pv diff`-visible contract edit. That single run discharges
REVIEW-06-02 and windows-ledger entry #4 together, and only then is there evidence to wire
a CI leg onto. The aarch64 evidence in `06-EVIDENCE.md` is unaffected either way.

**If no x86_64 host is reachable within this phase, this collapses to `defer`** — by plan
06-08's own option table. Record it that way explicitly: both hunks stay in the patch file,
and **D-18 clause 2 is a named CI gap**, evidenced locally by `just chronos-gate` and
`06-EVIDENCE.md` §4 only. It must not become a silent drop.

Hunk (b) (`cargo test -p aprender-mcp-forecast --test e2e_stdio` on the single
Integration-tests line) needs no runner provisioning and can be taken independently at any
time; it was not split out here because the decision was to gate on the measurement first.

### 5. OPEN ITEM for 06-09 — account for the uncommitted delta BEFORE committing it

Follow-on to item 3 above. Decision: **investigate, then commit** — not commit blind. The 18
uncommitted paths (14 source/doc + 3 `.pv/` artifacts, 510 insertions / 164 deletions) were
present in the session-start `git status` for plan 06-08, so they are neither 06-08's nor
06-07's. They are concentrated in exactly the crates 06-08 measured.

06-09 must establish their provenance and intent first, then commit them with a message that
says where they came from. Until that lands, `06-EVIDENCE.md`'s numbers are reproducible only
from `adc8a560a` **plus** a recorded sha256 delta
(`bd42a46dda0f0c660cc450b1c97d50f2c017d9111a6164a73272793b7dbcb2b6`, source-only) rather than
from a real commit. The delta digest stays as the honest record until then; it is not a
substitute for landing the work.

---

# Phase 06 close-out register (plan 06-09)

Everything above was logged AS IT WAS FOUND, keyed by the plan that found it
(`D-ITEM-06-<plan>-<letter>`). This section is the phase's CLOSING register, keyed by
DEFERRAL TOPIC (`D-ITEM-06-01` .. `D-ITEM-06-08`) as 06-09's plan specifies. **The two
numbering schemes overlap in text and mean different things** — `D-ITEM-06-03-a` above is
"the third plan's first finding" (cargo rebuild cost), while `D-ITEM-06-03` below is "the
third deferral topic" (the CI embedded-weights leg). Stated here rather than silently
renumbered, because renaming a record that other documents already cite is worse than an
explained collision.

Every entry carries its source, so the next milestone can start from the evidence rather
than from the claim.

## D-ITEM-06-01 — the two Lambda wrapper crates are deferred

`aprender-mcp-forecast-lambda` and `aprender-mcp-chronos-lambda` (thin copies of
`crates/aprender-mcp-setfit-lambda`) were explicitly Claude's discretion — 06-CONTEXT.md
"Claude's Discretion": "include if they fit in one plan". They did not, for two reasons that
are structural rather than budgetary:

1. Each wrapper adds a `bootstrap` `[[bin]]`, and every binary in this workspace has to be
   registered in `crates/aprender-core/tests/monorepo_invariants.rs`. Plan 06-02 made that
   register a POLICY decision (`allowed_bins` = migration debt vs `deployment_unit_bins` =
   `publish = false` thin servers), and 06-09 landed FALSIFY-MONO-011, which now also
   enforces `publish = false` on the deployment-unit register. Adding two names is a
   decision, not a same-PR edit — the invariant test says so in its own failure message.
2. `cargo pmcp deploy`'s `*-lambda` discovery is ambiguous with more than one candidate
   (06-RESEARCH Pitfall 11), so shipping two wrappers without resolving that would ship a
   deploy step nobody can predict.

Source: 06-CONTEXT.md "Claude's Discretion"; 06-RESEARCH.md Pitfall 11; 06-02 (the register).

## D-ITEM-06-02 — core `nn::loss::SmoothL1Loss` is detached from the autograd graph

**Filed, not fixed: https://github.com/paiml/aprender/issues/3034**

`crates/aprender-core/src/nn/loss.rs:170-195` computes the loss through a raw `Vec<f32>` map
and then builds a FRESH tensor from that data:

```rust
        let diff = pred.sub(target);
        let loss_data: Vec<f32> = diff
            .data()
            .iter()
            .map(|&x| {
                let abs_x = x.abs();
                if abs_x < self.beta {
                    0.5 * x * x / self.beta
                } else {
                    abs_x - 0.5 * self.beta
                }
            })
            .collect();

        let loss = Tensor::new(&loss_data, pred.shape());   // <-- the graph ends here
```

`diff` is on the graph; `diff.data()` leaves it; `Tensor::new` starts a new parentless tensor.
Backward through it contributes nothing, so a model trained with this loss receives zero
gradient from it and every value-based test still passes.

Workaround in the tree today: `aprender_forecast::np::weighted_huber`
(`crates/aprender-forecast/src/np.rs`), built from differentiable ops, asserted connected by
`np::tests::weighted_huber_is_graph_connected`, and pinned by the `huber_built_from_ops`
equation in `contracts/neuralprophet-parity-v1.yaml`. That workaround lives in a forecasting
crate and is not a fix for core.

Proposed fix, in the issue: compose from ops — `diff.abs()`,
`diff.pow(2.0).mul_scalar(0.5 / beta)`, `abs_diff.sub_scalar(0.5 * beta)`, selected by a
CONSTANT mask (the mask is data-dependent but not differentiable; the two branches must be).

**The gate matters more than the one fix:** a connectivity test for EVERY loss in `nn::loss`,
because "returns a plausible number and a dead graph" is invisible to value assertions.

NOT fixed in Phase 6 by explicit prohibition (06-09 plan): touching `aprender-core` pulls its
full lib suite into every cycle of a forecasting plan.

Source: 06-CONTEXT.md D-10 and "Claude's Discretion"; 06-RESEARCH.md Open Question 4; spike 002.

## D-ITEM-06-03 — the CI embedded-weights leg (D-18 clause 2), and the x86_64 measurement it waits on

**CI decision, read from 06-08-SUMMARY.md's single bare-token line: `measure-x86-first`.**
(Verbatim reply line, also in that SUMMARY: `CI decision (verbatim): CI decision:
measure-x86-first`. No `CI mount:` line was recorded — mount path used: **none given**.)

**Applied exactly, and applying it means NOT editing ci.yml.** `.github/` is provably
untouched by plan 06-09: both `git diff --name-only <plan-start>..HEAD -- .github` and
`git diff --name-only HEAD -- .github` are EMPTY, as they were for 06-07 and 06-08. The
06-09 plan's `files_modified` lists `.github/workflows/ci.yml` because the plan was written
before the decision existed; the recorded human decision supersedes that entry. This is a
deliberate non-edit, not a missed task.

Both hunks stay preserved in
`.planning/phases/06-native-time-series-forecasting-stack/06-ci-chronos-step.patch`, verified
by 06-09 to still pass `git apply --check` (rc=0) against ci.yml at plan-start
`18758190ba853665e39aec9b02d0c362e83f4aa7`.

**D-18 clause 2 is therefore a NAMED CI GAP, not a silent drop.** It is evidenced locally by
`just chronos-gate` and `06-EVIDENCE.md` §4 only. It carries three names for one obligation —
this entry (`D-ITEM-06-03`), **REVIEW-06-02**, and **windows-ledger entry #4** — and closing
any one of them closes all three. Do not treat them as three items.

The gating work, in order:

1. Run `just chronos-gate` ONCE on an x86_64 Linux host with the weights (lambda-vector
   qualifies and is pre-authorized compute per CLAUDE.md).
2. Read TWO things off it: the printed `f32_quantile_bar` line, and the measured max|delta|
   from `peyton_ladder_matches_oracle_f32`.
3. Tighten `quantiles_abs_f32_nonaarch64` (see D-ITEM-06-04) to measurement + margin, as a
   `pv diff`-visible contract edit.
4. THEN decide on the patch. Only after step 3 is there evidence to wire a CI leg onto.

Why not `apply-now` (06-08's finding, recorded so it is not rediscovered): hunk (a) has TWO
unprovisioned prerequisites — a runner-local weights mount (`/srv/models` in the diff is a
PLACEHOLDER) and a way for `uv run --with huggingface_hub --with safetensors --with numpy` to
resolve inside the network-less `sovereign-ci:stable` clean-room (a warm uv-cache mount, or
those three packages baked into the image). A step that dies at the weight-hash check before
running a single test is a red run that proves nothing.

Hunk (b) — `cargo test -p aprender-mcp-forecast --test e2e_stdio` appended to the single
Integration-tests line — needs NO runner provisioning and can be taken independently at any
time. See D-ITEM-06-06.

If no x86_64 host is reachable, this collapses to `defer` by 06-08's own option table, with
both hunks still in the patch file.

Source: 06-08-SUMMARY.md (`CI decision:`); 06-08 Task 3's blocking-human checkpoint;
06-CONTEXT.md D-18; REVIEW-06-02.

## D-ITEM-06-04 — the Chronos f32 tolerance is measured on aarch64 and PROVISIONAL everywhere else

The frozen bar `quantiles_abs_f32 = 1.0e-6` was measured on **aarch64 with the NEON kernel**:
Peyton Manning max|delta| = **9.5367e-7**, i.e. ~4.6 % margin (06-05). Every CI job in this
repo runs on `[self-hosted, X64, Linux, clean-room]`, where nothing has been measured.

Plan 06-05 therefore added a second, arch-keyed equation to
`contracts/chronos-bolt-parity-v1.yaml`:

    quantiles_abs_f32_nonaarch64 = 5.0e-6    <- PROVISIONAL, UNMEASURED

5x headroom, chosen so a first x86_64 run reports a NUMBER rather than a failure of unknown
size. The contract says so in those words. **A provisional bar is not a passing bar** — it is
a stated unknown, and it is the only equation in the phase whose value is not evidence.

The exact command that produces the missing number, on an x86_64 host:

    CHRONOS_MODEL_DIR=<f32 dir> cargo test -p aprender-forecast --lib \
        -- bolt::parity chronos::parity -- --nocapture

Read the printed `f32_quantile_bar` line and the measured max|delta|, then tighten
`quantiles_abs_f32_nonaarch64` to measurement + margin as a `pv diff`-visible contract edit.
(`just chronos-gate` wraps the same run with the weight re-verification — see D-ITEM-06-03.)

The aarch64 evidence in `06-EVIDENCE.md` is unaffected either way.

Source: 06-05-SUMMARY.md; REVIEW-06-02; `contracts/chronos-bolt-parity-v1.yaml`.

## D-ITEM-06-05 — everything 06-CONTEXT.md listed as out of scope, one line each

Product / scope tiers:

- **Chronos-2 tier** of `aprender-mcp-chronos` — 21 quantiles, 1024-step direct horizon, 8192
  context, 228 MB f16, ~0.5 s/forecast, and it needs a streaming loader to avoid a 1.45 GB
  load peak (spike 009). MANIFEST: "Chronos-Bolt first, Chronos-2 later".
  Source: 06-CONTEXT.md Deferred Ideas.
- **`model: auto` routing** ("monthly or <= 64 steps -> Chronos, else NeuralProphet/Prophet") —
  the policy is MEASURED (spike 006) but it spans two servers, so it is a product decision,
  documented not built. Source: 06-CONTEXT.md Deferred Ideas / Not in scope.
- **`freq` H** — needs fractional days through the whole Prophet pipeline; `future_days`
  refuses H today with a message (D-17). Not spiked. Source: 06-CONTEXT.md Not in scope.
- **Country-holiday calendars** — not spiked. Source: 06-CONTEXT.md Not in scope.
- **NeuralProphet quantile regression and `n_forecasts > 1`** — not spiked; the NP arm's band
  is residual-based today and its diagnostics say so rather than implying otherwise.
  Source: 06-CONTEXT.md Not in scope; `crates/aprender-forecast/src/forecast.rs`.
- **Multivariate / covariate inputs** — not spiked. Source: 06-CONTEXT.md Not in scope.
- **`apr forecast` CLI subcommand** — touches `contracts/apr-cli-commands-v1.yaml` and the
  111-command registry, so it needs its own ticket and its own contract edit.
  Source: 06-CONTEXT.md Not in scope.

Upstream / follow-up items surfaced by the spikes:

- **Opening the upstream NEON PR** — a HUMAN CHECKPOINT by D-05 and MANIFEST, explicitly not
  a Phase 6 task. The kernel is assumed and unmodified by this phase.
  Source: 06-CONTEXT.md D-05.
- **pmcp router-lock report to paiml/pmcp** — pmcp 2.19's streamable-HTTP router holds one
  `Arc<Mutex<Server>>` across the whole tool future, which serialises concurrent fits. The
  router pool exists only to work around it; the upstream fix would DELETE the pool, which is
  the better long-term resolution (see D-ITEM-06-07-a above).
  Source: 06-CONTEXT.md D-12 / Deferred Ideas; spike 010.
- **Parallel-GEMM N-split for M <= 256** — Source: 06-CONTEXT.md Deferred Ideas.
- **Streaming safetensors loader** — the prerequisite for the Chronos-2 tier.
  Source: 06-CONTEXT.md Deferred Ideas.
- **8x12 NEON tile** (the current kernel is 8x6) — Source: 06-CONTEXT.md Deferred Ideas.
- **`test_brick_profiler_reset_v2` timer-resolution flake** — Source: 06-CONTEXT.md Deferred
  Ideas.
- **Other core fixes the spikes surfaced**, beyond D-ITEM-06-02: `where` / `clamp` / `cat`
  autograd ops; `WolfeSearch` initial-step scaling and non-finite backtracking; `LbfgsF64`
  double evaluation at `x`; an f64 proximal solver. Source: 06-CONTEXT.md Deferred Ideas.

Operations:

- **Deployment to pmcp.run / Lambda and the MCP client configs** — the crates are
  Lambda-shaped; deploying is operations, after the crates exist. Blocked in practice on
  D-ITEM-06-01. Source: 06-CONTEXT.md Not in scope / Deferred Ideas.

## D-ITEM-06-06 — `tests/e2e_stdio.rs` is DARK in CI

CI runs `--lib` across the workspace plus ONE explicit line listing individual `--test`
targets. `crates/aprender-mcp-forecast/tests/e2e_stdio.rs` is NOT on that line, so it does not
run in CI — it passes locally and proves nothing about a merge.

The 06-08 decision did NOT list it: `measure-x86-first` leaves both patch hunks unapplied,
and hunk (b) is exactly the one line that would arm it. Hunk (b) needs no runner provisioning
and can be taken on its own at any time; it was not split out only because the recorded
decision was to gate on the x86_64 measurement first (D-ITEM-06-03).

Source: 06-CONTEXT.md "Claude's Discretion" (test organisation) and CLAUDE.md Testing;
06-ci-chronos-step.patch hunk (b).

## D-ITEM-06-07 — two cross-AI review findings REJECTED as REFUTED, with their falsifying observations

Recorded so a future reviewer re-raising either starts from the experiment rather than from
the claim. Neither was "fixed", because neither was real. Full rationale lives in 06-01's and
06-05's `<review_dispositions>` blocks and in 06-REVIEWS.md's orchestrator-verification table.

1. **"`[profile.dev.package.X]` does not reach test builds."** REFUTED. It does: the `test`
   profile INHERITS `dev`, including per-package overrides. The
   `[profile.dev.package.aprender-forecast] opt-level = 3` entry that the Prophet fits need is
   therefore live under `cargo test`, which is where it matters. No change was made.

2. **"`cargo test -- a b` silently drops the second positional filter."** REFUTED. It does
   not: modern libtest UNIONS positional filters, so
   `cargo test -p aprender-forecast --lib -- bolt::parity chronos::parity` runs both sets.
   This one matters because the Chronos gate depends on it — had the claim been true, the
   gate would have been measuring half of what it reports. No change was made.

## D-ITEM-06-08 — the D-13 memory clause was AMENDED; the one locked CONTEXT decision this phase overrode

D-13 AMENDED amend-memory-clause

**What was overridden.** 06-CONTEXT.md D-13's final sentence, quoted as written:

> "Only transposed `[in, out]` weights are kept in memory."

**Why it could not hold.** D-14's second clause, quoted as written:

> "single rows through the 8-accumulator dot"

`dot8` reads contiguous `[out, in]` rows (spike sources/007 `bolt.rs:20-31`, `66-73`). So the
layout D-13 deletes is exactly the layout D-14's kernel consumes. The two clauses cannot both
hold: honouring D-13's memory sentence would either delete the rows `dot8` needs, or force a
transpose per single-row call, which is not the kernel D-14 names.

**The branch taken and its evidence.** `amend-memory-clause`: BOTH layouts stay resident, and
`contracts/chronos-bolt-parity-v1.yaml`'s `weights_dual_layout_documented` equation says so
(it REPLACES `weights_transposed_only`, which no longer exists — REVIEW-06-01). The evidence
is that the frozen parity bar was measured with both layouts resident: aarch64 Peyton Manning
f32 max|delta| = **9.5367e-7** against the contract-read **1.0e-6**. Dropping a layout changes
accumulation order through a 12-layer T5, so the alternative (`drop-untransposed`) was not a
refactor — it required a RE-MEASUREMENT of every parity rung, which is why the choice was put
to a human rather than taken by the planner.

**The disclosed cost, stated rather than buried.** The weight arrays are resident roughly
TWICE — about **69 MB for f32 tiny** (34.6 MB per copy). Note explicitly: SC4's "< 30 MB" is a
**BINARY SIZE** bound and is NOT what this touches. Two different quantities; conflating them
would make this amendment look like a broken success criterion when it is not.

**Where the amendment now lives** (three places, so it cannot vanish with this file):
06-05's must-have truth; `contracts/chronos-bolt-parity-v1.yaml`'s
`weights_dual_layout_documented` equation, bound in `contracts/aprender/binding.yaml` to
`aprender_forecast::bolt` / `Bolt::load`; and 06-07's prohibition list.

**Governance.** The amendment was ratified by **06-05 Task 2's blocking human
`checkpoint:decision`**, not by the planner. The bare token is recorded in 06-05-SUMMARY.md as
`D-13 decision: amend-memory-clause`. 06-CONTEXT.md's D-13 text is deliberately LEFT AS
WRITTEN as the historical record, with this entry as its amendment — a locked decision
rewritten in place would leave no evidence that it ever changed.

## D-ITEM-06-09 — `cargo clippy -- -D warnings` is red workspace-wide, and SC5 is not scoped to it

Restating the boundary because it is the item most likely to be misread as a failed criterion.

SC5's clippy clause reads, verbatim in ROADMAP.md: "`cargo clippy -- -D warnings` on the new
crates". It is ALREADY SCOPED to the new crates, and on those it is GREEN. The 18 pre-existing
findings in `crates/aprender-compute` (plus one in `aprender-core`'s own lib) that make a
workspace-wide `-D warnings` red on aarch64 are OUT OF SCOPE for SC5 and were deliberately not
fixed here. They remain tracked as D-ITEM-06-01-b, D-ITEM-06-02-a, and item 2 of the 06-08
section above, which carry the measured controls.

Consequence still worth stating plainly: `make tier1` / `make tier2` cannot pass on an aarch64
macOS dev box today. Whoever picks it up should fix `aprender-compute` rather than add
`#[allow]`s, and should re-run on BOTH arches — three of the findings are arch-conditional and
therefore invisible to CI, which is X64. That is the same shape as CLAUDE.md #2370's "findings
accumulate where no gate looks".

## D-ITEM-06-16 — `chronos-coldstart`'s 150 ms bar has a different vacuous-pass mechanism, closed only by an upstream parse

Plan 06-16 Task 3 step (4) enumerated every wall-clock bar in the repository. Five were
converted to `scripts/assert_measurement_under.sh`. Two comparison sites were DELIBERATELY
EXCLUDED, and one of the two carries a residual worth writing down rather than leaving in a
commit message.

`justfile` `chronos-coldstart` asserts its bar as `if [ "$med" -ge 150 ]; then FAIL`, and
`chronos-embed-build` asserts a binary-size ceiling the same way. That is bash's INTEGER test,
not `awk`'s `+ 0`, and the difference was MEASURED rather than assumed:

```
  token=[]     -> refused rc=2  [: : integer expression expected
  token=[abc]  -> refused rc=2  [: abc: integer expression expected
  token=[1.5]  -> refused rc=2  [: 1.5: integer expression expected
  token=[150]  -> rc=0 (at the bar)
  token=[149]  -> rc=1 (under the bar)
```

So neither site can COERCE garbage to 0 the way IN-01 describes. That is the stated reason for
the exclusion.

**The residual.** The direction still matters. In `chronos-coldstart` the test is
`if [ "$med" -ge 150 ]; then FAIL; fi`, so a token that makes the test ERROR (rc=2) takes the
NOT-taken branch and the recipe goes on to print `COLD START OK`. `set -euo pipefail` does not
catch it: a command in an `if` condition is exempt from errexit. The hole is closed today only
because `med` is extracted by `sed -n 's/^median: ... \([0-9][0-9]*\) ms.*/\1/p'`, a pattern
that can emit nothing but digits, and an EMPTY `med` is caught by the explicit `[ -z "$med" ]`
guard immediately above. The bar is therefore safe by its upstream parse, not by itself.

status: open

**What to do.** Route it through `assert_measurement_under.sh under "$med" 150` like the other
five, so the guarantee lives at the bar instead of two lines above it. Out of scope for 06-16,
whose fence was the five `awk`-coercion sites named by IN-01.

## D-ITEM-06-16-b — the Phase 6 binding resolver could not resolve a PARAMETERIZED `just` recipe

`Makefile`'s `contract-audit-phase6` resolved a `justfile`-bound row with
`pattern="^$name[[:space:]]*:"`, which matches `forecast-pool-ratio:` but NOT
`forecast-sc1-sweep points="33" ...:`. It was invisible until 06-16 became the first plan to
bind a parameterized recipe, and it would equally have failed for `forecast-holiday-bench`.

FIXED in 06-16 (`^$name([[:space:]][^:]*)?:`), with a must-match / must-not-match table run over
the OLD and NEW patterns together, and re-mutated in its own scope: renaming the recipe makes
the audit exit 2 with `RESOLVE- forecast-tool-boundary-v1.yaml sc1_wall_swept`.

status: resolved
