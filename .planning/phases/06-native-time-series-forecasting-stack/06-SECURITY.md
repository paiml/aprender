---
phase: 06-native-time-series-forecasting-stack
asvs_level: 1
block_on: high
threats_total: 65
threats_open: 0
threats_closed: 59
threats_accepted: 6
verdict: SECURED
audited: 2026-09-07
remediated: 2026-09-07
auditor: gsd-security-auditor
register_source: 17 PLAN.md <threat_model> blocks (79 rows -> 65 unique threat_id x component pairs, 48 distinct IDs)
---

# Phase 06 — Security Audit

## Scope and method

The register was authored at plan time; this audit does not scan for new threats. Each of the
65 unique `(threat_id, component)` pairs was verified by its declared disposition at ASVS L1
(the mitigation must be PRESENT at the cited location). 59 pairs are `mitigate`, 6 are `accept`.
Implementation files were never modified.

Two facts established during the audit change how several verdicts should be read:

1. **The door bounds run in CI on every push.** `crates/aprender-forecast`,
   `crates/aprender-mcp-forecast` and `crates/aprender-mcp-chronos` are workspace members
   (`Cargo.toml:62-64`) and are not in `--exclude`, so `.github/workflows/ci.yml:289`
   (`cargo nextest run --profile ci --workspace --lib`) executes every door refusal test, every
   transport e2e refusal, every contract-mirror test and all three door-surface enumeration tests.
2. **The wall-clock and Chronos-parity gates run only when a human types them.**
   `.github/workflows/ci.yml` has zero `chronos`/`forecast` matches. `just chronos-gate`,
   `just forecast-sc1-sweep`, `chronos-bench`, `chronos-coldstart`, `forecast-bench`,
   `forecast-pool-ratio` and `forecast-holiday-bench` are manual. This was DECIDED, not
   overlooked (06-UAT.md item 3). Any mitigation whose only evidence is "a gate exists" is
   marked accordingly below. NOTE: since this audit, `just forecast-sc1-sweep` HAS been wired
   into `make tier3` via the `forecast-sc1-gate` target (UAT item 5, commit 6ce4ff7d8) — it is
   now a pre-push gate, though still not a CI one.

## Threat verification

### Closed — plans 06-01 … 06-05

| # | Threat | Category | Sev | Disp | Component | Status | Evidence |
|---|--------|----------|-----|------|-----------|--------|----------|
| 1 | T-06-01 | Tampering | high | mitigate | `ForecastArgs` deserialization | CLOSED | `types.rs:270` + `:285` `#[serde(deny_unknown_fields)]` on `HolidayArg` and `ForecastArgs`; strictness test passed, rc=0 |
| 2 | T-06-02 | DoS | high | mitigate | `forecast()` Prophet fit | CLOSED | `types.rs:17,19,21` (`MAX_POINTS` 20 000 / `MAX_HORIZON` 3 650 / `MIN_POINTS` 10) enforced `forecast.rs:44-61`; `fit.rs:63` `MAX_ITERS_PER_ROUND`, `:75` `FIT_BUDGET_SECS`; `spawn_blocking` at `aprender-mcp-forecast/src/lib.rs:61` |
| 3 | T-06-03 | Spoofing | med | mitigate | `--http` loopback surface | CLOSED | `main.rs:152` binds `("127.0.0.1", port)`; `lib.rs:85` `AllowedOrigins::localhost()`; pmcp layers SecurityHeaders + DnsRebinding + origin-locked CORS at `axum_router.rs:105-109` |
| 4 | T-06-04 | Tampering | med | mitigate | non-finite / constant `y` | CLOSED | `forecast.rs:62` `is_finite`; `:102` constant-`y`, both BEFORE any design build; e2e `refuses_constant_y` (`lib.rs:594`) |
| 5 | T-06-05 | Info Disclosure | low | accept | stderr banners | ACCEPTED | AR-1; premise verified `main.rs:159-166` |
| 6 | T-06-07 | Repudiation | med | mitigate | response non-determinism | CLOSED | `forecast.rs:95` `seed.unwrap_or(42)`; `lib.rs:1510`/`:1592` bit-identical 8/8 and 16/16 |
| 7 | T-06-SC | Tampering | high | mitigate | cargo dependency graph | CLOSED | `git show 725d5c7b2 -- Cargo.lock` adds exactly two `name =` entries and **zero** `source =` lines; no `[patch]`; `deny.toml` untouched |
| 8 | T-06-08 | Elev. of Priv. | med | mitigate | `test_no_unauthorized_binaries` | CLOSED | `monorepo_invariants.rs:235-400`: two registers, `publish = false` enforcement, two vacuity assertions |
| 9 | T-06-09 | Repudiation | low | mitigate | README claims | CLOSED | `readme_contract.rs` path gate. **See F-6** — reintroduced in a different file |
| 10 | T-06-SC | Tampering | high | accept | cargo installs (06-02/03/04/06/07/08/09) | ACCEPTED | AR-6; `Cargo.lock` has three commits phase-wide, all workspace members |
| 11 | T-06-10 | Tampering | med | mitigate | tolerance values | CLOSED | tolerances read at test time: `bolt.rs:1266,1329,1345,1443,1488,1552`; `pv validate` clean |
| 12 | T-06-11 | Repudiation | med | mitigate | a parity claim never falsified | CLOSED | 06-03-SUMMARY:87 — induced RED via YAML alone (32/0 -> 25/7 -> 32/0) |
| 13 | T-06-02 | DoS | high | mitigate | NP training with `n_lags` | CLOSED | `forecast.rs:427`, `:432`, `np::door_epochs` cap, `MAX_SPAN_DAYS` at `:81` |
| 14 | T-06-12 | Tampering | med | mitigate | shared autograd tape | CLOSED | `np.rs:667,690,710,719,750` `clear_graph()`; `:1129`; door-level `graph_tape_len()==0` at `forecast.rs:681` |
| 15 | T-06-06 | Tampering | high | mitigate | model weights supply chain | CLOSED | `justfile:364` pinned revision, VERIFY-ALWAYS re-hash, `:418` sha256-mismatch abort; `.gitignore:52` root-anchored `/models/` |
| 16 | T-06-02 | DoS | high | mitigate | Bolt forward / rollout | CLOSED | `chronos.rs:213-241`; `:287` context capped at 2 048; `door_context_used_is_capped_at_2048` `:692` |
| 17 | T-06-04 | Tampering | med | mitigate | non-finite / all-null `y` | CLOSED | `chronos.rs:243`, `:247` |
| 18 | T-06-13 | Repudiation | high | mitigate | parity passing only because weights were absent | CLOSED (manual gate) | counted `cfg_attr(..., ignore)`; two-sided proof 06-05-SUMMARY:152/155; `justfile:565-585` requires `0 ignored` AND >=1 pass |
| 19 | T-06-SC | Tampering | high | mitigate | cargo / python env | CLOSED | `git show 21732582c -- Cargo.lock` adds **zero** packages; `uv run --with` only inside `fetch-chronos-tiny` |

### Closed — plans 06-06 … 06-10

| # | Threat | Category | Sev | Disp | Component | Status | Evidence |
|---|--------|----------|-----|------|-----------|--------|----------|
| 20 | T-06-01 | Tampering | high | mitigate | unknown fields / enums | CLOSED | 22 e2e refusal cases `lib.rs:510-768`; `:1345` asserts `additionalProperties: false` |
| 21 | T-06-02 | DoS | high | mitigate | K concurrent fits | CLOSED | `types.rs:267` `DEFAULT_POOL = 8` (mirrored, passed); `pooled_app` `lib.rs:140-151` |
| 22 | T-06-07 | Repudiation | med | mitigate | responses under load | CLOSED | `lib.rs:1390` `mod pool_equality`, `:1510` (8/8), `:1592` (16/16) |
| 23 | T-06-03 | Spoofing | med | mitigate | loopback HTTP (pool) | CLOSED | `pooled_app` calls `http_app` per router (`lib.rs:150`) |
| 24 | T-06-01 | Tampering | high | mitigate | `ChronosArgs` | CLOSED | `chronos.rs:128` `deny_unknown_fields`; `aprender-mcp-chronos/src/lib.rs:494,703` |
| 25 | T-06-02 | DoS | high | mitigate | rollout count | CLOSED | `chronos.rs:228-241`; `spawn_blocking` at `aprender-mcp-chronos/src/lib.rs:109` |
| 26 | T-06-06 | Tampering | high | mitigate | embedded / on-disk weights | CLOSED | `lib.rs:66-79` `resolve_model()` reads env once; closure captures `Arc<Model>` (`:98-110`) |
| 27 | T-06-13 | Repudiation | high | mitigate | vacuous weight tests | CLOSED (manual gate) | as row 18 |
| 28 | T-06-05 | Info Disclosure | low | accept | stderr banner (chronos) | ACCEPTED | AR-2; `aprender-mcp-chronos/src/main.rs:225-232` |
| 29 | T-06-14 | Repudiation | high | mitigate | evidence table | CLOSED | `06-EVIDENCE.md:19-39` records arch, triple, commit, profile, per-row command and log path |
| 30 | T-06-15 | Tampering | high | mitigate | gate recipes reading status through a pipe | CLOSED | 10 sites: `justfile:486,530,541,547,606,643,670,720,830,922` — every one `rc=$?` after a redirect, never a pipe |
| 31 | T-06-16 | Elev. of Priv. | high | mitigate | CI workflow edit | CLOSED | `git log --since=2026-09-04 -- .github/workflows/ci.yml` EMPTY; proposal remains an unapplied patch |
| 32 | T-06-06 | Tampering | high | mitigate | weights in a CI cache / mount | CLOSED | `justfile:521-537` calls `fetch-chronos-tiny` UNCONDITIONALLY and aborts on non-zero |
| 33 | T-06-17 | Repudiation | high | mitigate | contracts validated by nothing | CLOSED | `Makefile:1996`, `:2367`, `:2385`, `:2396`, wired blocking into tier3 at `:380` |
| 34 | T-06-18 | Repudiation | med | mitigate | README counts drift | CLOSED | counts pasted from derivation commands and re-derived in verify |
| 35 | T-06-19 | Tampering | high | mitigate | `cap` on a non-logistic arm | CLOSED | `forecast.rs:160-164`; `checked_cap` `:167-186`; e2e `:672`, `:685`; contract `:519` |
| 36 | T-06-20 | Repudiation | med | mitigate | the `diagnostics` object | CLOSED | the accepted-and-dropped path no longer exists (row 35) |
| 37 | T-06-21 | DoS | low | accept | option-validation order | ACCEPTED | AR-3 |
| 38 | T-06-SC | Tampering | high | mitigate | installs (06-10 … 06-17) | CLOSED | no `Cargo.lock` commit after `c66da445e`; no install task in those plans |

### Closed — plans 06-11 … 06-17

| # | Threat | Category | Sev | Disp | Component | Status | Evidence |
|---|--------|----------|-----|------|-----------|--------|----------|
| 39 | T-06-22 | DoS | high | mitigate | holiday design build | CLOSED | `types.rs:82` `MAX_HOLIDAY_DESIGN_COST`, `:97` `MAX_HOLIDAY_DATES_TOTAL`; enforced `forecast.rs:302-313`, `:241-250`, both BEFORE `make_design` at `:372` |
| 40 | T-06-23 | DoS | high | mitigate | `FIT_BUDGET_SECS` blindness | CLOSED | bound is pre-fit (row 39); `forecast.rs:278-282` documents why the budget cannot cover it |
| 41 | T-06-24 | Tampering | med | mitigate | contract HARD-ceiling claim | CLOSED | `forecast-tool-boundary-v1.yaml:83,84`, mirrored by `cost_bounds_match_contract` (passed) |
| 42 | T-06-25 | DoS | med | mitigate | per-holiday date list | CLOSED | `forecast.rs:241-250` refuses the RUNNING sum inside the loop |
| 43 | T-06-22 | DoS | high | mitigate | same, via the transport | CLOSED | `lib.rs:807` refuse + `:832` accept-just-under, both through a live server, both in CI |
| 44 | T-06-26 | Tampering | med | mitigate | boundary contract as client doc | CLOSED | equation `:376`, obligation `:780`, falsification test; audit resolves the binding row |
| 45 | T-06-27 | DoS | med | mitigate | a gate that cannot fail | CLOSED | `justfile:830-872` — rc from redirect, absent-line/absent-token/`profile=release` guards, then the validator |
| 46 | T-06-28 | Tampering | high | mitigate | logistic band path | CLOSED | `prophet.rs:770-774` valid-domain branch above `POISSON_NORMAL_BRANCH_LAMBDA` (`:743`) |
| 47 | T-06-29 | Repudiation | med | mitigate | `diagnostics` on long-horizon logistic | CLOSED | degradation removed, not clamped; over-bound requests REFUSED loudly (row 49) |
| 48 | T-06-30 | DoS | med | mitigate | cost of the fix | CLOSED | 06-13-SUMMARY:115 — both reachable extremes on release: 0.225->0.233 s, 0.144->0.205 s vs the 2 s bar |
| 49 | T-06-31 | DoS | **critical** | mitigate | logistic uncertainty arm | CLOSED | `forecast.rs:350-371` — lambda via `changepoint_count`, the SAME function `make_design` uses, refused against `MAX_LOGISTIC_CHANGEPOINT_LAMBDA` BEFORE `make_design`; e2e `:910`/`:933`; contract `:413` |
| 50 | T-06-32 | DoS | high | mitigate | the door's un-enumerated surface | CLOSED | REMEDIATED 2026-09-07 — see Remediation |
| 51 | T-06-33 | Tampering | med | mitigate | `POISSON_NORMAL_BRANCH_LAMBDA` | CLOSED | contract `:128`, mirrored by `cost_bounds_match_contract` (passed) |
| 52 | T-06-34 | Repudiation | low | accept | over-lambda refusal message | ACCEPTED | AR-4; `forecast.rs:360-369` |
| 53 | T-06-35 | DoS | high | mitigate | `prophet::columns` via `holidays[].name` | CLOSED | `forecast.rs:208-216` FIRST check in the loop, in BYTES; `:1018`; e2e `:1057`/`:1078` |
| 54 | T-06-36 | DoS | high | mitigate | `np::train` (unbudgeted) | CLOSED (see F-1) | `types.rs:258` `MAX_NP_TRAIN_COST` enforced `forecast.rs:452-464` via `np::request_train_cost`; e2e `:1118`/`:1143` |
| 55 | T-06-37 | DoS | med | mitigate | late aggregate refusal | CLOSED (see F-2) | `forecast.rs:241-250` moved into the loop; position test `:832` asserts on the MESSAGE |
| 56 | T-06-38 | Info Disclosure | low | mitigate | holiday-name refusal message | CLOSED | `forecast.rs:209-215` names index and byte LENGTH, explicitly not the name |
| 57 | T-06-39 | DoS | high | mitigate | the SC1 gate surface | CLOSED (see F-4) | `sc1_wall.rs:408-463` over `prophet_matrix()` + an NP row; shared builder also used by `holiday_design_wall`; 19 compositions |
| 58 | T-06-40 | Spoofing | high | mitigate | wall-clock bar checks | CLOSED (see F-5) | five sites route through `scripts/assert_measurement_under.sh` (`justfile:625,686,751,869,960`); 23-row case table invoked at `:914` |
| 59 | T-06-41 | Repudiation | low | mitigate | `forecast-holiday-bench` comment | CLOSED | `justfile:807-809` — superlative removed |
| 60 | T-06-42 | DoS | med | accept | the sweep in CI's debug job | ACCEPTED | AR-5; `sc1_wall.rs:446-450` |
| 61 | T-06-43 | Tampering | high | mitigate | `prophet::poisson` falsification test | CLOSED | `prophet.rs:1080` with contract-owned `variance_tolerance` (`:1089`); zero-variance stub observed RED (rc=101) |
| 62 | T-06-44 | Tampering | high | mitigate | branch-constant detectability | CLOSED | `prophet.rs:1094` `zero_mass_tolerance` + non-vacuity `checked_zero_mass == 2` at `:1206`; lowered threshold observed RED |
| 63 | T-06-45 | DoS | med | mitigate | `prophet::feature_row` | CLOSED | `prophet.rs:218` total lookup; `:2414` miss-not-panic; `:2450` unchanged-behaviour control |
| 64 | T-06-46 | Tampering | med | mitigate | `bolt::transpose`'s `expect` | CLOSED (see F-6) | `bolt.rs:287` `debug_assert_eq!` |
| 65 | T-06-47 | Repudiation | med | mitigate | the round's own completion claim | CLOSED | 06-17-SUMMARY:463 ledger verified against the TREE; `:578-589` residual-risk statement |

## Remediation — T-06-32 (was the sole blocker, CLOSED 2026-09-07)

### T-06-32 — DoS — high — the door's un-enumerated surface as a whole (06-14)

**Declared mitigation:** "structural, not another point fix: `door_surface:` enumerates every knob
and every cost axis, and `every_request_knob_is_enumerated` goes red on a new field OR a phantom
field, while `every_cost_axis_names_a_real_bound` goes red on an axis claiming a bound key that
does not exist."

**What is present.** All three tests exist and pass (`cargo test -p aprender-forecast --lib types::`
-> `10 passed; 0 failed`, rc=0 read from a redirect). `no_cost_axis_is_pending` (`types.rs:704`) is
green: the declared-RED window that held C-07 and C-08 open is CLOSED.

**What is absent — two verified gaps in the completeness claim:**

1. **The enumeration covers one of the two doors.** `schema_knobs()` (`types.rs:542-563`) feeds
   exactly `ForecastArgs` and `HolidayArg` into the set equality. `grep -c ChronosArgs
   contracts/forecast-tool-boundary-v1.yaml` returns **0**, and all 16 `door_surface.knobs` rows
   are `owner: ForecastArgs` or `owner: HolidayArg`. `ChronosArgs`'s five caller-settable fields
   (`ds`, `y`, `horizon`, `freq`, `allow_long_horizon`, `chronos.rs:127-143`) have no knobs entry,
   no `enforced_by` prose and no cost axis — while the same contract's `constants:` block carries
   four `chronos_*` keys and the header at `:145` says "THE DOOR'S WHOLE SURFACE". A new
   `ChronosArgs` field enforced by nothing — the exact condition that found C-07 — is invisible to
   all three tests, on a live unauthenticated transport (`aprender-mcp-chronos`).
2. **Cost-axis completeness is not machine-checked.** `enumerated_axes()` (`types.rs:596-618`)
   reads `door_surface.cost_axes` out of the YAML and nothing else. Both axis tests iterate that
   list, so neither can observe an axis that is **not listed**. There is no `schema_for!` analogue
   for cost axes, which means a new cost axis is exactly as invisible as C-06 was before the
   previous round — and C-06 (the axis behind the critical T-06-31) would NOT have been caught by
   either test, because `growth`, `freq`, `horizon` and `ds` all carried knobs entries the whole
   time. The contract's own claim at `:457` ("MISSING (a field exists with no entry) is how CR-01's
   cost axis got in") is not what happened.

**Blocking:** severity high >= `block_on: high`. Counts as 1 toward `threats_open`.

**REMEDIATED — both halves, each observed RED first.**

*Half 1, the second door.* `schema_knobs()` (`types.rs:544-566`) now feeds `ChronosArgs`
alongside `ForecastArgs` and `HolidayArg`. Adding it turned
`every_request_knob_is_enumerated` RED naming exactly the five missing fields —
`[("ChronosArgs","allow_long_horizon"), ("ChronosArgs","ds"), ("ChronosArgs","freq"),
("ChronosArgs","horizon"), ("ChronosArgs","y")]` — which is the observed proof the gap was
real rather than theoretical. Five knobs rows were added to `door_surface.knobs` with the
enforcement read out of `chronos.rs` (`:206` length equality, `:213`/`:219` point bounds,
`:225`/`:228` horizon, `:234` the `allow_long_horizon` gate, `:243`/`:246` value checks,
`:254` monotonicity). Green.

*Half 2, cost-axis completeness, now DERIVABLE.* A new test
`every_cost_ceiling_constant_is_named_by_an_axis` (`types.rs`) derives the ceiling list from
the `constants:` mapping itself (`^(fit|chronos)_max_`) and requires each to be named by a
`cost_axes[].bound` or carry a `ceilings_subsumed` exemption. It shipped RED naming five
unreferenced ceilings — `["chronos_max_horizon", "chronos_max_points",
"fit_max_holiday_columns", "fit_max_holiday_dates", "fit_max_holiday_window"]`. Closed by two
real axes (C-15 Chronos rolling long-horizon forwards -> `chronos_max_horizon`; C-16 Chronos
history parse and context fill -> `chronos_max_points`) and three `ceilings_subsumed` entries
for the holiday component bounds, each verified against the code rather than asserted
(`forecast.rs:253` shows `holiday_columns += upper_window - lower_window + 1`, so both the
window and the column ceiling are factors of C-04's product; the per-holiday date count is one
term of C-03's sum). The exemption list is checked in BOTH directions — a phantom constant and
a dangling `subsumed_by` each fail — plus a vacuity guard requiring >= 10 ceilings so the
filter cannot silently stop matching.

*Overstated claims corrected in the same change,* since the audit showed the text asserted
more than the tests did: the `door_surface_is_complete` formula now names all three structs
and the ceiling direction; `inv[2]`'s claim that "MISSING is how CR-01's cost axis got in" is
replaced with the audited finding that it is **false**; `inv[4]`'s "applies no
DefaultBodyLimit" now records pmcp's 4 MiB `max_request_bytes` and 413 (F-3); the axis count
moved 14 -> 16.

**RESIDUAL, stated rather than closed.** An expensive path added inside `prophet.rs`/`np.rs`
that introduces NO constant and NO axis entry remains invisible to all four tests. That is
C-06's exact shape. Its detector is not a test here but `just forecast-sc1-sweep`, now wired
into `make tier3` (`forecast-sc1-gate`), which measures the wall rather than the enumeration.
The module comment at `types.rs` and the binding note both say so.

**Verification:** `cargo test -p aprender-forecast --lib types::` -> 11 passed / 0 failed
(was 10; the 11th is the new test). Full suites `aprender-forecast` + `aprender-mcp-forecast`
+ `aprender-mcp-chronos` -> 161 passed / 0 failed. `pv validate` rc=0 on the boundary
contract. `make contract-audit-phase6` rc=0, 63 rows, zero BIND-. clippy `-D warnings` rc=0,
`cargo fmt --all --check` rc=0.

---

**Original suggested resolution, retained for the record (either would have closed it):** add `("ChronosArgs", schema_for!(ChronosArgs))` to
`schema_knobs()` plus the five knobs entries and the Chronos cost axes (the rolling long-horizon
path at `ceil(horizon / chronos_native_horizon)` forward passes is the obvious one); AND make the
axes half derivable — the cheapest real check is the inverse direction, "every `constants.fit_max_*`
key must be named by at least one `cost_axes[].bound`", which is derivable today. If neither is
done, the invariant text in `door_surface_is_complete` and the module comment at `types.rs:513-519`
must be narrowed to what is actually checked: the knobs half is derived, for one door.

## Accepted risks

No SECURITY.md existed before this audit, so none of these acceptances was documented anywhere.
Each is recorded here with its factual premise independently verified.

**AR-1 — T-06-05, Information Disclosure, low (06-01).** The forecast server's stderr banners.
*Accepted because* they carry the server name, the bind address, the port and the router-pool mode
only; stdout is reserved for protocol. **Verified:** `aprender-mcp-forecast/src/main.rs:159-166`
and `:179-182`. No request content, no environment value and no filesystem path is echoed.

**AR-2 — T-06-05, Information Disclosure, low (06-07).** The Chronos server's stderr banner.
*Accepted because* it carries the model name, parameter count, dtype, source and load time only.
**Verified:** `aprender-mcp-chronos/src/main.rs:225-232`. One adjacent path echo exists —
`main.rs:57` prints the `--bench`/`--coldstart` directory argument on an I/O error — which is
operator-supplied, not caller-supplied, and reaches stderr only on a startup failure.

**AR-3 — T-06-21, Denial of Service, low (06-10).** The cost of the door's own option-validation
order. *Accepted as negligible:* the arm-scoping check is `Option::is_some` plus one enum
comparison, placed before any design build. **Verified:** `forecast.rs:112-130` and `:160-164`.

**AR-4 — T-06-34, Repudiation, low (06-14).** The over-lambda refusal message does not tell the
caller whether their horizon is too long or their history too short. *Accepted because* naming both
factors and their product is strictly more information than any neighbouring refusal gives, and a
decision tree inside an error string is worse for clients to parse. **Verified:**
`forecast.rs:360-369` names the observed Poisson mean, the changepoint count, `t_max`, both spans
in days, the bound key, its value, and three concrete fixes.

**AR-5 — T-06-42, Denial of Service, medium (06-16).** The SC1 sweep runs in CI's debug `--lib`
job with no wall-clock assertion. *Accepted by design:* a wall-clock assertion in a debug build
shared with 80 000+ other tests would flake on CPU throttling. **Verified:** `sc1_wall.rs:446-450`
returns before the 2 s bar under `cfg!(debug_assertions)`; every printed line carries `profile=`
derived from the cfg and never from intent; the only live CI assertion is a 120 s budget explicitly
labelled "NOT the SC1 bar". **Residual:** the release-profile bar runs only via
`make tier3` / `just forecast-sc1-sweep` (see F-4).

**AR-6 — T-06-SC, Tampering, high (06-02, 06-03, 06-04, 06-06, 06-07, 06-08, 06-09).** Seven plans
accepted the supply-chain threat on the ground that they install nothing and add no dependency.
*Accepted, and the premise is verified rather than asserted:* across the entire phase `Cargo.lock`
has exactly three commits — `725d5c7b2`, `21732582c`, `c66da445e` — which between them add exactly
three `name =` entries (`aprender-forecast`, `aprender-mcp-forecast`, `aprender-mcp-chronos`, all
workspace members) and **zero** `source =` lines. No registry package entered the graph, no `[patch]`
section exists in `Cargo.toml`, and `deny.toml` was not touched in the phase window.

## Findings not in the register

None of these blocks the phase. All are recorded so a future reader does not mistake this audit for
a clean bill on the surrounding claims.

**F-1 — CR-01: `MAX_NP_TRAIN_COST` over-refuses in-spec requests (availability, opposite sign).**
`types.rs:258` prices 20 000 daily points at `n_lags: 7` at 15 994 400 and refuses it by 6.6 %,
while the neighbouring 10 000-point request completes in ~1.3 s. This does not re-open T-06-36 —
the DoS direction is bounded, and the bound errs toward refusing — but the *accepted region* was
never characterised. **Disposition: DECIDED by the product owner** (06-UAT.md item 4, 2026-09-07,
option (c)): the 13 `pub const MAX_*` become a resolved `DoorLimits` profile with today's values as
the default; no profile may disable a bound or exceed a tier's structural maximum. Phase 7.

**F-2 — WR-01: the post-loop aggregate-dates refusal is unreachable, and three artifacts say it
fires.** `holiday_dates_total` has exactly one write site (`forecast.rs:241`) and is compared to the
ceiling on the very next statement with no `continue` in the loop, so `forecast.rs:289-295` is dead
for every input. `forecast-tool-boundary-v1.yaml:235` and cost axis C-03 both describe a refusal the
code cannot produce. The bound itself IS enforced; only the second message is dead.

**F-3 — WR-02: a load-bearing security claim in the D-15 contract is verifiably false.**
`types.rs:188-194` and `forecast-tool-boundary-v1.yaml:303, :401, :457` all state that neither
transport caps the request body, and C-07's `no_structural_maximum: true` disposition rests on it.
pmcp 2.19.3's `StreamableHttpServerConfig::stateless()` — configured at
`aprender-mcp-forecast/src/lib.rs:84` — sets `max_request_bytes = 4 MiB`
(`pmcp-2.19.3/src/server/limits.rs:46`), enforced with a 413 before any JSON parsing
(`streamable_http_server.rs:4571`), plus a 1 MiB `max_tool_args_bytes`. HTTP *does* have a
structural maximum; stdio genuinely does not. The 200-byte bound is not wrong, but its stated
justification was argued from the absence of a mechanism in one file rather than proved against the
transport (CLAUDE.md rule 2). Worth owning the limit explicitly rather than inheriting a dependency
default that can move.

**F-4 — WR-07/WR-08: the SC1 gate's shipped geometry misses the slowest known shapes.** The sweep's
`points` axis is pinned at 33 in both the harness default (`sc1_wall.rs:409`) and the recipe
(`justfile:907`), while the two slowest holiday compositions on record (800 x 50 at 1.692 s; the
4 700-point 5-column request at 4.2 s) lie outside the matrix. At audit time no automatic surface
ran it at all; `make tier3` now does (commit 6ce4ff7d8), but CI still does not.

**F-5 — WR-06: the validator's own scope claim overstates its reach.**
`scripts/assert_measurement_under.sh:2-3` calls itself "the ONE numeric bar check every wall-clock
gate in this repository calls". `chronos-coldstart` (`justfile:656`) is a wall-clock SC4 gate with
a 150 ms bar that does not call it. **Independently checked: that site fails closed** — the `sed`
at `:650` emits digits only, an absent-line guard fires at `:651-654`, and `[ "$med" -ge 150 ]`
under `set -e` aborts on a non-integer. `justfile:740` still runs the old `awk ... v + 0` form as a
max-*selector*, but its output is then shape-checked by the validator at `:751`. Neither is
exploitable; both contradict a claim the script makes about itself.

**F-6 — WR-09/IN-03: false semver framing in shipped documentation.**
`crates/aprender-forecast/README.md:99-113` describes `feature_row` and `safetensors::load` changes
as "breaking for external callers ... in the 0.63.0 line"; `Cargo.toml:15` is `publish = false` and
the crate has never been released. `bolt.rs:284-285` repeats the claim as the reason `transpose` was
not widened to `Result`, so a design decision rests on a constraint that does not exist.

**F-7 — process: two SUMMARYs carry no `## Threat Flags` section.** `06-08-SUMMARY.md` and
`06-13-SUMMARY.md` have none; the other fifteen do and all fifteen report "None". Every threat those
two plans owned was verified independently in this audit, so nothing is unverified as a result — but
the executor's own new-attack-surface declaration is missing for two plans.

**F-8 — unregistered surface: the demo pages.** Both servers serve a same-origin HTML MCP client at
`GET /` (`aprender-mcp-forecast/src/lib.rs:90-93`). 06-UAT.md test 1 found neither page could
complete a single `tools/call` for either model — broken for four plans and three gap-closure
rounds, undetected because no automated test drives them. Fixed in-session (`4da4980d5`); the page
is still covered by no test. An open design question rides on it: should an off-arm option carrying
its own DEFAULT (`n_lags: 0`, `growth: "linear"`) be refused, or only a non-neutral value?
Deferred to Phase 7. Security-relevant only as surface with no regression detector.

**Unregistered flags declared by executors:** none. All fifteen `## Threat Flags` sections report
"None", and this audit found no new attack surface outside F-8.

## Verification notes

- `cargo test -p aprender-forecast --lib types:: > /tmp/p06-sec-types.log 2>&1; rc=$?` -> **rc=0**,
  `10 passed; 0 failed; 0 ignored`, including `no_cost_axis_is_pending`,
  `every_request_knob_is_enumerated`, `every_cost_axis_names_a_real_bound`,
  `cost_bounds_match_contract`, `chronos_bounds_match_contract`, `pool_default_matches_contract`.
  Status read from a redirect, never through a pipe (CLAUDE.md rule 1).
- Supply chain: `git show <c> -- Cargo.lock | grep -cE '^\+name = '` for each of the three phase
  commits -> 2 / 0 / 1, and `grep -E '^\+source = '` -> empty for all three. Run through
  `rtk proxy`, because the `rtk` Bash hook silently empties piped output and the first attempt at
  this command returned three empty blocks that would have read as "nothing added".
- pmcp defaults read from the registry source, not inferred: `axum_router.rs:105-109`,
  `limits.rs:46`.
- CI coverage confirmed structurally: `Cargo.toml:62-64` lists the three crates as members, they are
  absent from `exclude`, and `.github/workflows/ci.yml:289` runs `--workspace --lib` with only the
  three GPU crates excluded. `grep -c 'chronos\|forecast' .github/workflows/ci.yml` -> **0**.
- No implementation file was modified by this audit.

## Next

Close T-06-32 (extend `schema_knobs()` to `ChronosArgs` + add the Chronos cost axes, and make the
cost-axis half derivable in at least one direction) OR narrow the `door_surface_is_complete`
invariant text to the half that is actually checked and register the Chronos door as explicitly out
of scope. Then re-run `/gsd-secure-phase 06`.
