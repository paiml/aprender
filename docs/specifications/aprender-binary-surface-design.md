# Aprender Binary Surface — Design Specification

**Status:** **DRAFT, revised against HEAD.** No pick has been implemented. Phase A (§13.3) has now been executed; its ledger is §1A and it moved several picks.

**Revision:** v2, 2026-08-15. Supersedes the v1 draft, whose repo facts were quoted from the Fable architectural review at `9f0b89fb` (2026-07-05). That snapshot is six weeks stale and **the tree has moved underneath four of its ten regression classes.** Where v1 and HEAD disagree, HEAD wins and v1 is struck through in place rather than deleted, so the delta is auditable.

**Verification basis:** `origin/main` @ `907bef27` ("fix(readme): three different contract counts, and the guard that watches them ran nowhere", #2485), 2026-08-15. Every VERIFIED line below cites a path:line in that tree.

| Pick | Tier | Status | Lands in |
|------|------|--------|----------|
| SURF-1 surface manifest emitter, derived from the clap tree (`apr surface --emit`) | B | **OPEN — rescoped.** The top-level command *set* is already locked (§1A/R2). The emitter's value is now everything *below* the command name | `apr-cli` |
| SURF-2 canonicalization + lock primitive (JCS descriptors, SHA-256 root) | B | OPEN | `aprender-contracts` |
| SURF-3 `apr surface --diff` semantic classifier + Kani proof | B | OPEN | `aprender-contracts` |
| SURF-4 L1 tier-1 gate — compiled workspace artifact, in `gate`'s `needs` | B | **NARROWED.** A tier-1 gate on the command set already exists and already blocks (§1A/R7). SURF-4 extends its *subject*, it does not create the lane | `.github/workflows/ci.yml` |
| SURF-5 L1 tier-2 gate — clean-room A4 verification of the installed binary | B | **OPEN, UNVERIFIABLE HERE.** The A1–A4 harness lives in `infra`, not in this repo; `clean-room` in `ci.yml` is a *runner label*, not that harness (§1A/T9) | `infra` |
| SURF-6 L0 rebind — Kaizen K2 enumerates subcommands from the lock | B | **OPEN, UNVERIFIABLE HERE.** `kaizen-binary-qa.md` is not in this repo | `infra` + `apr-cli` |
| SURF-7 L3 stream discipline (stdout purity, panic containment, subprocess stdio) | B | **OPEN — split.** The MCP subprocess boundary is already safe; the panic leg is genuinely unhandled (§1A/T5) | `apr-cli`, `aprender-mcp`, `aprender-serve` |
| SURF-8 L3 `oracle: real` non-simulation invariant per verb | B | OPEN | per-verb contracts |
| SURF-9 L2 transport equivalence + streaming reassembly | B | **UNBLOCKED, and smaller than v1 assumed.** §12.1/§12.3 resolved: 9 of 105 verbs have an MCP leg; HTTP is three divergent routers, not one | `apr-cli`, `aprender-mcp`, `aprender-serve` |
| SURF-10 verb registry for `apr` (UVS pattern, reimplemented) | B | **REJECT for the CLI leg.** The clap tree already *is* the single dispatch table (§1A/T1). A parallel registry would be copy #5 | — |
| SURF-11 re-lock churn instrumentation | — | OPEN | `infra` metrics |
| SURF-12 fleet-wide `pmat comply` rule | D | **REJECT** until SURF-11 reports | `pmat` |
| **SURF-13 nested-subcommand surface** (new) | B | **OPEN — the largest real gap found in Phase A: 45 ungated subcommand paths, measured** | `apr-cli` |
| **SURF-14 contract enforcement pointers must resolve** (new) | B | OPEN — cheap, and it falsifies itself today | `aprender-contracts` |

**Tag:** `SURF` (commit refs `Refs SURF-NN`; `pmat work` ids follow `PMAT-<SLUG>-NNN`)
**Parent doc:** `docs/specifications/fable-architectural-review.md`
**Scope:** the `apr` binary and every surface it exposes — CLI, MCP, HTTP — and the contracts, locks and gates that make functional regression on those surfaces detectable rather than discoverable-in-the-field. Out of scope: fleet rollout (SURF-12), non-`apr` binaries, and behaviour not observable from outside the process.

---

## 0. Why this spec exists — restated after Phase A

v1 opened: *"Aprender advertises 103 CLI commands. Three independent sources agree… The shipped binary has never been asked."*

**Both halves are false at HEAD, and the correction is more interesting than the claim.**

The number is **105**, not 103 (`README.md`: `**105** CLI commands`). And the binary *is* asked: `crates/apr-cli/tests/cli_commands.rs:147` spawns `env!("CARGO_BIN_EXE_apr")` — a compiled artifact, not source — parses its `--help`, and asserts the command **set** in both directions against a 105-entry mirror of `contracts/apr-cli-commands-v1.yaml`. That test runs in CI at `.github/workflows/ci.yml:327`, inside the `workspace-test` job, which is in `gate`'s `needs` (`ci.yml:540`). It blocks merges today.

A second gate landed the same week: `.github/workflows/ci.yml:516` runs `scripts/check_readme_claims.sh` in the `guard-runner-labels` job — also in `gate`'s `needs`. v1 asserted this script "does not execute in any workflow." It does, as of #2485.

So the premise that motivated this spec has been retired by ordinary maintenance. **That is the correct outcome of Phase A, and the spec is better for it.** What survives is narrower, sharper, and still worth building:

> **The command *name* is locked, at depth 1 only. Nothing below that is.**

The binary was built from HEAD and asked (§1A.5). It reports 105 top-level commands — agreeing exactly with all three paper copies — and **45 further subcommands one level down, across 15 parents, none of which any gate has ever seen.** 30% of the invocable command surface is unlocked. Beyond the names, nothing at any depth is locked: not flags, not value domains, not exit codes, not auth posture, not protocol revision. `apr registry` is verified to exist; that `apr registry aliases` exists, takes the arguments it took last release, and exits 5 rather than 2 on a bad manifest, is verified nowhere.

Two properties still drive every decision below, and Phase A strengthened both:

1. **The subject of a surface contract is the built artifact.** The tree already agrees — `CARGO_BIN_EXE_apr` and `route_surface_2376.rs` are both black-box. The remaining gap is not "source vs binary", it is *which* binary: the workspace debug artifact is gated, the published one is not.
2. **Surface regression is a derivation problem.** The tree already agrees here too, and has the scar tissue to prove it: `crates/aprender-serve/src/api/router.rs:66-78` documents three HTTP routes that were mounted, advertised in no list, and therefore invisible to the very guard written to catch them — fixed by deriving mount and advertisement from one table. That comment is the thesis of this spec, already implemented, for one transport.

---

## 1A. Phase A ledger

Verdicts: **VERIFIED** (claim holds, with path) · **STALE** (claim ≠ tree — both values given) · **UNVERIFIED** (could not fetch from this session).

### 1A.1 Regression classes — re-adjudicated

| # | v1 claim | Verdict | Evidence at `907bef27` |
|---|---|---|---|
| R1 | Binary leg never exercised; `check_readme_claims.sh` runs nowhere | **STALE — falsified** | `ci.yml:516` runs it in `guard-runner-labels`; `gate.needs` includes that job (`ci.yml:540`). Separately `cli_commands.rs:147` has exercised `CARGO_BIN_EXE_apr` all along, gated at `ci.yml:327` |
| R2 | Count assertion, not set assertion | **STALE — falsified for the top level** | `test_all_contract_commands_exist` (`cli_commands.rs:228`) and `test_no_unregistered_commands` (`:247`) assert the set **bidirectionally**. `test_command_count_matches` (`:272`) is the cardinality check *on top of* the set check, which is the right order. **Still true one level down** — see R12 |
| R3 | Nth drifting copy; contract prose says 77 | **PARTLY VERIFIED** | The three *machine-read* copies are in exact sync: README `**105**`, `apr-cli-commands-v1.yaml` 105 entries, `registered_commands()` 105 entries; a set-diff of the latter two is empty in both directions. **The prose is still rotten**: `apr-cli-commands-v1.yaml`'s `scope:` still reads "77 commands — original 57 + …". Nothing reads it, so nothing corrects it |
| R4 | Single-point sampling (CF-4) | **VERIFIED, unchanged** | No surface-level multi-step probe exists |
| R5 | Simulated engine advertised as real | **NOT REPRODUCED IN THIS REPO** | The cited instance is pmat's `refactor.*`. No aprender equivalent was found. The field is still worth requiring, but it is a *prophylactic* here, not a fix — say so rather than importing another repo's scar as our own |
| R6 | Process-level stream leakage | **VERIFIED — and narrower than stated.** See T4 | `crates/aprender-mcp/src/tools/subprocess.rs:174-175,342-343` sets `Stdio::piped()` on **both** child streams, so the MCP frame stream is already protected from child chatter |
| R7 | Gate exists, does not block | **STALE for the surface gates** | Both surface-relevant gates are inside `gate.needs`. The claim may still hold for the P4 decode harness (not re-checked — out of scope) |
| R8 | Surface invisible to the feature matrix; `cargo install aprender` may ship no `apr` | **STALE — falsified** | `Cargo.toml:524` `default = ["cli"]`; `Cargo.toml:517` documents this explicitly for GH-1599: *"`cargo install aprender` still works because `cli` is in default features."* The optional-dep observation is true; the inference drawn from it is not |
| R9 | Protocol drift; `aprender-mcp` revision UNVERIFIED | **VERIFIED — now pinned** | `crates/aprender-mcp/src/lib.rs:68`: `pub const PROTOCOL_VERSION: &str = "2024-11-05"`. Asserted at `server.rs:1016`. A newer proposal is accepted and answered with our version (`server.rs:921-944`) — deliberate, tested, and exactly the drift surface R9 describes |
| R10 | Toolchain drift; MSRV declared 1.89 | **VERIFIED, values STALE** | `rust-toolchain.toml` pins `1.93.0`; `Cargo.toml:123,491` declare `rust-version = "1.91"` (not 1.89). The gap is real and the direction of the argument survives |

### 1A.2 New regression classes found by Phase A

| # | Class | Instance | Fix |
|---|---|---|---|
| **R11** | **A contract's enforcement pointer is unverified free text** | `apr-cli-commands-v1.yaml` FALSIFY-CLI-001..005 each carry `enforcement: cargo test --test apr_cli_commands …`. **There is no `apr_cli_commands` test target** — the file is `crates/apr-cli/tests/cli_commands.rs`, target `cli_commands`. FALSIFY-CLI-003/-004 additionally name `test_all_commands_help`; the function is `test_all_commands_respond_to_help` (`cli_commands.rs:201`). Three of five conditions cite a command that exits non-zero. The *tests* are real and do run; the contract's account of how is fiction | **SURF-14**: `pv` resolves every `enforcement:` string to an existing target + test name, or the contract fails validation |
| **R12** | **The gated surface stops at depth 1** | **Measured against `./target/debug/apr` @ `907bef27` (§1A.5): 105 top-level commands, all gated; 45 depth-2 subcommands across 15 parents, none gated.** `registered_commands()` holds 105 entries and **zero contain a space** — no nested path is in it. `apr registry aliases` can be renamed or deleted and every surface gate stays green. 30% of the invocable command surface is unlocked | **SURF-13** |
| **R13** | **One binary, three HTTP surfaces, chosen by input file format** | `crates/aprender-serve/src/api/router.rs:232-236` states it outright: `GET /` and `GET /ready` *"are registered by the two OTHER routers in this repo (`apr-cli commands/serve/routes.rs`, `serve_run_model.rs`) and 404'd here, so which of three route surfaces you got depended on the format of the file you passed to `apr serve run`."* `serve_run_model.rs:76-81` mounts 6 routes; `crates/apr-cli/src/commands/serve/routes.rs:358-369` mounts 8; `crates/aprender-serve/src/api/router.rs` mounts 15+ from a derived table | Scopes SURF-9. A "transport equivalence" invariant over an ambiguous transport is unfalsifiable until the ambiguity is named |
| **R14** | **No panic containment anywhere on either server path** | `catch_unwind` / `CatchPanicLayer` appear **nowhere** in `crates/aprender-serve/src`, `crates/apr-cli/src` or `crates/aprender-mcp/src` outside a test helper (`popperian_tests.rs:240`) and `qualify.rs:91`. No `tower_http::catch_panic`. A handler panic is an aborted connection over HTTP and a killed frame stream over MCP | The `apr-surface-stream-v1` panic leg is a genuine finding, not a prophylactic |

### 1A.3 §13.3 question table — answered

| # | Question | Verdict | Answer |
|---|---|---|---|
| T1 | Is the `apr` surface enumerable from one dispatch table? | **VERIFIED — yes, but not the table v1 named** | Not from `Commands` — that enum has only ~36 variants (`commands_enum.rs:46`), because `ModelOps` (`:706`) and `Extended` (`:798`) are `#[command(flatten)]` and dissolve into the top level. The real table is the **clap `Command` tree**, reachable as `Cli::command()` via `clap::CommandFactory` — and it is **already used this way in-tree** at `help_producer_truth.rs:179-180, 212-213, 251-252`, which resolves backtick-quoted `apr …` invocations against the live parser. **SURF-1 is unblocked and its substrate is proven.** |
| T2 | Does `check_readme_claims.sh --claim cli_command_count` run anywhere? | **VERIFIED — yes** | `ci.yml:516`, in `gate.needs`. Caveat: the step's own comment says *"Text-only, no build"* and that is wrong — `measured_cli_command_count` (`check_readme_claims.sh:63`) runs `cargo run --quiet -p apr-cli --bin apr -- --help`. The step builds |
| T3 | Do HTTP and MCP surfaces exist, at what revision? | **VERIFIED — both exist** | MCP: `aprender-mcp`, protocol `2024-11-05` (`lib.rs:68`), **9 tools** (`contracts/apr-mcp-tool-schemas-v1.yaml`: `apr.version, apr.validate, apr.tensors, apr.bench, apr.qa, apr.trace, apr.run, apr.serve, apr.finetune`). HTTP: exists, but as **three** routers (R13). No `/v1/messages` route exists anywhere — v1's "Anthropic-compatible" HTTP leg is still a proposal |
| T4 | Does any verb spawn a subprocess with inherited stdio? | **VERIFIED — yes in the CLI, no at the MCP boundary** | 25+ non-test `Command::new` sites in `crates/apr-cli/src`. `.status()` without redirect **inherits**, at `crates/apr-cli/src/commands/pipeline.rs:32`, `crates/apr-cli/src/commands/train.rs:701,727`, `crates/apr-cli/src/commands/mono.rs:306,314,430`, `crates/apr-cli/src/commands/qualify.rs:183`. All are CLI-only paths where stdout is not a protocol stream. The MCP boundary pipes both streams (`crates/aprender-mcp/src/tools/subprocess.rs:174-175, 342-343`). **This inverts the SURF-7 mutation** — see §8 |
| T5 | Panic-catching layer? MCP stdio survives a handler panic? | **VERIFIED — no layer exists** | R14 |
| T6 | Are `schemars` and `clap` floating? | **VERIFIED, and v1's premise is wrong** | `clap = "4.5"` (`Cargo.toml:169`) resolves to **4.6.1** in `Cargo.lock` — floating across a minor. **`schemars` is not on this surface at all**: the only declaration is `crates/aprender-train/Cargo.toml:121`, `schemars = "0.8"`. Neither `apr-cli` nor `aprender-mcp` depends on it. MCP `inputSchema`s come from `build.rs` codegen off `contracts/apr-mcp-tool-schemas-v1.yaml` (`crates/aprender-mcp/src/lib.rs:44-59`) — **already contract-derived**. §2.1's `schemars:` field is deleted; `clap` is pinned in its place |
| T7 | Untagged enums in param types? | **VERIFIED — retroactive, not greenfield** | 17 `#[serde(untagged)]` sites workspace-wide; 4 on the surface path, including a request type at `crates/apr-cli/src/commands/serve/types.rs:295` and `crates/aprender-serve/src/api/realize_handlers.rs:280`. §3's constraint must be introduced with a grandfather list or it blocks on day one |
| T8 | Does `cargo install aprender` produce `apr`? | **VERIFIED — yes** | R8 |
| T9 | What does clean-room A4 assert today? | **UNVERIFIED** | The `infra` repo is not reachable from this session and is not in scope for this workspace. `clean-room` in `ci.yml` is a self-hosted **runner label** (`ci.yml:86,372,538`), not the A1–A4 harness. SURF-5 cannot be scoped from here |
| T10 | Does Kaizen K2 enumerate from the binary or a list? | **UNVERIFIED** | `kaizen-binary-qa.md` is not in this repo |
| T11 | `gate`'s `needs` at HEAD? | **VERIFIED** | `ci.yml:540`: `needs: [ci, workspace-test, mutants, guard-runner-labels]`. SURF-4 attaches to `workspace-test` (where `cli_commands` already runs) or `guard-runner-labels` (text/lightweight guards) |
| T12 | Is there prior art for a derived surface lock in-tree? | **VERIFIED — yes, and it is good** | `crates/aprender-serve/src/api/tests/route_surface_2376.rs` probes advertised⇒mounted **and** unadvertised⇒absent, black-box, across every `RouterConfig`. This is an L1 lock for one router, written before this spec existed. **SURF-1 should look like this, not like a new framework** |
| T13 | Tooling availability in the working container | **VERIFIED — absent** | `pv`, `pmat`, `apr`, `bashrs` are all absent from this session's container. Phase C (`pv validate`) and any `pmat work` step must run on a host that has them |
| T14 | What does the binary itself report? | **VERIFIED — measured, not inferred** | Built `apr 0.63.0 (907bef27)` from HEAD and asked it. See §1A.5 |

### 1A.4 Corrections to v1's own numbers

| v1 said | HEAD says |
|---|---|
| 103 CLI commands | **105** (README, contract, and Rust mirror all agree) |
| `commands_enum.rs:26` defines 104 − 1 dev-gated | The enum is at `commands_enum.rs:46` and has ~36 variants; the 105 arises after `#[command(flatten)]` on `ModelOps`/`Extended`. `Mono` is `#[cfg(feature = "dev")]` (`:801`) and is absent from the 105 |
| README contract count says 1331 vs 1,460 | Both stale. README says **1771** in three places; `find contracts -name '*.yaml' \| wc -l` = **1771**. In sync |
| MSRV declared 1.89 | `rust-version = "1.91"` |
| 20 advertised MCP tools (pmat) | aprender's MCP surface is **9** tools |

### 1A.5 The binary, asked directly

v1's animating complaint was that the binary had never been asked. It has now been built from HEAD and asked. `apr 0.63.0 (907bef27)`, `cargo build -p apr-cli --bin apr` (default features, no `dev`):

| Measurement | Method | Result |
|---|---|---|
| Top-level commands | `apr --help`, 2-space-indent rows, `help` excluded | **105** — exactly the README claim, the contract registry, and the Rust mirror. Three copies and the artifact all agree |
| …including clap's freebie | same, `help` included | 106 — the off-by-one `scripts/check_readme_claims.sh:57-61` documents having already hit |
| Parents with subcommands | `apr <cmd> --help` for all 105 | **15**: `serve, debug, canary, registry, runs, experiment, probar, modelfile, train, tokenize, data, pipeline, dataset, kernel, rosetta` |
| Depth-2 subcommands | same | **45** — `train`(8), `rosetta`(8), `tokenize`(6), `data`(5), `pipeline`(4), `runs`(3), `serve`(2), `canary`(2), and 7 parents with 1 each |
| Depth-3 subcommands | `apr <cmd> <sub> --help` for all 45 | **0** — the tree is exactly 2 deep |
| **Total invocable command paths** | 105 + 45 | **150, of which 105 (70%) are gated and 45 (30%) are not** |

Two things follow, and they are the strongest results in this ledger:

- **The top-level surface is genuinely sound.** Four independent representations — the artifact, `README.md`, `contracts/apr-cli-commands-v1.yaml`, and `cli_commands.rs::registered_commands()` — agree at 105, and two of the three copies are mechanically pinned to the artifact by a blocking gate. v1 assumed drift and found agreement. **Do not build SURF-1 to fix a problem that is already fixed.**
- **The gap has a number now: 45.** R12 stops being an argument and becomes a work item with a definition of done. That is what SURF-13 is for, and it is why it ranks above the lock primitive in §13.6 — it needs no new infrastructure at all, only the `Cli::command()` walk that `help_producer_truth.rs:179` already performs and the `CARGO_BIN_EXE_apr` harness `cli_commands.rs:147` already uses.

Depth being exactly 2 also bounds SURF-13: the emitter needs one level of recursion, not an unbounded walk, and the lock's `depth` field has domain `{1, 2}` today. Record it anyway — a future `apr registry alias add` is what the field is for.

---

## 2. The primitive — a surface lock emitted by the binary

```
<artifact> surface --emit json --features <combo>
  → JCS canonicalization → SHA-256 root
  → compare to contracts/surface/apr-<combo>-v1.lock
```

### 2.1 Manifest shape (revised)

```yaml
apiVersion: surface.paiml.dev/v1
binary: apr
contract_version: 1              # bump required on any BREAKING classification (§5.2)
build:
  features: [default]
  toolchain: "1.93.0"            # R10 — from rust-toolchain.toml
  rust_version: "1.91"           # R10 — declared MSRV; the two differing is the point
  clap: "4.6.1"                  # T6 — the LOCKED version, not the declared "4.5"
server:
  mcp_name: "apr"                # must match ^[a-z0-9_-]+$
  mcp_protocol: "2024-11-05"     # R9 — VERIFIED at crates/aprender-mcp/src/lib.rs:68
  http_router: native | serve_run_model | apr_cli_serve   # R13 — WHICH of the three
verbs:
  - name: registry.add           # dotted path; nesting is explicit (R12)
    depth: 2
    effects: read-write
    transports:
      cli:  { auth: none }
      mcp:  { present: false }
    params_digest: sha256:…      # over the canonical param descriptor, §2.4
    exit_codes: { ok: 0, invalid: 5 }
    oracle: real
```

Four changes from v1, each forced by Phase A:

- **`schemars:` is deleted.** It is not a dependency of `apr-cli` or `aprender-mcp` (T6). Pinning a crate the surface does not use is exactly the "verified against other claims" failure this spec exists to stop.
- **`clap` records the *locked* version, not the declared range.** `"4.5"` is not a fact about the artifact; `4.6.1` is.
- **`http_router` names which of three** (R13). A manifest that says "the HTTP surface" when the binary has three, selected by input file format, is false while passing.
- **`transports.mcp` may be `{present: false}`,** and for 96 of 105 verbs it will be. Transport parity is a claim about the 9 verbs that have two legs, not about all of them.

**`auth` remains a property of the (verb, transport) pair.** Unchanged from v1 and still correct: a verb reachable unauthenticated over CLI and token-gated over HTTP is the trust boundary working. Hoisting it to verb level makes the CLI leg false while green.

**`exit_codes` is contract-shaped already** — `apr validate` carries `exit_code = if score < 50 then 5 else 0` under FALSIFY-CLI-001. The manifest lifts an existing pattern.

### 2.2 `oracle: real | simulated`

Required, no default. **Demoted from "fix for R5" to prophylactic** (§1A/R5): no aprender verb was found to be a filename-pattern synthesizer. It is still worth requiring, because the field costs one line and the failure it prevents is undetectable from outside. Do not cite pmat's `refactor.*` as if it were our defect.

### 2.3 The emitter must be derived — and the substrate exists

**Resolved (T1).** The emitter walks `Cli::command()` recursively via `clap::CommandFactory`. This is not speculative: `crates/apr-cli/src/help_producer_truth.rs:179-180` already does it, to resolve every backtick-quoted `apr …` invocation in help text against the live parser. That file is the working proof that the clap tree is programmatically enumerable, complete, and includes flattened groups.

**Consequence for SURF-10: reject.** v1 proposed porting rmedia's `VerbRegistry` to `apr`. For the CLI leg that would create copy #5 of the surface — the thing R3 is about. The clap tree already *is* the single registry; a second one can only disagree with it. SURF-10 survives only if the MCP/HTTP legs later need a shared param type, and that is SURF-9's problem, not a precondition for it.

**The discrimination test in §7 remains the only thing standing between this design and R3.** Unchanged and non-negotiable: adding a hidden subcommand to the dispatcher, touching nothing else, must turn the lock red.

### 2.4 Hashing substrate

Descriptors → JCS canonicalization (RFC 8785) → SHA-256 → flat manifest root. **Floats forbidden anywhere in a hashed schema** — `serde_json` emits arbitrary-precision numbers that are not RFC 8785-conformant and canonicalizers coerce them to IEEE-754 silently. Do not hash YAML bytes: parse → typed struct → JCS → hash.

The substrate does not exist in this repo. See §12.2.

---

## 3. Constraints at registration, not translation at runtime

Unchanged in principle; **changed in cost.**

The three transports have genuinely different parameter shapes. A flattening rule mapping nested structs to CLI flags is **not injective** — dot-notation `{a:{b:1}}` collides with a field literally named `a.b` — so a translation layer accommodates the hazard instead of removing it. Reject non-representable schemas at registration:

- Bounded nesting depth. **v1 proposed 2 and admitted nobody had counted. Phase A counted the wrong thing to settle it** — command nesting is depth 2 (`apr registry add`), but *param struct* nesting is unmeasured. Still measure before fixing.
- **No untagged `serde` enums.** `schemars` renders them as `anyOf`, the loosest JSON Schema keyword: a payload satisfying two poorly-constrained arms passes silently with unusable errors. **Now retroactive (T7):** 17 sites exist workspace-wide, 4 on the surface path including a live request type at `crates/apr-cli/src/commands/serve/types.rs:295`. Introduce with an explicit, shrinking grandfather list — a ratchet, not a wall. A wall here fails on day one and gets `--skip`ped, which is worse than no rule.
- No maps with open key sets.
- No bare floats (§2.4).

The canonical schema is the JSON representation; CLI is a projection. A verb whose params cannot project losslessly is rejected at registration with a named error — never accepted with a note.

---

## 4. Four layers

| Layer | What it proves | Cost | Status at HEAD |
|---|---|---|---|
| **L0** liveness | binary runs, answers `--version`/`--help`, handles SIGTERM, stdin, error paths | low | Partly present: `test_version_flag`, `test_no_args_exits_usage_error` (`cli_commands.rs:292,309`). Kaizen K0–K7 is in `infra` — UNVERIFIED |
| **L1** shape | the verb set, schemas, auth, exit codes, protocol, toolchain are exactly as locked | very low | **Depth-1 command set: DONE and blocking** (`cli_commands.rs`, `ci.yml:327`). One HTTP router: DONE (`route_surface_2376.rs`). Everything else: absent |
| **L2** transport equivalence | same verb + params across legs → equivalent output and *identical error taxonomy* | medium | Absent. Scope is 9 verbs, not 105 (T3) |
| **L3** behaviour | per-verb oracle, stream discipline, multi-turn state | high | Fragmentary. Panic containment absent (R14) |

**L1 is the ratchet** — near-free per PR, catches deletion, rename, param narrowing, transport drop, protocol bump, toolchain shift. **And Phase A shows the ratchet already turns; it just has a short handle.** The work is extending its subject from `{name}` to `{name, path, params, exit_codes, protocol, toolchain}`, not building a lane.

**L2 equality is over the reassembled result, not bytes.** `apr serve` streams; the CLI batches. A single-shot comparison is a step-0 probe — the CF-4 signature. Streaming *order* is a separate invariant from streaming *content*.

**L3 is the corpus that grows by failure archaeology.** Every escaped defect adds one permanent falsifier. It never shrinks.

---

## 5. Gates

### 5.1 Three artifact tiers — tier 0 rehabilitated

v1 declared tier 0 (`cargo run`) *"proves nothing — this is R1 wearing a costume."* **That is too strong, and Phase A proves it too strong:** `check_readme_claims.sh:63` uses `cargo run` and it does compile and execute a real binary. What `cargo run` fails to prove is *feature-resolution and publish parity*, which is a narrower and more defensible objection.

| Tier | Command | Proves | Cadence |
|---|---|---|---|
| 0 | `cargo run --bin apr -- surface --emit` | the **dev-profile workspace** artifact under `apr-cli`'s own default features — which is **not** how the published `apr` is built (that goes through the facade's `cli` feature, `Cargo.toml:524-527`) | acceptable for a text claim; **not** for a lock |
| 1 | `env!("CARGO_BIN_EXE_apr")` or `./target/release/apr surface --emit` | the compiled workspace artifact, built the way the gate declares | per-PR, inside `gate.needs` |
| 2 | clean-room A4 verify installed binary (`cargo install` from crates.io, no sibling mounts) | the **published** artifact | release hard gate |

**Tier 1 green with tier 2 red is signal, not noise** — it means the workspace build and the crates.io build resolve different features or path deps. For `apr` this is concrete: the facade is what crates.io installs, and it reaches `apr-cli` through `cli` (`Cargo.toml:519,527`), so *"which verbs does `cargo install aprender` actually ship"* is answerable only at A4. **What A4 asserts today is UNVERIFIED from this session (T9)** — SURF-5 cannot be written until someone reads `infra`.

Clean-room remains a hard release gate. Note the huge-crate caps (`CARGO_BUILD_JOBS=2`, doctest `--test-threads=4`) or the container OOMs into spurious failures.

### 5.2 BREAKING is mechanized, never delegated to a reviewer

Unchanged, and Phase A reinforces it. The known fatal flaw of a lockfile gate is snapshot rubber-stamping: an opaque hash mismatch trains the reflex "regenerate and merge." The fix is **not** a manual-approval step — `main`'s protection matches required contexts by literal string (`ci / gate`, `workspace-test`), the org ruleset is never bypassed, and adding a human-approval context reintroduces exactly the attention that rubber-stamping defeats.

```
classify(head, lock) == BREAKING  ⇒  manifest.contract_version > lock.contract_version
```

CI asserts the implication. No human in the loop.

### 5.3 The classifier is a trusted component and needs its own proof

`surface --diff` deciding what counts as BREAKING is a new engine every merge decision routes through. It is also the best Kani target here — small, total, bounded:

```
classify(a, b) = BREAKING  ⟺  ∃ v : valid(v, a) ∧ ¬valid(v, b)
```

`metadata.kind: kernel` with mandatory Kani, alongside the canonicalization equations. `pv validate` defaults to `kernel`; set it explicitly on the pattern contracts so the default never silently applies.

### 5.4 Feature matrix

Every feature that alters the surface needs its own locked combination and its own matrix entry. At minimum: `default`, `--no-default-features`, and the combination that produces the published `apr`.

**Concretely, from `Cargo.toml:524-545`:** the facade exposes `cli, cuda, cuda-batch, wgpu, inference, training, training-gpu, visualization, zram, xet, whisper, ptx` — every one a passthrough to `apr-cli`. `apr-cli` itself carries `dev`, which gates `Mono` (`commands_enum.rs:801`) and therefore **changes the top-level command set**. `dev` is the one flag already known to move the surface; it must be in the matrix from day one.

---

## 6. Contracts

Every pick ships its `pv` contract in the same PR, bound in the registry.

| Contract | Kind | Key invariants | Falsifier that must turn RED |
|---|---|---|---|
| `apr-surface-canon-v1.yaml` | kernel | `canon(canon(x)) == canon(x)`; verb ordering total and stable; no float reaches the canonicalizer | skip the second sort pass |
| `apr-surface-lock-v1.yaml` | kernel | root changes **iff** any verb descriptor changes; verify refuses on mismatch | make the verifier return `Ok(())` unconditionally |
| `apr-surface-classify-v1.yaml` | kernel | `classify = BREAKING ⟺ ∃v: valid(v,a) ∧ ¬valid(v,b)`; total | classify `Option<T> → T` as ADDITIVE |
| `apr-surface-derivation-v1.yaml` | pattern | emitted verb set equals the clap tree; no hand-maintained list participates | add a hidden subcommand to the dispatcher, touch nothing else |
| **`apr-surface-depth-v1.yaml`** (new, SURF-13) | pattern | every nested subcommand path in the clap tree appears in the lock; depth is recorded, not truncated | delete a variant from `RegistryCommands` |
| **`apr-contract-enforcement-v1.yaml`** (new, SURF-14) | pattern | every `enforcement:` string in `contracts/` resolves to an existing cargo test target **and** an existing test fn | it fails today — FALSIFY-CLI-001..005 name `--test apr_cli_commands`, which does not exist (R11) |
| `apr-surface-transport-v1.yaml` | pattern | per-transport auth honoured; error taxonomy identical across legs; reassembled stream ≡ batch output. **Scoped to the 9 dual-leg verbs and to a named `http_router`** | HTTP returns 500 where CLI exits 5 |
| `apr-surface-stream-v1.yaml` | pattern | `stdout ⊆ valid protocol frames`; a handler `panic!` yields a valid JSON-RPC error frame and a valid JSON 500; stderr is host telemetry only | flip `crates/aprender-mcp/src/tools/subprocess.rs:174` to `Stdio::inherit()` — see §8 |

**Kani harnesses:** JCS idempotence, verb-set total ordering, root-hash sensitivity, classifier soundness. Bound collections explicitly with `#[kani::unwind]`.

**Property tests** (`proptest`, or `bolero` to run one harness under both proptest and Kani): emit → canonicalize → parse round-trip; `hash(x) == hash(parse(serialize(x)))`; determinism across repeated runs of the same artifact.

**SURF-14 is the cheapest pick in this table and it is red on arrival.** A contract registry whose enforcement pointers do not resolve is describing a gate rather than naming one — the same species of defect as R1, one level up. Ship it first; it costs a morning and it audits 1771 contracts.

---

## 7. Mutation set — and the discrimination requirement

A mutation that trips a *different* gate first proves nothing. **Every mutation must be checked for discrimination before it counts as evidence.**

| Layer | Mutation | Must turn RED | Discriminates against |
|---|---|---|---|
| L1 | add a hidden subcommand to the dispatcher, change nothing else | lock mismatch | **the R3 test — proves the manifest is derived.** *Discrimination check: `cli_commands.rs::test_no_unregistered_commands` fires first on a **visible** subcommand. The mutation must use `#[command(hide = true)]`, which that test cannot see because it parses `--help` output* |
| L1 | delete a variant from `RegistryCommands` | depth lock mismatch (SURF-13) | **R12 — nothing at HEAD notices this** |
| L1 | narrow a param from `Option<T>` to `T` | schema digest change **and** BREAKING classification | classifier soundness |
| L1 | bump `PROTOCOL_VERSION` without adapter change | lock mismatch | R9. *Discrimination check: `server.rs:1016` asserts the literal `"2024-11-05"` and fires first — the mutation must change the const **and** that assertion, which is exactly the two-copies problem the lock replaces* |
| L1 | build the gate artifact with a different toolchain | lock mismatch | R10 |
| L1 | rewrite an `enforcement:` string to a non-existent target | SURF-14 falsifier | R11. *No discrimination risk: nothing else reads that field* |
| L2 | HTTP adapter drops a field the CLI emits | parity falsifier | derived-adapter claim |
| L2 | HTTP returns 500 where CLI exits 5 | error-taxonomy falsifier | — |
| L2 | strip the bearer check on one transport | that transport's `unauth_status` falsifier | per-transport auth model |
| L3 | `println!("debug")` in a verb handler | stdout-purity falsifier | R6 |
| L3 | **`crates/aprender-mcp/src/tools/subprocess.rs:174` → `Stdio::inherit()`** | stdout-purity falsifier | **R6, the sharp case. Revised from v1 — see §8** |
| L3 | `panic!` inside a handler | valid JSON-RPC error frame + valid JSON 500; process survives | **fails at HEAD (R14) — this is a finding, not a future test** |
| L3 | flip a verb's engine to a constant or filename heuristic | `oracle` invariant | R5 (prophylactic) |
| L3 | drop tool_call id remapping on turn 2 | multi-turn falsifier | R4 |

**Canary requirement.** Land a deliberately unkilled mutant on a throwaway branch and confirm CI goes red before declaring any gate live. A gate nobody has watched fire is a gate nobody knows works.

---

## 8. The subprocess hazard — inverted by Phase A

v1: *"Whether `apr` shells out is UNVERIFIED. …the offending line contains no output statement."*

**Verified, and the conclusion flips.** `apr` shells out heavily — 25+ non-test `Command::new` sites in `crates/apr-cli/src`, to `cargo`, `nvidia-smi`, `python3`, `curl`, `bash`, `gh`, `ollama`, and to `apr` itself. Several inherit stdout by using `.status()` with no redirect: `crates/apr-cli/src/commands/pipeline.rs:32`, `crates/apr-cli/src/commands/train.rs:701,727`, `crates/apr-cli/src/commands/mono.rs:306,314,430`.

**But none of those is on a protocol stream.** They are CLI paths, where stdout is a terminal and inheritance is the desired behaviour. The one place a child could corrupt a frame stream — the MCP boundary — **already pipes both streams**: `crates/aprender-mcp/src/tools/subprocess.rs:174-175` and `:342-343` set `Stdio::piped()` on stdout and stderr, and every one of the 9 MCP tools routes through that module.

Three consequences:

1. **The v1 mutation "spawn a subprocess with inherited stdio" is not discriminating** — no MCP tool spawns directly, so the mutation has nowhere to land. Replace it with **flip `subprocess.rs:174` to `Stdio::inherit()`**, which is a one-token change to the single chokepoint that today prevents the class.
2. **The invariant to write is a structural one:** *no `Command` reachable from an MCP tool may be constructed outside `crates/aprender-mcp/src/tools/subprocess.rs`*. That is checkable by lint, cheap, and it is what actually holds the property — piping in one place only works while the chokepoint is the only door.
3. **`apr_bin::apr_binary()` (`crates/aprender-mcp/src/apr_bin.rs`) is the shadowed-artifact defense and it is already correct** — when the host process *is* `apr`, it delegates to itself rather than `$PATH`. Its module docs record the exact field failure (a 0.63.0 `apr mcp` running `/home/noah/.local/bin/apr` at 0.60.0 for all eight subprocess tools while `apr.version` answered 0.63.0). **The surface lock must record which binary the MCP leg delegated to,** or a tier-2 lock verifies a manifest emitted by one binary describing another.

---

## 9. Reference systems consulted (quorum)

- **Derived multi-transport surfaces:** rmedia `rmedia-verb` / Unified Verb Surface (GH-243) — one `VerbRegistry`, CLI + MCP stdio + HTTP/axum as derived adapters. **Still the model for SURF-9's MCP/HTTP legs; rejected for the CLI leg (§2.3).** And there is closer prior art in this repo: `crates/aprender-serve/src/api/router.rs` derives mount and advertisement from one `Route` table, for the same reason, with a written post-mortem of the three routes the two-copy version lost.
- **Lockfiles and pinning:** `Cargo.lock`; `package-lock.json` (`npm ci` refusal semantics — integrity is not authority); GitOps reconcilers (Flux/ArgoCD) pinning a revision and refusing to operate against a mismatch.
- **Content addressing:** OCI image descriptors (verify digest *and* size); Nix input-addressing and pre-hash canonicalization; RFC 8785 JCS.
- **Schema evolution:** Kubernetes CRDs — served vs storage versions, hub-and-spoke conversion, and the `status.storedVersions` data-loss guard, the model for `contract_version` bumps in §5.2.
- **Verification:** Kani (bit-precise bounded model checking); `cargo-mutants --in-diff` for PR-scoped blocking runs.

---

## 10. Do-not-do

1. **A lock built on `cargo run`.** §5.1 — acceptable for a text claim, never for a lock, because it resolves `apr-cli`'s features rather than the facade's.
2. **A verb-level `auth` field.** §2.1. Makes the CLI leg of the L2 invariant false while green.
3. **A CLI flattening rule for nested params.** §3. Not injective. Constrain at registration.
4. **Untagged `serde` enums in any *new* verb param type.** §3 — and ship the rule as a shrinking grandfather list, because 4 already exist on the surface path (T7).
5. **Bare floats in a hashed schema.** §2.4. JCS coerces them silently and lossily.
6. **Hashing YAML bytes.** §2.4. Certifies a byte string, not the value the consumer loaded.
7. **A required manual-approval context for BREAKING changes.** §5.2.
8. **A fourth lock primitive.** Extract or share one. §12.2.
9. **Shipping the lock without the classifier.** §5.2/§5.3.
10. **A feature flag absent from the feature matrix.** §5.4 — `dev` first, it already moves the command set.
11. **`apr surface --diff` that warns, repairs, or auto-relocks.** It classifies and it refuses.
12. **Fleet-wide rollout before SURF-11 reports.** §12.4.
13. **Declaring any gate live before its canary has been watched to fire.** §7.
14. **(new) A parallel verb registry for the CLI.** §2.3. The clap tree is the registry; a second one is copy #5, and R3 is about copies.
15. **(new) Citing another repo's scar as this repo's defect.** §1A/R5. v1 imported pmat's `refactor.*` and rmedia's `lint-doc-accuracy.sh` as if they were aprender findings. They are good *illustrations* and bad *evidence*; keeping the distinction is the whole point of §13.1.

---

## 11. What this spec does not fix

- **A surface lock pins shape, not behaviour.** L1 is the cheap total layer; what matters behaviourally is L2/L3, expensive and growing one falsifier at a time.
- **Schema digests are coarse.** A `clap` bump changes digests with zero semantic change — hence the pin in §2.1, and hence a bump is a deliberate reviewed re-lock, not an incident. `clap` currently floats from a declared `4.5` to a locked `4.6.1` (T6), so the first re-lock will be noisy.
- **Nothing here detects a verb that is present, correct, and useless.** Surface conformance is not product correctness; the beat gates remain the instrument for that.
- **(new) Nothing here fixes the three-router ambiguity (R13).** The lock can *record* which router answered; deciding that `apr serve` should have one HTTP surface instead of three is a separate architectural decision with its own ticket, and SURF-9 is partly blocked behind it.
- **(new) `infra`-side picks are unscoped.** SURF-5 and SURF-6 name artifacts this session cannot read (T9, T10). They stay OPEN with a named blocking artifact rather than being written speculatively.

---

## 12. Open decisions — status after Phase A

1. **Registry-first or lock-first? → RESOLVED: lock-first, and no new registry for the CLI.** T1 settles it: the clap tree is a real, complete, already-programmatically-walked dispatch table (`help_producer_truth.rs:179`). SURF-1..6 proceed on it. **SURF-10 is rejected for the CLI leg** (§2.3, do-not-do #14). It survives only as a possible shared param type for the 9 dual-leg verbs, which is inside SURF-9's scope, not a precondition for it.
2. **CAT-1 duplication → STILL OPEN, and still blocking SURF-2.** rmedia publishes nothing on crates.io, so a shared crate dependency remains impossible and Tier A/vendor is unavailable. Three options unchanged: implement twice at Tier B and accept divergence; publish the rmedia primitive and depend on it; or land it in aprender and have rmedia consume it. **Recommendation, new:** land it in `aprender-contracts` and let rmedia consume it. aprender already publishes to crates.io on a release cadence, already owns `pv`, and is the repo with 1771 contracts that will exercise the primitive hardest. **Decide before implementation starts, not after.**
3. **Does aprender expose HTTP and MCP? → RESOLVED, with a complication.** MCP: yes, `2024-11-05`, 9 tools, schemas already codegen'd from a contract. HTTP: yes, but **three routers** (R13), and **no `/v1/messages`** — v1's Anthropic-compatible leg is still a proposal. SURF-9's HTTP leg must therefore either name one router or wait on the decision to unify them.
4. **The churn threshold that reverses SURF-12 → STILL ASSERTED, NOT MEASURED.** Proposed: >1 re-lock per 3 PRs on `apr` over the first 30 PRs means the manifest is too fine-grained. SURF-11 exists to replace this with an observation. Record the value at which the fleet decision reverses.
5. **Nesting depth bound for §3 → HALF MEASURED.** *Command* nesting is now measured and is **exactly 2**, with 0 commands at depth 3 (§1A.5). *Param struct* nesting — which is what §3 actually constrains — is still unmeasured. v1 proposed a bound of 2 for the latter; that number remains asserted. Measure before fixing.
6. **(new) Should `apr serve` have one HTTP surface? → OPEN, blocks SURF-9's HTTP leg.** Three routers selected by input file format (R13) is a defect the tree has already documented against itself. Fixing it is out of this spec's scope but inside its critical path.

---

## 13. Refinement protocol

### 13.1 Grounding contract
Every claim is backed by an artifact fetched at HEAD: a path:line in a pinned worktree, a commit SHA, `gh` output, crates.io metadata, or CI logs. Nothing from training data, nothing from this document's snapshots. **§1A is the worked example**, including where it falsified this document's own previous revision.

### 13.2 Verdict discipline
Emit VERIFIED (with path) · STALE (claim ≠ tree — give both values) · UNVERIFIED (could not fetch). **Falsifying a claim in §1 is a success.** Distinguish *exists* / *wired into CI* / *wired into a BLOCKING gate*. "This design is wrong" is a permitted conclusion.

### 13.3 Phase A — **COMPLETE.** See §1A.
Four of ten v1 regression classes were falsified (R1, R2 at depth 1, R7, R8), two had their values corrected (R9, R10), one was demoted to prophylactic (R5), one was inverted (R6/§8), and four new classes were found (R11–R14). Two questions remain UNVERIFIED for lack of repo access (T9, T10).

### 13.4 Phase B — refine against what you found
**Done in this revision.** Where the tree contradicted the spec, the tree won and the section was rewritten: §0, §2.1, §2.3, §5.1, §8, §12.1, §12.3. Where the spec proposed something the tree makes harder, the cost is stated: §3 (retroactive untagged enums), §12.2 (no shared crate possible), §12.6 (three routers).

### 13.5 Phase C — contracts before code
For each pick, write the `pv` contract **first**: invariants, `falsification_tests`, `kani_harnesses`, explicit `metadata.kind`. Bind it in the registry. Run `pv validate`. **Note (T13): `pv` and `pmat` are absent from the standard dev container — Phase C must run on a host that has them.**

For every gate, name the exact mutation that must turn it RED **and check the mutation discriminates** (§7 now carries discrimination notes for the three mutations where an existing test fires first). A contract you cannot write a discriminating falsifier for marks its pick `UNDERSPECIFIED` — report it, do not ship it clean.

### 13.6 Phase D — ticket and implement
`pmat work add` per pick; check `pmat work list` first. Then branch → TDD, failing test first → PR → `ci / gate`. Required contexts are literally `ci / gate` and `workspace-test`; `main` is protected and never pushed directly. Small atomic commits with the ticket id in every trailer. Update §0's status table in the same PR as each pick.

**Suggested order, revised by Phase A cost/benefit:**

1. **SURF-14** (enforcement pointers resolve). Red on arrival, audits 1771 contracts, costs a morning, and fixes R11 — a defect in the *contract layer itself*, which everything else here depends on.
2. **SURF-13** (nested-subcommand surface). Largest genuine gap (R12); reuses the proven `Cli::command()` walk and the proven `CARGO_BIN_EXE_apr` harness; needs no new lock primitive.
3. **SURF-2 / SURF-1 / SURF-3** (lock primitive, emitter, classifier) — but only after §12.2 is decided.
4. **SURF-7 panic leg** (R14) — independent of everything above and independently valuable.
5. Everything else.

Two operational notes: PRs touching `.github/workflows/*` need a web-UI merge click (the `gh` token lacks `workflow` scope), and `pr-gate` auto-closes PRs from members whose org membership is private.

### 13.7 Constraints
Stop the line — no `--skip`, no suppressed lints, no lowered thresholds, no `|| true`, no `continue-on-error`. Five-whys to the owning crate; fix at root. Every feature ships its contract in the same PR. Anything ending in a publish names the clean-room gate first. If you find yourself reasoning that a check does not matter for this particular change, escalate rather than proceed.

### 13.8 Definition of done
A reviewer takes any row of §0 and, from this spec plus §1A, either opens the ticket or rejects the pick — with zero follow-up questions. Every §12 decision is resolved with evidence or restated with the exact artifact that would resolve it. Anything unverifiable is named (T9, T10), not smoothed over.

---

## 14. Provenance

**Design basis:** conversation of 2026-08-15 and its architectural audit. Regression classes R1–R10 were derived in v1 from documented failures; **R1, R2, R7 and R8 have since been falsified against this repo at HEAD**, and R11–R14 were derived from the Phase A sweep that falsified them.

**Repo state:** every aprender fact in §1A cites `origin/main` @ `907bef27`, 2026-08-15, read directly. v1's facts cited the Fable architectural review at `9f0b89fb` (2026-07-05); that snapshot is retained only where §1A confirms it.

**Known staleness carried forward, and what happened to it:**

| v1 recorded | Status at `907bef27` |
|---|---|
| `apr-cli-commands-v1.yaml:24` prose says 77 vs 103 registered | **Still rotten** — prose says 77, registry holds 105. Nothing reads the prose |
| README contract count 1331 vs 1460 | **Fixed** (#2485) — README says 1771 in three places, `find` counts 1771 |
| test count claims ~3.3× stale | Not re-checked; out of scope |
| `docs/BEATS.md:119` says 10/10 where contract says 11 | Not re-checked; out of scope |

**Crate versions drift.** `clap` floats from a declared `4.5` to a locked `4.6.1` today. Re-pin at implementation time. The design choices are stable; the semver is not.

**This document is a snapshot. Where it disagrees with HEAD, HEAD wins — and §1A is the record of the last time that happened to its own previous revision.**
