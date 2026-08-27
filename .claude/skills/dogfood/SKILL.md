---
name: dogfood
description: Sovereign-stack PRE-RELEASE protocol. Run this BEFORE generating a GitHub or crates.io release of any Rust crate in the fleet (copia, forjar, pmat, trueno, aprender, …). It runs every local release gate (fmt, clippy, tests, ≥95% coverage, cargo-deny, provable contracts) through the deterministic tools — pv, bashrs, pmat — verifies the release's CLI/HTTP/MCP interfaces against the spawned binary — including a DERIVED, SIMULTANEOUS cross-transport invariance check that invokes the same verb through every declared transport while all of them are live and requires byte-identical results, does a publish dry-run, and DOGFOODS the crate by using its own release binary on real data. Then it requires the MANDATORY clean-room gate and gives a go/no-go verdict with a receipt. Use when the user says "dogfood", "pre-release", "release checklist", "ready to publish/release", "cut a release", or asks whether a crate is release-ready.
---

# dogfood — pre-release protocol

The gate that stands between "CI is green" and `cargo publish`. It codifies the
sovereign quality bar (CLAUDE.md) into one repeatable, evidence-producing protocol.

**Toyota way, non-negotiable:** any RED gate STOPS the release. Fix the *root cause*
in the crate — never `--no-verify`, never `--skip`, never lower a floor to pass.

## There is exactly one of everything here

This file and `scripts/dogfood.sh` are the ONLY copies of the protocol, and both
live in the aprender repo, in git, reachable by CI and by `git diff`.

They did not used to be. Until #2640 the runner existed twice — once here and once
at `~/.claude/skills/dogfood/dogfood.sh` — and the two had silently diverged in nine
places. Every one was a real hardening the other copy lacked, so *neither copy was
safe to delete and neither was safe to trust*
(`docs/audits/dogfood-divergence-2640.md`). The user-scope path is now a ~50-line
shim that `exec`s this runner; `scripts/check_dogfood_shim.sh` keeps it one, and
`scripts/check_verifier_pinning.sh` keeps the merged hardenings honest. If you find
yourself about to copy either file somewhere, read that triage first.

## Every gate has a SUBJECT, and some of them are not this tree

Read this before adding a gate, and before concluding that a permanently-red one
is debt.

This runner's subject is **the local tree**: `$BINPATH` comes from
`cargo build --message-format=json`, and every gate inherits that. But three
gates ask about something else, and their subject is what makes them behave the
way they do:

| gate | subject | when it can pass |
|---|---|---|
| `version-unpublished`, `publish-dry-run` | the **registry** | before / after the cascade respectively (#2643) |
| `check_multiplatform_dogfood` | the **published artifact, on four hosts** | only AFTER publish (#2658) |
| everything else | **this tree** | now |

**A gate whose subject is a later phase is not broken and is not debt.** Its RED
is structural, and treating it as debt is how it becomes a step everyone learns
to walk past. `check_multiplatform_dogfood` had never passed for ANY release
until 0.64.0 — 0.63.0 published 2026-08-01 with receipts dated 2026-08-22 —
because the question it asks cannot be answered before the thing it asks about
exists.

That is the root cause recorded in aprender#2662: **phase was an undeclared
dimension of this protocol.** Gates carried an implicit subject, the runner
recognised only the first, and the producing half of the later ones went
unwritten — which is why the 0.64.0 four-host sweep was hand-rolled, in a
protocol whose own rule is *"Never dogfood by hand"*.

**So: when you add a gate, say what its subject is.** If the answer is not "this
tree", the gate belongs to a later phase, its receipt is produced there, and its
RED before that phase means *not yet measured* rather than *broken*.

**And the two phases do not measure the same thing** — but the earlier version
of this paragraph drew the wrong conclusion from that, and it cost a release.

It said: the post-publish phase CANNOT compute a llama.cpp ratio, because
`cargo install aprender` is CPU-only on every host (`default` carries no `cuda`,
no `wgpu`) while the comparator runs CUDA or Metal, so the ratio reads
~0.05–0.10 and the threshold never arms. Every clause of that is TRUE. The
conclusion drawn — *therefore do not compare post-publish* — is what was wrong.

What it cost: the published binary's CPU-only-ness sat in this file as a stated
fact while nothing ever measured what it costs a user. Measured for the first
time on 2026-08-24, on a host with an idle RTX 4090: **15.7 tok/s decode against
llama.cpp's 158.9, and 7.5 SECONDS to first token**, because `apr serve run
--gpu` accepts the flag, links no CUDA at all, warns nothing, and returns a
plausible number (aprender#2696).

The remedy is not to skip the comparison. It is to **compare within the compute
class**. The published apr takes the cpu path, so its comparator is llama.cpp
`-ngl 0`, which also takes the cpu path. That ratio is meaningful, it arms a
floor (**0.80 minimum, 1.50 stretch**), and no cross-class row is created. The
accelerated lanes stay pre-publish, from the tree, where the feature exists
(aprender#2667).

`scripts/lib/bench_receipt.py --parity` enforces the rest, and
`scripts/check_parity_receipt.sh` proves it discriminates across 17 cases before
any release reads its verdict:

| rule | what it stops |
|---|---|
| same class, or no verdict | a cpu-vs-cuda number reading as a kernel defect |
| the ratio is DERIVED from the samples | F12 — a stated ratio its own samples do not produce |
| the comparator is pinned | a denominator that moves silently between releases |
| the subject names its artifact | #2696 — local build and published binary differ by 6.6x |
| the verdict follows from the floor | a gate lying about its own rule |

Required lanes come from each host's **own declared accelerator**, so a host
that gains a GPU gains a required lane without anyone remembering to add one.

## Run it

From (or pointing at) the crate repo:

```bash
bash <aprender>/scripts/dogfood.sh [REPO_DIR]   # defaults to $PWD
cd <aprender> && bash scripts/dogfood.sh ../other-crate   # the relative form works too
# ~/.claude/skills/dogfood/dogfood.sh still works: it is a shim onto the above.
```

The relative form is the fleet path, and it is gated (PART 3 of
`scripts/check_verifier_pinning.sh`). It briefly was not: `SKILL_DIR` was resolved
after the runner had already `cd`-ed into the target repo, so a relative
`${BASH_SOURCE[0]}` pointed at the wrong tree and the fail-closed pin library was
"missing" — exit 2 before any gate ran, while the absolute form kept working.

### A crate's own release gates are DECLARED, never copied in

A repo that has written its own release guards declares them in its `Cargo.toml`,
beside `[package.metadata.transports]`, and the runner discovers and runs each one
with its own row in the receipt:

```toml
[package.metadata.dogfood]
gates = ["scripts/check_multiplatform_dogfood.sh", "scripts/check_verifier_pinning.sh"]
```

This is the anti-duplication rule applied to the runner itself. Without it, the way
to get a repo-specific gate into the protocol is to edit the protocol — which is how
it acquired a second copy of itself in the first place.

Both vacuity guards are hard FAILs, not SKIPs: a declared script that does not exist
is a *deleted* gate, and a missing or empty `gates` list is a clean sweep over an
empty set — the signature failure this protocol exists to refuse, reappearing one
layer up at discovery.

`DOGFOOD_GATES_ONLY=1` runs only that section, for exercising discovery without a
full sweep. It can never print GO; a partial run is not a verdict.

It writes a receipt to `<repo>/.dogfood/receipt-<ts>.json` — stamped with the
commit SHA it describes, written as `.partial` and atomically renamed on
completion, so a crashed run leaves NO completed receipt rather than a stale or
garbage one — and exits `0` = GO, `1` = NO-GO, `2` = setup error, `3` = the
receipt it just wrote is unreadable (verdict withheld). On NO-GO it prints every
red gate with its note before the verdict —
a verdict that says only "a gate failed" makes the reader hunt for it, and the
hunt is where bypasses start. Report the verdict and the failing gate(s) to the
user; do not proceed to release on a NO-GO.

It creates three artefact dirs in the repo — `.dogfood/` (receipts), `.pmat/`
(the context index CB-200 cannot run without) and `.pv/` (pv's lint cache). They
are excluded from the `git-clean` check because the protocol creates them; add
all three to the crate's `.gitignore`.

## What it checks (and why each matters)

1. **git-clean** — uncommitted work must be committed before a tag points at it.
2. **version-unpublished** — the `Cargo.toml` version is not already on crates.io
   (crates.io versions are immutable; a re-publish fails). Bump first if it is.
3. **changelog** — `CHANGELOG.md` has an entry for this version.
4. **fmt / clippy / test** — `cargo fmt --check`, `clippy -D warnings` (pedantic),
   the full test suite. Clean-room-published crates gate the binary behind the
   `cli` feature, so those run with `--features cli`.
5. **coverage** — `make coverage-check` (the ≥95% whole-crate floor).
6. **security** — `cargo deny check advisories`.
7. **contracts** — `make contracts`, where the crate has one. It runs what this
   protocol cannot generically replicate: the Lean (L4) proofs and the
   falsification suites. **It is not the authority on the pv layer** — 7b/7b′/7b″
   run `pv` directly so the exit codes are ours. Its recipe is also checked
   statically for laundered exit codes (**contracts-exit-integrity**, 7e).

   The earlier text here claimed `make contracts` runs "`pv validate` + `lean
   lean/*.lean` + `pv proof-status --binding` + the falsification suites". That
   was true of copia and false of the repo the skill was most often pointed at:
   rmedia has no `lean/` directory, its recipe never invokes `proof-status`, and
   it contains no falsification suite. A protocol description that does not
   match the repo is the same defect as a gate that does not run.

7a. **deterministic-tools** — `pv`, `bashrs`, `pmat` and `probador` must be
   INSTALLED. Their absence is a **NO-GO**, not a WARN.

   These are the verifiers the protocol is built on, and they are *deterministic*:
   the same input yields the same verdict on any machine, unlike reading a diff
   and forming an opinion. A release verified by tools that were not installed is
   an unverified release that prints GO.

   This used to WARN, contradicting this file's own rule that "a WARN in a
   release protocol is a step everybody learns to walk past". On 2026-08-16
   `pv validate` was run across forjar's contracts for the first time and **six
   had never passed it** — including the one `forjar prove` maps into its own
   `N/N proofs passed` line. Nothing had run the validator, so nothing knew.

7b. **pv-contracts** — every `contracts/*.yaml` validated individually, behind a
   **positive control**: a deliberately malformed contract is validated first and
   must be REJECTED. If `pv validate` accepts it, "N contracts valid" is a count
   of files, not a verdict, and the gate says so instead of reporting green.
   A `contracts/` directory that yields *zero* validatable files is a FAIL: a
   glob matching nothing passes vacuously. `binding.yaml` is excluded by name —
   it is a binding registry, not a contract.

   **Never replace this loop with `pv lint <FILE>`.** Verified three ways in
   pv 0.49.0, all exit 0 printing `Result: PASS`: on a contract `pv validate`
   rejects, on `binding.yaml`, and **on a path that does not exist**. `pv lint`
   reads directories; handed a file it lints zero contracts and calls it a pass.

7b′. **pv-lint** — `pv lint <DIR>` (directory form only), the 8-gate schema
   sweep: duplicate ids, dangling refs, reverse coverage, composition. Errors
   gate; its warnings do not.

7b″. **pv-bindings** — `pv verify-bindings <binding> --crate-name <crate>`, run
   from the crate root. This answers a question nothing else in the protocol
   asks: does every binding name a symbol that **exists**? `pv proof-status
   --binding` does not — pointed at
   `ThisFunctionDoesNotExistAnywhere::nope` it printed `Bindings 2/2` and exited
   0. It counts entries; it does not resolve them.

   Ghost findings are **REPORT, not FAIL**, and the reason is written down so it
   can be re-armed. pv 0.49.0 misses `pub(super) fn` (rmedia's `apply_loudnorm`
   is reported a ghost and is real at
   `crates/rmedia-core/src/audio_cleanup/normalize.rs:27`) and private `async fn`
   in a `[[bin]]` module — copia's **only** ghost, `deliver_local`, is real at
   `src/bin/copia/incremental.rs:330`. It also matches prose-qualified
   `function:` fields literally (`all_entries (slug field)`) and is module-blind:
   a binding with the right function name under a module that does not exist
   verifies clean. Making it blocking today would be permanent red on false
   positives, and a permanently red gate gets bypassed for substance. What DOES
   fail here is the tool producing **no verification line at all** — an unrun
   verifier is not a clean one. Enforce R1–R4 meanwhile with a crate-local linter
   (rmedia's `scripts/lint-contract-bindings.sh` is the reference; it is stricter
   than pv on module-path awareness and `pub(super)` visibility).

7c. **bashrs** — gated on the rules that are trustworthy, reporting the rest.
   `SEC*`/`DET*`/`IDEM*` findings **FAIL** the release: SEC011 caught a real
   unguarded `rm -rf "$tmp"` in a forjar resource this week. `SC1020/SC1035/SC1140`
   are *reported but not gating* — they fire inside string literals
   (paiml/bashrs#226, still OPEN in 6.66.2; even `echo "done"` trips SC1035), and
   gating on them would make this unpassable. An unpassable gate trains people to
   bypass the whole protocol, so the suppression is deliberate, narrow, and
   carries the issue number to be re-armed against.

   **A clean bashrs result has three causes and only one of them is "the shell is
   clean."** So the gate proves the other two are false before believing it:

   - **Positive control.** A sentinel script carrying a known `DET002` is linted
     first and must be flagged. Cost 83 ms. Watched fail: with `bashrs` shadowed
     by `#!/bin/sh exit 0`, the gate reported *POSITIVE CONTROL FAILED … do not
     read a clean result as clean* instead of "0 errors".
   - **The scan receipt.** bashrs prints `Linted N file(s): …` on **stderr**, and
     the gate asserts N equals the count it enumerated itself with `git ls-files`.
     Without it a repo-root `.bashrsignore` silently zeroes the gate: verified,
     `bashrs lint --level error sub/bad.sh` on a file containing `DET002` printed
     `Skipped: sub/bad.sh` and **exited 0 with no receipt at all**. `--no-ignore`
     restores the finding (exit 2) and is mandatory. The receipt is only emitted
     for N≥2, so a known-clean sentinel is appended to the argv — otherwise a
     one-script crate has no receipt to assert.
   - **No pipeline, no `xargs`.** `xargs bashrs lint` remaps any exit in 1..125
     to **123**, so errors and warnings become indistinguishable. argv is built
     with a read loop.

   Exit codes are overloaded and must not be read as a ladder: `0` = nothing at
   or above `--level` **or** everything was ignored; `1` = warnings **or** "No
   lintable files found"; `2` = errors **or** the path does not exist. Only the
   receipt disambiguates. `--level` drives the exit code, not `--fail-on` —
   verified, `--fail-on error` on a warning-only file still exits 1.

   **Not covered, and named rather than implied:** GitHub Actions `run:` blocks
   (bashrs parses a whole workflow YAML as bash — pointed at rmedia's `ci.yml` it
   flagged `SC1020: Missing space before closing ]` on line 29, which is
   `branches: [main, master]`), and DET/SEC inside Makefile *recipes* (filename
   dispatches the rule set; identical content named `Makefile` gets MAKE001-020
   and loses DET002/SEC008/SEC015). Covering `run:` blocks needs a real
   extractor. That is work, not coverage.

7d. **cli-surface / transport-decl / transport-absence / interface-parity** —
   RELEASE INTERFACE VERIFICATION (CLI / HTTP / MCP).

   **cli-surface** enumerates the subcommands the release binary *advertises in
   its own `--help`*, then requires each to answer `--help` with exit 0. It
   catches a release shipping a subcommand that panics, was renamed, or is listed
   but unimplemented — a CLI claiming a surface it does not have. The surface is
   read from the artifact, so it cannot drift the way a hand-written fixture
   does. (forjar: 159 advertised subcommands, all answering.)

   **transport-decl** — every interface the release ships is DECLARED in
   `Cargo.toml`, versioned with the code it describes:

   ```toml
   [package.metadata.transports]
   cli  = { e2e = "e2e_cli_t" }
   mcp  = { e2e = "e2e_mcp_stdio_t", features = ["mcp"] }
   http = { e2e = "e2e_http_serve_t", features = ["http", "lua"] }
   ```

   No declaration is a **FAIL**, not a skip. It is one line per transport, and it
   converts "we have no HTTP surface" from an assumption into an enforced fact.

   **transport-absence** — the binary must not advertise an `mcp`/`serve`/`http`
   subcommand that no declaration covers. An undeclared transport is an
   unverified one, and this makes adding one without declaring it a failure.

   **interface-parity** — each declared transport's e2e target is run **by name**:
   `cargo test -p <crate> --features <f> --test <target>`. Three properties, and
   each has been watched fail:

   1. **The target exists and its features are enabled.** Naming it with `--test`
      is load-bearing: `cargo test --test e2e_http_t` without the feature is
      `error: target 'e2e_http_t' … requires the features: 'http'`, **exit 101**,
      while a bare `cargo test` exits 0 and does not mention the target once.
      Naming it converts absence into a hard error; not naming it is a
      gate-shaped silence.
   2. **It spawns the shipped binary.** The target's source must reference
      `CARGO_BIN_EXE_`. A parity suite that calls the library cannot see
      reachability — rmedia's four-way transport-parity suite was **GREEN for the
      whole period `mcp::serve_stdio` and `http::serve` had no caller from
      `main.rs`** (GH-247). The transports agreed with each other perfectly and
      were unreachable from the process entry point. Agreement cannot falsify
      reachability; that is this file's own "a test you author cannot falsify a
      premise you hold", with the unexamined premise being *the transport is
      wired up*.
   3. **It actually runs tests.** `test result: ok. 0 passed` is a vacuous pass
      and FAILs.

7d′. **transport-invariance — every transport LIVE AT ONCE, verbs DERIVED**

   `interface-parity` proves each transport is reachable and green *in its own
   e2e*. Necessary, and not sufficient: three transports can each pass their own
   test file and still disagree about what a verb **returns**, because nothing
   compares them — each e2e only ever sees its own surface.

   This gate invokes the **same verb through every declared transport while all
   of them are standing**, and requires byte-identical results.

   **DERIVED, never hand-written.** The verb list comes from the *binary*, via
   the `list` command the crate declares. A hand-written probe tests the verbs
   someone remembered; a derived one tests the surface that actually shipped,
   and grows when the surface grows. The gate FAILs when the binary lists
   nothing — a parity check over an empty surface is vacuously true, the same
   defect as a registry that silently empties.

   **SIMULTANEOUS, never sequential.** One transport at a time cannot
   distinguish *"they agree"* from *"they share a process-global that only one
   of them may hold at a time"*. All of them up together is the configuration a
   real client fleet produces, and the one that surfaces a shared listener, a
   shared lock, or a runtime that tolerates a single owner.

   It also asserts the agreed payload **parses as JSON** — two transports
   returning identically-wrong bytes must not pass.

   Declare it beside `[package.metadata.transports]`:

   ```toml
   [package.metadata.unified_surface]
   list  = "verb list"                                    # one name per line
   cli   = "verb call {verb} --json {params}"
   http  = { serve = "verb serve --port {port}", path = "/v1/verbs/{verb}" }
   probe = { verb = "validate", params = "{\"path\":\"forjar.yaml\"}" }
   ```

   `probe` must name a verb that is **safe to invoke repeatedly and has no side
   effects** — the gate calls it once per transport, every run.

   Absent declaration → SKIP. Declaration present but `invariance.py` missing →
   **FAIL**, not SKIP: a gate that cannot run has not passed.

### Never dogfood by hand

**The protocol runs the binary. You do not.**

Manual verification — a hand-typed `curl`, a binary path you picked yourself, a
list of endpoints you remembered — is not a cheap version of this gate. It is a
different activity with a worse failure mode, because it silently tests
*something other than the artifact*.

Observed 2026-08-22, while building the very surface this gate checks: a manual
dogfood of a new HTTP transport reported `unrecognized subcommand 'serve'` and
appeared to prove the feature broken. The feature was fine. The binary at
`<repo>/target/debug/<crate>` was four minutes stale, because this workstation
sets cargo's target directory globally and the real output was under
`/mnt/nvme-raid0/targets/`. The e2e suite passed throughout — it uses
`CARGO_BIN_EXE_`, which cannot be stale.

`$BINPATH` here comes from `cargo build --message-format=json`'s own
`executable` field for exactly this reason: it is the artifact cargo just
produced, wherever cargo chose to put it. A path you type is a guess about where
that is.

So if you find yourself starting a server and curling it to check something,
that is a signal the **declaration is missing**, not that the gate is
unnecessary. Add `[package.metadata.unified_surface]` and let the protocol do it
— derived, simultaneous, and against the binary that will actually ship.


   **What the e2e target must assert** (the gate enforces that it exists, spawns
   the artifact and passes; what it checks is the crate's job). The reference
   implementation is rmedia — `crates/rmedia-cli/src/verbs/conformance.rs` plus
   `tests/e2e_mcp_stdio_t.rs` and `tests/e2e_http_serve_t.rs`:

   - **Surface equality** — the name sets from all transports (CLI leaf walk of
     the *derived* clap tree, MCP `tools/list`, HTTP `GET /v1/verbs`) are equal
     to each other and to a committed manifest that is **generated from the
     registry**, not maintained beside it. Docs as a projection make drift
     unrepresentable.
   - **One green verb through every transport, diffed as bytes** — with the MCP
     and HTTP params *derived from the CLI argv through the adapter*, so it
     compares adapters rather than three hand-written inputs that happen to agree.
   - **One red input through every transport** — same error class, each
     transport's own code (exit 2 / JSON-RPC -32602 / HTTP 400), no invented values.
   - **Lifecycle** — readiness read from the child (never `sleep`), SIGTERM
     exits 0, port released.

   **probador-suite** runs `probador test` where the crate configures it
   (`probar.toml`, or a `jugar-probar` dependency). Missing *configuration* is a
   SKIP with its reason; a missing *tool* is already a NO-GO above.

   ### probar/probador: what it gates, and the gap

   **probador 1.0.3 cannot verify a CLI, HTTP or MCP interface today.** It is
   "Playwright-Compatible Testing for WASM + TUI Applications". Its subcommands
   are `test, record, report, coverage, init, config, serve, build, watch,
   playbook, comply, av-sync, audio, video, animation, stress, llm` — none of
   which drives another process's CLI, an arbitrary HTTP endpoint, or a JSON-RPC
   server. Specifically, verified in the source:

   - `serve` is a **dev server for the WASM app under test** (axum
     `extract::ws`, hot reload, COOP/COEP) — it hosts a page; it is not an HTTP
     client.
   - `llm` **is** a real HTTP client and `llm bench --start <cmd>` even has the
     right shape, but the protocol is hardwired: `crates/probar/src/llm/client.rs`
     builds `{base_url}/v1/chat/completions` and POSTs a typed `ChatRequest`.
     There is no way to say "POST /v1/&lt;verb&gt; with this body and diff the bytes".
   - **MCP is inert**: `jsonrpc` appears in exactly two files, both
     `generated_contracts.rs`, and the generated `contract_jsonrpc_framing!`
     macros have **zero call sites** in `probar` or `probar-cli`. It carries the
     vocabulary, not the capability.
   - `probador comply` is **vacuous off-WASM**: on a plain 4-line Rust CLI crate
     with no WASM, no HTML and no browser it scored **8/10**, passing "Custom
     elements tested", "Threading modes tested", "WASM size limit". Gating on
     that would be precisely the defect this protocol exists to catch, so it is
     not gated.

   So probador's honest contribution to a Rust CLI release is its own suite where
   a crate configures it, and **nothing** for CLI/HTTP/MCP parity. That is why
   7d is built on cargo, which is already present, and why there is no `probador
   interface` invocation in `dogfood.sh` — writing one would be a step that reads
   as a pass without verifying anything.

   **The gap, as work rather than coverage.** For probar to become the home of
   this gate it needs five capabilities it does not have: (1) a process-under-test
   driver — spawn an arbitrary binary with argv/env/cwd and capture stdout,
   stderr and exit code separately (generalising `llm bench --start`);
   (2) a stdio JSON-RPC client with initialize / tools-list / tools-call as
   first-class ops (the `mcp-protocol-v1` contract macros it already carries are
   the schema for exactly this); (3) a general HTTP client — arbitrary
   method/path/body/headers with raw-byte capture, i.e. `llm/client.rs` with the
   hardcoded path and typed body lifted out; (4) readiness and lifecycle
   primitives — ephemeral port, readiness line read from the child, real SIGTERM,
   assert exit 0 and port release; (5) a cross-transport diff assertion. (1)+(4)
   would also fix `comply`'s vacuity, since C001 "Code execution verified" is the
   same *did it actually run* question. A plausible surface is
   `probador interface --manifest probador.interface.toml`, hosted by the existing
   `playbook` state machine. Until then, this section states what is missing
   rather than implying it is covered.

   The previous gate here was `probar run`: wrong binary (`probador`), and `run`
   is not a subcommand. `command -v probar` therefore failed and it WARNed on
   every release. **A verifier that has never executed is indistinguishable from
   one that passes** — which is the defect this entire skill exists to prevent,
   sitting inside the skill itself.

7e. **contracts-exit-integrity** — the `contracts:` recipe is inspected for
   laundered exit codes *before* its GREEN is believed. A POSIX `for` loop exits
   with its **last iteration's** status, so

   ```make
   contracts:
   	@for c in contracts/*.yaml; do pv validate "$$c"; done
   	pv lint contracts/binding.yaml 2>/dev/null || true
   ```

   reports whatever the alphabetically-last contract did. That is rmedia's actual
   recipe, and running its body against rmedia's real contracts **exits 0 with 17
   of 47 contracts failing validation** — because the last name in the glob,
   `visual-quality-v1.yaml`, passes. The largest false green found in this fleet.
   The second line is dead twice over (`pv lint` on a FILE passes over zero
   contracts, and `|| true` would swallow it anyway). The correct shape is
   copia's: `do pv validate "$$c" || exit 1; done`.

7f. **pmat-verify / pmat-comply** — `pmat comply check` runs 155 checks and **its
   exit code cannot see a skip**. Measured on a clean tiny crate:
   `{"pass":26,"warn":13,"fail":0,"skip":116}`, `"is_compliant": true`, exit 0 —
   and **three of those 116 skipped checks carry `"severity": "Error"`**.
   `--strict` does not help; it only adds a warnings tri-state, in which skips
   appear nowhere.

   The sharpest case is **CB-200 (TDG Grade Gate)**. In a fresh `git clone` with
   no `.pmat/` it reports *Skip: "Not measured: no .pmat/context.db … Run `pmat
   query \"x\"` to create the index."* Measured on rmedia: fresh clone
   `{"fail":3,"skip":95}` with CB-200 = Skip; after **one** `pmat query`,
   `{"fail":4,"skip":93}` with CB-200 = **Fail — "24 function(s) below minimum
   grade A"**. The green was the index's absence, not the tree's quality.

   So the gate **builds the index first**, then treats CB-200 `Skip` and `Fail`
   as equally NO-GO — unmeasured is not a pass — and REPORTS the rest. The rest
   stays non-gating because 17 further checks want state a release checkout
   structurally cannot have (a gitignored `.pmat-work/` ticket dir, a *sibling*
   `../provable-contracts/proof-status.json`, a ledger), which is genuinely a
   property of the workstation
   (paiml/paiml-mcp-agent-toolkit#1008). CB-200 is not in that category: it is
   fixable with one command, so it is gated. The old handling failed both ways —
   it piped comply into `python3`, so the number recorded was the *python stage's*
   result and comply's own exit code was lost, and it marked the outcome WARN.
8. **publish-dry-run** — `cargo publish --dry-run` packages cleanly (env
   `CARGO_REGISTRY_TOKEN` is unset so a stale token can't mask a real auth setup).
9. **dogfood-use** — THE dogfood step: the tool exercises its own generated
   artifacts against **real external tools and real on-disk shapes**, and fails
   if reality disagrees with what the code assumes.

   **Prefer a native subcommand over a shell script.** If the crate exposes one
   (`forjar dogfood`), run that — it lives with the code, is versioned with it,
   is itself unit-tested, and can enumerate its own surface exhaustively. Fall
   back to `scripts/dogfood-use.sh` only for crates that have no native gate yet.

   **A missing dogfood capability is a NO-GO, not a WARN.** A released tool
   nobody actually ran is not dogfooded, and a WARN in a release protocol is a
   step everybody learns to walk past.
10. **clean-room** — marked MANUAL because it runs on the intel CI box and is heavy.
    It is a **HARD release gate**: no `cargo publish` for any Sovereign AI Stack
    crate without it. Run it yourself before releasing:
    `make -C ~/src/infra/machines/clean-room clean-room-<crate>`.

## What dogfooding has to do — and why tests do not replace it

**A test you author cannot falsify a premise you hold.** forjar shipped three
releases in two days, each fixing the previous, and every one had passed 12,904
unit tests, a five-gate clean room and a 19-check CI run:

| release | bug | why the tests missed it |
|---|---|---|
| 1.13.0 | `backup_sync` read rclone's `--combined` status characters inverted, so files that were NOT backed up left the coverage denominator — a backup missing data reported *higher* coverage than one with everything | the test stub emitted whichever characters the author believed in, so it could never disagree |
| 1.13.2 | `disk_budget` required both `CACHEDIR.TAG` and `.rustc_info.json`; across a real 4.6 TB tree **zero of sixteen** marker-bearing dirs had the pair, so the reaper matched nothing and went inert at 94% used | the fixture had both markers because the author believed both were present |

Both are the same failure: the fixture and the code shared an author, so it was
confirmed rather than tested. Only the real tool and real data can break that
loop. So a dogfood exercise must:

- **invoke the real external tool** the code depends on (`rclone check` itself,
  not a stub of it) and assert on its actual output;
- **build the on-disk shapes that really occur**, taken from a machine, not from
  what the layout is assumed to be;
- **fail when the tool is absent** rather than skipping — a dogfood run that
  quietly skips the real dependency proves nothing;
- **use the interpreter production uses** (forjar executes with `bash`; checking
  emitted scripts under `sh` tests a configuration that never runs);
- **be exhaustive over the surface**, so a new feature cannot land with no
  coverage. `forjar dogfood` matches every `ResourceType` with no wildcard arm:
  a new type fails to compile until its coverage is declared, and
  `NotApplicable` requires a written reason. That is the property that stops the
  gate going quiet — the previous `dogfood-use.sh` covered only `file`
  resources and still returned GO while two new resource types shipped broken.

### Verifying the gate is real

A gate nobody has seen fail is not evidence. For each bug class it claims to
cover, apply the bug as a **named mutation** and confirm the gate turns RED,
then restore and confirm GREEN. Record both in the PR. forjar's:

```
both-markers cargo rule   -> FAIL disk_budget: repo target root NOT detected
inverted rclone +/-       -> FAIL backup_sync: counter keyed on wrong character
```

**This applies to the protocol's own gates.** Every gate in `dogfood.sh` was run
against a fixture crate at full GREEN (`VERDICT: GO`, exit 0), then mutated one
at a time; the tree was restored after each and re-confirmed GREEN. 2026-08-16
ledger — mutation → the gate that turned RED:

| mutation | gate | what it printed |
|---|---|---|
| `det002-timestamp` (add `$(date +%s%N)` to a script) | bashrs | 1 SEC/DET/IDEM error(s) over 3 file(s): DET002 |
| `bashrsignore+real-finding` (`.bashrsignore='*.sh'` hiding that DET002) | bashrs | *still* FAILs — `--no-ignore` is load-bearing |
| `bashrs-partial-scan` (stub silently lints 2 of 4 files, exit 0) | bashrs | NO SCAN RECEIPT: expected `Linted 4 file(s)`, got `Linted 2 file(s)` |
| `bashrs-noop-shadow` (`#!/bin/sh exit 0` on PATH) | bashrs | POSITIVE CONTROL FAILED … do not read a clean result as clean |
| `contract-schema-break` (rename a required key) | pv-contracts, pv-lint | 1 checked, FAILED: hash-integrity-v1.yaml / `Summary: 1 errors` |
| `pv-noop-shadow` | pv-contracts | POSITIVE CONTROL FAILED: `pv validate` accepted a malformed contract |
| `ghost-binding` (point a binding at a nonexistent fn) | pv-bindings | REPORT `0/1 verified; 1 ghost(s)` — by design, see 7b″ |
| `unreadable-binding` (corrupt `binding.yaml`) | pv-bindings | FAIL: produced no verification line — an unrun verifier is not a clean one |
| `laundered-for-loop` (drop `\|\| exit 1`) | contracts-exit-integrity | recipe launders its exit code: a for-loop with no `\|\| exit` |
| `or-true-swallow` (append `\|\| true`) | contracts-exit-integrity | `\|\| true` swallows a real failure |
| `pmat-index-absent` (`rm -rf .pmat`, `pmat query` stubbed out) | pmat-comply | CB-200 is UNMEASURED, not passing |
| `drop-transports-table` | transport-decl | no `[package.metadata.transports]` … an undeclared transport is an unverified one |
| `undeclare-http` (binary still ships `serve`) | transport-absence | binary advertises undeclared transport surface: serve(→http) |
| `library-only-e2e` (e2e calls the lib, not the binary) | interface-parity | e2e never references `CARGO_BIN_EXE_` — exercises the library, not the release artifact |
| `unmet-required-features` (drop `features=["http"]`) | interface-parity | http(exit=101) |
| `vacuous-zero-test-e2e` (target compiles, runs no tests) | interface-parity | target ran 0 tests — a vacuous pass |
| `render-compact-over-http` (one transport pretty, one compact) | transport-invariance | `validate` differs between cli and http — verified against forjar 2026-08-22 |
| `empty-verb-list` (binary lists no verbs) | transport-invariance | the binary lists NO verbs — a parity check over an empty surface is vacuous |
| `delete-invariance.py` | transport-invariance | invariance.py missing — a gate that cannot run is not a SKIP |
| `advertised-but-unusable` (list `diag` in `--help`, don't implement it) | cli-surface | advertised but unusable: diag (of 2 checked) |

Two negatives worth keeping, because "did not go red" is also a finding:
a duplicate contract id is a pv **warning**, not an error, so pv-lint stays
green on it; and `.bashrsignore` alone over *clean* scripts stays green, which
is correct — it is only dangerous when it hides a real finding, which is what
the mutation above tests.

### The gate that crashed instead of running

`dogfood.sh` referenced `$BINPATH` in the renacer gate ~100 lines **before** the
release binary was built. Under `set -u` that is a fatal abort, not a skipped
gate. Reproduced on a crate with a `renacer.toml` (trueno has one, and renacer
is installed): `dogfood.sh: line 181: BINPATH: unbound variable`, **exit 1** —
the same exit code this script uses for NO-GO. It never reached the contracts
step, the dogfood step, or the receipt, and a crashed protocol was
indistinguishable from a considered verdict. The build now happens once, near
the top, and records its own `release-binary` gate.

Two smaller versions of the same disease, also fixed: the note extractor printed
raw ANSI (`(B[m`) instead of the failure, and the crate's feature detection
matched `cli = { e2e = … }` inside `[package.metadata.transports]`, running every
gate with `--features cli` against a crate that has no such feature.

### Status vocabulary

`PASS` / `FAIL` / `SKIP` / `REPORT` / `MANUAL`. **`SKIP` must record the
enumeration that found nothing** ("0 files from: `git ls-files '*.sh' …`"), so
*no subject* can be told apart from *did not look*. **`REPORT` is a measurement
that is deliberately not gating, and must carry both the number and the upstream
issue it waits on** — a REPORT with no issue number is a WARN wearing a costume.
Do not add new `WARN`s: a WARN in a release protocol is a step everybody learns
to walk past.

### Legacy: the `scripts/dogfood-use.sh` contract

For crates without a native gate. The script receives `$BIN` (the built release
binary) and `$WORK` (a scratch dir) and exits non-zero if the tool misbehaves on
real input — e.g. copia round-trips its own source tree through sync + bisync +
hub and asserts byte-identity. Migrate these to native subcommands over time.

## Gaps this protocol did not catch (pmat 3.32.0) — now required checks

Every item below is a defect that survived a green CI, ~20,000 passing tests and
a NO-GO receipt that named neither. They are listed as *checks to run*, not
advice, because each was invisible to the version of this protocol that produced
that receipt.

**1. A gate that cannot fail is not a gate — prove it fails.**
`make coverage` ended with

    if [ -n "$COV_PCT" ] && [ below threshold ]; then fail; else PASS; fi

so an EMPTY percentage — a broken instrumentation run, a changed summary format,
anything that stops `TOTAL` parsing — took the else branch and printed
`✅ Coverage % meets threshold 95%` at exit 0. The test run above it was
`|| true`, so a failing suite reached the same place. **Grep every gate for the
shape `[ -n "$X" ] && [ …bad… ]`, and for `|| true` on the step that produces the
measurement.** An unmeasured value must FAIL, and must say which of the two
states it is in.

**2. A gate nobody runs is decoration — check it is WIRED, not merely present.**
Two falsification gates lived in the Makefile and in `tests/all.rs`. Both were
referenced **zero times** in any workflow, because every CI leg runs
`cargo test --lib` and they are integration targets. Run
`grep -c "<gate-target>" .github/workflows/` for each `make` gate. Then check it
is *blocking*: a job that reports but is not in a required context (directly or
via an aggregator's `needs:`) cannot stop a merge. Adding a gate that reports and
cannot block reproduces the exact defect the gate detects.

**3. A checker that measured nothing must not report success.**
Assert a floor on the population inside the checker itself — `assert!(checked >
1000)` in a repo with thousands of the thing being checked. Silence from a broken
walk is byte-identical to silence from a clean tree.

**4. A harness must not report the developer's shell.**
`run()` scrubbed `CARGO*`, `RUSTC*` and `PMAT_CONFIG` and never touched
`RUST_LOG`, so `--quiet` read as a no-op on ~43 commands with RUST_LOG unset and
as effective with it exported. Third instance of that class in one file — the
other two are `NO_COLOR=1` (turned ~40 `--color` flags into false no-ops) and a
nested `cargo check` under `cargo test` that made the harness *manufacture the
defect it claimed to detect*. **Scrub the environment, then let a flag that needs
one DECLARE it per-flag.** Never set it globally: the variable that makes
`--quiet` observable makes `--verbose` inert.

**5. Regenerate generated artifacts AFTER merging, never before.**
A committed ledger that counts every test in the tree is stale the moment the
base branch adds one. Regenerating and then merging produced a red PR that looked
like real breakage twice.

**6. Verify a constraint before repeating it.**
"PRs touching `.github/workflows/*` need a web-UI merge click — the token lacks
`workflow` scope" was carried for an entire session and was FALSE: the token had
the scope and the remote was SSH, which bypasses token scopes entirely. The
evidence — a successful push of workflow edits — was already in the transcript.
`gh auth status` and `git remote get-url origin` take one second between them.

**7. Read the file back after a scripted edit, and never `grep -c` a binary.**
A `python -c` inside double quotes let the shell execute backticks in a comment
and silently blank two words. `grep -c pattern <binary>` does not count matches —
use `grep -ac`. Both produce a confident, wrong "done".

**8. An external oracle is the point — comparing a tool to its own report proves
nothing.** `analyze clippy` reported `total_diagnostics: 0` on a crate
`cargo clippy` gives 76 warnings for, because the count was taken AFTER a
confidence filter. No unit test could catch it: the fixture and the code share an
author. The dogfood check that catches it runs **real cargo clippy** and compares.

## Adoption state (2026-08-16) — the baselines, stated rather than hidden

Both crates measured today are **NO-GO**, and that is the gates working:

- **copia 0.2.0** — RED on `version-unpublished` (0.2.0 is already on crates.io;
  bump it) and on `transport-decl` (a 7-subcommand CLI that has never declared
  its interfaces). Everything else green: 11 contracts valid, `pv lint` 0 errors,
  bashrs 2 files linted with the receipt confirmed, CB-200 measured and passing,
  `contracts:` recipe propagates failure, dogfood-use round-trips L1/L2/L3.
  `pv-bindings` REPORTs 14/15 with one ghost, which is a pv false positive
  (see 7b″).
- **rmedia** — measured read-only today, RED on four: `pv-contracts` (**17 of 47**
  contracts have never passed `pv validate`); `contracts-exit-integrity` (its
  `contracts:` recipe is exactly the laundered `for` loop above);
  `bashrs` (13 files linted, receipt confirmed, gating findings **DET002 ×2,
  SEC010 ×11, SEC011 ×1** — `scripts/bench-whisper-e2e.sh:83,89` is
  `START_NS=$(date +%s%N)` in a benchmark, a defensible use and precisely why
  this wants a dated baseline rather than a blanket block; 82 SC10xx suppressed
  as #226 and 150 further parse-noise errors reported but not gating); and
  `pmat-comply` (CB-200: 24 functions below grade A). Its interface work is the
  reference implementation for 7d and is already done — it needs the declaration.

Record a dated baseline in the crate rather than weakening a gate. A baseline may
only shrink.

### 9. A gate that is green here and red on the runner measured a different thing

Not a flaky gate — a gate whose CONTROL differs between environments. pmat's
flag-efficacy sweep was green on the workstation and red in CI, reporting
`analyze coverage-improve --fast` as a flag that "parses but changes nothing".
The flag is wired (`perfection_score/calculator.rs:212,297,316`). The runner's
corpus has no Makefile, so the command exits 1 with an empty stdout **before any
flag is read**, and the flagged run reproduced the baseline byte for byte
because both had already failed. The gate was accusing working code.

The general rule, which belongs in any differential harness:

> A baseline that exited non-zero **and rendered nothing** is not a control.
> Every flag compared against it reproduces it exactly.

Note both conjuncts. Keying on the exit code alone excuses every no-op flag on
every failing quality gate — `quality-gate`, `analyze lint-hotspot` and `enforce
extreme` all exit 1 while printing a full report, and that is their healthy
outcome. pmat had this guard on its two-value probe path and not on its boolean
path, so the rule lived in one of the two places that needed it; the fix moved it
into the single function both paths validate their baseline through.

**So: run the release gates in the CI environment before believing them, and
when a gate fires, reproduce the finding by hand before filing it.** A local
green is evidence about the local corpus, not about the gate.

### 9b. `cargo test --lib` does not run doctests, and the clean room does

Two doc comments in pmat 3.32.0 contained prose indented by four spaces. Markdown
reads an indented block as a code block and rustdoc COMPILES a code block as
Rust, so both were doctests, and both failed:

```
src/cli_exit.rs:75                 error: expected one of `!` or `::`, found `:`
src/services/unrun_tests/mod.rs:154 error: unknown start of token: \u{2014}
```

Every local check in that session was `cargo test --lib`, which excludes
doctests entirely, so the defects were green locally and red in the clean room's
B3 gate. Neither block was ever meant to be executable; both are now fenced
```` ```text ````.

**Run `cargo test --doc` before a release, not only `--lib`.** And note this
compounds with the coverage note elsewhere in this file: `make coverage[-broad]`
also uses `--lib`, so a doctest is unmeasured there too.

### 10. `cargo deny check advisories` is only as wide as RustSec

`advisories ok` answers "is anything in my tree **listed in RustSec**" — not "is
anything in my tree known-vulnerable". Measured in pmat 3.32.0: cargo-deny
printed `no crate matched advisory criteria / advisories ok` while thrift 0.17.0
sat in the tree (`parquet 57.3.1` <- `aprender-db`) carrying CVE-2026-43868
(medium, CVSS 5.3, patched 0.23.0). RustSec's db, pulled the same day with 1,208
advisories, had **no thrift entry at all**; GitHub's Advisory Database did.

The protocol now runs a `security-2nd-source` check against
`gh api "repos/{owner}/{repo}/dependabot/alerts" --paginate` — an independent
database, which is the entire point. FAIL on high/critical, WARN otherwise, and
WARN when `gh` is missing, because a source that did not run is not a source
that found nothing.

## 11. FLEET dogfood — the release binary against real repositories

**Every gate above tests pmat against fixtures pmat's own authors wrote**, so
the fixture and the code share an author and confirm each other. That is the
loop this whole protocol exists to break, and it was still unbroken at the top
of 3.32.0.

So: `cargo install --path . --root <scratch>` the release tree, and run that
binary against **every repo in the sovereign stack**. The roster is the one
`~/src/infra/machines/clean-room/Makefile` declares — read it from the
Makefile, never from a list restated here: the previous sentence enumerated
12 repos while the Makefile declared 13 (pmat itself was the one missing —
#2644 audit, SHIM-05), which is what a restated list does.
They span 10 to 60,980 files of code nobody wrote to make pmat look good.

Harness: `fleet-dogfood.sh` — an UNGUARDED user-scope helper
(`~/.claude/skills/dogfood/`), not tracked in any repo and not part of the
certified protocol (#2644 audit, SHIM-05: user-scope siblings already
demonstrate drift). Treat its output as a convenience, not evidence; the
certified path is `scripts/dogfood.sh` per repo. Tracking-or-deleting these
helpers is part of the one-runner consolidation (infra#270).

**Install to a SCRATCH root, never over `~/.cargo/bin`.** And use the built
artifact for pmat's own gates: the installed `pmat` and the release tree both
print `3.32.0` while being different commits, and in this cycle that difference
ran the OLD CB-200 against the new tree and produced a false NO-GO. Only the
`commit:` line in `--version` distinguishes them.

**Two kinds of finding, and conflating them is the failure mode:**

| | meaning | ticket goes to |
|---|---|---|
| `PMAT-DEFECT` | pmat crashed, hung, emitted unparseable JSON, or reported success having measured nothing | pmat |
| `REPO-FINDING` | pmat worked and found real debt in the target | that repo |

Conflating them files a tool's own bugs as other people's debt. **One
consolidated ticket per repo**, never one per finding — a maintainer opening six
issues from an automated sweep closes all six unread. **Reproduce every
PMAT-DEFECT by hand before filing.** If a repo is clean, file nothing and say so:
this must not manufacture work.

## 12. TRANSPORT PARITY — ask the same question three ways

A CLI-only sweep proves nothing about the other transports. This project's
history records **24 MCP-vs-CLI contradictions in a single round**, and the
3.32.0 surface audit found that of the 16 tools the shipped binary actually
serves over MCP, **zero** had a top-tier coverage row while the 18 it does not
serve scored higher. The tested surface and the served surface were nearly
disjoint.

Harness: `transport-parity.sh` — likewise an unguarded user-scope helper, not
tracked in any repo (#2644 audit, SHIM-05; see the fleet-dogfood.sh note
above). For each repo it asks
one question over **CLI, MCP stdio and HTTP** and compares the answers. The
finding it produces is not "a transport is broken" — it is **"the transports
DISAGREE"**, which is worse, because each looks correct alone.

Two traps, both hit on the first run:

* **Get the tool schema right.** The MCP tools take `paths` (an array), not
  `project_path`. Calling them wrongly yields `-32602 Validation error`, which
  reads exactly like a defect. The harness now classifies any JSON-RPC error as
  a HARNESS fault, because the first run produced three confident phantom
  findings that way.
* **A missing HTTP build is SKIPPED and said so**, never silently passed.

## 13. All three transports must be in the DEFAULT build

`mcp-http` was opt-in, so `cargo install pmat` produced a binary whose
`serve --help` read `[HTTP NOT COMPILED IN this build]` — while the crate
description said "(CLI, MCP, HTTP)". Two of three were true by default.

A transport nobody can reach without knowing a flag name is close to a transport
that does not ship, and it cannot be dogfooded by anyone who has not been told
the flag. As of 3.32.0 `mcp-http` is in `default`. **If a release moves it back
out, the description must move with it.**

## On GO — the release (only after clean-room is also green)

1. `git tag vX.Y.Z && git push origin vX.Y.Z` (or the repo's release workflow).
2. `env -u CARGO_REGISTRY_TOKEN cargo publish` from a box with valid crates.io
   credentials (lambda-labs/intel). If the tag-triggered release workflow is
   flaky, publishing manually is the sanctioned fallback.
3. Create the GitHub release from the tag (binaries via the release workflow).
4. Attach the dogfood receipt to the release notes as the pre-release evidence.

## On NO-GO

Stop. Report the failing gate + its note. Fix the root cause in the owning crate
(five-whys), commit, re-run `dogfood`. Never release on a NO-GO.
