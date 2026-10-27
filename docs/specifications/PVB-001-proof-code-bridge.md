# PVB-001 v1.3 — proof↔code bridge: Lean theorems that bind to the Rust that ships

Spec for the paiml-implement harness. **Public**: it lives in the public `aprender` repo, so it carries no private fleet information (§3 rule 10).

- Location: `~/src/aprender/docs/specifications/PVB-001-proof-code-bridge.md`.
- Launch from `~/src/aprender`; all work happens in `$WT="$SCRATCH/wt-aprender"`:
  `Implement docs/specifications/PVB-001-proof-code-bridge.md autonomously.`
- Idempotent: each run reads live state, picks the first eligible row, lands one PR (or hits one STOP), writes a receipt, and exits. Re-run the same line for the next row.

**Location rule.** A spec lives in the repo where ≥ 80 % of its row PRs merge; below that, it lives in the operator's infra repo.
- PVB: 13 of 13 rows land in aprender → aprender.
- PVL-001 and ONT-001 are cross-repo → infra.

## Changelog

**v1.3 (2026-10-27)** applied review `pvb-001-review-2026-10-27b` C-1…C-6, measured against the operator's real `scripts/pvl/lib.sh` and the pmat repo `[V 2026-10-27 reviewer]`:

| Finding | Change |
|---|---|
| C-1 | The operator's `json_object` ignored its jq filter (every check passed on any object), and `rc_is` neither stripped `--` nor let callers redirect. Both are redefined in a new public library, and its self-test fails if either regresses (Appendix B, BR-0). |
| C-2 | `scripts/pvb/lib.sh` is no longer a copy of the operator's library (which names private repos and finds probes only in its own tree). It is a dedicated public-safe library, written in full in Appendix B and pinned by its own blake3. |
| C-3 | A PR cannot know its own squash SHA, so ledger lines carry no `merged_sha`. A row is merged when its line is present on `main` with a matching `probe_sha256`. Every `probe:` ends `&& merged BR-<n>`. BR-7's probe now checks `BR-7` (was `PVB-6`). BR-10 runs after BR-9. BR-3b uses `absent`. |
| C-4 | BR-4 depends on BR-3a **and** BR-3b; pilot assertions are `>= 6` from BR-5 on. |
| C-5 | pmat facts frozen in PVB-PMAT-001 §1. |
| C-6 | Budget sidecar totals corrected (operator-side). |

**v1.2 (2026-10-27)**
- Moved spec, ledger and probes from infra to aprender (location rule).
- Budgets moved to a private operator sidecar (§0.5).
- Probe library vendored and hash-pinned (BR-0).
- Added BR-7b: `pv bridge fresh`, the hash-only freshness check that consumers call.
- Added §6: the consumer contract for pmat and other repos; pmat rows live in sibling spec PVB-PMAT-001.
- Hard rules 9 (dependency direction) and 10 (public-safe) added.

**v1.1 (2026-10-27)** applied review `pvb-001-review-2026-10-27` F-1…F-10:

| Finding | Change |
|---|---|
| F-1 | Superseded by v1.2's location rule |
| F-2 | Vacuous BR-0 rewritten; Mathlib pin left to PVL EV-5a |
| F-3 | PVL v3 row IDs; `discharge-summary.json` replaces `discharge.json` |
| F-4 | Every row has a cheap `probe:` and a heavier `accept:` |
| F-5 | Bridge commands under `pv bridge …`; existing `pv kani` extended |
| F-6 | Tracked, content-addressed `bridge-summary.json` |
| F-7 | Bridge block nested under ONT-001 `proof:`, reusing `requires`/`ensures` |
| F-8 | Pilot is the five computable `exact` Lean models that already exist; Q4_K follows as an authored row |
| F-9 | Typed `rust:` predicates |
| F-10 | `depends_on` per row; workflow STOP removed; decision STOPs added |

## Inherited from `PVL-001-pv-lean-gate.md` v3 (operator-side spec)

- §0 operating model: the canonical checkout `~/src/aprender` is never modified, checked out or built (launching from it reads the spec only); in-tree pv via `. scripts/pv_bin.sh` inside `$WT`; pv exit vocabulary `0 accept · 1 reject · 2 decline · 3 error`.
- §1 selector.
- §2 landing protocol: `gh issue create` → `pmat work add --github-issue N` (traffic cop only; single-minter rule) → branch `PMAT-N-pvb-<n>-<slug>` → RED → GREEN → mutation → `pmat verify` → quorum → `pmat-merge --auto --squash`. The orchestrator edits `ci.yml` itself; there is no workflow STOP.
- §3 `done_when` contract.
- §5 hard rules.
- §6 budget formula `K = 1.25 K̂` (values in §0.5, private).
- §9 receipt schema, header `# PVB-001 v1.3 session receipt`.

## PVB variables (all in aprender, public)

| Name | Path |
|---|---|
| `$LEDGER` | `docs/audits/PVB-001/ledger.jsonl`, written **inside each row's PR** (receipt + ledger line ride with the change; no separate ledger PRs) |
| Ledger line | `{"row": "BR-<n>", "probe_sha256": "<sha256 of scripts/pvb/done_when/BR-<n>.sh>", "ticket": "PMAT-<N>", "pr": <num>, "at": "<iso>"}`. **No `merged_sha`**: a PR cannot know its own squash SHA. |
| `$PVB_MAIN` | `origin/main` for aprender (pmat sets `origin/master`). The selector runs `git fetch origin` once per session; probes never fetch. |
| Probe library | `scripts/pvb/lib.sh`: a dedicated public-safe library, full text in **Appendix B**, pinned by `scripts/pvb/lib.sh.blake3`. Not a copy of the operator's library. |
| Probes | `scripts/pvb/done_when/<row>.sh`, each sourcing `scripts/pvb/lib.sh` |
| Fixtures | `scripts/pvb/fixtures/<row>/` |
| `$LEDGER_PVL` | path to the operator-side PVL ledger, **passed in by environment**, read-only, used only by `merged_pvl`. Never hard-coded, written or copied into aprender. Unset → `merged_pvl` exits 2 (decline). |

**Probe vocabulary** (Appendix B is the definition):

| Function | Meaning |
|---|---|
| `tracked <path>` | `git ls-files` knows the path |
| `present <regex> <path>` / `absent <regex> <path>` | ripgrep match / no match |
| `json_object <file> [<jq-filter>]` | file is exactly one JSON object **and** the filter is true (default `true`) |
| `rc_is <n> [--] <cmd...>` | `cmd` exits `n`. Caller redirections are honoured; the library never redirects. |
| `merged <row>` | `$PVB_MAIN:$LEDGER` has a line for `<row>` whose `probe_sha256` equals the current probe script's hash |
| `merged_remote <owner/repo> <ref> <ledger-path> <row>` | same presence test on another repo's ledger via `gh api` (used by PVB-PMAT-001) |
| `merged_pvl <row>` | the operator's PVL ledger marks `<row>` merged |

**Why "present on main" proves merge.** Ledger lines live only in the row's PR branch until squash-merge, so a line on `$PVB_MAIN` exists only after merge. Matching `probe_sha256` means editing a row's probe re-opens that row; that is intended (a changed definition of done must be re-earned).

**Probes are harness-only.** They run on the operator's machine in the selector, never in public CI. Public CI runs only `accept:` commands and the `provable-ladder` job.

**Probe/accept split:**
- `probe:` runs every session. It is cheap: file, grep or ledger checks, ≤ 5 s, no `lake`, `cargo` or `kani`.
- `accept:` runs once inside the row's PR and in CI. It may be heavy.
- No bare `| jq -e` on command output: assert through `json_object` on a file.
- No bare `test $? -eq N`: assert through `rc_is` plus a named stderr line.

Provenance marks: `[V]` verified by command (date) · `[C]` cited · `[A]` operator directive or estimate · `[U]` unknown · `[X]` refuted.

---

## §0 Thesis, on one page

**Problem.** aprender proves theorems about Lean models, not about the Rust kernels. Bindings resolved: 0 / 3133 `[C]` (ONT §1). The PVL finding "Rust connection ABSENT" (F5) is filed with no row `[C]`. After PVL-001, the project holds *checked proofs of models*: honest, but the step "model = code" is an unchecked human claim.

**Thesis.** A contract earns **L5** only when four legs agree on the same content hashes, each generated from or checked against the one contract YAML:

| Leg | Question it answers | Mechanism | Evidence |
|---|---|---|---|
| **B0 proof** | Is the model correct? | `lake` discharge | PVL EV-7a/7b/8a → tracked `crates/aprender-contracts-staging/discharge-summary.json` (keyed on `tree_sha`) |
| **B1 bind** | Does the named Rust function exist with the declared signature? | `binding.yaml` + resolver | PVL EV-2 |
| **B2 bound** | Does the Rust function satisfy `requires ⇒ ensures` up to bound *k*? | Kani harness generated from typed `rust:` predicates (§0.3) | `bridge-summary.json` `.kani_verified` |
| **B3 refine** | Does Rust compute the same function as the proved Lean model? | Differential oracle: Lean model compiled with `lake exe` vs the Rust function, on seeded and edge-case inputs | `bridge-summary.json` `.mismatches` |

What each leg covers:
- B0: spec ⇐ model.
- B3: model ≈ code.
- B2: code ⊨ spec.
- **B3 RED ∧ B2 GREEN** means the postcondition is too weak to catch a real bug. The bridge doubles as a contract-quality instrument (BR-10).

**Out of gate.** Aeneas and Charon (Rust→Lean, OCaml) are out-of-gate oracles only, the same doctrine as ONT R-1.

### §0.1 Numeric domains: the float gap, stated

| `numeric_domain` | Admissible outputs | B3 comparison | Label |
|---|---|---|---|
| `exact` | Integers, bits, indices, shapes. **No float outputs.** | Bit-identical | `L5` |
| `ieee` | Floats, Lean model on a same-width Float type | Bit-identical or `{ulp: k}` | `L5(±k ulp)` |
| `real` | Lean model over `ℝ`, compared through a computable Float twin | `{abs, rel}` | `L5(≈tol)` |

`pv validate` rejects `exact` with any float type in `io.out`.

### §0.2 Schema: nested under ONT-001 `proof:`, no second source of truth

ONT-3b and ONT-4e already define `requires:`/`ensures:` and `proof: {lean: {module, model_of, relation}, kani: {harnesses}}` `[C]`. PVB adds only what those blocks lack:

```yaml
requires:
  - prose: "input is one Q4_K scales block"          # prose stays legal
    rust: "scales.len() == 12"                        # typed, syn-parsed; required for B2
ensures:
  - prose: "every (scale, min) pair fits 6 bits"
    rust: "out.iter().all(|(s, m)| *s < 64 && *m < 64)"
proof:
  lean: {module: ProvableContracts.Defs.Q4KScaleMin, model_of: unpack_scale_min, relation: eq}  # ONT-3b, reused
  kani: {harnesses: [pvb_q4k_scale_min]}                                                           # ONT, reused; generated by BR-6
  bridge:                                                                                          # PVB adds only this
    numeric_domain: exact
    tolerance: null
    io: {in: "[u8; 12]", out: "[(u8, u8); 8]"}
    bound_k: full                                      # Kani unwind; full = whole input space when finite
```

`rust_fn` is **not** repeated here. B1 reads it from `binding.yaml`, the single source.

### §0.3 Typed predicates

Existing `formal:` and `invariants:` fields are prose and math pseudocode `[V 2026-10-27 reviewer]`, so they cannot generate a harness.
- A contract is **B2-eligible** only if every `requires`/`ensures` entry carries a `rust:` string that parses as a `syn::Expr` with free variables limited to the function's parameter names and `out`.
- Prose-only entries stay legal; the census marks such a contract `no:prose_predicate`.
- Generated harnesses are never hand-edited (rule 1).

### §0.4 Operator decisions: **ruled 2026-10-27**

| Id | Row | Ruling |
|---|---|---|
| **D-1** | BR-7 | **Approved.** L5 := B0 ∧ B1 ∧ B2 ∧ B3 in the single enum (PVL EV-3). Existing self-declared L5 credit drops to what is bridged; no legacy label. This is intended (ONT R-4). |
| **D-2** | BR-8 | **Approved.** PRs run `pv bridge fresh` and re-check only contracts whose hash keys changed; the full set runs nightly. |
| **D-3** | BR-9 | **Approved.** Target ≥ 20 % of kernel-kind contracts at bridged L5. |
| **D-4** | BR-0 | **Approved.** If PVL-001 rows are missing, the same worker runs them first (operator-side spec), then resumes at BR-0. No idle worker, no separate PVL worker. |
| **D-5** | scheduling | **Approved.** Train N+1 through the look-ahead L1 slot (APR-LOOKAHEAD-001); one new epic; one aprender worker; pmat's sibling spec starts when BR-7b merges and the crate is published. |

BR-0's PR writes one ledger line per ruling: `{"decision": "D-n", "ruled_by": "operator", "at": "2026-10-27T11:17:00Z"}`. A row whose decision line is absent from `$PVB_MAIN:$LEDGER` still exits `STOP(decision: D-n)`, so a reverted ruling takes effect without editing this spec.

### §0.5 Budgets live outside this repo

Per-row `K̂`, `K` and andon thresholds are operator-private. They live in the operator's sidecar `PVB-001.budget.yaml`, next to the operator-side copy of PVL-001.
- The harness reads the sidecar at launch. If it is absent → `STOP(budget: sidecar missing)`.
- Budget consumption is logged only to the operator's private budget ledger. **Never** to `$LEDGER`, a receipt, a PR body or an issue.
- This spec names rows; it never names their cost.

---

## §1 Ground truth (quote-frozen until BR-1 emits the census)

| Fact | Value | Mark | Re-verify |
|---|---|---|---|
| Bindings resolved | 0 / 3133 | `[C]` ONT §1 | `"$PV" verify-bindings contracts/binding.yaml --crate-name <crate>` in `$WT` |
| Mathlib `.rev` in `lake-manifest.json` | pinned SHA `1d042d3e…` | `[V 2026-10-27 reviewer]` | `jq '.packages[]\|select(.name=="mathlib")\|.rev'` |
| Mathlib `inputRev` in `lakefile.lean` | `master` (floating) | `[V 2026-10-27 reviewer]` | owned by **PVL EV-5a**, not PVB |
| Lean package root | `ProvableContracts.*` | `[V 2026-10-27 reviewer]` | `ls crates/aprender-contracts-staging/lean/ProvableContracts` |
| `Defs/` computability | all 30 files `noncomputable def` over `ℝ` | `[V 2026-10-27 reviewer]` | `rg -c 'noncomputable def' …/Defs` |
| Quant Lean theorems | `NF4Dequant.lean`, `RoundtripBound.lean`, both over `ℝ`; **no computable Q8_0/Q4_K block model** | `[V 2026-10-27 reviewer]` | `ls …/Theorems/Quantization` |
| Existing computable `exact` models | `F16/Conversion`, `TensorLayout/IndexAlgebra`, `MatMul/Shape`, `Gqa/HeadMapping`, `GgufExportSymmetry/Roundtrip` (under `Theorems/`) | `[V 2026-10-27 reviewer]` | BR-1 census |
| `pv kani <contract>` | exists, `cli.rs:45` | `[V 2026-10-27 reviewer]` | `pmat query "Commands::Kani"` |
| `pv codegen <contract_dir>` | positional dir, no subcommands | `[V 2026-10-27 reviewer]` | `cli.rs:37-43` |
| `requires`/`ensures` field format (ONT-4e) | `[U]` | `[U]` | BR-2 |
| Lean `Float32` in pinned toolchain | `[U]` | `[U]` | BR-0 |
| Kernel-kind contract count; B2-eligible count | `[U]` | `[U]` | BR-1 |
| `pv bridge diff` wall time per contract at N = 10⁵ | `[U]` | `[U]` | BR-5 |

---

## §2 Rows, in EV order

### Phase 0: preconditions

**BR-0** · "PVB-0: preflight — PVL v3 gate landed, public probe library, Float32 probe" · `depends_on: []`
- **Bootstrap.** Before BR-0 merges, `scripts/pvb/lib.sh` does not exist on `main`, so every probe script begins `[ -f scripts/pvb/lib.sh ] || exit 1`; a missing library means "row not done". BR-0 itself writes the library into `$WT` first and sources it from there for its own preflight.
- Write into `$WT`, before any check:
  - (a) `scripts/pvb/lib.sh`, byte-for-byte Appendix B, plus `scripts/pvb/lib.sh.blake3`.
  - (b) `scripts/pvb/lib-check.sh` (Appendix B), which asserts: the blake3 pin matches; every vocabulary function is defined; and a **self-test** on fixtures it builds in a temp dir (full text in Appendix B):
    - `json_object obj.json '.a==2'` → false when the file holds `{"a":1}`. This catches the C-1 regression, a filter that is silently ignored.
    - `json_object two-objects.jsonl` → false.
    - `rc_is 3 -- sh -c 'exit 3'` → true, and `rc_is 0 -- sh -c 'echo x >&2' 2>"$SCRATCH/e"` leaves `x` in `$SCRATCH/e`. This catches the C-1 redirect-swallow regression.
    - `merged BR-zz` on a fixture ledger → false; with a matching line and hash → true; with a stale hash → false.
  - (c) `crates/aprender-contracts-staging/lean/test/Float32Probe.lean`: `#eval`s one `Float32` op. If `Float32` is absent, the test records `float32: absent` and `ieee` contracts are limited to Float64 models.
- Then the read-only preflight, sourcing `$WT/scripts/pvb/lib.sh`:
  - For each of **EV-1, EV-2, EV-3, EV-5a, EV-6a, EV-6b, EV-7a, EV-7b, EV-8a, EV-8b, EV-9**: `merged_pvl <row>`. Exit 2 → `STOP(env: LEDGER_PVL unset)`. Exit 1 → per D-4, nothing is committed here; the worker switches to the operator-side PVL-001 spec, lands the first missing row by its own protocol, and re-runs this preflight next session. It repeats until all eleven pass. Each switch is logged in the session receipt as `handoff: PVL-001 <row>`.
  - The budget sidecar exists (§0.5), else `STOP(budget: sidecar missing)`.
  - Rule 10 scan of the whole BR-0 diff; any hit → `STOP(public-safety: <file>)`.
- Library changes later: any edit to `scripts/pvb/lib.sh` goes through its own PR that updates the pin and passes `lib-check.sh`. There is no upstream to drift from, so there is no drift STOP.
- Mutation (two, both in the PR body): replace `json_object` with the C-1 filter-ignoring version → self-test RED; edit one byte of `lib.sh` without updating the pin → RED.
- probe: `[ -f scripts/pvb/lib.sh ] && tracked scripts/pvb/lib.sh.blake3 && tracked crates/aprender-contracts-staging/lean/test/Float32Probe.lean && merged BR-0`
- accept: `bash scripts/pvb/lib-check.sh && (cd crates/aprender-contracts-staging/lean && lake env lean test/Float32Probe.lean)`

### Phase 1: measure

**BR-1** · "PVB-1: `pv bridge census`" · `depends_on: [BR-0]`
- Change: add a `Commands::Bridge` subcommand group: `pv bridge {census, codegen, diff, kani, fresh}`, never under `pv codegen`. `pv bridge census <contracts-dir> --out <file>` writes a derived, untracked JSON report. Per kernel-kind contract it records:
  - `binding_resolved`
  - `lean_discharged` (from `discharge-summary.json`)
  - `model_computable` (the `model_of` def is not `noncomputable`)
  - `b2_eligible` (§0.3)
  - `numeric_domain`
  - `bridgeable ∈ {yes, no:<reason>}`
- All paths come from arguments; the crate hard-codes no aprender layout, so other repos can use it (§6).
- Zero kernel-kind contracts → exit 2, stderr `decline: 0 kernel contracts`.
- RED: fixture `scripts/pvb/fixtures/br-1/` holds 4 contracts (bridgeable `exact`, noncomputable model, ghost binding, prose-only predicate). The test asserts the 4 verdicts exactly.
- Mutation: stub `model_computable = true` → fixture verdict flips → RED.
- probe: `present 'Bridge' "$WT/crates/aprender-contracts-cli/src/cli.rs" && tracked scripts/pvb/fixtures/br-1/expected.json && merged BR-1`
- accept: `"$PV" bridge census scripts/pvb/fixtures/br-1 --out "$SCRATCH/c.json" && json_object "$SCRATCH/c.json" '.rows|length==4 and ([.rows[].bridgeable]|sort)==(["no:ghost_binding","no:noncomputable","no:prose_predicate","yes"])'`

### Phase 2: mechanism

**BR-2** · "PVB-2: `proof.bridge` schema + typed `rust:` predicates" · `depends_on: [BR-1]`
- Change: `pv validate` parses `proof.bridge` and the `rust:` entries in `requires`/`ensures` (§0.2, §0.3). It reads ONT-4e's `requires`/`ensures` shape as found at HEAD (fills §1 `[U]`). It rejects, each with a named field:
  - `exact` with a float in `io.out`
  - missing `tolerance` on `ieee`/`real`
  - `tolerance` set on `exact`
  - a `rust:` string that fails `syn::parse_str::<Expr>` or names a free variable outside the parameters and `out`
  - a duplicate `rust_fn` or `lean_theorem` key inside `proof.bridge` (single source)
- RED: 5 fixtures, one per rejection. Each expects exit 1 **and** its named stderr line.
- Mutation: remove the float-in-exact check → that fixture is accepted → RED.
- probe: `tracked scripts/pvb/fixtures/br-2/exact-with-float.yaml && present 'numeric_domain' "$WT/crates/aprender-contracts/src" && merged BR-2`
- accept: `rc_is 1 -- "$PV" validate scripts/pvb/fixtures/br-2/exact-with-float.yaml 2>"$SCRATCH/e" && rg -q "proof.bridge.io.out: float type not admissible for numeric_domain=exact" "$SCRATCH/e"` (repeat per fixture)

**BR-3a** · "PVB-3a: pilot — five existing computable `exact` models bound" · `depends_on: [BR-2]`
- Fact: five computable `exact` Lean models already exist (§1). Proving the mechanism on them costs no new Lean authorship.
- Change: for each of the five that the census marks `bridgeable=yes` or `no:ghost_binding`:
  - add a `binding.yaml` entry to its Rust function
  - add the `proof.bridge` block
  - add `rust:` predicates
- If fewer than 5 have a Rust counterpart, take the next computable `exact` models from the census. If there are still fewer than 5, exit `STOP(scope: <k> of 5 bridgeable)` with the list. Contract IDs come from the census, never from this spec.
- Mutation: point one binding at a nonexistent path → census `no:ghost_binding` → RED.
- probe: `json_object contracts/.pvb-pilot.json '.pilot|length>=5' && merged BR-3a` (tracked pilot list written by this PR)
- accept: `"$PV" bridge census contracts/ --out "$SCRATCH/c.json" && json_object "$SCRATCH/c.json" '[.rows[]|select(.pilot and .bridgeable=="yes")]|length>=5'`

**BR-3b** · "PVB-3b: author computable Q4_K scale/min unpacker model + theorem" · `depends_on: [BR-2]`
- Fact: kernel validation is the point of the ontology `[A]`, yet no computable quant block model exists `[V]`. The reference contract is `contracts/q4k-interleaved-scale-min-v1.yaml` (`[u8; 12] → 8×(scale, min)`, 6-bit packed). Upstream ggml is the oracle, by reference to `scripts/llama_pin.toml` `build_commit` `[C]` (#3452 ruling).
- Change:
  - Add `ProvableContracts/Defs/Q4KScaleMin.lean`: a **computable** `def` over `UInt8`/`Fin 64`, no `ℝ`.
  - Add a theorem module stating (i) every output < 64 and (ii) the unpack∘pack round trip on the domain, discharged by `lake`.
  - Add `proof.lean`, `proof.bridge` and `rust:` predicates to the contract. This becomes pilot #6.
- RED: a golden test with three hand-decoded blocks taken from upstream ggml at the pinned commit, committed as a fixture with its derivation command.
- Mutation: shift one bit in the Lean def → golden RED while `lake build` stays green. This shows golden tests catch what a weak theorem misses.
- probe: `tracked crates/aprender-contracts-staging/lean/ProvableContracts/Defs/Q4KScaleMin.lean && absent 'noncomputable' crates/aprender-contracts-staging/lean/ProvableContracts/Defs/Q4KScaleMin.lean && merged BR-3b`
- accept: `(cd crates/aprender-contracts-staging/lean && lake build ProvableContracts.Defs.Q4KScaleMin) && "$PV" bridge census contracts/ --out "$SCRATCH/c.json" && json_object "$SCRATCH/c.json" '.rows[]|select(.id=="q4k-interleaved-scale-min-v1")|.bridgeable=="yes"'`

**BR-4** · "PVB-4: Lean reference executable `pv-ref`" · `depends_on: [BR-3a, BR-3b]`
- Change: a `lake exe pv-ref` target.
  - It reads JSONL `{model, input}` on stdin and writes JSONL `{output}` or `{decline: reason}`.
  - Inputs are batched (no per-case process).
  - A codec typeclass instance is registered per pilot `model_of`.
  - A noncomputable model → `decline`, never a crash.
- RED: a golden test of 3 cases per pilot model with hand-computed outputs, committed.
- Mutation: perturb one pilot def → golden RED, `lake build` green.
- probe: `present 'pv-ref' crates/aprender-contracts-staging/lean/lakefile.lean && tracked scripts/pvb/fixtures/br-4/golden.out.jsonl && merged BR-4`
- accept: `lake exe pv-ref < scripts/pvb/fixtures/br-4/golden.in.jsonl > "$SCRATCH/o.jsonl" && diff -q "$SCRATCH/o.jsonl" scripts/pvb/fixtures/br-4/golden.out.jsonl`

**BR-5** · "PVB-5: `pv bridge diff` — B3 + tracked `bridge-summary.json`" · `depends_on: [BR-4]`
- Change, in three parts:
  - `pv bridge codegen <contract>` emits a Rust integration test in the bound crate. Inputs come from (a) the contract's `falsification_tests`, (b) boundary values of the `io` types, (c) N seeded ChaCha8 cases. `exact` domains with a finite input space ≤ 2²⁴ are enumerated exhaustively instead.
  - `pv bridge diff <contract>` runs Rust vs `pv-ref` and writes **two** artifacts:
    - Untracked log `target/pvb/<id>/bridge.json`: `{git_sha, wall_s, n, seed, mismatches, first_mismatch, host}`, written on failure too. It is a CI artifact only, never committed; it may hold host names, which is why it stays untracked (rule 10).
    - **Tracked** summary at `--summary <path>` (default for aprender: `crates/aprender-contracts-staging/bridge-summary.json`; other repos pass their own, §6), one entry per contract, keyed and content-addressed: `{contract_blake3, lean_module_blake3, rust_fn_src_blake3, kani_harness_blake3, n, seed, mismatches, kani_verified}`. No `git_sha`, no `wall_s`, no host, sorted keys, deterministic bytes.
  - N = 10⁵ `[A]`. If measured `wall_s` > 120 s on the CI host class, N drops to the largest power of 10 under 120 s, and the summary records the N used.
- RED, three tests:
  - A planted Rust mutant → `mismatches > 0` and exit 1.
  - A stale summary entry (any hash ≠ current) is reported `stale` and contributes zero.
  - `n == 0` → exit 2, `decline: n=0` (ONT R-2).
- Mutation: flip one bit in a pilot Rust function → RED.
- probe: `tracked crates/aprender-contracts-staging/bridge-summary.json && json_object crates/aprender-contracts-staging/bridge-summary.json '(.entries|length)>=6 and all(.entries[]; has("rust_fn_src_blake3"))' && merged BR-5`
- accept: `"$PV" bridge diff --all-pilot && json_object crates/aprender-contracts-staging/bridge-summary.json 'all(.entries[]; .mismatches==0 and .n>=1000)'`

**BR-6** · "PVB-6: B2 — extend `pv kani` to generate from `rust:` predicates" · `depends_on: [BR-5]`
- Fact: `pv kani <contract>` exists `[V]` (`cli.rs:45`). Extend it; never duplicate it.
- Change: `pv bridge kani <contract>` (thin alias to the extended `pv kani`) generates `pvb_<id>`:
  - It asserts `requires ⇒ ensures` from the `rust:` predicates over `kani::any()`, unwinding to `bound_k`.
  - `kani::assume` appears **only** for `requires` entries, verbatim.
  - The result goes to `bridge-summary.json` `.kani_verified` with `kani_harness_blake3`.
- RED: the harness generated against the BR-5 planted mutant fails.
- Mutation: as BR-5. If B3 is RED and B2 is GREEN, write a summary `finding: spec_weakness` (fed to BR-10).
- probe: `json_object crates/aprender-contracts-staging/bridge-summary.json 'all(.entries[]; has("kani_verified") and has("kani_harness_blake3"))' && merged BR-6`
- accept: `"$PV" bridge kani --all-pilot && json_object crates/aprender-contracts-staging/bridge-summary.json 'all(.entries[]; .kani_verified==true)'`

### Phase 3: make it count, then make it block

**BR-7** · "PVB-7: L5 = B0∧B1∧B2∧B3, read from summaries" · `depends_on: [BR-6]`
- Precondition: ledger line for D-1, else `STOP(decision: D-1)`.
- Change:
  - `is_l5` = (`discharge-summary.json` discharged the `model_of` theorem at the current `tree_sha`) ∧ B1 resolved ∧ a `bridge-summary.json` entry whose four hashes all equal current content ∧ `mismatches==0` ∧ `kani_verified`.
  - It is never read from YAML, and it works on any clean checkout with no `target/` dependency.
  - `proof-status` prints `L5`, `L5(±k ulp)` or `L5(≈tol)`, with `self-declared` and `bridged` columns side by side. Totals use `bridged`.
- RED, three tests:
  - YAML `l5: true` without a summary entry is not counted.
  - `mismatches: 1` gives no L5.
  - A hash drift on the Rust source gives no L5 and is printed as `stale`.
- Mutation: hand-edit one entry to `mismatches: 1` → `l5_bridged` drops by 1.
- probe: `present 'l5_source' "$WT/crates/aprender-contracts/src/proof_status.rs" && merged BR-7`
- accept: `"$PV" proof-status --format json > "$SCRATCH/p.json" && json_object "$SCRATCH/p.json" '.l5_source=="bridge" and .l5_bridged>=6'`

**BR-7b** · "PVB-7b: `pv bridge fresh` — hash-only freshness for consumers" · `depends_on: [BR-7]`
- Purpose: one cheap, stable verdict that pmat and other consumers call (§6). The freshness definition lives only in `pv`.
- Change: `pv bridge fresh [--changed <file>...] --format json` recomputes only the four hash keys for contracts whose bound function, Lean module, contract or harness is in the changed set (or all, if no set is given), and compares them to `bridge-summary.json`.
  - It never runs `lake`, `cargo` or `kani`.
  - Exit codes: 0 all fresh · 1 any stale (lists them) · 2 decline (no summary, or 0 bridged contracts).
  - JSON output is versioned: `{"schema": "pvb-fresh/1", "fresh": [...], "stale": [...], "pv_version": "..."}`. A breaking change bumps `schema`.
- RED: on a fixture tree, editing one bound Rust file → exit 1 naming that contract; an untouched tree → exit 0.
- Mutation: skip the Rust-source hash → edited fixture reports fresh → RED.
- Budget: ≤ 5 s on the full pilot set, so it can sit in a pre-merge check.
- probe: `present 'fresh' "$WT/crates/aprender-contracts-cli/src/cli.rs" && tracked scripts/pvb/fixtures/br-7b/edited.rs && merged BR-7b`
- accept: `rc_is 1 -- "$PV" bridge fresh --changed scripts/pvb/fixtures/br-7b/edited.rs --format json > "$SCRATCH/f.json" && json_object "$SCRATCH/f.json" '.schema=="pvb-fresh/1" and (.stale|length)==1'`

**BR-8** · "PVB-8: bridge step in `provable-ladder` (PVL EV-9), freshness + never-down" · `depends_on: [BR-7b]`
- Precondition: ledger line for D-2, else `STOP(decision: D-2)`.
- Change, in the `provable-ladder` job:
  - On a PR: run `pv bridge fresh --changed <diff files>`. For any stale contract, re-run B2+B3 and fail if the committed `bridge-summary.json` entry differs from the recomputed one.
  - Nightly: run everything.
  - `l5_bridged` is base-vs-head never-down, with no baseline file (Decision 8).
  - The orchestrator edits `ci.yml` itself (PVL v3).
- Mutation: throwaway PR `pvb-8-mutation` with the BR-5 bit flip → `provable-ladder` RED. Close it unmerged and link the run in the main PR body.
- probe: `present 'pv bridge fresh' "$WT/.github/workflows/ci.yml" && merged BR-8`
- accept: the PR body links the mutation run (RED) and the main PR run (GREEN); the receipt records both run URLs.

### Phase 4: widen

**BR-9** · "PVB-9: expand to D-3" · `depends_on: [BR-8]` · repeatable
- Precondition: ledger line for D-3, else `STOP(decision: D-3)`.
- Each run takes up to 10 next `bridgeable=yes` contracts, in the order `exact` → `ieee` → `real`.
- When none remain below target, file one GH issue listing the `no:<reason>` buckets: `noncomputable` (needs a BR-3b-style authored model), `prose_predicate`, `ghost_binding`, `missing_tolerance`.
- probe: `json_object crates/aprender-contracts-staging/bridge-summary.json '(.entries|length)>=<D-3 count>' && merged BR-9` (count computed by BR-1, frozen into the ledger at the D-3 ruling)
- accept: `"$PV" proof-status --format json > "$SCRATCH/p.json" && json_object "$SCRATCH/p.json" '.l5_bridged/.kernel_n>=0.20'`

**BR-10** · "PVB-10: spec-weakness sweep" · `depends_on: [BR-9]`
- Change: for each `finding: spec_weakness` entry, strengthen the `ensures` `rust:` predicate, regenerate the harness, and re-run B2 against the recorded mutant until it is RED.
- probe: `json_object crates/aprender-contracts-staging/bridge-summary.json '[.entries[]|select(.finding=="spec_weakness")]|length==0' && merged BR-10`
- accept: the same check, plus each closed finding's mutant run is RED under the new harness.

---

## §3 Hard rules (on top of PVL v3 §5)

1. **One source.** The Lean model is reached through ONT `proof.lean`; the Rust function through `binding.yaml`; predicates through `requires`/`ensures` `rust:`. Hand-edited generated harnesses or tests are RED.
2. **Summaries, never YAML, carry credit.** No level above L3 comes from a YAML field.
3. **Tracked evidence is deterministic.** No `git_sha`, timestamps, wall times or hosts in tracked summaries; those stay in untracked logs.
4. **Zero is a decline.** `n == 0`, `kernel_n == 0`, or an empty pilot → exit 2 with a named reason.
5. **No label inflation.** `ieee`/`real` never print a bare `L5`; `exact` never has float outputs.
6. **No third-party translator in the gate.** No Python. Generators are Rust; glue is bash.
7. **Hash drift is an andon.** Changing a bound function, Lean module, contract or harness zeroes that contract's L5 until it is re-bridged.
8. **Probes are cheap.** Any `lake`, `cargo` or `kani` in a `probe:` line is a spec defect.
9. **Dependency direction.** Consumers depend on `pv`; `pv` never invokes pmat or any consumer. `pmat query` may help *author* bindings; it never sits in the census, the resolver or the gate.
10. **Public-safe.** Nothing written by this spec into aprender (spec text, ledger, receipts, fixtures, summaries, PR bodies, issues) names private hosts, budgets, token figures, revenue or repo tiers. Before each PR, the harness scans the diff with the operator's private-term list (kept operator-side, never vendored). Any hit → `STOP(public-safety: <file>)`.

---

## §4 Scoreboard (target + probe; baselines frozen in §1)

| Metric | Target | Probe |
|---|---|---|
| Bridged L5 count | ≥ 6 incl. Q4_K (BR-7); then D-3 | `proof-status --format json` → `.l5_bridged` |
| Self-declared L5 counted | 0 | `.l5_source=="bridge"` |
| Mismatches on `main` | 0 | `bridge-summary.json` |
| Non-`requires` `kani::assume` in generated harnesses | 0 | `rg -c 'kani::assume' <generated dir>` vs `requires` count |
| Planted-mutant recall (B2 ∨ B3) | 100 % | BR-5, BR-6, BR-8 mutation runs |
| Open spec-weakness findings | → 0 | BR-10 probe |
| Selector probe time per row | ≤ 5 s | receipt `probe_wall_s` |
| `pv bridge fresh` wall time, full pilot set | ≤ 5 s | BR-7b accept |
| PR-tier bridge time per changed contract | ≤ 120 s `[A]` | untracked `bridge.json` `.wall_s` |
| Public-safety hits merged | 0 | rule 10 scan in each receipt |
| External consumers with ≥ 1 bridged contract | ≥ 1 (pmat, via PVB-PMAT-001) | consumer's `pv proof-status` |

---

## §5 Five whys: why the rows exist

1. Why can't we claim the kernels are proved? The theorems are about Lean models.
2. Why don't they reach the code? Nothing links a model to a Rust function (bindings 0/3133).
3. Why not? Credit was self-declared, and no job built Lean.
4. Why was it self-declared? The producer graded itself.
5. Why was that allowed? No gate demanded evidence (PV-LEAN-AUDIT-001).

PVL-001 makes the checker run and own L4. PVB-001 makes content-addressed summaries own L5. Both countermeasures are mechanisms, not persons.

v1.0's own failure:
1. Why did v1.0 cite stale rows? It was written from a v2-era snapshot.
2. Why the snapshot? The author read no live spec.
3. Why didn't the format force a live read? It had no row requiring one.
4. Why does that matter? The fix belongs in the mechanism, not the author.
5. Countermeasure: BR-0 verifies every PVL dependency by `merged_pvl` before any write.

---

## §6 Consumers (pmat and any Rust repo)

`pv bridge` ships in `aprender-contracts-cli` on crates.io.

**Publishing gate.** The clean-room CI gate on the aprender tag is the hard gate for that release. A consumer may adopt a `pv` version only after it is published through that gate and pinned fleet-wide (PVL EV-13). Consumers never build `pv` from a branch.

**The stable consumer surface** (semver-governed; a breaking change bumps the major or the schema id):

| Surface | Contract |
|---|---|
| `pv proof-status --format json` | `.l5_source`, `.l5_bridged`, `.kernel_n`, per-contract level labels |
| `pv bridge fresh --format json` | `schema: pvb-fresh/1`; exit `0/1/2` as BR-7b |
| `pv --version` | compared by consumers to the fleet pin; mismatch → consumer declines |

Consumers parse only these, never `bridge-summary.json` directly, so the definition of fresh and of L5 lives in one place.

pmat's adoption (comply rule, `pmat verify` freshness tier, ticket ↔ contract binding, dogfood on pmat's own contracts) is specified in **PVB-PMAT-001** in the pmat repo. It depends on BR-7b plus a published `aprender-contracts-cli` carrying `pv bridge`.

---

## Appendix A: epic and sub-issues (the traffic cop mints; single-minter rule)

**Placement (D-5).** Milestone = train **N+1**, resolved by the cop at mint time as the first open release milestone after the current train. Worked through the look-ahead **L1** slot (APR-LOOKAHEAD-001). Parent = the standing contracts/ontology epic. 14 sub-issues, within the 7–20 epic rule.

**Epic title:** `EPIC: Proofs reach code — PVB-001 proof↔code bridge`
**Labels:** `epic`, `contracts`, `P1`

**Epic body:**
> Lean theorems prove properties of models; 0/3133 bindings resolve, so no proof reaches shipped code. Spec `docs/specifications/PVB-001-proof-code-bridge.md` v1.3 defines L5 = discharge ∧ binding ∧ Kani ∧ differential-oracle evidence, held in a tracked content-addressed `bridge-summary.json`, nested under ONT-001 `proof:`.
> Pilot: five existing computable `exact` Lean models, then an authored computable Q4_K scale/min unpacker.
> Consumer surface: `pv proof-status --format json` and `pv bridge fresh` (`pvb-fresh/1`), used by pmat via PVB-PMAT-001.
> Depends on: PVL-001 EV-1, EV-2, EV-3, EV-5a, EV-6a, EV-6b, EV-7a, EV-7b, EV-8a, EV-8b, EV-9 (D-4: the worker lands missing rows first).
> Rulings 2026-10-27: D-1 L5 = four legs, no legacy label · D-2 changed-only PR checks + full nightly · D-3 ≥ 20 % kernel-kind · D-4 PVL rows first · D-5 train N+1, L1 slot, one worker.
> Exit: `proof-status` → `.l5_source=="bridge" and .l5_bridged>=6`; planted bit-flip mutant RED in `provable-ladder`; BR-9 at ≥ 20 %; 0 open spec-weakness findings.

**Sub-issues, in order** (title · one-line acceptance, the row's `accept:`):

| # | Title | Kind |
|---|---|---|
| 0 | `PVB-S: land PVB-001 v1.3 spec` (docs-only tier) | docs |
| 1 | `PVB-0: preflight — PVL gate landed, public probe library, Float32 probe` | code |
| 2 | `PVB-1: pv bridge census` | code |
| 3 | `PVB-2: proof.bridge schema + typed rust: predicates` | code |
| 4 | `PVB-3a: pilot — five existing computable exact models bound` | code |
| 5 | `PVB-3b: computable Q4_K scale/min Lean model + theorem` | code |
| 6 | `PVB-4: Lean reference executable pv-ref` | code |
| 7 | `PVB-5: pv bridge diff (B3) + bridge-summary.json` | code |
| 8 | `PVB-6: B2 — pv kani from rust: predicates` | code |
| 9 | `PVB-7: L5 read from summaries` | code |
| 10 | `PVB-7b: pv bridge fresh` | code |
| 11 | `PVB-8: bridge step in provable-ladder` | CI |
| 12 | `PVB-9: widen to 20 %` (repeatable) | code |
| 13 | `PVB-10: spec-weakness sweep` | code |

**Mint commands** (aprender cop session only):
```bash
cd ~/src/aprender
sed -n '/^\*\*Epic body:\*\*/,/^\*\*Sub-issues/{/^> /s/^> //p}' docs/specifications/PVB-001-proof-code-bridge.md > "$SCRATCH/pvb-epic.md"
gh issue create -R paiml/aprender --title "EPIC: Proofs reach code — PVB-001 proof↔code bridge" \
  --body-file "$SCRATCH/pvb-epic.md" --label epic --label contracts --label P1 --milestone "<N+1>"
# then one `gh issue create` per sub-issue row, each linked as a sub-issue of the epic,
# same milestone, followed by `pmat work add --github-issue <n>` for each
```

---

## Appendix B: `scripts/pvb/lib.sh` and `scripts/pvb/lib-check.sh` (normative)

BR-0 commits these byte-for-byte. Pin (`scripts/pvb/lib.sh.blake3`): `7885d60a8eb095e2dccc554fe94c9afcd9b5014b585b3f112f75bde0c4a8681a`.

Verified `[V 2026-10-27]` in a scratch repo: the self-test passes on this text and fails (exit 1) on each of five mutants:
- M1 `json_object` ignores its filter (the C-1 defect) → 1 FAIL
- M2 `rc_is` redirects to `/dev/null` (C-1) → 2 FAIL
- M3 `rc_is` keeps `--` (C-1) → 5 FAIL
- M4 `merged` ignores `probe_sha256` (C-3) → 1 FAIL
- M5 one byte edited without re-pinning → pin FAIL

shellcheck was not available in that environment: `[U]`. BR-0 runs `shellcheck scripts/pvb/*.sh` in its accept if the CI image has it.

### `scripts/pvb/lib.sh`

```bash
#!/usr/bin/env bash
# scripts/pvb/lib.sh — public probe library for PVB-001 and PVB-PMAT-001.
# Public-safe: names no private repo, host, path or budget. Pinned by lib.sh.blake3.
# Never redirects caller output; never fetches. Exit 2 = decline (missing input).
# Requires: bash >= 4, git, jq, rg, sha256sum. merged_remote also needs gh.

: "${LEDGER:?set LEDGER to the repo-relative ledger path of the spec}"
: "${PVB_MAIN:=origin/main}"
: "${PVB_DONE_WHEN_DIRS:=scripts/pvb/done_when scripts/pvb-pmat/done_when}"

tracked() { git ls-files --error-unmatch -- "$1" >/dev/null 2>&1; }

present() { rg -q -- "$1" "$2"; }

absent() { [ -e "$2" ] || return 2; ! rg -q -- "$1" "$2"; }

# json_object FILE [FILTER]: FILE holds exactly one JSON object AND FILTER is true.
json_object() {
  [ -s "$1" ] || return 1
  jq -se 'length == 1 and (.[0] | type == "object")' "$1" >/dev/null 2>&1 || return 1
  jq -e "${2:-true}" "$1" >/dev/null
}

# rc_is N [--] CMD...: CMD exits N. Caller's redirections apply to CMD.
rc_is() {
  local want="$1"; shift
  [ "${1:-}" = "--" ] && shift
  local rc=0
  "$@" || rc=$?
  [ "$rc" -eq "$want" ]
}

row_script() {
  local d
  for d in $PVB_DONE_WHEN_DIRS; do
    [ -f "$d/$1.sh" ] && { printf '%s\n' "$d/$1.sh"; return 0; }
  done
  return 1
}

probe_sha256() {
  local s; s="$(row_script "$1")" || return 2
  sha256sum "$s" | cut -d' ' -f1
}

# merged ROW: a ledger line for ROW with the current probe hash is on PVB_MAIN.
merged() {
  local want; want="$(probe_sha256 "$1")" || return 2
  git show "$PVB_MAIN:$LEDGER" 2>/dev/null \
    | jq -se --arg r "$1" --arg h "$want" \
        'any(.[]; .row == $r and .probe_sha256 == $h)' >/dev/null
}

# merged_remote OWNER/REPO REF LEDGER_PATH ROW: presence on that repo's REF.
merged_remote() {
  gh api "repos/$1/contents/$3?ref=$2" --jq .content 2>/dev/null \
    | base64 -d 2>/dev/null \
    | jq -se --arg r "$4" 'any(.[]; .row == $r)' >/dev/null
}

# merged_pvl ROW: operator-side PVL ledger, path supplied by environment only.
merged_pvl() {
  [ -n "${LEDGER_PVL:-}" ] && [ -s "$LEDGER_PVL" ] || return 2
  jq -se --arg r "$1" 'any(.[]; .row == $r and ((.merged_sha // "") != ""))' \
    "$LEDGER_PVL" >/dev/null
}
```

### `scripts/pvb/lib-check.sh`

```bash
#!/usr/bin/env bash
# scripts/pvb/lib-check.sh — pin + vocabulary + behavioural self-test for lib.sh.
# Run from the repo root. Exit 0 only if every check passes.
set -u
here="$(cd "$(dirname "$0")" && pwd)"
fail=0
check() { if "$@"; then printf 'ok   %s\n' "$*"; else printf 'FAIL %s\n' "$*"; fail=1; fi; }
refute() { if "$@"; then printf 'FAIL (should be false) %s\n' "$*"; fail=1; else printf 'ok   not %s\n' "$*"; fi; }

# 1. pin
want="$(cat "$here/lib.sh.blake3")"; got="$(b3sum "$here/lib.sh" | cut -d' ' -f1)"
check test "$want" = "$got"

# 2. vocabulary (load in a scratch env so LEDGER is set)
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
export LEDGER="docs/audits/PVB-SELFTEST/ledger.jsonl"
# shellcheck source=/dev/null
. "$here/lib.sh"
for fn in tracked present absent json_object rc_is row_script probe_sha256 merged merged_remote merged_pvl; do
  check declare -F "$fn" >/dev/null
done

# 3. json_object honours its filter (C-1 regression) and rejects non-objects
printf '{"a":1}\n' > "$tmp/obj.json"
printf '{"a":1}\n{"a":2}\n' > "$tmp/two.jsonl"
printf '[1]\n' > "$tmp/arr.json"
: > "$tmp/empty.json"
check  json_object "$tmp/obj.json"
check  json_object "$tmp/obj.json" '.a==1'
refute json_object "$tmp/obj.json" '.a==2'
refute json_object "$tmp/two.jsonl"
refute json_object "$tmp/arr.json"
refute json_object "$tmp/empty.json"

# 4. rc_is strips -- and honours caller redirections (C-1 regression)
check  rc_is 3 -- sh -c 'exit 3'
check  rc_is 0 sh -c 'exit 0'
refute rc_is 0 -- sh -c 'exit 1'
rc_is 0 -- sh -c 'echo to-stderr >&2; echo to-stdout' 2>"$tmp/e" >"$tmp/o"
check grep -qx to-stderr "$tmp/e"
check grep -qx to-stdout "$tmp/o"

# 5. merged: presence on main with matching probe hash (C-3)
(
  cd "$tmp" && git init -q -b main repo && cd repo
  git config user.email t@t; git config user.name t
  mkdir -p scripts/pvb/done_when docs/audits/PVB-SELFTEST
  echo 'exit 0' > scripts/pvb/done_when/BR-zz.sh
  : > "$LEDGER"; git add -A; git commit -qm base
  export PVB_MAIN=main
  refute merged BR-zz                                  # no line yet
  h="$(sha256sum scripts/pvb/done_when/BR-zz.sh | cut -d' ' -f1)"
  printf '{"row":"BR-zz","probe_sha256":"%s"}\n' "$h" >> "$LEDGER"
  git commit -qam 'row merged'
  check  merged BR-zz                                  # line + matching hash
  echo 'exit 0 # edited' > scripts/pvb/done_when/BR-zz.sh
  refute merged BR-zz                                  # probe edited -> reopened
  check rc_is 2 -- merged BR-nope                      # no probe script -> decline
  exit "$fail"
) || fail=1

# 6. merged_pvl declines without LEDGER_PVL
( unset LEDGER_PVL; rc_is 2 -- merged_pvl EV-1 ) || { echo 'FAIL merged_pvl decline'; fail=1; }

exit "$fail"
```
