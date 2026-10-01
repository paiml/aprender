---
phase: 04-apr-artifact-and-production-parity
plan: 01
subsystem: contracts
tags: [contract, artifact-schema, setfit, apr, makefile-gate, claude-md]
requires: []
provides:
  - "contracts/setfit-apr-v1.yaml — the normative setfit-apr-v1 artifact schema"
  - "storage map: every APR-01 item -> location | bundle source | serialized form | loader destination"
  - "SetFitArtifactDoc normative field list (16 fields, deny_unknown_fields)"
  - "doc<->bundle bijection table, both directions, plus the closure equation"
  - "canonical tensor name table for all 21 HF templates + 3 schema-owned names"
  - "nullable-path allowlist: 4 paths over 5 walked sub-documents"
  - "constants: max_artifact_bytes 268435456, metadata bound 16777216"
  - "6 contract-resident synthetic probe strings + probe record schema"
  - "probe/parity tolerances citing setfit-encoder-conformance-v1"
  - "backend-identity grammar cpu:setfit-core:autograd-trueno-matmul"
  - "read_setfit_apr_bytes_bounded bounded-read obligation"
  - "ClassifyRequestDocument / ClassifyResponse schema obligations"
  - "selection-lock artifact lifecycle"
  - "make contract-audit-phase4 — blocking, tier3-wired, non-vacuous"
affects:
  - "04-02 (writer), 04-03 (loader), 04-04 (envelope), 04-05 (codec), 04-13 (bundle field 20 + completeness gate), 04-07/04-14 (lock CLI), 04-09 (parity), 04-10 (gates)"
tech-stack:
  added: []
  patterns:
    - "contract-before-code (Ph1 D-14): every constant committed before the comparison code exists"
    - "one new contract per phase, referencing never editing the others (Ph1 D-23)"
    - "scoped blocking binding audit (PHASE2/3/4_CONTRACTS), never the vacuous repo-wide contract-audit"
    - "status: pending (BIND-004 warning) for equations whose modules are not yet written"
key-files:
  created:
    - "contracts/setfit-apr-v1.yaml"
  modified:
    - "Makefile"
    - "CLAUDE.md"
    - "contracts/aprender/binding.yaml"
decisions:
  - "D-01 AMENDED: 6 of the 21 HF name templates have no canonical form in tensor-names-v1; this contract reserves them, with the non-collision checked"
  - "D-02(a) AMENDED (recorded, not silent): model_type is the only typed key; schema/schema_version/ordered_labels/tokenizer_sha256 are first-level doc fields"
  - "The nullable-path allowlist is 4 paths while the walk is 5 sub-documents, deliberately"
  - "contract-audit-phase4 tolerates BIND-004 (pending) and refuses BIND-001 (missing), because the schema lands before the code"
metrics:
  duration_seconds: 1549
  tasks_completed: 3
  files_changed: 4
  completed: 2026-08-15
---

# Phase 4 Plan 01: Contract and Wiring Summary

The `setfit-apr-v1` artifact schema — storage map, doc field list, doc↔bundle bijection, canonical
tensor names, the four-path nullable allowlist over five walked sub-documents, and every Phase 4
constant — committed as one pv-valid contract before any code that could quietly answer those
questions differently, and wired into a blocking tier3 gate that was shown able to fail.

## What Shipped

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | `contracts/setfit-apr-v1.yaml` — 15 equations, 18 proof obligations, 23 falsification tests, 4 declared-not-executed Kani harnesses, 1 qa_gate | `488e307d5` |
| 2 | `$(CONTRACTS)` append + `PHASE4_CONTRACTS` + `contract-audit-phase4` + tier3 wiring + 15 `pending` binding entries | `127dcc5db` |
| 3 | CLAUDE.md realizar-first SetFit row + D-09 rationale note | `697295f7a` |

## Verification Output

`pv validate` — the plan's Task 1 verify command, status captured directly, never through a pipe:

```
$ cargo run --release -p aprender-contracts-cli --bin pv -- validate contracts/setfit-apr-v1.yaml
0 error(s), 0 warning(s)
Contract is valid.
rc=0
```

`pv lint contracts/` (directory form) — **PASS, 0 errors**, 15 warnings for this file, all
`PV-ENF-002` (no `lean_theorem`). That is the same class and posture as Phase 3's
`setfit-train-lifecycle-v1.yaml`, which carries 13 warnings all of the same rule. Declaring a
`lean_theorem` name for a theorem that does not exist would contradict the contract's own
PROVABILITY HONESTY paragraph.

**A measurement caveat worth recording rather than burying.** `pv lint contracts/setfit-apr-v1.yaml`
— the single-FILE form — reports `Gate 1: validate ✓ (0 contracts, ...)` and `Result: PASS`. It
found **zero contracts** and passed anyway: a vacuous gate, the CR-02 failure class. `pv lint`
requires a DIRECTORY. Every lint figure quoted here is from the directory form. Had this gone
unnoticed, the summary would have reported a green lint that inspected nothing.

Task 2 verify (the plan's command verbatim, count captured into a variable and compared with
`test` — never read from `grep -c`'s exit status):

```
observed count (non-comment Makefile lines): 2
Phase 4 binding audit: 1 contract(s) audited, every equation is bound
VERIFY_RC=0
```

Task 3 verify: `grep -c "SetFit" CLAUDE.md` → **0 (exit 1) before, 3 (exit 0) after**. A real
RED→GREEN transition, not a criterion that was already satisfied.

`make contract-validate` reaches the new file (rc=0; `Running .../pv validate
contracts/setfit-apr-v1.yaml` appears in its output), so the `$(CONTRACTS)` append is not
decorative.

## The Committed Constants

| Constant | Value | Derivation |
| -------- | ----- | ---------- |
| `max_artifact_bytes` | `268435456` (256 MiB) | ~2.96x over the derived ~90.8 MB legitimate artifact; mirrors the `MAX_BUNDLE_BYTES` precedent (512 MiB over ~182 MB, ~2.9x) |
| metadata bound | `16777216` | the container's own `MAX_METADATA_SIZE` (`apr-format/src/v2/mod.rs:80`) — reused, not re-derived |
| encoder payload | `90261504` bytes | 22,565,376 f32 × 4; the element count matches `MAX_TOTAL_ELEMENTS`'s doc comment, derived independently here from the name table |
| tokenizer blob | `466247` bytes | restated from `setfit-train-lifecycle-v1`'s `bundle_completeness` |
| reference tensor count | `101` encoder + 3 schema-owned | 5 global + 16/layer × 6; matches `MAX_TENSOR_COUNT`'s recorded "the full pin carries 101" |
| probe embedding tolerance | `7.62939453e-06` | **cited** from `setfit-encoder-conformance-v1`'s pooled-output family, not re-derived |
| probe/parity logit + probability tolerance | `1.0e-5` absolute | labels compared EXACTLY; train-time round trip stays `Tolerance::EXACT` |
| probe count | `6` | fixed synthetic contract-resident strings |
| `max_batch_texts` | `256` | Security V5 |
| `max_request_body_bytes` | `1048576` | Security V5 |
| backend identity v1 | `cpu:setfit-core:autograd-trueno-matmul` | three-segment grammar `<device>:<implementation>:<kernel>` |

## The Storage Map As Shipped (review B1 + B2 closed)

| Item | Storage location | Bundle source | Serialized form | Loader destination |
| ---- | ---------------- | ------------- | --------------- | ------------------ |
| encoder tensors | named F32 tensors, canonical names | `tensors` | LE f32, row-major | HF-named map → `SetFitMiniLm::from_bundle_parts` |
| head weights | **tensor** `setfit.head.weight` `[num_labels, head_n_features]` | `head_weights_hex` | LE f32, row-major | `MultinomialLogisticRegression::from_stored_coefficients` |
| head intercepts | **tensor** `setfit.head.bias` `[num_labels]` | `head_intercepts_hex` | LE f32 | same call |
| tokenizer bytes | U8 tensor `tokenizer.blob` `[len]` | `tokenizer_bytes_hex` | raw, byte-identical to upstream | `from_bundle_parts` tokenizer input |
| family tag | TYPED metadata `model_type = "setfit"` | (constant) | JSON string | D-04 detection |
| everything else | the ONE custom key `"setfit"` → `SetFitArtifactDoc` | see bijection | one JSON object, BTreeMap-ordered | one `deny_unknown_fields` parse |

The head is **tensors, never metadata** — stated in-contract with the reason (a `K*d` float array in
JSON is lossy, unaligned, invisible to `apr tensors`, and pushes metadata toward the 16 MiB bound).

## The Bijection Table As Shipped (review B3 closed at the specification layer)

Forward, doc ← bundle: 16 rows. Thirteen are `= bundle.<field>` or a field of one
(`bundle_schema_version`, `format_id`, `architecture`, `tokenizer_sha256`, the six `preprocessing.*`,
`root_seed`, `head.n_features`, `ordered_labels`, `requested_config`, `resolved_config`, `evidence`,
`provenance`). Three are named deterministic functions: `head.num_labels =
bundle.ordered_labels.len()`, `hf_name_map = f(bundle.tensors.keys(), this contract's name table)`,
`probes = f(rebuild(bundle), this contract's probe inputs)`. Two are consts.

Reverse, bundle ← doc: all **20** bundle fields recovered — tensors via `invert(hf_name_map)`,
`tokenizer_bytes_hex` from the `tokenizer.blob` payload, the two head hex fields from the two head
tensors, the rest read directly.

Closure: `serialize(deserialize(bytes)) == bytes` at `Tolerance::EXACT` — a **byte** comparison, no
float tolerance participates. Two consequences recorded in-contract because they are easy to get
backwards: probes are **recomputed** on every serialize (which is what makes them closure-safe, and
why they never become a 21st bundle field), and closure **does not subsume** the null guard — a
`null` at an allowlisted path round-trips to `None` and back to `null`, so closure holds while the
value is gone. That is why the scan runs at write time and not only as a byte compare.

## The Four Allowlisted Nullable Paths

| Path | Owning type | Option field type | Source |
| ---- | ----------- | ----------------- | ------ |
| `architecture.vocab_remap` | `EncoderArchitecture` | `Option<Vec<u32>>` | `aprender-core/src/setfit/mod.rs:126-127` |
| `requested_config.pair_config.budget` | `PairConfigWire` | `Option<u64>` | `aprender-train/src/train/setfit/config.rs:716` |
| `requested_config.pair_config.hard_cap` | `PairConfigWire` | `Option<u64>` | `aprender-train/src/train/setfit/config.rs:717` |
| `evidence.epsilon_used` | `EvidenceSummary` | `Option<f64>` | `aprender-train/src/train/setfit/evidence.rs:599` |

Zero-contribution yet still walked: `resolved_config` (`ResolvedConfigRecord`, one `String`,
`bundle.rs:292-301`) and `provenance` (`ProvenanceRecord`, four `String` + `u64` + `u32`, bundle
field 20 from 04-13, which forbids an `Option` on it).

**The walk is FIVE, the allowlist is FOUR, and the contract says why.** An unwalked subtree cannot
reject anything, so a future `Option` on `ResolvedConfigRecord` or `ProvenanceRecord` would fail
silently on the first production artifact instead of loudly at the guard. Every claim above was
verified against the source, not restated from the plan: `import.rs:501` sets `vocab_remap: None`
(the full pin) and `:620` sets `Some` (the slice fixture), so an allowlist omitting that path would
refuse every pinned MiLM artifact while a fixture-based suite stayed green.

The `evidence.epsilon_used` residual is recorded rather than claimed away: it is the one allowlisted
path whose `Option` wraps a float, so `None` and a non-finite epsilon render identically. Today the
residual is empty — `epsilon_used` has exactly one assignment in the tree and it is `None`
(`evidence.rs:655`, asserted at `evidence.rs:1144`). `skip_serializing_if` is forbidden on all five
types, with the reason (it would change `SetFitBundle::to_canonical_bytes` and break Phase 3's
committed closure tests, while silently emptying this allowlist with no test turning red).

## The Makefile Wiring-Gate Falsification Transcript

Required by the plan, performed exactly once, statuses captured directly (`cmd > log 2>&1; rc=$?`),
never through a pipe:

| State | count | verify rc | Observed |
| ----- | ----- | --------- | -------- |
| `PHASE4_CONTRACTS` present | 2 | **0** | `Phase 4 binding audit: 1 contract(s) audited, every equation is bound` |
| `PHASE4_CONTRACTS` line deleted | **1** | **1** | `test "$n" -ge 2` fails, `&&` short-circuits, target never runs |
| `PHASE4_CONTRACTS := ` (empty list) | 2 | **2** (make's status) | `FAIL: PHASE4_CONTRACTS is empty — this gate audited nothing and would have reported success.` |
| line restored | 2 | **0** | green again |

The middle row is the point. A bare `grep -c` whose exit status is consumed directly would have
exited **0** at count 1 and PASSED — the same defect class as CLAUDE.md Verification rule 1. The
verify command captures the count into a variable and compares it with `test`, so it distinguishes 1
from 2.

The binding audit's own RED/GREEN was measured on a real difference, not assumed:

```
before bindings: rc=1, 15 x "[ERROR] BIND-001: ... has no binding entry"
after  bindings: rc=0, "Total equations: 15 / Bound equations: 15",
                 15 x "[WARN] BIND-004: ... is pending implementation"
```

## Contract-Shape Restructuring `pv` Forced

**None.** The contract fit the existing `KernelContract` schema on the first attempt —
`equations` + `proof_obligations` + `falsification_tests` + `kani_harnesses` + `qa_gate`, with the
large normative tables carried as extra keys inside their owning equation (the shape
`setfit-train-lifecycle-v1` already uses for `frozen_thresholds`). No bash/yq/python workaround was
written and no schema extension was needed.

One YAML syntax error was self-inflicted mid-task and fixed immediately: an `Edit` whose
`old_string` ended at a comment prefix left the comment's remainder dangling after a scalar,
producing `did not find expected '-' indicator at line 171`. Caught by re-running `pv validate`
(rc=1) rather than by reading the diff. Recorded because "the edit applied cleanly" and "the file is
still valid" are different claims.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking] `contracts/aprender/binding.yaml` needed 15 `pending` entries**

- **Found during:** Task 2
- **Issue:** The plan requires `make contract-audit-phase4` to exit 0 and says equations "are
  declared `status: pending`". `pending` is a property of the **binding registry**
  (`ImplStatus::Pending`, `binding.rs:65`), not of the contract, and `binding.yaml` is not in the
  plan's `files_modified`. Measured before acting: with no entries the audit returned **rc=1 with 15
  BIND-001 errors**, so the target could not pass.
- **Fix:** Added 15 entries with `status: pending`, each naming the `module_path`/`function` the
  later plan must make true, plus a comment block recording why they are pending and the measured
  rc=1→rc=0 transition. `pv audit` then returns rc=0 with 15 BIND-004 **warnings**.
- **Files modified:** `contracts/aprender/binding.yaml`
- **Commit:** `127dcc5db`

**2. [Rule 2 — Missing critical functionality] `contract-audit-phase4` could go vacuous**

- **Found during:** Task 2
- **Issue:** Modeled faithfully on `contract-audit-phase3`, the target would run its loop zero times
  if `PHASE4_CONTRACTS` were ever emptied, leave `unbound` empty, and **echo success**. The phase
  context states this explicitly ("a gate that can go vacuous is not a gate"; Phase 3's CR-02).
- **Fix:** Added an `audited` counter and a hard failure when it is zero, and made the success line
  print the count so the output shows work was done. Falsified: with the list emptied the target
  returns rc=2 with `FAIL: PHASE4_CONTRACTS is empty`.
- **Files modified:** `Makefile`
- **Commit:** `127dcc5db`

**3. [Rule 2 — Missing critical specification] Six HF names have no canonical form in
tensor-names-v1**

- **Found during:** Task 1
- **Issue:** The plan directs "every HF dotted name the encoder's `named_parameters()` emits, mapped
  to its tensor-names-v1 canonical form", anticipating only that the head has no role. In fact
  **six of the 21 templates have no canonical form**: four have no role at all
  (`embeddings.token_type_embeddings.weight`, both `embeddings.LayerNorm` leaves, and the three
  per-layer dense biases — there is no `o_proj_bias`, `ffn_up_bias` or `ffn_down_bias` role), and
  `position_embedding` has a `bert:` alias but an **empty `_fallback`** (line 265). On the pinned
  6-layer reference that is **22 of 101** encoder tensors with no canonical name, i.e. the writer
  would have had no name to write for a fifth of the artifact.
- **Fix:** Recorded as an explicit D-01 amendment in-contract. The six names are reserved by this
  contract (`position_embd.weight`, `token_types.weight`, `token_embd_norm.weight/.bias`,
  `blk.{n}.attn_output.bias`, `blk.{n}.ffn_up.bias`, `blk.{n}.ffn_down.bias`), follow the same
  `blk.{n}`/global convention so generic tooling sees one scheme, and the non-collision against
  tensor-names-v1's **complete** `_fallback` set was enumerated and checked rather than asserted.
  Upstreaming them as proper roles is logged as a deferred item — a deliberate `pv diff`-flagged
  edit to a contract this phase does not own (Ph1 D-23).
- **Files modified:** `contracts/setfit-apr-v1.yaml`
- **Commit:** `488e307d5`

**4. [Rule 2 — House standard] Equations shipped without preconditions/postconditions**

- **Found during:** Task 1 verification
- **Issue:** `pv lint contracts/` reported 45 warnings for the new file (15 equations × 3). Phase 3's
  contract reports 13, **all** `PV-ENF-002`, i.e. it supplies pre/postconditions on every equation
  and this one did not.
- **Fix:** Added specific, honest preconditions and postconditions to all 15 equations. `pv`'s
  suggested templates (`!input.is_empty()`, `result.len() > 0`) were **not** used — they are false
  for most of these equations and CLAUDE.md explicitly requires eliminating placeholder
  preconditions. Warnings 45 → 15, all now `PV-ENF-002`.
- **Files modified:** `contracts/setfit-apr-v1.yaml`
- **Commit:** `488e307d5`

**5. [Rule 1 — Bug] `pv lint` mutated a tracked baseline cache**

- **Found during:** Task 1 verification
- **Issue:** Running `pv lint contracts/` rewrote `.pv/lint-previous.json`, a tracked file, as a side
  effect. It is not this task's to change and would have ridden into the commit.
- **Fix:** `git checkout -- .pv/lint-previous.json` (a single named file, not a blanket reset).
- **Commit:** n/a — reverted before staging

## Criterion Defect Found (not a wiring defect)

Task 2's acceptance criterion states: *"tier3's recipe names contract-audit-phase4 (source
assertion: `grep -A 30 "^tier3" Makefile` includes it)"*. That assertion **fails**, and it fails for
a reason unrelated to this plan: `tier3:` is at line 285 and the wiring is at line 337, 52 lines
away, because of the extensive evidence-discipline comment blocks between them.

The control that settles it: the **pre-existing, working** `contract-audit-phase3` wiring (line 329)
also scores **0** under `grep -A 30 "^tier3"`. The 30-line window is stale, not the wiring.
`grep -A 60 "^tier3" Makefile | grep -c "contract-audit-phase4"` returns **1**, and reading lines
326-343 shows line 337 sitting in the contiguous tier3 recipe directly after the phase-3 call.
Reporting this criterion as a plain pass would have been the "check how it was measured" failure.

## Host Caveat

`bashrs` is **not installed on this host** (`command -v bashrs` → not found), so `bashrs make lint
Makefile` was not run. The new recipe was instead checked by inspection against the two properties
that matter here and that CLAUDE.md names: it contains **no pipe at all**, and `status=$$?` sits on
its own line immediately after the audit call. Stated as an unrun check rather than a passed one.

## Notes for Later Plans

- **04-02** must point the null walk at **all five** sub-documents and cite the four-path allowlist
  from **one named constant**, and must prove the full-pin (`vocab_remap: None`) shape is ACCEPTED.
- **04-13** owns the completeness gate asserting the allowlist is total against the shipped types,
  with `resolved_config` and `provenance` each asserted BY NAME to contribute an empty set.
- **Every plan that lands a module must flip its binding from `pending` to `implemented`.** A plan
  that lands the module and leaves the binding `pending` has recorded a claim nothing checks —
  `contract-audit-phase4` will still be green, because `pending` is by design a warning.
- The reference architecture block exists **only** for the size derivation and as a worked example.
  The validation rule is `expected(arch)` over the artifact's own `num_layers`; there is no
  six-layer list to update.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: contracts/setfit-apr-v1.yaml   (96.6K)
FOUND: Makefile                       (88.6K)
FOUND: CLAUDE.md                      (28.2K)
FOUND: contracts/aprender/binding.yaml (71.7K)
```

Commits claimed, checked in the log:

```
FOUND: 697295f7a docs(04-01): record the D-09 SetFit exception in the realizar-first table
FOUND: 127dcc5db chore(04-01): wire setfit-apr-v1 into the Makefile contract gates
FOUND: 488e307d5 feat(04-01): author contracts/setfit-apr-v1.yaml — the Phase 4 artifact schema
```

No file deletions in any of the three commits (`git diff --diff-filter=D HEAD~1 HEAD` empty for
each). Working tree clean before this summary.

## Known Stubs

None. This plan ships specification and gate wiring only; the 15 `pending` binding entries are not
stubs but the recorded, audited state of code that later plans land — and they are the mechanism by
which those plans are held to the schema.
