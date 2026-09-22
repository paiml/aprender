# ONE GATE FRAMEWORK — one and only one way to build a gate

**Status:** DESIGN, for quorum. Not implemented.
**Anchors:** #3418/#3421 (alfredodeza: dispatch duplicated with no single source), #3844, #3836, #3837, #3849, #3852
**Operator ruling, 2026-09-22:** *"we need ONE and ONLY one way to do things. toyota way"*

## 1. The problem, measured

```
scripts/check_*.sh                                   143
  ... with their own --self-test                      94
  ... with their own ratchet                          39
  ... with their own mutation proof                   38
scripts/lib/*cases*  (case-table directories)          9
separate baseline / acknowledgement / waiver files    ~25
```

Twenty-five baselines. Thirty-nine ratchets. Thirty-eight mutation proofs. Every one
hand-rolled, each slightly different, none sharing a line of code.

**This is not untidiness. It is the defect source, and tonight measured it.**

## 2. The evidence that duplication causes the defect

On 2026-09-22, one release, three sessions, **five** gates were found that could not
fail — and the fifth was found only because the first four taught nobody anything:

| gate | how it could not fail |
|---|---|
| `make contracts` | `pv lint \| tail -5` took **tail's** exit status. Verdict printed, never enforced. Third time this idiom shipped (#2336, #2360) |
| `ollama-binary` completion check | grepped `ollama --version`, which **asks the daemon** — so the mismatch warning supplied the string being grepped for. **The failure mode made the check true** |
| `ollama-service` check | discarded `/api/version`'s body and asked only whether the daemon answered. A process on the old inode answers fine, so the pin could never generate drift |
| `FALSIFY-README-003` | **passed while printing its own finding**: `cli_command_count: 111 (README lags at 110)` |
| `dogfood.sh gate()` | note picked by unanchored substring, so a FAILING row's whole explanation was `2 pass / 0 fail / 2 skip` |

And the decisive fact for this design: **fixing one told us nothing about the others.**
The cop fixed `check_no_claim_literals`, declared it done, and found
`check_perf_claims_cite_receipts` red only on a wider re-run — having run that exact
guard *before* the change and seen green. `check_apr_bin_pinned` surfaced only on a
third pass over all fifteen. Three guards, one surface, three separate discoveries.

Two more from the same night, both structural:

- **`interface-parity`'s own comment:** *"agreement cannot falsify reachability —
  rmedia's four-way parity suite was GREEN for the whole period `mcp::serve_stdio` and
  `http::serve` had no caller from main.rs."* Four transports agreeing perfectly while
  two were unreachable.
- **A kernel that never executed one instruction** while four mutants, stride
  assertions and an entry-name guard sat green above it (an em dash in PTX; `ptxas`
  rejected the module). **Every guard above a thing that never ran will pass.**

## 3. Design

**One framework. A gate DECLARES; the framework ENFORCES.**

### 3.1 One declaration
A contract with `metadata.kind: registry` — the kind `pv validate` already accepts and
five contracts already use. Per gate: id, what it judges, its surfaces, its ratchet
file, its case table, its exemption file, its runtime tier.

### 3.2 One ratchet
Shrink-only, comparand **measured from the protected base**, never read from a file the
PR can rewrite. One implementation, not 39. `check_no_fabricated_baselines.sh` already
exists because hand-rolled baselines were being fabricated — that gate is evidence for
this design, not an alternative to it.

### 3.3 One case table
Must-match **and** must-not-match rows, with each known false positive recorded **as a
row**. The `apr`-invocation pattern was wrong five times; every one was caught by a
table, none by review.

### 3.4 One mutation harness
A gate ships a mutant that makes it RED and the framework runs it. **A gate whose
mutant does not fire is refused.** This is the only mechanism that catches §2's five,
and it must be the framework's, not each gate's.

### 3.5 One exemption mechanism
An exemption requires a **non-empty reason**, is printed as `DEFERRED`/`ACKNOWLEDGED`
and **never counted as a pass**, and carries a **maintenance obligation**: when the
underlying issue closes, the row is deleted in the same commit. A stale
acknowledgement silently exempts the next instance — the hazard is real and named
(`check_unwired_capabilities.sh` header).

### 3.6 One note extractor
Anchored, never an unanchored substring, and the full output is **kept**, not
discarded. `dogfood.sh gate()` discarded its output, so the keep-the-worklog fix could
not reach the four rows that needed it most.

## 4. What the framework must be able to detect about ITSELF

The framework is a gate, so it is subject to its own rules. It must detect:

1. **A gate with no mutant** → refuse.
2. **A gate whose mutant does not fire** → refuse.
3. **A gate that prints a discrepancy and exits 0** → refuse (`FALSIFY-README-003`).
4. **A gate whose subject never ran** → the em-dash class. A gate must state what it
   would take for its subject to be absent, and the framework must be able to check it.
5. **A gate whose ratchet baseline is not derived from the protected base.**

## 5. Migration — and the gate on the migration

**Not a rewrite.** The framework proves itself by adopting existing gates one at a
time, and a migrated gate must be **byte-identical in verdict** on the current tree
before and after. Five volunteers first, chosen because each already has a mutation
proof: `check_model_ladder`, `check_apr_bin_pinned`, `check_no_claim_literals`,
`check_coverage_has_producers`, `check_unwired_capabilities`.

**Adoption is ratcheted, not mandated:** the count of framework-adopting gates may
only rise. New gates must use it.

## 6. Explicitly out of scope

Deleting any existing gate; changing any verdict; CI workflow edits (a
check-in-before-acting surface); anything in 0.69.1.

## 7. Open questions for the quorum

1. Is `kind: registry` the right contract shape, or does a gate registry need its own kind?
2. Should §4.4 (subject-never-ran) be mandatory, given only one instance is known?
3. Is byte-identical-verdict the right migration acceptance, or too strict to ever pass?
4. Does one framework for 143 gates create a single point of failure worse than 143 independent ones?
