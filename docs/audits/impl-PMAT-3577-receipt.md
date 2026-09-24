# PMAT-3577 / #3577 — receipts under contract: `parity-receipt-v2` + `extract:parity-receipt` + the back-fill

`checkout-only: true` — `pv` is 0.65.2 fleet-wide and has neither `lint --gate` nor `extract`, so every
verdict below was rendered in a checkout at the HEAD-built `pv` 0.68.2. A verdict on ≥ 1 fleet host is owed
once the pin lands (#3567), and this row does not wait for it.

Worktree `/mnt/nvme-raid0/agent-wt/rel-3577`, branch `PMAT-3577-receipts-under-contract`, cut from
`origin/main` 863ba48ef. Measurements re-run by me, not taken from a lane.

## What the row found before it built anything

**1. The committed denominator is 7, not 8.** The ticket and the cop both said 8 (7 legacy + the #3574
receipt). Measured over `git ls-files 'evidence/parity/**/*.json'`: 113 files, **7** parity records, **0**
with a `comparator`. The #3574 receipt is in PR #3575, which is **still open** — it is not on `main`. The
cop confirmed and will bump the denominator to 8 as part of landing #3575, which is this row's own
falsifier firing on its first real use.

**2. Item 6 of the ticket is mis-aimed, in the way the ticket itself warns about.** The ruling said
"`receipt-lint` is deleted, replaced by `pv lint`"; the ticket corrected that (`receipt-lint` is
paiml-implement's quorum linter) and re-aimed item 6 at `scripts/check_parity_receipt.sh`. Measured: that
script validates a **different artifact family**. Its fixtures carry `instrument`, `protocol_ref`, `lanes[]`
with `decode_tok_per_sec` and a llama.cpp `build_commit`; `bench_receipt.py::validate_parity` requires
`instrument`, `protocol_ref`, `model`, `lanes`. A logit record has never carried one of them. Its callers
are `check_perf_claims_cite_receipts.sh`, `check_multiplatform_dogfood.sh` and `parity_host_receipt.sh`.
Folding it in would have deleted the validator for #2696 — published-apr-takes-the-CPU-path-and-reports-0.099x
— from a family nobody was watching. **`check_parity_receipt.sh` is out of scope and untouched**, agreed with
the cop, and the `done_when` line is satisfied by the honest answer: nothing to fold, different family.

The ruling's premise gets *stronger*: "receipts are the only artifact family with no shape" is **literally
true** of the logit records. No validator of any kind, ever. That is why seven of them carried no comparator
for months — there was nothing that could have noticed.

**3. `model_sha256` is `pattern` with no `minCount`, deliberately.** Six of the seven never recorded a model
hash. Hashing the files on the hosts today and attaching that to a receipt about 2026-09-06 would be a claim
about a different world wearing a witness's clothes. Absent means no measurement, never a match (ONT-4c1);
`partially_receipted: true` and an `unmeasured` entry carry the honesty. Confirmed by the cop before use.

## What landed

| | |
|---|---|
| `contracts/parity-receipt-v2.yaml` | 3 shapes: `parity-receipt-complete` (closed, `ignoredProperties: []`), `parity-comparator-self`, `parity-comparator-oracle` |
| `contracts/parity-receipt-v1.yaml` | the retired layout, recorded; **no shape** — see below |
| `crates/aprender-contracts/src/ontology/extract/parity_receipt.rs` | the extractor + 10 unit cases |
| `scripts/parity_receipt_denominator.sh` | the INDEPENDENT predicate + 4-case self-test |
| `evidence/parity/EXPECTED_RECEIPTS` | the committed denominator (7) |
| `crates/aprender-contracts-cli/tests/ont4c3_parity_receipts.rs` | 8 CLI cases over 5 fixtures |
| `evidence/parity/**` (7 files) | migrated to v2 + back-filled, one commit |
| `contracts/{contracts.nt,shapes.ttl,census.json}` | regenerated (derived, R-18 / decision 8) |

**v1 declares no shape, on purpose.** All seven instances were migrated, so a shape over the retired layout
would target a class nothing instantiates and pass vacuously — worth less than no shape. The enforcement that
replaces it fires: the extractor **refuses** an unmigrated record by name, and the denominator predicate
refuses it independently.

**Nothing is armed.** Arming lives in `contracts/lint-baseline.json` `armed_shapes[]`, a shared file this row
is forbidden to touch (decision 7). The three shapes are computed and reported, exactly as `ladder-green` was
at ONT-4c1; arming them is a follow-up carrying the `touches-shared-contracts` label.

## Controls, every one measured in both directions

| control | before | after |
|---|---|---|
| **the back-fill** — comparator stripped from all 7 vs as committed | **7** parity-shape violations | **0** |
| **the plant** — comparator removed from one record | **exactly 1**, naming `ont:parity/comparator` and `minCount` | 0 on restore |
| **the mutation** — `in:` widened to accept `oracle`, contract + all 5 fixtures | `ont4c3_parity_receipts` **FAILS** (`a_comparator_kind_the_shape_does_not_accept_fails`) | 8/8 on restore |
| **the mutation, other half** — only the real contract widened | **FAILS** (`every_fixture_carries_the_real_contract_byte_for_byte`) | 8/8 on restore |
| **an unmigrated record** (hit for real when a `git checkout` reverted the migration mid-run) | `Unknown{WrongCorpus}`, **exit 2**, each legacy file refused by name | Pass after re-migration |
| **denominator drift** — 2 receipts, committed 1 | exit 2, "matched 2 … says 1" | — |
| **the three answers are distinct** | `parity-green` 0 · `parity-nocomparator` 1 · `parity-unmigrated` 2 | — |

```
cargo test -p aprender-contracts --lib ontology::extract::parity_receipt   10 passed
cargo test -p aprender-contracts --lib ontology::verdict                    8 passed
cargo test -p aprender-contracts-cli --test ont4c3_parity_receipts          8 passed
bash scripts/parity_receipt_denominator.sh --self-test                      4 ok
bash scripts/parity_receipt_denominator.sh        PASS 7 receipt(s), EXPECTED_RECEIPTS says 7
pv lint contracts --gate shapes                   Pass, violations 0, parity-shape violations 0
pv extract contracts --check                      rc 0
```

`by_shape` on the committed tree: `parity-receipt-complete=7`, `parity-comparator-self=7`,
`parity-comparator-oracle=0`.

## The back-fill is a relabel, and every field cites committed evidence

`raw` in each migrated record is the original `apr parity --json` document, key for key. The envelope was
quoted from files already in the tree, named in each record's `provenance.record`:
`evidence/parity/l0-1/{lambda,gx10}/RECORD.md` and `evidence/parity/l0-1b/gx10/n5/DETERMINISM.md`.

**One derived field was wrong on the first pass and is worth recording.** `result.verdict` initially copied
`raw.parity` — apr's own per-position band flag — which says PASS for the two 1.5B cells that their own
`RECORD.md` calls RED. It now resolves the threshold from `evidence/parity/thresholds.yaml` (`default.min_cosine`
0.98, `min_positions` 64) and judges min cosine over ≥ 64 positions, which reproduces all seven readings the
RECORD.md files state, including both REDs. A threshold is **never** typed into the shape — that is this row's
STOP condition — and `thresholdSource` is `resolves:`, with a missing path materialised as
`thresholdSourceMissing` for a shape to refuse with `maxCount 0`.

Cross-check: the migrated `min_cosine` values reproduce `evidence/parity/thresholds.yaml`'s basis text to six
decimals (7B lambda 0.998607, 7B gx10 0.998465, 1.5B lambda 0.950827, 1.5B gx10 0.950611).

## Not done, and why

**`ONT-4c3 bound in the ONT-001 ledger` — cannot be done from this repository.** The ledger is
`docs/specifications/paiml-ontology.md` in **paiml/infra**, where v4.8 defines ONT-4c3 as **kernel** receipts
(`entity type kernel`, `apr-kernel-receipt/v1`, the `kernel-parity`/`kernel-timing`/`kernel-safety` shapes).
**Filed as paiml/infra#814** with both resolutions (renumber the parity row, or record the re-scope in a
v4.9) and no preference between them — the point is that one identifier should not mean two things in two
repositories. The re-scope is an aprender-side ruling that the infra spec does not yet carry, so binding
it is an **infra PR**, not this one. Raised with the cop rather than left as a checked box. The kernel
sub-row with the `gated_rmsnorm` fixture stays a follow-up either way.

## Follow-ups

- Arm the three shapes in `contracts/lint-baseline.json` (`touches-shared-contracts`, group of one).
- Bind ONT-4c3 in infra's ONT-001 ledger once **paiml/infra#814** decides which meaning the identifier keeps.
- #3575 bumps `EXPECTED_RECEIPTS` to 8 when it lands, and its receipt needs the v2 envelope
  (`comparator`, `partially_receipted`, `backend`, `generated_at`) or these shapes will report it.
- `Receipt` as a shared parent class once quorum and dispatch receipts join (NOT this row).
- The dense oracle re-measurement, comparator ruling item (d).
