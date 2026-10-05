# #4814 receipt: slices 0 to 4 at `9884b2c7ac`, intel

- **Commit:** `9884b2c7acfb3ff05973c1e12e9cc7cfc4fb5360`. The script checks `git rev-parse HEAD` against it before anything runs.
- **Host:** intel (x86_64, Xeon W-3245). Run under the shared `/tmp/a7-pf-intel.lock`, 19:22Z to 20:20Z on 2026-10-05.
- **Target dir:** private, `CARGO_TARGET_DIR=/home/noah/wshacl-4814-target`.
- **Script:** `intel-run.sh`. Every mutant fails closed: a patch that does not apply aborts the run, and the worktree is
  checked clean after each restore.

This is the first compile of slices 1 to 4.

| step | result |
|---|---|
| `cargo test -p aprender-contracts --tests` | rc 0: **2535 passed, 0 failed**, 16 binaries |
| `cargo test -p aprender-contracts-cli --tests` | rc 0: **592 passed, 0 failed**, 39 binaries (FALSIFY-SHP-004 fixture still exits 3 naming `targetNode`) |
| `pv lint contracts --gate shapes` | rc 0, verdict Pass, 66 shapes, 3787 focus nodes, 0 violations |
| W3C cases in the gate | `every_embedded_case_parses_and_passes` ok, so 29 of 29 |

## Mutants (each must turn its test RED)

Each one is RED with a test that compiled, ran and panicked on its assertion, not on a compile error.

| # | slice | mutation | RED test |
|---|---|---|---|
| 1 | step 2 (fail closed) | `str_key` reads a wrong-typed value as absent | `a_known_key_with_a_wrong_typed_value_is_refused_never_read_as_absent` |
| 2 | 0 | one suite case dropped from `NOT_VENDORED` | `the_table_of_ont_0_is_accounted_for_case_by_case` |
| 3 | 1 | an incomparable value is skipped, not reported | `every_embedded_case_parses_and_passes` |
| 4 | 2 | `equals` checks one direction only | `every_embedded_case_parses_and_passes` |
| 5 | 3 | `hasValue` requires every value to be the term | `every_embedded_case_parses_and_passes` |
| 6 | 4 | `targetSubjectsOf` names no focus node | `every_embedded_case_parses_and_passes`. targetSubjectsOf-001 conforms when it should not, and -002 loses InvalidInstance2 |

Mutant 1 is the step-2 RED proof (due 22:00Z).

## `make oracle` (out of gate): rc 2, OPEN

- **W3C arm:** the pinned oracle (shacl 0.3.21) agrees with every one of the 29 vendored expectations.
- **Corpus arm:** it **disagrees**. The oracle finds 736 violations and pv finds 730. The first difference is
  `(csv/csv-train, closed)` from the oracle against `(csv/csv-train, in)` from pv.
- **The branch leaves the graph unchanged:** `contracts/contracts.nt` and `contracts/shapes.ttl` are identical to
  the merge-base `11f844a772`. The tracked `differential.json` (2026-09-19) records 103 = 103, from a smaller
  corpus.
- **Lead:** pv admits `rdf:type` on every closed shape (baseline §1, `closed` is partial). The exported closed
  shapes carry no `sh:ignoredProperties rdf:type`, so the oracle reports it. The closed shapes have 5 focus
  nodes between them, so this explains at most 5 of the 6. Not proven.
- **Decisive check:** queued. It runs `main`'s pv (merge-base `11f844a772`) through the same oracle binary on the
  same graph.
- `tests/oracle/differential.json` is **not** updated until that check answers.
