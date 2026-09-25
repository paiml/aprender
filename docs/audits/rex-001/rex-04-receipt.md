# REX-04 receipt — per-cell admission, `rex-cell-admission-v1` (PMAT-4359, epic #4354)

prereg_sha `ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0` (unchanged).

## Result: 6/6 cells resolved, 0 silent; every cell `NotRun{NoDeclaredExecutor}` → **S-7**

Admission file: `docs/audits/rex-001/rex-04-admission.jsonl` (6 rows). Every row carries:
- apr `v0.69.3`, released binary sha256 `8a67a0103cbb036332908cc184d5a8a8ae422fb9025bd5c78e42d65df86cfbd7`;
- `Qwen3.5-4B-Q4_K_M`, weights sha256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`.

| Cell | Host / backend | State | Why |
|---|---|---|---|
| C1 intel-wgpu | intel / wgpu | NotRun{NoDeclaredExecutor} | infra#1088 open |
| C2 intel-cpu (reference) | intel / cpu | NotRun{NoDeclaredExecutor} | infra#1088 open |
| C3 lambda-cpu | lambda-labs / cpu | NotRun{NoDeclaredExecutor} | infra#1088 open |
| C4 gx10-cuda | gx10 / cuda | NotRun{NoDeclaredExecutor} | infra#1088 open (gx10 is P1) |
| C5a mini-cpu | mini / cpu | NotRun{NoDeclaredExecutor} | infra#1088 open |
| C5b mini-metal | mini / metal | NotRun{NoDeclaredExecutor} | infra#1088 open |

`rex admission-check --file docs/audits/rex-001/rex-04-admission.jsonl` →
`{"admitted":[],"refused":[],"not_run":["C1","C2","C3","C4","C5a","C5b"],"s7":true}`, exit **10**.

The spec expects C1 and C5b to be "likely `Refused` before 0.71". They are still recorded
as NotRun: a refusal is a result (§0.5), so it has to be observed on the declared executor,
never assumed. R-4 forbids ad-hoc SSH or hand-installed `apr`, and the REX-01 Phase 0 check
found no existing declared label that admits these jobs. No cell can therefore run until
infra#1088 lands.

## Premise check (§0.6 genchi genbutsu)

§7 REX-04 says "`apr parity` per cell against llama.cpp `d1d3c3396`". This premise is
**false at the released tag**: `apr parity` in v0.69.3 is a GPU-vs-CPU check
(`apr parity --help`: "GPU/CPU parity check"), and no released apr verb compares logits
against llama.cpp. The admission schema therefore records the oracle explicitly
(`parity.oracle`, e.g. `llama.cpp@d1d3c3396` or `apr-parity-gpu-cpu`), with a cosine, a
threshold and a cited `threshold_basis` (R-7). The llama.cpp oracle binary must be a
declared executor input like the weights (added to infra#1088).

## Falsifiers, planted RED → restored GREEN (`admission.rs`, restored from a sha-checked copy)

| Falsifier | Plant | Planted | Restored |
|---|---|---|---|
| RCA-001 a silent cell rejects the file | silent-cell arm made a no-op | rc=101 `admission_tests.rs:80` "a silent cell must reject the file" | 1 passed |
| RCA-002 Admitted needs a passing parity receipt | drop `cosine >= threshold` | rc=101 `admission_tests.rs:106` "below threshold" | 1 passed |
| RCA-003 no admitted cell raises S-7 | `s7 = false` | rc=101 `admission_tests.rs:122` "all NotRun is S-7" | 1 passed |

Source was byte-identical after restore (`restored-ok`). 52/52 lib tests pass; clippy
`-D warnings` is clean on lib, examples and tests; `pv validate` reports 0 errors.

## S-7 → STOP

§8 S-7: "Every cell is `Refused` or `NotRun`, so there is no admissible cell for (A)". This
is a STOP: write the §9 report and do not work around it. REX-05..REX-12 all need a running
cell or the gx10 declaration, so none proceeds. **Resume condition:** infra#1088 lands at
least one declared executor. Then re-run `rex admit` for that cell with a parity receipt,
and `rex admission-check` exits 0.

## Operator ruling recorded here (the prereg is frozen)

S-6 was resolved by the operator (verbatim, 2026-09-25, relayed by aprender-77 as
"OPERATOR, verbatim"): **"yes it can train on claude/agy outputs"**. B2 may train on silver
labels (Claude/agy outputs) alongside gold labels and the 27B teacher. Silver and gold are
kept in separate files, and the label source is recorded per row.
