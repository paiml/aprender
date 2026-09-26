# REX-03 receipt — harness, scorer, `review-experiment-receipt-v1` (PMAT-4358, epic #4354)

prereg_sha `ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0` (unchanged; `stats.rs`
is frozen and was not edited — the scorer calls it). Corpus `review-corpus-v1@787d2026256cc08b`.

## What landed (crate `aprender-review-experiment`)

| Module | Role |
|---|---|
| `receipt.rs` | The row schema (`deny_unknown_fields`), the `VERDICT:` parser (§2.3), localization, admissibility |
| `score.rs` | §2.3 metrics with Wilson CIs over the **expected** item set; H1/H2 divergence, H3 McNemar input, H4/H5 pairwise-parsed pairs |
| `harness.rs` | `/v1/chat/completions` client (ureq), fixed prompt composition, seeded run order, 10 % re-run subset, load snapshot, receipt construction |
| `examples/rex.rs` | `review`, `not-run`, `score` (with `--pubkey` → `minisign -V`, otherwise labelled `unsigned (exploratory)`) |
| `contracts/review-experiment-receipt-v1.yaml` | FALSIFY-RXR-001..003; `pv validate` → 0 errors |

## Pre-registered choices made here `[A]` (the spec left them open; fixed before any model run)

- **Decoding:** temperature 0, seed 4354, `max_tokens` 512.
- **Request:** the prompt file verbatim, a blank line, then the diff in a ```diff fence. It is never
  truncated; a 4xx that names the context becomes `NotRun{ContextOverflow}`. The sha of the exact
  body is recorded (`request_sha256`).
- **Verdict:** the first `VERDICT:` line after the last `</think>`. Markdown decoration is ignored.
  The first word must be exactly `PASS`/`FAIL`, and a line naming both is `Unparsed` (an echo of
  the prompt). The table has 12 cases in `receipt_tests.rs`.
- **Localization:** a finding after the verdict names the defect file by its full path or by its
  last two components. A bare file name does not count (`mod.rs`).
- **Scoring denominator:** the expected item set. An item without exactly one admissible row
  (raw output present at the recorded sha, and re-parsing to the recorded verdict) is
  `NotRun{Inadmissible}`. Dropping a row can therefore never raise a score.
- **Hosted arms** (Haiku, agy) record `hosted` for the apr/weights shas plus their exact model id.
- **A row whose serve reply has no `usage`** is `NotRun{ServeError}`; it cannot be admitted as a run.
- **Nullable by design:** `timings.server` (TTFT/tok/s is then `NotRun{NoSensor}`, never 0),
  `load.co_running` and `load.peak_anon_bytes` (executor-reported, infra#1088).

## Falsifiers, planted RED → restored GREEN (mutations in `receipt.rs`, restored from a sha-checked copy)

| Falsifier | Plant | Planted | Restored |
|---|---|---|---|
| RXR-001 missing model sha → inadmissible | `#[serde(default)]` on `weights_sha256` + drop its hex check | rc=101 "missing weights sha must be inadmissible" | 1 passed |
| RXR-002 NotRun counted as correct → rejected | `NotRun(_) => true` in `Verdict::correct` | rc=101 "NotRun counted as correct", left (10,10) right (0,10) | 1 passed |
| RXR-003 Unparsed counted as PASS → rejected | `Unparsed => !defect` | rc=101 "Unparsed counted as PASS on a good item", left (2,3) right (0,3) | 1 passed |

Scorer fixtures are hand-computed (`score_tests.rs`):
- 6 defects / 4 good → recall 3/6 with Wilson (0.18762, 0.81238) by hand; precision 3/4;
  false-refute 1/4; localization 2/3; parse 6/8.
- McNemar b=5, c=1 → p = 7/64.
- `rfc3339` checked against `date -u`.
- A loopback HTTP test drives `post` end to end.

Totals: 47/47 lib tests; clippy `-D warnings` clean on lib, examples and tests.

## CLI smoke on the real corpus (no serve, no GPU)

- `rex not-run --why NoDeclaredExecutor --split test` (C2, apr-4b, v0.69.3 `~/.cargo/bin/apr`
  sha) wrote 105 rows.
- `rex score` gave: items 105, recall 0/70 (the NotRun items stay in the denominator),
  correct 0/105, `not_run {NoDeclaredExecutor: 105}`, rejected rows 0.
- Signed with a throwaway minisign key: `provenance: signed`.
- After a one-byte edit to the receipts file: `signature does not verify` → exit 1.

## Not done here, by design

- No model run: cells need forjar-declared executors (R-4, infra#1088). REX-05/06 run through
  `rex review` on those executors.
- Receipt push/transport is executor-side (infra#1088).
