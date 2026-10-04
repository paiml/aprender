# REX-02 receipt — review corpus v1 (PMAT-4357, epic #4354)

prereg_sha `ef51087dc79bab0ad160e8a14f5b13e2ea43986b30c05dafc63c84c2dc21cdc0` (unchanged; `rex prereg-check` OK).

- **Corpus version:** `review-corpus-v1@787d2026256cc08b`
- **Sealed manifest:** `test-manifest-v1.txt`, sha256 `787d2026256cc08b2b219c2f315f3ada5ceeca5e5b47ef93081c95e7f0fa1b2b` (105 test items)
- **Item metadata:** `corpus-v1.jsonl`, sha256 `7acf0ef708021a62b55f03393f47ffdee26303a0902ee33bbc6753d3869f37f5`
- **Diffs (outside git):** `/mnt/nvme-raid0/rex-corpus/v1/items-v1.tar` on lambda-vector, sha256
  `70dffe5625e1849661303b82bb1fdf5254ab1d4560e2fe3eacc147e5d2588bba`. Deterministic tar (sorted,
  fixed mtime/owner); hosts get it through the infra#1088 declaration.
- **Built by:**
  `rex corpus-build prs.json 2026-09-11T00:00:00Z <items> <4 mutant lists>` at base `80437647e`.
  Inputs:
  - `gh pr list --state merged --limit 1500` snapshot (1468 PRs, sha256 `e185a34fa87f1ee1…`, copied to
    `/mnt/nvme-raid0/rex-corpus/v1/prs-snapshot.json`);
  - `cargo mutants --list --diff --json` for aprender-core, aprender-serve, apr-cli and
    aprender-contracts (76 264 mutants, concatenated sha256 `80e6f2773c0a6127…`). Listed only; nothing was
    compiled or run.

## Counts (class × stratum × split)

| Class | S dev/test | M dev/test | L dev/test | Total |
|---|---|---|---|---|
| P planted | 15 / 35 | — | — | 50 |
| R real (revert-the-fix) | 7 / 15 | 6 / 14 | 2 / 6 | 50 |
| G good | 5 / 12 | 5 / 12 | 5 / 11 | 50 |
| **Total** | 27 / 62 | 11 / 26 | 7 / 17 | **150 (45 dev / 105 test)** |

Test defect items (P ∪ R) = **70**. This matches the §2.2 sample-size basis. P is all S, because a
single mutant diff is small; the population does not allow balancing it (§2.2 "where the
population allows").

## Construction notes (all `[A]`, fixed before any measurement)

1. **P source.** No catalogue of 50 committed planted-defect fixtures exists: 9 registered
   MUTATION-POINTs and about 47 sed plants are spread across guard scripts. So P is the fleet's
   mutation tool (cargo-mutants 27.0.0, the CI `mutants` gate) listing mutants as diffs. The
   sample is seeded (4354), one per file, on review paths only. The `+++ replace …` header and the
   `/* ~ changed by cargo-mutants ~ */` marker are stripped (FALSIFY-RCV-001); 0 of 150 item
   files contain either.
   - Caveat: some listed mutants may be equivalent (no behaviour change). This is reported as a
     label-noise bound, not corrected.
2. **R.** Merged PRs with a closing-issue link and a fix title. Only code hunks on review paths
   (.rs/.sh; not tests, benches or examples) are reverse-applied; comment-only hunks are dropped
   (FALSIFY-RCV-003). Defect label = the new-side lines of the reversed hunks.
3. **G.** Merged before 2026-09-11 (14 days before the build), not a fix, not named by a
   `Revert` PR, and no `regression` label. Green means `ci / gate` and `workspace-test`
   SUCCESS with no FAILURE/TIMED_OUT/ERROR in the rollup; this was checked per PR via gh.
4. **Size.** Strata use a bytes/4 token proxy (not a model tokenizer). Candidates over 32 000
   proxy tokens are dropped at construction. Nothing kept is ever truncated.
5. **Seal.** No test item was read by the author: the builder prints counts only, and the one
   dev item inspected was an R dev item.

## Contamination falsifier, planted on disk (RED → GREEN)

A test R item was copied (by script, unread) into
`docs/audits/review-corpus/train/leak.jsonl` as a JSONL `diff` field:

```
falsify_rcc_003: planted_rc=101  test result: FAILED  item: "R-pr1411"  how: "hunk"
```

Removed by trap → 29/29 lib tests pass.
