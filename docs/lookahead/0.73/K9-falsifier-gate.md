# K9: which gate fails a contract that has too few falsifiers?

Look-ahead draft for 0.73 (#3999), 2026-10-03. Measured on main @316dee2cd4 with the installed
pv 0.70.0 (2add1c644). 2add1c644 is an ancestor of 316dee2cd4, and no commit between them touches
`crates/aprender-contracts` or `crates/aprender-contracts-cli`, so this pv runs main's contract
rules. The numbers are dated samples. Most come from the two commands below; the others name
their command where they appear. Where a number disagrees with its command, the command wins.

```
python3 docs/lookahead/0.73/falsifier_gate.py                      # the case table in §5; exit 1 if a cell changes
python3 docs/lookahead/0.73/falsifier_gate.py --corpus 316dee2cd4  # the numbers for main's contracts/
```

## 1. The question

K9 (2026-09-27) found that `pv validate` and plain `pv lint` pass a contract with zero
falsification_tests. It proposed checking the 0.73 drafts with `pv lint --min-score 0.6`. This
recheck finds that the first half holds for every kind of contract except kernel, and that the
second half does not work: a score floor is not a falsifier gate. It also gives a gate that does
work today, using only pv (§4).

## 2. Where the decision is made

Both decision surfaces run plain `pv lint contracts/`.

| Surface | Command | Where (@316dee2cd4) |
|---|---|---|
| CI, step "pv lint contracts/" | `"$PV" lint contracts/` | `ci/sections.yml:3714-3717` |
| Release, `make contracts`, step_lint | `"$PV" lint contracts/` | `scripts/contracts_gate.sh:43` |

Neither passes `--min-score`, and `.pv.toml` sets no score floor. So lint's score gate runs at
threshold 0.00. On an extract of main's tree (`git archive 316dee2cd4 contracts .pv.toml`),
`pv lint contracts/` reports that gate as passing: "(1889 contracts, mean=0.48, threshold=0.00)".
The release script's other lint calls run named gates only (`--gate shapes` at :81;
`ont-consistency`, `refines` and `bindings` at :96).

## 3. Findings

**F-K9-1. The count rule covers kernel contracts only.** PROVABILITY-001 fails a contract that has
no proof_obligations, no falsification_tests or no kani_harnesses, or fewer falsification_tests
than proof_obligations (`crates/aprender-contracts/src/schema/types.rs:353`, count clause at
:367). It is an error in validate (`schema/validator.rs:475-486`), and lint reports it as
PV-PRV-001 (`lint/gates.rs:290`). It runs only when `requires_proofs()`, which is
`kind() == Kernel` (`schema/types.rs:324-327`). A contract with no `kind` is a kernel
(`schema/kind.rs:54-56`), but `kind()` turns a kernel with `registry: true` into Registry
(`schema/types.rs:316-322`). Every other kind passes with no falsifiers at all.

- Case table: the two schema drafts (BPM, WGF) with their falsifiers stripped pass validate,
  lint and `lint --strict`. The two kernel drafts (CPU, NEON) stripped fail all three.
- Main: 123 of the 907 contracts that declare proof_obligations have fewer falsification_tests
  than obligations, and 19 of them have none. By their metadata, 92 have no kind and
  `registry: true`, 7 have `kind: kernel` and `registry: true`, 18 are `kind: pattern`,
  3 `beat-benchmark`, 2 `training-loop`, and 1 `pretraining-corpus` with `registry: true`. None
  resolves to kernel. Lint's validate gate passes the tree with no PV-PRV-001 finding.

**F-K9-2. pv names INERT contracts and fails none of them.** The 19 contracts in F-K9-1 that
have no falsification_tests all keep their falsifiers in a legacy top-level block
(`falsification:` or `falsification_conditions:`). pv counts those entries
(`legacy_falsification_entries()`, `schema/types.rs:338`) but does not read them as tests.
Outside a unit test, the only caller of that count is `pv status`
(`crates/aprender-contracts-cli/src/commands/status.rs:24`). For each of the 19 it says the block
is one "which NO pv gate enforces — this contract is INERT: it reads as enforced and enforces
nothing". On main, 414 YAML files carry such a block. 411 are INERT (19 with proof_obligations
and 392 without), and 3 have falsification_tests beside the unread entries. No lint rule reads
the count, and lint's output on the extract in §2 has no finding about a legacy block or INERT.

**F-K9-3. `--min-score 0.6` is not a falsifier gate.** It fails NEON as drafted (composite 0.508,
with 10 falsification_tests for 9 obligations), and it fails every stripped copy (0.258 to
0.412). So it does not separate an honest draft from a planted one. On main, 1185 of 1889
contracts (62.7%) score below 0.6, so the floor would fail the tree as it stands.

**F-K9-4. A score floor rewards deleting obligations.** NEON cut back to its first five
obligations (the count it had on 2026-09-27), with the same 10 falsification_tests and
3 Kani harnesses, scores 0.575, up from 0.508. D3 (Kani) is the summed harness weight over the
obligation count (`scoring/mod.rs:330-331`), so the four obligations added since then cut D3 from 0.60 to 0.33. A gate on
the composite passes a contract more easily the less it promises.

**F-K9-5. D2 counts falsifiers; it does not match them to obligations.** D2 is
min(tests, obligations) / obligations (`scoring/mod.rs:260-262`). For a contract with no
obligations it is 0.0 without tests and 1.0 with any (:252-257). For each obligation, pv also
looks for a falsification test whose `rule` equals the obligation's `property` (:222, "(no test)"
at :231), but that probe does not enter the score. On main, 680 of 3888 obligations (17.5%) have
such a test. In the four drafts, 0 of 39 do (`falsifier_gate.py` prints the figure). So D2 = 1.00 means "at least as many tests as
obligations", not "each obligation has a test".

## 4. The gate that works today

```
pv score DIR --weights '{"spec_depth":0,"falsification":1,"kani":0,"lean":0,"binding":0}' --min-score 1.0 --exit-code
```

With all the weight on D2, the composite is D2, and `--exit-code` exits 1 when any contract scores
below `--min-score` (`pv score --help`). At 1.0 it fails any contract with fewer
falsification_tests than proof_obligations, whatever its kind. It uses only pv. It has two limits:

- It counts and does not match (F-K9-5).
- It also fails a contract that declares neither obligations nor tests. On main's tree it would
  fail 866 rows: the 123 of F-K9-1, and 743 that declare neither.

So it is a gate for a directory of new contracts, such as `contracts-draft/`, not for the tree.
All four drafts pass it today.

## 5. Case table

Exit codes from `falsifier_gate.py`. The script fails if one changes. The composite uses pv's
default weights and is a dated sample (2026-10-03). BPM and WGF are `kind: schema`. NEON is
`kind: kernel`, and CPU declares no kind, so it is a kernel too (F-K9-1).

| Case | validate | lint | lint --strict | lint --min-score 0.6 | gate (§4) | composite |
|---|---|---|---|---|---|---|
| BPM as drafted | 0 | 0 | 0 | 0 | 0 | 0.650 |
| CPU as drafted | 0 | 0 | 0 | 0 | 0 | 0.662 |
| NEON as drafted | 0 | 0 | 0 | **1** | 0 | 0.508 |
| WGF as drafted | 0 | 0 | 0 | 0 | 0 | 0.650 |
| BPM, falsifiers stripped | 0 | 0 | 0 | 1 | **1** | 0.400 |
| WGF, falsifiers stripped | 0 | 0 | 0 | 1 | **1** | 0.400 |
| CPU, falsifiers stripped | 1 | 1 | 1 | 1 | 1 | 0.412 |
| NEON, falsifiers stripped | 1 | 1 | 1 | 1 | 1 | 0.258 |
| The four drafts and stripped BPM | 0 | 0 | 0 | 1 | **1** | as above |
| NEON cut to its first 5 obligations | 0 | 0 | 0 | 1 | 0 | 0.575 |

- The BPM and WGF stripped rows are the K9 hole. A schema contract with no falsifiers passes
  validate and lint, and only the gate fails it.
- The NEON as-drafted row is why the 0.6 floor is wrong: it fails an honest draft.
- The mixed row shows the gate failing one bad file among good ones while lint passes the set.
- The script also checks that the cut copy scores higher than the draft (F-K9-4).
- Both modes were seen to fail once: the case table with one expected exit code changed, and
  corpus mode with a stripped kernel copy planted in main's tree. Each exited 1. Neither plant
  is kept in the script.

## 6. Proposed pv change

This is proposed as one pv ticket. It is not filed, because the look-ahead mints no tickets.

1. Apply the count clause of PROVABILITY-001 to every contract that declares proof_obligations,
   whatever its kind. Make it a ratchet: the 123 that fail it on main today are baselined, and a
   new one fails.
2. Make the INERT signature a lint error, ratcheted the same way (411 on main today).
3. Score D2 from the per-obligation probe, or drop the probe, so that a reader of the score does
   not take it as matched.

Side note: the doc comment at `scoring/mod.rs:23-28` gives D1 20% and D4 10%, but the defaults in
`scoring/types.rs:20-24` are spec_depth 0.25 and lean 0.05. The code is what runs.

## 7. For the 0.73 drafts

- Check the drafts with the gate in §4, not with `--min-score 0.6`, which fails NEON for
  stating more obligations.
- Until change 1 lands, no gate in CI or the release holds the falsifiers of the two schema
  drafts once they move to `contracts/`. Only review does, or this gate run by hand.
- This supersedes the K9 bullet in `R3-neon-q4k-q6k.md` §10.
