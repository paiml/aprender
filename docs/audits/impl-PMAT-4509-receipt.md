# PMAT-4509 implementation receipt

Base: origin/car/0.70.0 @ 1274d040de, merged into this branch. Measured 2026-09-27T07:33:52Z in this worktree. Each block is the command's FULL output.

## Scope of the judged diff (read before judging)

This branch is a FOLD, not a single-file fix. Acceptance criterion 4 of `docs/roadmaps/entries/PMAT-4509.yaml` reads: "folded onto car/0.70.0 together with its base (e6 carry-forward gate da88d81d1: carry_forward_gate.py, decode_floor.py, rc_cut.sh wiring), whose self-tests also PASS". The #4509 fix (infra-64, 0c4f9b88f8) was authored ON TOP of e6's branch `e6/rc-carry-forward-gate` @ da88d81d1, and that base is not yet on car/0.70.0, so `car...HEAD` necessarily contains it:

| Files | Origin | Why it is here |
|---|---|---|
| `scripts/release/carry_forward_gate.py`, `carry-forward-drops.tsv`, `rc_cut.sh` (+12/-1) | e6 e0ec2d6dfd | the carry-forward gate base (AC 4); self-tests below, rc=0 |
| `scripts/release/decode_floor.py`, `scripts/perf-matrix.yaml`, and **98 lines of `scripts/release/rc_fleet_stage.sh`** (`DECODE_SH`, `decode_floor_row()`, their wiring into `stage()`/the verdict, and the matching self-test rows) | e6 da88d81d17 | the decode-floor base (AC 4): an rc is not published while it decodes slower than the previous line; self-tests below, rc=0 |
| the rest of `scripts/release/rc_fleet_stage.sh` (`FLEET_HOSTS`, `catalogue_suffix()`, `fleet_table()`, the catalogue self-test rows and the #4509 hand-copy mutant) | infra-64 0c4f9b88f8 + aprender-57 3cafab5852 | the #4509 fix proper (AC 1-3) |
| `docs/roadmaps/entries/PMAT-4509.yaml`, `roadmap.yaml` (+21) | aprender-57 | the ticket fragment; roadmap.yaml adds only the PMAT-4509 block over car |

Correction: commit 78c45809ef's subject says "the judged diff adds only PMAT-4509". That is true of `docs/roadmaps/roadmap.yaml` only (car's file + the PMAT-4509 block). The judged diff as a whole is the fold in the table above.

## `bash scripts/release/rc_fleet_stage.sh --self-test` → rc=0

```
rc_fleet_stage self-test: stage_verdict
  ok   every host passes -> publish
  ok   ONE host failing holds the rc
  ok   an unreachable host is RED, never skipped (#4328 C7)
  ok   a dated waiver through today covers an unreachable host
  ok   an expired waiver covers nothing
  ok   a waiver never covers a host that was reached and failed
  ok   no hosts is not N/N
  ok   an unknown state holds
  ok   mutant 'fail arm is a no-op' publishes past the bad host: the refusal is load-bearing
  ok   mutant 'unreachable counts as pass' publishes past the bad host: the refusal is load-bearing
  ok   catalogue -> intel stages -wgpu, mini stages the darwin pv, {arch} per host
  ok   a host no pv row covers -> pv "-" (none ships), not a guess
  ok   a host the catalogue ships no apr to -> refused, never a default
  ok   an asset not <bin>-{tag}-<suffix>.tar.gz -> refused
  ok   no catalogue -> refused (fail closed)
  ok   stage() without RC_FLEET_HOSTS_FILE reads the catalogue, and stops before the network
  ok   mutant (host match dropped) still yields a table
  ok   host-match mutant killed
  ok   mutant (#4509 hand copy: intel -cpu, mini no pv) still yields a table
  ok   #4509 hand-copy mutant killed: intel -cpu is not the catalogue's table
  ok   catalogue says intel -cpu -> the table says -cpu (no second list to disagree with)
rc_fleet_stage self-test: a fake fleet, end to end
  ok   every host installs and verifies -> the draft is published (rc 0, published=1)
  ok   ONE host whose PATH resolves an old apr -> NOT published (operator acceptance, #4327) (rc 1, published=0)
  ok   an unreachable host -> NOT published (#4328 C7) (rc 1, published=0)
  ok   the floor host absent from the fleet -> NOT published (never skipped) (rc 1, published=0)
  ok   an rc decoding 20x slower than the previous line -> NOT published (#4273) (rc 1, published=0)
  ok   mutant (floor row not in the verdict) publishes the slow rc: the row is what holds it (rc 0, published=1)
  ok   mutant (asset verify deleted) publishes past the lying host: the verify is what holds it (rc 0, published=1)
rc_fleet_stage self-test: PASS
```

## `bash scripts/release/rc_cut.sh --self-test` → rc=0

```
rc_cut self-test: rc_decide case table
  ok   first green run on a fresh release branch -> rc.1 (a red non-required job does not block)
  ok   rc.1 exists on an older commit -> rc.2
  ok   gap in numbering -> max+1, never reuse a name
  ok   numeric max over the API's LEXICAL order (rc.10 sorts before rc.9)
  ok   other versions, the final tag and near-miss names (dots are literal) do not count
  ok   re-run on an already-cut commit cuts nothing
  ok   main is not a release branch
  ok   suffixed release branch refused
  ok   two-part version refused
  ok   fork PR with a release/X.Y.Z head refused
  ok   red ci / gate cuts nothing
  ok   cancelled workspace-test cuts nothing
  ok   a missing required check is not green
  ok   every job of the name must pass
  ok   skipped is not success
  ok   superseded push cuts nothing
  ok   empty head sha never cuts
  ok   mutant (check loop deleted) cuts on a red gate: the red-gate row can see that defect
rc_cut self-test: PASS
carry_forward_gate self-test
  ok   prev tag = newest of the lower line, rc.10 > rc.9
  ok   a final outranks its own rcs
  ok   the candidate's own line never counts
  ok   a lower major.minor across a major
  ok   merge-base found
  ok   cherry-pick is carried by patch-id
  ok   squash-merged fix is carried by content
  ok   a fix absent from cand is MISSING
  ok   subject drop row drops the bump
  ok   sha-prefix drop row drops it
  ok   deletion-only commit NOT applied on cand is MISSING
  ok   short-line commit is 'empty', not silently carried
carry-forward v0.69.5-rc.1 -> main (base 5515ada29): 2 MISSING, 2 carried, 2 dropped, 1 empty
  MISSING 07f7378a3  perf: lost (#4273)  [content 0.00 of 1]
  dropped 9b8a0a77f  release: bump to 0.69.5  [version bumps are per line]
  dropped 4c9d37a30  fix: superseded  [superseded by car's rewrite, ruled 2026-09-26]
  MISSING ffc422a01  refactor: delete the original line  [content 0.00 of 1]
  empty   6e0af5cc3  tiny  [no judgeable line]
  ok   the gate refuses while anything is MISSING
  ok   without drops, both dropped rows are MISSING
carry-forward v0.69.5-rc.1 -> main (base 5515ada29): 4 carried, 2 dropped, 1 empty
  dropped 9b8a0a77f  release: bump to 0.69.5  [version bumps are per line]
  dropped 4c9d37a30  fix: superseded  [superseded by car's rewrite, ruled 2026-09-26]
  empty   6e0af5cc3  tiny  [no judgeable line]
  ok   all carried or dropped -> exit 0
  ok   full history: range is S only, carried
  ok   a boundary on base's ancestry is ENV, never judged
  ok   malformed drops row is ENV, never a pass
  ok   unreadable cand is ENV
```

## `python3 scripts/release/carry_forward_gate.py --self-test` → rc=0

```
carry_forward_gate self-test
  ok   prev tag = newest of the lower line, rc.10 > rc.9
  ok   a final outranks its own rcs
  ok   the candidate's own line never counts
  ok   a lower major.minor across a major
  ok   merge-base found
  ok   cherry-pick is carried by patch-id
  ok   squash-merged fix is carried by content
  ok   a fix absent from cand is MISSING
  ok   subject drop row drops the bump
  ok   sha-prefix drop row drops it
  ok   deletion-only commit NOT applied on cand is MISSING
  ok   short-line commit is 'empty', not silently carried
carry-forward v0.69.5-rc.1 -> main (base 4f392faa7): 2 MISSING, 2 carried, 2 dropped, 1 empty
  MISSING 2245f258b  perf: lost (#4273)  [content 0.00 of 1]
  dropped 8a28a6609  release: bump to 0.69.5  [version bumps are per line]
  dropped 4f79a69a6  fix: superseded  [superseded by car's rewrite, ruled 2026-09-26]
  MISSING 6ccdfb470  refactor: delete the original line  [content 0.00 of 1]
  empty   9e138ede7  tiny  [no judgeable line]
  ok   the gate refuses while anything is MISSING
  ok   without drops, both dropped rows are MISSING
carry-forward v0.69.5-rc.1 -> main (base 4f392faa7): 4 carried, 2 dropped, 1 empty
  dropped 8a28a6609  release: bump to 0.69.5  [version bumps are per line]
  dropped 4f79a69a6  fix: superseded  [superseded by car's rewrite, ruled 2026-09-26]
  empty   9e138ede7  tiny  [no judgeable line]
  ok   all carried or dropped -> exit 0
  ok   full history: range is S only, carried
  ok   a boundary on base's ancestry is ENV, never judged
  ok   malformed drops row is ENV, never a pass
  ok   unreadable cand is ENV
```

## `python3 scripts/release/decode_floor.py --self-test` → rc=0

```
  ok   equal speed passes -> pass
  ok   20x decode regression fails (#4273) -> fail
  ok   just under the floor fails -> fail
  ok   at the floor passes -> pass
  ok   median, not mean: one slow outlier rc run does not fail -> pass
  ok   rc on CPU is a fail, never compared -> fail
  ok   prev on CPU is unmeasured (the rc would win on a lie) -> unmeasured
  ok   rc receipt from another commit fails -> fail
  ok   rc receipt with unknown commit fails -> fail
  ok   no runs at all is unmeasured -> unmeasured
  ok   no prev runs is unmeasured -> unmeasured
  ok   a crashed run is unmeasured -> unmeasured
  ok   zero tok/s rc fails -> fail
  ok   0.70 rc measures against the newest 0.69 tag -> v0.69.5-rc.1
  ok   never the previous rc of its own line -> v0.69.5-rc.1
  ok   0.69 rc measures against 0.68 -> v0.68.2
  ok   nothing older -> none -> None
  ok   a non-version tag -> none -> None
  ok   matrix release_gates.decode_floor present, floor=0.9
  ok   a non-JSON bench line is kept as a crash, not dropped
  ok   mutant: the floor comparison is load-bearing (case '20x decode regression fails (#4273)' turns red)
  ok   mutant: the rc cuda check is load-bearing (case 'rc on CPU is a fail, never compared' turns red)
  ok   mutant: the build_commit check is load-bearing (case 'rc receipt from another commit fails' turns red)
  ok   mutant: median, not mean (case 'median, not mean: one slow outlier rc run does not fail' turns red)
```

## Roadmap

`git diff origin/car/0.70.0...HEAD -- docs/roadmaps/roadmap.yaml` adds exactly one entry, `- id: PMAT-4509` (21 lines, appended; no re-sort). `aggregate --check` was already RED on car before this branch; re-sorting is left to the car owner.
