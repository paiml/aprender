# PMAT-4509 implementation receipt

Head base: origin/car/0.70.0 @ 1274d040de. Measured 2026-09-27T07:15:02Z in this worktree.

## `scripts/release/rc_fleet_stage.sh --self-test` → rc=0

```
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

## `scripts/release/rc_cut.sh --self-test` → rc=0

```
  ok   merge-base found
  ok   cherry-pick is carried by patch-id
  ok   squash-merged fix is carried by content
  ok   a fix absent from cand is MISSING
  ok   subject drop row drops the bump
  ok   sha-prefix drop row drops it
  ok   deletion-only commit NOT applied on cand is MISSING
  ok   short-line commit is 'empty', not silently carried
carry-forward v0.69.5-rc.1 -> main (base 9745278c4): 2 MISSING, 2 carried, 2 dropped, 1 empty
  MISSING 488b772fa  perf: lost (#4273)  [content 0.00 of 1]
  dropped bd95d3ce5  release: bump to 0.69.5  [version bumps are per line]
  dropped 87b7e06e1  fix: superseded  [superseded by car's rewrite, ruled 2026-09-26]
  MISSING c8e4ed8f2  refactor: delete the original line  [content 0.00 of 1]
  empty   1c129aa46  tiny  [no judgeable line]
  ok   the gate refuses while anything is MISSING
  ok   without drops, both dropped rows are MISSING
carry-forward v0.69.5-rc.1 -> main (base 9745278c4): 4 carried, 2 dropped, 1 empty
  dropped bd95d3ce5  release: bump to 0.69.5  [version bumps are per line]
  dropped 87b7e06e1  fix: superseded  [superseded by car's rewrite, ruled 2026-09-26]
  empty   1c129aa46  tiny  [no judgeable line]
  ok   all carried or dropped -> exit 0
  ok   full history: range is S only, carried
  ok   a boundary on base's ancestry is ENV, never judged
  ok   malformed drops row is ENV, never a pass
  ok   unreadable cand is ENV
```

## `scripts/release/carry_forward_gate.py --self-test` → rc=0

```
  ok   merge-base found
  ok   cherry-pick is carried by patch-id
  ok   squash-merged fix is carried by content
  ok   a fix absent from cand is MISSING
  ok   subject drop row drops the bump
  ok   sha-prefix drop row drops it
  ok   deletion-only commit NOT applied on cand is MISSING
  ok   short-line commit is 'empty', not silently carried
carry-forward v0.69.5-rc.1 -> main (base 274427b33): 2 MISSING, 2 carried, 2 dropped, 1 empty
  MISSING 8fde565ad  perf: lost (#4273)  [content 0.00 of 1]
  dropped 204be241d  release: bump to 0.69.5  [version bumps are per line]
  dropped 1ce45abfd  fix: superseded  [superseded by car's rewrite, ruled 2026-09-26]
  MISSING 39aea7bf0  refactor: delete the original line  [content 0.00 of 1]
  empty   3f2fb51fa  tiny  [no judgeable line]
  ok   the gate refuses while anything is MISSING
  ok   without drops, both dropped rows are MISSING
carry-forward v0.69.5-rc.1 -> main (base 274427b33): 4 carried, 2 dropped, 1 empty
  dropped 204be241d  release: bump to 0.69.5  [version bumps are per line]
  dropped 1ce45abfd  fix: superseded  [superseded by car's rewrite, ruled 2026-09-26]
  empty   3f2fb51fa  tiny  [no judgeable line]
  ok   all carried or dropped -> exit 0
  ok   full history: range is S only, carried
  ok   a boundary on base's ancestry is ENV, never judged
  ok   malformed drops row is ENV, never a pass
  ok   unreadable cand is ENV
```

## `scripts/release/decode_floor.py --self-test` → rc=0

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

roadmap.yaml = car's file + the PMAT-4509 block only (21 lines, appended; no re-sort). `aggregate --check` was already RED on car before this branch; re-sorting is left to the car owner.
