# PMAT-4076 — ONT-7 `valid_under`: implementation receipt

## Identity

| | |
|---|---|
| Ticket | PMAT-4076 (derived from paiml/aprender#4076 with `pmat work add --github-issue`), `kind:code` |
| Row | paiml/infra `docs/specifications/paiml-ontology.md` v4.12 @948ae923, ONT-7 (:656), probe (:657) |
| Branch | `PMAT-4076-ont7-valid-under`, off `origin/main` 49fe19c28 |
| Reviewed head | `e6cf9f10b` |
| Assigned by | cop aprender-cf (release-gate item: the operator ruled the ontology spec must be finished in 0.70) |
| Author | aprender-98, claude-opus-5-5 (model-gate: admit, basis=file) |
| Verdict | **DONE at the receipt.** Batching is the default: not armed, no PR opened; the cop folds |

## What landed

- `pv lint --gate valid-under` and **gate 13** of every `pv lint` run (R-8: computed everywhere, armed per repo;
  not armed here). Module: `crates/aprender-contracts/src/lint/valid_under_gate.rs`.
- Schema, the author's design, since the spec pins none: `metadata.valid_under{world, toolchain, host_class,
  backend, features}`. `world` indexes Σ `worlds:`; omitted, it reads `committed`, per Σ's own doc line. So the
  spec's Appendix B example is admitted as written. infra-83 confirms Appendix B's top-level
  `metadata_valid_under:` is a spec typo for §4.2; the fix is due in spec v4.14.
- Rules: PV-ONT-013 (shape, empty, closed keys), PV-ONT-014 (world), PV-ONT-015 (qualifiers), PV-ONT-016 (the
  shrink-only `contracts_without_valid_under` ratchet, TOP-LEVEL in `contracts/lint-baseline.json`, where the
  probe reads it). No Σ → decline; malformed Σ → error; no kernel contract AND no `valid_under` → decline.
- First real witness: `contracts/ont-verdict-lattice-v1.yaml` carries `valid_under: {world: committed}`.
- Gate contract: `contracts/ont-valid-under-v1.yaml` (6 falsification tests).
- `scripts/check_ont_ratchet.sh --write` now carries the new key verbatim. Before this change it would have
  deleted it.

## The probe, on the real corpus at the reviewed head

`contracts_without_valid_under` = 386, non-null. `pv lint contracts/ --gate valid-under --format json` →
`verdict: Pass`: 387 kernel-kind (non-registry) contracts, 1 carrying `valid_under`, `by_world: committed=1`.
Lane 3 re-derived 387 − 1 = 386 from the census.

## Verification (author-run, then judged by the lanes from the record)

| Check | Result |
|---|---|
| `make gate` at e6cf9f10b | exit 0, gate-reduce sha256 `3887b52c…` |
| aprender-contracts `--lib` | 1717 passed, 0 failed (round 2); +1 test in round 3 |
| aprender-contracts-cli, all 21 test binaries | 0 failing after the ont6 not_armed list fix |
| aprender-core `--test readme_contract` | 15/0 |
| clippy `-D warnings`, fmt | clean |
| guards | check_ont_ratchet, check_complexity_ratchet (round 1 RED at cognitive 40, split, then PASS), check_tree_reader_tests, check_explicit_test_commands, check_readme_claims, readme_sync, `pv extract --check` |

**MUST-RED, measured.** Each mutant was planted on the committed tree and restored from it, and each FAILS its
witness:
- M1: world lookup removed.
- M2: ratchet off.
- M3: zero-kernel decline removed.
- M4: zero-kernel return drops findings.
- M5: omitted world rejects.
- M6: gate 13 not pushed into the full run.
- M7: gate 13's validation branch inverted.
- The ratchet script's key carry-over, disabled: 2 self-test rows FAIL.

## Quorum — `docs/audits/quorum-PMAT-4076.json`

Claude Code lanes, claude-sonnet-5 (measured from each run's `modelUsage`), author claude-opus-5-5, so
**degraded: same-family**. The operator's words: "no wait, fall back fast". `receipt-lint` refuses a
same-family artifact by design, so this record is for the cop's ruling, not for `pmat-merge`.

| Round | Head | Result |
|---|---|---|
| 1 | e27e782a5 | FAIL: R-8, Appendix B rejected, no gate contract, zero-kernel hid findings, no `--lib` witness. All fixed |
| 2 | 3e60721f2 | PASS, PASS, lane 3 no verdict (account session limit). The notes were fixed in e6cf9f10b |
| 3 | **e6cf9f10b** | **PASS ×3 — agreed** |

## Open, non-blocking (round 3), for the fold or a follow-up

- Σ `readers.worlds` still lists only `ontology/sigma.rs`; `lint/valid_under_gate.rs` now reads `worlds` too.
  All three lanes noted it. It is a one-line change to `contracts/ontology.yaml`, and it changes the Σ
  checksum, so it was not made on a reviewed head.
- VU-INV-003's `formal:` is a tautology. The rule itself is enforced (PV-ONT-013 on `{}`); the formula should
  read `len(keys(valid_under(c))) ≥ 1`.
- Disclosed design, kept: a non-integer baseline value reads as "no baseline", and `make ont-ratchet` carries
  the key rather than re-measuring it. Both follow the formal_prose precedent.
- Not run: the plan-grill quorum the skill prescribes for a spec section (Q2). The design was instead judged
  inside the three review rounds.
