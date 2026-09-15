# PP-066 plan quorum record (STEP 4 of the driver; epic #2873)

**Subject:** `docs/audits/pp-066-plan.md` + `docs/specifications/pp-066-dag.yaml` (with spec v1.5 §0–§5 and the S0 ledger verdicts as the brief). **Schema:** `~/.claude/skills/paiml-implement/agy/quorum-schema.json` (verdict ∈ {PASS, FAIL, do-not-implement-as-written}; findings with grounding cited | measured | asserted). **Runner:** `agy 1.1.27`, `--sandbox --dangerously-skip-permissions --print-timeout 25m --output-format json --json-schema <schema>`, one process per lane, stdout and stderr to separate files (never `2>&1`); only `.structured_output` was read; a lane with no parseable structured output is **INDETERMINATE** and blocks the seat.

**Deviations from the driver's invocation, measured, not chosen:** (1) the brief is 204 KB and Linux refuses a single argv element above 131,072 bytes (`MAX_ARG_STRLEN`, measured in the pr-review skill §3.E), so the brief was mounted as a directory with `--add-dir` and the prompt carried the read-only clause plus a pointer — round 0 without `--add-dir` reviewed agy's own scratch directory and was discarded (lane-2, conversation `7f7996f1-f1b8-431b-b4bb-b5e03bb90037`); (2) the teamwork pass was invoked as `agy … -p='/teamwork-preview <clause>'` directly (the skill's `agy-lane.sh` does not pass `--add-dir`); (3) the GPT-OSS seat produced no usable review in two rounds (round 1: no structured output, asked what to do; round 2: a vacuous PASS with 0 findings whose summary describes another task), so the third seat was re-run on `gemini-3.8-flash-high` — a distinct model id (PRQ-22), same vendor family as lane 1; the family overlap is recorded here, not hidden. agy emits no field naming the model that answered; `model_id` is the requested id.

## Round 1 — brief = plan/DAG at `d95e19617` (69 rows)

| seat | model_id | conversation | verdict | findings |
|---|---|---|---|---|
| teamwork | (agy /teamwork-preview, default model) | `72a36f4b-c159-4e3f-bce7-5697ab150117` | do-not-implement-as-written | 10 (8 measured, 2 cited) |
| lane 1 | gemini-3.1-pro-high | `7911f9b2-3591-4631-8c08-08fd5d74d143` | do-not-implement-as-written | 11 (2 cited, 9 asserted) |
| lane 2 | gpt-oss-120b-medium | `03de5563-f055-41e9-b425-855ca664cc8c` | INDETERMINATE | no structured output |
| lane 3 | claude-opus-4-6-thinking | `88ae707a-6f56-48a0-a700-ad6a2632f623` | PASS | 26 (25 cited confirmations, 1 asserted) |

**Folded (cited/measured):** decisions as DAG blockers (DEC-D-9 → R-2, DEC-D-10 → R-5, DEC-D-2 → T-3, DEC-D-11 → every Track P row, DEC-D-3 → B-A1 with the expiry anchored on it); a `TAG-0.66.0` release-cut row blocked by C0-1..C0-4 so C0 precedes the tag structurally; P-1.2 as its own row (P-1.1 tightened 10-10 → 10-03 for slack); §6 rows unbatched (B-W1..W5, B-M1..M4, B-S1..S4, T67-1..T67-5); S0 premises a card named as blockers recorded as `[premises discharged: …]`; SPEC-1.6 given a RED test (`tests/spec/pp066_v16_defects.sh`); P-0.6's `--min-score 0.48` given `basis=` (measured mean at `b1a6324b8`, #2870); T-1 bundled in T-0's gx10 window (spec §5); R-0's A names FX-4b/REG-9/REG-10 on the real mixed host (S0-3); #2841 attached to B-M1. **Recorded, not applied (asserted):** "a --selftest cannot be turned RED by a data mutation" — the plan now states the RED-first test and the guard mutation are distinct artifacts; "C0-1's dry-run mutation cannot fire" — replaced by a live, reversible mutation with before/after API reads; "#2854 / #2863 missing from the DAG" — both are merged PRs, not issues; "R-0/R-2 miss S0 blockers" — S0 premises are discharged (ledger), recorded on the rows.

## Round 2 — brief = plan/DAG at `489275958` (86 rows, every #2873 sub-issue attached)

| seat | model_id | conversation | verdict | findings |
|---|---|---|---|---|
| teamwork | (agy /teamwork-preview) | `8a91ce01-d682-409a-aa6f-e98f797331cc` | do-not-implement-as-written | 6 (3 measured, 2 cited, 1 asserted: round-1 folds confirmed resolved) |
| lane 1 | gemini-3.1-pro-high | `7cf11124-0db6-488f-92c8-03149cca7ba0` | do-not-implement-as-written | 10 (8 measured, 1 cited, 1 asserted) |
| lane 2 | gpt-oss-120b-medium | `629a8fd1-28bb-4410-918c-26ed8026529e` | INDETERMINATE | vacuous PASS (0 findings; summary describes another task) |
| lane 3 | claude-opus-4-6-thinking | `d43af23f-d21c-4312-94cc-03084b0dbf83` | PASS | 26 (24 cited, 2 asserted) |
| lane 4 (replacement seat) | gemini-3.8-flash-high | `7a1f9b14-87c1-43ab-9407-56b8a0491e00` (agy `.status: ERROR` on a recovered retry; structured output present and schema-valid, so consulted) | PASS | 5 (all cited) |

**Folded after round 2 (cited/measured) — `c59ead5e2`:** T-3's master-8 blocker noted as LANDED (S0-2); R-0's A carries S0-21's stress-ng half; S-1/S-2/S-3 thresholds carry `basis=` (registered predictions: master §12 rows 19/20, report §10, and 0.55× of the comparator median in `evidence/parity-http/findings.json`); `$PV` spelled as `. scripts/pv_bin.sh && "$PV"`; R-5's blockers match the spec (R-2 dropped); D-2doc (S0-9 README defect) added; TAG-0.66.0 blocked by every 0.66 code row (49); G-6 and R-0 acceptance commands made exact; lane 3's TAG-blocker notes (P-1.2, T-0h, T-1, T-2, S-3, S-3g, I-16, I-17, S-2) resolved by the same fold. **Recorded, not applied (asserted):** "C0-1..4 must block every C-criterion producer" (lane 1) — spec §4 says C0 is *credited* first and other criteria are recorded `[U]` until it holds; it does not sequence work behind it (lane 3, cited, reads it the same way); "I-15/I-18 expiries are hard anchors, not 'derived'" (teamwork) — `{anchor, days}` is the derived form the DAG uses.

**Folded after lane 4 (cited) — this commit:** P-1.1's acceptance no longer asserts the 10 call sites resolve (that is P-1.2, same PR); C0-1's DAG mutation text matches the plan's live reversible mutation; D-1doc records its spec id D-1; S-1 records master rows 6 and 9 as discharged premises. Its P-1.2-in-TAG finding was already folded at `c59ead5e2` (the lane judged `489275958`).

## Verdict and dissent

Two seats in each round returned **do-not-implement-as-written**; every cited or measured finding they raised was folded before the next step, and their asserted findings are recorded above with the reason they were not applied. One seat (Claude via agy) returned PASS in both rounds with cited confirmations of each fold; the replacement Gemini 3.8 Flash seat returned PASS with five cited card-hygiene findings, all folded. The GPT-OSS seat is INDETERMINATE twice. Round-2 tally over the four seats that produced structured output: 2 do-not-implement-as-written (teamwork, Gemini 3.1 Pro — judged before the round-2 folds), 2 PASS. The driver caps the review at two rounds and names an aggregate do-not-implement-as-written verdict as a STOP; the operator decides whether the folded plan proceeds to STEP 5. **Recommendation:** proceed — the remaining dissent is one asserted reading of §4 that contradicts the spec's own wording, and the artifact checks that the lanes could not run (slack, cycles, queue order, 56/56 sub-issues attached) pass at `c59ead5e2`.

## Override record (STEP A2 of the driver, 2026-09-05)

```yaml
override:
  by: noahgift
  date: 2026-09-05
  dissent: "C0 as temporal precedence — agy teamwork + Gemini 3.1 Pro, rounds 1–2"
  disposition: proceed
  root_cause: "§4 'credited first' admits two readings → SPEC-1.6"
```

| seat | model_id (requested) | model_family | round 1 | round 2 | in M |
|---|---|---|---|---|---|
| teamwork | agy `/teamwork-preview`, no `--model` (agy's default; agy emits no field naming the model that answered) | [U] | do-not-implement-as-written | do-not-implement-as-written | yes |
| lane 1 | gemini-3.1-pro-high | google/gemini | do-not-implement-as-written | do-not-implement-as-written | yes |
| lane 2 | gpt-oss-120b-medium | openai/gpt-oss | INDETERMINATE | INDETERMINATE | **no** (INDETERMINATE in both rounds) |
| lane 3 | claude-opus-4-6-thinking | anthropic/claude | PASS | PASS | yes |
| lane 4 (replacement) | gemini-3.8-flash-high | google/gemini | — | PASS | yes |

M = {teamwork, lane 1, lane 3, lane 4}. Distinct model families with a known label in M: **2** (google/gemini, anthropic/claude); the teamwork seat's family is [U]. Fewer than 3 distinct known families remain → **`quorum: DEGRADED`** (recorded; no re-run, per the driver). The anthropic seat shares the author's vendor (the session model is claude-fable-5-1), so the cross-vendor property rests on the google seats alone.
