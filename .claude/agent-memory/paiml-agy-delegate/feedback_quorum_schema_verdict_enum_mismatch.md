---
name: quorum-schema-verdict-enum-mismatch
description: The quorum schema's verdict enum is PASS|FAIL|do-not-implement-as-written — briefs asking for design verdicts must carry an explicit mapping or lanes emit invalid JSON
metadata:
  type: feedback
---

`~/.claude/skills/paiml-implement/agy/quorum-schema.json` constrains `verdict` to exactly
`PASS | FAIL | do-not-implement-as-written`. Briefs regularly ask lanes for a *design*
verdict (`implement-as-written | implement-with-changes | do-not-implement-as-written`),
which does not fit. When that happens, inject an explicit mapping into every lane prompt
(implement-as-written→PASS, implement-with-changes→FAIL, third passes through) AND require
the lane to begin `summary` with a literal `DESIGN_VERDICT=<value> | ` token, so the real
verdict survives the lossy enum.

**Why:** without the mapping the lane either fails schema validation or silently picks a
value that collapses "ship it with fixes" and "the design is wrong" into one bucket — the
orchestrator then cannot tell a FAIL that means "3 changes" from a FAIL that means "stop".

**How to apply:** check the pinned schema's enum against the verdict vocabulary the brief's
prompt uses, before launching. Also: when a brief omits `mode` but pins `--schema`
explicitly, `--mode plan` is the identity wrapper (`agy-lane.sh` sets `FULL="$PROMPT"`,
adds no doctrine text) — it is the only mode that does not mutate a self-contained prompt.
Name the inference in the receipt's `open_questions` rather than bouncing the brief.
See [[agy-lane-calling-form]].

**Addendum (PMAT-1065):** the mapping instruction leaks. Given the mapping plus a
required `summary` prefix `ROOT_CAUSE_VERDICT=<root-cause-named|root-cause-hypothesis|
cannot-say> | `, lane 3 of 3 emitted `ROOT_CAUSE_VERDICT=FAIL` — it wrote the *mapped*
schema enum into the token meant to carry the *unmapped* verdict, collapsing exactly the
distinction the token exists to preserve. Lanes 1 and 2 obeyed. So the token is worth
keeping, but treat it as best-effort: when the prefix value is one of the schema enum
members rather than the brief's vocabulary, recover the real verdict from the lane's
prose and say in the receipt that the lane's self-label was unusable.
