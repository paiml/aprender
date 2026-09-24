# Cop ruling on the PMAT-4113 quorum round (2026-09-24)

- **Round judged head:** `0891dfcc3`
- **Lanes:** gemini-3.1-pro-high, gemini-3.8-flash-high and gemini-3.7-flash-high all PASS. The models were measured, not just declared.
- **Artifact:** `docs/audits/quorum-PMAT-4113.json`, with `agreed: true` and receipt-lint ok.
- **Why the script said "NOT AGREED":** `quorum-review.sh` printed "NOT AGREED" (and posted it, on #4023) because its status line reads lane-reduce's rc. That rc was non-zero from one partial_reason: agy's own `.err` narration from lane 1.
- **Ruling:** release cop aprender-cf ruled that the round **COUNTS**. The only partial reason is agy's narration, a known rail false-partial (precedent #4110/#4114, fixed by paiml-implement PR #356).
- **Status:** unarmed; this branch goes into the batch.

This is an evidence-only commit on top of the judged head. It changes no code.
