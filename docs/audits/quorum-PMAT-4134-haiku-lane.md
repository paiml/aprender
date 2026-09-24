# quorum-PMAT-4134: the Claude Code lane (operator rule 2026-09-24)

Operator, verbatim, relayed by the cop (aprender-cf) 2026-09-24: "Since we are running out of agy credits too often
over 5 hour window we need to not allow three agy quorum workers, one must always be a CHEAP claude code...perhaps
haiku". Every quorum is therefore 2 agy non-Claude lanes + 1 Claude Code lane on `claude-haiku-4-5`.

This round, all at head `bbdc7d70c`, was planned that way from the start. It is not a replacement lane, and the
round-1 history is in `docs/audits/history/`.
- `quorum-PMAT-4134.json`: `--width 2`, gemini-3.1-pro-high and gemini-3.8/3.7-flash (measured), **2/2 PASS,
  agreed**.
- The Claude Code lane, `claude-haiku-4-5` (Agent tool, model `haiku`), read-only, on the same head, diff, receipt
  and ticket: **PASS, no blocking findings**.
  - It measured the guard refusing (exit 2) on x86_64.
  - It confirmed each fix is cfg-gated with no allow, the self-test's trapped restore, the workflow edit as its own
    commit `0a2acd0d2`, and the committed gx10 logs (self-test RED on `unused_on_arm`, real run PASS, 0 findings).
  - It confirmed the receipt's apr-cli crate-wide-allow blind spot is accurate.

Net: 3/3 PASS, 2 non-Claude plus 1 claude-haiku-4-5 per the operator rule. The cop overrides receipt-lint's refusal
of the single haiku seat.
