# GH-4162: measured fleet token usage (Claude Code + agy) and what drives it

**Subject.** ONE file, `scripts/token_usage_report.py` (plus this receipt and its work entry). Reviewers: judge that script against #4162. Reading any other file is out of scope.

**What it does.** Read-only; prints markdown to stdout (`python3 scripts/token_usage_report.py [--hours N] [--json]`, about 20 s on lambda).
- **Claude:** every transcript under `~/.claude/projects/**/*.jsonl`, using the per-message `message.usage`.
  - Messages are de-duplicated by message id.
  - Usage is split by project (from the message's `cwd`), model, main turn vs subagent (`isSidechain` / `subagents/`, typed by `.meta.json` `agentType`), session and hour.
  - Also reported: image payload sizes and `compact_boundary` records (`preTokens`, `postTokens`).
- **agy:** every quorum lane + probe envelope on disk (agy's own `usage`, `duration_seconds`), agy's conversation files and 429 log lines. Repeat rounds are counted both from on-disk lane dirs and from committed receipts.
- **Privacy:** it prints counters and sizes only, never message content.

**Estimates, labelled as such** (E1–E3, F2). Each is arithmetic on the measured per-turn records.
- F2 replays each session's real per-turn context growth under a cap. Each forced compaction is charged at the MEASURED cost: the summarizer reads the whole context once, then the median summary and post-compaction baseline of the real compactions in the window.
- Stated limit: rework after a compaction can't be seen, so F2's NET is an upper bound.

**Measured (24 h, lambda, posted on #4162).**
- **Claude:** 12.9 B tokens, 98.5% of them cache reads.
- **The driver:** 66.7% of main turns run above 300k context and carry 87% of main-turn cache reads, and every top-10 session peaks around 966k.
- **E2 (poll turns):** 1.0%, which refutes polling as a driver.
- **F2:** a 200k cap would force about 300 compactions for about 69% net, 300k about 60%, 400k about 50% (upper bounds).

## Quorum round 1 (2 agy lanes, NOT AGREED): both findings real, both fixed

The artifact is kept in `docs/audits/quorum-GH-4162-r1/`.
- **No agy breakdown by model** (lanes 1 + 2). The ticket asks for "by project / model / lane / hour", and agy was reported only by total, project and hour.
  - Added **B2b**. The model is read per lane from agy's own `model="…"` log line beside the envelope, because the envelope names none.
  - 18 lanes have no log on disk (probes and older rounds) and are shown as `(unrecorded)` rather than guessed.
- **F2 counted the restart baseline B twice** (lanes 1 + 2): once in the compaction cost and again in `read_sim` on the next turn. Removed from the compaction cost.
  - Re-measured, 24 h: 200k → 303 compactions, net 69.4%; 300k → 59.6%; 400k → 50.1%; 600k → 32.3%. That's within 0.6 points of the posted figures.
