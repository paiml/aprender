# Driver kaizen — one line per prompt defect met

- 2026-09-06: a capped paiml-impl-worker keeps its slot lock until it is told to stop (the hook proves pid liveness against the HARNESS pid, which never dies mid-session); every orchestrator push/PR was refused until a SendMessage asked the worker for its receipt — the hook needs SubagentStop-on-cap, or the cap must count as a stop
- 2026-09-06: I12 `gh pr checks <n> --watch --fail-fast` exits immediately whenever an advisory check (present / pr-review-receipt) is failing, so it cannot serve as the blocking wait; a bounded until-loop on the run's or the guard job's conclusion is what works
- 2026-09-06: the delegate ran every N-lane and rescope quorum on ONE model family (Gemini 3.1 Pro ×3) although the driver requires three families; the brief must name the families and the delegate must refuse otherwise
- 2026-09-06: v4/v5 'nothing is written after a merge' met G-10a's receipt, which was partial when #3011 merged under v3; flipped by the docs commit once, applied from G-11a on
- 2026-09-06: the driver's G-10b baseline '281' is the pre-PR-A count; measured on the PR-A tip it is 243 — baselines are measured at the commit named, never carried from a prompt
- 2026-09-06: the L0-1 report was named 'verbatim' but was not in-tree; v5.1 named #2971 — the ticket IS the report; my #3017 was closed as its duplicate
- 2026-09-06: I13's write set does not name evidence/<row>/ (row records) or .pr/<row>/accept.sh; G-11b gitignores .pr/** except accept.sh and treats row evidence as row-owned
- 2026-09-06: C3 (--backend on every surface) moved to 0.67 while R-0b (the resolution) stays in 0.66: the rescope quorum read that as scope-fails until the spec text was re-read; the criteria table now says why
- 2026-09-06: the claims ratchet turned RED on the rescope record itself (it quoted the literals it was cataloguing); records cite file:line, never repeat a number
- 2026-09-06: the CI job stops at its first failing step, so one mutant commit with three guards needed three RED runs (one per guard) — one mutant per PR-run, or the case tables in separate jobs
- 2026-09-06: bashrs' multi-file ratchet counts a branch's INHERITED lint debt (G-11b based on G-11's pre-fix commit): re-cut onto main before judging the count
- 2026-09-06: apr-agent --help creates a worktree named --help (no flag parsing before the slug); removed by hand; a slug guard for leading dashes belongs in the launcher
- 2026-09-06: a Bash call refused by the subagent-lock hook aborts the WHOLE command line, including the file writes before the gh call — writes and gh calls in separate calls
