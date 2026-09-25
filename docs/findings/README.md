# Findings ledger

A defect a session finds is recorded **here**, not as a new GitHub issue
(APR-EPIC-001 v1.3 intake control, `contracts/findings-ledger-v1.yaml`, #4455).
Only the cop mints issues, and only when the target epic has budget; the cop
reads this ledger when it mints.

## Writing a finding

Append ONE line of JSON per finding to `docs/findings/<YYYY-MM-DD>-<session>.jsonl`
and land it in the PR you are already working on (or the next one). Every key is
required, every value is a non-empty string, and no other keys are allowed:

| key | what it holds |
|-----|---------------|
| `id` | unique across every ledger file, `[A-Za-z0-9._-]+` — e.g. `FND-20260925-flow011-author` |
| `title` | one line: what is wrong |
| `evidence` | what you measured: the command's output, a CI run id, a log line |
| `repro` | the command that shows it again |
| `suspected_epic` | the epic it belongs under (`#NNNN` or an epic id); the cop decides |
| `severity` | `P0` \| `P1` \| `P2` \| `P3` |
| `found_at_sha` | the commit you measured at, 7–40 lowercase hex |

`bash scripts/check_findings_ledger.sh` validates every row (case table first,
then this directory) and is RED on a missing key, an unknown key, a duplicate
id, or an empty ledger.
