---
name: read-response-when-findings-are-labels
description: agy lanes sometimes emit axis-label findings with no detail — the substance is in the raw `response` field, so always diff findings against response before summarizing
metadata:
  type: feedback
---

When a lane's `structured_output.findings[].claim` reads like a heading ("JSON schema
consistency", "Receipt mutations") rather than an assertion, the real review is in the
lane's top-level `response` string. Read `jq -r '.response'` for that lane before writing
the receipt.

**Why:** on PMAT-989 (width 3, `--mode plan`) lane 3 filled `findings` with the brief's
own question labels and put every verdict, citation and required change in `response`.
Summarizing from `findings` alone would have reported "14 findings, no content" and lost
two of the three dissents against the other lanes.

**How to apply:** after collecting lanes, for each one check whether `claim` strings
assert something. If any lane's claims are label-shaped, pull its `response` and
reconcile. Also: lanes disagreeing on the same file:line (schema strictness, a YAML
duplicate, whether a decomposition is behaviour-preserving) is the normal case — name
each dissent with its lane index verbatim, since the orchestrator re-runs the checks.
See [[lanes-need-a-cd-wrapper-script]].
