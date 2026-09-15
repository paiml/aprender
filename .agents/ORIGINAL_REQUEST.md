# Original User Request

## Initial Request — 2026-09-06T09:57:29Z

# Teamwork Project Prompt — Draft

> Status: Ready for launch — awaiting user approval
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: [none — teamwork routes from the description]

Review `docs/specifications/PP-066-release-spec.md` (checking for spec compliance, grammar/clarity, and code metrics/status), generate an audit report, and copy both the audit and the original report to the Desktop.

Working directory: /home/noah/src/aprender-worktrees/pp-066-spec
Integrity mode: development

## Requirements

### R1. Document Review and Audit Generation
Conduct a comprehensive review of `docs/specifications/PP-066-release-spec.md`. The review should evaluate compliance with the PP-066 specification, check for grammar and clarity, and audit project metrics or status. Produce an audit document summarizing these findings.

### R2. File Delivery
Copy the newly generated audit document and the original `PP-066-release-spec.md` file to the `~/Desktop` directory.

## Acceptance Criteria

### Audit Quality
- [ ] The generated audit document covers all requested review dimensions (spec compliance, grammar/clarity, and project metrics). (Verified via agent-as-judge)

### Delivery Verification
- [ ] Both the generated audit document and the copied `PP-066-release-spec.md` file are verified to exist in `~/Desktop` (Verified programmatically via a script testing for file existence).

---
*Next: when approved → delegate via invoke_subagent (see Delegation Protocol)*
