# Agent models — the canonical registry

This file is the single place where model identifiers are declared. MACS F6
(`pmat comply check`, CB-1657) reads it: a superseded model id appearing in any
other document is reported as **doc model drift**, on the theory that a model id
quoted in prose is a claim about what we run, and claims scatter.

The check was failing on this repository for a reason worth recording — **this
file did not exist**. With no registry, every model id anywhere in `docs/` is by
definition "outside the registry", so the gate reported drift it had no way to
resolve. The fix is the registry, not edits to the documents that mention models.

## Current

| role | id | notes |
|---|---|---|
| Opus 5 | `claude-opus-5` | most capable; the default for agent work in this repo |
| Sonnet 5 | `claude-sonnet-5` | balanced cost/capability |
| Fable 5 | `claude-fable-5` | |
| Haiku 4.5 | `claude-haiku-4-5-20251001` | fastest; date-stamped id |

When building anything in this repository that calls a model, prefer the latest
and most capable of these unless there is a measured reason not to.

## Superseded

These ids appear in archived documents under `docs/archive/`. They are **not**
in use and must not be introduced into new code or docs.

| id | superseded by |
|---|---|
| `gpt-4-turbo`, `gpt-4-turbo-preview` | not applicable — non-Anthropic, quoted only in a historical cost table |
| `claude-3-opus` | `claude-opus-5` |
| `claude-3-sonnet` | `claude-sonnet-5` |
| `claude-3-haiku` | `claude-haiku-4-5-20251001` |

### Why the archive is not rewritten

`docs/archive/entrenar-contract-falsification-sweep.md` (dated 2026-02-23) quotes
a cost-tier match expression containing four of these ids. That document is a
record of an analysis performed on a specific day against a specific tree. Editing
its quoted code so a present-day gate goes green would make the record describe
something that never ran — the same class of defect as a test asserting a
behaviour it did not observe, which is what the 0.63.0 audit
(`docs/audits/dogfood-0.63.0-hansei.md`) was about.

An archive earns its name by being left alone. Declaring the ids here, with what
replaced them, satisfies the gate's actual intent — one place to look — without
falsifying history.

## Adding a model

Add it to **Current** here first, then use the id. Moving an id from Current to
Superseded is the only supported way to retire one; deleting it leaves every
existing mention looking like drift with nothing to point at.
