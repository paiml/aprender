# Phase 8: Laya Decision Model: Local Fine-Tune and Thin MCP Server - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-25
**Phase:** 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
**Areas discussed:** Local training harness, Dataset in / quality gate out, MCP tool surface, Crate home and deploy

---

## Local training harness

| Question | Options | Selected |
|---|---|---|
| Training language | Python (Laya's own code) / Rust autograd / Python now, Rust later | Python (Laya's own code) |
| Invocation | `just laya-train` + pinned uv / `apr laya train` / plain script | `just laya-train` + pinned uv |
| Device | Auto MPS→CUDA→CPU with proof / MPS only / any silently | Auto with proof |
| Recipe | Locked / overridable flags / allow typed-decisions base | Free text (below) |

**User's choice (recipe):** "It is OK to start with the English root base, however, we will want to add in the later phase the support for the multilang models, and the typed-decisions when we get use cases for it."
**Notes:** Recorded as English-root-only with the base as a declared manifest field; recipe values kept locked.

---

## Dataset in, quality gate out

| Question | Options | Selected |
|---|---|---|
| Input | task.json + train.jsonl / `apr data` directory / CSV | task.json + train.jsonl |
| Eval data | Required eval.jsonl + seeded calibration slice / optional carve / none | Required eval + calibration slice |
| Gate | Fail-closed, deploy refuses / report only / none | Fail-closed |
| Seeds | One declared + `--seeds N` / always 3, ship median / one, silent | One declared + `--seeds N` |

---

## MCP tool surface

| Question | Options | Selected |
|---|---|---|
| Tool shape | Task-bound `classify(text)` / generic `decide` / both | Task-bound `classify` |
| Batch | List with contract max / single text | List with contract max |
| Output | Label + probs + model identity / label + confidence / label | Label + probs + identity |
| Overlength | Truncate like Laya + flag / refuse | Truncate + `truncated: true` |

---

## Crate home and deploy

| Question | Options | Selected |
|---|---|---|
| Crate home | New `aprender-laya` + servers / aprender-serve / aprender-core | Free text (below) |
| Artifact | Native files + manifest / convert to .apr | Convert to .apr |
| Deploy | Live pmcp.run / deploy-ready only / stdio only | Live pmcp.run |
| Demo task | TweetEval stance / emotion / business task | TweetEval stance |

**User's choice (crate home):** "We can use option 1 suggestion to have aprender-laya, however, it will be better if we can generalize it. If we are missing the ModernBERT we can add the support to it in a similar way that other BERT models are supported. Another direction can be to generalize it to the aprender-llm-decide or aprender-llm-classify to support other methods that will pop up soon with the success of Jev, Kev, and Laya."

Follow-ups:

| Question | Options | Selected |
|---|---|---|
| Crate name | `aprender-decide` / `aprender-llm-decide` / `aprender-llm-classify` | `aprender-decide` |
| Generality | Trait + Laya only / Laya + Kev | Trait + Laya only |
| ModernBERT home | aprender-core `models/modernbert/` / inside aprender-decide | aprender-core `models/modernbert/` |
| APR dtype | F16 widen at load / F32 | F16, widen at load |

---

## Claude's Discretion

- uv project layout and recipe names; Python-export vs `apr import` for the .apr conversion; S3/IAM layout; the
  declared ECE ceiling and zero-shot margin; the `classify` list maximum.

## Deferred Ideas

- Multilingual (mmBERT) and `typed-decisions` bases; Kev/Jev methods; a decide training MCP server; a Rust trainer;
  batched-row GEMM.
