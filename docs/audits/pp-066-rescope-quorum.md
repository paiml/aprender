# PP-066 SPEC-2.0 — rescope quorum record (2026-09-06)

Three adversarial agy lanes (agy 1.1.27, `--mode plan`, `writes=false`; conversations `e6d9c78b-faab-4c6c-aa7c-db548906e2d9`, `788aa31f-bef7-46e7-9a98-4e8898788e7f`, `b57a46a2-2ba3-49b4-a6dc-0c5b80d947b1`; families: Gemini 3.1 Pro ×3 — **not three families**, a gap recorded in `docs/audits/driver-kaizen.md`) were asked: *find a claim README/notes/CLI still makes that a cut row protected*. Every lane ran with `num_turns=1` (no tool calls); the delegate hand-verified 13 of 27 citations against the tree; the orchestrator re-read the load-bearing ones below.

## Verdict: **scope-holds-with-changes** (lane 1); lanes 2/3 said scope-fails on a premise the spec refutes

The 2-of-3 "scope-fails" rested on *C3 moved to 0.67 ⇒ `--backend` resolution unenforced*. The spec says the resolution ships in **R-0b** (`docs/specifications/PP-066-release-spec.md:267` — "R-0b ships the resolution; the per-surface case table over every host is credited in 0.67 with its instrument"; R-0b is kept, #3002, PMAT-1073, a TAG-0.66.0 dependency) and C11 (credited) carries "zero `cfg!(feature)` reads in `apr-cli` backend decisions"; only the per-surface × per-host case table is deferred with its instrument. A moved criterion is not a moved capability. Verdict adopted: **scope holds with the changes below.**

## Rulings (every hit returns a row or removes a claim)

| hit (file:line, verbatim) | cut row that protected it | ruling |
|---|---|---|
| `README.md:44` `**1812** provable contracts`, `README.md:265` `1812 contracts across …` | P-0.3 (proof credit) | **remove the claim in R-7**: the count stays (a derivable file count, `find contracts -name '*.yaml' \| wc -l`, held by `check_readme_claims.sh`), the word *provable* goes — 0.66 ships no proof-count claim |
| `crates/apr-cli/src/dispatch.rs:177` — a refusal message that ships a slowdown ratio and two throughput figures (quoted at that line; not repeated here, the ratchet forbids the literal) | S-1 / S-3 | **remove in R-7** (CLI output is in the ratchet's universe): the refusal names the path, never a ratio |
| `crates/apr-cli/src/commands/chat_load_tokenizers.rs:230` (a speed ratio in a user-facing hint) | S-1 | **remove in R-7** |
| `crates/apr-cli/src/extended_commands.rs:87`, `:208` (tok/s in `--help`) | S-1 | **remove in R-7** |
| `crates/apr-cli/src/extended_commands.rs:1811`, `crates/apr-cli/src/model_ops_commands.rs:67` (`wgpu` / `metal` accepted values in `--help`) | B-W1 / B-M1 | **keep as vocabulary** — R-0b widens `BACKEND_VALUES` to the five REG-11 kinds as a static clap vocabulary with the refusal at runtime as `Unavailable(NotCompiled)` / `FeatureDisabled` (D-9); a listed value is not a coverage promise once the registry refuses it by name |
| `book/src/examples/qwen-inference.md:9` (a GPU throughput figure), `book/src/examples/showcase-benchmark.md:20` (a throughput figure), `book/src/best-practices/performance.md:250` | S-1 / I-18 | **remove in R-7** (the book is in the guard's universe; R-7 owns the user-facing rewrite) |
| `crates/aprender-zram/README.md:25` (a speed ratio), `crates/aprender-viz/README.md:35` (a speed ratio vs btop) | S-* (speed) | **remove in R-7** AND widen the ratchet: `crates/*/README.md` and `Cargo.toml` `description` are outside `check_no_claim_literals.sh`'s universe (`:1091` unions `book/**/*.md`, `docs/**/*.md`, root `*.md`, `crates/*/src/**/*.rs`) — SPEC-2.0 follow-up: widen the universe and RE-MUTATE in the widened scope (baseline set-aperture) |
| `docs/BEATS.md:121` (the cold-start ratio), `:131` (the Ollama decode parity band) | — | **keep — receipted** (the beat scoreboard is gated against `contracts/` by `readme_contract.rs`; the withdrawn headline at `:245` is the WITHDRAWN record and must never be quoted as live — lane 2's `:38` ruling was a misplaced citation and is rejected) |

## Q1 — the ratchet's universe
The spec's sentence ("README, release notes and CLI output") **understates** the guard: it already covers the book, `docs/**`, root `*.md` and every `crates/*/src/**/*.rs`. Genuine gaps: `crates/*/README.md`, `Cargo.toml` descriptions, `apr --version` banners (the last is `.rs`, covered). Action: SPEC-2.0 widens the universe (guard edit with re-mutation) and the spec names the universe the guard has.

## Q2 — sufficiency of the ten credited criteria
Sufficient for the three claims once R-0b lands (C11) — see the verdict. Two instruments the criteria name are not on main yet: `evidence/models/supported.yaml` (on `agent/L0-1`, L0-1a) and `scripts/check_backend_registry.sh` (R-0b); `release_criteria.sh` reports a missing instrument as ENV (exit 2), never a pass, until they land.

## Q3 — C5
**Unanimous: dishonest to keep credited** while every training number is cut and no claim needs it. **C5 moves to 0.67 with Track T.** Claim protected: none of the three (a training-parity receipt schema); the residue — the harness and its receipt schema — is carried by row **T-0h** (`release: 0.67`, `claim_protected:` names it) and credited in 0.67 with Track T's numbers. The credited set is C0 C4 C6 C7 C8 C9 C11 C13 C14 (nine).

## Disposition
- `scripts/release_criteria.sh`: C5 → 0.67 (this commit).
- Spec §4.1: C5 moved; the ratchet-universe sentence corrected (this commit).
- R-7's card gains the removal list above (the DAG row notes, this commit).
- Kaizen: the delegate ran three lanes of one family for an adversarial quorum; N-lane and rescope quorums must name three families in the brief and refuse otherwise.
