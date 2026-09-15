# Release cell receipt — gx10 / cpu / qwen2.5-coder-7b-instruct-q4_k_m

**Pre-tag dry run.** There is no v0.66.0 tag and no release asset yet, so this cell was run
BY HAND per epic #3058 §F1 ("needing none of the missing automation"). Every gate that
depends on automation 0.66 does not ship is EXCLUDED BY NAME below, never silently skipped —
§5.5 is explicit that a missing cell is NO-GO rather than a pass, and these exclusions are
why this directory is `v0.66.0-pre` and not a tag.

| | |
|---|---|
| host | `gx10` (aarch64) |
| GPU | NVIDIA GB10 (not used in this cell) |
| backend under test | `cpu` |
| binary | `apr 0.65.2`, sha256 `8cdb7c1a668127db95cfca70952a566c3c1519a17b2656e6c3cd82bd8691b584` — built from the L0-1b tree, NOT a release asset (see M1) |
| model | `qwen2.5-coder-7b-instruct-q4_k_m.gguf`, sha256 `509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c`, 4683073536 bytes |

## Gates

| id | gate | result |
|---|---|---|
| **M3** | CPU/GPU parity | **N/A for a CPU cell** — M3 compares the two backends of a host and is recorded on that host's GPU cell |
| **M4** | determinism — two greedy runs, byte-identical token stream | **GREEN** — stream sha256 `6f38f13e5debd285` |
| **M6** | 7B service smoke — loads, > 0 tokens, no OOM | **GREEN** — non-empty output, zero OOM lines, exit 0 |
| **M1** | artifact identity (installed sha256 == manifest) | **EXCLUDED** — 0.66 ships no release assets and no manifest (D-13, D-14). The binary is locally built, so there is nothing to compare and M1 cannot be claimed |
| **M2** | registry readback via `apr devices --json` | **EXCLUDED** — `apr devices` is R-0a, moved to 0.67 by D-14 |
| **M5** | refusal semantics FX-16/17/18 | **PARTIAL** — the FX fixtures are R-0a (0.67). The admission level IS covered: `cargo test -p apr-cli --test reg15_admission` 7/7, and it is FALSIFY-CPU-GPU-014 in the parity contract |
| **M7** | transport parity (`apr serve` + effective-config) | **NOT RUN** — named, not skipped; it needs a port and a client and was out of this by-hand pass |
| **M8** | performance, report-only | **NOT RECORDED, DELIBERATELY** — 0.66's claims ratchet refuses a throughput literal on a documented surface, and 0.66 makes no speed claim. Recording one here would be the thing D-13/D-14 forbid |

## Cross-cell result worth more than any single cell

The greedy stream sha256 is **`6f38f13e5debd285` in all four cells** — lambda GPU, lambda CPU,
gx10 GPU, gx10 CPU. Same model bytes, same prompt, two architectures (x86_64 sm_89 and
aarch64 sm_121) and both backends produce a byte-identical token stream. M4 only asks for
determinism within a cell; this is determinism across the fleet.

## Verdict

**Not GO and not NO-GO — this is a dry run, and §5.5 admits no third verdict for a real
tag.** M3, M4 and M6 are green on every cell that can run today. The release verdict is
withheld because M1 and M2 are structurally unavailable in 0.66 and a receipt that called
itself GO with two gates excluded would be the theater §1.1 forbids.
