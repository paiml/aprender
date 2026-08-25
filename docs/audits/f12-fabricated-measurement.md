# F12 — fabricated measurement

**A value carrying the form, units and provenance-shape of a measurement,
produced without the measurement having been taken.**

Catalogued 2026-08-24 while building the llama.cpp parity gates (#2667). The
class belongs to `APR-QUALITY-001`; this file is the repo-side record of the
instances and the mechanism, so a future reader can find both from the code.

## Why it is its own class

It is not its neighbours, and the distinction is what makes it hard to see:

| | why F12 is not this |
|---|---|
| **F7 — silent pass** | Nothing passes falsely. The number may even be *correct* for some past run on some host. |
| **F9 — coupled falsifier** | There is no coupled oracle. **There is no oracle at all.** |
| **F10 — stale self-description** | A stale description was once true of something. A fabricated measurement was never a measurement. |

The harm is that it is **indistinguishable from evidence at the point of
consumption**, and it survives review because it *looks* like the thing it
replaces. A reviewer checking "does the receipt have a baseline?" sees a
number and moves on.

Its specific damage to a ratio: the denominator becomes a constant of unknown
date, host and build, so the ratio stops measuring change over time while
continuing to report one.

## Instances found

| # | site | what was asserted |
|---|---|---|
| 1 | `scripts/benchmark-2x-ollama.sh:27-29` | `OLLAMA_BASELINE=291`, `OLLAMA_SINGLE=120`, `OLLAMA_CPU=15` — **ollama never invoked** |
| 2 | `scripts/benchmark-matrix.sh:396` | the same three literals emitted into JSON as `ollama_baselines` |
| 3 | `benches/external_matrix.rs:331-336` | `HardwareSpec { cpu: "Benchmark CPU", gpu: Some("Benchmark GPU"), memory_gb: 32 }` |
| 4 | `src/bench/backend.rs:140,226,336` | `version: "b2345"  // Would be detected from binary`, `"0.4.0"`, `"0.1.0"` |

Instances 1–2 were deleted (#2672, with a triage table first). Instances 3–4
were replaced by detection (#2674).

Note the comment on instance 4: *"Would be detected from binary."* The author
knew. F12 is not usually deception — it is a placeholder that outlived the
intention to replace it, which is why a mechanism beats a review.

## The general remedy

**A field that CAN be measured is derived or absent — never asserted.** Where
derivation is impossible, the receipt records the absence explicitly
(`"unknown"`, `null`) so the consuming gate can treat it as RED. A gate cannot
detect a plausible literal at all; it can trivially detect an honest gap.

## Mechanisms in the tree

| mechanism | catches |
|---|---|
| `scripts/check_no_fabricated_baselines.sh` | a competitor-named variable or JSON key assigned a numeric literal, in shell |
| `scripts/lib/bench_receipt.py` | a receipt whose `samples_ms` are all identical — a timing distribution is not constant |
| `HardwareSpec::detect()` + its tests | host identity read from `/proc`/`sysctl`, asserted against this host's real values |
| `detect_version()` in `bench/backend.rs` | comparator version read from the binary the config points at |

The shell guard ships a **must-match / must-not-match case table** because its
pattern was wrong twice before it was right: the first version missed the
JSON-literal spelling entirely, and the second used a suffix allowlist that
missed `OLLAMA_CPU=15` — one of the three live instances. Neither error was
visible on review.

## Detection heuristic worth generalising

A literal in a receipt-producing path whose *name* matches a measurable
quantity — `*_baseline`, `version`, `cpu`, `gpu`, `memory_*`, `*_tps` — is a
candidate. The `check_no_hand_rolled_parsers.sh` construct-ban shape applies:
ban the construct, not the file, or the pattern returns under a new name.

Refs #2679 · #2667 · #2672 · #2674
