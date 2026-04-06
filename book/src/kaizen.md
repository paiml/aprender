# Fleet Enforcement (Kaizen)

`pv kaizen` implements continuous improvement for contract enforcement
across the entire PAIML sovereign stack.

## Overview

The PAIML sovereign stack comprises 25 Rust crates. Each has a
`binding.yaml` mapping contract equations to implementation functions.
`pv kaizen` measures how many of those functions actually have
contract assertion call sites, classifies their quality (E0/E1/E2),
and reports per-repo letter grades.

## Tiered Scoring

The fleet is split into two tiers with different expectations:

### Kernel Tier

**Repos**: aprender, entrenar, realizar, trueno

These implement mathematical kernels (softmax, matmul, attention, etc.)
where contract enforcement catches real numerical bugs. Quality metric:
`penetration x quality` where E0=0.1, E1=0.5, E2=1.0.

### Tool Tier

**Repos**: 21 other repos (batuta, decy, depyler, forjar, etc.)

These implement tools, infrastructure, and services where contract
enforcement provides structural wiring. E0 is acceptable since these
functions don't do numerical computation. Metric: penetration only.

## Enforcement Levels

| Level | Name | Criteria | Weight |
|-------|------|----------|--------|
| **E0** | Generic | Empty macro body or `!is_empty()` only | 0.1 |
| **E1** | Precondition | Domain-specific pre-checks (`is_finite`, `len() >`) | 0.5 |
| **E2** | Full DbC | Both `contract_pre_*` and `contract_post_*` call sites | 1.0 |

## Letter Grades

| Grade | Score | Meaning |
|-------|-------|---------|
| **A** | >= 0.60 | Strong DbC — domain-specific pre+post |
| **B** | >= 0.40 | Good coverage — majority E1+ |
| **C** | >= 0.25 | Moderate — wired but many E0 |
| **D** | >= 0.10 | Weak — low quality |
| **F** | < 0.10 | Minimal or no enforcement |

## Usage

```bash
# Measure entire fleet (dry-run, default)
pv kaizen --src-root /path/to/repos

# Measure single repo
pv kaizen --src-root /path/to/repos --repo aprender

# JSON output for CI
pv kaizen --src-root /path/to/repos --json

# CI gate: fail if below threshold
pv kaizen --src-root /path/to/repos --min-score 0.50
```

## Current State (v0.2.1)

```
Fleet: Grade A (0.92) — 636 sites, 376 E2, 98% pen
Kernel: Grade A — 259/239 sites, 53% E2
Tool: Grade A — 287/295 sites, 97% pen

24/25 repos at Grade A
```

## Five-Whys Root Cause Analysis

The kaizen process uses Toyota Production System five-whys to trace
enforcement gaps to root causes:

1. **Grade F**: Missing `generated_contracts.rs` or `#[macro_use]`
2. **Grade D**: Stale codegen, YAML variable names codegen can't map
3. **Grade C**: Missing postcondition call sites
4. **Grade B**: Need 20%+ E2 sites (pre+post) for Grade A
5. **Grade A**: Achieved — maintain via CI gate (`--min-score 0.60`)
