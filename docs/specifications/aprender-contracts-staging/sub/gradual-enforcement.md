# Sub-spec: Gradual Enforcement

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [pv-spec.md](../pv-spec.md) Section 17

---

## Motivation

Falsification of orthogonal annotation enforcement systems (mypy, TypeScript,
Rust `#[must_use]`/`#[forbid]`, C# nullable, JSpecify, Haskell/LiquidHaskell,
ty, Elm) revealed 5 gaps in provable-contracts. This sub-spec closes them.

**Design principle from the research:** Tools that achieve highest adoption
have the **smoothest on-ramp** (gradual stages) and the **hardest off-ramp**
(ratchets, locks, stale detection). provable-contracts has a hard on-ramp
(AllImplemented from day 1) but no off-ramp protection.

---

## Gap 1: Per-Contract Enforcement Levels

**Pattern:** mypy per-module `[mypy-module.*]`, C# per-file `#nullable enable`,
JSpecify per-package `@NullMarked`.

### YAML Extension

```yaml
metadata:
  enforcement_level: strict  # basic | standard | strict | proven
```

| Level | Requirements |
|---|---|
| `basic` | Schema valid, has equations |
| `standard` | + falsification tests + Kani harnesses (PROVABILITY-001) |
| `strict` | + all bindings implemented + `#[contract]` annotations |
| `proven` | + Lean 4 proved (no sorry) |

Default: `standard` for new contracts. Existing contracts without the field
are treated as `standard`.

### CLI

```bash
pv lint contracts/ --min-level standard   # Fail if any contract below standard
pv lint contracts/ --min-level strict     # Require bindings + annotations
```

### Implementation Status

- ✅ `crates/provable-contracts/src/schema/types.rs` — `EnforcementLevel` enum (Basic/Standard/Strict/Proven)
- ✅ `crates/provable-contracts/src/lint/gates_extended.rs` — Gate 6: enforcement level checking
- ✅ `--min-level` CLI flag implemented and parsed
- ✅ `locked_level` field with `pv unlock --reason` command
- ⚠️ `compute_actual_level()` does NOT detect Strict (no binding/annotation check)
- ❌ `.pv.toml` config `default_level` — NOT IMPLEMENTED (CLI-only for now)
- ❌ Per-glob enforcement overrides — NOT IMPLEMENTED

---

## Gap 2: Stale Suppression Detection

**Pattern:** TypeScript `@ts-expect-error` (errors if suppression unnecessary),
Rust `#[expect(lint)]` (warns on unused), ty `unused-ignore-comment`.

### Rule: PV-SUP-001 (Stale Suppression)

When a finding ID in `suppress.findings` or `suppress.rules` no longer matches
any active finding, emit `PV-SUP-001` warning:

```
[WARN] PV-SUP-001: Suppression 'PV-AUD-001' for contracts/softmax-kernel-v1.yaml
       is stale — the finding no longer fires. Remove the suppression.
```

### Ratchet Property

Once a suppression becomes stale, removing it is safe and encouraged. With
`--strict`, stale suppressions become errors — preventing accumulation.

### YAML-Level Suppression

```yaml
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    suppress: PV-ENF-001  # Inline suppression with stale detection
```

### Implementation

- `crates/provable-contracts/src/lint/mod.rs` — After computing findings,
  diff against suppressions. Any suppression not consumed is stale.
- Rule `PV-SUP-001` severity: `Warning` (default), `Error` (with `--strict`)

---

## Gap 3: Multi-Stage Enforcement Pipeline

**Pattern:** C# nullable `disable → warnings → annotations → enable`,
mypy individual strict flags enabled one at a time.

### Four Verification Tiers (maps to existing Verification Ladder)

```
Tier 1 (L1-L2): Schema + Tests
  ✓ Contract parses (metadata, equations, obligations)
  ✓ Falsification tests exist (>= obligations)
  Gate: pv lint gates 1-2 (validate + audit)

Tier 2 (L3): Bounded Verification
  ✓ Kani harnesses exist
  ✓ All bindings implemented (AllImplemented)
  Gate: pv lint gates 1-5 + Gate 6 (build-rs)

Tier 3 (L4): Full Verification
  ✓ Lean 4 theorems proved (no sorry)
  ✓ #[contract] annotations on all bound functions
  Gate: pv lint gates 1-7 + reverse coverage > 50%

Tier 4 (L5): Locked
  ✓ All of Tier 3
  ✓ metadata.locked_level set
  ✓ Cannot regress without --unlock
  Gate: pv lint --forbid-regression
```

### CI Integration

```yaml
# Progressive enforcement in CI
- name: Tier 1 gate
  run: pv lint contracts/ --min-level basic
- name: Tier 2 gate (new contracts only)
  run: pv lint contracts/ --min-level standard --diff main
```

---

## Gap 4: Aggregate Coverage Metric with CI Ratchet

**Pattern:** TypeScript `type-coverage`, mypy typed def count,
C# nullable-enabled file count.

### `pv lint --coverage`

Reports aggregate contract coverage as a single percentage:

```
$ pv lint --coverage contracts/
Contract Coverage: 89/121 at Tier 2+ (73.6%)
  Tier 1 (basic):    32 contracts
  Tier 2 (standard): 75 contracts
  Tier 3 (strict):   12 contracts
  Tier 4 (proven):    2 contracts
```

### CI Ratchet

```bash
pv lint --coverage contracts/ --min-coverage 0.70  # Exit 1 if below 70%
```

The coverage percentage is tracked in `.pv/trend/` alongside the existing
score trend. Each CI run records the coverage level. Coverage must be
monotonically non-decreasing (ratchet) — any drop fails CI.

### Implementation

- `crates/provable-contracts/src/lint/mod.rs` — Add `CoverageGate` after Gate 7
- Output: percentage of contracts at or above `--min-level`
- `.pv/trend/coverage.json` — Historical coverage tracking

---

## Gap 5: Irreversible Level Lock (`#![forbid]` pattern)

**Pattern:** Rust `#![forbid(unsafe_code)]` — cannot be overridden by inner
`#[allow]`. Elm totality — no escape hatch exists.

### `metadata.locked_level`

```yaml
metadata:
  locked_level: L3  # Once set, contract cannot drop below L3
```

`pv lint` enforces: if `locked_level` is set and the contract's actual
verification level is below it, emit `PV-LCK-001` **Error** (not warning).

### Unlock Protocol

Removing `locked_level` from YAML is not sufficient — `pv lint` tracks
previously-locked contracts in `.pv/locks.json`. To genuinely unlock:

```bash
pv unlock contracts/softmax-kernel-v1.yaml --reason "Refactoring proof structure"
```

This records an audit trail entry. The `--reason` is mandatory.

### Implementation

- `crates/provable-contracts/src/schema/types.rs` — Add `locked_level: Option<String>`
- `crates/provable-contracts/src/lint/mod.rs` — Gate 9: lock enforcement
- `.pv/locks.json` — Lock audit trail
- Rule `PV-LCK-001`: `Error` severity, cannot be suppressed

---

## References

### Gradual Typing and Verification

1. Siek, J. G., & Taha, W. (2006). "Gradual Typing for Functional Languages."
   *Scheme and Functional Programming Workshop*.

2. Bader, J., Aldrich, J., & Tanter, E. (2018). "Gradual Program Verification."
   *VMCAI 2018*. arXiv:1710.06422.

3. Lehmann, N., & Tanter, E. (2023). "Gradual Liquid Type Inference."
   *OOPSLA 2023*. DOI:10.1145/3622843.

4. Jafery, K. A., & Dunfield, J. (2017). "Sums of Uncertainty: Refinements Go
   Gradual." *POPL 2017*.

5. Garcia, R., Clark, A., & Tanter, E. (2016). "Abstracting Gradual Typing."
   *POPL 2016*.

### Contract Verification

6. Meyer, B. (2025). "Software engineering as a domain to formalize."
   arXiv:2502.11434.

7. Li, Y., et al. (2025). "Do Large Language Models Respect Contracts?"
   arXiv:2510.12047.

8. Bruni, R., et al. (2026). "Agent Behavioral Contracts." arXiv:2602.22302.

9. Huang, L., Meyer, B., & Weber, R. (2025). "Loop Unrolling: Formal Definition
   and Application to Testing." arXiv:2509.xxxxx.

### Type System Enforcement

10. Bierman, G., et al. (2014). "Null Safety in C#."
    *Microsoft Technical Report*.

11. Dietl, W., et al. (2011). "Building and Using Pluggable Type-Checkers."
    *ICSE 2011*.

12. Rondon, P. M., Kawaguci, M., & Jhala, R. (2008). "Liquid Types."
    *PLDI 2008*.

### Tool-Specific

13. mypy Contributors (2012-2026). "mypy: Optional Static Typing for Python."
    github.com/python/mypy.

14. Astral (2025-2026). "ty: An extremely fast Python type checker."
    docs.astral.sh/ty.

15. Microsoft (2012-2026). "TypeScript: JavaScript with Syntax for Types."
    typescriptlang.org.

16. GHC Contributors (1990-2026). "The Glasgow Haskell Compiler User's Guide."
    ghc.gitlab.haskell.org.

17. Rust Project (2015-2026). "The Rust Reference: Diagnostic Attributes."
    doc.rust-lang.org/reference/attributes/diagnostics.html.

18. JSpecify Contributors (2023-2026). "JSpecify: Standard Java Null-Safety
    Annotations." jspecify.dev.

---

## Implementation Roadmap

| Gap | Priority | Effort | Dependencies |
|---|---|---|---|
| Gap 2: Stale suppression | **High** | Small | None |
| Gap 4: Coverage metric | **High** | Small | None |
| Gap 1: Per-contract levels | **Medium** | Medium | Schema change |
| Gap 5: Level lock | **Medium** | Medium | Gap 1 |
| Gap 3: Multi-stage pipeline | **Low** | Large | Gaps 1, 4, 5 |

Recommended order: 2 → 4 → 1 → 5 → 3.
