# Toyota Way Overview

Presentar is built on Toyota Production System principles.

## Core Pillars

| Pillar | Japanese | Application |
|--------|----------|-------------|
| Continuous Improvement | Kaizen | Red-Green-Refactor cycle |
| Respect for People | | Clear documentation, accessibility |
| Long-term Thinking | | Quality over shortcuts |
| Problem Solving | Genchi Genbutsu | Go to the source code |

## Key Concepts

### Jidoka (Built-in Quality)

Stop when defects detected:
- Tests must pass before commit
- Clippy warnings are errors
- Type safety catches bugs at compile time

### Muda (Waste Elimination)

Remove non-value activities:
- No Python GIL overhead
- No runtime interpretation
- Minimal dependencies

### Poka-yoke (Mistake Proofing)

Design to prevent errors:
- Strict type system
- Schema validation
- Compile-time checks

## The Three-Tier Pipeline

Inspired by Toyota's quality gates:

```
Tier 1: On-save     → Andon light (immediate feedback)
Tier 2: Pre-commit  → Quality inspection
Tier 3: Nightly     → Full audit
```

## Genchi Genbutsu

"Go and see":
- Debug with real data
- Profile actual workloads
- Test on target devices

## Verified Test

```rust
#[test]
fn test_toyota_principles_applied() {
    // Jidoka: Tests catch defects
    let button = presentar_widgets::Button::new("Test");
    assert!(button.is_interactive());  // Built-in quality

    // Poka-yoke: Type system prevents errors
    // let bad: Button = 42;  // Won't compile!
}
```
