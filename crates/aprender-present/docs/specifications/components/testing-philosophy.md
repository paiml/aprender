# Testing Philosophy

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** Popperian falsificationism applied to software testing. Severity levels, bold conjectures, anti-patterns.

---

## Epistemological Foundation

> "The criterion of the scientific status of a theory is its falsifiability, or refutability, or testability." -- Karl Popper, *Conjectures and Refutations* (1963)

**We do not verify. We falsify.**

A test that cannot fail is worthless. A test designed to pass is theater. The only meaningful test is one that **tries to prove the code is broken**.

Our tests do not PROVE correctness. They FAIL TO FALSIFY incorrectness.

## The Three Laws

### LAW 1: Every test must be capable of failing

```rust
// WORTHLESS (unfalsifiable):
let _field: &Vec<f32> = &snapshot.per_core_temp;  // Always passes if it compiles

// FALSIFIABLE:
assert!(temps[0] > 0.0, "FALSIFIED: Core 0 has no temperature");
```

### LAW 2: Tests must make bold, specific predictions

```rust
// WEAK:
assert!(!temps.is_empty());

// BOLD:
assert_eq!(temps.len(), 48, "FALSIFIED: Expected 48 core temps, got {}", temps.len());
assert!(temps[47] > 20.0 && temps[47] < 105.0);
```

### LAW 3: Actively seek failure conditions

```rust
// CONFIRMATION BIAS (seeks success):
fn test_temp_works() { assert!(read_temps().len() > 0); }

// FALSIFICATIONIST (seeks failure):
fn falsify_temp_on_amd_k10temp() {
    // k10temp has temp1,temp3-6 - NO temp2
    let temps = read_temps_with_mock(K10TEMP_LAYOUT);
    assert!(temps[0] > 0.0, "FALSIFIED: k10temp temp2 gap not handled");
}
```

## Confirmation vs Corroboration

| Approach | Question | Value |
|----------|----------|-------|
| Confirmation | "Does this work?" | Zero - unfalsifiable |
| Corroboration | "I tried to break this and couldn't" | High - survived falsification |

## Severity Levels

| Level | Description | Example | Allowed? |
|-------|-------------|---------|----------|
| S0 | Cannot fail (Coconut Radio) | `assert!(true)` | FORBIDDEN |
| S1 | Unlikely to fail | `assert!(!temps.is_empty())` | FORBIDDEN |
| S2 | Might fail | `assert!(temps[0] > 0.0)` | FORBIDDEN |
| S3 | Likely to fail if bug exists | `assert!(temps[47] > 0.0)` on 48-core | REQUIRED |
| S4 | Will definitely fail if bug exists | Mock k10temp, check all cores | REQUIRED |

**All tests MUST be S3 or S4.**

## Bold Conjectures

Specific enough to be wrong:

| Weak (Hard to Falsify) | Bold (Easy to Falsify) |
|------------------------|------------------------|
| "Renders something" | "Renders exactly 48 core rows" |
| "Shows temperatures" | "Shows temps 20-105C for all cores" |
| "Updates data" | "Frequency changes within 2s of CPU load" |
| "Displays processes" | "Top 10 by CPU% match `top` within 5%" |

### Example Bold Conjectures

1. **Temperature Universality:** For any Linux system with hwmon sensors, all CPU cores display temperatures in 20-105C range, regardless of vendor/driver.
2. **Real-Time Accuracy:** CPU percentages are within 5% of `top` sampled at the same instant.
3. **Complete Visibility:** In exploded CPU view at 180x60, ALL cores are visible without scrolling.
4. **Data Freshness:** All displayed values are less than 2 seconds old.

## Anti-Patterns (FORBIDDEN)

### Coconut Radio Pattern
Looks like a test, isn't a test:
```rust
fn test_cpu_panel_interface() {
    let panel = CpuPanel::new();
    let _ = panel.render();
    // No assertions. Just vibes.
}
```

### Confirmation Bias Tests
Tests designed to pass:
```rust
fn test_temperature_works() {
    let temps = get_temps();
    assert!(!temps.is_empty());  // This ALWAYS passes
}
```

### "Works On My Machine" Pattern
```rust
fn test_temps() {
    let temps = read_temps();
    // Works on Intel, fails on AMD
    assert!(temps[0] > 0.0);
}
```

### Implementation Testing (not behavior)
```rust
fn test_has_temp_field() {
    let _: &Vec<f32> = &snapshot.per_core_temp;  // Compiles = passes
}
```

## The Falsification Protocol

Before claiming "It Works":

1. State the falsifiable claim explicitly
2. Design a test that TRIES TO BREAK IT
3. Run the test on real hardware
4. Document the failure modes tested
5. Only then claim "not yet falsified"

### Panel Checklist Template

```markdown
## Panel: CPU
### Falsifiable Claims
1. [ ] All 48 cores show temperatures > 0
2. [ ] Frequencies match /proc/cpuinfo within 100MHz
### Falsification Tests Run
1. [ ] AMD k10temp mock (no temp2) - PASSED/FAILED
2. [ ] Real hardware test (48-core TR) - PASSED/FAILED
### Known Falsifications (BUGS)
### Status: FALSIFIED / NOT YET FALSIFIED
```

## Severe Tests

A **severe test** has high probability of failing if the claim is false.

Every panel MUST have tests that:
1. Test boundary conditions (core 0 AND core 47)
2. Test vendor variations (AMD k10temp, Intel coretemp)
3. Test failure modes (missing sensors, permission denied)
4. Test real hardware (not just mocks)
5. Compare to ground truth (`top`, `free`, `sensors`, `ps`)

## The Only Metric That Matters

> "How many ways did we TRY to break it?"

Not: "How many tests pass?" Not: "What's the coverage?" Not: "Do the tests compile?"

**Only: "What falsification attempts survived?"**

## References

1. Popper, K. (1963). *Conjectures and Refutations*. Routledge.
2. Popper, K. (1959). *The Logic of Scientific Discovery*. Hutchinson.
3. Mayo, D. (1996). *Error and the Growth of Experimental Knowledge*. University of Chicago Press.
4. Lakatos, I. (1978). *The Methodology of Scientific Research Programmes*. Cambridge University Press.
