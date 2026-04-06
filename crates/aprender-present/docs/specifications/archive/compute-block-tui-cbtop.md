# ComputeBlock TUI Specification (cbtop)

**Version:** 3.0.0
**Status:** ENFORCED
**Date:** 2026-01-11
**Philosophy:** Popperian Falsificationism

---

## SECTION 0: EPISTEMOLOGICAL FOUNDATION

### 0.1 The Popper Principle

**We do not verify. We falsify.**

> "The criterion of the scientific status of a theory is its falsifiability, or refutability, or testability." — Karl Popper, *Conjectures and Refutations* (1963)

A test that cannot fail is worthless. A test designed to pass is theater. The only meaningful test is one that **tries to prove the code is broken**.

### 0.2 The Three Laws of Falsificationist Testing

**LAW 1: Every test must be capable of failing**
```rust
// WORTHLESS (unfalsifiable):
let _field: &Vec<f32> = &snapshot.per_core_temp;  // Always passes if it compiles

// FALSIFIABLE:
assert!(temps[0] > 0.0, "FALSIFIED: Core 0 has no temperature");
```

**LAW 2: Tests must make bold, specific predictions**
```rust
// WEAK (vague):
assert!(!temps.is_empty());  // Almost never fails

// BOLD (specific):
assert_eq!(temps.len(), 48, "FALSIFIED: Expected 48 core temps, got {}", temps.len());
assert!(temps[47] > 20.0 && temps[47] < 105.0, "FALSIFIED: Core 47 temp {} out of range", temps[47]);
```

**LAW 3: Actively seek failure conditions**
```rust
// CONFIRMATION BIAS (seeks success):
#[test]
fn test_temp_works() {
    let temps = read_temps();
    assert!(temps.len() > 0);  // Designed to pass
}

// FALSIFICATIONIST (seeks failure):
#[test]
fn falsify_temp_on_amd_k10temp() {
    // k10temp has temp1,temp3,temp4,temp5,temp6 - NO temp2
    // This WILL fail if code assumes temp2 exists
    let temps = read_temps_with_mock(K10TEMP_LAYOUT);
    assert!(temps[0] > 0.0, "FALSIFIED: k10temp temp2 gap not handled");
}
```

### 0.3 Confirmation vs Corroboration

| Approach | Question Asked | Value |
|----------|----------------|-------|
| **Confirmation** | "Does this work?" | Zero - unfalsifiable |
| **Corroboration** | "I tried to break this and couldn't" | High - survived falsification |

**Our tests do not PROVE correctness. They FAIL TO FALSIFY incorrectness.**

---

## SECTION 1: FALSIFIABLE CLAIMS

Every feature must be expressed as a **falsifiable claim**. If we cannot write a test that could disprove the claim, the claim is meaningless.

### 1.1 CPU Panel Claims

| Claim | Falsification Test | Falsified By |
|-------|-------------------|--------------|
| "Shows temperature for all cores" | Check temps[0..47] all > 0 | Any temp showing "-" |
| "Frequency updates in real-time" | Compare freq at t=0 vs t=1s | Values identical after CPU load change |
| "Core count matches system" | Compare to `nproc` output | Mismatch |
| "Load average matches /proc/loadavg" | Parse and compare | Delta > 1.0 |
| "Works on AMD k10temp" | Mock k10temp sysfs (no temp2) | Any core shows "-" |
| "Works on Intel coretemp" | Mock coretemp sysfs | Any core shows "-" |
| "All 48 cores visible" | Count visible core rows | Count < 48 |
| "Histogram buckets sum to core count" | Sum all buckets | Sum != nproc |

### 1.2 Memory Panel Claims

| Claim | Falsification Test | Falsified By |
|-------|-------------------|--------------|
| "Total matches /proc/meminfo" | Compare MemTotal | Delta > 1% |
| "Used + Available = Total" | Math check | Doesn't sum |
| "Swap shown when enabled" | Check on system with swap | Swap line missing |
| "Bar percentage matches numeric" | Compare bar fill to number | Mismatch > 5% |

### 1.3 Process Panel Claims

| Claim | Falsification Test | Falsified By |
|-------|-------------------|--------------|
| "USER column shows usernames" | Check for actual names | Any "-" in USER column |
| "PID matches ps output" | Compare to `ps aux` | Missing PIDs |
| "CPU% within 10% of top" | Compare top 10 processes | Delta > 10% |
| "Command not garbled" | Check for control chars | Non-printable chars in CMD |
| "Full command in exploded view" | Check truncation | Command truncated when space available |

### 1.4 Network Panel Claims

| Claim | Falsification Test | Falsified By |
|-------|-------------------|--------------|
| "All interfaces shown" | Compare to `ip link` | Missing interface |
| "RX/TX rates update" | Generate traffic, check delta | Rates stuck at 0 |
| "Rate formatting correct" | 1500 B/s shows as "1.5K/s" | Wrong format |

---

## SECTION 2: FALSIFICATION TEST PATTERNS

### 2.1 The AMD k10temp Falsification Test

This test **would have caught** the temperature bug:

```rust
/// FALSIFICATION TEST: Temperature display on AMD systems
///
/// AMD k10temp driver exposes: temp1 (Tctl), temp3-6 (Tccd1-4)
/// There is NO temp2. Code that assumes temp{n+2} exists will fail.
#[test]
fn falsify_temp_on_amd_k10temp() {
    // Arrange: Mock k10temp sysfs layout
    let mock = MockHwmon::new("k10temp")
        .with_temp(1, "Tctl", 65000)      // temp1_input
        // NO temp2_input - this is the trap
        .with_temp(3, "Tccd1", 60000)     // temp3_input
        .with_temp(4, "Tccd2", 55000)     // temp4_input
        .with_temp(5, "Tccd3", 58000)     // temp5_input
        .with_temp(6, "Tccd4", 52000);    // temp6_input

    // Act: Read temperatures for 48 cores
    let temps = read_core_temperatures_with_hwmon(&mock, 48);

    // FALSIFICATION ATTEMPTS:

    // Attempt 1: Core 0 must have a temperature
    assert!(
        temps[0] > 0.0,
        "FALSIFIED: Core 0 shows '-' because code tried to read temp2_input which doesn't exist"
    );

    // Attempt 2: All cores must have temperatures (mapped from CCDs)
    for i in 0..48 {
        assert!(
            temps[i] > 0.0,
            "FALSIFIED: Core {} has no temperature. CCD mapping is broken.", i
        );
    }

    // Attempt 3: Temperatures should be distributed across CCDs
    // Cores 0-11 → Tccd1, 12-23 → Tccd2, 24-35 → Tccd3, 36-47 → Tccd4
    assert!(
        (temps[0] - 60.0).abs() < 1.0,
        "FALSIFIED: Core 0 should map to Tccd1 (60°C), got {}", temps[0]
    );
    assert!(
        (temps[36] - 52.0).abs() < 1.0,
        "FALSIFIED: Core 36 should map to Tccd4 (52°C), got {}", temps[36]
    );
}
```

### 2.2 The Process USER Column Falsification Test

```rust
/// FALSIFICATION TEST: USER column shows actual usernames
///
/// Previously failed because ProcessRefreshKind didn't include .with_user()
#[test]
fn falsify_user_column_shows_dashes() {
    let app = App::new_real();  // Non-deterministic, real data

    // Render process panel
    let output = render_process_panel(&app);

    // FALSIFICATION: Look for the failure pattern
    let lines: Vec<&str> = output.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        // Parse USER column (typically column 2)
        if let Some(user) = parse_user_column(line) {
            assert!(
                user != "-",
                "FALSIFIED at line {}: USER column shows '-' instead of username.\n\
                 This means .with_user() is missing from ProcessRefreshKind.\n\
                 Line: {}", i, line
            );

            assert!(
                !user.is_empty(),
                "FALSIFIED at line {}: USER column is empty", i
            );

            // Username should be alphanumeric (not garbage)
            assert!(
                user.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-'),
                "FALSIFIED at line {}: USER '{}' contains invalid characters", i, user
            );
        }
    }
}
```

### 2.3 The Garbled Command Falsification Test

```rust
/// FALSIFICATION TEST: Command text is not corrupted
#[test]
fn falsify_command_text_garbled() {
    let app = App::new_real();
    let output = render_process_panel(&app);

    for line in output.lines() {
        if let Some(cmd) = parse_command_column(line) {
            // FALSIFICATION: Commands should not contain garbage

            // No control characters
            assert!(
                !cmd.chars().any(|c| c.is_control() && c != '\t'),
                "FALSIFIED: Command contains control characters: {:?}", cmd
            );

            // No sequences of random alphanumerics that look like memory corruption
            // Pattern: lowercase + digits + lowercase (like "4h89vwbbsb3")
            let garbage_pattern = regex::Regex::new(r"[a-z][0-9][a-z0-9]{6,}").unwrap();
            assert!(
                !garbage_pattern.is_match(&cmd),
                "FALSIFIED: Command looks like memory garbage: {}", cmd
            );

            // Should start with a valid path character or alphanumeric
            assert!(
                cmd.starts_with('/') || cmd.starts_with('.') ||
                cmd.chars().next().map(|c| c.is_alphanumeric()).unwrap_or(false),
                "FALSIFIED: Command starts with invalid character: {}", cmd
            );
        }
    }
}
```

### 2.4 The Core Count Falsification Test

```rust
/// FALSIFICATION TEST: All cores are visible
#[test]
fn falsify_missing_cores() {
    let expected_cores = num_cpus::get();  // e.g., 48

    let app = App::new_real();
    app.exploded_panel = Some(PanelType::Cpu);

    let output = render_full_screen(&app, 180, 60);

    // Count visible core entries
    let visible_cores = count_core_rows(&output);

    // FALSIFICATION: We should see ALL cores
    assert_eq!(
        visible_cores, expected_cores,
        "FALSIFIED: Only {} of {} cores visible. Missing cores: {:?}",
        visible_cores, expected_cores,
        find_missing_cores(&output, expected_cores)
    );
}
```

---

## SECTION 3: BOLD CONJECTURES (Claims That Risk Being Wrong)

Popper emphasized **bold conjectures** - claims that are specific enough to be wrong.

### 3.1 Weak vs Bold Claims

| Weak (Hard to Falsify) | Bold (Easy to Falsify) |
|------------------------|------------------------|
| "Renders something" | "Renders exactly 48 core rows" |
| "Shows temperatures" | "Shows temps 20-105°C for all cores" |
| "Updates data" | "Frequency changes within 2s of CPU load" |
| "Displays processes" | "Top 10 by CPU% match `top` within 5%" |

### 3.2 Our Bold Conjectures

**CONJECTURE 1: Temperature Universality**
> "For any Linux system with hwmon sensors, all CPU cores will display temperatures in the range 20-105°C, regardless of vendor (Intel/AMD) or driver (coretemp/k10temp/zenpower)."

Falsification: Find any system where cores show "-"

**CONJECTURE 2: Real-Time Accuracy**
> "CPU percentages displayed are within 5% of values reported by `top` sampled at the same instant."

Falsification: Simultaneous capture showing >5% delta

**CONJECTURE 3: Complete Visibility**
> "In exploded CPU view at 180x60, ALL cores are visible without scrolling."

Falsification: Any core not visible

**CONJECTURE 4: Data Freshness**
> "All displayed values are less than 2 seconds old."

Falsification: Value unchanged after known system state change

---

## SECTION 4: SEVERE TESTS

A **severe test** is one with high probability of failing if the claim is false.

### 4.1 Severity Levels

| Level | Description | Example |
|-------|-------------|---------|
| **S0** | Cannot fail | `assert!(true)` |
| **S1** | Unlikely to fail | `assert!(!temps.is_empty())` |
| **S2** | Might fail | `assert!(temps[0] > 0.0)` |
| **S3** | Likely to fail if bug exists | `assert!(temps[47] > 0.0)` on 48-core |
| **S4** | Will definitely fail if bug exists | Mock k10temp, check all cores |

**All tests MUST be S3 or S4.**

### 4.2 Severity Analysis of Current Tests

| Test | Current Severity | Required Severity | Gap |
|------|------------------|-------------------|-----|
| `test_metrics_snapshot_includes_per_core_temp` | S0 (checks field exists) | S4 (checks values correct) | FAIL |
| `test_app_has_per_core_temp_field` | S0 | S4 | FAIL |
| `test_apply_snapshot_updates_freq_temp` | S2 (checks one value) | S4 (checks all values) | PARTIAL |
| `test_render_uses_async_updated_data` | S2 | S4 | PARTIAL |

### 4.3 Required Severe Tests

Every panel MUST have tests that:

1. **Test boundary conditions** (core 0, core 47, not just "some core")
2. **Test vendor variations** (AMD k10temp, Intel coretemp)
3. **Test failure modes** (missing sensors, permission denied)
4. **Test real hardware** (not just mocks)
5. **Compare to ground truth** (`top`, `free`, `sensors`, `ps`)

---

## SECTION 5: THE FALSIFICATION PROTOCOL

### 5.1 Before Claiming "It Works"

Before any claim of correctness, you MUST:

1. **State the falsifiable claim** explicitly
2. **Design a test that TRIES TO BREAK IT**
3. **Run the test on real hardware**
4. **Document the failure modes tested**
5. **Only then claim "not yet falsified"**

### 5.2 The Falsification Checklist

For each panel:

```markdown
## Panel: CPU

### Falsifiable Claims
1. [ ] All 48 cores show temperatures > 0
2. [ ] Frequencies match /proc/cpuinfo within 100MHz
3. [ ] Load average matches /proc/loadavg within 0.5
4. [ ] Histogram buckets sum to core count

### Falsification Tests Run
1. [ ] AMD k10temp mock (no temp2) - PASSED/FAILED
2. [ ] Intel coretemp mock - PASSED/FAILED
3. [ ] Real hardware test (48-core TR) - PASSED/FAILED
4. [ ] Comparison with `sensors` output - PASSED/FAILED

### Known Falsifications (BUGS)
1. k10temp: temp2 doesn't exist, cores 0,5-47 show "-"
2. High core count: only 25/48 visible in exploded view

### Status: FALSIFIED / NOT YET FALSIFIED
```

### 5.3 CI Enforcement

```yaml
# .github/workflows/falsification.yml
name: Falsification Tests

on: [push, pull_request]

jobs:
  falsify:
    runs-on: ubuntu-latest
    steps:
      - name: Run Severe Tests (S3+)
        run: cargo test --features falsification -- --include-ignored

      - name: Check Severity Levels
        run: |
          # Fail if any S0/S1 tests exist
          cargo test --features falsification -- --list | \
            grep -E "^test_.*: test$" | \
            while read test; do
              severity=$(cargo test $test -- --nocapture 2>&1 | grep "SEVERITY:")
              if [[ "$severity" =~ S[01] ]]; then
                echo "FAIL: $test is severity $severity (must be S3+)"
                exit 1
              fi
            done
```

---

## SECTION 6: THE ANTI-PATTERNS

### 6.1 Confirmation Bias Tests (FORBIDDEN)

```rust
// FORBIDDEN: Tests designed to pass
#[test]
fn test_temperature_works() {
    let temps = get_temps();
    assert!(!temps.is_empty());  // This ALWAYS passes
}

// FORBIDDEN: Testing implementation, not behavior
#[test]
fn test_has_temp_field() {
    let _: &Vec<f32> = &snapshot.per_core_temp;  // Compiles = passes
}

// FORBIDDEN: Vague assertions
#[test]
fn test_renders_something() {
    let output = render();
    assert!(output.len() > 0);  // Useless
}
```

### 6.2 The Coconut Radio Pattern (FORBIDDEN)

```rust
// COCONUT RADIO: Looks like a test, isn't a test
#[test]
fn test_cpu_panel_interface() {
    // "Look, we have a test!"
    let panel = CpuPanel::new();
    let _ = panel.render();
    // No assertions. Just vibes.
}
```

### 6.3 The "Works On My Machine" Pattern (FORBIDDEN)

```rust
// FORBIDDEN: Only tests happy path on developer's machine
#[test]
fn test_temps() {
    let temps = read_temps();
    // Works on my Intel machine, fails on AMD
    assert!(temps[0] > 0.0);
}
```

---

## SECTION 7: IMPLEMENTATION REQUIREMENTS

### 7.1 Temperature Reading (Falsification-Driven Fix)

The current code FAILS the k10temp falsification test. Here's the fix:

```rust
fn read_core_temperatures(core_count: usize) -> Vec<f32> {
    let mut temps = vec![0.0f32; core_count];

    // Find hwmon device
    let hwmon = find_cpu_hwmon();

    // Read ALL available temp*_input files (don't assume layout)
    let available_temps = read_all_temps(&hwmon);

    // Map sensors to cores based on labels and driver
    match hwmon.driver_name() {
        "k10temp" => {
            // AMD: Tccd1-4 map to core groups
            // Tccd1 (temp3) → cores 0-11
            // Tccd2 (temp4) → cores 12-23
            // Tccd3 (temp5) → cores 24-35
            // Tccd4 (temp6) → cores 36-47
            if let Some(tccd1) = available_temps.get("Tccd1") {
                for i in 0..12.min(core_count) { temps[i] = *tccd1; }
            }
            if let Some(tccd2) = available_temps.get("Tccd2") {
                for i in 12..24.min(core_count) { temps[i] = *tccd2; }
            }
            // ... etc
        }
        "coretemp" => {
            // Intel: temp2_input = Core 0, temp3_input = Core 1, etc.
            for (i, temp) in available_temps.iter().enumerate() {
                if i < core_count { temps[i] = *temp; }
            }
        }
        _ => {
            // Unknown driver: use package temp for all
            if let Some(pkg) = available_temps.get("Package") {
                temps.fill(*pkg);
            }
        }
    }

    temps
}
```

### 7.2 Process USER Column (Falsification-Driven Fix)

```rust
// The falsification test revealed: .with_user() was missing
let refresh_kind = ProcessRefreshKind::nothing()
    .with_cpu()
    .with_memory()
    .with_user(UpdateKind::OnlyIfNotSet);  // THIS WAS MISSING
```

---

## SECTION 8: METRICS

### 8.1 Test Quality Metrics

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Tests at S3+ severity | 100% | ~20% | FAIL |
| Falsifiable claims covered | 100% | ~30% | FAIL |
| Vendor variations tested | AMD, Intel | Intel only | FAIL |
| Real hardware CI | Yes | No | FAIL |

### 8.2 The Only Metric That Matters

> **"How many ways did we TRY to break it?"**

Not: "How many tests pass?"
Not: "What's the coverage?"
Not: "Do the tests compile?"

**Only: "What falsification attempts survived?"**

---

## SECTION 9: REFERENCES

1. Popper, K. (1963). *Conjectures and Refutations: The Growth of Scientific Knowledge*. Routledge.

2. Popper, K. (1959). *The Logic of Scientific Discovery*. Hutchinson.

3. Mayo, D. (1996). *Error and the Growth of Experimental Knowledge*. University of Chicago Press.

4. Lakatos, I. (1978). *The Methodology of Scientific Research Programmes*. Cambridge University Press.

---

## SECTION 10: SUMMARY

**OLD APPROACH (Confirmation):**
- Write code
- Write tests that check "does it work?"
- Tests pass
- Claim success
- Ship bugs

**NEW APPROACH (Falsification):**
- State falsifiable claim
- Write tests that TRY TO BREAK the claim
- Run on multiple vendors/configurations
- If tests fail: claim is falsified, fix code
- If tests pass: claim is "not yet falsified"
- Ship with humility

---

*"The game of science is, in principle, without end. He who decides one day that scientific statements do not call for any further test, and that they can be regarded as finally verified, retires from the game."* — Karl Popper
