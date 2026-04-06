# Lean-Scientific Code Review Protocol

## Executive Summary

Code review in trueno-viz is treated as **scientific peer review**, not administrative gatekeeping. Every Pull Request is a **manuscript submission**—the author makes a claim ("This code implements Feature X"), and the reviewer acts as the **peer referee** validating that claim.

This document provides:
1. The Reproducibility Audit Checklist
2. The Annotation Taxonomy
3. Evidence Requirements
4. Stop the Line Protocol
5. Example Annotations

---

## 1. The Reproducibility Audit Checklist

**Before reading a single line of code**, reviewers MUST complete this checklist:

### Table 1: Reproducibility Audit

| Category | Verification Step | Evidence Required | Pass Criteria |
|----------|-------------------|-------------------|---------------|
| **Dependencies** | Are all libraries declared? | `Cargo.toml` complete | `cargo build` succeeds |
| **Environment** | Does code run on clean install? | Build instructions | Reviewer can build from scratch |
| **Data Availability** | Is test data accessible? | Test fixtures or generators | Tests don't require external data |
| **Determinism** | Do tests pass consistently? | CI logs | 3+ consecutive green runs |
| **Results** | Does behavior match description? | Reviewer verification | "Reproduced locally" comment |

### Audit Commands

```bash
# 1. Checkout the PR
git fetch origin
git checkout <pr-branch-or-commit>

# 2. Clean build
cargo clean && cargo build

# 3. Run tests (3x for flakiness detection)
cargo test && cargo test && cargo test

# 4. Verify specific functionality
cargo run --example <example-name>

# 5. Record reproducibility
# Add comment: "Reproducibility Audit: PASS - Verified on [date]"
```

### Audit Failure Protocol

If ANY audit item fails:

1. Tag the review: `[REPRODUCIBILITY:BLOCKING]`
2. Stop the review (do not proceed to code reading)
3. Request clarification from author
4. Author must provide evidence before re-review

---

## 2. The Annotation Taxonomy

Every review comment MUST follow this structured format:

```
[CATEGORY:SEVERITY]: <Narrative>
Evidence: <citation, data, or reference>
```

### 2.1 Category Definitions (The "What")

| Category | Definition | Scientific Parallel |
|----------|------------|---------------------|
| **CORRECTNESS** | Logic is demonstrably false | Mathematical error in paper |
| **SECURITY** | Code introduces risk | Ethics/safety violation |
| **PERFORMANCE** | Functionally correct but wasteful | Methodology inefficiency |
| **MAINTAINABILITY** | Difficult to read or extend | Paper structure/clarity |
| **REPRODUCIBILITY** | Cannot verify behavior | Non-reproducible experiment |
| **TESTABILITY** | Insufficient proof of correctness | Missing control variables |

### 2.2 Sub-Category Taxonomy

#### CORRECTNESS Sub-Categories

| Sub-Category | Definition | Example |
|--------------|------------|---------|
| `CORRECTNESS:Logic` | Control flow, boolean logic, off-by-one | `for i in 0..len` should be `0..=len` |
| `CORRECTNESS:Data` | Type errors, precision loss, schema | `f32` truncation when `f64` required |
| `CORRECTNESS:Concurrency` | Race conditions, deadlocks | Non-atomic counter access |
| `CORRECTNESS:API` | Contract violations | Missing required field |

#### SECURITY Sub-Categories

| Sub-Category | Definition | Example |
|--------------|------------|---------|
| `SECURITY:Injection` | SQLi, command injection, XSS | Unsanitized user input in query |
| `SECURITY:Auth` | Broken authentication/authorization | Missing permission check |
| `SECURITY:Privacy` | PII leakage, GDPR violations | Logging email addresses |
| `SECURITY:Crypto` | Weak algorithms, key exposure | MD5 for password hashing |

#### PERFORMANCE Sub-Categories

| Sub-Category | Definition | Example |
|--------------|------------|---------|
| `PERFORMANCE:Complexity` | Algorithmic inefficiency | O(n²) when O(n log n) possible |
| `PERFORMANCE:Resource` | Memory leaks, handle exhaustion | Unclosed file handles |
| `PERFORMANCE:Latency` | Blocking operations | Sync I/O on render thread |
| `PERFORMANCE:Cache` | Cache misses, false sharing | Column-major access in row-major data |

#### MAINTAINABILITY Sub-Categories

| Sub-Category | Definition | Example |
|--------------|------------|---------|
| `MAINTAINABILITY:Readability` | Confusing naming, structure | Single-letter variable names |
| `MAINTAINABILITY:Modularity` | Tight coupling, poor separation | God object pattern |
| `MAINTAINABILITY:Debt` | TODOs without tickets, hacks | `// FIXME: hack for demo` |
| `MAINTAINABILITY:Duplication` | Copy-paste code | Same logic in 3 places |

#### TESTABILITY Sub-Categories

| Sub-Category | Definition | Example |
|--------------|------------|---------|
| `TESTABILITY:Coverage` | Missing tests for new logic | New function with 0% coverage |
| `TESTABILITY:Design` | Flaky tests, testing implementation | Mocking private methods |
| `TESTABILITY:Scenarios` | Missing edge cases | No test for empty input |
| `TESTABILITY:Assertions` | Weak assertions | Only checking `is_ok()` |

### 2.3 Severity Levels (The "Action")

| Severity | Definition | Journal Equivalent | Required Action |
|----------|------------|-------------------|-----------------|
| **BLOCKING** | Fundamental flaw | Reject | **Stop the Line** - Cannot merge |
| **REQUIRED** | Significant issue | Major Revision | Must fix before approval |
| **SUGGESTION** | Minor improvement | Minor Revision | Should fix, no re-review needed |
| **DISCUSSION** | Future improvement | Editorial Note | Optional, create Kaizen ticket |
| **PRAISE** | Excellent work | Highlight | Recognition for quality |

### 2.4 Decision Matrix

| Category | BLOCKING | REQUIRED | SUGGESTION |
|----------|----------|----------|------------|
| CORRECTNESS | Logic errors, data corruption | Edge case bugs | Naming clarity |
| SECURITY | Vulnerabilities (OWASP Top 10) | Hardened defaults | Defense in depth |
| PERFORMANCE | 10x+ regression | 2x regression | Minor inefficiency |
| MAINTAINABILITY | Unmaintainable code | Refactoring needed | Style preference |
| REPRODUCIBILITY | Cannot run locally | Missing instructions | Documentation gaps |
| TESTABILITY | 0% coverage on new code | < 80% coverage | Edge case coverage |

---

## 3. Evidence Requirements

**A reviewer cannot simply state an opinion; they must cite a source.**

### Types of Admissible Evidence

| Evidence Type | Description | Example |
|---------------|-------------|---------|
| **Specification** | Language/protocol spec | "Violates IEEE 754 Section 6.3" |
| **Documentation** | Internal docs, ADRs | "Contradicts ADR-0012" |
| **Empirical Data** | Benchmarks, profiles | "Benchmark shows 5x regression" |
| **Static Analysis** | Tool output | "Clippy lint rust-analyzer/E0001" |
| **Literature** | Papers, best practices | "Known issue per CVE-2024-XXXX" |
| **Code Reference** | Existing codebase patterns | "See `render.rs:142` for pattern" |

### Evidence Quality Hierarchy

1. **Empirical data** (benchmarks, profiles, logs) - Strongest
2. **Specification/standard violations** - Strong
3. **Static analysis tool output** - Strong
4. **Literature/best practice citations** - Moderate
5. **Internal documentation references** - Moderate
6. **Code pattern references** - Acceptable
7. **Personal experience** - Weakest (requires additional support)

---

## 4. Stop the Line Protocol (Andon)

### When to Stop

The line MUST be stopped for:

| Trigger | Category | Example |
|---------|----------|---------|
| Cannot reproduce | `REPRODUCIBILITY:BLOCKING` | Tests fail locally |
| Security vulnerability | `SECURITY:BLOCKING` | SQL injection possible |
| Data corruption risk | `CORRECTNESS:BLOCKING` | Race condition on shared state |
| Missing critical tests | `TESTABILITY:BLOCKING` | 0% coverage on security code |

### Andon Protocol

```
1. DETECT    → Identify the abnormality
2. STOP      → Mark review as "Request Changes"
3. SIGNAL    → Notify author immediately (don't let them discover later)
4. ANNOTATE  → Write [CATEGORY:BLOCKING] comment with evidence
5. SWARM     → Offer to pair program if fix is complex
```

### Stop the Line Template

```markdown
## STOP THE LINE

**Category**: [SECURITY:BLOCKING]
**Issue**: SQL injection vulnerability in user input handling
**Evidence**: Input from `request.params["id"]` passed directly to query without sanitization

### Impact
- Allows arbitrary database queries
- Potential data exfiltration
- OWASP A03:2021 - Injection

### Required Action
Parameterize the query using prepared statements.

### Resolution Path
Happy to pair on this if helpful. Ping me on Slack.
```

---

## 5. Example Annotations

### BLOCKING Examples

#### Security Vulnerability
```markdown
[SECURITY:Injection:BLOCKING]: User input passed directly to shell command.

```rust
let output = Command::new("sh")
    .arg("-c")
    .arg(format!("echo {}", user_input))  // VULNERABLE
    .output()?;
```

Evidence: CWE-78 (OS Command Injection). User-controlled `user_input` can execute arbitrary commands via shell metacharacters.

Fix: Use `Command::new("echo").arg(user_input)` to avoid shell interpretation.
```

#### Correctness Issue
```markdown
[CORRECTNESS:Concurrency:BLOCKING]: Race condition on `point_count` access.

```rust
// Thread 1                    // Thread 2
self.point_count += 1;         let count = self.point_count;
```

Evidence: Non-atomic read-modify-write on shared state violates Rust's memory model. Can cause data races and undefined behavior.

Fix: Use `AtomicUsize` or protect with `Mutex`.
```

### REQUIRED Examples

#### Performance Regression
```markdown
[PERFORMANCE:Complexity:REQUIRED]: O(n²) nested loop will timeout on target datasets.

```rust
for i in 0..points.len() {
    for j in 0..points.len() {  // O(n²)
        distances.push(distance(points[i], points[j]));
    }
}
```

Evidence: Benchmark on 100K points: 45 seconds. Target SLA: < 1 second.
Per spec, datasets up to 1M points must render in < 5s.

Fix: Use spatial index (R-tree/KD-tree) for O(n log n) or parallel processing.
```

#### Missing Tests
```markdown
[TESTABILITY:Coverage:REQUIRED]: New public API `Heatmap::render()` has 0% test coverage.

Evidence: `cargo tarpaulin` shows lines 142-189 uncovered. This function handles color interpolation which is critical for correctness.

Fix: Add unit tests covering:
- Empty data input
- Single cell
- Color scale boundaries
- NaN handling
```

### SUGGESTION Examples

#### Readability
```markdown
[MAINTAINABILITY:Readability:SUGGESTION]: Variable `x` does not convey intent.

```rust
let x = points.iter().map(|p| p.0).collect::<Vec<_>>();
```

Evidence: Per style guide Section 3.2, use domain-specific names.

Suggestion: `let x_coordinates = points.iter().map(|p| p.x).collect();`
```

#### Minor Performance
```markdown
[PERFORMANCE:Cache:SUGGESTION]: Column-major access pattern in row-major array.

```rust
for col in 0..width {
    for row in 0..height {
        pixels[row * width + col] = color;  // Cache-unfriendly
    }
}
```

Evidence: Loop interchange benchmark shows 2.3x improvement for this pattern.

Suggestion: Swap loop order to `for row ... for col ...`
```

### PRAISE Examples

```markdown
[PRAISE]: Excellent use of `dispatch_binary_op!` macro for SIMD acceleration.
This follows the trueno pattern perfectly and will provide 4-8x speedup on AVX2.
The fallback path is also correct.
```

```markdown
[PRAISE]: Thorough property-based testing for color interpolation.
The proptest coverage of edge cases (NaN, infinity, denormals) is exactly what
safety-critical rendering code needs.
```

---

## 6. Reviewer Checklist

### Pre-Review Checklist

- [ ] Checked out the code locally
- [ ] Ran `cargo build` successfully
- [ ] Ran `cargo test` (3x for flakiness)
- [ ] Verified described behavior manually
- [ ] Ran `pmat quality-gate` locally

### During Review Checklist

- [ ] All comments use `[CATEGORY:SEVERITY]` format
- [ ] All comments include evidence
- [ ] BLOCKING issues use Andon protocol
- [ ] Looked for OWASP Top 10 issues
- [ ] Verified test coverage for new code
- [ ] Checked for performance regressions

### Post-Review Checklist

- [ ] Summarized findings in review comment
- [ ] Indicated LGTM or Request Changes
- [ ] For BLOCKING: notified author immediately
- [ ] For LGTM: verified all previous comments addressed

---

## 7. Author Response Protocol

### Addressing Comments

For each comment:

1. **Fix and acknowledge**: "Fixed in commit abc123"
2. **Provide counter-evidence**: "Benchmark shows X, see attachment"
3. **Request clarification**: "Could you elaborate on the concern?"
4. **Defer with ticket**: "Created ISSUE-123 for follow-up"

### Rebuttal Format

```markdown
**Rebuttal to [PERFORMANCE:Complexity:REQUIRED]**

Thank you for the review. I have counter-evidence:

The dataset in production is capped at 10K points (see `config.rs:42`).
Benchmark on 10K: 45ms (well under 200ms SLA).

The O(n log n) solution adds 200 LOC and a tree-sitter dependency.
Given the hard cap, I propose we defer optimization to ISSUE-456.

Attached: benchmark_results.png
```

---

## 8. Metrics and Continuous Improvement

### Review Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Defect Escape Rate | < 1% | Bugs found post-merge / total bugs |
| Review Cycle Time | < 24h | Time from review-ready to merge |
| Comment Density | 2-5/PR | Scientific annotations per PR |
| Stop the Line Rate | 5-15% | PRs blocked / total PRs |
| Rebuttal Success | 20-30% | Rebuttals accepted / total rebuttals |

### Kaizen Actions

After each escaped defect:
1. **Root cause analysis**: Why did review miss this?
2. **Systemic fix**: Can we automate detection?
3. **Checklist update**: Add to review checklist
4. **Training**: Share in calibration session

---

## References

1. Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
2. Fagan, M. E. (1976). "Design and code inspections." *IBM Systems Journal*.
3. Bacchelli, A., & Bird, C. (2013). "Expectations, outcomes, and challenges of modern code review." *ICSE '13*.
4. Toyota Production System principles: Genchi Genbutsu, Jidoka, Kaizen.
5. OWASP Top 10 (2021). Web Application Security Risks.
6. NASA Software Engineering Handbook (NASA-HDBK-2203).

---

*"In science, the purpose of peer review is not approval—it is verification."*
