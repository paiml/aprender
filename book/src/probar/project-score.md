# Project Testing Score

![Project Score Coverage](../assets/coverage_combined.png)

The `probador serve score` command generates a comprehensive 115-point score evaluating how thoroughly your project implements probar's testing capabilities across 10 categories.

## Score Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    115-POINT SCORING SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Score Categories                          │ │
│  │                                                               │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │ │
│  │  │   Runtime      │  │   Playbook     │  │    Pixel       │  │ │
│  │  │   Health       │  │   Coverage     │  │   Testing      │  │ │
│  │  │   (15 pts)     │  │   (15 pts)     │  │   (13 pts)     │  │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘  │ │
│  │                                                               │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │ │
│  │  │     GUI        │  │  Performance   │  │ Load Testing   │  │ │
│  │  │  Interaction   │  │  Benchmarks    │  │   Config       │  │ │
│  │  │   (13 pts)     │  │   (14 pts)     │  │   (10 pts)     │  │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘  │ │
│  │                                                               │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  │ │
│  │  │ Deterministic  │  │ Cross-Browser  │  │ Accessibility  │  │ │
│  │  │    Replay      │  │   Testing      │  │   Testing      │  │ │
│  │  │   (10 pts)     │  │   (10 pts)     │  │   (10 pts)     │  │ │
│  │  └────────────────┘  └────────────────┘  └────────────────┘  │ │
│  │                                                               │ │
│  │  ┌────────────────┐                                          │ │
│  │  │ Documentation  │   Note: Runtime Health gates grade       │ │
│  │  │    Quality     │   caps (failures cap at C grade)        │ │
│  │  │    (5 pts)     │                                          │ │
│  │  └────────────────┘                                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  Grade: A (90%+), B (80-89%), C (70-79%), D (60-69%), F (<60%)  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Generate score for current directory
probador serve score

# With detailed breakdown
probador serve score --verbose

# Set minimum threshold (CI gate)
probador serve score --min 80

# Output as JSON
probador serve score --format json

# Generate binary report (view with TUI)
probador serve score --report score-report.msgpack
```

## Score Output

```
PROJECT TESTING SCORE: demos/realtime-transcription
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Score: 85/115 (74%, B)

┌─────────────────────┬────────┬────────┬─────────────────────────────────┐
│ Category            │ Score  │ Max    │ Status                          │
├─────────────────────┼────────┼────────┼─────────────────────────────────┤
│ Runtime Health      │ 15/15  │ 15     │ ✓ WASM loads, no JS errors      │
│ Playbook Coverage   │ 12/15  │ 15     │ ⚠ Missing: error state coverage │
│ Pixel Testing       │ 10/13  │ 13     │ ⚠ Missing: error state snapshot │
│ GUI Interaction     │ 10/13  │ 13     │ ⚠ Missing: keyboard navigation  │
│ Performance         │ 14/14  │ 14     │ ✓ All benchmarks defined        │
│ Load Testing        │ 8/10   │ 10     │ ⚠ No sustained load config      │
│ Deterministic Replay│ 8/10   │ 10     │ ⚠ No edge case recordings       │
│ Cross-Browser       │ 5/10   │ 10     │ ✗ Only Chrome tested            │
│ Accessibility       │ 3/10   │ 10     │ ✗ No ARIA labels tested         │
│ Documentation       │ 0/5    │ 5      │ ✗ Missing test docs             │
└─────────────────────┴────────┴────────┴─────────────────────────────────┘

Grade Scale: A (90%+), B (80-89%), C (70-79%), D (60-69%), F (<60%)

Top 3 Recommendations:
1. Add Firefox/Safari to cross-browser matrix (+5 points)
2. Add ARIA label assertions to GUI tests (+4 points)
3. Add tests/README.md documentation (+5 points)

Run `probador serve score --verbose` for detailed breakdown.
```

## Scoring Categories

### Runtime Health (15 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| WASM loads successfully | 5 | Module instantiation without errors |
| No JS console errors | 4 | Zero uncaught exceptions |
| No memory leaks | 3 | Stable memory after warm-up |
| Graceful error handling | 3 | Errors caught and reported |

### Playbook Coverage (15 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| Playbook exists | 4 | `playbooks/*.yaml` present |
| All states defined | 4 | States match actual UI states |
| Invariants per state | 4 | At least 1 invariant per state |
| Forbidden transitions | 3 | Edge cases documented |

### Pixel Testing (13 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| Baseline snapshots exist | 4 | `snapshots/*.png` present |
| Coverage of states | 4 | Snapshots for 80%+ of states |
| Responsive variants | 3 | Mobile/tablet/desktop snapshots |
| Dark mode variants | 2 | Theme-aware snapshots |

### GUI Interaction Testing (13 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| Click handlers tested | 4 | All buttons have click tests |
| Form inputs tested | 4 | All inputs have validation tests |
| Keyboard navigation | 3 | Tab order and shortcuts tested |
| Touch events | 2 | Swipe/pinch gestures (if applicable) |

### Performance Benchmarks (14 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| RTF target defined | 5 | `performance.rtf_target` in playbook |
| Memory threshold | 4 | `performance.max_memory_mb` defined |
| Latency targets | 3 | p95/p99 latency assertions |
| Baseline file exists | 2 | `baseline.json` present |

### Load Testing (10 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| Load test config exists | 3 | `load_test.yaml` or equivalent |
| Concurrent user targets | 3 | Defined user load levels |
| Sustained load duration | 2 | Tests run for adequate duration |
| Resource monitoring | 2 | CPU/memory tracked during load |

### Deterministic Replay (10 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| Happy path recording | 4 | Main user flow recorded |
| Error path recordings | 3 | Error scenarios captured |
| Edge case recordings | 3 | Boundary conditions recorded |

### Cross-Browser Testing (10 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| Chrome tested | 3 | Chromium-based browser in matrix |
| Firefox tested | 3 | Gecko engine in matrix |
| Safari/WebKit tested | 3 | WebKit engine in matrix |
| Mobile browser tested | 1 | iOS Safari or Chrome Android |

### Accessibility Testing (10 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| ARIA labels | 3 | Interactive elements have labels |
| Color contrast | 3 | WCAG AA contrast ratios |
| Screen reader flow | 2 | Logical reading order |
| Focus indicators | 2 | Visible focus states |

### Documentation (5 points)

| Criterion | Points | Measurement |
|-----------|--------|-------------|
| Test README exists | 2 | `tests/README.md` present |
| Test rationale documented | 2 | Why, not just what |
| Running instructions | 1 | Clear setup/execution steps |

## CI/CD Integration

Use score as a quality gate in CI:

```yaml
# .github/workflows/test-score.yml
name: Test Score Gate
on: [push, pull_request]

jobs:
  score:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install probador
        run: cargo install probador

      - name: Check test score
        run: probador serve score --min 80 --format json > score.json

      - name: Upload score artifact
        uses: actions/upload-artifact@v4
        with:
          name: test-score
          path: score.json

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const score = require('./score.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Test Score: ${score.total}/${score.max} (${score.grade})\n\n${score.summary}`
            });
```

## Score History

Track score over time:

```bash
# Append to history file
probador serve score --history scores.jsonl

# View trend
probador serve score --trend
```

### Trend Output

```
SCORE TREND: demos/realtime-transcription
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     100 ┤
      90 ┤                              ╭──
      80 ┤                    ╭─────────╯
      70 ┤          ╭────────╯
      60 ┤    ╭─────╯
      50 ┤────╯
      40 ┤
         └────────────────────────────────
         Dec 1   Dec 5   Dec 10   Dec 14

Current: 73/100 (+8 from last week)
Target:  80/100 by Dec 21
```

## CLI Reference

```bash
probador serve score [OPTIONS] [PATH]

Arguments:
  [PATH]  Project directory [default: .]

Options:
      --verbose           Show detailed breakdown
      --format <FORMAT>   Output format (console, json)
      --min <SCORE>       Minimum required score (exit non-zero if below)
      --report <FILE>     Generate HTML report
      --history <FILE>    Append to JSONL history file
      --trend             Show score trend chart
  -h, --help              Print help
```

## Programmatic API

```rust
use probador::score::{ProjectScore, calculate_score};

let score = calculate_score("./demos/realtime-transcription")?;

println!("Total: {}/{} ({})", score.total, score.max, score.grade);

for category in &score.categories {
    println!("{}: {}/{}", category.name, category.score, category.max);
}

for rec in &score.recommendations {
    println!("{}. {} (+{} points)", rec.priority, rec.action, rec.potential_points);
}
```

## Improving Your Score

### Quick Wins (Low Effort, High Points)

1. **Add a playbook** - 5 points for just having one
2. **Create baseline snapshots** - 5 points for visual regression
3. **Add Chrome to test matrix** - 3 points

### Medium Effort

1. **Define invariants** - 5 points for state validation
2. **Add keyboard tests** - 3 points for accessibility
3. **Record happy path** - 4 points for replay testing

### High Effort, High Value

1. **Cross-browser testing** - Up to 10 points
2. **Full accessibility audit** - Up to 10 points
3. **Complete state coverage** - Up to 20 points

## Best Practices

1. **Run score regularly** - Track progress over time
2. **Set minimum thresholds** - Prevent quality regression
3. **Focus on recommendations** - Prioritized by impact
4. **Review in PRs** - Comment score changes on pull requests
5. **Celebrate milestones** - Team visibility on improvements
