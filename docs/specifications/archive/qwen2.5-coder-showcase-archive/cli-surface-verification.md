# Full CLI Surface Area Verification (Archived from Section 17)

> Archived from qwen2.5-coder-showcase-demo.md, Section 17 (lines 1888-1960).

## 17. Full CLI Surface Area Verification

### 17.1 Complete Subcommand Registry (49 Total)

**39 top-level commands:**

| # | Command | Category | Contract Gate | Showcase Test |
|---|---------|----------|---------------|---------------|
| 1 | `apr run` | Inference | Gated | Matrix cells 1-6 |
| 2 | `apr chat` | Inference | Gated | Matrix cells 7-12 |
| 3 | `apr serve` | Inference | Gated | Matrix cells 13-18 |
| 4 | `apr pull` | Provenance | Exempt | Step 1 of protocol |
| 5 | `apr import` | Provenance | Gated | Step 3 of protocol |
| 6 | `apr export` | Provenance | Gated | Step 4 of protocol |
| 7 | `apr convert` | Provenance | Gated | Section 2.1 |
| 8 | `apr inspect` | Diagnostic | Exempt | Section 2.3 |
| 9 | `apr debug` | Diagnostic | Exempt | Section 2.3 |
| 10 | `apr validate` | Diagnostic | Exempt | Section 2.3 |
| 11 | `apr lint` | Diagnostic | Exempt | Section 2.3 |
| 12 | `apr check` | Pipeline | Gated | Section 3 |
| 13 | `apr tensors` | Diagnostic | Exempt | Section 2.3 |
| 14 | `apr hex` | Diagnostic | Exempt | Section 2.3 |
| 15 | `apr tree` | Diagnostic | Exempt | Section 2.3 |
| 16 | `apr flow` | Diagnostic | Exempt | Section 2.3 |
| 17 | `apr diff` | Diagnostic | Exempt | Section 2.3 |
| 18 | `apr compare-hf` | Diagnostic | Gated | Section 2.3 |
| 19 | `apr trace` | Analysis | Gated | Section 2.4 |
| 20 | `apr bench` | Analysis | Gated | Section 2.4 |
| 21 | `apr eval` | Analysis | Gated | Section 2.4 |
| 22 | `apr profile` | Analysis | Gated | Section 2.4 |
| 23 | `apr cbtop` | Analysis | Gated | Section 2.4 |
| 24 | `apr qa` | Quality | Exempt | Section 2.4 |
| 25 | `apr showcase` | Quality | Exempt | Section 2.4 |
| 26 | `apr list` | Management | Exempt | Section 2.5 |
| 27 | `apr rm` | Management | Exempt | Section 2.5 |
| 28 | `apr publish` | Management | Exempt | Section 2.5 |
| 29 | `apr oracle` | Contract | Exempt | Section 16.4 |
| 30 | `apr tune` | Advanced | Exempt | Section 2.6 |
| 31 | `apr parity` | Diagnostic | Exempt | GPU/CPU divergence check |
| 32 | `apr merge` | Advanced | Gated | Section 2.6 |
| 33 | `apr canary` | Regression | Exempt | Section 2.6 |
| 34 | `apr probar` | Visual Test | Gated | Section 2.6 |
| 35 | `apr explain` | Help | Exempt | Section 2.6 |
| 36 | `apr tui` | Interactive | Gated | Section 2.6 |
| 37 | `apr rosetta` | Conversion | Mixed | Section 2.7 |
| 38 | `apr ptx-map` | Diagnostic | Exempt | Section 13.10 |
| 39 | `apr ptx` | Analysis | Exempt | Section 13.11 |

**10 nested subcommands (under `rosetta` and `canary`):**

| # | Command | Parent | Showcase Test |
|---|---------|--------|---------------|
| 40 | `apr rosetta inspect` | rosetta | Section 2.7 |
| 41 | `apr rosetta convert` | rosetta | Section 2.7 |
| 42 | `apr rosetta chain` | rosetta | Section 2.7 |
| 43 | `apr rosetta verify` | rosetta | Section 2.7 |
| 44 | `apr rosetta compare-inference` | rosetta | Section 0.6 |
| 45 | `apr rosetta diff-tensors` | rosetta | Section 2.7 |
| 46 | `apr rosetta fingerprint` | rosetta | Section 2.7 |
| 47 | `apr rosetta validate-stats` | rosetta | Section 2.7 |
| 48 | `apr canary create` | canary | Section 2.6 |
| 49 | `apr canary check` | canary | Section 2.6 |

### 17.2 CLI Surface Falsification Gates (F-SURFACE-*)

| ID | Prediction | Test | Expected | Status |
|----|-----------|------|----------|--------|
| F-SURFACE-001 | All 39 top-level commands exist | `apr <cmd> --help` for each | All 39 return help text | **Pass** (39 variants in Commands enum confirmed, including `parity`) |
| F-SURFACE-002 | All 10 nested commands exist | `apr rosetta <sub> --help`, `apr canary <sub> --help` | All 10 return help text | **Pass** (8 rosetta + 2 canary = 10 nested verified) |
| F-SURFACE-003 | No undocumented commands | `apr --help` lists all commands | Count matches 39 | **Pass** (all enum variants documented in spec) |
| F-SURFACE-004 | Every command referenced in spec | grep this spec for each command | 49/49 referenced | **Pass** (all 39 top-level + 10 nested found in spec) |
| F-SURFACE-005 | Contract classification matches code | Compare table above vs `extract_model_paths()` | 17 gated, rest exempt | **Pass** (action vs diagnostic classification verified) |
