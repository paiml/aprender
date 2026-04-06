# Design by Contract Falsification Sweep Report
## /home/noah/src/batuta/src/

**Date:** 2026-02-23  
**Scope:** Non-test Rust source files only  
**Focus:** Contract violations, silent defaults, architecture heuristics, hardcoded constants

---

## CRITICAL FINDINGS

### 1. Architecture String Heuristics (HIGH SEVERITY)

**File:** `/home/noah/src/batuta/src/serve/templates.rs:85-104`  
**Pattern:** Model name string matching without validation  
**Code:**
```rust
pub fn from_model_name(name: &str) -> Self {
    let lower = name.to_lowercase();
    if lower.contains("llama-2") || lower.contains("llama2") {
        Self::Llama2
    } else if lower.contains("mistral") || lower.contains("mixtral") {
        Self::Mistral
    } else if lower.contains("chatml") || lower.contains("openhermes") {
        Self::ChatML
    } else if lower.contains("alpaca") {
        Self::Alpaca
    } else if lower.contains("vicuna") {
        Self::Vicuna
    } else {
        Self::Raw  // SILENT DEFAULT
    }
}
```

**Issues:**
- Line 103: Falls back to `Self::Raw` for unknown models (silent default)
- String heuristics brittle: "llama-2-uncensored" ≠ "llama-2"
- No validation that name matches architecture metadata
- Substring matching causes collisions: "mistral" matches "mistral-instruct", "mistral-large", "mistral-moe"
- **Runtime Contract:** Template format MUST match model's actual architecture
- **Test vs Production:** Tests use exact names; production sees user-provided names

**Severity:** HIGH  
**Category:** Contract Violation + Silent Default  
**Recommendation:** Validate against known architectures or require explicit schema from model metadata

---

### 2. Silent Default on Model Workspace Version (MEDIUM SEVERITY)

**File:** `/home/noah/src/batuta/src/oracle/local_workspace.rs:282`  
**Pattern:** `unwrap_or("0.0.0")` on workspace version  
**Code:**
```rust
let version = workspace
    .get("package")
    .and_then(|p| p.get("version"))
    .and_then(|v| v.as_str())
    .unwrap_or("0.0.0")  // <-- SILENT DEFAULT
    .to_string();
```

**Issues:**
- Falls back to "0.0.0" if version field missing
- Silently masks invalid/missing workspace config
- Could cause version comparison failures downstream
- **Runtime Contract:** Version MUST be parsed from valid semver in Cargo.toml
- **Test vs Production:** Tests may use minimal Cargo.toml; production uses real workspaces

**Severity:** MEDIUM  
**Category:** Silent Defensive Default  
**Recommendation:** Return error on missing version; log warning if fallback occurs

---

### 3. Circuit Breaker Date Calculation Silent Default (MEDIUM SEVERITY)

**File:** `/home/noah/src/batuta/src/serve/circuit_breaker.rs:118`  
**Pattern:** `unwrap_or_default()` on SystemTime duration  
**Code:**
```rust
pub fn current_date() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()  // <-- RETURNS 0 SECS ON ERROR
        .as_secs();
    // Simple date calculation (not timezone aware)
    let days = now / 86400;
    let year = 1970 + (days / 365);  // Computes epoch (1970-01-01)
    ...
}
```

**Issues:**
- `unwrap_or_default()` returns Duration(0) on SystemTime error (rare but possible)
- Silently returns 1970-01-01 in rare failure cases
- No indication to caller that date calculation failed
- **Runtime Contract:** Date MUST reflect current wall-clock time
- **Impact:** Circuit breaker thinks we're in 1970 → usage records misaligned

**Severity:** MEDIUM  
**Category:** Silent Default on Critical State  
**Recommendation:** Return `Result<String>` and propagate error; abort if current_date() fails

---

### 4. Hardcoded Token IDs and Dimensions (MEDIUM SEVERITY)

**File:** `/home/noah/src/batuta/src/pacha/handlers.rs:139, 145`  
**Pattern:** Hardcoded vocabulary size and tensor dimensions  
**Code:**
```rust
println!("  Vocab:      32000");  // Line 139

let tensors = [
    ("token_embd.weight", "[32000, 4096]", "Q4_K"),
    ("blk.0.attn_q.weight", "[4096, 4096]", "Q4_K"),
    ...
];
```

**Issues:**
- Hardcoded vocab=32000 (LLaMA default, not Qwen/Mistral)
- Hardcoded hidden_dim=4096 (7B model size)
- Only showing first 10 tensors for all models
- No validation that these match actual model metadata
- **Runtime Contract:** Tensor shapes MUST match model's actual config
- **Test vs Production:** Works for LLaMA-7B; fails for 13B, Qwen, Mistral, etc.

**Severity:** MEDIUM  
**Category:** Hardcoded Architecture Constants  
**Recommendation:** Load from model.config_json or ModelMetadata; validate before printing

---

### 5. Unwrap_or with Numeric Defaults (MEDIUM SEVERITY)

**File:** `/home/noah/src/batuta/src/audit.rs:342, 552`  
**Pattern:** `unwrap_or(0)` on model-related counts  
**Code:**
```rust
src/audit.rs:342:            .unwrap_or(0);
src/audit.rs:552:            .unwrap_or(0)
```

**Issues:**
- Missing counts silently default to 0
- No indication whether 0 is "no findings" or "parse error"
- Downstream aggregations can't distinguish missing from empty
- **Runtime Contract:** Counts MUST come from valid audit data

**Severity:** MEDIUM  
**Category:** Numeric Silent Default  
**Recommendation:** Use `expect("count must exist")` or return error; log if key missing

---

### 6. Graph Depth Calculation Silent Default (LOW SEVERITY)

**File:** `/home/noah/src/batuta/src/tui/graph_layout.rs:342`  
**Pattern:** `unwrap_or(1)` on max depth  
**Code:**
```rust
let max_depth = visited.values().max().copied().unwrap_or(1);
```

**Issues:**
- Falls back to depth=1 if no nodes visited
- Could mask graph construction failures
- Layout calculations assume depth ≥ 1
- **Runtime Contract:** Depth MUST reflect actual node hierarchy

**Severity:** LOW  
**Category:** Silent Default on Layout Computation  
**Recommendation:** Assert depth > 0; return error if graph empty

---

### 7. Model Parity Claim Parsing With Unwrap (MEDIUM SEVERITY)

**File:** `/home/noah/src/batuta/src/bug_hunter/model_parity.rs:134`  
**Pattern:** `unwrap()` on claim header parsing  
**Code:**
```rust
let claim_header = line.strip_prefix("### Claim ");
if claim_header.is_none() {
    continue;
}
let header = claim_header.unwrap();  // <-- UNWRAP AFTER is_none() CHECK
```

**Issues:**
- Unnecessary `unwrap()` after checked `is_none()`
- Should use `if let Some(header)` instead
- Pattern is safe but verbose; makes code harder to audit
- **Runtime Contract:** Claim headers MUST be present and parseable

**Severity:** MEDIUM  
**Category:** Defensive Unwrap Pattern  
**Recommendation:** Use `if let Some(header) = claim_header` pattern (idiomatic)

---

### 8. HuggingFace Hub Author Extraction Silent Default (LOW SEVERITY)

**File:** `/home/noah/src/batuta/src/hf/hub_client.rs:83`  
**Pattern:** `unwrap_or("unknown")` on model author  
**Code:**
```rust
let author = id_str.split('/').next().unwrap_or("unknown").to_string();
```

**Issues:**
- Falls back to "unknown" author if model ID has no '/'
- Model ID format: "author/model-name" but invalid IDs default silently
- **Runtime Contract:** Model ID MUST be "author/model" format

**Severity:** LOW  
**Category:** Silent Default on Metadata  
**Recommendation:** Validate model ID format; return error if invalid

---

### 9. Content Line Extraction With Silent Default (LOW SEVERITY)

**File:** `/home/noah/src/batuta/src/bug_hunter/localization.rs:249`  
**Pattern:** `unwrap_or("")` on line content  
**Code:**
```rust
let line_content = content.lines().nth(line.saturating_sub(1)).unwrap_or("");
```

**Issues:**
- Returns empty string if line number out of bounds
- No indication that line was missing
- Could silently skip error messages for invalid line numbers
- **Runtime Contract:** Line number MUST be within file bounds

**Severity:** LOW  
**Category:** Silent Default on Content Access  
**Recommendation:** Return error if line out of bounds; validate line number

---

### 10. Blame Author Parsing Silent Defaults (LOW SEVERITY)

**File:** `/home/noah/src/batuta/src/bug_hunter/blame.rs:131, 136`  
**Pattern:** Multiple `unwrap_or()` on blame data  
**Code:**
```rust
author = line.strip_prefix("author ").unwrap_or("").to_string();  // Line 131
// Later...
.unwrap_or("0")  // Line 136
```

**Issues:**
- Empty author string if "author " prefix missing
- "0" default for numeric fields (probably timestamp)
- Could mask git blame parsing failures
- **Runtime Contract:** Blame data MUST have author and timestamp

**Severity:** LOW  
**Category:** Multiple Silent Defaults in Parsing  
**Recommendation:** Validate blame format; return error on parse failures

---

### 11. Communities Count Silent Default (LOW SEVERITY)

**File:** `/home/noah/src/batuta/src/tui/graph_analytics.rs:207`  
**Pattern:** `unwrap_or(0)` on communities count  
**Code:**
```rust
let num_communities = communities.values().max().map(|&m| m + 1).unwrap_or(0);
```

**Issues:**
- Returns 0 if no communities found (could mask empty graph)
- No indication whether 0 is "no communities" or "computation failed"
- **Runtime Contract:** Communities count MUST reflect actual graph structure

**Severity:** LOW  
**Category:** Silent Default on Analytics  
**Recommendation:** Validate graph has nodes before computing communities

---

## SUMMARY TABLE

| File | Line(s) | Pattern | Severity | Category | Test Risk |
|------|---------|---------|----------|----------|-----------|
| serve/templates.rs | 103 | `Self::Raw` fallback | HIGH | Arch Heuristic | YES |
| oracle/local_workspace.rs | 282 | `"0.0.0"` version | MEDIUM | Silent Default | YES |
| serve/circuit_breaker.rs | 118 | `unwrap_or_default()` date | MEDIUM | Critical State | RARE |
| pacha/handlers.rs | 139, 145 | Hardcoded dims/vocab | MEDIUM | Constants | YES |
| audit.rs | 342, 552 | `unwrap_or(0)` counts | MEDIUM | Numeric Default | YES |
| tui/graph_layout.rs | 342 | `unwrap_or(1)` depth | LOW | Layout Computation | NO |
| bug_hunter/model_parity.rs | 134 | `unwrap()` after check | MEDIUM | Defensive Pattern | NO |
| hf/hub_client.rs | 83 | `"unknown"` author | LOW | Metadata | RARE |
| bug_hunter/localization.rs | 249 | `""` line content | LOW | Content Access | RARE |
| bug_hunter/blame.rs | 131, 136 | Multiple defaults | LOW | Parse Failures | RARE |
| tui/graph_analytics.rs | 207 | `unwrap_or(0)` | LOW | Analytics | NO |

---

## RECOMMENDATIONS (Priority Order)

### P0 (Fix Immediately)
1. **serve/templates.rs:103** — Remove `Self::Raw` fallback; return `Result<Self>` or require explicit architecture
2. **pacha/handlers.rs** — Load tensor shapes from actual model metadata, not hardcoded

### P1 (Fix Before Release)
3. **oracle/local_workspace.rs:282** — Return error on missing version; log warning
4. **serve/circuit_breaker.rs:118** — Return `Result<String>`, propagate errors
5. **audit.rs** — Use `expect()` with message; log missing keys

### P2 (Clean Up)
6. **bug_hunter/model_parity.rs:134** — Use `if let Some` pattern
7. **bug_hunter/blame.rs, localization.rs** — Validate parse formats; return errors

---

## Notes for Code Review

- **No test code included** in this scan (per request)
- **Architecture heuristics are the biggest risk** — string matching is fundamentally fragile
- **Silent defaults on model state are critical** — circuit breaker date, workspace version, tensor dimensions
- **Most violations are defensive but unhelpful** — would be clearer to fail fast with errors
- **Recommend contract annotations** before fixing (see aprender/CLAUDE.md LAYOUT-001/002 for example)

