Continuously improve contract enforcement across the entire PAIML sovereign stack. Measure, regenerate, inject, validate, and report — then fix root causes using five-whys analysis. Target: Kernel tier Grade A, Tool tier Grade A.

## Grading

| Grade | Score | Meaning |
|-------|-------|---------|
| A | >= 0.60 | Strong DbC — most bindings have domain-specific pre+post |
| B | >= 0.40 | Good coverage — majority E1+ |
| C | >= 0.25 | Moderate — wired but many E0 |
| D | >= 0.10 | Weak — low quality |
| F | < 0.10 | Minimal or no enforcement |

Tool tier uses penetration-only: A >= 90%, B >= 75%, C >= 50%.

## Process

### Phase 1: Measure Fleet State

```bash
pv kaizen --src-root /home/noah/src
```

Read the report: fleet grade, kernel tier grade + E2%, tool tier grade + pen%, per-repo grades. Identify F and D repos.

### Phase 2: Prioritize by Tier

**Kernel tier** (aprender, entrenar, realizar, trueno):
- Target: Grade A (score >= 0.60, E2 >= 60%)
- Focus: add postcondition call sites to upgrade E1→E2
- Only 9 postcondition macros exist: softmax, matmul, rmsnorm, cross_entropy, attention, layernorm, swiglu, rope, embedding_lookup

**Tool tier** (21 other repos):
- Target: Grade A (penetration >= 90%)
- Focus: add call sites for unbound functions
- E0 is acceptable — these repos don't do numerical computation

### Phase 3: Fix Root Causes (Five-Whys)

**Grade F (zero call sites):**
1. Does `src/generated_contracts.rs` exist? If not: `pv codegen contracts/ -o <repo>/src/generated_contracts.rs`
2. Is `#[macro_use] mod generated_contracts;` in `lib.rs`? If not: add it
3. For workspace crates, also check `crates/*/src/` — each subcrate needs its own `generated_contracts.rs` + `#[macro_use]`
4. Are there functions that match binding names? If not: binding.yaml is stale
5. Does the macro compile with the function's parameter types? If not: use `contract_pre_X!()` zero-arg form

**Grade D (low quality, E0-heavy):**
1. Check if `generated_contracts.rs` is stale (<100 lines or old): regenerate with `pv codegen`
2. YAML preconditions using `!config.is_empty()` → codegen can't map `config` to `_contract_input`. Change YAML to use `input.len() > 0` instead
3. Call sites passing non-slice types (bool, structs, scalars) → use `contract_pre_X!()` zero-arg form
4. After regeneration, E0 should become E1 for contracts with `input.len() > 0`, `x.is_finite()`, etc.

**Grade C→B (need postconditions):**
1. Check which `contract_post_*` macros exist in `generated_contracts.rs`
2. Find functions with `contract_pre_*` that have matching `contract_post_*`
3. Insert `contract_post_<eq>!(&result);` before return statements
4. Only for numeric return types (`Vec<f32>`, `&[f32]`, `f32`) — skip `Result<>`, `Tensor`, structs

### Phase 4: Apply Fixes

For each repo:

1. **Regenerate macros**: `pv codegen /home/noah/src/provable-contracts/contracts/ -o src/generated_contracts.rs`
2. **Add call sites** at function entry points (after early-return guards)
3. **Add postconditions** before return statements (kernel tier only)
4. **Compile check**: `cargo check` (or `cargo check -p <crate>` for workspaces)
5. **Fix compilation errors**: wrong types → zero-arg form, moved values → pass by reference
6. **Run tests**: `cargo test` to verify no regressions

### Phase 5: Re-Measure and Report

```bash
pv kaizen --src-root /home/noah/src
```

Report the grade changes:
```
Fleet:  C → B (0.37 → 0.45)
Kernel: B → A (0.43 → 0.62)
Tool:   B → A (86% → 91%)

Grade changes:
  realizar: C → B (added 15 postconditions)
  batuta: F → D (regenerated macros)
  ...
```

### Phase 6: Update Spec

Update Section 31 of `docs/specifications/pv-spec.md` with new measured state.

## Sovereign Stack Repos (25 bound repos)

**Kernel Tier** (mathematical contracts, quality matters):
- aprender (72 bindings, Grade D — needs more postconditions)
- entrenar (50 bindings, Grade A — 33 E2)
- realizar (100 bindings, Grade C — 22 E2, largest repo)
- trueno (17 bindings, Grade A — 9 E2)

**Tool Tier** (infrastructure wiring, penetration matters):
- Grade A: alimentar, trueno-rag
- Grade B: forjar, pacha, renacer, ruchy, simular, trueno-zram
- Grade D: depyler, presentar, rmedia, trueno-viz
- Grade F: batuta, bashrs, certeza, repartir, apr-model-qa-playbook

## Key Patterns

- Sibling path: `CARGO_MANIFEST_DIR/../provable-contracts/contracts/<crate>/binding.yaml`
- Macro inclusion: `#[macro_use] #[allow(unused_macros)] mod generated_contracts;`
- Guard placement: always insert `contract_pre_*!()` AFTER early-return guards
- Zero-arg fallback: `contract_pre_X!()` for types without `.len()` or `.is_finite()`
- Workspace crates: each subcrate needs its own copy of `generated_contracts.rs`
- Postcondition dereference: YAML `result >= 0.0` → codegen emits `*_contract_result >= 0.0`
- Feature-gated tests: `#[cfg(feature = "gpu")]` tests are Warning, not Error, in `pv lint`

## Do NOT

- Create git branches (work directly on main in each repo)
- Run `cargo kani` (too slow for fleet sweep)
- Modify YAML preconditions for numerical kernel contracts without understanding the math
- Skip `cargo check` after injection
- Inject into `#[test]` functions
- Push to remote without user confirmation
- Use `awk` dedup on YAML files (destroys duplicate precondition lines across equations)

## Push Frequency

Commit after every 1-3 repo fixes with descriptive messages. Format:
```
fix(<repo>): contract enforcement — <what changed> (Refs PMAT-NNN)
```
