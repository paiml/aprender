# 31. Kaizen Fleet Enforcement

## Motivation

v2.4.0 produced 540 domain-specific assertions across 170+ contracts,
but only 27 call sites existed across 25 repos. Fleet enforcement score
was ~1.5%. The bottleneck was not contract quality — it was call site
injection. Manual dogfooding (switching to each repo, regenerating,
fixing compilation) took hours per cycle.

**Kaizen** (改善, continuous improvement) automates this: a single
command measures, regenerates, injects, validates, and reports across
the entire sovereign stack.

## The Fleet

The PAIML sovereign stack comprises 40 Rust crates with contract bindings
(41 binding directories, 40 with resolvable sibling source):

| Repo | Domain | Binding Count |
|------|--------|---------------|
| aprender | ML inference + apr-cli | 284 |
| realizar | quantization + GPU | 100 |
| entrenar | ML training | 50 |
| trueno | SIMD tensor ops | 38 |
| decy | code analysis | 31 |
| rurl | HTTP client | 24 |
| duende | daemon orchestration | 23 |
| pmcp | MCP protocol SDK | 23 |
| batuta | agent orchestration | 20 |
| ruchy | bytecode VM | 20 |
| simular | simulation framework | 20 |
| renacer | golden tracing | 20 |
| trueno-rag | RAG pipeline | 20 |
| trueno-zram | ZRAM compression | 20 |
| presentar | TUI framework | 20 |
| depyler | Python→Rust transpiler | 18 |
| bashrs | shell/Make linting | 16 |
| forjar | secret management | 13 |
| rmedia | media processing | 13 |
| pmat | quality analysis (CLI) | 12 |
| probar | property testing | 13 |
| 19 others | various | ~160 |

Each repo follows the sibling-path convention:
`CARGO_MANIFEST_DIR/../provable-contracts/contracts/<crate>/binding.yaml`

**v2.6.0 addition:** The `source_dir` field in binding.yaml allows repos
whose contract directory name differs from the source directory name:

```yaml
# contracts/pmat/binding.yaml
version: 1.0.0
target_crate: pmat
source_dir: paiml-mcp-agent-toolkit  # resolves to ../paiml-mcp-agent-toolkit/
bindings: [...]
```

When present, `source_dir` takes priority over the direct name match.
This enables pmat (source in `paiml-mcp-agent-toolkit/`) and pmcp
(source in `rust-mcp-sdk/`) to participate in kaizen fleet scans.

## The Kaizen Loop

```
pv kaizen [--fleet | --repo <name>] [--dry-run] [--fix]
```

**Phase 1: Measure** (read-only)
```
For each repo in fleet:
  1. Parse contracts/<repo>/binding.yaml → binding count
  2. Scan <repo>/src/ for contract_pre_*/contract_post_* call sites
  3. Read <repo>/src/generated_contracts.rs → classify E0/E1/E2
  4. Compute enforcement score = penetration × quality
```

**Phase 2: Codegen** (regenerate macros)
```
For each repo in fleet:
  1. pv codegen contracts/ -o <repo>/src/generated_contracts.rs
  2. Diff old vs new → report assertion count changes
```

**Phase 3: Inject** (insert call sites at function entry/exit)
```
For each binding with zero call sites:
  1. Locate the bound function in <repo>/src/ via fn signature
  2. Parse function body (find opening brace)
  3. Insert contract_pre_<macro>!(<first_param>) after opening brace
  4. If postcondition macro exists, insert contract_post_<macro>!(<result>)
     before return expressions
```

**Phase 4: Validate** (compile check)
```
For each modified repo:
  1. cargo check --message-format=short
  2. On failure: revert injection, log error, continue
  3. On success: report new enforcement level
```

**Phase 5: Report** (aggregate fleet status)
```
Fleet Enforcement Report
========================
  Repos:              40
  Total bindings:     678
  Call sites:         424
  Penetration:        62.5%

  E0 (generic):       54
  E1 (domain pre):    137
  E2 (pre + post):    233

  Assertions:         14,436
  Enforcement:        0.4527 (Grade B)

  Tiered:
    Kernel (4 repos):  Grade F — 34/283 sites, E2 44%, pen 12.0%
    Tool (36 repos):   Grade A — 390/395 sites, pen 98.7%
```

## Injection Strategy

Call site injection is AST-aware but lightweight — it uses line-level
heuristics rather than a full Rust parser:

1. **Find the function**: grep for `fn <binding_function_name>` in
   `<repo>/src/**/*.rs`, skip `generated_contracts.rs` and test files
2. **Find the opening brace**: scan forward from `fn` to the first `{`
3. **Choose the macro**: map binding's contract stem + equation to
   `contract_pre_<macro_name>`
4. **Choose the argument**: use the first parameter name from the `fn`
   signature (heuristic: first identifier after `(` that isn't `&`,
   `mut`, or `self`)
5. **Insert**: add `contract_pre_<macro>!(<arg>);` on the line after `{`
6. **Guard placement**: if the function has early-return guards
   (`if <cond> { return ... }`), insert after all guards

**What injection does NOT do:**
- Does not modify function signatures
- Does not add `use` statements (macros are `#[macro_use]`)
- Does not inject into `#[test]` functions
- Does not inject if a call site already exists

## Enforcement Grading

`pv kaizen` reports letter grades per-repo and per-tier:

| Grade | Score | Meaning |
|-------|-------|---------|
| **A** | >= 0.60 | Strong DbC — most bindings have domain-specific pre+post |
| **B** | >= 0.40 | Good coverage — majority E1+, infrastructure solid |
| **C** | >= 0.25 | Moderate — wired but many E0 or low penetration |
| **D** | >= 0.10 | Weak — call sites exist but low quality |
| **F** | < 0.10 | Minimal or no enforcement |

Tool-tier repos use penetration-only grading (E0 is acceptable for
non-numerical code):

| Grade | Penetration | Meaning |
|-------|-------------|---------|
| **A** | >= 90% | Nearly all bindings have call sites |
| **B** | >= 75% | Good wiring |
| **C** | >= 50% | Partial |
| **D** | >= 25% | Sparse |
| **F** | < 25% | Not wired |

## Tiered Scoring

The fleet is split into two tiers with different quality expectations:

**Kernel tier** (aprender, entrenar, realizar, trueno): Mathematical
contracts with real invariants (finiteness, shape, bounds). Quality
metric: `penetration × quality` where E0=0.1, E1=0.5, E2=1.0.
Target: Grade A (score >= 0.60, E2 >= 60%).

**Tool tier** (36 other repos): Infrastructure wiring. E0 is acceptable
because tool functions pass strings/structs, not numeric slices.
Target: Grade A (penetration >= 90%).

## Enforcement Quality Targets

| Milestone | Fleet | Kernel | Tool | How |
|-----------|-------|--------|------|-----|
| v2.3.0 | F (0.015) | — | — | Manual dogfooding |
| v2.4.0 | F (0.019) | — | — | Codegen fix + trueno injection |
| v2.4.1 | D (0.178) | — | — | `/kaizen` fleet sweep |
| v2.4.2 | C (0.258) | — | — | Postconditions + E1 classifier |
| v2.4.3 | C (0.374) | B (0.43) | B (86%) | YAML rewrite + tiered grading |
| v2.4.4 | B (0.590) | A | A (93%) | realizar 74 new sites, E0→E1 fleet-wide |
| **v2.6.0** | **B (0.453)** | **F (12%)** | **A (98.7%)** | **40-repo fleet (+15), source_dir, postconditions, DBC lifecycle** |
| v3.0.0 | A (0.65) | A (0.80) | A (95%) | E2 for remaining P1/P2 contracts |

> **Note:** v2.6.0 fleet score dropped from v2.4.4's 0.590 because the
> fleet expanded from 25→40 repos. The 15 new repos (including pmat, pmcp,
> probar, rurl, duende, and others) mostly have bindings but few call sites
> yet, diluting the penetration rate. Tool tier remains Grade A (98.7%).
> Kernel tier dropped to F because realized/aprender call site injection
> from v2.4.4 was not persisted in the new repo checkouts.

**v2.6.0 measured state (2026-04-04):**

```
Fleet: 40 repos, 678 bindings, 424 call sites
Penetration: 62.5%
Fleet enforcement: 0.4527 (Grade B)

Kernel (4 repos): Grade F — 34/283 sites, E2 44%, pen 12.0%
Tool (36 repos):  Grade A — 390/395 sites, pen 98.7%

E0 (generic):      54 call sites
E1 (domain pre):  137 call sites
E2 (pre+post):    233 call sites
Assertions:       14,436

Grade A repos (17): alimentar, bashrs, batuta, certeza, decy, depyler,
                    forjar, pacha, pepita, presentar, renacer, repartir,
                    rmedia, ruchy, simular, trueno-viz, trueno-zram
Grade C repos (1):  trueno
Grade F repos (15): aprender, apr-model-qa-playbook, copia, duende,
                    entrenar, faro, pmat, pmcp, probar, pzsh, rclean,
                    realizar, rurl, verificar, zenith
```

**Key improvements v2.4.4→v2.6.0:**
- Fleet expanded 25→40 repos via source_dir binding.yaml feature
- pmat now in kaizen (12 bindings, 11 E0 sites) via source_dir → paiml-mcp-agent-toolkit
- pmcp now in kaizen (23 bindings) via source_dir → rust-mcp-sdk
- 4 new binding stubs: cohete, manzana, nviwatch, promogen
- PV-AUD-003 eliminated: 3 entrenar contracts gained proof_obligations + falsification tests
- pmat infrastructure contracts: 8 postconditions added, 0 lint warnings (was 8)
- apr-cli-v1 score: D→C (0.55→0.75) with 6 new bindings to aprender registry
- 146 book pages regenerated

## CLI Reference

```
pv kaizen [OPTIONS]

OPTIONS:
    --src-root <PATH>    Root directory for sibling repos (default: ../)
    --repo <NAME>        Run for a single repo only
    --dry-run            Measure and report only (default)
    --codegen            Regenerate generated_contracts.rs in each repo
    --fix                Inject call sites and validate with cargo check
    --json               Output as JSON (includes kernel_score, kernel_e2_pct)
    --min-score <F>      Exit 1 if fleet score below threshold
```

**Output includes:**
- Fleet-wide enforcement score with letter grade
- Tiered breakdown: kernel (4 repos) vs tool (36 repos) with separate grades
- Per-repo table with bindings, sites, E0/E1/E2, and letter grade
- Workspace subcrate scanning (scans `crates/*/src/` in addition to `src/`)

**Exit codes:** 0 = success, 1 = below `--min-score`.

## Falsification Criteria (Section 31)

1. **Injection safety**: After `pv kaizen --fleet --fix`, every modified
   repo must `cargo check` clean. Any compilation failure is a bug in
   the injector, not the repo.
2. **No behavioral change**: Contract macros use `debug_assert!` only.
   Release builds (`--release`) must produce identical binaries before
   and after injection. Verify with `cargo build --release` checksum.
3. **Score monotonicity**: Each `pv kaizen` run must produce
   `enforcement_score >= previous_score`. Regression = bug.
4. **Detection test**: Inject a known-bad input into softmax (NaN in
   input slice). Debug build must panic at the contract assertion.
   Release build must not panic.

## References (Section 31)

- Imai (1986). *Kaizen: The Key to Japan's Competitive Success.*
  McGraw-Hill. The continuous improvement methodology.
- Ohno (1988). *Toyota Production System.* Productivity Press.
  Five-whys root cause analysis applied to enforcement gaps.
- Meyer (1992). "Applying Design by Contract." IEEE Computer 25(10).
  Pre/postcondition methodology that kaizen automates fleet-wide.
