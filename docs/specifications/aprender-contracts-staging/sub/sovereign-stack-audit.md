# Sub-spec: Sovereign Stack Audit & No-Escape Plan

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [pv-spec.md](../pv-spec.md) Section 19

---

## Audit Findings

Full audit of all 13 repos in the PAIML sovereign AI stack.
**6.4M LOC total. Only 1.9M LOC (30%) under contract enforcement.**

### Tier 1: Enforced (AllImplemented — build fails on gaps)

| Repo | LOC | Bindings | Reverse Cov | Status |
|---|---|---|---|---|
| aprender | 531K | 377 | 7.3% | AllImplemented ✓ |
| realizar | 525K | 168 | 7.4% | AllImplemented ✓ |
| entrenar | 255K | 117 | 3.6% | AllImplemented ✓ |
| trueno | 175K | 90 | 5.0% | AllImplemented ✓ |
| rmedia | 156K | 31 | 5.4% | AllImplemented ✓ |
| presentar | 241K | 27 | 1.0% | AllImplemented ✓ |
| forjar | 336K | 13 | N/A | AllImplemented ✓ |
| **Subtotal** | **2.2M** | **823** | **~5%** | |

### Tier 2: Newly Registered (binding.yaml created, build.rs pending)

| Repo | LOC | Role | Bindings | Status |
|---|---|---|---|---|
| **depyler** | **2.5M** | Python→Rust transpiler | 7 | Registered, needs build.rs |
| **ruchy** | **829K** | Scripting→Rust transpiler | 4 | Registered, needs build.rs |
| **decy** | **460K** | Rust tooling | 5 | Registered, needs build.rs |
| **bashrs/rash** | **388K** | Shell safety tool | 6 | Registered, needs build.rs |
| **pmat** | **146K** | Quality toolkit | 4 | Registered, needs build.rs |
| **simular** | **73K** | Simulation engine | 3 | Registered, needs build.rs |
| **Subtotal** | **4.4M** | | **29** | **Bindings exist, enforcement pending** |

### Five-Whys: Why Are 6 Repos Unenforced?

1. **Why no contracts?** — These repos were built before provable-contracts existed or matured
2. **Why not retrofitted?** — Contract-first workflow assumes contracts precede code; these need contract-after
3. **Why no build.rs?** — No one added the provable-contracts dependency to these repos
4. **Why no bindings?** — `pv infer` didn't exist until this session; manual binding creation is prohibitive for 4.4M LOC
5. **Root cause:** No organizational-level enforcement requiring ALL repos to have contracts. CB-1200 checks IF contracts exist but doesn't REQUIRE them.

### The pmat Irony

pmat's CB-1200 checks other repos for contract compliance:
```
CB-1200: Provable Contracts — runs pv lint + pv score + binding coverage
```
But pmat itself has:
- 146K LOC
- 0 contracts
- 0 bindings
- 0 build.rs enforcement
- 0 `#[contract]` annotations

**The enforcer doesn't enforce itself.**

---

## No-Escape Plan

### Phase 1: Organizational Mandate (Week 1)

**Rule: Every repo in `github.com/paiml` with >10K LOC Rust MUST have:**

1. A `contracts/` directory with at least 1 valid contract YAML
2. A `contracts/binding.yaml` with at least 1 binding
3. A `build.rs` that reads binding.yaml and sets `CONTRACT_*` env vars
4. AllImplemented policy (build fails on `not_implemented`)

**Enforcement:** Add CB-1300 to pmat comply:
```
CB-1300: Contract Mandate — FAIL if Rust project >10K LOC lacks contracts/binding.yaml
```

This check runs on pmat itself, creating a bootstrap paradox that forces
pmat to add its own contracts.

### Phase 2: Transpiler Contracts (Weeks 2-4)

depyler and ruchy are the highest-risk repos. A transpiler bug
propagates to every program it produces.

**Critical contracts needed:**

| Repo | Contract | Why |
|---|---|---|
| depyler | `type-preservation-v1` | Transpiled Rust must preserve Python type semantics |
| depyler | `memory-safety-v1` | Generated Rust must not have UB |
| depyler | `semantic-equivalence-v1` | `depyler(f)(x) == f(x)` for all valid inputs |
| ruchy | `parse-roundtrip-v1` | `emit(parse(src)) ≈ src` |
| ruchy | `type-inference-v1` | Inferred types must be sound |
| ruchy | `codegen-safety-v1` | Generated Rust passes `cargo clippy -- -D warnings` |

**Method:** Use `pv infer` to auto-detect contractable functions,
accept high-confidence matches, create contracts for the rest.

### Phase 3: Self-Enforcement (Weeks 2-3)

pmat must eat its own dogfood.

**pmat contracts needed:**

| Contract | What it covers |
|---|---|
| `tdg-scoring-v1` | TDG dimensions are computed correctly |
| `comply-check-v1` | CB rules produce deterministic pass/fail |
| `score-geometric-mean-v1` | Geometric mean formula is correct |
| `context-generation-v1` | Agent context is complete and consistent |

### Phase 4: Security Tool Contracts (Week 3)

bashrs/rash is a shell safety tool — its correctness IS the security guarantee.

**bashrs contracts needed:**

| Contract | What it covers |
|---|---|
| `parser-soundness-v1` | Parser accepts all valid bash, rejects all invalid |
| `classifier-accuracy-v1` | Safety classifier has no false negatives |
| `encoder-invertibility-v1` | encode(decode(x)) = x for all shell commands |
| `cwe-mapping-v1` | CWE classification is exhaustive |

### Phase 5: Reverse Coverage Ratchet (Ongoing)

Target reverse coverage by repo:

| Repo | Current | 6-month target | 12-month target |
|---|---|---|---|
| aprender | 7% | 25% | 50% |
| realizar | 7% | 25% | 50% |
| trueno | 5% | 20% | 40% |
| entrenar | 4% | 15% | 30% |
| depyler | 0% | 10% | 25% |
| ruchy | 0% | 10% | 25% |
| pmat | 0% | 15% | 30% |
| bashrs | 0% | 10% | 25% |

CI ratchet: `pv score . --min-coverage <target> --exit-code`

### Phase 6: PVScore Gate (Month 3+)

Once all repos have basic contracts, enable PVScore (10-dim) as
the unified quality gate:

```yaml
# .github/workflows/contracts.yml (every repo)
- name: PVScore gate
  run: pv score . --min-score 90 --exit-code
```

---

## Timeline

```
Week 1:  CB-1300 mandate + pmat self-contracts
Week 2:  depyler type-preservation + memory-safety contracts
Week 3:  ruchy parse-roundtrip + bashrs parser-soundness
Week 4:  decy + simular initial contracts
Month 2: pv infer rollout across all 13 repos
Month 3: PVScore gate enabled (90 threshold)
Month 6: 25% reverse coverage target
Month 12: 50% reverse coverage target
```

---

## References

1. Meyer, B. (2025). "Software engineering as a domain to formalize." arXiv:2502.11434.
   — "A program without a specification is untestable."

2. Tu, H. et al. (2025). "Agentic Program Verification." arXiv:2511.17330.
   — Documents the cost barrier to formal verification adoption at scale.

3. Zahan, N. et al. (2023). "OpenSSF Scorecard." arXiv:2208.03412.
   — Organizational-level security scoring across entire ecosystems.

4. Sun, S. et al. (2025). "Good and Bad Failures in CI/CD." arXiv:2504.11839.
   — Pre-merge quality gates outperform post-merge approaches.

---

## Current Status (2026-03-24)

```
SOVEREIGN STACK — 13 REPOS, 852 BINDINGS
─────────────────────────────────────────
trueno        90 bindings  AllImplemented ✓
aprender     377 bindings  AllImplemented ✓ 
entrenar     117 bindings  AllImplemented ✓
realizar     168 bindings  AllImplemented ✓
forjar        13 bindings  AllImplemented ✓
presentar     27 bindings  AllImplemented ✓
rmedia        31 bindings  AllImplemented ✓
bashrs         6 bindings  AllImplemented ✓
depyler        7 bindings  AllImplemented ✓
decy           5 bindings  AllImplemented ✓
ruchy          4 bindings  AllImplemented ✓
simular        3 bindings  AllImplemented ✓
pmat           4 bindings  AllImplemented ✓
─────────────────────────────────────────
TOTAL        852 bindings  13/13 enforced
```

All 13 repos have:
1. `contracts/<repo>/binding.yaml` in provable-contracts
2. `build.rs` with AllImplemented policy (panic on gaps)
3. `serde` + `serde_yaml_ng` build-dependencies

Zero unenforced repos remain. The enforcer (pmat) enforces itself.
