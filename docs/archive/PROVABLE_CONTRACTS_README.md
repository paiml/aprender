# Provable Contracts System — Complete Analysis

This directory contains a comprehensive analysis of the **provable-contracts** system used by aprender and the broader PAIML stack.

## Quick Navigation

### Start Here
1. **[PROVABLE_CONTRACTS_SUMMARY.txt](./PROVABLE_CONTRACTS_SUMMARY.txt)** — Quick reference (5 min read)
   - Overview of 89 contracts across 6 tiers
   - Critical gaps and missing contracts
   - Priority roadmap (6 weeks)

### Deep Dive
2. **[PROVABLE_CONTRACTS_ANALYSIS.md](./PROVABLE_CONTRACTS_ANALYSIS.md)** — Full analysis (30 min read)
   - Executive summary with 5 critical findings
   - Detailed inventory by tier (Kernels, Architecture, ML, Time Series, Data/Format, E2E)
   - Gap analysis with code examples
   - Models needing contracts
   - Implementation roadmap with code examples
   - Hardcoded values to contractify

### Reference
3. **[PROVABLE_CONTRACTS_INVENTORY.txt](./PROVABLE_CONTRACTS_INVENTORY.txt)** — Complete file listing (5 min read)
   - All 89 YAML contract files organized by tier
   - Statistics and completion percentages
   - Missing contracts checklist

## Key Findings at a Glance

### Current State: 89/97 Contracts (81% Complete)

| Tier | Count | Status | Notes |
|------|-------|--------|-------|
| **Kernels** | 26/26 | ✅ 100% | All ML kernels formalized |
| **Architecture** | 20/20 | ✅ 100% | Qwen2/3/3.5/3-MoE + generic |
| **ML Algorithms** | 16/16 | ✅ 100% | Linear, Tree, Cluster, Bayes |
| **Time Series** | 8/8 | ✅ 100% | ARIMA, drift detection, etc. |
| **Data/Format** | 11/11 | ⚠️ 82% | 2 files partial (tokenizer, tensor names) |
| **E2E Verify** | 8/8 | ⚠️ 50% | 4 files partial (GPU dispatch, routing) |
| **Missing** | 8/0 | ❌ 0% | **3 P0 blockers, 3 P1, 2 P2** |

### Critical Gaps (Publishing Blockers)

1. **special-tokens-registry-v1.yaml** — EOS/BOS/PAD token IDs per model
   - Status: 0% (not started)
   - Risk: Infinite token generation loops with wrong EOS ID
   - Timeline: 2 weeks

2. **model-metadata-bounds-v1.yaml** — Plausibility bounds for model config
   - Status: 0% (ad-hoc checks exist, no contract)
   - Risk: Accepts corrupt models (94.5% zeros)
   - Timeline: 1 week

3. **tokenizer-vocab-v1.yaml** — Per-model vocab size + BPE merge rules
   - Status: 0% (hardcoded per model)
   - Risk: Encoding consistency not validated
   - Timeline: 2 weeks

## File Descriptions

### PROVABLE_CONTRACTS_SUMMARY.txt
**Format:** Fixed-width ASCII art with visual hierarchy
**Length:** 266 lines
**Best for:** Quick reference, executive briefings, decision making

Contains:
- Tier-by-tier contract overview with ASCII boxes
- Detailed gap analysis with risk severity
- Hardcoded values inventory (vocab, special tokens, rope_theta, chat markers)
- Summary table: component vs completion vs blocker status
- 6-week roadmap with week-by-week breakdown
- Command reference for `pv` CLI

### PROVABLE_CONTRACTS_ANALYSIS.md
**Format:** Markdown with detailed explanations
**Length:** 421 lines
**Best for:** Technical understanding, implementation planning, architecture reviews

Contains:
- Executive summary with 5 critical findings
- Detailed inventory by tier with full descriptions
- 5 critical gap analyses with code examples and impact
- All models needing contracts (Qwen, LLaMA, Mistral, Phi, Whisper)
- P0/P1/P2 prioritization with timeline estimates
- Implementation roadmap with Rust code examples
- Hardcoded values in aprender/src with locations
- Governance workflow for adding new models

### PROVABLE_CONTRACTS_INVENTORY.txt
**Format:** Plain text with hierarchical outline
**Length:** 229 lines
**Best for:** Contract discovery, audit, completeness checking

Contains:
- All 89 YAML files organized by tier and category
- One-liner descriptions for each contract
- Completion percentages per tier
- List of missing contracts with severity levels
- Statistics table

## Related Files

In provable-contracts repository:
- `/home/noah/src/provable-contracts/contracts/` — 89 YAML contract files
- `/home/noah/src/provable-contracts/contracts/aprender/binding.yaml` — 284 kernel→function bindings
- `/home/noah/src/provable-contracts/contracts/model-config-algebra-v1.yaml` — 5-level proof hierarchy
- `/home/noah/src/provable-contracts/contracts/aprender/tensor-layout-v1.yaml` — ROW-MAJOR enforcement

## What is a Provable Contract?

A provable contract is a YAML file that formalizes ML/transformer math:

```yaml
metadata:
  version: "1.0.0"
  description: "Special token IDs per model"

equations:
  token_bounds:
    formula: "0 <= token_id < vocab_size"
    domain: "All model families"

proof_obligations:
  - property: "Token validity"
    formal: "pad_id < vocab_size ∧ eos_id < vocab_size"

falsification_tests:
  - id: FALSIFY-STR-001
    prediction: "All token IDs valid"
    test: "proptest token_id < vocab_size"
```

From a contract, the `pv` CLI can generate:
- Rust trait stubs
- Failing test scaffolds
- Property-based tests (proptest)
- Kani bounded model checking harnesses
- Wired integration tests (binding registry)

## Governance

**Commitment:** All new models added to aprender MUST have contracts before publishing.

**Workflow:**
1. Add model to aprender
2. Create/update contracts in provable-contracts/
3. Run `pv validate contracts/model-*.yaml`
4. Generate bindings: `pv generate-bindings`
5. Tests pass: `cargo test` + `make coverage`
6. Publish: `batuta stack release aprender --publish`

## Next Action

Start with **P0 Blocker #1**: Create `special-tokens-registry-v1.yaml`

```bash
cd /home/noah/src/provable-contracts
cat > contracts/special-tokens-registry-v1.yaml << 'EOF'
metadata:
  version: "1.0.0"
  description: "Special token IDs per model family"
  references:
    - "aprender/src/format/converter/tokenizer_loader.rs"

registries:
  qwen2:
    pad_token: 151643
    eos_token: 151643
    bos_token: 151644
    # ... etc for all models

falsification_tests:
  - id: FALSIFY-STR-001
    prediction: "token_id < vocab_size for all models"
    test: "proptest: generate (vocab_size, token_id) pairs"
EOF

pv validate contracts/special-tokens-registry-v1.yaml
```

## Questions?

See detailed analysis in:
- **How many contracts?** → PROVABLE_CONTRACTS_SUMMARY.txt (lines 1-50)
- **What's missing?** → PROVABLE_CONTRACTS_ANALYSIS.md (section "CRITICAL GAPS")
- **Which models need contracts?** → PROVABLE_CONTRACTS_ANALYSIS.md (section "Models Needing Contracts")
- **How long to add a contract?** → PROVABLE_CONTRACTS_SUMMARY.txt (lines 180-210)
- **What hardcoded values should be contracts?** → PROVABLE_CONTRACTS_ANALYSIS.md (section "Hardcoded Values in aprender/src")

---

**Generated:** 2026-02-23
**Status:** Complete and ready for action
