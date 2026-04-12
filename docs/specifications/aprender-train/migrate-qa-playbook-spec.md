# SPEC-QA-MIGRATE: Migrate apr-model-qa-playbook into Monorepo

Version: 1.0
Status: proposed
Date: 2026-04-10

**Document ID:** SPEC-QA-MIGRATE-001
**Version:** 1.0.0
**Status:** PROPOSED
**Author:** PAIML Engineering
**Date:** 2026-04-10
**Priority:** P1
**Parent:** APR-MONO (Phase 2g)
**Source Repo:** `paiml/apr-model-qa-playbook` (629 commits, 257 rs files, 5 crates)
**Target:** `crates/aprender-qa-*` in `paiml/aprender`
**PMAT Epic:** PMAT-532 (subtasks: PMAT-533..537)

---

## 1. Abstract

Migrate the `apr-model-qa-playbook` repository into the aprender monorepo as Phase 2g
of APR-MONO. The repo contains 5 workspace crates implementing property-based model
qualification testing with MQS scoring, gateway validation (G0-G4), kernel class
taxonomy, and certification pipelines. After migration, the source repo is archived.

---

## 2. Motivation

1. **Dependency alignment.** The QA playbook depends on `aprender` for format validation,
   `realizar` for inference, and `trueno` for tensor ops. Cross-repo version skew causes
   the same diamond dependency problems documented in APR-MONO §3.
2. **Contract integration.** SPEC-MODEL-TYPE-001 identified that the G0 gateway needs
   `is_llm()` from `converter_types.rs`. This requires the QA crates to import
   `aprender-core` as a workspace path dep, not a crates.io version.
3. **Single test suite.** 1,891 QA tests should run in `cargo test --workspace --lib`.
4. **Archive cleanup.** One fewer external repo to maintain.

---

## 3. Source Inventory

### Crates

| Source Crate | Target Crate | Role | Files | Tests |
|-------------|-------------|------|-------|-------|
| `apr-qa-gen` | `aprender-qa-gen` | Scenario generation, oracles, kernel profiles | ~80 rs | ~600 |
| `apr-qa-runner` | `aprender-qa-runner` | Playbook execution (Rayon parallel) | ~60 rs | ~800 |
| `apr-qa-report` | `aprender-qa-report` | MQS scoring, JUnit/HTML/Markdown reports | ~40 rs | ~300 |
| `apr-qa-certify` | `aprender-qa-certify` | Tier-aware scoring, README sync, CSV export | ~50 rs | ~150 |
| `apr-qa-cli` | `aprender-qa-cli` | CLI binary with 15 subcommands | ~30 rs | ~40 |

### Non-Crate Assets

| Asset | Target Location | Notes |
|-------|----------------|-------|
| `contracts/` (5 YAML) | `contracts/aprender-qa/` | gateway, format invariants, garbage oracle, MQS, binding |
| `playbooks/` (256 YAML) | `playbooks/` (root level) | Model qualification playbooks |
| `certifications/` | `certifications/` (root level) | Certification evidence artifacts |
| `book/` | `book/src/qa/` or discard | mdBook documentation |
| `docs/specifications/` | Merge into `docs/specifications/aprender-qa/` | 5 spec files |

### Key Types to Preserve

- `QaScenario`, `Evidence`, `MqsScore`, `Oracle`, `KernelProfile`, `KernelClass`
- Gateway logic G0-G4 (zeroing invariant)
- `from_family()` kernel class mapping (A-F + SSM + Linear)

---

## 4. Migration Steps

### Phase 2g-1: Copy crates (1 hour)

```bash
# Create target directories
for crate in gen runner report certify cli; do
    cp -r ../apr-model-qa-playbook/crates/apr-qa-$crate crates/aprender-qa-$crate
done

# Rename package names in Cargo.toml
for crate in crates/aprender-qa-*/Cargo.toml; do
    sed -i 's/^name = "apr-qa-/name = "aprender-qa-/' "$crate"
done
```

### Phase 2g-2: Update dependencies (1 hour)

- Replace crates.io deps with workspace path deps:
  - `aprender = { path = "../aprender-core" }`
  - `realizar = { path = "../aprender-serve" }`
  - `trueno = { path = "../aprender-compute" }`
- Add `version.workspace = true` inheritance
- Add to root `Cargo.toml` `[workspace] members`

### Phase 2g-3: Copy contracts (30 min)

```bash
mkdir -p contracts/aprender-qa
cp ../apr-model-qa-playbook/contracts/*.yaml contracts/aprender-qa/
```

### Phase 2g-4: Copy playbooks + certifications (30 min)

```bash
cp -r ../apr-model-qa-playbook/playbooks .
cp -r ../apr-model-qa-playbook/certifications .
```

### Phase 2g-5: Verify compilation (30 min)

```bash
cargo check --workspace
cargo test -p aprender-qa-gen --lib
cargo test -p aprender-qa-runner --lib
cargo test -p aprender-qa-report --lib
cargo test -p aprender-qa-certify --lib
```

### Phase 2g-6: Integration (1 hour)

- Wire `apr qa` subcommand in `apr-cli` to use `aprender-qa-cli` as library
- Add `aprender-qa-gen` dependency for kernel class queries in `apr explain`
- Update `is_llm()` integration per SPEC-MODEL-TYPE-001

### Phase 2g-7: Archive source repo (10 min)

```bash
gh repo edit paiml/apr-model-qa-playbook \
    --description "ARCHIVED — merged into paiml/aprender (crates/aprender-qa-*)" \
    --homepage "https://github.com/paiml/aprender"
gh repo archive paiml/apr-model-qa-playbook --yes
```

---

## 5. Acceptance Criteria

| ID | Criterion | Threshold | Measurement |
|----|-----------|-----------|-------------|
| AC-QAM-001 | All 5 crates compile in workspace | `cargo check --workspace` | Zero errors |
| AC-QAM-002 | 1891 tests pass | `cargo test --workspace --lib` | >= 1890 pass (1 pre-existing failure) |
| AC-QAM-003 | 5 QA contracts in `contracts/aprender-qa/` | ls count | 5 YAML files |
| AC-QAM-004 | 256 playbooks in `playbooks/models/` | ls count | 256 YAML files |
| AC-QAM-005 | Source repo archived | `gh repo view` | Archived flag true |
| AC-QAM-006 | `apr qa` subcommand works | `apr qa --help` | Exit 0 |
| AC-QAM-007 | Workspace member count increases by 5 | Root Cargo.toml | 74 + 5 = 79 members |

---

## 6. Falsification Tests

| ID | Hypothesis Falsified If... | Mitigation |
|----|---------------------------|------------|
| FALSIFY-QAM-001 | Any QA crate fails to compile in workspace | Fix dep version conflicts; use path deps |
| FALSIFY-QAM-002 | Test count drops below 1890 | Investigate missing test files; check module visibility |
| FALSIFY-QAM-003 | `KernelClass::from_family` breaks after migration | Preserve kernel_class.rs exactly; add regression test |
| FALSIFY-QAM-004 | Gateway G0-G4 contract YAML fails `pv validate` | Fix contract paths after relocation |
| FALSIFY-QAM-005 | Playbook schema validation fails | Update schema paths in playbook YAML |

---

## 7. Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Dep version conflict (trueno 0.16 vs 0.17) | Medium | HIGH | Use workspace path deps (version skew eliminated) |
| apr-qa-cli binary conflicts with apr binary | Low | MEDIUM | Wire as library into apr-cli, remove standalone binary |
| Playbook YAML paths break | Medium | LOW | Grep-replace all relative paths |
| 1 pre-existing test failure cascades | Low | LOW | Document as known; fix separately |

---

## 8. References

| Reference | Location |
|-----------|----------|
| APR-MONO spec | `docs/specifications/aprender-monorepo-consolidation.md` (Phase 2g) |
| Gateway contract | `apr-model-qa-playbook/contracts/gateway-contract-v1.yaml` |
| Format invariants | `apr-model-qa-playbook/contracts/apr-format-invariants-v1.yaml` |
| Kernel class taxonomy | `apr-model-qa-playbook/crates/apr-qa-gen/src/kernel_class.rs` |
| Model type taxonomy | `docs/specifications/aprender-train/model-type-tokenizer-contract-spec.md` |
| Source repo | `paiml/apr-model-qa-playbook` (629 commits) |

---

*End of specification SPEC-QA-MIGRATE-001.*
