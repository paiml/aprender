# Sovereign AI Assurance Protocol: Popperian Falsification Checklist v2.5

**Version:** 2.5.0-draft
**Date:** 2026-02-04
**Status:** Draft for Team Review
**Philosophy:** *"A theory that explains everything, explains nothing."* — Karl Popper
**Governance:** Toyota Way + Scientific Method

---

## Part I: Theoretical Framework

### 1. The Crisis of Verification in Sovereign AI

The contemporary landscape of software engineering witnesses a profound bifurcation. The commercial sector prioritizes velocity over structural integrity. Meanwhile, the domain of critical infrastructure, national security, and regulated enterprise grapples with **Sovereign AI**—systems where data provenance, logic determinism, and computation residency are not mere preferences but existential mandates.

> "The right process will produce the right results." — *The Toyota Way* [9]

Traditional code review—asynchronous, superficial stylistic checks—has proven dangerously insufficient. The stochastic nature of ML, combined with neural network opacity, introduces "hidden technical debt" and "entanglement" that resists conventional debugging [5]. This protocol transforms code review from bureaucratic checkpoint into the **central scientific instrument** of the organization.

#### 1.1 Definition of Sovereign AI

Sovereignty is not binary but a **design pattern** encompassing architectural, operational, and governance choices [1]. The literature identifies five pillars:

| Pillar | Definition | Verification Mechanism |
|--------|------------|----------------------|
| **AI Capabilities** | Functional ML operations | Numerical parity tests |
| **Data Residency** | Geographic data containment | Static analysis + runtime audit |
| **Data Privacy** | PII protection, differential privacy | Formal verification |
| **Legal Controls** | Contractual/regulatory compliance | Policy-as-code checks |
| **Security/Resiliency** | Attack resistance, failover | Fuzzing, circuit breakers |

*Sources: [1] Petronella Tech, [2] Oracle Sovereign AI Brief, [3] Katonic.ai, [62] Hummel et al.*

#### 1.2 The Failure of Modern Code Review in ML

Research by Sculley et al. [5] at NeurIPS characterizes ML systems as possessing "high-interest credit cards" of technical debt:

- **Entanglement** (CACE: Changing Anything Changes Everything)
- **Hidden Feedback Loops**
- **Undeclared Consumers**
- **Correction Cascades**

Modern Code Review (MCR), as characterized by Bacchelli and Bird [4], examines "diffs" in isolation—structurally blind to systemic ML risks. A preprocessing change may look syntactically correct but fundamentally alter feature distributions, invalidating training assumptions.

The "reproducibility crisis" in ML [7] underscores verification difficulty. If reviewers cannot reproduce model performance, review is performative [60]. **This protocol elevates code review to scientific peer review standards.**

---

### 2. Toyota Way as Engineering Epistemology

The Toyota Production System (TPS) is fundamentally a system for knowledge work and problem-solving. We adapt three principles [9, 64]:

#### 2.1 Jidoka: Automation with Human Intelligence

**Origin:** Sakichi Toyoda's automatic loom stopped instantly upon thread breakage, preventing defective cloth production [10].

**In Sovereign AI:** The CI/CD pipeline must detect abnormalities and **stop automatically** before human review begins. This prevents waste (Muda) of compute and reviewer attention on structurally invalid code.

**Four Steps of Jidoka** [10]:
1. **Detect** the abnormality (automated tests, linters, formal proofs)
2. **Stop** the pipeline (circuit breaker)
3. **Fix** the immediate condition (block merge)
4. **Investigate** root cause to prevent recurrence (blameless post-mortem)

> Research indicates that Jidoka transforms incidents from bug fixes into learning events [14].

#### 2.2 Genchi Genbutsu: Go and See

**Principle 12 of Toyota Way:** Decision-makers must "go and see" the problem at its source [9].

**In Sovereign AI:** Reviewers practicing Genchi Genbutsu do not:
- Assume data is clean because the author says so—they **inspect distributions**
- Assume inference is fast enough—they **run it locally**
- Trust residency claims—they **verify physical data locations**

> "Desk" reviews miss context-dependent errors. The "source" in AI is the data itself and runtime behavior, not just Python scripts [15].

#### 2.3 Kaizen: Continuous Improvement

**Philosophy:** An "unending sense of crisis" regarding the status quo [9].

**In Sovereign AI:** Every defect found signals process failure. The question shifts from "Who broke this?" to "How did our process allow this to be written?" [19]

---

### 3. The Seven Wastes (Muda) in AI Development

Adapted from Lean literature [9] for AI systems:

| Waste Type | AI Manifestation | Mitigation |
|------------|------------------|------------|
| **Overproduction** | Models/features solving no user problem | Hypothesis-Driven Development |
| **Waiting** | Training delays, review feedback loops | Jidoka parallelization |
| **Transport** | Unnecessary data movement (security risk) | Federated Learning, VPC isolation |
| **Overprocessing** | Transformer when Random Forest suffices | Baseline comparison requirement |
| **Inventory** | Data hoarding "just in case" | Data minimization, lifecycle policies |
| **Motion** | Developer context switching | Integrated tooling |
| **Defects** | Bugs requiring rework | Pre-merge inspection gates |

---

### 4. Scientific Method for High-Assurance Engineering

#### 4.1 Hypothesis-Driven Development (HDD)

Every proposed change is a **hypothesis to be tested**, not a requirement to be implemented [20-23]:

1. **Observation:** Analysis of current system (e.g., "15% false positive rate on edge cases")
2. **Hypothesis:** Precise prediction (e.g., "Focal loss reduces FP on class B by ≥5%")
3. **Experiment Design:** Code changes, data selection, evaluation metrics
4. **Execution:** Implementation and training
5. **Analysis:** Results vs hypothesis comparison
6. **Peer Review:** Independent validation

**Integration:** PR templates require hypothesis statement and evidence. If evidence doesn't support hypothesis, code is not merged—even if bug-free.

#### 4.2 Reproducibility Standards

Derived from NeurIPS/ICLR standards [26, 27] and Heil et al. [7]:

| Level | Requirements | Sovereign AI Mandate |
|-------|--------------|---------------------|
| **Bronze** | Code available | Minimum |
| **Silver** | Code + data + environment documented | Required |
| **Gold** | Third-party reproduction with single command | **Mandatory** |

**Implementation:**
- Pin random seeds
- Containerized environments (Docker/Singularity)
- Data versioning (DVC/Pachyderm)
- `make reproduce` target

#### 4.3 Falsifiability and Null Hypothesis

Reviewers act as skeptics. They start with the **Null Hypothesis**—the new model is no better than baseline. Authors must provide:
- Statistical evidence (p-values, confidence intervals)
- Effect sizes
- Ablation studies [39]

This prevents "state-of-the-art" chasing based on statistical noise rather than genuine improvement [28].

#### 4.4 Equation-Driven Development (EDD)

For simulation systems (e.g., `simular`), code follows **Equation-Driven Development**:

1. **Prove First:** Mathematical equation verified analytically or via symbolic computation
2. **Implement Second:** Simulation built on verified equation foundation
3. **Document Always:** Equation Model Card (EMC) accompanies every shared simulation

**EDD Workflow:**
```
Equation → Analytical Proof → Implementation → Numerical Validation → EMC
```

**Equation Model Card (EMC)** requirements:
- LaTeX-rendered governing equations
- Domain of validity (parameter ranges, boundary conditions)
- Analytical derivation or citation to peer-reviewed source
- Numerical stability analysis
- Verification test cases with known solutions

This ensures simulations are not "curve fitting" but grounded in verified physics/mathematics.

#### 4.5 Architectural Invariants (Hard Requirements)

Two non-negotiable requirements apply to **all** stack projects:

**1. Declarative Configuration (No-Code YAML)**

Every project must offer full functionality via YAML configuration without writing code:
```yaml
# Example: aprender declarative ML pipeline
pipeline:
  model: random_forest
  features: [age, income, score]
  target: churn
  cross_validation: 5
```

**Rationale:** Enables domain experts, reduces bugs, ensures reproducibility, facilitates auditing.

**2. Zero Scripting Policy**

No Python, JavaScript, or other interpreted scripting languages in production paths:
- **Allowed:** Rust, WASM, declarative YAML/TOML
- **Prohibited:** Python, JavaScript, Lua, shell scripts in runtime
- **Exception:** Build-time code generation only

**Example Violation:** `presentar` using JavaScript vs `jugar/probar` pure-Rust testing framework.

**Rationale:** Type safety, performance, deterministic builds, supply chain security.

---

## Part II: Three-Layer Operational Architecture

### Layer 1: Jidoka — Automated Sovereignty and Quality Gate

Before human engagement, code passes an automated gauntlet. **Defects never pass downstream** [11-14].

#### 1.1 Automated Sovereignty Checks

| Check | Tool | Rejection Trigger |
|-------|------|-------------------|
| **AI BOM/SBOM** | cargo-sbom, custom | Unapproved library or pre-trained weight |
| **Supply Chain** | cargo-audit, cargo-deny | Known vulnerability, untrusted source |
| **Sovereignty Linter** | Custom static analysis | Hardcoded IPs, non-compliant APIs |
| **Formal Verification** | Miri, Kani, custom | Safety proof failure |
| **Reproducibility** | Build verification | Loose dependencies (e.g., `numpy` vs `numpy==1.21.0`) |

*Reference: [29, 31] on supply chain risks in LLM applications*

#### 1.2 Model Circuit Breakers

Jidoka-inspired automated stops:
- **Training divergence** — Loss exceeds threshold
- **Data drift detection** — Distribution shift beyond tolerance
- **Fairness regression** — Protected class metrics degrade
- **Latency regression** — P99 exceeds SLA

### Layer 2: Genchi Genbutsu — Evidence-Based Peer Review

Once automation clears, **structured audit** begins (not "LGTM" culture).

#### 2.1 The Sovereign AI Checklist

Empirical studies [35, 52] demonstrate checklist-based reading significantly outperforms ad-hoc review (60-90% defect removal vs 30-50% for testing).

| Category | Inspection Item | Theoretical Basis | Source |
|----------|-----------------|-------------------|--------|
| **Hypothesis** | Falsifiable hypothesis stated? | Scientific Method | [20] |
| **Data Sovereignty** | Residency boundaries respected? | Five Pillars | [3] |
| **Reproducibility** | Reviewer can execute training and match metrics? | Gold Standard | [7] |
| **ML Technical Debt** | Entanglement introduced? | CACE Principle | [5] |
| **Waste (Muda)** | Complexity justified vs baseline? | Toyota Way | [9] |
| **Robustness** | Adversarial testing performed? | AI Safety | [40] |
| **Auditability** | Decision logging sufficient? | Compliance | [42] |

#### 2.2 Deterministic LLM-Assisted Development

LLM-assisted code generation is permitted when conducted through controlled tooling (e.g., `paiml-mcp-agent-toolkit`) that ensures:
- Reproducible prompts with version-controlled context
- Audit trail of generation sessions
- Human review of all generated code against checklist

### Layer 3: Governance — Artifacts of Sovereignty

#### 3.1 Model Cards and Datasheets

Every significant model change requires updates to:
- **Model Card** [45, 46, 67]: Intended use, limitations, training data provenance, ethical considerations
- **Datasheet for Datasets** [47, 68]: Collection methodology, preprocessing, known biases

Mismatch between code behavior and Model Card = **critical defect**.

#### 3.2 Sovereign Audit Trail

The review system acts as a ledger:
- Who reviewed (certified Sovereign Code Reviewer)
- What tests ran (with result hashes)
- Which sovereignty checks passed
- Full provenance for regulatory inquiry

*"Auditability by Design" [42]*

---

## Part III: The 100-Item Falsification Checklist

### Nullification Protocol

For each claim:
- **Claim:** Assertion made by the stack
- **Falsification Test:** Experiment to attempt disproof (*Genchi Genbutsu*)
- **Null Hypothesis (H₀):** Assumed true until disproven
- **Rejection Criteria:** Conditions that falsify claim (*Jidoka trigger*)
- **TPS Principle:** Toyota Way mapping
- **Evidence Required:** Data for evaluation

### Severity Levels

| Level | Impact | TPS Response |
|-------|--------|--------------|
| **Critical** | Invalidates core claims | Stop the Line (Andon), retract claim |
| **Major** | Significantly weakens validity | Kaizen required, revise with caveats |
| **Minor** | Edge case/boundary | Document limitation |
| **Informational** | Clarification needed | Update documentation |

---

## Section 1: Sovereign Data Governance [15 Items]

*Five Pillars verification: Data Residency, Privacy, Legal Controls*

### SDG-01: Data Residency Boundary Enforcement

**Claim:** Sovereign-critical data never crosses defined geographic boundaries.

**Falsification Test:**
```bash
cargo test --package batuta --test data_residency_enforcement
./scripts/network_egress_audit.sh
```

**Null Hypothesis:** Data crosses boundaries during processing.

**Rejection Criteria:** Any network call to non-compliant regions during Sovereign-tier operations.

**TPS Principle:** Jidoka — automatic boundary violation detection

**Evidence Required:**
- [ ] Network egress logs during test workloads
- [ ] Static analysis of all HTTP/gRPC endpoints
- [ ] VPC isolation verification in IaC

*Reference: [2] Oracle Sovereign AI, [3] Katonic Data Sovereignty*

---

### SDG-02: Data Inventory Completeness

**Claim:** All ingested data has documented purpose, classification, and lifecycle policy.

**Falsification Test:**
```bash
./scripts/data_inventory_audit.sh
```

**Null Hypothesis:** Undocumented data exists in the system.

**Rejection Criteria:** Any data asset without classification tag or lifecycle policy.

**TPS Principle:** Muda (Inventory waste) — prevent data hoarding liability

**Evidence Required:**
- [ ] Data catalog completeness report
- [ ] Classification schema documentation
- [ ] Automated classification enforcement

*Reference: [3] Katonic Data Inventory*

---

### SDG-03: Privacy-Preserving Computation

**Claim:** Differential privacy correctly implemented where specified.

**Falsification Test:**
```bash
cargo test --package trueno --test differential_privacy
```

**Null Hypothesis:** Privacy guarantees are not mathematically sound.

**Rejection Criteria:**
- ε-δ guarantees not met
- Composition budget exceeded

**TPS Principle:** Poka-Yoke — privacy by design

**Evidence Required:**
- [ ] Privacy budget accounting
- [ ] Formal ε-δ verification
- [ ] Noise calibration documentation

*Reference: [38] IEEE Privacy-Preserving ML Survey*

---

### SDG-04: Federated Learning Client Isolation

**Claim:** Federated learning clients send only model updates, never raw data.

**Falsification Test:**
```bash
cargo test --package batuta --test federated_client_isolation
./scripts/fl_traffic_inspection.sh
```

**Null Hypothesis:** Raw data escapes client boundaries.

**Rejection Criteria:** Any payload containing raw training samples in network trace.

**TPS Principle:** Jidoka — client-side enforcement

**Evidence Required:**
- [ ] Packet inspection logs
- [ ] Gradient-only serialization verification
- [ ] Secure aggregation implementation audit

*Reference: [48] Google Federated Learning, [49, 50] STL Partners/Duality*

---

### SDG-05: Supply Chain Provenance (AI BOM)

**Claim:** All model weights and training data have verified provenance.

**Falsification Test:**
```bash
batuta stack ai-bom --verify
cargo audit
```

**Null Hypothesis:** Unverified external artifacts exist in the stack.

**Rejection Criteria:**
- Pre-trained weight without Ed25519 signature
- Training data without chain-of-custody documentation

**TPS Principle:** Jidoka — supply chain circuit breaker

**Evidence Required:**
- [ ] AI Bill of Materials generation
- [ ] Signature verification for all artifacts
- [ ] Approved vendor whitelist

*Reference: [29, 31] Supply Chain Risks in LLM Applications*

---

### SDG-06: VPC Isolation Verification

**Claim:** Compute resources spin up only in compliant sovereign regions.

**Falsification Test:**
```bash
./scripts/iac_region_audit.sh
terraform plan -out=plan.out && terraform show -json plan.out | jq '.resource_changes[].change.after.region'
```

**Null Hypothesis:** Infrastructure deploys to non-compliant regions.

**Rejection Criteria:** Any resource in region not on approved list.

**TPS Principle:** Genchi Genbutsu — verify physical location

**Evidence Required:**
- [ ] IaC policy-as-code checks
- [ ] Runtime region verification
- [ ] Cloud provider compliance attestation

*Reference: [37] IBM Data Sovereignty vs Residency*

---

### SDG-07: Data Classification Enforcement

**Claim:** Code paths enforce data classification (Public/Internal/Confidential/Sovereign).

**Falsification Test:**
```bash
cargo test --package batuta --test classification_enforcement
```

**Null Hypothesis:** Classification labels are advisory only.

**Rejection Criteria:** Sovereign-classified data processed by Public-tier code path.

**TPS Principle:** Poka-Yoke — classification-based routing

**Evidence Required:**
- [ ] Type-level classification enforcement
- [ ] Runtime label verification
- [ ] Cross-tier data flow analysis

---

### SDG-08: Consent and Purpose Limitation

**Claim:** Data usage matches documented consent scope.

**Falsification Test:**
```bash
./scripts/consent_audit.sh
```

**Null Hypothesis:** Data used beyond consent scope.

**Rejection Criteria:** Any data processing without matching consent record.

**TPS Principle:** Legal controls pillar

**Evidence Required:**
- [ ] Consent management integration
- [ ] Purpose binding verification
- [ ] Consent withdrawal propagation

---

### SDG-09: Right to Erasure (RTBF) Compliance

**Claim:** Data deletion requests fully propagate through all storage.

**Falsification Test:**
```bash
cargo test --package batuta --test rtbf_propagation
```

**Null Hypothesis:** Deleted data persists in some storage layer.

**Rejection Criteria:** Any trace of erased identity in storage or model.

**TPS Principle:** Muda — data inventory hygiene

**Evidence Required:**
- [ ] Deletion cascade verification
- [ ] Model unlearning documentation
- [ ] Backup purge confirmation

---

### SDG-10: Cross-Border Transfer Logging

**Claim:** All cross-border data transfers are logged and justified.

**Falsification Test:**
```bash
./scripts/cross_border_audit.sh
```

**Null Hypothesis:** Unlogged transfers occur.

**Rejection Criteria:** Any cross-border transfer without audit log entry.

**TPS Principle:** Auditability requirement

**Evidence Required:**
- [ ] Transfer logging completeness
- [ ] Legal basis documentation
- [ ] Transfer impact assessment

---

### SDG-11: Model Weight Sovereignty

**Claim:** Model weights trained on sovereign data remain under sovereign control.

**Falsification Test:**
```bash
cargo test --package pacha --test weight_sovereignty
```

**Null Hypothesis:** Weights escape sovereign control.

**Rejection Criteria:** Model weights accessible from non-sovereign context.

**TPS Principle:** Jidoka — weight exfiltration prevention

**Evidence Required:**
- [ ] Access control verification
- [ ] Encryption at rest/in transit
- [ ] Key management sovereignty

---

### SDG-12: Inference Result Classification

**Claim:** Inference outputs inherit classification from input data.

**Falsification Test:**
```bash
cargo test --package realizar --test output_classification
```

**Null Hypothesis:** Output classification is lower than input classification.

**Rejection Criteria:** Sovereign input produces unclassified output.

**TPS Principle:** Poka-Yoke — automatic classification propagation

**Evidence Required:**
- [ ] Classification inheritance logic
- [ ] Output tagging verification
- [ ] Downstream consumer enforcement

---

### SDG-13: Audit Log Immutability

**Claim:** Audit logs cannot be modified or deleted.

**Falsification Test:**
```bash
./scripts/audit_log_integrity.sh
```

**Null Hypothesis:** Audit logs can be tampered with.

**Rejection Criteria:** Any successful modification of historical log entry.

**TPS Principle:** Governance layer integrity

**Evidence Required:**
- [ ] Append-only storage verification
- [ ] Cryptographic chaining (Merkle tree)
- [ ] Independent log verification

---

### SDG-14: Third-Party API Isolation

**Claim:** No third-party API calls during sovereign-tier operations.

**Falsification Test:**
```bash
cargo test --package batuta --test third_party_isolation
```

**Null Hypothesis:** Third-party APIs are called.

**Rejection Criteria:** Any outbound call to non-sovereign endpoint.

**TPS Principle:** Jidoka — network isolation

**Evidence Required:**
- [ ] Network allowlist enforcement
- [ ] Static analysis for external calls
- [ ] Runtime network monitoring

---

### SDG-15: Homomorphic/Secure Computation Verification

**Claim:** Secure computation primitives correctly implemented.

**Falsification Test:**
```bash
cargo test --package trueno --test secure_computation
```

**Null Hypothesis:** Cryptographic guarantees are violated.

**Rejection Criteria:** Any information leakage beyond specified bounds.

**TPS Principle:** Formal verification requirement

**Evidence Required:**
- [ ] Cryptographic protocol audit
- [ ] Reference implementation comparison
- [ ] Side-channel analysis

*Reference: [38] IEEE Privacy-Preserving ML Survey*

---

## Section 2: ML Technical Debt Prevention [10 Items]

*Sculley et al. "Hidden Technical Debt" mitigation [5, 39]*

### MTD-01: Entanglement (CACE) Detection

**Claim:** Feature changes are isolated; changing one doesn't silently affect others.

**Falsification Test:**
```bash
cargo test --package aprender --test feature_isolation
./scripts/ablation_study.sh
```

**Null Hypothesis:** CACE principle is violated.

**Rejection Criteria:** Ablation study shows unexpected cross-feature impact.

**TPS Principle:** Kaizen — root cause analysis

**Evidence Required:**
- [ ] Feature importance analysis
- [ ] Ablation study results
- [ ] Feature dependency documentation

*Reference: [5] Sculley NeurIPS "CACE Principle"*

---

### MTD-02: Correction Cascade Prevention

**Claim:** No model exists solely to correct another model's errors.

**Falsification Test:**
```bash
./scripts/model_dependency_audit.sh
```

**Null Hypothesis:** Correction cascades exist in production.

**Rejection Criteria:** Model B exists only to patch Model A outputs.

**TPS Principle:** Kaizen — fix root cause in Model A

**Evidence Required:**
- [ ] Model dependency graph
- [ ] Purpose documentation for each model
- [ ] Refactoring plan for any cascades found

*Reference: [5] Sculley "Correction Cascades"*

---

### MTD-03: Undeclared Consumer Detection

**Claim:** All model consumers are documented and access-controlled.

**Falsification Test:**
```bash
./scripts/consumer_audit.sh
```

**Null Hypothesis:** Unknown consumers access model outputs.

**Rejection Criteria:** Any access from unregistered consumer.

**TPS Principle:** Visibility across downstream supply chain

**Evidence Required:**
- [ ] Consumer registry completeness
- [ ] Access log analysis
- [ ] Authentication enforcement

*Reference: [5] "Undeclared Consumers", [42] Auditability*

---

### MTD-04: Data Dependency Freshness

**Claim:** Training data dependencies are current and maintained.

**Falsification Test:**
```bash
cargo test --package alimentar --test data_freshness
```

**Null Hypothesis:** Stale data dependencies exist.

**Rejection Criteria:** Data source unchanged for >N days without explicit acknowledgment.

**TPS Principle:** Muda (Inventory) — prevent data staleness

**Evidence Required:**
- [ ] Data freshness monitoring
- [ ] Staleness alerting
- [ ] Explicit freeze documentation

---

### MTD-05: Pipeline Glue Code Minimization

**Claim:** Pipeline code uses standardized connectors, not ad-hoc scripts.

**Falsification Test:**
```bash
./scripts/glue_code_audit.sh
```

**Null Hypothesis:** Excessive glue code exists.

**Rejection Criteria:** >10% of pipeline LOC is custom data transformation.

**TPS Principle:** Muda (Motion) — standardization

**Evidence Required:**
- [ ] Pipeline code analysis
- [ ] Standard connector usage metrics
- [ ] Refactoring plan for glue code

---

### MTD-06: Configuration Debt Prevention

**Claim:** All hyperparameters and configurations are version-controlled.

**Falsification Test:**
```bash
./scripts/config_audit.sh
```

**Null Hypothesis:** Configuration drift exists.

**Rejection Criteria:** Any configuration not in version control.

**TPS Principle:** Reproducibility requirement

**Evidence Required:**
- [ ] Config file version control
- [ ] Environment variable documentation
- [ ] Configuration change audit trail

---

### MTD-07: Dead Code Elimination

**Claim:** No unused model code paths exist in production.

**Falsification Test:**
```bash
cargo +nightly udeps --workspace
./scripts/dead_feature_audit.sh
```

**Null Hypothesis:** Dead code exists.

**Rejection Criteria:** Any unreachable code path in model inference.

**TPS Principle:** Muda (Inventory) — code hygiene

**Evidence Required:**
- [ ] Dead code analysis
- [ ] Feature flag cleanup
- [ ] Deprecated path removal

---

### MTD-08: Abstraction Boundary Verification

**Claim:** ML code respects clean abstraction boundaries.

**Falsification Test:**
```bash
./scripts/abstraction_audit.sh
```

**Null Hypothesis:** Leaky abstractions exist.

**Rejection Criteria:** Business logic leaks into model code or vice versa.

**TPS Principle:** Clean Architecture principle

**Evidence Required:**
- [ ] Dependency direction analysis
- [ ] Interface segregation verification
- [ ] Layer violation detection

*Reference: [63] Lewis et al. "Mismatch in ML Systems"*

---

### MTD-09: Feedback Loop Detection

**Claim:** No hidden feedback loops where model outputs influence future training.

**Falsification Test:**
```bash
./scripts/feedback_loop_audit.sh
```

**Null Hypothesis:** Hidden feedback loops exist.

**Rejection Criteria:** Model output appears in training data pipeline.

**TPS Principle:** Entanglement prevention

**Evidence Required:**
- [ ] Data flow tracing
- [ ] Training/inference separation verification
- [ ] Feedback loop documentation (if intentional)

*Reference: [5] "Hidden Feedback Loops"*

---

### MTD-10: Technical Debt Quantification

**Claim:** ML technical debt is measured and trending downward.

**Falsification Test:**
```bash
pmat analyze tdg . --output tdg_history.json
```

**Null Hypothesis:** Technical debt is unmeasured or increasing.

**Rejection Criteria:**
- No TDG measurement
- TDG score declining over 3 releases

**TPS Principle:** Kaizen — continuous measurement

**Evidence Required:**
- [ ] TDG score history
- [ ] Debt remediation velocity
- [ ] Prioritized debt backlog

---

## Section 3: Hypothesis-Driven & Equation-Driven Development [13 Items]

*Scientific Method integration [20-27]*

### HDD-01: Hypothesis Statement Requirement

**Claim:** Every model change PR includes falsifiable hypothesis.

**Falsification Test:**
```bash
./scripts/pr_hypothesis_audit.sh
```

**Null Hypothesis:** PRs merge without hypothesis.

**Rejection Criteria:** Model PR without "Hypothesis:" section in template.

**TPS Principle:** Scientific Method integration

**Evidence Required:**
- [ ] PR template enforcement
- [ ] Hypothesis statement audit
- [ ] Rejection rate for missing hypothesis

*Reference: [20-23] Hypothesis-Driven Development*

---

### HDD-02: Baseline Comparison Requirement

**Claim:** Complex models must beat simple baselines to be merged.

**Falsification Test:**
```bash
cargo test --package aprender --test baseline_comparison
```

**Null Hypothesis:** Complex models merge without baseline comparison.

**Rejection Criteria:**
- Transformer merges without Random Forest baseline
- <5% improvement over baseline without justification

**TPS Principle:** Muda (Overprocessing) prevention

**Evidence Required:**
- [ ] Baseline benchmark results
- [ ] Complexity vs improvement analysis
- [ ] Justification for marginal improvements

---

### HDD-03: Gold Standard Reproducibility

**Claim:** `make reproduce` recreates training results from scratch.

**Falsification Test:**
```bash
make clean && make reproduce
```

**Null Hypothesis:** Reproduction fails or results differ.

**Rejection Criteria:**
- Build fails from clean state
- Metrics differ by >1% from documented results

**TPS Principle:** Scientific reproducibility

**Evidence Required:**
- [ ] Fresh environment reproduction
- [ ] Hash comparison of outputs
- [ ] CI reproduction verification

*Reference: [7] Heil et al. "Reproducibility Standards"*

---

### HDD-04: Random Seed Documentation

**Claim:** All stochastic operations have documented, pinned seeds.

**Falsification Test:**
```bash
cargo test --package aprender --test seed_documentation
```

**Null Hypothesis:** Seeds are undocumented or unpinned.

**Rejection Criteria:** Any stochastic operation without explicit seed.

**TPS Principle:** Deterministic reproducibility

**Evidence Required:**
- [ ] Seed inventory
- [ ] Seed pinning verification
- [ ] Cross-run consistency testing

---

### HDD-05: Environment Containerization

**Claim:** Training environment is fully containerized and versioned.

**Falsification Test:**
```bash
./scripts/container_reproducibility.sh
```

**Null Hypothesis:** Environment has uncontrolled dependencies.

**Rejection Criteria:**
- Dockerfile missing
- Unpinned dependency in requirements

**TPS Principle:** Silver → Gold reproducibility

**Evidence Required:**
- [ ] Dockerfile verification
- [ ] Dependency lock file
- [ ] Container hash tracking

*Reference: [25] "Defining Reproducibility"*

---

### HDD-06: Data Version Control

**Claim:** Training data is versioned with content-addressable storage.

**Falsification Test:**
```bash
dvc status
./scripts/data_version_audit.sh
```

**Null Hypothesis:** Data changes without version tracking.

**Rejection Criteria:** Any data modification not captured in version control.

**TPS Principle:** Reproducibility requirement

**Evidence Required:**
- [ ] DVC or equivalent setup
- [ ] Data hash verification
- [ ] Historical data retrieval test

---

### HDD-07: Statistical Significance Requirement

**Claim:** Performance claims include statistical significance tests.

**Falsification Test:**
```bash
./scripts/statistical_significance_audit.sh
```

**Null Hypothesis:** Claims made without significance testing.

**Rejection Criteria:**
- p ≥ 0.05 for claimed improvement
- No confidence interval reported

**TPS Principle:** Scientific rigor

**Evidence Required:**
- [ ] Hypothesis test results
- [ ] Effect size calculations
- [ ] Multiple comparison corrections

*Reference: [28] "Systematic Literature Review on ML Code Review"*

---

### HDD-08: Ablation Study Requirement

**Claim:** Multi-component changes include ablation studies.

**Falsification Test:**
```bash
./scripts/ablation_requirement_audit.sh
```

**Null Hypothesis:** Compound changes merge without ablation.

**Rejection Criteria:** PR with >2 model changes without per-component analysis.

**TPS Principle:** Scientific Method — isolation of variables

**Evidence Required:**
- [ ] Per-component contribution analysis
- [ ] Interaction effect documentation
- [ ] Sensitivity analysis

*Reference: [39] Google Research "Hidden Technical Debt Extended"*

---

### HDD-09: Negative Result Documentation

**Claim:** Failed experiments are documented, not just successes.

**Falsification Test:**
```bash
./scripts/negative_result_audit.sh
```

**Null Hypothesis:** Negative results are hidden.

**Rejection Criteria:** Experiment log shows only successful attempts.

**TPS Principle:** Kaizen — learning from failures

**Evidence Required:**
- [ ] Failed experiment documentation
- [ ] Why-it-failed analysis
- [ ] Publication of negative results

---

### HDD-10: Pre-registration of Metrics

**Claim:** Evaluation metrics defined before experimentation.

**Falsification Test:**
```bash
git log --grep="evaluation metric" --before="experiment-date"
```

**Null Hypothesis:** Metrics selected after seeing results.

**Rejection Criteria:** Metric definition commit after experiment commit.

**TPS Principle:** Scientific pre-registration

**Evidence Required:**
- [ ] Metric definition timestamp
- [ ] Experiment start timestamp
- [ ] No metric modification after results

---

### EDD-01: Equation Verification Before Implementation

**Claim:** Every simular simulation has analytically verified governing equation.

**Falsification Test:**
```bash
cargo test --package simular --test equation_verification
./scripts/emc_audit.sh
```

**Null Hypothesis:** Simulation implemented without equation proof.

**Rejection Criteria:** Simulation code exists without corresponding EMC or analytical derivation.

**TPS Principle:** EDD — prove first, implement second

**Evidence Required:**
- [ ] EMC file exists for each simulation
- [ ] Analytical derivation or peer-reviewed citation
- [ ] Symbolic computation verification (where applicable)

---

### EDD-02: Equation Model Card (EMC) Completeness

**Claim:** Every shared simulation includes complete Equation Model Card.

**Falsification Test:**
```bash
./scripts/emc_completeness_audit.sh
```

**Null Hypothesis:** EMC is incomplete or missing.

**Rejection Criteria:** EMC missing required sections:
- Governing equations (LaTeX)
- Domain of validity
- Derivation/citation
- Stability analysis
- Verification test cases

**TPS Principle:** Governance — equation traceability

**Evidence Required:**
- [ ] EMC schema validation
- [ ] LaTeX equation rendering
- [ ] Parameter range documentation

---

### EDD-03: Numerical vs Analytical Validation

**Claim:** Simulation results match analytical solutions within tolerance.

**Falsification Test:**
```bash
cargo test --package simular --test analytical_validation
```

**Null Hypothesis:** Numerical results diverge from analytical solutions.

**Rejection Criteria:** |numerical - analytical| > documented tolerance for any test case with known solution.

**TPS Principle:** Genchi Genbutsu — verify against ground truth

**Evidence Required:**
- [ ] Test cases with closed-form solutions
- [ ] Convergence analysis
- [ ] Error bound documentation

---

## Section 4: Numerical Reproducibility [15 Items]

*IEEE 754 compliance, reference implementation parity*

### NR-01: IEEE 754 Floating-Point Compliance

**Claim:** trueno SIMD operations produce IEEE 754-compliant results.

**Falsification Test:**
```bash
cargo test --package trueno --test ieee754_compliance
```

**Null Hypothesis:** Results deviate from IEEE 754 specification.

**Rejection Criteria:** Any operation differs from reference by >1 ULP.

**TPS Principle:** Jidoka — automatic compliance verification

**Evidence Required:**
- [ ] Berkeley TestFloat results
- [ ] Special case documentation (NaN, Inf, subnormals)
- [ ] Kahan paranoia test results

*Reference: [IEEE 754-2019], Goldberg (1991), Kahan (1965)*

---

### NR-02: Cross-Platform Numerical Determinism

**Claim:** Identical inputs produce identical outputs across x86-64, ARM64, WASM.

**Falsification Test:**
```bash
./scripts/cross-platform-numeric-test.sh
```

**Null Hypothesis:** Results differ across platforms.

**Rejection Criteria:** Any bit-level difference for identical inputs.

**TPS Principle:** Genchi Genbutsu — verify on actual hardware

**Evidence Required:**
- [ ] SHA-256 hash comparison across platforms
- [ ] Platform-specific rounding documentation
- [ ] CI matrix coverage

*Reference: [61] Nagarajan et al. "Determinism in Deep Learning"*

---

### NR-03: NumPy Reference Parity

**Claim:** trueno operations match NumPy within documented epsilon.

**Falsification Test:**
```bash
cargo test --package trueno --test numpy_parity
python tests/reference/generate_numpy_golden.py
```

**Null Hypothesis:** Results diverge beyond tolerance.

**Rejection Criteria:** |trueno - numpy| > ε for documented ε per operation.

**TPS Principle:** Baseline comparison

**Evidence Required:**
- [ ] Per-operation epsilon documentation
- [ ] 1000+ random inputs per operation
- [ ] Edge case testing

---

### NR-04: scikit-learn Algorithm Parity

**Claim:** aprender algorithms produce statistically equivalent sklearn results.

**Falsification Test:**
```bash
cargo test --package aprender --test sklearn_parity
```

**Null Hypothesis:** Results not statistically equivalent.

**Rejection Criteria:**
- Regression: R² difference > 0.01
- Classification: Accuracy difference > 1%
- Clustering: ARI difference > 0.05

**TPS Principle:** Scientific validation

**Evidence Required:**
- [ ] UCI dataset results
- [ ] Paired t-test (p < 0.05)
- [ ] Seed reproducibility

---

### NR-05: Linear Algebra Decomposition Accuracy

**Claim:** Decompositions (Cholesky, SVD, QR, LU) meet LAPACK standards.

**Falsification Test:**
```bash
cargo test --package trueno --test decomposition_accuracy
```

**Null Hypothesis:** Accuracy worse than LAPACK.

**Rejection Criteria:** ||A - reconstruct(decompose(A))|| / ||A|| > 1e-12 (f64).

**TPS Principle:** Reference baseline

**Evidence Required:**
- [ ] LAPACK test matrix results
- [ ] Condition number analysis
- [ ] Ill-conditioning behavior

*Reference: Higham "Accuracy and Stability of Numerical Algorithms"*

---

### NR-06: Kahan Summation Implementation

**Claim:** Summation uses compensated summation for error minimization.

**Falsification Test:**
```bash
cargo test --package trueno --test kahan_summation
```

**Null Hypothesis:** Naive summation accumulates error.

**Rejection Criteria:** Error growth O(n) instead of O(√n) or O(1).

**TPS Principle:** Quality built-in

**Evidence Required:**
- [ ] Error growth analysis
- [ ] Naive vs compensated comparison
- [ ] Pathological case testing

*Reference: Kahan (1965)*

---

### NR-07: RNG Statistical Quality

**Claim:** RNG passes NIST SP 800-22 statistical tests.

**Falsification Test:**
```bash
cargo test --package trueno --test rng_quality
./scripts/nist_sts_test.sh
```

**Null Hypothesis:** RNG fails randomness tests.

**Rejection Criteria:** Any NIST test failure at α = 0.01.

**TPS Principle:** Formal verification

**Evidence Required:**
- [ ] Full NIST suite results
- [ ] Dieharder results
- [ ] PractRand extended test

---

### NR-08: Quantization Error Bounds

**Claim:** Quantization maintains perplexity within bounds.

**Falsification Test:**
```bash
realizar eval --quant q4_0 --metric perplexity
```

**Null Hypothesis:** Error exceeds bounds.

**Rejection Criteria:**
- Q4_0: >5% perplexity increase
- Q8_0: >1% perplexity increase

**TPS Principle:** Documented tradeoffs

**Evidence Required:**
- [ ] Perplexity measurements
- [ ] Per-layer error analysis
- [ ] llama.cpp comparison

*Reference: Dettmers et al. (2023) QLoRA*

---

### NR-09: Gradient Computation Correctness

**Claim:** Autograd produces correct gradients.

**Falsification Test:**
```bash
cargo test --package entrenar --test gradient_check
```

**Null Hypothesis:** Gradients differ from numerical gradients.

**Rejection Criteria:** Relative difference > 1e-5.

**TPS Principle:** Mathematical correctness

**Evidence Required:**
- [ ] Finite difference verification
- [ ] PyTorch comparison
- [ ] All operation coverage

---

### NR-10: Tokenization Parity

**Claim:** Tokenizer matches HuggingFace output.

**Falsification Test:**
```bash
cargo test --package realizar --test tokenizer_parity
```

**Null Hypothesis:** Token IDs differ.

**Rejection Criteria:** Any string where tokens differ.

**TPS Principle:** Reference baseline

**Evidence Required:**
- [ ] 10,000+ diverse string tests
- [ ] Unicode edge cases
- [ ] Special token handling

---

### NR-11: Attention Mechanism Correctness

**Claim:** Attention computes softmax(QK^T/√d)V correctly.

**Falsification Test:**
```bash
cargo test --package realizar --test attention_correctness
```

**Null Hypothesis:** Attention computation incorrect.

**Rejection Criteria:**
- Difference from reference > 1e-5
- Attention weights don't sum to 1.0

**TPS Principle:** Mathematical specification

**Evidence Required:**
- [ ] PyTorch SDPA comparison
- [ ] Causal mask verification
- [ ] Multi-head correctness

---

### NR-12: Loss Function Accuracy

**Claim:** Loss functions match reference implementations.

**Falsification Test:**
```bash
cargo test --package aprender --test loss_accuracy
```

**Null Hypothesis:** Loss values differ.

**Rejection Criteria:** |computed - reference| > 1e-6.

**TPS Principle:** Baseline comparison

**Evidence Required:**
- [ ] PyTorch/sklearn comparison
- [ ] Numerical stability at extremes
- [ ] Gradient verification

---

### NR-13: Optimizer State Correctness

**Claim:** Optimizers maintain correct state updates.

**Falsification Test:**
```bash
cargo test --package entrenar --test optimizer_state
```

**Null Hypothesis:** State diverges from reference.

**Rejection Criteria:** Parameter update difference > 1e-6 after N steps.

**TPS Principle:** Step-by-step verification

**Evidence Required:**
- [ ] PyTorch optimizer comparison
- [ ] Weight decay verification
- [ ] LR scheduler correctness

---

### NR-14: Normalization Layer Correctness

**Claim:** BatchNorm/LayerNorm/RMSNorm produce correct outputs.

**Falsification Test:**
```bash
cargo test --package trueno --test normalization
```

**Null Hypothesis:** Normalized outputs incorrect.

**Rejection Criteria:** Mean ≠ 0 or variance ≠ 1 (within tolerance).

**TPS Principle:** Statistical verification

**Evidence Required:**
- [ ] Output statistics verification
- [ ] Training vs inference mode
- [ ] PyTorch comparison

---

### NR-15: Matrix Multiplication Numerical Stability

**Claim:** Matmul handles ill-conditioned matrices gracefully.

**Falsification Test:**
```bash
cargo test --package trueno --test matmul_stability
```

**Null Hypothesis:** Unstable under ill-conditioning.

**Rejection Criteria:** Error exceeds theoretical bounds for condition number.

**TPS Principle:** Graceful degradation

**Evidence Required:**
- [ ] Condition number analysis
- [ ] Error bound documentation
- [ ] Higham reference compliance

---

## Section 5: Performance & Waste (Muda) Elimination [15 Items]

*Toyota Way efficiency principles*

### PW-01: 5× PCIe Rule Validation

**Claim:** GPU dispatch beneficial only when compute > 5× transfer time.

**Falsification Test:**
```bash
cargo bench --package trueno -- pcie_crossover
```

**Null Hypothesis:** 5× rule incorrect for hardware.

**Rejection Criteria:** Optimal crossover differs from 5× by >50%.

**TPS Principle:** Cost-based backend selection

**Evidence Required:**
- [ ] Crossover on 3+ GPU types
- [ ] PCIe generation impact
- [ ] Reference: Gregg & Hazelwood (2011)

---

### PW-02: SIMD Speedup Verification

**Claim:** AVX2 provides >2× speedup over scalar.

**Falsification Test:**
```bash
cargo bench --package trueno -- simd_speedup
```

**Null Hypothesis:** Speedup ≤2×.

**Rejection Criteria:** Geometric mean ≤ 2×.

**TPS Principle:** Muda (Waiting) reduction

**Evidence Required:**
- [ ] Per-operation speedups
- [ ] Memory bandwidth analysis
- [ ] Autovectorization comparison

---

### PW-03: WASM Performance Ratio

**Claim:** WASM achieves 50-90% native performance.

**Falsification Test:**
```bash
./scripts/wasm_native_comparison.sh
```

**Null Hypothesis:** Performance outside range.

**Rejection Criteria:** Geometric mean < 50%.

**TPS Principle:** Edge deployment efficiency

**Evidence Required:**
- [ ] Browser benchmarks
- [ ] SIMD128 comparison
- [ ] Reference: Haas (2017), Lin (2020)

---

### PW-04: Inference Latency SLA

**Claim:** realizar <50ms/token on CPU (AVX2).

**Falsification Test:**
```bash
realizar bench --tokens 100
```

**Null Hypothesis:** Latency exceeds 50ms.

**Rejection Criteria:** Median > 50ms or P99 > 100ms.

**TPS Principle:** Muda (Waiting) elimination

**Evidence Required:**
- [ ] Latency distribution
- [ ] Model/quant specification
- [ ] Hardware specification

---

### PW-05: Batch Processing Efficiency

**Claim:** Batching provides near-linear throughput scaling.

**Falsification Test:**
```bash
cargo bench --package realizar -- batch_scaling
```

**Null Hypothesis:** Sublinear scaling.

**Rejection Criteria:** batch_size=32 throughput < 20× batch_size=1.

**TPS Principle:** Resource utilization

**Evidence Required:**
- [ ] Throughput vs batch curve
- [ ] Memory overhead analysis
- [ ] Optimal batch determination

---

### PW-06: Parallel Scaling Efficiency

**Claim:** Multi-threaded operations scale with core count.

**Falsification Test:**
```bash
cargo bench --package trueno -- parallel_scaling
```

**Null Hypothesis:** Efficiency < 50%.

**Rejection Criteria:** 8-core speedup < 4×.

**TPS Principle:** Resource utilization

**Evidence Required:**
- [ ] Scaling curve
- [ ] Amdahl's Law analysis
- [ ] Lock contention measurement

---

### PW-07: Model Loading Time

**Claim:** GGUF loading <2s per GB.

**Falsification Test:**
```bash
realizar bench --load-only
```

**Null Hypothesis:** Loading > 2s/GB.

**Rejection Criteria:** Time > 2 × size_gb seconds.

**TPS Principle:** Muda (Waiting)

**Evidence Required:**
- [ ] mmap vs read comparison
- [ ] SSD/HDD measurements
- [ ] Lazy loading verification

---

### PW-08: Startup Time

**Claim:** CLI startup <100ms.

**Falsification Test:**
```bash
hyperfine --warmup 3 'batuta --version'
```

**Null Hypothesis:** Startup > 100ms.

**Rejection Criteria:** Median > 100ms.

**TPS Principle:** Developer experience

**Evidence Required:**
- [ ] Cold/warm measurements
- [ ] Initialization breakdown
- [ ] Dynamic linking overhead

---

### PW-09: Test Suite Performance

**Claim:** Full test suite <5 minutes.

**Falsification Test:**
```bash
time cargo nextest run --workspace
```

**Null Hypothesis:** Suite > 5 minutes.

**Rejection Criteria:** Runtime > 5 minutes.

**TPS Principle:** Muda (Waiting) in CI

**Evidence Required:**
- [ ] Parallelization efficiency
- [ ] Slowest test identification
- [ ] I/O vs CPU analysis

---

### PW-10: Overprocessing Detection

**Claim:** Model complexity justified by improvement over baseline.

**Falsification Test:**
```bash
./scripts/complexity_justification_audit.sh
```

**Null Hypothesis:** Unjustified complexity exists.

**Rejection Criteria:** Complex model with <5% improvement over simple baseline.

**TPS Principle:** Muda (Overprocessing)

**Evidence Required:**
- [ ] Complexity metrics
- [ ] Baseline comparison
- [ ] Justification documentation

---

### PW-11: Zero-Copy Operation Verification

**Claim:** Hot paths operate without allocation.

**Falsification Test:**
```bash
cargo test --package trueno --test allocation_free
```

**Null Hypothesis:** Hot paths allocate.

**Rejection Criteria:** Any allocation in hot path.

**TPS Principle:** Muda (Motion) - memory efficiency

**Evidence Required:**
- [ ] Custom allocator instrumentation
- [ ] Heaptrack analysis
- [ ] Hot path identification

---

### PW-12: Cache Efficiency

**Claim:** Publish status cache >90% hit ratio.

**Falsification Test:**
```bash
cargo test --package batuta --test cache_efficiency
```

**Null Hypothesis:** Hit ratio < 90%.

**Rejection Criteria:** Steady-state hit ratio < 90%.

**TPS Principle:** Muda (Waiting) reduction

**Evidence Required:**
- [ ] Hit/miss statistics
- [ ] Cache size optimization
- [ ] Eviction policy analysis

---

### PW-13: Cost Model Accuracy

**Claim:** Backend selection predicts optimal choice >90%.

**Falsification Test:**
```bash
cargo test --package trueno --test cost_model
```

**Null Hypothesis:** Accuracy < 90%.

**Rejection Criteria:** Misprediction rate > 10%.

**TPS Principle:** Intelligent automation (Jidoka)

**Evidence Required:**
- [ ] Prediction vs actual comparison
- [ ] Error distribution
- [ ] Hardware calibration

---

### PW-14: Data Transport Minimization

**Claim:** Data movement minimized through co-location.

**Falsification Test:**
```bash
./scripts/data_transport_audit.sh
```

**Null Hypothesis:** Excessive data transport.

**Rejection Criteria:** Data crosses network boundary unnecessarily.

**TPS Principle:** Muda (Transport)

**Evidence Required:**
- [ ] Data locality analysis
- [ ] Network transfer measurements
- [ ] Co-location opportunities

---

### PW-15: Inventory (Data) Minimization

**Claim:** No hoarded unused data.

**Falsification Test:**
```bash
./scripts/data_usage_audit.sh
```

**Null Hypothesis:** Unused data exists.

**Rejection Criteria:** Data asset unused for >90 days without explicit retention justification.

**TPS Principle:** Muda (Inventory)

**Evidence Required:**
- [ ] Data usage tracking
- [ ] Retention policy enforcement
- [ ] Storage cost analysis

---

## Section 6: Safety & Formal Verification [10 Items]

*Jidoka automated safety, formal methods [32, 33, 40, 41]*

### SF-01: Unsafe Code Isolation

**Claim:** All unsafe code isolated in marked internal modules.

**Falsification Test:**
```bash
./scripts/unsafe_audit.sh
```

**Null Hypothesis:** Unsafe in public API.

**Rejection Criteria:** Unsafe block outside designated module.

**TPS Principle:** Jidoka — containment

**Evidence Required:**
- [ ] Unsafe block inventory
- [ ] Safety comment audit
- [ ] Encapsulation verification

*Reference: Jung et al. (2017) RustBelt*

---

### SF-02: Memory Safety Under Fuzzing

**Claim:** No memory safety violations under fuzzing.

**Falsification Test:**
```bash
cargo +nightly fuzz run fuzz_target -- -max_total_time=3600
```

**Null Hypothesis:** Fuzzing reveals safety issues.

**Rejection Criteria:** Any ASan/MSan/UBSan violation.

**TPS Principle:** Jidoka — defect detection

**Evidence Required:**
- [ ] Fuzzing campaign logs (>1M iterations)
- [ ] Corpus coverage
- [ ] Sanitizer configurations

---

### SF-03: Miri Undefined Behavior Detection

**Claim:** Core operations pass Miri validation.

**Falsification Test:**
```bash
cargo +nightly miri test --package trueno
```

**Null Hypothesis:** Miri detects UB.

**Rejection Criteria:** Any Miri error.

**TPS Principle:** Jidoka — automatic UB detection

**Evidence Required:**
- [ ] Full Miri test logs
- [ ] Stacked borrows validation
- [ ] Tree borrows (if enabled)

---

### SF-04: Formal Safety Properties

**Claim:** Safety-critical components have formal proofs.

**Falsification Test:**
```bash
./scripts/formal_verification.sh
```

**Null Hypothesis:** Proofs fail or missing.

**Rejection Criteria:** Safety property unproven for critical path.

**TPS Principle:** Formal verification requirement

**Evidence Required:**
- [ ] Kani/Creusot proofs
- [ ] Safety bubble verification
- [ ] Adversarial robustness bounds

*Reference: [32, 33] Seshia et al., [65] Katz et al. "Reluplex"*

---

### SF-05: Adversarial Robustness Verification

**Claim:** Models tested against adversarial examples.

**Falsification Test:**
```bash
cargo test --package aprender --test adversarial_robustness
```

**Null Hypothesis:** Adversarial vulnerability exists.

**Rejection Criteria:** Model fails under documented attack types.

**TPS Principle:** AI Safety requirement

**Evidence Required:**
- [ ] Attack type coverage
- [ ] Perturbation bounds
- [ ] Certified robustness (if applicable)

*Reference: [40] "Adversarial Robustness Verification Survey", [66] Madry et al.*

---

### SF-06: Thread Safety (Send + Sync)

**Claim:** All Send + Sync implementations correct.

**Falsification Test:**
```bash
cargo test --package trueno --test thread_safety
./scripts/tsan_test.sh
```

**Null Hypothesis:** Data races exist.

**Rejection Criteria:** TSan detects any race.

**TPS Principle:** Jidoka — race detection

**Evidence Required:**
- [ ] TSan clean run
- [ ] Concurrent patterns tested
- [ ] Lock ordering verification

---

### SF-07: Resource Leak Prevention

**Claim:** No resource leaks.

**Falsification Test:**
```bash
valgrind --leak-check=full ./target/release/test_binary
```

**Null Hypothesis:** Leaks exist.

**Rejection Criteria:** "definitely lost" > 0 bytes.

**TPS Principle:** Muda (Defects)

**Evidence Required:**
- [ ] Valgrind report
- [ ] Long-running stress test
- [ ] GPU resource tracking

---

### SF-08: Panic Safety

**Claim:** Panics don't corrupt data structures.

**Falsification Test:**
```bash
cargo test --package trueno --test panic_safety
```

**Null Hypothesis:** Panic corrupts state.

**Rejection Criteria:** Post-panic access causes UB.

**TPS Principle:** Graceful degradation

**Evidence Required:**
- [ ] Panic injection testing
- [ ] Drop order verification
- [ ] catch_unwind safety

---

### SF-09: Input Validation

**Claim:** All public APIs validate inputs.

**Falsification Test:**
```bash
cargo test --package trueno --test input_validation
```

**Null Hypothesis:** Invalid input causes UB/panic.

**Rejection Criteria:** Any panic from malformed input.

**TPS Principle:** Poka-Yoke — error prevention

**Evidence Required:**
- [ ] Boundary testing
- [ ] Type confusion prevention
- [ ] Size overflow checking

---

### SF-10: Supply Chain Security

**Claim:** All dependencies audited.

**Falsification Test:**
```bash
cargo audit
cargo deny check
```

**Null Hypothesis:** Vulnerable dependencies exist.

**Rejection Criteria:** Known vulnerability or unmaintained critical dependency.

**TPS Principle:** Jidoka — supply chain circuit breaker

**Evidence Required:**
- [ ] cargo audit clean
- [ ] cargo deny policy
- [ ] Review process documentation

*Reference: [29, 31] Supply Chain Risk*

---

## Section 7: Jidoka Automated Gates [10 Items]

*CI/CD circuit breakers, automated quality enforcement*

### JA-01: Pre-Commit Hook Enforcement

**Claim:** Pre-commit hooks catch basic issues locally.

**Falsification Test:**
```bash
./scripts/precommit_effectiveness.sh
```

**Null Hypothesis:** Issues reach CI that pre-commit should catch.

**Rejection Criteria:** >5% of CI failures are pre-commit-detectable.

**TPS Principle:** Jidoka — early detection

**Evidence Required:**
- [ ] Pre-commit configuration
- [ ] CI failure analysis
- [ ] Hook effectiveness metrics

---

### JA-02: Automated Sovereignty Linting

**Claim:** Static analysis catches sovereignty violations.

**Falsification Test:**
```bash
./scripts/sovereignty_lint.sh
```

**Null Hypothesis:** Violations pass linting.

**Rejection Criteria:** Known violation pattern not flagged.

**TPS Principle:** Jidoka — automated sovereignty check

**Evidence Required:**
- [ ] Lint rule coverage
- [ ] False positive rate
- [ ] Known pattern detection

---

### JA-03: Data Drift Circuit Breaker

**Claim:** Training stops on significant data drift.

**Falsification Test:**
```bash
cargo test --package batuta --test drift_circuit_breaker
```

**Null Hypothesis:** Training continues despite drift.

**Rejection Criteria:** Training completes with >20% distribution shift.

**TPS Principle:** Jidoka — automatic stop

**Evidence Required:**
- [ ] Drift detection implementation
- [ ] Threshold calibration
- [ ] Alert/stop verification

---

### JA-04: Model Performance Regression Gate

**Claim:** Deployment blocked on performance regression.

**Falsification Test:**
```bash
./scripts/regression_gate_test.sh
```

**Null Hypothesis:** Regression reaches production.

**Rejection Criteria:** Model with <baseline metrics deploys.

**TPS Principle:** Jidoka — quality gate

**Evidence Required:**
- [ ] Baseline metric tracking
- [ ] Regression detection logic
- [ ] Gate enforcement verification

---

### JA-05: Fairness Metric Circuit Breaker

**Claim:** Training stops on fairness regression.

**Falsification Test:**
```bash
cargo test --package aprender --test fairness_circuit_breaker
```

**Null Hypothesis:** Fairness regression undetected.

**Rejection Criteria:** Protected class metric degrades >5% without alert.

**TPS Principle:** Jidoka — ethical safeguard

**Evidence Required:**
- [ ] Fairness metric definition
- [ ] Threshold configuration
- [ ] Enforcement verification

---

### JA-06: Latency SLA Circuit Breaker

**Claim:** Deployment blocked on latency regression.

**Falsification Test:**
```bash
./scripts/latency_gate_test.sh
```

**Null Hypothesis:** Slow model deploys.

**Rejection Criteria:** P99 latency exceeds SLA in staging.

**TPS Principle:** Jidoka — SLA enforcement

**Evidence Required:**
- [ ] SLA definition
- [ ] Staging benchmark
- [ ] Gate enforcement

---

### JA-07: Memory Footprint Gate

**Claim:** Deployment blocked on excessive memory.

**Falsification Test:**
```bash
./scripts/memory_gate_test.sh
```

**Null Hypothesis:** Memory-hungry model deploys.

**Rejection Criteria:** Peak memory exceeds target by >20%.

**TPS Principle:** Muda (Inventory) prevention

**Evidence Required:**
- [ ] Memory budget
- [ ] Profiling automation
- [ ] Gate enforcement

*Reference: [69] Strubell et al. "Energy and Policy Considerations"*

---

### JA-08: Security Scan Gate

**Claim:** Build blocked on security findings.

**Falsification Test:**
```bash
./scripts/security_gate_test.sh
```

**Null Hypothesis:** Vulnerability reaches artifact.

**Rejection Criteria:** High/Critical vulnerability in build.

**TPS Principle:** Jidoka — security gate

**Evidence Required:**
- [ ] SAST/DAST integration
- [ ] Severity thresholds
- [ ] Gate enforcement

---

### JA-09: License Compliance Gate

**Claim:** Build blocked on license violation.

**Falsification Test:**
```bash
cargo deny check licenses
```

**Null Hypothesis:** Non-compliant license passes.

**Rejection Criteria:** Disallowed license in dependency tree.

**TPS Principle:** Legal controls pillar

**Evidence Required:**
- [ ] License policy
- [ ] Allowlist/denylist
- [ ] Gate enforcement

---

### JA-10: Documentation Gate

**Claim:** PR blocked without documentation updates.

**Falsification Test:**
```bash
./scripts/doc_gate_test.sh
```

**Null Hypothesis:** Undocumented API merges.

**Rejection Criteria:** Public API change without doc update.

**TPS Principle:** Knowledge transfer

**Evidence Required:**
- [ ] Doc coverage tracking
- [ ] API change detection
- [ ] Gate enforcement

---

## Section 8: Model Cards & Auditability [10 Items]

*Governance artifacts [42, 45-47]*

### MA-01: Model Card Completeness

**Claim:** Every model has complete Model Card.

**Falsification Test:**
```bash
./scripts/model_card_audit.sh
```

**Null Hypothesis:** Incomplete Model Cards exist.

**Rejection Criteria:** Required section missing.

**TPS Principle:** Governance documentation

**Evidence Required:**
- [ ] Model Card schema
- [ ] Completeness check
- [ ] Update tracking

*Reference: [45, 46] Model Cards*

---

### MA-02: Datasheet Completeness

**Claim:** Every dataset has Datasheet.

**Falsification Test:**
```bash
./scripts/datasheet_audit.sh
```

**Null Hypothesis:** Incomplete Datasheets exist.

**Rejection Criteria:** Required section missing.

**TPS Principle:** Data governance

**Evidence Required:**
- [ ] Datasheet schema
- [ ] Completeness check
- [ ] Update tracking

*Reference: [47] Datasheets for Datasets*

---

### MA-03: Model Card Accuracy

**Claim:** Model Card reflects current model behavior.

**Falsification Test:**
```bash
./scripts/model_card_accuracy.sh
```

**Null Hypothesis:** Model Card is stale.

**Rejection Criteria:** Documented behavior differs from actual.

**TPS Principle:** Genchi Genbutsu — verify claims

**Evidence Required:**
- [ ] Behavior verification
- [ ] Last-updated tracking
- [ ] Automated drift detection

---

### MA-04: Decision Logging Completeness

**Claim:** All model decisions are logged with sufficient context.

**Falsification Test:**
```bash
cargo test --package realizar --test decision_logging
```

**Null Hypothesis:** Decisions logged without context.

**Rejection Criteria:** Log entry missing version, input hash, or timestamp.

**TPS Principle:** Auditability requirement

**Evidence Required:**
- [ ] Log schema verification
- [ ] Context completeness
- [ ] Reconstruction capability

*Reference: [42, 43] Auditability in ML*

---

### MA-05: Provenance Chain Completeness

**Claim:** Full provenance from data to prediction.

**Falsification Test:**
```bash
./scripts/provenance_audit.sh
```

**Null Hypothesis:** Provenance gaps exist.

**Rejection Criteria:** Any step in data→model→prediction without tracking.

**TPS Principle:** Audit trail integrity

**Evidence Required:**
- [ ] Data lineage
- [ ] Training lineage
- [ ] Inference lineage

---

### MA-06: Version Tracking

**Claim:** All model versions uniquely identified.

**Falsification Test:**
```bash
cargo test --package pacha --test version_tracking
```

**Null Hypothesis:** Ambiguous versions exist.

**Rejection Criteria:** Two models with same version identifier differ.

**TPS Principle:** Configuration management

**Evidence Required:**
- [ ] Version schema
- [ ] Uniqueness verification
- [ ] Content addressing

---

### MA-07: Rollback Capability

**Claim:** Any model version can be restored.

**Falsification Test:**
```bash
./scripts/rollback_test.sh
```

**Null Hypothesis:** Rollback fails.

**Rejection Criteria:** Historical version unavailable or corrupted.

**TPS Principle:** Recovery capability

**Evidence Required:**
- [ ] Version retention policy
- [ ] Restoration testing
- [ ] Integrity verification

---

### MA-08: A/B Test Logging

**Claim:** A/B tests fully logged for analysis.

**Falsification Test:**
```bash
./scripts/ab_test_audit.sh
```

**Null Hypothesis:** A/B test data incomplete.

**Rejection Criteria:** Assignment or outcome data missing.

**TPS Principle:** Scientific experimentation

**Evidence Required:**
- [ ] Assignment logging
- [ ] Outcome logging
- [ ] Analysis reproducibility

---

### MA-09: Bias Audit Trail

**Claim:** Bias assessments documented per model.

**Falsification Test:**
```bash
./scripts/bias_audit.sh
```

**Null Hypothesis:** Bias assessment missing.

**Rejection Criteria:** Model without documented bias analysis.

**TPS Principle:** Ethical governance

**Evidence Required:**
- [ ] Bias metric definition
- [ ] Assessment results
- [ ] Mitigation documentation

---

### MA-10: Incident Response Logging

**Claim:** Model incidents fully documented.

**Falsification Test:**
```bash
./scripts/incident_audit.sh
```

**Null Hypothesis:** Incident documentation incomplete.

**Rejection Criteria:** Incident without root cause and remediation.

**TPS Principle:** Kaizen — learning from failures

**Evidence Required:**
- [ ] Incident log completeness
- [ ] Root cause analysis
- [ ] Preventive measures

---

## Section 9: Cross-Platform & API Completeness [5 Items]

*Portability and coverage claims*

### CP-01: Linux Distribution Compatibility

**Claim:** Stack runs on major Linux distributions.

**Falsification Test:**
```bash
./scripts/linux_distro_test.sh
```

**Null Hypothesis:** Distribution-specific failures.

**Rejection Criteria:** Failure on Ubuntu, Fedora, Debian, or Arch.

**TPS Principle:** Portability

**Evidence Required:**
- [ ] CI matrix
- [ ] glibc requirements
- [ ] Kernel requirements

---

### CP-02: macOS/Windows Compatibility

**Claim:** Stack runs on macOS and Windows.

**Falsification Test:**
```bash
./scripts/cross_platform_test.sh
```

**Null Hypothesis:** Platform-specific failures.

**Rejection Criteria:** Failure on macOS ARM64 or Windows 11.

**TPS Principle:** Portability

**Evidence Required:**
- [ ] CI matrix
- [ ] Platform-specific testing
- [ ] Path handling verification

---

### CP-03: WASM Browser Compatibility

**Claim:** WASM build works in major browsers.

**Falsification Test:**
```bash
./scripts/browser_test.sh
```

**Null Hypothesis:** Browser-specific failures.

**Rejection Criteria:** Failure in Chrome, Firefox, or Safari.

**TPS Principle:** Edge deployment

**Evidence Required:**
- [ ] Browser version matrix
- [ ] WebGPU detection
- [ ] SharedArrayBuffer handling

---

### CP-04: NumPy API Coverage

**Claim:** trueno supports >90% of NumPy operations.

**Falsification Test:**
```bash
python scripts/numpy_api_coverage.py
```

**Null Hypothesis:** Coverage ≤90%.

**Rejection Criteria:** <90% of numpy.ndarray methods.

**TPS Principle:** API completeness

**Evidence Required:**
- [ ] API enumeration
- [ ] Coverage matrix
- [ ] Gap documentation

---

### CP-05: sklearn Estimator Coverage

**Claim:** aprender supports >80% of sklearn estimators.

**Falsification Test:**
```bash
python scripts/sklearn_coverage.py
```

**Null Hypothesis:** Coverage ≤80%.

**Rejection Criteria:** <80% of sklearn estimators.

**TPS Principle:** API completeness

**Evidence Required:**
- [ ] Estimator enumeration
- [ ] Implementation matrix
- [ ] Priority ranking

---

## Section 10: Architectural Invariants [5 Items] — CRITICAL

*Hard requirements. Any failure = project FAIL. No exceptions.*

### AI-01: Declarative YAML Configuration

**Claim:** Project offers full functionality via declarative YAML without code.

**Falsification Test:**
```bash
./scripts/declarative_coverage_audit.sh
```

**Null Hypothesis:** Code required for basic functionality.

**Rejection Criteria (CRITICAL):**
- Any core feature unavailable via YAML config
- User must write Rust/code to use basic functionality

**TPS Principle:** Poka-Yoke — enable non-developers

**Evidence Required:**
- [ ] YAML schema documentation
- [ ] Feature coverage matrix (YAML vs API)
- [ ] No-code quickstart example

**Severity:** CRITICAL — Project FAIL if not met

---

### AI-02: Zero Scripting in Production

**Claim:** No Python/JavaScript/Lua in production runtime paths.

**Falsification Test:**
```bash
./scripts/scripting_audit.sh
cargo tree --edges no-dev | grep -E "pyo3|napi|mlua"
```

**Null Hypothesis:** Scripting language dependencies exist in runtime.

**Rejection Criteria (CRITICAL):**
- Any `.py`, `.js`, `.lua` in src/ or runtime
- pyo3, napi-rs, mlua in non-dev dependencies
- Interpreter embedded in binary

**TPS Principle:** Jidoka — type safety, determinism

**Evidence Required:**
- [ ] Dependency audit (no scripting runtimes)
- [ ] File extension audit
- [ ] Build artifact inspection

**Severity:** CRITICAL — Project FAIL if not met

---

### AI-03: Pure Rust Testing (No Jest/Pytest)

**Claim:** All tests written in Rust, no external test frameworks.

**Falsification Test:**
```bash
find . -name "*.test.js" -o -name "test_*.py" -o -name "*_test.py"
```

**Null Hypothesis:** Non-Rust test files exist.

**Rejection Criteria (CRITICAL):**
- Any Jest, Mocha, Pytest, unittest files
- package.json with test scripts
- requirements-dev.txt with pytest

**TPS Principle:** Zero scripting policy

**Evidence Required:**
- [ ] Test file audit
- [ ] CI config uses cargo test/nextest only
- [ ] No node_modules or __pycache__ in repo

**Severity:** CRITICAL — Project FAIL if not met

**Reference:** `jugar/probar` as canonical pure-Rust testing pattern

---

### AI-04: WASM-First Browser Support

**Claim:** Browser functionality via WASM, not JavaScript.

**Falsification Test:**
```bash
./scripts/wasm_browser_audit.sh
```

**Null Hypothesis:** JavaScript used for browser features.

**Rejection Criteria (CRITICAL):**
- JS files beyond minimal WASM glue
- npm dependencies for core functionality
- React/Vue/Svelte instead of WASM UI

**TPS Principle:** Zero scripting, sovereignty

**Evidence Required:**
- [ ] WASM build target exists
- [ ] JS limited to wasm-bindgen glue
- [ ] No npm runtime dependencies

**Severity:** CRITICAL — Project FAIL if not met

---

### AI-05: Declarative Schema Validation

**Claim:** YAML configs validated against typed schema.

**Falsification Test:**
```bash
cargo test --package $PKG --test yaml_schema_validation
```

**Null Hypothesis:** YAML accepted without schema validation.

**Rejection Criteria (CRITICAL):**
- Invalid YAML silently accepted
- No JSON Schema or serde validation
- Runtime panics on bad config

**TPS Principle:** Poka-Yoke — prevent config errors

**Evidence Required:**
- [ ] JSON Schema or Rust struct with serde
- [ ] Validation error messages
- [ ] Invalid config test cases

**Severity:** CRITICAL — Project FAIL if not met

---

## Section 11: Proactive Bug Hunting [20 Items]

*Popperian falsification applied to proactive defect discovery. Rather than waiting for bugs to manifest, we actively attempt to falsify the claim "this code is correct."*

**Philosophy:** "A theory that explains everything, explains nothing." — Karl Popper. Bug hunting operationalizes this: we systematically attempt to break code, not merely verify it works.

**Integration with OIP:** These checks leverage OIP's SBFL (Tarantula/Ochiai/DStar), defect classification, and RAG enhancement to proactively identify bugs before they reach production.

---

### BH-01: Mutation-Based Invariant Falsification (FDV)

**Claim:** Code invariants survive mutation testing with >80% kill rate.

**Falsification Test:**
```bash
batuta bug-hunter falsify --target src/ --min-kill-rate 80
cargo mutants --package $PKG --timeout 60 --in-place
```

**Null Hypothesis:** Invariants are weak; mutants survive undetected.

**Rejection Criteria:**
- Mutation kill rate < 80%
- Equivalent mutant ratio > 20%
- Critical path mutants survive

**TPS Principle:** Genchi Genbutsu — test the actual behavior, not assumptions

**Evidence Required:**
- [ ] Mutation testing report
- [ ] Per-function kill rates
- [ ] Surviving mutant analysis

*Reference: Groce et al. (2018) "Falsification-driven verification and testing." Automated Software Engineering.*

---

### BH-02: SBFL Without Failing Tests (SBEST)

**Claim:** Stack traces from crashes/logs enable fault localization without explicit failing tests.

**Falsification Test:**
```bash
batuta bug-hunter hunt --stack-trace crash.log --coverage baseline.lcov
oip localize --passed-coverage baseline.lcov --formula ochiai --top-n 10
```

**Null Hypothesis:** Stack trace provides no signal for bug location.

**Rejection Criteria:**
- Buggy method not in top-10 suspicious
- False positive rate > 50%
- Stack trace reachability < 70%

**TPS Principle:** Jidoka — detect abnormality from available signals

**Evidence Required:**
- [ ] Stack trace parsing accuracy
- [ ] SBFL ranking correlation with actual bugs
- [ ] Reachability analysis from crash point

*Reference: Rafi et al. (2025) "SBEST: Spectrum-Based Fault Localization Without Fault-Triggering Tests." Empirical Software Engineering.*

---

### BH-03: LLM-Augmented Static Analysis (LLIFT Pattern)

**Claim:** LLM filtering reduces static analysis false positives by >50%.

**Falsification Test:**
```bash
batuta bug-hunter analyze --llm-filter --category memory-safety
cargo clippy --all-targets 2>&1 | batuta bug-hunter filter --model gpt-4
```

**Null Hypothesis:** LLM provides no improvement over raw static analysis.

**Rejection Criteria:**
- False positive reduction < 50%
- True positive loss > 10%
- LLM hallucination rate > 5%

**TPS Principle:** Jidoka — human-machine collaboration for defect detection

**Evidence Required:**
- [ ] Baseline clippy/miri warning count
- [ ] Post-LLM-filter warning count
- [ ] Manual verification of filtered warnings

*Reference: Li et al. (2024) "Enhancing Static Analysis for Practical Bug Detection: An LLM-Integrated Approach." OOPSLA.*

---

### BH-04: Targeted Unsafe Rust Fuzzing (FourFuzz Pattern)

**Claim:** Fuzzing coverage of `unsafe` blocks exceeds 80%.

**Falsification Test:**
```bash
batuta bug-hunter fuzz --target-unsafe --duration 1h --min-coverage 80
cargo +nightly fuzz run fuzz_target -- -max_total_time=3600
```

**Null Hypothesis:** Unsafe blocks remain inadequately tested.

**Rejection Criteria:**
- Unsafe block coverage < 80%
- No crashes found in known-vulnerable patterns
- Fuzzing corpus stagnation

**TPS Principle:** Genchi Genbutsu — go to the source (unsafe = highest risk)

**Evidence Required:**
- [ ] Unsafe block inventory
- [ ] Per-block coverage report
- [ ] Crash triage and CVE correlation

*Reference: FourFuzz (2025) "Targeted Fuzzing for Unsafe Rust Code: Leveraging Selective Instrumentation." arXiv.*

---

### BH-05: Hybrid Concolic + SBFL (COTTONTAIL Pattern)

**Claim:** Concolic execution reaches >90% branch coverage on complex conditionals.

**Falsification Test:**
```bash
batuta bug-hunter deep-hunt --concolic --sbfl-ensemble --target src/pipeline.rs
```

**Null Hypothesis:** Path explosion prevents meaningful coverage.

**Rejection Criteria:**
- Branch coverage < 90% on target
- Constraint solver timeout rate > 20%
- SBFL ranking accuracy < 70%

**TPS Principle:** Kaizen — continuous improvement through deeper analysis

**Evidence Required:**
- [ ] Branch coverage report
- [ ] Solver statistics
- [ ] Generated test corpus

*Reference: Böhme et al. (2026) "COTTONTAIL: LLM-Driven Concolic Execution." IEEE S&P.*

---

### BH-06: Defect Category Classification

**Claim:** Defects are automatically classified into OIP categories with >85% accuracy.

**Falsification Test:**
```bash
batuta bug-hunter classify --model oip-classifier --validation-set ground-truth.json
oip extract-training-data --repo . --validate
```

**Null Hypothesis:** Classification is no better than random.

**Rejection Criteria:**
- Classification accuracy < 85%
- Category confusion > 15%
- Novel defect types misclassified

**TPS Principle:** Poka-Yoke — route defects to appropriate remediation

**Evidence Required:**
- [ ] Confusion matrix
- [ ] Per-category precision/recall
- [ ] Ground truth validation

---

### BH-07: Git History Defect Mining

**Claim:** Historical defect patterns predict future bug locations.

**Falsification Test:**
```bash
batuta bug-hunter mine --repo . --lookback 1y --predict next-commit
oip extract-training-data --repo . --max-commits 500
```

**Null Hypothesis:** History provides no predictive signal.

**Rejection Criteria:**
- Predictive AUC < 0.7
- Churn correlation < 0.3
- False alarm rate > 30%

**TPS Principle:** Kaizen — learn from past defects

**Evidence Required:**
- [ ] Historical defect density map
- [ ] Prediction accuracy on holdout
- [ ] Churn-to-defect correlation

*Reference: Kamei et al. (2024) "ML-based defect prediction." Journal of Systems and Software.*

---

### BH-08: Property-Based Test Generation

**Claim:** Property-based tests discover edge cases missed by unit tests.

**Falsification Test:**
```bash
batuta bug-hunter proptest --target src/lib.rs --iterations 10000
cargo test --features proptest
```

**Null Hypothesis:** Property tests find no new bugs.

**Rejection Criteria:**
- No new failures discovered
- Property coverage < 70%
- Shrinking fails to minimize

**TPS Principle:** Genchi Genbutsu — explore the actual input space

**Evidence Required:**
- [ ] Property test coverage
- [ ] Discovered edge cases
- [ ] Shrunk failure examples

*Reference: Facebook propfuzz (2021) "Property-based testing meets fuzzing."*

---

### BH-09: Cross-Function Data Flow Analysis

**Claim:** Inter-procedural data flow analysis detects taint propagation.

**Falsification Test:**
```bash
batuta bug-hunter dataflow --taint-sources user_input --taint-sinks sql_query
cargo clippy -- -W clippy::unwrap_used
```

**Null Hypothesis:** Taint analysis produces excessive false positives.

**Rejection Criteria:**
- False positive rate > 40%
- Missed taint paths > 10%
- Analysis timeout on large functions

**TPS Principle:** Poka-Yoke — prevent injection at design time

**Evidence Required:**
- [ ] Taint path enumeration
- [ ] Source-to-sink validation
- [ ] Sanitizer coverage

---

### BH-10: Ensemble Bug Ranking

**Claim:** Ensemble of SBFL + ML + LLM provides better ranking than any single method.

**Falsification Test:**
```bash
batuta bug-hunter ensemble --sbfl ochiai --ml oip-model --llm gpt-4 --fusion rrf
oip localize --ensemble --include-churn
```

**Null Hypothesis:** Ensemble provides no improvement over best single method.

**Rejection Criteria:**
- Ensemble MRR < best single method MRR
- Rank correlation < 0.8
- Computational overhead > 3x

**TPS Principle:** Heijunka — balance multiple analysis techniques

**Evidence Required:**
- [ ] Per-method ranking accuracy
- [ ] Ensemble vs single comparison
- [ ] RRF fusion weights

*Reference: AutoFL (2024) "LLM-based fault localization with ensemble methods." ICSE.*

---

### BH-11: Spec-Driven Bug Hunting

**Claim:** Bug hunting guided by specification files yields higher-relevance findings than undirected analysis.

**Falsification Test:**
```bash
# Hunt bugs related to claims in a specification
batuta bug-hunter hunt --spec docs/specifications/popperian-falsification-checklist.md

# Parse spec, find implementing code, hunt bugs in that code
batuta bug-hunter analyze --spec docs/specifications/serving-api.md --section "Authentication"
```

**Null Hypothesis:** Spec-driven hunting provides no improvement over undirected hunting.

**Rejection Criteria:**
- Finding relevance < 80% (findings unrelated to spec claims)
- Spec parsing fails on standard markdown headers
- Implementation-to-claim mapping accuracy < 70%

**TPS Principle:** Genchi Genbutsu — "go to the source" (the spec defines intent)

**Evidence Required:**
- [ ] Spec claim parser (markdown headers → claims)
- [ ] Code-to-claim mapping via grep/AST
- [ ] Relevance score comparison (spec-driven vs undirected)

*Reference: Requirements-based testing literature; IEEE 829 Test Documentation Standard.*

---

### BH-12: PMAT Work Ticket Integration

**Claim:** Bug hunting scoped to PMAT work tickets focuses effort on active development areas.

**Falsification Test:**
```bash
# Hunt bugs in code areas defined by PMAT ticket
batuta bug-hunter hunt --ticket PMAT-1234
batuta bug-hunter hunt --ticket .pmat/tickets/feature-x.md

# Ticket defines: affected files, expected behavior, acceptance criteria
```

**Null Hypothesis:** Ticket-scoped hunting is no more efficient than project-wide hunting.

**Rejection Criteria:**
- Ticket parsing fails for standard PMAT format
- Scoped analysis time > 50% of full analysis time
- Findings outside ticket scope > 20%

**TPS Principle:** Heijunka — level the workload by focusing on current priorities

**Evidence Required:**
- [ ] PMAT ticket parser (YAML/markdown)
- [ ] File scope extraction from tickets
- [ ] Scope adherence metrics

*Reference: PMAT Methodology, Certeza Quality Gates.*

---

### BH-13: Scoped Analysis

**Claim:** Scoped analysis (--lib, --bin, --path) reduces noise and analysis time.

**Falsification Test:**
```bash
# Scope to library code only
batuta bug-hunter analyze --lib

# Scope to specific binary
batuta bug-hunter analyze --bin batuta

# Scope to specific path
batuta bug-hunter analyze --path src/oracle

# Exclude test code explicitly
batuta bug-hunter analyze --no-tests
```

**Null Hypothesis:** Scoped analysis provides same signal-to-noise ratio as full analysis.

**Rejection Criteria:**
- `--lib` includes binary-only code
- `--bin` includes library-only code
- `--path` scope leaks to other directories
- `--no-tests` fails to exclude `#[cfg(test)]` modules

**TPS Principle:** Muda elimination — reduce waste by analyzing only relevant code

**Evidence Required:**
- [ ] Cargo manifest parsing for lib/bin targets
- [ ] Path isolation verification
- [ ] Test code detection accuracy

---

### BH-14: Bidirectional Spec-Code Linking

**Claim:** Bug-hunter can update specification files with implementation locations and finding links.

**Falsification Test:**
```bash
# Hunt and update spec with findings
batuta bug-hunter hunt --spec docs/specifications/auth-spec.md --update-spec

# Expected spec update:
# ### AUTH-01: Token Validation
# - **Implementation**: `src/auth/token.rs:42` ✓
# - **Findings**: [BH-PAT-0012](src/auth/token.rs:87) - unwrap in error path
# - **Status**: ⚠️ 1 bug found
```

**Null Hypothesis:** Manual spec-code traceability is equally maintainable.

**Rejection Criteria:**
- Spec update corrupts existing content
- Links point to non-existent code
- Update fails to preserve markdown formatting
- No backup created before modification

**TPS Principle:** Visual Management — make quality status visible in the spec itself

**Evidence Required:**
- [ ] Safe spec modification (backup + atomic write)
- [ ] Link validation (file:line exists)
- [ ] Markdown preservation tests
- [ ] Incremental update (only changed sections)

*Reference: Traceability matrices in requirements engineering; IEEE 830.*

---

### BH-15: False Positive Suppression

**Claim:** Bug-hunter should suppress known false positive patterns to maintain signal quality.

**Falsification Test:**
```bash
# Issue #17: Intentional identical if-blocks should not be flagged
batuta bug-hunter analyze --suppress-intentional-patterns

# Known false positive patterns:
# - Mapper functions returning same enum for different conditions
# - Pattern matching with intentional fallthrough
# - Documentation strings containing pattern keywords
```

**Null Hypothesis:** All clippy warnings are actionable without filtering.

**Rejection Criteria:**
- Suppression removes true positives
- Suppression logic too aggressive (>10% valid findings removed)
- No way to disable suppression (`--no-suppress`)

**TPS Principle:** Poka-Yoke — mistake-proof the tool itself

**Evidence Required:**
- [ ] Heuristic for "mapper function" detection
- [ ] Heuristic for "intentional pattern" comments
- [ ] False positive rate measurement
- [ ] `#[allow(...)]` suggestion in findings

*Reference: GitHub Issue #17 - False positive for intentional identical if-blocks.*

---

### BH-16: Mutation-Based Fault Localization (MBFL)

**Claim:** Combining mutation analysis with fault localization improves accuracy over pure SBFL.

**Research Basis:** Papadakis & Le Traon (2015) "Metallaxis-FL: Mutation-based Fault Localization" - IEEE TSE. MBFL outperforms SBFL on 75% of real faults (Pearson et al., ICSE 2017).

**Falsification Test:**
```bash
# Run MBFL mode
batuta bug-hunter hunt --strategy mbfl

# Compare with pure SBFL
batuta bug-hunter hunt --strategy sbfl
batuta bug-hunter hunt --strategy hybrid --sbfl-weight 0.5

# Evaluate on known bug corpus
batuta bug-hunter benchmark --corpus defects4j --strategy mbfl
```

**Algorithm:**
```
Suspiciousness(L) = |killed_mutants(L) ∩ failing_tests| / |total_mutants(L)|
```

**Null Hypothesis:** SBFL alone is sufficient for accurate fault localization.

**Rejection Criteria:**
- MBFL does not improve ranking over SBFL on ≥50% of test cases
- MBFL runtime exceeds 10× SBFL runtime
- False positive rate increases by >5%

**TPS Principle:** Kaizen — continuous improvement through research integration

**Evidence Required:**
- [ ] MBFL scoring implementation
- [ ] Hybrid strategy with configurable weights
- [ ] Benchmark results on Defects4J or equivalent corpus
- [ ] Runtime comparison metrics

*Reference: Papadakis, M., & Le Traon, Y. (2015). IEEE TSE.*

---

### BH-17: Causal Fault Localization

**Claim:** Causal inference eliminates confounding from coincidental coverage patterns.

**Research Basis:** Baah et al. (2010) "Causal Inference for Statistical Fault Localization" - ISSTA. Reports 15-30% improvement in fault ranking.

**Falsification Test:**
```bash
# Run causal analysis mode
batuta bug-hunter hunt --causal

# Compare ACE scores with correlation-based SBFL
batuta bug-hunter hunt --causal --show-confounders
```

**Algorithm (Average Causal Effect):**
```
ACE(L) = P(fail | do(L=executed)) - P(fail | do(L=not_executed))
```

**Null Hypothesis:** Correlation-based SBFL captures the true fault-inducing relationship.

**Rejection Criteria:**
- Causal analysis does not improve ranking on confounded test suites
- Intervention estimation is computationally intractable (>100× overhead)
- Causal model assumptions violated in >20% of cases

**TPS Principle:** Genchi Genbutsu — go to the source (causal root, not correlation)

**Evidence Required:**
- [ ] Causal graph construction from coverage data
- [ ] Intervention estimation (propensity scoring or do-calculus)
- [ ] Confounding detection and reporting
- [ ] Comparison with pure SBFL on confounded cases

*Reference: Baah, G. K., Podgurski, A., & Harrold, M. J. (2010). ISSTA.*

---

### BH-18: Predictive Mutation Testing

**Claim:** ML can predict mutation kill outcomes, reducing mutation testing cost by 70%+ while maintaining 95% accuracy.

**Research Basis:** Zhang et al. (2018) "Predictive Mutation Testing" - IEEE TSE. Achieves 73% cost reduction with 95% accuracy.

**Falsification Test:**
```bash
# Run predictive mutation analysis
batuta bug-hunter falsify --predictive

# Train predictor on project history
batuta bug-hunter falsify --train-predictor --corpus ./test-results/

# Evaluate prediction accuracy
batuta bug-hunter falsify --predictive --validate
```

**Features for Prediction:**
- Code complexity (cyclomatic, cognitive)
- Test coverage density at mutation site
- Historical kill rates for mutation operator
- AST context embeddings (CodeBERT-style)

**Null Hypothesis:** Full mutation testing is required for accurate mutation scores.

**Rejection Criteria:**
- Prediction accuracy <90% on held-out mutants
- Cost reduction <50% compared to full mutation testing
- High-value mutants (boundary, null) have >10% false negative rate

**TPS Principle:** Muda elimination — reduce waste of compute on predictable outcomes

**Evidence Required:**
- [ ] Feature extraction from mutant context
- [ ] Trained predictor model (Random Forest or XGBoost)
- [ ] Accuracy metrics on validation set
- [ ] Prioritized mutant queue based on uncertainty

*Reference: Zhang, J., et al. (2018). IEEE TSE.*

---

### BH-19: Multi-Channel Fault Localization

**Claim:** Combining multiple fault signals (spectrum, mutation, static, semantic) via learning-to-rank outperforms single-channel approaches.

**Research Basis:** Li et al. (2021) "DeepFL: Integrating Multiple Fault Diagnosis Dimensions" - ISSTA. Reports 50% improvement over single-channel.

**Falsification Test:**
```bash
# Run multi-channel analysis
batuta bug-hunter hunt --multi-channel

# Configure channel weights
batuta bug-hunter hunt --multi-channel \
  --spectrum-weight 0.3 \
  --mutation-weight 0.3 \
  --static-weight 0.2 \
  --semantic-weight 0.2

# Train weights on historical bug fixes
batuta bug-hunter train-weights --corpus ./bug-fixes/
```

**Channels:**
1. **Spectrum** — SBFL coverage-based scores
2. **Mutation** — MBFL mutation-based scores
3. **Static** — Clippy/pattern analysis findings
4. **Semantic** — Embedding similarity to error messages

**Algorithm:**
```
FinalScore(L) = Σᵢ wᵢ · channelᵢ(L)
```

**Null Hypothesis:** Single best channel is sufficient; combining adds noise.

**Rejection Criteria:**
- Multi-channel does not outperform best single channel on ≥60% of bugs
- Weight learning overfits to training corpus
- Semantic channel degrades results when error messages unavailable

**TPS Principle:** Heijunka — level the signal by combining multiple sources

**Evidence Required:**
- [ ] Channel score normalization (0-1 scale)
- [ ] Learning-to-rank implementation (LambdaMART or linear)
- [ ] Cross-validation on bug-fix corpus
- [ ] Ablation study showing each channel's contribution

*Reference: Li, X., et al. (2021). ISSTA.*

---

### BH-20: Semantic Crash Bucketing

**Claim:** Grouping crashes by causal root cause reduces duplicate findings by 60%+.

**Research Basis:** Cui et al. (2016) "RETracer: Triaging Crashes by Reverse Execution" - ICSE. Achieves 60-80% reduction in duplicate reports.

**Falsification Test:**
```bash
# Run crash analysis with semantic bucketing
batuta bug-hunter hunt --crash-bucket semantic

# Compare with stack-trace-only bucketing
batuta bug-hunter hunt --crash-bucket stack-trace

# Show bucket summary
batuta bug-hunter hunt --crash-bucket semantic --show-buckets
```

**Algorithm:**
```
1. Parse crash trace to identify failure point
2. Reverse dataflow to find "root cause variable"
3. Classify root cause pattern (e.g., index_oob, null_deref, overflow)
4. Bucket crashes by (pattern, taint_source) tuple
5. Select representative crash per bucket
```

**Null Hypothesis:** Stack trace similarity is sufficient for crash deduplication.

**Rejection Criteria:**
- Semantic bucketing does not reduce duplicates by ≥40%
- Different root causes incorrectly merged into same bucket (>5% error)
- Reverse dataflow analysis exceeds 10s per crash

**TPS Principle:** Muda elimination — remove waste of reviewing duplicate findings

**Evidence Required:**
- [ ] Reverse dataflow implementation for Rust MIR or LLVM IR
- [ ] Root cause pattern taxonomy (10+ patterns)
- [ ] Bucket quality metrics (precision, recall)
- [ ] Representative crash selection algorithm

*Reference: Cui, W., et al. (2016). ICSE.*

---

## Part IV: Evaluation Protocol

### Scoring Methodology

Each item scored as:
- **Pass (1):** Rejection criteria avoided, evidence provided
- **Partial (0.5):** Some evidence, minor issues
- **Fail (0):** Rejection criteria met, claim falsified

**Total Score:** Sum / 128 × 100%

### TPS-Aligned Assessment

| Score | Assessment | TPS Response |
|-------|------------|--------------|
| 95-100% | Toyota Standard | "Good Thinking, Good Products" — Release |
| 85-94% | Kaizen Required | Beta/Preview with documented issues |
| 70-84% | Andon Warning | Significant revision, no release |
| <70% | Stop the Line | Major rework, halt development |

### Three-Layer Review Process

1. **Layer 1 (Jidoka):** Automated tests run on every commit
2. **Layer 2 (Genchi Genbutsu):** Human review with checklist on every PR
3. **Layer 3 (Governance):** Quarterly full audit, external verification

### Claim Revision Protocol (Kaizen)

When claim falsified:
1. **Detect:** Document in issue tracker with severity
2. **Stop:** Block affected releases
3. **Fix:** Either fix implementation OR revise claim with caveats
4. **Prevent:** Root cause analysis, process improvement

---

## Appendix A: Peer-Reviewed References

### Toyota Production System
- [9] Liker, J. (2004). *The Toyota Way*. McGraw-Hill.
- [10] 6Sigma.us. "Jidoka: Toyota Production System."
- [11-14] Six Sigma Online, KPI Fire, Agilitest, Interlake Mecalux on Jidoka.
- [16, 19] TeamHub, Kaizen.com on Kaizen in Software.
- [56] Toyota Global. "Vision and Philosophy."
- [64] Poppendieck, M., & Poppendieck, T. (2003). *Lean Software Development: An Agile Toolkit*. Addison-Wesley.

### Machine Learning Technical Debt & Engineering
- [5] Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*.
- [39] Google Research. "Hidden Technical Debt" (Extended).
- [60] Zhang, J. M., et al. (2020). "Machine learning testing: Survey, landscapes and horizons." *IEEE Transactions on Software Engineering*.
- [63] Lewis, G. A., et al. (2021). "Characterizing and detecting mismatch in machine learning-enabled systems." *ICSE-SEIP*.

### Code Review
- [4] Bacchelli, A., & Bird, C. (2013). "Expectations, outcomes, and challenges of modern code review." *ICSE*.
- [35] Wurzel Gonçalves, et al. (2022). "Checklist-based code review."
- [52] Laitenberger, O. (2002). "State-of-the-art Software Inspections."
- [53] IEEE. (2025). "Michael Fagan's work on software inspection."

### Reproducibility & Determinism
- [7] Heil, B. J., et al. (2021). "Reproducibility standards for machine learning in the life sciences." *Nature Methods*.
- [25] arXiv. (2024). "Defining Reproducibility."
- [26, 27] PMC, JMLR. "Reproducibility checklist."
- [34] arXiv. (2025). "Best Practices for ML Reproducibility."
- [61] Nagarajan, P., et al. (2019). "Determinism in deep learning systems." *NeurIPS*.

### Hypothesis-Driven Development
- [20] ThoughtWorks. "How to implement Hypothesis-Driven Development."
- [21] OpenSource.com. (2020). "Hypothesis-Driven Development."
- [22] Statsig. "How to apply Hypothesis-Driven Development."
- [23] LaunchDarkly. "Hypothesis-Driven Development for Software Engineers."

### Sovereign AI & Data Governance
- [1] Petronella Tech. "Sovereign AI by Design."
- [2] Oracle. "Sovereign AI Brief."
- [3] Katonic.ai. "Building Your AI Stack: Data Sovereignty."
- [37] IBM. "Data Sovereignty vs Data Residency."
- [58] arXiv. (2025). "Systematic Literature Review on Data Sovereignty."
- [62] Hummel, P., et al. (2021). "Data sovereignty: A review." *Big Data & Society*.

### Supply Chain Security
- [29] arXiv. (2025). "Model Transparency and Supply Chain Security."
- [30] arXiv. (2025). "Software Bill of Materials (SBOM) for Java."
- [31] arXiv. (2025). "Supply Chain Risks in LLM Applications."

### Privacy & Federated Learning
- [38] IEEE. (2024). "Privacy-Preserving Machine Learning Survey."
- [48] Google Cloud. "What is Federated Learning."
- [49] STL Partners. "Federated Learning: Decentralised training."
- [50] Duality Tech. "Federated Learning in Meeting Global Data Sovereignty."

### Formal Verification & Safety
- [32] Seshia, S. A., et al. (2018). "Formal Verification of Deep Neural Networks." *ATVA*.
- [33] arXiv. (2018). "Formal Verification of Learning Enabled Components."
- [40] arXiv. (2022). "Adversarial Robustness Verification Survey."
- [41] arXiv. (2020). "Formal Verification of Deep Neural Networks."
- [65] Katz, G., et al. (2017). "Reluplex: An efficient SMT solver for verifying deep neural networks." *CAV*.
- [66] Madry, A., et al. (2018). "Towards deep learning models resistant to adversarial attacks." *ICLR*.

### Model Cards, Auditability & Ethics
- [42] arXiv. (2024). "Audits contribute to the trustworthiness of Learning Analytics."
- [43] Innovative Journal of Applied Science. (2025). "Auditability in Machine Learning Pipelines."
- [45] PMC. (2024). "Model Cards and High Assurance AI."
- [46] Google DeepMind. (2024). "Gemini Model Card."
- [47] Notre Dame Law School. "Datasheets for Datasets and Sovereignty."
- [67] Mitchell, M., et al. (2019). "Model cards for model reporting." *FAT* (now FAccT)*.
- [68] Gebru, T., et al. (2021). "Datasheets for datasets." *CACM*.
- [69] Strubell, E., et al. (2019). "Energy and policy considerations for deep learning in NLP." *ACL*.

### Numerical Computing
- IEEE 754-2019. *IEEE Standard for Floating-Point Arithmetic*.
- Goldberg, D. (1991). "What Every Computer Scientist Should Know About Floating-Point Arithmetic." *ACM Computing Surveys*.
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*. SIAM.
- Kahan, W. (1965). "Further Remarks on Reducing Truncation Errors." *CACM*.
- Gregg, C., & Hazelwood, K. (2011). "Where is the Data?" *ISPASS*.

### Rust Safety
- Jung, R., et al. (2017). "RustBelt: Securing the Foundations of Rust." *POPL 2018*.

### Proactive Bug Hunting & Fault Localization
- [70] Groce, A., et al. (2018). "Falsification-driven verification and testing." *Automated Software Engineering*.
- [71] Rafi, Q., et al. (2025). "SBEST: Spectrum-Based Fault Localization Without Fault-Triggering Tests." *Empirical Software Engineering*.
- [72] Li, H., et al. (2024). "Enhancing Static Analysis for Practical Bug Detection: An LLM-Integrated Approach." *OOPSLA*.
- [73] FourFuzz (2025). "Targeted Fuzzing for Unsafe Rust Code: Leveraging Selective Instrumentation." *arXiv*.
- [74] Böhme, M., et al. (2026). "COTTONTAIL: LLM-Driven Concolic Execution." *IEEE S&P*.
- [75] Hu, J., et al. (2024). "Marco: A Stochastic Asynchronous Concolic Explorer." *ICSE*.
- [76] Kang, S., et al. (2024). "AutoFL: LLM-based fault localization." *ICSE*.
- [77] Facebook (2021). "propfuzz: Property-based testing meets fuzzing." *GitHub*.
- [78] Kamei, Y., et al. (2024). "ML-based defect prediction: A systematic review." *Journal of Systems and Software*.
- [79] Papadakis, M., & Le Traon, Y. (2015). "Metallaxis-FL: Mutation-based Fault Localization." *IEEE TSE*.
- [80] Pearson, S., et al. (2017). "Evaluating and Improving Fault Localization." *ICSE*.
- [81] Baah, G. K., Podgurski, A., & Harrold, M. J. (2010). "Causal Inference for Statistical Fault Localization." *ISSTA*.
- [82] Zhang, J., et al. (2018). "Predictive Mutation Testing." *IEEE TSE*.
- [83] Li, X., et al. (2021). "DeepFL: Integrating Multiple Fault Diagnosis Dimensions for Deep Fault Localization." *ISSTA*.
- [84] Cui, W., et al. (2016). "RETracer: Triaging Crashes by Reverse Execution from Partial Memory Dumps." *ICSE*.

---

## Appendix B: Checklist Summary Table

| Section | Items | Focus | TPS Principle |
|---------|-------|-------|---------------|
| Sovereign Data Governance | 15 | Five Pillars compliance | Poka-Yoke |
| ML Technical Debt Prevention | 10 | Sculley debt categories | Kaizen |
| HDD & EDD | 13 | Scientific Method, Equation Model Cards | Scientific Rigor |
| Numerical Reproducibility | 15 | IEEE 754, reference parity | Genchi Genbutsu |
| Performance & Waste Elimination | 15 | Efficiency, latency | Muda Elimination |
| Safety & Formal Verification | 10 | Memory safety, proofs | Jidoka |
| Jidoka Automated Gates | 10 | CI/CD circuit breakers | Jidoka |
| Model Cards & Auditability | 10 | Governance artifacts | Governance Layer |
| Cross-Platform & API | 5 | Portability, coverage | Completeness |
| **Architectural Invariants** | **5** | **Declarative YAML, Zero Scripting** | **CRITICAL** |
| **Proactive Bug Hunting** | **20** | **Falsification-driven defect discovery** | **Genchi Genbutsu** |
| **Total** | **123** | | |

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0-draft | 2025-12-11 | Team | Initial draft |
| 2.0.0-draft | 2025-12-11 | Team | Integrated Toyota Way framework, Sovereign AI pillars, ML Technical Debt, three-layer architecture, expanded peer-reviewed citations |
| 2.1.0-draft | 2025-12-11 | Team | Added EDD (Equation-Driven Development) for simular, EMC (Equation Model Card), deterministic LLM-assisted development |
| 2.2.0-draft | 2025-12-11 | Team | Added Architectural Invariants (CRITICAL): Declarative YAML, Zero Scripting, Pure Rust Testing, WASM-First |
| 2.3.0-draft | 2026-02-04 | Team | Added Section 11: Proactive Bug Hunting (BH-01 to BH-10) - 10 items for falsification-driven defect discovery integrating mutation testing, SBFL, LLM-augmented analysis, targeted fuzzing, and hybrid concolic execution. Total items now 118. |
| 2.4.0-draft | 2026-02-04 | Team | Added BH-11 to BH-15: Spec-Driven Bug Hunting, PMAT Work Ticket Integration, Scoped Analysis, Bidirectional Spec-Code Linking, False Positive Suppression (Issue #17). Total items now 123. |
| 2.5.0-draft | 2026-02-04 | Team | Added BH-16 to BH-20: Research-based fault localization improvements - MBFL (Papadakis 2015), Causal Inference (Baah 2010), Predictive Mutation Testing (Zhang 2018), Multi-Channel Localization (Li 2021), Semantic Crash Bucketing (Cui 2016). Total items now 128. |

---

**Status:** DRAFT v2.5 - Pending Team Review

**Document Philosophy:**
> "The goal is not to just build a product, but to build a capacity to produce." — Toyota Way

**Next Steps:**
1. Team review of theoretical framework alignment
2. Prioritize checklist items by implementation effort
3. Create automated test infrastructure for Jidoka layer
4. Establish Sovereign Code Reviewer certification program
5. Schedule quarterly full audit cadence
6. Publish results with version 2.0.0
