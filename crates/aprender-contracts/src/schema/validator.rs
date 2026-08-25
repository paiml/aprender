use std::collections::HashSet;

use crate::error::{Severity, Violation};
use crate::schema::types::{Contract, ContractKind, CONTRACT_TOP_LEVEL_FIELDS};

/// Validate a parsed contract for completeness and consistency.
///
/// Returns a list of violations. If any violation has
/// [`Severity::Error`], the contract is considered invalid.
///
/// Validation is kind-aware: non-kernel contracts (registries, model-family
/// schemas, reference documents) are validated only for metadata consistency;
/// the provability invariant, equations, and proof/kani/falsification checks
/// only apply to `ContractKind::Kernel`.
pub fn validate_contract(contract: &Contract) -> Vec<Violation> {
    let mut violations = Vec::new();

    validate_metadata(contract, &mut violations);
    // Runs BEFORE the kind split below on purpose: a top-level `kind:` is
    // exactly the key that would otherwise decide which branch runs, and the
    // whole point of SCHEMA-018 is that it silently decides nothing.
    validate_top_level_keys(contract, &mut violations);

    // Kernel-only checks: these enforce the provability invariant and
    // require equations + proof obligations + tests + Kani harnesses.
    if contract.kind() == ContractKind::Kernel && !contract.is_registry() {
        validate_equations(contract, &mut violations);
        validate_provability_invariant(contract, &mut violations);
        validate_proof_obligations(contract, &mut violations);
        validate_falsification_tests(contract, &mut violations);
        validate_kani_harnesses(contract, &mut violations);
        validate_qa_gate(contract, &mut violations);
    } else {
        // Non-kernel kinds (registry, model-family, schema): still validate
        // any proof obligations/falsification/kani data that IS present, so
        // mistakes are caught even on exempt contracts.
        validate_proof_obligations(contract, &mut violations);
        validate_falsification_tests(contract, &mut violations);
        validate_kani_harnesses(contract, &mut violations);
    }

    // BeatBenchmark-only checks (PMAT-741): the `beat:` block must pin a
    // falsifiable, four-pillar incumbent baseline. Independent of the
    // kernel/non-kernel split above.
    if contract.kind() == ContractKind::BeatBenchmark {
        validate_beat_benchmark(contract, &mut violations);
    }

    // CRUX competitive-research metadata (aprender#2555): kind-independent —
    // the three fields are carried by 275 `crux-*` contracts of several kinds
    // and by non-crux contracts that reuse the vocabulary.
    validate_crux_intake(contract, &mut violations);

    violations
}

/// The closed set of competitors a CRUX story may be extracted from
/// (`metadata.competitor`, rule CRUX-002).
///
/// # Why this is NOT `BEAT_INCUMBENTS`
///
/// Reusing [`BEAT_INCUMBENTS`] was considered and REJECTED — it names a
/// different domain and would import a defect. `BEAT_INCUMBENTS` answers "whom
/// does aprender claim to *beat* on a pinned benchmark" (the four-pillar
/// mission); `metadata.competitor` answers "whose UX was this story *extracted
/// from*". MEASURED on this branch: 275 contract FILES carry the field, in 292
/// declarations (17 crux contracts carry a second `competitor` inside an
/// equivalence obligation). `BEAT_INCUMBENTS.iter().any(|p| c.contains(p))`
/// accepts only `pytorch` (37) and `ollama` (21) — 58 of 292. It cannot name:
///
/// - `huggingface` (88 contracts — the single largest source), nor `vllm` (32),
/// - `llama_cpp` (37): the BEAT list spells it `llama.cpp`, and `"llama.cpp"`
///   is not a substring of `"llama_cpp"`, so even the pillar it does name is
///   missed under the underscore spelling the crux corpus uses,
/// - `ecosystem` (30), `openclaw` (20), `hf-kernels-community` (15),
///   `apr-qa-playbook` (9), `openclip` (2), `none` (1).
///
/// So this registry is the corpus vocabulary, exactly. Every member is
/// exercised by at least one contract in `contracts/`; adding a competitor is a
/// deliberate one-line edit here plus a test, which is the point — an open
/// domain is what let `THIS-COMPETITOR-DOES-NOT-EXIST` validate.
pub(crate) const CRUX_COMPETITORS: [&str; 12] = [
    "apr-qa-playbook",
    "ecosystem",
    "hf-kernels-community",
    "huggingface",
    "llama_cpp",
    "none",
    "ollama",
    "openclaw",
    "openclip",
    // Orange Sun Pulp Free Chat — a local-first desktop chat app (CRUX-C-37).
    // Admitted because it competes on the SAME axis this project sells on
    // (private, on-device, no subscription) while publishing no throughput
    // number at all, which is itself a competitive datapoint.
    "pulp-free-chat",
    "pytorch",
    "vllm",
];

/// Documented inclusive bounds of `metadata.demand_score`, from
/// `contracts/crux-competitive-research-ux-v1.yaml`: "a demand_score (1..5) …
/// demand_score maps directly to pmat priority".
const DEMAND_SCORE_RANGE: std::ops::RangeInclusive<i64> = 1..=5;

/// Validate the CRUX competitive-research domains (aprender#2555).
///
/// Two SURFACES carry these fields, and both are checked here:
///
/// 1. `metadata.{competitor,demand_score,intake_status}` on an individual
///    `crux-*` contract.
/// 2. The `stories:` rows of the MASTER REGISTRY,
///    `contracts/crux-competitive-research-ux-v1.yaml`.
///
/// Surface 2 was added because the original rationale for this rule did not
/// survive measurement. #2555 justified CRUX-001 as guarding "the ranking
/// signal the whole competitive-research programme sorts by" — but MEASURED,
/// nothing in the repo reads `metadata.demand_score`. The signal §12.1 of
/// `docs/specifications/crux-competitive-research-ux-workflows.md` maps to
/// `pmat work` priority is `stories[].demand_score` in the registry: 250 rows,
/// entirely ungated. Checking only surface 1 left the stated justification
/// unsupported by the code.
///
/// On a registry row the three fields are also REQUIRED, not optional. On
/// surface 1 they cannot be: `Option` is right there, because 1500-odd non-crux
/// contracts carry none of them (see the presence obligation in
/// `contracts/crux-intake-metadata-domains-v1.yaml`). A registry row has no
/// such excuse — it exists to be ranked.
///
/// `intake_status` / `status` values are absent from the checks below ON
/// PURPOSE: both are the closed enum `IntakeStatus`, so an invented value is
/// rejected during deserialization and never reaches a validator. That is the
/// stronger guarantee — a lint can be read and ignored, a parse failure cannot.
fn validate_crux_intake(contract: &Contract, violations: &mut Vec<Violation>) {
    // CRUX-001: demand_score is the ranking signal the whole competitive-research
    // programme sorts by. An unvalidated out-of-range value silently dominates
    // every ranking it appears in.
    if let Some(score) = contract.metadata.demand_score {
        if !DEMAND_SCORE_RANGE.contains(&score) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "CRUX-001".to_string(),
                message: format!(
                    "metadata.demand_score {score} is outside the documented range {}..={} \
                     — it is the priority signal pmat work sorts by, so an out-of-range \
                     value silently outranks every real story",
                    DEMAND_SCORE_RANGE.start(),
                    DEMAND_SCORE_RANGE.end(),
                ),
                location: Some("metadata.demand_score".to_string()),
            });
        }
    }

    // CRUX-002: competitor must name a source in the registry above.
    //
    // No `.trim()` here, deliberately. It used to trim before comparing, which
    // made `competitor: "  ecosystem  "` validate clean while the STORED value
    // kept its padding — the check laundered a value it did not fix, so every
    // consumer reading `metadata.competitor` still saw the untrimmed string.
    // Normalisation now happens once, at parse time
    // (`deserialize_trimmed_opt_string` in `schema/types.rs`), so what is
    // compared is exactly what is stored.
    if let Some(competitor) = contract.metadata.competitor.as_deref() {
        if !CRUX_COMPETITORS.contains(&competitor) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "CRUX-002".to_string(),
                message: format!(
                    "metadata.competitor {competitor:?} is not a known competitive-research \
                     source — must be one of: {}",
                    CRUX_COMPETITORS.join(", ")
                ),
                location: Some("metadata.competitor".to_string()),
            });
        }
    }

    validate_crux_registry_stories(contract, violations);
}

/// Hold every MASTER-REGISTRY story row to the same two domains.
///
/// These are the rows that carry the ranking signal, so here the fields are
/// required as well as bounded: a row with no `demand_score` cannot be sorted,
/// and a row with no `competitor` cannot be attributed.
fn validate_crux_registry_stories(contract: &Contract, violations: &mut Vec<Violation>) {
    for story in &contract.stories {
        let at = |field: &str| Some(format!("stories[{}].{field}", story.id));

        match story.demand_score {
            None => violations.push(Violation {
                severity: Severity::Error,
                rule: "CRUX-001".to_string(),
                message: format!(
                    "registry story {} has no demand_score — it is the priority signal \
                     pmat work sorts by, and an absent one sorts arbitrarily",
                    story.id
                ),
                location: at("demand_score"),
            }),
            Some(score) if !DEMAND_SCORE_RANGE.contains(&score) => violations.push(Violation {
                severity: Severity::Error,
                rule: "CRUX-001".to_string(),
                message: format!(
                    "registry story {} has demand_score {score}, outside the documented \
                     range {}..={} — a single fabricated score reorders the whole queue",
                    story.id,
                    DEMAND_SCORE_RANGE.start(),
                    DEMAND_SCORE_RANGE.end(),
                ),
                location: at("demand_score"),
            }),
            Some(_) => {}
        }

        match story.competitor.as_deref() {
            None => violations.push(Violation {
                severity: Severity::Error,
                rule: "CRUX-002".to_string(),
                message: format!(
                    "registry story {} has no competitor — the row cannot be attributed \
                     to the UX it was extracted from",
                    story.id
                ),
                location: at("competitor"),
            }),
            Some(c) if !CRUX_COMPETITORS.contains(&c) => violations.push(Violation {
                severity: Severity::Error,
                rule: "CRUX-002".to_string(),
                message: format!(
                    "registry story {} names competitor {c:?}, which is not a known \
                     competitive-research source — must be one of: {}",
                    story.id,
                    CRUX_COMPETITORS.join(", ")
                ),
                location: at("competitor"),
            }),
            Some(_) => {}
        }
    }
}

/// The four incumbents a BEAT may target (case-insensitive substring match, so
/// `ollama` and `llama.cpp` both satisfy Pillar 4).
const BEAT_INCUMBENTS: [&str; 5] = ["scikit-learn", "pytorch", "unsloth", "ollama", "llama.cpp"];

/// Enforce the BeatBenchmark shape (PMAT-741): a `beat-benchmark` contract MUST
/// carry a well-formed `beat:` block so the claim is a falsifiable CI gate, not
/// prose. Rules BEAT-001..007.
fn validate_beat_benchmark(contract: &Contract, violations: &mut Vec<Violation>) {
    let push = |violations: &mut Vec<Violation>, rule: &str, message: String, field: &str| {
        violations.push(Violation {
            severity: Severity::Error,
            rule: rule.to_string(),
            message,
            location: Some(format!("beat.{field}")),
        });
    };

    let Some(beat) = contract.beat.as_ref() else {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "BEAT-001".to_string(),
            message: "beat-benchmark contract must define a `beat:` block \
                      (incumbent, metric, direction, beat_threshold, ci_gate_name)"
                .to_string(),
            location: Some("beat".to_string()),
        });
        return;
    };

    // BEAT-002: incumbent must name one of the four pillars.
    let incumbent = beat.incumbent.trim().to_lowercase();
    if incumbent.is_empty() {
        push(
            violations,
            "BEAT-002",
            "beat.incumbent must not be empty".to_string(),
            "incumbent",
        );
    } else if !BEAT_INCUMBENTS.iter().any(|p| incumbent.contains(p)) {
        push(
            violations,
            "BEAT-002",
            format!(
                "beat.incumbent {:?} must name one of the four pillars ({})",
                beat.incumbent,
                BEAT_INCUMBENTS.join(", ")
            ),
            "incumbent",
        );
    }

    // BEAT-003: a measured metric is required.
    if beat.metric.trim().is_empty() {
        push(
            violations,
            "BEAT-003",
            "beat.metric must name the measured quantity (e.g. accuracy, wall_clock_ms, \
             tokens_per_sec)"
                .to_string(),
            "metric",
        );
    }

    // BEAT-004: direction fixes which way is a regression.
    match beat.direction.trim() {
        "higher_is_better" | "lower_is_better" => {}
        other => push(
            violations,
            "BEAT-004",
            format!(
                "beat.direction must be `higher_is_better` or `lower_is_better`, got {other:?}"
            ),
            "direction",
        ),
    }

    // BEAT-005: a finite, machine-pinned threshold is required (the gate value).
    match beat.beat_threshold {
        None => push(
            violations,
            "BEAT-005",
            "beat.beat_threshold is required — the pinned value CI fails below".to_string(),
            "beat_threshold",
        ),
        Some(t) if !t.is_finite() => push(
            violations,
            "BEAT-005",
            format!("beat.beat_threshold must be finite, got {t}"),
            "beat_threshold",
        ),
        Some(_) => {}
    }

    // BEAT-006: the enforcing CI gate must be named.
    if beat.ci_gate_name.trim().is_empty() {
        push(
            violations,
            "BEAT-006",
            "beat.ci_gate_name must name the CI test that enforces this gate".to_string(),
            "ci_gate_name",
        );
    }

    // BEAT-007: approved_compute is required and must be CPU or GPU (the
    // autonomous-vs-operator track distinction depends on it).
    match beat
        .approved_compute
        .as_deref()
        .map(|c| c.trim().to_uppercase())
    {
        None => push(
            violations,
            "BEAT-007",
            "beat.approved_compute is required — must be `CPU` or `GPU`".to_string(),
            "approved_compute",
        ),
        Some(ref c) if c != "CPU" && c != "GPU" => push(
            violations,
            "BEAT-007",
            format!(
                "beat.approved_compute must be `CPU` or `GPU`, got {:?}",
                beat.approved_compute
            ),
            "approved_compute",
        ),
        Some(_) => {}
    }
}

/// Enforce the provability invariant: kernel contracts (non-registry) MUST have
/// `proof_obligations`, `falsification_tests`, and `kani_harnesses`.
fn validate_provability_invariant(contract: &Contract, violations: &mut Vec<Violation>) {
    for v in contract.provability_violations() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "PROVABILITY-001".to_string(),
            message: v,
            location: None,
        });
    }
}

/// The forms a YAML key could be a plural/case/separator variant of.
///
/// Case-folded with separators dropped, then the key itself plus its `-s` and
/// `-es` singularizations. Comparing SETS rather than normalizing to one
/// canonical string is what makes `qa_gates` ~ `qa_gate` and
/// `kani_harness` ~ `kani_harnesses` both work: a single-pass normalizer has to
/// choose between stripping `es` (right for `harnesses`, wrong for `gates`) and
/// stripping `s` (vice versa), and gets one of the two wrong whichever it picks.
fn key_forms(key: &str) -> Vec<String> {
    let squashed: String = key
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let mut forms = vec![squashed.clone()];
    for suffix in ["es", "s"] {
        if let Some(stem) = squashed.strip_suffix(suffix) {
            if !stem.is_empty() {
                forms.push(stem.to_string());
            }
        }
    }
    forms
}

/// The real block name an unknown top-level key is a near-miss of, if any.
///
/// Exact field names never reach here (the parser filters them out), so a hit
/// is always a misspelling, a case/separator variant, or a singular/plural slip
/// — never a legitimate downstream-owned block. The near-collisions this must
/// NOT fire on are pinned by `legitimate_downstream_keys_are_not_flagged`:
/// `invariants` is not `type_invariants`, `gates` is not `qa_gate`, `spec` is
/// not `coq_spec`.
fn near_miss_of(key: &str) -> Option<&'static str> {
    let forms = key_forms(key);
    CONTRACT_TOP_LEVEL_FIELDS
        .iter()
        .copied()
        .find(|field| key_forms(field).iter().any(|f| forms.contains(f)))
}

/// SCHEMA-018 / SCHEMA-019: reject the two top-level shapes that are never
/// legitimate.
///
/// `Contract` tolerates unknown top-level keys by design — see
/// [`crate::schema::parse_contract_str`]. This check does not change that; it
/// carves out the two cases where serde's silence is a defect:
///
/// * **SCHEMA-018** — a top-level `kind:`. 119 contracts carried one. It is
///   dropped, so the contract silently falls back to `metadata.kind` (or to the
///   `kernel` default), and in 72 of those files the top-level value said
///   `KernelContract` while `metadata.registry: true` made the contract an
///   exempt registry. The key does not just fail to help, it lies.
/// * **SCHEMA-019** — a near-miss of a real block name. This is how
///   `contracts/publish-workspace-v1.yaml` lost four FALSIFY-PUB-* entries:
///   they sat under a key serde did not recognise, `pv status` printed
///   "Falsification tests: 0", and nothing anywhere said why.
fn validate_top_level_keys(contract: &Contract, violations: &mut Vec<Violation>) {
    // SCHEMA-020: the document is not valid YAML to a strict reader even though
    // the derived deserializer accepted it — today that means a duplicate
    // mapping key, one of whose values is being thrown away silently.
    if let Some(err) = contract.strict_yaml_error.as_ref() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-020".to_string(),
            message: format!(
                "the contract schema accepted this document but a strict YAML reader \
                 rejects it ({err}) — `yq`, PyYAML and any `serde_yaml::Value` consumer \
                 will drop content here. A duplicate mapping key is the usual cause: \
                 merge the two blocks into one"
            ),
            location: None,
        });
    }

    for key in &contract.unknown_top_level_keys {
        if key == "kind" {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-018".to_string(),
                message: "top-level `kind:` is not part of the contract schema and is \
                          silently dropped — the contract's kind comes from \
                          `metadata.kind:` (or defaults to `kernel`). Move it under \
                          `metadata:` if it names a real kind, or delete it"
                    .to_string(),
                location: Some("kind".to_string()),
            });
        } else if let Some(field) = near_miss_of(key) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-019".to_string(),
                message: format!(
                    "top-level `{key}:` is not a contract field and is silently dropped \
                     — did you mean `{field}:`? Everything under `{key}:` is invisible \
                     to every pv gate"
                ),
                location: Some(key.clone()),
            });
        }
    }
}

fn validate_metadata(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.metadata.references.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-001".to_string(),
            message: "metadata.references must not be empty — \
                      every contract must cite its source paper(s)"
                .to_string(),
            location: Some("metadata.references".to_string()),
        });
    }

    if contract.metadata.version.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-002".to_string(),
            message: "metadata.version must not be empty".to_string(),
            location: Some("metadata.version".to_string()),
        });
    }
}

fn validate_equations(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.equations.is_empty() {
        violations.push(Violation {
            severity: Severity::Error,
            rule: "SCHEMA-003".to_string(),
            message: "equations must contain at least one equation".to_string(),
            location: Some("equations".to_string()),
        });
    }

    for (name, eq) in &contract.equations {
        if eq.formula.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-004".to_string(),
                message: format!("equations.{name}.formula must not be empty"),
                location: Some(format!("equations.{name}.formula")),
            });
        }
    }
}

fn validate_proof_obligations(contract: &Contract, violations: &mut Vec<Violation>) {
    use crate::schema::types::ObligationType;

    let mut seen_ids = HashSet::new();
    for (i, ob) in contract.proof_obligations.iter().enumerate() {
        if ob.property.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-005".to_string(),
                message: format!("proof_obligations[{i}].property must not be empty"),
                location: Some(format!("proof_obligations[{i}].property")),
            });
        }
        if let Some(ref formal) = ob.formal {
            if !seen_ids.insert(formal.clone()) {
                violations.push(Violation {
                    severity: Severity::Warning,
                    rule: "SCHEMA-006".to_string(),
                    message: format!("Duplicate formal predicate: {formal}"),
                    location: Some(format!("proof_obligations[{i}].formal")),
                });
            }
        }

        // DbC field/type constraints
        if ob.requires.is_some() && ob.obligation_type != ObligationType::Postcondition {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-014".to_string(),
                message: format!(
                    "proof_obligations[{i}].requires is only valid on \
                     postcondition obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].requires")),
            });
        }

        if ob.applies_to_phase.is_some()
            && ob.obligation_type != ObligationType::LoopInvariant
            && ob.obligation_type != ObligationType::LoopVariant
        {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-015".to_string(),
                message: format!(
                    "proof_obligations[{i}].applies_to_phase is only valid on \
                     loop_invariant or loop_variant obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].applies_to_phase")),
            });
        }

        if ob.parent_contract.is_some() && ob.obligation_type != ObligationType::Subcontract {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-016".to_string(),
                message: format!(
                    "proof_obligations[{i}].parent_contract is only valid on \
                     subcontract obligations (found on {})",
                    ob.obligation_type
                ),
                location: Some(format!("proof_obligations[{i}].parent_contract")),
            });
        }

        // Subcontract parent_contract must be in depends_on
        if let Some(ref parent) = ob.parent_contract {
            if ob.obligation_type == ObligationType::Subcontract
                && !contract.metadata.depends_on.contains(parent)
            {
                violations.push(Violation {
                    severity: Severity::Error,
                    rule: "SCHEMA-017".to_string(),
                    message: format!(
                        "proof_obligations[{i}].parent_contract \"{parent}\" \
                         must be listed in metadata.depends_on"
                    ),
                    location: Some(format!("proof_obligations[{i}].parent_contract")),
                });
            }
        }
    }
}

fn validate_falsification_tests(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut ids = HashSet::new();
    for test in &contract.falsification_tests {
        if !ids.insert(&test.id) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-007".to_string(),
                message: format!("Duplicate falsification test ID: {}", test.id),
                location: Some(format!("falsification_tests.{}", test.id)),
            });
        }
        if test.prediction.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-008".to_string(),
                message: format!(
                    "falsification_tests.{}.prediction must not be empty — \
                     every test must make a falsifiable prediction",
                    test.id
                ),
                location: Some(format!("falsification_tests.{}.prediction", test.id)),
            });
        }
        if test.if_fails.is_empty() {
            violations.push(Violation {
                severity: Severity::Warning,
                rule: "SCHEMA-009".to_string(),
                message: format!(
                    "falsification_tests.{}.if_fails is empty — \
                     should describe root cause diagnosis",
                    test.id
                ),
                location: Some(format!("falsification_tests.{}.if_fails", test.id)),
            });
        }
    }
}

fn validate_kani_harnesses(contract: &Contract, violations: &mut Vec<Violation>) {
    let mut ids = HashSet::new();
    for harness in &contract.kani_harnesses {
        if !ids.insert(&harness.id) {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-010".to_string(),
                message: format!("Duplicate Kani harness ID: {}", harness.id),
                location: Some(format!("kani_harnesses.{}", harness.id)),
            });
        }
        if harness.obligation.is_empty() {
            violations.push(Violation {
                severity: Severity::Error,
                rule: "SCHEMA-011".to_string(),
                message: format!(
                    "kani_harnesses.{}.obligation must not be empty — \
                     every harness must reference a proof obligation",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.obligation", harness.id)),
            });
        }
        if harness.bound.is_none() {
            violations.push(Violation {
                severity: Severity::Warning,
                rule: "SCHEMA-012".to_string(),
                message: format!(
                    "kani_harnesses.{}.bound not specified — \
                     Kani requires an unwind bound",
                    harness.id
                ),
                location: Some(format!("kani_harnesses.{}.bound", harness.id)),
            });
        }
    }
}

fn validate_qa_gate(contract: &Contract, violations: &mut Vec<Violation>) {
    if contract.qa_gate.is_none() {
        violations.push(Violation {
            severity: Severity::Warning,
            rule: "SCHEMA-013".to_string(),
            message: "No qa_gate defined — contract should define a \
                      certeza quality gate"
                .to_string(),
            location: Some("qa_gate".to_string()),
        });
    }
}

#[cfg(test)]
mod tests {
    include!("validator_tests.rs");
}
