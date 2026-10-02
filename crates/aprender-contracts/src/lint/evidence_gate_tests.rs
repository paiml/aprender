//! ONT-8 — the rules of `evidence`, one case per rule, against the real Σ; then the gate over a corpus.

use super::*;

fn sigma_text() -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/ontology.yaml"),
    )
    .expect("the repo's Σ is readable")
}

fn sigma() -> Sigma {
    Sigma::from_yaml(&sigma_text()).expect("the repo's Σ parses")
}

fn rules_with(census: Option<&str>, yaml: &str) -> (Vec<String>, bool) {
    let v: serde_yaml::Value = serde_yaml::from_str(yaml).expect("case is YAML");
    let mut out = Vec::new();
    let pending = check_evidence(
        &sigma(),
        census,
        &v,
        "case",
        Path::new("case.yaml"),
        &mut out,
    );
    (
        out.into_iter().map(|f| f.rule_id).collect(),
        pending.is_some(),
    )
}

fn rules_for(yaml: &str) -> Vec<String> {
    rules_with(None, yaml).0
}

const FULL: &str = "level: L2\nmark: C\nprovenance:\n  wasGeneratedBy: {command: \"pv extract README.md\"}\n  wasAttributedTo: pv\n  generatedAtTime: \"2026-09-24T00:00:00Z\"\n";
const SHA: &str = "2b4eba7250000000000000000000000000000000";

#[test]
fn a_full_block_draws_nothing() {
    assert!(rules_for(FULL).is_empty());
}

#[test]
fn the_minimal_block_is_a_level_and_an_agent() {
    assert!(rules_for("level: L1\nprovenance: {wasAttributedTo: human}\n").is_empty());
}

/// F-16: provenance keys are exactly the PROV-O set — `author:` is RED, at every level of the block.
#[test]
fn f16_author_is_pv_ont_017_wherever_it_is_written() {
    let under_provenance = FULL.replace(
        "  wasAttributedTo: pv\n",
        "  wasAttributedTo: pv\n  author: noah\n",
    );
    assert_eq!(rules_for(&under_provenance), ["PV-ONT-017"]);
    assert_eq!(rules_for(&format!("{FULL}author: noah\n")), ["PV-ONT-017"]);
    let under_generated_by = FULL.replace("{command: ", "{author: noah, command: ");
    assert_eq!(rules_for(&under_generated_by), ["PV-ONT-017"]);
}

#[test]
fn a_block_that_is_not_a_mapping_is_pv_ont_017() {
    assert_eq!(rules_for("L2\n"), ["PV-ONT-017"]);
    assert_eq!(rules_for("level: L2\nprovenance: pv\n"), ["PV-ONT-017"]);
}

#[test]
fn a_level_outside_the_one_enum_is_pv_ont_018() {
    // L0 was the spec's; EV-3 removed it. The gate learns that from `ProofLevel`, not from a list of its own.
    assert_eq!(rules_for(&FULL.replace("L2", "L0")), ["PV-ONT-018"]);
    assert_eq!(rules_for(&FULL.replace("L2", "l2")), ["PV-ONT-018"]);
    assert_eq!(rules_for(&FULL.replace("L2", "L6")), ["PV-ONT-018"]);
    assert_eq!(rules_for(&FULL.replace("level: L2\n", "")), ["PV-ONT-018"]);
}

#[test]
fn every_proof_level_is_admitted() {
    for l in ["L1", "L2", "L3", "L4", "L5"] {
        assert!(rules_for(&FULL.replace("L2", l)).is_empty(), "{l}");
    }
}

#[test]
fn an_unknown_mark_is_pv_ont_019() {
    assert_eq!(
        rules_for(&FULL.replace("mark: C", "mark: X")),
        ["PV-ONT-019"]
    );
    assert_eq!(
        rules_for(&FULL.replace("mark: C", "mark: \"[C]\"")),
        ["PV-ONT-019"]
    );
}

#[test]
fn a_cited_claim_without_a_command_is_pv_ont_019() {
    let no_command = FULL.replace(
        "  wasGeneratedBy: {command: \"pv extract README.md\"}\n",
        "",
    );
    assert_eq!(rules_for(&no_command), ["PV-ONT-019"]);
    let blank = FULL.replace("\"pv extract README.md\"", "\"  \"");
    assert_eq!(rules_for(&blank), ["PV-ONT-019"]);
    // A and U claims make no claim about how they were produced.
    for m in ["A", "U"] {
        assert!(
            rules_for(&no_command.replace("mark: C", &format!("mark: {m}"))).is_empty(),
            "{m}"
        );
    }
}

#[test]
fn a_verified_claim_needs_a_full_sha() {
    let v = FULL.replace("mark: C", "mark: V");
    assert_eq!(rules_for(&v), ["PV-ONT-020"]);
    assert_eq!(
        rules_for(&v.replace("}\n", ", git_sha: 2b4eba725}\n")),
        ["PV-ONT-020"]
    );
    let upper = SHA.to_uppercase().replace('0', "A");
    assert_eq!(
        rules_for(&v.replace("}\n", &format!(", git_sha: \"{upper}\"}}\n"))),
        ["PV-ONT-020"]
    );
}

#[test]
fn a_verified_claim_binds_to_the_census_sha_when_the_census_records_one() {
    let v = FULL
        .replace("mark: C", "mark: V")
        .replace("}\n", &format!(", git_sha: \"{SHA}\"}}\n"));
    assert_eq!(rules_with(Some(SHA), &v), (vec![], false));
    let other = "f".repeat(40);
    assert_eq!(rules_with(Some(&other), &v).0, ["PV-ONT-020"]);
    // A null census (the 2026-09-16 ruling) leaves the sha for the repository to resolve.
    assert_eq!(rules_with(None, &v), (vec![], true));
}

#[test]
fn a_verified_claim_without_a_command_draws_both_rules() {
    let v = format!("level: L3\nmark: V\nprovenance:\n  wasGeneratedBy: {{git_sha: \"{SHA}\"}}\n  wasAttributedTo: pv\n");
    assert_eq!(rules_with(Some(SHA), &v).0, ["PV-ONT-019"]);
}

#[test]
fn an_agent_sigma_does_not_declare_is_pv_ont_021() {
    assert_eq!(
        rules_for(&FULL.replace("wasAttributedTo: pv", "wasAttributedTo: orchestrator")),
        ["PV-ONT-021"]
    );
    assert_eq!(
        rules_for(&FULL.replace("wasAttributedTo: pv", "wasAttributedTo: [pv]")),
        ["PV-ONT-021"]
    );
    assert_eq!(
        rules_for(&FULL.replace("  wasAttributedTo: pv\n", "")),
        ["PV-ONT-021"]
    );
    assert_eq!(rules_for("level: L2\n"), ["PV-ONT-021"]);
    assert_eq!(
        rules_for(&FULL.replace("\"2026-09-24T00:00:00Z\"", "7")),
        ["PV-ONT-021"]
    );
}

#[test]
fn git_that_cannot_look_is_unresolved_not_a_finding() {
    let pending = vec![PendingSha {
        sha: SHA.to_string(),
        stem: "case".into(),
        file: "case.yaml".into(),
    }];
    let mut out = Vec::new();
    let unresolved = resolve_pending(Path::new("/nonexistent/ont-8/corpus"), pending, &mut out);
    assert_eq!((unresolved, out.len()), (1, 0));
}

// ---- the gate over a corpus ----------------------------------------------------------------------------------

fn corpus(files: &[(&str, &str)]) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("ontology.yaml"), sigma_text()).unwrap();
    for (name, body) in files {
        std::fs::write(tmp.path().join(name), body).unwrap();
    }
    tmp
}

fn contract(entity_type: &str, evidence: &str) -> String {
    let indented: String = evidence.lines().map(|l| format!("  {l}\n")).collect();
    format!(
        "metadata:\n  version: \"1.0.0\"\nentity:\n  type: {entity_type}\nevidence:\n{indented}"
    )
}

fn ran(outcome: EvidenceOutcome) -> (GateResult, Vec<LintFinding>) {
    match outcome {
        EvidenceOutcome::Ran { result, findings } => (*result, findings),
        other => panic!("expected Ran, got {other:?}"),
    }
}

fn extra(r: &GateResult) -> (usize, usize, String) {
    match r.extra.as_ref() {
        Some(GateExtra::Evidence {
            entity_types_checked,
            contracts_with_evidence,
            levels_source,
            ..
        }) => (
            *entity_types_checked,
            *contracts_with_evidence,
            levels_source.clone(),
        ),
        other => panic!("expected GateExtra::Evidence, got {other:?}"),
    }
}

/// R-17: a README, a model file and a code contract meet the same rules — the one block, three entity types.
#[test]
fn r17_the_same_gate_passes_readme_model_and_code_contracts() {
    let tmp = corpus(&[
        ("readme-v1.yaml", &contract("readme", FULL)),
        ("model-v1.yaml", &contract("apr-model", FULL)),
        ("code-v1.yaml", &contract("code", FULL)),
    ]);
    let (r, findings) = ran(run_evidence_gate(tmp.path()));
    assert!(findings.is_empty(), "{findings:?}");
    assert_eq!(r.verdict, Verdict::Pass);
    assert_eq!(extra(&r), (3, 3, "enum".to_string()));
}

/// R-17 from the other side: the same defect draws the same finding whatever the entity type.
#[test]
fn r17_the_same_defect_is_rejected_on_every_entity_type() {
    let bad = FULL.replace(
        "  wasAttributedTo: pv\n",
        "  wasAttributedTo: pv\n  author: noah\n",
    );
    for t in ["readme", "apr-model", "code", "gguf", "csv"] {
        let tmp = corpus(&[("c-v1.yaml", &contract(t, &bad))]);
        let (r, findings) = ran(run_evidence_gate(tmp.path()));
        assert_eq!(r.verdict, Verdict::Fail, "{t}");
        let rules: Vec<_> = findings.iter().map(|f| f.rule_id.as_str()).collect();
        assert_eq!(rules, ["PV-ONT-017"], "{t}");
    }
}

#[test]
fn a_contract_without_evidence_is_counted_not_judged() {
    let tmp = corpus(&[
        ("with-v1.yaml", &contract("code", FULL)),
        ("without-v1.yaml", "metadata:\n  version: \"1.0.0\"\n"),
    ]);
    let (r, _) = ran(run_evidence_gate(tmp.path()));
    assert_eq!(r.verdict, Verdict::Pass);
    assert!(matches!(
        r.detail,
        GateDetail::Validate {
            contracts: 2,
            errors: 0,
            ..
        }
    ));
}

/// R-2: zero is a decline, never an accept.
#[test]
fn no_evidence_anywhere_is_a_decline() {
    let tmp = corpus(&[("without-v1.yaml", "metadata:\n  version: \"1.0.0\"\n")]);
    assert!(matches!(
        run_evidence_gate(tmp.path()),
        EvidenceOutcome::NoEvidence {
            contracts_checked: 1
        }
    ));
}

#[test]
fn no_sigma_is_a_decline() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("c-v1.yaml"), contract("code", FULL)).unwrap();
    assert!(matches!(
        run_evidence_gate(tmp.path()),
        EvidenceOutcome::NoSigma
    ));
}

#[test]
fn a_census_sha_binds_verified_claims_in_the_gate() {
    let v = FULL
        .replace("mark: C", "mark: V")
        .replace("}\n", &format!(", git_sha: \"{SHA}\"}}\n"));
    let tmp = corpus(&[("c-v1.yaml", &contract("code", &v))]);
    std::fs::write(
        tmp.path().join("census.json"),
        format!("{{\"git_sha\": \"{SHA}\"}}"),
    )
    .unwrap();
    let (r, _) = ran(run_evidence_gate(tmp.path()));
    assert_eq!(r.verdict, Verdict::Pass);
    std::fs::write(
        tmp.path().join("census.json"),
        format!("{{\"git_sha\": \"{}\"}}", "e".repeat(40)),
    )
    .unwrap();
    let (r, findings) = ran(run_evidence_gate(tmp.path()));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(findings[0].rule_id, "PV-ONT-020");
}

fn git(dir: &Path, args: &[&str]) -> String {
    let o = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            "user.name=ont8",
            "-c",
            "user.email=ont8@example.invalid",
        ])
        .args(args)
        .output()
        .expect("git runs");
    assert!(
        o.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8_lossy(&o.stdout).trim().to_string()
}

/// A null census binds a `[V]` sha to the corpus's own repository: a commit it holds passes, one it does not
/// is rejected. Both arms, so the resolution can fail.
#[test]
fn a_null_census_binds_verified_claims_to_the_repository() {
    let tmp = corpus(&[]);
    git(tmp.path(), &["init", "-q"]);
    git(
        tmp.path(),
        &["commit", "-q", "--allow-empty", "-m", "ont-8 fixture"],
    );
    let head = git(tmp.path(), &["rev-parse", "HEAD"]);
    let v = |sha: &str| {
        FULL.replace("mark: C", "mark: V")
            .replace("}\n", &format!(", git_sha: \"{sha}\"}}\n"))
    };

    std::fs::write(tmp.path().join("c-v1.yaml"), contract("code", &v(&head))).unwrap();
    let (r, findings) = ran(run_evidence_gate(tmp.path()));
    assert_eq!(r.verdict, Verdict::Pass, "{findings:?}");

    std::fs::write(
        tmp.path().join("c-v1.yaml"),
        contract("code", &v(&"a".repeat(40))),
    )
    .unwrap();
    let (r, findings) = ran(run_evidence_gate(tmp.path()));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(findings[0].rule_id, "PV-ONT-020");
}

fn levels_and_unresolved(r: &GateResult) -> (Vec<String>, usize) {
    match r.extra.as_ref() {
        Some(GateExtra::Evidence {
            by_level,
            unresolved_git_shas,
            ..
        }) => (by_level.clone(), *unresolved_git_shas),
        other => panic!("expected GateExtra::Evidence, got {other:?}"),
    }
}

/// Kills `&&` -> `||` on `passed` and `>` -> `<` on the unresolved branch in `run_evidence_gate`: a verified
/// claim that git cannot look up (no census, no repository) is no finding, yet the gate neither passes nor
/// reports Pass — it is Unknown.
#[test]
fn an_unresolved_git_sha_with_no_findings_is_unknown_and_not_passed() {
    let v = FULL
        .replace("mark: C", "mark: V")
        .replace("}\n", &format!(", git_sha: \"{SHA}\"}}\n"));
    let tmp = corpus(&[("c-v1.yaml", &contract("code", &v))]);
    let (r, findings) = ran(run_evidence_gate(tmp.path()));
    assert!(findings.is_empty(), "{findings:?}");
    assert_eq!(levels_and_unresolved(&r).1, 1);
    assert_eq!(r.verdict, Verdict::Unknown(Reason::ToolAbsent));
    assert!(!r.passed);
}

/// Kills `==` -> `!=` and `+=` -> `*=` in `Census::observe`: a level is counted once per block that drew no
/// finding, and a block with a finding is not counted.
#[test]
fn by_level_counts_only_clean_blocks() {
    let bad = FULL.replace(
        "  wasAttributedTo: pv\n",
        "  wasAttributedTo: pv\n  author: noah\n",
    );
    let tmp = corpus(&[
        ("a-v1.yaml", &contract("code", FULL)),
        ("b-v1.yaml", &contract("readme", FULL)),
        ("c-v1.yaml", &contract("code", &bad)),
    ]);
    let (r, findings) = ran(run_evidence_gate(tmp.path()));
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert_eq!(levels_and_unresolved(&r).0, vec!["L2=2".to_string()]);
}

/// Kills `!shallow` -> `true` and `unresolved += 1` -> `*=`/`-=` in `resolve_pending`: in a shallow clone an
/// absent sha is unresolved (git cannot say), never a PV-ONT-020 finding.
#[test]
fn a_shallow_repository_leaves_an_absent_sha_unresolved() {
    let src = corpus(&[]);
    git(src.path(), &["init", "-q"]);
    git(src.path(), &["commit", "-q", "--allow-empty", "-m", "one"]);
    git(src.path(), &["commit", "-q", "--allow-empty", "-m", "two"]);
    let dst = tempfile::tempdir().unwrap();
    let clone = dst.path().join("shallow");
    let o = Command::new("git")
        .args(["clone", "-q", "--depth", "1"])
        .arg(format!("file://{}", src.path().display()))
        .arg(&clone)
        .output()
        .expect("git runs");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert_eq!(
        git(&clone, &["rev-parse", "--is-shallow-repository"]),
        "true"
    );
    let pending = vec![PendingSha {
        sha: "a".repeat(40),
        stem: "case".into(),
        file: "case.yaml".into(),
    }];
    let mut out = Vec::new();
    let unresolved = resolve_pending(&clone, pending, &mut out);
    assert_eq!((unresolved, out.len()), (1, 0));
}

fn by_level_of(r: &GateResult) -> Vec<String> {
    match r.extra.as_ref() {
        Some(GateExtra::Evidence { by_level, .. }) => by_level.clone(),
        other => panic!("expected GateExtra::Evidence, got {other:?}"),
    }
}

/// k9: only a contract that drew NO finding is counted at its level; the count is a running sum.
#[test]
fn by_level_counts_clean_contracts_only_and_sums_them() {
    let bad = FULL.replace(
        "  wasAttributedTo: pv\n",
        "  wasAttributedTo: pv\n  author: noah\n",
    );
    let tmp = corpus(&[
        ("a-v1.yaml", &contract("code", FULL)),
        ("b-v1.yaml", &contract("code", FULL)),
        ("c-v1.yaml", &contract("code", &bad)),
    ]);
    let (r, findings) = ran(run_evidence_gate(tmp.path()));
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert_eq!(by_level_of(&r), vec!["L2=2".to_string()]);
}

/// k9: a sha git cannot look up (shallow clone) with no other finding is Unknown, never Pass.
#[test]
fn an_unresolved_sha_without_findings_is_unknown_not_pass() {
    let src = corpus(&[]);
    git(src.path(), &["init", "-q"]);
    git(src.path(), &["commit", "-q", "--allow-empty", "-m", "one"]);
    git(src.path(), &["commit", "-q", "--allow-empty", "-m", "two"]);
    let dst = tempfile::tempdir().unwrap();
    let clone = dst.path().join("shallow");
    let o = Command::new("git")
        .args(["clone", "-q", "--depth", "1"])
        .arg(format!("file://{}", src.path().display()))
        .arg(&clone)
        .output()
        .expect("git runs");
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    std::fs::write(clone.join("ontology.yaml"), sigma_text()).unwrap();
    let v = FULL
        .replace("mark: C", "mark: V")
        .replace("}\n", &format!(", git_sha: \"{}\"}}\n", "a".repeat(40)));
    std::fs::write(clone.join("c-v1.yaml"), contract("code", &v)).unwrap();
    let (r, findings) = ran(run_evidence_gate(&clone));
    assert!(findings.is_empty(), "{findings:?}");
    assert_eq!(r.verdict, Verdict::Unknown(Reason::ToolAbsent));
    assert!(!r.passed);
}
