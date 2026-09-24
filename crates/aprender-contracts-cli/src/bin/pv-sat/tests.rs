//! The reasoner is only trusted through the checker, so the property here is "whatever `solve` says, `check`
//! accepts" — swept over seeded random clause sets (stdlib LCG, no proptest) — plus the controls and the writer's
//! file discipline over a copy of `tests/fixtures/ont/relations-ok`.

use std::collections::BTreeSet;
use std::path::Path;

use provable_contracts::lint::consistency_gate::{run_consistency_gate, ConsistencyOutcome};
use provable_contracts::ontology::witness::{Checked, Core, Step, WitnessResult};

use super::*;

fn s(x: &str) -> String {
    x.to_string()
}

#[test]
fn the_plant_draws_its_core_in_derivation_order() {
    assert_eq!(
        sat::solve(&plant::plant()),
        WitnessResult::UnsatCore(Core {
            steps: vec![
                Step::Unit(s("A")),
                Step::Implies(s("A"), s("B")),
                Step::Implies(s("B"), s("C")),
            ],
            conflict: (s("A"), s("C")),
        })
    );
    assert_eq!(plant::pc_reasoner(), Ok(FIRED));
    assert_eq!(plant::pc_model(), Ok(()));
    assert_eq!(pc_checker(), Ok(FIRED));
}

/// Deterministic LCG (Knuth MMIX constants).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: u64) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from((self.0 >> 33) % n).expect("small")
    }
}

#[test]
fn every_answer_the_reasoner_gives_checks() {
    let names: Vec<String> = (0..8).map(|i| format!("v{i}")).collect();
    let (mut sat_n, mut unsat_n) = (0, 0);
    for seed in 0..500_u64 {
        let mut r = Lcg(seed);
        let mut cs = ClauseSet {
            units: BTreeSet::new(),
            implies: BTreeSet::new(),
            conflicts: BTreeSet::new(),
        };
        for _ in 0..=r.next(3) {
            cs.units.insert(names[r.next(8)].clone());
        }
        for _ in 0..r.next(10) {
            let (a, b) = (r.next(8), r.next(8));
            if a != b {
                cs.implies.insert((names[a].clone(), names[b].clone()));
            }
        }
        for _ in 0..r.next(5) {
            let (a, b) = (r.next(8), r.next(8));
            if a != b {
                cs.conflicts
                    .insert((names[a.min(b)].clone(), names[a.max(b)].clone()));
            }
        }
        let first = sat::solve(&cs);
        assert_eq!(first, sat::solve(&cs), "seed {seed}: not deterministic");
        match check(&cs, &first) {
            Ok(Checked::Sat) => sat_n += 1,
            Ok(Checked::Unsat { .. }) => unsat_n += 1,
            Err(e) => panic!(
                "seed {seed}: the reasoner's own answer does not check: {e}\n{cs:?}\n{first:?}"
            ),
        }
    }
    // The sweep must visit both answers, or it proves half the claim.
    assert!(sat_n > 50 && unsat_n > 50, "sat {sat_n} / unsat {unsat_n}");
}

fn corpus() -> tempfile::TempDir {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ont/relations-ok");
    let dir = tempfile::tempdir().expect("tempdir");
    for entry in std::fs::read_dir(src).expect("fixture") {
        let entry = entry.expect("entry");
        std::fs::copy(entry.path(), dir.path().join(entry.file_name())).expect("copy");
    }
    dir
}

fn exit(code: ExitCode) -> String {
    format!("{code:?}")
}

#[test]
fn the_written_witness_is_the_one_the_gate_checks_and_a_rerun_leaves_it() {
    let dir = corpus();
    let stray = dir
        .path()
        .join("witness")
        .join(format!("{}.json", "a".repeat(64)));
    std::fs::create_dir_all(stray.parent().expect("dir")).expect("mkdir");
    std::fs::write(&stray, "{}").expect("stray");
    let notes = dir.path().join("witness").join("README.json");
    std::fs::write(&notes, "{}").expect("notes");

    assert_eq!(exit(write_witness(dir.path())), exit(ExitCode::SUCCESS));
    assert!(!stray.exists(), "another graph's witness was not pruned");
    assert!(notes.exists(), "a file that is not a witness was pruned");

    // relations-ok is inconsistent (`a contradicts d`, both live): the gate must reach a verdict, and it is a Fail.
    match run_consistency_gate(dir.path()) {
        ConsistencyOutcome::Ran { findings, .. } => {
            let ids: Vec<&str> = findings.iter().map(|f| f.rule_id.as_str()).collect();
            assert_eq!(ids, ["PV-ONT-022"]);
        }
        other => panic!("expected a verdict, got {other:?}"),
    }

    let written: Vec<_> = std::fs::read_dir(dir.path().join("witness"))
        .expect("witness dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.file_name() != notes.file_name())
        .collect();
    assert_eq!(written.len(), 1);
    let before = std::fs::read(&written[0]).expect("witness");
    let mtime = std::fs::metadata(&written[0])
        .and_then(|m| m.modified())
        .expect("mtime");
    assert_eq!(exit(write_witness(dir.path())), exit(ExitCode::SUCCESS));
    assert_eq!(std::fs::read(&written[0]).expect("witness"), before);
    assert_eq!(
        std::fs::metadata(&written[0])
            .and_then(|m| m.modified())
            .expect("mtime"),
        mtime
    );
}

#[test]
fn nothing_to_reason_over_is_exit_2_and_a_broken_sigma_exit_3() {
    let dir = corpus();
    std::fs::remove_file(dir.path().join("ontology.yaml")).expect("rm");
    assert_eq!(exit(write_witness(dir.path())), exit(ExitCode::from(2)));
    assert!(!dir.path().join("witness").exists());

    let dir = corpus();
    std::fs::write(dir.path().join("ontology.yaml"), "roles: [unclosed\n").expect("write");
    assert_eq!(exit(write_witness(dir.path())), exit(ExitCode::from(3)));
}
