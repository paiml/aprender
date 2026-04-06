//! Playbook Operations Benchmarks
//!
//! Benchmarks for YAML parsing, state machine validation, and mutation testing.
//!
//! Run with: `cargo bench --bench playbook_ops`

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jugar_probar::playbook::{
    to_dot, to_svg, ComplexityAnalyzer, MutationClass, MutationGenerator, Playbook,
    StateMachineValidator,
};

const SIMPLE_PLAYBOOK: &str = r#"
version: "1.0"
name: "Simple Flow"
machine:
  id: "simple"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "go"
      from: "start"
      to: "end"
      event: "proceed"
"#;

const MEDIUM_PLAYBOOK: &str = r#"
version: "1.0"
name: "Login Flow"
machine:
  id: "login"
  initial: "logged_out"
  states:
    logged_out:
      id: "logged_out"
      invariants:
        - description: "Login button visible"
          condition: "has_element('#login')"
    authenticating:
      id: "authenticating"
    logged_in:
      id: "logged_in"
      final_state: true
    error:
      id: "error"
  transitions:
    - id: "submit"
      from: "logged_out"
      to: "authenticating"
      event: "submit"
    - id: "success"
      from: "authenticating"
      to: "logged_in"
      event: "auth_ok"
    - id: "failure"
      from: "authenticating"
      to: "error"
      event: "auth_fail"
    - id: "retry"
      from: "error"
      to: "logged_out"
      event: "retry"
    - id: "logout"
      from: "logged_in"
      to: "logged_out"
      event: "logout"
  forbidden:
    - from: "logged_out"
      to: "logged_in"
      reason: "Cannot skip auth"
"#;

fn generate_large_playbook(states: usize, transitions_per_state: usize) -> String {
    let mut yaml = String::from(
        r#"version: "1.0"
name: "Large Flow"
machine:
  id: "large"
  initial: "state_0"
  states:
"#,
    );

    for i in 0..states {
        yaml.push_str(&format!("    state_{}:\n      id: \"state_{}\"\n", i, i));
        if i == states - 1 {
            yaml.push_str("      final_state: true\n");
        }
    }

    yaml.push_str("  transitions:\n");

    let mut trans_id = 0;
    for i in 0..states {
        for j in 0..transitions_per_state {
            let target = (i + j + 1) % states;
            yaml.push_str(&format!(
                "    - id: \"trans_{}\"\n      from: \"state_{}\"\n      to: \"state_{}\"\n      event: \"event_{}\"\n",
                trans_id, i, target, trans_id
            ));
            trans_id += 1;
        }
    }

    yaml
}

fn bench_yaml_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("yaml_parsing");

    let playbooks = vec![
        ("simple_2_states", SIMPLE_PLAYBOOK.to_string()),
        ("medium_4_states", MEDIUM_PLAYBOOK.to_string()),
        ("large_10_states", generate_large_playbook(10, 2)),
        ("large_50_states", generate_large_playbook(50, 2)),
    ];

    for (name, yaml) in playbooks {
        group.bench_with_input(BenchmarkId::from_parameter(name), &yaml, |bench, yaml| {
            bench.iter(|| {
                let playbook = Playbook::from_yaml(black_box(yaml)).unwrap();
                black_box(playbook);
            });
        });
    }

    group.finish();
}

fn bench_state_machine_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_machine_validation");

    let playbooks = vec![
        ("simple", Playbook::from_yaml(SIMPLE_PLAYBOOK).unwrap()),
        ("medium", Playbook::from_yaml(MEDIUM_PLAYBOOK).unwrap()),
        (
            "large_10",
            Playbook::from_yaml(&generate_large_playbook(10, 2)).unwrap(),
        ),
        (
            "large_50",
            Playbook::from_yaml(&generate_large_playbook(50, 2)).unwrap(),
        ),
    ];

    for (name, playbook) in playbooks {
        group.bench_with_input(BenchmarkId::from_parameter(name), &playbook, |bench, pb| {
            bench.iter(|| {
                let validator = StateMachineValidator::new(black_box(pb));
                let result = validator.validate();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_dot_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_generation");

    let playbooks = vec![
        ("simple", Playbook::from_yaml(SIMPLE_PLAYBOOK).unwrap()),
        ("medium", Playbook::from_yaml(MEDIUM_PLAYBOOK).unwrap()),
        (
            "large_10",
            Playbook::from_yaml(&generate_large_playbook(10, 2)).unwrap(),
        ),
        (
            "large_50",
            Playbook::from_yaml(&generate_large_playbook(50, 2)).unwrap(),
        ),
    ];

    for (name, playbook) in playbooks {
        group.bench_with_input(BenchmarkId::from_parameter(name), &playbook, |bench, pb| {
            bench.iter(|| {
                let dot = to_dot(black_box(pb));
                black_box(dot);
            });
        });
    }

    group.finish();
}

fn bench_svg_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("svg_generation");

    let playbooks = vec![
        ("simple", Playbook::from_yaml(SIMPLE_PLAYBOOK).unwrap()),
        ("medium", Playbook::from_yaml(MEDIUM_PLAYBOOK).unwrap()),
        (
            "large_10",
            Playbook::from_yaml(&generate_large_playbook(10, 2)).unwrap(),
        ),
    ];

    for (name, playbook) in playbooks {
        group.bench_with_input(BenchmarkId::from_parameter(name), &playbook, |bench, pb| {
            bench.iter(|| {
                let svg = to_svg(black_box(pb));
                black_box(svg);
            });
        });
    }

    group.finish();
}

fn bench_mutation_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutation_generation");

    let playbooks = vec![
        ("simple", Playbook::from_yaml(SIMPLE_PLAYBOOK).unwrap()),
        ("medium", Playbook::from_yaml(MEDIUM_PLAYBOOK).unwrap()),
        (
            "large_10",
            Playbook::from_yaml(&generate_large_playbook(10, 2)).unwrap(),
        ),
    ];

    for (name, playbook) in playbooks {
        group.bench_with_input(BenchmarkId::from_parameter(name), &playbook, |bench, pb| {
            bench.iter(|| {
                let generator = MutationGenerator::new(black_box(pb));
                let mutants = generator.generate_all();
                black_box(mutants);
            });
        });
    }

    group.finish();
}

fn bench_mutation_by_class(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutation_by_class");

    let playbook = Playbook::from_yaml(MEDIUM_PLAYBOOK).unwrap();

    let classes = vec![
        MutationClass::StateRemoval,
        MutationClass::TransitionRemoval,
        MutationClass::EventSwap,
        MutationClass::TargetSwap,
        MutationClass::GuardNegation,
    ];

    for class in classes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", class)),
            &(&playbook, class),
            |bench, (pb, cls)| {
                bench.iter(|| {
                    let generator = MutationGenerator::new(pb);
                    let mutants = generator.generate(black_box(*cls));
                    black_box(mutants);
                });
            },
        );
    }

    group.finish();
}

fn bench_complexity_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("complexity_analysis");

    let data_sizes = vec![10, 50, 100, 500];

    for size in data_sizes {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_points", size)),
            &size,
            |bench, &n| {
                let data: Vec<(usize, f64)> = (0..n).map(|i| (i, (i as f64) * 2.0)).collect();
                bench.iter(|| {
                    let analyzer = ComplexityAnalyzer::new(black_box(data.clone()));
                    let result = analyzer.analyze(None);
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_yaml_parsing,
    bench_state_machine_validation,
    bench_dot_generation,
    bench_svg_generation,
    bench_mutation_generation,
    bench_mutation_by_class,
    bench_complexity_analysis
);
criterion_main!(benches);
