use simular::engine::state::Vec3;
use simular::prelude::*;

fn relaxed_jidoka() -> simular::engine::jidoka::JidokaConfig {
    simular::engine::jidoka::JidokaConfig {
        energy_tolerance: 100.0, // Very relaxed for repro tests
        ..Default::default()
    }
}

// H0: Different random seeds produce identical outputs
// Falsification: Run simulation with seeds 42, 43, 44; compare bitwise
#[test]
fn h0_1_different_seeds_produce_different_outputs() {
    let seeds = [42, 43, 44];
    let mut outputs = Vec::new();

    for seed in seeds {
        let config = SimConfig::builder()
            .seed(seed)
            .jidoka(relaxed_jidoka())
            .build();
        let mut engine = SimEngine::new(config).unwrap();

        // Add a body to ensure state changes
        engine
            .state_mut()
            .add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));

        // Use RNG to perturb state
        let perturbation = engine.rng_mut().gen_f64();
        engine.state_mut().add_body(
            1.0,
            Vec3::new(perturbation, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );

        engine.step().unwrap();

        // Serialize state for comparison
        let state_str = serde_json::to_string(engine.state()).unwrap();
        outputs.push(state_str);
    }

    assert_ne!(
        outputs[0], outputs[1],
        "Seed 42 and 43 produced identical output"
    );
    assert_ne!(
        outputs[1], outputs[2],
        "Seed 43 and 44 produced identical output"
    );
    assert_ne!(
        outputs[0], outputs[2],
        "Seed 42 and 44 produced identical output"
    );
}

// H0: Same seed produces different outputs across runs
// Falsification: Run 100 iterations with seed=42; hash all outputs
#[test]
fn h0_2_same_seed_produces_identical_outputs() {
    let seed = 42;
    let mut first_output = String::new();

    for i in 0..100 {
        let config = SimConfig::builder()
            .seed(seed)
            .jidoka(relaxed_jidoka())
            .build();
        let mut engine = SimEngine::new(config).unwrap();

        // Setup state consistently
        engine
            .state_mut()
            .add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));

        // Run for some steps
        engine
            .run_for(simular::engine::SimTime::from_secs(0.1))
            .unwrap();

        let state_str = serde_json::to_string(engine.state()).unwrap();

        if i == 0 {
            first_output = state_str;
        } else {
            assert_eq!(
                state_str, first_output,
                "Run {} produced different output",
                i
            );
        }
    }
}

// H0: Thread count affects results
#[test]
fn h0_4_thread_count_invariance() {
    use std::thread;

    let handles: Vec<_> = (0..8)
        .map(|_| {
            thread::spawn(|| {
                let config = SimConfig::builder()
                    .seed(42)
                    .jidoka(relaxed_jidoka())
                    .build();
                let mut engine = SimEngine::new(config).unwrap();
                engine.state_mut().add_body(
                    1.0,
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                );
                engine
                    .run_for(simular::engine::SimTime::from_secs(0.1))
                    .unwrap();
                serde_json::to_string(engine.state()).unwrap()
            })
        })
        .collect();

    let mut results = Vec::new();
    for handle in handles {
        results.push(handle.join().unwrap());
    }

    for i in 1..results.len() {
        assert_eq!(
            results[0], results[i],
            "Thread {} produced different result",
            i
        );
    }
}

// H0: Checkpoint restore changes trajectory
#[test]
fn h0_6_checkpoint_restore_continuity() {
    let config = SimConfig::builder()
        .seed(42)
        .jidoka(relaxed_jidoka())
        .build();

    // Run 1: Uninterrupted
    let mut engine1 = SimEngine::new(config.clone()).unwrap();
    engine1
        .state_mut()
        .add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
    engine1
        .run_for(simular::engine::SimTime::from_secs(0.1))
        .unwrap();
    let final_state1 = serde_json::to_string(engine1.state()).unwrap();

    // Run 2: Interrupted
    let mut engine2 = SimEngine::new(config).unwrap();
    engine2
        .state_mut()
        .add_body(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));

    // Run halfway
    engine2
        .run_for(simular::engine::SimTime::from_secs(0.05))
        .unwrap();

    // Snapshot state
    let checkpoint = serde_json::to_string(engine2.state()).unwrap();

    // Restore state
    let restored_state: simular::engine::SimState = serde_json::from_str(&checkpoint).unwrap();
    *engine2.state_mut() = restored_state;

    // Continue
    engine2
        .run_for(simular::engine::SimTime::from_secs(0.05))
        .unwrap();
    let final_state2 = serde_json::to_string(engine2.state()).unwrap();

    assert_eq!(
        final_state1, final_state2,
        "Checkpoint restore produced different state"
    );
}

// H0: RNG state serialization loses information
#[test]
fn h0_7_rng_state_serialization() {
    let mut rng1 = simular::engine::SimRng::new(42);
    let _ = rng1.gen_f64();

    // Snapshot
    let rng_snapshot = serde_json::to_string(&rng1).unwrap();

    // Continue rng1
    let val1 = rng1.gen_f64();

    // Restore to rng2
    let mut rng2: simular::engine::SimRng = serde_json::from_str(&rng_snapshot).unwrap();
    let val2 = rng2.gen_f64();

    assert_eq!(val1, val2, "Restored RNG produced different value");

    // Check subsequent values
    assert_eq!(rng1.gen_u64(), rng2.gen_u64());
}

// H0: Time discretization causes drift
#[test]
fn h0_9_time_discretization_convergence() {
    // Run with dt=0.01
    let config1 = SimConfig::builder()
        .timestep(0.01)
        .jidoka(relaxed_jidoka())
        .build();
    let mut engine1 = SimEngine::new(config1).unwrap();
    engine1
        .state_mut()
        .add_body(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    engine1
        .run_for(simular::engine::SimTime::from_secs(1.0))
        .unwrap();
    let pos1 = engine1.state().positions()[0].x;

    // Run with dt=0.001
    let config2 = SimConfig::builder()
        .timestep(0.001)
        .jidoka(relaxed_jidoka())
        .build();
    let mut engine2 = SimEngine::new(config2).unwrap();
    engine2
        .state_mut()
        .add_body(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    engine2
        .run_for(simular::engine::SimTime::from_secs(1.0))
        .unwrap();
    let pos2 = engine2.state().positions()[0].x;

    assert!((pos1 - 1.0).abs() < 1e-6);
    assert!((pos2 - 1.0).abs() < 1e-6);
    assert!((pos1 - pos2).abs() < 1e-9);
}
