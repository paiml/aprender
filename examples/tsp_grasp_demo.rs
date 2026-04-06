//! TSP GRASP Demo Example
//!
//! Demonstrates the GRASP (Greedy Randomized Adaptive Search Procedure)
//! methodology for solving the Traveling Salesman Problem.
//!
//! This example showcases the EDD (Equation-Driven Development) 5-phase cycle:
//! 1. Equation - Tour length, 2-opt improvement, BHH constant
//! 2. Failing Test - Random construction yields poor tours
//! 3. Implementation - GRASP with randomized greedy + 2-opt
//! 4. Verification - Achieves good optimality gap
//! 5. Falsification - Compares greedy vs random construction
//!
//! Run with: cargo run --example tsp_grasp_demo

use simular::demos::tsp_grasp::{ConstructionMethod, TspGraspDemo};
use simular::demos::EddDemo;

fn main() {
    print_header();
    run_phase_1_equations();
    let (random_length, greedy_length) = run_phase_2_failing_test(25, 42);
    print_failing_test_results(random_length, greedy_length);
    let demo = run_phase_3_implementation(42, 25, 20);
    run_phase_4_verification(&demo);
    run_phase_5_falsification();
    run_visualization(&demo);
    run_edd_trait_info(&demo);
    print_footer();
}

fn print_header() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           TSP GRASP Demo - EDD Showcase Demo 6              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Greedy Randomized Adaptive Search Procedure for TSP        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

fn run_phase_1_equations() {
    println!("═══ Phase 1: Governing Equations ═══");
    println!();
    println!("Tour Length:           L(π) = Σᵢ d(π(i), π(i+1)) + d(π(n), π(1))");
    println!("2-Opt Improvement:     Δ = d(i,i+1) + d(j,j+1) - d(i,j) - d(i+1,j+1)");
    println!("Expected Greedy Tour:  E[L] ≈ 0.7124·√(n·A)  [Beardwood-Halton-Hammersley]");
    println!("Lower Bound:           L* ≥ 1-tree(G)  [Held-Karp style bound]");
    println!();
}

fn run_phase_2_failing_test(n: usize, seed: u64) -> (f64, f64) {
    println!("═══ Phase 2: Failing Test - Random vs Greedy Construction ═══");
    println!();

    let mut demo_random = TspGraspDemo::new(seed, n);
    demo_random.set_construction_method(ConstructionMethod::Random);
    demo_random.construct_tour();
    let random_tour_length = demo_random.tour_length;

    let mut demo_greedy = TspGraspDemo::new(seed, n);
    demo_greedy.set_construction_method(ConstructionMethod::RandomizedGreedy);
    demo_greedy.construct_tour();
    let greedy_tour_length = demo_greedy.tour_length;

    println!("Problem: {} random cities in unit square", n);
    (random_tour_length, greedy_tour_length)
}

fn print_failing_test_results(random_length: f64, greedy_length: f64) {
    println!("Random construction tour length:  {:.4}", random_length);
    println!("Greedy construction tour length:  {:.4}", greedy_length);
    println!(
        "Greedy improvement: {:.1}%",
        (random_length - greedy_length) / random_length * 100.0
    );
    println!();
    println!(
        "✗ Random construction fails: tour is {:.1}% longer than greedy",
        (random_length - greedy_length) / greedy_length * 100.0
    );
    println!();
}

fn run_phase_3_implementation(seed: u64, n: usize, iterations: usize) -> TspGraspDemo {
    println!("═══ Phase 3: GRASP Implementation ═══");
    println!();

    let mut demo = TspGraspDemo::new(seed, n);
    demo.set_rcl_size(5);
    println!("Configuration:");
    println!("  • Cities: {}", n);
    println!("  • RCL size: 5 (top 5 nearest candidates)");
    println!("  • Construction: Randomized Greedy");
    println!("  • Local Search: 2-opt to local optimum");
    println!();

    println!("Running GRASP iterations...");
    println!();

    for i in 1..=iterations {
        demo.grasp_iteration();
        if i <= 5 || i == 10 || i == 15 || i == iterations {
            let gap = demo.optimality_gap() * 100.0;
            println!(
                "  Iteration {:2}: best_tour = {:.4}, gap = {:.1}%",
                i, demo.best_tour_length, gap
            );
        }
    }
    println!();
    demo
}

fn run_phase_4_verification(demo: &TspGraspDemo) {
    println!("═══ Phase 4: Verification ═══");
    println!();

    let gap = demo.optimality_gap();
    let cv = demo.restart_cv();

    println!("Final Results:");
    println!("  • Best tour length: {:.4}", demo.best_tour_length);
    println!("  • 1-tree bound:     {:.4}", demo.lower_bound);
    println!("  • Optimality gap:   {:.1}%", gap * 100.0);
    println!("  • Restart CV:       {:.1}%", cv * 100.0);
    println!("  • Edge crossings:   {}", demo.count_crossings());
    println!("  • 2-opt iterations: {}", demo.two_opt_iterations);
    println!("  • 2-opt improvements: {}", demo.two_opt_improvements);
    println!();

    let verified = demo.verify_equation();
    if verified {
        println!("✓ VERIFIED: GRASP achieves reasonable optimality gap");
    } else {
        println!("✗ NOT VERIFIED: Gap exceeds threshold");
    }
    println!();
}

fn run_phase_5_falsification() {
    println!("═══ Phase 5: Falsification ═══");
    println!();
    println!("Testing: Does greedy construction outperform random?");
    println!();

    let (greedy_wins, total_improvement, trials) = run_falsification_trials(10, 30);

    println!("Results over {} trials:", trials);
    println!("  • Greedy wins: {}/{}", greedy_wins, trials);
    println!(
        "  • Average improvement: {:.1}%",
        total_improvement / trials as f64 * 100.0
    );
    println!();

    if greedy_wins > trials / 2 {
        println!("✓ FALSIFICATION CONFIRMED: Random construction is inferior");
    } else {
        println!("✗ Falsification failed: Random performed surprisingly well");
    }
    println!();
}

fn run_falsification_trials(trials: usize, n: usize) -> (usize, f64, usize) {
    let mut greedy_wins = 0;
    let mut total_greedy_improvement = 0.0;

    for trial_seed in 0..trials {
        let mut g = TspGraspDemo::new(trial_seed as u64, n);
        g.set_construction_method(ConstructionMethod::RandomizedGreedy);
        g.construct_tour();

        let mut r = TspGraspDemo::new(trial_seed as u64, n);
        r.set_construction_method(ConstructionMethod::Random);
        r.construct_tour();

        if g.tour_length < r.tour_length {
            greedy_wins += 1;
        }
        total_greedy_improvement += (r.tour_length - g.tour_length) / r.tour_length;
    }

    (greedy_wins, total_greedy_improvement, trials)
}

fn run_visualization(demo: &TspGraspDemo) {
    println!("═══ Best Tour Visualization ═══");
    println!();

    let tour = &demo.best_tour;
    let cities = &demo.cities;

    println!("Tour path ({} cities):", tour.len());
    for (i, &city_idx) in tour.iter().enumerate() {
        let city = &cities[city_idx];
        if i < 5 || i >= tour.len() - 2 {
            println!(
                "  {:2}. City {:2} at ({:.3}, {:.3})",
                i + 1,
                city_idx,
                city.x,
                city.y
            );
        } else if i == 5 {
            println!("  ... ({} more cities) ...", tour.len() - 7);
        }
    }
    let first_city = &cities[tour[0]];
    println!(
        "  →  City {:2} at ({:.3}, {:.3}) [return]",
        tour[0], first_city.x, first_city.y
    );
    println!();
}

fn run_edd_trait_info(demo: &TspGraspDemo) {
    println!("═══ EDD Demo Trait Info ═══");
    println!();
    println!("Demo name: {}", demo.name());
    println!("EMC reference: {}", demo.emc_ref());

    let status = demo.get_falsification_status();
    println!("Status: {}", status.message);
    for criterion in &status.criteria {
        let mark = if criterion.passed { "✓" } else { "✗" };
        println!(
            "  {} {} = {:.4} (threshold: {:.4})",
            mark, criterion.name, criterion.value, criterion.threshold
        );
    }
    println!();
}

fn print_footer() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Demo Complete                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
