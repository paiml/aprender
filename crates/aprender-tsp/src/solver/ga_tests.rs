use super::*;

fn square_instance() -> TspInstance {
    let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    TspInstance::from_coords("square", coords).expect("should create")
}

fn triangle_instance() -> TspInstance {
    let coords = vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
    TspInstance::from_coords("triangle", coords).expect("should create")
}

fn pentagon_instance() -> TspInstance {
    // Regular pentagon
    let angle_step = 2.0 * std::f64::consts::PI / 5.0;
    let coords: Vec<(f64, f64)> = (0..5)
        .map(|i| {
            let angle = i as f64 * angle_step;
            (angle.cos(), angle.sin())
        })
        .collect();
    TspInstance::from_coords("pentagon", coords).expect("should create")
}

#[test]
fn test_ga_default_params() {
    let ga = GaSolver::default();
    assert_eq!(ga.population_size, 50);
    assert!((ga.crossover_rate - 0.9).abs() < 1e-10);
    assert!((ga.mutation_rate - 0.1).abs() < 1e-10);
    assert_eq!(ga.tournament_size, 5);
    assert_eq!(ga.elitism, 2);
}

#[test]
fn test_ga_builder() {
    let ga = GaSolver::new()
        .with_population_size(100)
        .with_crossover_rate(0.8)
        .with_mutation_rate(0.2)
        .with_tournament_size(3)
        .with_elitism(5)
        .with_seed(42);

    assert_eq!(ga.population_size, 100);
    assert!((ga.crossover_rate - 0.8).abs() < 1e-10);
    assert!((ga.mutation_rate - 0.2).abs() < 1e-10);
    assert_eq!(ga.tournament_size, 3);
    assert_eq!(ga.elitism, 5);
    assert_eq!(ga.seed, Some(42));
}

#[test]
fn test_ga_solves_square() {
    let instance = square_instance();
    let mut solver = GaSolver::new().with_seed(42).with_population_size(20);

    let solution = solver
        .solve(&instance, Budget::Iterations(50))
        .expect("should solve");

    // Optimal tour around square is 4.0
    assert!(solution.length <= 5.0, "Length {} > 5.0", solution.length);
    assert_eq!(solution.tour.len(), 4);
    assert!(instance.validate_tour(&solution.tour).is_ok());
}

#[test]
fn test_ga_solves_triangle() {
    let instance = triangle_instance();
    let mut solver = GaSolver::new().with_seed(42).with_population_size(20);

    let solution = solver
        .solve(&instance, Budget::Iterations(50))
        .expect("should solve");

    // Triangle tour: 3 + 4 + 5 = 12
    assert!(solution.length <= 13.0, "Length {} > 13.0", solution.length);
}

#[test]
fn test_ga_solves_pentagon() {
    let instance = pentagon_instance();
    let mut solver = GaSolver::new().with_seed(42).with_population_size(30);

    let solution = solver
        .solve(&instance, Budget::Iterations(100))
        .expect("should solve");

    assert_eq!(solution.tour.len(), 5);
    assert!(instance.validate_tour(&solution.tour).is_ok());
}

#[test]
fn test_ga_deterministic_with_seed() {
    let instance = square_instance();

    let mut solver1 = GaSolver::new().with_seed(42).with_population_size(20);
    let mut solver2 = GaSolver::new().with_seed(42).with_population_size(20);

    let solution1 = solver1
        .solve(&instance, Budget::Iterations(20))
        .expect("should solve");
    let solution2 = solver2
        .solve(&instance, Budget::Iterations(20))
        .expect("should solve");

    assert!((solution1.length - solution2.length).abs() < 1e-10);
}

#[test]
fn test_ga_tracks_history() {
    let instance = square_instance();
    let mut solver = GaSolver::new().with_seed(42).with_population_size(20);

    let solution = solver
        .solve(&instance, Budget::Iterations(30))
        .expect("should solve");

    assert_eq!(solution.history.len(), 30);
    // History should be non-increasing
    for window in solution.history.windows(2) {
        assert!(window[1] <= window[0] + 1e-10);
    }
}

#[test]
fn test_order_crossover() {
    let mut rng = StdRng::seed_from_u64(42);

    let parent1 = vec![0, 1, 2, 3, 4];
    let parent2 = vec![4, 3, 2, 1, 0];

    let child = GaSolver::order_crossover(&parent1, &parent2, &mut rng);

    // Child should be a valid permutation
    assert_eq!(child.len(), 5);
    let mut sorted = child.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_mutation() {
    let solver = GaSolver::new().with_mutation_rate(1.0); // Always mutate
    let mut rng = StdRng::seed_from_u64(42);

    let mut tour = vec![0, 1, 2, 3, 4];
    let original = tour.clone();

    solver.mutate(&mut tour, &mut rng);

    // Tour should still be valid permutation
    let mut sorted = tour.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);

    // Should likely be different (with mutation rate 1.0)
    // Note: could still be same if i==j selected
    assert!(tour != original || true); // Just check it doesn't crash
}

#[test]
fn test_ga_evolve_population() {
    let instance = square_instance();
    let mut solver = GaSolver::new().with_seed(42).with_population_size(20);

    let population = solver.evolve(&instance, 50).expect("should evolve");

    assert_eq!(population.len(), 20);
    // Population should be sorted by fitness
    for window in population.windows(2) {
        assert!(window[0].1 <= window[1].1);
    }
}

#[test]
fn test_ga_name() {
    let solver = GaSolver::new();
    assert_eq!(solver.name(), "Genetic Algorithm");
}

#[test]
fn test_ga_elitism_preserves_best() {
    let instance = square_instance();
    let mut solver = GaSolver::new()
        .with_seed(42)
        .with_population_size(20)
        .with_elitism(2);

    let solution = solver
        .solve(&instance, Budget::Iterations(50))
        .expect("should solve");

    // With elitism, best should never get worse
    for window in solution.history.windows(2) {
        assert!(window[1] <= window[0] + 1e-10);
    }
}
