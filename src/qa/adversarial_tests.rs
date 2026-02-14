use super::*;

#[test]
fn test_fgsm_perturb() {
    let attack = FgsmAttack::new(0.1);
    let input = vec![1.0, 2.0, 3.0];
    let gradient = vec![0.5, -0.5, 0.0];

    let perturbed = attack.perturb(&input, &gradient);

    assert_eq!(perturbed.len(), 3);
    assert!((perturbed[0] - 1.1).abs() < 1e-6); // positive grad -> +epsilon
    assert!((perturbed[1] - 1.9).abs() < 1e-6); // negative grad -> -epsilon
    assert!((perturbed[2] - 3.1).abs() < 1e-6); // zero grad -> +epsilon (sign(0) = 1)
}

#[test]
fn test_fgsm_batch() {
    let attack = FgsmAttack::new(0.1);
    let inputs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let gradients = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];

    let perturbed = attack.attack_batch(&inputs, &gradients);

    assert_eq!(perturbed.len(), 2);
    assert_eq!(perturbed[0].len(), 2);
}

#[test]
fn test_pgd_step() {
    let attack = PgdAttack::new(10, 0.01, 0.1);
    let original = vec![1.0, 2.0];
    let current = vec![1.05, 1.95];
    let gradient = vec![1.0, -1.0];

    let next = attack.step(&current, &original, &gradient);

    // Should move in direction of gradient sign
    assert!(next[0] > current[0]);
    assert!(next[1] < current[1]);

    // Should stay within epsilon of original
    assert!((next[0] - original[0]).abs() <= attack.epsilon + 1e-6);
    assert!((next[1] - original[1]).abs() <= attack.epsilon + 1e-6);
}

#[test]
fn test_pgd_attack() {
    let attack = PgdAttack::new(5, 0.01, 0.1);
    let input = vec![1.0, 2.0, 3.0];

    // Mock gradient function
    let result = attack.attack(&input, |_x| vec![1.0, 1.0, 1.0]);

    assert_eq!(result.len(), 3);
    // After attack, values should be perturbed up to epsilon
    for (orig, perturbed) in input.iter().zip(result.iter()) {
        assert!((perturbed - orig).abs() <= attack.epsilon + 1e-6);
    }
}

#[test]
fn test_gaussian_noise() {
    let attack = GaussianNoiseAttack::new(0.1, 42);
    let input = vec![1.0, 2.0, 3.0];

    let perturbed = attack.perturb(&input);

    assert_eq!(perturbed.len(), 3);
    // Values should be different from original
    let any_different = input
        .iter()
        .zip(perturbed.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(any_different);
}

#[test]
fn test_gaussian_noise_reproducibility() {
    let attack1 = GaussianNoiseAttack::new(0.1, 42);
    let attack2 = GaussianNoiseAttack::new(0.1, 42);
    let input = vec![1.0, 2.0, 3.0];

    let perturbed1 = attack1.perturb(&input);
    let perturbed2 = attack2.perturb(&input);

    // Same seed should give same results
    for (a, b) in perturbed1.iter().zip(perturbed2.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn test_attack_result() {
    let result = AttackResult::new("FGSM", 0.95, 0.92, 0.05, Duration::from_millis(100));

    assert_eq!(result.attack_name, "FGSM");
    assert!((result.accuracy_drop - 0.03).abs() < 1e-6);
    assert!(result.is_robust); // 3% < 5%

    let result2 = AttackResult::new("PGD", 0.95, 0.85, 0.05, Duration::from_millis(100));
    assert!(!result2.is_robust); // 10% > 5%
}

#[test]
fn test_adversarial_config_default() {
    let config = AdversarialConfig::default();
    assert!((config.fgsm_epsilon - 0.1).abs() < 1e-6);
    assert_eq!(config.pgd_steps, 10);
    assert!((config.max_accuracy_drop - 0.05).abs() < 1e-6);
}

#[test]
fn test_run_robustness_tests() {
    let config = AdversarialConfig::default();
    let (score, issues) = run_robustness_tests(&config);

    assert_eq!(score.points_possible, 20);
    assert!(score.tests_passed + score.tests_failed > 0);
    // PGD typically fails with default config
    assert!(!issues.is_empty());
}

#[test]
fn test_run_robustness_tests_all_robust() {
    // Create config where all tests pass (large max_accuracy_drop)
    let config = AdversarialConfig {
        max_accuracy_drop: 0.20, // 20% allowed drop - all should pass
        ..Default::default()
    };
    let (score, issues) = run_robustness_tests(&config);

    // With large allowed drop, all tests should pass
    assert!(score.tests_passed >= 3);
    assert!(issues.is_empty());
}

#[test]
fn test_run_robustness_tests_none_robust() {
    // Create config where all tests fail (tiny max_accuracy_drop)
    let config = AdversarialConfig {
        max_accuracy_drop: 0.001, // 0.1% allowed - all should fail
        ..Default::default()
    };
    let (score, issues) = run_robustness_tests(&config);

    // With tiny allowed drop, all tests should fail
    assert!(score.tests_failed >= 3);
    // Should have issues for FGSM, PGD, and noise
    assert!(issues.len() >= 2); // At least FGSM and PGD issues
}

#[test]
fn test_attack_result_negative_drop_clamped() {
    // Test that negative accuracy drop is clamped to 0
    let result = AttackResult::new(
        "Test",
        0.90,
        0.95, // attacked is higher than original
        0.05,
        Duration::from_millis(100),
    );

    assert!((result.accuracy_drop - 0.0).abs() < 1e-6);
    assert!(result.is_robust); // 0% drop is within threshold
}

#[test]
fn test_fgsm_perturb_empty_input() {
    let attack = FgsmAttack::new(0.1);
    let input: Vec<f32> = vec![];
    let gradient: Vec<f32> = vec![];

    let perturbed = attack.perturb(&input, &gradient);
    assert!(perturbed.is_empty());
}

#[test]
fn test_pgd_attack_zero_steps() {
    let attack = PgdAttack::new(0, 0.01, 0.1);
    let input = vec![1.0, 2.0, 3.0];

    // With zero steps, output should equal input
    let result = attack.attack(&input, |_x| vec![1.0, 1.0, 1.0]);
    assert_eq!(result, input);
}

#[test]
fn test_gaussian_noise_different_seeds() {
    let attack1 = GaussianNoiseAttack::new(0.1, 42);
    let attack2 = GaussianNoiseAttack::new(0.1, 43); // Different seed
    let input = vec![1.0, 2.0, 3.0];

    let perturbed1 = attack1.perturb(&input);
    let perturbed2 = attack2.perturb(&input);

    // Different seeds should give different results
    let any_different = perturbed1
        .iter()
        .zip(perturbed2.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(any_different);
}

#[test]
fn test_gaussian_noise_empty_input() {
    let attack = GaussianNoiseAttack::new(0.1, 42);
    let input: Vec<f32> = vec![];

    let perturbed = attack.perturb(&input);
    assert!(perturbed.is_empty());
}

#[test]
fn test_adversarial_config_clone() {
    let config = AdversarialConfig::default();
    let cloned = config.clone();

    assert!((config.fgsm_epsilon - cloned.fgsm_epsilon).abs() < 1e-6);
    assert_eq!(config.pgd_steps, cloned.pgd_steps);
}

#[test]
fn test_attack_result_clone() {
    let result = AttackResult::new("Test", 0.95, 0.90, 0.05, Duration::from_millis(100));
    let cloned = result.clone();

    assert_eq!(result.attack_name, cloned.attack_name);
    assert!((result.accuracy_drop - cloned.accuracy_drop).abs() < 1e-6);
}
