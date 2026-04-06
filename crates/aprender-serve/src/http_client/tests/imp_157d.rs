
/// IMP-157d: Test reproducibility hash
#[test]
fn test_imp_157d_reproducibility_hash() {
    let env = EnvironmentMetadata::capture();
    let bench = BenchmarkMetadata::new("test_bench");

    let hash1 = ReproducibilityHash::compute(&env, &bench);
    let hash2 = ReproducibilityHash::compute(&env, &bench);

    assert_eq!(
        hash1.combined_hash, hash2.combined_hash,
        "IMP-157d: Same inputs should produce same hash"
    );
    assert_eq!(
        hash1.config_hash.len(),
        16,
        "IMP-157d: Config hash should be 16 chars"
    );
    assert_eq!(
        hash1.environment_hash.len(),
        16,
        "IMP-157d: Env hash should be 16 chars"
    );

    println!("\nIMP-157d: Reproducibility Hash:");
    println!("  Config: {}", hash1.config_hash);
    println!("  Environment: {}", hash1.environment_hash);
    println!("  Combined: {}", hash1.combined_hash);
}

// =========================================================================
// IMP-158: Benchmark Result JSON Schema Validation (EXTREME TDD)
// Per spec QA-040: JSON schema validation for benchmark results
