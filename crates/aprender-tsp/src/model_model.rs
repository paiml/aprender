
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_model_new_aco() {
        let model = TspModel::new(TspAlgorithm::Aco);
        assert_eq!(model.algorithm, TspAlgorithm::Aco);
        assert!(matches!(model.params, TspParams::Aco { .. }));
    }

    #[test]
    fn test_model_new_tabu() {
        let model = TspModel::new(TspAlgorithm::Tabu);
        assert_eq!(model.algorithm, TspAlgorithm::Tabu);
        assert!(matches!(model.params, TspParams::Tabu { .. }));
    }

    #[test]
    fn test_model_new_ga() {
        let model = TspModel::new(TspAlgorithm::Ga);
        assert_eq!(model.algorithm, TspAlgorithm::Ga);
        assert!(matches!(model.params, TspParams::Ga { .. }));
    }

    #[test]
    fn test_model_new_hybrid() {
        let model = TspModel::new(TspAlgorithm::Hybrid);
        assert_eq!(model.algorithm, TspAlgorithm::Hybrid);
        assert!(matches!(model.params, TspParams::Hybrid { .. }));
    }

    #[test]
    fn test_model_save_load_aco() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let model = TspModel::new(TspAlgorithm::Aco).with_params(TspParams::Aco {
            alpha: 2.0,
            beta: 3.5,
            rho: 0.2,
            q0: 0.85,
            num_ants: 30,
        });

        model.save(&path).expect("should save");
        let loaded = TspModel::load(&path).expect("should load");

        assert_eq!(loaded.algorithm, TspAlgorithm::Aco);
        if let TspParams::Aco {
            alpha,
            beta,
            rho,
            q0,
            num_ants,
        } = loaded.params
        {
            assert!((alpha - 2.0).abs() < 1e-10);
            assert!((beta - 3.5).abs() < 1e-10);
            assert!((rho - 0.2).abs() < 1e-10);
            assert!((q0 - 0.85).abs() < 1e-10);
            assert_eq!(num_ants, 30);
        } else {
            panic!("Expected ACO params");
        }
    }

    #[test]
    fn test_model_save_load_tabu() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let model = TspModel::new(TspAlgorithm::Tabu).with_params(TspParams::Tabu {
            tenure: 25,
            max_neighbors: 150,
        });

        model.save(&path).expect("should save");
        let loaded = TspModel::load(&path).expect("should load");

        assert_eq!(loaded.algorithm, TspAlgorithm::Tabu);
        if let TspParams::Tabu {
            tenure,
            max_neighbors,
        } = loaded.params
        {
            assert_eq!(tenure, 25);
            assert_eq!(max_neighbors, 150);
        } else {
            panic!("Expected Tabu params");
        }
    }

    #[test]
    fn test_model_save_load_ga() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let model = TspModel::new(TspAlgorithm::Ga).with_params(TspParams::Ga {
            population_size: 100,
            crossover_rate: 0.85,
            mutation_rate: 0.15,
        });

        model.save(&path).expect("should save");
        let loaded = TspModel::load(&path).expect("should load");

        assert_eq!(loaded.algorithm, TspAlgorithm::Ga);
        if let TspParams::Ga {
            population_size,
            crossover_rate,
            mutation_rate,
        } = loaded.params
        {
            assert_eq!(population_size, 100);
            assert!((crossover_rate - 0.85).abs() < 1e-10);
            assert!((mutation_rate - 0.15).abs() < 1e-10);
        } else {
            panic!("Expected GA params");
        }
    }

    #[test]
    fn test_model_save_load_hybrid() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let model = TspModel::new(TspAlgorithm::Hybrid).with_params(TspParams::Hybrid {
            ga_fraction: 0.5,
            tabu_fraction: 0.25,
            aco_fraction: 0.25,
        });

        model.save(&path).expect("should save");
        let loaded = TspModel::load(&path).expect("should load");

        assert_eq!(loaded.algorithm, TspAlgorithm::Hybrid);
        if let TspParams::Hybrid {
            ga_fraction,
            tabu_fraction,
            aco_fraction,
        } = loaded.params
        {
            assert!((ga_fraction - 0.5).abs() < 1e-10);
            assert!((tabu_fraction - 0.25).abs() < 1e-10);
            assert!((aco_fraction - 0.25).abs() < 1e-10);
        } else {
            panic!("Expected Hybrid params");
        }
    }

    #[test]
    fn test_model_metadata_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.apr");

        let metadata = TspModelMetadata {
            trained_instances: 10,
            avg_instance_size: 52,
            best_known_gap: 0.03,
            training_time_secs: 2.5,
        };

        let model = TspModel::new(TspAlgorithm::Aco).with_metadata(metadata);
        model.save(&path).expect("should save");
        let loaded = TspModel::load(&path).expect("should load");

        assert_eq!(loaded.metadata.trained_instances, 10);
        assert_eq!(loaded.metadata.avg_instance_size, 52);
        assert!((loaded.metadata.best_known_gap - 0.03).abs() < 1e-10);
        assert!((loaded.metadata.training_time_secs - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_model_invalid_magic() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("bad.apr");

        // Write invalid magic (must be at least 16 bytes to pass size check)
        let mut data = vec![0u8; 20];
        data[0..4].copy_from_slice(b"BAD\x00");
        std::fs::write(&path, &data).unwrap();

        let result = TspModel::load(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Not an .apr file"), "Unexpected error: {err}");
    }

    #[test]
    fn test_model_invalid_checksum() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("corrupt.apr");

        // Create valid model
        let model = TspModel::new(TspAlgorithm::Aco);
        model.save(&path).expect("should save");

        // Corrupt the checksum
        let mut data = std::fs::read(&path).unwrap();
        data[12] ^= 0xFF; // Flip bits in checksum
        std::fs::write(&path, &data).unwrap();

        let result = TspModel::load(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("checksum mismatch"));
    }

    #[test]
    fn test_model_file_too_small() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("small.apr");

        // Write too-small file
        std::fs::write(&path, b"APR\x00").unwrap();

        let result = TspModel::load(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("too small"));
    }

    #[test]
    fn test_model_unsupported_version() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("future.apr");

        // Write header with future version
        let mut data = Vec::new();
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes()); // Future version
        data.extend_from_slice(&MODEL_TYPE_TSP.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes()); // Fake checksum
        std::fs::write(&path, &data).unwrap();

        let result = TspModel::load(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("version"));
    }
}
