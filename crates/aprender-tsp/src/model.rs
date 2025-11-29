//! TSP model persistence in .apr format.
//!
//! Toyota Way Principle: *Standardized Work* - Consistent .apr format enables
//! reproducible results across environments.

use crate::error::{TspError, TspResult};
use crate::solver::TspAlgorithm;
use std::io::{Read, Write};
use std::path::Path;

/// Magic number for TSP .apr files
const MAGIC: &[u8; 4] = b"APR\x00";

/// Version number
const VERSION: u32 = 1;

/// Model type identifier for TSP
const MODEL_TYPE_TSP: u32 = 0x54_53_50_00; // "TSP\x00"

/// Algorithm-specific parameters
#[derive(Debug, Clone)]
pub enum TspParams {
    /// ACO parameters
    Aco {
        alpha: f64,
        beta: f64,
        rho: f64,
        q0: f64,
        num_ants: usize,
    },
    /// Tabu Search parameters
    Tabu { tenure: usize, max_neighbors: usize },
    /// Genetic Algorithm parameters
    Ga {
        population_size: usize,
        crossover_rate: f64,
        mutation_rate: f64,
    },
    /// Hybrid parameters
    Hybrid {
        ga_fraction: f64,
        tabu_fraction: f64,
        aco_fraction: f64,
    },
}

impl Default for TspParams {
    fn default() -> Self {
        Self::Aco {
            alpha: 1.0,
            beta: 2.5,
            rho: 0.1,
            q0: 0.9,
            num_ants: 20,
        }
    }
}

/// Training metadata
#[derive(Debug, Clone)]
pub struct TspModelMetadata {
    /// Number of instances used for training
    pub trained_instances: u32,
    /// Average instance size
    pub avg_instance_size: u32,
    /// Best known gap achieved during training
    pub best_known_gap: f64,
    /// Training time in seconds
    pub training_time_secs: f64,
}

impl Default for TspModelMetadata {
    fn default() -> Self {
        Self {
            trained_instances: 0,
            avg_instance_size: 0,
            best_known_gap: 0.0,
            training_time_secs: 0.0,
        }
    }
}

/// TSP model persisted in .apr format
#[derive(Debug, Clone)]
pub struct TspModel {
    /// Solver algorithm
    pub algorithm: TspAlgorithm,
    /// Learned parameters (algorithm-specific)
    pub params: TspParams,
    /// Training metadata
    pub metadata: TspModelMetadata,
}

impl TspModel {
    /// Create a new TSP model with default ACO parameters
    pub fn new(algorithm: TspAlgorithm) -> Self {
        let params = match algorithm {
            TspAlgorithm::Aco => TspParams::Aco {
                alpha: 1.0,
                beta: 2.5,
                rho: 0.1,
                q0: 0.9,
                num_ants: 20,
            },
            TspAlgorithm::Tabu => TspParams::Tabu {
                tenure: 20,
                max_neighbors: 100,
            },
            TspAlgorithm::Ga => TspParams::Ga {
                population_size: 50,
                crossover_rate: 0.9,
                mutation_rate: 0.1,
            },
            TspAlgorithm::Hybrid => TspParams::Hybrid {
                ga_fraction: 0.4,
                tabu_fraction: 0.3,
                aco_fraction: 0.3,
            },
        };

        Self {
            algorithm,
            params,
            metadata: TspModelMetadata::default(),
        }
    }

    /// Set parameters
    pub fn with_params(mut self, params: TspParams) -> Self {
        self.params = params;
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: TspModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Save model to .apr file
    pub fn save(&self, path: &Path) -> TspResult<()> {
        let mut file = std::fs::File::create(path)?;

        // Serialize payload first to compute checksum
        let payload = self.serialize_payload();

        // Write header
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&MODEL_TYPE_TSP.to_le_bytes())?;

        // Compute and write checksum
        let checksum = crc32fast::hash(&payload);
        file.write_all(&checksum.to_le_bytes())?;

        // Write payload
        file.write_all(&payload)?;

        Ok(())
    }

    /// Load model from .apr file
    pub fn load(path: &Path) -> TspResult<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        Self::from_bytes(&data, path)
    }

    /// Load from bytes
    fn from_bytes(data: &[u8], path: &Path) -> TspResult<Self> {
        // Minimum size: magic(4) + version(4) + type(4) + checksum(4) + min_payload
        if data.len() < 16 {
            return Err(TspError::InvalidFormat {
                message: "File too small".into(),
                hint: "Ensure this is a valid .apr file".into(),
            });
        }

        // Verify magic
        if &data[0..4] != MAGIC {
            return Err(TspError::InvalidFormat {
                message: "Not an .apr file".into(),
                hint: format!("Expected magic 'APR\\x00', got {:?}", &data[0..4]),
            });
        }

        // Verify version
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != VERSION {
            return Err(TspError::InvalidFormat {
                message: format!("Unsupported version: {version}"),
                hint: format!("This tool supports version {VERSION}"),
            });
        }

        // Verify model type
        let model_type = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        if model_type != MODEL_TYPE_TSP {
            return Err(TspError::InvalidFormat {
                message: "Not a TSP model".into(),
                hint: "This file contains a different model type".into(),
            });
        }

        // Verify checksum
        let stored_checksum = u32::from_le_bytes([data[12], data[13], data[14], data[15]]);
        let payload = &data[16..];
        let computed_checksum = crc32fast::hash(payload);

        if stored_checksum != computed_checksum {
            return Err(TspError::ChecksumMismatch {
                expected: stored_checksum,
                computed: computed_checksum,
            });
        }

        Self::deserialize_payload(payload, path)
    }

    /// Serialize payload (without header)
    fn serialize_payload(&self) -> Vec<u8> {
        let mut payload = Vec::new();

        // Algorithm type (1 byte)
        let algo_byte = match self.algorithm {
            TspAlgorithm::Aco => 0u8,
            TspAlgorithm::Tabu => 1u8,
            TspAlgorithm::Ga => 2u8,
            TspAlgorithm::Hybrid => 3u8,
        };
        payload.push(algo_byte);

        // Metadata
        payload.extend_from_slice(&self.metadata.trained_instances.to_le_bytes());
        payload.extend_from_slice(&self.metadata.avg_instance_size.to_le_bytes());
        payload.extend_from_slice(&self.metadata.best_known_gap.to_le_bytes());
        payload.extend_from_slice(&self.metadata.training_time_secs.to_le_bytes());

        // Algorithm-specific parameters
        match &self.params {
            TspParams::Aco {
                alpha,
                beta,
                rho,
                q0,
                num_ants,
            } => {
                payload.extend_from_slice(&alpha.to_le_bytes());
                payload.extend_from_slice(&beta.to_le_bytes());
                payload.extend_from_slice(&rho.to_le_bytes());
                payload.extend_from_slice(&q0.to_le_bytes());
                payload.extend_from_slice(&(*num_ants as u32).to_le_bytes());
            }
            TspParams::Tabu {
                tenure,
                max_neighbors,
            } => {
                payload.extend_from_slice(&(*tenure as u32).to_le_bytes());
                payload.extend_from_slice(&(*max_neighbors as u32).to_le_bytes());
            }
            TspParams::Ga {
                population_size,
                crossover_rate,
                mutation_rate,
            } => {
                payload.extend_from_slice(&(*population_size as u32).to_le_bytes());
                payload.extend_from_slice(&crossover_rate.to_le_bytes());
                payload.extend_from_slice(&mutation_rate.to_le_bytes());
            }
            TspParams::Hybrid {
                ga_fraction,
                tabu_fraction,
                aco_fraction,
            } => {
                payload.extend_from_slice(&ga_fraction.to_le_bytes());
                payload.extend_from_slice(&tabu_fraction.to_le_bytes());
                payload.extend_from_slice(&aco_fraction.to_le_bytes());
            }
        }

        payload
    }

    /// Deserialize payload
    #[allow(clippy::too_many_lines)]
    fn deserialize_payload(payload: &[u8], path: &Path) -> TspResult<Self> {
        if payload.is_empty() {
            return Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: None,
                cause: "Empty payload".into(),
            });
        }

        let algo_byte = payload[0];
        let algorithm = match algo_byte {
            0 => TspAlgorithm::Aco,
            1 => TspAlgorithm::Tabu,
            2 => TspAlgorithm::Ga,
            3 => TspAlgorithm::Hybrid,
            _ => {
                return Err(TspError::InvalidFormat {
                    message: format!("Unknown algorithm type: {algo_byte}"),
                    hint: "Supported: aco (0), tabu (1), ga (2), hybrid (3)".into(),
                });
            }
        };

        // Minimum payload size check
        let min_size = 1 + 4 + 4 + 8 + 8; // algo + metadata
        if payload.len() < min_size {
            return Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: None,
                cause: "Payload too small for metadata".into(),
            });
        }

        let mut offset = 1;

        // Metadata
        let trained_instances = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        offset += 4;

        let avg_instance_size = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        offset += 4;

        let best_known_gap = f64::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
            payload[offset + 4],
            payload[offset + 5],
            payload[offset + 6],
            payload[offset + 7],
        ]);
        offset += 8;

        let training_time_secs = f64::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
            payload[offset + 4],
            payload[offset + 5],
            payload[offset + 6],
            payload[offset + 7],
        ]);
        offset += 8;

        let metadata = TspModelMetadata {
            trained_instances,
            avg_instance_size,
            best_known_gap,
            training_time_secs,
        };

        // Algorithm-specific parameters
        let params =
            match algorithm {
                TspAlgorithm::Aco => {
                    let alpha =
                        f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read alpha".into(),
                            },
                        )?);
                    offset += 8;

                    let beta = f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                        |_| TspError::ParseError {
                            file: path.to_path_buf(),
                            line: None,
                            cause: "Failed to read beta".into(),
                        },
                    )?);
                    offset += 8;

                    let rho = f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                        |_| TspError::ParseError {
                            file: path.to_path_buf(),
                            line: None,
                            cause: "Failed to read rho".into(),
                        },
                    )?);
                    offset += 8;

                    let q0 = f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                        |_| TspError::ParseError {
                            file: path.to_path_buf(),
                            line: None,
                            cause: "Failed to read q0".into(),
                        },
                    )?);
                    offset += 8;

                    let num_ants =
                        u32::from_le_bytes(payload[offset..offset + 4].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read num_ants".into(),
                            },
                        )?) as usize;

                    TspParams::Aco {
                        alpha,
                        beta,
                        rho,
                        q0,
                        num_ants,
                    }
                }
                TspAlgorithm::Tabu => {
                    let tenure =
                        u32::from_le_bytes(payload[offset..offset + 4].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read tenure".into(),
                            },
                        )?) as usize;
                    offset += 4;

                    let max_neighbors =
                        u32::from_le_bytes(payload[offset..offset + 4].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read max_neighbors".into(),
                            },
                        )?) as usize;

                    TspParams::Tabu {
                        tenure,
                        max_neighbors,
                    }
                }
                TspAlgorithm::Ga => {
                    let population_size =
                        u32::from_le_bytes(payload[offset..offset + 4].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read population_size".into(),
                            },
                        )?) as usize;
                    offset += 4;

                    let crossover_rate =
                        f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read crossover_rate".into(),
                            },
                        )?);
                    offset += 8;

                    let mutation_rate =
                        f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read mutation_rate".into(),
                            },
                        )?);

                    TspParams::Ga {
                        population_size,
                        crossover_rate,
                        mutation_rate,
                    }
                }
                TspAlgorithm::Hybrid => {
                    let ga_fraction =
                        f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read ga_fraction".into(),
                            },
                        )?);
                    offset += 8;

                    let tabu_fraction =
                        f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read tabu_fraction".into(),
                            },
                        )?);
                    offset += 8;

                    let aco_fraction =
                        f64::from_le_bytes(payload[offset..offset + 8].try_into().map_err(
                            |_| TspError::ParseError {
                                file: path.to_path_buf(),
                                line: None,
                                cause: "Failed to read aco_fraction".into(),
                            },
                        )?);

                    TspParams::Hybrid {
                        ga_fraction,
                        tabu_fraction,
                        aco_fraction,
                    }
                }
            };

        Ok(Self {
            algorithm,
            params,
            metadata,
        })
    }
}

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
