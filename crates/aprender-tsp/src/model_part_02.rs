
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
