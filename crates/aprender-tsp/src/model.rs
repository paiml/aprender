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

include!("payload_reader.rs");
include!("model_part_03.rs");
