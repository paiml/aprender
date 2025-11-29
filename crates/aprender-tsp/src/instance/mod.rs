//! TSP instance representation and parsers.
//!
//! Supports multiple formats:
//! - TSPLIB (standard benchmark format)
//! - CSV (simple coordinate format)
//! - Distance matrix (JSON)

mod tsplib;

pub use tsplib::TsplibParser;

use crate::error::{TspError, TspResult};
use std::path::Path;

/// Edge weight computation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeWeightType {
    /// Euclidean 2D distance
    Euc2d,
    /// Euclidean 3D distance
    Euc3d,
    /// Ceiling of Euclidean 2D
    Ceil2d,
    /// Geographical distance
    Geo,
    /// Explicit distance matrix
    Explicit,
    /// Pseudo-Euclidean (ATSP)
    Att,
}

/// A TSP problem instance
#[derive(Debug, Clone)]
pub struct TspInstance {
    /// Instance name
    pub name: String,
    /// Number of cities
    pub dimension: usize,
    /// Optional comment
    pub comment: Option<String>,
    /// Edge weight type
    pub edge_weight_type: EdgeWeightType,
    /// City coordinates (if available)
    pub coords: Option<Vec<(f64, f64)>>,
    /// Distance matrix (computed or explicit)
    pub distances: Vec<Vec<f64>>,
    /// Best known solution length (if available)
    pub best_known: Option<f64>,
}

impl TspInstance {
    /// Create a new TSP instance from coordinates
    pub fn from_coords(name: &str, coords: Vec<(f64, f64)>) -> TspResult<Self> {
        if coords.is_empty() {
            return Err(TspError::InvalidInstance {
                message: "Instance must have at least one city".into(),
            });
        }

        let dimension = coords.len();
        let distances = Self::compute_distance_matrix(&coords);

        Ok(Self {
            name: name.to_string(),
            dimension,
            comment: None,
            edge_weight_type: EdgeWeightType::Euc2d,
            coords: Some(coords),
            distances,
            best_known: None,
        })
    }

    /// Create a new TSP instance from a distance matrix
    pub fn from_matrix(name: &str, distances: Vec<Vec<f64>>) -> TspResult<Self> {
        if distances.is_empty() {
            return Err(TspError::InvalidInstance {
                message: "Distance matrix cannot be empty".into(),
            });
        }

        let dimension = distances.len();

        // Validate matrix is square
        for (i, row) in distances.iter().enumerate() {
            if row.len() != dimension {
                return Err(TspError::InvalidInstance {
                    message: format!(
                        "Distance matrix row {i} has {} elements, expected {dimension}",
                        row.len()
                    ),
                });
            }
        }

        Ok(Self {
            name: name.to_string(),
            dimension,
            comment: None,
            edge_weight_type: EdgeWeightType::Explicit,
            coords: None,
            distances,
            best_known: None,
        })
    }

    /// Load instance from file (auto-detects format)
    pub fn load(path: &Path) -> TspResult<Self> {
        let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match extension.to_lowercase().as_str() {
            "tsp" => TsplibParser::parse_file(path),
            "csv" => Self::load_csv(path),
            _ => Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: None,
                cause: format!("Unknown file extension: {extension}"),
            }),
        }
    }

    /// Load from CSV format (id,x,y)
    fn load_csv(path: &Path) -> TspResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut coords = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 3 {
                return Err(TspError::ParseError {
                    file: path.to_path_buf(),
                    line: Some(line_num + 1),
                    cause: "Expected at least 3 columns: id,x,y".into(),
                });
            }

            let x: f64 = parts[1].trim().parse().map_err(|_| TspError::ParseError {
                file: path.to_path_buf(),
                line: Some(line_num + 1),
                cause: format!("Invalid x coordinate: {}", parts[1]),
            })?;

            let y: f64 = parts[2].trim().parse().map_err(|_| TspError::ParseError {
                file: path.to_path_buf(),
                line: Some(line_num + 1),
                cause: format!("Invalid y coordinate: {}", parts[2]),
            })?;

            coords.push((x, y));
        }

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unnamed")
            .to_string();

        Self::from_coords(&name, coords)
    }

    /// Compute Euclidean distance matrix from coordinates
    fn compute_distance_matrix(coords: &[(f64, f64)]) -> Vec<Vec<f64>> {
        let n = coords.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i + 1..n {
                let dx = coords[i].0 - coords[j].0;
                let dy = coords[i].1 - coords[j].1;
                let dist = (dx * dx + dy * dy).sqrt();
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }

        matrix
    }

    /// Get number of cities
    #[inline]
    pub fn num_cities(&self) -> usize {
        self.dimension
    }

    /// Get distance between two cities
    #[inline]
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        self.distances[i][j]
    }

    /// Calculate tour length
    pub fn tour_length(&self, tour: &[usize]) -> f64 {
        if tour.is_empty() {
            return 0.0;
        }

        let mut length = 0.0;
        for window in tour.windows(2) {
            length += self.distance(window[0], window[1]);
        }
        // Return to start
        length += self.distance(tour[tour.len() - 1], tour[0]);
        length
    }

    /// Validate a tour
    pub fn validate_tour(&self, tour: &[usize]) -> TspResult<()> {
        if tour.len() != self.dimension {
            return Err(TspError::InvalidInstance {
                message: format!(
                    "Tour has {} cities, expected {}",
                    tour.len(),
                    self.dimension
                ),
            });
        }

        let mut visited = vec![false; self.dimension];
        for &city in tour {
            if city >= self.dimension {
                return Err(TspError::InvalidInstance {
                    message: format!("City {city} out of range [0, {})", self.dimension),
                });
            }
            if visited[city] {
                return Err(TspError::InvalidInstance {
                    message: format!("City {city} visited multiple times"),
                });
            }
            visited[city] = true;
        }

        Ok(())
    }

    /// Get nearest neighbor rank of city j from city i
    pub fn nearest_neighbor_rank(&self, i: usize, j: usize) -> usize {
        let mut neighbors: Vec<(usize, f64)> = (0..self.dimension)
            .filter(|&k| k != i)
            .map(|k| (k, self.distance(i, k)))
            .collect();
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        neighbors
            .iter()
            .position(|&(k, _)| k == j)
            .map_or(self.dimension, |p| p + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_coords_creates_instance() {
        let coords = vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
        let instance = TspInstance::from_coords("test", coords).expect("should create instance");

        assert_eq!(instance.name, "test");
        assert_eq!(instance.dimension, 3);
        assert_eq!(instance.num_cities(), 3);
    }

    #[test]
    fn test_from_coords_computes_distances() {
        // 3-4-5 right triangle
        let coords = vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)];
        let instance = TspInstance::from_coords("test", coords).expect("should create instance");

        assert!((instance.distance(0, 1) - 3.0).abs() < 1e-10);
        assert!((instance.distance(1, 2) - 4.0).abs() < 1e-10);
        assert!((instance.distance(0, 2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_coords_empty_fails() {
        let result = TspInstance::from_coords("test", vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_matrix_creates_instance() {
        let matrix = vec![
            vec![0.0, 10.0, 15.0],
            vec![10.0, 0.0, 20.0],
            vec![15.0, 20.0, 0.0],
        ];
        let instance = TspInstance::from_matrix("test", matrix).expect("should create instance");

        assert_eq!(instance.dimension, 3);
        assert!((instance.distance(0, 1) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_matrix_non_square_fails() {
        let matrix = vec![vec![0.0, 10.0], vec![10.0, 0.0, 5.0]];
        let result = TspInstance::from_matrix("test", matrix);
        assert!(result.is_err());
    }

    #[test]
    fn test_tour_length_calculation() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let instance = TspInstance::from_coords("square", coords).expect("should create");

        // Tour around the square: 0->1->2->3->0 = 1+1+1+1 = 4
        let tour = vec![0, 1, 2, 3];
        let length = instance.tour_length(&tour);
        assert!((length - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_validate_tour_correct() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let instance = TspInstance::from_coords("test", coords).expect("should create");

        let tour = vec![0, 1, 2];
        assert!(instance.validate_tour(&tour).is_ok());
    }

    #[test]
    fn test_validate_tour_wrong_length() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let instance = TspInstance::from_coords("test", coords).expect("should create");

        let tour = vec![0, 1];
        assert!(instance.validate_tour(&tour).is_err());
    }

    #[test]
    fn test_validate_tour_duplicate_city() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let instance = TspInstance::from_coords("test", coords).expect("should create");

        let tour = vec![0, 1, 1];
        assert!(instance.validate_tour(&tour).is_err());
    }

    #[test]
    fn test_validate_tour_out_of_range() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)];
        let instance = TspInstance::from_coords("test", coords).expect("should create");

        let tour = vec![0, 1, 5];
        assert!(instance.validate_tour(&tour).is_err());
    }

    #[test]
    fn test_nearest_neighbor_rank() {
        // Cities: A(0,0), B(1,0), C(3,0)
        // From A: B is nearest (rank 1), C is 2nd (rank 2)
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)];
        let instance = TspInstance::from_coords("test", coords).expect("should create");

        assert_eq!(instance.nearest_neighbor_rank(0, 1), 1); // B is nearest to A
        assert_eq!(instance.nearest_neighbor_rank(0, 2), 2); // C is 2nd nearest to A
    }

    #[test]
    fn test_empty_tour_length() {
        let coords = vec![(0.0, 0.0), (1.0, 0.0)];
        let instance = TspInstance::from_coords("test", coords).expect("should create");

        assert!((instance.tour_length(&[]) - 0.0).abs() < 1e-10);
    }
}
