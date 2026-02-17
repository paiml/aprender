//! TSPLIB format parser.
//!
//! Reference: Reinelt (1991) "TSPLIB—A Traveling Salesman Problem Library"

use crate::error::{TspError, TspResult};
use crate::instance::{EdgeWeightType, TspInstance};
use std::path::Path;

/// Intermediate state during TSPLIB parsing
#[derive(Default)]
struct TsplibParseState {
    name: String,
    comment: Option<String>,
    dimension: usize,
    edge_weight_type: EdgeWeightType,
    coords: Vec<(f64, f64)>,
    edge_weights: Vec<f64>,
    best_known: Option<f64>,
}

impl TsplibParseState {
    fn into_instance(self, path: &Path) -> Result<TspInstance, TspError> {
        if self.dimension == 0 {
            return Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: None,
                cause: "Missing DIMENSION field".into(),
            });
        }

        let distances = if !self.edge_weights.is_empty() {
            TsplibParser::build_matrix_from_weights(&self.edge_weights, self.dimension, path)?
        } else if !self.coords.is_empty() {
            TsplibParser::compute_distance_matrix(&self.coords, self.edge_weight_type)
        } else {
            return Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: None,
                cause: "No coordinates or edge weights found".into(),
            });
        };

        let name = if self.name.is_empty() {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unnamed")
                .to_string()
        } else {
            self.name
        };

        Ok(TspInstance {
            name,
            dimension: self.dimension,
            comment: self.comment,
            edge_weight_type: self.edge_weight_type,
            coords: if self.coords.is_empty() { None } else { Some(self.coords) },
            distances,
            best_known: self.best_known,
        })
    }
}

/// Parser for TSPLIB format files
#[derive(Debug)]
pub struct TsplibParser;

impl TsplibParser {
    /// Parse a TSPLIB file
    pub fn parse_file(path: &Path) -> TspResult<TspInstance> {
        let content = std::fs::read_to_string(path)?;
        Self::parse(&content, path)
    }

    /// Parse a single node coordinate line ("id x y")
    fn parse_coord_line(
        line: &str,
        path: &Path,
        line_num: usize,
    ) -> TspResult<Option<(f64, f64)>> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            let x: f64 = parts[1].parse().map_err(|_| TspError::ParseError {
                file: path.to_path_buf(),
                line: Some(line_num + 1),
                cause: format!("Invalid x coordinate: {}", parts[1]),
            })?;
            let y: f64 = parts[2].parse().map_err(|_| TspError::ParseError {
                file: path.to_path_buf(),
                line: Some(line_num + 1),
                cause: format!("Invalid y coordinate: {}", parts[2]),
            })?;
            Ok(Some((x, y)))
        } else {
            Ok(None)
        }
    }

    /// Parse edge weight values from a line
    fn parse_weight_line(
        line: &str,
        path: &Path,
        line_num: usize,
        weights: &mut Vec<f64>,
    ) -> TspResult<()> {
        for part in line.split_whitespace() {
            let weight: f64 = part.parse().map_err(|_| TspError::ParseError {
                file: path.to_path_buf(),
                line: Some(line_num + 1),
                cause: format!("Invalid edge weight: {part}"),
            })?;
            weights.push(weight);
        }
        Ok(())
    }

    /// Parse a header key-value field
    fn parse_header_field(
        key: &str,
        value: &str,
        path: &Path,
        line_num: usize,
        state: &mut TsplibParseState,
    ) -> TspResult<()> {
        match key {
            "NAME" => state.name = value.to_string(),
            "COMMENT" => {
                state.comment = Some(value.to_string());
                if let Some(opt) = Self::extract_optimal_from_comment(value) {
                    state.best_known = Some(opt);
                }
            }
            "BEST_KNOWN" | "OPTIMAL" => {
                if let Ok(opt) = value.parse::<f64>() {
                    state.best_known = Some(opt);
                }
            }
            "DIMENSION" => {
                state.dimension = value.parse().map_err(|_| TspError::ParseError {
                    file: path.to_path_buf(),
                    line: Some(line_num + 1),
                    cause: format!("Invalid dimension: {value}"),
                })?;
            }
            "EDGE_WEIGHT_TYPE" => {
                state.edge_weight_type = Self::parse_edge_weight_type(value, path, line_num)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Parse TSPLIB content
    pub fn parse(content: &str, path: &Path) -> TspResult<TspInstance> {
        let mut state = TsplibParseState::default();
        let mut in_node_coord = false;
        let mut in_edge_weight = false;

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line == "EOF" {
                continue;
            }

            // Section headers
            match line {
                "NODE_COORD_SECTION" => { in_node_coord = true; in_edge_weight = false; continue; }
                "EDGE_WEIGHT_SECTION" => { in_edge_weight = true; in_node_coord = false; continue; }
                "DISPLAY_DATA_SECTION" => { in_node_coord = false; in_edge_weight = false; continue; }
                _ => {}
            }

            if in_node_coord {
                if let Some(coord) = Self::parse_coord_line(line, path, line_num)? {
                    state.coords.push(coord);
                }
            } else if in_edge_weight {
                Self::parse_weight_line(line, path, line_num, &mut state.edge_weights)?;
            } else if let Some((key, value)) = line.split_once(':') {
                Self::parse_header_field(
                    key.trim().to_uppercase().as_str(),
                    value.trim(),
                    path,
                    line_num,
                    &mut state,
                )?;
            }
        }

        state.into_instance(path)
    }

    fn parse_edge_weight_type(
        value: &str,
        path: &Path,
        line_num: usize,
    ) -> TspResult<EdgeWeightType> {
        match value.to_uppercase().as_str() {
            "EUC_2D" => Ok(EdgeWeightType::Euc2d),
            "EUC_3D" => Ok(EdgeWeightType::Euc3d),
            "CEIL_2D" => Ok(EdgeWeightType::Ceil2d),
            "GEO" => Ok(EdgeWeightType::Geo),
            "ATT" => Ok(EdgeWeightType::Att),
            "EXPLICIT" => Ok(EdgeWeightType::Explicit),
            _ => Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: Some(line_num + 1),
                cause: format!("Unsupported edge weight type: {value}"),
            }),
        }
    }

    fn compute_distance_matrix(coords: &[(f64, f64)], edge_type: EdgeWeightType) -> Vec<Vec<f64>> {
        let n = coords.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i + 1..n {
                let dist = match edge_type {
                    EdgeWeightType::Euc2d => {
                        let dx = coords[i].0 - coords[j].0;
                        let dy = coords[i].1 - coords[j].1;
                        (dx * dx + dy * dy).sqrt()
                    }
                    EdgeWeightType::Ceil2d => {
                        let dx = coords[i].0 - coords[j].0;
                        let dy = coords[i].1 - coords[j].1;
                        (dx * dx + dy * dy).sqrt().ceil()
                    }
                    EdgeWeightType::Att => {
                        // ATT distance (pseudo-Euclidean)
                        // TSPLIB formula: rij = sqrt((xd*xd + yd*yd) / 10.0)
                        // Note: division by 10 is INSIDE sqrt, not after
                        let dx = coords[i].0 - coords[j].0;
                        let dy = coords[i].1 - coords[j].1;
                        let r = ((dx * dx + dy * dy) / 10.0).sqrt();
                        let t = r.round();
                        if t < r {
                            t + 1.0
                        } else {
                            t
                        }
                    }
                    EdgeWeightType::Geo => {
                        // Geographic distance
                        Self::geo_distance(coords[i], coords[j])
                    }
                    _ => {
                        // Default to Euclidean
                        let dx = coords[i].0 - coords[j].0;
                        let dy = coords[i].1 - coords[j].1;
                        (dx * dx + dy * dy).sqrt()
                    }
                };
                matrix[i][j] = dist;
                matrix[j][i] = dist;
            }
        }

        matrix
    }

    fn geo_distance(c1: (f64, f64), c2: (f64, f64)) -> f64 {
        const PI: f64 = std::f64::consts::PI;
        const RRR: f64 = 6378.388; // Earth radius in km

        let deg_to_rad = |deg: f64| -> f64 {
            let deg_int = deg.trunc();
            let min = deg - deg_int;
            PI * (deg_int + 5.0 * min / 3.0) / 180.0
        };

        let lat1 = deg_to_rad(c1.0);
        let lon1 = deg_to_rad(c1.1);
        let lat2 = deg_to_rad(c2.0);
        let lon2 = deg_to_rad(c2.1);

        let q1 = (lon1 - lon2).cos();
        let q2 = (lat1 - lat2).cos();
        let q3 = (lat1 + lat2).cos();

        let dij = RRR * (0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)).acos() + 1.0;
        dij.floor()
    }

    /// Extract optimal tour length from COMMENT field.
    ///
    /// Recognizes patterns like:
    /// - "Optimal tour: 7542"
    /// - "Optimal: 7542"
    /// - "Best known: 426"
    /// - "optimal solution: 10628"
    /// - "Length = 7542"
    fn extract_optimal_from_comment(comment: &str) -> Option<f64> {
        let comment_lower = comment.to_lowercase();

        // List of patterns to try
        let patterns = [
            "optimal tour:",
            "optimal:",
            "best known:",
            "optimal solution:",
            "best:",
            "length =",
            "length:",
            "tour length:",
        ];

        for pattern in patterns {
            if let Some(pos) = comment_lower.find(pattern) {
                let after_pattern = &comment[pos + pattern.len()..];
                // Extract the number following the pattern
                let num_str: String = after_pattern
                    .chars()
                    .skip_while(|c| c.is_whitespace())
                    .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == ',')
                    .filter(|c| *c != ',') // Remove thousand separators
                    .collect();

                if let Ok(val) = num_str.parse::<f64>() {
                    return Some(val);
                }
            }
        }

        // Also try to find a standalone number at the end like "(7542)"
        if let Some(start) = comment.rfind('(') {
            if let Some(end) = comment.rfind(')') {
                if start < end {
                    let num_str: String = comment[start + 1..end]
                        .chars()
                        .filter(|c| c.is_ascii_digit() || *c == '.')
                        .collect();
                    if let Ok(val) = num_str.parse::<f64>() {
                        return Some(val);
                    }
                }
            }
        }

        None
    }

    fn build_matrix_from_weights(
        weights: &[f64],
        dimension: usize,
        path: &Path,
    ) -> TspResult<Vec<Vec<f64>>> {
        let expected = dimension * (dimension - 1) / 2;
        if weights.len() < expected {
            return Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: None,
                cause: format!(
                    "Not enough edge weights: got {}, expected at least {} for dimension {}",
                    weights.len(),
                    expected,
                    dimension
                ),
            });
        }

        let mut matrix = vec![vec![0.0; dimension]; dimension];
        let mut weight_iter = weights.iter();

        // Assume lower triangular format
        // Use range loops here as we need indices for symmetric matrix assignment
        #[allow(clippy::needless_range_loop)]
        for i in 1..dimension {
            for j in 0..i {
                if let Some(&weight) = weight_iter.next() {
                    matrix[i][j] = weight;
                    matrix[j][i] = weight;
                }
            }
        }

        Ok(matrix)
    }
}

#[cfg(test)]
#[path = "tsplib_tests.rs"]
mod tests;
