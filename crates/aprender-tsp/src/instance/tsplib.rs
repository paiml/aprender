//! TSPLIB format parser.
//!
//! Reference: Reinelt (1991) "TSPLIB—A Traveling Salesman Problem Library"

use crate::error::{TspError, TspResult};
use crate::instance::{EdgeWeightType, TspInstance};
use std::path::Path;

/// Parser for TSPLIB format files
#[derive(Debug)]
pub struct TsplibParser;

impl TsplibParser {
    /// Parse a TSPLIB file
    pub fn parse_file(path: &Path) -> TspResult<TspInstance> {
        let content = std::fs::read_to_string(path)?;
        Self::parse(&content, path)
    }

    /// Parse TSPLIB content
    #[allow(clippy::too_many_lines)]
    pub fn parse(content: &str, path: &Path) -> TspResult<TspInstance> {
        let mut name = String::new();
        let mut comment = None;
        let mut dimension = 0;
        let mut edge_weight_type = EdgeWeightType::Euc2d;
        let mut coords: Vec<(f64, f64)> = Vec::new();
        let mut in_node_coord_section = false;
        let mut in_edge_weight_section = false;
        let mut edge_weights: Vec<f64> = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();

            if line.is_empty() || line == "EOF" {
                continue;
            }

            // Check for section headers
            if line == "NODE_COORD_SECTION" {
                in_node_coord_section = true;
                in_edge_weight_section = false;
                continue;
            }
            if line == "EDGE_WEIGHT_SECTION" {
                in_edge_weight_section = true;
                in_node_coord_section = false;
                continue;
            }
            if line == "DISPLAY_DATA_SECTION" {
                in_node_coord_section = false;
                in_edge_weight_section = false;
                continue;
            }

            // Parse node coordinates
            if in_node_coord_section {
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
                    coords.push((x, y));
                }
                continue;
            }

            // Parse edge weights
            if in_edge_weight_section {
                for part in line.split_whitespace() {
                    let weight: f64 = part.parse().map_err(|_| TspError::ParseError {
                        file: path.to_path_buf(),
                        line: Some(line_num + 1),
                        cause: format!("Invalid edge weight: {part}"),
                    })?;
                    edge_weights.push(weight);
                }
                continue;
            }

            // Parse header fields
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().to_uppercase();
                let value = value.trim();

                match key.as_str() {
                    "NAME" => name = value.to_string(),
                    "COMMENT" => comment = Some(value.to_string()),
                    "DIMENSION" => {
                        dimension = value.parse().map_err(|_| TspError::ParseError {
                            file: path.to_path_buf(),
                            line: Some(line_num + 1),
                            cause: format!("Invalid dimension: {value}"),
                        })?;
                    }
                    "EDGE_WEIGHT_TYPE" => {
                        edge_weight_type = Self::parse_edge_weight_type(value, path, line_num)?;
                    }
                    // Unknown or informational fields - ignore
                    _ => {}
                }
            }
        }

        // Validate parsed data
        if dimension == 0 {
            return Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: None,
                cause: "Missing DIMENSION field".into(),
            });
        }

        // Build distance matrix
        let distances = if !edge_weights.is_empty() {
            Self::build_matrix_from_weights(&edge_weights, dimension, path)?
        } else if !coords.is_empty() {
            Self::compute_distance_matrix(&coords, edge_weight_type)
        } else {
            return Err(TspError::ParseError {
                file: path.to_path_buf(),
                line: None,
                cause: "No coordinates or edge weights found".into(),
            });
        };

        if name.is_empty() {
            name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unnamed")
                .to_string();
        }

        Ok(TspInstance {
            name,
            dimension,
            comment,
            edge_weight_type,
            coords: if coords.is_empty() {
                None
            } else {
                Some(coords)
            },
            distances,
            best_known: None,
        })
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
                        let dx = coords[i].0 - coords[j].0;
                        let dy = coords[i].1 - coords[j].1;
                        let r = (dx * dx + dy * dy).sqrt() / 10.0;
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
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_path() -> PathBuf {
        PathBuf::from("test.tsp")
    }

    #[test]
    fn test_parse_simple_tsplib() {
        let content = r#"
NAME: test
TYPE: TSP
COMMENT: A simple test
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 3.0 0.0
3 3.0 4.0
EOF
"#;

        let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

        assert_eq!(instance.name, "test");
        assert_eq!(instance.dimension, 3);
        assert_eq!(instance.comment, Some("A simple test".into()));
        assert!(instance.coords.is_some());
        assert_eq!(instance.coords.as_ref().map(|c| c.len()), Some(3));
    }

    #[test]
    fn test_parse_computes_distances() {
        let content = r#"
NAME: triangle
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 3.0 0.0
3 3.0 4.0
EOF
"#;

        let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

        // 3-4-5 triangle
        assert!((instance.distance(0, 1) - 3.0).abs() < 1e-10);
        assert!((instance.distance(1, 2) - 4.0).abs() < 1e-10);
        assert!((instance.distance(0, 2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_ceil_2d() {
        let content = r#"
NAME: ceil_test
DIMENSION: 2
EDGE_WEIGHT_TYPE: CEIL_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.5 0.0
EOF
"#;

        let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

        // Distance 1.5 should be ceiling'd to 2.0
        assert!((instance.distance(0, 1) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_missing_dimension() {
        let content = r#"
NAME: test
NODE_COORD_SECTION
1 0.0 0.0
EOF
"#;

        let result = TsplibParser::parse(content, &test_path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("DIMENSION"));
    }

    #[test]
    fn test_parse_no_coords_or_weights() {
        let content = r#"
NAME: test
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
EOF
"#;

        let result = TsplibParser::parse(content, &test_path());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_coordinate() {
        let content = r#"
NAME: test
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 abc 0.0
EOF
"#;

        let result = TsplibParser::parse(content, &test_path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Invalid x coordinate"));
    }

    #[test]
    fn test_parse_explicit_weights() {
        let content = r#"
NAME: explicit_test
DIMENSION: 3
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_SECTION
10
20 30
EOF
"#;

        let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

        // Lower triangular: [1,0]=10, [2,0]=20, [2,1]=30
        assert!((instance.distance(1, 0) - 10.0).abs() < 1e-10);
        assert!((instance.distance(2, 0) - 20.0).abs() < 1e-10);
        assert!((instance.distance(2, 1) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_unsupported_edge_type() {
        let content = r#"
NAME: test
DIMENSION: 3
EDGE_WEIGHT_TYPE: UNKNOWN_TYPE
NODE_COORD_SECTION
1 0.0 0.0
EOF
"#;

        let result = TsplibParser::parse(content, &test_path());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unsupported edge weight type"));
    }

    #[test]
    fn test_att_distance() {
        let content = r#"
NAME: att_test
DIMENSION: 2
EDGE_WEIGHT_TYPE: ATT
NODE_COORD_SECTION
1 0.0 0.0
2 100.0 0.0
EOF
"#;

        let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

        // ATT distance should be different from Euclidean
        let dist = instance.distance(0, 1);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_name_defaults_to_filename() {
        let content = r#"
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.0 0.0
EOF
"#;

        let instance =
            TsplibParser::parse(content, &PathBuf::from("my_instance.tsp")).expect("should parse");
        assert_eq!(instance.name, "my_instance");
    }
}
