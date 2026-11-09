//! More unit tests for `tsp_instance` (split out of tsp_instance.rs for the pre-commit complexity limit, #6025; code unchanged).

use super::*;

// =========================================================================
// Error display tests
// =========================================================================

#[test]
fn test_error_display_parse() {
    let err = TspInstanceError::ParseError("test error".to_string());
    assert!(err.to_string().contains("YAML parse error"));
}

#[test]
fn test_error_display_matrix_dimension() {
    let err = TspInstanceError::MatrixDimensionMismatch {
        expected: 6,
        got_rows: 4,
    };
    let msg = err.to_string();
    assert!(msg.contains("6x6"));
    assert!(msg.contains("4 rows"));
}

#[test]
fn test_error_display_matrix_row() {
    let err = TspInstanceError::MatrixRowMismatch {
        row: 2,
        expected: 6,
        got: 4,
    };
    let msg = err.to_string();
    assert!(msg.contains("row 2"));
    assert!(msg.contains("4 columns"));
}

#[test]
fn test_error_display_triangle() {
    let err = TspInstanceError::TriangleInequalityViolation { i: 0, j: 1, k: 2 };
    assert!(err.to_string().contains("Triangle inequality"));
}

#[test]
fn test_error_display_asymmetric() {
    let err = TspInstanceError::AsymmetricMatrix {
        i: 0,
        j: 1,
        forward: 10,
        backward: 20,
    };
    let msg = err.to_string();
    assert!(msg.contains("Asymmetric"));
    assert!(msg.contains("10"));
    assert!(msg.contains("20"));
}

#[test]
fn test_error_display_invalid_city() {
    let err = TspInstanceError::InvalidCityId { id: 10, max: 5 };
    assert!(err.to_string().contains("Invalid city ID"));
}

#[test]
fn test_error_display_io() {
    let err = TspInstanceError::IoError("file not found".to_string());
    assert!(err.to_string().contains("IO error"));
}

// =========================================================================
// File loading tests
// =========================================================================

#[test]
fn test_from_yaml_file_not_found() {
    let result = TspInstanceYaml::from_yaml_file("/nonexistent/path/file.yaml");
    assert!(matches!(result, Err(TspInstanceError::IoError(_))));
}

#[test]
fn test_from_yaml_file_success() {
    // Use the actual bay_area_tsp.yaml file (now 20-city California instance)
    let result = TspInstanceYaml::from_yaml_file("examples/experiments/bay_area_tsp.yaml");
    assert!(result.is_ok());
    let instance = result.unwrap();
    assert_eq!(instance.meta.id, "TSP-CA-020");
    assert_eq!(instance.city_count(), 20);
}

// =========================================================================
// Trait implementations
// =========================================================================

#[test]
fn test_coords_clone_and_copy() {
    let coords = Coords {
        lat: 37.0,
        lon: -122.0,
    };
    let cloned = coords.clone();
    let copied = coords;
    assert_eq!(coords, cloned);
    assert_eq!(coords, copied);
}

#[test]
fn test_coords_partial_eq() {
    let c1 = Coords {
        lat: 37.0,
        lon: -122.0,
    };
    let c2 = Coords {
        lat: 37.0,
        lon: -122.0,
    };
    let c3 = Coords {
        lat: 38.0,
        lon: -122.0,
    };
    assert_eq!(c1, c2);
    assert_ne!(c1, c3);
}

#[test]
fn test_tsp_city_clone() {
    let city = TspCity::new(0, "SF", "SF", 37.0, -122.0);
    let cloned = city.clone();
    assert_eq!(city, cloned);
}

#[test]
fn test_tsp_meta_clone() {
    let meta = TspMeta::default();
    let cloned = meta.clone();
    assert_eq!(meta, cloned);
}

#[test]
fn test_tsp_params_clone() {
    let params = TspParams::default();
    let cloned = params.clone();
    assert_eq!(params, cloned);
}

#[test]
fn test_tsp_algorithm_config_clone() {
    let config = TspAlgorithmConfig::default();
    let cloned = config.clone();
    assert_eq!(config, cloned);
}

#[test]
fn test_tsp_instance_yaml_clone() {
    let instance = TspInstanceYaml::from_yaml(MINIMAL_YAML).expect("Parse");
    let cloned = instance.clone();
    assert_eq!(instance, cloned);
}

#[test]
fn test_error_is_error_trait() {
    let err: Box<dyn std::error::Error> =
        Box::new(TspInstanceError::ParseError("test".to_string()));
    assert!(!err.to_string().is_empty());
}

// =========================================================================
// Debug trait tests
// =========================================================================

#[test]
fn test_coords_debug() {
    let coords = Coords {
        lat: 37.0,
        lon: -122.0,
    };
    let debug = format!("{:?}", coords);
    assert!(debug.contains("Coords"));
    assert!(debug.contains("37"));
}

#[test]
fn test_tsp_instance_error_debug() {
    let err = TspInstanceError::ParseError("test".to_string());
    let debug = format!("{:?}", err);
    assert!(debug.contains("ParseError"));
}
