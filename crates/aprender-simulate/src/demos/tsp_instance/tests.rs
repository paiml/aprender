//! Unit tests for `tsp_instance` (split out of tsp_instance.rs for the pre-commit complexity limit, #6025; code unchanged).

use super::*;

const MINIMAL_YAML: &str = r#"
meta:
  id: "TEST-001"
  description: "Minimal test instance"
cities:
  - id: 0
    name: "City A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "City B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10, 0]
"#;

const BAY_AREA_YAML: &str = r#"
meta:
  id: "TSP-BAY-006"
  version: "1.0.0"
  description: "6-city Bay Area ground truth instance"
  source: "Google Maps (Dec 2024)"
  units: "miles"
  optimal_known: 115
cities:
  - id: 0
    name: "San Francisco"
    alias: "SF"
    coords: { lat: 37.7749, lon: -122.4194 }
  - id: 1
    name: "Oakland"
    alias: "OAK"
    coords: { lat: 37.8044, lon: -122.2712 }
  - id: 2
    name: "San Jose"
    alias: "SJ"
    coords: { lat: 37.3382, lon: -121.8863 }
  - id: 3
    name: "Palo Alto"
    alias: "PA"
    coords: { lat: 37.4419, lon: -122.1430 }
  - id: 4
    name: "Berkeley"
    alias: "BRK"
    coords: { lat: 37.8716, lon: -122.2727 }
  - id: 5
    name: "Fremont"
    alias: "FRE"
    coords: { lat: 37.5485, lon: -121.9886 }
matrix:
  - [ 0, 12, 48, 35, 14, 42]
  - [12,  0, 42, 30,  4, 30]
  - [48, 42,  0, 15, 46, 17]
  - [35, 30, 15,  0, 32, 18]
  - [14,  4, 46, 32,  0, 32]
  - [42, 30, 17, 18, 32,  0]
algorithm:
  method: "grasp"
  params:
    rcl_size: 3
    restarts: 10
    two_opt: true
    seed: 42
"#;

// =========================================================================
// OR-001-01: TspInstanceYaml struct + serde tests
// =========================================================================

#[test]
fn test_deserialize_valid_yaml() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Should parse valid YAML");
    assert_eq!(instance.meta.id, "TSP-BAY-006");
    assert_eq!(instance.cities.len(), 6);
    assert_eq!(instance.matrix.len(), 6);
    assert_eq!(instance.algorithm.method, "grasp");
}

#[test]
fn test_deserialize_minimal_yaml() {
    let instance = TspInstanceYaml::from_yaml(MINIMAL_YAML).expect("Should parse minimal YAML");
    assert_eq!(instance.meta.id, "TEST-001");
    assert_eq!(instance.cities.len(), 2);
    assert_eq!(instance.matrix.len(), 2);
    // Defaults should apply
    assert_eq!(instance.algorithm.method, "grasp");
    assert_eq!(instance.algorithm.params.rcl_size, 3);
}

#[test]
fn test_deserialize_invalid_yaml() {
    let invalid = "this is not valid yaml: [[[";
    let result = TspInstanceYaml::from_yaml(invalid);
    assert!(result.is_err());
    if let Err(TspInstanceError::ParseError(msg)) = result {
        assert!(!msg.is_empty());
    } else {
        panic!("Expected ParseError");
    }
}

#[test]
fn test_deserialize_missing_required_fields() {
    let missing_cities = r#"
meta:
  id: "TEST"
  description: "No cities"
matrix: []
"#;
    let result = TspInstanceYaml::from_yaml(missing_cities);
    assert!(result.is_err());
}

#[test]
fn test_serialize_roundtrip() {
    let original = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
    let yaml = original.to_yaml().expect("Serialize");
    let restored = TspInstanceYaml::from_yaml(&yaml).expect("Reparse");
    assert_eq!(original.meta.id, restored.meta.id);
    assert_eq!(original.cities.len(), restored.cities.len());
    assert_eq!(original.matrix, restored.matrix);
}

#[test]
fn test_default_algorithm_params() {
    let params = TspParams::default();
    assert_eq!(params.rcl_size, 3);
    assert_eq!(params.restarts, 10);
    assert!(params.two_opt);
    assert_eq!(params.seed, 42);
}

#[test]
fn test_default_meta() {
    let meta = TspMeta::default();
    assert_eq!(meta.id, "TSP-UNNAMED");
    assert_eq!(meta.version, "1.0.0");
    assert_eq!(meta.units, "miles");
    assert!(meta.optimal_known.is_none());
}

#[test]
fn test_default_algorithm_config() {
    let config = TspAlgorithmConfig::default();
    assert_eq!(config.method, "grasp");
    assert_eq!(config.params.rcl_size, 3);
}

#[test]
fn test_coords_default() {
    let coords = Coords::default();
    assert_eq!(coords.lat, 0.0);
    assert_eq!(coords.lon, 0.0);
}

#[test]
fn test_tsp_city_new() {
    let city = TspCity::new(0, "San Francisco", "SF", 37.7749, -122.4194);
    assert_eq!(city.id, 0);
    assert_eq!(city.name, "San Francisco");
    assert_eq!(city.alias, "SF");
    assert!((city.coords.lat - 37.7749).abs() < 0.0001);
}

// =========================================================================
// Validation tests (Jidoka)
// =========================================================================

#[test]
fn test_validate_valid_instance() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
    assert!(instance.validate().is_ok());
}

#[test]
fn test_validate_matrix_dimension_mismatch() {
    let yaml = r#"
meta:
  id: "TEST"
  description: "Bad matrix"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10, 20]
  - [10, 0, 30]
  - [20, 30, 0]
"#;
    let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
    let result = instance.validate();
    assert!(matches!(
        result,
        Err(TspInstanceError::MatrixDimensionMismatch { .. })
    ));
}

#[test]
fn test_validate_matrix_row_mismatch() {
    let yaml = r#"
meta:
  id: "TEST"
  description: "Bad row"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10]
"#;
    let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
    let result = instance.validate();
    assert!(matches!(
        result,
        Err(TspInstanceError::MatrixRowMismatch { .. })
    ));
}

#[test]
fn test_validate_invalid_city_id() {
    let yaml = r#"
meta:
  id: "TEST"
  description: "Bad city ID"
cities:
  - id: 5
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [10, 0]
"#;
    let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
    let result = instance.validate();
    assert!(matches!(
        result,
        Err(TspInstanceError::InvalidCityId { .. })
    ));
}

#[test]
fn test_check_symmetry_valid() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
    assert!(instance.check_symmetry().is_ok());
}

#[test]
fn test_check_symmetry_invalid() {
    let yaml = r#"
meta:
  id: "TEST"
  description: "Asymmetric"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
matrix:
  - [0, 10]
  - [20, 0]
"#;
    let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
    let result = instance.check_symmetry();
    assert!(matches!(
        result,
        Err(TspInstanceError::AsymmetricMatrix { .. })
    ));
}

#[test]
fn test_check_triangle_inequality_valid() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
    assert!(instance.check_triangle_inequality().is_ok());
}

#[test]
fn test_check_triangle_inequality_violation() {
    // d(0,2) = 100 > d(0,1) + d(1,2) = 10 + 10 = 20
    let yaml = r#"
meta:
  id: "TEST"
  description: "Triangle violation"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
  - id: 1
    name: "B"
    alias: "B"
    coords: { lat: 1.0, lon: 1.0 }
  - id: 2
    name: "C"
    alias: "C"
    coords: { lat: 2.0, lon: 2.0 }
matrix:
  - [0, 10, 100]
  - [10, 0, 10]
  - [100, 10, 0]
"#;
    let instance = TspInstanceYaml::from_yaml(yaml).expect("Parse");
    let result = instance.check_triangle_inequality();
    assert!(matches!(
        result,
        Err(TspInstanceError::TriangleInequalityViolation { .. })
    ));
}

// =========================================================================
// Tour length tests
// =========================================================================

#[test]
fn test_tour_length_bay_area_optimal() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
    // Optimal tour: SF(0) → OAK(1) → BRK(4) → FRE(5) → SJ(2) → PA(3) → SF(0)
    let tour = vec![0, 1, 4, 5, 2, 3];
    let length = instance.tour_length(&tour);
    // 12 + 4 + 32 + 17 + 15 + 35 = 115
    assert_eq!(length, 115);
}

#[test]
fn test_tour_length_empty() {
    let instance = TspInstanceYaml::from_yaml(MINIMAL_YAML).expect("Parse");
    let length = instance.tour_length(&[]);
    assert_eq!(length, 0);
}

#[test]
fn test_tour_length_single_city() {
    let instance = TspInstanceYaml::from_yaml(MINIMAL_YAML).expect("Parse");
    let length = instance.tour_length(&[0]);
    assert_eq!(length, 0); // d(0,0) = 0
}

#[test]
fn test_tour_length_verified() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
    let tour = vec![0, 1, 4, 5, 2, 3];
    let (length, steps) = instance.tour_length_verified(&tour);
    assert_eq!(length, 115);
    assert_eq!(steps.len(), 6);
    assert!(steps[0].contains("San Francisco"));
    assert!(steps[0].contains("Oakland"));
    assert!(steps[0].contains("12"));
}

// =========================================================================
// Accessor tests
// =========================================================================

#[test]
fn test_city_count() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
    assert_eq!(instance.city_count(), 6);
}

#[test]
fn test_distance() {
    let instance = TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("Parse");
    assert_eq!(instance.distance(0, 1), 12); // SF to Oakland
    assert_eq!(instance.distance(1, 4), 4); // Oakland to Berkeley
}

mod more;
