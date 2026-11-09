//! WASM binding tests (split out of tsp_instance.rs for the pre-commit complexity limit, #6025; code unchanged).

use super::wasm::TspWasmInstance;

const BAY_AREA_YAML: &str = include_str!("../../../examples/experiments/bay_area_tsp.yaml");

const MINIMAL_YAML: &str = r#"
meta:
  id: "TEST-001"
  description: "Minimal test instance"
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
  - [10, 0]
"#;

#[test]
fn test_wasm_from_yaml() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML);
    assert!(instance.is_ok());
}

#[test]
fn test_wasm_from_yaml_invalid() {
    let result = TspWasmInstance::from_yaml("invalid yaml: [[[");
    assert!(result.is_err());
}

#[test]
fn test_wasm_to_yaml() {
    let instance = TspWasmInstance::from_yaml(MINIMAL_YAML).expect("parse");
    let yaml = instance.to_yaml();
    assert!(yaml.is_ok());
    assert!(yaml.unwrap().contains("TEST-001"));
}

#[test]
fn test_wasm_validate() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert!(instance.validate().is_ok());
}

#[test]
fn test_wasm_check_symmetry() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert!(instance.check_symmetry().is_ok());
}

#[test]
fn test_wasm_check_triangle_inequality() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    // Real-world driving distances may violate triangle inequality
    // (roads don't always follow straight lines). This is expected.
    let result = instance.check_triangle_inequality();
    // The Bay Area data does violate it (native probe: Err at i=0, j=1, k=18), so the wrapper must report Err
    assert!(
        result.is_err(),
        "wasm check must surface the violation: {result:?}"
    );
}

#[test]
fn test_wasm_city_count() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.city_count(), 20); // 20-city California instance
}

#[test]
fn test_wasm_distance() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.distance(0, 1), 12); // SF to Oakland
}

#[test]
fn test_wasm_tour_length() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    let tour = [0, 1, 4, 5, 2, 3]; // Optimal tour
    assert_eq!(instance.tour_length(&tour), 115);
}

#[test]
fn test_wasm_get_id() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.get_id(), "TSP-CA-020"); // Updated to 20-city California instance
}

#[test]
fn test_wasm_get_description() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert!(instance.get_description().contains("Bay Area"));
}

#[test]
fn test_wasm_get_units() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.get_units(), "miles");
}

#[test]
fn test_wasm_get_optimal_known() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.get_optimal_known(), None); // Unknown for 20-city instance
}

#[test]
fn test_wasm_get_algorithm_method() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.get_algorithm_method(), "grasp");
}

#[test]
fn test_wasm_get_seed() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.get_seed(), 42);
}

#[test]
fn test_wasm_get_rcl_size() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.get_rcl_size(), 3);
}

#[test]
fn test_wasm_get_restarts() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert_eq!(instance.get_restarts(), 100); // More restarts for 20-city instance
}

#[test]
fn test_wasm_get_two_opt_enabled() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    assert!(instance.get_two_opt_enabled());
}

#[test]
fn test_wasm_get_city_names_json() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    let json = instance.get_city_names_json();
    assert!(json.contains("San Francisco"));
    assert!(json.contains("Oakland"));
}

#[test]
fn test_wasm_get_city_aliases_json() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    let json = instance.get_city_aliases_json();
    assert!(json.contains("SF"));
    assert!(json.contains("OAK"));
}

#[test]
fn test_wasm_get_matrix_json() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    let json = instance.get_matrix_json();
    assert!(json.contains("[0,12")); // First row starts with 0,12
}

#[test]
fn test_wasm_get_city_coords_json() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    let json = instance.get_city_coords_json();
    assert!(json.contains("37.7749")); // SF latitude
}

#[test]
fn test_wasm_tour_length_verified_json() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    let tour = [0, 1, 4, 5, 2, 3];
    let json = instance.tour_length_verified_json(&tour);
    assert!(json.contains("\"length\":115"));
    assert!(json.contains("steps"));
    assert!(json.contains("San Francisco"));
}

#[test]
fn test_wasm_to_json() {
    let instance = TspWasmInstance::from_yaml(BAY_AREA_YAML).expect("parse");
    let json = instance.to_json();
    assert!(json.contains("TSP-CA-020")); // Updated to 20-city California instance
    assert!(json.contains("cities"));
    assert!(json.contains("matrix"));
}

#[test]
fn test_wasm_check_symmetry_fails() {
    let asymmetric_yaml = r#"
meta:
  id: "ASYM"
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
  - [99, 0]
"#;
    let instance = TspWasmInstance::from_yaml(asymmetric_yaml).expect("parse");
    assert!(instance.check_symmetry().is_err());
}

#[test]
fn test_wasm_validate_fails() {
    let bad_yaml = r#"
meta:
  id: "BAD"
  description: "Bad matrix"
cities:
  - id: 0
    name: "A"
    alias: "A"
    coords: { lat: 0.0, lon: 0.0 }
matrix:
  - [0, 10, 20]
  - [10, 0, 30]
  - [20, 30, 0]
"#;
    let instance = TspWasmInstance::from_yaml(bad_yaml).expect("parse");
    assert!(instance.validate().is_err());
}
