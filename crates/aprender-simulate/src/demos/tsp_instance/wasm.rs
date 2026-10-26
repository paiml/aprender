//! WASM bindings for TSP instances (split out of tsp_instance.rs for the pre-commit complexity limit, #6025; code unchanged).

use super::TspInstanceYaml;
use wasm_bindgen::prelude::*;

/// WASM-exported TSP instance for JavaScript/TypeScript.
///
/// Provides YAML-first configuration for TSP instances accessible from
/// web browsers via WebAssembly.
///
/// # JavaScript Example
///
/// ```javascript
/// import { TspWasmInstance } from 'simular';
///
/// const yaml = `
/// meta:
///   id: "MY-TSP"
///   description: "Custom instance"
/// cities:
///   - id: 0
///     name: "A"
///     alias: "A"
///     coords: { lat: 0.0, lon: 0.0 }
/// matrix:
///   - [0]
/// `;
///
/// const instance = TspWasmInstance.from_yaml(yaml);
/// console.log(instance.city_count());
/// ```
#[wasm_bindgen]
pub struct TspWasmInstance {
    inner: TspInstanceYaml,
}

// WASM exports don't need #[must_use] - values returned to JS
#[allow(clippy::must_use_candidate)]
#[wasm_bindgen]
impl TspWasmInstance {
    /// Parse a TSP instance from YAML string.
    ///
    /// # Errors
    ///
    /// Returns error string if YAML is invalid or fails validation.
    #[wasm_bindgen(js_name = fromYaml)]
    pub fn from_yaml(yaml: &str) -> Result<Self, String> {
        let inner = TspInstanceYaml::from_yaml(yaml).map_err(|e| e.to_string())?;
        Ok(Self { inner })
    }

    /// Serialize the instance back to YAML.
    #[wasm_bindgen(js_name = toYaml)]
    pub fn to_yaml(&self) -> Result<String, String> {
        self.inner.to_yaml().map_err(|e| e.to_string())
    }

    /// Validate the instance (Jidoka).
    ///
    /// Checks matrix dimensions, city IDs, symmetry, and triangle inequality.
    #[wasm_bindgen]
    pub fn validate(&self) -> Result<(), String> {
        self.inner.validate().map_err(|e| e.to_string())
    }

    /// Check matrix symmetry.
    #[wasm_bindgen(js_name = checkSymmetry)]
    pub fn check_symmetry(&self) -> Result<(), String> {
        self.inner.check_symmetry().map_err(|e| e.to_string())
    }

    /// Check triangle inequality.
    #[wasm_bindgen(js_name = checkTriangleInequality)]
    pub fn check_triangle_inequality(&self) -> Result<(), String> {
        self.inner
            .check_triangle_inequality()
            .map_err(|e| e.to_string())
    }

    /// Get the number of cities.
    #[wasm_bindgen(js_name = cityCount)]
    pub fn city_count(&self) -> usize {
        self.inner.city_count()
    }

    /// Get distance between two cities.
    #[wasm_bindgen]
    pub fn distance(&self, from: usize, to: usize) -> u32 {
        self.inner.distance(from, to)
    }

    /// Compute tour length for a given tour (city indices).
    ///
    /// Tour is specified as a JavaScript array of city indices.
    #[wasm_bindgen(js_name = tourLength)]
    pub fn tour_length(&self, tour: &[usize]) -> u32 {
        self.inner.tour_length(tour)
    }

    /// Get instance ID.
    #[wasm_bindgen(js_name = getId)]
    pub fn get_id(&self) -> String {
        self.inner.meta.id.clone()
    }

    /// Get instance description.
    #[wasm_bindgen(js_name = getDescription)]
    pub fn get_description(&self) -> String {
        self.inner.meta.description.clone()
    }

    /// Get units (e.g., "miles", "km").
    #[wasm_bindgen(js_name = getUnits)]
    pub fn get_units(&self) -> String {
        self.inner.meta.units.clone()
    }

    /// Get known optimal value (if any).
    #[wasm_bindgen(js_name = getOptimalKnown)]
    pub fn get_optimal_known(&self) -> Option<u32> {
        self.inner.meta.optimal_known
    }

    /// Get algorithm method.
    #[wasm_bindgen(js_name = getAlgorithmMethod)]
    pub fn get_algorithm_method(&self) -> String {
        self.inner.algorithm.method.clone()
    }

    /// Get algorithm seed.
    #[wasm_bindgen(js_name = getSeed)]
    pub fn get_seed(&self) -> u64 {
        self.inner.algorithm.params.seed
    }

    /// Get RCL size parameter.
    #[wasm_bindgen(js_name = getRclSize)]
    pub fn get_rcl_size(&self) -> usize {
        self.inner.algorithm.params.rcl_size
    }

    /// Get number of restarts.
    #[wasm_bindgen(js_name = getRestarts)]
    pub fn get_restarts(&self) -> usize {
        self.inner.algorithm.params.restarts
    }

    /// Check if 2-opt is enabled.
    #[wasm_bindgen(js_name = getTwoOptEnabled)]
    pub fn get_two_opt_enabled(&self) -> bool {
        self.inner.algorithm.params.two_opt
    }

    /// Get city names as JSON array.
    #[wasm_bindgen(js_name = getCityNamesJson)]
    pub fn get_city_names_json(&self) -> String {
        let names: Vec<&str> = self.inner.cities.iter().map(|c| c.name.as_str()).collect();
        serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get city aliases as JSON array.
    #[wasm_bindgen(js_name = getCityAliasesJson)]
    pub fn get_city_aliases_json(&self) -> String {
        let aliases: Vec<&str> = self.inner.cities.iter().map(|c| c.alias.as_str()).collect();
        serde_json::to_string(&aliases).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get distance matrix as JSON (2D array).
    #[wasm_bindgen(js_name = getMatrixJson)]
    pub fn get_matrix_json(&self) -> String {
        serde_json::to_string(&self.inner.matrix).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get city coordinates as JSON array of {lat, lon} objects.
    #[wasm_bindgen(js_name = getCityCoordsJson)]
    pub fn get_city_coords_json(&self) -> String {
        let coords: Vec<_> = self.inner.cities.iter().map(|c| &c.coords).collect();
        serde_json::to_string(&coords).unwrap_or_else(|_| "[]".to_string())
    }

    /// Compute tour length with step-by-step verification.
    ///
    /// Returns JSON: `{"length": 115, "steps": ["Step 1: SF → OAK = 12 miles (total: 12)", ...]}`
    #[wasm_bindgen(js_name = tourLengthVerifiedJson)]
    pub fn tour_length_verified_json(&self, tour: &[usize]) -> String {
        let (length, steps) = self.inner.tour_length_verified(tour);
        serde_json::json!({
            "length": length,
            "steps": steps
        })
        .to_string()
    }

    /// Get full instance as JSON.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).unwrap_or_else(|_| "{}".to_string())
    }
}
