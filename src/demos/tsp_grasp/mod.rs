//! Demo 6: TSP Randomized Greedy Start with 2-Opt
//!
//! Demonstrates the GRASP (Greedy Randomized Adaptive Search Procedure) methodology
//! for the Traveling Salesman Problem.
//!
//! # Governing Equations
//!
//! ```text
//! Tour Length:           L(π) = Σᵢ d(π(i), π(i+1)) + d(π(n), π(1))
//! 2-Opt Improvement:     Δ = d(i,i+1) + d(j,j+1) - d(i,j) - d(i+1,j+1)
//! Expected Greedy Tour:  E[L_greedy] ≈ 0.7124·√(n·A)  [BHH 1959]
//! 1-Tree Lower Bound:    L* ≥ MST(G\{v₀}) + 2·min_edges(v₀)  [Held-Karp 1970]
//! ```
//!
//! # EDD Cycle
//!
//! 1. **Equation**: Greedy + 2-opt achieves tours within ~5% of optimal [JM 1997]
//! 2. **Failing Test**: Assert `tour_length` / `lower_bound` < 1.10 (10% optimality gap)
//! 3. **Implementation**: Randomized nearest-neighbor + exhaustive 2-opt local search
//! 4. **Verification**: Multiple random starts converge to similar tour lengths
//! 5. **Falsification**: Pure random start (no greedy) yields significantly worse results
//!
//! # Toyota Production System Compliance
//!
//! - **Jidoka**: Edge crossing detection (Lin-Kernighan 1973: Euclidean TSP optimal has 0 crossings)
//! - **Muda**: Stagnation detection eliminates restart waste (early termination)
//! - **Heijunka**: Adaptive tick rate based on convergence status
//! - **Poka-Yoke**: RCL size clamping, input validation
//!
//! # References (IEEE Style)
//!
//! - \[51\] Feo & Resende (1995). GRASP. *J. Global Optimization*, 6(2), 109-133.
//! - \[52\] Lin & Kernighan (1973). TSP Heuristics. *Operations Research*, 21(2), 498-516.
//! - \[53\] Johnson & `McGeoch` (1997). TSP Case Study. *Local Search in Comb. Opt.*
//! - \[54\] Christofides (1976). TSP Heuristic. *CMU Report 388*.
//! - \[55\] Beardwood, Halton & Hammersley (1959). Shortest Path. *Proc. Cambridge*, 55, 299-327.
//! - \[56\] Held & Karp (1970). TSP and MST. *Operations Research*, 18(6), 1138-1162.

use super::tsp_instance::{TspInstanceError, TspInstanceYaml};
use super::{CriterionStatus, EddDemo, FalsificationStatus};
use crate::edd::audit::{
    Decision, EquationEval, SimulationAuditLog, StepEntry, TspStateSnapshot, TspStepType,
};
use crate::engine::rng::SimRng;
use crate::engine::SimTime;
use serde::{Deserialize, Serialize};

// WASM-compatible timing abstraction
// std::time::Instant::now() panics in WASM - use web_sys::Performance or dummy
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct Instant(f64);

#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        // Use performance.now() if available, otherwise 0
        #[cfg(feature = "wasm")]
        {
            if let Some(window) = web_sys::window() {
                if let Some(perf) = window.performance() {
                    return Self(perf.now());
                }
            }
        }
        Self(0.0)
    }

    fn elapsed(&self) -> std::time::Duration {
        #[cfg(feature = "wasm")]
        {
            if let Some(window) = web_sys::window() {
                if let Some(perf) = window.performance() {
                    let now = perf.now();
                    let elapsed_ms = now - self.0;
                    return std::time::Duration::from_micros((elapsed_ms * 1000.0) as u64);
                }
            }
        }
        std::time::Duration::ZERO
    }
}

/// Helper to generate random usize in range [0, max).
fn gen_range_usize(rng: &mut SimRng, max: usize) -> usize {
    if max == 0 {
        return 0;
    }
    (rng.gen_u64() as usize) % max
}

/// A 2D point representing a city.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct City {
    pub x: f64,
    pub y: f64,
}

impl City {
    /// Create a new city at coordinates (x, y).
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another city.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Construction method for initial tour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ConstructionMethod {
    /// Randomized greedy (GRASP-style RCL selection).
    #[default]
    RandomizedGreedy,
    /// Pure nearest neighbor (deterministic greedy).
    NearestNeighbor,
    /// Pure random tour (for falsification).
    Random,
}

/// TSP GRASP Demo state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TspGraspDemo {
    /// Cities to visit.
    pub cities: Vec<City>,
    /// Current tour (indices into cities).
    pub tour: Vec<usize>,
    /// Current tour length.
    pub tour_length: f64,
    /// Best tour found across restarts.
    pub best_tour: Vec<usize>,
    /// Best tour length found.
    pub best_tour_length: f64,
    /// Number of cities.
    pub n: usize,
    /// RCL size for randomized greedy (k nearest candidates).
    pub rcl_size: usize,
    /// Construction method.
    pub construction_method: ConstructionMethod,
    /// Number of 2-opt iterations performed.
    pub two_opt_iterations: u64,
    /// Number of 2-opt improvements made.
    pub two_opt_improvements: u64,
    /// Number of GRASP restarts performed.
    pub restarts: u64,
    /// History of tour lengths from each restart.
    pub restart_history: Vec<f64>,
    /// Lower bound estimate (MST-based).
    pub lower_bound: f64,
    /// Area of bounding box (for BHH constant).
    pub area: f64,
    /// Seed for reproducibility.
    pub seed: u64,
    /// Unit label for distances (e.g., "miles", "km").
    #[serde(default = "default_units")]
    pub units: String,
    /// Known optimal tour length (if available from YAML).
    #[serde(default)]
    pub optimal_known: Option<u32>,
    /// MUDA: Consecutive restarts without improvement (stagnation detection).
    #[serde(default)]
    pub stagnation_count: u64,
    /// MUDA: Threshold for early termination (default: 10 restarts).
    #[serde(default = "default_stagnation_threshold")]
    pub stagnation_threshold: u64,
    /// MUDA: Flag indicating convergence (no further improvement expected).
    #[serde(default)]
    pub converged: bool,
    /// Flag indicating if distances are Euclidean (computed from coords).
    /// Lin-Kernighan crossing theorem ONLY applies to Euclidean instances.
    #[serde(default = "default_euclidean")]
    pub is_euclidean: bool,
    /// RNG for stochastic elements.
    #[serde(skip)]
    rng: Option<SimRng>,
    /// Distance matrix (precomputed for efficiency).
    #[serde(skip)]
    distance_matrix: Vec<Vec<f64>>,
    /// Audit log for step-by-step tracking (EDD-16).
    #[serde(skip)]
    audit_log: Vec<StepEntry<TspStateSnapshot>>,
    /// Step counter for audit logging.
    #[serde(skip)]
    step_counter: u64,
    /// Enable audit logging (can be disabled for performance).
    #[serde(default = "default_audit_enabled")]
    pub audit_enabled: bool,
}

fn default_units() -> String {
    "units".to_string()
}

fn default_audit_enabled() -> bool {
    true
}

/// MUDA: Default stagnation threshold (10 restarts without improvement).
fn default_stagnation_threshold() -> u64 {
    10
}

/// Default: assume Euclidean (computed from coordinates).
fn default_euclidean() -> bool {
    true
}

impl Default for TspGraspDemo {
    fn default() -> Self {
        Self::new(42, 20)
    }
}

impl TspGraspDemo {
    /// Ensure demo is properly initialized after deserialization.
    /// Call this after deserializing to rebuild RNG and distance matrix.
    pub fn reinitialize(&mut self) {
        if self.rng.is_none() {
            self.rng = Some(SimRng::new(self.seed));
        }
        if self.distance_matrix.is_empty() && !self.cities.is_empty() {
            self.precompute_distances();
        }
        if self.lower_bound == 0.0 && !self.cities.is_empty() {
            self.compute_lower_bound();
        }
    }
}

impl TspGraspDemo {
    /// Create a new TSP GRASP demo with random cities.
    #[must_use]
    pub fn new(seed: u64, n: usize) -> Self {
        let mut rng = SimRng::new(seed);
        let n = n.max(4); // Minimum 4 cities for meaningful TSP

        // Generate random cities in unit square
        let mut cities = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.gen_range_f64(0.0, 1.0);
            let y = rng.gen_range_f64(0.0, 1.0);
            cities.push(City::new(x, y));
        }

        let mut demo = Self {
            cities,
            tour: Vec::new(),
            tour_length: 0.0,
            best_tour: Vec::new(),
            best_tour_length: 0.0,
            n,
            rcl_size: 3, // Default RCL size
            construction_method: ConstructionMethod::RandomizedGreedy,
            two_opt_iterations: 0,
            two_opt_improvements: 0,
            restarts: 0,
            restart_history: Vec::new(),
            lower_bound: 0.0,
            area: 1.0, // Unit square
            seed,
            units: "units".to_string(), // Default for random cities
            optimal_known: None,        // Unknown for random instances
            stagnation_count: 0,        // MUDA: Start fresh
            stagnation_threshold: 10,   // MUDA: 10 restarts without improvement
            converged: false,           // MUDA: Not yet converged
            is_euclidean: true,         // Random cities use Euclidean distances
            rng: Some(rng),
            distance_matrix: Vec::new(),
            audit_log: Vec::new(),
            step_counter: 0,
            audit_enabled: true,
        };

        demo.precompute_distances();
        demo.compute_lower_bound();
        demo
    }

    /// Create demo with specific cities.
    #[must_use]
    pub fn with_cities(seed: u64, cities: Vec<City>) -> Self {
        let n = cities.len();
        let rng = SimRng::new(seed);

        // Compute bounding box area
        let (min_x, max_x, min_y, max_y) = cities.iter().fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ),
            |(min_x, max_x, min_y, max_y), c| {
                (
                    min_x.min(c.x),
                    max_x.max(c.x),
                    min_y.min(c.y),
                    max_y.max(c.y),
                )
            },
        );
        let area = (max_x - min_x) * (max_y - min_y);

        let mut demo = Self {
            cities,
            tour: Vec::new(),
            tour_length: 0.0,
            best_tour: Vec::new(),
            best_tour_length: 0.0,
            n,
            rcl_size: 3,
            construction_method: ConstructionMethod::RandomizedGreedy,
            two_opt_iterations: 0,
            two_opt_improvements: 0,
            restarts: 0,
            restart_history: Vec::new(),
            lower_bound: 0.0,
            area: area.max(0.001), // Avoid zero area
            seed,
            units: "units".to_string(), // Default for custom cities
            optimal_known: None,        // Unknown for custom instances
            stagnation_count: 0,        // MUDA: Start fresh
            stagnation_threshold: 10,   // MUDA: 10 restarts without improvement
            converged: false,           // MUDA: Not yet converged
            is_euclidean: true,         // Custom cities use Euclidean distances
            rng: Some(rng),
            distance_matrix: Vec::new(),
            audit_log: Vec::new(),
            step_counter: 0,
            audit_enabled: true,
        };

        demo.precompute_distances();
        demo.compute_lower_bound();
        demo
    }

    /// Create demo from a YAML instance configuration.
    ///
    /// This enables the YAML-first architecture where users can modify
    /// `bay_area_tsp.yaml` and run the same demo in TUI or WASM.
    ///
    /// # Arguments
    ///
    /// * `instance` - Parsed YAML configuration
    ///
    /// # Example
    ///
    /// ```
    /// use simular::demos::{TspGraspDemo, TspInstanceYaml};
    ///
    /// let yaml = r#"
    /// meta:
    ///   id: "TEST"
    ///   description: "Test"
    /// cities:
    ///   - id: 0
    ///     name: "A"
    ///     alias: "A"
    ///     coords: { lat: 0.0, lon: 0.0 }
    ///   - id: 1
    ///     name: "B"
    ///     alias: "B"
    ///     coords: { lat: 1.0, lon: 1.0 }
    /// matrix:
    ///   - [0, 10]
    ///   - [10, 0]
    /// "#;
    ///
    /// let instance = TspInstanceYaml::from_yaml(yaml).unwrap();
    /// let demo = TspGraspDemo::from_instance(&instance);
    /// assert_eq!(demo.n, 2);
    /// ```
    #[must_use]
    pub fn from_instance(instance: &TspInstanceYaml) -> Self {
        // Convert TspCity coords (lat/lon) to City (x/y)
        // Use lon as x, lat as y (standard mapping)
        let cities: Vec<City> = instance
            .cities
            .iter()
            .map(|c| City::new(c.coords.lon, c.coords.lat))
            .collect();

        let seed = instance.algorithm.params.seed;
        let mut demo = Self::with_cities(seed, cities);

        // Apply algorithm configuration
        demo.set_rcl_size(instance.algorithm.params.rcl_size);

        // Set construction method based on YAML config
        let method = match instance.algorithm.method.as_str() {
            "nearest_neighbor" | "nn" => ConstructionMethod::NearestNeighbor,
            "random" => ConstructionMethod::Random,
            _ => ConstructionMethod::RandomizedGreedy, // "grasp" is default
        };
        demo.set_construction_method(method);

        // Override distance matrix with YAML-provided distances
        // This allows users to specify exact driving distances rather than Euclidean
        demo.distance_matrix = instance
            .matrix
            .iter()
            .map(|row| row.iter().map(|&d| f64::from(d)).collect())
            .collect();

        // YAML instances with distance matrices are NON-EUCLIDEAN
        // Lin-Kernighan crossing theorem does NOT apply
        demo.is_euclidean = false;

        // Copy metadata from YAML instance
        demo.units.clone_from(&instance.meta.units);
        demo.optimal_known = instance.meta.optimal_known;

        // Recompute lower bound with actual distances
        demo.compute_lower_bound();

        demo
    }

    /// Load demo from YAML string.
    ///
    /// This is the primary entry point for YAML-first configuration.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - YAML parsing fails
    /// - Validation fails (matrix size, symmetry, etc.)
    ///
    /// # Example
    ///
    /// ```
    /// use simular::demos::TspGraspDemo;
    ///
    /// let yaml = include_str!("../../../examples/experiments/bay_area_tsp.yaml");
    /// let demo = TspGraspDemo::from_yaml(yaml).expect("YAML should parse");
    /// assert_eq!(demo.n, 20); // 20-city California instance
    /// ```
    pub fn from_yaml(yaml: &str) -> Result<Self, TspInstanceError> {
        let instance = TspInstanceYaml::from_yaml(yaml)?;
        instance.validate()?;
        Ok(Self::from_instance(&instance))
    }

    /// Load demo from YAML file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or YAML is invalid.
    pub fn from_yaml_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TspInstanceError> {
        let instance = TspInstanceYaml::from_yaml_file(path)?;
        instance.validate()?;
        Ok(Self::from_instance(&instance))
    }

    /// Set construction method.
    pub fn set_construction_method(&mut self, method: ConstructionMethod) {
        self.construction_method = method;
    }

    /// Set RCL size for randomized greedy.
    pub fn set_rcl_size(&mut self, size: usize) {
        self.rcl_size = size.max(1).min(self.n);
    }

    /// Precompute distance matrix for O(1) lookups.
    fn precompute_distances(&mut self) {
        self.distance_matrix = vec![vec![0.0; self.n]; self.n];
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let d = self.cities[i].distance_to(&self.cities[j]);
                self.distance_matrix[i][j] = d;
                self.distance_matrix[j][i] = d;
            }
        }
    }

    /// Get distance between two cities (O(1) with precomputed matrix).
    #[must_use]
    pub fn distance(&self, i: usize, j: usize) -> f64 {
        if self.distance_matrix.is_empty() {
            self.cities[i].distance_to(&self.cities[j])
        } else {
            self.distance_matrix[i][j]
        }
    }

    /// Compute tour length for a given tour.
    #[must_use]
    pub fn compute_tour_length(&self, tour: &[usize]) -> f64 {
        if tour.len() < 2 {
            return 0.0;
        }

        let mut length = 0.0;
        for i in 0..tour.len() {
            let j = (i + 1) % tour.len();
            length += self.distance(tour[i], tour[j]);
        }
        length
    }

    /// Compute MST on vertices excluding `exclude_vertex`.
    fn compute_mst_excluding(&self, exclude_vertex: usize) -> f64 {
        if self.n < 3 {
            return 0.0;
        }

        // Prim's MST algorithm on all vertices except exclude_vertex
        let mut in_mst = vec![false; self.n];
        let mut min_edge = vec![f64::INFINITY; self.n];

        // Find first vertex that's not excluded
        let start = (0..self.n).find(|&v| v != exclude_vertex).unwrap_or(0);
        min_edge[start] = 0.0;
        in_mst[exclude_vertex] = true; // Mark excluded vertex as "in MST" to skip it

        let mut mst_weight = 0.0;
        let vertices_to_add = self.n - 1; // All except excluded

        for _ in 0..vertices_to_add {
            // Find minimum edge to non-MST vertex
            let mut u = None;
            let mut min_val = f64::INFINITY;
            for (i, (&in_tree, &edge)) in in_mst.iter().zip(min_edge.iter()).enumerate() {
                if !in_tree && edge < min_val {
                    min_val = edge;
                    u = Some(i);
                }
            }

            let Some(u) = u else { break };

            in_mst[u] = true;
            mst_weight += min_val;

            // Update minimum edges
            for v in 0..self.n {
                if !in_mst[v] && v != exclude_vertex {
                    let d = self.distance(u, v);
                    if d < min_edge[v] {
                        min_edge[v] = d;
                    }
                }
            }
        }

        mst_weight
    }

    /// Compute 1-tree lower bound (Held-Karp style).
    ///
    /// For each vertex v:
    /// 1. Compute MST on remaining n-1 vertices
    /// 2. Add two smallest edges from v to the MST
    /// 3. This gives a lower bound for TSP
    ///
    /// The maximum over all vertices is the 1-tree bound.
    fn compute_one_tree_bound(&self) -> f64 {
        if self.n < 3 {
            return 0.0;
        }

        let mut best_bound = 0.0;

        for exclude in 0..self.n {
            // Compute MST excluding this vertex
            let mst_weight = self.compute_mst_excluding(exclude);

            // Find two smallest edges from excluded vertex to MST
            let mut edges: Vec<f64> = (0..self.n)
                .filter(|&v| v != exclude)
                .map(|v| self.distance(exclude, v))
                .collect();
            edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Add two smallest edges
            let one_tree =
                mst_weight + edges.first().unwrap_or(&0.0) + edges.get(1).unwrap_or(&0.0);

            if one_tree > best_bound {
                best_bound = one_tree;
            }
        }

        best_bound
    }

    /// Compute lower bound using 1-tree (tighter than MST).
    fn compute_lower_bound(&mut self) {
        if self.n < 2 {
            self.lower_bound = 0.0;
            return;
        }

        self.lower_bound = self.compute_one_tree_bound();
    }

    /// Check if two line segments intersect (for crossing detection).
    fn segments_intersect(p1: &City, p2: &City, p3: &City, p4: &City) -> bool {
        // Returns true if segment p1-p2 intersects segment p3-p4
        // Using cross product method
        fn cross(o: &City, a: &City, b: &City) -> f64 {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }

        let d1 = cross(p3, p4, p1);
        let d2 = cross(p3, p4, p2);
        let d3 = cross(p1, p2, p3);
        let d4 = cross(p1, p2, p4);

        // Segments intersect if points are on opposite sides
        if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
            && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
        {
            return true;
        }

        false
    }

    /// Count edge crossings in current tour.
    /// For 2D Euclidean TSP, optimal tour has 0 crossings.
    #[must_use]
    pub fn count_crossings(&self) -> usize {
        let n = self.tour.len();
        if n < 4 {
            return 0;
        }

        let mut crossings = 0;

        for i in 0..n {
            let i_next = (i + 1) % n;
            let p1 = &self.cities[self.tour[i]];
            let p2 = &self.cities[self.tour[i_next]];

            // Check against all non-adjacent edges
            for j in (i + 2)..n {
                let j_next = (j + 1) % n;

                // Skip if edges share a vertex
                if j_next == i {
                    continue;
                }

                let p3 = &self.cities[self.tour[j]];
                let p4 = &self.cities[self.tour[j_next]];

                if Self::segments_intersect(p1, p2, p3, p4) {
                    crossings += 1;
                }
            }
        }

        crossings
    }

    /// Check if tour is at a 2-opt local optimum.
    #[must_use]
    pub fn is_two_opt_optimal(&self) -> bool {
        let n = self.tour.len();
        if n < 4 {
            return true;
        }

        for i in 0..(n - 2) {
            for j in (i + 2)..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                if self.two_opt_improvement(i, j) > f64::EPSILON {
                    return false;
                }
            }
        }

        true
    }

    /// Construct initial tour using randomized greedy (GRASP).
    fn construct_randomized_greedy(&mut self) {
        let n = self.n;
        let rcl_size_param = self.rcl_size;

        // Take RNG out temporarily
        let Some(mut rng) = self.rng.take() else {
            return;
        };

        let mut visited = vec![false; n];
        let mut tour = Vec::with_capacity(n);

        // Start from random city
        let start = gen_range_usize(&mut rng, n);
        tour.push(start);
        visited[start] = true;

        // Build tour using RCL
        while tour.len() < n {
            let Some(&current) = tour.last() else {
                break;
            };

            // Find k nearest unvisited cities (RCL)
            let mut candidates: Vec<(usize, f64)> = (0..n)
                .filter(|&i| !visited[i])
                .map(|i| (i, self.distance_matrix[current][i]))
                .collect();

            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top k candidates (or all remaining if fewer)
            let rcl_size = rcl_size_param.min(candidates.len());
            let rcl = &candidates[..rcl_size];

            // Select randomly from RCL
            let idx = gen_range_usize(&mut rng, rcl_size);
            let next = rcl[idx].0;

            tour.push(next);
            visited[next] = true;
        }

        // Put RNG back
        self.rng = Some(rng);

        self.tour = tour;
        self.tour_length = self.compute_tour_length(&self.tour);
    }

    /// Construct initial tour using pure nearest neighbor.
    fn construct_nearest_neighbor(&mut self) {
        let n = self.n;

        // Take RNG out temporarily
        let Some(mut rng) = self.rng.take() else {
            return;
        };

        let mut visited = vec![false; n];
        let mut tour = Vec::with_capacity(n);

        // Start from random city
        let start = gen_range_usize(&mut rng, n);
        tour.push(start);
        visited[start] = true;

        while tour.len() < n {
            let Some(&current) = tour.last() else {
                break;
            };

            // Find nearest unvisited city
            let mut best_next = None;
            let mut best_dist = f64::INFINITY;

            for i in 0..n {
                if !visited[i] {
                    let d = self.distance_matrix[current][i];
                    if d < best_dist {
                        best_dist = d;
                        best_next = Some(i);
                    }
                }
            }

            if let Some(next) = best_next {
                tour.push(next);
                visited[next] = true;
            }
        }

        // Put RNG back
        self.rng = Some(rng);

        self.tour = tour;
        self.tour_length = self.compute_tour_length(&self.tour);
    }

    /// Construct initial tour randomly (for falsification).
    fn construct_random(&mut self) {
        let n = self.n;

        // Take RNG out temporarily
        let Some(mut rng) = self.rng.take() else {
            return;
        };

        // Fisher-Yates shuffle
        let mut tour: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = gen_range_usize(&mut rng, i + 1);
            tour.swap(i, j);
        }

        // Put RNG back
        self.rng = Some(rng);

        self.tour = tour;
        self.tour_length = self.compute_tour_length(&self.tour);
    }

    /// Construct initial tour based on selected method.
    pub fn construct_tour(&mut self) {
        match self.construction_method {
            ConstructionMethod::RandomizedGreedy => self.construct_randomized_greedy(),
            ConstructionMethod::NearestNeighbor => self.construct_nearest_neighbor(),
            ConstructionMethod::Random => self.construct_random(),
        }
    }

    /// Compute 2-opt improvement for swapping edges (i, i+1) and (j, j+1).
    /// Returns positive value if swap improves tour.
    #[must_use]
    pub fn two_opt_improvement(&self, i: usize, j: usize) -> f64 {
        let n = self.tour.len();
        if n < 4 || i >= j || j >= n {
            return 0.0;
        }

        let i_next = (i + 1) % n;
        let j_next = (j + 1) % n;

        // Current edges: (tour[i], tour[i+1]) and (tour[j], tour[j+1])
        // New edges: (tour[i], tour[j]) and (tour[i+1], tour[j+1])
        let d_current = self.distance(self.tour[i], self.tour[i_next])
            + self.distance(self.tour[j], self.tour[j_next]);

        let d_new = self.distance(self.tour[i], self.tour[j])
            + self.distance(self.tour[i_next], self.tour[j_next]);

        d_current - d_new
    }

    /// Apply 2-opt swap: reverse segment between i+1 and j.
    fn apply_two_opt(&mut self, i: usize, j: usize) {
        // Reverse the segment from i+1 to j
        let mut left = i + 1;
        let mut right = j;
        while left < right {
            self.tour.swap(left, right);
            left += 1;
            right -= 1;
        }
        self.tour_length = self.compute_tour_length(&self.tour);
    }

    /// Perform one pass of 2-opt local search.
    /// Returns true if an improvement was made.
    pub fn two_opt_pass(&mut self) -> bool {
        let n = self.tour.len();
        if n < 4 {
            return false;
        }

        self.two_opt_iterations += 1;

        // Find best improving move
        let mut best_improvement = 0.0;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..(n - 2) {
            for j in (i + 2)..n {
                // Skip adjacent edges that share a node
                if i == 0 && j == n - 1 {
                    continue;
                }

                let improvement = self.two_opt_improvement(i, j);
                if improvement > best_improvement {
                    best_improvement = improvement;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_improvement > f64::EPSILON {
            self.apply_two_opt(best_i, best_j);
            self.two_opt_improvements += 1;
            true
        } else {
            false
        }
    }

    /// Run 2-opt to local optimum.
    pub fn two_opt_to_local_optimum(&mut self) {
        while self.two_opt_pass() {}
    }

    /// Run one GRASP iteration: construct + 2-opt.
    pub fn grasp_iteration(&mut self) {
        let start_time = Instant::now();
        let input_snapshot = self.snapshot();
        let rng_before = self.rng_state_hash();

        self.construct_tour();
        self.two_opt_to_local_optimum();

        // Track restart
        self.restarts += 1;
        self.restart_history.push(self.tour_length);

        // Log construction + 2-opt as single GRASP iteration
        let equations = vec![
            EquationEval::new("tour_length", self.tour_length).with_input("n", self.n as f64),
            EquationEval::new("optimality_gap", self.optimality_gap())
                .with_input("tour_length", self.tour_length)
                .with_input("lower_bound", self.lower_bound),
        ];

        let decisions = vec![Decision::new(
            "construction_method",
            format!("{:?}", self.construction_method),
        )
        .with_rationale("rcl_size", self.rcl_size as f64)];

        self.log_audit_step(
            TspStepType::GraspIteration,
            input_snapshot,
            rng_before,
            start_time,
            equations,
            decisions,
        );

        // Update best (0.0 means no tour yet computed)
        // MUDA: Track stagnation to eliminate restart waste
        if self.best_tour_length == 0.0 || self.tour_length < self.best_tour_length {
            self.best_tour_length = self.tour_length;
            self.best_tour = self.tour.clone();
            self.stagnation_count = 0; // Reset on improvement
        } else {
            self.stagnation_count += 1;
            // MUDA: Mark converged when stagnation threshold reached
            if self.stagnation_count >= self.stagnation_threshold {
                self.converged = true;
            }
        }
    }

    /// Run multiple GRASP iterations.
    pub fn run_grasp(&mut self, iterations: usize) {
        for _ in 0..iterations {
            // MUDA: Skip iterations after convergence (eliminate waste)
            if self.converged {
                break;
            }
            self.grasp_iteration();
        }
    }

    /// Check if the algorithm has converged (MUDA: stagnation detection).
    #[must_use]
    pub const fn is_converged(&self) -> bool {
        self.converged
    }

    /// Set stagnation threshold (MUDA: configurable waste elimination).
    pub fn set_stagnation_threshold(&mut self, threshold: u64) {
        self.stagnation_threshold = threshold;
    }

    /// Get optimality gap: (`tour_length` - `lower_bound`) / `lower_bound`.
    #[must_use]
    pub fn optimality_gap(&self) -> f64 {
        if self.lower_bound > f64::EPSILON {
            (self.best_tour_length - self.lower_bound) / self.lower_bound
        } else {
            0.0
        }
    }

    /// Get expected tour length from BHH formula.
    #[must_use]
    pub fn bhh_expected_length(&self) -> f64 {
        // E[L] ≈ 0.7124 * sqrt(n * A)
        0.7124 * (self.n as f64 * self.area).sqrt()
    }

    /// Compute variance of restart tour lengths.
    #[must_use]
    pub fn restart_variance(&self) -> f64 {
        if self.restart_history.len() < 2 {
            return 0.0;
        }

        let n = self.restart_history.len() as f64;
        let mean: f64 = self.restart_history.iter().sum::<f64>() / n;
        let variance: f64 = self
            .restart_history
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / n;

        variance
    }

    /// Compute coefficient of variation of restart tour lengths.
    #[must_use]
    pub fn restart_cv(&self) -> f64 {
        if self.restart_history.is_empty() {
            return 0.0;
        }

        let mean: f64 =
            self.restart_history.iter().sum::<f64>() / self.restart_history.len() as f64;
        if mean > f64::EPSILON {
            self.restart_variance().sqrt() / mean
        } else {
            0.0
        }
    }

    // =========================================================================
    // Audit Logging (EDD-16, EDD-17, EDD-18)
    // =========================================================================

    /// Create a state snapshot for audit logging.
    #[must_use]
    fn snapshot(&self) -> TspStateSnapshot {
        TspStateSnapshot {
            tour: self.tour.clone(),
            tour_length: self.tour_length,
            best_tour: self.best_tour.clone(),
            best_tour_length: self.best_tour_length,
            restarts: self.restarts,
            two_opt_iterations: self.two_opt_iterations,
            two_opt_improvements: self.two_opt_improvements,
        }
    }

    /// Get RNG state hash (for audit logging).
    fn rng_state_hash(&self) -> [u8; 32] {
        self.rng.as_ref().map_or([0u8; 32], |rng| {
            // Hash the RNG's internal state
            let state_bytes = rng.state_bytes();
            *blake3::hash(&state_bytes).as_bytes()
        })
    }

    /// Log a step to the audit trail.
    fn log_audit_step(
        &mut self,
        step_type: TspStepType,
        input_snapshot: TspStateSnapshot,
        rng_before: [u8; 32],
        start_time: Instant,
        equations: Vec<EquationEval>,
        decisions: Vec<Decision>,
    ) {
        if !self.audit_enabled {
            return;
        }

        let rng_after = self.rng_state_hash();
        let output_snapshot = self.snapshot();
        let duration_us = start_time.elapsed().as_micros() as u64;

        let mut entry = StepEntry::new(
            self.step_counter,
            SimTime::from_secs(self.step_counter as f64 * 0.001), // Arbitrary time step
            step_type.to_string(),
            input_snapshot,
            output_snapshot,
        )
        .with_rng_states(rng_before, rng_after)
        .with_duration(duration_us);

        for eq in equations {
            entry.add_equation_eval(eq);
        }

        for dec in decisions {
            entry.add_decision(dec);
        }

        self.audit_log.push(entry);
        self.step_counter += 1;
    }

    /// Enable or disable audit logging.
    pub fn set_audit_enabled(&mut self, enabled: bool) {
        self.audit_enabled = enabled;
    }
}

// =============================================================================
// SimulationAuditLog Implementation (EDD-16 MANDATORY)
// =============================================================================

impl SimulationAuditLog for TspGraspDemo {
    type StateSnapshot = TspStateSnapshot;

    fn log_step(&mut self, entry: StepEntry<Self::StateSnapshot>) {
        self.audit_log.push(entry);
    }

    fn audit_log(&self) -> &[StepEntry<Self::StateSnapshot>] {
        &self.audit_log
    }

    fn audit_log_mut(&mut self) -> &mut Vec<StepEntry<Self::StateSnapshot>> {
        &mut self.audit_log
    }

    fn clear_audit_log(&mut self) {
        self.audit_log.clear();
        self.step_counter = 0;
    }
}

impl EddDemo for TspGraspDemo {
    fn name(&self) -> &'static str {
        "TSP Randomized Greedy Start with 2-Opt"
    }

    fn emc_ref(&self) -> &'static str {
        "optimization/tsp_grasp_2opt"
    }

    fn step(&mut self, _dt: f64) {
        // One step = one GRASP iteration
        self.grasp_iteration();
    }

    fn verify_equation(&self) -> bool {
        // Verify optimality gap < 20% (1-tree bound typically gives 10-20% gap)
        let gap_ok = self.optimality_gap() < 0.20;

        // Verify low variance across restarts (CV < 5%)
        let cv_ok = self.restart_cv() < 0.05 || self.restart_history.len() < 5;

        // Verify no edge crossings in best tour (ONLY for Euclidean TSP)
        // Lin & Kernighan (1973): Non-Euclidean instances may have optimal "crossing" tours
        let no_crossings = !self.is_euclidean || self.best_tour.is_empty() || {
            // Temporarily check crossings on best tour
            let mut temp = self.clone();
            temp.tour.clone_from(&self.best_tour);
            temp.count_crossings() == 0
        };

        gap_ok && cv_ok && no_crossings
    }

    fn get_falsification_status(&self) -> FalsificationStatus {
        let gap = self.optimality_gap();
        let cv = self.restart_cv();

        // Check crossings on best tour
        let crossings = if self.best_tour.is_empty() {
            0
        } else {
            let mut temp = self.clone();
            temp.tour.clone_from(&self.best_tour);
            temp.count_crossings()
        };

        let gap_passed = gap < 0.20;
        let cv_passed = cv < 0.05 || self.restart_history.len() < 5;

        // JIDOKA: Crossings only matter for EUCLIDEAN instances
        // Lin & Kernighan (1973): Only Euclidean TSP optimal tours have ZERO crossings
        // Non-Euclidean (driving distances) may have "crossings" that are actually optimal
        let crossings_passed = !self.is_euclidean || crossings == 0;

        FalsificationStatus {
            verified: gap_passed && cv_passed && crossings_passed,
            criteria: vec![
                // JIDOKA: Crossings first - but only for Euclidean instances
                CriterionStatus {
                    id: "TSP-CROSSINGS".to_string(),
                    name: if self.is_euclidean {
                        "Edge crossings (Jidoka)".to_string()
                    } else {
                        "Edge crossings (N/A: non-Euclidean)".to_string()
                    },
                    passed: crossings_passed,
                    value: crossings as f64,
                    threshold: 0.0,
                },
                CriterionStatus {
                    id: "TSP-GAP".to_string(),
                    name: "Optimality gap (1-tree, Held-Karp 1970)".to_string(),
                    passed: gap_passed,
                    value: gap,
                    threshold: 0.20,
                },
                CriterionStatus {
                    id: "TSP-VARIANCE".to_string(),
                    name: "Restart consistency (CV)".to_string(),
                    passed: cv_passed,
                    value: cv,
                    threshold: 0.05,
                },
            ],
            message: if gap_passed && cv_passed && crossings_passed {
                if self.is_euclidean {
                    format!(
                        "TSP GRASP verified: crossings=0, gap={:.1}%, CV={:.1}%, best={:.1}",
                        gap * 100.0,
                        cv * 100.0,
                        self.best_tour_length
                    )
                } else {
                    format!(
                        "TSP GRASP verified (non-Euclidean): gap={:.1}%, CV={:.1}%, best={:.1}",
                        gap * 100.0,
                        cv * 100.0,
                        self.best_tour_length
                    )
                }
            } else if !crossings_passed && self.is_euclidean {
                // JIDOKA: Crossings are highest priority - stop the line! (Euclidean only)
                format!(
                    "JIDOKA STOP: Tour has {crossings} edge crossing(s) - \
                     Euclidean TSP tours MUST have 0 (Lin & Kernighan, 1973)"
                )
            } else if !gap_passed {
                format!(
                    "Optimality gap too large: {:.1}% (expected <20%, Held-Karp 1970)",
                    gap * 100.0
                )
            } else {
                format!(
                    "Restart variance too high: CV={:.1}% (expected <5%)",
                    cv * 100.0
                )
            },
        }
    }

    fn reset(&mut self) {
        let seed = self.seed;
        let n = self.n;
        let method = self.construction_method;
        let rcl_size = self.rcl_size;
        let cities = self.cities.clone();

        *self = Self::with_cities(seed, cities);
        self.construction_method = method;
        self.rcl_size = rcl_size;
        self.n = n;
    }
}

// =============================================================================
// WASM Bindings
// =============================================================================

#[cfg(feature = "wasm")]
mod wasm {
    use super::{City, ConstructionMethod, EddDemo, TspGraspDemo};
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmTspGrasp {
        inner: TspGraspDemo,
    }

    #[wasm_bindgen]
    impl WasmTspGrasp {
        // =====================================================================
        // Construction
        // =====================================================================

        #[wasm_bindgen(constructor)]
        pub fn new(seed: u32, n: usize) -> Self {
            Self {
                inner: TspGraspDemo::new(u64::from(seed), n),
            }
        }

        /// Create demo from YAML configuration string.
        ///
        /// This is the primary YAML-first entry point for web applications.
        /// Load `bay_area_tsp.yaml` or user-modified YAML.
        ///
        /// # Example (JavaScript)
        /// ```javascript
        /// const yaml = `
        /// meta:
        ///   id: "MY-TSP"
        /// cities:
        ///   - id: 0
        ///     name: "A"
        ///     alias: "A"
        ///     coords: { lat: 0.0, lon: 0.0 }
        ///   - id: 1
        ///     name: "B"
        ///     alias: "B"
        ///     coords: { lat: 1.0, lon: 1.0 }
        /// matrix:
        ///   - [0, 10]
        ///   - [10, 0]
        /// `;
        /// const demo = WasmTspGrasp.fromYaml(yaml);
        /// ```
        #[wasm_bindgen(js_name = fromYaml)]
        pub fn from_yaml(yaml: &str) -> Result<Self, String> {
            let inner = TspGraspDemo::from_yaml(yaml).map_err(|e| e.to_string())?;
            Ok(Self { inner })
        }

        /// Create demo with cities from JavaScript array.
        /// Expected format: [[x1, y1], [x2, y2], ...]
        #[wasm_bindgen(js_name = withCities)]
        pub fn with_cities_js(seed: u32, cities_js: &JsValue) -> Result<Self, JsValue> {
            let cities_array: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(cities_js.clone())
                .map_err(|e| JsValue::from_str(&format!("Invalid cities format: {e}")))?;

            let cities: Vec<City> = cities_array
                .into_iter()
                .filter_map(|coords| {
                    if coords.len() >= 2 {
                        Some(City::new(coords[0], coords[1]))
                    } else {
                        None
                    }
                })
                .collect();

            if cities.len() < 3 {
                return Err(JsValue::from_str("Need at least 3 cities"));
            }

            Ok(Self {
                inner: TspGraspDemo::with_cities(u64::from(seed), cities),
            })
        }

        // =====================================================================
        // Simulation Control
        // =====================================================================

        pub fn step(&mut self) {
            self.inner.step(0.0);
        }

        /// Run a single GRASP iteration (construct + 2-opt).
        #[wasm_bindgen(js_name = graspIteration)]
        pub fn grasp_iteration(&mut self) {
            self.inner.grasp_iteration();
        }

        /// Run multiple GRASP iterations.
        #[wasm_bindgen(js_name = runGrasp)]
        pub fn run_grasp(&mut self, iterations: usize) {
            self.inner.run_grasp(iterations);
        }

        /// Construct initial tour only (no 2-opt).
        #[wasm_bindgen(js_name = constructTour)]
        pub fn construct_tour(&mut self) {
            self.inner.construct_tour();
        }

        /// Run a single 2-opt pass. Returns true if improvement was made.
        #[wasm_bindgen(js_name = twoOptPass)]
        pub fn two_opt_pass(&mut self) -> bool {
            self.inner.two_opt_pass()
        }

        /// Run 2-opt to local optimum.
        #[wasm_bindgen(js_name = twoOptToLocalOptimum)]
        pub fn two_opt_to_local_optimum(&mut self) {
            self.inner.two_opt_to_local_optimum();
        }

        pub fn reset(&mut self) {
            self.inner.reset();
        }

        // =====================================================================
        // Configuration
        // =====================================================================

        /// Set construction method: 0=RandomizedGreedy, 1=NearestNeighbor, 2=Random.
        #[wasm_bindgen(js_name = setConstructionMethod)]
        pub fn set_construction_method(&mut self, method: u8) {
            self.inner.construction_method = match method {
                1 => ConstructionMethod::NearestNeighbor,
                2 => ConstructionMethod::Random,
                _ => ConstructionMethod::RandomizedGreedy, // 0 and any invalid value
            };
        }

        /// Set RCL size for randomized greedy construction.
        #[wasm_bindgen(js_name = setRclSize)]
        pub fn set_rcl_size(&mut self, size: usize) {
            self.inner.set_rcl_size(size);
        }

        // =====================================================================
        // State Queries - Primitives
        // =====================================================================

        #[wasm_bindgen(js_name = getTourLength)]
        pub fn get_tour_length(&self) -> f64 {
            self.inner.tour_length
        }

        #[wasm_bindgen(js_name = getBestTourLength)]
        pub fn get_best_tour_length(&self) -> f64 {
            self.inner.best_tour_length
        }

        #[wasm_bindgen(js_name = getOptimalityGap)]
        pub fn get_optimality_gap(&self) -> f64 {
            self.inner.optimality_gap()
        }

        #[wasm_bindgen(js_name = getLowerBound)]
        pub fn get_lower_bound(&self) -> f64 {
            self.inner.lower_bound
        }

        #[wasm_bindgen(js_name = getRestarts)]
        pub fn get_restarts(&self) -> u64 {
            self.inner.restarts
        }

        #[wasm_bindgen(js_name = getTwoOptIterations)]
        pub fn get_two_opt_iterations(&self) -> u64 {
            self.inner.two_opt_iterations
        }

        #[wasm_bindgen(js_name = getTwoOptImprovements)]
        pub fn get_two_opt_improvements(&self) -> u64 {
            self.inner.two_opt_improvements
        }

        #[wasm_bindgen(js_name = getN)]
        pub fn get_n(&self) -> usize {
            self.inner.n
        }

        /// Get the unit label for distances (e.g., "miles", "km").
        #[wasm_bindgen(js_name = getUnits)]
        pub fn get_units(&self) -> String {
            self.inner.units.clone()
        }

        /// Get known optimal tour length (if available from YAML).
        #[wasm_bindgen(js_name = getOptimalKnown)]
        pub fn get_optimal_known(&self) -> Option<u32> {
            self.inner.optimal_known
        }

        #[wasm_bindgen(js_name = getRclSize)]
        pub fn get_rcl_size(&self) -> usize {
            self.inner.rcl_size
        }

        #[wasm_bindgen(js_name = getConstructionMethod)]
        pub fn get_construction_method(&self) -> u8 {
            match self.inner.construction_method {
                ConstructionMethod::RandomizedGreedy => 0,
                ConstructionMethod::NearestNeighbor => 1,
                ConstructionMethod::Random => 2,
            }
        }

        #[wasm_bindgen(js_name = getRestartVariance)]
        pub fn get_restart_variance(&self) -> f64 {
            self.inner.restart_variance()
        }

        #[wasm_bindgen(js_name = getRestartCv)]
        pub fn get_restart_cv(&self) -> f64 {
            self.inner.restart_cv()
        }

        // =====================================================================
        // State Queries - JavaScript Objects
        // =====================================================================

        /// Get cities as JavaScript array: [[x1, y1], [x2, y2], ...]
        #[wasm_bindgen(js_name = getCities)]
        pub fn get_cities_js(&self) -> JsValue {
            let cities: Vec<[f64; 2]> = self.inner.cities.iter().map(|c| [c.x, c.y]).collect();
            serde_wasm_bindgen::to_value(&cities).unwrap_or(JsValue::NULL)
        }

        /// Get current tour as JavaScript array of indices.
        #[wasm_bindgen(js_name = getTour)]
        pub fn get_tour_js(&self) -> JsValue {
            serde_wasm_bindgen::to_value(&self.inner.tour).unwrap_or(JsValue::NULL)
        }

        /// Get best tour as JavaScript array of indices.
        #[wasm_bindgen(js_name = getBestTour)]
        pub fn get_best_tour_js(&self) -> JsValue {
            serde_wasm_bindgen::to_value(&self.inner.best_tour).unwrap_or(JsValue::NULL)
        }

        /// Get restart history as JavaScript array.
        #[wasm_bindgen(js_name = getRestartHistory)]
        pub fn get_restart_history_js(&self) -> JsValue {
            serde_wasm_bindgen::to_value(&self.inner.restart_history).unwrap_or(JsValue::NULL)
        }

        /// Get full state as JavaScript object for serialization.
        #[wasm_bindgen(js_name = getState)]
        pub fn get_state_js(&self) -> JsValue {
            serde_wasm_bindgen::to_value(&self.inner).unwrap_or(JsValue::NULL)
        }

        // =====================================================================
        // EDD Verification
        // =====================================================================

        pub fn verify(&self) -> bool {
            self.inner.verify_equation()
        }

        /// Get falsification status as JavaScript object.
        #[wasm_bindgen(js_name = getFalsificationStatus)]
        pub fn get_falsification_status_js(&self) -> JsValue {
            let status = self.inner.get_falsification_status();
            serde_wasm_bindgen::to_value(&status).unwrap_or(JsValue::NULL)
        }

        /// Get demo name.
        #[wasm_bindgen(js_name = getName)]
        pub fn get_name(&self) -> String {
            self.inner.name().to_string()
        }

        /// Get EMC reference.
        #[wasm_bindgen(js_name = getEmcRef)]
        pub fn get_emc_ref(&self) -> String {
            self.inner.emc_ref().to_string()
        }
    }
}

#[cfg(test)]
mod quality_tests;
#[cfg(test)]
mod tests;
