//! Configuration system with YAML schema and validation.
//!
//! Implements Poka-Yoke (mistake-proofing) through:
//! - Type-safe configuration structs
//! - Compile-time validation via serde
//! - Runtime semantic validation

use serde::{Deserialize, Serialize};
use std::path::Path;
use validator::Validate;

use crate::engine::jidoka::JidokaConfig;
use crate::error::{SimError, SimResult};

/// Top-level simulation configuration.
///
/// Loaded from YAML files with full schema validation.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(deny_unknown_fields)]
pub struct SimConfig {
    /// Schema version for forward compatibility.
    #[validate(length(min = 1))]
    #[serde(default = "default_schema_version")]
    pub schema_version: String,

    /// Simulation metadata.
    #[validate(nested)]
    #[serde(default)]
    pub simulation: SimulationMeta,

    /// Reproducibility settings.
    #[validate(nested)]
    pub reproducibility: ReproducibilityConfig,

    /// Domain-specific configurations.
    #[validate(nested)]
    #[serde(default)]
    pub domains: DomainsConfig,

    /// Jidoka (stop-on-error) configuration.
    #[serde(default)]
    pub jidoka: JidokaConfig,

    /// Replay configuration.
    #[validate(nested)]
    #[serde(default)]
    pub replay: ReplayConfig,

    /// Visualization configuration.
    #[serde(default)]
    pub visualization: VisualizationConfig,

    /// Falsification testing configuration.
    #[serde(default)]
    pub falsification: FalsificationConfig,
}

fn default_schema_version() -> String {
    "1.0".to_string()
}

impl SimConfig {
    /// Load configuration from a YAML file.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - File cannot be read
    /// - YAML parsing fails
    /// - Validation fails
    pub fn load<P: AsRef<Path>>(path: P) -> SimResult<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::from_yaml(&content)
    }

    /// Parse configuration from YAML string.
    ///
    /// # Errors
    ///
    /// Returns error if parsing or validation fails.
    pub fn from_yaml(yaml: &str) -> SimResult<Self> {
        let config: Self = serde_yaml::from_str(yaml)?;

        // Poka-Yoke: validate all constraints
        config.validate()?;

        // Additional semantic validation
        config.validate_semantic()?;

        Ok(config)
    }

    /// Create a builder for configuration.
    #[must_use]
    pub fn builder() -> SimConfigBuilder {
        SimConfigBuilder::default()
    }

    /// Validate semantic constraints beyond schema.
    fn validate_semantic(&self) -> SimResult<()> {
        // Ensure Monte Carlo has sufficient samples
        if self.domains.monte_carlo.enabled && self.domains.monte_carlo.samples < 100 {
            return Err(SimError::config(format!(
                "Monte Carlo requires at least 100 samples, got {}",
                self.domains.monte_carlo.samples
            )));
        }

        // Ensure timestep is reasonable
        let dt = self.domains.physics.timestep.dt;
        if dt <= 0.0 {
            return Err(SimError::config("Timestep must be positive"));
        }
        if dt > 1.0 {
            return Err(SimError::config("Timestep should not exceed 1 second"));
        }

        Ok(())
    }

    /// Get the timestep in seconds.
    #[must_use]
    pub const fn get_timestep(&self) -> f64 {
        self.domains.physics.timestep.dt
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            schema_version: default_schema_version(),
            simulation: SimulationMeta::default(),
            reproducibility: ReproducibilityConfig::default(),
            domains: DomainsConfig::default(),
            jidoka: JidokaConfig::default(),
            replay: ReplayConfig::default(),
            visualization: VisualizationConfig::default(),
            falsification: FalsificationConfig::default(),
        }
    }
}

/// Configuration builder for programmatic construction.
#[derive(Debug, Default)]
pub struct SimConfigBuilder {
    seed: Option<u64>,
    timestep: Option<f64>,
    jidoka: Option<JidokaConfig>,
}

impl SimConfigBuilder {
    /// Set the random seed.
    #[must_use]
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the timestep in seconds.
    #[must_use]
    pub const fn timestep(mut self, dt: f64) -> Self {
        self.timestep = Some(dt);
        self
    }

    /// Set Jidoka configuration.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // JidokaConfig doesn't impl Copy
    pub fn jidoka(mut self, config: JidokaConfig) -> Self {
        self.jidoka = Some(config);
        self
    }

    /// Build the configuration.
    #[must_use]
    pub fn build(self) -> SimConfig {
        let mut config = SimConfig::default();

        if let Some(seed) = self.seed {
            config.reproducibility.seed = seed;
        }

        if let Some(dt) = self.timestep {
            config.domains.physics.timestep.dt = dt;
        }

        if let Some(jidoka) = self.jidoka {
            config.jidoka = jidoka;
        }

        config
    }
}

/// Simulation metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct SimulationMeta {
    /// Simulation name.
    #[serde(default)]
    pub name: String,
    /// Description.
    #[serde(default)]
    pub description: String,
    /// Version.
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "0.1.0".to_string()
}

/// Reproducibility settings.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ReproducibilityConfig {
    /// Master seed for all RNG.
    pub seed: u64,
    /// IEEE 754 strict mode for cross-platform reproducibility.
    #[serde(default = "default_true")]
    pub ieee_strict: bool,
    /// Record RNG state in journal for perfect replay.
    #[serde(default = "default_true")]
    pub record_rng_state: bool,
}

const fn default_true() -> bool {
    true
}

impl Default for ReproducibilityConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            ieee_strict: true,
            record_rng_state: true,
        }
    }
}

/// Domain-specific configurations.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct DomainsConfig {
    /// Physics domain configuration.
    #[validate(nested)]
    #[serde(default)]
    pub physics: PhysicsConfig,
    /// Monte Carlo domain configuration.
    #[validate(nested)]
    #[serde(default)]
    pub monte_carlo: MonteCarloConfig,
    /// Optimization domain configuration.
    #[serde(default)]
    pub optimization: OptimizationConfig,
}

/// Physics domain configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PhysicsConfig {
    /// Whether physics is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Physics engine type.
    #[serde(default)]
    pub engine: PhysicsEngine,
    /// Integrator configuration.
    #[serde(default)]
    pub integrator: IntegratorConfig,
    /// Timestep configuration.
    #[validate(nested)]
    #[serde(default)]
    pub timestep: TimestepConfig,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            engine: PhysicsEngine::default(),
            integrator: IntegratorConfig::default(),
            timestep: TimestepConfig::default(),
        }
    }
}

/// Physics engine type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PhysicsEngine {
    /// Rigid body dynamics.
    #[default]
    RigidBody,
    /// Orbital mechanics.
    Orbital,
    /// Fluid dynamics.
    Fluid,
    /// Discrete event.
    Discrete,
}

/// Integrator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratorConfig {
    /// Integrator type.
    #[serde(default)]
    pub integrator_type: IntegratorType,
}

impl Default for IntegratorConfig {
    fn default() -> Self {
        Self {
            integrator_type: IntegratorType::Verlet,
        }
    }
}

/// Integrator type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IntegratorType {
    /// Euler method (1st order).
    Euler,
    /// Störmer-Verlet (2nd order, symplectic).
    #[default]
    Verlet,
    /// Runge-Kutta 4th order.
    Rk4,
    /// Dormand-Prince 7(8).
    Rk78,
    /// Symplectic Euler.
    SymplecticEuler,
}

/// Timestep configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TimestepConfig {
    /// Timestep mode.
    #[serde(default)]
    pub mode: TimestepMode,
    /// Fixed timestep in seconds.
    #[validate(range(min = 0.000_001, max = 1.0))]
    #[serde(default = "default_timestep")]
    pub dt: f64,
    /// Minimum timestep for adaptive mode.
    #[serde(default = "default_min_timestep")]
    pub min_dt: f64,
    /// Maximum timestep for adaptive mode.
    #[serde(default = "default_max_timestep")]
    pub max_dt: f64,
    /// Tolerance for adaptive mode.
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
}

const fn default_timestep() -> f64 {
    0.001
}

const fn default_min_timestep() -> f64 {
    0.0001
}

const fn default_max_timestep() -> f64 {
    0.01
}

const fn default_tolerance() -> f64 {
    1e-9
}

impl Default for TimestepConfig {
    fn default() -> Self {
        Self {
            mode: TimestepMode::Fixed,
            dt: default_timestep(),
            min_dt: default_min_timestep(),
            max_dt: default_max_timestep(),
            tolerance: default_tolerance(),
        }
    }
}

/// Timestep mode.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TimestepMode {
    /// Fixed timestep.
    #[default]
    Fixed,
    /// Adaptive timestep.
    Adaptive,
}

/// Monte Carlo configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MonteCarloConfig {
    /// Whether Monte Carlo is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Number of samples.
    #[validate(range(min = 1))]
    #[serde(default = "default_samples")]
    pub samples: usize,
    /// Variance reduction method.
    #[serde(default)]
    pub variance_reduction: VarianceReductionMethod,
}

const fn default_samples() -> usize {
    10_000
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            samples: default_samples(),
            variance_reduction: VarianceReductionMethod::None,
        }
    }
}

/// Variance reduction method.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum VarianceReductionMethod {
    /// No variance reduction.
    #[default]
    None,
    /// Antithetic variates.
    Antithetic,
    /// Control variates.
    ControlVariate,
    /// Importance sampling.
    Importance,
    /// Stratified sampling.
    Stratified,
}

/// Optimization configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Whether optimization is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Optimization algorithm.
    #[serde(default)]
    pub algorithm: OptimizationAlgorithm,
}

/// Optimization algorithm.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OptimizationAlgorithm {
    /// Bayesian optimization with GP surrogate.
    #[default]
    Bayesian,
    /// CMA-ES evolutionary strategy.
    CmaEs,
    /// Genetic algorithm.
    Genetic,
    /// Gradient-based optimization.
    Gradient,
}

/// Replay configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ReplayConfig {
    /// Whether replay is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Checkpoint interval in steps.
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: u64,
    /// Maximum checkpoint storage in bytes.
    #[serde(default = "default_max_storage")]
    pub max_storage: usize,
    /// Compression algorithm.
    #[serde(default)]
    pub compression: CompressionAlgorithm,
    /// Compression level (1-22 for zstd).
    #[validate(range(min = 1, max = 22))]
    #[serde(default = "default_compression_level")]
    pub compression_level: i32,
}

const fn default_checkpoint_interval() -> u64 {
    1000
}

const fn default_max_storage() -> usize {
    1024 * 1024 * 1024 // 1 GB
}

const fn default_compression_level() -> i32 {
    3
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            checkpoint_interval: default_checkpoint_interval(),
            max_storage: default_max_storage(),
            compression: CompressionAlgorithm::Zstd,
            compression_level: default_compression_level(),
        }
    }
}

/// Compression algorithm.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CompressionAlgorithm {
    /// No compression.
    None,
    /// LZ4 fast compression.
    Lz4,
    /// Zstandard compression.
    #[default]
    Zstd,
}

/// Visualization configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// TUI configuration.
    #[serde(default)]
    pub tui: TuiConfig,
    /// Web visualization configuration.
    #[serde(default)]
    pub web: WebConfig,
}

/// TUI configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiConfig {
    /// Whether TUI is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Refresh rate in Hz.
    #[serde(default = "default_refresh_hz")]
    pub refresh_hz: u32,
}

const fn default_refresh_hz() -> u32 {
    30
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            refresh_hz: default_refresh_hz(),
        }
    }
}

/// Web visualization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConfig {
    /// Whether web visualization is enabled.
    #[serde(default)]
    pub enabled: bool,
    /// Web server port.
    #[serde(default = "default_port")]
    pub port: u16,
}

const fn default_port() -> u16 {
    8080
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            port: default_port(),
        }
    }
}

/// Falsification testing configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FalsificationConfig {
    /// Null hypothesis description.
    #[serde(default)]
    pub null_hypothesis: String,
    /// Significance level (alpha).
    #[serde(default = "default_significance")]
    pub significance: f64,
}

const fn default_significance() -> f64 {
    0.05
}

// =============================================================================
// Poka-Yoke: Explicit Units in Configuration (Section 4.3.6)
// =============================================================================

/// Poka-Yoke velocity with mandatory explicit units [56].
///
/// Prevents unit confusion by requiring explicit unit strings in YAML.
///
/// # YAML Examples
///
/// ```yaml
/// separation_velocity: "10.0 m/s"
/// orbital_velocity: "7.8 km/s"
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct Velocity {
    /// Value in meters per second (canonical unit).
    pub meters_per_second: f64,
    /// Original unit string for display.
    pub original_unit: String,
}

impl Velocity {
    /// Create velocity from meters per second.
    #[must_use]
    pub fn from_mps(value: f64) -> Self {
        Self {
            meters_per_second: value,
            original_unit: "m/s".to_string(),
        }
    }

    /// Get value in meters per second.
    #[must_use]
    pub const fn as_mps(&self) -> f64 {
        self.meters_per_second
    }

    /// Get value in kilometers per second.
    #[must_use]
    pub fn as_kps(&self) -> f64 {
        self.meters_per_second / 1000.0
    }

    /// Get value in kilometers per hour.
    #[must_use]
    pub fn as_kph(&self) -> f64 {
        self.meters_per_second * 3.6
    }
}

impl<'de> Deserialize<'de> for Velocity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        parse_velocity(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "Invalid velocity '{s}'. Expected format: '<number> <unit>' \
                 where unit is 'm/s', 'km/s', 'km/h', 'ft/s', or 'kn' (knots)"
            ))
        })
    }
}

/// Parse velocity string with explicit units.
fn parse_velocity(s: &str) -> Option<Velocity> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }

    let value: f64 = parts[0].parse().ok()?;
    let unit = parts[1].to_lowercase();

    let meters_per_second = match unit.as_str() {
        "m/s" => value,
        "km/s" => value * 1000.0,
        "km/h" | "kph" => value / 3.6,
        "ft/s" => value * 0.3048,
        "kn" | "knots" | "kt" => value * 0.514_444,
        "mph" => value * 0.447_04,
        _ => return None,
    };

    Some(Velocity {
        meters_per_second,
        original_unit: parts[1].to_string(),
    })
}

/// Poka-Yoke distance/length with mandatory explicit units [56].
#[derive(Debug, Clone, Serialize)]
pub struct Length {
    /// Value in meters (canonical unit).
    pub meters: f64,
    /// Original unit string for display.
    pub original_unit: String,
}

impl Length {
    /// Create length from meters.
    #[must_use]
    pub fn from_meters(value: f64) -> Self {
        Self {
            meters: value,
            original_unit: "m".to_string(),
        }
    }

    /// Get value in meters.
    #[must_use]
    pub const fn as_meters(&self) -> f64 {
        self.meters
    }

    /// Get value in kilometers.
    #[must_use]
    pub fn as_km(&self) -> f64 {
        self.meters / 1000.0
    }
}

impl<'de> Deserialize<'de> for Length {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        parse_length(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "Invalid length '{s}'. Expected format: '<number> <unit>' \
                 where unit is 'm', 'km', 'cm', 'mm', 'ft', 'mi', or 'nm' (nautical miles)"
            ))
        })
    }
}

/// Parse length string with explicit units.
fn parse_length(s: &str) -> Option<Length> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }

    let value: f64 = parts[0].parse().ok()?;
    let unit = parts[1].to_lowercase();

    let meters = match unit.as_str() {
        "m" | "meters" => value,
        "km" | "kilometers" => value * 1000.0,
        "cm" | "centimeters" => value / 100.0,
        "mm" | "millimeters" => value / 1000.0,
        "ft" | "feet" => value * 0.3048,
        "mi" | "miles" => value * 1609.344,
        "nm" | "nmi" => value * 1852.0,    // Nautical miles
        "au" => value * 149_597_870_700.0, // Astronomical units
        _ => return None,
    };

    Some(Length {
        meters,
        original_unit: parts[1].to_string(),
    })
}

/// Poka-Yoke mass with mandatory explicit units [56].
#[derive(Debug, Clone, Serialize)]
pub struct Mass {
    /// Value in kilograms (canonical unit).
    pub kilograms: f64,
    /// Original unit string for display.
    pub original_unit: String,
}

impl Mass {
    /// Create mass from kilograms.
    #[must_use]
    pub fn from_kg(value: f64) -> Self {
        Self {
            kilograms: value,
            original_unit: "kg".to_string(),
        }
    }

    /// Get value in kilograms.
    #[must_use]
    pub const fn as_kg(&self) -> f64 {
        self.kilograms
    }

    /// Get value in grams.
    #[must_use]
    pub fn as_grams(&self) -> f64 {
        self.kilograms * 1000.0
    }
}

impl<'de> Deserialize<'de> for Mass {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        parse_mass(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "Invalid mass '{s}'. Expected format: '<number> <unit>' \
                 where unit is 'kg', 'g', 'mg', 't' (metric ton), or 'lb'"
            ))
        })
    }
}

/// Parse mass string with explicit units.
fn parse_mass(s: &str) -> Option<Mass> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }

    let value: f64 = parts[0].parse().ok()?;
    let unit = parts[1].to_lowercase();

    let kilograms = match unit.as_str() {
        "kg" | "kilograms" => value,
        "g" | "grams" => value / 1000.0,
        "mg" | "milligrams" => value / 1_000_000.0,
        "t" | "tonnes" | "metric_ton" => value * 1000.0,
        "lb" | "lbs" | "pounds" => value * 0.453_592,
        _ => return None,
    };

    Some(Mass {
        kilograms,
        original_unit: parts[1].to_string(),
    })
}

/// Poka-Yoke time duration with mandatory explicit units [56].
#[derive(Debug, Clone, Serialize)]
pub struct Duration {
    /// Value in seconds (canonical unit).
    pub seconds: f64,
    /// Original unit string for display.
    pub original_unit: String,
}

impl Duration {
    /// Create duration from seconds.
    #[must_use]
    pub fn from_seconds(value: f64) -> Self {
        Self {
            seconds: value,
            original_unit: "s".to_string(),
        }
    }

    /// Get value in seconds.
    #[must_use]
    pub const fn as_seconds(&self) -> f64 {
        self.seconds
    }

    /// Get value in milliseconds.
    #[must_use]
    pub fn as_millis(&self) -> f64 {
        self.seconds * 1000.0
    }
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        parse_duration(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "Invalid duration '{s}'. Expected format: '<number> <unit>' \
                 where unit is 's', 'ms', 'us', 'ns', 'min', 'h', or 'd'"
            ))
        })
    }
}

/// Parse duration string with explicit units.
fn parse_duration(s: &str) -> Option<Duration> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }

    let value: f64 = parts[0].parse().ok()?;
    let unit = parts[1].to_lowercase();

    let seconds = match unit.as_str() {
        "s" | "sec" | "seconds" => value,
        "ms" | "milliseconds" => value / 1000.0,
        "us" | "microseconds" | "µs" => value / 1_000_000.0,
        "ns" | "nanoseconds" => value / 1_000_000_000.0,
        "min" | "minutes" => value * 60.0,
        "h" | "hr" | "hours" => value * 3600.0,
        "d" | "days" => value * 86400.0,
        _ => return None,
    };

    Some(Duration {
        seconds,
        original_unit: parts[1].to_string(),
    })
}

/// Poka-Yoke angle with mandatory explicit units [56].
#[derive(Debug, Clone, Serialize)]
pub struct Angle {
    /// Value in radians (canonical unit).
    pub radians: f64,
    /// Original unit string for display.
    pub original_unit: String,
}

impl Angle {
    /// Create angle from radians.
    #[must_use]
    pub fn from_radians(value: f64) -> Self {
        Self {
            radians: value,
            original_unit: "rad".to_string(),
        }
    }

    /// Create angle from degrees.
    #[must_use]
    pub fn from_degrees(value: f64) -> Self {
        Self {
            radians: value.to_radians(),
            original_unit: "deg".to_string(),
        }
    }

    /// Get value in radians.
    #[must_use]
    pub const fn as_radians(&self) -> f64 {
        self.radians
    }

    /// Get value in degrees.
    #[must_use]
    pub fn as_degrees(&self) -> f64 {
        self.radians.to_degrees()
    }
}

impl<'de> Deserialize<'de> for Angle {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        parse_angle(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "Invalid angle '{s}'. Expected format: '<number> <unit>' \
                 where unit is 'rad', 'deg', 'arcmin', or 'arcsec'"
            ))
        })
    }
}

/// Parse angle string with explicit units.
fn parse_angle(s: &str) -> Option<Angle> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }

    let value: f64 = parts[0].parse().ok()?;
    let unit = parts[1].to_lowercase();

    let radians = match unit.as_str() {
        "rad" | "radians" => value,
        "deg" | "degrees" | "°" => value.to_radians(),
        "arcmin" => (value / 60.0).to_radians(),
        "arcsec" => (value / 3600.0).to_radians(),
        _ => return None,
    };

    Some(Angle {
        radians,
        original_unit: parts[1].to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = SimConfig::default();

        assert_eq!(config.schema_version, "1.0");
        assert_eq!(config.reproducibility.seed, 42);
        assert!(config.reproducibility.ieee_strict);
        assert!((config.domains.physics.timestep.dt - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_builder() {
        let config = SimConfig::builder().seed(12345).timestep(0.01).build();

        assert_eq!(config.reproducibility.seed, 12345);
        assert!((config.domains.physics.timestep.dt - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_yaml_parse() {
        let yaml = r"
reproducibility:
  seed: 42
domains:
  physics:
    enabled: true
    timestep:
      dt: 0.001
";
        let config = SimConfig::from_yaml(yaml);
        assert!(config.is_ok());

        let config = config.ok();
        assert!(config.is_some());
        assert_eq!(config.as_ref().map(|c| c.reproducibility.seed), Some(42));
    }

    #[test]
    fn test_config_validation_fails_invalid_samples() {
        let yaml = r"
reproducibility:
  seed: 42
domains:
  monte_carlo:
    enabled: true
    samples: 10
";
        let config = SimConfig::from_yaml(yaml);
        assert!(config.is_err());
    }

    #[test]
    fn test_config_validation_fails_negative_timestep() {
        let yaml = r"
reproducibility:
  seed: 42
domains:
  physics:
    timestep:
      dt: -0.001
";
        // Negative timestep should fail semantic validation
        let config = SimConfig::from_yaml(yaml);
        assert!(config.is_err());
    }

    #[test]
    fn test_integrator_types() {
        let yaml_verlet = r"
reproducibility:
  seed: 42
domains:
  physics:
    integrator:
      integrator_type: verlet
";
        let config = SimConfig::from_yaml(yaml_verlet);
        assert!(config.is_ok());

        let yaml_rk4 = r"
reproducibility:
  seed: 42
domains:
  physics:
    integrator:
      integrator_type: rk4
";
        let config = SimConfig::from_yaml(yaml_rk4);
        assert!(config.is_ok());
    }

    // === Poka-Yoke Unit Parsing Tests (Section 4.3.6) ===

    #[test]
    fn test_velocity_parsing() {
        // m/s
        let v = parse_velocity("10.0 m/s");
        assert!(v.is_some());
        assert!((v.as_ref().unwrap().meters_per_second - 10.0).abs() < 0.01);

        // km/s
        let v = parse_velocity("7.8 km/s");
        assert!(v.is_some());
        assert!((v.as_ref().unwrap().meters_per_second - 7800.0).abs() < 0.01);

        // km/h
        let v = parse_velocity("100 km/h");
        assert!(v.is_some());
        assert!((v.as_ref().unwrap().meters_per_second - 27.778).abs() < 0.01);

        // Invalid
        let v = parse_velocity("10.0");
        assert!(v.is_none());

        let v = parse_velocity("10.0 furlongs/fortnight");
        assert!(v.is_none());
    }

    #[test]
    fn test_velocity_conversions() {
        let v = Velocity::from_mps(1000.0);
        assert!((v.as_kps() - 1.0).abs() < f64::EPSILON);
        assert!((v.as_kph() - 3600.0).abs() < 0.01);
    }

    #[test]
    fn test_length_parsing() {
        // meters
        let l = parse_length("100 m");
        assert!(l.is_some());
        assert!((l.as_ref().unwrap().meters - 100.0).abs() < 0.01);

        // km
        let l = parse_length("1.5 km");
        assert!(l.is_some());
        assert!((l.as_ref().unwrap().meters - 1500.0).abs() < 0.01);

        // astronomical units
        let l = parse_length("1 au");
        assert!(l.is_some());
        assert!((l.as_ref().unwrap().meters - 149_597_870_700.0).abs() < 1000.0);

        // Invalid
        let l = parse_length("100");
        assert!(l.is_none());
    }

    #[test]
    fn test_mass_parsing() {
        // kg
        let m = parse_mass("100 kg");
        assert!(m.is_some());
        assert!((m.as_ref().unwrap().kilograms - 100.0).abs() < 0.01);

        // tonnes
        let m = parse_mass("1.5 t");
        assert!(m.is_some());
        assert!((m.as_ref().unwrap().kilograms - 1500.0).abs() < 0.01);

        // pounds
        let m = parse_mass("100 lb");
        assert!(m.is_some());
        assert!((m.as_ref().unwrap().kilograms - 45.36).abs() < 0.01);
    }

    #[test]
    fn test_duration_parsing() {
        // seconds
        let d = parse_duration("10 s");
        assert!(d.is_some());
        assert!((d.as_ref().unwrap().seconds - 10.0).abs() < f64::EPSILON);

        // milliseconds
        let d = parse_duration("1000 ms");
        assert!(d.is_some());
        assert!((d.as_ref().unwrap().seconds - 1.0).abs() < 0.001);

        // hours
        let d = parse_duration("2 h");
        assert!(d.is_some());
        assert!((d.as_ref().unwrap().seconds - 7200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_angle_parsing() {
        // radians
        let a = parse_angle("3.14159 rad");
        assert!(a.is_some());
        assert!((a.as_ref().unwrap().radians - 3.14159).abs() < 0.00001);

        // degrees
        let a = parse_angle("180 deg");
        assert!(a.is_some());
        assert!((a.as_ref().unwrap().radians - std::f64::consts::PI).abs() < 0.0001);

        // arcmin
        let a = parse_angle("60 arcmin");
        assert!(a.is_some());
        assert!((a.as_ref().unwrap().as_degrees() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_poka_yoke_rejects_unitless() {
        // All parse functions should reject unitless values
        assert!(parse_velocity("100").is_none());
        assert!(parse_length("100").is_none());
        assert!(parse_mass("100").is_none());
        assert!(parse_duration("100").is_none());
        assert!(parse_angle("100").is_none());
    }

    #[test]
    fn test_poka_yoke_yaml_deserialization() {
        #[derive(Debug, Deserialize)]
        struct TestConfig {
            velocity: Velocity,
            length: Length,
        }

        let yaml = r#"
velocity: "100 m/s"
length: "10 km"
"#;
        let config: Result<TestConfig, _> = serde_yaml::from_str(yaml);
        assert!(config.is_ok());

        let config = config.ok().unwrap();
        assert!((config.velocity.meters_per_second - 100.0).abs() < 0.01);
        assert!((config.length.meters - 10000.0).abs() < 0.01);
    }

    #[test]
    fn test_poka_yoke_yaml_rejects_invalid() {
        #[derive(Debug, Deserialize)]
        #[allow(dead_code)]
        struct TestConfig {
            velocity: Velocity,
        }

        // Unitless value should fail
        let yaml = r#"
velocity: "100"
"#;
        let config: Result<TestConfig, _> = serde_yaml::from_str(yaml);
        assert!(config.is_err());

        // Invalid unit should fail
        let yaml = r#"
velocity: "100 parsecs"
"#;
        let config: Result<TestConfig, _> = serde_yaml::from_str(yaml);
        assert!(config.is_err());
    }

    #[test]
    fn test_config_get_timestep() {
        let config = SimConfig::default();
        assert!((config.get_timestep() - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_builder_with_jidoka() {
        let jidoka = JidokaConfig::default();
        let config = SimConfig::builder().jidoka(jidoka).build();
        // JidokaConfig has energy_tolerance field
        assert!(config.jidoka.energy_tolerance > 0.0);
    }

    #[test]
    fn test_config_validation_fails_large_timestep() {
        let yaml = r"
reproducibility:
  seed: 42
domains:
  physics:
    timestep:
      dt: 2.0
";
        let config = SimConfig::from_yaml(yaml);
        assert!(config.is_err());
    }

    #[test]
    fn test_variance_reduction_methods() {
        let _none = VarianceReductionMethod::None;
        let _anti = VarianceReductionMethod::Antithetic;
        let _control = VarianceReductionMethod::ControlVariate;
        let _importance = VarianceReductionMethod::Importance;
        let _strat = VarianceReductionMethod::Stratified;
    }

    #[test]
    fn test_optimization_algorithms() {
        let _bayesian = OptimizationAlgorithm::Bayesian;
        let _cmaes = OptimizationAlgorithm::CmaEs;
        let _genetic = OptimizationAlgorithm::Genetic;
        let _gradient = OptimizationAlgorithm::Gradient;
    }

    #[test]
    fn test_compression_algorithms() {
        let _none = CompressionAlgorithm::None;
        let _lz4 = CompressionAlgorithm::Lz4;
        let _zstd = CompressionAlgorithm::Zstd;
    }

    #[test]
    fn test_velocity_all_units() {
        // ft/s
        let v = parse_velocity("100 ft/s");
        assert!(v.is_some());
        assert!((v.as_ref().unwrap().meters_per_second - 30.48).abs() < 0.01);

        // mph
        let v = parse_velocity("60 mph");
        assert!(v.is_some());
        assert!((v.as_ref().unwrap().meters_per_second - 26.82).abs() < 0.1);

        // knots
        let v = parse_velocity("100 kn");
        assert!(v.is_some());
        assert!((v.as_ref().unwrap().meters_per_second - 51.44).abs() < 0.1);

        // kph alias
        let v = parse_velocity("36 kph");
        assert!(v.is_some());
        assert!((v.as_ref().unwrap().meters_per_second - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_length_all_units() {
        // cm
        let l = parse_length("100 cm");
        assert!(l.is_some());
        assert!((l.as_ref().unwrap().meters - 1.0).abs() < 0.01);

        // mm
        let l = parse_length("1000 mm");
        assert!(l.is_some());
        assert!((l.as_ref().unwrap().meters - 1.0).abs() < 0.01);

        // ft
        let l = parse_length("100 ft");
        assert!(l.is_some());
        assert!((l.as_ref().unwrap().meters - 30.48).abs() < 0.01);

        // mi
        let l = parse_length("1 mi");
        assert!(l.is_some());
        assert!((l.as_ref().unwrap().meters - 1609.344).abs() < 0.01);

        // nm (nautical miles)
        let l = parse_length("1 nm");
        assert!(l.is_some());
        assert!((l.as_ref().unwrap().meters - 1852.0).abs() < 0.01);

        // full word
        let l = parse_length("1 meters");
        assert!(l.is_some());
    }

    #[test]
    fn test_length_conversions() {
        let l = Length::from_meters(1000.0);
        assert!((l.as_km() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mass_all_units() {
        // g
        let m = parse_mass("1000 g");
        assert!(m.is_some());
        assert!((m.as_ref().unwrap().kilograms - 1.0).abs() < 0.01);

        // mg
        let m = parse_mass("1000000 mg");
        assert!(m.is_some());
        assert!((m.as_ref().unwrap().kilograms - 1.0).abs() < 0.01);

        // lbs alias
        let m = parse_mass("2.2 lbs");
        assert!(m.is_some());
        assert!((m.as_ref().unwrap().kilograms - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mass_conversions() {
        let m = Mass::from_kg(1.0);
        assert!((m.as_grams() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_duration_all_units() {
        // us
        let d = parse_duration("1000000 us");
        assert!(d.is_some());
        assert!((d.as_ref().unwrap().seconds - 1.0).abs() < 0.001);

        // ns
        let d = parse_duration("1000000000 ns");
        assert!(d.is_some());
        assert!((d.as_ref().unwrap().seconds - 1.0).abs() < 0.001);

        // min
        let d = parse_duration("1 min");
        assert!(d.is_some());
        assert!((d.as_ref().unwrap().seconds - 60.0).abs() < f64::EPSILON);

        // d (days)
        let d = parse_duration("1 d");
        assert!(d.is_some());
        assert!((d.as_ref().unwrap().seconds - 86400.0).abs() < f64::EPSILON);

        // sec alias
        let d = parse_duration("10 sec");
        assert!(d.is_some());
    }

    #[test]
    fn test_duration_conversions() {
        let d = Duration::from_seconds(1.0);
        assert!((d.as_millis() - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_angle_all_units() {
        // arcsec
        let a = parse_angle("3600 arcsec");
        assert!(a.is_some());
        assert!((a.as_ref().unwrap().as_degrees() - 1.0).abs() < 0.001);

        // degrees alias
        let a = parse_angle("90 degrees");
        assert!(a.is_some());

        // radians alias
        let a = parse_angle("1 radians");
        assert!(a.is_some());
    }

    #[test]
    fn test_angle_conversions() {
        let a = Angle::from_degrees(180.0);
        assert!((a.as_radians() - std::f64::consts::PI).abs() < 0.0001);
        assert!((a.as_degrees() - 180.0).abs() < 0.0001);

        let a2 = Angle::from_radians(std::f64::consts::PI);
        assert!((a2.as_degrees() - 180.0).abs() < 0.0001);
    }

    #[test]
    fn test_parse_invalid_number() {
        assert!(parse_velocity("abc m/s").is_none());
        assert!(parse_length("abc m").is_none());
        assert!(parse_mass("abc kg").is_none());
        assert!(parse_duration("abc s").is_none());
        assert!(parse_angle("abc rad").is_none());
    }

    #[test]
    fn test_parse_empty_string() {
        assert!(parse_velocity("").is_none());
        assert!(parse_length("").is_none());
        assert!(parse_mass("").is_none());
        assert!(parse_duration("").is_none());
        assert!(parse_angle("").is_none());
    }

    #[test]
    fn test_parse_too_many_parts() {
        assert!(parse_velocity("100 m per second").is_none());
        assert!(parse_length("100 meters long").is_none());
    }

    #[test]
    fn test_simulation_meta_default() {
        let meta = SimulationMeta::default();
        assert!(meta.name.is_empty());
    }

    #[test]
    fn test_replay_config_default() {
        let config = ReplayConfig::default();
        assert!(config.enabled);
        assert_eq!(config.checkpoint_interval, 1000);
    }

    #[test]
    fn test_tui_config_default() {
        let config = TuiConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.refresh_hz, 30);
    }

    #[test]
    fn test_web_config_default() {
        let config = WebConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn test_falsification_config_default() {
        let config = FalsificationConfig::default();
        assert!(config.null_hypothesis.is_empty());
        // Rust Default for f64 is 0.0; serde default is 0.05 (via default_significance)
        assert!((config.significance - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monte_carlo_config_default() {
        let config = MonteCarloConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.samples, 10_000);
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_domains_config_default() {
        let config = DomainsConfig::default();
        assert!(config.physics.enabled);
        assert!(!config.monte_carlo.enabled);
    }

    #[test]
    fn test_reproducibility_config_default() {
        let config = ReproducibilityConfig::default();
        assert_eq!(config.seed, 42);
        assert!(config.ieee_strict);
    }
}
