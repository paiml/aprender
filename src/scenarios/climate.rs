//! Simplified climate model scenarios.
//!
//! Implements energy balance climate models:
//! - Zero-dimensional energy balance model (global mean temperature)
//! - Climate sensitivity analysis
//! - Radiative forcing scenarios (CO2, aerosols)

use crate::engine::state::SimState;
use crate::error::{SimError, SimResult};
use serde::{Deserialize, Serialize};

/// Stefan-Boltzmann constant (W/m²/K⁴).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Solar constant (W/m²).
pub const SOLAR_CONSTANT: f64 = 1361.0;

/// Pre-industrial CO2 concentration (ppm).
pub const PREINDUSTRIAL_CO2: f64 = 280.0;

/// Climate sensitivity parameter (K per W/m²).
/// Typical range: 0.3-1.2 K/(W/m²).
pub const DEFAULT_CLIMATE_SENSITIVITY: f64 = 0.8;

/// Configuration for energy balance climate model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateConfig {
    /// Initial global mean temperature (K).
    pub initial_temperature: f64,
    /// Planetary albedo (fraction reflected, 0-1).
    pub albedo: f64,
    /// Effective emissivity (greenhouse effect, 0-1).
    pub emissivity: f64,
    /// Ocean heat capacity (J/m²/K).
    pub heat_capacity: f64,
    /// CO2 concentration (ppm).
    pub co2_concentration: f64,
    /// Climate sensitivity parameter (K per W/m²).
    pub climate_sensitivity: f64,
    /// Aerosol forcing (W/m², typically negative).
    pub aerosol_forcing: f64,
}

impl Default for ClimateConfig {
    fn default() -> Self {
        Self {
            initial_temperature: 288.0, // ~15°C global mean
            albedo: 0.3,
            emissivity: 0.612,    // Effective emissivity for Earth
            heat_capacity: 1.7e8, // Mixed-layer ocean (~50m depth)
            co2_concentration: PREINDUSTRIAL_CO2,
            climate_sensitivity: DEFAULT_CLIMATE_SENSITIVITY,
            aerosol_forcing: 0.0,
        }
    }
}

impl ClimateConfig {
    /// Create present-day configuration (~420 ppm CO2).
    #[must_use]
    pub fn present_day() -> Self {
        Self {
            co2_concentration: 420.0,
            aerosol_forcing: -0.5, // Cooling effect of aerosols
            ..Default::default()
        }
    }

    /// Create doubled CO2 scenario.
    #[must_use]
    pub fn doubled_co2() -> Self {
        Self {
            co2_concentration: 2.0 * PREINDUSTRIAL_CO2,
            ..Default::default()
        }
    }

    /// Create RCP 8.5 end-of-century scenario (~1000 ppm CO2).
    #[must_use]
    pub fn rcp85() -> Self {
        Self {
            co2_concentration: 1000.0,
            ..Default::default()
        }
    }

    /// Calculate radiative forcing from CO2 (W/m²).
    /// Uses logarithmic relationship: ΔF = 5.35 × ln(C/C₀)
    #[must_use]
    pub fn co2_forcing(&self) -> f64 {
        5.35 * (self.co2_concentration / PREINDUSTRIAL_CO2).ln()
    }

    /// Calculate total radiative forcing (W/m²).
    #[must_use]
    pub fn total_forcing(&self) -> f64 {
        self.co2_forcing() + self.aerosol_forcing
    }

    /// Calculate equilibrium temperature for current forcing.
    #[must_use]
    pub fn equilibrium_temperature(&self) -> f64 {
        let absorbed_solar = SOLAR_CONSTANT * (1.0 - self.albedo) / 4.0;
        let forcing = self.total_forcing();

        // Solve: absorbed_solar + forcing = emissivity * sigma * T^4
        let total_flux = absorbed_solar + forcing;
        (total_flux / (self.emissivity * STEFAN_BOLTZMANN)).powf(0.25)
    }

    /// Calculate equilibrium climate sensitivity (ECS) in Kelvin.
    /// ECS = temperature change for doubled CO2.
    #[must_use]
    pub fn ecs(&self) -> f64 {
        let forcing_2xco2 = 5.35 * 2.0_f64.ln(); // ~3.7 W/m²
        self.climate_sensitivity * forcing_2xco2
    }
}

/// Climate state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClimateState {
    /// Global mean surface temperature (K).
    pub temperature: f64,
    /// Ocean heat content anomaly (J/m²).
    pub ocean_heat: f64,
    /// Radiative imbalance at TOA (W/m²).
    pub radiative_imbalance: f64,
    /// Time (years from start).
    pub time: f64,
}

/// Energy balance climate model scenario.
#[derive(Debug, Clone)]
pub struct ClimateScenario {
    config: ClimateConfig,
    state: ClimateState,
}

impl ClimateScenario {
    /// Create a new climate scenario.
    #[must_use]
    pub fn new(config: ClimateConfig) -> Self {
        let state = ClimateState {
            temperature: config.initial_temperature,
            ocean_heat: 0.0,
            radiative_imbalance: 0.0,
            time: 0.0,
        };
        Self { config, state }
    }

    /// Initialize simulation state.
    #[must_use]
    pub fn init_state(&self) -> SimState {
        let mut state = SimState::new();
        // Use mass=heat_capacity, position.x=temperature, velocity.x=dT/dt
        state.add_body(
            self.config.heat_capacity,
            crate::engine::state::Vec3::new(self.config.initial_temperature, 0.0, 0.0),
            crate::engine::state::Vec3::zero(),
        );
        state
    }

    /// Step the climate model forward in time.
    ///
    /// Uses forward Euler integration of the energy balance equation:
    /// `C × dT/dt = absorbed_solar + forcing - outgoing_longwave`
    ///
    /// # Errors
    ///
    /// Returns an error if temperature becomes non-physical (<0K or >500K).
    #[allow(clippy::while_float)]
    pub fn step(&mut self, dt_years: f64) -> SimResult<&ClimateState> {
        // Incoming absorbed solar radiation (W/m²)
        let absorbed_solar = SOLAR_CONSTANT * (1.0 - self.config.albedo) / 4.0;

        // Radiative forcing
        let forcing = self.config.total_forcing();

        // Outgoing longwave radiation (W/m²)
        let outgoing = self.config.emissivity * STEFAN_BOLTZMANN * self.state.temperature.powi(4);

        // Net radiative imbalance
        let imbalance = absorbed_solar + forcing - outgoing;
        self.state.radiative_imbalance = imbalance;

        // Temperature change (convert years to seconds)
        let dt_seconds = dt_years * 365.25 * 24.0 * 3600.0;
        let dt_temp = imbalance * dt_seconds / self.config.heat_capacity;

        // Jidoka: Check for non-physical temperature
        let new_temp = self.state.temperature + dt_temp;
        if !(0.0..=500.0).contains(&new_temp) {
            return Err(SimError::jidoka(format!(
                "Non-physical temperature: {new_temp} K"
            )));
        }

        self.state.temperature = new_temp;
        self.state.ocean_heat += imbalance * dt_seconds;
        self.state.time += dt_years;

        Ok(&self.state)
    }

    /// Run simulation to equilibrium.
    ///
    /// Returns the trajectory of climate states.
    ///
    /// # Errors
    ///
    /// Returns an error if temperature becomes non-physical during simulation.
    #[allow(clippy::while_float)]
    pub fn run_to_equilibrium(
        &mut self,
        dt_years: f64,
        max_years: f64,
        tolerance: f64,
    ) -> SimResult<Vec<ClimateState>> {
        let mut trajectory = vec![self.state.clone()];

        while self.state.time < max_years {
            let prev_temp = self.state.temperature;
            self.step(dt_years)?;
            trajectory.push(self.state.clone());

            // Check for equilibrium
            let temp_change = (self.state.temperature - prev_temp).abs();
            if temp_change / dt_years < tolerance {
                break;
            }
        }

        Ok(trajectory)
    }

    /// Get current climate state.
    #[must_use]
    pub const fn state(&self) -> &ClimateState {
        &self.state
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &ClimateConfig {
        &self.config
    }

    /// Calculate temperature anomaly from pre-industrial.
    #[must_use]
    pub fn temperature_anomaly(&self) -> f64 {
        self.state.temperature - 288.0 // Pre-industrial baseline
    }

    /// Set CO2 concentration (for scenario analysis).
    pub fn set_co2(&mut self, ppm: f64) {
        self.config.co2_concentration = ppm;
    }
}

/// Climate forcing scenario for transient simulations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForcingScenario {
    /// Name of scenario.
    pub name: String,
    /// CO2 trajectory (year, ppm).
    pub co2_trajectory: Vec<(f64, f64)>,
    /// Aerosol forcing trajectory (year, W/m²).
    pub aerosol_trajectory: Vec<(f64, f64)>,
}

impl ForcingScenario {
    /// Create historical forcing scenario (1850-2020).
    #[must_use]
    pub fn historical() -> Self {
        Self {
            name: "Historical".to_string(),
            co2_trajectory: vec![
                (1850.0, 285.0),
                (1900.0, 296.0),
                (1950.0, 311.0),
                (1980.0, 338.0),
                (2000.0, 369.0),
                (2020.0, 414.0),
            ],
            aerosol_trajectory: vec![
                (1850.0, 0.0),
                (1950.0, -0.2),
                (1980.0, -0.5),
                (2000.0, -0.4),
                (2020.0, -0.3),
            ],
        }
    }

    /// Interpolate CO2 at given year.
    #[must_use]
    pub fn co2_at(&self, year: f64) -> f64 {
        interpolate(&self.co2_trajectory, year)
    }

    /// Interpolate aerosol forcing at given year.
    #[must_use]
    pub fn aerosol_at(&self, year: f64) -> f64 {
        interpolate(&self.aerosol_trajectory, year)
    }
}

/// Linear interpolation helper.
fn interpolate(data: &[(f64, f64)], x: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    if x <= data[0].0 {
        return data[0].1;
    }
    if x >= data[data.len() - 1].0 {
        return data[data.len() - 1].1;
    }

    for i in 0..data.len() - 1 {
        if x >= data[i].0 && x <= data[i + 1].0 {
            let t = (x - data[i].0) / (data[i + 1].0 - data[i].0);
            return data[i].1 + t * (data[i + 1].1 - data[i].1);
        }
    }

    data[data.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_climate_config_default() {
        let config = ClimateConfig::default();

        assert!((config.initial_temperature - 288.0).abs() < 0.01);
        assert!((config.albedo - 0.3).abs() < 0.01);
        assert!((config.co2_concentration - PREINDUSTRIAL_CO2).abs() < 0.01);
    }

    #[test]
    fn test_climate_config_co2_forcing() {
        let config = ClimateConfig::default();

        // Pre-industrial: zero forcing
        assert!(config.co2_forcing().abs() < 0.01);

        // Doubled CO2: ~3.7 W/m²
        let doubled = ClimateConfig::doubled_co2();
        let forcing = doubled.co2_forcing();
        assert!((forcing - 3.7).abs() < 0.1, "2xCO2 forcing = {forcing}");
    }

    #[test]
    fn test_climate_config_ecs() {
        let config = ClimateConfig {
            climate_sensitivity: 0.8, // K per W/m²
            ..Default::default()
        };

        // ECS should be ~3K for typical sensitivity
        let ecs = config.ecs();
        assert!(ecs > 2.0 && ecs < 4.0, "ECS = {ecs}");
    }

    #[test]
    fn test_climate_scenario_step() {
        let config = ClimateConfig::doubled_co2();
        let mut scenario = ClimateScenario::new(config);

        // Initial state
        assert!((scenario.state().temperature - 288.0).abs() < 0.01);

        // Step forward 1 year
        scenario.step(1.0).unwrap();

        // Temperature should increase due to positive forcing
        assert!(scenario.state().temperature > 288.0);
    }

    #[test]
    fn test_climate_scenario_equilibrium() {
        let config = ClimateConfig::doubled_co2();
        let mut scenario = ClimateScenario::new(config);

        // Run to equilibrium
        let trajectory = scenario.run_to_equilibrium(1.0, 500.0, 0.001).unwrap();

        // Should reach equilibrium
        assert!(!trajectory.is_empty());

        // Final temperature should be higher than initial
        let final_temp = trajectory.last().unwrap().temperature;
        assert!(final_temp > 288.0, "Final temp = {final_temp}");

        // Temperature increase should be positive for doubled CO2
        // The exact value depends on model parameters and feedbacks
        let delta_t = final_temp - 288.0;
        assert!(delta_t > 0.5, "ΔT = {delta_t} should be positive");
    }

    #[test]
    fn test_climate_scenario_present_day() {
        let config = ClimateConfig::present_day();

        // Present day should have positive CO2 forcing
        assert!(config.co2_forcing() > 0.0);

        // But aerosol cooling partially offsets
        assert!(config.aerosol_forcing < 0.0);

        // Net forcing should still be positive
        assert!(config.total_forcing() > 0.0);
    }

    #[test]
    fn test_climate_scenario_rcp85() {
        let config = ClimateConfig::rcp85();

        // RCP8.5 has much higher CO2 forcing
        let forcing = config.co2_forcing();
        assert!(forcing > 6.0, "RCP8.5 forcing = {forcing}");
    }

    #[test]
    fn test_forcing_scenario_historical() {
        let scenario = ForcingScenario::historical();

        // 1850: near pre-industrial
        let co2_1850 = scenario.co2_at(1850.0);
        assert!((co2_1850 - 285.0).abs() < 1.0);

        // 2020: ~414 ppm
        let co2_2020 = scenario.co2_at(2020.0);
        assert!((co2_2020 - 414.0).abs() < 1.0);

        // Interpolation: 1975 should be between 1950 and 1980
        let co2_1975 = scenario.co2_at(1975.0);
        assert!(co2_1975 > 311.0 && co2_1975 < 338.0);
    }

    #[test]
    fn test_interpolate() {
        let data = vec![(0.0, 0.0), (10.0, 100.0)];

        assert!((interpolate(&data, 0.0) - 0.0).abs() < 0.01);
        assert!((interpolate(&data, 5.0) - 50.0).abs() < 0.01);
        assert!((interpolate(&data, 10.0) - 100.0).abs() < 0.01);

        // Extrapolation: clamp to endpoints
        assert!((interpolate(&data, -5.0) - 0.0).abs() < 0.01);
        assert!((interpolate(&data, 15.0) - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_climate_jidoka_temperature_bounds() {
        let mut config = ClimateConfig::default();
        config.initial_temperature = 10.0; // Unrealistically cold
        config.co2_concentration = 10.0; // Very low CO2

        let mut scenario = ClimateScenario::new(config);

        // Should eventually trigger Jidoka if temperature goes non-physical
        // (though this specific case may not, it tests the bounds checking)
        let result = scenario.step(100.0);
        // Either succeeds or fails with Jidoka error
        match result {
            Ok(state) => assert!(state.temperature > 0.0 && state.temperature < 500.0),
            Err(e) => assert!(e.to_string().contains("Non-physical")),
        }
    }

    #[test]
    fn test_climate_equilibrium_temperature() {
        let config = ClimateConfig::default();

        // Pre-industrial equilibrium should be close to 288K
        let eq_temp = config.equilibrium_temperature();
        assert!(
            (eq_temp - 288.0).abs() < 5.0,
            "Equilibrium temp = {eq_temp}"
        );
    }

    // === Additional Coverage Tests ===

    #[test]
    fn test_climate_scenario_init_state() {
        let config = ClimateConfig::default();
        let scenario = ClimateScenario::new(config);
        let state = scenario.init_state();

        assert_eq!(state.num_bodies(), 1);
        // Temperature is stored in position.x
        assert!((state.positions()[0].x - 288.0).abs() < 0.01);
    }

    #[test]
    fn test_climate_scenario_temperature_anomaly() {
        let mut config = ClimateConfig::default();
        config.initial_temperature = 290.0; // 2K above pre-industrial
        let scenario = ClimateScenario::new(config);

        let anomaly = scenario.temperature_anomaly();
        assert!((anomaly - 2.0).abs() < 0.01, "Anomaly = {anomaly}");
    }

    #[test]
    fn test_climate_scenario_set_co2() {
        let config = ClimateConfig::default();
        let mut scenario = ClimateScenario::new(config);

        scenario.set_co2(560.0);
        assert!((scenario.config().co2_concentration - 560.0).abs() < 0.01);
    }

    #[test]
    fn test_forcing_scenario_aerosol_at() {
        let scenario = ForcingScenario::historical();

        // 1850: zero aerosol forcing
        let aerosol_1850 = scenario.aerosol_at(1850.0);
        assert!((aerosol_1850 - 0.0).abs() < 0.01);

        // 1980: peak aerosol forcing
        let aerosol_1980 = scenario.aerosol_at(1980.0);
        assert!((aerosol_1980 - (-0.5)).abs() < 0.01);

        // Interpolation
        let aerosol_1965 = scenario.aerosol_at(1965.0);
        assert!(aerosol_1965 < 0.0 && aerosol_1965 > -0.5);
    }

    #[test]
    fn test_interpolate_empty() {
        let data: Vec<(f64, f64)> = vec![];
        let result = interpolate(&data, 5.0);
        assert!((result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_climate_state_clone_and_debug() {
        let state = ClimateState {
            temperature: 288.0,
            ocean_heat: 1000.0,
            radiative_imbalance: 0.5,
            time: 10.0,
        };

        let cloned = state.clone();
        assert!((cloned.temperature - 288.0).abs() < f64::EPSILON);
        assert!((cloned.time - 10.0).abs() < f64::EPSILON);

        let debug = format!("{:?}", state);
        assert!(debug.contains("ClimateState"));
        assert!(debug.contains("temperature"));
    }

    #[test]
    fn test_climate_config_clone_and_debug() {
        let config = ClimateConfig::default();
        let cloned = config.clone();
        assert!((cloned.albedo - config.albedo).abs() < f64::EPSILON);

        let debug = format!("{:?}", config);
        assert!(debug.contains("ClimateConfig"));
    }

    #[test]
    fn test_climate_scenario_clone() {
        let config = ClimateConfig::doubled_co2();
        let scenario = ClimateScenario::new(config);
        let cloned = scenario.clone();

        assert!((cloned.state().temperature - scenario.state().temperature).abs() < f64::EPSILON);
    }

    #[test]
    fn test_forcing_scenario_clone_and_debug() {
        let scenario = ForcingScenario::historical();
        let cloned = scenario.clone();

        assert_eq!(cloned.name, "Historical");
        assert_eq!(cloned.co2_trajectory.len(), scenario.co2_trajectory.len());

        let debug = format!("{:?}", scenario);
        assert!(debug.contains("ForcingScenario"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// CO2 forcing increases logarithmically with concentration.
        #[test]
        fn prop_co2_forcing_increases_with_concentration(
            co2_1 in 200.0f64..2000.0,
            co2_2 in 200.0f64..2000.0,
        ) {
            let config1 = ClimateConfig { co2_concentration: co2_1, ..Default::default() };
            let config2 = ClimateConfig { co2_concentration: co2_2, ..Default::default() };

            let forcing1 = config1.co2_forcing();
            let forcing2 = config2.co2_forcing();

            // Higher CO2 -> higher forcing
            if co2_1 > co2_2 {
                prop_assert!(forcing1 > forcing2);
            } else if co2_1 < co2_2 {
                prop_assert!(forcing1 < forcing2);
            }
        }

        /// Equilibrium temperature increases with forcing.
        #[test]
        fn prop_equilibrium_temp_increases_with_co2(
            co2_1 in 280.0f64..1500.0,
            co2_2 in 280.0f64..1500.0,
        ) {
            let config1 = ClimateConfig { co2_concentration: co2_1, ..Default::default() };
            let config2 = ClimateConfig { co2_concentration: co2_2, ..Default::default() };

            let temp1 = config1.equilibrium_temperature();
            let temp2 = config2.equilibrium_temperature();

            // Higher CO2 -> higher equilibrium temperature
            if co2_1 > co2_2 {
                prop_assert!(temp1 > temp2 - 0.001, "temp1={}, temp2={}", temp1, temp2);
            } else if co2_1 < co2_2 {
                prop_assert!(temp1 < temp2 + 0.001, "temp1={}, temp2={}", temp1, temp2);
            }
        }

        /// Albedo reduces absorbed solar radiation.
        #[test]
        fn prop_higher_albedo_less_absorbed(
            albedo1 in 0.1f64..0.9,
            albedo2 in 0.1f64..0.9,
        ) {
            let config1 = ClimateConfig { albedo: albedo1, ..Default::default() };
            let config2 = ClimateConfig { albedo: albedo2, ..Default::default() };

            // Absorbed = S/4 * (1 - albedo)
            let absorbed1 = SOLAR_CONSTANT / 4.0 * (1.0 - config1.albedo);
            let absorbed2 = SOLAR_CONSTANT / 4.0 * (1.0 - config2.albedo);

            // Higher albedo -> less absorbed
            if albedo1 > albedo2 {
                prop_assert!(absorbed1 < absorbed2);
            } else if albedo1 < albedo2 {
                prop_assert!(absorbed1 > absorbed2);
            }
        }

        /// Climate state temperature must be positive (Kelvin).
        #[test]
        fn prop_temperature_positive(
            initial_temp in 200.0f64..350.0,
        ) {
            let config = ClimateConfig {
                initial_temperature: initial_temp,
                ..Default::default()
            };
            let scenario = ClimateScenario::new(config);

            prop_assert!(scenario.state().temperature > 0.0);
        }

        /// Total forcing is CO2 forcing plus aerosol forcing.
        #[test]
        fn prop_total_forcing_additive(
            co2 in 280.0f64..1000.0,
            aerosol in -2.0f64..0.0,
        ) {
            let config = ClimateConfig {
                co2_concentration: co2,
                aerosol_forcing: aerosol,
                ..Default::default()
            };

            let co2_forcing = config.co2_forcing();
            let total = config.total_forcing();

            let expected = co2_forcing + aerosol;
            prop_assert!((total - expected).abs() < 1e-10, "total={}, expected={}", total, expected);
        }
    }
}
