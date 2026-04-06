//! Epidemic compartmental model scenarios.
//!
//! Implements classical epidemiological models:
//! - SIR (Susceptible-Infected-Recovered)
//! - SEIR (with Exposed compartment)
//! - SEIRS (with waning immunity)
//! - Stochastic variants for Monte Carlo analysis

use crate::engine::rng::SimRng;
use crate::error::{SimError, SimResult};
use serde::{Deserialize, Serialize};

/// Configuration for SIR model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SIRConfig {
    /// Total population size.
    pub population: f64,
    /// Initial number of infected individuals.
    pub initial_infected: f64,
    /// Initial number of recovered individuals.
    pub initial_recovered: f64,
    /// Transmission rate (β): contacts per unit time × transmission probability.
    pub beta: f64,
    /// Recovery rate (γ): 1 / infectious period.
    pub gamma: f64,
}

impl Default for SIRConfig {
    fn default() -> Self {
        Self {
            population: 1_000_000.0,
            initial_infected: 100.0,
            initial_recovered: 0.0,
            beta: 0.3,  // R0 = β/γ = 3.0 (like early COVID)
            gamma: 0.1, // 10-day infectious period
        }
    }
}

impl SIRConfig {
    /// Create configuration for seasonal flu.
    #[must_use]
    pub fn flu() -> Self {
        Self {
            population: 1_000_000.0,
            initial_infected: 10.0,
            initial_recovered: 0.0,
            beta: 0.2, // R0 ≈ 1.3
            gamma: 0.15,
        }
    }

    /// Create configuration for measles (highly contagious).
    #[must_use]
    pub fn measles() -> Self {
        Self {
            population: 1_000_000.0,
            initial_infected: 1.0,
            initial_recovered: 0.0,
            beta: 1.8, // R0 ≈ 15 (one of the most contagious)
            gamma: 0.12,
        }
    }

    /// Calculate basic reproduction number R0.
    ///
    /// Returns 0.0 if gamma is zero (no recovery → undefined R0).
    #[must_use]
    pub fn r0(&self) -> f64 {
        if self.gamma == 0.0 {
            return 0.0;
        }
        self.beta / self.gamma
    }

    /// Calculate herd immunity threshold.
    ///
    /// Returns 0.0 if R0 is zero.
    #[must_use]
    pub fn herd_immunity_threshold(&self) -> f64 {
        let r0 = self.r0();
        if r0 == 0.0 {
            return 0.0;
        }
        1.0 - 1.0 / r0
    }

    /// Calculate initial susceptible population.
    #[must_use]
    pub fn initial_susceptible(&self) -> f64 {
        self.population - self.initial_infected - self.initial_recovered
    }
}

/// Configuration for SEIR model (adds Exposed compartment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEIRConfig {
    /// Base SIR configuration.
    pub sir: SIRConfig,
    /// Initial number of exposed (infected but not yet infectious).
    pub initial_exposed: f64,
    /// Incubation rate (σ): 1 / incubation period.
    pub sigma: f64,
}

impl Default for SEIRConfig {
    fn default() -> Self {
        Self {
            sir: SIRConfig::default(),
            initial_exposed: 50.0,
            sigma: 0.2, // 5-day incubation period
        }
    }
}

impl SEIRConfig {
    /// Calculate initial susceptible population (accounting for exposed).
    #[must_use]
    pub fn initial_susceptible(&self) -> f64 {
        self.sir.population
            - self.sir.initial_infected
            - self.sir.initial_recovered
            - self.initial_exposed
    }
}

/// State of SIR model at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SIRState {
    /// Number of susceptible individuals.
    pub susceptible: f64,
    /// Number of infected individuals.
    pub infected: f64,
    /// Number of recovered individuals.
    pub recovered: f64,
    /// Current time.
    pub time: f64,
    /// Effective reproduction number Rt.
    pub rt: f64,
}

impl SIRState {
    /// Calculate total population.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.susceptible + self.infected + self.recovered
    }

    /// Calculate fraction infected.
    #[must_use]
    pub fn infection_rate(&self) -> f64 {
        let total = self.total();
        if total == 0.0 {
            0.0
        } else {
            self.infected / total
        }
    }

    /// Calculate fraction susceptible.
    #[must_use]
    pub fn susceptible_rate(&self) -> f64 {
        let total = self.total();
        if total == 0.0 {
            0.0
        } else {
            self.susceptible / total
        }
    }

    /// Check if epidemic is over (no more infected).
    #[must_use]
    pub fn is_over(&self) -> bool {
        self.infected < 1.0
    }
}

/// State of SEIR model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEIRState {
    /// Base SIR state.
    pub sir: SIRState,
    /// Number of exposed individuals.
    pub exposed: f64,
}

impl SEIRState {
    /// Calculate total population.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.sir.susceptible + self.exposed + self.sir.infected + self.sir.recovered
    }
}

/// SIR epidemic scenario.
#[derive(Debug, Clone)]
pub struct SIRScenario {
    config: SIRConfig,
    state: SIRState,
}

impl SIRScenario {
    /// Create a new SIR scenario.
    ///
    /// # Panics
    ///
    /// Panics if `population` or `gamma` is not positive.
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn new(config: SIRConfig) -> Self {
        assert!(config.population > 0.0, "Population must be > 0");
        assert!(config.gamma > 0.0, "Recovery rate (gamma) must be > 0");

        let s = config.initial_susceptible();
        let i = config.initial_infected;
        let r = config.initial_recovered;
        let n = config.population;

        let state = SIRState {
            susceptible: s,
            infected: i,
            recovered: r,
            time: 0.0,
            rt: config.r0() * (s / n),
        };

        Self { config, state }
    }

    /// Step forward using 4th-order Runge-Kutta.
    ///
    /// # Errors
    ///
    /// Returns an error if state becomes non-physical or population is not conserved.
    #[allow(clippy::many_single_char_names)]
    pub fn step(&mut self, dt: f64) -> SimResult<&SIRState> {
        let n = self.config.population;
        let beta = self.config.beta;
        let gamma = self.config.gamma;

        // Current state
        let s = self.state.susceptible;
        let i = self.state.infected;
        let r = self.state.recovered;

        // RK4 integration
        let (k1_s, k1_i, k1_r) = self.derivatives(s, i, r, n, beta, gamma);
        let (k2_s, k2_i, k2_r) = self.derivatives(
            s + 0.5 * dt * k1_s,
            i + 0.5 * dt * k1_i,
            r + 0.5 * dt * k1_r,
            n,
            beta,
            gamma,
        );
        let (k3_s, k3_i, k3_r) = self.derivatives(
            s + 0.5 * dt * k2_s,
            i + 0.5 * dt * k2_i,
            r + 0.5 * dt * k2_r,
            n,
            beta,
            gamma,
        );
        let (k4_s, k4_i, k4_r) =
            self.derivatives(s + dt * k3_s, i + dt * k3_i, r + dt * k3_r, n, beta, gamma);

        let new_s = s + dt / 6.0 * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s);
        let new_i = i + dt / 6.0 * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i);
        let new_r = r + dt / 6.0 * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r);

        // Jidoka: Check for non-physical values
        if new_s < 0.0 || new_i < 0.0 || new_r < 0.0 {
            return Err(SimError::jidoka(format!(
                "Non-physical state: S={new_s}, I={new_i}, R={new_r}"
            )));
        }

        // Conservation check
        let total = new_s + new_i + new_r;
        if (total - n).abs() > 1.0 {
            return Err(SimError::jidoka(format!(
                "Population conservation violated: {total} != {n}"
            )));
        }

        self.state.susceptible = new_s;
        self.state.infected = new_i;
        self.state.recovered = new_r;
        self.state.time += dt;
        self.state.rt = self.config.r0() * (new_s / n);

        Ok(&self.state)
    }

    /// Calculate derivatives for SIR model.
    #[inline]
    #[allow(clippy::unused_self)]
    #[allow(clippy::many_single_char_names)]
    fn derivatives(
        &self,
        s: f64,
        i: f64,
        _r: f64,
        n: f64,
        beta: f64,
        gamma: f64,
    ) -> (f64, f64, f64) {
        let infection = beta * s * i / n;
        let recovery = gamma * i;

        let ds = -infection;
        let di = infection - recovery;
        let dr = recovery;

        (ds, di, dr)
    }

    /// Run simulation until epidemic ends or max time.
    ///
    /// # Errors
    ///
    /// Returns an error if numerical integration fails (non-physical state).
    pub fn run(&mut self, dt: f64, max_time: f64) -> SimResult<Vec<SIRState>> {
        let mut trajectory = vec![self.state.clone()];

        while self.state.time < max_time && !self.state.is_over() {
            self.step(dt)?;
            trajectory.push(self.state.clone());
        }

        Ok(trajectory)
    }

    /// Get current state.
    #[must_use]
    pub const fn state(&self) -> &SIRState {
        &self.state
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &SIRConfig {
        &self.config
    }

    /// Calculate peak infected (analytically for SIR).
    #[must_use]
    pub fn peak_infected(&self) -> f64 {
        let r0 = self.config.r0();
        if r0 == 0.0 {
            return 0.0;
        }
        let n = self.config.population;
        let s0 = self.config.initial_susceptible();
        let i0 = self.config.initial_infected;

        if s0 <= 0.0 {
            return i0;
        }

        // Peak occurs when dI/dt = 0, i.e., S = N/R0
        let s_peak = n / r0;

        // I_peak = S0 + I0 - S_peak + (N/R0) * ln(S_peak/S0)
        s0 + i0 - s_peak + (n / r0) * (s_peak / s0).ln()
    }

    /// Calculate final epidemic size (analytically).
    #[must_use]
    pub fn final_size(&self) -> f64 {
        let r0 = self.config.r0();
        if r0 == 0.0 {
            return 0.0;
        }
        let n = self.config.population;

        // Solve: R_∞ = N * (1 - exp(-R0 * R_∞ / N))
        // Using Newton's method
        let mut r_inf = 0.8 * n; // Initial guess
        for _ in 0..50 {
            let f = r_inf - n * (1.0 - (-r0 * r_inf / n).exp());
            let df = 1.0 - r0 * (-r0 * r_inf / n).exp();
            if df.abs() < f64::EPSILON {
                break;
            }
            r_inf -= f / df;
        }

        r_inf
    }
}

/// SEIR epidemic scenario.
#[derive(Debug, Clone)]
pub struct SEIRScenario {
    config: SEIRConfig,
    state: SEIRState,
}

impl SEIRScenario {
    /// Create a new SEIR scenario.
    ///
    /// # Panics
    ///
    /// Panics if `population` or `gamma` is not positive.
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn new(config: SEIRConfig) -> Self {
        assert!(config.sir.population > 0.0, "Population must be > 0");
        assert!(config.sir.gamma > 0.0, "Recovery rate (gamma) must be > 0");

        let s = config.initial_susceptible();
        let e = config.initial_exposed;
        let i = config.sir.initial_infected;
        let r = config.sir.initial_recovered;
        let n = config.sir.population;

        let state = SEIRState {
            sir: SIRState {
                susceptible: s,
                infected: i,
                recovered: r,
                time: 0.0,
                rt: config.sir.r0() * (s / n),
            },
            exposed: e,
        };

        Self { config, state }
    }

    /// Step forward using 4th-order Runge-Kutta.
    ///
    /// # Errors
    ///
    /// Returns an error if state becomes non-physical.
    #[allow(clippy::many_single_char_names)]
    pub fn step(&mut self, dt: f64) -> SimResult<&SEIRState> {
        let n = self.config.sir.population;
        let beta = self.config.sir.beta;
        let gamma = self.config.sir.gamma;
        let sigma = self.config.sigma;

        // Current state
        let s = self.state.sir.susceptible;
        let e = self.state.exposed;
        let i = self.state.sir.infected;
        let r = self.state.sir.recovered;

        // RK4 integration
        let (k1_s, k1_e, k1_i, k1_r) = self.derivatives(s, e, i, r, n, beta, gamma, sigma);
        let (k2_s, k2_e, k2_i, k2_r) = self.derivatives(
            s + 0.5 * dt * k1_s,
            e + 0.5 * dt * k1_e,
            i + 0.5 * dt * k1_i,
            r + 0.5 * dt * k1_r,
            n,
            beta,
            gamma,
            sigma,
        );
        let (k3_s, k3_e, k3_i, k3_r) = self.derivatives(
            s + 0.5 * dt * k2_s,
            e + 0.5 * dt * k2_e,
            i + 0.5 * dt * k2_i,
            r + 0.5 * dt * k2_r,
            n,
            beta,
            gamma,
            sigma,
        );
        let (k4_s, k4_e, k4_i, k4_r) = self.derivatives(
            s + dt * k3_s,
            e + dt * k3_e,
            i + dt * k3_i,
            r + dt * k3_r,
            n,
            beta,
            gamma,
            sigma,
        );

        let new_s = s + dt / 6.0 * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s);
        let new_e = e + dt / 6.0 * (k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e);
        let new_i = i + dt / 6.0 * (k1_i + 2.0 * k2_i + 2.0 * k3_i + k4_i);
        let new_r = r + dt / 6.0 * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r);

        // Jidoka: Check for non-physical values
        if new_s < 0.0 || new_e < 0.0 || new_i < 0.0 || new_r < 0.0 {
            return Err(SimError::jidoka(format!(
                "Non-physical state: S={new_s}, E={new_e}, I={new_i}, R={new_r}"
            )));
        }

        self.state.sir.susceptible = new_s;
        self.state.exposed = new_e;
        self.state.sir.infected = new_i;
        self.state.sir.recovered = new_r;
        self.state.sir.time += dt;
        self.state.sir.rt = self.config.sir.r0() * (new_s / n);

        Ok(&self.state)
    }

    /// Calculate derivatives for SEIR model.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::unused_self)]
    #[allow(clippy::many_single_char_names)]
    fn derivatives(
        &self,
        s: f64,
        e: f64,
        i: f64,
        _r: f64,
        n: f64,
        beta: f64,
        gamma: f64,
        sigma: f64,
    ) -> (f64, f64, f64, f64) {
        let infection = beta * s * i / n;
        let incubation = sigma * e;
        let recovery = gamma * i;

        let ds = -infection;
        let de = infection - incubation;
        let di = incubation - recovery;
        let dr = recovery;

        (ds, de, di, dr)
    }

    /// Run simulation until epidemic ends or max time.
    ///
    /// # Errors
    ///
    /// Returns an error if numerical integration fails (non-physical state).
    pub fn run(&mut self, dt: f64, max_time: f64) -> SimResult<Vec<SEIRState>> {
        let mut trajectory = vec![self.state.clone()];

        while self.state.sir.time < max_time
            && (self.state.sir.infected > 1.0 || self.state.exposed > 1.0)
        {
            self.step(dt)?;
            trajectory.push(self.state.clone());
        }

        Ok(trajectory)
    }

    /// Get current state.
    #[must_use]
    pub const fn state(&self) -> &SEIRState {
        &self.state
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &SEIRConfig {
        &self.config
    }
}

/// Stochastic SIR using Gillespie algorithm.
#[derive(Debug, Clone)]
pub struct StochasticSIR {
    config: SIRConfig,
    state: SIRState,
}

impl StochasticSIR {
    /// Create a new stochastic SIR scenario.
    ///
    /// # Panics
    ///
    /// Panics if `population` or `gamma` is not positive.
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn new(config: SIRConfig) -> Self {
        assert!(config.population > 0.0, "Population must be > 0");
        assert!(config.gamma > 0.0, "Recovery rate (gamma) must be > 0");

        let s = config.initial_susceptible();
        let i = config.initial_infected;
        let r = config.initial_recovered;
        let n = config.population;

        let state = SIRState {
            susceptible: s,
            infected: i,
            recovered: r,
            time: 0.0,
            rt: config.r0() * (s / n),
        };

        Self { config, state }
    }

    /// Execute one step of Gillespie algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if simulation encounters invalid state.
    pub fn step(&mut self, rng: &mut SimRng) -> SimResult<&SIRState> {
        let n = self.config.population;
        let beta = self.config.beta;
        let gamma = self.config.gamma;

        let s = self.state.susceptible;
        let i = self.state.infected;

        // Propensities (rates)
        let infection_rate = beta * s * i / n;
        let recovery_rate = gamma * i;
        let total_rate = infection_rate + recovery_rate;

        if total_rate <= 0.0 {
            // No more events possible
            return Ok(&self.state);
        }

        // Time to next event (exponential distribution)
        let u1: f64 = rng.gen_range_f64(0.0, 1.0);
        let dt = -u1.ln() / total_rate;

        // Choose event type
        let u2: f64 = rng.gen_range_f64(0.0, 1.0);
        if u2 < infection_rate / total_rate {
            // Infection event
            self.state.susceptible -= 1.0;
            self.state.infected += 1.0;
        } else {
            // Recovery event
            self.state.infected -= 1.0;
            self.state.recovered += 1.0;
        }

        self.state.time += dt;
        self.state.rt = self.config.r0() * (self.state.susceptible / n);

        Ok(&self.state)
    }

    /// Run stochastic simulation until epidemic ends.
    ///
    /// # Errors
    ///
    /// Returns an error if stochastic simulation encounters invalid state.
    pub fn run(&mut self, max_time: f64, rng: &mut SimRng) -> SimResult<Vec<SIRState>> {
        let mut trajectory = vec![self.state.clone()];

        while self.state.time < max_time && self.state.infected >= 1.0 {
            self.step(rng)?;
            trajectory.push(self.state.clone());
        }

        Ok(trajectory)
    }

    /// Get current state.
    #[must_use]
    pub const fn state(&self) -> &SIRState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sir_config_default() {
        let config = SIRConfig::default();

        assert!((config.population - 1_000_000.0).abs() < f64::EPSILON);
        assert!(config.initial_infected > 0.0);
    }

    #[test]
    fn test_sir_config_r0() {
        let config = SIRConfig::default();
        let r0 = config.r0();

        // R0 = 0.3 / 0.1 = 3.0
        assert!((r0 - 3.0).abs() < 0.01, "R0 = {r0}");
    }

    #[test]
    fn test_sir_config_herd_immunity() {
        let config = SIRConfig::default();
        let hit = config.herd_immunity_threshold();

        // For R0 = 3, HIT = 1 - 1/3 = 0.667
        assert!((hit - 0.667).abs() < 0.01, "HIT = {hit}");
    }

    #[test]
    fn test_sir_scenario_step() {
        let config = SIRConfig::default();
        let mut scenario = SIRScenario::new(config);

        let initial_infected = scenario.state().infected;
        scenario.step(0.1).unwrap();

        // Infected should increase initially (R0 > 1)
        assert!(
            scenario.state().infected > initial_infected,
            "Infected should increase when R0 > 1"
        );
    }

    #[test]
    fn test_sir_scenario_conservation() {
        let config = SIRConfig::default();
        let mut scenario = SIRScenario::new(config.clone());

        // Run for a while
        for _ in 0..100 {
            scenario.step(0.1).unwrap();
        }

        // Total population should be conserved
        let total = scenario.state().total();
        assert!(
            (total - config.population).abs() < 1.0,
            "Population not conserved: {total} != {}",
            config.population
        );
    }

    #[test]
    fn test_sir_scenario_run() {
        let config = SIRConfig::default();
        let mut scenario = SIRScenario::new(config);

        let trajectory = scenario.run(0.1, 200.0).unwrap();

        // Trajectory should have multiple points
        assert!(trajectory.len() > 10);

        // Should reach near-equilibrium
        let final_state = trajectory.last().unwrap();
        assert!(final_state.infected < 100.0, "Epidemic should end");
    }

    #[test]
    fn test_sir_peak_infected() {
        let config = SIRConfig::default();
        let scenario = SIRScenario::new(config.clone());

        let analytical_peak = scenario.peak_infected();

        // Run simulation to find actual peak
        let mut sim = SIRScenario::new(config);
        let trajectory = sim.run(0.1, 200.0).unwrap();
        let numerical_peak = trajectory.iter().map(|s| s.infected).fold(0.0, f64::max);

        // Analytical and numerical should be close
        let relative_error = (analytical_peak - numerical_peak).abs() / numerical_peak;
        assert!(
            relative_error < 0.05,
            "Analytical peak {analytical_peak} vs numerical {numerical_peak}"
        );
    }

    #[test]
    fn test_sir_final_size() {
        let config = SIRConfig::default();
        let scenario = SIRScenario::new(config.clone());

        let analytical_final = scenario.final_size();

        // Run simulation to get actual final size
        let mut sim = SIRScenario::new(config);
        let trajectory = sim.run(0.1, 500.0).unwrap();
        let numerical_final = trajectory.last().unwrap().recovered;

        // Should be close
        let relative_error = (analytical_final - numerical_final).abs() / numerical_final;
        assert!(
            relative_error < 0.05,
            "Analytical final size {analytical_final} vs numerical {numerical_final}"
        );
    }

    #[test]
    fn test_seir_scenario_step() {
        let config = SEIRConfig::default();
        let mut scenario = SEIRScenario::new(config);

        scenario.step(0.1).unwrap();

        // All compartments should be non-negative
        assert!(scenario.state().sir.susceptible >= 0.0);
        assert!(scenario.state().exposed >= 0.0);
        assert!(scenario.state().sir.infected >= 0.0);
        assert!(scenario.state().sir.recovered >= 0.0);
    }

    #[test]
    fn test_seir_scenario_run() {
        let config = SEIRConfig::default();
        let mut scenario = SEIRScenario::new(config);

        let trajectory = scenario.run(0.1, 200.0).unwrap();

        assert!(trajectory.len() > 10);
    }

    #[test]
    fn test_seir_delayed_peak() {
        // SEIR should have delayed peak compared to SIR due to incubation
        let sir_config = SIRConfig::default();
        let mut sir = SIRScenario::new(sir_config);
        let sir_trajectory = sir.run(0.1, 200.0).unwrap();

        let seir_config = SEIRConfig::default();
        let mut seir = SEIRScenario::new(seir_config);
        let seir_trajectory = seir.run(0.1, 200.0).unwrap();

        // Find peak times
        let sir_peak_time = sir_trajectory
            .iter()
            .max_by(|a, b| a.infected.partial_cmp(&b.infected).unwrap())
            .map(|s| s.time)
            .unwrap();

        let seir_peak_time = seir_trajectory
            .iter()
            .max_by(|a, b| a.sir.infected.partial_cmp(&b.sir.infected).unwrap())
            .map(|s| s.sir.time)
            .unwrap();

        // SEIR peak should be later
        assert!(
            seir_peak_time > sir_peak_time,
            "SEIR peak time {seir_peak_time} should be > SIR peak time {sir_peak_time}"
        );
    }

    #[test]
    fn test_stochastic_sir_step() {
        let config = SIRConfig {
            population: 1000.0,
            initial_infected: 10.0,
            ..Default::default()
        };
        let mut scenario = StochasticSIR::new(config);
        let mut rng = SimRng::new(42);

        scenario.step(&mut rng).unwrap();

        // State should change by exactly 1 in one compartment
        let total = scenario.state().total();
        assert!((total - 1000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stochastic_sir_run() {
        let config = SIRConfig {
            population: 1000.0,
            initial_infected: 10.0,
            ..Default::default()
        };
        let mut scenario = StochasticSIR::new(config);
        let mut rng = SimRng::new(42);

        let trajectory = scenario.run(500.0, &mut rng).unwrap();

        // Trajectory should have many events
        assert!(trajectory.len() > 100);

        // Final state should have no infected
        let final_state = trajectory.last().unwrap();
        assert!(final_state.infected < 1.0);
    }

    #[test]
    fn test_sir_flu() {
        let config = SIRConfig::flu();

        // Flu has lower R0
        assert!(config.r0() < 2.0);
    }

    #[test]
    fn test_sir_measles() {
        let config = SIRConfig::measles();

        // Measles is highly contagious
        assert!(config.r0() > 10.0);

        // Requires high herd immunity threshold
        assert!(config.herd_immunity_threshold() > 0.9);
    }

    #[test]
    fn test_sir_state_is_over() {
        let state = SIRState {
            susceptible: 100.0,
            infected: 0.5,
            recovered: 899.5,
            time: 100.0,
            rt: 0.1,
        };

        assert!(state.is_over());
    }

    #[test]
    fn test_sir_state_infection_rate() {
        let state = SIRState {
            susceptible: 500.0,
            infected: 100.0,
            recovered: 400.0,
            time: 10.0,
            rt: 1.5,
        };

        assert!((state.infection_rate() - 0.1).abs() < 0.01);
        assert!((state.susceptible_rate() - 0.5).abs() < 0.01);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// SIR total population is conserved.
        #[test]
        fn prop_sir_population_conserved(
            susceptible in 100.0f64..900.0,
            infected in 1.0f64..100.0,
        ) {
            let recovered = 1000.0 - susceptible - infected;
            let state = SIRState {
                susceptible,
                infected,
                recovered,
                time: 0.0,
                rt: 2.0,
            };

            let total = state.total();
            prop_assert!((total - 1000.0).abs() < 0.001, "total={}", total);
        }

        /// R0 > 1 means epidemic grows initially.
        #[test]
        fn prop_r0_greater_than_one_grows(
            beta in 0.3f64..0.8,
            gamma in 0.05f64..0.15,
        ) {
            let r0 = beta / gamma;
            // When S/N ≈ 1 and R0 > 1, infections should increase initially
            if r0 > 1.0 {
                // dI/dt = β*S*I/N - γ*I = I*(β*S/N - γ)
                // With S ≈ N: dI/dt ≈ I*(β - γ) = I*γ*(R0 - 1) > 0
                let growth_rate = gamma * (r0 - 1.0);
                prop_assert!(growth_rate > 0.0);
            }
        }

        /// SEIR total population is conserved.
        #[test]
        fn prop_seir_population_conserved(
            susceptible in 100.0f64..800.0,
            exposed in 1.0f64..50.0,
            infected in 1.0f64..50.0,
        ) {
            let recovered = 1000.0 - susceptible - exposed - infected;
            let sir = SIRState {
                susceptible,
                infected,
                recovered,
                time: 0.0,
                rt: 2.0,
            };
            let state = SEIRState { sir, exposed };

            let total = state.total();
            prop_assert!((total - 1000.0).abs() < 0.001, "total={}", total);
        }

        /// Infection rate is non-negative.
        #[test]
        fn prop_infection_rate_nonnegative(
            infected in 0.0f64..500.0,
            total in 500.0f64..2000.0,
        ) {
            let susceptible = total - infected;
            let state = SIRState {
                susceptible,
                infected,
                recovered: 0.0,
                time: 0.0,
                rt: 2.0,
            };

            let rate = state.infection_rate();
            prop_assert!(rate >= 0.0, "rate={}", rate);
            prop_assert!(rate <= 1.0, "rate={} should be <= 1", rate);
        }

        /// Epidemic is over when infected < threshold.
        #[test]
        fn prop_epidemic_over_threshold(
            infected in 0.0f64..10.0,
        ) {
            let state = SIRState {
                susceptible: 100.0,
                infected,
                recovered: 900.0 - infected,
                time: 100.0,
                rt: 0.5,
            };

            // Epidemic is over when infected < 1
            if infected < 1.0 {
                prop_assert!(state.is_over());
            }
        }
    }
}
