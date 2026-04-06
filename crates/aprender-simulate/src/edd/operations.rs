//! Operations Science equations for manufacturing and service systems.
//!
//! This module implements the fundamental equations from operations science
//! that govern queueing, inventory, and production systems. These equations
//! are derived from decades of peer-reviewed research and form the mathematical
//! foundation for simulating production and service systems.
//!
//! # References
//!
//! - [30] Little, J.D.C. (1961). "A Proof for the Queuing Formula: L = λW"
//! - [31] Kingman, J.F.C. (1961). "The single server queue in heavy traffic"
//! - [32] Lee, H.L., et al. (1997). "The Bullwhip Effect in Supply Chains"
//! - [33] Hopp, W.J. & Spearman, M.L. (2004). "To Pull or Not to Pull"

use super::equation::{
    Citation, EquationClass, EquationVariable, GoverningEquation, VariableConstraints,
};

// =============================================================================
// Little's Law: L = λW
// =============================================================================

/// Little's Law: The fundamental equation relating WIP, throughput, and cycle time.
///
/// `L = λW` where:
/// - L: Average number of items in the system (WIP)
/// - λ: Average arrival rate (throughput at steady state)
/// - W: Average time an item spends in the system (cycle time)
///
/// This law holds for ANY stable queueing system regardless of arrival distribution,
/// service distribution, or queue discipline. It is one of the most fundamental
/// results in queueing theory.
#[derive(Debug, Clone)]
pub struct LittlesLaw {
    latex: String,
    description: String,
}

impl LittlesLaw {
    /// Create a new Little's Law equation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            latex: r"L = \lambda W".to_string(),
            description: "Little's Law relates the average number of items in a stable system (L) \
                         to the arrival rate (λ) and the average time spent in the system (W)."
                .to_string(),
        }
    }

    /// Evaluate L = λW given arrival rate and wait time.
    #[must_use]
    pub fn evaluate(&self, lambda: f64, w: f64) -> f64 {
        lambda * w
    }

    /// Validate that observed values satisfy Little's Law within tolerance.
    ///
    /// # Arguments
    /// - `l`: Observed average WIP
    /// - `lambda`: Observed throughput/arrival rate
    /// - `w`: Observed average cycle time
    /// - `tolerance`: Relative tolerance for validation
    ///
    /// # Errors
    /// Returns error message if `|L - λW| / L >= tolerance`.
    pub fn validate(&self, l: f64, lambda: f64, w: f64, tolerance: f64) -> Result<(), String> {
        let expected = lambda * w;
        let relative_error = if l.abs() > f64::EPSILON {
            (l - expected).abs() / l
        } else if expected.abs() > f64::EPSILON {
            (l - expected).abs() / expected
        } else {
            0.0
        };

        if relative_error <= tolerance {
            Ok(())
        } else {
            Err(format!(
                "Little's Law violation: L={l:.4}, λW={expected:.4}, relative_error={relative_error:.4} > tolerance={tolerance:.4}"
            ))
        }
    }

    /// Solve for L given λ and W.
    #[must_use]
    pub fn solve_l(&self, lambda: f64, w: f64) -> f64 {
        lambda * w
    }

    /// Solve for λ given L and W.
    #[must_use]
    pub fn solve_lambda(&self, l: f64, w: f64) -> f64 {
        if w.abs() < f64::EPSILON {
            f64::INFINITY
        } else {
            l / w
        }
    }

    /// Solve for W given L and λ.
    #[must_use]
    pub fn solve_w(&self, l: f64, lambda: f64) -> f64 {
        if lambda.abs() < f64::EPSILON {
            f64::INFINITY
        } else {
            l / lambda
        }
    }
}

impl Default for LittlesLaw {
    fn default() -> Self {
        Self::new()
    }
}

impl GoverningEquation for LittlesLaw {
    fn latex(&self) -> &str {
        &self.latex
    }

    fn class(&self) -> EquationClass {
        EquationClass::Queueing
    }

    fn citation(&self) -> Citation {
        Citation::new(&["Little, J.D.C."], "Operations Research", 1961)
            .with_title("A Proof for the Queuing Formula: L = λW")
            .with_doi("10.1287/opre.9.3.383")
    }

    fn variables(&self) -> Vec<EquationVariable> {
        vec![
            EquationVariable::new("L", "wip", "items")
                .with_description("Average number of items in the system (work in progress)")
                .with_constraints(VariableConstraints {
                    min: Some(0.0),
                    max: None,
                    positive: false,
                    integer: false,
                }),
            EquationVariable::new("λ", "arrival_rate", "items/time")
                .with_description("Average arrival rate (throughput at steady state)")
                .with_constraints(VariableConstraints {
                    min: Some(0.0),
                    max: None,
                    positive: true,
                    integer: false,
                }),
            EquationVariable::new("W", "cycle_time", "time")
                .with_description("Average time an item spends in the system")
                .with_constraints(VariableConstraints {
                    min: Some(0.0),
                    max: None,
                    positive: true,
                    integer: false,
                }),
        ]
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn name(&self) -> &'static str {
        "Little's Law"
    }

    fn validate_consistency(&self, values: &[(&str, f64)], tolerance: f64) -> Result<(), String> {
        let mut l = None;
        let mut lambda = None;
        let mut w = None;

        for (name, value) in values {
            match *name {
                "L" | "l" | "wip" => l = Some(*value),
                "λ" | "lambda" | "arrival_rate" | "throughput" => lambda = Some(*value),
                "W" | "w" | "cycle_time" => w = Some(*value),
                _ => {}
            }
        }

        match (l, lambda, w) {
            (Some(l_val), Some(lambda_val), Some(w_val)) => {
                self.validate(l_val, lambda_val, w_val, tolerance)
            }
            _ => Err("Missing required variables for Little's Law validation".to_string()),
        }
    }
}

// =============================================================================
// Kingman's Formula: VUT Equation
// =============================================================================

/// Kingman's VUT Formula for queue wait times.
///
/// The expected wait time in queue is approximated by:
/// `W_q ≈ (ρ / (1-ρ)) × ((c_a² + c_s²) / 2) × t_s`
///
/// where:
/// - ρ: Utilization (arrival rate / service rate)
/// - `c_a`: Coefficient of variation of inter-arrival times
/// - `c_s`: Coefficient of variation of service times
/// - `t_s`: Mean service time
///
/// This formula reveals the "hockey stick" effect: wait times grow exponentially
/// as utilization approaches 100%.
#[derive(Debug, Clone)]
pub struct KingmanFormula {
    latex: String,
    description: String,
}

impl KingmanFormula {
    /// Create a new Kingman's Formula equation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            latex: r"W_q \approx \frac{\rho}{1-\rho} \cdot \frac{c_a^2 + c_s^2}{2} \cdot t_s"
                .to_string(),
            description: "Kingman's VUT equation approximates expected queue wait time as a \
                         function of Variability, Utilization, and Time."
                .to_string(),
        }
    }

    /// Calculate expected wait time in queue.
    ///
    /// # Arguments
    /// - `rho`: Utilization (0 < ρ < 1)
    /// - `c_a`: Coefficient of variation of arrivals
    /// - `c_s`: Coefficient of variation of service
    /// - `t_s`: Mean service time
    ///
    /// # Returns
    /// Expected wait time in queue.
    #[must_use]
    pub fn expected_wait_time(&self, rho: f64, c_a: f64, c_s: f64, t_s: f64) -> f64 {
        if rho >= 1.0 {
            return f64::INFINITY;
        }
        if rho <= 0.0 {
            return 0.0;
        }

        let utilization_factor = rho / (1.0 - rho);
        let variability_factor = (c_a * c_a + c_s * c_s) / 2.0;

        utilization_factor * variability_factor * t_s
    }

    /// Get the utilization factor ρ/(1-ρ).
    #[must_use]
    pub fn utilization_factor(&self, rho: f64) -> f64 {
        if rho >= 1.0 {
            f64::INFINITY
        } else if rho <= 0.0 {
            0.0
        } else {
            rho / (1.0 - rho)
        }
    }

    /// Get the variability factor `(c_a² + c_s²)/2`.
    #[must_use]
    pub fn variability_factor(&self, c_a: f64, c_s: f64) -> f64 {
        (c_a * c_a + c_s * c_s) / 2.0
    }
}

impl Default for KingmanFormula {
    fn default() -> Self {
        Self::new()
    }
}

impl GoverningEquation for KingmanFormula {
    fn latex(&self) -> &str {
        &self.latex
    }

    fn class(&self) -> EquationClass {
        EquationClass::Queueing
    }

    fn citation(&self) -> Citation {
        Citation::new(
            &["Kingman, J.F.C."],
            "Annals of Mathematical Statistics",
            1961,
        )
        .with_title("The single server queue in heavy traffic")
        .with_doi("10.1214/aoms/1177704567")
    }

    fn variables(&self) -> Vec<EquationVariable> {
        vec![
            EquationVariable::new("ρ", "utilization", "dimensionless")
                .with_description("Server utilization ratio (arrival rate / service rate)")
                .with_constraints(VariableConstraints {
                    min: Some(0.0),
                    max: Some(1.0),
                    positive: false,
                    integer: false,
                }),
            EquationVariable::new("c_a", "arrival_cv", "dimensionless")
                .with_description("Coefficient of variation of inter-arrival times"),
            EquationVariable::new("c_s", "service_cv", "dimensionless")
                .with_description("Coefficient of variation of service times"),
            EquationVariable::new("t_s", "service_time", "time")
                .with_description("Mean service time"),
            EquationVariable::new("W_q", "queue_wait", "time")
                .with_description("Expected wait time in queue"),
        ]
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn name(&self) -> &'static str {
        "Kingman's VUT Formula"
    }

    fn validate_consistency(&self, values: &[(&str, f64)], tolerance: f64) -> Result<(), String> {
        let mut rho = None;
        let mut c_a = None;
        let mut c_s = None;
        let mut t_s = None;
        let mut w_q = None;

        for (name, value) in values {
            match *name {
                "ρ" | "rho" | "utilization" => rho = Some(*value),
                "c_a" | "arrival_cv" => c_a = Some(*value),
                "c_s" | "service_cv" => c_s = Some(*value),
                "t_s" | "service_time" => t_s = Some(*value),
                "W_q" | "queue_wait" => w_q = Some(*value),
                _ => {}
            }
        }

        match (rho, c_a, c_s, t_s, w_q) {
            (Some(r), Some(ca), Some(cs), Some(ts), Some(wq)) => {
                let expected = self.expected_wait_time(r, ca, cs, ts);
                let relative_error = if expected.abs() > f64::EPSILON {
                    (wq - expected).abs() / expected
                } else if wq.abs() > f64::EPSILON {
                    (wq - expected).abs() / wq
                } else {
                    0.0
                };

                if relative_error <= tolerance {
                    Ok(())
                } else {
                    Err(format!(
                        "Kingman violation: W_q={wq:.4}, expected={expected:.4}, error={relative_error:.4}"
                    ))
                }
            }
            _ => Err("Missing required variables for Kingman validation".to_string()),
        }
    }
}

// =============================================================================
// Square Root Law for Safety Stock
// =============================================================================

/// The Square Root Law for safety stock scaling.
///
/// Safety stock scales with the square root of demand, not linearly:
/// `I_safety = z × σ_D × √L`
///
/// where:
/// - z: Service level z-score (e.g., 1.96 for 95% service)
/// - `σ_D`: Standard deviation of demand per period
/// - L: Lead time in periods
///
/// This means pooling inventory reduces total safety stock requirements.
#[derive(Debug, Clone)]
pub struct SquareRootLaw {
    latex: String,
    description: String,
}

impl SquareRootLaw {
    /// Create a new Square Root Law equation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            latex: r"I_{safety} = z \cdot \sigma_D \cdot \sqrt{L}".to_string(),
            description: "The Square Root Law shows that safety stock requirements scale with \
                         the square root of demand or lead time, not linearly."
                .to_string(),
        }
    }

    /// Calculate required safety stock.
    ///
    /// # Arguments
    /// - `demand_std`: Standard deviation of demand per period (`σ_D`)
    /// - `lead_time`: Lead time in periods (L)
    /// - `z_score`: Service level z-score (z)
    ///
    /// # Returns
    /// Required safety stock units.
    #[must_use]
    pub fn safety_stock(&self, demand_std: f64, lead_time: f64, z_score: f64) -> f64 {
        z_score * demand_std * lead_time.sqrt()
    }

    /// Calculate safety stock reduction from pooling.
    ///
    /// When combining n locations with equal demand variance, the total
    /// safety stock is reduced by a factor of √n.
    #[must_use]
    pub fn pooling_reduction(&self, num_locations: usize) -> f64 {
        if num_locations == 0 {
            return 0.0;
        }
        1.0 / (num_locations as f64).sqrt()
    }

    /// Common z-scores for service levels.
    #[must_use]
    pub fn z_score_for_service_level(service_level: f64) -> f64 {
        match service_level {
            x if (x - 0.90).abs() < 0.001 => 1.28,
            x if (x - 0.95).abs() < 0.001 => 1.65,
            x if (x - 0.99).abs() < 0.001 => 2.33,
            x if (x - 0.999).abs() < 0.001 => 3.09,
            _ => {
                // Approximation using inverse normal
                // For 95% confidence interval (2-sided), use 1.96
                1.96
            }
        }
    }
}

impl Default for SquareRootLaw {
    fn default() -> Self {
        Self::new()
    }
}

impl GoverningEquation for SquareRootLaw {
    fn latex(&self) -> &str {
        &self.latex
    }

    fn class(&self) -> EquationClass {
        EquationClass::Inventory
    }

    fn citation(&self) -> Citation {
        Citation::new(&["Eppen, G.D."], "Management Science", 1979)
            .with_title(
                "Effects of Centralization on Expected Costs in a Multi-Location Newsboy Problem",
            )
            .with_doi("10.1287/mnsc.25.5.498")
    }

    fn variables(&self) -> Vec<EquationVariable> {
        vec![
            EquationVariable::new("I_safety", "safety_stock", "units")
                .with_description("Required safety stock quantity"),
            EquationVariable::new("z", "z_score", "dimensionless")
                .with_description("Service level z-score from normal distribution"),
            EquationVariable::new("σ_D", "demand_std", "units/period")
                .with_description("Standard deviation of demand per period"),
            EquationVariable::new("L", "lead_time", "periods")
                .with_description("Lead time in periods"),
        ]
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn name(&self) -> &'static str {
        "Square Root Law"
    }

    fn validate_consistency(&self, values: &[(&str, f64)], tolerance: f64) -> Result<(), String> {
        let mut z = None;
        let mut sigma_d = None;
        let mut lead_time = None;
        let mut i_safety = None;

        for (name, value) in values {
            match *name {
                "z" | "z_score" => z = Some(*value),
                "σ_D" | "demand_std" => sigma_d = Some(*value),
                "L" | "lead_time" => lead_time = Some(*value),
                "I_safety" | "safety_stock" => i_safety = Some(*value),
                _ => {}
            }
        }

        match (z, sigma_d, lead_time, i_safety) {
            (Some(z_val), Some(sd), Some(lt), Some(is)) => {
                let expected = self.safety_stock(sd, lt, z_val);
                let relative_error = if expected.abs() > f64::EPSILON {
                    (is - expected).abs() / expected
                } else {
                    0.0
                };

                if relative_error <= tolerance {
                    Ok(())
                } else {
                    Err(format!(
                        "Square Root Law violation: I_safety={is:.4}, expected={expected:.4}"
                    ))
                }
            }
            _ => Err("Missing required variables for Square Root Law validation".to_string()),
        }
    }
}

// =============================================================================
// Bullwhip Effect
// =============================================================================

/// The Bullwhip Effect: Variance amplification in supply chains.
///
/// Demand variance amplifies as you move upstream in a supply chain:
/// `Var(Orders) / Var(Demand) ≥ 1`
///
/// The amplification factor depends on lead time, order batching,
/// price fluctuations, and demand signal processing.
#[derive(Debug, Clone)]
pub struct BullwhipEffect {
    latex: String,
    description: String,
}

impl BullwhipEffect {
    /// Create a new Bullwhip Effect equation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            latex: r"\frac{Var(Orders)}{Var(Demand)} \geq 1 + \frac{2L}{p} + \frac{2L^2}{p^2}"
                .to_string(),
            description: "The Bullwhip Effect describes how demand variance amplifies upstream \
                         in supply chains due to demand signal processing."
                .to_string(),
        }
    }

    /// Calculate the minimum amplification factor for order-up-to policy.
    ///
    /// # Arguments
    /// - `lead_time`: Lead time in periods (L)
    /// - `review_period`: Review period (p)
    ///
    /// # Returns
    /// Minimum variance amplification factor.
    #[must_use]
    pub fn amplification_factor(&self, lead_time: f64, review_period: f64) -> f64 {
        if review_period <= 0.0 {
            return f64::INFINITY;
        }
        let ratio = lead_time / review_period;
        1.0 + 2.0 * ratio + 2.0 * ratio * ratio
    }

    /// Calculate total variance amplification across n echelons.
    #[must_use]
    pub fn multi_echelon_amplification(&self, lead_times: &[f64], review_periods: &[f64]) -> f64 {
        if lead_times.len() != review_periods.len() {
            return f64::NAN;
        }

        lead_times
            .iter()
            .zip(review_periods.iter())
            .map(|(lt, rp)| self.amplification_factor(*lt, *rp))
            .product()
    }
}

impl Default for BullwhipEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl GoverningEquation for BullwhipEffect {
    fn latex(&self) -> &str {
        &self.latex
    }

    fn class(&self) -> EquationClass {
        EquationClass::Inventory
    }

    fn citation(&self) -> Citation {
        Citation::new(
            &["Lee, H.L.", "Padmanabhan, V.", "Whang, S."],
            "Management Science",
            1997,
        )
        .with_title("The Bullwhip Effect in Supply Chains")
        .with_doi("10.1287/mnsc.43.4.546")
    }

    fn variables(&self) -> Vec<EquationVariable> {
        vec![
            EquationVariable::new("L", "lead_time", "periods")
                .with_description("Lead time between ordering and receiving"),
            EquationVariable::new("p", "review_period", "periods")
                .with_description("Time between inventory reviews"),
            EquationVariable::new("Var(Orders)", "order_variance", "units²")
                .with_description("Variance of orders placed"),
            EquationVariable::new("Var(Demand)", "demand_variance", "units²")
                .with_description("Variance of end-customer demand"),
        ]
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn name(&self) -> &'static str {
        "Bullwhip Effect"
    }

    fn validate_consistency(&self, values: &[(&str, f64)], tolerance: f64) -> Result<(), String> {
        let mut lead_time = None;
        let mut review_period = None;
        let mut order_var = None;
        let mut demand_var = None;

        for (name, value) in values {
            match *name {
                "L" | "lead_time" => lead_time = Some(*value),
                "p" | "review_period" => review_period = Some(*value),
                "Var(Orders)" | "order_variance" => order_var = Some(*value),
                "Var(Demand)" | "demand_variance" => demand_var = Some(*value),
                _ => {}
            }
        }

        match (lead_time, review_period, order_var, demand_var) {
            (Some(lt), Some(rp), Some(ov), Some(dv)) => {
                let min_factor = self.amplification_factor(lt, rp);
                let observed_factor = if dv > f64::EPSILON { ov / dv } else { 0.0 };

                // Allow some tolerance below minimum (measurement noise)
                if observed_factor >= min_factor * (1.0 - tolerance) {
                    Ok(())
                } else {
                    Err(format!(
                        "Bullwhip violation: observed amplification {observed_factor:.4} < minimum {min_factor:.4}"
                    ))
                }
            }
            _ => Err("Missing required variables for Bullwhip validation".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Little's Law Tests
    // =========================================================================

    #[test]
    fn test_littles_law_basic() {
        let law = LittlesLaw::new();

        // L = λW: If λ=5 and W=2, then L=10
        let result = law.evaluate(5.0, 2.0);
        assert!((result - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_littles_law_validation_pass() {
        let law = LittlesLaw::new();
        let result = law.validate(10.0, 5.0, 2.0, 0.01);
        assert!(result.is_ok());
    }

    #[test]
    fn test_littles_law_validation_fail() {
        let law = LittlesLaw::new();
        let result = law.validate(15.0, 5.0, 2.0, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_littles_law_solvers() {
        let law = LittlesLaw::new();

        assert!((law.solve_l(5.0, 2.0) - 10.0).abs() < f64::EPSILON);
        assert!((law.solve_lambda(10.0, 2.0) - 5.0).abs() < f64::EPSILON);
        assert!((law.solve_w(10.0, 5.0) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_littles_law_has_citation() {
        let law = LittlesLaw::new();
        let citation = law.citation();
        assert_eq!(citation.year, 1961);
        assert!(!citation.authors.is_empty());
    }

    #[test]
    fn test_littles_law_has_variables() {
        let law = LittlesLaw::new();
        let vars = law.variables();
        assert!(vars.len() >= 3);
    }

    // =========================================================================
    // Kingman's Formula Tests
    // =========================================================================

    #[test]
    fn test_kingman_basic() {
        let formula = KingmanFormula::new();

        // At 50% utilization with CV=1, wait = (0.5/0.5) * 1 * t_s = t_s
        let wait = formula.expected_wait_time(0.5, 1.0, 1.0, 1.0);
        assert!((wait - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_kingman_hockey_stick() {
        let formula = KingmanFormula::new();

        let wait_50 = formula.expected_wait_time(0.5, 1.0, 1.0, 1.0);
        let wait_95 = formula.expected_wait_time(0.95, 1.0, 1.0, 1.0);

        // At 95% utilization, wait should be much higher than at 50%
        assert!(wait_95 > wait_50 * 10.0);
    }

    #[test]
    fn test_kingman_100_percent_utilization() {
        let formula = KingmanFormula::new();
        let wait = formula.expected_wait_time(1.0, 1.0, 1.0, 1.0);
        assert!(wait.is_infinite());
    }

    #[test]
    fn test_kingman_has_citation() {
        let formula = KingmanFormula::new();
        let citation = formula.citation();
        assert_eq!(citation.year, 1961);
    }

    // =========================================================================
    // Square Root Law Tests
    // =========================================================================

    #[test]
    fn test_square_root_law_basic() {
        let law = SquareRootLaw::new();

        // z=1.96, σ_D=100, L=1 => I = 1.96 * 100 * 1 = 196
        let stock = law.safety_stock(100.0, 1.0, 1.96);
        assert!((stock - 196.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_square_root_law_scaling() {
        let law = SquareRootLaw::new();

        // demand_std scaling is linear (not the square root property)
        let _stock_100 = law.safety_stock(100.0, 1.0, 1.96);
        let _stock_400 = law.safety_stock(400.0, 1.0, 1.96);

        // The sqrt applies to lead time - this is the key property:
        // If lead_time quadruples (1→4), safety stock doubles (√4 = 2)
        let stock_l1 = law.safety_stock(100.0, 1.0, 1.96);
        let stock_l4 = law.safety_stock(100.0, 4.0, 1.96);

        let lt_ratio = stock_l4 / stock_l1;
        assert!((lt_ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_square_root_law_pooling() {
        let law = SquareRootLaw::new();

        // Pooling 4 locations should reduce safety stock by factor of 2
        let reduction = law.pooling_reduction(4);
        assert!((reduction - 0.5).abs() < f64::EPSILON);
    }

    // =========================================================================
    // Bullwhip Effect Tests
    // =========================================================================

    #[test]
    fn test_bullwhip_basic() {
        let effect = BullwhipEffect::new();

        // With L=1, p=1: factor = 1 + 2*1 + 2*1 = 5
        let factor = effect.amplification_factor(1.0, 1.0);
        assert!((factor - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bullwhip_multi_echelon() {
        let effect = BullwhipEffect::new();

        // Two echelons, each with L=1, p=1 => 5 * 5 = 25
        let factor = effect.multi_echelon_amplification(&[1.0, 1.0], &[1.0, 1.0]);
        assert!((factor - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bullwhip_has_citation() {
        let effect = BullwhipEffect::new();
        let citation = effect.citation();
        assert_eq!(citation.year, 1997);
        assert!(citation.authors.len() >= 3);
    }

    // =========================================================================
    // GoverningEquation Trait Tests (for coverage)
    // =========================================================================

    #[test]
    fn test_littles_law_trait_methods() {
        let law = LittlesLaw::new();

        // Test latex
        assert!(law.latex().contains("lambda"));

        // Test class
        assert_eq!(law.class(), EquationClass::Queueing);

        // Test name
        assert_eq!(law.name(), "Little's Law");

        // Test description
        assert!(!law.description().is_empty());
    }

    #[test]
    fn test_littles_law_validate_consistency() {
        let law = LittlesLaw::new();

        // Valid consistency
        let values = vec![("L", 10.0), ("lambda", 5.0), ("W", 2.0)];
        assert!(law.validate_consistency(&values, 0.01).is_ok());

        // Alternative names
        let values2 = vec![("wip", 10.0), ("arrival_rate", 5.0), ("cycle_time", 2.0)];
        assert!(law.validate_consistency(&values2, 0.01).is_ok());

        // Missing variables
        let incomplete = vec![("L", 10.0), ("lambda", 5.0)];
        assert!(law.validate_consistency(&incomplete, 0.01).is_err());
    }

    #[test]
    fn test_littles_law_default() {
        let law = LittlesLaw::default();
        assert_eq!(law.name(), "Little's Law");
    }

    #[test]
    fn test_littles_law_solve_edge_cases() {
        let law = LittlesLaw::new();

        // Division by zero handling
        assert!(law.solve_lambda(10.0, 0.0).is_infinite());
        assert!(law.solve_w(10.0, 0.0).is_infinite());
    }

    #[test]
    fn test_kingman_trait_methods() {
        let formula = KingmanFormula::new();

        assert!(formula.latex().contains("rho"));
        assert_eq!(formula.class(), EquationClass::Queueing);
        assert_eq!(formula.name(), "Kingman's VUT Formula");
        assert!(!formula.description().is_empty());
    }

    #[test]
    fn test_kingman_default() {
        let formula = KingmanFormula::default();
        assert_eq!(formula.name(), "Kingman's VUT Formula");
    }

    #[test]
    fn test_kingman_variables() {
        let formula = KingmanFormula::new();
        let vars = formula.variables();
        assert!(vars.len() >= 4);
    }

    #[test]
    fn test_kingman_validate_consistency() {
        let formula = KingmanFormula::new();

        // At 50% utilization, CV=1, t_s=1: wait = 1.0
        let values = vec![
            ("rho", 0.5),
            ("c_a", 1.0),
            ("c_s", 1.0),
            ("t_s", 1.0),
            ("W_q", 1.0),
        ];
        assert!(formula.validate_consistency(&values, 0.1).is_ok());

        // Missing variables
        let incomplete = vec![("rho", 0.5)];
        assert!(formula.validate_consistency(&incomplete, 0.1).is_err());
    }

    #[test]
    fn test_kingman_utilization_and_variability() {
        let formula = KingmanFormula::new();

        // Utilization factor
        assert_eq!(formula.utilization_factor(0.5), 1.0);
        assert!(formula.utilization_factor(0.9) > 1.0);

        // Variability factor
        assert_eq!(formula.variability_factor(1.0, 1.0), 1.0);
        assert!(formula.variability_factor(2.0, 2.0) > 1.0);
    }

    #[test]
    fn test_square_root_law_trait_methods() {
        let law = SquareRootLaw::new();

        assert!(law.latex().contains("sqrt"));
        assert_eq!(law.class(), EquationClass::Inventory);
        assert_eq!(law.name(), "Square Root Law");
        assert!(!law.description().is_empty());
    }

    #[test]
    fn test_square_root_law_default() {
        let law = SquareRootLaw::default();
        assert_eq!(law.name(), "Square Root Law");
    }

    #[test]
    fn test_square_root_law_variables() {
        let law = SquareRootLaw::new();
        let vars = law.variables();
        assert!(!vars.is_empty());
    }

    #[test]
    fn test_square_root_law_citation() {
        let law = SquareRootLaw::new();
        let citation = law.citation();
        assert_eq!(citation.year, 1979); // Eppen 1979
    }

    #[test]
    fn test_square_root_law_validate_consistency() {
        let law = SquareRootLaw::new();

        // z=1.96, σ_D=100, L=1 => I_safety = 1.96 * 100 * sqrt(1) = 196
        let values = vec![
            ("demand_std", 100.0),
            ("lead_time", 1.0),
            ("z_score", 1.96),
            ("I_safety", 196.0),
        ];
        assert!(law.validate_consistency(&values, 0.01).is_ok());
    }

    #[test]
    fn test_square_root_law_z_score() {
        // Test z_score_for_service_level - specific values
        let z_90 = SquareRootLaw::z_score_for_service_level(0.90);
        assert!((z_90 - 1.28).abs() < 0.01);

        let z_95 = SquareRootLaw::z_score_for_service_level(0.95);
        assert!((z_95 - 1.65).abs() < 0.01);

        let z_99 = SquareRootLaw::z_score_for_service_level(0.99);
        assert!((z_99 - 2.33).abs() < 0.01);

        let z_999 = SquareRootLaw::z_score_for_service_level(0.999);
        assert!((z_999 - 3.09).abs() < 0.01);

        // Unknown values default to 1.96
        let z_other = SquareRootLaw::z_score_for_service_level(0.50);
        assert!((z_other - 1.96).abs() < 0.01);
    }

    #[test]
    fn test_bullwhip_trait_methods() {
        let effect = BullwhipEffect::new();

        assert!(effect.latex().contains("L"));
        assert_eq!(effect.class(), EquationClass::Inventory);
        assert_eq!(effect.name(), "Bullwhip Effect");
        assert!(!effect.description().is_empty());
    }

    #[test]
    fn test_bullwhip_default() {
        let effect = BullwhipEffect::default();
        assert_eq!(effect.name(), "Bullwhip Effect");
    }

    #[test]
    fn test_bullwhip_variables() {
        let effect = BullwhipEffect::new();
        let vars = effect.variables();
        assert!(!vars.is_empty());
    }

    #[test]
    fn test_bullwhip_validate_consistency() {
        let effect = BullwhipEffect::new();

        // L=1, p=1 => factor = 5, so Var(Orders)/Var(Demand) >= 5
        let values = vec![
            ("lead_time", 1.0),
            ("review_period", 1.0),
            ("order_variance", 50.0),
            ("demand_variance", 10.0),
        ];
        // 50/10 = 5, which equals the minimum amplification factor
        assert!(effect.validate_consistency(&values, 0.1).is_ok());
    }

    #[test]
    fn test_bullwhip_empty_echelons() {
        let effect = BullwhipEffect::new();

        // Empty arrays should return 1.0
        let factor = effect.multi_echelon_amplification(&[], &[]);
        assert!((factor - 1.0).abs() < f64::EPSILON);
    }
}
