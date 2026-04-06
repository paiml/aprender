//! Performance predictor with curve fitting and model selection.

use std::collections::HashMap;

use super::types::{DataPoint, FittedModel, ModelType, Prediction, MIN_SAMPLES_FOR_FIT};

/// Performance predictor
#[derive(Debug)]
pub struct PerformancePredictor {
    /// Data points
    data_points: Vec<DataPoint>,
    /// Fitted models
    models: HashMap<ModelType, FittedModel>,
    /// Best model
    best_model: Option<ModelType>,
    /// Confidence level for bounds
    confidence_level: f64,
}

impl Default for PerformancePredictor {
    fn default() -> Self {
        Self {
            data_points: Vec::new(),
            models: HashMap::new(),
            best_model: None,
            confidence_level: 0.95,
        }
    }
}

impl PerformancePredictor {
    /// Create new predictor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set confidence level
    pub fn with_confidence(mut self, level: f64) -> Self {
        self.confidence_level = level.clamp(0.5, 0.99);
        self
    }

    /// Add data point
    pub fn add_point(&mut self, point: DataPoint) {
        self.data_points.push(point);
        // Invalidate cached models
        self.models.clear();
        self.best_model = None;
    }

    /// Add data from values
    pub fn add(&mut self, size: usize, performance: f64, latency_us: f64) {
        self.add_point(DataPoint::new(size, performance, latency_us));
    }

    /// Get data point count
    pub fn point_count(&self) -> usize {
        self.data_points.len()
    }

    /// Check if enough data for fitting
    pub fn has_sufficient_data(&self) -> bool {
        self.data_points.len() >= MIN_SAMPLES_FOR_FIT
    }

    /// Get size range of data
    pub fn size_range(&self) -> Option<(usize, usize)> {
        if self.data_points.is_empty() {
            return None;
        }

        let min = self.data_points.iter().map(|p| p.size).min().expect("non-empty collection");
        let max = self.data_points.iter().map(|p| p.size).max().expect("non-empty collection");
        Some((min, max))
    }

    /// Fit linear model
    pub fn fit_linear(&mut self) -> Option<FittedModel> {
        if !self.has_sufficient_data() {
            return None;
        }

        let n = self.data_points.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for p in &self.data_points {
            let x = p.size as f64;
            let y = p.performance;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return None;
        }

        let a = (n * sum_xy - sum_x * sum_y) / denom;
        let b = (sum_y - a * sum_x) / n;

        let mean_y = sum_y / n;
        let (ss_res, ss_tot) = self.compute_ss(|x| a * x + b, mean_y);
        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 1.0 };

        let model = FittedModel {
            model_type: ModelType::Linear,
            coefficients: vec![a, b],
            r_squared,
            rss: ss_res,
            sample_count: self.data_points.len(),
        };

        self.models.insert(ModelType::Linear, model.clone());
        Some(model)
    }

    /// Fit polynomial model (quadratic)
    pub fn fit_polynomial(&mut self) -> Option<FittedModel> {
        if !self.has_sufficient_data() {
            return None;
        }

        // Use least squares for quadratic fit
        // y = a*x^2 + b*x + c
        let n = self.data_points.len() as f64;
        let mut _sum_x = 0.0;
        let mut _sum_x2 = 0.0;
        let mut _sum_x3 = 0.0;
        let mut _sum_x4 = 0.0;
        let mut sum_y = 0.0;
        let mut _sum_xy = 0.0;
        let mut _sum_x2y = 0.0;

        for p in &self.data_points {
            let x = p.size as f64;
            let y = p.performance;
            _sum_x += x;
            _sum_x2 += x * x;
            _sum_x3 += x * x * x;
            _sum_x4 += x * x * x * x;
            sum_y += y;
            _sum_xy += x * y;
            _sum_x2y += x * x * y;
        }

        // Solve 3x3 system (simplified - use Cramer's rule)
        // This is a simplified implementation
        let _mean_y = sum_y / n;

        // Fallback to linear for now if polynomial fails
        if let Some(linear) = self.fit_linear() {
            let a = 0.0; // No quadratic term
            let b = linear.coefficients[0];
            let c = linear.coefficients[1];

            let model = FittedModel {
                model_type: ModelType::Polynomial,
                coefficients: vec![a, b, c],
                r_squared: linear.r_squared,
                rss: linear.rss,
                sample_count: self.data_points.len(),
            };

            self.models.insert(ModelType::Polynomial, model.clone());
            return Some(model);
        }

        None
    }

    /// Fit logarithmic model
    pub fn fit_logarithmic(&mut self) -> Option<FittedModel> {
        if !self.has_sufficient_data() {
            return None;
        }

        // y = a * log(x) + b
        let n = self.data_points.len() as f64;
        let mut sum_lnx = 0.0;
        let mut sum_y = 0.0;
        let mut sum_lnx_y = 0.0;
        let mut sum_lnx2 = 0.0;

        for p in &self.data_points {
            let x = p.size as f64;
            if x <= 0.0 {
                continue;
            }
            let lnx = x.ln();
            let y = p.performance;
            sum_lnx += lnx;
            sum_y += y;
            sum_lnx_y += lnx * y;
            sum_lnx2 += lnx * lnx;
        }

        let denom = n * sum_lnx2 - sum_lnx * sum_lnx;
        if denom.abs() < 1e-10 {
            return None;
        }

        let a = (n * sum_lnx_y - sum_lnx * sum_y) / denom;
        let b = (sum_y - a * sum_lnx) / n;

        let mean_y = sum_y / n;
        let (ss_res, ss_tot) = self.compute_ss(|x| a * x.ln() + b, mean_y);
        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 1.0 };

        let model = FittedModel {
            model_type: ModelType::Logarithmic,
            coefficients: vec![a, b],
            r_squared,
            rss: ss_res,
            sample_count: self.data_points.len(),
        };

        self.models.insert(ModelType::Logarithmic, model.clone());
        Some(model)
    }

    /// Compute SS_res and SS_tot
    fn compute_ss<F: Fn(f64) -> f64>(&self, predict_fn: F, mean_y: f64) -> (f64, f64) {
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for p in &self.data_points {
            let x = p.size as f64;
            let y_pred = predict_fn(x);
            ss_res += (p.performance - y_pred).powi(2);
            ss_tot += (p.performance - mean_y).powi(2);
        }

        (ss_res, ss_tot)
    }

    /// Fit all models and select best
    pub fn fit_all(&mut self) -> Option<ModelType> {
        self.fit_linear();
        self.fit_polynomial();
        self.fit_logarithmic();

        // Select best by R-squared
        let best = self
            .models
            .iter()
            .max_by(|a, b| {
                a.1.r_squared.partial_cmp(&b.1.r_squared).expect("values should be comparable")
            })
            .map(|(t, _)| *t);

        self.best_model = best;
        best
    }

    /// Get best model
    pub fn best_model(&mut self) -> Option<&FittedModel> {
        if self.best_model.is_none() {
            self.fit_all();
        }

        self.best_model.and_then(|t| self.models.get(&t))
    }

    /// Predict at size
    pub fn predict_at_size(&mut self, size: usize) -> Option<Prediction> {
        let model = self.best_model()?.clone();
        let predicted = model.predict(size);

        let (min_size, max_size) = self.size_range()?;
        let is_extrapolation = size < min_size || size > max_size;

        // Compute confidence bounds based on R-squared and extrapolation
        let base_uncertainty = 1.0 - model.r_squared;
        let extrapolation_penalty = if is_extrapolation {
            let distance = if size < min_size {
                (min_size - size) as f64 / min_size as f64
            } else {
                (size - max_size) as f64 / max_size as f64
            };
            distance * 0.5 // 50% more uncertainty per distance ratio
        } else {
            0.0
        };

        let total_uncertainty = (base_uncertainty + extrapolation_penalty).min(1.0);
        let z = 1.96; // 95% confidence

        let half_width = predicted * total_uncertainty * z;
        let lower_bound = (predicted - half_width).max(0.0);
        let upper_bound = predicted + half_width;

        Some(Prediction {
            size,
            predicted,
            lower_bound,
            upper_bound,
            confidence_level: self.confidence_level,
            model_type: model.model_type,
            is_extrapolation,
        })
    }

    /// Get model by type
    pub fn get_model(&self, model_type: ModelType) -> Option<&FittedModel> {
        self.models.get(&model_type)
    }

    /// Compare models
    pub fn compare_models(&self) -> Vec<(&ModelType, f64)> {
        let mut comparisons: Vec<_> = self.models.iter().map(|(t, m)| (t, m.r_squared)).collect();

        comparisons.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("values should be comparable"));
        comparisons
    }

    /// Export model (serialize to JSON-like format)
    pub fn export_model(&self, model_type: ModelType) -> Option<String> {
        let model = self.models.get(&model_type)?;
        Some(format!(
            "{{\"type\":\"{}\",\"coefficients\":{:?},\"r_squared\":{:.6}}}",
            model.model_type.name(),
            model.coefficients,
            model.r_squared
        ))
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.data_points.clear();
        self.models.clear();
        self.best_model = None;
    }
}
