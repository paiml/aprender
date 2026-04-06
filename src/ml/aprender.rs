//! aprender integration for bug prediction
//!
//! This module provides integration with the aprender ML library
//! for training and inference of bug prediction models.

#[cfg(feature = "ml")]
use aprender::tree::RandomForestClassifier;
#[cfg(feature = "ml")]
use aprender::Matrix;

use crate::data::CodeFeatures;
use crate::Result;

/// Trained bug prediction model using aprender RandomForest
#[derive(Debug)]
pub struct AprenderBugPredictor {
    #[cfg(feature = "ml")]
    model: RandomForestClassifier,
    #[cfg(not(feature = "ml"))]
    _phantom: std::marker::PhantomData<()>,
}

impl AprenderBugPredictor {
    /// Train a new bug prediction model
    ///
    /// # Arguments
    ///
    /// * `features` - Training features extracted from code
    /// * `labels` - Ground truth labels (true if bug, false if correct)
    ///
    /// # Errors
    ///
    /// Returns error if training fails
    #[cfg(feature = "ml")]
    pub fn train(features: &[CodeFeatures], labels: &[bool]) -> Result<Self> {
        // Convert CodeFeatures to Matrix<f32>
        let n_samples = features.len();
        let n_features = 5; // ast_depth, num_operators, num_control_flow, cyclomatic_complexity, uses_edge_values

        let mut data = Vec::with_capacity(n_samples * n_features);
        for f in features {
            data.push(f.ast_depth as f32);
            data.push(f.num_operators as f32);
            data.push(f.num_control_flow as f32);
            data.push(f.cyclomatic_complexity);
            data.push(if f.uses_edge_values { 1.0 } else { 0.0 });
        }

        let x = Matrix::from_vec(n_samples, n_features, data)
            .map_err(|e| crate::Error::Data(format!("Failed to create matrix: {e}")))?;

        let y: Vec<usize> = labels.iter().map(|&b| usize::from(b)).collect();

        let mut model = RandomForestClassifier::new(100)
            .with_max_depth(10)
            .with_random_state(42);

        model
            .fit(&x, &y)
            .map_err(|e| crate::Error::Data(format!("Training failed: {e}")))?;

        Ok(Self { model })
    }

    /// Train a new bug prediction model (no-op without ml feature)
    ///
    /// # Errors
    ///
    /// Always returns error without 'ml' feature enabled
    #[cfg(not(feature = "ml"))]
    pub fn train(_features: &[CodeFeatures], _labels: &[bool]) -> Result<Self> {
        Err(crate::Error::Data(
            "aprender integration requires 'ml' feature".to_string(),
        ))
    }

    /// Predict probability of a bug
    ///
    /// Returns probability in range [0, 1]
    ///
    /// Note: Currently returns hard predictions (0.0 or 1.0) as aprender's
    /// RandomForestClassifier doesn't expose predict_proba yet.
    #[cfg(feature = "ml")]
    pub fn predict(&self, features: &CodeFeatures) -> f32 {
        let data = vec![
            features.ast_depth as f32,
            features.num_operators as f32,
            features.num_control_flow as f32,
            features.cyclomatic_complexity,
            if features.uses_edge_values { 1.0 } else { 0.0 },
        ];

        let Ok(x) = Matrix::from_vec(1, 5, data) else {
            return 0.5; // Fallback
        };

        let predictions = self.model.predict(&x);
        // Convert class label (0 or 1) to probability
        if predictions.is_empty() {
            0.5
        } else {
            predictions[0] as f32
        }
    }

    /// Predict probability of a bug (fallback without ml feature)
    #[cfg(not(feature = "ml"))]
    pub fn predict(&self, _features: &CodeFeatures) -> f32 {
        0.5 // Neutral probability
    }

    /// Save model to file
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails
    ///
    /// Note: Model persistence requires serde serialization support in aprender.
    /// Planned for future release with SafeTensors format.
    pub fn save(&self, _path: &str) -> Result<()> {
        Err(crate::Error::Data(
            "Model serialization not yet implemented".to_string(),
        ))
    }

    /// Load model from file
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails
    ///
    /// Note: Model persistence requires serde serialization support in aprender.
    /// Planned for future release with SafeTensors format.
    pub fn load(_path: &str) -> Result<Self> {
        Err(crate::Error::Data(
            "Model deserialization not yet implemented".to_string(),
        ))
    }
}

#[cfg(all(test, feature = "ml"))]
mod tests {
    use super::*;

    #[test]
    fn test_train_and_predict() {
        let features = vec![
            CodeFeatures {
                ast_depth: 5,
                num_operators: 10,
                num_control_flow: 2,
                cyclomatic_complexity: 3.0,
                uses_edge_values: false,
                ..Default::default()
            },
            CodeFeatures {
                ast_depth: 10,
                num_operators: 50,
                num_control_flow: 10,
                cyclomatic_complexity: 15.0,
                uses_edge_values: true,
                ..Default::default()
            },
        ];

        let labels = vec![false, true]; // Second one is buggy

        let predictor = AprenderBugPredictor::train(&features, &labels).unwrap();

        // Predict on new data
        let test_simple = CodeFeatures {
            ast_depth: 3,
            num_operators: 5,
            num_control_flow: 1,
            cyclomatic_complexity: 2.0,
            uses_edge_values: false,
            ..Default::default()
        };

        let prob = predictor.predict(&test_simple);
        assert!((0.0..=1.0).contains(&prob));
    }
}
