//! Mixture of Experts implementation

use super::gating::GatingNetwork;
use crate::traits::Estimator;
use crate::Result;
use serde::{Deserialize, Serialize};

/// MoE routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoeConfig {
    pub top_k: usize,
    pub capacity_factor: f32,
    pub expert_dropout: f32,
    pub load_balance_weight: f32,
}

impl Default for MoeConfig {
    fn default() -> Self {
        Self {
            top_k: 1,
            capacity_factor: 1.0,
            expert_dropout: 0.0,
            load_balance_weight: 0.01,
        }
    }
}

impl MoeConfig {
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    #[must_use]
    pub fn with_capacity_factor(mut self, factor: f32) -> Self {
        self.capacity_factor = factor;
        self
    }

    #[must_use]
    pub fn with_expert_dropout(mut self, dropout: f32) -> Self {
        self.expert_dropout = dropout;
        self
    }

    #[must_use]
    pub fn with_load_balance_weight(mut self, weight: f32) -> Self {
        self.load_balance_weight = weight;
        self
    }
}

/// Mixture of Experts ensemble
pub struct MixtureOfExperts<E: Estimator, G: GatingNetwork> {
    experts: Vec<E>,
    gating: G,
    config: MoeConfig,
}

impl<E: Estimator + std::fmt::Debug, G: GatingNetwork + std::fmt::Debug> std::fmt::Debug
    for MixtureOfExperts<E, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixtureOfExperts")
            .field("experts", &self.experts)
            .field("gating", &self.gating)
            .field("config", &self.config)
            .finish()
    }
}

impl<E: Estimator, G: GatingNetwork> MixtureOfExperts<E, G> {
    pub fn builder() -> MoeBuilder<E, G> {
        MoeBuilder::new()
    }

    #[must_use]
    pub fn n_experts(&self) -> usize {
        self.experts.len()
    }

    #[must_use]
    pub fn config(&self) -> &MoeConfig {
        &self.config
    }

    pub fn predict(&self, input: &[f32]) -> f32 {
        let weights = self.gating.forward(input);
        let mut indexed_weights: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
        indexed_weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k = self.config.top_k.min(self.experts.len());
        let top_weights: Vec<(usize, f32)> = indexed_weights.into_iter().take(top_k).collect();
        let weight_sum: f32 = top_weights.iter().map(|(_, w)| w).sum();

        let mut output = 0.0f32;
        for (expert_idx, weight) in top_weights {
            let x = crate::Matrix::from_vec(1, input.len(), input.to_vec())
                .expect("valid input matrix");
            let pred = self.experts[expert_idx].predict(&x);
            let expert_output = pred.as_slice()[0];
            output += (weight / weight_sum) * expert_output;
        }
        output
    }
}

impl<E, G> MixtureOfExperts<E, G>
where
    E: Estimator + Serialize + serde::de::DeserializeOwned,
    G: GatingNetwork + Serialize + serde::de::DeserializeOwned,
{
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let bytes = bincode::serialize(&SerializableMoe {
            experts: &self.experts,
            gating: &self.gating,
            config: &self.config,
        })
        .map_err(|e| crate::AprenderError::FormatError {
            message: format!("MoE serialization failed: {e}"),
        })?;
        std::fs::write(path, bytes).map_err(crate::AprenderError::Io)?;
        Ok(())
    }

    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(crate::AprenderError::Io)?;
        let data: DeserializableMoe<E, G> =
            bincode::deserialize(&bytes).map_err(|e| crate::AprenderError::FormatError {
                message: format!("MoE deserialization failed: {e}"),
            })?;
        Ok(Self {
            experts: data.experts,
            gating: data.gating,
            config: data.config,
        })
    }

    pub fn save_apr<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        use crate::format::{ModelType, SaveOptions};
        let data = SerializableMoe {
            experts: &self.experts,
            gating: &self.gating,
            config: &self.config,
        };
        crate::format::save(
            &data,
            ModelType::MixtureOfExperts,
            path,
            SaveOptions::default(),
        )
    }
}

#[derive(Serialize)]
struct SerializableMoe<'a, E, G> {
    experts: &'a Vec<E>,
    gating: &'a G,
    config: &'a MoeConfig,
}

#[derive(Deserialize)]
struct DeserializableMoe<E, G> {
    experts: Vec<E>,
    gating: G,
    config: MoeConfig,
}

/// Builder for MixtureOfExperts
pub struct MoeBuilder<E: Estimator, G: GatingNetwork> {
    experts: Vec<E>,
    gating: Option<G>,
    config: MoeConfig,
}

impl<E: Estimator + std::fmt::Debug, G: GatingNetwork + std::fmt::Debug> std::fmt::Debug
    for MoeBuilder<E, G>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoeBuilder")
            .field("experts", &self.experts)
            .field("gating", &self.gating)
            .field("config", &self.config)
            .finish()
    }
}

impl<E: Estimator, G: GatingNetwork> MoeBuilder<E, G> {
    fn new() -> Self {
        Self {
            experts: Vec::new(),
            gating: None,
            config: MoeConfig::default(),
        }
    }

    #[must_use]
    pub fn gating(mut self, g: G) -> Self {
        self.gating = Some(g);
        self
    }

    #[must_use]
    pub fn expert(mut self, e: E) -> Self {
        self.experts.push(e);
        self
    }

    #[must_use]
    pub fn config(mut self, c: MoeConfig) -> Self {
        self.config = c;
        self
    }

    pub fn build(self) -> Result<MixtureOfExperts<E, G>> {
        let gating = self
            .gating
            .ok_or_else(|| crate::AprenderError::InvalidHyperparameter {
                param: "gating".into(),
                value: "None".into(),
                constraint: "GatingNetwork required".into(),
            })?;
        Ok(MixtureOfExperts {
            experts: self.experts,
            gating,
            config: self.config,
        })
    }
}
