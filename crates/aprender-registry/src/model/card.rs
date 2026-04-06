//! Model Card for standardized model documentation.
//!
//! Based on "Model Cards for Model Reporting" (Mitchell et al., 2019).

use crate::data::DatasetReference;
use crate::recipe::RecipeReference;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ModelReference;

/// Model Card with standardized documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Model description.
    pub description: String,

    /// Reference to training dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_data: Option<DatasetReference>,
    /// Reference to training recipe.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_recipe: Option<RecipeReference>,
    /// When training was performed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_date: Option<DateTime<Utc>>,
    /// Training duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub training_duration_secs: Option<i64>,

    /// Performance metrics (e.g., accuracy, F1 score).
    #[serde(default)]
    pub metrics: HashMap<String, f64>,
    /// Reference to evaluation dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluation_data: Option<DatasetReference>,

    /// Primary intended use cases.
    #[serde(default)]
    pub primary_uses: Vec<String>,
    /// Out-of-scope use cases.
    #[serde(default)]
    pub out_of_scope_uses: Vec<String>,

    /// Known limitations.
    #[serde(default)]
    pub limitations: Vec<String>,
    /// Ethical considerations.
    #[serde(default)]
    pub ethical_considerations: Vec<String>,

    /// Parent model (if fine-tuned).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_model: Option<ModelReference>,
    /// Models this was derived from.
    #[serde(default)]
    pub derived_from: Vec<ModelReference>,

    /// Additional metadata.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ModelCard {
    /// Create a new model card builder.
    #[must_use]
    pub fn builder() -> ModelCardBuilder {
        ModelCardBuilder::new()
    }

    /// Create a minimal model card with just a description.
    #[must_use]
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            training_data: None,
            training_recipe: None,
            training_date: None,
            training_duration_secs: None,
            metrics: HashMap::new(),
            evaluation_data: None,
            primary_uses: Vec::new(),
            out_of_scope_uses: Vec::new(),
            limitations: Vec::new(),
            ethical_considerations: Vec::new(),
            parent_model: None,
            derived_from: Vec::new(),
            extra: HashMap::new(),
        }
    }

    /// Get training duration as a Duration.
    #[must_use]
    pub fn training_duration(&self) -> Option<Duration> {
        self.training_duration_secs.map(Duration::seconds)
    }

    /// Add a metric.
    pub fn add_metric(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.insert(name.into(), value);
    }

    /// Add a primary use case.
    pub fn add_primary_use(&mut self, use_case: impl Into<String>) {
        self.primary_uses.push(use_case.into());
    }

    /// Add a limitation.
    pub fn add_limitation(&mut self, limitation: impl Into<String>) {
        self.limitations.push(limitation.into());
    }
}

impl Default for ModelCard {
    fn default() -> Self {
        Self::new("")
    }
}

/// Builder for creating model cards.
#[derive(Debug, Default)]
pub struct ModelCardBuilder {
    card: ModelCard,
}

impl ModelCardBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self { card: ModelCard::default() }
    }

    /// Set the description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.card.description = description.into();
        self
    }

    /// Set the training data reference.
    #[must_use]
    pub fn training_data(mut self, data: DatasetReference) -> Self {
        self.card.training_data = Some(data);
        self
    }

    /// Set the training recipe reference.
    #[must_use]
    pub fn training_recipe(mut self, recipe: RecipeReference) -> Self {
        self.card.training_recipe = Some(recipe);
        self
    }

    /// Set the training date.
    #[must_use]
    pub fn training_date(mut self, date: DateTime<Utc>) -> Self {
        self.card.training_date = Some(date);
        self
    }

    /// Set the training duration.
    #[must_use]
    pub fn training_duration(mut self, duration: Duration) -> Self {
        self.card.training_duration_secs = Some(duration.num_seconds());
        self
    }

    /// Add metrics from an iterator.
    #[must_use]
    pub fn metrics<I, K>(mut self, metrics: I) -> Self
    where
        I: IntoIterator<Item = (K, f64)>,
        K: Into<String>,
    {
        for (k, v) in metrics {
            self.card.metrics.insert(k.into(), v);
        }
        self
    }

    /// Set evaluation data reference.
    #[must_use]
    pub fn evaluation_data(mut self, data: DatasetReference) -> Self {
        self.card.evaluation_data = Some(data);
        self
    }

    /// Add primary uses.
    #[must_use]
    pub fn primary_uses<I, S>(mut self, uses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.card.primary_uses = uses.into_iter().map(Into::into).collect();
        self
    }

    /// Add out-of-scope uses.
    #[must_use]
    pub fn out_of_scope_uses<I, S>(mut self, uses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.card.out_of_scope_uses = uses.into_iter().map(Into::into).collect();
        self
    }

    /// Add limitations.
    #[must_use]
    pub fn limitations<I, S>(mut self, limitations: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.card.limitations = limitations.into_iter().map(Into::into).collect();
        self
    }

    /// Add ethical considerations.
    #[must_use]
    pub fn ethical_considerations<I, S>(mut self, considerations: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.card.ethical_considerations = considerations.into_iter().map(Into::into).collect();
        self
    }

    /// Set parent model reference.
    #[must_use]
    pub fn parent_model(mut self, parent: ModelReference) -> Self {
        self.card.parent_model = Some(parent);
        self
    }

    /// Set derived-from models.
    #[must_use]
    pub fn derived_from(mut self, models: Vec<ModelReference>) -> Self {
        self.card.derived_from = models;
        self
    }

    /// Build the model card.
    #[must_use]
    pub fn build(self) -> ModelCard {
        self.card
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DatasetVersion;
    use crate::model::ModelVersion;
    use crate::recipe::RecipeVersion;

    #[test]
    fn test_model_card_new() {
        let card = ModelCard::new("A fraud detection model");
        assert_eq!(card.description, "A fraud detection model");
        assert!(card.metrics.is_empty());
    }

    #[test]
    fn test_model_card_builder() {
        let card = ModelCard::builder()
            .description("Fraud detector v1")
            .metrics([("auc", 0.95), ("f1", 0.88)])
            .primary_uses(["Fraud detection in payment transactions"])
            .limitations(["May have reduced accuracy on international transactions"])
            .build();

        assert_eq!(card.description, "Fraud detector v1");
        assert_eq!(card.metrics.get("auc"), Some(&0.95));
        assert_eq!(card.metrics.get("f1"), Some(&0.88));
        assert_eq!(card.primary_uses.len(), 1);
        assert_eq!(card.limitations.len(), 1);
    }

    #[test]
    fn test_model_card_with_references() {
        let dataset_ref = DatasetReference::new("transactions", DatasetVersion::new(1, 0, 0));
        let recipe_ref = RecipeReference::new("fraud-training", RecipeVersion::new(1, 0, 0));
        let parent_ref = ModelReference::new("base-classifier", ModelVersion::new(1, 0, 0));

        let card = ModelCard::builder()
            .description("Fine-tuned fraud detector")
            .training_data(dataset_ref.clone())
            .training_recipe(recipe_ref.clone())
            .parent_model(parent_ref.clone())
            .build();

        assert_eq!(card.training_data.unwrap().name, "transactions");
        assert_eq!(card.training_recipe.unwrap().name, "fraud-training");
        assert_eq!(card.parent_model.unwrap().name, "base-classifier");
    }

    #[test]
    fn test_model_card_add_methods() {
        let mut card = ModelCard::new("Test model");
        card.add_metric("accuracy", 0.92);
        card.add_primary_use("Classification");
        card.add_limitation("Requires normalized inputs");

        assert_eq!(card.metrics.get("accuracy"), Some(&0.92));
        assert_eq!(card.primary_uses, vec!["Classification"]);
        assert_eq!(card.limitations, vec!["Requires normalized inputs"]);
    }

    #[test]
    fn test_model_card_serialization() {
        let card =
            ModelCard::builder().description("Test model").metrics([("accuracy", 0.95)]).build();

        let json = serde_json::to_string(&card).unwrap();
        let deserialized: ModelCard = serde_json::from_str(&json).unwrap();

        assert_eq!(card.description, deserialized.description);
        assert_eq!(card.metrics, deserialized.metrics);
    }

    #[test]
    fn test_training_duration() {
        let card =
            ModelCard::builder().description("Model").training_duration(Duration::hours(2)).build();

        let duration = card.training_duration().unwrap();
        assert_eq!(duration.num_hours(), 2);
    }
}
