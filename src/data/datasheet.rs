//! Datasheet for standardized dataset documentation.
//!
//! Based on "Datasheets for Datasets" (Gebru et al., 2021).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Datasheet with standardized dataset documentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Datasheet {
    // Motivation
    /// Purpose of the dataset.
    pub purpose: String,
    /// Creators of the dataset.
    #[serde(default)]
    pub creators: Vec<String>,
    /// Funding source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub funding: Option<String>,

    // Composition
    /// Number of instances.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instance_count: Option<u64>,
    /// Feature descriptions.
    #[serde(default)]
    pub features: HashMap<String, FeatureInfo>,
    /// Sensitive features that require special handling.
    #[serde(default)]
    pub sensitive_features: Vec<String>,

    // Collection process
    /// How the data was collected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection_method: Option<String>,
    /// When the data collection started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection_start: Option<DateTime<Utc>>,
    /// When the data collection ended.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collection_end: Option<DateTime<Utc>>,
    /// Preprocessing steps applied.
    #[serde(default)]
    pub preprocessing: Vec<PreprocessingStep>,

    // Distribution
    /// License for the dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Access restrictions.
    #[serde(default)]
    pub access_restrictions: Vec<String>,

    // Maintenance
    /// Who maintains the dataset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maintainer: Option<String>,
    /// How often the dataset is updated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update_frequency: Option<String>,
    /// Deprecation policy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deprecation_policy: Option<String>,

    /// Additional metadata.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Information about a feature in the dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInfo {
    /// Data type of the feature.
    pub dtype: String,
    /// Description of the feature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Whether the feature can be null.
    #[serde(default)]
    pub nullable: bool,
    /// Statistics about the feature.
    #[serde(default)]
    pub statistics: HashMap<String, f64>,
}

impl FeatureInfo {
    /// Create a new feature info.
    #[must_use]
    pub fn new(dtype: impl Into<String>) -> Self {
        Self { dtype: dtype.into(), description: None, nullable: false, statistics: HashMap::new() }
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set nullable.
    #[must_use]
    pub fn with_nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }
}

/// A preprocessing step applied to the data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingStep {
    /// Name of the step.
    pub name: String,
    /// Description of what the step does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Parameters used.
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

impl PreprocessingStep {
    /// Create a new preprocessing step.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), description: None, parameters: HashMap::new() }
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

impl Datasheet {
    /// Create a new datasheet builder.
    #[must_use]
    pub fn builder() -> DatasheetBuilder {
        DatasheetBuilder::new()
    }

    /// Create a minimal datasheet with just a purpose.
    #[must_use]
    pub fn new(purpose: impl Into<String>) -> Self {
        Self {
            purpose: purpose.into(),
            creators: Vec::new(),
            funding: None,
            instance_count: None,
            features: HashMap::new(),
            sensitive_features: Vec::new(),
            collection_method: None,
            collection_start: None,
            collection_end: None,
            preprocessing: Vec::new(),
            license: None,
            access_restrictions: Vec::new(),
            maintainer: None,
            update_frequency: None,
            deprecation_policy: None,
            extra: HashMap::new(),
        }
    }

    /// Add a feature.
    pub fn add_feature(&mut self, name: impl Into<String>, info: FeatureInfo) {
        self.features.insert(name.into(), info);
    }

    /// Add a preprocessing step.
    pub fn add_preprocessing(&mut self, step: PreprocessingStep) {
        self.preprocessing.push(step);
    }
}

impl Default for Datasheet {
    fn default() -> Self {
        Self::new("")
    }
}

/// Builder for creating datasheets.
#[derive(Debug, Default)]
pub struct DatasheetBuilder {
    sheet: Datasheet,
}

impl DatasheetBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self { sheet: Datasheet::default() }
    }

    /// Set the purpose.
    #[must_use]
    pub fn purpose(mut self, purpose: impl Into<String>) -> Self {
        self.sheet.purpose = purpose.into();
        self
    }

    /// Set creators.
    #[must_use]
    pub fn creators<I, S>(mut self, creators: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.sheet.creators = creators.into_iter().map(Into::into).collect();
        self
    }

    /// Set funding.
    #[must_use]
    pub fn funding(mut self, funding: impl Into<String>) -> Self {
        self.sheet.funding = Some(funding.into());
        self
    }

    /// Set instance count.
    #[must_use]
    pub fn instance_count(mut self, count: u64) -> Self {
        self.sheet.instance_count = Some(count);
        self
    }

    /// Add a feature.
    #[must_use]
    pub fn feature(mut self, name: impl Into<String>, info: FeatureInfo) -> Self {
        self.sheet.features.insert(name.into(), info);
        self
    }

    /// Set sensitive features.
    #[must_use]
    pub fn sensitive_features<I, S>(mut self, features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.sheet.sensitive_features = features.into_iter().map(Into::into).collect();
        self
    }

    /// Set collection method.
    #[must_use]
    pub fn collection_method(mut self, method: impl Into<String>) -> Self {
        self.sheet.collection_method = Some(method.into());
        self
    }

    /// Set license.
    #[must_use]
    pub fn license(mut self, license: impl Into<String>) -> Self {
        self.sheet.license = Some(license.into());
        self
    }

    /// Set maintainer.
    #[must_use]
    pub fn maintainer(mut self, maintainer: impl Into<String>) -> Self {
        self.sheet.maintainer = Some(maintainer.into());
        self
    }

    /// Build the datasheet.
    #[must_use]
    pub fn build(self) -> Datasheet {
        self.sheet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datasheet_new() {
        let sheet = Datasheet::new("Customer transactions for fraud detection");
        assert_eq!(sheet.purpose, "Customer transactions for fraud detection");
        assert!(sheet.features.is_empty());
    }

    #[test]
    fn test_datasheet_builder() {
        let sheet = Datasheet::builder()
            .purpose("Training data for fraud detection")
            .creators(["Alice", "Bob"])
            .instance_count(1_000_000)
            .feature(
                "amount",
                FeatureInfo::new("float64").with_description("Transaction amount in USD"),
            )
            .feature("timestamp", FeatureInfo::new("datetime").with_nullable(true))
            .sensitive_features(["customer_id", "card_number"])
            .license("MIT")
            .maintainer("data-team@company.com")
            .build();

        assert_eq!(sheet.purpose, "Training data for fraud detection");
        assert_eq!(sheet.creators, vec!["Alice", "Bob"]);
        assert_eq!(sheet.instance_count, Some(1_000_000));
        assert_eq!(sheet.features.len(), 2);
        assert!(sheet.features.contains_key("amount"));
        assert_eq!(sheet.sensitive_features.len(), 2);
        assert_eq!(sheet.license, Some("MIT".to_string()));
    }

    #[test]
    fn test_feature_info() {
        let info = FeatureInfo::new("int64").with_description("User ID").with_nullable(false);

        assert_eq!(info.dtype, "int64");
        assert_eq!(info.description, Some("User ID".to_string()));
        assert!(!info.nullable);
    }

    #[test]
    fn test_preprocessing_step() {
        let step = PreprocessingStep::new("normalize").with_description("Min-max normalization");

        assert_eq!(step.name, "normalize");
        assert_eq!(step.description, Some("Min-max normalization".to_string()));
    }

    #[test]
    fn test_datasheet_add_methods() {
        let mut sheet = Datasheet::new("Test dataset");
        sheet.add_feature("col1", FeatureInfo::new("string"));
        sheet.add_preprocessing(PreprocessingStep::new("clean"));

        assert_eq!(sheet.features.len(), 1);
        assert_eq!(sheet.preprocessing.len(), 1);
    }

    #[test]
    fn test_datasheet_serialization() {
        let sheet = Datasheet::builder().purpose("Test").instance_count(100).build();

        let json = serde_json::to_string(&sheet).unwrap();
        let deserialized: Datasheet = serde_json::from_str(&json).unwrap();

        assert_eq!(sheet.purpose, deserialized.purpose);
        assert_eq!(sheet.instance_count, deserialized.instance_count);
    }
}
