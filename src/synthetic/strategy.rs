//! Generation strategies for synthetic data.

use std::fmt;

/// Available synthetic data generation strategies.
///
/// Each strategy uses different techniques for creating synthetic samples,
/// with trade-offs between quality, diversity, and computational cost.
///
/// # Example
///
/// ```
/// use aprender::synthetic::GenerationStrategy;
///
/// let strategy = GenerationStrategy::EDA;
/// assert_eq!(strategy.name(), "eda");
/// assert!(strategy.description().contains("Augmentation"));
/// ```
///
/// # References
///
/// - Template: Cubuk et al. (2019). AutoAugment. CVPR.
/// - EDA: Wei & Zou (2019). Easy Data Augmentation. EMNLP.
/// - BackTranslation: Sennrich et al. (2016). Back-translation. ACL.
/// - MixUp: Zhang et al. (2018). MixUp. ICLR.
/// - GrammarBased: Jia & Liang (2016). Data Recombination. ACL.
/// - SelfTraining: Xie et al. (2020). Noisy Student. CVPR.
/// - WeakSupervision: Ratner et al. (2017). Snorkel. VLDB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerationStrategy {
    /// Template-based generation with slot filling.
    ///
    /// Uses parameterized templates to generate variations by filling slots
    /// with different values. Fast and controllable, but limited diversity.
    Template,

    /// Easy Data Augmentation (synonym replacement, random insertion/swap/deletion).
    ///
    /// Simple text transformations that preserve semantics while creating
    /// surface-level variations. Good baseline with low computational cost.
    EDA,

    /// Back-translation through pivot representation.
    ///
    /// Translates to intermediate representation and back, creating paraphrases.
    /// High quality but requires translation models or intermediate formats.
    BackTranslation,

    /// MixUp interpolation in embedding space.
    ///
    /// Creates virtual examples by interpolating between existing samples
    /// in latent space. Improves generalization and decision boundaries.
    MixUp,

    /// Grammar-based recombination from rules.
    ///
    /// Uses formal grammar to generate valid samples by recombining
    /// syntactic components. Guarantees syntactic correctness.
    GrammarBased,

    /// Self-training with pseudo-labels.
    ///
    /// Uses model predictions on unlabeled data as synthetic labels.
    /// Iteratively improves with curriculum ordering.
    SelfTraining,

    /// Programmatic weak supervision with labeling functions.
    ///
    /// Uses heuristic rules to label unlabeled data, then combines
    /// noisy labels using a generative model.
    WeakSupervision,
}

impl GenerationStrategy {
    /// Get the strategy name as a string identifier.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Template => "template",
            Self::EDA => "eda",
            Self::BackTranslation => "back_translation",
            Self::MixUp => "mixup",
            Self::GrammarBased => "grammar_based",
            Self::SelfTraining => "self_training",
            Self::WeakSupervision => "weak_supervision",
        }
    }

    /// Get a human-readable description of the strategy.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Template => "Template-based generation with slot filling",
            Self::EDA => "Easy Data Augmentation (synonym replacement, random ops)",
            Self::BackTranslation => "Back-translation through pivot representation",
            Self::MixUp => "MixUp interpolation in embedding space",
            Self::GrammarBased => "Grammar-based recombination from rules",
            Self::SelfTraining => "Self-training with pseudo-labels",
            Self::WeakSupervision => "Programmatic weak supervision with labeling functions",
        }
    }

    /// Get the relative computational cost (1 = lowest, 5 = highest).
    #[must_use]
    pub fn computational_cost(&self) -> u8 {
        match self {
            Self::Template | Self::EDA => 1,
            Self::MixUp | Self::GrammarBased => 2,
            Self::WeakSupervision => 3,
            Self::SelfTraining | Self::BackTranslation => 4,
        }
    }

    /// Check if this strategy requires a trained model.
    #[must_use]
    pub fn requires_model(&self) -> bool {
        matches!(
            self,
            Self::BackTranslation | Self::SelfTraining | Self::MixUp
        )
    }

    /// Check if this strategy preserves label semantics.
    #[must_use]
    pub fn preserves_labels(&self) -> bool {
        matches!(
            self,
            Self::Template | Self::EDA | Self::BackTranslation | Self::GrammarBased
        )
    }

    /// Get all available strategies.
    #[must_use]
    pub fn all() -> &'static [GenerationStrategy] {
        &[
            Self::Template,
            Self::EDA,
            Self::BackTranslation,
            Self::MixUp,
            Self::GrammarBased,
            Self::SelfTraining,
            Self::WeakSupervision,
        ]
    }

    /// Parse strategy from string name.
    ///
    /// # Arguments
    ///
    /// * `name` - Strategy name (case-insensitive)
    ///
    /// # Returns
    ///
    /// `Some(strategy)` if valid, `None` if unrecognized.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "template" => Some(Self::Template),
            "eda" => Some(Self::EDA),
            "back_translation" | "backtranslation" => Some(Self::BackTranslation),
            "mixup" | "mix_up" => Some(Self::MixUp),
            "grammar_based" | "grammarbased" | "grammar" => Some(Self::GrammarBased),
            "self_training" | "selftraining" => Some(Self::SelfTraining),
            "weak_supervision" | "weaksupervision" | "snorkel" => Some(Self::WeakSupervision),
            _ => None,
        }
    }
}

impl fmt::Display for GenerationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for GenerationStrategy {
    /// Default strategy is EDA (simple, effective, low cost).
    fn default() -> Self {
        Self::EDA
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_names() {
        assert_eq!(GenerationStrategy::Template.name(), "template");
        assert_eq!(GenerationStrategy::EDA.name(), "eda");
        assert_eq!(
            GenerationStrategy::BackTranslation.name(),
            "back_translation"
        );
        assert_eq!(GenerationStrategy::MixUp.name(), "mixup");
        assert_eq!(GenerationStrategy::GrammarBased.name(), "grammar_based");
        assert_eq!(GenerationStrategy::SelfTraining.name(), "self_training");
        assert_eq!(
            GenerationStrategy::WeakSupervision.name(),
            "weak_supervision"
        );
    }

    #[test]
    fn test_strategy_descriptions() {
        for strategy in GenerationStrategy::all() {
            let desc = strategy.description();
            assert!(!desc.is_empty());
            assert!(desc.len() > 10); // Meaningful description
        }
    }

    #[test]
    fn test_computational_cost() {
        // Template and EDA should be cheapest
        assert_eq!(GenerationStrategy::Template.computational_cost(), 1);
        assert_eq!(GenerationStrategy::EDA.computational_cost(), 1);

        // Self-training and back-translation require models, more expensive
        assert!(GenerationStrategy::SelfTraining.computational_cost() >= 3);
        assert!(GenerationStrategy::BackTranslation.computational_cost() >= 3);
    }

    #[test]
    fn test_requires_model() {
        assert!(!GenerationStrategy::Template.requires_model());
        assert!(!GenerationStrategy::EDA.requires_model());
        assert!(!GenerationStrategy::GrammarBased.requires_model());
        assert!(!GenerationStrategy::WeakSupervision.requires_model());

        assert!(GenerationStrategy::BackTranslation.requires_model());
        assert!(GenerationStrategy::SelfTraining.requires_model());
        assert!(GenerationStrategy::MixUp.requires_model());
    }

    #[test]
    fn test_preserves_labels() {
        assert!(GenerationStrategy::Template.preserves_labels());
        assert!(GenerationStrategy::EDA.preserves_labels());
        assert!(GenerationStrategy::BackTranslation.preserves_labels());
        assert!(GenerationStrategy::GrammarBased.preserves_labels());

        assert!(!GenerationStrategy::MixUp.preserves_labels());
        assert!(!GenerationStrategy::SelfTraining.preserves_labels());
        assert!(!GenerationStrategy::WeakSupervision.preserves_labels());
    }

    #[test]
    fn test_all_strategies() {
        let all = GenerationStrategy::all();
        assert_eq!(all.len(), 7);

        // Check all unique
        use std::collections::HashSet;
        let unique: HashSet<_> = all.iter().collect();
        assert_eq!(unique.len(), 7);
    }

    #[test]
    fn test_from_name_exact() {
        assert_eq!(
            GenerationStrategy::from_name("template"),
            Some(GenerationStrategy::Template)
        );
        assert_eq!(
            GenerationStrategy::from_name("eda"),
            Some(GenerationStrategy::EDA)
        );
        assert_eq!(
            GenerationStrategy::from_name("back_translation"),
            Some(GenerationStrategy::BackTranslation)
        );
    }

    #[test]
    fn test_from_name_case_insensitive() {
        assert_eq!(
            GenerationStrategy::from_name("TEMPLATE"),
            Some(GenerationStrategy::Template)
        );
        assert_eq!(
            GenerationStrategy::from_name("EDA"),
            Some(GenerationStrategy::EDA)
        );
        assert_eq!(
            GenerationStrategy::from_name("MixUp"),
            Some(GenerationStrategy::MixUp)
        );
    }

    #[test]
    fn test_from_name_aliases() {
        assert_eq!(
            GenerationStrategy::from_name("backtranslation"),
            Some(GenerationStrategy::BackTranslation)
        );
        assert_eq!(
            GenerationStrategy::from_name("mix_up"),
            Some(GenerationStrategy::MixUp)
        );
        assert_eq!(
            GenerationStrategy::from_name("grammar"),
            Some(GenerationStrategy::GrammarBased)
        );
        assert_eq!(
            GenerationStrategy::from_name("snorkel"),
            Some(GenerationStrategy::WeakSupervision)
        );
    }

    #[test]
    fn test_from_name_invalid() {
        assert_eq!(GenerationStrategy::from_name("unknown"), None);
        assert_eq!(GenerationStrategy::from_name(""), None);
        assert_eq!(GenerationStrategy::from_name("random"), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", GenerationStrategy::Template), "template");
        assert_eq!(format!("{}", GenerationStrategy::EDA), "eda");
    }

    #[test]
    fn test_default() {
        assert_eq!(GenerationStrategy::default(), GenerationStrategy::EDA);
    }

    #[test]
    fn test_clone_and_copy() {
        let s1 = GenerationStrategy::Template;
        let s2 = s1; // Copy
        let s3 = s1; // Also copy (Clone is auto-derived for Copy types)
        assert_eq!(s1, s2);
        assert_eq!(s1, s3);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        map.insert(GenerationStrategy::Template, "template_value");
        map.insert(GenerationStrategy::EDA, "eda_value");

        assert_eq!(
            map.get(&GenerationStrategy::Template),
            Some(&"template_value")
        );
        assert_eq!(map.get(&GenerationStrategy::EDA), Some(&"eda_value"));
    }

    #[test]
    fn test_roundtrip_name() {
        for strategy in GenerationStrategy::all() {
            let name = strategy.name();
            let parsed = GenerationStrategy::from_name(name);
            assert_eq!(parsed, Some(*strategy), "Roundtrip failed for {name}");
        }
    }
}
