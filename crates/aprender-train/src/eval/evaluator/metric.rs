//! Evaluation metric definitions

use super::super::classification::Average;
use std::fmt;

/// ROUGE variant for text generation evaluation
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RougeVariant {
    /// Unigram overlap
    Rouge1,
    /// Bigram overlap
    Rouge2,
    /// Longest common subsequence
    RougeL,
}

impl fmt::Display for RougeVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RougeVariant::Rouge1 => write!(f, "ROUGE-1"),
            RougeVariant::Rouge2 => write!(f, "ROUGE-2"),
            RougeVariant::RougeL => write!(f, "ROUGE-L"),
        }
    }
}

/// Available evaluation metrics
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    // Classification
    /// Classification accuracy
    Accuracy,
    /// Precision with averaging strategy
    Precision(Average),
    /// Recall with averaging strategy
    Recall(Average),
    /// F1 score with averaging strategy
    F1(Average),
    // Regression
    /// R² coefficient of determination
    R2,
    /// Mean Squared Error
    MSE,
    /// Mean Absolute Error
    MAE,
    /// Root Mean Squared Error
    RMSE,
    // Clustering
    /// Silhouette score
    Silhouette,
    /// Inertia
    Inertia,
    // ASR (Automatic Speech Recognition)
    /// Word Error Rate (lower is better)
    WER,
    /// Inverse Real-Time Factor (higher is better: RTFx=100 means 100x real-time)
    RTFx,
    // Text Generation
    /// BLEU score (higher is better)
    BLEU,
    /// ROUGE score with variant (higher is better)
    ROUGE(RougeVariant),
    /// Perplexity (lower is better)
    Perplexity,
    // LLM Benchmarks
    /// MMLU accuracy (higher is better, covers MMLU-PRO, BBH, etc.)
    MMLUAccuracy,
    // Code
    /// pass@k — unbiased estimator, parameterized by k (higher is better)
    PassAtK(usize),
    // Retrieval
    /// NDCG@k — normalized discounted cumulative gain (higher is better)
    NDCGAtK(usize),
}

impl Metric {
    /// Whether higher values are better for this metric
    pub fn higher_is_better(&self) -> bool {
        !matches!(
            self,
            Metric::MSE
                | Metric::MAE
                | Metric::RMSE
                | Metric::Inertia
                | Metric::WER
                | Metric::Perplexity
        )
    }

    /// Get metric name as string
    pub fn name(&self) -> &'static str {
        match self {
            Metric::Accuracy => "Accuracy",
            Metric::Precision(_) => "Precision",
            Metric::Recall(_) => "Recall",
            Metric::F1(_) => "F1",
            Metric::R2 => "R²",
            Metric::MSE => "MSE",
            Metric::MAE => "MAE",
            Metric::RMSE => "RMSE",
            Metric::Silhouette => "Silhouette",
            Metric::Inertia => "Inertia",
            Metric::WER => "WER",
            Metric::RTFx => "RTFx",
            Metric::BLEU => "BLEU",
            Metric::ROUGE(_) => "ROUGE",
            Metric::Perplexity => "Perplexity",
            Metric::MMLUAccuracy => "MMLU",
            Metric::PassAtK(_) => "pass@k",
            Metric::NDCGAtK(_) => "NDCG@k",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Metric::Accuracy
            | Metric::R2
            | Metric::MSE
            | Metric::MAE
            | Metric::RMSE
            | Metric::Silhouette
            | Metric::Inertia
            | Metric::WER
            | Metric::RTFx
            | Metric::BLEU
            | Metric::Perplexity
            | Metric::MMLUAccuracy => write!(f, "{}", self.name()),
            Metric::Precision(avg) => write!(f, "Precision({avg:?})"),
            Metric::Recall(avg) => write!(f, "Recall({avg:?})"),
            Metric::F1(avg) => write!(f, "F1({avg:?})"),
            Metric::ROUGE(variant) => write!(f, "{variant}"),
            Metric::PassAtK(k) => write!(f, "pass@{k}"),
            Metric::NDCGAtK(k) => write!(f, "NDCG@{k}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // name() arm tests — each exercises its exact match arm with `_` wildcard
    #[test]
    fn test_name_precision_arm() {
        let m = Metric::Precision(Average::Macro);
        match m {
            Metric::Precision(_) => assert_eq!(m.name(), "Precision"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_name_recall_arm() {
        let m = Metric::Recall(Average::Micro);
        match m {
            Metric::Recall(_) => assert_eq!(m.name(), "Recall"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_name_f1_arm() {
        let m = Metric::F1(Average::Weighted);
        match m {
            Metric::F1(_) => assert_eq!(m.name(), "F1"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_name_rouge_arm() {
        let m = Metric::ROUGE(RougeVariant::RougeL);
        match m {
            Metric::ROUGE(_) => assert_eq!(m.name(), "ROUGE"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_name_pass_at_k_arm() {
        let m = Metric::PassAtK(1);
        match m {
            Metric::PassAtK(_) => assert_eq!(m.name(), "pass@k"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_name_ndcg_at_k_arm() {
        let m = Metric::NDCGAtK(5);
        match m {
            Metric::NDCGAtK(_) => assert_eq!(m.name(), "NDCG@k"),
            _ => unreachable!(),
        }
    }

    // Display arm tests — each exercises its exact match arm with named binding
    #[test]
    fn test_display_precision_avg_arm() {
        let m = Metric::Precision(Average::Macro);
        match m {
            Metric::Precision(avg) => {
                let _ = avg;
                assert_eq!(m.to_string(), "Precision(Macro)");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_display_recall_avg_arm() {
        let m = Metric::Recall(Average::Micro);
        match m {
            Metric::Recall(avg) => {
                let _ = avg;
                assert_eq!(m.to_string(), "Recall(Micro)");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_display_f1_avg_arm() {
        let m = Metric::F1(Average::Weighted);
        match m {
            Metric::F1(avg) => {
                let _ = avg;
                assert_eq!(m.to_string(), "F1(Weighted)");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_display_rouge_variant_arm() {
        let m = Metric::ROUGE(RougeVariant::Rouge1);
        match m {
            Metric::ROUGE(variant) => {
                let _ = variant;
                assert_eq!(m.to_string(), "ROUGE-1");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_display_pass_at_k_arm() {
        let m = Metric::PassAtK(5);
        match m {
            Metric::PassAtK(k) => {
                let _ = k;
                assert_eq!(m.to_string(), "pass@5");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_display_ndcg_at_k_arm() {
        let m = Metric::NDCGAtK(10);
        match m {
            Metric::NDCGAtK(k) => {
                let _ = k;
                assert_eq!(m.to_string(), "NDCG@10");
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_display_simple_variants() {
        assert_eq!(Metric::Accuracy.to_string(), "Accuracy");
        assert_eq!(Metric::R2.to_string(), "R²");
        assert_eq!(Metric::MSE.to_string(), "MSE");
        assert_eq!(Metric::MAE.to_string(), "MAE");
        assert_eq!(Metric::RMSE.to_string(), "RMSE");
        assert_eq!(Metric::Silhouette.to_string(), "Silhouette");
        assert_eq!(Metric::Inertia.to_string(), "Inertia");
        assert_eq!(Metric::WER.to_string(), "WER");
        assert_eq!(Metric::RTFx.to_string(), "RTFx");
        assert_eq!(Metric::BLEU.to_string(), "BLEU");
        assert_eq!(Metric::Perplexity.to_string(), "Perplexity");
        assert_eq!(Metric::MMLUAccuracy.to_string(), "MMLU");
    }

    #[test]
    fn test_name_simple_variants() {
        assert_eq!(Metric::Accuracy.name(), "Accuracy");
        assert_eq!(Metric::R2.name(), "R²");
        assert_eq!(Metric::MSE.name(), "MSE");
        assert_eq!(Metric::MAE.name(), "MAE");
        assert_eq!(Metric::RMSE.name(), "RMSE");
        assert_eq!(Metric::Silhouette.name(), "Silhouette");
        assert_eq!(Metric::Inertia.name(), "Inertia");
        assert_eq!(Metric::WER.name(), "WER");
        assert_eq!(Metric::RTFx.name(), "RTFx");
        assert_eq!(Metric::BLEU.name(), "BLEU");
        assert_eq!(Metric::Perplexity.name(), "Perplexity");
        assert_eq!(Metric::MMLUAccuracy.name(), "MMLU");
    }

    #[test]
    fn test_higher_is_better_all_variants() {
        assert!(Metric::Accuracy.higher_is_better());
        assert!(Metric::Precision(Average::Macro).higher_is_better());
        assert!(Metric::Recall(Average::Micro).higher_is_better());
        assert!(Metric::F1(Average::Weighted).higher_is_better());
        assert!(Metric::R2.higher_is_better());
        assert!(!Metric::MSE.higher_is_better());
        assert!(!Metric::MAE.higher_is_better());
        assert!(!Metric::RMSE.higher_is_better());
        assert!(Metric::Silhouette.higher_is_better());
        assert!(!Metric::Inertia.higher_is_better());
        assert!(!Metric::WER.higher_is_better());
        assert!(Metric::RTFx.higher_is_better());
        assert!(Metric::BLEU.higher_is_better());
        assert!(Metric::ROUGE(RougeVariant::Rouge1).higher_is_better());
        assert!(!Metric::Perplexity.higher_is_better());
        assert!(Metric::MMLUAccuracy.higher_is_better());
        assert!(Metric::PassAtK(1).higher_is_better());
        assert!(Metric::NDCGAtK(5).higher_is_better());
    }

    #[test]
    fn test_rouge_variant_display() {
        assert_eq!(RougeVariant::Rouge1.to_string(), "ROUGE-1");
        assert_eq!(RougeVariant::Rouge2.to_string(), "ROUGE-2");
        assert_eq!(RougeVariant::RougeL.to_string(), "ROUGE-L");
    }
}
