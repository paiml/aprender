//! Classification evaluation report and checkpoint evaluation.
//!
//! Extracted from `classify_trainer.rs` to reduce file size.

use super::classify_pipeline::ClassifyPipeline;
use crate::eval::classification::{ConfusionMatrix, MultiClassMetrics};
use std::path::Path;

/// Evaluation report from the classification pipeline.
///
/// Contains per-class precision/recall/F1, confusion matrix, aggregate metrics,
/// and advanced diagnostics (Cohen's kappa, MCC, calibration, confidence distribution).
/// Produced by [`ClassifyTrainer::evaluate`] or [`evaluate_checkpoint`].
#[derive(Debug, Clone)]
pub struct ClassifyEvalReport {
    /// Overall accuracy (0.0-1.0)
    pub accuracy: f64,
    /// Average cross-entropy loss
    pub avg_loss: f32,
    /// Per-class precision (0.0-1.0)
    pub per_class_precision: Vec<f64>,
    /// Per-class recall (0.0-1.0)
    pub per_class_recall: Vec<f64>,
    /// Per-class F1 score (0.0-1.0)
    pub per_class_f1: Vec<f64>,
    /// Per-class support (sample count)
    pub per_class_support: Vec<usize>,
    /// Confusion matrix: `confusion_matrix[true][predicted]`
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Number of classes
    pub num_classes: usize,
    /// Total samples evaluated
    pub total_samples: usize,
    /// Evaluation wall-clock time in milliseconds
    pub eval_time_ms: u64,
    /// Human-readable class names
    pub label_names: Vec<String>,
    /// Cohen's kappa (chance-corrected agreement, -1 to 1)
    pub cohens_kappa: f64,
    /// Matthews Correlation Coefficient (-1 to 1, robust to class imbalance)
    pub mcc: f64,
    /// Top-2 accuracy (correct class in top 2 predictions)
    pub top2_accuracy: f64,
    /// Mean prediction confidence (max softmax probability)
    pub mean_confidence: f64,
    /// Mean confidence when prediction is correct
    pub mean_confidence_correct: f64,
    /// Mean confidence when prediction is wrong
    pub mean_confidence_wrong: f64,
    /// Samples per second throughput
    pub samples_per_sec: f64,
    /// Calibration bins: (mean_confidence, mean_accuracy, count) for 10 bins
    pub calibration_bins: Vec<(f64, f64, usize)>,
    /// Expected Calibration Error (lower is better, 0 = perfectly calibrated)
    pub ece: f64,
    /// Brier score (multi-class mean squared error of probabilities, lower is better)
    pub brier_score: f64,
    /// Log loss (negative log-likelihood of true class, lower is better)
    pub log_loss: f64,
    /// Bootstrap 95% confidence intervals: (lower, upper) for (accuracy, macro_f1, mcc)
    pub ci_accuracy: (f64, f64),
    pub ci_macro_f1: (f64, f64),
    pub ci_mcc: (f64, f64),
    /// Baseline accuracies: (random, majority_class, stratified_random)
    pub baseline_random: f64,
    pub baseline_majority: f64,
    /// Most confused class pairs: (true_class, pred_class, count)
    pub top_confusions: Vec<(usize, usize, usize)>,
}

impl ClassifyEvalReport {
    /// Build a report from raw predictions with full probability distributions.
    ///
    /// Computes all metrics including Cohen's kappa, MCC, calibration ECE,
    /// top-2 accuracy, and confidence analysis.
    pub(crate) fn from_predictions_with_probs(
        y_pred: &[usize],
        y_true: &[usize],
        all_probs: &[Vec<f32>],
        total_loss: f32,
        num_classes: usize,
        label_names: &[String],
        eval_time_ms: u64,
    ) -> Self {
        let total_samples = y_pred.len();
        let avg_loss = if total_samples > 0 { total_loss / total_samples as f32 } else { 0.0 };

        let cm = ConfusionMatrix::from_predictions_with_min_classes(y_pred, y_true, num_classes);
        let metrics = MultiClassMetrics::from_confusion_matrix(&cm);
        let accuracy = cm.accuracy();

        let cohens_kappa = Self::compute_cohens_kappa(&cm, total_samples);
        let mcc = Self::compute_mcc(&cm, cm.n_classes(), total_samples);
        let top2_accuracy = Self::compute_top2_accuracy(all_probs, y_true, total_samples);

        let confidences: Vec<f64> =
            all_probs.iter().map(|p| f64::from(p.iter().copied().fold(0.0f32, f32::max))).collect();

        let (mean_confidence, mean_confidence_correct, mean_confidence_wrong) =
            Self::compute_confidence_stats(&confidences, y_pred, y_true);

        let (calibration_bins, ece) =
            Self::compute_calibration(&confidences, y_pred, y_true, total_samples);

        let samples_per_sec = if eval_time_ms > 0 {
            total_samples as f64 / (eval_time_ms as f64 / 1000.0)
        } else {
            0.0
        };

        let brier_score = Self::compute_brier_score(all_probs, y_true, num_classes);
        let log_loss = Self::compute_log_loss(all_probs, y_true);

        let (ci_accuracy, ci_macro_f1, ci_mcc) =
            Self::compute_bootstrap_cis(y_pred, y_true, num_classes, 1000);

        let (baseline_random, baseline_majority) =
            Self::compute_baselines(&metrics.support, total_samples, num_classes);

        let top_confusions = Self::compute_top_confusions(cm.matrix(), 5);

        Self {
            accuracy,
            avg_loss,
            per_class_precision: metrics.precision,
            per_class_recall: metrics.recall,
            per_class_f1: metrics.f1,
            per_class_support: metrics.support,
            confusion_matrix: cm.matrix().clone(),
            num_classes,
            total_samples,
            eval_time_ms,
            label_names: label_names.to_vec(),
            cohens_kappa,
            mcc,
            top2_accuracy,
            mean_confidence,
            mean_confidence_correct,
            mean_confidence_wrong,
            samples_per_sec,
            calibration_bins,
            ece,
            brier_score,
            log_loss,
            ci_accuracy,
            ci_macro_f1,
            ci_mcc,
            baseline_random,
            baseline_majority,
            top_confusions,
        }
    }

    /// Compute top-2 accuracy: fraction of samples where the true label is in the top 2 predictions.
    pub(crate) fn compute_top2_accuracy(
        all_probs: &[Vec<f32>],
        y_true: &[usize],
        total: usize,
    ) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let correct = all_probs
            .iter()
            .zip(y_true.iter())
            .filter(|(probs, &true_label)| {
                let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                indexed.len() >= 2 && (indexed[0].0 == true_label || indexed[1].0 == true_label)
            })
            .count();
        correct as f64 / total as f64
    }

    /// Compute mean confidence overall, for correct predictions, and for wrong predictions.
    pub(crate) fn compute_confidence_stats(
        confidences: &[f64],
        y_pred: &[usize],
        y_true: &[usize],
    ) -> (f64, f64, f64) {
        let mean = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f64>() / confidences.len() as f64
        };

        let (mut sum_correct, mut n_correct) = (0.0f64, 0usize);
        let (mut sum_wrong, mut n_wrong) = (0.0f64, 0usize);
        for (i, &conf) in confidences.iter().enumerate() {
            if y_pred[i] == y_true[i] {
                sum_correct += conf;
                n_correct += 1;
            } else {
                sum_wrong += conf;
                n_wrong += 1;
            }
        }

        let mean_correct = if n_correct > 0 { sum_correct / n_correct as f64 } else { 0.0 };
        let mean_wrong = if n_wrong > 0 { sum_wrong / n_wrong as f64 } else { 0.0 };

        (mean, mean_correct, mean_wrong)
    }

    /// Compute calibration bins (10 equal-width bins) and Expected Calibration Error.
    pub(crate) fn compute_calibration(
        confidences: &[f64],
        y_pred: &[usize],
        y_true: &[usize],
        total_samples: usize,
    ) -> (Vec<(f64, f64, usize)>, f64) {
        let num_bins = 10;
        let mut bin_sum_conf = vec![0.0f64; num_bins];
        let mut bin_sum_acc = vec![0.0f64; num_bins];
        let mut bin_count = vec![0usize; num_bins];

        for (i, &conf) in confidences.iter().enumerate() {
            let bin = ((conf * num_bins as f64) as usize).min(num_bins - 1);
            bin_sum_conf[bin] += conf;
            bin_sum_acc[bin] += if y_pred[i] == y_true[i] { 1.0 } else { 0.0 };
            bin_count[bin] += 1;
        }

        let bins: Vec<(f64, f64, usize)> = (0..num_bins)
            .map(|b| {
                if bin_count[b] > 0 {
                    (
                        bin_sum_conf[b] / bin_count[b] as f64,
                        bin_sum_acc[b] / bin_count[b] as f64,
                        bin_count[b],
                    )
                } else {
                    (0.0, 0.0, 0)
                }
            })
            .collect();

        let ece: f64 = bins
            .iter()
            .map(|&(conf, acc, count)| {
                if count > 0 {
                    (conf - acc).abs() * count as f64 / total_samples as f64
                } else {
                    0.0
                }
            })
            .sum();

        (bins, ece)
    }

    /// Cohen's kappa: chance-corrected agreement.
    ///
    /// kappa = (p_o - p_e) / (1 - p_e)
    /// where p_o = observed agreement (accuracy), p_e = expected agreement by chance
    pub(crate) fn compute_cohens_kappa(cm: &ConfusionMatrix, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let mat = cm.matrix();
        let n = total as f64;
        let p_o = cm.accuracy();

        // p_e = sum_k (row_k_total * col_k_total) / n^2
        let k = mat.len();
        let mut p_e = 0.0f64;
        for class in 0..k {
            let row_sum: f64 = mat[class].iter().sum::<usize>() as f64;
            let col_sum: f64 = mat.iter().map(|row| row[class]).sum::<usize>() as f64;
            p_e += (row_sum * col_sum) / (n * n);
        }

        if (1.0 - p_e).abs() < 1e-10 {
            return if (p_o - 1.0).abs() < 1e-10 { 1.0 } else { 0.0 };
        }

        (p_o - p_e) / (1.0 - p_e)
    }

    /// Matthews Correlation Coefficient for multiclass.
    ///
    /// MCC = (c * s - sum_k(p_k * t_k)) / sqrt((s^2 - sum_k(p_k^2)) * (s^2 - sum_k(t_k^2)))
    /// where c = correct predictions, s = total samples, p_k = predicted counts, t_k = true counts
    pub(crate) fn compute_mcc(cm: &ConfusionMatrix, num_classes: usize, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        let mat = cm.matrix();
        let s = total as f64;

        // c = sum of diagonal (correct predictions)
        let c: f64 = (0..num_classes).map(|k| mat[k][k] as f64).sum();

        // p_k = column sums (predicted counts per class)
        let p: Vec<f64> =
            (0..num_classes).map(|k| mat.iter().map(|row| row[k] as f64).sum()).collect();

        // t_k = row sums (true counts per class)
        let t: Vec<f64> = (0..num_classes).map(|k| mat[k].iter().sum::<usize>() as f64).collect();

        let sum_pk_tk: f64 = p.iter().zip(t.iter()).map(|(pk, tk)| pk * tk).sum();
        let sum_pk_sq: f64 = p.iter().map(|pk| pk * pk).sum();
        let sum_tk_sq: f64 = t.iter().map(|tk| tk * tk).sum();

        let numer = c * s - sum_pk_tk;
        let denom_sq = (s * s - sum_pk_sq) * (s * s - sum_tk_sq);

        if denom_sq <= 0.0 {
            return 0.0;
        }

        numer / denom_sq.sqrt()
    }

    /// Multi-class Brier score: mean of sum_k (p_k - y_k)^2 across samples.
    pub(crate) fn compute_brier_score(
        all_probs: &[Vec<f32>],
        y_true: &[usize],
        num_classes: usize,
    ) -> f64 {
        if all_probs.is_empty() {
            return 0.0;
        }
        let total: f64 = all_probs
            .iter()
            .zip(y_true.iter())
            .map(|(probs, &true_label)| {
                (0..num_classes)
                    .map(|k| {
                        let p = f64::from(*probs.get(k).unwrap_or(&0.0));
                        let y = if k == true_label { 1.0 } else { 0.0 };
                        (p - y) * (p - y)
                    })
                    .sum::<f64>()
            })
            .sum();
        total / all_probs.len() as f64
    }

    /// Log loss: -mean(log(p_true_class)).
    pub(crate) fn compute_log_loss(all_probs: &[Vec<f32>], y_true: &[usize]) -> f64 {
        if all_probs.is_empty() {
            return 0.0;
        }
        let eps = 1e-15_f64;
        let total: f64 = all_probs
            .iter()
            .zip(y_true.iter())
            .map(|(probs, &true_label)| {
                let p = f64::from(probs.get(true_label).copied().unwrap_or(0.0));
                -p.clamp(eps, 1.0 - eps).ln()
            })
            .sum();
        total / all_probs.len() as f64
    }

    /// Bootstrap 95% confidence intervals for accuracy, macro F1, and MCC.
    ///
    /// Uses percentile method with `n_boot` resamples. Deterministic seed
    /// for reproducibility.
    pub(crate) fn compute_bootstrap_cis(
        y_pred: &[usize],
        y_true: &[usize],
        num_classes: usize,
        n_boot: usize,
    ) -> ((f64, f64), (f64, f64), (f64, f64)) {
        let n = y_pred.len();
        if n == 0 {
            return ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0));
        }

        let mut accs = Vec::with_capacity(n_boot);
        let mut f1s = Vec::with_capacity(n_boot);
        let mut mccs = Vec::with_capacity(n_boot);

        // Simple LCG PRNG (deterministic, no dependency needed)
        let mut rng_state: u64 = 42;
        let lcg_next = |state: &mut u64| -> usize {
            *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((*state >> 33) as usize) % n
        };

        for _ in 0..n_boot {
            // Resample with replacement
            let mut boot_pred = Vec::with_capacity(n);
            let mut boot_true = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = lcg_next(&mut rng_state);
                boot_pred.push(y_pred[idx]);
                boot_true.push(y_true[idx]);
            }

            let cm = ConfusionMatrix::from_predictions_with_min_classes(
                &boot_pred,
                &boot_true,
                num_classes,
            );
            let metrics = MultiClassMetrics::from_confusion_matrix(&cm);

            accs.push(cm.accuracy());

            // Macro F1
            let valid_f1: Vec<f64> = metrics.f1.iter().copied().filter(|v| !v.is_nan()).collect();
            let macro_f1 = if valid_f1.is_empty() {
                0.0
            } else {
                valid_f1.iter().sum::<f64>() / valid_f1.len() as f64
            };
            f1s.push(macro_f1);

            mccs.push(Self::compute_mcc(&cm, cm.n_classes(), n));
        }

        accs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        f1s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        mccs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lo = (0.025 * n_boot as f64) as usize;
        let hi = (0.975 * n_boot as f64).ceil() as usize;
        let hi = hi.min(n_boot - 1);

        ((accs[lo], accs[hi]), (f1s[lo], f1s[hi]), (mccs[lo], mccs[hi]))
    }

    /// Compute baseline accuracies: random and majority-class.
    pub(crate) fn compute_baselines(
        support: &[usize],
        total: usize,
        num_classes: usize,
    ) -> (f64, f64) {
        let random = if num_classes > 0 { 1.0 / num_classes as f64 } else { 0.0 };
        let majority = if total > 0 {
            support.iter().copied().max().unwrap_or(0) as f64 / total as f64
        } else {
            0.0
        };
        (random, majority)
    }

    /// Extract top-N most confused class pairs from the confusion matrix (off-diagonal).
    pub(crate) fn compute_top_confusions(
        matrix: &[Vec<usize>],
        top_n: usize,
    ) -> Vec<(usize, usize, usize)> {
        let mut pairs: Vec<(usize, usize, usize)> = Vec::new();
        for (i, row) in matrix.iter().enumerate() {
            for (j, &count) in row.iter().enumerate() {
                if i != j && count > 0 {
                    pairs.push((i, j, count));
                }
            }
        }
        pairs.sort_by(|a, b| b.2.cmp(&a.2));
        pairs.truncate(top_n);
        pairs
    }

    /// Format as a human-readable sklearn-style classification report.
    #[must_use]
    pub fn to_report(&self) -> String {
        use crate::eval::classification::Average;

        let mut out = String::new();

        // Header
        out.push_str(&format!(
            "{:>18} {:>10} {:>10} {:>10} {:>10}\n",
            "", "precision", "recall", "f1-score", "support"
        ));
        out.push_str(&format!("{}\n", "-".repeat(62)));

        // Per-class rows
        for i in 0..self.num_classes {
            let name = self
                .label_names
                .get(i)
                .map_or_else(|| format!("Class {i}"), std::clone::Clone::clone);
            out.push_str(&format!(
                "{:>18} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
                name,
                self.per_class_precision[i],
                self.per_class_recall[i],
                self.per_class_f1[i],
                self.per_class_support[i],
            ));
        }

        out.push_str(&format!("{}\n", "-".repeat(62)));

        let total_support: usize = self.per_class_support.iter().sum();

        // Macro average
        let macro_p = self.avg_metric(&self.per_class_precision, Average::Macro);
        let macro_r = self.avg_metric(&self.per_class_recall, Average::Macro);
        let macro_f1 = self.avg_metric(&self.per_class_f1, Average::Macro);
        out.push_str(&format!(
            "{:>18} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "macro avg", macro_p, macro_r, macro_f1, total_support,
        ));

        // Weighted average
        let weighted_p = self.avg_metric(&self.per_class_precision, Average::Weighted);
        let weighted_r = self.avg_metric(&self.per_class_recall, Average::Weighted);
        let weighted_f1 = self.avg_metric(&self.per_class_f1, Average::Weighted);
        out.push_str(&format!(
            "{:>18} {:>10.4} {:>10.4} {:>10.4} {:>10}\n",
            "weighted avg", weighted_p, weighted_r, weighted_f1, total_support,
        ));

        // ── Summary metrics ───────────────────────────────────────
        self.report_summary(&mut out);
        self.report_confidence(&mut out);
        self.report_scoring_rules(&mut out);
        self.report_calibration(&mut out);
        self.report_baselines(&mut out);
        self.report_top_confusions(&mut out);
        self.report_throughput(&mut out);
        out
    }

    pub(crate) fn report_summary(&self, out: &mut String) {
        out.push_str(&format!(
            "\nAccuracy:       {:.4}  ({:.1}%)  95% CI [{:.4}, {:.4}]\n",
            self.accuracy,
            self.accuracy * 100.0,
            self.ci_accuracy.0,
            self.ci_accuracy.1
        ));
        out.push_str(&format!(
            "Top-2 accuracy: {:.4}  ({:.1}%)\n",
            self.top2_accuracy,
            self.top2_accuracy * 100.0
        ));
        out.push_str(&format!(
            "Cohen's kappa:  {:.4}  ({})\n",
            self.cohens_kappa,
            Self::kappa_interpretation(self.cohens_kappa)
        ));
        out.push_str(&format!(
            "MCC:            {:.4}  95% CI [{:.4}, {:.4}]\n",
            self.mcc, self.ci_mcc.0, self.ci_mcc.1
        ));
        out.push_str(&format!(
            "Macro F1:       {:.4}  95% CI [{:.4}, {:.4}]\n",
            self.avg_metric(&self.per_class_f1, crate::eval::classification::Average::Macro),
            self.ci_macro_f1.0,
            self.ci_macro_f1.1
        ));
        out.push_str(&format!("Avg loss:       {:.4}\n", self.avg_loss));
    }

    pub(crate) fn report_confidence(&self, out: &mut String) {
        out.push_str(&format!("\nConfidence (mean): {:.4}\n", self.mean_confidence));
        out.push_str(&format!("  correct preds:   {:.4}\n", self.mean_confidence_correct));
        out.push_str(&format!("  wrong preds:     {:.4}\n", self.mean_confidence_wrong));
        let gap = self.mean_confidence_correct - self.mean_confidence_wrong;
        out.push_str(&format!("  gap (higher=better): {gap:.4}\n"));
    }

    pub(crate) fn report_scoring_rules(&self, out: &mut String) {
        out.push_str(&format!(
            "\nBrier score:    {:.4}  (perfect=0, random={:.4})\n",
            self.brier_score,
            1.0 - 1.0 / self.num_classes as f64
        ));
        out.push_str(&format!(
            "Log loss:       {:.4}  (random={:.4})\n",
            self.log_loss,
            (self.num_classes as f64).ln()
        ));
    }

    pub(crate) fn report_calibration(&self, out: &mut String) {
        out.push_str(&format!("\nECE (Expected Calibration Error): {:.4}\n", self.ece));
        out.push_str("Calibration:\n");
        out.push_str("  Bin       Confidence  Accuracy    Count\n");
        for (i, &(conf, acc, count)) in self.calibration_bins.iter().enumerate() {
            if count > 0 {
                let lo = i as f64 * 0.1;
                let hi = lo + 0.1;
                let overconf = if conf > acc { "+" } else { "" };
                out.push_str(&format!(
                    "  [{:.1}-{:.1})  {:.4}      {:.4}      {:>5}  {overconf}{:.3}\n",
                    lo,
                    hi,
                    conf,
                    acc,
                    count,
                    conf - acc,
                ));
            }
        }
    }

    pub(crate) fn report_baselines(&self, out: &mut String) {
        out.push_str(&format!(
            "\nBaselines:  random={:.1}%  majority={:.1}%  model={:.1}%  lift={:.1}x\n",
            self.baseline_random * 100.0,
            self.baseline_majority * 100.0,
            self.accuracy * 100.0,
            if self.baseline_majority > 0.0 { self.accuracy / self.baseline_majority } else { 0.0 },
        ));
    }

    pub(crate) fn report_top_confusions(&self, out: &mut String) {
        if self.top_confusions.is_empty() {
            return;
        }
        out.push_str("\nTop confusions (true → predicted, count):\n");
        for &(true_c, pred_c, count) in &self.top_confusions {
            let true_name = self.label_names.get(true_c).map_or("?", |n| n.as_str());
            let pred_name = self.label_names.get(pred_c).map_or("?", |n| n.as_str());
            out.push_str(&format!("  {true_name} → {pred_name}: {count}\n"));
        }
    }

    pub(crate) fn report_throughput(&self, out: &mut String) {
        out.push_str(&format!("\nSamples:   {}\n", self.total_samples));
        out.push_str(&format!(
            "Time:      {}ms ({:.1} samples/sec)\n",
            self.eval_time_ms, self.samples_per_sec
        ));
    }

    /// Interpret Cohen's kappa value.
    pub(crate) fn kappa_interpretation(kappa: f64) -> &'static str {
        if kappa < 0.0 {
            "worse than chance"
        } else if kappa < 0.20 {
            "slight"
        } else if kappa < 0.40 {
            "fair"
        } else if kappa < 0.60 {
            "moderate"
        } else if kappa < 0.80 {
            "substantial"
        } else {
            "almost perfect"
        }
    }

    /// Format as JSON string.
    ///
    /// Uses `serde_json::json!` internally — infallible.
    #[must_use]
    #[allow(clippy::disallowed_methods)]
    pub fn to_json(&self) -> String {
        let per_class: Vec<serde_json::Value> = (0..self.num_classes)
            .map(|i| {
                let name = self
                    .label_names
                    .get(i)
                    .map_or_else(|| format!("class_{i}"), std::clone::Clone::clone);
                serde_json::json!({
                    "label": name,
                    "precision": self.per_class_precision[i],
                    "recall": self.per_class_recall[i],
                    "f1": self.per_class_f1[i],
                    "support": self.per_class_support[i],
                })
            })
            .collect();

        let calibration: Vec<serde_json::Value> = self
            .calibration_bins
            .iter()
            .enumerate()
            .filter(|(_, &(_, _, count))| count > 0)
            .map(|(i, &(conf, acc, count))| {
                serde_json::json!({
                    "bin": format!("[{:.1}-{:.1})", i as f64 * 0.1, (i + 1) as f64 * 0.1),
                    "mean_confidence": conf,
                    "mean_accuracy": acc,
                    "count": count,
                })
            })
            .collect();

        let confusions: Vec<serde_json::Value> = self.top_confusions.iter().map(|&(t, p, c)| {
            serde_json::json!({
                "true_class": self.label_names.get(t).cloned().unwrap_or_else(|| format!("class_{t}")),
                "pred_class": self.label_names.get(p).cloned().unwrap_or_else(|| format!("class_{p}")),
                "count": c,
            })
        }).collect();

        let json = serde_json::json!({
            "accuracy": self.accuracy,
            "top2_accuracy": self.top2_accuracy,
            "cohens_kappa": self.cohens_kappa,
            "mcc": self.mcc,
            "avg_loss": self.avg_loss,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "eval_time_ms": self.eval_time_ms,
            "samples_per_sec": self.samples_per_sec,
            "confidence_intervals_95": {
                "accuracy": [self.ci_accuracy.0, self.ci_accuracy.1],
                "macro_f1": [self.ci_macro_f1.0, self.ci_macro_f1.1],
                "mcc": [self.ci_mcc.0, self.ci_mcc.1],
            },
            "baselines": {
                "random": self.baseline_random,
                "majority_class": self.baseline_majority,
                "lift_over_majority": if self.baseline_majority > 0.0 { self.accuracy / self.baseline_majority } else { 0.0 },
            },
            "per_class": per_class,
            "confusion_matrix": self.confusion_matrix,
            "top_confusions": confusions,
            "confidence": {
                "mean": self.mean_confidence,
                "mean_correct": self.mean_confidence_correct,
                "mean_wrong": self.mean_confidence_wrong,
                "gap": self.mean_confidence_correct - self.mean_confidence_wrong,
            },
            "calibration": {
                "ece": self.ece,
                "brier_score": self.brier_score,
                "log_loss": self.log_loss,
                "bins": calibration,
            },
        });

        serde_json::to_string_pretty(&json).unwrap_or_default()
    }

    /// Generate a HuggingFace-compatible model card (README.md) from evaluation results.
    ///
    /// Produces a publication-quality model card with YAML front matter, summary metrics,
    /// per-class breakdown, confusion matrix (raw + normalized), calibration analysis,
    /// intended use, limitations, and ethical considerations.
    #[must_use]
    pub fn to_model_card(&self, model_name: &str, base_model: Option<&str>) -> String {
        use crate::eval::classification::Average;

        let macro_f1 = self.avg_metric(&self.per_class_f1, Average::Macro);
        let weighted_f1 = self.avg_metric(&self.per_class_f1, Average::Weighted);

        let mut out = String::new();
        self.card_yaml_front_matter(&mut out, model_name, base_model, macro_f1, weighted_f1);
        self.card_title(&mut out, model_name, base_model);
        self.card_summary(&mut out, macro_f1, weighted_f1);
        self.card_labels(&mut out);
        self.card_per_class_metrics(&mut out);
        self.card_confusion_matrix(&mut out);
        self.card_error_analysis(&mut out);
        self.card_calibration(&mut out);
        Self::card_intended_use(&mut out);
        self.card_limitations(&mut out);
        Self::card_ethical_considerations(&mut out);
        self.card_training(&mut out, base_model);
        out.push_str("---\n*Generated by [entrenar](https://github.com/paiml/entrenar)*\n");
        out
    }

    pub(crate) fn card_yaml_front_matter(
        &self,
        out: &mut String,
        model_name: &str,
        base_model: Option<&str>,
        macro_f1: f64,
        weighted_f1: f64,
    ) {
        out.push_str("---\n");
        out.push_str("license: apache-2.0\n");
        out.push_str("language:\n- en\n");
        out.push_str(
            "tags:\n- shell-safety\n- code-classification\n- lora\n- entrenar\n- security\n",
        );
        if let Some(base) = base_model {
            out.push_str(&format!("base_model: {base}\n"));
        }
        out.push_str("pipeline_tag: text-classification\n");
        out.push_str("model-index:\n");
        out.push_str(&format!("- name: {model_name}\n"));
        out.push_str("  results:\n");
        out.push_str("  - task:\n");
        out.push_str("      type: text-classification\n");
        out.push_str("      name: Shell Safety Classification\n");
        out.push_str("    metrics:\n");
        out.push_str(&format!("    - type: accuracy\n      value: {:.4}\n", self.accuracy));
        out.push_str(&format!(
            "    - type: f1\n      value: {macro_f1:.4}\n      name: Macro F1\n"
        ));
        out.push_str(&format!(
            "    - type: f1\n      value: {weighted_f1:.4}\n      name: Weighted F1\n"
        ));
        out.push_str(&format!("    - type: mcc\n      value: {:.4}\n", self.mcc));
        out.push_str(&format!("    - type: cohens_kappa\n      value: {:.4}\n", self.cohens_kappa));
        out.push_str("---\n\n");
    }

    pub(crate) fn card_title(&self, out: &mut String, model_name: &str, base_model: Option<&str>) {
        out.push_str(&format!("# {model_name}\n\n"));
        out.push_str("A shell command safety classifier that categorizes shell commands into safety classes, ");
        out.push_str("enabling automated triage of commands before execution.\n\n");
        out.push_str(
            "Trained with [entrenar](https://github.com/paiml/entrenar) using LoRA fine-tuning",
        );
        if let Some(base) = base_model {
            out.push_str(&format!(" on [`{base}`](https://huggingface.co/{base})"));
        }
        out.push_str(".\n\n");
    }

    pub(crate) fn card_summary(&self, out: &mut String, macro_f1: f64, weighted_f1: f64) {
        out.push_str("## Summary\n\n");
        out.push_str("| Metric | Value | 95% CI |\n");
        out.push_str("|--------|-------|--------|\n");
        out.push_str(&format!(
            "| Accuracy | {:.2}% | [{:.2}%, {:.2}%] |\n",
            self.accuracy * 100.0,
            self.ci_accuracy.0 * 100.0,
            self.ci_accuracy.1 * 100.0
        ));
        out.push_str(&format!("| Top-2 Accuracy | {:.2}% | — |\n", self.top2_accuracy * 100.0));
        out.push_str(&format!(
            "| Macro F1 | {macro_f1:.4} | [{:.4}, {:.4}] |\n",
            self.ci_macro_f1.0, self.ci_macro_f1.1
        ));
        out.push_str(&format!("| Weighted F1 | {weighted_f1:.4} | — |\n"));
        out.push_str(&format!(
            "| Cohen's Kappa | {:.4} ({}) | — |\n",
            self.cohens_kappa,
            Self::kappa_interpretation(self.cohens_kappa)
        ));
        out.push_str(&format!(
            "| MCC | {:.4} | [{:.4}, {:.4}] |\n",
            self.mcc, self.ci_mcc.0, self.ci_mcc.1
        ));
        out.push_str(&format!("| Brier Score | {:.4} | — |\n", self.brier_score));
        out.push_str(&format!("| Log Loss | {:.4} | — |\n", self.log_loss));
        out.push_str(&format!("| ECE | {:.4} | — |\n", self.ece));
        out.push_str(&format!("| Avg Loss | {:.4} | — |\n", self.avg_loss));
        out.push_str(&format!("| Eval Samples | {} | — |\n", self.total_samples));
        out.push_str(&format!("| Throughput | {:.1} samples/sec | — |\n\n", self.samples_per_sec));

        let lift =
            if self.baseline_majority > 0.0 { self.accuracy / self.baseline_majority } else { 0.0 };
        out.push_str("**Baselines**: ");
        out.push_str(&format!(
            "random={:.1}%, majority={:.1}%, model={:.1}% ({:.1}x lift over majority)\n\n",
            self.baseline_random * 100.0,
            self.baseline_majority * 100.0,
            self.accuracy * 100.0,
            lift
        ));
    }

    pub(crate) fn card_labels(&self, out: &mut String) {
        out.push_str("## Labels\n\n");
        out.push_str("| ID | Label | Description |\n");
        out.push_str("|----|-------|-------------|\n");
        let descriptions = [
            "Command is safe to execute as-is",
            "Command contains unquoted variable expansions (word splitting/globbing risk)",
            "Command uses non-deterministic sources ($RANDOM, $$, date, etc.)",
            "Command is not idempotent (unsafe to re-run: mkdir without -p, etc.)",
            "Command is destructive or has injection risk (rm -rf, eval, etc.)",
        ];
        for (i, name) in self.label_names.iter().enumerate() {
            let desc = descriptions.get(i).unwrap_or(&"");
            out.push_str(&format!("| {i} | {name} | {desc} |\n"));
        }
        out.push('\n');
    }

    pub(crate) fn card_per_class_metrics(&self, out: &mut String) {
        out.push_str("## Per-Class Metrics\n\n");
        out.push_str("| Label | Precision | Recall | F1 | Support |\n");
        out.push_str("|-------|-----------|--------|----|---------|\n");
        for i in 0..self.num_classes {
            let name = self
                .label_names
                .get(i)
                .map_or_else(|| format!("class_{i}"), std::clone::Clone::clone);
            out.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:.4} | {} |\n",
                name,
                self.per_class_precision[i],
                self.per_class_recall[i],
                self.per_class_f1[i],
                self.per_class_support[i],
            ));
        }
        out.push('\n');
    }

    fn card_confusion_matrix(&self, out: &mut String) {
        out.push_str("## Confusion Matrix\n\n");
        self.card_confusion_raw(out);
        self.card_confusion_normalized(out);
    }

    pub(crate) fn card_confusion_raw(&self, out: &mut String) {
        out.push_str("### Raw Counts\n\n```\n");
        self.card_confusion_header(out);
        for (i, row) in self.confusion_matrix.iter().enumerate() {
            self.card_confusion_row_label(out, i);
            for val in row {
                out.push_str(&format!(" {val:>8}"));
            }
            out.push('\n');
        }
        out.push_str("```\n\n");
    }

    pub(crate) fn card_confusion_normalized(&self, out: &mut String) {
        out.push_str("### Normalized (row %)\n\n```\n");
        self.card_confusion_header(out);
        for (i, row) in self.confusion_matrix.iter().enumerate() {
            self.card_confusion_row_label(out, i);
            let row_sum: usize = row.iter().sum();
            for val in row {
                if row_sum > 0 {
                    out.push_str(&format!(" {:>7.1}%", *val as f64 / row_sum as f64 * 100.0));
                } else {
                    out.push_str("     0.0%");
                }
            }
            out.push('\n');
        }
        out.push_str("```\n\n");
    }

    pub(crate) fn card_confusion_header(&self, out: &mut String) {
        out.push_str(&format!("{:>18}", "Predicted →"));
        for name in &self.label_names {
            let short = if name.len() > 8 { &name[..8] } else { name.as_str() };
            out.push_str(&format!(" {short:>8}"));
        }
        out.push('\n');
    }

    pub(crate) fn card_confusion_row_label(&self, out: &mut String, i: usize) {
        let name =
            self.label_names.get(i).map_or_else(|| format!("class_{i}"), std::clone::Clone::clone);
        let short = if name.len() > 18 { &name[..18] } else { name.as_str() };
        out.push_str(&format!("{short:>18}"));
    }

    pub(crate) fn card_calibration(&self, out: &mut String) {
        out.push_str("## Confidence & Calibration\n\n");
        out.push_str("| Metric | Value |\n");
        out.push_str("|--------|-------|\n");
        out.push_str(&format!("| Mean confidence | {:.4} |\n", self.mean_confidence));
        out.push_str(&format!("| Confidence (correct) | {:.4} |\n", self.mean_confidence_correct));
        out.push_str(&format!("| Confidence (wrong) | {:.4} |\n", self.mean_confidence_wrong));
        let gap = self.mean_confidence_correct - self.mean_confidence_wrong;
        out.push_str(&format!("| Confidence gap | {gap:.4} |\n"));
        out.push_str(&format!("| ECE | {:.4} |\n\n", self.ece));

        out.push_str("**Calibration curve** (reliability diagram):\n\n");
        out.push_str("```\n");
        out.push_str("Bin         Conf    Acc     Count\n");
        for (i, &(conf, acc, count)) in self.calibration_bins.iter().enumerate() {
            if count > 0 {
                let lo = i as f64 * 0.1;
                let hi = lo + 0.1;
                out.push_str(&format!("[{lo:.1}-{hi:.1})   {conf:.3}   {acc:.3}   {count:>5}\n",));
            }
        }
        out.push_str("```\n\n");
    }

    pub(crate) fn card_error_analysis(&self, out: &mut String) {
        if self.top_confusions.is_empty() {
            return;
        }
        out.push_str("## Error Analysis\n\n");
        out.push_str("Most frequent misclassifications:\n\n");
        out.push_str("| True Class | Predicted As | Count |\n");
        out.push_str("|------------|-------------|-------|\n");
        for &(true_c, pred_c, count) in &self.top_confusions {
            let true_name = self.label_names.get(true_c).map_or("?", |n| n.as_str());
            let pred_name = self.label_names.get(pred_c).map_or("?", |n| n.as_str());
            out.push_str(&format!("| {true_name} | {pred_name} | {count} |\n"));
        }
        out.push('\n');
    }

    pub(crate) fn card_intended_use(out: &mut String) {
        out.push_str("## Intended Use\n\n");
        out.push_str(
            "This model is designed for **automated shell command safety triage** in:\n\n",
        );
        out.push_str(
            "- **CI/CD pipelines**: Pre-flight safety check before executing generated scripts\n",
        );
        out.push_str("- **Shell purification tools**: Classify commands to determine transformation strategy\n");
        out.push_str(
            "- **Code review**: Flag potentially unsafe shell commands in pull requests\n",
        );
        out.push_str("- **Interactive shells**: Warn users before executing risky commands\n\n");
    }

    fn card_limitations(&self, out: &mut String) {
        out.push_str("## Limitations\n\n");
        out.push_str("- **Not a security oracle**: This model provides *classification hints*, not security guarantees\n");
        out.push_str("- **Context-blind**: Cannot assess safety based on the execution environment or user permissions\n");
        out.push_str("- **Training distribution**: Trained on synthetic shell scripts; may underperform on novel patterns\n");
        out.push_str(
            "- **English only**: Command names and variable patterns are English-centric\n",
        );
        self.card_weak_classes(out);
        out.push('\n');
    }

    pub(crate) fn card_weak_classes(&self, out: &mut String) {
        let min_f1_idx = self
            .per_class_f1
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i);
        if let Some(idx) = min_f1_idx {
            if self.per_class_f1[idx] < 0.5 {
                let name = self.label_names.get(idx).map_or("unknown", |n| n.as_str());
                out.push_str(&format!(
                    "- **Weak class**: `{name}` (F1={:.2}) — consider additional training data\n",
                    self.per_class_f1[idx]
                ));
            }
        }
    }

    pub(crate) fn card_ethical_considerations(out: &mut String) {
        out.push_str("## Ethical Considerations\n\n");
        out.push_str("- **False negatives are dangerous**: An `unsafe` command classified as `safe` could lead to data loss\n");
        out.push_str("- **Defense in depth**: Always combine this classifier with other safety mechanisms (sandboxing, dry-run)\n");
        out.push_str("- **Not adversarial-robust**: Determined attackers can craft commands to evade classification\n\n");
    }

    pub(crate) fn card_training(&self, out: &mut String, base_model: Option<&str>) {
        out.push_str("## Training\n\n");
        out.push_str("| Parameter | Value |\n");
        out.push_str("|-----------|-------|\n");
        out.push_str("| Framework | [entrenar](https://github.com/paiml/entrenar) (Rust) |\n");
        out.push_str("| Method | LoRA (Low-Rank Adaptation) |\n");
        if let Some(base) = base_model {
            out.push_str(&format!("| Base model | `{base}` |\n"));
        }
        out.push_str(&format!("| Num classes | {} |\n\n", self.num_classes));
    }

    /// Average a metric vector using the given strategy.
    pub(crate) fn avg_metric(
        &self,
        values: &[f64],
        average: crate::eval::classification::Average,
    ) -> f64 {
        match average {
            crate::eval::classification::Average::Macro => {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            crate::eval::classification::Average::Weighted => {
                let total: usize = self.per_class_support.iter().sum();
                if total == 0 {
                    return 0.0;
                }
                values
                    .iter()
                    .zip(self.per_class_support.iter())
                    .map(|(&v, &s)| v * s as f64)
                    .sum::<f64>()
                    / total as f64
            }
            _ => {
                // Fallback to macro
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
        }
    }
}

/// SSC label names used across the shell safety classifier.
pub const SSC_LABELS: [&str; 5] =
    ["safe", "needs-quoting", "non-deterministic", "non-idempotent", "unsafe"];

/// Evaluate a saved checkpoint against a test JSONL dataset.
///
/// Standalone function that loads a checkpoint, builds a pipeline, and runs
/// evaluation without needing a full `ClassifyTrainer` setup.
///
/// Handles LoRA adapter checkpoints: reads `adapter_config.json` to find the
/// base model path, loads the full transformer from that path, then restores
/// trained LoRA + classifier head weights from the checkpoint's `model.safetensors`.
/// Restore class_weights from checkpoint metadata.json if present.
/// Training runs that use `auto_balance_classes()` save weights to metadata;
/// without this, evaluation would use uniform weights while training used weighted loss.
pub(crate) fn restore_class_weights_from_metadata(
    checkpoint_dir: &std::path::Path,
    num_classes: usize,
) -> Option<Vec<f32>> {
    let meta_str = std::fs::read_to_string(checkpoint_dir.join("metadata.json")).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&meta_str).ok()?;
    let arr = meta.get("class_weights")?.as_array()?;
    let weights: Vec<f32> = arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect();
    (weights.len() == num_classes).then_some(weights)
}

///
/// # Arguments
/// * `checkpoint_dir` - Directory containing `model.safetensors` + `adapter_config.json`
/// * `test_data` - JSONL file with `{"input": "...", "label": N}` entries
/// * `model_config` - Transformer architecture config (must match checkpoint)
/// * `classify_config` - Classification config (num_classes, etc.)
/// * `label_names` - Human-readable class names
///
/// # Errors
/// Returns error if checkpoint or test data cannot be loaded.
pub fn evaluate_checkpoint(
    checkpoint_dir: &Path,
    test_data: &Path,
    model_config: &crate::transformer::TransformerConfig,
    classify_config: super::classify_pipeline::ClassifyConfig,
    label_names: &[String],
) -> crate::Result<ClassifyEvalReport> {
    use super::classification::load_safety_corpus;

    let start = std::time::Instant::now();
    let num_classes = classify_config.num_classes;

    // Restore class_weights from checkpoint metadata if not provided by caller.
    let mut classify_config = classify_config;
    if classify_config.class_weights.is_none() {
        if let Some(weights) = restore_class_weights_from_metadata(checkpoint_dir, num_classes) {
            println!("Restored class_weights from checkpoint: {weights:?}");
            classify_config.class_weights = Some(weights);
        }
    }

    // Resolve the base model directory from adapter_config.json (LoRA checkpoint)
    // or fall back to loading directly from checkpoint_dir (full model checkpoint)
    let adapter_config_path = checkpoint_dir.join("adapter_config.json");
    let mut pipeline = if adapter_config_path.exists() {
        // LoRA adapter checkpoint: load base model, then restore adapter weights
        let adapter_json = std::fs::read_to_string(&adapter_config_path)
            .map_err(|e| crate::Error::Io(format!("Failed to read adapter_config.json: {e}")))?;
        let peft_config: crate::lora::PeftAdapterConfig = serde_json::from_str(&adapter_json)
            .map_err(|e| {
                crate::Error::Serialization(format!("Invalid adapter_config.json: {e}"))
            })?;

        // Update classify_config with LoRA rank/alpha from the checkpoint's
        // adapter_config.json so LoRA layers are created with matching dimensions.
        if peft_config.r > 0 {
            classify_config.lora_rank = peft_config.r;
        }
        if peft_config.lora_alpha > 0.0 {
            classify_config.lora_alpha = peft_config.lora_alpha;
        }

        // Try to load base model from pretrained weights.  Fall back to random
        // init when the path is missing, points to an .apr file (not a
        // SafeTensors directory), or otherwise fails to load.
        let mut pipe = match peft_config.base_model_name_or_path.as_deref() {
            Some(base_model_path)
                if std::path::Path::new(base_model_path).is_dir()
                    || std::path::Path::new(base_model_path)
                        .extension()
                        .is_some_and(|e| e == "safetensors") =>
            {
                println!("Loading base model from: {base_model_path}");
                ClassifyPipeline::from_pretrained(
                    base_model_path,
                    model_config,
                    classify_config.clone(),
                )?
            }
            Some(base_model_path) => {
                println!("Base model path is not a SafeTensors directory: {base_model_path}");
                println!("Using random-init base model (adapter weights will be restored from checkpoint)");
                ClassifyPipeline::new(model_config, classify_config.clone())
            }
            None => {
                println!("No base_model_name_or_path in adapter_config.json");
                println!("Using random-init base model (adapter weights will be restored from checkpoint)");
                ClassifyPipeline::new(model_config, classify_config.clone())
            }
        };

        // Load trained LoRA + classifier weights from checkpoint
        let st_path = checkpoint_dir.join("model.safetensors");
        let st_data = std::fs::read(&st_path).map_err(|e| {
            crate::Error::Io(format!("Failed to read checkpoint model.safetensors: {e}"))
        })?;
        let tensors = safetensors::SafeTensors::deserialize(&st_data).map_err(|e| {
            crate::Error::Serialization(format!("Failed to deserialize checkpoint: {e}"))
        })?;

        // Restore classifier head weights
        if let Ok(w) = tensors.tensor("classifier.weight") {
            let w_data: &[f32] = bytemuck::cast_slice(w.data());
            pipe.classifier
                .weight
                .data_mut()
                .as_slice_mut()
                .expect("contiguous classifier.weight")
                .copy_from_slice(w_data);
        }
        if let Ok(b) = tensors.tensor("classifier.bias") {
            let b_data: &[f32] = bytemuck::cast_slice(b.data());
            pipe.classifier
                .bias
                .data_mut()
                .as_slice_mut()
                .expect("contiguous classifier.bias")
                .copy_from_slice(b_data);
        }

        // Restore LoRA adapter weights (convention: 2 per layer, Q=even V=odd)
        for (idx, lora) in pipe.lora_layers.iter_mut().enumerate() {
            let layer = idx / 2;
            let proj = if idx % 2 == 0 { "q" } else { "v" };

            if let Ok(a) = tensors.tensor(&format!("lora.{layer}.{proj}_proj.lora_a")) {
                let a_data: &[f32] = bytemuck::cast_slice(a.data());
                lora.lora_a_mut()
                    .data_mut()
                    .as_slice_mut()
                    .expect("contiguous lora_a")
                    .copy_from_slice(a_data);
            }
            if let Ok(b) = tensors.tensor(&format!("lora.{layer}.{proj}_proj.lora_b")) {
                let b_data: &[f32] = bytemuck::cast_slice(b.data());
                lora.lora_b_mut()
                    .data_mut()
                    .as_slice_mut()
                    .expect("contiguous lora_b")
                    .copy_from_slice(b_data);
            }
        }

        let loaded_count = tensors.names().len();
        println!("Restored {loaded_count} tensors from checkpoint");
        pipe
    } else {
        // Full model checkpoint: load directly
        ClassifyPipeline::from_pretrained(checkpoint_dir, model_config, classify_config)?
    };

    // Load test corpus
    let samples = load_safety_corpus(test_data, num_classes)?;

    // Run forward-only on all samples, collecting full probability distributions
    let mut y_true: Vec<usize> = Vec::with_capacity(samples.len());
    let mut y_pred: Vec<usize> = Vec::with_capacity(samples.len());
    let mut all_probs: Vec<Vec<f32>> = Vec::with_capacity(samples.len());
    let mut total_loss = 0.0f32;

    for (i, sample) in samples.iter().enumerate() {
        let ids = pipeline.tokenize(&sample.input);
        let (loss, predicted, probs) = pipeline.forward_only_with_probs(&ids, sample.label);
        total_loss += loss;
        y_true.push(sample.label);
        y_pred.push(predicted);
        all_probs.push(probs);

        // Progress indicator every 100 samples
        if (i + 1) % 100 == 0 {
            println!("  Evaluated {}/{} samples...", i + 1, samples.len());
        }
    }
    println!("  Evaluated {}/{} samples (done)", samples.len(), samples.len());

    Ok(ClassifyEvalReport::from_predictions_with_probs(
        &y_pred,
        &y_true,
        &all_probs,
        total_loss,
        num_classes,
        label_names,
        start.elapsed().as_millis() as u64,
    ))
}
