//! Leaderboard for comparing multiple models

use super::metric::Metric;
use super::result::EvalResult;
use std::fmt;

/// Leaderboard for comparing multiple models
#[derive(Clone, Debug)]
pub struct Leaderboard {
    /// Evaluation results for each model
    pub results: Vec<EvalResult>,
    /// Primary metric for ranking
    pub primary_metric: Metric,
}

impl Leaderboard {
    /// Create a new leaderboard
    pub fn new(primary_metric: Metric) -> Self {
        Self { results: Vec::new(), primary_metric }
    }

    /// Add evaluation result
    pub fn add(&mut self, result: EvalResult) {
        self.results.push(result);
        self.sort();
    }

    /// Sort by primary metric
    ///
    /// N-06 (Meyer DbC): Models with missing scores sort last, not to an
    /// arbitrary position. `NEG_INFINITY` for "higher is better" and
    /// `INFINITY` for "lower is better" ensure worst-possible semantics.
    pub fn sort(&mut self) {
        let higher_is_better = self.primary_metric.higher_is_better();
        let missing = if higher_is_better { f64::NEG_INFINITY } else { f64::INFINITY };
        self.results.sort_by(|a, b| {
            let score_a = a.get_score(self.primary_metric).unwrap_or(missing);
            let score_b = b.get_score(self.primary_metric).unwrap_or(missing);
            if higher_is_better {
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
    }

    /// Sort by a specific metric
    pub fn sort_by(&mut self, metric: Metric) {
        let higher_is_better = metric.higher_is_better();
        let missing = if higher_is_better { f64::NEG_INFINITY } else { f64::INFINITY };
        self.results.sort_by(|a, b| {
            let score_a = a.get_score(metric).unwrap_or(missing);
            let score_b = b.get_score(metric).unwrap_or(missing);
            if higher_is_better {
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            }
        });
    }

    /// Get best model by primary metric
    pub fn best(&self) -> Option<&EvalResult> {
        self.results.first()
    }

    /// Print formatted leaderboard to stdout (Mieruka - visual control)
    pub fn print(&self) {
        println!("{self}");
    }

    /// Export as markdown table
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        // Collect all metrics
        let metrics: Vec<Metric> = if let Some(first) = self.results.first() {
            first.scores.keys().copied().collect()
        } else {
            return md;
        };

        // Header
        md.push_str("| Model |");
        for metric in &metrics {
            md.push_str(&format!(" {metric} |"));
        }
        md.push_str(" Inference (ms) |\n");

        // Separator
        md.push_str("|-------|");
        for _ in &metrics {
            md.push_str("----------|");
        }
        md.push_str("---------------|\n");

        // Rows
        for result in &self.results {
            md.push_str(&format!("| {} |", result.model_name));
            for metric in &metrics {
                match result.get_score(*metric) {
                    Some(score) => md.push_str(&format!(" {score:.4} |")),
                    // N-06: Missing scores display as "—" not "0.0000"
                    None => md.push_str(" — |"),
                }
            }
            md.push_str(&format!(" {:.2} |\n", result.inference_time_ms));
        }

        md
    }
}

impl fmt::Display for Leaderboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.results.is_empty() {
            return writeln!(f, "Leaderboard: (empty)");
        }

        // Collect all metrics
        let metrics: Vec<Metric> = if let Some(first) = self.results.first() {
            first.scores.keys().copied().collect()
        } else {
            return Ok(());
        };

        // Calculate column widths
        let model_width = self.results.iter().map(|r| r.model_name.len()).max().unwrap_or(5).max(5);

        // Header
        write!(f, "┌{:─<width$}┬", "", width = model_width + 2)?;
        for _ in &metrics {
            write!(f, "{:─<12}┬", "")?;
        }
        writeln!(f, "{:─<15}┐", "")?;

        write!(f, "│ {:width$} │", "Model", width = model_width)?;
        for metric in &metrics {
            write!(f, " {:>10} │", metric.name())?;
        }
        writeln!(f, " Inference (ms)│")?;

        // Separator
        write!(f, "├{:─<width$}┼", "", width = model_width + 2)?;
        for _ in &metrics {
            write!(f, "{:─<12}┼", "")?;
        }
        writeln!(f, "{:─<15}┤", "")?;

        // Rows
        for result in &self.results {
            write!(f, "│ {:width$} │", result.model_name, width = model_width)?;
            for metric in &metrics {
                match result.get_score(*metric) {
                    Some(score) => write!(f, " {score:>10.4} │")?,
                    // N-06: Missing scores display as "—" not "0.0000"
                    None => write!(f, " {:>10} │", "—")?,
                }
            }
            writeln!(f, " {:>13.2} │", result.inference_time_ms)?;
        }

        // Footer
        write!(f, "└{:─<width$}┴", "", width = model_width + 2)?;
        for _ in &metrics {
            write!(f, "{:─<12}┴", "")?;
        }
        writeln!(f, "{:─<15}┘", "")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(name: &str, metric: Metric, score: Option<f64>) -> EvalResult {
        let mut r = EvalResult::new(name);
        if let Some(s) = score {
            r.add_score(metric, s);
        }
        r
    }

    // =========================================================================
    // FALSIFY tests — contract violation sweep (N-06)
    // =========================================================================

    #[test]
    fn test_falsify_n06_missing_score_sorts_last_higher_is_better() {
        // N-06: A model with a missing score for a "higher is better" metric
        // must sort LAST, not to an arbitrary middle position.
        let metric = Metric::Accuracy;
        assert!(metric.higher_is_better());

        let mut lb = Leaderboard::new(metric);
        lb.results.push(make_result("good", metric, Some(0.9)));
        lb.results.push(make_result("missing", metric, None));
        lb.results.push(make_result("bad", metric, Some(0.1)));
        lb.sort();

        assert_eq!(lb.results[0].model_name, "good");
        assert_eq!(lb.results[1].model_name, "bad");
        assert_eq!(
            lb.results[2].model_name, "missing",
            "Model with missing score must sort last for higher-is-better metric"
        );
    }

    #[test]
    fn test_falsify_n06_missing_score_sorts_last_lower_is_better() {
        // N-06: A model with a missing score for a "lower is better" metric
        // must also sort LAST.
        let metric = Metric::MSE;
        assert!(!metric.higher_is_better());

        let mut lb = Leaderboard::new(metric);
        lb.results.push(make_result("good", metric, Some(0.01)));
        lb.results.push(make_result("missing", metric, None));
        lb.results.push(make_result("bad", metric, Some(10.0)));
        lb.sort();

        assert_eq!(lb.results[0].model_name, "good");
        assert_eq!(lb.results[1].model_name, "bad");
        assert_eq!(
            lb.results[2].model_name, "missing",
            "Model with missing score must sort last for lower-is-better metric"
        );
    }

    #[test]
    fn test_falsify_n06_sort_by_missing_score_sorts_last() {
        // N-06: sort_by() also uses correct missing-score semantics.
        let primary = Metric::Accuracy;
        let secondary = Metric::Perplexity; // lower is better
        assert!(!secondary.higher_is_better());

        let mut lb = Leaderboard::new(primary);

        let mut r1 = make_result("model_a", primary, Some(0.8));
        r1.add_score(secondary, 5.0);
        lb.results.push(r1);

        let r2 = make_result("model_b", primary, Some(0.9));
        // model_b has NO perplexity score
        lb.results.push(r2);

        let mut r3 = make_result("model_c", primary, Some(0.7));
        r3.add_score(secondary, 100.0);
        lb.results.push(r3);

        lb.sort_by(secondary);

        assert_eq!(lb.results[0].model_name, "model_a", "lowest perplexity first");
        assert_eq!(lb.results[1].model_name, "model_c");
        assert_eq!(lb.results[2].model_name, "model_b", "Missing perplexity score must sort last");
    }

    #[test]
    fn test_falsify_n06_display_missing_score_shows_dash() {
        // N-06: Missing scores must display as "—", never "0.0000".
        // A zero score is a valid measurement; a missing score is not.
        let metric = Metric::Accuracy;
        let mut lb = Leaderboard::new(metric);
        lb.results.push(make_result("has_score", metric, Some(0.95)));
        lb.results.push(make_result("no_score", metric, None));

        let md = lb.to_markdown();
        // Model with score should show numeric value
        assert!(md.contains("0.95"), "scored model must show numeric value in markdown");
        // Model without score must show dash, NOT "0.0000"
        assert!(md.contains('—'), "missing score must show '—' in markdown, got:\n{md}");
        assert!(
            !md.contains("0.0000") || md.contains("0.9500"),
            "markdown must not contain '0.0000' for missing scores"
        );

        // Also test Display trait
        let display = format!("{lb}");
        assert!(display.contains('—'), "missing score must show '—' in display output");
    }

    #[test]
    fn test_leaderboard_add_and_best() {
        let metric = Metric::Accuracy;
        let mut lb = Leaderboard::new(metric);

        lb.add(make_result("bad", metric, Some(0.5)));
        lb.add(make_result("best", metric, Some(0.99)));
        lb.add(make_result("mid", metric, Some(0.75)));

        let best = lb.best().expect("should have a best");
        assert_eq!(best.model_name, "best");
    }
}
