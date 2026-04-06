//! ML analysis generation and printing for tracer output
//!
//! Contains KMeans, Isolation Forest, and Autoencoder analysis functions.

/// Generate and populate ML analysis for JSON output
pub(super) fn generate_ml_analysis_for_json(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    ml_clusters: usize,
) -> Option<crate::ml_anomaly::MlAnomalyReport> {
    if let Some(ref tracker) = stats_tracker {
        let mut ml_data = std::collections::HashMap::new();
        for (syscall_name, stats) in tracker.stats_map() {
            let total_time_ns = stats.total_time_us * 1000;
            ml_data.insert(syscall_name.clone(), (stats.count, total_time_ns));
        }
        let analyzer = crate::ml_anomaly::MlAnomalyAnalyzer::new(ml_clusters);
        Some(analyzer.analyze(&ml_data))
    } else {
        None
    }
}

/// Generate Isolation Forest analysis for JSON output (Sprint 22)
pub(super) fn generate_isolation_forest_analysis_for_json(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    num_trees: usize,
    contamination: f32,
    explain: bool,
) -> Option<crate::isolation_forest::OutlierReport> {
    if let Some(ref tracker) = stats_tracker {
        let mut data = std::collections::HashMap::new();
        for (syscall_name, stats) in tracker.stats_map() {
            let total_time_ns = stats.total_time_us * 1000;
            data.insert(syscall_name.clone(), (stats.count, total_time_ns));
        }
        Some(crate::isolation_forest::analyze_outliers(&data, num_trees, contamination, explain))
    } else {
        None
    }
}

/// Generate Autoencoder analysis for JSON output (Sprint 23)
pub(super) fn generate_autoencoder_analysis_for_json(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    hidden_size: usize,
    epochs: usize,
    threshold: f64,
    explain: bool,
) -> Option<crate::autoencoder::AutoencoderReport> {
    if let Some(ref tracker) = stats_tracker {
        let mut data = std::collections::HashMap::new();
        for (syscall_name, stats) in tracker.stats_map() {
            let total_time_ns = stats.total_time_us * 1000;
            data.insert(syscall_name.clone(), (stats.count, total_time_ns));
        }
        Some(crate::autoencoder::analyze_anomalies(&data, hidden_size, epochs, threshold, explain))
    } else {
        None
    }
}

/// Print ML anomaly analysis report (Sprint 23)
pub(super) fn print_ml_analysis(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    ml_clusters: usize,
    ml_compare: bool,
    anomaly_threshold: f32,
) {
    if let Some(ref tracker) = stats_tracker {
        let mut ml_data = std::collections::HashMap::new();
        for (syscall_name, stats) in tracker.stats_map() {
            let total_time_ns = stats.total_time_us * 1000;
            ml_data.insert(syscall_name.clone(), (stats.count, total_time_ns));
        }
        let analyzer = crate::ml_anomaly::MlAnomalyAnalyzer::new(ml_clusters);
        let report = analyzer.analyze(&ml_data);

        if ml_compare {
            // Compare with z-score anomaly detection
            let mut zscore_anomalies = Vec::new();
            for syscall_name in tracker.stats_map().keys() {
                // Note: Pass None - this is for comparison, not primary compute
                #[cfg(feature = "otlp")]
                let extended = tracker.calculate_extended_statistics(syscall_name, None);
                #[cfg(not(feature = "otlp"))]
                let extended = tracker.calculate_extended_statistics(syscall_name, None);

                if let Some(extended) = extended {
                    if extended.stddev > 0.0 {
                        let z_score = (extended.max - extended.mean) / extended.stddev;
                        if z_score > anomaly_threshold {
                            zscore_anomalies.push((syscall_name.clone(), f64::from(z_score)));
                        }
                    }
                }
            }
            eprint!("{}", report.format_comparison(&zscore_anomalies));
        } else {
            eprint!("{}", report.format());
        }
    }
}

/// Print Isolation Forest outlier analysis report (Sprint 22)
pub(super) fn print_isolation_forest_analysis(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    num_trees: usize,
    contamination: f32,
    explain: bool,
) {
    if let Some(ref tracker) = stats_tracker {
        let mut data = std::collections::HashMap::new();
        for (syscall_name, stats) in tracker.stats_map() {
            let total_time_ns = stats.total_time_us * 1000;
            data.insert(syscall_name.clone(), (stats.count, total_time_ns));
        }

        let report =
            crate::isolation_forest::analyze_outliers(&data, num_trees, contamination, explain);

        // Print report
        eprintln!("\n=== Isolation Forest Anomaly Detection ===");
        eprintln!(
            "Trees: {}, Contamination: {:.1}%, Samples: {}\n",
            report.num_trees,
            report.contamination * 100.0,
            report.total_samples
        );

        if report.outliers.is_empty() {
            eprintln!("No outliers detected.");
        } else {
            eprintln!("Detected {} outlier(s):\n", report.outliers.len());
            for outlier in &report.outliers {
                eprintln!("  {} (anomaly score: {:.3})", outlier.syscall, outlier.anomaly_score);
                eprintln!(
                    "    Avg duration: {:.2} \u{03bc}s, Calls: {}",
                    outlier.avg_duration_us, outlier.call_count
                );

                if explain && !outlier.feature_importance.is_empty() {
                    eprintln!("    Feature Importance:");
                    for (feature, importance) in &outlier.feature_importance {
                        eprintln!("      {feature}: {importance:.1}%");
                    }
                }
                eprintln!();
            }
        }
        eprintln!("=========================================\n");
    }
}

/// Print Autoencoder anomaly detection report (Sprint 23)
pub(super) fn print_autoencoder_analysis(
    stats_tracker: &Option<crate::stats::StatsTracker>,
    hidden_size: usize,
    epochs: usize,
    threshold: f32,
    explain: bool,
) {
    if let Some(ref tracker) = stats_tracker {
        let mut data = std::collections::HashMap::new();
        for (syscall_name, stats) in tracker.stats_map() {
            let total_time_ns = stats.total_time_us * 1000;
            data.insert(syscall_name.clone(), (stats.count, total_time_ns));
        }

        let report = crate::autoencoder::analyze_anomalies(
            &data,
            hidden_size,
            epochs,
            f64::from(threshold),
            explain,
        );

        // Print report
        eprintln!("\n=== Autoencoder Anomaly Detection ===");
        eprintln!(
            "Hidden Size: {}, Epochs: {}, Threshold: {:.2}\u{03c3}",
            report.hidden_size, report.epochs, threshold
        );
        eprintln!(
            "Samples: {}, Adaptive Threshold: {:.4}\n",
            report.total_samples, report.threshold
        );

        if report.anomalies.is_empty() {
            eprintln!("No anomalies detected.");
        } else {
            eprintln!("Detected {} anomal(y/ies):\n", report.anomalies.len());
            for anomaly in &report.anomalies {
                eprintln!(
                    "  {} (reconstruction error: {:.4})",
                    anomaly.syscall, anomaly.reconstruction_error
                );
                eprintln!(
                    "    Avg duration: {:.2} \u{03bc}s, Calls: {}",
                    anomaly.avg_duration_us, anomaly.call_count
                );

                if explain && !anomaly.feature_contributions.is_empty() {
                    eprintln!("    Feature Contributions to Error:");
                    for (feature, contribution) in &anomaly.feature_contributions {
                        eprintln!("      {feature}: {contribution:.1}%");
                    }
                }
                eprintln!();
            }
        }
        eprintln!("======================================\n");
    }
}
