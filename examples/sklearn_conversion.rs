//! sklearn to Aprender Conversion Example (BATUTA-009)
//!
//! Demonstrates the conversion mapping from Python scikit-learn (sklearn)
//! algorithms to Rust Aprender equivalents with MoE backend selection.
//!
//! Run with: cargo run --example sklearn_conversion

use batuta::sklearn_converter::{SklearnAlgorithm, SklearnConverter};

fn main() {
    println!("🔄 sklearn → Aprender Conversion Demo (BATUTA-009)");
    println!("================================================\n");

    let converter = SklearnConverter::new();

    // Show conversion mapping
    println!("📋 Algorithm Conversion Mapping");
    println!("-------------------------------\n");

    let algorithms = vec![
        (SklearnAlgorithm::LinearRegression, "sklearn.linear_model.LinearRegression()"),
        (SklearnAlgorithm::LogisticRegression, "sklearn.linear_model.LogisticRegression()"),
        (SklearnAlgorithm::KMeans, "sklearn.cluster.KMeans(n_clusters=3)"),
        (SklearnAlgorithm::DecisionTreeClassifier, "sklearn.tree.DecisionTreeClassifier()"),
        (SklearnAlgorithm::StandardScaler, "sklearn.preprocessing.StandardScaler()"),
        (SklearnAlgorithm::TrainTestSplit, "sklearn.model_selection.train_test_split()"),
    ];

    for (alg, sklearn_code) in algorithms {
        if let Some(aprender_alg) = converter.convert(&alg) {
            println!("sklearn:  {}", sklearn_code);
            println!("Aprender: {}", aprender_alg.code_template);
            println!("          Complexity: {:?}", aprender_alg.complexity);
            println!();
        }
    }

    // Show backend recommendations
    println!("🎯 Backend Recommendations by Data Size");
    println!("----------------------------------------\n");

    let workloads = vec![
        (SklearnAlgorithm::StandardScaler, "Data preprocessing"),
        (SklearnAlgorithm::LinearRegression, "Linear regression"),
        (SklearnAlgorithm::KMeans, "K-Means clustering"),
    ];

    let sizes = vec![100, 10_000, 50_000, 100_000];

    for (alg, desc) in workloads {
        println!("{} ({:?} complexity):", desc, alg.complexity());
        for &size in &sizes {
            let backend = converter.recommend_backend(&alg, size);
            println!("  {:>8} samples → {}", format_size(size), backend);
        }
        println!();
    }

    // Show Python→Rust conversion example
    println!("💡 Practical Conversion Example");
    println!("-------------------------------\n");

    println!("Python sklearn:");
    println!("```python");
    println!("from sklearn.linear_model import LinearRegression");
    println!("from sklearn.model_selection import train_test_split");
    println!("from sklearn.metrics import mean_squared_error");
    println!();
    println!("# Split data");
    println!("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)");
    println!();
    println!("# Train model");
    println!("model = LinearRegression()");
    println!("model.fit(X_train, y_train)");
    println!();
    println!("# Predict and evaluate");
    println!("predictions = model.predict(X_test)");
    println!("mse = mean_squared_error(y_test, predictions)");
    println!("```\n");

    println!("Rust Aprender (converted):");
    println!("```rust");
    println!("use aprender::linear_model::LinearRegression;");
    println!("use aprender::model_selection::train_test_split;");
    println!("use aprender::metrics::mean_squared_error;");
    println!("use aprender::Estimator;");
    println!();
    println!("// Split data");
    println!("let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.25)?;");
    println!();
    println!("// Train model");
    println!("let mut model = LinearRegression::new();");
    println!("model.fit(&X_train, &y_train)?;");
    println!();
    println!("// Predict and evaluate");
    println!("let predictions = model.predict(&X_test)?;");
    println!("let mse = mean_squared_error(&y_test, &predictions)?;");
    println!("```\n");

    // Show performance analysis
    println!("⚡ Performance Analysis");
    println!("----------------------\n");

    println!("Backend selection via MoE routing:");
    println!(
        "  • 100 samples (preprocessing): {} (small dataset)",
        converter.recommend_backend(&SklearnAlgorithm::StandardScaler, 100)
    );
    println!(
        "  • 50K samples (linear regression): {}",
        converter.recommend_backend(&SklearnAlgorithm::LinearRegression, 50_000)
    );
    println!(
        "  • 100K samples (K-Means clustering): {} (high-compute)",
        converter.recommend_backend(&SklearnAlgorithm::KMeans, 100_000)
    );
    println!(
        "  • 200K samples (Random Forest): {} (tree ensemble)",
        converter.recommend_backend(&SklearnAlgorithm::RandomForestClassifier, 200_000)
    );

    println!("\n✨ Benefits:");
    println!("  • API preservation: sklearn-like ergonomic interface");
    println!("  • Performance: SIMD/GPU acceleration via Aprender");
    println!("  • Safety: Rust's memory safety and thread safety");
    println!("  • Adaptive: MoE routing selects optimal backend");
    println!("  • Type safety: Compile-time error checking");

    // Show common patterns
    println!("\n📚 Common Conversion Patterns");
    println!("-----------------------------\n");

    let patterns = vec![
        ("Supervised Learning", SklearnAlgorithm::LinearRegression),
        ("Unsupervised Learning", SklearnAlgorithm::KMeans),
        ("Data Preprocessing", SklearnAlgorithm::StandardScaler),
        ("Model Evaluation", SklearnAlgorithm::Accuracy),
    ];

    for (category, alg) in patterns {
        if let Some(aprender_alg) = converter.convert(&alg) {
            println!("{} ({:?}):", category, alg);
            println!("  sklearn:  {}.{:?}()", alg.sklearn_module(), alg);
            println!("  Aprender: {}", aprender_alg.code_template);
            println!("  Usage:\n    {}", aprender_alg.usage_pattern.replace('\n', "\n    "));
            println!();
        }
    }

    // Show module organization
    println!("🗂️  Aprender Module Organization");
    println!("--------------------------------\n");

    let modules = vec![
        ("linear_model", "Linear regression algorithms"),
        ("classification", "Logistic regression & classifiers"),
        ("cluster", "K-Means, DBSCAN clustering"),
        ("tree", "Decision trees & ensembles"),
        ("preprocessing", "Data transformers & scalers"),
        ("model_selection", "Cross-validation utilities"),
        ("metrics", "Model evaluation metrics"),
    ];

    for (module, description) in modules {
        println!("  aprender::{:<20} - {}", module, description);
    }

    println!("\n✅ Conversion Complete!");
    println!("   - 8 algorithm mappings demonstrated");
    println!("   - MoE backend selection enabled");
    println!("   - sklearn API ergonomics preserved");
}

fn format_size(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}
