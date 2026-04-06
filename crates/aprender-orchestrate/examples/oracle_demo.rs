/// Oracle Mode demonstration
/// Based on oracle-mode-spec.md - Intelligent query interface for Sovereign AI Stack
use batuta::oracle::{
    arxiv::{ArxivEnricher, ArxivSource},
    Backend, DataSize, HardwareSpec, KnowledgeGraph, OracleQuery, ProblemDomain, QueryConstraints,
    QueryEngine, Recommender, StackLayer,
};

fn main() {
    println!("🔮 Oracle Mode Demo");
    println!("Intelligent query interface for the Sovereign AI Stack\n");

    // Initialize the Oracle components
    let graph = KnowledgeGraph::sovereign_stack();
    let recommender = Recommender::new();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1. KNOWLEDGE GRAPH: Stack Component Registry");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // List all stack layers
    println!("📚 Sovereign AI Stack Layers:");
    for layer in StackLayer::all() {
        let components = graph.components_in_layer(layer);
        println!("  {}: {} components", layer, components.len());
        for comp in components {
            println!("    - {} v{}", comp.name, comp.version);
        }
    }
    println!();

    // Show capability search
    println!("🔍 Finding components with SIMD capability:");
    for comp in graph.find_by_capability("simd") {
        println!("  - {} ({})", comp.name, comp.layer);
    }
    println!();

    // Show domain search
    println!("🧠 Finding components for Supervised Learning domain:");
    for comp in graph.find_by_domain(ProblemDomain::SupervisedLearning) {
        println!("  - {}: {}", comp.name, comp.description);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2. NATURAL LANGUAGE QUERIES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Process natural language queries
    let queries = [
        "How do I run K-means clustering on a large dataset?",
        "What's the fastest way to do matrix multiplication?",
        "I need to transpile PyTorch to Rust with GPU support",
        "Help me optimize random forest training for 10GB of data",
    ];

    for query in &queries {
        println!("📝 Query: \"{}\"", query);
        let response = recommender.query(query);
        println!("   Problem: {}", response.problem_class);
        if let Some(algo) = &response.algorithm {
            println!("   Algorithm: {}", algo);
        }
        println!(
            "   Primary: {} ({:.0}% confidence)",
            response.primary.component,
            response.primary.confidence * 100.0
        );
        println!("   Backend: {:?}", response.compute.backend);
        println!();
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3. RECOMMENDER: Intelligent Component Selection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Backend selection scenarios
    println!("🎯 Backend Selection (Amdahl's Law + PCIe 5× Rule):\n");

    // Scenario 1: Small data, simple operation
    let query1 = OracleQuery::new("matrix multiply").with_data_size(DataSize::samples(64 * 64));
    let result1 = recommender.query_structured(&query1);
    println!("Scenario: Small matrix (64×64)");
    println!("  Backend: {:?}", result1.compute.backend);
    println!("  Rationale: {}\n", result1.compute.rationale);

    // Scenario 2: Large data, complex operation
    let constraints2 = QueryConstraints {
        data_size: Some(DataSize::samples(1_000_000)),
        hardware: HardwareSpec::with_gpu(16.0),
        ..Default::default()
    };
    let query2 =
        OracleQuery::new("neural network training large dataset").with_constraints(constraints2);
    let result2 = recommender.query_structured(&query2);
    println!("Scenario: Neural network training (1M samples, GPU available)");
    println!("  Backend: {:?}", result2.compute.backend);
    println!("  Rationale: {}\n", result2.compute.rationale);

    // Scenario 3: Very large distributed workload
    let constraints3 = QueryConstraints {
        data_size: Some(DataSize::samples(1_000_000_000)),
        hardware: HardwareSpec { is_distributed: true, node_count: Some(8), ..Default::default() },
        ..Default::default()
    };
    let query3 =
        OracleQuery::new("random forest distributed repartir").with_constraints(constraints3);
    let result3 = recommender.query_structured(&query3);
    println!("Scenario: Billion-scale random forest (8 nodes)");
    println!("  Backend: {:?}", result3.compute.backend);
    println!("  Distribution: {}", if result3.distribution.needed { "Yes" } else { "No" });
    println!("  Rationale: {}\n", result3.distribution.rationale);

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4. INTEGRATION PATTERNS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Show integration patterns
    println!("🔗 Integration patterns from trueno:");
    for pattern in graph.integrations_from("trueno") {
        println!("  trueno → {}: {}", pattern.to, pattern.description);
    }
    println!();

    println!("🔗 Components that integrate with aprender:");
    for pattern in graph.integrations_to("aprender") {
        println!("  {} → aprender: {}", pattern.from, pattern.description);
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("5. CODE GENERATION EXAMPLES");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Show code examples for different queries
    println!("💻 Code example for SIMD matrix operations:");
    let result_simd = recommender.query("matrix multiply simd");
    if let Some(code) = &result_simd.code_example {
        println!("{}\n", code);
    }

    println!("💻 Code example for distributed computing:");
    let dist_constraints = QueryConstraints {
        data_size: Some(DataSize::samples(100_000_000)),
        hardware: HardwareSpec { is_distributed: true, node_count: Some(4), ..Default::default() },
        ..Default::default()
    };
    let query_dist = OracleQuery::new("distributed computing cluster repartir")
        .with_constraints(dist_constraints);
    let result_dist = recommender.query_structured(&query_dist);
    if let Some(code) = &result_dist.code_example {
        println!("{}\n", code);
    }

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("6. BACKEND DECISION CRITERIA");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Backend selection follows Gregg & Hazelwood (2011):\n");
    println!("  {:12} | Criteria", "Backend");
    println!("  {:-<12}-+-{:-<50}", "", "");
    println!("  {:12} | Simple operations, small data", "Scalar");
    println!("  {:12} | SIMD-friendly (matmul, conv, reductions)", Backend::SIMD);
    println!("  {:12} | compute_time > 5× PCIe_transfer_time", Backend::GPU);
    println!("  {:12} | Data > single node capacity OR Amdahl P > 0.95", Backend::Distributed);
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("7. SYNTAX HIGHLIGHTING (syntect)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Demonstrate syntax highlighting with sample code
    println!("🎨 24-bit True Color Highlighting (base16-ocean.dark):\n");

    println!("   Oracle output uses syntect for rich syntax highlighting.");
    println!("   Try running: batuta oracle --recipe ml-random-forest\n");

    // Show the color legend using raw ANSI escapes
    println!("   Color Legend:");
    println!("     • Keywords (fn, let, use):     \x1b[38;2;180;142;173mpink\x1b[0m");
    println!("     • Comments (// ...):           \x1b[38;2;101;115;126mgray\x1b[0m");
    println!("     • Strings (\"...\"):             \x1b[38;2;163;190;140mgreen\x1b[0m");
    println!("     • Numbers (100, 10):           \x1b[38;2;208;135;112morange\x1b[0m");
    println!("     • Attributes (#[test]):        \x1b[38;2;191;97;106mred\x1b[0m");
    println!();

    println!("   Example highlighted output:");
    println!("   ─────────────────────────────────────────────────");
    // Manually show what highlighted code looks like (base16-ocean.dark colors)
    println!("   \x1b[38;2;180;142;173muse\x1b[0m aprender::tree::RandomForest;");
    println!();
    println!("   \x1b[38;2;101;115;126m// Train a random forest classifier\x1b[0m");
    println!("   \x1b[38;2;180;142;173mlet\x1b[0m model = RandomForest::new()");
    println!("       .n_estimators(\x1b[38;2;208;135;112m100\x1b[0m)");
    println!("       .max_depth(Some(\x1b[38;2;208;135;112m10\x1b[0m))");
    println!("       .fit(&x, &y)?;");
    println!("   ─────────────────────────────────────────────────");
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("8. TDD TEST COMPANIONS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Show TDD test companions from cookbook recipes
    let cookbook = batuta::oracle::cookbook::Cookbook::standard();
    let recipes = cookbook.recipes();
    let total = recipes.len();
    let with_tests = recipes.iter().filter(|r| !r.test_code.is_empty()).count();
    let with_cfg = recipes.iter().filter(|r| r.test_code.contains("#[cfg(")).count();

    println!("📋 Cookbook Test Companion Coverage:");
    println!("   Total recipes: {}", total);
    println!("   With test companions: {}", with_tests);
    println!("   With #[cfg(test)]: {}\n", with_cfg);

    // Show a specific recipe's test companion
    if let Some(recipe) = recipes.iter().find(|r| r.id == "ml-random-forest") {
        println!("📝 Recipe: {} ({})", recipe.title, recipe.id);
        println!("   Components: {}", recipe.components.join(", "));
        println!("\n   Code (first 3 lines):");
        for line in recipe.code.lines().take(3) {
            println!("   | {}", line);
        }
        println!("   | ...\n");
        println!("   Test Companion (first 5 lines):");
        for line in recipe.test_code.lines().take(5) {
            println!("   | {}", line);
        }
        println!("   | ...\n");
    }

    // Show recommender code examples also include test companions
    println!("💻 Recommender code examples with test companions:");
    let test_queries = ["matrix multiply simd", "train a model"];
    for query in &test_queries {
        let result = recommender.query(query);
        if let Some(code) = &result.code_example {
            let has_test = code.contains("#[cfg(test)]");
            println!(
                "   \"{}\" → {} (test companion: {})",
                query,
                result.primary.component,
                if has_test { "yes" } else { "no" }
            );
        }
    }
    println!();

    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("9. arXiv PAPER ENRICHMENT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create a QueryEngine to parse natural language and an ArxivEnricher
    let engine = QueryEngine::new();
    let enricher = ArxivEnricher::new();

    // Parse a speech-recognition query and enrich with the builtin curated DB
    let parsed = engine.parse("whisper speech recognition");
    let enrichment = enricher.enrich_builtin(&parsed, 3);

    println!("📄 Query: \"whisper speech recognition\"");
    println!("   Source: {:?}", enrichment.source);
    println!("   Search terms: {}\n", enrichment.query_terms.join(", "));

    if enrichment.papers.is_empty() {
        println!("   (no papers found)");
    } else {
        for (i, paper) in enrichment.papers.iter().enumerate() {
            println!("   {}. {}", i + 1, paper.title);
            println!("      Authors: {}", paper.authors);
            println!("      Year: {}", paper.year);
            println!("      URL: {}", paper.url);
            println!();
        }
    }

    // Show empty-result case with a nonsense query
    let parsed_empty = engine.parse("xyzzy zyxwv nonexistent gibberish");
    let empty_enrichment = enricher.enrich_builtin(&parsed_empty, 3);

    println!("📄 Query: \"xyzzy zyxwv nonexistent gibberish\"");
    println!("   Source: {:?}", empty_enrichment.source);
    assert_eq!(empty_enrichment.source, ArxivSource::Builtin);
    println!(
        "   Papers found: {} (nonsense queries yield no results)\n",
        empty_enrichment.papers.len()
    );

    println!("✅ Oracle Mode ready for production queries!");
    println!("   Run: batuta oracle --interactive");
}
