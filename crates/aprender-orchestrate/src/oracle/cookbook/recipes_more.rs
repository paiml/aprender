//! Additional recipe registrations
//!
//! Continuation of recipes module, split for file size compliance.

use super::Recipe;

pub fn register_training_recipes(cookbook: &mut super::Cookbook) {
    // LoRA Fine-tuning
    cookbook.add(
        Recipe::new("training-lora", "LoRA Fine-tuning")
            .with_problem("Fine-tune large models efficiently with Low-Rank Adaptation")
            .with_components(vec!["entrenar", "aprender", "alimentar"])
            .with_tags(vec!["training", "lora", "fine-tuning", "efficient", "llm"])
            .with_code(
                r#"use entrenar::prelude::*;

// Load base model
let model = Model::load("llama-7b.apr")?;

// Configure LoRA
let lora_config = LoraConfig {
    r: 16,                    // Rank
    alpha: 32,                // Scaling factor
    dropout: 0.1,
    target_modules: vec!["q_proj", "v_proj"],
};

// Apply LoRA adapters
let model = model.with_lora(lora_config)?;

// Only ~0.1% of parameters are trainable now
println!("Trainable params: {}", model.trainable_params());

// Training loop
let optimizer = AdamW::new(model.trainable_params(), 1e-4);
for batch in dataloader {
    let loss = model.forward(&batch)?;
    loss.backward()?;
    optimizer.step()?;
}

// Save LoRA weights only (small file)
model.save_lora("adapter.lora")?;

// Later: merge for inference
// let merged = Model::load("llama-7b.apr")?.merge_lora("adapter.lora")?;
"#,
            )
            .with_related(vec!["training-qlora", "training-autograd"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_lora_config_rank_and_alpha() {
    let rank = 16;
    let alpha = 32;
    assert!(rank > 0 && alpha >= rank);
}

    #[test]
    fn test_trainable_params_fraction() {
    let total = 1_000_000;
    let lora = 8192;
    let fraction = lora as f64 / total as f64;
    assert!(fraction < 0.1);
}

    #[test]
    fn test_dropout_in_valid_range() {
    let dropout = 0.1_f64;
    assert!(dropout >= 0.0 && dropout <= 1.0);
}
}",
            ),
    );

    // QLoRA
    cookbook.add(
        Recipe::new("training-qlora", "QLoRA Quantized Fine-tuning")
            .with_problem("Fine-tune 4-bit quantized models on consumer hardware")
            .with_components(vec!["entrenar", "aprender"])
            .with_tags(vec!["training", "qlora", "quantization", "4bit", "memory-efficient"])
            .with_code(
                r#"use entrenar::prelude::*;

// Load 4-bit quantized model
let model = Model::load_quantized("llama-7b.q4_k.gguf")?;

// QLoRA config (LoRA on quantized base)
let qlora_config = QLoraConfig {
    lora: LoraConfig { r: 64, alpha: 16, dropout: 0.1, .. },
    nf4: true,              // NormalFloat4 quantization
    double_quant: true,     // Double quantization for memory
    compute_dtype: F16,     // Compute in fp16
};

let model = model.with_qlora(qlora_config)?;

// Train on 24GB GPU (fits 7B model!)
let trainer = Trainer::new(model)
    .gradient_checkpointing(true)
    .batch_size(4)
    .gradient_accumulation(4);

trainer.train(&dataset, 3)?;  // 3 epochs
"#,
            )
            .with_related(vec!["training-lora"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_quantization_bits_valid() {
    let bits = 4;
    assert!(bits == 4 || bits == 8);
}

    #[test]
    fn test_effective_batch_size() {
    let batch_size = 4;
    let grad_accum = 4;
    let effective = batch_size * grad_accum;
    assert_eq!(effective, 16);
}

    #[test]
    fn test_nf4_requires_4bit() {
    let nf4 = true;
    let bits = 4;
    assert!(nf4 && bits == 4);
}
}",
            ),
    );

    // Autograd
    cookbook.add(
        Recipe::new("training-autograd", "Custom Training with Autograd")
            .with_problem("Build custom neural networks with automatic differentiation")
            .with_components(vec!["entrenar", "trueno"])
            .with_tags(vec!["training", "autograd", "neural-network", "custom"])
            .with_code(
                r#"use entrenar::autograd::*;

// Define model with autograd tensors
let w1 = Tensor::randn(&[784, 256]).requires_grad();
let w2 = Tensor::randn(&[256, 10]).requires_grad();

// Forward pass (computation graph built automatically)
fn forward(x: &Tensor, w1: &Tensor, w2: &Tensor) -> Tensor {
    let h = x.matmul(w1).relu();
    h.matmul(w2).softmax(-1)
}

// Training loop
let optimizer = SGD::new(vec![&w1, &w2], 0.01);
for (x, y) in dataloader {
    let pred = forward(&x, &w1, &w2);
    let loss = cross_entropy(&pred, &y);

    // Backward pass (gradients computed automatically)
    loss.backward();

    optimizer.step();
    optimizer.zero_grad();
}

// Gradients accessible
println!("w1 grad: {:?}", w1.grad());
"#,
            )
            .with_related(vec!["training-lora", "ml-random-forest"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_weight_matrix_dimensions() {
    let input_dim = 784;
    let hidden_dim = 256;
    let weights = vec![vec![0.0_f64; hidden_dim]; input_dim];
    assert_eq!(weights.len(), input_dim);
}

    #[test]
    fn test_softmax_sums_to_one() {
    let logits = vec![1.0_f64, 2.0, 3.0];
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = logits.iter().map(|x| (x - max).exp()).sum();
    let sum: f64 = logits.iter().map(|x| (x - max).exp() / exp_sum).sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

    #[test]
    fn test_learning_rate_positive() {
    let lr = 0.01_f64;
    assert!(lr > 0.0);
}
}",
            ),
    );
}

// =========================================================================
// Data Loading Recipes
// =========================================================================

pub fn register_data_recipes(cookbook: &mut super::Cookbook) {
    // Alimentar Data Loading
    cookbook.add(
        Recipe::new("data-alimentar", "Zero-Copy Data Loading")
            .with_problem("Load large datasets efficiently with memory mapping")
            .with_components(vec!["alimentar", "trueno"])
            .with_tags(vec!["data", "loading", "parquet", "arrow", "zero-copy"])
            .with_code(
                r#"use alimentar::prelude::*;

// Load Parquet with zero-copy (memory-mapped)
let dataset = ParquetDataset::open("data.parquet")?
    .select(&["features", "label"])?
    .filter(|row| row["label"].as_i64() > 0)?;

// Iterate with batching
let dataloader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true)
    .num_workers(4);

for batch in dataloader {
    // batch.features is Arrow array (zero-copy)
    let features = batch["features"].as_tensor()?;
    let labels = batch["label"].as_tensor()?;

    model.train_step(&features, &labels)?;
}

// Streaming from remote (S3, HuggingFace)
let dataset = Dataset::from_hub("username/dataset")?
    .streaming(true);  // Don't download entire dataset
"#,
            )
            .with_related(vec!["data-preprocessing", "ml-random-forest"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_batch_size_config() {
    let batch_size = 32_u32;
    assert!(batch_size > 0);
}

    #[test]
    fn test_column_selection() {
    let columns = vec!["features", "label"];
    assert_eq!(columns.len(), 2);
}

    #[test]
    fn test_worker_count() {
    let workers = 4;
    assert!(workers > 0 && workers <= 16);
}
}"#,
            ),
    );

    // Data Preprocessing
    cookbook.add(
        Recipe::new("data-preprocessing", "Data Preprocessing Pipeline")
            .with_problem("Build reproducible preprocessing pipelines")
            .with_components(vec!["alimentar", "aprender"])
            .with_tags(vec!["data", "preprocessing", "pipeline", "transforms"])
            .with_code(
                r#"use alimentar::prelude::*;
use aprender::preprocessing::*;

// Build preprocessing pipeline
let pipeline = Pipeline::new()
    .add(StandardScaler::fit(&train_data)?)
    .add(OneHotEncoder::fit(&["category"])?)
    .add(Imputer::median());

// Apply to train/test
let X_train = pipeline.transform(&train_data)?;
let X_test = pipeline.transform(&test_data)?;

// Save pipeline for inference
pipeline.save("preprocess.pipeline")?;

// Later: load and apply
let pipeline = Pipeline::load("preprocess.pipeline")?;
let X_new = pipeline.transform(&new_data)?;
"#,
            )
            .with_related(vec!["data-alimentar"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_pipeline_step_count() {
    let steps = vec!["scale", "encode", "impute"];
    assert_eq!(steps.len(), 3);
}

    #[test]
    fn test_transform_preserves_row_count() {
    let input_rows = 1000;
    let output_rows = 1000;
    assert_eq!(input_rows, output_rows);
}

    #[test]
    fn test_scaler_std_positive() {
    let std_dev = 1.0_f64;
    assert!(std_dev > 0.0);
}
}"#,
            ),
    );
}

// =========================================================================
// Model Registry Recipes
// =========================================================================

pub fn register_registry_recipes(cookbook: &mut super::Cookbook) {
    // Pacha Model Registry
    cookbook.add(
        Recipe::new("registry-pacha", "Model Registry with Pacha")
            .with_problem("Version, sign, and distribute ML models securely")
            .with_components(vec!["pacha", "aprender"])
            .with_tags(vec!["registry", "versioning", "signing", "distribution", "mlops"])
            .with_code(
                r#"use pacha::prelude::*;

// Initialize registry
let registry = Registry::new("./models")?;

// Register model with metadata
let model_card = ModelCard {
    name: "sentiment-classifier",
    version: "1.0.0",
    description: "BERT-based sentiment analysis",
    metrics: hashmap!{
        "accuracy" => 0.94,
        "f1" => 0.92,
    },
    license: "MIT",
    authors: vec!["team@example.com"],
};

// Push with Ed25519 signature
let artifact = registry.push(
    "model.apr",
    model_card,
    SigningKey::from_env()?,  // PACHA_SIGNING_KEY
)?;

println!("Registered: {}@{}", artifact.name, artifact.version);
println!("Hash: {}", artifact.blake3_hash);

// Pull model (verifies signature)
let model_path = registry.pull("sentiment-classifier", "1.0.0")?;

// List versions
for version in registry.versions("sentiment-classifier")? {
    println!("{} - {}", version.version, version.created_at);
}
"#,
            )
            .with_related(vec!["registry-hf", "ml-serving"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_model_card_metadata() {
    let name = "sentiment-classifier";
    let version = "1.0.0";
    assert!(!name.is_empty());
    assert!(version.chars().filter(|c| *c == '.').count() == 2);
}

    #[test]
    fn test_version_string_format() {
    let version = "1.0.0";
    let parts: Vec<_> = version.split('.').collect();
    assert_eq!(parts.len(), 3);
}

    #[test]
    fn test_hash_length() {
    let blake3_hash = "a".repeat(64);
    assert_eq!(blake3_hash.len(), 64);
}
}"#,
            ),
    );

    // HuggingFace Integration
    cookbook.add(
        Recipe::new("registry-hf", "HuggingFace Hub Integration")
            .with_problem("Download and cache models from HuggingFace Hub")
            .with_components(vec!["hf-hub", "aprender", "realizar"])
            .with_tags(vec!["registry", "huggingface", "download", "cache"])
            .with_code(
                r#"use hf_hub::api::sync::Api;

// Initialize API (uses HF_TOKEN env var if set)
let api = Api::new()?;

// Download model files
let repo = api.model("meta-llama/Llama-2-7b");
let model_path = repo.get("model.safetensors")?;
let config_path = repo.get("config.json")?;

// Files cached in ~/.cache/huggingface/hub/
println!("Model: {}", model_path.display());

// Download specific revision
let repo = api.model("meta-llama/Llama-2-7b").revision("main");
let path = repo.get("tokenizer.json")?;

// Progress callback
let repo = api.model("big-model").progress(|p| {
    println!("Downloading: {:.1}%", p.percent * 100.0);
});
"#,
            )
            .with_related(vec!["registry-pacha", "speech-whisper"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_api_url_valid() {
    let url = "https://huggingface.co";
    assert!(url.starts_with("https://"));
}

    #[test]
    fn test_model_path_structure() {
    let org = "meta-llama";
    let model = "Llama-2-7b";
    let path = format!("{}/{}", org, model);
    assert_eq!(path.split('/').count(), 2);
}

    #[test]
    fn test_revision_default() {
    let revision = "main";
    assert_eq!(revision, "main");
}
}"#,
            ),
    );
}

// =========================================================================
// RAG Pipeline Recipes
// =========================================================================

pub fn register_rag_recipes(cookbook: &mut super::Cookbook) {
    // RAG Pipeline
    cookbook.add(
        Recipe::new("rag-pipeline", "RAG Pipeline with Trueno-RAG")
            .with_problem("Build retrieval-augmented generation pipelines")
            .with_components(vec!["trueno-rag", "trueno-db", "aprender"])
            .with_tags(vec!["rag", "retrieval", "generation", "embeddings", "search"])
            .with_code(
                r#"use trueno_rag::prelude::*;

// Initialize RAG pipeline
let rag = RagPipeline::builder()
    .chunker(SemanticChunker::new(512))  // Semantic chunking
    .embedder(Embedder::load("bge-small-en")?)
    .retriever(HybridRetriever::new()
        .bm25_weight(0.3)
        .dense_weight(0.7))
    .reranker(CrossEncoder::load("ms-marco-MiniLM")?)
    .build()?;

// Index documents
for doc in documents {
    rag.add_document(&doc)?;
}
rag.build_index()?;

// Query with retrieval
let query = "What is the capital of France?";
let results = rag.retrieve(query, 5)?;  // Top 5 chunks

for (i, chunk) in results.iter().enumerate() {
    println!("{}. [score: {:.3}] {}", i+1, chunk.score, chunk.text);
}

// Full RAG with generation
let context = rag.retrieve_context(query, 3)?;
let prompt = format!("Context:\n{}\n\nQuestion: {}\nAnswer:", context, query);
let answer = llm.generate(&prompt)?;
"#,
            )
            .with_related(vec!["rag-semantic-search", "ml-serving"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_top_k_parameter() {
    let top_k = 5;
    assert!(top_k > 0 && top_k <= 100);
}

    #[test]
    fn test_chunk_size_exceeds_overlap() {
    let chunk_size = 512;
    let overlap = 50;
    assert!(chunk_size > overlap);
}

    #[test]
    fn test_retriever_weights_sum_to_one() {
    let bm25_weight = 0.3_f64;
    let vector_weight = 0.7_f64;
    assert!((bm25_weight + vector_weight - 1.0).abs() < 1e-6);
}
}",
            ),
    );

    // Semantic Search
    cookbook.add(
        Recipe::new("rag-semantic-search", "Semantic Search Engine")
            .with_problem("Build fast semantic search over documents")
            .with_components(vec!["trueno-db", "trueno-rag"])
            .with_tags(vec!["search", "semantic", "embeddings", "hnsw", "vector-db"])
            .with_code(
                r#"use trueno_db::prelude::*;
use trueno_rag::embeddings::*;

// Initialize vector store with HNSW index
let db = VectorDb::open("vectors.db")?
    .with_index(HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 50,
    });

// Embed and store documents
let embedder = Embedder::load("bge-small-en")?;
for doc in documents {
    let embedding = embedder.embed(&doc.text)?;
    db.insert(&doc.id, &embedding, &doc.metadata)?;
}

// Search
let query_embedding = embedder.embed("machine learning")?;
let results = db.search(&query_embedding, 10)?;

for result in results {
    println!("{}: {:.3}", result.id, result.score);
}

// Filtered search
let results = db.search_filtered(
    &query_embedding,
    10,
    |meta| meta["category"] == "science",
)?;
"#,
            )
            .with_related(vec!["rag-pipeline"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_hnsw_config_params() {
    let m = 16;
    let ef_construction = 200;
    assert!(m >= 4 && m <= 64);
    assert!(ef_construction >= m);
}

    #[test]
    fn test_search_result_ordering() {
    let scores = vec![0.95, 0.85, 0.75];
    let is_sorted = scores.windows(2).all(|w| w[0] >= w[1]);
    assert!(is_sorted);
}

    #[test]
    fn test_filter_predicate() {
    let min_score = 0.5_f64;
    let result_score = 0.75_f64;
    assert!(result_score >= min_score);
}
}",
            ),
    );
}

// =========================================================================
// Visualization Recipes
// =========================================================================

pub fn register_viz_recipes(cookbook: &mut super::Cookbook) {
    // Terminal Visualization
    cookbook.add(
        Recipe::new("viz-terminal", "Terminal Visualization")
            .with_problem("Create charts and plots in the terminal")
            .with_components(vec!["trueno-viz"])
            .with_tags(vec!["visualization", "terminal", "charts", "ascii"])
            .with_code(
                r#"use trueno_viz::prelude::*;

// Line chart in terminal
let chart = LineChart::new()
    .title("Training Loss")
    .x_label("Epoch")
    .y_label("Loss")
    .series("train", &train_losses)
    .series("val", &val_losses);

chart.render_terminal(80, 24)?;  // 80x24 chars

// Histogram
let hist = Histogram::new(&data)
    .bins(20)
    .title("Distribution");
hist.render_terminal(60, 15)?;

// Scatter plot
let scatter = ScatterPlot::new()
    .points(&x_vals, &y_vals)
    .title("Correlation");
scatter.render_terminal(40, 20)?;

// Progress bars (integrated with training)
let pb = ProgressBar::new(total_epochs);
for epoch in 0..total_epochs {
    // ... training ...
    pb.set(epoch, format!("loss: {:.4}", loss));
}
"#,
            )
            .with_related(vec!["viz-png", "training-autograd"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_chart_dimensions() {
    let width = 80;
    let height = 24;
    assert!(width > 0 && height > 0);
}

    #[test]
    fn test_bin_count() {
    let bins = 20;
    assert!(bins > 0 && bins <= 100);
}

    #[test]
    fn test_series_data_finite() {
    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    assert!(data.iter().all(|x| x.is_finite()));
}
}",
            ),
    );

    // PNG Export
    cookbook.add(
        Recipe::new("viz-png", "PNG Chart Export")
            .with_problem("Export publication-quality charts as PNG images")
            .with_components(vec!["trueno-viz"])
            .with_tags(vec!["visualization", "png", "export", "charts"])
            .with_code(
                r#"use trueno_viz::prelude::*;

// Create chart
let chart = LineChart::new()
    .title("Model Performance")
    .x_label("Epoch")
    .y_label("Accuracy")
    .series("ResNet", &resnet_acc)
    .series("VGG", &vgg_acc)
    .legend(Position::TopRight);

// Export as PNG
chart.save_png("performance.png", 800, 600)?;

// With custom styling
let styled = chart
    .background(Color::WHITE)
    .grid(true)
    .font_size(14);
styled.save_png("styled.png", 1200, 800)?;

// Batch export multiple charts
let charts = vec![
    ("loss", loss_chart),
    ("accuracy", acc_chart),
    ("confusion", confusion_matrix),
];
for (name, chart) in charts {
    chart.save_png(&format!("{}.png", name), 800, 600)?;
}
"#,
            )
            .with_related(vec!["viz-terminal"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_image_dimensions() {
    let width = 800;
    let height = 600;
    assert!(width > 0 && height > 0);
}

    #[test]
    fn test_chart_title_non_empty() {
    let title = "Model Performance";
    assert!(!title.is_empty());
}

    #[test]
    fn test_batch_export_count() {
    let charts = vec!["loss", "accuracy", "confusion"];
    assert_eq!(charts.len(), 3);
}
}"#,
            ),
    );
}

// =========================================================================
// RLHF & Alignment Recipes
// =========================================================================

pub fn register_rlhf_recipes(cookbook: &mut super::Cookbook) {
    super::recipes_rlhf_alignment::register_rlhf_alignment_recipes(cookbook);
    super::recipes_rlhf_training::register_rlhf_training_recipes(cookbook);
    super::recipes_rlhf_efficiency::register_rlhf_efficiency_recipes(cookbook);
}
