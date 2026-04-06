//! Entrenar LLM fine-tuning integration
//!
//! Exports verified transpilation tuples for LoRA fine-tuning with entrenar.
//! See VERIFICAR-090.
//!
//! # Knowledge Distillation
//!
//! From spec Section 5.4: Multi-teacher distillation via temperature-scaled
//! KL divergence (Hinton et al. 2015).
//!
//! ```text
//! L_distill = α * KL(softmax(z_s/T) || softmax(z_t/T)) + (1-α) * L_CE
//! ```

use crate::data::VerifiedTuple;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Training example for code-to-code translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeTranslationExample {
    /// Unique identifier
    pub id: String,
    /// Source language
    pub source_language: String,
    /// Target language
    pub target_language: String,
    /// Source code
    pub source_code: String,
    /// Target code (correct translation)
    pub target_code: String,
    /// Prompt for LLM (formatted input)
    pub prompt: String,
    /// Completion for LLM (expected output)
    pub completion: String,
}

/// Prompt template for code translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Template name
    pub name: String,
    /// System prompt (optional)
    pub system: Option<String>,
    /// User prompt template with placeholders: {source_lang}, {target_lang}, {source_code}
    pub user_template: String,
    /// Whether to include language tags
    pub include_lang_tags: bool,
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self::instruction_following()
    }
}

impl PromptTemplate {
    /// Instruction-following style prompt (Alpaca/Vicuna format)
    #[must_use]
    pub fn instruction_following() -> Self {
        Self {
            name: "instruction".to_string(),
            system: Some("You are an expert code translator.".to_string()),
            user_template: "Translate the following {source_lang} code to {target_lang}:\n\n```{source_lang}\n{source_code}\n```".to_string(),
            include_lang_tags: true,
        }
    }

    /// Chat-style prompt (ChatML format)
    #[must_use]
    pub fn chat_style() -> Self {
        Self {
            name: "chat".to_string(),
            system: Some("You are a helpful assistant that translates code between programming languages.".to_string()),
            user_template: "Please convert this {source_lang} code to idiomatic {target_lang}:\n\n{source_code}".to_string(),
            include_lang_tags: false,
        }
    }

    /// Completion-style prompt (minimal, for base models)
    #[must_use]
    pub fn completion_style() -> Self {
        Self {
            name: "completion".to_string(),
            system: None,
            user_template: "# {source_lang}\n{source_code}\n\n# {target_lang}\n".to_string(),
            include_lang_tags: false,
        }
    }

    /// Format a prompt using this template
    #[must_use]
    pub fn format(&self, source_lang: &str, target_lang: &str, source_code: &str) -> String {
        self.user_template
            .replace("{source_lang}", source_lang)
            .replace("{target_lang}", target_lang)
            .replace("{source_code}", source_code)
    }

    /// Format completion (target code with optional language tag)
    #[must_use]
    pub fn format_completion(&self, target_lang: &str, target_code: &str) -> String {
        if self.include_lang_tags {
            format!("```{target_lang}\n{target_code}\n```")
        } else {
            target_code.to_string()
        }
    }
}

/// Export configuration for entrenar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Output format: "json", "jsonl", "parquet"
    pub format: ExportFormat,
    /// Prompt template to use
    pub template: PromptTemplate,
    /// Train/val split ratio (0.0 to 1.0)
    pub train_ratio: f64,
    /// Random seed for splitting
    pub seed: u64,
    /// Maximum examples to export (None for all)
    pub max_examples: Option<usize>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Jsonl,
            template: PromptTemplate::default(),
            train_ratio: 0.9,
            seed: 42,
            max_examples: None,
        }
    }
}

/// Export format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// Single JSON array
    Json,
    /// JSON Lines (one object per line)
    Jsonl,
    /// Apache Parquet
    Parquet,
}

/// Export statistics
#[derive(Debug, Clone, Default)]
pub struct ExportStats {
    /// Total examples exported
    pub total: usize,
    /// Training examples
    pub train_count: usize,
    /// Validation examples
    pub val_count: usize,
    /// Average source code length
    pub avg_source_len: f64,
    /// Average target code length
    pub avg_target_len: f64,
}

// ============================================================================
// Knowledge Distillation Configuration (Spec Section 5.4)
// ============================================================================

/// Configuration for knowledge distillation training
///
/// Implements multi-teacher distillation via temperature-scaled KL divergence
/// (Hinton et al. 2015). The loss function is:
///
/// ```text
/// L_distill = α * KL(softmax(z_s/T) || softmax(z_t/T)) + (1-α) * L_CE
/// ```
///
/// Where:
/// - `T` is the temperature (higher = softer distributions)
/// - `α` is the balance between distillation and cross-entropy loss
/// - `z_s` and `z_t` are student and teacher logits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softmax (higher = softer probabilities)
    /// Typical values: 1.0-10.0. Default: 3.0
    pub temperature: f32,

    /// Balance between distillation loss and CE loss
    /// α=1.0 means pure distillation, α=0.0 means pure CE
    /// Typical values: 0.5-0.9. Default: 0.7
    pub alpha: f32,

    /// Number of teacher models for ensemble distillation
    pub num_teachers: usize,

    /// Student model configuration
    pub student: StudentConfig,

    /// Training hyperparameters
    pub training: DistillTrainingConfig,

    /// Output directory for distilled model
    pub output_dir: std::path::PathBuf,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 3.0,
            alpha: 0.7,
            num_teachers: 1,
            student: StudentConfig::default(),
            training: DistillTrainingConfig::default(),
            output_dir: std::path::PathBuf::from("distilled_model"),
        }
    }
}

/// Student model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentConfig {
    /// Model type identifier
    pub model_type: String,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of output classes (for classification)
    pub num_classes: usize,
}

impl Default for StudentConfig {
    fn default() -> Self {
        Self {
            model_type: "distilled_student".to_string(),
            hidden_size: 256,
            num_layers: 4,
            num_classes: 18, // 18 defect categories from org-intel
        }
    }
}

/// Training configuration for distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillTrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Gradient clipping norm
    pub grad_clip: f32,
    /// Whether to use mixed precision training
    pub fp16: bool,
}

impl Default for DistillTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 32,
            learning_rate: 1e-4,
            grad_clip: 1.0,
            fp16: false,
        }
    }
}

/// Result from distillation training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationResult {
    /// Final distillation loss
    pub final_loss: f32,
    /// Loss history per epoch
    pub loss_history: Vec<f32>,
    /// Number of teachers used
    pub teacher_count: usize,
    /// Student model configuration
    pub student_config: StudentConfig,
    /// Temperature used
    pub temperature: f32,
    /// Alpha used
    pub alpha: f32,
    /// Training status
    pub status: String,
    /// Additional notes
    pub note: String,
}

impl DistillationConfig {
    /// Create a new distillation config
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Builder: set alpha (distillation weight)
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Builder: set number of teachers
    #[must_use]
    pub fn with_teachers(mut self, num_teachers: usize) -> Self {
        self.num_teachers = num_teachers;
        self
    }

    /// Builder: set student config
    #[must_use]
    pub fn with_student(mut self, student: StudentConfig) -> Self {
        self.student = student;
        self
    }

    /// Builder: set training config
    #[must_use]
    pub fn with_training(mut self, training: DistillTrainingConfig) -> Self {
        self.training = training;
        self
    }

    /// Builder: set output directory
    #[must_use]
    pub fn with_output_dir(mut self, output_dir: impl Into<std::path::PathBuf>) -> Self {
        self.output_dir = output_dir.into();
        self
    }

    /// Generate YAML configuration file for entrenar distillation
    #[must_use]
    pub fn to_yaml(&self) -> String {
        format!(
            "# Entrenar Distillation Config\n\
             # Generated by verificar distill\n\
             \n\
             model:\n\
             \x20 type: student\n\
             \x20 hidden_size: {}\n\
             \x20 num_layers: {}\n\
             \n\
             distillation:\n\
             \x20 temperature: {}\n\
             \x20 alpha: {}\n\
             \x20 num_teachers: {}\n\
             \n\
             training:\n\
             \x20 epochs: {}\n\
             \x20 batch_size: {}\n\
             \x20 learning_rate: {:e}\n\
             \n\
             data:\n\
             \x20 teacher_logits: \"/tmp/teacher_logits\"\n\
             \x20 output_dir: {:?}\n",
            self.student.hidden_size,
            self.student.num_layers,
            self.temperature,
            self.alpha,
            self.num_teachers,
            self.training.epochs,
            self.training.batch_size,
            self.training.learning_rate,
            self.output_dir.display()
        )
    }

    /// Run placeholder distillation (simulates training)
    ///
    /// Full distillation requires entrenar LLM feature and teacher model weights.
    /// This returns a placeholder result for testing the pipeline.
    #[must_use]
    pub fn run_placeholder(&self) -> DistillationResult {
        // Simulate decreasing loss over epochs
        let mut loss_history = Vec::with_capacity(self.training.epochs);
        let mut loss = 2.6_f32;

        for _ in 0..self.training.epochs {
            loss *= 0.75; // Simulate 25% improvement per epoch
            loss_history.push(loss);
        }

        DistillationResult {
            final_loss: loss,
            loss_history,
            teacher_count: self.num_teachers,
            student_config: self.student.clone(),
            temperature: self.temperature,
            alpha: self.alpha,
            status: "placeholder".to_string(),
            note: "Full distillation requires entrenar llm feature and teacher model weights"
                .to_string(),
        }
    }

    /// Validate configuration parameters
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid
    pub fn validate(&self) -> Result<(), String> {
        if self.temperature <= 0.0 {
            return Err("temperature must be positive".to_string());
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err("alpha must be in [0.0, 1.0]".to_string());
        }
        if self.num_teachers == 0 {
            return Err("num_teachers must be at least 1".to_string());
        }
        if self.student.hidden_size == 0 {
            return Err("hidden_size must be positive".to_string());
        }
        if self.student.num_layers == 0 {
            return Err("num_layers must be at least 1".to_string());
        }
        if self.training.epochs == 0 {
            return Err("epochs must be at least 1".to_string());
        }
        if self.training.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".to_string());
        }
        Ok(())
    }
}

/// Exporter for entrenar training data
#[derive(Debug)]
pub struct EntrenarExporter {
    config: ExportConfig,
}

impl EntrenarExporter {
    /// Create a new exporter with configuration
    #[must_use]
    pub fn new(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Convert verified tuple to training example
    #[must_use]
    pub fn to_example(&self, tuple: &VerifiedTuple, id: &str) -> CodeTranslationExample {
        let source_lang = tuple.source_language.to_string();
        let target_lang = tuple.target_language.to_string();

        let prompt = self
            .config
            .template
            .format(&source_lang, &target_lang, &tuple.source_code);
        let completion = self
            .config
            .template
            .format_completion(&target_lang, &tuple.target_code);

        CodeTranslationExample {
            id: id.to_string(),
            source_language: source_lang,
            target_language: target_lang,
            source_code: tuple.source_code.clone(),
            target_code: tuple.target_code.clone(),
            prompt,
            completion,
        }
    }

    /// Export verified tuples to training data
    ///
    /// # Errors
    ///
    /// Returns error if export fails
    pub fn export(
        &self,
        tuples: &[VerifiedTuple],
        output_dir: &Path,
    ) -> std::io::Result<ExportStats> {
        let examples: Vec<_> = tuples
            .iter()
            .take(self.config.max_examples.unwrap_or(usize::MAX))
            .enumerate()
            .map(|(i, t)| self.to_example(t, &format!("ex_{i:06}")))
            .collect();

        let (train, val) = self.split_train_val(&examples);

        let stats = ExportStats {
            total: examples.len(),
            train_count: train.len(),
            val_count: val.len(),
            avg_source_len: examples.iter().map(|e| e.source_code.len()).sum::<usize>() as f64
                / examples.len().max(1) as f64,
            avg_target_len: examples.iter().map(|e| e.target_code.len()).sum::<usize>() as f64
                / examples.len().max(1) as f64,
        };

        std::fs::create_dir_all(output_dir)?;

        match self.config.format {
            ExportFormat::Json => {
                self.write_json(&train, &output_dir.join("train.json"))?;
                self.write_json(&val, &output_dir.join("val.json"))?;
            }
            ExportFormat::Jsonl => {
                self.write_jsonl(&train, &output_dir.join("train.jsonl"))?;
                self.write_jsonl(&val, &output_dir.join("val.jsonl"))?;
            }
            ExportFormat::Parquet => {
                // Parquet export requires additional dependencies
                // For now, fall back to JSONL
                self.write_jsonl(&train, &output_dir.join("train.jsonl"))?;
                self.write_jsonl(&val, &output_dir.join("val.jsonl"))?;
            }
        }

        Ok(stats)
    }

    /// Split examples into train/val sets
    fn split_train_val(
        &self,
        examples: &[CodeTranslationExample],
    ) -> (Vec<CodeTranslationExample>, Vec<CodeTranslationExample>) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut train = Vec::new();
        let mut val = Vec::new();

        for (i, example) in examples.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            (self.config.seed, i).hash(&mut hasher);
            let hash = hasher.finish();

            #[allow(clippy::cast_sign_loss)]
            let threshold = (self.config.train_ratio * u64::MAX as f64) as u64;

            if hash < threshold {
                train.push(example.clone());
            } else {
                val.push(example.clone());
            }
        }

        (train, val)
    }

    fn write_json(&self, examples: &[CodeTranslationExample], path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(examples)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(path, json)
    }

    fn write_jsonl(&self, examples: &[CodeTranslationExample], path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        for example in examples {
            let line = serde_json::to_string(example)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            writeln!(file, "{line}")?;
        }
        Ok(())
    }
}

/// Generate entrenar YAML config for the exported data
#[must_use]
pub fn generate_entrenar_config(data_dir: &Path, output_dir: &Path, lora_rank: usize) -> String {
    format!(
        r"# Entrenar configuration for verificar training data
# Generated by verificar v{}

model:
  path: codellama-7b.gguf  # Replace with your base model
  layers: [q_proj, k_proj, v_proj, o_proj]

data:
  train: {}
  val: {}
  batch_size: 4
  seq_len: 2048

optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01

lora:
  rank: {}
  alpha: {}
  target_modules: [q_proj, v_proj]
  dropout: 0.05

training:
  epochs: 3
  grad_clip: 1.0
  lr_scheduler: cosine
  warmup_steps: 100
  save_interval: 1
  output_dir: {}
",
        env!("CARGO_PKG_VERSION"),
        data_dir.join("train.jsonl").display(),
        data_dir.join("val.jsonl").display(),
        lora_rank,
        lora_rank * 2, // alpha = 2 * rank is common
        output_dir.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Language;

    fn sample_tuple() -> VerifiedTuple {
        VerifiedTuple {
            source_language: Language::Python,
            target_language: Language::Rust,
            source_code: "def add(a: int, b: int) -> int:\n    return a + b".to_string(),
            target_code: "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}".to_string(),
            is_correct: true,
            execution_time_ms: 10,
        }
    }

    #[test]
    fn test_prompt_template_instruction() {
        let template = PromptTemplate::instruction_following();
        let prompt = template.format("Python", "Rust", "x = 1");

        assert!(prompt.contains("Python"));
        assert!(prompt.contains("Rust"));
        assert!(prompt.contains("x = 1"));
        assert!(prompt.contains("```Python"));
    }

    #[test]
    fn test_prompt_template_chat() {
        let template = PromptTemplate::chat_style();
        let prompt = template.format("Python", "Rust", "x = 1");

        assert!(prompt.contains("Python"));
        assert!(prompt.contains("idiomatic Rust"));
        assert!(!prompt.contains("```")); // No code blocks in chat style
    }

    #[test]
    fn test_prompt_template_completion() {
        let template = PromptTemplate::completion_style();
        let prompt = template.format("Python", "Rust", "x = 1");

        assert!(prompt.contains("# Python"));
        assert!(prompt.contains("# Rust"));
    }

    #[test]
    fn test_format_completion_with_tags() {
        let template = PromptTemplate::instruction_following();
        let completion = template.format_completion("Rust", "fn main() {}");

        assert!(completion.contains("```Rust"));
        assert!(completion.contains("fn main() {}"));
    }

    #[test]
    fn test_format_completion_without_tags() {
        let template = PromptTemplate::completion_style();
        let completion = template.format_completion("Rust", "fn main() {}");

        assert_eq!(completion, "fn main() {}");
        assert!(!completion.contains("```"));
    }

    #[test]
    fn test_to_example() {
        let config = ExportConfig::default();
        let exporter = EntrenarExporter::new(config);
        let tuple = sample_tuple();

        let example = exporter.to_example(&tuple, "test_001");

        assert_eq!(example.id, "test_001");
        assert_eq!(example.source_language, "python");
        assert_eq!(example.target_language, "rust");
        assert!(example.prompt.contains("def add"));
        assert!(example.completion.contains("fn add"));
    }

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();

        assert_eq!(config.format, ExportFormat::Jsonl);
        assert!((config.train_ratio - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.seed, 42);
        assert!(config.max_examples.is_none());
    }

    #[test]
    fn test_split_train_val_ratio() {
        let config = ExportConfig {
            train_ratio: 0.8,
            ..Default::default()
        };
        let exporter = EntrenarExporter::new(config);

        let examples: Vec<_> = (0..1000)
            .map(|i| CodeTranslationExample {
                id: format!("ex_{i}"),
                source_language: "Python".to_string(),
                target_language: "Rust".to_string(),
                source_code: format!("x = {i}"),
                target_code: format!("let x = {i};"),
                prompt: String::new(),
                completion: String::new(),
            })
            .collect();

        let (train, val) = exporter.split_train_val(&examples);

        // Should be approximately 80/20 split
        let train_ratio = train.len() as f64 / examples.len() as f64;
        assert!(train_ratio > 0.7 && train_ratio < 0.9);
        assert_eq!(train.len() + val.len(), examples.len());
    }

    #[test]
    fn test_split_deterministic() {
        let config = ExportConfig::default();
        let exporter = EntrenarExporter::new(config);

        let examples: Vec<_> = (0..100)
            .map(|i| CodeTranslationExample {
                id: format!("ex_{i}"),
                source_language: "Python".to_string(),
                target_language: "Rust".to_string(),
                source_code: format!("x = {i}"),
                target_code: format!("let x = {i};"),
                prompt: String::new(),
                completion: String::new(),
            })
            .collect();

        let (train1, _) = exporter.split_train_val(&examples);
        let (train2, _) = exporter.split_train_val(&examples);

        assert_eq!(train1.len(), train2.len());
    }

    #[test]
    fn test_generate_entrenar_config() {
        let config =
            generate_entrenar_config(Path::new("data/train"), Path::new("outputs/model"), 16);

        assert!(config.contains("lora:"));
        assert!(config.contains("rank: 16"));
        assert!(config.contains("alpha: 32"));
        assert!(config.contains("train.jsonl"));
        assert!(config.contains("val.jsonl"));
    }

    #[test]
    fn test_export_format_serde() {
        let json = serde_json::to_string(&ExportFormat::Jsonl).unwrap();
        assert_eq!(json, "\"jsonl\"");

        let parsed: ExportFormat = serde_json::from_str("\"parquet\"").unwrap();
        assert_eq!(parsed, ExportFormat::Parquet);
    }

    // RED PHASE: Tests that require full entrenar integration

    #[test]
    #[ignore = "requires filesystem setup"]
    fn test_export_to_jsonl() {
        let config = ExportConfig::default();
        let exporter = EntrenarExporter::new(config);
        let tuples = vec![sample_tuple()];

        let dir = tempfile::tempdir().unwrap();
        let stats = exporter.export(&tuples, dir.path()).unwrap();

        assert_eq!(stats.total, 1);
        assert!(dir.path().join("train.jsonl").exists() || dir.path().join("val.jsonl").exists());
    }

    #[test]
    #[ignore = "requires entrenar integration"]
    fn test_export_to_parquet() {
        // TODO: Implement Parquet export
        // let config = ExportConfig { format: ExportFormat::Parquet, ..Default::default() };
        // let exporter = EntrenarExporter::new(config);
        // let stats = exporter.export(&tuples, dir.path()).unwrap();
        // assert!(dir.path().join("train.parquet").exists());
        unimplemented!("Parquet export not yet implemented")
    }

    #[test]
    #[ignore = "requires entrenar integration"]
    fn test_load_in_entrenar() {
        // TODO: Verify exported data loads correctly in entrenar
        // let config = entrenar::config::load_config("train_config.yaml").unwrap();
        // assert!(config.data.train.exists());
        unimplemented!("Entrenar integration test not yet implemented")
    }

    #[test]
    #[ignore = "requires LLM evaluation"]
    fn test_prompt_quality() {
        // TODO: Evaluate prompt quality with actual LLM
        // - Measure translation accuracy
        // - Compare different prompt templates
        // - Validate on held-out test set
        unimplemented!("LLM evaluation not yet implemented")
    }

    // ========== DISTILLATION CONFIG TESTS ==========

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();

        assert!((config.temperature - 3.0).abs() < f32::EPSILON);
        assert!((config.alpha - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.num_teachers, 1);
        assert_eq!(config.student.hidden_size, 256);
        assert_eq!(config.student.num_layers, 4);
        assert_eq!(config.student.num_classes, 18);
        assert_eq!(config.training.epochs, 3);
    }

    #[test]
    fn test_distillation_config_builder() {
        let config = DistillationConfig::new()
            .with_temperature(5.0)
            .with_alpha(0.9)
            .with_teachers(3)
            .with_output_dir("/tmp/model");

        assert!((config.temperature - 5.0).abs() < f32::EPSILON);
        assert!((config.alpha - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.num_teachers, 3);
        assert_eq!(config.output_dir.to_str().unwrap(), "/tmp/model");
    }

    #[test]
    fn test_distillation_config_with_student() {
        let student = StudentConfig {
            model_type: "custom".to_string(),
            hidden_size: 512,
            num_layers: 8,
            num_classes: 10,
        };

        let config = DistillationConfig::new().with_student(student);

        assert_eq!(config.student.model_type, "custom");
        assert_eq!(config.student.hidden_size, 512);
        assert_eq!(config.student.num_layers, 8);
        assert_eq!(config.student.num_classes, 10);
    }

    #[test]
    fn test_distillation_config_with_training() {
        let training = DistillTrainingConfig {
            epochs: 10,
            batch_size: 64,
            learning_rate: 5e-5,
            grad_clip: 0.5,
            fp16: true,
        };

        let config = DistillationConfig::new().with_training(training);

        assert_eq!(config.training.epochs, 10);
        assert_eq!(config.training.batch_size, 64);
        assert!((config.training.learning_rate - 5e-5).abs() < f64::EPSILON);
        assert!(config.training.fp16);
    }

    #[test]
    fn test_distillation_config_to_yaml() {
        let config = DistillationConfig::default();
        let yaml = config.to_yaml();

        assert!(yaml.contains("temperature: 3"));
        assert!(yaml.contains("alpha: 0.7"));
        assert!(yaml.contains("hidden_size: 256"));
        assert!(yaml.contains("num_layers: 4"));
        assert!(yaml.contains("epochs: 3"));
    }

    #[test]
    fn test_distillation_config_validate_valid() {
        let config = DistillationConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_distillation_config_validate_invalid_temperature() {
        let config = DistillationConfig::default().with_temperature(0.0);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("temperature"));
    }

    #[test]
    fn test_distillation_config_validate_invalid_alpha() {
        let config = DistillationConfig::default().with_alpha(1.5);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("alpha"));
    }

    #[test]
    fn test_distillation_config_validate_invalid_teachers() {
        let config = DistillationConfig::default().with_teachers(0);
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("teachers"));
    }

    #[test]
    fn test_distillation_config_validate_invalid_hidden_size() {
        let mut config = DistillationConfig::default();
        config.student.hidden_size = 0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("hidden_size"));
    }

    #[test]
    fn test_distillation_config_validate_invalid_layers() {
        let mut config = DistillationConfig::default();
        config.student.num_layers = 0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("num_layers"));
    }

    #[test]
    fn test_distillation_config_validate_invalid_epochs() {
        let mut config = DistillationConfig::default();
        config.training.epochs = 0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("epochs"));
    }

    #[test]
    fn test_distillation_config_validate_invalid_lr() {
        let mut config = DistillationConfig::default();
        config.training.learning_rate = 0.0;
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("learning_rate"));
    }

    #[test]
    fn test_run_placeholder() {
        let config = DistillationConfig::default();
        let result = config.run_placeholder();

        assert_eq!(result.teacher_count, 1);
        assert!((result.temperature - 3.0).abs() < f32::EPSILON);
        assert!((result.alpha - 0.7).abs() < f32::EPSILON);
        assert_eq!(result.loss_history.len(), 3); // 3 epochs
        assert!(result.final_loss < 2.6); // Should decrease
        assert_eq!(result.status, "placeholder");
    }

    #[test]
    fn test_distillation_result_serde() {
        let result = DistillationResult {
            final_loss: 0.5,
            loss_history: vec![1.0, 0.75, 0.5],
            teacher_count: 2,
            student_config: StudentConfig::default(),
            temperature: 3.0,
            alpha: 0.7,
            status: "complete".to_string(),
            note: "test".to_string(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let parsed: DistillationResult = serde_json::from_str(&json).unwrap();

        assert!((parsed.final_loss - 0.5).abs() < f32::EPSILON);
        assert_eq!(parsed.teacher_count, 2);
        assert_eq!(parsed.loss_history.len(), 3);
    }

    #[test]
    fn test_student_config_default() {
        let config = StudentConfig::default();

        assert_eq!(config.model_type, "distilled_student");
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 4);
        assert_eq!(config.num_classes, 18);
    }

    #[test]
    fn test_distill_training_config_default() {
        let config = DistillTrainingConfig::default();

        assert_eq!(config.epochs, 3);
        assert_eq!(config.batch_size, 32);
        assert!((config.learning_rate - 1e-4).abs() < f64::EPSILON);
        assert!((config.grad_clip - 1.0).abs() < f32::EPSILON);
        assert!(!config.fp16);
    }

    #[test]
    fn test_distillation_config_debug() {
        let config = DistillationConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("DistillationConfig"));
        assert!(debug.contains("temperature"));
    }

    #[test]
    fn test_distillation_config_clone() {
        let config = DistillationConfig::default();
        let cloned = config.clone();
        assert!((cloned.temperature - config.temperature).abs() < f32::EPSILON);
        assert_eq!(cloned.num_teachers, config.num_teachers);
    }

    #[test]
    fn test_loss_history_decreasing() {
        let config = DistillationConfig::new().with_training(DistillTrainingConfig {
            epochs: 5,
            ..Default::default()
        });
        let result = config.run_placeholder();

        // Verify loss decreases over epochs
        for i in 1..result.loss_history.len() {
            assert!(result.loss_history[i] < result.loss_history[i - 1]);
        }
    }
}
