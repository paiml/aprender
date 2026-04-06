//! Command parsing and execution for the REPL.

use crate::state::{HistoryEntry, LoadedModel, ModelRole, SessionState};
use entrenar_common::{EntrenarError, Result};

/// A parsed command.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Fetch a model from HuggingFace
    Fetch { model_id: String, role: ModelRole },
    /// Inspect a loaded model
    Inspect { target: InspectTarget },
    /// Estimate memory requirements
    Memory { batch_size: Option<u32>, seq_len: Option<usize> },
    /// Set configuration values
    Set { key: String, value: String },
    /// Run distillation
    Distill { dry_run: bool },
    /// Export model to file
    Export { format: String, path: String },
    /// Show command history
    History,
    /// Show help
    Help { topic: Option<String> },
    /// Clear screen
    Clear,
    /// Quit the shell
    Quit,
    /// Unknown command
    Unknown { input: String },
}

/// Target for inspect command.
#[derive(Debug, Clone, PartialEq)]
pub enum InspectTarget {
    /// Inspect layer structure
    Layers,
    /// Inspect memory usage
    Memory,
    /// Inspect all info
    All,
    /// Inspect specific model by name
    Model(String),
}

/// Parse a command string into a Command.
pub fn parse(input: &str) -> Result<Command> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(Command::Unknown { input: String::new() });
    }

    let parts: Vec<&str> = input.split_whitespace().collect();
    let cmd = parts[0].to_lowercase();
    let args = &parts[1..];

    match cmd.as_str() {
        "fetch" | "download" => parse_fetch(args),
        "inspect" | "show" => parse_inspect(args),
        "memory" | "mem" => parse_memory(args),
        "set" => parse_set(args),
        "distill" | "train" => parse_distill(args),
        "export" | "save" => parse_export(args),
        "history" | "hist" => Ok(Command::History),
        "help" | "?" => parse_help(args),
        "clear" | "cls" => Ok(Command::Clear),
        "quit" | "exit" | "q" => Ok(Command::Quit),
        _ => Ok(Command::Unknown { input: input.to_string() }),
    }
}

fn parse_fetch(args: &[&str]) -> Result<Command> {
    if args.is_empty() {
        return Err(EntrenarError::ConfigValue {
            field: "model_id".into(),
            message: "No model ID provided".into(),
            suggestion: "Usage: fetch <model_id> [--teacher|--student]".into(),
        });
    }

    let model_id = args[0].to_string();
    let role = if args.contains(&"--teacher") {
        ModelRole::Teacher
    } else if args.contains(&"--student") {
        ModelRole::Student
    } else {
        ModelRole::None
    };

    Ok(Command::Fetch { model_id, role })
}

fn parse_inspect(args: &[&str]) -> Result<Command> {
    let target = if args.is_empty() {
        InspectTarget::All
    } else {
        match args[0].to_lowercase().as_str() {
            "layers" | "layer" => InspectTarget::Layers,
            "memory" | "mem" => InspectTarget::Memory,
            "all" => InspectTarget::All,
            name => InspectTarget::Model(name.to_string()),
        }
    };

    Ok(Command::Inspect { target })
}

fn parse_memory(args: &[&str]) -> Result<Command> {
    let mut batch_size = None;
    let mut seq_len = None;

    let mut i = 0;
    while i < args.len() {
        match args[i] {
            "--batch" | "-b" if i + 1 < args.len() => {
                batch_size = args[i + 1].parse().ok();
                i += 2;
            }
            "--seq" | "-s" if i + 1 < args.len() => {
                seq_len = args[i + 1].parse().ok();
                i += 2;
            }
            _ => i += 1,
        }
    }

    Ok(Command::Memory { batch_size, seq_len })
}

fn parse_set(args: &[&str]) -> Result<Command> {
    if args.len() < 2 {
        return Err(EntrenarError::ConfigValue {
            field: "set".into(),
            message: "Not enough arguments".into(),
            suggestion: "Usage: set <key> <value>".into(),
        });
    }

    Ok(Command::Set { key: args[0].to_string(), value: args[1..].join(" ") })
}

fn parse_distill(args: &[&str]) -> Result<Command> {
    let dry_run = args.contains(&"--dry-run") || args.contains(&"-n");
    Ok(Command::Distill { dry_run })
}

fn parse_export(args: &[&str]) -> Result<Command> {
    if args.len() < 2 {
        return Err(EntrenarError::ConfigValue {
            field: "export".into(),
            message: "Not enough arguments".into(),
            suggestion: "Usage: export <format> <path>".into(),
        });
    }

    Ok(Command::Export { format: args[0].to_string(), path: args[1].to_string() })
}

fn parse_help(args: &[&str]) -> Result<Command> {
    let topic = args.first().map(ToString::to_string);
    Ok(Command::Help { topic })
}

/// Execute a command and update state.
pub fn execute(cmd: &Command, state: &mut SessionState) -> Result<String> {
    let start = std::time::Instant::now();

    let result = match cmd {
        Command::Fetch { model_id, role } => execute_fetch(model_id, *role, state),
        Command::Inspect { target } => execute_inspect(target, state),
        Command::Memory { batch_size, seq_len } => execute_memory(*batch_size, *seq_len, state),
        Command::Set { key, value } => execute_set(key, value, state),
        Command::Distill { dry_run } => execute_distill(*dry_run, state),
        Command::Export { format, path } => execute_export(format, path, state),
        Command::History => execute_history(state),
        Command::Help { topic } => execute_help(topic.as_deref()),
        Command::Clear => Ok(String::new()),
        Command::Quit => Ok("Goodbye!".to_string()),
        Command::Unknown { input } => {
            if input.is_empty() {
                Ok(String::new())
            } else {
                Err(EntrenarError::ConfigValue {
                    field: "command".into(),
                    message: format!("Unknown command: {input}"),
                    suggestion: "Type 'help' for available commands".into(),
                })
            }
        }
    };

    let duration_ms = start.elapsed().as_millis() as u64;
    let success = result.is_ok();

    // Record in history (except for help/history/clear/quit)
    if !matches!(
        cmd,
        Command::Help { .. }
            | Command::History
            | Command::Clear
            | Command::Quit
            | Command::Unknown { .. }
    ) {
        let cmd_str = format!("{cmd:?}");
        state.add_to_history(HistoryEntry::new(cmd_str, duration_ms, success));
        state.record_command(duration_ms, success);
    }

    result
}

fn execute_fetch(model_id: &str, role: ModelRole, state: &mut SessionState) -> Result<String> {
    // Simulate model fetching
    let model = LoadedModel {
        id: model_id.to_string(),
        path: std::path::PathBuf::from(format!("/tmp/models/{}", model_id.replace('/', "_"))),
        architecture: detect_architecture(model_id),
        parameters: estimate_params(model_id),
        layers: estimate_layers(model_id),
        hidden_dim: 4096,
        role,
    };

    let name = if role == ModelRole::Teacher {
        "teacher"
    } else if role == ModelRole::Student {
        "student"
    } else {
        model_id.split('/').next_back().unwrap_or(model_id)
    };

    state.add_model(name.to_string(), model.clone());

    Ok(format!(
        "✓ Fetched {}\n  Architecture: {}\n  Parameters: {:.1}B\n  Layers: {}",
        model_id,
        model.architecture,
        model.parameters as f64 / 1e9,
        model.layers
    ))
}

fn execute_inspect(target: &InspectTarget, state: &SessionState) -> Result<String> {
    match target {
        InspectTarget::All => {
            if state.loaded_models().is_empty() {
                return Ok("No models loaded. Use 'fetch <model_id>' to load a model.".to_string());
            }

            let mut output = String::from("Loaded Models:\n");
            for (name, model) in state.loaded_models() {
                output.push_str(&format!(
                    "  {} ({}): {:.1}B params, {} layers\n",
                    name,
                    model.id,
                    model.parameters as f64 / 1e9,
                    model.layers
                ));
            }
            Ok(output)
        }
        InspectTarget::Layers => {
            let mut output = String::from("Layer Analysis:\n");
            for (name, model) in state.loaded_models() {
                output.push_str(&format!(
                    "  {}: {} layers, hidden_dim={}\n",
                    name, model.layers, model.hidden_dim
                ));
            }
            Ok(output)
        }
        InspectTarget::Memory => execute_memory(None, None, state),
        InspectTarget::Model(name) => {
            if let Some(model) = state.get_model(name) {
                Ok(format!(
                    "Model: {}\n  ID: {}\n  Path: {}\n  Architecture: {}\n  Parameters: {:.1}B\n  Layers: {}\n  Hidden Dim: {}",
                    name, model.id, model.path.display(), model.architecture,
                    model.parameters as f64 / 1e9, model.layers, model.hidden_dim
                ))
            } else {
                Err(EntrenarError::ModelNotFound { path: name.into() })
            }
        }
    }
}

fn execute_memory(
    batch_size: Option<u32>,
    seq_len: Option<usize>,
    state: &SessionState,
) -> Result<String> {
    let batch = batch_size.unwrap_or(state.preferences().default_batch_size);
    let seq = seq_len.unwrap_or(state.preferences().default_seq_len);

    let total_params: u64 = state.loaded_models().values().map(|m| m.parameters).sum();
    let model_mem = total_params * 2; // FP16
    let activation_mem = u64::from(batch) * (seq as u64) * 4096 * 32 * 2;
    let total = model_mem + activation_mem;

    Ok(format!(
        "Memory Estimate (batch={}, seq={}):\n  Model: {:.1} GB\n  Activations: {:.1} GB\n  Total: {:.1} GB",
        batch, seq,
        model_mem as f64 / 1e9,
        activation_mem as f64 / 1e9,
        total as f64 / 1e9
    ))
}

fn execute_set(key: &str, value: &str, state: &mut SessionState) -> Result<String> {
    match key {
        "batch_size" | "batch" => {
            let v: u32 = value.parse().map_err(|_| EntrenarError::ConfigValue {
                field: "batch_size".into(),
                message: "Invalid number".into(),
                suggestion: "Use a positive integer".into(),
            })?;
            state.preferences_mut().default_batch_size = v;
            Ok(format!("Set batch_size = {v}"))
        }
        "seq_len" | "seq" => {
            let v: usize = value.parse().map_err(|_| EntrenarError::ConfigValue {
                field: "seq_len".into(),
                message: "Invalid number".into(),
                suggestion: "Use a positive integer".into(),
            })?;
            state.preferences_mut().default_seq_len = v;
            Ok(format!("Set seq_len = {v}"))
        }
        _ => Err(EntrenarError::ConfigValue {
            field: key.into(),
            message: "Unknown setting".into(),
            suggestion: "Available settings: batch_size, seq_len".into(),
        }),
    }
}

fn execute_distill(dry_run: bool, state: &SessionState) -> Result<String> {
    let teacher = state.loaded_models().values().find(|m| m.role == ModelRole::Teacher);
    let student = state.loaded_models().values().find(|m| m.role == ModelRole::Student);

    if teacher.is_none() {
        return Err(EntrenarError::ConfigValue {
            field: "teacher".into(),
            message: "No teacher model loaded".into(),
            suggestion: "Use 'fetch <model_id> --teacher' to load a teacher model".into(),
        });
    }

    if student.is_none() {
        return Err(EntrenarError::ConfigValue {
            field: "student".into(),
            message: "No student model loaded".into(),
            suggestion: "Use 'fetch <model_id> --student' to load a student model".into(),
        });
    }

    if dry_run {
        let teacher = teacher.expect("teacher validated non-None above");
        let student = student.expect("student validated non-None above");
        Ok(format!(
            "Dry run configuration:\n  Teacher: {} ({:.1}B)\n  Student: {} ({:.1}B)\n  Ready to train",
            teacher.id, teacher.parameters as f64 / 1e9,
            student.id, student.parameters as f64 / 1e9
        ))
    } else {
        Ok("Training started... (simulated)".to_string())
    }
}

fn execute_export(format: &str, path: &str, _state: &SessionState) -> Result<String> {
    Ok(format!("Exported to {path} in {format} format"))
}

fn execute_history(state: &SessionState) -> Result<String> {
    if state.history().is_empty() {
        return Ok("No command history.".to_string());
    }

    let mut output = String::from("Command History:\n");
    for (i, entry) in state.history().iter().enumerate() {
        let status = if entry.success { "✓" } else { "✗" };
        output.push_str(&format!(
            "  {}. {} {} ({}ms)\n",
            i + 1,
            status,
            entry.command,
            entry.duration_ms
        ));
    }
    Ok(output)
}

fn execute_help(topic: Option<&str>) -> Result<String> {
    match topic {
        Some("fetch") => {
            Ok("fetch <model_id> [--teacher|--student]\n  Download a model from HuggingFace"
                .to_string())
        }
        Some("inspect") => {
            Ok("inspect [layers|memory|all|<model>]\n  Inspect loaded models".to_string())
        }
        Some("memory") => {
            Ok("memory [--batch <n>] [--seq <n>]\n  Estimate memory requirements".to_string())
        }
        Some("distill") => Ok("distill [--dry-run]\n  Run distillation training".to_string()),
        _ => Ok("Available commands:
  fetch <model>      Download model from HuggingFace
  inspect [target]   Inspect loaded models
  memory             Estimate memory requirements
  set <key> <value>  Configure settings
  distill            Run distillation
  export <fmt> <path> Export model
  history            Show command history
  help [topic]       Show help
  quit               Exit shell"
            .to_string()),
    }
}

/// N-04 (Meyer DbC): Best-effort architecture guess from model ID string.
/// This is informational only (shell display) — actual architecture detection
/// for inference uses tensor-name-based `ArchitectureDetector::detect()`.
/// Order matters: more specific patterns must come before generic ones
/// (e.g., "mistral" before "llama" since Mistral inherits LLaMA naming).
const ARCH_PATTERNS: &[(&[&str], &str)] = &[
    (&["qwen"], "qwen"),
    (&["phi"], "phi"),
    (&["falcon"], "falcon"),
    (&["mistral", "mixtral"], "mistral"),
    (&["llama"], "llama"),
    (&["bert"], "bert"),
    (&["gpt"], "gpt"),
];

fn detect_architecture(model_id: &str) -> String {
    let lower = model_id.to_lowercase();
    for (patterns, arch) in ARCH_PATTERNS {
        if patterns.iter().any(|p| lower.contains(p)) {
            return (*arch).to_string();
        }
    }
    eprintln!(
        "Warning: could not detect architecture from model ID '{model_id}', \
         defaulting to 'unknown' (use tensor-based detection for accuracy)"
    );
    "unknown".to_string()
}

fn estimate_params(model_id: &str) -> u64 {
    let lower = model_id.to_lowercase();
    if lower.contains("70b") {
        70_000_000_000
    } else if lower.contains("13b") {
        13_000_000_000
    } else if lower.contains("7b") {
        7_000_000_000
    } else if lower.contains("1.1b") || lower.contains("1b") {
        1_100_000_000
    } else if lower.contains("base") {
        350_000_000
    } else {
        1_000_000_000
    }
}

fn estimate_layers(model_id: &str) -> u32 {
    let lower = model_id.to_lowercase();
    if lower.contains("70b") {
        80
    } else if lower.contains("13b") {
        40
    } else if lower.contains("7b") {
        32
    } else if lower.contains("base") {
        12
    } else {
        24
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fetch() {
        let cmd = parse("fetch meta-llama/Llama-2-7b --teacher").expect("parsing should succeed");
        assert!(matches!(cmd, Command::Fetch { role: ModelRole::Teacher, .. }));

        let cmd =
            parse("fetch TinyLlama/TinyLlama-1.1B --student").expect("parsing should succeed");
        assert!(matches!(cmd, Command::Fetch { role: ModelRole::Student, .. }));
    }

    #[test]
    fn test_parse_inspect() {
        assert!(matches!(
            parse("inspect").expect("parsing should succeed"),
            Command::Inspect { target: InspectTarget::All }
        ));
        assert!(matches!(
            parse("inspect layers").expect("parsing should succeed"),
            Command::Inspect { target: InspectTarget::Layers }
        ));
    }

    #[test]
    fn test_parse_memory() {
        let cmd = parse("memory --batch 64 --seq 1024").expect("parsing should succeed");
        if let Command::Memory { batch_size, seq_len } = cmd {
            assert_eq!(batch_size, Some(64));
            assert_eq!(seq_len, Some(1024));
        } else {
            panic!("Expected Memory command");
        }
    }

    #[test]
    fn test_parse_quit_variants() {
        assert!(matches!(parse("quit").expect("parsing should succeed"), Command::Quit));
        assert!(matches!(parse("exit").expect("parsing should succeed"), Command::Quit));
        assert!(matches!(parse("q").expect("parsing should succeed"), Command::Quit));
    }

    #[test]
    fn test_execute_fetch() {
        let mut state = SessionState::new();
        let result = execute_fetch("meta-llama/Llama-2-7b", ModelRole::Teacher, &mut state);

        assert!(result.is_ok());
        assert!(state.get_model("teacher").is_some());
    }

    #[test]
    fn test_execute_set() {
        let mut state = SessionState::new();

        execute_set("batch_size", "64", &mut state).expect("operation should succeed");
        assert_eq!(state.preferences().default_batch_size, 64);

        execute_set("seq_len", "1024", &mut state).expect("operation should succeed");
        assert_eq!(state.preferences().default_seq_len, 1024);
    }

    #[test]
    fn test_unknown_command() {
        let cmd = parse("foobar").expect("parsing should succeed");
        assert!(matches!(cmd, Command::Unknown { .. }));
    }

    #[test]
    fn test_parse_fetch_missing_model() {
        let result = parse("fetch");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_set_not_enough_args() {
        let result = parse("set batch_size");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_export_not_enough_args() {
        let result = parse("export safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_export_valid() {
        let cmd = parse("export safetensors /tmp/model.st").expect("parsing should succeed");
        if let Command::Export { format, path } = cmd {
            assert_eq!(format, "safetensors");
            assert_eq!(path, "/tmp/model.st");
        } else {
            panic!("Expected Export command");
        }
    }

    #[test]
    fn test_parse_help_with_topic() {
        let cmd = parse("help fetch").expect("parsing should succeed");
        if let Command::Help { topic } = cmd {
            assert_eq!(topic, Some("fetch".to_string()));
        } else {
            panic!("Expected Help command");
        }
    }

    #[test]
    fn test_parse_distill_dry_run() {
        let cmd = parse("distill --dry-run").expect("parsing should succeed");
        if let Command::Distill { dry_run } = cmd {
            assert!(dry_run);
        } else {
            panic!("Expected Distill command");
        }
    }

    #[test]
    fn test_parse_distill_short_flag() {
        let cmd = parse("distill -n").expect("parsing should succeed");
        if let Command::Distill { dry_run } = cmd {
            assert!(dry_run);
        } else {
            panic!("Expected Distill command");
        }
    }

    #[test]
    fn test_parse_inspect_model() {
        let cmd = parse("inspect teacher").expect("parsing should succeed");
        if let Command::Inspect { target } = cmd {
            assert_eq!(target, InspectTarget::Model("teacher".to_string()));
        } else {
            panic!("Expected Inspect command");
        }
    }

    #[test]
    fn test_parse_inspect_memory() {
        let cmd = parse("inspect memory").expect("parsing should succeed");
        assert!(matches!(cmd, Command::Inspect { target: InspectTarget::Memory }));
    }

    #[test]
    fn test_parse_command_aliases() {
        // download = fetch
        assert!(matches!(
            parse("download model").expect("parsing should succeed"),
            Command::Fetch { .. }
        ));
        // show = inspect
        assert!(matches!(
            parse("show layers").expect("parsing should succeed"),
            Command::Inspect { .. }
        ));
        // mem = memory
        assert!(matches!(parse("mem").expect("parsing should succeed"), Command::Memory { .. }));
        // train = distill
        assert!(matches!(parse("train").expect("parsing should succeed"), Command::Distill { .. }));
        // save = export (needs args)
        assert!(matches!(
            parse("save gguf /tmp/out").expect("parsing should succeed"),
            Command::Export { .. }
        ));
        // cls = clear
        assert!(matches!(parse("cls").expect("parsing should succeed"), Command::Clear));
        // ? = help
        assert!(matches!(parse("?").expect("parsing should succeed"), Command::Help { .. }));
        // hist = history
        assert!(matches!(parse("hist").expect("parsing should succeed"), Command::History));
    }

    #[test]
    fn test_execute_inspect_no_models() {
        let state = SessionState::new();
        let result = execute_inspect(&InspectTarget::All, &state);
        assert!(result.expect("load should succeed").contains("No models loaded"));
    }

    #[test]
    fn test_execute_inspect_layers() {
        let mut state = SessionState::new();
        let model = LoadedModel {
            id: "test".to_string(),
            path: std::path::PathBuf::from("/tmp"),
            architecture: "llama".to_string(),
            parameters: 7_000_000_000,
            layers: 32,
            hidden_dim: 4096,
            role: ModelRole::None,
        };
        state.add_model("test".to_string(), model);

        let result =
            execute_inspect(&InspectTarget::Layers, &state).expect("operation should succeed");
        assert!(result.contains("32 layers"));
    }

    #[test]
    fn test_execute_inspect_model_not_found() {
        let state = SessionState::new();
        let result = execute_inspect(&InspectTarget::Model("unknown".to_string()), &state);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_history_empty() {
        let state = SessionState::new();
        let result = execute_history(&state).expect("operation should succeed");
        assert!(result.contains("No command history"));
    }

    #[test]
    fn test_execute_help_topics() {
        let fetch_help = execute_help(Some("fetch")).expect("operation should succeed");
        assert!(fetch_help.contains("Download"));

        let inspect_help = execute_help(Some("inspect")).expect("operation should succeed");
        assert!(inspect_help.contains("Inspect"));

        let memory_help = execute_help(Some("memory")).expect("operation should succeed");
        assert!(memory_help.contains("memory"));

        let distill_help = execute_help(Some("distill")).expect("operation should succeed");
        assert!(distill_help.contains("distill"));

        let general_help = execute_help(None).expect("operation should succeed");
        assert!(general_help.contains("Available commands"));
    }

    #[test]
    fn test_detect_architecture_variants() {
        assert_eq!(detect_architecture("meta-llama/Llama-2-7b"), "llama");
        assert_eq!(detect_architecture("bert-base-uncased"), "bert");
        assert_eq!(detect_architecture("openai-gpt"), "gpt");
        assert_eq!(detect_architecture("mistralai/Mistral-7B"), "mistral");
        assert_eq!(detect_architecture("Qwen/Qwen2.5-Coder-0.5B"), "qwen");
        assert_eq!(detect_architecture("microsoft/phi-2"), "phi");
        assert_eq!(detect_architecture("tiiuae/falcon-7b"), "falcon");
        assert_eq!(detect_architecture("mistralai/Mixtral-8x7B"), "mistral");
        assert_eq!(detect_architecture("custom-model"), "unknown");
    }

    // =========================================================================
    // FALSIFY tests — contract violation sweep (N-04)
    // =========================================================================

    #[test]
    fn test_falsify_n04_mistral_before_llama_in_ambiguous_id() {
        // N-04: "mistral" must be checked before "llama" since some Mistral
        // model IDs contain both substrings. Mistral-specific detection
        // should take priority.
        assert_eq!(
            detect_architecture("mistral-llama-variant"),
            "mistral",
            "Mistral should take priority over LLaMA in ambiguous IDs"
        );
    }

    #[test]
    fn test_falsify_n04_unknown_is_explicit() {
        // N-04: Unknown architecture must be an explicit "unknown" string,
        // never an empty string or a silent default to a known architecture.
        let result = detect_architecture("totally-custom-model-v3");
        assert_eq!(result, "unknown");
        assert!(!result.is_empty());
    }

    #[test]
    fn test_estimate_params_variants() {
        assert_eq!(estimate_params("model-70b"), 70_000_000_000);
        assert_eq!(estimate_params("model-13b"), 13_000_000_000);
        assert_eq!(estimate_params("model-7b"), 7_000_000_000);
        assert_eq!(estimate_params("model-1.1b"), 1_100_000_000);
        assert_eq!(estimate_params("bert-base"), 350_000_000);
    }

    #[test]
    fn test_estimate_layers_variants() {
        assert_eq!(estimate_layers("model-70b"), 80);
        assert_eq!(estimate_layers("model-13b"), 40);
        assert_eq!(estimate_layers("model-7b"), 32);
        assert_eq!(estimate_layers("bert-base"), 12);
    }

    #[test]
    fn test_execute_set_invalid_number() {
        let mut state = SessionState::new();
        let result = execute_set("batch_size", "not_a_number", &mut state);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_set_unknown_key() {
        let mut state = SessionState::new();
        let result = execute_set("unknown_setting", "value", &mut state);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_distill_no_teacher() {
        let state = SessionState::new();
        let result = execute_distill(true, &state);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_distill_no_student() {
        let mut state = SessionState::new();
        let model = LoadedModel {
            id: "teacher".to_string(),
            path: std::path::PathBuf::from("/tmp"),
            architecture: "llama".to_string(),
            parameters: 7_000_000_000,
            layers: 32,
            hidden_dim: 4096,
            role: ModelRole::Teacher,
        };
        state.add_model("teacher".to_string(), model);

        let result = execute_distill(true, &state);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_distill_success() {
        let mut state = SessionState::new();

        let teacher = LoadedModel {
            id: "teacher/model".to_string(),
            path: std::path::PathBuf::from("/tmp/t"),
            architecture: "llama".to_string(),
            parameters: 7_000_000_000,
            layers: 32,
            hidden_dim: 4096,
            role: ModelRole::Teacher,
        };
        state.add_model("teacher".to_string(), teacher);

        let student = LoadedModel {
            id: "student/model".to_string(),
            path: std::path::PathBuf::from("/tmp/s"),
            architecture: "llama".to_string(),
            parameters: 1_000_000_000,
            layers: 12,
            hidden_dim: 2048,
            role: ModelRole::Student,
        };
        state.add_model("student".to_string(), student);

        let result = execute_distill(true, &state).expect("operation should succeed");
        assert!(result.contains("Dry run"));
    }

    #[test]
    fn test_execute_export() {
        let state = SessionState::new();
        let result = execute_export("safetensors", "/tmp/model.st", &state)
            .expect("operation should succeed");
        assert!(result.contains("Exported"));
    }

    #[test]
    fn test_parse_empty_input() {
        let cmd = parse("").expect("parsing should succeed");
        assert!(matches!(cmd, Command::Unknown { .. }));
    }

    #[test]
    fn test_execute_memory_with_args() {
        let state = SessionState::new();
        let result =
            execute_memory(Some(64), Some(1024), &state).expect("operation should succeed");
        assert!(result.contains("batch=64"));
        assert!(result.contains("seq=1024"));
    }

    #[test]
    fn test_command_enum_equality() {
        assert_eq!(Command::Quit, Command::Quit);
        assert_eq!(Command::Clear, Command::Clear);
        assert_eq!(Command::History, Command::History);
    }
}
