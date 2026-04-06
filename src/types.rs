use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Workflow phase in the 5-phase Batuta pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkflowPhase {
    Analysis,
    Transpilation,
    Optimization,
    Validation,
    Deployment,
}

impl WorkflowPhase {
    /// Get the next phase in the workflow
    pub fn next(&self) -> Option<WorkflowPhase> {
        match self {
            WorkflowPhase::Analysis => Some(WorkflowPhase::Transpilation),
            WorkflowPhase::Transpilation => Some(WorkflowPhase::Optimization),
            WorkflowPhase::Optimization => Some(WorkflowPhase::Validation),
            WorkflowPhase::Validation => Some(WorkflowPhase::Deployment),
            WorkflowPhase::Deployment => None,
        }
    }

    /// Get all phases in order
    pub fn all() -> Vec<WorkflowPhase> {
        vec![
            WorkflowPhase::Analysis,
            WorkflowPhase::Transpilation,
            WorkflowPhase::Optimization,
            WorkflowPhase::Validation,
            WorkflowPhase::Deployment,
        ]
    }
}

impl std::fmt::Display for WorkflowPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowPhase::Analysis => write!(f, "Analysis"),
            WorkflowPhase::Transpilation => write!(f, "Transpilation"),
            WorkflowPhase::Optimization => write!(f, "Optimization"),
            WorkflowPhase::Validation => write!(f, "Validation"),
            WorkflowPhase::Deployment => write!(f, "Deployment"),
        }
    }
}

/// Status of a workflow phase
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseStatus {
    NotStarted,
    InProgress,
    Completed,
    Failed,
}

impl std::fmt::Display for PhaseStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhaseStatus::NotStarted => write!(f, "Not Started"),
            PhaseStatus::InProgress => write!(f, "In Progress"),
            PhaseStatus::Completed => write!(f, "Completed"),
            PhaseStatus::Failed => write!(f, "Failed"),
        }
    }
}

/// Information about a single phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseInfo {
    pub phase: WorkflowPhase,
    pub status: PhaseStatus,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub error: Option<String>,
}

impl PhaseInfo {
    pub fn new(phase: WorkflowPhase) -> Self {
        Self {
            phase,
            status: PhaseStatus::NotStarted,
            started_at: None,
            completed_at: None,
            error: None,
        }
    }

    pub fn start(&mut self) {
        self.status = PhaseStatus::InProgress;
        self.started_at = Some(chrono::Utc::now());
    }

    pub fn complete(&mut self) {
        self.status = PhaseStatus::Completed;
        self.completed_at = Some(chrono::Utc::now());
    }

    pub fn fail(&mut self, error: String) {
        self.status = PhaseStatus::Failed;
        self.error = Some(error);
        self.completed_at = Some(chrono::Utc::now());
    }
}

/// Complete workflow state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowState {
    pub current_phase: Option<WorkflowPhase>,
    pub phases: std::collections::HashMap<WorkflowPhase, PhaseInfo>,
}

impl Default for WorkflowState {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkflowState {
    pub fn new() -> Self {
        let mut phases = std::collections::HashMap::new();
        for phase in WorkflowPhase::all() {
            phases.insert(phase, PhaseInfo::new(phase));
        }

        Self { current_phase: None, phases }
    }

    /// Load workflow state from file
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let content = std::fs::read_to_string(path)?;
        let state = serde_json::from_str(&content)?;
        Ok(state)
    }

    /// Save workflow state to file
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Start a phase
    pub fn start_phase(&mut self, phase: WorkflowPhase) {
        self.current_phase = Some(phase);
        if let Some(info) = self.phases.get_mut(&phase) {
            info.start();
        }
    }

    /// Complete the current phase
    pub fn complete_phase(&mut self, phase: WorkflowPhase) {
        if let Some(info) = self.phases.get_mut(&phase) {
            info.complete();
        }

        // Move to next phase if available
        if let Some(next) = phase.next() {
            self.current_phase = Some(next);
        } else {
            self.current_phase = None;
        }
    }

    /// Fail the current phase
    pub fn fail_phase(&mut self, phase: WorkflowPhase, error: String) {
        if let Some(info) = self.phases.get_mut(&phase) {
            info.fail(error);
        }
        self.current_phase = Some(phase);
    }

    /// Get status of a specific phase
    pub fn get_phase_status(&self, phase: WorkflowPhase) -> PhaseStatus {
        self.phases.get(&phase).map(|info| info.status).unwrap_or(PhaseStatus::NotStarted)
    }

    /// Check if a phase is completed
    pub fn is_phase_completed(&self, phase: WorkflowPhase) -> bool {
        self.get_phase_status(phase) == PhaseStatus::Completed
    }

    /// Get overall progress percentage
    pub fn progress_percentage(&self) -> f64 {
        let total = WorkflowPhase::all().len() as f64;
        let completed =
            self.phases.values().filter(|info| info.status == PhaseStatus::Completed).count()
                as f64;

        (completed / total) * 100.0
    }
}

/// Programming language detected in the project
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    Python,
    C,
    Cpp,
    Rust,
    Shell,
    JavaScript,
    TypeScript,
    Go,
    Java,
    Other(String),
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Python => write!(f, "Python"),
            Language::C => write!(f, "C"),
            Language::Cpp => write!(f, "C++"),
            Language::Rust => write!(f, "Rust"),
            Language::Shell => write!(f, "Shell"),
            Language::JavaScript => write!(f, "JavaScript"),
            Language::TypeScript => write!(f, "TypeScript"),
            Language::Go => write!(f, "Go"),
            Language::Java => write!(f, "Java"),
            Language::Other(name) => write!(f, "{}", name),
        }
    }
}

impl std::str::FromStr for Language {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "python" => Language::Python,
            "c" => Language::C,
            "c++" | "cpp" => Language::Cpp,
            "rust" => Language::Rust,
            "shell" | "bash" | "sh" => Language::Shell,
            "javascript" | "js" => Language::JavaScript,
            "typescript" | "ts" => Language::TypeScript,
            "go" | "golang" => Language::Go,
            "java" => Language::Java,
            other => Language::Other(other.to_string()),
        })
    }
}

/// Dependency manager type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyManager {
    Pip,    // requirements.txt
    Pipenv, // Pipfile
    Poetry, // pyproject.toml
    Conda,  // environment.yml
    Cargo,  // Cargo.toml
    Npm,    // package.json
    Yarn,   // yarn.lock
    GoMod,  // go.mod
    Maven,  // pom.xml
    Gradle, // build.gradle
    Make,   // Makefile
}

impl std::fmt::Display for DependencyManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DependencyManager::Pip => write!(f, "pip (requirements.txt)"),
            DependencyManager::Pipenv => write!(f, "Pipenv"),
            DependencyManager::Poetry => write!(f, "Poetry"),
            DependencyManager::Conda => write!(f, "Conda"),
            DependencyManager::Cargo => write!(f, "Cargo"),
            DependencyManager::Npm => write!(f, "npm"),
            DependencyManager::Yarn => write!(f, "Yarn"),
            DependencyManager::GoMod => write!(f, "Go modules"),
            DependencyManager::Maven => write!(f, "Maven"),
            DependencyManager::Gradle => write!(f, "Gradle"),
            DependencyManager::Make => write!(f, "Make"),
        }
    }
}

/// Information about detected dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    pub manager: DependencyManager,
    pub file_path: PathBuf,
    pub count: Option<usize>,
}

/// Language statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageStats {
    pub language: Language,
    pub file_count: usize,
    pub line_count: usize,
    pub percentage: f64,
}

/// Complete project analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectAnalysis {
    pub root_path: PathBuf,
    pub languages: Vec<LanguageStats>,
    pub primary_language: Option<Language>,
    pub dependencies: Vec<DependencyInfo>,
    pub total_files: usize,
    pub total_lines: usize,
    pub tdg_score: Option<f64>,
}

impl ProjectAnalysis {
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
            languages: Vec::new(),
            primary_language: None,
            dependencies: Vec::new(),
            total_files: 0,
            total_lines: 0,
            tdg_score: None,
        }
    }

    pub fn recommend_transpiler(&self) -> Option<&'static str> {
        contract_pre_transpile!(self);
        match self.primary_language.as_ref()? {
            Language::Python => Some("Depyler (Python → Rust)"),
            Language::C | Language::Cpp => Some("Decy (C/C++ → Rust)"),
            Language::Shell => Some("Bashrs (Shell → Rust)"),
            Language::Rust => Some("Already Rust! Consider Ruchy for gradual typing."),
            _ => None,
        }
    }

    pub fn has_ml_dependencies(&self) -> bool {
        // Check if project uses ML frameworks
        self.dependencies.iter().any(|dep| {
            matches!(
                dep.manager,
                DependencyManager::Pip | DependencyManager::Conda | DependencyManager::Poetry
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // WorkflowPhase Tests
    // =========================================================================

    #[test]
    fn test_workflow_phase_next() {
        assert_eq!(WorkflowPhase::Analysis.next(), Some(WorkflowPhase::Transpilation));
        assert_eq!(WorkflowPhase::Transpilation.next(), Some(WorkflowPhase::Optimization));
        assert_eq!(WorkflowPhase::Optimization.next(), Some(WorkflowPhase::Validation));
        assert_eq!(WorkflowPhase::Validation.next(), Some(WorkflowPhase::Deployment));
        assert_eq!(WorkflowPhase::Deployment.next(), None);
    }

    #[test]
    fn test_workflow_phase_all() {
        let phases = WorkflowPhase::all();
        assert_eq!(phases.len(), 5);
        assert_eq!(phases[0], WorkflowPhase::Analysis);
        assert_eq!(phases[1], WorkflowPhase::Transpilation);
        assert_eq!(phases[2], WorkflowPhase::Optimization);
        assert_eq!(phases[3], WorkflowPhase::Validation);
        assert_eq!(phases[4], WorkflowPhase::Deployment);
    }

    #[test]
    fn test_workflow_phase_display() {
        assert_eq!(WorkflowPhase::Analysis.to_string(), "Analysis");
        assert_eq!(WorkflowPhase::Transpilation.to_string(), "Transpilation");
        assert_eq!(WorkflowPhase::Optimization.to_string(), "Optimization");
        assert_eq!(WorkflowPhase::Validation.to_string(), "Validation");
        assert_eq!(WorkflowPhase::Deployment.to_string(), "Deployment");
    }

    #[test]
    fn test_workflow_phase_serialization() {
        let phase = WorkflowPhase::Analysis;
        let json = serde_json::to_string(&phase).expect("json serialize failed");
        let deserialized: WorkflowPhase =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(phase, deserialized);
    }

    // =========================================================================
    // PhaseStatus Tests
    // =========================================================================

    #[test]
    fn test_phase_status_display() {
        assert_eq!(PhaseStatus::NotStarted.to_string(), "Not Started");
        assert_eq!(PhaseStatus::InProgress.to_string(), "In Progress");
        assert_eq!(PhaseStatus::Completed.to_string(), "Completed");
        assert_eq!(PhaseStatus::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_phase_status_equality() {
        assert_eq!(PhaseStatus::NotStarted, PhaseStatus::NotStarted);
        assert_eq!(PhaseStatus::Completed, PhaseStatus::Completed);
        assert_ne!(PhaseStatus::NotStarted, PhaseStatus::Completed);
        assert_ne!(PhaseStatus::InProgress, PhaseStatus::Failed);
    }

    #[test]
    fn test_phase_status_serialization() {
        let status = PhaseStatus::Completed;
        let json = serde_json::to_string(&status).expect("json serialize failed");
        let deserialized: PhaseStatus =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(status, deserialized);
    }

    // =========================================================================
    // PhaseInfo Tests
    // =========================================================================

    #[test]
    fn test_phase_info_new() {
        let info = PhaseInfo::new(WorkflowPhase::Analysis);
        assert_eq!(info.phase, WorkflowPhase::Analysis);
        assert_eq!(info.status, PhaseStatus::NotStarted);
        assert!(info.started_at.is_none());
        assert!(info.completed_at.is_none());
        assert!(info.error.is_none());
    }

    #[test]
    fn test_phase_info_start() {
        let mut info = PhaseInfo::new(WorkflowPhase::Analysis);
        info.start();
        assert_eq!(info.status, PhaseStatus::InProgress);
        assert!(info.started_at.is_some());
        assert!(info.completed_at.is_none());
    }

    #[test]
    fn test_phase_info_complete() {
        let mut info = PhaseInfo::new(WorkflowPhase::Analysis);
        info.start();
        info.complete();
        assert_eq!(info.status, PhaseStatus::Completed);
        assert!(info.started_at.is_some());
        assert!(info.completed_at.is_some());
        assert!(info.error.is_none());
    }

    #[test]
    fn test_phase_info_fail() {
        let mut info = PhaseInfo::new(WorkflowPhase::Analysis);
        info.start();
        info.fail("Test error".to_string());
        assert_eq!(info.status, PhaseStatus::Failed);
        assert_eq!(info.error.as_deref(), Some("Test error"));
        assert!(info.completed_at.is_some());
    }

    #[test]
    fn test_phase_info_serialization() {
        let info = PhaseInfo::new(WorkflowPhase::Transpilation);
        let json = serde_json::to_string(&info).expect("json serialize failed");
        let deserialized: PhaseInfo = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(info.phase, deserialized.phase);
        assert_eq!(info.status, deserialized.status);
    }

    // =========================================================================
    // WorkflowState Tests
    // =========================================================================

    #[test]
    fn test_workflow_state_new() {
        let state = WorkflowState::new();
        assert!(state.current_phase.is_none());
        assert_eq!(state.phases.len(), 5);

        for phase in WorkflowPhase::all() {
            assert!(state.phases.contains_key(&phase));
            assert_eq!(state.get_phase_status(phase), PhaseStatus::NotStarted);
        }
    }

    #[test]
    fn test_workflow_state_default() {
        let state = WorkflowState::default();
        assert!(state.current_phase.is_none());
        assert_eq!(state.phases.len(), 5);
    }

    #[test]
    fn test_workflow_state_start_phase() {
        let mut state = WorkflowState::new();
        state.start_phase(WorkflowPhase::Analysis);

        assert_eq!(state.current_phase, Some(WorkflowPhase::Analysis));
        assert_eq!(state.get_phase_status(WorkflowPhase::Analysis), PhaseStatus::InProgress);
    }

    #[test]
    fn test_workflow_state_complete_phase() {
        let mut state = WorkflowState::new();
        state.start_phase(WorkflowPhase::Analysis);
        state.complete_phase(WorkflowPhase::Analysis);

        assert_eq!(state.get_phase_status(WorkflowPhase::Analysis), PhaseStatus::Completed);
        // Should automatically move to next phase
        assert_eq!(state.current_phase, Some(WorkflowPhase::Transpilation));
    }

    #[test]
    fn test_workflow_state_complete_final_phase() {
        let mut state = WorkflowState::new();
        state.start_phase(WorkflowPhase::Deployment);
        state.complete_phase(WorkflowPhase::Deployment);

        assert_eq!(state.get_phase_status(WorkflowPhase::Deployment), PhaseStatus::Completed);
        // No next phase after Deployment
        assert!(state.current_phase.is_none());
    }

    #[test]
    fn test_workflow_state_fail_phase() {
        let mut state = WorkflowState::new();
        state.start_phase(WorkflowPhase::Analysis);
        state.fail_phase(WorkflowPhase::Analysis, "Analysis failed".to_string());

        assert_eq!(state.get_phase_status(WorkflowPhase::Analysis), PhaseStatus::Failed);
        assert_eq!(state.current_phase, Some(WorkflowPhase::Analysis));

        let phase_info = state.phases.get(&WorkflowPhase::Analysis).expect("key not found");
        assert_eq!(phase_info.error.as_deref(), Some("Analysis failed"));
    }

    #[test]
    fn test_workflow_state_is_phase_completed() {
        let mut state = WorkflowState::new();
        assert!(!state.is_phase_completed(WorkflowPhase::Analysis));

        state.start_phase(WorkflowPhase::Analysis);
        assert!(!state.is_phase_completed(WorkflowPhase::Analysis));

        state.complete_phase(WorkflowPhase::Analysis);
        assert!(state.is_phase_completed(WorkflowPhase::Analysis));
    }

    #[test]
    fn test_workflow_state_progress_percentage() {
        let mut state = WorkflowState::new();
        assert_eq!(state.progress_percentage(), 0.0);

        state.start_phase(WorkflowPhase::Analysis);
        state.complete_phase(WorkflowPhase::Analysis);
        assert_eq!(state.progress_percentage(), 20.0); // 1/5 = 20%

        state.start_phase(WorkflowPhase::Transpilation);
        state.complete_phase(WorkflowPhase::Transpilation);
        assert_eq!(state.progress_percentage(), 40.0); // 2/5 = 40%
    }

    #[test]
    fn test_workflow_state_save_and_load() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let state_path = temp_dir.path().join("workflow-state.json");

        let mut state = WorkflowState::new();
        state.start_phase(WorkflowPhase::Analysis);
        state.complete_phase(WorkflowPhase::Analysis);

        state.save(&state_path).expect("save failed");
        assert!(state_path.exists());

        let loaded_state = WorkflowState::load(&state_path).expect("unexpected failure");
        assert_eq!(loaded_state.current_phase, state.current_phase);
        assert_eq!(loaded_state.get_phase_status(WorkflowPhase::Analysis), PhaseStatus::Completed);
    }

    #[test]
    fn test_workflow_state_load_nonexistent() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let state_path = temp_dir.path().join("nonexistent.json");

        let state = WorkflowState::load(&state_path).expect("unexpected failure");
        assert!(state.current_phase.is_none());
        assert_eq!(state.progress_percentage(), 0.0);
    }

    // =========================================================================
    // Language Tests
    // =========================================================================

    #[test]
    fn test_language_display() {
        assert_eq!(Language::Python.to_string(), "Python");
        assert_eq!(Language::C.to_string(), "C");
        assert_eq!(Language::Cpp.to_string(), "C++");
        assert_eq!(Language::Rust.to_string(), "Rust");
        assert_eq!(Language::Shell.to_string(), "Shell");
        assert_eq!(Language::JavaScript.to_string(), "JavaScript");
        assert_eq!(Language::TypeScript.to_string(), "TypeScript");
        assert_eq!(Language::Go.to_string(), "Go");
        assert_eq!(Language::Java.to_string(), "Java");
        assert_eq!(Language::Other("Ruby".to_string()).to_string(), "Ruby");
    }

    #[test]
    fn test_language_equality() {
        assert_eq!(Language::Python, Language::Python);
        assert_ne!(Language::Python, Language::Rust);
        assert_eq!(Language::Other("Kotlin".to_string()), Language::Other("Kotlin".to_string()));
    }

    #[test]
    fn test_language_serialization() {
        let lang = Language::Python;
        let json = serde_json::to_string(&lang).expect("json serialize failed");
        let deserialized: Language = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(lang, deserialized);

        let other_lang = Language::Other("Haskell".to_string());
        let json2 = serde_json::to_string(&other_lang).expect("json serialize failed");
        let deserialized2: Language =
            serde_json::from_str(&json2).expect("json deserialize failed");
        assert_eq!(other_lang, deserialized2);
    }

    // =========================================================================
    // DependencyManager Tests
    // =========================================================================

    #[test]
    fn test_dependency_manager_display() {
        assert_eq!(DependencyManager::Pip.to_string(), "pip (requirements.txt)");
        assert_eq!(DependencyManager::Pipenv.to_string(), "Pipenv");
        assert_eq!(DependencyManager::Poetry.to_string(), "Poetry");
        assert_eq!(DependencyManager::Conda.to_string(), "Conda");
        assert_eq!(DependencyManager::Cargo.to_string(), "Cargo");
        assert_eq!(DependencyManager::Npm.to_string(), "npm");
        assert_eq!(DependencyManager::Yarn.to_string(), "Yarn");
        assert_eq!(DependencyManager::GoMod.to_string(), "Go modules");
        assert_eq!(DependencyManager::Maven.to_string(), "Maven");
        assert_eq!(DependencyManager::Gradle.to_string(), "Gradle");
        assert_eq!(DependencyManager::Make.to_string(), "Make");
    }

    #[test]
    fn test_dependency_manager_equality() {
        assert_eq!(DependencyManager::Pip, DependencyManager::Pip);
        assert_ne!(DependencyManager::Pip, DependencyManager::Cargo);
    }

    #[test]
    fn test_dependency_manager_serialization() {
        let manager = DependencyManager::Cargo;
        let json = serde_json::to_string(&manager).expect("json serialize failed");
        let deserialized: DependencyManager =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(manager, deserialized);
    }

    // =========================================================================
    // DependencyInfo Tests
    // =========================================================================

    #[test]
    fn test_dependency_info_creation() {
        let dep_info = DependencyInfo {
            manager: DependencyManager::Pip,
            file_path: PathBuf::from("requirements.txt"),
            count: Some(10),
        };

        assert_eq!(dep_info.manager, DependencyManager::Pip);
        assert_eq!(dep_info.file_path, PathBuf::from("requirements.txt"));
        assert_eq!(dep_info.count, Some(10));
    }

    #[test]
    fn test_dependency_info_serialization() {
        let dep_info = DependencyInfo {
            manager: DependencyManager::Cargo,
            file_path: PathBuf::from("Cargo.toml"),
            count: Some(5),
        };

        let json = serde_json::to_string(&dep_info).expect("json serialize failed");
        let deserialized: DependencyInfo =
            serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(dep_info.manager, deserialized.manager);
        assert_eq!(dep_info.file_path, deserialized.file_path);
        assert_eq!(dep_info.count, deserialized.count);
    }

    // =========================================================================
    // LanguageStats Tests
    // =========================================================================

    #[test]
    fn test_language_stats_creation() {
        let stats = LanguageStats {
            language: Language::Python,
            file_count: 50,
            line_count: 10000,
            percentage: 75.5,
        };

        assert_eq!(stats.language, Language::Python);
        assert_eq!(stats.file_count, 50);
        assert_eq!(stats.line_count, 10000);
        assert_eq!(stats.percentage, 75.5);
    }

    #[test]
    fn test_language_stats_serialization() {
        let stats = LanguageStats {
            language: Language::Rust,
            file_count: 30,
            line_count: 5000,
            percentage: 24.5,
        };

        let json = serde_json::to_string(&stats).expect("json serialize failed");
        let deserialized: LanguageStats =
            serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(stats.language, deserialized.language);
        assert_eq!(stats.file_count, deserialized.file_count);
        assert_eq!(stats.line_count, deserialized.line_count);
        assert_eq!(stats.percentage, deserialized.percentage);
    }

    // =========================================================================
    // ProjectAnalysis Tests
    // =========================================================================

    #[test]
    fn test_project_analysis_new() {
        let analysis = ProjectAnalysis::new(PathBuf::from("/test/project"));

        assert_eq!(analysis.root_path, PathBuf::from("/test/project"));
        assert_eq!(analysis.languages.len(), 0);
        assert!(analysis.primary_language.is_none());
        assert_eq!(analysis.dependencies.len(), 0);
        assert_eq!(analysis.total_files, 0);
        assert_eq!(analysis.total_lines, 0);
        assert!(analysis.tdg_score.is_none());
    }

    #[test]
    fn test_project_analysis_recommend_transpiler_python() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.primary_language = Some(Language::Python);

        assert_eq!(analysis.recommend_transpiler(), Some("Depyler (Python → Rust)"));
    }

    #[test]
    fn test_project_analysis_recommend_transpiler_c() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.primary_language = Some(Language::C);

        assert_eq!(analysis.recommend_transpiler(), Some("Decy (C/C++ → Rust)"));
    }

    #[test]
    fn test_project_analysis_recommend_transpiler_cpp() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.primary_language = Some(Language::Cpp);

        assert_eq!(analysis.recommend_transpiler(), Some("Decy (C/C++ → Rust)"));
    }

    #[test]
    fn test_project_analysis_recommend_transpiler_shell() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.primary_language = Some(Language::Shell);

        assert_eq!(analysis.recommend_transpiler(), Some("Bashrs (Shell → Rust)"));
    }

    #[test]
    fn test_project_analysis_recommend_transpiler_rust() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.primary_language = Some(Language::Rust);

        assert_eq!(
            analysis.recommend_transpiler(),
            Some("Already Rust! Consider Ruchy for gradual typing.")
        );
    }

    #[test]
    fn test_project_analysis_recommend_transpiler_other() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.primary_language = Some(Language::Java);

        assert_eq!(analysis.recommend_transpiler(), None);
    }

    #[test]
    fn test_project_analysis_recommend_transpiler_none() {
        let analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        assert_eq!(analysis.recommend_transpiler(), None);
    }

    #[test]
    fn test_project_analysis_has_ml_dependencies_pip() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.dependencies.push(DependencyInfo {
            manager: DependencyManager::Pip,
            file_path: PathBuf::from("requirements.txt"),
            count: Some(10),
        });

        assert!(analysis.has_ml_dependencies());
    }

    #[test]
    fn test_project_analysis_has_ml_dependencies_conda() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.dependencies.push(DependencyInfo {
            manager: DependencyManager::Conda,
            file_path: PathBuf::from("environment.yml"),
            count: Some(5),
        });

        assert!(analysis.has_ml_dependencies());
    }

    #[test]
    fn test_project_analysis_has_ml_dependencies_poetry() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.dependencies.push(DependencyInfo {
            manager: DependencyManager::Poetry,
            file_path: PathBuf::from("pyproject.toml"),
            count: Some(8),
        });

        assert!(analysis.has_ml_dependencies());
    }

    #[test]
    fn test_project_analysis_has_ml_dependencies_false() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.dependencies.push(DependencyInfo {
            manager: DependencyManager::Cargo,
            file_path: PathBuf::from("Cargo.toml"),
            count: Some(5),
        });

        assert!(!analysis.has_ml_dependencies());
    }

    #[test]
    fn test_project_analysis_has_ml_dependencies_empty() {
        let analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        assert!(!analysis.has_ml_dependencies());
    }

    #[test]
    fn test_project_analysis_serialization() {
        let mut analysis = ProjectAnalysis::new(PathBuf::from("/test"));
        analysis.primary_language = Some(Language::Python);
        analysis.total_files = 10;
        analysis.total_lines = 1000;
        analysis.tdg_score = Some(85.5);

        let json = serde_json::to_string(&analysis).expect("json serialize failed");
        let deserialized: ProjectAnalysis =
            serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(analysis.root_path, deserialized.root_path);
        assert_eq!(analysis.primary_language, deserialized.primary_language);
        assert_eq!(analysis.total_files, deserialized.total_files);
        assert_eq!(analysis.total_lines, deserialized.total_lines);
        assert_eq!(analysis.tdg_score, deserialized.tdg_score);
    }
}
