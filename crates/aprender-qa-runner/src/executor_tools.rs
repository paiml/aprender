/// APR tool coverage executor for validating tool integration.
pub struct ToolExecutor {
    model_path: String,
    no_gpu: bool,
    timeout_ms: u64,
    command_runner: Arc<dyn CommandRunner>,
}

impl std::fmt::Debug for ToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolExecutor")
            .field("model_path", &self.model_path)
            .field("no_gpu", &self.no_gpu)
            .field("timeout_ms", &self.timeout_ms)
            .field("command_runner", &"<dyn CommandRunner>")
            .finish()
    }
}

include!("execute.rs");
include!("executor_tools_result_types.rs");
