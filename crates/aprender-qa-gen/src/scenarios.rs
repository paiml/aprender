impl ScenarioGenerator {
    /// Create a new generator
    #[must_use]
    pub fn new(model: ModelId) -> Self {
        Self {
            model,
            scenarios_per_combination: 100,
            prompts: default_prompts(),
        }
    }

    /// Set the number of scenarios per combination
    #[must_use]
    pub const fn with_scenarios_per_combination(mut self, count: usize) -> Self {
        self.scenarios_per_combination = count;
        self
    }

    /// Set custom prompts
    #[must_use]
    pub fn with_prompts(mut self, prompts: Vec<String>) -> Self {
        self.prompts = prompts;
        self
    }

    /// Append architecture-targeted prompts from a kernel profile.
    ///
    /// This adds the kernel-specific prompts to the existing prompt set,
    /// so scenarios will exercise both default and architecture-specific prompts.
    #[must_use]
    pub fn with_kernel_profile(mut self, profile: &crate::kernel_profile::KernelProfile) -> Self {
        let extra = profile.all_prompts();
        self.prompts.extend(extra);
        self
    }

    /// Generate all scenarios for a model
    #[must_use]
    pub fn generate(&self) -> Vec<QaScenario> {
        if self.prompts.is_empty() {
            return Vec::new();
        }
        let mut scenarios = Vec::new();
        let mut seed: u64 = 0;

        for modality in Modality::inference_modalities() {
            for backend in Backend::all() {
                for format in Format::all() {
                    for i in 0..self.scenarios_per_combination {
                        let prompt_idx = i % self.prompts.len();
                        let prompt = &self.prompts[prompt_idx];

                        scenarios.push(QaScenario::new(
                            self.model.clone(),
                            modality,
                            backend,
                            format,
                            prompt.clone(),
                            seed,
                        ));

                        seed = seed.wrapping_add(1);
                    }
                }
            }
        }

        scenarios
    }

    /// Generate scenarios for a specific combination
    #[must_use]
    pub fn generate_for(
        &self,
        modality: Modality,
        backend: Backend,
        format: Format,
    ) -> Vec<QaScenario> {
        let mut scenarios = Vec::new();
        let base_seed: u64 = (modality as u64) << 32 | (backend as u64) << 16 | (format as u64);

        for (i, prompt) in self
            .prompts
            .iter()
            .enumerate()
            .take(self.scenarios_per_combination)
        {
            scenarios.push(QaScenario::new(
                self.model.clone(),
                modality,
                backend,
                format,
                prompt.clone(),
                base_seed.wrapping_add(i as u64),
            ));
        }

        scenarios
    }
}

/// Get default test prompts
fn default_prompts() -> Vec<String> {
    vec![
        // Arithmetic (verifiable)
        "What is 2+2?".to_string(),
        "Calculate 7*8".to_string(),
        "What is 15-7?".to_string(),
        "What is 100/4?".to_string(),
        "2+2=".to_string(),
        // Code completion
        "def fibonacci(n):".to_string(),
        "fn main() {".to_string(),
        "async function fetch() {".to_string(),
        "class Person:".to_string(),
        // Instruction following
        "Write a haiku about programming.".to_string(),
        "List three colors.".to_string(),
        "Explain what a variable is in one sentence.".to_string(),
        "Say hello in three languages.".to_string(),
        // Edge cases
        String::new(),            // Empty prompt
        " ".to_string(),          // Whitespace only
        "Hello!".to_string(),     // Simple greeting
        "你好".to_string(),       // Chinese
        "こんにちは".to_string(), // Japanese
    ]
}

