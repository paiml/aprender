/// Register model families into the registry by architecture group
impl ModelRegistry {
    /// Register Meta LLaMA and CodeLlama model variants
    fn add_llama_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.2-1B", "Instruct"),
            size: SizeCategory::Small,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.2-3B", "Instruct"),
            size: SizeCategory::Medium,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.1-8B", "Instruct"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.1-70B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("meta-llama", "Llama-3.3-70B", "Instruct"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // CodeLlama family
        self.add(ModelMetadata {
            id: ModelId::new("meta-llama", "CodeLlama-7b-Instruct-hf"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("meta-llama", "CodeLlama-13b-Instruct-hf"),
            size: SizeCategory::XLarge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("meta-llama", "CodeLlama-34b-Instruct-hf"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("meta-llama", "CodeLlama-70b-Instruct-hf"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    /// Register Mistral AI model variants including Codestral
    fn add_mistral_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::with_variant("mistralai", "Mistral-7B", "Instruct-v0.3"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("mistralai", "Mistral-Nemo-Instruct-2407"),
            size: SizeCategory::XLarge,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("mistralai", "Mistral-Small-24B-Instruct-2501"),
            size: SizeCategory::Huge,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("mistralai", "Codestral-22B-v0.1"),
            size: SizeCategory::Huge,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    /// Register Google Gemma 2, Gemma 3, and CodeGemma model variants
    fn add_gemma_models(&mut self) {
        // Gemma 2 family
        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-2-2b", "it"),
            size: SizeCategory::Small,
            architecture: "gemma2".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-2-9b", "it"),
            size: SizeCategory::Large,
            architecture: "gemma2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-2-27b", "it"),
            size: SizeCategory::Huge,
            architecture: "gemma2".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "codegemma-7b", "it"),
            size: SizeCategory::Large,
            architecture: "gemma".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        // Gemma 3 family
        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-3-1b", "it"),
            size: SizeCategory::Small,
            architecture: "gemma3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-3-4b", "it"),
            size: SizeCategory::Medium,
            architecture: "gemma3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string(), "f16".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-3-12b", "it"),
            size: SizeCategory::XLarge,
            architecture: "gemma3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::with_variant("google", "gemma-3-27b", "it"),
            size: SizeCategory::Huge,
            architecture: "gemma3".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    /// Register Microsoft Phi-3, Phi-3.5, and Phi-4 model variants
    fn add_phi_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-3-mini-4k-instruct"),
            size: SizeCategory::Medium,
            architecture: "phi3".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-3-small-8k-instruct"),
            size: SizeCategory::Large,
            architecture: "phi3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-3-medium-4k-instruct"),
            size: SizeCategory::XLarge,
            architecture: "phi3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-3.5-mini-instruct"),
            size: SizeCategory::Medium,
            architecture: "phi3".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("microsoft", "Phi-4-mini-instruct"),
            size: SizeCategory::Medium,
            architecture: "phi4".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    /// Register DeepSeek Coder and V2 model variants
    fn add_deepseek_coder_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "deepseek-coder-1.3b-instruct"),
            size: SizeCategory::Small,
            architecture: "deepseek".to_string(),
            quantizations: vec![
                "q4_k_m".to_string(),
                "q5_k_m".to_string(),
                "q8_0".to_string(),
            ],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "deepseek-coder-6.7b-instruct"),
            size: SizeCategory::Large,
            architecture: "deepseek".to_string(),
            quantizations: vec![
                "q4_k_m".to_string(),
                "q5_k_m".to_string(),
                "q8_0".to_string(),
            ],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "deepseek-coder-7b-instruct"),
            size: SizeCategory::Large,
            architecture: "deepseek".to_string(),
            quantizations: vec![
                "q4_k_m".to_string(),
                "q5_k_m".to_string(),
                "q8_0".to_string(),
            ],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "deepseek-coder-33b-instruct"),
            size: SizeCategory::Huge,
            architecture: "deepseek".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-Coder-V2-Lite-Instruct"),
            size: SizeCategory::XLarge,
            architecture: "deepseek2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

}
