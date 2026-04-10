/// Register additional model families into the registry
impl ModelRegistry {
    /// Register DeepSeek R1 distilled model variants (Qwen and Llama architectures)
    fn add_deepseek_r1_models(&mut self) {
        // Qwen architecture distills
        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Qwen-1.5B"),
            size: SizeCategory::Small,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Qwen-7B"),
            size: SizeCategory::Large,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Qwen-14B"),
            size: SizeCategory::XLarge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Qwen-32B"),
            size: SizeCategory::Huge,
            architecture: "qwen2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // Llama architecture distills
        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Llama-8B"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("deepseek-ai", "DeepSeek-R1-Distill-Llama-70B"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    /// Register BigCode StarCoder2 model variants
    fn add_starcoder_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("bigcode", "starcoder2-3b"),
            size: SizeCategory::Medium,
            architecture: "starcoder2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                arithmetic: false,
                instruction_following: false,
                multi_turn: false,
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("bigcode", "starcoder2-7b"),
            size: SizeCategory::Large,
            architecture: "starcoder2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                arithmetic: false,
                instruction_following: false,
                multi_turn: false,
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("bigcode", "starcoder2-15b"),
            size: SizeCategory::XLarge,
            architecture: "starcoder2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                arithmetic: false,
                instruction_following: false,
                multi_turn: false,
            },
        });
    }

    /// Register 01.AI Yi 1.5 model variants
    fn add_yi_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("01-ai", "Yi-1.5-6B-Chat"),
            size: SizeCategory::Large,
            architecture: "yi".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("01-ai", "Yi-1.5-9B-Chat"),
            size: SizeCategory::Large,
            architecture: "yi".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("01-ai", "Yi-1.5-34B-Chat"),
            size: SizeCategory::Huge,
            architecture: "yi".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    /// Register SmolLM2, TinyLlama, and StableLM small model variants
    fn add_small_models(&mut self) {
        // SmolLM2 family
        self.add(ModelMetadata {
            id: ModelId::new("HuggingFaceTB", "SmolLM2-135M-Instruct"),
            size: SizeCategory::Tiny,
            architecture: "smollm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                arithmetic: false,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("HuggingFaceTB", "SmolLM2-360M-Instruct"),
            size: SizeCategory::Tiny,
            architecture: "smollm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                arithmetic: false,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("HuggingFaceTB", "SmolLM2-1.7B-Instruct"),
            size: SizeCategory::Small,
            architecture: "smollm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // TinyLlama
        self.add(ModelMetadata {
            id: ModelId::new("TinyLlama", "TinyLlama-1.1B-Chat-v1.0"),
            size: SizeCategory::Small,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                arithmetic: false,
                ..Default::default()
            },
        });

        // StableLM family
        self.add(ModelMetadata {
            id: ModelId::new("stabilityai", "stablelm-2-zephyr-1_6b"),
            size: SizeCategory::Small,
            architecture: "stablelm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("stabilityai", "stablelm-zephyr-3b"),
            size: SizeCategory::Medium,
            architecture: "stablelm".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    /// Register TII UAE Falcon model variants
    fn add_falcon_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("tiiuae", "falcon-7b-instruct"),
            size: SizeCategory::Large,
            architecture: "falcon".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("tiiuae", "falcon-40b-instruct"),
            size: SizeCategory::Huge,
            architecture: "falcon".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities::default(),
        });
    }

    /// Register InternLM 2.5 model variants
    fn add_internlm_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("internlm", "internlm2_5-7b-chat"),
            size: SizeCategory::Large,
            architecture: "internlm2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("internlm", "internlm2_5-20b-chat"),
            size: SizeCategory::Huge,
            architecture: "internlm2".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    /// Register IBM Granite 3.1 model variants
    fn add_granite_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("ibm-granite", "granite-3.1-2b-instruct"),
            size: SizeCategory::Small,
            architecture: "granite".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("ibm-granite", "granite-3.1-8b-instruct"),
            size: SizeCategory::Large,
            architecture: "granite".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("ibm-granite", "granite-3b-code-instruct-128k"),
            size: SizeCategory::Medium,
            architecture: "granite".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }

    /// Register AllenAI OLMo 2 model variants
    fn add_olmo_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("allenai", "OLMo-2-1124-7B-Instruct"),
            size: SizeCategory::Large,
            architecture: "olmo".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("allenai", "OLMo-2-1124-13B-Instruct"),
            size: SizeCategory::XLarge,
            architecture: "olmo".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    /// Register NVIDIA Nemotron model variants
    fn add_nvidia_models(&mut self) {
        self.add(ModelMetadata {
            id: ModelId::new("nvidia", "Llama-3.1-Nemotron-Nano-4B-v1.1"),
            size: SizeCategory::Medium,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("nvidia", "Llama-3.1-Nemotron-70B-Instruct-HF"),
            size: SizeCategory::Huge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });
    }

    /// Register community fine-tuned models (Hermes, OpenChat, Zephyr, Dolphin, etc)
    fn add_community_models(&mut self) {
        // NousResearch Hermes
        self.add(ModelMetadata {
            id: ModelId::new("NousResearch", "Hermes-3-Llama-3.1-8B"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // OpenChat
        self.add(ModelMetadata {
            id: ModelId::new("openchat", "openchat-3.5-0106"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // Zephyr
        self.add(ModelMetadata {
            id: ModelId::new("HuggingFaceH4", "zephyr-7b-beta"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // Dolphin
        self.add(ModelMetadata {
            id: ModelId::new("cognitivecomputations", "dolphin-2.6-mistral-7b"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("cognitivecomputations", "Dolphin3.0-Llama3.1-8B"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // Vicuna
        self.add(ModelMetadata {
            id: ModelId::new("lmsys", "vicuna-7b-v1.5"),
            size: SizeCategory::Large,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        self.add(ModelMetadata {
            id: ModelId::new("lmsys", "vicuna-13b-v1.5"),
            size: SizeCategory::XLarge,
            architecture: "llama".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // OpenHermes
        self.add(ModelMetadata {
            id: ModelId::new("teknium", "OpenHermes-2.5-Mistral-7B"),
            size: SizeCategory::Large,
            architecture: "mistral".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: true,
            supports_system_prompt: true,
            capabilities: ModelCapabilities::default(),
        });

        // WizardCoder
        self.add(ModelMetadata {
            id: ModelId::new("WizardLMTeam", "WizardCoder-15B-V1.0"),
            size: SizeCategory::XLarge,
            architecture: "starcoder".to_string(),
            quantizations: vec!["q4_k_m".to_string(), "q8_0".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });

        self.add(ModelMetadata {
            id: ModelId::new("WizardLMTeam", "WizardCoder-33B-V1.1"),
            size: SizeCategory::Huge,
            architecture: "deepseek".to_string(),
            quantizations: vec!["q4_k_m".to_string()],
            has_chat_template: false,
            supports_system_prompt: false,
            capabilities: ModelCapabilities {
                code_completion: true,
                ..Default::default()
            },
        });
    }
}
