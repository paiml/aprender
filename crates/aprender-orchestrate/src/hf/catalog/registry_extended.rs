//! Extended component registration for the HuggingFace Ecosystem Catalog
//!
//! Contains Training, Collaboration, Community, and Integration component definitions.

use super::core::HfCatalog;
use super::types::{AssetType, CatalogComponent, CourseAlignment, HfComponentCategory};

impl HfCatalog {
    pub(crate) fn register_training_components(&mut self) {
        // PEFT
        self.add(
            CatalogComponent::new("peft", "PEFT", HfComponentCategory::Training)
                .with_description("Parameter-efficient finetuning for large language models")
                .with_docs("https://huggingface.co/docs/peft")
                .with_repo("https://github.com/huggingface/peft")
                .with_pypi("peft")
                .with_tags(&["finetuning", "lora", "qlora", "efficient"])
                .with_deps(&["transformers", "bitsandbytes"])
                .with_course(
                    CourseAlignment::new(4, 1)
                        .with_lessons(&["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                ),
        );

        // Accelerate
        self.add(
            CatalogComponent::new("accelerate", "Accelerate", HfComponentCategory::Training)
                .with_description("Train PyTorch models with multi-GPU, TPU, mixed precision")
                .with_docs("https://huggingface.co/docs/accelerate")
                .with_repo("https://github.com/huggingface/accelerate")
                .with_pypi("accelerate")
                .with_tags(&["distributed", "multi-gpu", "tpu", "mixed-precision"])
                .with_course(
                    CourseAlignment::new(1, 2)
                        .with_lessons(&["2.8"])
                        .with_assets(&[AssetType::Lab]),
                ),
        );

        // Optimum
        self.add(
            CatalogComponent::new("optimum", "Optimum", HfComponentCategory::Training)
                .with_description("Optimize HF Transformers for faster training/inference")
                .with_docs("https://huggingface.co/docs/optimum")
                .with_repo("https://github.com/huggingface/optimum")
                .with_pypi("optimum")
                .with_tags(&["optimization", "onnx", "quantization", "hardware"])
                .with_deps(&["transformers"])
                .with_course(
                    CourseAlignment::new(5, 3)
                        .with_lessons(&["3.1", "3.2", "3.3", "3.4", "3.5"])
                        .with_assets(&[AssetType::Video, AssetType::Lab, AssetType::Reading]),
                ),
        );

        // AWS Trainium/Inferentia
        self.add(
            CatalogComponent::new(
                "aws-trainium",
                "AWS Trainium & Inferentia",
                HfComponentCategory::Training,
            )
            .with_description("Train/deploy Transformers/Diffusers on AWS custom silicon")
            .with_docs("https://huggingface.co/docs/optimum-neuron")
            .with_pypi("optimum-neuron")
            .with_tags(&["aws", "trainium", "inferentia", "hardware"]),
        );

        // Google TPUs
        self.add(
            CatalogComponent::new("tpu", "Google TPUs", HfComponentCategory::Training)
                .with_description("Train and deploy models on Google TPUs via Optimum")
                .with_docs("https://huggingface.co/docs/optimum-tpu")
                .with_tags(&["gcp", "tpu", "hardware"]),
        );

        // TRL
        self.add(
            CatalogComponent::new("trl", "TRL", HfComponentCategory::Training)
                .with_description("Train transformer LMs with reinforcement learning")
                .with_docs("https://huggingface.co/docs/trl")
                .with_repo("https://github.com/huggingface/trl")
                .with_pypi("trl")
                .with_tags(&["rlhf", "dpo", "ppo", "alignment", "sft"])
                .with_deps(&["transformers", "peft"])
                .with_course(
                    CourseAlignment::new(4, 2)
                        .with_lessons(&["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Discussion,
                            AssetType::Quiz,
                        ]),
                )
                .with_course(
                    CourseAlignment::new(4, 3)
                        .with_lessons(&["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                ),
        );

        // Bitsandbytes
        self.add(
            CatalogComponent::new("bitsandbytes", "Bitsandbytes", HfComponentCategory::Training)
                .with_description("Optimize and quantize models with bitsandbytes")
                .with_docs("https://huggingface.co/docs/bitsandbytes")
                .with_repo("https://github.com/TimDettmers/bitsandbytes")
                .with_pypi("bitsandbytes")
                .with_tags(&["quantization", "4bit", "8bit", "nf4", "qlora"])
                .with_course(
                    CourseAlignment::new(4, 1)
                        .with_lessons(&["1.4", "1.5"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Lighteval
        self.add(
            CatalogComponent::new("lighteval", "Lighteval", HfComponentCategory::Training)
                .with_description("All-in-one toolkit to evaluate LLMs across multiple backends")
                .with_docs("https://huggingface.co/docs/lighteval")
                .with_repo("https://github.com/huggingface/lighteval")
                .with_pypi("lighteval")
                .with_tags(&["evaluation", "llm", "benchmarking"]),
        );

        // Trainer API
        self.add(
            CatalogComponent::new("trainer", "Trainer API", HfComponentCategory::Training)
                .with_description("High-level training loops for transformers models")
                .with_docs("https://huggingface.co/docs/transformers/main_classes/trainer")
                .with_tags(&["training", "api", "loops"])
                .with_deps(&["transformers", "datasets"])
                .with_course(
                    CourseAlignment::new(2, 2)
                        .with_lessons(&["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                ),
        );

        // AutoTrain
        self.add(
            CatalogComponent::new("autotrain", "AutoTrain", HfComponentCategory::Training)
                .with_description("AutoTrain API and UI for seamless model training")
                .with_docs("https://huggingface.co/docs/autotrain")
                .with_repo("https://github.com/huggingface/autotrain-advanced")
                .with_pypi("autotrain-advanced")
                .with_tags(&["automl", "no-code", "training"]),
        );
    }

    pub(crate) fn register_collaboration_components(&mut self) {
        // Gradio
        self.add(
            CatalogComponent::new("gradio", "Gradio", HfComponentCategory::Collaboration)
                .with_description("Build ML demos and web apps with a few lines of Python")
                .with_docs("https://www.gradio.app/docs")
                .with_repo("https://github.com/gradio-app/gradio")
                .with_pypi("gradio")
                .with_tags(&["demos", "ui", "web-apps", "interactive"])
                .with_course(
                    CourseAlignment::new(5, 2)
                        .with_lessons(&["2.5", "2.6", "2.7", "2.8"])
                        .with_assets(&[AssetType::Video, AssetType::Lab, AssetType::Quiz]),
                ),
        );

        // Trackio
        self.add(
            CatalogComponent::new("trackio", "Trackio", HfComponentCategory::Collaboration)
                .with_description("Lightweight, local-first experiment tracking library")
                .with_docs("https://huggingface.co/docs/trackio")
                .with_pypi("trackio")
                .with_tags(&["experiment-tracking", "logging", "local"]),
        );

        // smolagents
        self.add(
            CatalogComponent::new("smolagents", "smolagents", HfComponentCategory::Collaboration)
                .with_description("Smol library to build great agents in Python")
                .with_docs("https://huggingface.co/docs/smolagents")
                .with_repo("https://github.com/huggingface/smolagents")
                .with_pypi("smolagents")
                .with_tags(&["agents", "tools", "llm"]),
        );

        // LeRobot
        self.add(
            CatalogComponent::new("lerobot", "LeRobot", HfComponentCategory::Collaboration)
                .with_description("Making AI for Robotics more accessible with end-to-end learning")
                .with_docs("https://huggingface.co/docs/lerobot")
                .with_repo("https://github.com/huggingface/lerobot")
                .with_pypi("lerobot")
                .with_tags(&["robotics", "embodied-ai", "imitation-learning"]),
        );

        // Chat UI
        self.add(
            CatalogComponent::new("chat-ui", "Chat UI", HfComponentCategory::Collaboration)
                .with_description("Open source chat frontend powering HuggingChat")
                .with_docs("https://huggingface.co/docs/chat-ui")
                .with_repo("https://github.com/huggingface/chat-ui")
                .with_tags(&["chat", "ui", "frontend", "huggingchat"]),
        );

        // Leaderboards
        self.add(
            CatalogComponent::new(
                "leaderboards",
                "Leaderboards",
                HfComponentCategory::Collaboration,
            )
            .with_description("Create custom leaderboards on HuggingFace")
            .with_docs("https://huggingface.co/docs/leaderboards")
            .with_tags(&["leaderboards", "benchmarking", "comparison"]),
        );

        // Argilla
        self.add(
            CatalogComponent::new("argilla", "Argilla", HfComponentCategory::Collaboration)
                .with_description("Collaboration tool for building high-quality datasets")
                .with_docs("https://docs.argilla.io/")
                .with_repo("https://github.com/argilla-io/argilla")
                .with_pypi("argilla")
                .with_tags(&["annotation", "labeling", "data-quality"]),
        );

        // Distilabel
        self.add(
            CatalogComponent::new("distilabel", "Distilabel", HfComponentCategory::Collaboration)
                .with_description("Framework for synthetic data generation and AI feedback")
                .with_docs("https://distilabel.argilla.io/")
                .with_repo("https://github.com/argilla-io/distilabel")
                .with_pypi("distilabel")
                .with_tags(&["synthetic-data", "ai-feedback", "data-generation"]),
        );
    }

    pub(crate) fn register_community_components(&mut self) {
        // Blog
        self.add(
            CatalogComponent::new("blog", "Blog", HfComponentCategory::Community)
                .with_description("HuggingFace official blog with tutorials and announcements")
                .with_docs("https://huggingface.co/blog")
                .with_tags(&["blog", "tutorials", "announcements"]),
        );

        // Learn
        self.add(
            CatalogComponent::new("learn", "Learn", HfComponentCategory::Community)
                .with_description("HuggingFace learning resources and courses")
                .with_docs("https://huggingface.co/learn")
                .with_tags(&["learning", "courses", "education"]),
        );

        // Discord
        self.add(
            CatalogComponent::new("discord", "Discord", HfComponentCategory::Community)
                .with_description("HuggingFace community Discord server")
                .with_docs("https://discord.gg/huggingface")
                .with_tags(&["community", "discord", "chat"]),
        );

        // Forum
        self.add(
            CatalogComponent::new("forum", "Forum", HfComponentCategory::Community)
                .with_description("HuggingFace discussion forum")
                .with_docs("https://discuss.huggingface.co/")
                .with_tags(&["community", "forum", "discussion"]),
        );

        // Open LLM Leaderboard
        self.add(
            CatalogComponent::new(
                "open-llm-leaderboard",
                "Open LLM Leaderboard",
                HfComponentCategory::Community,
            )
            .with_description("Track and compare open-source LLM performance")
            .with_docs("https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard")
            .with_tags(&["leaderboard", "llm", "benchmarking", "evaluation"])
            .with_related(&["lighteval", "leaderboards"]),
        );

        // Arena
        self.add(
            CatalogComponent::new("arena", "Chatbot Arena", HfComponentCategory::Community)
                .with_description("Anonymous LLM benchmark via human preference voting")
                .with_docs("https://lmarena.ai/")
                .with_tags(&["arena", "llm", "human-preference", "elo"])
                .with_related(&["open-llm-leaderboard"]),
        );
    }

    pub(crate) fn register_integration_components(&mut self) {
        // Outlines
        self.add(
            CatalogComponent::new("outlines", "Outlines", HfComponentCategory::Collaboration)
                .with_description("Structured text generation with grammar constraints")
                .with_docs("https://outlines-dev.github.io/outlines/")
                .with_repo("https://github.com/outlines-dev/outlines")
                .with_pypi("outlines")
                .with_tags(&["structured-output", "json", "grammar", "constrained-generation"])
                .with_deps(&["transformers"])
                .with_course(
                    CourseAlignment::new(3, 2)
                        .with_lessons(&["2.5"])
                        .with_assets(&[AssetType::Lab]),
                ),
        );

        // Wandb
        self.add(
            CatalogComponent::new("wandb", "Weights & Biases", HfComponentCategory::Collaboration)
                .with_description("Experiment tracking, visualization, and model registry")
                .with_docs("https://docs.wandb.ai/")
                .with_pypi("wandb")
                .with_tags(&["experiment-tracking", "logging", "mlops", "visualization"])
                .with_course(
                    CourseAlignment::new(2, 2)
                        .with_lessons(&["2.5"])
                        .with_assets(&[AssetType::Lab]),
                ),
        );

        // FAISS
        self.add(
            CatalogComponent::new("faiss", "FAISS", HfComponentCategory::Collaboration)
                .with_description(
                    "Facebook's efficient similarity search and clustering of dense vectors",
                )
                .with_docs("https://faiss.ai/")
                .with_repo("https://github.com/facebookresearch/faiss")
                .with_pypi("faiss-cpu")
                .with_tags(&["vector-search", "similarity", "indexing", "rag"])
                .with_course(
                    CourseAlignment::new(3, 2)
                        .with_lessons(&["2.3", "2.4"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );
    }
}
