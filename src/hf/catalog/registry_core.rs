//! Core component registration for the HuggingFace Ecosystem Catalog
//!
//! Contains Hub, Deployment, and Library component definitions.

use super::core::HfCatalog;
use super::types::{AssetType, CatalogComponent, CourseAlignment, HfComponentCategory};

impl HfCatalog {
    pub(crate) fn register_hub_components(&mut self) {
        // Hub models
        self.add(
            CatalogComponent::new("hub-models", "Models", HfComponentCategory::Hub)
                .with_description("700K+ ML models on HuggingFace Hub")
                .with_docs("https://huggingface.co/models")
                .with_tags(&["hub", "models", "repository"])
                .with_course(
                    CourseAlignment::new(1, 1)
                        .with_lessons(&["1.1", "1.3"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Hub datasets
        self.add(
            CatalogComponent::new("hub-datasets", "Datasets", HfComponentCategory::Hub)
                .with_description("100K+ datasets on HuggingFace Hub")
                .with_docs("https://huggingface.co/datasets")
                .with_tags(&["hub", "datasets", "repository"])
                .with_course(
                    CourseAlignment::new(1, 1)
                        .with_lessons(&["1.6", "1.7"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Hub spaces
        self.add(
            CatalogComponent::new("hub-spaces", "Spaces", HfComponentCategory::Hub)
                .with_description("300K+ ML demos and apps")
                .with_docs("https://huggingface.co/spaces")
                .with_tags(&["hub", "spaces", "demos", "apps"])
                .with_course(
                    CourseAlignment::new(5, 2)
                        .with_lessons(&["2.7", "2.8"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Hub Python Library
        self.add(
            CatalogComponent::new(
                "huggingface-hub",
                "Hub Python Library",
                HfComponentCategory::Hub,
            )
            .with_description("Python client to interact with the HuggingFace Hub")
            .with_docs("https://huggingface.co/docs/huggingface_hub")
            .with_repo("https://github.com/huggingface/huggingface_hub")
            .with_pypi("huggingface-hub")
            .with_tags(&["hub", "client", "python", "api"])
            .with_course(
                CourseAlignment::new(1, 1)
                    .with_lessons(&["1.1"])
                    .with_assets(&[AssetType::Reading]),
            ),
        );

        // Huggingface.js
        self.add(
            CatalogComponent::new("huggingface-js", "Huggingface.js", HfComponentCategory::Hub)
                .with_description("JavaScript libraries for HuggingFace with TypeScript types")
                .with_docs("https://huggingface.co/docs/huggingface.js")
                .with_repo("https://github.com/huggingface/huggingface.js")
                .with_npm("@huggingface/hub")
                .with_tags(&["hub", "client", "javascript", "typescript"])
                .with_course(
                    CourseAlignment::new(5, 3)
                        .with_lessons(&["3.6", "3.7"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Tasks
        self.add(
            CatalogComponent::new("tasks", "Tasks", HfComponentCategory::Hub)
                .with_description("Explore demos, models, and datasets for any ML task")
                .with_docs("https://huggingface.co/tasks")
                .with_tags(&["hub", "tasks", "taxonomy"])
                .with_course(
                    CourseAlignment::new(1, 3)
                        .with_lessons(&["3.7"])
                        .with_assets(&[AssetType::Reading]),
                ),
        );

        // Dataset Viewer
        self.add(
            CatalogComponent::new("dataset-viewer", "Dataset Viewer", HfComponentCategory::Hub)
                .with_description("API for metadata, stats, and content of Hub datasets")
                .with_docs("https://huggingface.co/docs/dataset-viewer")
                .with_tags(&["hub", "datasets", "api", "viewer"])
                .with_course(
                    CourseAlignment::new(2, 1)
                        .with_lessons(&["1.2"])
                        .with_assets(&[AssetType::Lab]),
                ),
        );
    }

    pub(crate) fn register_deployment_components(&mut self) {
        // Inference Providers
        self.add(
            CatalogComponent::new(
                "inference-providers",
                "Inference Providers",
                HfComponentCategory::Deployment,
            )
            .with_description("Call 200k+ models hosted by 10+ inference partners")
            .with_docs("https://huggingface.co/docs/api-inference")
            .with_tags(&["inference", "api", "serverless"])
            .with_course(
                CourseAlignment::new(5, 1)
                    .with_lessons(&["1.6", "1.7"])
                    .with_assets(&[AssetType::Video, AssetType::Lab]),
            ),
        );

        // Inference Endpoints
        self.add(
            CatalogComponent::new(
                "inference-endpoints",
                "Inference Endpoints",
                HfComponentCategory::Deployment,
            )
            .with_description("Deploy models on dedicated & fully managed infrastructure")
            .with_docs("https://huggingface.co/docs/inference-endpoints")
            .with_tags(&["inference", "deployment", "dedicated", "managed"])
            .with_course(
                CourseAlignment::new(5, 2)
                    .with_lessons(&["2.1", "2.2", "2.4"])
                    .with_assets(&[AssetType::Video, AssetType::Lab]),
            ),
        );

        // TGI - Text Generation Inference
        self.add(
            CatalogComponent::new(
                "tgi",
                "Text Generation Inference",
                HfComponentCategory::Deployment,
            )
            .with_description("Serve language models with TGI optimized toolkit")
            .with_docs("https://huggingface.co/docs/text-generation-inference")
            .with_repo("https://github.com/huggingface/text-generation-inference")
            .with_tags(&["inference", "llm", "serving", "tgi", "production"])
            .with_deps(&["transformers"])
            .with_course(
                CourseAlignment::new(5, 1)
                    .with_lessons(&["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"])
                    .with_assets(&[
                        AssetType::Video,
                        AssetType::Lab,
                        AssetType::Reading,
                        AssetType::Quiz,
                    ]),
            ),
        );

        // TEI - Text Embeddings Inference
        self.add(
            CatalogComponent::new(
                "tei",
                "Text Embeddings Inference",
                HfComponentCategory::Deployment,
            )
            .with_description("Serve embeddings models with TEI optimized toolkit")
            .with_docs("https://huggingface.co/docs/text-embeddings-inference")
            .with_repo("https://github.com/huggingface/text-embeddings-inference")
            .with_tags(&["inference", "embeddings", "serving", "tei"])
            .with_deps(&["sentence-transformers"]),
        );

        // AWS DLCs
        self.add(
            CatalogComponent::new(
                "aws-dlcs",
                "AWS Deep Learning Containers",
                HfComponentCategory::Deployment,
            )
            .with_description("Train/deploy models from HuggingFace to AWS with DLCs")
            .with_docs("https://huggingface.co/docs/sagemaker")
            .with_tags(&["aws", "sagemaker", "deployment", "cloud"]),
        );

        // Azure
        self.add(
            CatalogComponent::new("azure", "Microsoft Azure", HfComponentCategory::Deployment)
                .with_description("Deploy HuggingFace models on Microsoft Azure")
                .with_docs("https://huggingface.co/docs/hub/azure")
                .with_tags(&["azure", "deployment", "cloud"]),
        );

        // GCP
        self.add(
            CatalogComponent::new("gcp", "Google Cloud", HfComponentCategory::Deployment)
                .with_description("Train and deploy HuggingFace models on Google Cloud")
                .with_docs("https://huggingface.co/docs/hub/google-cloud")
                .with_tags(&["gcp", "deployment", "cloud"]),
        );
    }

    pub(crate) fn register_library_components(&mut self) {
        // Transformers
        self.add(
            CatalogComponent::new("transformers", "Transformers", HfComponentCategory::Library)
                .with_description("State-of-the-art AI models for PyTorch, TensorFlow, JAX")
                .with_docs("https://huggingface.co/docs/transformers")
                .with_repo("https://github.com/huggingface/transformers")
                .with_pypi("transformers")
                .with_tags(&["models", "nlp", "vision", "audio", "multimodal"])
                .with_deps(&["tokenizers", "safetensors", "huggingface-hub"])
                .with_related(&["diffusers", "peft", "trl"])
                .with_course(
                    CourseAlignment::new(1, 2)
                        .with_lessons(&["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                )
                .with_course(
                    CourseAlignment::new(1, 3)
                        .with_lessons(&["3.1", "3.2", "3.3", "3.4", "3.5", "3.6"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Diffusers
        self.add(
            CatalogComponent::new("diffusers", "Diffusers", HfComponentCategory::Library)
                .with_description("State-of-the-art diffusion models in PyTorch")
                .with_docs("https://huggingface.co/docs/diffusers")
                .with_repo("https://github.com/huggingface/diffusers")
                .with_pypi("diffusers")
                .with_tags(&["diffusion", "image-generation", "stable-diffusion"])
                .with_deps(&["transformers", "safetensors"]),
        );

        // Datasets
        self.add(
            CatalogComponent::new("datasets", "Datasets", HfComponentCategory::Library)
                .with_description("Access & share datasets for any ML task")
                .with_docs("https://huggingface.co/docs/datasets")
                .with_repo("https://github.com/huggingface/datasets")
                .with_pypi("datasets")
                .with_tags(&["datasets", "data-loading", "preprocessing"])
                .with_deps(&["huggingface-hub"])
                .with_course(
                    CourseAlignment::new(2, 1)
                        .with_lessons(&["1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7"])
                        .with_assets(&[
                            AssetType::Video,
                            AssetType::Lab,
                            AssetType::Reading,
                            AssetType::Quiz,
                        ]),
                ),
        );

        // Transformers.js
        self.add(
            CatalogComponent::new(
                "transformers-js",
                "Transformers.js",
                HfComponentCategory::Library,
            )
            .with_description("State-of-the-art ML running directly in your browser")
            .with_docs("https://huggingface.co/docs/transformers.js")
            .with_repo("https://github.com/xenova/transformers.js")
            .with_npm("@xenova/transformers")
            .with_tags(&["javascript", "browser", "wasm", "onnx"])
            .with_course(
                CourseAlignment::new(5, 3)
                    .with_lessons(&["3.6", "3.7"])
                    .with_assets(&[AssetType::Video, AssetType::Lab]),
            ),
        );

        // Tokenizers
        self.add(
            CatalogComponent::new("tokenizers", "Tokenizers", HfComponentCategory::Library)
                .with_description("Fast tokenizers optimized for research & production")
                .with_docs("https://huggingface.co/docs/tokenizers")
                .with_repo("https://github.com/huggingface/tokenizers")
                .with_pypi("tokenizers")
                .with_tags(&["tokenization", "bpe", "wordpiece", "sentencepiece"])
                .with_course(
                    CourseAlignment::new(1, 2)
                        .with_lessons(&["2.4"])
                        .with_assets(&[AssetType::Reading]),
                ),
        );

        // Evaluate
        self.add(
            CatalogComponent::new("evaluate", "Evaluate", HfComponentCategory::Library)
                .with_description("Evaluate and compare model performance")
                .with_docs("https://huggingface.co/docs/evaluate")
                .with_repo("https://github.com/huggingface/evaluate")
                .with_pypi("evaluate")
                .with_tags(&["evaluation", "metrics", "benchmarking"])
                .with_course(
                    CourseAlignment::new(2, 3)
                        .with_lessons(&["3.1", "3.2", "3.3", "3.4"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // timm
        self.add(
            CatalogComponent::new("timm", "timm", HfComponentCategory::Library)
                .with_description("State-of-the-art vision models: layers, optimizers, utilities")
                .with_docs("https://huggingface.co/docs/timm")
                .with_repo("https://github.com/huggingface/pytorch-image-models")
                .with_pypi("timm")
                .with_tags(&["vision", "image-classification", "pretrained"])
                .with_course(
                    CourseAlignment::new(1, 3)
                        .with_lessons(&["3.1", "3.2"])
                        .with_assets(&[AssetType::Video, AssetType::Lab]),
                ),
        );

        // Sentence Transformers
        self.add(
            CatalogComponent::new(
                "sentence-transformers",
                "Sentence Transformers",
                HfComponentCategory::Library,
            )
            .with_description("Embeddings, retrieval, and reranking")
            .with_docs("https://www.sbert.net/")
            .with_repo("https://github.com/UKPLab/sentence-transformers")
            .with_pypi("sentence-transformers")
            .with_tags(&["embeddings", "semantic-search", "retrieval", "rag"])
            .with_deps(&["transformers"])
            .with_course(
                CourseAlignment::new(3, 2)
                    .with_lessons(&["2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"])
                    .with_assets(&[
                        AssetType::Video,
                        AssetType::Lab,
                        AssetType::Reading,
                        AssetType::Discussion,
                        AssetType::Quiz,
                    ]),
            ),
        );

        // Kernels
        self.add(
            CatalogComponent::new("kernels", "Kernels", HfComponentCategory::Library)
                .with_description("Load and run compute kernels from the HuggingFace Hub")
                .with_docs("https://huggingface.co/docs/kernels")
                .with_tags(&["kernels", "cuda", "triton", "optimization"]),
        );

        // Safetensors
        self.add(
            CatalogComponent::new("safetensors", "Safetensors", HfComponentCategory::Library)
                .with_description("Safe way to store/distribute neural network weights")
                .with_docs("https://huggingface.co/docs/safetensors")
                .with_repo("https://github.com/huggingface/safetensors")
                .with_pypi("safetensors")
                .with_tags(&["serialization", "safe", "tensors", "format"])
                .with_course(
                    CourseAlignment::new(1, 1)
                        .with_lessons(&["1.4"])
                        .with_assets(&[AssetType::Video]),
                ),
        );
    }
}
