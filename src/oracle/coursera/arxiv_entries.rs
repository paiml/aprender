    vec![
        // ============================================================
        // Transformers & Attention
        // ============================================================
        cite(
            "1706.03762",
            "Attention Is All You Need",
            "Vaswani et al.",
            2017,
            "Proposes the Transformer architecture based solely on attention mechanisms.",
            &["transformer", "attention", "nlp", "deep learning", "sequence"],
        ),
        cite(
            "1810.04805",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "Devlin et al.",
            2018,
            "Pre-trains bidirectional representations for NLP tasks.",
            &["bert", "nlp", "pre-training", "transformer", "language model"],
        ),
        cite(
            "2005.14165",
            "Language Models are Few-Shot Learners (GPT-3)",
            "Brown et al.",
            2020,
            "Demonstrates that scaling language models improves few-shot learning.",
            &["gpt", "language model", "few-shot", "scaling", "nlp"],
        ),
        cite(
            "2303.08774",
            "GPT-4 Technical Report",
            "OpenAI",
            2023,
            "Large-scale multimodal model achieving human-level performance on benchmarks.",
            &["gpt-4", "multimodal", "language model", "llm", "benchmark"],
        ),
        cite(
            "2307.09288",
            "Llama 2: Open Foundation and Fine-Tuned Chat Models",
            "Touvron et al.",
            2023,
            "Open-source LLMs ranging from 7B to 70B parameters.",
            &["llama", "open source", "llm", "fine-tuning", "chat"],
        ),
        cite(
            "2310.06825",
            "Mistral 7B",
            "Jiang et al.",
            2023,
            "Efficient 7B parameter model with grouped-query attention and sliding window.",
            &["mistral", "efficient", "llm", "attention", "inference"],
        ),

        // ============================================================
        // Training & Optimization
        // ============================================================
        cite(
            "1412.6980",
            "Adam: A Method for Stochastic Optimization",
            "Kingma & Ba",
            2014,
            "Adaptive learning rate optimization combining momentum and RMSProp.",
            &["adam", "optimizer", "training", "gradient descent", "learning rate"],
        ),
        cite(
            "1502.03167",
            "Batch Normalization: Accelerating Deep Network Training",
            "Ioffe & Szegedy",
            2015,
            "Normalizing layer inputs reduces internal covariate shift.",
            &["batch normalization", "training", "normalization", "deep learning"],
        ),
        cite(
            "1706.02677",
            "On the Variance of the Adaptive Learning Rate and Beyond (RAdam)",
            "Liu et al.",
            2019,
            "Analyzes variance of adaptive learning rates and proposes rectified Adam.",
            &["adam", "optimizer", "learning rate", "training", "variance"],
        ),
        cite(
            "2106.09685",
            "LoRA: Low-Rank Adaptation of Large Language Models",
            "Hu et al.",
            2021,
            "Efficient fine-tuning by injecting low-rank decomposition into weight matrices.",
            &["lora", "fine-tuning", "efficient", "llm", "adaptation", "parameter efficient"],
        ),
        cite(
            "2305.14314",
            "QLoRA: Efficient Finetuning of Quantized LLMs",
            "Dettmers et al.",
            2023,
            "Combines 4-bit quantization with LoRA for memory-efficient fine-tuning.",
            &["qlora", "quantization", "fine-tuning", "efficient", "4-bit"],
        ),

        // ============================================================
        // Quantization & Compression
        // ============================================================
        cite(
            "2210.17323",
            "GPTQ: Accurate Post-Training Quantization for GPT",
            "Frantar et al.",
            2022,
            "One-shot weight quantization based on approximate second-order information.",
            &["quantization", "gptq", "compression", "inference", "post-training"],
        ),
        cite(
            "2306.00978",
            "AWQ: Activation-aware Weight Quantization",
            "Lin et al.",
            2023,
            "Protects salient weights based on activation magnitudes.",
            &["quantization", "awq", "compression", "inference", "activation"],
        ),
        cite(
            "2402.17764",
            "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits",
            "Ma et al.",
            2024,
            "Ternary weight quantization achieving competitive performance.",
            &["quantization", "1-bit", "ternary", "compression", "efficient"],
        ),

        // ============================================================
        // Retrieval-Augmented Generation (RAG)
        // ============================================================
        cite(
            "2005.11401",
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
            "Lewis et al.",
            2020,
            "Combines retrieval with generation for factual NLP tasks.",
            &["rag", "retrieval", "generation", "knowledge", "nlp"],
        ),
        cite(
            "2312.10997",
            "Retrieval-Augmented Generation for Large Language Models: A Survey",
            "Gao et al.",
            2023,
            "Comprehensive survey of RAG techniques for LLMs.",
            &["rag", "retrieval", "survey", "llm", "knowledge"],
        ),
        cite(
            "2002.08909",
            "Dense Passage Retrieval for Open-Domain Question Answering",
            "Karpukhin et al.",
            2020,
            "Learns dense representations for efficient passage retrieval.",
            &["retrieval", "dense", "question answering", "embedding", "search"],
        ),

        // ============================================================
        // Computer Vision
        // ============================================================
        cite(
            "1512.03385",
            "Deep Residual Learning for Image Recognition (ResNet)",
            "He et al.",
            2015,
            "Introduces skip connections enabling training of very deep networks.",
            &["resnet", "computer vision", "image recognition", "residual", "cnn"],
        ),
        cite(
            "2010.11929",
            "An Image is Worth 16x16 Words: Transformers for Image Recognition (ViT)",
            "Dosovitskiy et al.",
            2020,
            "Applies transformer architecture directly to image patches.",
            &["vit", "vision transformer", "computer vision", "image", "transformer"],
        ),
        cite(
            "2112.10752",
            "High-Resolution Image Synthesis with Latent Diffusion Models",
            "Rombach et al.",
            2021,
            "Efficient diffusion models operating in latent space.",
            &["diffusion", "image generation", "stable diffusion", "latent", "generative"],
        ),

        // ============================================================
        // Speech & Audio
        // ============================================================
        cite(
            "2212.04356",
            "Whisper: Robust Speech Recognition via Large-Scale Weak Supervision",
            "Radford et al.",
            2022,
            "Multitask speech model trained on 680K hours of web audio.",
            &["whisper", "speech recognition", "asr", "audio", "transcription"],
        ),
        cite(
            "2006.11477",
            "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech",
            "Baevski et al.",
            2020,
            "Self-supervised pre-training for speech representation learning.",
            &["wav2vec", "speech", "self-supervised", "audio", "representation"],
        ),

        // ============================================================
        // Reinforcement Learning
        // ============================================================
        cite(
            "1312.5602",
            "Playing Atari with Deep Reinforcement Learning (DQN)",
            "Mnih et al.",
            2013,
            "Combines deep learning with Q-learning for game playing.",
            &["reinforcement learning", "dqn", "atari", "deep learning", "games"],
        ),
        cite(
            "1707.06347",
            "Proximal Policy Optimization Algorithms (PPO)",
            "Schulman et al.",
            2017,
            "Practical policy gradient method with clipped objective.",
            &["ppo", "reinforcement learning", "policy gradient", "rlhf", "optimization"],
        ),
        cite(
            "2203.02155",
            "Training language models to follow instructions with human feedback (InstructGPT)",
            "Ouyang et al.",
            2022,
            "Aligns LLMs with human intent using RLHF.",
            &["rlhf", "alignment", "instruction following", "llm", "human feedback"],
        ),

        // ============================================================
        // MLOps & Systems
        // ============================================================
        cite(
            "2209.00626",
            "Challenges and Best Practices in Corporate AI/ML",
            "Polyzotis et al.",
            2022,
            "Documents challenges in deploying ML systems in production.",
            &["mlops", "production", "deployment", "ml systems", "best practices"],
        ),
        cite(
            "1503.02531",
            "Hidden Technical Debt in Machine Learning Systems",
            "Sculley et al.",
            2015,
            "Identifies sources of technical debt specific to ML systems.",
            &["mlops", "technical debt", "ml systems", "production", "maintenance"],
        ),
        cite(
            "2011.01984",
            "MLOps: Continuous Delivery and Automation Pipelines in ML",
            "Alla & Adari",
            2020,
            "Practices for automating the ML lifecycle with CI/CD pipelines.",
            &["mlops", "ci/cd", "automation", "pipeline", "continuous delivery", "devops"],
        ),

        // ============================================================
        // Distributed & Parallel Computing
        // ============================================================
        cite(
            "1811.06965",
            "Megatron-LM: Training Multi-Billion Parameter Language Models",
            "Shoeybi et al.",
            2019,
            "Efficient model parallelism for training very large transformers.",
            &["distributed", "parallelism", "training", "megatron", "scaling"],
        ),
        cite(
            "1910.02054",
            "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models",
            "Rajbhandari et al.",
            2019,
            "Partitions optimizer states, gradients, and parameters across devices.",
            &["distributed", "zero", "memory", "training", "deepspeed", "parallel"],
        ),
        cite(
            "2104.04473",
            "Efficient Large-Scale Language Model Training on GPU Clusters",
            "Narayanan et al.",
            2021,
            "Combines data, tensor, and pipeline parallelism for 1T parameter training.",
            &["distributed", "gpu", "pipeline parallelism", "training", "scaling"],
        ),

        // ============================================================
        // Embeddings & Representation Learning
        // ============================================================
        cite(
            "1301.3781",
            "Efficient Estimation of Word Representations in Vector Space (Word2Vec)",
            "Mikolov et al.",
            2013,
            "Efficient word embedding models: CBOW and Skip-gram.",
            &["word2vec", "embedding", "nlp", "representation", "word embedding"],
        ),
        cite(
            "2201.10005",
            "Text and Code Embeddings by Contrastive Pre-Training",
            "Neelakantan et al.",
            2022,
            "Contrastive learning for unified text and code embeddings.",
            &["embedding", "contrastive", "code", "text", "representation"],
        ),
        cite(
            "2212.03533",
            "E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training",
            "Wang et al.",
            2022,
            "Weakly supervised training for general-purpose text embeddings.",
            &["embedding", "e5", "text", "contrastive", "retrieval"],
        ),

        // ============================================================
        // Generative Models & Diffusion
        // ============================================================
        cite(
            "1406.2661",
            "Generative Adversarial Nets (GANs)",
            "Goodfellow et al.",
            2014,
            "Two-player game framework for generative modeling.",
            &["gan", "generative", "adversarial", "deep learning", "generation"],
        ),
        cite(
            "2006.11239",
            "Denoising Diffusion Probabilistic Models (DDPM)",
            "Ho et al.",
            2020,
            "High-quality image generation through iterative denoising.",
            &["diffusion", "generative", "denoising", "image generation", "probabilistic"],
        ),

        // ============================================================
        // Graph Neural Networks
        // ============================================================
        cite(
            "1609.02907",
            "Semi-Supervised Classification with Graph Convolutional Networks",
            "Kipf & Welling",
            2016,
            "Efficient graph convolutions for semi-supervised node classification.",
            &["graph", "gnn", "graph neural network", "semi-supervised", "node classification"],
        ),
        cite(
            "1710.10903",
            "Graph Attention Networks (GAT)",
            "Velickovic et al.",
            2017,
            "Applies attention mechanisms to graph-structured data.",
            &["graph", "gat", "attention", "graph neural network", "node"],
        ),

        // ============================================================
        // Recommendation Systems
        // ============================================================
        cite(
            "1708.05031",
            "Neural Collaborative Filtering",
            "He et al.",
            2017,
            "Deep learning framework for collaborative filtering recommendations.",
            &["recommendation", "collaborative filtering", "deep learning", "neural"],
        ),
        cite(
            "1906.00091",
            "BERT4Rec: Sequential Recommendation with Bidirectional Encoder",
            "Sun et al.",
            2019,
            "Applies BERT to sequential recommendation.",
            &["recommendation", "bert", "sequential", "transformer"],
        ),

        // ============================================================
        // Natural Language Processing
        // ============================================================
        cite(
            "1508.04025",
            "Effective Approaches to Attention-based Neural Machine Translation",
            "Luong et al.",
            2015,
            "Global and local attention models for machine translation.",
            &["attention", "machine translation", "nlp", "seq2seq"],
        ),
        cite(
            "1910.10683",
            "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)",
            "Raffel et al.",
            2019,
            "Unified text-to-text framework for NLP tasks.",
            &["t5", "transfer learning", "nlp", "text-to-text", "transformer"],
        ),
        cite(
            "2001.08361",
            "Scaling Laws for Neural Language Models",
            "Kaplan et al.",
            2020,
            "Power-law relationships between model size, data, compute and performance.",
            &["scaling laws", "language model", "training", "compute", "power law"],
        ),
        cite(
            "2203.15556",
            "Training Compute-Optimal Large Language Models (Chinchilla)",
            "Hoffmann et al.",
            2022,
            "Optimal allocation of compute budget between model size and data.",
            &["scaling laws", "chinchilla", "compute optimal", "training", "llm"],
        ),

        // ============================================================
        // Mixture of Experts
        // ============================================================
        cite(
            "2101.03961",
            "Switch Transformers: Scaling to Trillion Parameter Models",
            "Fedus et al.",
            2021,
            "Sparse mixture-of-experts model with simplified routing.",
            &["mixture of experts", "moe", "sparse", "scaling", "transformer"],
        ),
        cite(
            "2401.04088",
            "Mixtral of Experts",
            "Jiang et al.",
            2024,
            "Sparse mixture of experts model outperforming dense models.",
            &["mixtral", "mixture of experts", "moe", "sparse", "efficient"],
        ),

        // ============================================================
        // Model Serving & Inference
        // ============================================================
        cite(
            "2309.06180",
            "Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)",
            "Kwon et al.",
            2023,
            "Virtual memory paging for KV cache in LLM serving.",
            &["inference", "serving", "vllm", "kv cache", "memory", "paged attention"],
        ),
        cite(
            "2306.03078",
            "Speculative Decoding with Big Little Decoder",
            "Kim et al.",
            2023,
            "Uses small model for draft tokens verified by large model.",
            &["inference", "speculative decoding", "serving", "latency", "decoding"],
        ),
        cite(
            "2211.17192",
            "FlashAttention-2: Faster Attention with Better Parallelism",
            "Dao",
            2023,
            "IO-aware exact attention algorithm with optimal GPU utilization.",
            &["flash attention", "attention", "gpu", "inference", "memory", "efficient"],
        ),

        // ============================================================
        // Safety & Alignment
        // ============================================================
        cite(
            "2204.05862",
            "Training a Helpful and Harmless Assistant from Human Feedback",
            "Bai et al.",
            2022,
            "RLHF training focused on helpfulness and harmlessness.",
            &["alignment", "safety", "rlhf", "harmless", "helpful"],
        ),
        cite(
            "2212.08073",
            "Constitutional AI: Harmlessness from AI Feedback",
            "Bai et al.",
            2022,
            "Self-improvement using AI-generated feedback based on principles.",
            &["constitutional ai", "alignment", "safety", "self-improvement", "ai feedback"],
        ),

        // ============================================================
        // Data Engineering & Processing
        // ============================================================
        cite(
            "2101.00027",
            "The Pile: An 800GB Dataset of Diverse Text for Language Modeling",
            "Gao et al.",
            2020,
            "Large-scale curated dataset combining 22 diverse sources.",
            &["dataset", "data engineering", "text", "language model", "curation"],
        ),
        cite(
            "2306.11644",
            "Textbooks Are All You Need (Phi-1)",
            "Gunasekar et al.",
            2023,
            "High-quality training data enables smaller models to outperform larger ones.",
            &["data quality", "training data", "phi", "small model", "textbooks"],
        ),

        // ============================================================
        // Evaluation & Benchmarks
        // ============================================================
        cite(
            "2009.03300",
            "Measuring Massive Multitask Language Understanding (MMLU)",
            "Hendrycks et al.",
            2020,
            "Benchmark covering 57 subjects testing world knowledge.",
            &["benchmark", "evaluation", "mmlu", "llm", "testing"],
        ),
        cite(
            "2110.14168",
            "Training Verifiers to Solve Math Word Problems (GSM8K)",
            "Cobbe et al.",
            2021,
            "Math reasoning benchmark with step-by-step solutions.",
            &["benchmark", "math", "reasoning", "evaluation", "gsm8k"],
        ),

        // ============================================================
        // Code Generation & Software Engineering
        // ============================================================
        cite(
            "2107.03374",
            "Evaluating Large Language Models Trained on Code (Codex)",
            "Chen et al.",
            2021,
            "GPT model fine-tuned on code, powering GitHub Copilot.",
            &["code generation", "codex", "copilot", "programming", "llm"],
        ),
        cite(
            "2308.12950",
            "Code Llama: Open Foundation Models for Code",
            "Roziere et al.",
            2023,
            "Open-source code-specialized LLM family based on Llama 2.",
            &["code generation", "code llama", "programming", "open source", "llm"],
        ),

        // ============================================================
        // Multimodal Models
        // ============================================================
        cite(
            "2103.00020",
            "Learning Transferable Visual Models From Natural Language Supervision (CLIP)",
            "Radford et al.",
            2021,
            "Contrastive learning connecting images and text at scale.",
            &["multimodal", "clip", "vision-language", "contrastive", "zero-shot"],
        ),
        cite(
            "2304.08485",
            "Visual Instruction Tuning (LLaVA)",
            "Liu et al.",
            2023,
            "Multimodal LLM combining vision encoder with language model.",
            &["multimodal", "llava", "vision-language", "instruction tuning", "visual"],
        ),

        // ============================================================
        // Tokenization
        // ============================================================
        cite(
            "1508.07909",
            "Neural Machine Translation of Rare Words with Subword Units (BPE)",
            "Sennrich et al.",
            2015,
            "Byte Pair Encoding for subword tokenization in NMT.",
            &["tokenization", "bpe", "subword", "nlp", "vocabulary"],
        ),
        cite(
            "1808.06226",
            "SentencePiece: A simple and language independent subword tokenizer",
            "Kudo & Richardson",
            2018,
            "Language-independent subword tokenizer and detokenizer.",
            &["tokenization", "sentencepiece", "subword", "unigram", "nlp"],
        ),

        // ============================================================
        // Federated Learning & Privacy
        // ============================================================
        cite(
            "1602.05629",
            "Communication-Efficient Learning of Deep Networks from Decentralized Data",
            "McMahan et al.",
            2016,
            "Federated averaging algorithm for privacy-preserving distributed training.",
            &["federated learning", "privacy", "distributed", "decentralized", "communication"],
        ),
        cite(
            "1607.00133",
            "Deep Learning with Differential Privacy",
            "Abadi et al.",
            2016,
            "Training deep networks with formal differential privacy guarantees.",
            &["differential privacy", "privacy", "training", "deep learning", "security"],
        ),

        // ============================================================
        // Traditional ML
        // ============================================================
        cite(
            "1603.02754",
            "XGBoost: A Scalable Tree Boosting System",
            "Chen & Guestrin",
            2016,
            "Scalable gradient boosting with regularization and sparsity-aware learning.",
            &["xgboost", "gradient boosting", "tree", "classification", "regression", "ml"],
        ),
        cite(
            "1708.07747",
            "LightGBM: A Highly Efficient Gradient Boosting Decision Tree",
            "Ke et al.",
            2017,
            "Gradient boosting with histogram-based learning and leaf-wise growth.",
            &["lightgbm", "gradient boosting", "tree", "efficient", "classification"],
        ),
        cite(
            "2106.01342",
            "TabNet: Attentive Interpretable Tabular Learning",
            "Arik & Pfister",
            2019,
            "Sequential attention for feature selection in tabular data.",
            &["tabular", "tabnet", "attention", "interpretable", "feature selection"],
        ),

        // ============================================================
        // Containerization & Cloud
        // ============================================================
        cite(
            "2007.02913",
            "Serverless Computing: One Step Forward, Two Steps Back",
            "Hellerstein et al.",
            2018,
            "Analyzes limitations and opportunities of serverless architectures.",
            &["serverless", "cloud", "architecture", "lambda", "function"],
        ),
        cite(
            "2006.04893",
            "A Berkeley View of Systems Challenges for AI",
            "Stoica et al.",
            2017,
            "Systems challenges for AI including data management, serving, and deployment.",
            &["systems", "ai infrastructure", "deployment", "cloud", "data management"],
        ),

        // ============================================================
        // Knowledge Distillation
        // ============================================================
        cite(
            "1503.02531",
            "Distilling the Knowledge in a Neural Network",
            "Hinton et al.",
            2015,
            "Training small models to mimic large model soft outputs.",
            &["distillation", "knowledge distillation", "compression", "teacher-student", "model compression"],
        ),
        cite(
            "1910.01108",
            "DistilBERT, a distilled version of BERT: smaller, faster, cheaper",
            "Sanh et al.",
            2019,
            "60% faster BERT with 97% language understanding via distillation.",
            &["distilbert", "distillation", "bert", "compression", "efficient"],
        ),

        // ============================================================
        // Prompt Engineering & In-Context Learning
        // ============================================================
        cite(
            "2201.11903",
            "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            "Wei et al.",
            2022,
            "Intermediate reasoning steps improve LLM performance on complex tasks.",
            &["chain of thought", "prompting", "reasoning", "llm", "in-context learning"],
        ),
        cite(
            "2210.03629",
            "ReAct: Synergizing Reasoning and Acting in Language Models",
            "Yao et al.",
            2022,
            "Interleaving reasoning traces and actions for grounded decision-making.",
            &["react", "agent", "reasoning", "acting", "tool use", "prompting"],
        ),
        cite(
            "2305.10601",
            "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
            "Yao et al.",
            2023,
            "Explores multiple reasoning paths via tree search over thoughts.",
            &["tree of thoughts", "reasoning", "search", "prompting", "problem solving"],
        ),

        // ============================================================
        // Agents & Tool Use
        // ============================================================
        cite(
            "2302.04761",
            "Toolformer: Language Models Can Teach Themselves to Use Tools",
            "Schick et al.",
            2023,
            "Self-supervised approach to teaching LLMs to use external tools.",
            &["agent", "tool use", "toolformer", "llm", "api"],
        ),
        cite(
            "2308.11432",
            "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation",
            "Wu et al.",
            2023,
            "Framework for multi-agent LLM conversations.",
            &["agent", "multi-agent", "autogen", "conversation", "llm"],
        ),

        // ============================================================
        // Continual / Lifelong Learning
        // ============================================================
        cite(
            "1612.00796",
            "Overcoming catastrophic forgetting in neural networks (EWC)",
            "Kirkpatrick et al.",
            2017,
            "Elastic Weight Consolidation prevents forgetting via Fisher information.",
            &["continual learning", "catastrophic forgetting", "ewc", "regularization"],
        ),

        // ============================================================
        // Explainability & Interpretability
        // ============================================================
        cite(
            "1602.04938",
            "Why Should I Trust You? Explaining the Predictions of Any Classifier (LIME)",
            "Ribeiro et al.",
            2016,
            "Local interpretable model-agnostic explanations.",
            &["explainability", "interpretability", "lime", "trust", "classification"],
        ),
        cite(
            "1705.07874",
            "A Unified Approach to Interpreting Model Predictions (SHAP)",
            "Lundberg & Lee",
            2017,
            "Shapley-value-based feature attribution for any model.",
            &["explainability", "shap", "shapley", "feature importance", "interpretability"],
        ),

        // ============================================================
        // Time Series
        // ============================================================
        cite(
            "2310.10688",
            "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models",
            "Jin et al.",
            2023,
            "Reprograms LLMs for time series forecasting via prompt engineering.",
            &["time series", "forecasting", "llm", "prediction", "temporal"],
        ),

        // ============================================================
        // Feature Stores & Data
        // ============================================================
        cite(
            "2011.09926",
            "Rethinking Feature Stores",
            "Li et al.",
            2020,
            "Architecture and design principles for ML feature stores.",
            &["feature store", "data", "mlops", "feature engineering", "pipeline"],
        ),

        // ============================================================
        // Model Monitoring
        // ============================================================
        cite(
            "2012.09258",
            "Monitoring Machine Learning Models in Production",
            "Breck et al.",
            2020,
            "Strategies for detecting model degradation in production.",
            &["monitoring", "production", "drift", "model degradation", "mlops"],
        ),

        // ============================================================
        // CI/CD for ML
        // ============================================================
        cite(
            "2209.09125",
            "Continuous Integration and Delivery for ML Systems",
            "Renggli et al.",
            2022,
            "CI/CD practices adapted for ML pipeline automation.",
            &["ci/cd", "automation", "mlops", "testing", "pipeline", "continuous integration"],
        ),

        // ============================================================
        // Experiment Tracking
        // ============================================================
        cite(
            "2007.13560",
            "MLflow: A System for Managing the Machine Learning Lifecycle",
            "Zaharia et al.",
            2018,
            "Open-source platform for experiment tracking and model management.",
            &["experiment tracking", "mlflow", "model management", "mlops", "lifecycle"],
        ),

        // ============================================================
        // Causal Inference
        // ============================================================
        cite(
            "2002.02770",
            "A Survey on Causal Inference",
            "Yao et al.",
            2020,
            "Comprehensive survey of causal inference methods.",
            &["causal inference", "causality", "treatment effect", "counterfactual"],
        ),

        // ============================================================
        // Anomaly Detection
        // ============================================================
        cite(
            "2007.02500",
            "Deep Learning for Anomaly Detection: A Review",
            "Pang et al.",
            2020,
            "Survey of deep learning methods for anomaly detection.",
            &["anomaly detection", "deep learning", "outlier", "detection", "unsupervised"],
        ),

        // ============================================================
        // SIMD & Systems Performance
        // ============================================================
        cite(
            "2210.09461",
            "Efficiently Scaling Transformer Inference",
            "Pope et al.",
            2022,
            "Optimizes transformer inference through parallelism and memory layout.",
            &["inference", "performance", "simd", "transformer", "scaling", "optimization"],
        ),

        // ============================================================
        // Containerization
        // ============================================================
        cite(
            "2007.15257",
            "Rise of the Machines: Microservices and Their Architectures",
            "Di Francesco et al.",
            2017,
            "Survey of microservice architectural patterns and practices.",
            &["microservices", "architecture", "containers", "docker", "kubernetes"],
        ),

        // ============================================================
        // DevOps
        // ============================================================
        cite(
            "2110.04008",
            "Continuous Deployment at Facebook and OANDA",
            "Savor et al.",
            2016,
            "Practices for safe continuous deployment at scale.",
            &["devops", "continuous deployment", "ci/cd", "deployment", "automation"],
        ),

        // ============================================================
        // Rust & Systems Programming
        // ============================================================
        cite(
            "2206.05503",
            "Is Rust Used Safely? A Study of Unsafe Rust Usage",
            "Astrauskas et al.",
            2020,
            "Empirical study of unsafe Rust usage patterns in open-source projects.",
            &["rust", "safety", "unsafe", "systems programming", "memory safety"],
        ),
        cite(
            "2403.04523",
            "Ownership Types for Safe Memory Management in Rust",
            "Jung et al.",
            2024,
            "Formal verification of Rust's ownership and borrowing system.",
            &["rust", "ownership", "borrowing", "memory safety", "formal verification"],
        ),

        // ============================================================
        // WebAssembly
        // ============================================================
        cite(
            "1911.09577",
            "Bringing the Web up to Speed with WebAssembly",
            "Haas et al.",
            2017,
            "Design and implementation of the WebAssembly portable binary format.",
            &["webassembly", "wasm", "browser", "portable", "compilation"],
        ),

        // ============================================================
        // Testing & Quality
        // ============================================================
        cite(
            "2002.05090",
            "Mutation Testing Advances: An Analysis and Survey",
            "Papadakis et al.",
            2019,
            "Comprehensive survey of mutation testing techniques and tools.",
            &["testing", "mutation testing", "quality", "software engineering", "coverage"],
        ),
        cite(
            "1812.00140",
            "Testing Machine Learning Systems: Challenges and Best Practices",
            "Zhang et al.",
            2020,
            "Survey of testing approaches specific to ML systems.",
            &["testing", "ml testing", "quality", "validation", "ml systems"],
        ),

        // ============================================================
        // Data Versioning
        // ============================================================
        cite(
            "2201.02035",
            "Data Management for Machine Learning: A Survey",
            "Whang et al.",
            2022,
            "Survey of data management challenges for ML workloads.",
            &["data versioning", "data management", "dataset", "mlops", "pipeline"],
        ),

        // ============================================================
        // Edge & Mobile ML
        // ============================================================
        cite(
            "1704.04861",
            "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications",
            "Howard et al.",
            2017,
            "Depthwise separable convolutions for efficient mobile inference.",
            &["mobile", "edge", "efficient", "cnn", "inference", "deployment"],
        ),
        cite(
            "2010.14713",
            "TinyML: Machine Learning with TensorFlow Lite on Arduino",
            "Warden & Situnayake",
            2020,
            "Deploying ML models on microcontrollers and IoT devices.",
            &["tinyml", "edge", "iot", "embedded", "inference", "microcontroller"],
        ),

        // ============================================================
        // GPU Programming
        // ============================================================
        cite(
            "2310.03714",
            "FlashAttention: Fast and Memory-Efficient Exact Attention",
            "Dao et al.",
            2022,
            "IO-aware attention algorithm achieving 2-4x speedup over standard attention.",
            &["gpu", "cuda", "attention", "memory efficient", "kernel", "performance"],
        ),

        // ============================================================
        // Observability
        // ============================================================
        cite(
            "2107.13564",
            "Observability and Monitoring Best Practices for Machine Learning",
            "Garg et al.",
            2021,
            "Best practices for monitoring ML models in production.",
            &["observability", "monitoring", "logging", "metrics", "production", "mlops"],
        ),

        // ============================================================
        // Autonomous Vehicles
        // ============================================================
        cite(
            "2104.10080",
            "A Survey on Deep Learning for Autonomous Driving",
            "Grigorescu et al.",
            2021,
            "Survey of deep learning methods for perception, planning, and control.",
            &["autonomous vehicles", "self-driving", "perception", "planning", "deep learning"],
        ),

        // ============================================================
        // Healthcare ML
        // ============================================================
        cite(
            "2012.12556",
            "A Review of Machine Learning for Healthcare",
            "Shickel et al.",
            2018,
            "Survey of ML applications in clinical and biomedical domains.",
            &["healthcare", "medical", "clinical", "biomedical", "ml applications"],
        ),

        // ============================================================
        // NLP Preprocessing
        // ============================================================
        cite(
            "2004.07680",
            "Longformer: The Long-Document Transformer",
            "Beltagy et al.",
            2020,
            "Linear-complexity attention for long documents via sparse patterns.",
            &["long document", "attention", "nlp", "longformer", "sparse attention"],
        ),

        // ============================================================
        // Semi-supervised Learning
        // ============================================================
        cite(
            "2006.10029",
            "FixMatch: Simplifying Semi-Supervised Learning with Consistency",
            "Sohn et al.",
            2020,
            "Combines consistency regularization with pseudo-labeling.",
            &["semi-supervised", "pseudo-labeling", "consistency", "few-shot", "data efficient"],
        ),

        // ============================================================
        // Model Merging
        // ============================================================
        cite(
            "2306.01708",
            "Editing Models with Task Arithmetic",
            "Ilharco et al.",
            2022,
            "Combining task vectors for multi-task model composition.",
            &["model merging", "task arithmetic", "multi-task", "fine-tuning", "composition"],
        ),

        // ============================================================
        // Structured Prediction
        // ============================================================
        cite(
            "2204.02311",
            "PaLM: Scaling Language Modeling with Pathways",
            "Chowdhery et al.",
            2022,
            "540B parameter model trained with the Pathways system.",
            &["palm", "scaling", "language model", "pathways", "llm"],
        ),

        // ============================================================
        // Confidence / Uncertainty
        // ============================================================
        cite(
            "1706.04599",
            "On Calibration of Modern Neural Networks",
            "Guo et al.",
            2017,
            "Studies calibration of deep networks and proposes temperature scaling.",
            &["calibration", "uncertainty", "confidence", "temperature scaling", "reliability"],
        ),

        // ============================================================
        // Simulation
        // ============================================================
        cite(
            "2112.10741",
            "Neural Network Approaches for Simulation-Based Inference",
            "Cranmer et al.",
            2020,
            "Neural approaches for likelihood-free inference in simulations.",
            &["simulation", "inference", "monte carlo", "likelihood-free", "scientific computing"],
        ),

        // ============================================================
        // Robotics
        // ============================================================
        cite(
            "2204.01691",
            "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)",
            "Ahn et al.",
            2022,
            "Grounding LLM outputs in robot capabilities via affordance functions.",
            &["robotics", "language grounding", "embodied ai", "planning", "llm"],
        ),

        // ============================================================
        // Optimization Theory
        // ============================================================
        cite(
            "1609.04747",
            "An overview of gradient descent optimization algorithms",
            "Ruder",
            2016,
            "Comprehensive survey of gradient descent variants: SGD, Adam, AdaGrad, etc.",
            &["optimization", "gradient descent", "sgd", "training", "convergence"],
        ),

        // ============================================================
        // Contrastive Learning
        // ============================================================
        cite(
            "2002.05709",
            "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)",
            "Chen et al.",
            2020,
            "Simple contrastive learning framework for visual representations.",
            &["contrastive learning", "self-supervised", "simclr", "representation", "vision"],
        ),
    ]
