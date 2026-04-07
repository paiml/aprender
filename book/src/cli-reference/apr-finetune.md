# apr finetune

```
Fine-tune model with LoRA/QLoRA (GH-244)

Usage: apr finetune [OPTIONS] [FILE]

Arguments:
  [FILE]  Input model file

Options:
  -m, --method <METHOD>                Fine-tuning method: auto, full, lora, qlora [default: auto]
  -r, --rank <RANK>                    LoRA rank (default: auto-selected)
      --vram <VRAM>                    Available VRAM in GB [default: 16.0]
      --plan                           Plan mode (estimate only)
  -d, --data <FILE>                    Training data file (JSONL format)
  -o, --output <OUTPUT>                Output path (adapter dir or merged model)
      --adapter <ADAPTER>              Adapter path for merge mode
      --merge                          Merge adapter into base model
      --epochs <EPOCHS>                Training epochs [default: 3]
      --learning-rate <LEARNING_RATE>  Learning rate [default: 0.0002]
      --model-size <SIZE>              Model size for planning (e.g., "7B", "1.5B")
      --task <TASK>                    Fine-tuning task: classify (sequence classification)
      --num-classes <NUM_CLASSES>      Number of classes for classification task [default: 5]
      --checkpoint-format <FORMAT>     Output format for checkpoints: apr, safetensors, or both
                                       (comma-separated) [default: apr,safetensors]
      --oversample                     Oversample minority classes to match majority (for imbalanced
                                       datasets)
      --max-seq-len <LEN>              Maximum sequence length for GPU buffer allocation (lower =
                                       less VRAM)
      --quantize-nf4                   Quantize frozen weights to NF4 (4-bit) for QLoRA training
                                       (~8x VRAM savings)
      --gpus <INDICES>                 GPU indices for data-parallel training (e.g., "0,1" for dual
                                       GPU)
      --gpu-backend <GPU_BACKEND>      GPU backend selection: auto, cuda, wgpu [default: auto]
      --role <ROLE>                    Distributed training role: coordinator or worker
      --bind <ADDR>                    Address to bind (coordinator) or connect to (worker)
      --coordinator <ADDR>             Coordinator address for worker nodes (e.g., "intel:9000")
      --expect-workers <N>             Expected number of workers (coordinator only)
      --wait-gpu <SECS>                Wait for VRAM availability before training (timeout in
                                       seconds, 0 = no wait) [default: 0]
      --adapters <DATA:CHECKPOINT>     Multi-adapter training: data:checkpoint pairs (GPU-SHARE
                                       Phase 2) Format: --adapters
                                       data/corpus-a.jsonl:checkpoints/adapter-a Can be specified
                                       multiple times for concurrent adapter training
      --adapters-config <FILE>         Multi-adapter config file: TOML with [[adapter]] entries
                                       (GPU-SHARE §2.4)
      --experimental-mps               Enable experimental CUDA MPS for concurrent GPU sharing
                                       (GPU-SHARE §1.5). WARNING: A GPU fault in any MPS client will
                                       crash ALL clients on that GPU
      --gpu-share <PCT>                MPS thread percentage (1-100). Controls SM allocation per
                                       process. Only effective with --experimental-mps. Default: 50
                                       [default: 50]
      --profile                        PMAT-486: Enable StepProfiler for per-phase wall-clock timing
      --json                           Output as JSON
  -v, --verbose                        Verbose output
  -q, --quiet                          Quiet mode (errors only)
      --offline                        Disable network access (Sovereign AI compliance, Section 9)
      --skip-contract                  Skip tensor contract validation (PMAT-237: use with
                                       diagnostic tooling)
  -h, --help                           Print help
  -V, --version                        Print version
```
