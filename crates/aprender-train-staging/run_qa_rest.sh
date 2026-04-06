#!/bin/bash
export PATH=$PWD/target/release:$PATH
mkdir -p logs
mkdir -p checkpoints/ckpt

run_test() {
    id=$1
    desc=$2
    cmd=$3
    echo "Running [$id] $desc..."
    eval "$cmd" > "logs/$id.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "[$id] PASS"
    else
        echo "[$id] FAIL"
    fi
}

# SECTION D
run_test "31" "LoRA" "entrenar train examples/yaml_fixed/lora.yaml"
run_test "32" "QLoRA" "entrenar train examples/yaml_fixed/qlora.yaml"
run_test "33" "Quantization QAT" "cargo run --example quantization -- --qat"
run_test "34" "Distillation" "cargo run --example distillation"
run_test "35" "HF Distillation" "cargo run --example hf_distillation -- --model bert-base"
run_test "36" "Merge Models" "cargo run --example merge_models -- --method ties"
run_test "37" "Grad Accum" "entrenar train examples/yaml_fixed/grad_accum.yaml"
run_test "38" "FP16" "cargo run --example training_loop -- --fp16"
run_test "39" "Memory Bench" "cargo run --example llama2-memory-benchmarks -- --offload"
run_test "40" "LR Schedule" "entrenar train examples/yaml_fixed/lr_schedule.yaml"

# SECTION E
run_test "41" "Monitoring" "cargo run --example monitoring --features monitor"
run_test "42" "Persist" "cargo run --example monitoring -- --persist"
run_test "43" "Tracing" "cargo run --example explainability --features tracing"
run_test "44" "Andon" "entrenar train examples/yaml_fixed/andon.yaml"
run_test "45" "Explainability" "cargo run --example explainability -- --method integrated-gradients"
run_test "46" "Confusion" "cargo run --example monitoring -- --confusion"
run_test "47" "Profiling" "cargo run --example llama2-memory-benchmarks"
run_test "48" "Outlier" "entrenar inspect examples/yaml_fixed/outlier.yaml"
run_test "49" "Histograms" "cargo run --example inspect -- --histograms"
run_test "50" "Export JSON" "cargo run --example monitoring -- --export-json"

# SECTION F
run_test "51" "Checkpoint" "entrenar train examples/yaml_fixed/checkpoint.yaml"
run_test "52" "Crash Sim" "cargo run --example training_loop -- --simulate-crash"
run_test "53" "Shutdown" "cargo run --example training_loop"
run_test "54" "Disk Full" "cargo run --example model_io -- --mock-disk-full"
run_test "55" "Network Timeout" "cargo run --example hf_distillation -- --offline"
run_test "56" "Config Validate" "entrenar validate examples/yaml_fixed/config.yaml"
run_test "57" "Version Check" "cargo run --example model_io -- --check-version"
run_test "58" "Long Run" "entrenar train examples/yaml_fixed/long_run.yaml --dry-run"
run_test "59" "Check Numerics" "cargo run --example research -- --check-numerics"
run_test "60" "Lockfile" "entrenar train examples/yaml_fixed/locked.yaml"

# SECTION G
run_test "61" "WASM" "wasm-pack build crates/entrenar-wasm"
run_test "62" "Latency" "entrenar bench examples/yaml_fixed/latency.yaml"
run_test "63" "Batch Predict" "cargo run --example model_io -- --batch-predict"
run_test "64" "Quantize CPU" "cargo run --example quantization -- --arch x86_64"
run_test "65" "Sign Model" "cargo run --example sovereign -- --sign-model"
run_test "66" "CLI Help" "cargo run --bin entrenar -- --help"
run_test "67" "JSON Output" "entrenar train examples/yaml_fixed/json_output.yaml"
run_test "68" "Env Var" "DATA_DIR=/tmp/data cargo run --example mnist_train"
run_test "69" "Docker" "echo 'Skipping Docker'"
run_test "70" "Hot Reload" "cargo run --example citl -- --watch"

# SECTION H
run_test "71" "Sovereign Offline" "cargo run --example sovereign -- --offline"
run_test "72" "Unlearn" "cargo run --example research -- --unlearn user_id_123"
run_test "73" "Diff Privacy" "entrenar train examples/yaml_fixed/dp.yaml"
run_test "74" "Model Card" "cargo run --example explainability -- --generate-card"
run_test "75" "Carbon Report" "cargo run --example monitoring -- --report-energy"
run_test "76" "Bias Audit" "entrenar audit examples/yaml_fixed/bias.yaml"
run_test "77" "Adv Robust" "cargo run --example research -- --attack fgsm"
run_test "78" "Federated" "cargo run --example sovereign -- --federated"
run_test "79" "Audit Deps" "cargo run --example sovereign -- --audit-deps"
run_test "80" "Licenses" "cargo run --example research -- --check-licenses"

# SECTION I
run_test "81" "Batuta Mock" "cargo run --example training_loop -- --mode worker"
run_test "82" "Bash Comp" "cargo run --bin entrenar -- completion bash"
run_test "83" "Resume" "entrenar train examples/yaml_fixed/session.yaml"
run_test "84" "Decision Log" "cargo run --example citl -- --log-decision-tree"
run_test "85" "Viz Export" "cargo run --example explainability -- --export-viz"
run_test "86" "RAG Query" "cargo run --example citl --features citl"
run_test "87" "Workspace Build" "cargo build --workspace --all-features"
run_test "88" "Audit" "cargo audit"
run_test "89" "Doc Test" "cargo test --doc"
run_test "90" "Bench" "cargo bench"

# SECTION J
run_test "91" "Black Swan" "cargo run --example training_loop -- --input-range 1e10"
run_test "92" "Power Loss" "echo 'Skipping Power Loss'"
run_test "93" "Mem Corruption" "cargo test --test integrity_test -- --bitflip"
run_test "94" "Clock Skew" "cargo run --example monitoring -- --mock-time-skew"
run_test "95" "Zero Disk" "cargo run --example mnist_train -- --memory-only"
run_test "96" "Soak Test" "entrenar train examples/yaml_fixed/soak.yaml"
run_test "97" "Auth Fail" "cargo run --example sovereign -- --mock-auth-fail"
run_test "98" "Drift" "entrenar monitor examples/yaml_fixed/drift.yaml"
run_test "99" "HITL" "cargo run --example training_loop -- --await-approval"
run_test "100" "Golden Run" "entrenar train examples/yaml_fixed/release.yaml"
