#!/bin/bash
export PATH=$PWD/target/release:$PATH
mkdir -p logs

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
        # echo "--- Last 5 lines of logs/$id.log ---"
        # cat "logs/$id.log" | tail -n 5
        # echo "------------------------------------"
    fi
}

# SECTION B: CITL
# 11. CITL Index
run_test "11" "CITL Index" "cargo run --example citl --features citl -- --mode index"

# 12. Tarantula
run_test "12" "Tarantula" "cargo run --example citl --features citl -- --mode tarantula"

# 13. Decision Trace
run_test "13" "Decision Trace" "cargo run --example citl --features citl -- --trace-file execution.log"

# 14. Fix Suggestion (using citl example)
run_test "14" "Fix Suggestion" "cargo run --example citl --features citl -- examples/citl_suggest.yaml"

# 15. APR Persistence
run_test "15" "APR Persistence" "cargo run --example citl --features citl -- --save patterns.apr"

# 16. Oracle Prediction
run_test "16" "Oracle Prediction" "cargo run --example citl --features citl -- --predict-outcome source.rs"

# 17. Workspace
run_test "17" "Workspace" "cargo run --example citl --features citl -- examples/citl_workspace.yaml"

# 18. Watch
run_test "18" "Watch" "cargo run --example citl --features citl -- --watch ./src"

# 19. Feedback Loop
run_test "19" "Feedback Loop" "cargo run --example citl --features citl -- --simulate-feedback 100"

# 20. Version Compat
run_test "20" "Version Compat" "cargo run --example citl --features citl -- --target-version 1.75.0"


# SECTION C: Model Architecture
# 21. Llama2 Mock
run_test "21" "Llama2 Mock" "entrenar train examples/yaml_fixed/llama2_mock.yaml --dry-run"

# 22. XOR
run_test "22" "XOR" "cargo run --example training_loop -- --xor"

# 23. Custom Arch
run_test "23" "Custom Arch" "entrenar train examples/yaml_fixed/custom_arch.yaml"

# 24. Activation Sweep
run_test "24" "Activation Sweep" "cargo run --example research -- --sweep activations"

# 25. Dropout
run_test "25" "Dropout" "entrenar train examples/yaml_fixed/dropout.yaml"

# 26. Batch Norm (Inspect) - EXPECT FAIL
run_test "26" "Batch Norm" "cargo run --example inspect -- --layer batch_norm"

# 27. Init Audit
run_test "27" "Init Audit" "cargo run --example research -- --check-init"

# 28. Grad Clip
run_test "28" "Grad Clip" "entrenar train examples/yaml_fixed/grad_clip.yaml"

# 29. Resnet Grads
run_test "29" "Resnet Grads" "cargo run --example research -- --check-gradients resnet"

# 30. Dynamic Search
run_test "30" "Dynamic Search" "cargo run --example research -- --prune-threshold 0.01"
