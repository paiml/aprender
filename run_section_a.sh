#!/bin/bash
export PATH=$PWD/target/release:$PATH
mkdir -p logs
mkdir -p checkpoints
mkdir -p outputs/deterministic

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
        echo "--- Last 5 lines of logs/$id.log ---"
        cat "logs/$id.log" | tail -n 5
        echo "------------------------------------"
    fi
}

# 1. MNIST Baseline (CPU)
run_test "01" "MNIST Baseline" "entrenar train examples/yaml_fixed/mnist_cpu.yaml"

# 2. MNIST GPU
run_test "02" "MNIST GPU" "cargo run --example mnist_train_gpu --features gpu --release"

# 3. Custom CSV
run_test "03" "Custom CSV" "entrenar train examples/yaml_fixed/csv_data.yaml"

# 4. Parquet
run_test "04" "Parquet" "entrenar train examples/yaml_fixed/parquet_data.yaml"

# 5. Deterministic
run_test "05" "Deterministic" "entrenar train examples/yaml_fixed/deterministic.yaml"

# 6. Stratified Splitting
run_test "06" "Stratified Split" "cargo run --example research -- --check-split --target label"

# 7. Corrupt Data
echo "Running [07] Corrupt Data..."
cargo run --example training_loop -- --inject-nan-input > logs/07.log 2>&1
if [ $? -ne 0 ]; then
    echo "[07] PASS (Correctly halted)"
else
    echo "[07] FAIL (Did not halt - Example likely ignores flag)"
fi

# 8. Large Dataset
run_test "08" "Large Dataset" "cargo run --example model_io -- --stream-data --batch-size 128"

# 9. Multi-Worker
run_test "09" "Multi-Worker" "entrenar train examples/yaml_fixed/multiworker.yaml"

# 10. Data Poisoning (Running test instead of example)
run_test "10" "Data Poisoning" "cargo test --test integrity_test"