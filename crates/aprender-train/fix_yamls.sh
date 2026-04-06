#!/bin/bash
mkdir -p examples/yaml_fixed

# ... (previous content) ...
# I will overwrite the file with ALL yamls including previous ones to be safe.

cat <<EOF > examples/yaml_fixed/mnist_cpu.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001

training:
  epochs: 1
EOF

cat <<EOF > examples/yaml_fixed/csv_data.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.csv"
  batch_size: 32
  auto_infer_types: true

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/parquet_data.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/deterministic.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001

training:
  epochs: 1
  output_dir: "outputs/deterministic"
EOF

cat <<EOF > examples/yaml_fixed/multiworker.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

# Section C
cat <<EOF > examples/yaml_fixed/llama2_mock.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 4

optimizer:
  name: "adam"
  lr: 0.001

training:
  epochs: 1
EOF

cat <<EOF > examples/yaml_fixed/custom_arch.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/dropout.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001

lora:
  rank: 8
  alpha: 16
  target_modules: [q_proj]
  dropout: 0.5

training:
  epochs: 1
EOF

cat <<EOF > examples/yaml_fixed/grad_clip.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001

training:
  epochs: 1
  grad_clip: 1.0
EOF

# Section D
cat <<EOF > examples/yaml_fixed/lora.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001

lora:
  rank: 8
  alpha: 16
  target_modules: [q_proj]
EOF

cat <<EOF > examples/yaml_fixed/qlora.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001

lora:
  rank: 64
  alpha: 16
  target_modules: [q_proj]

quantize:
  bits: 4
  symmetric: true
  per_channel: true
EOF

cat <<EOF > examples/yaml_fixed/distillation.yaml
# 'entrenar distill' command doesn't exist, but creating yaml anyway
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/grad_accum.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 4

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/lr_schedule.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.0001

training:
  epochs: 1
  lr_scheduler: cosine
  warmup_steps: 10
EOF

# Section E
cat <<EOF > examples/yaml_fixed/andon.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/outlier.yaml
# 'entrenar inspect' doesn't exist
model:
  path: "model.gguf"
  layers: []
data:
  train: "data/train.parquet"
  batch_size: 32
optimizer:
  name: "adam"
  lr: 0.001
EOF

# Section F
cat <<EOF > examples/yaml_fixed/checkpoint.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001

training:
  epochs: 1
  save_interval: 1
  output_dir: "checkpoints/ckpt"
EOF

cat <<EOF > examples/yaml_fixed/config.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/long_run.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001

training:
  epochs: 1000
EOF

cat <<EOF > examples/yaml_fixed/locked.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

# Section G
cat <<EOF > examples/yaml_fixed/latency.yaml
# 'entrenar bench' doesn't exist
model:
  path: "model.gguf"
  layers: []
data:
  train: "data/train.parquet"
  batch_size: 32
optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/json_output.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

# Section H
cat <<EOF > examples/yaml_fixed/dp.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/bias.yaml
# 'entrenar audit' doesn't exist
model:
  path: "model.gguf"
  layers: []
data:
  train: "data/train.parquet"
  batch_size: 32
optimizer:
  name: "adam"
  lr: 0.001
EOF

# Section I
cat <<EOF > examples/yaml_fixed/session.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

# Section J
cat <<EOF > examples/yaml_fixed/soak.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/drift.yaml
# 'entrenar monitor' doesn't exist
model:
  path: "model.gguf"
  layers: []
data:
  train: "data/train.parquet"
  batch_size: 32
optimizer:
  name: "adam"
  lr: 0.001
EOF

cat <<EOF > examples/yaml_fixed/release.yaml
model:
  path: "model.gguf"
  layers: []

data:
  train: "data/train.parquet"
  batch_size: 32

optimizer:
  name: "adam"
  lr: 0.001
EOF
