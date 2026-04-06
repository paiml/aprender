#!/bin/bash
mkdir -p data

# Create train.csv
echo "feature1,feature2,label" > data/train.csv
for i in {1..100}; do
  echo "$((RANDOM % 100)),$((RANDOM % 100)),$((RANDOM % 2))" >> data/train.csv
done

# Create train.parquet using uv to handle dependencies
cat <<EOF > generate_parquet.py
import pandas as pd
import numpy as np

# Create data where features is a list of floats (simulating a vector)
# The example parquet_data.yaml expects columns [features, label]
data = {
    'features': [np.random.rand(10).tolist() for _ in range(100)],
    'label': np.random.randint(0, 2, 100)
}
df = pd.DataFrame(data)
df.to_parquet('data/train.parquet')
print("Successfully created data/train.parquet")
EOF

uv run --with pandas --with pyarrow --with numpy generate_parquet.py