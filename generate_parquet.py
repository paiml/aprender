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
