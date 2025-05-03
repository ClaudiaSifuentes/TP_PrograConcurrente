import numpy as np
import pandas as pd

# Generate synthetic data
rows, cols = 1000000, 4
data = np.random.rand(rows, cols)  # Decimal columns
target = np.random.randint(0, 2, size=(rows, 1))  # Binary target column

# Combine data and target
synthetic_data = np.hstack((data, target))

# Create a DataFrame
columns = [f'feature_{i+1}' for i in range(cols)] + ['target']
df = pd.DataFrame(synthetic_data, columns=columns)

# Save to CSV
df.to_csv('synthetic_data.csv', index=False)