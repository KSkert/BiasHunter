import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/processed_law_school_cleaned.csv")

sensitive_features = ['male', 'race']
non_sensitive_cols = [col for col in df.columns if col not in sensitive_features]

correlations = []

for i in range(0, len(df), 2):
    if i + 1 >= len(df):
        break  # incomplete pair

    row1 = df.iloc[i]
    row2 = df.iloc[i + 1]

    # Only consider pairs differing in race
    if row1['race'] == row2['race']:
        continue

    # Compute Spearman correlation
    corr, _ = spearmanr(row1[non_sensitive_cols], row2[non_sensitive_cols])
    correlations.append(corr)

print(f"Number of race-differing pairs: {len(correlations)}")
print(f"Mean Spearman correlation: {pd.Series(correlations).mean():.3f}")
print(f"Min: {min(correlations):.3f}, Max: {max(correlations):.3f}")

plt.hist(correlations, bins=20, edgecolor='black')
plt.title("Spearman Correlation of Real Pairs (Differing Only in 'race')")
plt.xlabel("Spearman Correlation")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()