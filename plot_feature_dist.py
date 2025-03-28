import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


columns = ['BalanceCheque', 'SavingsBalance', 'Mths_employ']

df1 = pd.read_csv("aligned_greman_samples_synthetic_instances.csv").dropna(subset=columns)
df2 = pd.read_csv("dataset/processed_greman_cleaned.csv").dropna(subset=columns)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

normalized1 = scaler1.fit_transform(df1[columns])
normalized2 = scaler2.fit_transform(df2[columns])

norm_df1 = pd.DataFrame(normalized1, columns=columns)
norm_df2 = pd.DataFrame(normalized2, columns=columns)

plt.figure(figsize=(10, 6))

line_handles = {} # legend

# solid lines only
for col in columns:
    counts, bin_edges = np.histogram(norm_df1[col], bins=50, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    line, = plt.plot(bin_centers, counts, label=f"{col} (fake)", linewidth=2)
    line_handles[col] = line

# dotted lines only
for col in columns:
    counts, bin_edges = np.histogram(norm_df2[col], bins=50, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    plt.plot(bin_centers, counts, linestyle='dotted', color=line_handles[col].get_color(), label=f"{col} (real)", linewidth=2)

plt.title("Normalized Distributions from synthetic and real data")
plt.xlabel("Normalized Value (0–1)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
