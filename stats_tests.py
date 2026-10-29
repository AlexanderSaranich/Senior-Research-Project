import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Import the CSV datasets
df1 = pd.read_csv('data/fake_survey_dataset_1.csv')
df2 = pd.read_csv('data/fake_survey_dataset_2.csv')

# 2. Find the averages (mean) and standard deviations for each field in each dataframe
print("=== DATASET 1: MEANS & STANDARD DEVIATIONS ===")
stats_df1 = pd.DataFrame({
    'Mean': df1.mean(),
    'StdDev': df1.std()
})
print(stats_df1)

print("\n=== DATASET 2: MEANS & STANDARD DEVIATIONS ===")
stats_df2 = pd.DataFrame({
    'Mean': df2.mean(),
    'StdDev': df2.std()
})
print(stats_df2)

# 3. Find the correlation between different fields
print("\n=== DATASET 1: CORRELATION MATRIX ===")
corr_df1 = df1.corr()
print(corr_df1)

print("\n=== DATASET 2: CORRELATION MATRIX ===")
corr_df2 = df2.corr()
print(corr_df2)

# 4. Find the difference in means between Group 1 and Group 2
print("\n=== DIFFERENCE IN MEANS (Dataset 1 - Dataset 2) ===")
mean_diff = df1.mean() - df2.mean()
print(mean_diff)

# Bonus: Independent samples t-test to determine if mean differences are statistically significant
print("\n=== INDEPENDENT T-TEST RESULTS (Dataset 1 vs Dataset 2) ===")
ttest_results = []
for col in df1.columns:
    t_stat, p_val = stats.ttest_ind(df1[col], df2[col], equal_var=False)
    ttest_results.append({
        'Field': col,
        'Mean_Diff': df1[col].mean() - df2[col].mean(),
        't_statistic': t_stat,
        'p_value': p_val
    })

ttest_df = pd.DataFrame(ttest_results).set_index('Field')
print(ttest_df)

ttest_results = []
for col in df1.columns:
    t_stat, p_val = stats.ttest_ind(df1[col], df2[col], equal_var=False)
    mean1 = df1[col].mean()
    mean2 = df2[col].mean()
    std1 = df1[col].std()
    std2 = df2[col].std()

    if p_val < 0.20:
        ttest_results.append({
            'Field': col,
            'Dataset 1 Mean': mean1,
            'Dataset 2 Mean': mean2,
            'Dataset 1 Std': std1,
            'Dataset 2 Std': std2,
            'p_value': p_val,
        })

filtered_df = pd.DataFrame(ttest_results)
print("=== FIELDS WITH P-VALUE < 0.20 ===")
print(filtered_df[['Field', 'Dataset 1 Mean', 'Dataset 2 Mean', 'p_value']])

# 3. Create a bar chart for each significant/marginal field
sns.set_theme(style='whitegrid')

for idx, row in filtered_df.iterrows():
    field = row['Field']

    plt.figure(figsize=(6, 4))
    means = [row['Dataset 1 Mean'], row['Dataset 2 Mean']]
    stds = [row['Dataset 1 Std'], row['Dataset 2 Std']]
    groups = ['Group 1\n(Dataset 1)', 'Group 2\n(Dataset 2)']

    # Plot bars with standard error / standard deviation error bars
    bars = plt.bar(
        groups,
        means,
        yerr=stds,
        capsize=5,
        color=['#3498db', '#e74c3c'],
        alpha=0.85,
    )

    # Title and Labels
    plt.title(f'{field} (p-value = {row["p_value"]:.4f})', fontsize=12, pad=10)
    plt.ylabel('Mean Rating (1-7 Scale)', fontsize=10)
    plt.ylim(1, 7)

    # Add data values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval / 2,
            f'{yval:.2f}',
            ha='center',
            va='center',
            color='white',
            fontweight='bold',
            fontsize=11,
        )

    plt.tight_layout()
    plt.show()