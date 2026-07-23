import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

sns.set(style="whitegrid")

df = sns.load_dataset("titanic")   # pulls the Titanic dataset built into seaborn
print(df.shape)
df.info()
print(df.isnull().sum())

# Drop columns that duplicate info or have too many nulls
df = df.drop(columns=["deck", "embark_town", "alive", "class", "who", "adult_male"])

# age -> fill with median (numeric, robust to outliers/skew — better than mean here)
df["age"] = df["age"].fillna(df["age"].median())

# embarked -> fill with mode (most common category, since it's categorical)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

print(df.isnull().sum())

# Label Encoding for 'sex' — it's binary (male/female), so this is fine
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])   # female=0, male=1

# One-Hot Encoding for 'embarked' — it has 3 categories (S, C, Q) with no natural order
df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

# 'alone' is currently True/False (boolean) — convert to 0/1
df["alone"] = df["alone"].astype(int)

print(df.columns.tolist())
print(df.head())

num_cols = ["age", "fare", "sibsp", "parch"]

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print(df[num_cols].head())
print(df[num_cols].describe())

# Boxplots BEFORE removing outliers
fig, axes = plt.subplots(1, len(num_cols), figsize=(16, 4))
for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f"Before: {col}")
plt.tight_layout()
plt.savefig("boxplots_before.png")
plt.close()

def remove_outliers_iqr(data, cols):
    data = data.copy()
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

df_clean = remove_outliers_iqr(df, num_cols)
print(f"Rows before: {len(df)}")
print(f"Rows after: {len(df_clean)}")

# Boxplots AFTER removing outliers
fig, axes = plt.subplots(1, len(num_cols), figsize=(16, 4))
for i, col in enumerate(num_cols):
    sns.boxplot(y=df_clean[col], ax=axes[i])
    axes[i].set_title(f"After: {col}")
plt.tight_layout()
plt.savefig("boxplots_after.png")
plt.close()

# Save cleaned dataset
df_clean.to_csv("titanic_cleaned.csv", index=False)
print("Saved! Final shape:", df_clean.shape)