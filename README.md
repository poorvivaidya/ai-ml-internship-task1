# Task 1: Data Cleaning & Preprocessing

**Dataset:** Titanic (loaded via seaborn.load_dataset("titanic"))
**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## What I did
1. Explored the dataset (shape, dtypes, nulls) and dropped redundant/sparse columns.
2. Filled missing `age` with median, `embarked` with mode.
3. Encoded `sex` with Label Encoding, `embarked` with One-Hot Encoding.
4. Standardized `age`, `fare`, `sibsp`, `parch` using StandardScaler.
5. Visualized outliers with boxplots and removed them using the IQR method.

## Results
- Original: 891 rows
- After cleaning: 577 rows
- Saved cleaned dataset to `titanic_cleaned.csv`