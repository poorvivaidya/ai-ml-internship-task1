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

## Interview Questions

**1. What are the different types of missing data?**
- MCAR (Missing Completely at Random): missingness has no relation to any other data.
- MAR (Missing at Random): missingness relates to other observed variables, not the missing value itself.
- MNAR (Missing Not at Random): missingness relates to the value that is missing (e.g., people with high income not disclosing it).

**2. How do you handle categorical variables?**
By encoding them into numbers — using Label Encoding (assigns each category an integer, good for binary/ordinal data) or One-Hot Encoding (creates a separate binary column per category, good for nominal data with no order).

**3. What is the difference between normalization and standardization?**
Normalization (Min-Max scaling) rescales values into a fixed range, usually [0,1]. Standardization (Z-score scaling) rescales data to have mean = 0 and standard deviation = 1. Standardization is less affected by outliers and works better when data isn't bounded.

**4. How do you detect outliers?**
Using boxplots (visual method based on IQR), the IQR rule (values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are outliers), or Z-scores (values with |z| > 3 are often flagged).

**5. Why is preprocessing important in ML?**
Raw data is often incomplete, inconsistent, or on different scales. Models trained on unclean data learn noise or biased patterns, leading to poor accuracy. Preprocessing ensures data quality and puts features on comparable scales so the model learns correctly.

**6. What is one-hot encoding vs label encoding?**
Label Encoding assigns each category an integer (0, 1, 2...), which implies an order — best for ordinal or binary categories. One-Hot Encoding creates a separate 0/1 column per category, avoiding any false ordering — best for nominal categories.

**7. How do you handle data imbalance?**
Oversampling the minority class (e.g., SMOTE), undersampling the majority class, using class-weighted loss functions, or choosing evaluation metrics like precision, recall, F1, and ROC-AUC instead of plain accuracy, which can be misleading on imbalanced data.

**8. Can preprocessing affect model accuracy?**
Yes, significantly. Poor handling of missing values, unencoded categoricals, unscaled features, or leftover outliers can bias the model or slow its learning — directly lowering accuracy. Good preprocessing is often the biggest lever for model performance.