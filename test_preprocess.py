# test_preprocess.py

from src.preprocess import load_and_clean_data

# ✅ Path to your dataset
csv_path = "data/online_retail.csv"

# ✅ Load and clean data
df_cleaned = load_and_clean_data(csv_path)

# ✅ Preview the result
print("\n🔹 Sample Cleaned Data:\n")
print(df_cleaned.head())

print("\n🔹 Data Info:\n")
print(df_cleaned.info())
