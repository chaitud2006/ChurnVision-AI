import pandas as pd

# 1. Load the dataset
print("🔄 Loading your new bank dataset...")
df = pd.read_csv("dataset/customer_churn.csv")
print(f"✅ Success! Loaded {len(df)} customer records.")

# 2. Let's see what columns we actually have now
print("\n--- Columns in our Bank Dataset ---")
print(list(df.columns))

# 3. Drop columns that are completely useless for AI training
# RowNumber, CustomerId, and Surname are personal tags, not patterns!
print("\n🧹 Removing columns that aren't useful (RowNumber, CustomerId, Surname)...")
df_cleaned = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# 4. Check for any missing (empty) values
missing_data = df_cleaned.isnull().sum()
print("\n🔍 Checking for missing values in each column:")
print(missing_data)

# 5. Save this clean, simplified version
cleaned_file_path = "dataset/customer_churn_clean.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)
print(f"\n🎉 Success! Cleaned dataset saved at: {cleaned_file_path}")