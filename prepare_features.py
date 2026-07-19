import pandas as pd

# 1. Load our cleaned dataset
print("🔄 Loading clean data...")
df = pd.read_csv("dataset/customer_churn_clean.csv")

# 2. Translate Gender to numbers (Female = 0, Male = 1)
print("⚧ Encoding Gender...")
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

# 3. Translate Geography using One-Hot Encoding
print("🌍 Encoding Geography countries...")
# This automatically creates separate columns for France, Germany, and Spain!
df = pd.get_dummies(df, columns=['Geography'], dtype=int)

# 4. Let's look at our new mathematical dataset columns
print("\n--- Our New AI-Ready Columns ---")
print(list(df.columns))

# 5. Save this final file!
final_file_path = "dataset/customer_churn_final.csv"
df.to_csv(final_file_path, index=False)
print(f"\n🎉 SUCCESS! Week 1 complete! Final dataset saved at: {final_file_path}")