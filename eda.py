import pandas as pd
import os

# 1. Load our cleaned dataset
df = pd.read_csv("dataset/customer_churn_clean.csv")
print("📊 Clean dataset loaded! Ready for detective work.\n")

# 2. Question 1: What percentage of people actually leave the bank?
# 'Exited' is 1 if they left, and 0 if they stayed.
total_customers = len(df)
left_customers = df['Exited'].sum()
stayed_customers = total_customers - left_customers

churn_rate = (left_customers / total_customers) * 100

print("--- 🏦 OVERALL BANK CHURN STATS ---")
print(f"Total Customers Analyzed: {total_customers}")
print(f"Customers who stayed (0): {stayed_customers}")
print(f"Customers who left (1): {left_customers}")
print(f"Overall Churn Rate: {churn_rate:.2f}%\n")

# 3. Question 2: Does Age affect churn? Let's find the average age!
avg_age_stayed = df[df['Exited'] == 0]['Age'].mean()
avg_age_left = df[df['Exited'] == 1]['Age'].mean()

print("--- 👴 AGE PATTERNS ---")
print(f"Average age of customers who STAYED: {avg_age_stayed:.1f} years old")
print(f"Average age of customers who LEFT: {avg_age_left:.1f} years old")
if avg_age_left > avg_age_stayed:
    print("💡 Pattern Detected: Older customers seem more likely to leave the bank!")
else:
    print("💡 Pattern Detected: Younger customers seem more likely to leave the bank!")

# 4. Question 3: Does country play a role? 
print("\n--- 🌍 GEOGRAPHY PATTERNS (Where do they live?) ---")
country_churn = df.groupby('Geography')['Exited'].mean() * 100
for country, rate in country_churn.items():
    print(f"Churn rate in {country}: {rate:.2f}%")