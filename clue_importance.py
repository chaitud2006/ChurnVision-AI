import pandas as pd
import joblib

# 1. Load our champion model and the final dataset columns
print("🔄 Loading our champion AI brain...")
model = joblib.load("models/churn_model.pkl")

df = pd.read_csv("dataset/customer_churn_final.csv")
# Get the list of clues (everything except the 'Exited' column)
clue_names = df.drop(columns=['Exited']).columns

# 2. Get importance scores from the Random Forest
# This is a built-in function that measures how much the model used each column
importances = model.feature_importances_

# 3. Pair each clue name with its score and sort them from highest to lowest
clue_scores = pd.DataFrame({
    'Clue': clue_names,
    'Importance (%)': importances * 100
}).sort_values(by='Importance (%)', ascending=False)

# 4. Print the results beautifully!
print("\n--- 🔍 THE MOST IMPORTANT CLUES FOR CHURN ---")
print("Here is what our AI looks at when making a decision:\n")

for index, row in clue_scores.iterrows():
    print(f"📍 {row['Clue']}: {row['Importance (%)']:.2f}%")

print("\n💡 Fact: The clue with the highest percentage is the #1 driver of customer churn!")