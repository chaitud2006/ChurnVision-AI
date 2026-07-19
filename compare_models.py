import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 1. Load the features
print("🔄 Loading data...")
df = pd.read_csv("dataset/customer_churn_final.csv")
X = df.drop(columns=['Exited'])
y = df['Exited']

# 2. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize our three contenders!
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
}

# 4. Train and evaluate each one
results = {}
print("\n🏁 Starting the AI Race...")

for name, model in models.items():
    print(f"🏋️ Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions) * 100
    results[name] = acc
    print(f"➡️ {name} got an accuracy of: {acc:.2f}%\n")

# 5. Declare the winner!
print("--- 🏆 FINAL RESULTS ---")
for name, score in results.items():
    print(f"• {name}: {score:.2f}%")

best_model = max(results, key=results.get)
print(f"\n🥇 The winner is: {best_model} with {results[best_model]:.2f}% accuracy!")