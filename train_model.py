import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # This is the magic tool that "saves" the brain!

# 1. Create a folder named 'models' if it doesn't exist
os.makedirs("models", exist_ok=True)

# 2. Load our final AI-ready dataset
print("🔄 Loading our prepared features...")
df = pd.read_csv("dataset/customer_churn_final.csv")

# 3. Separate our Features (X) from our Target (y)
X = df.drop(columns=['Exited'])
y = df['Exited']

# 4. Split into Train (80%) and Test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and Train our AI Brain (The Random Forest)
print("🧠 Training the Random Forest AI...")
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)
print("✅ Training complete!")

# 6. Time for the Final Exam!
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\n🎓 AI Accuracy Score: {accuracy:.2f}%")

# 7. SAVE THE BRAIN!
model_path = "models/churn_model.pkl"
joblib.dump(model, model_path)
print(f"💾 Success! Saved the trained AI brain to: {model_path}")