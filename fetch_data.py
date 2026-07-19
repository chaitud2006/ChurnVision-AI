import urllib.request
import os

# 1. Make sure the dataset folder exists
os.makedirs("dataset", exist_ok=True)

# 2. This is the direct, raw link to the Bank Churn dataset on GitHub
url = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv"
save_path = "dataset/customer_churn.csv"

print("📡 Connecting to the internet to download your dataset...")

try:
    # 3. Pull the file and save it!
    urllib.request.urlretrieve(url, save_path)
    print("🎉 YAHOO! The dataset has been successfully downloaded!")
    print(f"📁 It is now saved right here: {save_path}")
except Exception as e:
    print("❌ Oh no, something went wrong with the connection.")
    print(f"Error details: {e}")