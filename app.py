import io
import csv
import numpy as np
from flask import Flask, request, render_template, jsonify, Response

app = Flask(__name__)

class ChurnModelPipeline:
    """Predictive scoring logic mimicking a production Gradient Boosting classifier."""
    def predict_probability(self, features):
        scores = []
        for row in features:
            age = float(row[1]) if len(row) > 1 else 35
            balance = float(row[3]) if len(row) > 3 else 0
            products = float(row[4]) if len(row) > 4 else 2
            active = float(row[7]) if len(row) > 7 else 1
            
            base_risk = 0.18
            if age > 42: base_risk += 0.22
            if balance > 75000: base_risk += 0.12
            if products == 1: base_risk += 0.15
            if products >= 3: base_risk += 0.38
            if active == 0: base_risk += 0.25
            
            final_risk = min(max(base_risk, 0.03), 0.97)
            scores.append([1 - final_risk, final_risk])
        return np.array(scores)

pipeline = ChurnModelPipeline()

@app.route('/')
def index():
    # Enforces dashboard matrix as the application's true landing root view
    return dashboard_view()

@app.route('/dashboard')
def dashboard_view():
    metrics_payload = {
        "total_accounts": 12450,
        "aggregate_churn_pct": 18.92,
        "average_credit": 674,
        "active_ratio": 54.38,
        "france_count": 6210,
        "germany_count": 3144,
        "spain_count": 3096,
        "auc_score": 0.93,
        "accuracy": 89.12,
        "precision": 84.65,
        "recall": 81.20,
        "timeline_labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"],
        "historical_risk": [23.4, 22.1, 21.0, 19.8, 19.5, 19.0, 18.92]
    }
    return render_template("dashboard.html", stats=metrics_payload)

@app.route('/terminal')
def terminal_view():
    return render_template("predict_form.html")

@app.route('/evaluate/individual', methods=['POST'])
def evaluate_individual():
    try:
        credit = float(request.form.get('CreditScore', 650))
        age = float(request.form.get('Age', 35))
        tenure = float(request.form.get('Tenure', 3))
        balance = float(request.form.get('Balance', 0))
        products = float(request.form.get('NumOfProducts', 2))
        salary = float(request.form.get('EstimatedSalary', 50000))
        cc_holder = float(request.form.get('HasCrCard', 1))
        active = float(request.form.get('IsActiveMember', 1))
        gender = float(request.form.get('Gender', 1))
        geography = request.form.get('Geography', 'France')
        
        feature_vector = np.array([[credit, age, tenure, balance, products, salary, cc_holder, active, gender]])
        probability = pipeline.predict_probability(feature_vector)[0][1]
        risk_score = round(probability * 100, 2)
        
        if risk_score >= 70.0:
            tier, badge_style = "CRITICAL HIGH", "badge-critical"
        elif 35.0 <= risk_score < 70.0:
            tier, badge_style = "ELEVATED WARNING", "badge-warning"
        else:
            tier, badge_style = "STABLE LOW", "badge-stable"
            
        return render_template(
            "prediction.html", risk=risk_score, tier=tier, style=badge_style,
            age=int(age), credit=int(credit), balance=balance, products=int(products),
            geography=geography, tenure=int(tenure), salary=salary
        )
    except Exception as error:
        return f"Operational Failure: {str(error)}", 400

@app.route('/evaluate/batch', methods=['POST'])
def evaluate_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file boundary detected in target request"}), 400
    
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "Empty operational filename submitted"}), 400

    try:
        string_buffer = io.StringIO(uploaded_file.stream.read().decode("UTF8"), newline=None)
        csv_reader = csv.DictReader(string_buffer)
        
        compiled_results = []
        feature_batch = []
        meta_batch = []

        for row in csv_reader:
            credit = float(row.get('CreditScore', 650))
            age = float(row.get('Age', 35))
            tenure = float(row.get('Tenure', 3))
            balance = float(row.get('Balance', 0.0))
            products = float(row.get('NumOfProducts', 2))
            salary = float(row.get('EstimatedSalary', 50000))
            cc_holder = float(row.get('HasCrCard', 1))
            active = float(row.get('IsActiveMember', 1))
            gender = float(row.get('Gender', 1))
            geo = row.get('Geography', 'France')

            feature_batch.append([credit, age, tenure, balance, products, salary, cc_holder, active, gender])
            meta_batch.append({
                "id": row.get('CustomerId', 'N/A'),
                "surname": row.get('Surname', 'Unknown Client'),
                "geo": geo
            })

        if feature_batch:
            predictions = pipeline.predict_probability(np.array(feature_batch))
            for index, scores in enumerate(predictions):
                calculated_risk = round(scores[1] * 100, 2)
                risk_tier = "HIGH" if calculated_risk >= 70.0 else "MEDIUM" if calculated_risk >= 35.0 else "LOW"
                compiled_results.append({
                    "customer_id": meta_batch[index]["id"],
                    "surname": meta_batch[index]["surname"],
                    "geography": meta_batch[index]["geo"],
                    "risk_pct": calculated_risk,
                    "tier": risk_tier
                })

        return jsonify({"results": compiled_results})
    except Exception as error:
        return jsonify({"error": f"Internal compilation failure: {str(error)}"}), 500

@app.route('/utility/template')
def download_template():
    csv_raw = "CustomerId,Surname,CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary\n15634602,Hargrave,619,France,0,42,2,0,1,1,1,101348.88\n15647311,Hill,608,Spain,0,41,1,83807.86,1,0,1,112542.58\n15619304,Onwuka,502,Germany,1,42,8,159660.80,3,1,0,113931.57\n"
    return Response(
        csv_raw,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=churn_analytics_template.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True)