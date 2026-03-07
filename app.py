from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from config import MODEL_PATH, MEDIAN_DAYS
import os 

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
median_days = np.load(MEDIAN_DAYS)

def risk_level(prob):
    if prob < 0.3:
        return 'Low'
    elif prob < 0.7:
        return 'Medium'
    else:
        return 'High'
    
@app.route("/")
def home():
    return render_template('home.html')

from flask import request, redirect, url_for
import pandas as pd
import numpy as np

@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if request.method == "POST":

        data = {
            "age_of_driver": int(request.form["age_of_driver"]),
            "gender": request.form["gender"],
            "marital_status": request.form["marital_status"],
            "safety_rating": int(request.form["safety_rating"]),
            "annual_income": float(request.form["annual_income"]),
            "high_education": request.form["high_education"],
            "address_change": request.form["address_change"],
            "property_status": request.form["property_status"],
            "claim_date": request.form["claim_date"],
            "claim_day_of_week": request.form["claim_day_of_week"],
            "accident_site": request.form["accident_site"],
            "past_num_of_claims": int(request.form["past_num_of_claims"]),
            "witness_present": request.form["witness_present"],
            "liab_prct": float(request.form["liab_prct"]),
            "channel": request.form["channel"],
            "police_report": request.form["police_report"],
            "age_of_vehicle": int(request.form["age_of_vehicle"]),
            "vehicle_category": request.form["vehicle_category"],
            "vehicle_price": float(request.form["vehicle_price"]),
            "total_claim": float(request.form["total_claim"]),
            "injury_claim": float(request.form["injury_claim"]),
            "policy deductible": float(request.form["policy_deductible"]),
            "annual premium": float(request.form["annual_premium"]),
            "days open": float(request.form["days_open"]),
            "form defects": request.form["form_defects"]
        }

        df = pd.DataFrame([data])

        binary_cols = [
            "witness_present",
            "police_report",
            "form defects",
            "address_change",
            "high_education"
        ]

        for col in binary_cols:
            df[col] = df[col].map({"Yes": 1, "No": 0})

        df["claim_date"] = pd.to_datetime(df["claim_date"], errors="coerce")

        df["claim_month"] = df["claim_date"].dt.month
        df["claim_weekend"] = (df["claim_date"].dt.weekday >= 5).astype(int)

        df = df.drop(columns=["claim_date"])

        df["claim_to_vehicle"] = df["total_claim"] / df["vehicle_price"]
        df["injury_to_claim"] = df["injury_claim"] / df["total_claim"]
        df["income_to_claim"] = df["total_claim"] / df["annual_income"]
        df["deductible_to_claim"] = df["policy deductible"] / df["total_claim"]

        df["repeat_claim_flag"] = np.where(
            df["past_num_of_claims"] > 2, 1, 0
        )

        df["days_open_flag"] = np.where(
            df["days open"] > median_days, 1, 0
        )

        df["claim_severity_index"] = (
            df["total_claim"] / (df["policy deductible"] + 1)
        )

        df["high_claim_flag"] = np.where(
            df["total_claim"] > 50000, 1, 0
        )

        categorical_cols = [
            "gender",
            "marital_status",
            "property_status",
            "accident_site",
            "channel",
            "vehicle_category",
            "claim_day_of_week"
        ]

        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        train_df = pd.read_csv("data/processed_data.csv")
        train_columns = train_df.drop("Fraud", axis=1).columns

        df = df.reindex(columns=train_columns, fill_value=0)

        prob = float(model.predict_proba(df)[:,1][0])

        if prob > 0.5:
            return redirect(url_for("fraud_yes"))
        else:
            return redirect(url_for("fraud_no"))

    return render_template("prediction.html")


@app.route('/prediction/yes')
def fraud_yes():
    return render_template("fraud_yes.html")

@app.route('/prediction/no')
def fraud_no():
    return render_template("fraud_no.html")

@app.route('/dataset')
def dataset():
    df = pd.read_csv('data/raw_data_1.csv')

    preview = df.head(10)
    table = preview.to_html(classes = 'table table-striped table-bordered', index = False)

    total_records = len(df)
    fraud_counts = df['fraud reported'].value_counts().to_dict()
    fraud_yes = fraud_counts.get('Y', 0)
    fraud_no = fraud_counts.get('N', 0)

    return render_template(
        'dataset.html', 
        table = table,
        total_records = total_records,
        fraud_yes = fraud_yes,
        fraud_no = fraud_no   
    )


@app.route('/feature_engineering')
def feature_engineering():
    return render_template('feature_engineering.html')

@app.route('/model')
def model_info():
    return render_template('model.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
    