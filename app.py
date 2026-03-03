from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os 

app = Flask(__name__)

model = joblib.load('model/fraud_detection.pkl')
median_days = np.load('model/median_days.npy')

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

@app.route('/prediction', methods = ['POST', 'GET'])
def prediction():

    if request.method == 'POST':
        try:
            data = {
                'total_claim' : float(request.form['total_claim']),
                'injury_claim': float(request.form['injury_claim']),
                'vehicle_price': float(request.form['vehicle_price']),
            }

            df = pd.DataFrame([data])

            df['claim_to_vehicle'] = df['total_claim']/ df['vehicle_price']
            df['injury_to_claim'] = df['injury_claim']/ df['total_claim']
            df['income_to_claim'] = df['total_claim']/df['annual_income']
            df['deductible_to_claim'] = df['policy_deductible']/df['total_claim']

            df['repeat_claim_flag'] = np.where(df['past_num_of_claims'] > 2, 1, 0)
            df['days_open_flag'] = np.where(df['days_open'] > median_days, 1, 0)
            df['claim_severity_index'] = (df['total_claim']/ (df['policy_deductible'] +1))
            df['claim_flag'] = np.where(df['total_claim'] > 50000, 1, 0)

            df = df.fillna(0)

            train_df = pd.read_csv('data/processed_data.csv')
            train_cols = train_df.drop('Fraud', axis = 1).columns

            df = df.reindex(columns = train_cols, fill_value = 0)

            prob = float(model.predict_proba(df)[:, 1][0])
            risk = risk_level(prob)
            return render_template('prediction.html', prob = round(prob*100, 2), risk = risk)
        
        except Exception as e:
            return render_template('prediction.html', probability = 'Error', risk = str(e))
        
    return render_template('prediction.html')

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
    