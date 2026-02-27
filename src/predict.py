import joblib
import pandas as pd
import numpy as np

model = joblib.load('model/fraud_detection.pkl')

def predict_fraud(input_data):
    input_df = pd.DataFrame([input_data])

    input_df['claim_date'] = pd.to_datetime(input_df['claim_date'], errors='coerce')
    input_df['claim_month'] = input_df['claim_date'].dt.month
    input_df['claim_weekend'] = (input_df['claim_date'].dt.weekday >= 5).astype(int)
    input_df  = input_df.drop(columns=['claim_date'])

    input_df['claim_to_vehicle'] = input_df['total_claim']/input_df['vehicle_price']
    input_df['injury_to_claim'] = input_df['injury_claim']/input_df['total_claim']
    input_df['income_to_claim'] = input_df['total_claim']/input_df['annual_income']
    input_df['deductible_to_claim'] = input_df['policy deductible']/input_df['total_claim']
    input_df['repeat_claim_flag'] = np.where(input_df['past_num_of_claims'] > 2, 1, 0)

    median_days = np.load("model/median_days.npy") 
    input_df['days_open_flag'] = np.where(input_df['days open'] > median_days, 1, 0)

    input_df['claim_severity_index'] = input_df['total_claim'] / (input_df['policy deductible'] + 1)
    input_df['high_claim_flag'] = np.where(
        input_df['total_claim'] > input_df['total_claim'].quantile(0.90), 1, 0
        )
    
    categorical_col = [
    "gender",
    "marital_status",
    "property_status",
    "accident_site",
    "channel",
    "vehicle_category",
    "claim_day_of_week" 
    ]

    input_df = pd.get_dummies(
        input_df,
        columns=[col for col in categorical_col if col in input_df.columns],
        drop_first=True
    )

    train_df = pd.read_csv("data/processed_data.csv")
    train_columns = train_df.drop("Fraud", axis=1).columns

    input_df = input_df.reindex(columns=train_columns, fill_value=0)
    probability = float(model.predict_proba(input_df)[:, 1][0])

    def risk_level(prob):
        if prob < 0.3:
            return "Low Risk"
        elif prob < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"

    risk = risk_level(probability)

    print("\n----- Fraud Risk Assessment -----")
    print(f"Fraud Probability : {probability * 100:.2f}%")
    print(f"Risk Category     : {risk}")
    print("----------------------------------\n")


claim_1 = {
    "age_of_driver": 52,
    "annual_income": 95000,
    "past_num_of_claims": 0,
    "liab_prct": 30,
    "age_of_vehicle": 6,
    "vehicle_price": 30000,
    "total_claim": 4000,
    "injury_claim": 500,
    "policy deductible": 1000,
    "annual premium": 1500,
    "days open": 18,
    "witness_present": 1,
    "police_report": 1,
    "form defects": 0,
    "address_change": 0,
    "claim_day_of_week": "Tuesday",
    "gender": "Female",
    "marital_status": "Married",
    "property_status": "Own",
    "accident_site": "Street",
    "channel": "Agent",
    "vehicle_category": "Sedan",
    "claim_date": "2024-04-10"
}

claim_2 = {
    "age_of_driver": 34,
    "annual_income": 42000,
    "past_num_of_claims": 2,
    "liab_prct": 60,
    "age_of_vehicle": 2,
    "vehicle_price": 25000,
    "total_claim": 18000,
    "injury_claim": 12000,
    "policy deductible": 500,
    "annual premium": 1100,
    "days open": 4,
    "witness_present": 0,
    "police_report": 1,
    "form defects": 0,
    "address_change": 1,
    "claim_day_of_week": "Saturday",
    "gender": "Male",
    "marital_status": "Single",
    "property_status": "Rent",
    "accident_site": "Highway",
    "channel": "Online",
    "vehicle_category": "SUV",
    "claim_date": "2024-07-20"
}

claim_3 = {
    "age_of_driver": 27,
    "annual_income": 28000,
    "past_num_of_claims": 4,
    "liab_prct": 90,
    "age_of_vehicle": 1,
    "vehicle_price": 22000,
    "total_claim": 21000,
    "injury_claim": 19000,
    "policy deductible": 250,
    "annual premium": 900,
    "days open": 1,
    "witness_present": 0,
    "police_report": 0,
    "form defects": 1,
    "address_change": 1,
    "claim_day_of_week": "Sunday",
    "gender": "Male",
    "marital_status": "Single",
    "property_status": "Rent",
    "accident_site": "Parking Lot",
    "channel": "Online",
    "vehicle_category": "Sports",
    "claim_date": "2024-08-18"
}

print("Claim 1:")
predict_fraud(claim_1)
print('\nClaim 2:')
predict_fraud(claim_2)
print('\nClaim 3:')
predict_fraud(claim_3)
