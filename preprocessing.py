import pandas as pd
import numpy as np 

df = pd.read_csv('data/raw_data_1.csv')

numeric_col = [
    'total_claim',
    'injury_claim',
    'vehicle_price',
    'annual_income',
    'policy deductible',
    'past_num_of_claims',
    'days_open'
]
for col in numeric_col:
    df[col] = pd.to_numeric(df[col], errors = 'coerce')

df.drop(columns = [
    'claim_number',
    'zip_code', 
    'vehicle_color', 
    'claim_date'
    ], errors = 'ignore')

df['Fraud'] = df['fraud reported'].map({'Yes': 1, 'No': 0})
df.drop(columns = ['fraud reported'], inplace = True)


df['claim_to_vehicle'] = df['total_claim'] / df['vehicle_price'].replace(0, np.nan)
df['injury_to_claim'] = df['injury_claim'] / df['total_claim'].replace(0, np.nan)
df['income_to_claim'] = df['total_claim'] / df['annual_income'].replace(0, np.nan)
df['deductible_to_claim'] = df['policy deductible'] / df['total_claim'].replace(0, np.nan)  

df['repeat_claim_flag'] = np.where(df['past_num_of_claims'] > 2, 1, 0)
median_days = df['days_open'].median()
df['days_open_flag'] = np.where(df['days_open'] > median_days, 1, 0)

categorical_col = [
    'gender',
    'marital_status',
    'property_status',
    'accident_site',
    'channel',
    'police_report',
    'vehicle_category'
]

df = pd.get_dummies(df, 
                    columns = [col for col in categorical_col],
                    drop_first = True)

df = df.fillna(0)

df.to_csv('data/processed_data.csv', index = False)