import pandas as pd
import numpy as np 

df = pd.read_csv('data/raw_data_1.csv')   

df = df.replace('*', np.nan)


numeric_col = [
    'total_claim',
    'injury_claim',
    'vehicle_price',
    'annual_income',
    'policy deductible',
    'past_num_of_claims',
    'days open'
]
for col in numeric_col:
    df[col] = pd.to_numeric(df[col], errors = 'coerce')

df = df.drop(columns = [
    'claim_number',
    'zip_code', 
    'vehicle_color', 
    ], errors = 'ignore')

df['Fraud'] = df['fraud reported'].map({'Y': 1, 'N': 0})
df.drop(columns = ['fraud reported'], inplace = True)

binary_cols = [
    'witness_present',
    'police_report',
    'form defects',
    'address_change'
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})



df['claim_to_vehicle'] = df['total_claim'] / df['vehicle_price'].replace(0, np.nan)
df['injury_to_claim'] = df['injury_claim'] / df['total_claim'].replace(0, np.nan)
df['income_to_claim'] = df['total_claim'] / df['annual_income'].replace(0, np.nan)
df['deductible_to_claim'] = df['policy deductible'] / df['total_claim'].replace(0, np.nan)  

df['repeat_claim_flag'] = np.where(df['past_num_of_claims'] > 2, 1, 0)
median_days = df['days open'].median()
np.save("model/median_days.npy", median_days)

df['days_open_flag'] = np.where(df['days open'] > median_days, 1, 0)

categorical_col = [
    'gender',
    'marital_status',
    'property_status',
    'accident_site',
    'channel',
    'police_report',
    'vehicle_category',
    'claim_day_of_week'
]

df = pd.get_dummies(df, 
                    columns = [col for col in categorical_col],
                    drop_first = True)

df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')

# Extract features
df['claim_month'] = df['claim_date'].dt.month
df['claim_weekend'] = (df['claim_date'].dt.weekday >= 5).astype(int)

df = df.drop(columns=['claim_date'])

df['claim_severity_index'] = (
    df['total_claim'] / (df['policy deductible'] + 1)
)

df['high_claim_flag'] = np.where(
    df['total_claim'] > df['total_claim'].quantile(0.90),
    1,
    0
)


df = df.fillna(0)

df.to_csv('data/processed_data.csv', index = False)
print('done')