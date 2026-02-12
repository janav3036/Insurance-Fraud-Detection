import pandas as pd
import numpy as np 

df = pd.read_csv('data/raw_data_1.csv')

df.drop(columns = [
    'claim_number',
    'zip_code', 
    'vehicle_color', 
    'claim_date'
    ])

df['Fraud'] = df['fraud reported'].map({'Yes': 1, 'No': 0})
df.drop(columns = ['fraud reported'], inplace = True)
