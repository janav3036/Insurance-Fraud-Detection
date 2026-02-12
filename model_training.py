import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score



df = pd.read_csv('data/processed_data.csv')

x = df.drop(columns=['Fraud'])
y = df['Fraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, max_depth=None, class_weight='balanced', random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_prob)}')

joblib.dump(model, 'model/fraud_detection.pkl') 
print('done')
