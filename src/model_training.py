import pandas as pd
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve




df = pd.read_csv('data/processed_data.csv')

x = df.drop(columns=['Fraud'])
y = df['Fraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=1,
    min_child_weight=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="auc"
)


model.fit(x_train, y_train)
y_prob = model.predict_proba(x_test)[:, 1]

threshold = 0.30
y_pred = (y_prob > threshold).astype(int)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f'ROC AUC Score: {roc_auc_score(y_test, y_prob)}')

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.plot([0,1], [0,1], 'k--', label='Random Classifier')
plt.show()

joblib.dump(model, 'model/fraud_detection.pkl') 
print('done')
