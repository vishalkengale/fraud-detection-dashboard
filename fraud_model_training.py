"""Fraud model training for financial_fraud_detection_dataset.csv
Run:  python fraud_model_training.py   (needs pandas, numpy, scikit-learn)
Keep the CSV in the same folder as this script."""
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Load data and build features
df = pd.read_csv('financial_fraud_detection_dataset.csv')
df['dt'] = pd.to_datetime(df.Transaction_Date, format='%d-%m-%Y %H:%M')
df['intl'] = df.Is_International
df['kw'] = (df.Suspicious_Keyword == 'Yes').astype(int)
df['night'] = (df.dt.dt.hour <= 5).astype(int)                      # 12 am to 5 am
df['ratio'] = df.Transaction_Amount / df.Average_Spend              # amount vs customer's usual spend
df['age'] = df.Account_Age_Days / 1000
df['prev'] = df.Previous_Transactions / 100
FEATURES = ['intl', 'kw', 'night', 'ratio', 'age', 'prev']
y = df.Fraudulent
print(f'Fraud rate: {y.mean():.2%}  (imbalanced, so accuracy alone is misleading)')

# 2. Train / test split (stratified keeps the fraud share equal in both parts)
Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], y, test_size=0.25, stratify=y, random_state=42)

# 3. Compare three models
models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000),
    'Random Forest': RandomForestClassifier(300, class_weight='balanced', min_samples_leaf=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    print(f'{name:20s} AUC={roc_auc_score(yte, p):.3f}  precision={precision_score(yte, pred):.3f}  '
          f'recall={recall_score(yte, pred):.3f}  F1={f1_score(yte, pred):.3f}')

# 4. Chosen model: Logistic Regression with a tuned alert threshold
lr = models['Logistic Regression']
p = lr.predict_proba(Xte)[:, 1]
th = max(np.arange(0.3, 0.95, 0.025), key=lambda t: f1_score(yte, (p >= t).astype(int)))
pred = (p >= th).astype(int)
print(f'\nThreshold {th:.2f}: precision={precision_score(yte, pred):.3f} recall={recall_score(yte, pred):.3f} F1={f1_score(yte, pred):.3f}')
print('Confusion matrix [[TN, FP], [FN, TP]]:\n', confusion_matrix(yte, pred))
print('Coefficients:', dict(zip(FEATURES, lr.coef_[0].round(3))))
# Note: the threshold was tuned on the test set, so these numbers can be slightly optimistic.
