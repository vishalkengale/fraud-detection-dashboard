# Financial Fraud Detection Dashboard

An interactive dashboard and machine learning model that show where fraud happens in card transactions and score how risky a single transaction is.

**Live demo:** https://vishalkengale.github.io/fraud-detection-dashboard/

> Built for the ZIDIO Data Science & Analytics program by Vishal Kengale.

## Screenshots

![Dashboard overview](screenshots/dashboard-overview.png)
![Fraud by hour and segments](screenshots/hour-and-segments.png)
![Model results and transaction scorer](screenshots/model-and-scorer.png)

## What it does

- Summarizes 5,000 transactions: total transactions, fraud cases, fraud rate and value in fraud cases.
- Shows the three signals that explain most of the fraud: international transactions, suspicious keywords and night-time activity.
- Charts fraud by hour of day and fraud cases per month.
- Compares segments (city, merchant category, payment method, device type).
- Updates every chart and number when you use the filters (city, category, payment, device, month) or click a bar.
- Compares three machine learning models and explains the results.
- Scores a single transaction with the trained model and rates it as low, medium or high risk.
- Lists the ten highest-risk transactions for the current selection.
- Works in light and dark mode and on mobile screens.

## Key findings

| Signal | Fraud rate with signal | Fraud rate without |
|---|---|---|
| International transaction | 37.0% | 7.0% |
| Suspicious keyword | 47.3% | 7.6% |
| Night time (12 am to 5 am) | 25.6% | 4.5% |

- Overall, 482 of 5,000 transactions (9.64%) are fraud.
- City, merchant category, payment method and device type stay close to the average fraud rate, so they are weak signals on their own.

## Model results

Trained on 75% of the data and tested on the other 25% (1,250 transactions, stratified split).

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** (used) | 0.908 | 0.368 | 0.758 | 0.496 |
| Random Forest | 0.895 | 0.348 | 0.650 | 0.453 |
| Gradient Boosting | 0.900 | 0.510 | 0.217 | 0.304 |

- Logistic Regression uses balanced class weights and an alert threshold of 0.70. The other two models use the default 0.5 threshold.
- The model catches 91 of the 120 fraud cases in the test set, with 156 false alarms.
- Accuracy alone is misleading here because only about 1 in 10 transactions is fraud, so the models are judged by AUC, precision and recall.

## Dataset

`financial_fraud_detection_dataset.csv` has 5,000 transactions from January 2023 to February 2024.

Columns: `Transaction_ID`, `Customer_ID`, `Transaction_Date`, `Transaction_Amount`, `Merchant_Category`, `Payment_Method`, `Device_Type`, `Location`, `Is_International`, `Previous_Transactions`, `Average_Spend`, `Account_Age_Days`, `Suspicious_Keyword`, `Fraudulent`.

## Tech stack

- **Python:** pandas, NumPy, scikit-learn (data preparation and model training)
- **Dashboard:** HTML, CSS and JavaScript in a single file, with no external libraries. The data and the trained Logistic Regression coefficients are embedded in the page.

## Project structure

```
fraud-detection-dashboard/
├── index.html                              # the dashboard
├── fraud_model_training.py                 # trains and compares the models
├── financial_fraud_detection_dataset.csv   # dataset
├── screenshots/                            # images used in this README
└── README.md
```

## How to run

**Dashboard:** open `index.html` in any modern browser. No installation is needed.

**Model training:**

```bash
pip install pandas numpy scikit-learn
python fraud_model_training.py
```

Keep the CSV file in the same folder as the script.

## How the model works

1. Features: international flag, suspicious keyword flag, night-time flag (hour 0 to 5), amount compared with the customer's average spend, account age and number of previous transactions.
2. The data is split into training and test sets, keeping the same fraud share in both.
3. Three models are trained and compared.
4. Logistic Regression is used in the dashboard. Its coefficients are copied into the page, so the scorer works in the browser without a server.

## Limitations

- Precision is 37%, so about 6 in 10 flagged transactions are false alarms. This is a deliberate trade-off to catch more fraud.
- The 0.70 alert threshold was chosen on the test set, so the reported scores may be slightly optimistic.
- The ten highest-risk transactions are scored on data the model was trained on.
- Differences across cities and categories are small, which suggests the data may be synthetic.
- The dataset has no currency column, so amounts are shown as plain numbers.

## Future work

- Real-time monitoring with streaming data.
- Email alerts for high-risk transactions.
- Testing on real bank data and retraining over time.
- Unsupervised methods such as Isolation Forest.

## Author

Vishal Kengale, ZIDIO Data Science & Analytics.
