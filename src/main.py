# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 17:29:14 2025

@author: Mauricio Ruiz
"""


import os
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# Ensure folders exist (GitHub-safe: data and outputs go into folders)
os.makedirs("data", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


'STOCK PRICES'
# Select Stocks Top20 and Dates
tickers = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'TSLA', 'AVGO', 'ADBE', 'PEP',
    'COST', 'AMD', 'NFLX', 'INTC', 'CSCO', 'TXN', 'QCOM', 'AMAT', 'PYPL', 'SBUX'
]
start = '2015-08-01'
end = '2024-12-31'

# Getting data
data = yf.download(tickers, start=start, end=end, group_by='ticker')

# Reorganize the data into a clean wide format
clean_data = pd.DataFrame()

for ticker in tickers:
    if ticker in data:
        ticker_data = data[ticker].copy()
        ticker_data.columns = [f'{ticker}_{col}' for col in ticker_data.columns]

        # Join on index (Date)
        if clean_data.empty:
            clean_data = ticker_data
        else:
            clean_data = clean_data.join(ticker_data, how='outer')

# Check values
missing_values = clean_data.isnull().sum().sum()
print(f"Missing values: {missing_values}")

# Reset index -> Have Date as a column to merge with NDX
clean_data.reset_index(inplace=True)

# Save to csv (UPDATED PATH)
clean_data.to_csv("data/nasdaq_top20_data.csv", index=False)


'NASDAQ Closing and Returns'
# gettting Nasdaq Closing Price - NDX
ndx = yf.download('^NDX', start='2015-08-01', end='2024-12-31')
ndx.reset_index(inplace=True)

# Check missing values
print("Missing values in NDX after download:")
print(ndx.isnull().sum())

# Keeping only Date and Close
ndx = ndx[['Date', 'Close']]
ndx.rename(columns={'Close': 'NDX_Close'}, inplace=True)

# Calculate the target — next-day return % change daily
ndx['NDX_Return_Tomorrow'] = ndx['NDX_Close'].shift(-1) / ndx['NDX_Close'] - 1

# Drop last row (no next day to calculate return)
ndx.dropna(inplace=True)

# Save to CSV (UPDATED PATH)
ndx.to_csv('data/ndx_with_returns.csv', index=False)

# Load NDX return target and merge by Date (UPDATED PATH)
ndx = pd.read_csv("data/ndx_with_returns.csv")

# Merge on Date from Stock prices column
# Convert both Date columns to datetime
clean_data['Date'] = pd.to_datetime(clean_data['Date'])
ndx['Date'] = pd.to_datetime(ndx['Date'])

# Merge
merged_data = pd.merge(clean_data, ndx, on='Date', how='inner')

# Final merged dataset (UPDATED PATH)
merged_data.to_csv("data/nasdaq_top20_with_target.csv", index=False)
merged_data.shape


'FRED Data'
start_date = datetime(2015, 8, 1)
end_date = datetime(2024, 12, 31)

# Federal Funds Rate
fed_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)

# Consumer Price Index (CPI)
cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)

# Unemployment Rate
unemployment = web.DataReader('UNRATE', 'fred', start_date, end_date)

# Real GDP (Quarterly)
gdp = web.DataReader('GDPC1', 'fred', start_date, end_date)

# 10-Year Treasury Constant Maturity Rate
ten_year = web.DataReader('GS10', 'fred', start_date, end_date)

# M2 Money Supply
m2 = web.DataReader('M2SL', 'fred', start_date, end_date)

# Merge all into one DataFrame
fred_data = fed_rate.join([cpi, unemployment, gdp, ten_year, m2], how='outer')
fred_data.columns = ['FedFunds', 'CPI', 'Unemployment', 'GDP', 'TenYear', 'M2']

# Checking missing values
print("Missing values:")
print(fred_data.isnull().sum())

# Plot missing values
plt.figure(figsize=(10, 4))
fred_data['GDP'].isnull().astype(int).plot()
plt.title("Missing GDP Over Time")
plt.ylabel("Missing (1) / Present (0)")
plt.yticks([0, 1])  # Only show 0 and 1 on y-axis
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Forward fill for missing values in GDP
fred_data = fred_data.sort_index().fillna(method='ffill')

# Checking missing values
print("Missing values:")
print(fred_data.isnull().sum())

# 2 missing values at the beginning so using bfill
fred_data = fred_data.ffill().bfill()

# Checking missing values
print("Missing values:")
print(fred_data.isnull().sum())

# Save the merged dataset (UPDATED PATH)
fred_data.to_csv('data/fred_combined_data.csv')


' Merging FRED and NASDAQ with NDX returns'
# reading csv data (UPDATED PATHS)
fred_data = pd.read_csv("data/fred_combined_data.csv", parse_dates=['DATE'])
merged_data = pd.read_csv("data/nasdaq_top20_with_target.csv", parse_dates=['Date'])

# Rename date column
fred_data.rename(columns={'DATE': 'Date'}, inplace=True)

# Merge macro with nasdaq
fred_data = fred_data.sort_values('Date').set_index('Date')  # sort and set index
merged_data = merged_data.sort_values('Date').set_index('Date')  # merge

# Forward fill FRED
full_data = merged_data.join(fred_data, how='left')
full_data = full_data.ffill()

# Check missing values
print("\nMissing values after forward-fill:")
print(full_data.isnull().sum())

# backfill for missing values
full_data = full_data.ffill().bfill()

# Reset index - turn date into column
full_data.reset_index(inplace=True)

# DF saved independently for the 2 models
regression_df = full_data.copy()
classification_df = full_data.copy()

# Save CSV (UPDATED PATH)
full_data.to_csv("data/final_modeling_dataset.csv", index=False)

'Engineering Features'

# Regression Model
# Lagged macro indicators (1-day lag)
regression_df['FedFunds_lag1'] = regression_df['FedFunds'].shift(1)
regression_df['CPI_lag1'] = regression_df['CPI'].shift(1)
regression_df['Unemployment_lag1'] = regression_df['Unemployment'].shift(1)
regression_df['GDP_lag1'] = regression_df['GDP'].shift(1)
regression_df['TenYear_lag1'] = regression_df['TenYear'].shift(1)
regression_df['M2_lag1'] = regression_df['M2'].shift(1)

# Rolling averages (3-period window)
regression_df['FedFunds_ma3'] = regression_df['FedFunds'].rolling(window=3).mean()
regression_df['Unemployment_ma3'] = regression_df['Unemployment'].rolling(window=3).mean()

# Rolling volatility (standard deviation)
regression_df['CPI_rolling_std3'] = regression_df['CPI'].rolling(window=3).std()

# NDX return rolling stats
regression_df['NDX_Return_rolling_mean3'] = regression_df['NDX_Return_Tomorrow'].rolling(window=3).mean()
regression_df['NDX_Return_volatility3'] = regression_df['NDX_Return_Tomorrow'].rolling(window=3).std()

# Binary flag GDP: Identify when GPD changed (new report)
regression_df['GDP_Updated'] = regression_df['GDP'].diff().ne(0).astype(int)

# Stock Returns
close_cols = [col for col in regression_df.columns if col.endswith('_Close') and not col.startswith('NDX')]
for col in close_cols:
    return_col = col.replace('_Close', '_Return')
    regression_df[return_col] = regression_df[col].pct_change()
    
# Stock lagged (1-day)
for col in regression_df.columns:
    if col.endswith('_Return'):
        regression_df[col + '_lag1'] = regression_df[col].shift(1)

# Check missing values
print("Missing values:")
print(regression_df.isnull().sum())

print("\nRows with missing values:", regression_df.isnull().any(axis=1).sum())

regression_df[regression_df.isnull().any(axis=1)].head()

# Drop NAs
regression_df.dropna(inplace=True)

## Z Score - Normalize Macro indicators
macro_engineered_cols = [col for col in regression_df.columns 
                         if col.startswith(('FedFunds', 'CPI', 'Unemployment', 'GDP', 'TenYear', 'M2')) 
                         and col not in ['GDP_Updated']]

scaler = StandardScaler()
regression_df[macro_engineered_cols] = scaler.fit_transform(regression_df[macro_engineered_cols])



# Drop raw stock features - unnecessary features
columns_to_drop = [col for col in regression_df.columns if any(col.endswith(suffix) for suffix in ['_Open', '_High', '_Low', '_Volume', 'Close'])]

# Drop raw macro - unnecessary 
columns_to_drop += ['FedFunds', 'CPI', 'Unemployment', 'GDP', 'TenYear', 'M2']

# Drop unlagged stock returns 
drop_unlagged_returns = [col for col in regression_df.columns 
                         if col.endswith('_Return') and not col.endswith('_Return_lag1')]
columns_to_drop += drop_unlagged_returns
# Dropping columns
regression_df.drop(columns=columns_to_drop, inplace=True)

#Data Analysis Section

# List of macro indicators 
macro_vars = [
    'FedFunds_lag1',
    'CPI_lag1',
    'Unemployment_lag1',
    'GDP_lag1',
    'TenYear_lag1',
    'M2_lag1'
]

# Plot one chart per macroindicator
for macro in macro_vars:
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.set_title(f"NDX Returns vs {macro}")
    ax1.set_xlabel("Date")

    # Plot macro indicator (left y-axis)
    ax1.plot(regression_df.index, regression_df[macro], color='blue', label=macro)
    ax1.set_ylabel(macro, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot NDX returns (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(regression_df.index, regression_df['NDX_Return_Tomorrow'], color='orange', alpha=0.6, label='NDX Return (Tomorrow)')
    ax2.set_ylabel('NDX Return (Tomorrow)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.tight_layout()
    plt.show()



# Save to csv
regression_df.to_csv("macro_regression_dataset.csv", index=False)


# Classification model

## Calsification target
classification_df['NDX_Direction_Tomorrow'] = (classification_df['NDX_Return_Tomorrow'] > 0).astype(int)

# Lagged macro indicators (1-day lag)
classification_df['FedFunds_lag1'] = classification_df['FedFunds'].shift(1)
classification_df['CPI_lag1'] = classification_df['CPI'].shift(1)
classification_df['Unemployment_lag1'] = classification_df['Unemployment'].shift(1)
classification_df['GDP_lag1'] = classification_df['GDP'].shift(1)
classification_df['TenYear_lag1'] = classification_df['TenYear'].shift(1)
classification_df['M2_lag1'] = classification_df['M2'].shift(1)

# Changes in macro indicators %
classification_df['FedFunds_pct_change'] = classification_df['FedFunds'].pct_change()
classification_df['CPI_pct_change'] = classification_df['CPI'].pct_change()
classification_df['Unemployment_pct_change'] = classification_df['Unemployment'].pct_change()
classification_df['GDP_pct_change'] = classification_df['GDP'].pct_change()
classification_df['TenYear_pct_change'] = classification_df['TenYear'].pct_change()
classification_df['M2_pct_change'] = classification_df['M2'].pct_change()

# Volatility (std) and momentum (mean) 3 days

classification_df['NDX_Return_rolling_mean3'] = classification_df['NDX_Return_Tomorrow'].rolling(window=3).mean()
classification_df['NDX_Return_volatility3'] = classification_df['NDX_Return_Tomorrow'].rolling(window=3).std()

# Nasdaq lag -1
classification_df['NDX_Close_lag1'] = classification_df['NDX_Close'].shift(1)

## Binary flag GDP: Identify when GPD changed (new report)
classification_df['GDP_Updated'] = classification_df['GDP'].diff().ne(0).astype(int)

# Retunrs for each stock
close_cols = [col for col in classification_df.columns if col.endswith('_Close') and not col.startswith('NDX')]
for col in close_cols:
    return_col = col.replace('_Close', '_Return')
    classification_df[return_col] = classification_df[col].pct_change()

# 1-day lagged returns
for col in classification_df.columns:
    if col.endswith('_Return') and not col.endswith('_lag1'):
        classification_df[col + '_lag1'] = classification_df[col].shift(1)


# Check missing values
print("Missing values:")
print(classification_df.isnull().sum())

print("\nRows with missing values:", classification_df.isnull().any(axis=1).sum())

classification_df[classification_df.isnull().any(axis=1)].head()

# Drop NAs
classification_df.dropna(inplace=True)

# Z score normalisation for macro indicators
macro_cols_to_scale = [col for col in classification_df.columns
                       if col.startswith(('FedFunds', 'CPI', 'Unemployment', 'GDP', 'TenYear', 'M2'))
                       and col not in ['GDP_Updated']]  

# Drop raw stock features - unnecessary 
drop_stock_cols = [col for col in classification_df.columns 
                   if any(col.endswith(suffix) for suffix in ['_Open', '_High', '_Low', '_Volume', '_Close'])]

# Drop raw macro - unnecessary 
drop_macro_raw = ['FedFunds', 'CPI', 'Unemployment', 'GDP', 'TenYear', 'M2']

# Drop unlagged stock returns 
drop_stock_returns = [col for col in classification_df.columns 
                      if col.endswith('_Return') and not col.endswith('_Return_lag1')]

# Combine all columns to drop
columns_to_drop = drop_stock_cols + drop_macro_raw + drop_stock_returns

# Drop columns
classification_df.drop(columns=columns_to_drop, inplace=True)

# Save to CSV
classification_df.to_csv("market_classification_dataset.csv", index=False)




'Models'

'Regression Model '
# Checking features before modelling

print(f"Rows: {regression_df.shape[0]}")
print(f"Columns: {regression_df.shape[1]}")

# Show all columns
print("\nColumns:")
print(regression_df.columns.tolist())

# data sorted chronologically
regression_df = regression_df.sort_values(by='Date').reset_index(drop=True)

# Selecting features for X and target y
X = regression_df.drop(columns=['Date', 'NDX_Return_Tomorrow'])
y = regression_df['NDX_Return_Tomorrow']

## Chronological split (80% train, 20% test)
split_index = int(len(regression_df) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Hyperparameters
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

# Early stopping
xgb_reg.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=40,
    eval_metric='rmse',
    verbose=True
)

# Predictions
y_pred = xgb_reg.predict(X_test)

# Model evaluation
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.6f}") 
print(f"R² Score: {r2:.4f}")
print(f"Best iteration: {xgb_reg.best_iteration}")

## PLOTS
# Predictions vs Actual
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color='royalblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual NDX Returns (Tomorrow)")
plt.ylabel("Predicted NDX Returns")
plt.title("Predicted vs. Actual NDX Returns")
plt.grid(True)
plt.tight_layout()
plt.show()


# Training vs validation (early stop)
results = xgb_reg.evals_result()
epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 5))
plt.plot(x_axis, results['validation_0']['rmse'], label='Test')
plt.xlabel('Boosting Rounds')
plt.ylabel('RMSE')
plt.title('Early Stopping Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#features
#High importance features
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_reg.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel("Gain Importance")
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()  # Highest at top
plt.tight_layout()
plt.show()

# low importance features
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_reg.feature_importances_
}).sort_values(by='Importance', ascending=True).head(10)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel("Gain Importance")
plt.title("Bottom 10 Feature Importances")
plt.gca().invert_yaxis()  # Highest at top
plt.tight_layout()
plt.show()

# comparison predictions vs actual
comparison_df = pd.DataFrame({
    'Date': regression_df.loc[y_test.index, 'Date'],
    'Actual': y_test,
    'Predicted': y_pred
}).sort_values('Date')

# Line plot
plt.figure(figsize=(14, 6))
plt.plot(comparison_df['Date'], comparison_df['Actual'], label='Actual Returns', linewidth=2)
plt.plot(comparison_df['Date'], comparison_df['Predicted'], label='Predicted Returns', linewidth=2, linestyle='--')
plt.xlabel("Date")
plt.ylabel("NDX Return (Tomorrow)")
plt.title("Actual vs. Predicted NDX Returns Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'Classification mddel'

# Checking features before modelling

print(f"Rows: {classification_df.shape[0]}")
print(f"Columns: {classification_df.shape[1]}")

# Show all columns
print("\nColumns:")
print(classification_df.columns.tolist())

# data sorted chronologically
classification_df = classification_df.sort_values(by='Date').reset_index(drop=True)

# Selecting features for X and target y
X_cls = classification_df.drop(columns=['Date', 'NDX_Direction_Tomorrow', 'NDX_Return_Tomorrow'])
y_cls = classification_df['NDX_Direction_Tomorrow']  # should be 0/1 (Down/Up)

# Chronological split (80% train, 20% test)
split_index_cls = int(len(classification_df) * 0.8)
X_train_cls = X_cls.iloc[:split_index_cls]
X_test_cls = X_cls.iloc[split_index_cls:]
y_train_cls = y_cls.iloc[:split_index_cls]
y_test_cls = y_cls.iloc[split_index_cls:]

#Hyerparameters
xgb_clf = XGBClassifier(
    objective='binary:logistic',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Early stopping
xgb_clf.fit(
    X_train_cls, y_train_cls,
    eval_set=[(X_test_cls, y_test_cls)],
    early_stopping_rounds=40,
    verbose=True
)

#predictions 
y_pred_cls = xgb_clf.predict(X_test_cls)

# Model evaluation
accuracy = accuracy_score(y_test_cls, y_pred_cls)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls))
print(f"Best iteration: {xgb_clf.best_iteration}")
# Confusion matrix
conf_matrix = confusion_matrix(y_test_cls, y_pred_cls)
print("\nConfusion Matrix:\n", conf_matrix)

#PLOTS
#features
#High importance features
importances = xgb_clf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_cls.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel("Gain Importance")
plt.title("Top 10 Feature Importances (by Gain)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Training vs validation (early stop)
results_cls = xgb_clf.evals_result()
epochs_cls = len(results_cls['validation_0']['logloss'])
x_axis_cls = range(0, epochs_cls)

plt.figure(figsize=(10, 5))
plt.plot(x_axis_cls, results_cls['validation_0']['logloss'], label='Validation LogLoss')
plt.xlabel('Boosting Rounds')
plt.ylabel('LogLoss')
plt.title('Early Stopping Curve (Classification)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# low importance features
importances = xgb_clf.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_cls.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=True).head(10)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel("Gain Importance")
plt.title("Top 10 Feature Importances (by Gain)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#comparison predictions vs actual
plt.figure(figsize=(12, 5))
plt.plot(y_test_cls.values, label='Actual', alpha=0.7, marker='o')
plt.plot(y_pred_cls, label='Predicted', alpha=0.7, marker='x')
plt.title("Actual vs Predicted Directions (Line Plot)")
plt.xlabel("Observation")
plt.ylabel("Direction (0=Down, 1=Up)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test_cls, y_pred_cls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# Prediction probabilities
probs = xgb_clf.predict_proba(X_test_cls)[:, 1]
preds = xgb_clf.predict(X_test_cls)

# Create a DataFrame for analysis
results_df = pd.DataFrame({
    'True': y_test_cls.values,
    'Predicted': preds,
    'Prob': probs
})
results_df['Correct'] = results_df['True'] == results_df['Predicted']

# Separating correct and incorrect predictions
correct_probs = results_df[results_df['Correct']]['Prob']
incorrect_probs = results_df[~results_df['Correct']]['Prob']

# Plot histograms
plt.figure(figsize=(10, 5))
plt.hist([incorrect_probs, correct_probs], 
         bins=20, 
         stacked=True, 
         label=['False', 'True'], 
         color=['brown', 'lightgreen'],
         edgecolor='black')

# Labels and title
plt.title("Prediction Confidence vs. Correctness")
plt.xlabel("Predicted Probability (Class 1)")
plt.ylabel("Count")
plt.legend(title="Correct")
plt.grid(True)
plt.tight_layout()
plt.show()


