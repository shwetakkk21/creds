import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier # Using this for the placeholder income model
import joblib
import json

print("--- Starting Model and Asset Training ---")

# --- Load and Prepare Original Data ---
try:
    internal_data = pd.read_excel('Internal_Bank_Dataset.xlsx')
    external_data = pd.read_excel('External_Cibil_Dataset.xlsx')
    train_df = pd.merge(internal_data, external_data, on='PROSPECTID')
    train_df.replace(-99999, np.nan, inplace=True)
    print("Datasets loaded and merged successfully.")
except FileNotFoundError:
    print("ERROR: Could not find the original data files. Please place them in the same folder.")
    exit()

# --- 1. Train and Save Model A (Repayment Score) ---

print("\n--- Training Model A: Repayment Score Model ---")
# Define the safe, non-leaky features for the repayment model
repayment_features = [
    'Tot_Missed_Pmnt', 'Home_TL', 'PL_TL', 'Secured_TL', 'Unsecured_TL',
    'Age_Oldest_TL', 'Age_Newest_TL', 'time_since_recent_payment',
    'CC_enq_L12m', 'PL_enq_L12m', 'time_since_recent_enq', 'enq_L3m',
    'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag', 'HL_Flag', 'GL_Flag'
]

# Define the target variable
train_df['is_default'] = (train_df['num_times_delinquent'].fillna(0) > 0).astype(int)

# Calculate medians for imputation and save them
repayment_medians = train_df[repayment_features].median().to_dict()
with open('repayment_medians.json', 'w') as f:
    json.dump(repayment_medians, f)
print("Repayment medians saved.")

# Fill missing values
for col, median in repayment_medians.items():
    train_df[col].fillna(median, inplace=True)

X_repayment = train_df[repayment_features]
y_repayment = train_df['is_default']

# Train the final XGBoost model
weight_ratio = y_repayment.value_counts()[0] / y_repayment.value_counts()[1]
repayment_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, 
                                    eval_metric='logloss', scale_pos_weight=weight_ratio)
repayment_model.fit(X_repayment, y_repayment)

# Save the model and feature list
joblib.dump(repayment_model, 'repayment_model.joblib')
with open('repayment_features.json', 'w') as f:
    json.dump(repayment_features, f)
print("Repayment Model (Model A) trained and saved.")


# --- 2. Train and Save Placeholder Model B (Income Score) ---

print("\n--- Training Model B: Income Score Model ---")
# Define features for the income model based on dashboard inputs
income_features = [
    'person_age', 'household_size_calculated', 'avg_education_years_adults',
    'NETMONTHLYINCOME', 'Time_With_Curr_Empr', 'Possess_Car', 'Possess_Refrigerator', 
    'Possess_WashingMachine' 
    # Add other categorical features here after one-hot encoding if desired
]

# Create a placeholder target for income categories
# This is a simplified logic for demonstration
bins = [0, 15000, 30000, 50000, np.inf]
labels = ['Very Low', 'Low', 'Medium', 'High']
train_df['income_category'] = pd.cut(train_df['NETMONTHLYINCOME'], bins=bins, labels=labels, right=False)

# For this placeholder, we'll just use a few features that exist in the original data
# In a real scenario, you'd have data for all dashboard features
placeholder_income_features = ['NETMONTHLYINCOME', 'Time_With_Curr_Empr']
train_df.rename(columns={'AGE': 'person_age'}, inplace=True) # Align one column for demonstration

# Fill NaNs for the target and features
train_df['income_category'].fillna('Low', inplace=True)
for col in placeholder_income_features:
    train_df[col].fillna(train_df[col].median(), inplace=True)

X_income = train_df[placeholder_income_features]
y_income = train_df['income_category']

# Train a simple RandomForest model
income_model = RandomForestClassifier(random_state=42, class_weight='balanced')
income_model.fit(X_income, y_income)

# Save the model and the full feature list it expects
joblib.dump(income_model, 'income_model.joblib')
with open('income_features.json', 'w') as f:
    json.dump(income_features, f) # Save the list of features the dashboard will provide
print("Income Model (Model B) trained and saved.")
print("\n--- All models and assets are ready! ---")
