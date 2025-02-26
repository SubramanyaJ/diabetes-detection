import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# File paths
DATA_PATH = 'data.csv'
MODEL_PATH = 'xgb_model.pkl'
FEATURES_PATH = 'feature_names.pkl'

# Load dataset
data = pd.read_csv(DATA_PATH)

# Feature-target separation
X = data.drop(columns=['Target'])
y = data['Target']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Apply SMOTE to balance classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# XGBoost parameters
xgb_params = {
    'n_estimators': 96,
    'max_depth': 6,
    'min_child_weight': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'scale_pos_weight': 1,
    'eval_metric': 'mlogloss',
    'random_state': 42
}

# Train and save model
xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train_res, y_train_res)

# Save model and feature names
joblib.dump(xgb, MODEL_PATH)
joblib.dump(X.columns.tolist(), FEATURES_PATH)
print("Model trained and saved successfully.")

# Load model and features
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# Ask user for input
print("Enter values for the following features:")
user_input = []
for feature in feature_names:
    value = float(input(f"{feature}: "))
    user_input.append(value)

# Convert input to numpy array and reshape
user_array = np.array(user_input).reshape(1, -1)

# Predict
prediction = model.predict(user_array)
print(f"Predicted diabetes type: {prediction[0]}")
