# ==============================
# Used Car Price Prediction 
# ==============================

# Step 1: Import Libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Suppress FutureWarnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load Datasets
train = pd.read_csv('/kaggle/input/playground-series-s4e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e9/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')

# Step 3: Handle Missing / Infinite Values
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)
train.fillna('missing', inplace=True)
test.fillna('missing', inplace=True)

# Step 4: Identify Target Column
target = 'price'
X = train.drop(target, axis=1)
y = train[target]

# Drop ID column if exists
if 'id' in X.columns:
    X = X.drop('id', axis=1)
    test = test.drop('id', axis=1)

# Step 5: Encode Categorical Columns
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    combined = pd.concat([X[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    X[col] = le.transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Step 6: Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train RandomForest Regressor
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Step 8: Evaluate Model - Metrics
val_preds = model.predict(X_valid)

mae = mean_absolute_error(y_valid, val_preds)
mse = mean_squared_error(y_valid, val_preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, val_preds)

print("📊 Validation Metrics:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==============================
# Step 9: Important Graphs
# ==============================

# Ensure numeric target for plotting
train_numeric = train.copy()
train_numeric[target] = pd.to_numeric(train_numeric[target], errors='coerce')

# 1️⃣ Target Price Distribution
plt.figure(figsize=(8,4))
sns.histplot(train_numeric[target].dropna(), bins=40, kde=True, color='teal')
plt.title("Distribution of Used Car Prices")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()

# 2️⃣ Random Forest Feature Importance
importances = pd.Series(model.feature_importances_, index=X_train.columns)
plt.figure(figsize=(10,5))
importances.sort_values(ascending=False)[:15].plot(kind='bar', color='orange')
plt.title("Top 15 Features Affecting Car Price")
plt.ylabel("Importance Score")
plt.show()

# Step 10: Predict Test Data & Create Submission
test_preds = model.predict(test)
submission = sample_submission.copy()
submission[submission.columns[-1]] = test_preds
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("✅ Submission file created!")
print(submission.head())
