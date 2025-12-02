import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import joblib

# Load the data
df = pd.read_csv('final_v4.csv')

# Define feature columns (excluding target variable)
feature_cols = [
    'is_weekend',
    'month_sin',
    'weekday_sin',
    'relative_humidity',
    'vapor_pressure',
    'coolingHeating',
    'ms_diff',
    'hr_diff',
    'CPI_diff',
    'daily_rainfall',
    'seasons_2',
    'seasons_3',
    'seasons_4'
]

# Prepare X and y
X = df[feature_cols]
y = df['maxElec_diff']

# Create and train the model
dt_model = DecisionTreeRegressor(
    criterion='absolute_error',
    max_depth=10,
    min_samples_leaf=4,
    min_samples_split=10
)

# Train the model
dt_model.fit(X, y)

# Save the model
joblib.dump(dt_model, 'model_dt.pkl')

# Print model performance metrics
y_pred = dt_model.predict(X)
mae = np.mean(np.abs(y - y_pred))
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
mape = np.mean(np.abs((y - y_pred) / y)) * 100
r2 = dt_model.score(X, y)

print(f"Model Performance Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R2 Score: {r2:.4f}")

# Print feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': dt_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('importance', ascending=False)) 