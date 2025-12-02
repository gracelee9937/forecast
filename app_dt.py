from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

def safe_float(val, default=0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

# Load the trained model and data
dt_model = joblib.load('model_dt.pkl')
df = pd.read_csv('final_v4.csv')

# Load or create MinMaxScaler for target variable (예측 결과 역정규화용)
try:
    # 기존에 저장된 target scaler가 있으면 로드
    target_scaler = joblib.load('target_scaler.pkl')
except:
    # 없으면 새로 생성하고 target 데이터로 fit
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler.fit(df[['maxElec_diff']])
    # target scaler 저장
    joblib.dump(target_scaler, 'target_scaler.pkl')

# Feature columns used in the model
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

def get_historical_average(df, date, col):
    """Get historical average for a specific date, fallback to global mean if NaN"""
    mask = (df['일시'] < date)
    avg = df.loc[mask, col].mean()
    if pd.isna(avg):
        avg = df[col].mean()
    return avg

@app.route('/')
def main():
    return render_template('home_dt.html')

@app.route('/dt')
def home():
    return render_template('dt_index.html')

@app.route('/dt/predict', methods=['POST'])
def predict():
    try:
        # Get date from request
        date_str = request.form['date']
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Prepare input data
        input_data = {
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'month_sin': np.sin(2 * np.pi * date.month / 12),
            'weekday_sin': np.sin(2 * np.pi * date.weekday() / 7),
            'relative_humidity': safe_float(request.form.get('relative_humidity')),
            'vapor_pressure': safe_float(request.form.get('vapor_pressure')),
            'coolingHeating': safe_float(request.form.get('coolingHeating')),
            'ms_diff': safe_float(request.form.get('ms_diff')),
            'hr_diff': safe_float(request.form.get('hr_diff')),
            'CPI_diff': safe_float(request.form.get('CPI_diff')),
            'daily_rainfall': safe_float(request.form.get('daily_rainfall')),
            'seasons_2': 1 if date.month in [4, 5, 6] else 0,
            'seasons_3': 1 if date.month in [7, 8, 9] else 0,
            'seasons_4': 1 if date.month in [10, 11, 12] else 0
        }
        
        # Create DataFrame for prediction (피처는 이미 정규화되어 있음)
        X_pred = pd.DataFrame([input_data])
        
        # Make prediction (결과는 (-1, 1) 범위로 정규화된 상태)
        prediction_scaled = dt_model.predict(X_pred)[0]
        
        # 예측 결과를 (-1, 1) 범위에서 역정규화
        prediction_scaled_reshaped = np.array([[prediction_scaled]])
        prediction = target_scaler.inverse_transform(prediction_scaled_reshaped)[0][0]
        
        # 역정규화된 prediction 값에 69168 더하기
        prediction = prediction + 69168
        
        # Calculate supply recommendation (add 12.5%)
        supply_recommendation = prediction * 1.125

        # Get actual value if available
        actual_value = None
        error_abs = None
        error_pct = None
        is_future_date = date > datetime.now()

        if not is_future_date:
            # Try to get actual value from the dataset
            mask = (pd.to_datetime(df['일시']).dt.date == date.date())
            if mask.any():
                actual_value = df.loc[mask, 'maxElec_diff'].iloc[0]
                error_abs = abs(prediction - actual_value)
                error_pct = (error_abs / actual_value) * 100 if actual_value != 0 else None

        # Chart.js용 데이터 준비
        historical_data = df.copy()
        historical_data['date'] = pd.to_datetime(historical_data['일시'])
        historical_data = historical_data[historical_data['date'] >= date - pd.DateOffset(years=1)]
        line_labels = historical_data['date'].dt.strftime('%Y-%m-%d').tolist()
        line_values = historical_data['maxElec_diff'].tolist()

        # Calculate model performance metrics
        recent_data = df.copy()
        recent_data['date'] = pd.to_datetime(recent_data['일시'])
        recent_data = recent_data[recent_data['date'] <= datetime.now()]
        
        if len(recent_data) > 0:
            X_recent = recent_data[feature_cols]
            
            # 예측 (피처는 이미 정규화되어 있음)
            y_pred_scaled = dt_model.predict(X_recent)
            
            # 예측 결과 역정규화
            y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)
            y_pred = target_scaler.inverse_transform(y_pred_scaled_reshaped).flatten()
            
            # 역정규화된 예측값에 69168 더하기
            y_pred = y_pred + 69168
            
            y_true = recent_data['maxElec_diff']
            
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # R2 계산을 위해 정규화된 y_true 필요
            y_true_scaled = target_scaler.transform(y_true.values.reshape(-1, 1)).flatten()
            r2 = dt_model.score(X_recent, y_true_scaled)
        else:
            mae = rmse = mape = r2 = None

        # 실제값이 있는 최근 날짜 추천
        recommend_date = None
        available_dates = df.dropna(subset=['maxElec_diff'])['일시']
        if not available_dates.empty:
            recommend_date = pd.to_datetime(available_dates.iloc[-1]).strftime('%Y-%m-%d')

        # Feature importance
        feature_importance = {
            'labels': feature_cols,
            'values': dt_model.feature_importances_.tolist()
        }

        return jsonify({
            'prediction': float(prediction),
            'supply_recommendation': float(supply_recommendation),
            'actual_value': float(actual_value) if actual_value is not None else None,
            'error_abs': float(error_abs) if error_abs is not None else None,
            'error_pct': float(error_pct) if error_pct is not None else None,
            'is_future_date': bool(is_future_date),
            'line_labels': [str(label) for label in line_labels],
            'line_values': [float(val) for val in line_values],
            'pie_values': [float(supply_recommendation), float(prediction)],
            'model_metrics': {
                'mae': float(mae) if mae is not None else None,
                'rmse': float(rmse) if rmse is not None else None,
                'mape': float(mape) if mape is not None else None,
                'r2': float(r2) if r2 is not None else None
            },
            'recommend_date': str(recommend_date) if recommend_date is not None else None,
            'feature_importance': feature_importance,
            # 디버깅을 위한 정보 추가
            'debug_info': {
                'original_features': input_data,
                'prediction_scaled': float(prediction_scaled),
                'prediction_unscaled': float(prediction - 69168),
                'prediction_final': float(prediction)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5001) 