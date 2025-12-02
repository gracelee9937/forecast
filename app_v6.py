from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

app = Flask(__name__)

# Load the trained model and data (v6)
model = joblib.load('model_rf_v4.pkl')  # v6 모델이 학습되면 경로 수정 필요
df = pd.read_csv('final_v4.csv', encoding='utf-8')
first_col = df.columns[0]  # '일시' 컬럼(혹은 깨진 한글)

# Feature columns: '일시', 'maxElec_diff' 제외
feature_cols = [col for col in df.columns if col not in [first_col, 'maxElec_diff']]

def get_group(name):
    if name.startswith('year_'):
        return 'year'
    elif name.startswith('seasons_'):
        return 'season'
    else:
        return 'other'

def get_historical_average(df, year, month, day, col):
    if not all(x in df.columns for x in ['year', 'month', 'day']):
        if '일시_dt' not in df.columns:
            df['일시_dt'] = pd.to_datetime(df[df.columns[0]])
        mask = (df['일시_dt'].dt.year < year) & (df['일시_dt'].dt.month == month) & (df['일시_dt'].dt.day == day)
    else:
        mask = (df['year'] < year) & (df['month'] == month) & (df['day'] == day)
    avg = df.loc[mask, col].mean()
    if pd.isna(avg):
        avg = df[col].mean()
    return avg

@app.route('/v6')
def v6_home():
    return render_template('v6_home.html')

@app.route('/v6/predict', methods=['GET'])
def v6_predict_form():
    return render_template('v6_predict.html')

@app.route('/v6/predict', methods=['POST'])
def v6_predict():
    try:
        # 1. 입력값 받기
        date_str = request.form['date']
        try:
            date_obj = pd.to_datetime(date_str)
            date_str = date_obj.strftime('%Y-%m-%d')
        except:
            return jsonify({'error': '날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요.'}), 400
        
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        weekday = date_obj.weekday()

        # 2. feature 자동 생성
        input_data = {}
        # (1) 연도 더미(year_2002~year_2025)
        year_cols = [col for col in feature_cols if col.startswith('year_')]
        year_in_range = False
        for col in year_cols:
            col_year = int(col.split('_')[1])
            if col_year == year:
                input_data[col] = 1.0
                year_in_range = True
            else:
                input_data[col] = 0.0
        if not year_in_range:
            for col in year_cols:
                input_data[col] = 0.0

        # (2) 날짜 파생 변수
        if 'month_sin' in feature_cols:
            input_data['month_sin'] = np.sin(2 * np.pi * month / 12)
        if 'weekday_sin' in feature_cols:
            input_data['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
        if 'is_weekend' in feature_cols:
            input_data['is_weekend'] = 1.0 if weekday >= 5 else 0.0

        # (3) 계절 더미
        for col in feature_cols:
            if col.startswith('seasons_'):
                col_season = int(col.split('_')[1])
                if month in [3,4,5]: season = 1
                elif month in [6,7,8]: season = 2
                elif month in [9,10,11]: season = 3
                else: season = 4
                input_data[col] = 1.0 if col_season == season else 0.0

        # (4) 날씨 및 기타 feature 기본값
        defaults = {
            'daily_rainfall': 0.0,
            'coolingHeating': 0.0,
            'relative_humidity': 0.5,
            'vapor_pressure': 0.5,
            'CPI_diff': 0.0,
            'ms_diff': 0.0,
            'hr_diff': 0.0
        }
        for col, val in defaults.items():
            if col in feature_cols:
                input_data[col] = val

        # (5) 나머지 feature 결측치 처리
        for col in feature_cols:
            if col not in input_data:
                hist_avg = get_historical_average(df, year, month, day, col)
                input_data[col] = hist_avg

        # 3. 예측
        X_pred = pd.DataFrame([input_data])
        prediction = model.predict(X_pred)[0]

        # 4. 결과 계산
        if prediction is None or (isinstance(prediction, float) and np.isnan(prediction)):
            prediction = None
            predicted_diff = None
            predicted_maxElec = None
            recommended_supply_10 = None
            recommended_supply_15 = None
        else:
            predicted_diff = prediction

            # 역차분
            if 'last_maxElec' in df.columns:
                last_maxElec = df['last_maxElec'].iloc[-1]
            else:
                last_maxElec = 0.0

            if 'maxElec' in df.columns:
                last_maxElec = df['maxElec'].iloc[-1]

            predicted_maxElec = last_maxElec + predicted_diff
            recommended_supply_10 = predicted_maxElec * 1.10
            recommended_supply_15 = predicted_maxElec * 1.15

        # 5. Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importances = np.array(model.feature_importances_)
            feature_labels = [feature_cols[i] for i in range(len(importances))]
            feature_groups = [get_group(f) for f in feature_labels]
            indices = np.argsort(importances)
            feature_importance = {
                'labels': feature_labels,
                'values': [float(importances[i]) for i in indices],
                'groups': [feature_groups[i] for i in indices]
            }

        # 6. 모델 성능 지표
        recent_df = df.copy()
        if '일시_dt' not in recent_df.columns:
            recent_df['일시_dt'] = pd.to_datetime(recent_df[first_col])
        recent_df = recent_df.sort_values('일시_dt')
        
        if len(recent_df) > 0:
            X_recent = recent_df[feature_cols]
            y_true = recent_df['maxElec_diff']
            y_pred = model.predict(X_recent)
            mae = float(np.mean(np.abs(y_pred - y_true)))
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            model_metrics = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            }
        else:
            model_metrics = {
                'mae': None, 'rmse': None, 'mape': None, 'r2': None
            }

        # 7. 실제값 추출 및 오차 계산
        df['일시_dt'] = pd.to_datetime(df[first_col])
        actual_row = df[df['일시_dt'] == date_obj]
        if not actual_row.empty and 'maxElec' in actual_row.columns:
            actual_value = actual_row['maxElec'].values[0]
        elif not actual_row.empty and 'maxElec_diff' in actual_row.columns:
            actual_value = actual_row['maxElec_diff'].values[0]
        else:
            actual_value = None

        error_abs = None
        error_pct = None
        if actual_value is not None and predicted_maxElec is not None:
            error_abs = abs(predicted_maxElec - actual_value)
            error_pct = (error_abs / actual_value) * 100 if actual_value != 0 else None

        return jsonify({
            'prediction': predicted_maxElec,
            'predicted_diff': predicted_diff,
            'recommended_supply_10': recommended_supply_10,
            'recommended_supply_15': recommended_supply_15,
            'used_features': input_data,
            'feature_importance': feature_importance,
            'model_metrics': model_metrics,
            'actual_value': actual_value,
            'error_abs': error_abs,
            'error_pct': error_pct
        })

    except Exception as e:
        return jsonify({'error': f'예측 중 오류가 발생했습니다: {str(e)}'}), 400

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5051, debug=True) 