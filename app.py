from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

app = Flask(__name__)
@app.route('/heatmap-data')
def heatmap_data():
    # '일시' 컬럼이 'YYYY-MM-DD HH:MM:SS' 형식이라고 가정
    df['일시'] = pd.to_datetime(df['일시'])
    df['hour'] = df['일시'].dt.hour
    df['weekday'] = df['일시'].dt.weekday  # 0=월, 6=일

    # 시간별 데이터 개수 분포 출력
    print('시간별 데이터 개수:', json.dumps(df['hour'].value_counts().sort_index().to_dict(), ensure_ascii=False))

    # 요일/시간 전체 인덱스
    all_weekdays = list(range(7))
    all_hours = list(range(24))

    # 피벗테이블 생성 (평균)
    pivot = df.pivot_table(index='weekday', columns='hour', values='최대전력(MW)', aggfunc='mean')
    pivot = pivot.reindex(index=all_weekdays, columns=all_hours)

    # 1차: 시간별 전체 평균으로 결측치 채우기
    for hour in all_hours:
        hour_mean = df[df['hour'] == hour]['최대전력(MW)'].mean()
        pivot[hour] = pivot[hour].fillna(hour_mean)
    # 2차: 요일별 평균으로 결측치 채우기
    pivot = pivot.apply(lambda row: row.fillna(row.mean()), axis=1)
    # 3차: 전체 평균으로 결측치 채우기
    overall_mean = df['최대전력(MW)'].mean()
    pivot = pivot.fillna(overall_mean)

    # shape 점검 및 로그
    print('pivot shape:', pivot.shape)  # (7, 24)여야 정상

    # 요일 한글 라벨
    days = ['월', '화', '수', '목', '금', '토', '일']
    values = pivot.values.tolist()

    data = {
        "hours": all_hours,
        "days": days,
        "values": values
    }
    return jsonify(data)

@app.route('/correlation-heatmap-data')
def correlation_heatmap_data():
    # 모든 수치형 변수만 추출
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 상관계수 행렬 계산
    corr_df = df[numeric_cols].corr(method='pearson')
    corr_matrix = corr_df.fillna(0).values.tolist()
    return jsonify({
        'labels': corr_df.columns.tolist(),
        'matrix': corr_matrix
    })

# Load the trained model and data
rf_model = joblib.load('model_rf.pkl')
df = pd.read_csv('cleaned_power_data.csv')

# Feature columns used in the model
feature_cols = [
    'year', 'month', 'day',
    '일강수량(mm)_norm',
    '평균 풍속(m/s)_norm',
    '최대 풍속(m/s)_norm',
    '평균기온(°C)_norm',
    '최저기온(°C)_norm',
    '최고기온(°C)_norm'
]

def get_historical_average(df, year, month, day, col):
    """Get historical average for a specific date, fallback to global mean if NaN"""
    mask = (df['year'] < year) & (df['month'] == month) & (df['day'] == day)
    avg = df.loc[mask, col].mean()
    if pd.isna(avg):
        avg = df[col].mean()
    return avg

def normalize_value(value, col):
    min_val = df[col].min()
    max_val = df[col].max()
    if pd.isna(value) or pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return 0.5  # fallback default
    return (value - min_val) / (max_val - min_val)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get date from request
        date_str = request.form['date']
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Prepare input data
        input_data = {
            'year': date.year,
            'month': date.month,
            'day': date.day
        }
        
        # Handle optional weather inputs
        weather_cols = {
            '일강수량(mm)_norm': '일강수량(mm)_norm',
            '평균 풍속(m/s)_norm': '평균 풍속(m/s)_norm',
            '최대 풍속(m/s)_norm': '최대 풍속(m/s)_norm',
            '평균기온(°C)_norm': '평균기온(°C)_norm',
            '최저기온(°C)_norm': '최저기온(°C)_norm',
            '최고기온(°C)_norm': '최고기온(°C)_norm'
        }
        
        for col, norm_col in weather_cols.items():
            if col in request.form and request.form[col]:
                value = float(request.form[col])
                input_data[norm_col] = normalize_value(value, col)
            else:
                # Get historical average and normalize, fallback to global mean if NaN
                hist_avg = get_historical_average(df, date.year, date.month, date.day, col)
                input_data[norm_col] = normalize_value(hist_avg, col)
        
        # Create DataFrame for prediction
        X_pred = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = rf_model.predict(X_pred)[0]
        # 예측 신뢰도(트리별 예측 표준편차)
        if hasattr(rf_model, 'estimators_'):
            tree_preds = np.array([est.predict(X_pred)[0] for est in rf_model.estimators_])
            confidence = float(np.std(tree_preds))
        else:
            confidence = None
        # Calculate supply recommendation (add 12.5%)
        supply_recommendation = prediction * 1.125

        # Get actual value if available
        actual_value = None
        error_abs = None
        error_pct = None
        is_future_date = date > datetime.now()

        if not is_future_date:
            # Try to get actual value from the dataset
            mask = (df['year'] == date.year) & (df['month'] == date.month) & (df['day'] == date.day)
            if mask.any():
                actual_value = df.loc[mask, '최대전력(MW)'].iloc[0]
                error_abs = abs(prediction - actual_value)
                error_pct = (error_abs / actual_value) * 100 if actual_value != 0 else None

        # Chart.js용 데이터 준비
        historical_data = df[df['year'] >= date.year - 1].copy()
        historical_data['date'] = pd.to_datetime(historical_data[['year', 'month', 'day']])
        line_labels = historical_data['date'].dt.strftime('%Y-%m-%d').tolist()
        line_values = historical_data['최대전력(MW)'].tolist()

        # Calculate model performance metrics
        recent_data = df[df['year'] >= date.year - 1].copy()
        recent_data['date'] = pd.to_datetime(recent_data[['year', 'month', 'day']])
        recent_data = recent_data[recent_data['date'] <= datetime.now()]
        
        if len(recent_data) > 0:
            X_recent = recent_data[feature_cols]
            y_pred = rf_model.predict(X_recent)
            y_true = recent_data['최대전력(MW)']
            
            mae = np.mean(np.abs(y_pred - y_true))
            rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            # R2 score
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else None
        else:
            mae = rmse = mape = r2 = None

        # 실제값이 있는 최근 날짜 추천
        recommend_date = None
        available_dates = df.dropna(subset=['최대전력(MW)'])[['year', 'month', 'day']]
        if not available_dates.empty:
            # 가장 최근 날짜를 찾음
            last_row = available_dates.iloc[-1]
            recommend_date = f"{int(last_row['year']):04d}-{int(last_row['month']):02d}-{int(last_row['day']):02d}"

        # Feature importance (랜덤포레스트 모델에서만 지원)
        feature_importance = None
        if hasattr(rf_model, 'feature_importances_'):
            importances = np.array(rf_model.feature_importances_)
            indices = np.argsort(importances)  # 오름차순 정렬
            feature_importance = {
                'labels': [feature_cols[i] for i in indices],
                'values': [float(importances[i]) for i in indices]
            }

        return jsonify({
            'prediction': float(prediction),
            'confidence': confidence,
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
            'feature_importance': feature_importance
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

if __name__ == '__main__':
    app.run(debug=True) 
    print(df.columns.tolist())

