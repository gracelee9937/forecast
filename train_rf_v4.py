import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 데이터 불러오기 (encoding은 상황에 따라 cp949 또는 utf-8로 변경)
df = pd.read_csv('final_v4.csv', encoding='utf-8')

# 컬럼명 확인 및 feature/target 분리
# '일시' 컬럼명이 깨져있으므로 첫 번째 컬럼을 사용
first_col = df.columns[0]
X = df.drop([first_col, 'maxElec_diff'], axis=1)
y = df['maxElec_diff']

# 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 모델 저장
joblib.dump(model, 'model_rf_v4.pkl')

print('모델 학습 및 저장 완료: model_rf_v4.pkl') 