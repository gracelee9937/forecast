import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import joblib

# 데이터 불러오기 (encoding은 상황에 따라 cp949 또는 utf-8로 변경)
df = pd.read_csv("final_v4.csv", index_col='일시', parse_dates=True)

# 컬럼명 확인 및 feature/target 분리
# '일시' 컬럼명이 깨져있으므로 첫 번째 컬럼을 사용
first_col = df.columns[0]
X = df.drop([first_col, 'maxElec_diff'], axis=1)
y = df['maxElec_diff']

def split_X_y_80_20(X, y):
    """
    X, y 데이터를 8:2 비율로 분할
    """
    split_idx = int(len(X) * 0.8)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test

# 사용 예시
X_train, X_test, y_train, y_test = split_X_y_80_20(X, y)

# 모델 학습
model = DecisionTreeRegressor(criterion= 'absolute_error', max_depth = 10, min_samples_leaf = 4, min_samples_split = 10)
model.fit(X_train, y_train)

# 모델 저장
joblib.dump(model, 'model_rf_v4.pkl')

print('모델 학습 및 저장 완료: model_rf_v4.pkl') 