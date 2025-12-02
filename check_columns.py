import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('final_v4.csv')

# 컬럼명 출력
print("컬럼명 목록:")
for col in df.columns:
    print(f"- {col}") 