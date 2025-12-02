import pandas as pd

# 데이터 읽기
df = pd.read_csv('전력_날씨_통합데이터_선택변수.csv')

# 결측치 처리: 연속형 변수는 ffill, 나머지는 0
columns_ffill = [
    '평균기온(°C)', '최저기온(°C)', '최고기온(°C)',
    '일강수량(mm)', '평균 풍속(m/s)', '최대 풍속(m/s)',
    '평균 상대습도(%)', '평균 증기압(hPa)', '합계 일조시간(hr)'
]
for col in columns_ffill:
    if col in df.columns:
        df[col] = df[col].fillna(method='ffill')

# 나머지 결측치는 0으로 채움
df = df.fillna(0)

# 처리 후 저장
output_file = '전력_날씨_통합데이터_선택변수_결측치처리.csv'
df.to_csv(output_file, index=False)
print(f"결측치 처리 완료! 파일이 생성되었습니다: {output_file}") 