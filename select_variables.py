import pandas as pd

# 데이터 읽기
df = pd.read_csv('전력_날씨_통합데이터_v2.csv')

# 선택할 변수 리스트
variables = [
    '년월', '지역',  # '일시' 대신 '년월' 사용
    '평균기온(°C)', '최저기온(°C)', '최저기온 시각(hhmi)', '최고기온(°C)',
    '일강수량(mm)',
    '평균 풍속(m/s)', '최대 풍속(m/s)',
    '평균 상대습도(%)',
    '평균 증기압(hPa)',
    '합계 일조시간(hr)'
]

# 선택된 변수만 추출
selected_df = df[variables]

# 새로운 파일로 저장
selected_df.to_csv('전력_날씨_통합데이터_선택변수.csv', index=False)
print("파일이 생성되었습니다: 전력_날씨_통합데이터_선택변수.csv")

# 선택된 변수 목록 출력
print("\n선택된 변수 목록:")
for var in variables:
    print(var) 