import pandas as pd
import numpy as np

# 데이터 읽기
df = pd.read_csv('전력_날씨_통합데이터_v2.csv')

# 컬럼명 출력
print('컬럼명 목록:')
for col in df.columns:
    print(col)

# 결측치 확인
print("=== 결측치 현황 ===")
print(df.isnull().sum())

# 결측치 처리
# 1. 날씨 관련 컬럼의 결측치는 전날 같은 시간대의 값으로 채우기
weather_columns = [
    '평균기온(°C)', '최저기온(°C)', '최고기온(°C)',
    '강수 계속시간(hr)', '10분 최다 강수량(mm)', '1시간 최다강수량(mm)', '일강수량(mm)',
    '최대 순간 풍속(m/s)', '최대 풍속(m/s)', '평균 풍속(m/s)',
    '평균 상대습도(%)', '평균 현지기압(hPa)', '평균 해면기압(hPa)',
    '합계 일조시간(hr)', '합계 일사량(MJ/m2)',
    '평균 지면온도(°C)', '평균 5cm 지중온도(°C)', '평균 10cm 지중온도(°C)',
    '평균 20cm 지중온도(°C)', '평균 30cm 지중온도(°C)',
    '0.5m 지중온도(°C)', '1.0m 지중온도(°C)', '1.5m 지중온도(°C)',
    '3.0m 지중온도(°C)', '5.0m 지중온도(°C)',
    '합계 대형증발량(mm)', '합계 소형증발량(mm)', '9-9강수(mm)',
    '안개 계속시간(hr)'
]

for col in weather_columns:
    if col in df.columns:
        df[col] = df[col].fillna(method='ffill')

# 2. 전력 관련 컬럼의 결측치는 0으로 채우기
power_columns = [
    '고객호수(호)', '평균판매단가(원/kWh)', '전력판매단가(원/kWh)',
    '실질판매단가_CPI', '실질판매단가_PPI',
    '전력판매단가_CPI', '전력판매단가_PPI'
]

for col in power_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 3. 시간 관련 컬럼의 결측치는 '0000'으로 채우기
time_columns = [
    '최저기온 시각(hhmi)', '최고기온 시각(hhmi)',
    '10분 최다강수량 시각(hhmi)', '1시간 최다 강수량 시각(hhmi)',
    '최대 순간풍속 시각(hhmi)', '최대 풍속 시각(hhmi)',
    '최소 상대습도 시각(hhmi)', '최고 해면기압 시각(hhmi)',
    '최저 해면기압 시각(hhmi)', '1시간 최다일사 시각(hhmi)',
    '일 최심신적설 시각(hhmi)', '일 최심적설 시각(hhmi)'
]

for col in time_columns:
    if col in df.columns:
        df[col] = df[col].fillna('0000')

# 4. 방향 관련 컬럼의 결측치는 0으로 채우기
direction_columns = [
    '최대 순간 풍속 풍향(16방위)', '최대 풍속 풍향(16방위)',
    '최다풍향(16방위)'
]

for col in direction_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 5. 남은 결측치가 있다면 0으로 채우기
df = df.fillna(0)

# 처리 후 결측치 확인
print("\n=== 결측치 처리 후 현황 ===")
print(df.isnull().sum())

# 처리된 데이터 저장
df.to_csv('전력_날씨_통합데이터_v2_결측치처리.csv', index=False)
print("\n결측치 처리 완료! '전력_날씨_통합데이터_v2_결측치처리.csv' 파일이 생성되었습니다.") 