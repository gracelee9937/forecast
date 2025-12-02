import pandas as pd

# 전력수급 파일 변환
try:
    elec = pd.read_csv('전력수급_날짜변환완료.csv', encoding='utf-8')
    elec.to_csv('전력수급_날짜변환완료_utf8.csv', encoding='utf-8', index=False)
    print('전력수급_날짜변환완료.csv 변환 완료')
except Exception as e:
    print(f'전력수급_날짜변환완료.csv 변환 실패: {e}')

# price 파일 변환
try:
    price = pd.read_csv('merged_result 1.csv', encoding='cp949')
    price.to_csv('merged_result_utf8.csv', encoding='utf-8', index=False)
    print('merged_result 1.csv 변환 완료')
except Exception as e:
    print(f'merged_result 1.csv 변환 실패: {e}') 