import chardet

# 전력수급 파일 인코딩 확인
with open('전력수급_날짜변환완료.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    print(f"전력수급_날짜변환완료.csv 인코딩: {result}")

# merged_result 파일 인코딩 확인
with open('merged_result 1.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    print(f"merged_result 1.csv 인코딩: {result}") 