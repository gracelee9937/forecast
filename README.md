# 전력 예측 프로젝트 (Power Forecast)

전력 사용량 예측을 위한 머신러닝 프로젝트입니다.

## 프로젝트 개요

이 프로젝트는 과거 전력 사용량 데이터를 기반으로 미래 전력 사용량을 예측하는 웹 애플리케이션입니다.

## 주요 기능

- 전력 사용량 데이터 분석 및 전처리
- 머신러닝 모델을 활용한 전력 사용량 예측
- Flask 기반 웹 애플리케이션
- 시각화를 통한 예측 결과 제공

## 기술 스택

- Python 3.x
- Flask
- pandas
- scikit-learn
- plotly

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/gracelee9937/forecast.git
cd forecast
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python app.py
```

또는 특정 버전 실행:
```bash
python app_v4.py
python app_dt.py
```

## 프로젝트 구조

```
.
├── app.py              # 메인 애플리케이션
├── app_v4.py           # 버전 4 애플리케이션
├── app_dt.py           # 의사결정트리 모델 애플리케이션
├── train_rf_v4.py     # 랜덤포레스트 모델 학습
├── train_dt_model.py   # 의사결정트리 모델 학습
├── powercast/          # 배포용 디렉토리
├── templates/          # HTML 템플릿
├── static/             # 정적 파일 (CSS, JS, 이미지)
└── requirements.txt    # 의존성 목록
```





