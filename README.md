#  FSI Threat Intelligence Dashboard

이 저장소는 금융 이상거래 탐지를 위한 **데모 대시보드**입니다.  
Streamlit 기반으로 거래 데이터를 합성하고, 이상탐지 모델을 적용하며,  
설명 가능성(Explainability) 및 위협 인텔 연동까지 한눈에 체험할 수 있습니다.  

---

##  주요 기능
- **합성 거래 시뮬레이터**
  - 계정, 금액, 위치, 디바이스, IP 등 속성을 가진 가상 거래 생성
  - 정상 + 이상 거래(비정상 금액, 빠른 이동 등)를 자동 삽입
- **특징 추출 (Feature Engineering)**
  - 시간 패턴(sin/cos), 거래 간격, 거리/속도, 금액 잔차, 기기 변경 등
- **이상탐지 모델링**
  - Isolation Forest 기반
  - **Top N 다거래 계정 전용 모델 + 공용(Global) 모델** 세분화 전략
  - 분위수 기반 threshold로 이상 여부 판단
- **실시간 모드 (Live Mode)**
  - Streamlit 자동 새로고침으로 거래 스트리밍 시뮬레이션
- **설명 가능성 (Explainability)**
  - XGBoost surrogate + SHAP으로 이상탐지 근거 시각화
- **위협 인텔 연동**
  - VirusTotal API를 통한 URL IoC 조회
- **라벨링 & 챗봇**
  - 탐지 거래 라벨링 CSV 저장
  - 업로드된 라벨 데이터를 기반으로 LLM 챗봇 요약/질의응답
- **운영 알림**
  - Slack Webhook을 통한 이상치 경보 발송

---

##  프로젝트 구조
FSI/
├─ app/
│ ├─ combined_full.py # 메인 대시보드
│ ├─ labeling_ui.py # 라벨링 UI
│ ├─ pages/fsi_chat_bot.py # 챗봇 페이지
│ ├─ requirements.txt # 의존성
│ └─ data/labels.csv # 샘플 라벨 데이터
└─ core/
├─ feature_engineering.py # 특징 추출
├─ detectors.py # IForest/LOF
├─ thresholding.py # 임계치 방법
├─ drift.py # 드리프트 감지
└─ rules.py # 규칙 기반 가점

##  설치 및 실행

### 의존성 설치
```bash
pip install -r app/requirements.txt
