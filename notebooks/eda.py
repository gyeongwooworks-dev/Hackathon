import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import warnings

# 기초 설정
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 시각화 한글 깨짐 방지
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터 로드 (경로 내 r 붙여서 에러 방지) - 파일 경로 바꿔주세요
file_path = r'C:/Users/anywa/OneDrive/바탕 화면/ai_project/data/train.csv'
df = pd.read_csv(file_path)

# 컬럼 공백 제거 
df.columns = [c.strip() for c in df.columns]

# 1. 결측치, 이상치 확인  

print(df.describe())
print(f"데이터 크기: {df.shape}")
print(df.info())

# 결측치 비율 확인 (상위 5개)
missing_series = df.isnull().sum().sort_values(ascending=False)
print("결측치 상위 컬럼:\n", missing_series.head(10))

# 타겟 변수(임신 성공 여부) 불균형 확인
print("타겟 클래스 비율:\n", df['임신 성공 여부'].value_counts(normalize=True))


# 중복된 데이터 확인
print(df.duplicated().sum())

df_columns = df[[
    'ID', '시술 시기 코드', '시술 당시 나이', '임신 시도 또는 마지막 임신 경과 연수', '시술 유형',
    '특정 시술 유형', '배란 자극 여부', '배란 유도 유형', '단일 배아 이식 여부', '착상 전 유전 검사 사용 여부',
    '착상 전 유전 진단 사용 여부', '남성 주 불임 원인', '남성 부 불임 원인', '여성 주 불임 원인', 
    '여성 부 불임 원인', '부부 주 불임 원인', '부부 부 불임 원인', '불명확 불임 원인', '불임 원인 - 난관 질환',
    '불임 원인 - 남성 요인', '불임 원인 - 배란 장애', '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제',
    '불임 원인 - 자궁내막증', '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성',
    '불임 원인 - 정자 형태', '배아 생성 주요 이유', '총 시술 횟수', '클리닉 내 총 시술 횟수', 'IVF 시술 횟수',
    'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수',
    'DI 출산 횟수', '총 생성 배아 수', '미세주입된 난자 수', '미세주입에서 생성된 배아 수', '이식된 배아 수',
    '미세주입 배아 이식 수', '저장된 배아 수', '미세주입 후 저장된 배아 수', '해동된 배아 수', '해동 난자 수',
    '수집된 신선 난자 수', '저장된 신선 난자 수', '혼합된 난자 수', '파트너 정자와 혼합된 난자 수',
    '기증자 정자와 혼합된 난자 수', '난자 출처', '정자 출처', '난자 기증자 나이', '정자 기증자 나이',
    '동결 배아 사용 여부', '신선 배아 사용 여부', '기증 배아 사용 여부', '대리모 여부', 'PGD 시술 여부',
    'PGS 시술 여부', '난자 채취 경과일', '난자 해동 경과일', '난자 혼합 경과일', '배아 이식 경과일', 
    '배아 해동 경과일', '임신 성공 여부'
]]


# 3. 타겟 변수 (Target)
target_col = '임신 성공 여부'

# 1. 원본 보존을 위한 Deep Copy
df_clean = df.copy()

# 2. 나이 데이터 수치화 및 그룹화 매핑 정의
# 연령별 중간값(val)과 임상적 위험도(risk)를 멀티 트랙으로 관리
age_info = {
    '만18-34세': {'val': 26, 'risk': '정상_임신군'},
    '만35-37세': {'val': 36, 'risk': '고위험_임신군'},
    '만38-39세': {'val': 38.5, 'risk': '고위험_임신군'},
    '만40-42세': {'val': 41, 'risk': '초고위험_임신군'},
    '만43-44세': {'val': 43.5, 'risk': '초고위험_임신군'},
    '만45-50세': {'val': 47.5, 'risk': '초고위험_임신군'},
    '알 수 없음': {'val': np.nan, 'risk': '미분류'}
}

df_clean['나이_수치'] = df_clean['시술 당시 나이'].apply(lambda x: age_info[x]['val'])
df_clean['임신_위험도_범주'] = df_clean['시술 당시 나이'].apply(lambda x: age_info[x]['risk'])

# 3. 데이터 정제: 불필요한 특성 제거 (Feature Selection) 
# (1) 결측치 과다 항목 (90% 이상)
missing_drop = ['임신 시도 또는 마지막 임신 경과 연수', '난자 해동 경과일']

# (2) 극심한 불균형 항목 (상수 수준 변수)
# 분석 결과 1의 빈도가 0.1% 미만으로 학습에 악영향을 줄 수 있는 컬럼
imbalanced_drop = [
    '불임 원인 - 여성 요인', 
    '불임 원인 - 자궁경부 문제', 
    '불임 원인 - 정자 면역학적 요인', 
    '불임 원인 - 정자 운동성', 
    '불임 원인 - 정자 농도', 
    '불임 원인 - 정자 형태'
]

# (3) 통합 삭제 리스트 및 실행
final_drop_list = missing_drop + imbalanced_drop
df_clean = df_clean.drop(columns=final_drop_list)

# 4. 최종 결과 검증
print("-" * 30)
print(f"1. 원본 대비 삭제된 컬럼 수: {len(final_drop_list)}")
print(f"2. 전처리 후 최종 남은 컬럼 수: {len(df_clean.columns)}")
print("-" * 30)
print("3. 연령대별 평균 성공률 트렌드 (인사이트 확인):")
print(df_clean.groupby('시술 당시 나이')['임신 성공 여부'].mean().sort_values(ascending=False))