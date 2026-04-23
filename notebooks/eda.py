# 한글 깨짐 방지
import koreanize_matplotlib
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import koreanize_matplotlib
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
# from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')

# import shap
# shap.initjs()  # JavaScript 시각화 초기화
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from lightgbm import LGBMRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import root_mean_squared_error, classification_report, confusion_matrix
# 랜덤 시드 고정
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
# 차트에서 '-' 깨짐 현상 방지
plt.rcParams['axes.unicode_minus'] = False


# CSV 파일 읽기
df = pd.read_csv('C:/Users/anywa/OneDrive/바탕 화면/ai_project/data/train.csv')
df.columns = [c.strip() for c in df.columns] 
# 데이터의 상위 5개 행 출력하여 눈으로 확인
print(df.head())

# 1. 결측치, 이상치 확인
print(df.shape)    
print(df.info())
print(df.describe())
print(df.isnull().sum())

print("결측치 종류 확인")
df.isnull().sum().sort_values(ascending=False)

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

# 1. 특정 컬럼의 실제 값 종류 확인 (예: 나이 컬럼)
print(df['시술 당시 나이'].unique())

# 2. 임신 성공 여부(정답)의 비율 확인 (불균형 데이터인지 체크)
print(df['임신 성공 여부'].value_counts(normalize=True))

def classify_age_risk(age_str):
    if age_str == '만18-34세':
        return '정상_임신군'
    elif age_str in ['만35-37세', '만38-39세']:
        return '고위험_임신군'
    elif age_str in ['만40-42세', '만43-44세', '만45-50세']:
        return '초고위험_임신군'
    else:
        return '미분류' # '알 수 없음' 처리

# 1. 매핑 딕셔너리 정의 (대표값과 위험군 동시 정의)
age_info = {
    '만18-34세': {'val': 26, 'risk': 'Normal'},
    '만35-37세': {'val': 36, 'risk': 'High_Early'},
    '만38-39세': {'val': 38.5, 'risk': 'High_Early'},
    '만40-42세': {'val': 41, 'risk': 'High_Extreme'},
    '만43-44세': {'val': 43.5, 'risk': 'High_Extreme'},
    '만45-50세': {'val': 47.5, 'risk': 'High_Extreme'},
    '알 수 없음': {'val': None, 'risk': 'Unknown'}
}
df['임신_위험도_범주'] = df['시술 당시 나이'].apply(classify_age_risk)

# 잘 바뀌었는지 확인
print(df[['시술 당시 나이', '임신_위험도_범주']].value_counts())


# 2. 파생 변수 생성
df['Age_Median'] = df['시술 당시 나이'].map(lambda x: age_info[x]['val'])
df['Age_Risk_Group'] = df['시술 당시 나이'].map(lambda x: age_info[x]['risk'])

# 3. 결과 확인 (나이별로 잘 쪼개졌는지 검증)
result = df.groupby('시술 당시 나이')[['Age_Median', '임신 성공 여부']].mean()
print(result)


