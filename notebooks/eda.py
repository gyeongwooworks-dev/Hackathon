# 한글 깨짐 방지
# pip install koreanize-matplotlib -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import koreanize_matplotlib
import warnings
import math


# 기초 설정
warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 시각화 한글 깨짐 방지
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# 데이터 로드 (경로 내 r 붙여서 에러 방지) - 파일 경로 바꿔주세요
train_path = r'/content/train.csv'
test_path  = r'/content/test.csv'

df      = pd.read_csv(train_path)   # Train 원본
df_test = pd.read_csv(test_path)    # Test 원본

# 컬럼 공백 제거
df.columns      = [c.strip() for c in df.columns]
df_test.columns = [c.strip() for c in df_test.columns]

# 1. 결측치, 이상치 확인

print(df.describe())
print(f"데이터 크기: {df.shape}")
print(df.info())

# 결측치 비율 확인 (상위 5개)
missing_series = df.isnull().sum().sort_values(ascending=False)
print("결측치 상위 컬럼:\n", missing_series.head(10))

# 타겟 변수(임신 성공 여부) 불균형 확인
print("타겟 클래스 비율:\n", df['임신 성공 여부'].value_counts(normalize=True))


#  중복된 행의 전체 개수 확인
duplicate_count = df.duplicated().sum()
print(f"완전 중복 행 개수: {duplicate_count}")

# 2. 어떤 데이터가 중복되었는지 상위 5개만 눈으로 확인
if duplicate_count > 0:
    print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns[:3])).head())

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


# 타겟 변수 (Target)
target_col = '임신 성공 여부'

#  원본 보존을 위한 Deep Copy
df_clean      = df.copy()
df_clean_test = df_test.copy()   # Test 전처리용 복사본

# ================================================================
# [Data Leakage 방지] Train 기준 fit 값 저장소
# ================================================================
iqr_clip_bounds = {}  # IQR 클리핑 상한값 (Train 기준 계산 후 Test에 재사용)

# ID별 출현 빈도 계산
id_counts = df['ID'].value_counts()

# 중복된 ID만 추출 (2회 이상 나타난 경우)
duplicate_ids = id_counts[id_counts > 1]

print(f"중복된 ID 개수: {len(duplicate_ids)}")
print("-" * 30)
if len(duplicate_ids) > 0:
    print("상위 중복 ID 샘플:\n", duplicate_ids.head())

#  실제 중복된 ID 중 하나를 골라 데이터 내용 확인
    sample_id = duplicate_ids.index[0]
    print(f"\n[ID: {sample_id}]의 상세 데이터:")
    display(df[df['ID'] == sample_id])
else:
    print("모든 ID가 고유합니다. (중복 없음)")

df_unique = df.drop_duplicates(subset=['ID'], keep='first')

print(f"제거 전 행 수: {len(df)}")
print(f"제거 후 행 수: {len(df_unique)}")
print(f"삭제된 중복 행 수: {len(df) - len(df_unique)}")

#  데이터 정제: 불필요한 특성 제거 (Feature Selection)

# (1) 결측치 과다 항목 (90% 이상)
missing_drop = ['임신 시도 또는 마지막 임신 경과 연수', '난자 해동 경과일']

# (2) 극심한 불균형 항목 (상수 수준 변수), (ID 포함)
# 분석 결과 1의 빈도가 0.1% 미만으로 학습에 악영향을 줄 수 있는 컬럼
imbalanced_drop = [
    'ID', '시술 시기 코드',
    '배란 유도 유형',  # 유효값 2건(0.0008%)으로 drop
    '불임 원인 - 여성 요인',
    '불임 원인 - 자궁경부 문제',
    '불임 원인 - 정자 면역학적 요인',
    '불임 원인 - 정자 운동성',
    '불임 원인 - 정자 농도',
    '불임 원인 - 정자 형태',
    '정자 출처',  # 카테고리별 성공률 차이 1.6%p로 미미, 극소수 카테고리 존재
]

final_drop_list = missing_drop + imbalanced_drop
df_clean = df_clean.drop(columns=final_drop_list)

df_clean['특정 시술 유형'] = df_clean['특정 시술 유형'].fillna('Unknown')

# ================================================================
# 결측치 처리 및 파생변수 생성 
# ================================================================
# (1) 유전 검사 컬럼 → 비수행(0) 처리
# 고위험/초고위험군에서만 선택적으로 수행하는 시술이라 결측 = 미수행

genetic_cols = [
    'PGS 시술 여부',
    'PGD 시술 여부',
    '착상 전 유전 검사 사용 여부',
    '착상 전 유전 진단 사용 여부',   # 2.4% 결측도 동일 처리
]

df_clean[genetic_cols] = df_clean[genetic_cols].fillna(0)

# (3) [그룹 A] 배아 해동 경과일 (84% 결측)
# 동결 배아를 사용한 케이스에서만 기록 → 결측 = 미수행 → 플래그 전환 후 원본 삭제
df_clean['배아해동_수행'] = df_clean['배아 해동 경과일'].notna().astype(int)
df_clean = df_clean.drop(columns=['배아 해동 경과일'])

# (4) [그룹 B] 날짜 경과일 3개 (17~22% 결측)
# 각 시술 단계 미수행 시 기록 없음 → 결측 = 미수행 → 플래그 전환 후 원본 삭제
date_cols = {
    '난자 채취 경과일': '난자채취_수행',
    '난자 혼합 경과일': '난자혼합_수행',
}

for col, flag in date_cols.items():
    df_clean[flag] = df_clean[col].notna().astype(int)
df_clean = df_clean.drop(columns=list(date_cols.keys()))

# ----------------------------------------------------------
#   배아 이식 경과일 구간화 (배아이식_수행 플래그 대체)
#   5~6일(배반포 단계) 성공률 40.1% vs 3~4일 26.5% → +13.6%p 차이
# ----------------------------------------------------------
def make_elapsed_bin(series):
    """경과일 → 구간 숫자 (0일/1-2일/3-4일/5-6일/7일+)"""
    return pd.cut(
        pd.to_numeric(series, errors='coerce').fillna(-1),
        bins=[-2, 0, 2, 4, 6, 999],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

df_clean['이식경과일_구간'] = make_elapsed_bin(df_clean['배아 이식 경과일'])
df_clean['배반포_이식_추정'] = (df_clean['이식경과일_구간'].fillna(0) >= 3).astype(int)
df_clean = df_clean.drop(columns=['배아 이식 경과일'])

print("[추가 완료] 배아 이식 경과일 구간 피처")
print(f"  배반포_이식_추정 분포: {df_clean['배반포_이식_추정'].value_counts().to_dict()}")
print(f"  배반포 이식 성공률: {df_clean[df_clean['배반포_이식_추정']==1]['임신 성공 여부'].mean():.3f}")
print(f"  일반 이식 성공률:   {df_clean[df_clean['배반포_이식_추정']==0]['임신 성공 여부'].mean():.3f}")

# (5) [그룹 C] 배아/난자 수치 20개 (2.4% 결측)
# IUI, DI 등 체외수정 미수행 시 수집 자체가 안 되는 구조적 결측
# 결측 여부 플래그 1개 추가 후 0으로 채움
embryo_cols = [c for c in df_clean.columns if df_clean[c].isnull().sum() == 6291]
df_clean['배아정보_결측'] = df_clean[embryo_cols[0]].isna().astype(int)
df_clean[embryo_cols] = df_clean[embryo_cols].fillna(0)

missing_left = df_clean.isnull().sum()
print("남은 결측치:")
print(missing_left[missing_left > 0] if missing_left.any() else "없음")
print()
print(f"전체 행 수: {len(df_clean)}")
print(f"전체 컬럼 수: {len(df_clean.columns)}")

# ================================================================
# 파생변수 생성
# ================================================================

# (1) 나이 수치화 및 임상적 위험도 그룹화 (도메인 지식 기반)
age_info = {
    '만18-34세': {'val': 26,   'risk': '정상_임신군'},
    '만35-37세': {'val': 36,   'risk': '고위험_임신군'},
    '만38-39세': {'val': 38.5, 'risk': '고위험_임신군'},
    '만40-42세': {'val': 41,   'risk': '초고위험_임신군'},
    '만43-44세': {'val': 43.5, 'risk': '초고위험_임신군'},
    '만45-50세': {'val': 47.5, 'risk': '초고위험_임신군'},
    '알 수 없음': {'val': np.nan, 'risk': '미분류'}
}
df_clean['나이_수치'] = df_clean['시술 당시 나이'].apply(
    lambda x: age_info.get(x, {'val': np.nan, 'risk': '미분류'})['val']
)
df_clean['임신_위험도_범주'] = df_clean['시술 당시 나이'].apply(lambda x: age_info[x]['risk'])

# 나이_수치 결측치 329건 (0.13%) → Train 중앙값으로 대체
# [Fix #2] 전체 데이터 중앙값 대신 Train 기준 중앙값만 사용하여 Test 영향 차단
AGE_MEDIAN_FILLNA = df_clean['나이_수치'].median()   # Train에서만 계산
print(f"[나이 결측 대체값] Train 중앙값: {AGE_MEDIAN_FILLNA}")
df_clean['나이_수치'] = df_clean['나이_수치'].fillna(AGE_MEDIAN_FILLNA)

# (2) 시술 유형 5개 그룹으로 압축
def classify_treatment_logic(x):
    if pd.isna(x):
        return 'Unknown'
    target = str(x).upper().strip()
    if 'BLASTOCYST' in target:
        return 'Blastocyst_Transfer'
    elif 'ICSI' in target:
        return 'ICSI'
    elif 'IVF' in target or 'VF' in target:
        return 'IVF'
    elif 'IUI' in target:
        return 'IUI'
    elif 'UNKNOWN' in target or target == '' or target == 'NAN':
        return 'Unknown'
    else:
        return 'Unknown'

df_clean['시술_분류_그룹'] = df_clean['특정 시술 유형'].apply(classify_treatment_logic)

# 검증
print("시술_분류_그룹 분포:")
print(df_clean['시술_분류_그룹'].value_counts())
print()
print("연령대별 평균 성공률:")
print(df_clean.groupby('시술 당시 나이')['임신 성공 여부'].mean().sort_values(ascending=False))

df_clean.columns = df_clean.columns.str.replace(' ', '_')

print(df_clean.columns.tolist())
df_clean

df_clean.columns

desc = df_clean.select_dtypes(include='number').describe().T
print(desc)
print("==" * 30 )
# 이상치 의심 컬럼 추출
# Carl Friedrich Gauss의 정규분포 이론
# 정규분포를 따를 때 평균 ± 3σ 안에 전체 데이터의 99.7%가 들어옵니다.
# 즉 max값이 평균 + 3σ를 초과하면 상위 0.15% 밖에 있는 극단값이라는 의미

desc['outlier_flag'] = desc['max'] > (desc['mean'] + 3 * desc['std'])
print(desc[desc['outlier_flag'] == True][['mean', 'std', 'max']])
print("==" * 30 )
# 이진 변수 제외하고 연속형만 필터링
continuous_outliers = desc[
    (desc['outlier_flag'] == True) & (desc['max'] > 1)
][['mean', 'std', 'max']]
print(continuous_outliers)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'NanumGothic'

outlier_cols = [
    '총_생성_배아_수', '미세주입된_난자_수', '미세주입에서_생성된_배아_수',
    '저장된_배아_수', '미세주입_후_저장된_배아_수', '해동된_배아_수',
    '해동_난자_수', '수집된_신선_난자_수', '저장된_신선_난자_수',
    '혼합된_난자_수', '파트너_정자와_혼합된_난자_수', '기증자_정자와_혼합된_난자_수'
]

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()

for i, col in enumerate(outlier_cols):
    axes[i].boxplot(df_clean[col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[i].set_title(col, fontsize=9)
    axes[i].set_ylabel('값', fontsize=8)

plt.suptitle('이상치 의심 컬럼 박스플롯', fontsize=14)
plt.tight_layout()
plt.show()

outlier_cols = [
    '총_생성_배아_수', '미세주입된_난자_수', '미세주입에서_생성된_배아_수',
    '저장된_배아_수', '미세주입_후_저장된_배아_수', '해동된_배아_수',
    '해동_난자_수', '수집된_신선_난자_수', '저장된_신선_난자_수',
    '혼합된_난자_수', '파트너_정자와_혼합된_난자_수', '기증자_정자와_혼합된_난자_수'
]

# log(x+1) 변환 — 0값이 있어서 +1 필수
for col in outlier_cols:
    df_clean[col + '_log'] = np.log1p(df_clean[col])

# 변환 전후 비교
fig, axes = plt.subplots(4, 6, figsize=(24, 16))
axes = axes.flatten()

for i, col in enumerate(outlier_cols):
    # 원본
    axes[i*2].boxplot(df_clean[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6))
    axes[i*2].set_title(f'{col}\n(원본)', fontsize=8)

    # 로그 변환
    axes[i*2+1].boxplot(df_clean[col + '_log'].dropna(), patch_artist=True,
                        boxprops=dict(facecolor='coral', alpha=0.6))
    axes[i*2+1].set_title(f'{col}\n(log)', fontsize=8)

plt.suptitle('로그 변환 전후 비교', fontsize=14)
plt.tight_layout()
plt.show()

# 0값 비율 확인
weak_cols = [
    '저장된_배아_수', '미세주입_후_저장된_배아_수', '해동된_배아_수',
    '해동_난자_수', '저장된_신선_난자_수', '기증자_정자와_혼합된_난자_수'
]
for col in weak_cols:
    zero_ratio = (df_clean[col] == 0).sum() / len(df_clean) * 100
    print(f'{col}: 0값 비율 {zero_ratio:.1f}%')

# ================================================================
# 이상치 처리 및 파생변수 생성
# ================================================================

# (1) 로그 변환 — 효과 좋은 6개
log_cols = [
    '총_생성_배아_수', '미세주입된_난자_수', '미세주입에서_생성된_배아_수',
    '수집된_신선_난자_수', '혼합된_난자_수', '파트너_정자와_혼합된_난자_수'
]
for col in log_cols:
    df_clean[col + '_log'] = np.log1p(df_clean[col])

# (2) 이진 변수 전환 — 0값 93%+ (수행 여부가 핵심 정보)
binary_cols = [
    '해동_난자_수', '저장된_신선_난자_수', '기증자_정자와_혼합된_난자_수'
]
for col in binary_cols:
    df_clean[col + '_수행'] = (df_clean[col] > 0).astype(int)

# (3) IQR 클리핑 — 0값 67~84% (수치 자체도 의미 있음)
# 모델이 특정 수치에 과적합(Overfitting)되는 것을 방지하기 위해 수치가 50개 이상인 극단값들을 상한선으로 고정하여
clip_cols = [
    '저장된_배아_수', '미세주입_후_저장된_배아_수', '해동된_배아_수'
]
# [Fix #1] Train 데이터에서만 IQR 계산 → 계산값을 iqr_clip_bounds에 저장 → Test에 재사용
for col in clip_cols:
    Q1 = df_clean[col].quantile(0.25)   # Train 기준
    Q3 = df_clean[col].quantile(0.75)   # Train 기준
    IQR = Q3 - Q1
    upper = Q3 + 1.5 * IQR
    iqr_clip_bounds[col] = upper         # Test에서 재사용할 상한값 저장
    df_clean[col + '_clip'] = df_clean[col].clip(upper=upper)

# ================================================================
# 검증
# ================================================================
print("=== 로그 변환 컬럼 ===")
print(df_clean[[col + '_log' for col in log_cols]].describe().T[['mean', 'std', 'max']])

print("\n=== 이진 변환 컬럼 ===")
for col in binary_cols:
    print(f"{col}_수행: 1값 비율 {df_clean[col + '_수행'].mean()*100:.1f}%")

print("\n=== IQR 클리핑 컬럼 ===")
for col in clip_cols:
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - df_clean[col].quantile(0.25)
    upper = Q3 + 1.5 * IQR
    print(f"{col}_clip: 상한값 {upper:.1f}, 원본 max {df_clean[col].max():.1f} → 클리핑 후 max {df_clean[col+'_clip'].max():.1f}")

print(df_clean.columns.tolist())
print(f"\n전체 컬럼 수: {len(df_clean.columns)}")

pythonfig, axes = plt.subplots(1, 3, figsize=(20, 6))

# (1) 연령대별 임신 성공률
age_success = df_clean[df_clean['시술_당시_나이'] != '알 수 없음'].groupby('시술_당시_나이')['임신_성공_여부'].mean().sort_values(ascending=False)
bars = axes[0].bar(range(len(age_success)), age_success.values, color='steelblue', alpha=0.7)
axes[0].set_xticks(range(len(age_success)))
axes[0].set_xticklabels(age_success.index, rotation=30, ha='right', fontsize=9)
axes[0].set_title('연령대별 임신 성공률', fontsize=13)
axes[0].set_ylabel('임신 성공률')
axes[0].set_ylim(0, max(age_success.values) * 1.25)
axes[0].axhline(y=df_clean['임신_성공_여부'].mean(), color='red', linestyle='--', label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[0].legend()
for bar, val in zip(bars, age_success.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='steelblue', linewidth=1))

# (2) 위험도 범주별 임신 성공률
risk_success = df_clean[df_clean['임신_위험도_범주'] != '미분류'].groupby('임신_위험도_범주')['임신_성공_여부'].mean().sort_values(ascending=False)
risk_colors = {'정상_임신군': 'steelblue', '고위험_임신군': 'orange', '초고위험_임신군': 'red'}
bar_colors = [risk_colors.get(r, 'steelblue') for r in risk_success.index]
bars2 = axes[1].bar(range(len(risk_success)), risk_success.values, color=bar_colors, alpha=0.7)
axes[1].set_xticks(range(len(risk_success)))
axes[1].set_xticklabels(risk_success.index, rotation=20, ha='right', fontsize=9)
axes[1].set_title('위험도 범주별 임신 성공률', fontsize=13)
axes[1].set_ylabel('임신 성공률')
axes[1].set_ylim(0, max(risk_success.values) * 1.25)
axes[1].axhline(y=df_clean['임신_성공_여부'].mean(), color='red', linestyle='--', label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[1].legend()
for bar, val, color in zip(bars2, risk_success.values, bar_colors):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=1))

# (3) 나이대별 위험도 범주 구성
age_order = ['만18-34세', '만35-37세', '만38-39세', '만40-42세', '만43-44세', '만45-50세']
age_risk = df_clean[df_clean['시술_당시_나이'].isin(age_order)].groupby(
    ['시술_당시_나이', '임신_위험도_범주']).size().unstack(fill_value=0)
age_risk = age_risk.reindex(age_order)

color_map = {'정상_임신군': 'steelblue', '고위험_임신군': 'orange', '초고위험_임신군': 'red'}
bottom = None
for risk_cat in ['정상_임신군', '고위험_임신군', '초고위험_임신군']:
    if risk_cat in age_risk.columns:
        vals = age_risk[risk_cat].values
        axes[2].bar(range(len(age_order)), vals, bottom=bottom,
                label=risk_cat, color=color_map[risk_cat], alpha=0.7)
        bottom = vals if bottom is None else bottom + vals

axes[2].set_xticks(range(len(age_order)))
axes[2].set_xticklabels(age_order, rotation=30, ha='right', fontsize=9)
axes[2].set_title('나이대별 위험도 범주 구성', fontsize=13)
axes[2].set_ylabel('케이스 수')
axes[2].legend()

plt.suptitle('나이/위험도 vs 임신 성공률', fontsize=15)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 나이대별 위험도 범주 케이스 수
age_order = ['만18-34세', '만35-37세', '만38-39세', '만40-42세', '만43-44세', '만45-50세']
risk_counts = df_clean[df_clean['시술_당시_나이'].isin(age_order)].groupby('임신_위험도_범주').size()

# (1) 위험도 범주별 인구 비율 파이차트
risk_order = ['정상_임신군', '고위험_임신군', '초고위험_임신군']
risk_vals = [risk_counts.get(r, 0) for r in risk_order]
risk_colors_pie = ['steelblue', 'orange', 'red']

wedges, texts, autotexts = axes[0].pie(
    risk_vals,
    labels=risk_order,
    colors=risk_colors_pie,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(alpha=0.7, edgecolor='white', linewidth=2)
)
for text in autotexts:
    text.set_fontsize(11)
    text.set_fontweight('bold')
axes[0].set_title('위험도 범주별 인구 비율', fontsize=13)

# 케이스 수 범례 추가
legend_labels = [f'{r}: {v:,}명' for r, v in zip(risk_order, risk_vals)]
axes[0].legend(legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=9)

# (2) 나이대별 인구 비율 파이차트
age_counts = df_clean[df_clean['시술_당시_나이'].isin(age_order)].groupby('시술_당시_나이').size().reindex(age_order)
age_colors_pie = ['steelblue', 'orange', 'orange', 'red', 'red', 'red']

wedges2, texts2, autotexts2 = axes[1].pie(
    age_counts.values,
    labels=age_counts.index,
    colors=age_colors_pie,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    wedgeprops=dict(alpha=0.7, edgecolor='white', linewidth=2)
)
for text in autotexts2:
    text.set_fontsize(10)
    text.set_fontweight('bold')
axes[1].set_title('나이대별 인구 비율\n(파란색=정상, 주황=고위험, 빨강=초고위험)', fontsize=13)

legend_labels2 = [f'{age}: {cnt:,}명' for age, cnt in zip(age_order, age_counts.values)]
axes[1].legend(legend_labels2, loc='lower center', bbox_to_anchor=(0.5, -0.15), fontsize=9)

plt.suptitle('위험도 범주 및 나이대별 인구 분포', fontsize=15)
plt.tight_layout()
plt.show()

# 시술별 파스텔 색상 지정
treatment_colors = {
    'Blastocyst_Transfer': '#C9B8E8',  # 연한 보라
    'ICSI':                '#FFF3B0',  # 연한 노랑
    'IVF':                 '#B8D8E8',  # 연한 하늘
    'IUI':                 '#FFD9B0',  # 연한 주황
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (1) 시술 유형별 임신 성공률
treatment_success = df_clean[df_clean['시술_분류_그룹'] != 'Unknown'].groupby('시술_분류_그룹')['임신_성공_여부'].mean().sort_values(ascending=False)
bar_colors = [treatment_colors[t] for t in treatment_success.index]
bars = axes[0].bar(range(len(treatment_success)), treatment_success.values, color=bar_colors, edgecolor='gray', linewidth=0.5)
axes[0].set_xticks(range(len(treatment_success)))
axes[0].set_xticklabels(treatment_success.index, rotation=20, ha='right', fontsize=10)
axes[0].set_title('시술 유형별 임신 성공률', fontsize=13)
axes[0].set_ylabel('임신 성공률')
axes[0].set_ylim(0, max(treatment_success.values) * 1.25)
axes[0].axhline(y=df_clean['임신_성공_여부'].mean(), color='red', linestyle='--', label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[0].legend()
for bar, val, col in zip(bars, treatment_success.values, bar_colors):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=1))

# (2) 시술 유형별 케이스 수
treatment_counts = df_clean[df_clean['시술_분류_그룹'] != 'Unknown'].groupby('시술_분류_그룹').size().sort_values(ascending=False)
bar_colors2 = [treatment_colors[t] for t in treatment_counts.index]
bars2 = axes[1].bar(range(len(treatment_counts)), treatment_counts.values, color=bar_colors2, edgecolor='gray', linewidth=0.5)
axes[1].set_xticks(range(len(treatment_counts)))
axes[1].set_xticklabels(treatment_counts.index, rotation=20, ha='right', fontsize=10)
axes[1].set_title('시술 유형별 케이스 수', fontsize=13)
axes[1].set_ylabel('케이스 수')
axes[1].set_ylim(0, max(treatment_counts.values) * 1.25)
for bar, val in zip(bars2, treatment_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,}건', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=1))

plt.suptitle('시술 유형 vs 임신 성공률', fontsize=15)
plt.tight_layout()
plt.show()

age_order = ['만18-34세', '만35-37세', '만38-39세', '만40-42세', '만43-44세', '만45-50세']
treatment_order = ['Blastocyst_Transfer', 'ICSI', 'IVF', 'IUI']

df_filtered = df_clean[
    (df_clean['시술_당시_나이'].isin(age_order)) &
    (df_clean['시술_분류_그룹'].isin(treatment_order))
]

pivot = df_filtered.groupby(['시술_당시_나이', '시술_분류_그룹'])['임신_성공_여부'].mean().unstack()
pivot = pivot.reindex(age_order)[treatment_order]
pivot_pct = (pivot * 100).round(1)

plt.figure(figsize=(12, 7))
sns.heatmap(
    pivot_pct,
    annot=True,
    fmt='.1f',
    cmap='coolwarm',
    vmin=0, vmax=50,
    linewidths=0.5,
    linecolor='white',
    annot_kws={'size': 12, 'weight': 'bold'},
    cbar_kws={'label': '임신 성공률 (%)'}
)
plt.title('나이대 x 시술 유형별 임신 성공률 (%)', fontsize=14, pad=15)
plt.xlabel('시술 유형', fontsize=11)
plt.ylabel('나이대', fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)
plt.tight_layout()
plt.show()

print(df_clean[df_clean['난자_출처'] == '본인 제공']['시술_분류_그룹'].value_counts())
print("==" * 60)
# IUI 케이스 난자_출처 → 본인 제공으로 변경
df_clean.loc[df_clean['시술_분류_그룹'] == 'IUI', '난자_출처'] = '본인 제공'
# 확인
print(df_clean['난자_출처'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(22, 7))

for ax, source, title in zip(
    axes,
    ['본인 제공', '기증 제공'],
    ['본인 난자 기준', '기증 난자 기준']
):
    df_source = df_clean[
        (df_clean['시술_당시_나이'].isin(age_order)) &
        (df_clean['시술_분류_그룹'].isin(treatment_order)) &
        (df_clean['난자_출처'] == source)
    ]

    pivot = df_source.groupby(['시술_당시_나이', '시술_분류_그룹'])['임신_성공_여부'].mean().unstack()
    pivot = pivot.reindex(age_order).reindex(columns=treatment_order)
    pivot_pct = (pivot * 100).round(1)

    mask = pivot_pct.isna()

    sns.heatmap(
        pivot_pct,
        annot=True,
        fmt='.1f',
        cmap='coolwarm',
        vmin=0, vmax=50,
        linewidths=0.5,
        linecolor='white',
        annot_kws={'size': 11, 'weight': 'bold'},
        cbar_kws={'label': '임신 성공률 (%)'},
        mask=mask,
        ax=ax
    )

    for i in range(len(age_order)):
        for j in range(len(treatment_order)):
            if mask.iloc[i, j]:
                ax.text(j + 0.5, i + 0.5, '데이터 없음',
                        ha='center', va='center', fontsize=9,
                        color='gray',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0', edgecolor='lightgray'))

    ax.set_title(f'나이대 x 시술 유형별 임신 성공률 — {title} (%)', fontsize=13)
    ax.set_xlabel('시술 유형', fontsize=10)
    ax.set_ylabel('나이대', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)

plt.suptitle('난자 출처별 임신 성공률 비교', fontsize=15)
plt.tight_layout()
plt.show()

# ================================================================
# 난자 출처 기반 파생변수 생성
#  난자_출처 '알 수 없음' 191건 — Unknown 시술 케이스로 출처 확인 불가
# 전체의 0.07%로 모델 영향 미미하여 별도 처리 없이 유지
# ================================================================

# (1) 기증 난자 여부 이진 변수
df_clean['기증_난자_여부'] = (df_clean['난자_출처'] == '기증 제공').astype(int)

# (2) 초고위험군 + 기증 난자 교호작용 변수
# 히트맵에서 확인된 인사이트 — 고령일수록 기증 난자 효과가 극적으로 나타남
df_clean['초고위험_기증난자_조합'] = (
    (df_clean['임신_위험도_범주'] == '초고위험_임신군') &
    (df_clean['난자_출처'] == '기증 제공')
).astype(int)

# 검증
print("기증_난자_여부 분포:")
print(df_clean['기증_난자_여부'].value_counts())
print(f"기증 난자 비율: {df_clean['기증_난자_여부'].mean()*100:.1f}%")
print()
print("초고위험_기증난자_조합 분포:")
print(df_clean['초고위험_기증난자_조합'].value_counts())
print(f"초고위험 + 기증 난자 비율: {df_clean['초고위험_기증난자_조합'].mean()*100:.1f}%")

# 불임 원인 컬럼 목록
infertility_cols = [
    '남성_주_불임_원인', '남성_부_불임_원인',
    '여성_주_불임_원인', '여성_부_불임_원인',
    '부부_주_불임_원인', '부부_부_불임_원인',
    '불명확_불임_원인', '불임_원인_-_난관_질환',
    '불임_원인_-_남성_요인', '불임_원인_-_배란_장애',
    '불임_원인_-_자궁내막증'
]

# 불임 원인별 임신 성공률
success_rates = {}
for col in infertility_cols:
    # 해당 원인이 있는 케이스(1)와 없는 케이스(0) 성공률 비교
    rate_1 = df_clean[df_clean[col] == 1]['임신_성공_여부'].mean()
    rate_0 = df_clean[df_clean[col] == 0]['임신_성공_여부'].mean()
    count_1 = df_clean[col].sum()
    success_rates[col] = {'있음': rate_1, '없음': rate_0, '건수': int(count_1)}

# 결과 출력
print(f"{'컬럼':<25} {'있음(%)':>8} {'없음(%)':>8} {'건수':>8}")
print("-" * 55)
for col, vals in success_rates.items():
    print(f"{col:<25} {vals['있음']*100:>7.1f}% {vals['없음']*100:>7.1f}% {vals['건수']:>8,}")

fig, ax = plt.subplots(figsize=(14, 6))

cols = list(success_rates.keys())
있음 = [success_rates[c]['있음']*100 for c in cols]
없음 = [success_rates[c]['없음']*100 for c in cols]

x = np.arange(len(cols))
width = 0.35

bars1 = ax.bar(x - width/2, 있음, width, label='원인 있음', color='#FFB3B3', edgecolor='gray', linewidth=0.5)
bars2 = ax.bar(x + width/2, 없음, width, label='원인 없음', color='#B3D4FF', edgecolor='gray', linewidth=0.5)

ax.axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--',
        label=f'전체 평균 임신 성공률 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
ax.set_xticks(x)
ax.set_xticklabels(cols, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('임신 성공률 (%)')
ax.set_ylim(0, 35)
ax.set_title('불임 원인별 임신 성공률 비교 (있음 vs 없음)', fontsize=13)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#FFB3B3', linewidth=0.8))
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#B3D4FF', linewidth=0.8))

plt.tight_layout()
plt.show()

# # 불임 원인 총개수 생성
# infertility_cols = ['남성_주_불임_원인', '남성_부_불임_원인', '여성_주_불임_원인',
#                     '여성_부_불임_원인', '부부_주_불임_원인', '부부_부_불임_원인']
# df_clean['불임_원인_총개수'] = df_clean[infertility_cols].sum(axis=1)

# # 원인 개수별 성공률 확인
# print(df_clean.groupby('불임_원인_총개수')['임신_성공_여부'].agg(['mean', 'count']))
# 0개가 245,209건으로 압도적이라 사실상 상수에 가깝고, 개수와 성공률 사이에 명확한 선형 패턴이 없어 변수 생성 안함

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(log_cols):
    bp = df_clean.boxplot(column=col, by='임신_성공_여부', ax=axes[i],
                    patch_artist=True,
                    boxprops=dict(facecolor='#B3D4FF', color='steelblue'), 
                    medianprops=dict(color='red', linewidth=2.5),
                    whiskerprops=dict(color='steelblue'),
                    capprops=dict(color='steelblue'),
                    flierprops=dict(marker='o', color='steelblue', alpha=0.5))
    axes[i].set_title(col.replace('_log', ''), fontsize=9)
    axes[i].set_xlabel('임신 성공 여부 (0=실패, 1=성공)', fontsize=8)
    axes[i].set_ylabel('log 변환값', fontsize=8)

plt.suptitle('배아/난자 수치(log) vs 임신 성공 여부', fontsize=14)
plt.tight_layout()
plt.show()

# 5단계 기타 변수 vs 임신 성공률
# 배란 자극, 단일 배아 이식, 유전 검사, 동결/신선 배아 사용 여부

binary_vars = {
    '배란_자극_여부': '배란 자극',
    '단일_배아_이식_여부': '단일 배아 이식',
    'PGS_시술_여부': 'PGS 시술',
    'PGD_시술_여부': 'PGD 시술',
    '착상_전_유전_검사_사용_여부': '착상 전 유전 검사',
    '착상_전_유전_진단_사용_여부': '착상 전 유전 진단',
    '동결_배아_사용_여부': '동결 배아 사용',
    '신선_배아_사용_여부': '신선 배아 사용',
    '기증_배아_사용_여부': '기증 배아 사용',
    '대리모_여부': '대리모',
}

results = {}
for col, label in binary_vars.items():
    rate_1 = df_clean[df_clean[col] == 1]['임신_성공_여부'].mean()
    rate_0 = df_clean[df_clean[col] == 0]['임신_성공_여부'].mean()
    count_1 = int(df_clean[col].sum())
    results[label] = {'있음': rate_1, '없음': rate_0, '건수': count_1}

fig, ax = plt.subplots(figsize=(16, 6))

labels = list(results.keys())
있음 = [results[l]['있음']*100 for l in labels]
없음 = [results[l]['없음']*100 for l in labels]

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, 있음, width, label='해당 있음', color='#FFB3B3', edgecolor='gray', linewidth=0.5)
bars2 = ax.bar(x + width/2, 없음, width, label='해당 없음', color='#B3D4FF', edgecolor='gray', linewidth=0.5)

ax.axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--',
        label=f'전체 평균 임신 성공률 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('임신 성공률 (%)')
ax.set_ylim(0, 45)
ax.set_title('기타 변수별 임신 성공률 비교 (있음 vs 없음)', fontsize=13)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#FFB3B3', linewidth=0.8))
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#B3D4FF', linewidth=0.8))

plt.tight_layout()
plt.show()

# 수치 확인
print(f"{'변수':<20} {'있음(%)':>8} {'없음(%)':>8} {'건수':>8}")
print("-" * 50)
for label, vals in results.items():
    print(f"{label:<20} {vals['있음']*100:>7.1f}% {vals['없음']*100:>7.1f}% {vals['건수']:>8,}")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# (1) 배란 자극 여부 x 시술 유형별 임신 성공률
treatment_order = ['Blastocyst_Transfer', 'ICSI', 'IVF', 'IUI']
colors_stim = {0: '#B3D4FF', 1: '#FFB3B3'}

stim_results = df_clean[df_clean['시술_분류_그룹'].isin(treatment_order)].groupby(
    ['시술_분류_그룹', '배란_자극_여부'])['임신_성공_여부'].mean().unstack()
stim_results = stim_results.reindex(treatment_order)

x = np.arange(len(treatment_order))
width = 0.35
bars1 = axes[0].bar(x - width/2, stim_results[0]*100, width, label='배란 자극 없음', color='#B3D4FF', edgecolor='gray', linewidth=0.5)
bars2 = axes[0].bar(x + width/2, stim_results[1]*100, width, label='배란 자극 있음', color='#FFB3B3', edgecolor='gray', linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(treatment_order, rotation=20, ha='right', fontsize=9)
axes[0].set_title('배란 자극 여부 x 시술 유형별 임신 성공률', fontsize=12)
axes[0].set_ylabel('임신 성공률 (%)')
axes[0].set_ylim(0, 55)
axes[0].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--', label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[0].legend(fontsize=9)
for bar in list(bars1) + list(bars2):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', linewidth=0.8))

# (2) 시술 유형 x 난자 출처별 임신 성공률 (나이대 위험도로 색 구분)
source_results = df_clean[
    (df_clean['시술_분류_그룹'].isin(treatment_order)) &
    (df_clean['난자_출처'].isin(['본인 제공', '기증 제공']))
].groupby(['시술_분류_그룹', '난자_출처'])['임신_성공_여부'].mean().unstack()
source_results = source_results.reindex(treatment_order)

bars3 = axes[1].bar(x - width/2, source_results['본인 제공']*100, width, label='본인 난자', color='#B3D4FF', edgecolor='gray', linewidth=0.5)
bars4 = axes[1].bar(x + width/2, source_results['기증 제공']*100, width, label='기증 난자', color='#FFB3B3', edgecolor='gray', linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(treatment_order, rotation=20, ha='right', fontsize=9)
axes[1].set_title('시술 유형 x 난자 출처별 임신 성공률', fontsize=12)
axes[1].set_ylabel('임신 성공률 (%)')
axes[1].set_ylim(0, 55)
axes[1].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--', label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[1].legend(fontsize=9)
for bar in list(bars3) + list(bars4):
    if bar.get_height() > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', linewidth=0.8))

plt.suptitle('시술 유형별 상호작용 분석', fontsize=14)
plt.tight_layout()
plt.show()

# ================================================================
# 시술/임신/출산 횟수 컬럼 전처리 및 파생변수 생성
# ================================================================

cycle_cols = [
    '총_시술_횟수', '클리닉_내_총_시술_횟수', 'IVF_시술_횟수', 'DI_시술_횟수',
    '총_임신_횟수', 'IVF_임신_횟수', 'DI_임신_횟수',
    '총_출산_횟수', 'IVF_출산_횟수', 'DI_출산_횟수'
]

# 문자열인 경우에만 변환 (이미 수치형이면 스킵)
for col in cycle_cols:
    if df_clean[col].dtype == object:
        df_clean[col] = (
            df_clean[col]
            .str.replace('회 이상', '')
            .str.replace('회', '')
            .str.strip()
            .astype(int)
        )

# 클리핑 파생변수 생성
# 시각화 결과 0~3회 구간에서 횟수가 늘수록 임신 성공률이 선형적으로 하락하는
# 강력한 예측 신호를 확인함 → 이 패턴을 보존하기 위해 원본 유지
# 단, 6회 이상 케이스는 샘플이 극소수(노이즈 우려)라 5로 클리핑하여 제어
for col in cycle_cols:
    df_clean[col + '_clip'] = df_clean[col].clip(upper=5)

# 확인
print(df_clean[[col + '_clip' for col in cycle_cols]].describe().T[['min', 'max', 'mean']])
print(f"\n전체 컬럼 수: {len(df_clean.columns)}")

fig, axes = plt.subplots(2, 5, figsize=(22, 10))
axes = axes.flatten()

clip_cols = [col + '_clip' for col in cycle_cols]

for i, col in enumerate(clip_cols):
    success = df_clean.groupby(col)['임신_성공_여부'].mean() * 100
    count = df_clean.groupby(col).size()

    bars = axes[i].bar(success.index, success.values, color='#B3D4FF', edgecolor='gray', linewidth=0.5)
    axes[i].set_title(col.replace('_clip', ''), fontsize=9)
    axes[i].set_xlabel('횟수', fontsize=8)
    axes[i].set_ylabel('임신 성공률 (%)', fontsize=8)
    axes[i].set_ylim(0, max(success.values) * 1.3)
    axes[i].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--', linewidth=1)

    for bar, val, cnt in zip(bars, success.values, count.values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%\n({cnt:,}건)', ha='center', va='bottom', fontsize=7)

plt.suptitle('시술/임신/출산 횟수별 임신 성공률 (클리핑 적용)', fontsize=14)
plt.tight_layout()
plt.show()

# ⚠️⚠️원본 컬럼 제거 — clip 버전으로 대체
# 이유: 6회 이상 극소수 샘플이 노이즈로 작용할 수 있어 clip 버전만 사용
df_clean = df_clean.drop(columns=cycle_cols, errors='ignore')

print(f"전체 컬럼 수: {len(df_clean.columns)}")
print("\n잔존 확인 (원본 없어야 함):")
print([c for c in df_clean.columns if c in cycle_cols])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (1) 배아해동_수행 여부
for_plot = df_clean.groupby('배아해동_수행')['임신_성공_여부'].mean() * 100
bars = axes[0].bar(['미수행(0)', '수행(1)'], for_plot.values,
                color=['#B3D4FF', '#FFB3B3'], edgecolor='gray', linewidth=0.5)
axes[0].set_title('배아 해동 수행 여부별 임신 성공률', fontsize=11)
axes[0].set_ylabel('임신 성공률 (%)')
axes[0].set_ylim(0, 40)
axes[0].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--',
                label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[0].legend()
for bar, val in zip(bars, for_plot.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=0.8))

# (2) 동결 배아 사용 여부
for_plot2 = df_clean.groupby('동결_배아_사용_여부')['임신_성공_여부'].mean() * 100
bars2 = axes[1].bar(['미사용(0)', '사용(1)'], for_plot2.values,
                    color=['#B3D4FF', '#FFB3B3'], edgecolor='gray', linewidth=0.5)
axes[1].set_title('동결 배아 사용 여부별 임신 성공률', fontsize=11)
axes[1].set_ylabel('임신 성공률 (%)')
axes[1].set_ylim(0, 40)
axes[1].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--',
                label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[1].legend()
for bar, val in zip(bars2, for_plot2.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=0.8))

# (3) 신선 배아 사용 여부
for_plot3 = df_clean.groupby('신선_배아_사용_여부')['임신_성공_여부'].mean() * 100
bars3 = axes[2].bar(['미사용(0)', '사용(1)'], for_plot3.values,
                    color=['#B3D4FF', '#FFB3B3'], edgecolor='gray', linewidth=0.5)
axes[2].set_title('신선 배아 사용 여부별 임신 성공률', fontsize=11)
axes[2].set_ylabel('임신 성공률 (%)')
axes[2].set_ylim(0, 40)
axes[2].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--',
                label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[2].legend()
for bar, val in zip(bars3, for_plot3.values):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', linewidth=0.8))

plt.suptitle('배아 종류별 임신 성공률 비교', fontsize=14)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

age_order = ['만18-34세', '만35-37세', '만38-39세', '만40-42세', '만43-44세', '만45-50세']

# (1) 나이대별 동결 배아 사용 여부 임신 성공률
pivot_frozen = df_clean[df_clean['시술_당시_나이'].isin(age_order)].groupby(
    ['시술_당시_나이', '동결_배아_사용_여부'])['임신_성공_여부'].mean().unstack() * 100
pivot_frozen = pivot_frozen.reindex(age_order)

x = np.arange(len(age_order))
width = 0.35
bars1 = axes[0].bar(x - width/2, pivot_frozen[0], width, label='동결 미사용', color='#B3D4FF', edgecolor='gray', linewidth=0.5)
bars2 = axes[0].bar(x + width/2, pivot_frozen[1], width, label='동결 사용', color='#FFB3B3', edgecolor='gray', linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(age_order, rotation=30, ha='right', fontsize=9)
axes[0].set_title('나이대별 동결 배아 사용 여부 임신 성공률', fontsize=11)
axes[0].set_ylabel('임신 성공률 (%)')
axes[0].set_ylim(0, 45)
axes[0].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--', label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[0].legend(fontsize=9)
for bar in list(bars1) + list(bars2):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', linewidth=0.8))

# (2) 나이대별 신선 배아 사용 여부 임신 성공률
pivot_fresh = df_clean[df_clean['시술_당시_나이'].isin(age_order)].groupby(
    ['시술_당시_나이', '신선_배아_사용_여부'])['임신_성공_여부'].mean().unstack() * 100
pivot_fresh = pivot_fresh.reindex(age_order)

bars3 = axes[1].bar(x - width/2, pivot_fresh[0], width, label='신선 미사용', color='#B3D4FF', edgecolor='gray', linewidth=0.5)
bars4 = axes[1].bar(x + width/2, pivot_fresh[1], width, label='신선 사용', color='#FFB3B3', edgecolor='gray', linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(age_order, rotation=30, ha='right', fontsize=9)
axes[1].set_title('나이대별 신선 배아 사용 여부 임신 성공률', fontsize=11)
axes[1].set_ylabel('임신 성공률 (%)')
axes[1].set_ylim(0, 45)
axes[1].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--', label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[1].legend(fontsize=9)
for bar in list(bars3) + list(bars4):
    if bar.get_height() > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', linewidth=0.8))

plt.suptitle('나이대별 배아 종류 임신 성공률 비교', fontsize=14)
plt.tight_layout()
plt.show()

# ================================================================
# 배아 종류 x 나이대 교호작용 파생변수 생성
# ================================================================

# 고령(초고위험군) + 동결 배아 사용 교호작용
# 시각화 결과: 만40세 이상에서 동결 배아 사용 시 미사용 대비 성공률이 역전되는 패턴 확인
# 젊을 때 채취한 고품질 배아를 고령에 사용하는 전략이 유효함을 시사
df_clean['고령_동결배아_조합'] = (
    (df_clean['임신_위험도_범주'] == '초고위험_임신군') &
    (df_clean['동결_배아_사용_여부'] == 1)
).astype(int)

# 검증
print("고령_동결배아_조합 분포:")
print(df_clean['고령_동결배아_조합'].value_counts())
print(f"비율: {df_clean['고령_동결배아_조합'].mean()*100:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

age_order = ['만18-34세', '만35-37세', '만38-39세', '만40-42세', '만43-44세', '만45-50세']

# 총_시술_횟수_clip 구간화 (0회 / 1-2회 / 3회 이상)
df_clean['시술횟수_구간'] = pd.cut(
    df_clean['총_시술_횟수_clip'],
    bins=[-1, 0, 2, 5],
    labels=['0회', '1-2회', '3회 이상']
)

# (1) 나이대 x 시술 횟수 구간별 임신 성공률
pivot = df_clean[df_clean['시술_당시_나이'].isin(age_order)].groupby(
    ['시술_당시_나이', '시술횟수_구간'])['임신_성공_여부'].mean().unstack() * 100
pivot = pivot.reindex(age_order)

x = np.arange(len(age_order))
width = 0.25
colors = ['#B3D4FF', '#FFB3B3', '#B3E8B3']

for j, (col, color) in enumerate(zip(['0회', '1-2회', '3회 이상'], colors)):
    bars = axes[0].bar(x + (j-1)*width, pivot[col], width,
                    label=col, color=color, edgecolor='gray', linewidth=0.5)
    for bar in bars:
        if bar.get_height() > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, linewidth=0.8))

axes[0].set_xticks(x)
axes[0].set_xticklabels(age_order, rotation=30, ha='right', fontsize=9)
axes[0].set_title('나이대 x 시술 횟수 구간별 임신 성공률', fontsize=11)
axes[0].set_ylabel('임신 성공률 (%)')
axes[0].set_ylim(0, 45)
axes[0].axhline(y=df_clean['임신_성공_여부'].mean()*100, color='red', linestyle='--',
                label=f'전체 평균 {df_clean["임신_성공_여부"].mean()*100:.1f}%')
axes[0].legend(fontsize=9)

# (2) 히트맵으로도 확인
pivot2 = df_clean[df_clean['시술_당시_나이'].isin(age_order)].groupby(
    ['시술_당시_나이', '시술횟수_구간'])['임신_성공_여부'].mean().unstack() * 100
pivot2 = pivot2.reindex(age_order)

sns.heatmap(
    pivot2.round(1),
    annot=True, fmt='.1f',
    cmap='coolwarm', vmin=0, vmax=40,
    linewidths=0.5, linecolor='white',
    annot_kws={'size': 11, 'weight': 'bold'},
    ax=axes[1]
)
axes[1].set_title('나이대 x 시술 횟수 구간별 임신 성공률 히트맵', fontsize=11)
axes[1].set_xlabel('시술 횟수 구간', fontsize=10)
axes[1].set_ylabel('나이대', fontsize=10)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=9)

plt.suptitle('나이대 x 과거 시술 횟수 교차 분석', fontsize=14)
plt.tight_layout()
plt.show()

# 정상_임신군 + 첫 시술(0회) 조합
# 젊고 첫 시도일 때 성공률이 가장 높은 패턴 반영
df_clean['정상군_첫시술'] = (
    (df_clean['임신_위험도_범주'] == '정상_임신군') &
    (df_clean['총_시술_횟수_clip'] == 0)
).astype(int)

# ================================================================
# 남성/여성 불임 요인 통합 파생변수 생성
# ================================================================

# 남성 요인 관련 컬럼 중 하나라도 1이면 남성 요인 있음
df_clean['남성_요인_존재'] = (
    (df_clean['남성_주_불임_원인'] == 1) |
    (df_clean['남성_부_불임_원인'] == 1) |
    (df_clean['불임_원인_-_남성_요인'] == 1)
).astype(int)

# 여성 요인 관련 컬럼 중 하나라도 1이면 여성 요인 있음
df_clean['여성_요인_존재'] = (
    (df_clean['여성_주_불임_원인'] == 1) |
    (df_clean['여성_부_불임_원인'] == 1) |
    (df_clean['불임_원인_-_난관_질환'] == 1) |
    (df_clean['불임_원인_-_배란_장애'] == 1) |
    (df_clean['불임_원인_-_자궁내막증'] == 1)
).astype(int)

# 비교
print(f"남성 요인 있음: {df_clean[df_clean['남성_요인_존재']==1]['임신_성공_여부'].mean()*100:.1f}%")
print(f"남성 요인 없음: {df_clean[df_clean['남성_요인_존재']==0]['임신_성공_여부'].mean()*100:.1f}%")
print()
print(f"여성 요인 있음: {df_clean[df_clean['여성_요인_존재']==1]['임신_성공_여부'].mean()*100:.1f}%")
print(f"여성 요인 없음: {df_clean[df_clean['여성_요인_존재']==0]['임신_성공_여부'].mean()*100:.1f}%")
print()
print(f"전체 평균: {df_clean['임신_성공_여부'].mean()*100:.1f}%")

# 분포 확인
print(f"\n남성_요인_존재: {df_clean['남성_요인_존재'].value_counts().to_dict()}")
print(f"여성_요인_존재: {df_clean['여성_요인_존재'].value_counts().to_dict()}")

fig, ax = plt.subplots(figsize=(9, 5))

labels = ['남성 요인\n있음\n(n=101,051)', '남성 요인\n없음\n(n=155,300)',
        '여성 요인\n있음\n(n=86,375)', '여성 요인\n없음\n(n=169,976)']
values = [27.6, 24.7, 26.3, 25.6]
colors = ['#1D9E75', '#B4B2A9', '#7F77DD', '#D3D1C7']

x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, width=0.55, zorder=3)

# 전체 평균 기준선
avg = 25.8
ax.axhline(avg, color='#E24B4A', linewidth=1.5, linestyle='--', zorder=4, label=f'전체 평균 {avg}%')

# 수치 박스 (잘림 방지)
for bar, val in zip(bars, values):
    ax.annotate(
        f'{val}%',
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 6),
        textcoords='offset points',
        ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='#2C2C2A',
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#D3D1C7', lw=1)
    )

# 차이 화살표 (남성)
ax.annotate('', xy=(0, 27.6), xytext=(1, 24.7),
            arrowprops=dict(arrowstyle='<->', color='#1D9E75', lw=1.5))
ax.text(0.5, 26.4, '+2.9%p', ha='center', va='bottom', fontsize=10,
        color='#1D9E75', fontweight='bold')

# 차이 화살표 (여성)
ax.annotate('', xy=(2, 26.3), xytext=(3, 25.6),
            arrowprops=dict(arrowstyle='<->', color='#7F77DD', lw=1.5))
ax.text(2.5, 26.1, '+0.7%p', ha='center', va='bottom', fontsize=10,
        color='#7F77DD', fontweight='bold')

# 축 정리
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(22, 31)
ax.set_ylabel('임신 성공률 (%)', fontsize=11)
ax.yaxis.grid(True, linestyle=':', color='#D3D1C7', zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

# 범례
patches = [
    mpatches.Patch(color='#1D9E75', label='남성 요인 있음'),
    mpatches.Patch(color='#B4B2A9', label='남성 요인 없음'),
    mpatches.Patch(color='#7F77DD', label='여성 요인 있음'),
    mpatches.Patch(color='#D3D1C7', label='여성 요인 없음'),
    plt.Line2D([0], [0], color='#E24B4A', lw=1.5, linestyle='--', label=f'전체 평균 {avg}%'),
]
ax.legend(handles=patches, loc='lower right', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('factor_success_rate.png', dpi=150, bbox_inches='tight')
plt.show()

# 이식된_배아_수별 성공률
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 집계
emb_data = df_clean.groupby('이식된_배아_수')['임신_성공_여부'].agg(['mean', 'count']).reset_index()
emb_data.columns = ['이식수', '성공률', '샘플수']
emb_data = emb_data[emb_data['이식수'] > 0]  # 0개 이식은 제외

# 성공률에 따라 색상 그라데이션
colors = ['#B4B2A9' if v < 0.258 else '#1D9E75' for v in emb_data['성공률']]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(emb_data['이식수'].astype(int), emb_data['성공률'] * 100,
            color=colors, width=0.55, zorder=3)

# 전체 평균선
ax.axhline(25.8, color='#E24B4A', lw=1.5, ls='--', zorder=4, label='전체 평균 25.8%')

# 수치 박스 (성공률 + 샘플수)
for bar, (_, row) in zip(bars, emb_data.iterrows()):
    ax.annotate(
        f"{row['성공률']*100:.1f}%\n(n={int(row['샘플수']):,})",
        xy=(bar.get_x() + bar.get_width() / 2, row['성공률'] * 100),
        xytext=(0, 7), textcoords='offset points',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2C2C2A',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#aaa', lw=1)
    )

# 꺾은선
ax.plot(emb_data['이식수'].astype(int), emb_data['성공률'] * 100,
        'o--', color='#534AB7', lw=1.8, ms=6, zorder=5, label='추세선')

ax.set_xlabel('이식된 배아 수 (개)', fontsize=12)
ax.set_ylabel('임신 성공률 (%)', fontsize=12)
ax.set_xticks(emb_data['이식수'].astype(int))
ax.set_ylim(0, 45)
ax.yaxis.grid(True, linestyle=':', color='#D3D1C7', zorder=0)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)

legend_patches = [
    mpatches.Patch(color='#1D9E75', label='평균 이상'),
    mpatches.Patch(color='#B4B2A9', label='평균 이하'),
    plt.Line2D([0], [0], color='#E24B4A', lw=1.5, ls='--', label='전체 평균 25.8%'),
    plt.Line2D([0], [0], color='#534AB7', lw=1.8, ls='--',
            marker='o', ms=5, label='추세선'),
]
ax.legend(handles=legend_patches, fontsize=10, framealpha=0.9, loc='upper right')

plt.suptitle('이식된 배아 수별 임신 성공률', fontsize=14)
plt.tight_layout()
plt.savefig('embryo_success.png', dpi=150, bbox_inches='tight')
plt.show()

# 1. 단일배아 여부 (1개 이식 = 최적 조건)
df_clean['단일배아_이식'] = (df_clean['이식된_배아_수'] == 1).astype(int)

# 2. 과다이식 여부 (3개 이상 = 위험 신호)
df_clean['과다배아_이식'] = (df_clean['이식된_배아_수'] >= 3).astype(int)

# 3. 나이 × 배아수 조합 (핵심 교호작용)
# 고령인데 배아 3개 이상 이식 = 이중 위험
df_clean['고령_과다이식'] = (
    (df_clean['나이_수치'] >= 38) & (df_clean['이식된_배아_수'] >= 3)
).astype(int)

cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
print(f"범주형 컬럼 수: {len(cat_cols)}")
for col in cat_cols:
    print(f"{col}: {df_clean[col].nunique()}개 → {df_clean[col].unique()}")

# ================================================================
# 범주형 컬럼 인코딩
# ================================================================

# (1) 원본 파생변수로 대체된 컬럼 제거
drop_encoded = ['특정_시술_유형', '시술_당시_나이', '난자_출처']
df_clean = df_clean.drop(columns=drop_encoded, errors='ignore')

# (2) 라벨 인코딩 — 시술_유형 (IVF=0, DI=1)
df_clean['시술_유형'] = df_clean['시술_유형'].map({'IVF': 0, 'DI': 1})

# (3) 순서형 인코딩 — 임신_위험도_범주
risk_order = {'미분류': 0, '정상_임신군': 1, '고위험_임신군': 2, '초고위험_임신군': 3}
df_clean['임신_위험도_범주'] = df_clean['임신_위험도_범주'].map(risk_order)

# (4) 순서형 인코딩 — 난자_기증자_나이
egg_donor_order = {'알 수 없음': 0, '만20세 이하': 1, '만21-25세': 2, '만26-30세': 3, '만31-35세': 4}
df_clean['난자_기증자_나이'] = df_clean['난자_기증자_나이'].map(egg_donor_order)

# (5) 순서형 인코딩 — 정자_기증자_나이
sperm_donor_order = {'알 수 없음': 0, '만20세 이하': 1, '만21-25세': 2, '만26-30세': 3,
                    '만31-35세': 4, '만36-40세': 5, '만41-45세': 6}
df_clean['정자_기증자_나이'] = df_clean['정자_기증자_나이'].map(sperm_donor_order)

# (6) 원핫 인코딩 — 시술_분류_그룹, 배아_생성_주요_이유
# [Fix #3] pd.get_dummies(전체) → sklearn OneHotEncoder로 교체
#           Train에서만 fit → Test에는 transform만 적용 (handle_unknown='ignore'로 미지 카테고리 안전 처리)
from sklearn.preprocessing import OneHotEncoder

ohe_cols = ['시술_분류_그룹', '배아_생성_주요_이유']

# ohe.fit 전 타입 통일 (혼합 타입으로 인한 오류 방지)
for col in ohe_cols:
    df_clean[col] = df_clean[col].astype(str)

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
ohe.fit(df_clean[ohe_cols])   # Train에서만 fit

# Train 변환
train_encoded = ohe.transform(df_clean[ohe_cols])
ohe_feature_names = ohe.get_feature_names_out(ohe_cols)
df_encoded = pd.DataFrame(train_encoded, columns=ohe_feature_names, index=df_clean.index)
df_clean = pd.concat([df_clean.drop(columns=ohe_cols), df_encoded], axis=1)

# Unknown = ICI, Generic DI, FER, GIFT 등 기타 시술 포함 → 컬럼명 변경
# 해당 컬럼이 없을 경우 에러 방지를 위해 errors='ignore' 추가
df_clean = df_clean.rename(columns={'시술_분류_그룹_Unknown': '시술_분류_그룹_기타'}, errors='ignore')

# 검증
print("남은 범주형 컬럼:")
print(df_clean.select_dtypes(include='object').columns.tolist())
print(f"\n전체 컬럼 수: {len(df_clean.columns)}")

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

cont_cols = [
    '나이_수치',
    '이식된_배아_수',
    '총_시술_횟수_clip',
    '총_임신_횟수_clip',
    '수집된_신선_난자_수_log',
    '임신_성공_여부',
]

g = sns.pairplot(
    df_clean[cont_cols].sample(3000, random_state=42),
    hue='임신_성공_여부',
    palette={0: '#B4B2A9', 1: '#1D9E75'},
    kind='kde',          # 산점도 → KDE
    diag_kind='kde',     # 대각선도 KDE
    plot_kws={'alpha': 0.5, 'fill': True},
    diag_kws={'fill': True, 'alpha': 0.5},
)

# 범례 라벨 변경
g.legend.set_title('임신 성공 여부')
for t, label in zip(g.legend.texts, ['실패 (0)', '성공 (1)']):
    t.set_text(label)

plt.suptitle('주요 연속형 변수 KDE 페어플롯', fontsize=14, y=1.02)
plt.savefig('pairplot_kde.png', dpi=120, bbox_inches='tight')
plt.show()

cont_cols = [
    '나이_수치',
    '이식된_배아_수',
    '총_시술_횟수_clip',
    '총_임신_횟수_clip',
    '수집된_신선_난자_수_log'
]

corr = df_clean[cont_cols + ['임신_성공_여부']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.title('연속형 변수 상관관계', fontsize=13)
plt.tight_layout()
plt.savefig('corr_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ================================================================
# 파생변수 최종 추가
# ================================================================

# 1. 기존 누락 확인용 (Blastocyst_Transfer)
print('Blastocyst 있음:', any('Blastocyst' in c for c in df_clean.columns))

# 2. 단일배아/과다배아/고령과다이식 (대화 중 생성 — 혹시 없으면 여기서 생성)
df_clean['단일배아_이식'] = (df_clean['이식된_배아_수'] == 1).astype(int)
df_clean['과다배아_이식'] = (df_clean['이식된_배아_수'] >= 3).astype(int)
df_clean['고령_과다이식'] = (
    (df_clean['나이_수치'] >= 38) & (df_clean['이식된_배아_수'] >= 3)
).astype(int)

# 3. 신규 파생변수
df_clean['유전검사_시행'] = (
    (df_clean['PGD_시술_여부'] == 1) | (df_clean['PGS_시술_여부'] == 1)
).astype(int)

df_clean['복합_불임_원인'] = (
    (df_clean['남성_요인_존재'] == 1) & (df_clean['여성_요인_존재'] == 1)
).astype(int)

df_clean['고령_반복시술'] = (
    (df_clean['나이_수치'] >= 38) & (df_clean['총_시술_횟수_clip'] >= 3)
).astype(int)

df_clean['동결_기증_복합'] = (
    (df_clean['동결_배아_사용_여부'] == 1) & (df_clean['기증_배아_사용_여부'] == 1)
).astype(int)

df_clean['배아출처_불명'] = (
    (df_clean['신선_배아_사용_여부'] == 0) & (df_clean['동결_배아_사용_여부'] == 0)
).astype(int)

df_clean['배아_이식_효율'] = (
    df_clean['이식된_배아_수'] / (df_clean['총_생성_배아_수_log'] + 1)
).round(4)

# 파생변수로 대체된 컬럼이 아직 남아있는지 확인
# 제거했어야 할 원본 컬럼들
check_cols = [
    '특정_시술_유형',   # → 시술_분류_그룹으로 대체
    '시술_당시_나이',   # → 나이_수치, 임신_위험도_범주로 대체
    '난자_출처',       # → 기증_난자_여부로 대체
]

print("제거 대상 컬럼 잔존 여부:")
for col in check_cols:
    print(f"{col}: {'존재' if col in df_clean.columns else '제거됨'}")

print()

# 컬럼명 확인 — 공백/특수문자 포함된 컬럼 찾기
problematic = [c for c in df_clean.columns if ' ' in c or ',' in c]
print("공백/쉼표 포함 컬럼:")
for c in problematic:
    print(f"  '{c}'")

# 컬럼명 정리 — 공백→언더바, 쉼표/슬래시 제거
df_clean.columns = (df_clean.columns
    .str.replace(' ', '_')
    .str.replace(',', '')
    .str.replace('/', '_')
)

# 확인
problematic = [c for c in df_clean.columns if ' ' in c or ',' in c]
print(f"공백/쉼표 포함 컬럼 잔존: {len(problematic)}개")
print(f"전체 컬럼 수: {len(df_clean.columns)}")
print("현재 컬럼 목록:")
print(df_clean.columns.tolist())

# 다중공선성 문제로 log/clip/이진 버전 최종 컬럼 정리
remove_cols = [
    # 원본 배아/난자 수치 (log/clip/이진 버전으로 대체)
    '총_생성_배아_수', '미세주입된_난자_수', '미세주입에서_생성된_배아_수',
    '저장된_배아_수', '미세주입_후_저장된_배아_수', '해동된_배아_수',
    '해동_난자_수', '수집된_신선_난자_수', '저장된_신선_난자_수',
    '혼합된_난자_수', '파트너_정자와_혼합된_난자_수', '기증자_정자와_혼합된_난자_수',
    # 시각화용 임시 컬럼
    '시술횟수_구간',
]

df_clean = df_clean.drop(columns=remove_cols, errors='ignore')

print(f"전체 컬럼 수: {len(df_clean.columns)}")
print(f"전체 행 수: {len(df_clean)}")

df_clean.columns

# 전처리 완료된 데이터 저장
df_clean.to_csv('df_clean_preprocessed_v2.csv', index=False)
print(f"저장 완료")
print(f"shape: {df_clean.shape}")

# ================================================================
# [Fix 종합] Test 데이터에 Train 기준값 그대로 적용 (transform only)
# ================================================================
print("\n=== Test 데이터 전처리 (Train 기준값 재사용) ===")

# --- 기본 전처리 (구조적 결측·파생변수 동일 적용) ---
df_clean_test.columns = df_clean_test.columns.str.strip()

# Fix #2: 나이 결측 → Train 중앙값(AGE_MEDIAN_FILLNA) 그대로 사용
df_clean_test['나이_수치'] = df_clean_test['시술_당시_나이'].apply(
    lambda x: age_info.get(x, {'val': np.nan})['val']
)
df_clean_test['나이_수치'] = df_clean_test['나이_수치'].fillna(AGE_MEDIAN_FILLNA)
print(f"[Test 나이 결측 대체] 사용값: {AGE_MEDIAN_FILLNA} (Train 중앙값 고정)")

# Fix #1: IQR 클리핑 → Train에서 계산한 iqr_clip_bounds 재사용
for col, upper in iqr_clip_bounds.items():
    if col in df_clean_test.columns:
        df_clean_test[col + '_clip'] = df_clean_test[col].clip(upper=upper)
        print(f"[Test IQR 클리핑] {col} 상한값: {upper:.1f} (Train 기준 고정)")

# Fix #3: OHE → 이미 fit된 ohe 객체로 transform만 수행
if '시술_분류_그룹' in df_clean_test.columns and '배아_생성_주요_이유' in df_clean_test.columns:
    test_encoded = ohe.transform(df_clean_test[ohe_cols])   # fit 없이 transform만
    df_test_enc  = pd.DataFrame(test_encoded, columns=ohe_feature_names, index=df_clean_test.index)
    df_clean_test = pd.concat([df_clean_test.drop(columns=ohe_cols), df_test_enc], axis=1)
    df_clean_test = df_clean_test.rename(columns={'시술_분류_그룹_Unknown': '시술_분류_그룹_기타'})
    print(f"[Test OHE] Train fit 객체로 transform 완료 → 컬럼 수: {len(df_test_enc.columns)}")

# 컬럼 일치 검증
train_cols = set(df_clean.columns) - {'임신_성공_여부'}
test_cols  = set(df_clean_test.columns)
missing_in_test  = train_cols - test_cols
extra_in_test    = test_cols  - train_cols
print(f"\n[컬럼 일치 검증]")
print(f"  Test에 없는 컬럼: {missing_in_test  if missing_in_test  else '없음 ✅'}")
print(f"  Test에만 있는 컬럼: {extra_in_test  if extra_in_test    else '없음 ✅'}")

df_clean_test.to_csv('df_clean_test_preprocessed_v2.csv', index=False)
print(f"\nTest 저장 완료 | shape: {df_clean_test.shape}")