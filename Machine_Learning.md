 Hands-On Machine Learning with Scikit-Learn & TensorFlow
========================================================
Sohee Hwang, Start: 2019/8/8
--------------------------------------------------------

# Chapter1. 한눈에 보는 머신러닝

## 1.1. 머신러닝이란?
> 머신러닝이란, 명시적인 프로그래밍 없이 컴퓨터가 학습하는 능력을 갖추게 하는 연구 분야다.<br>
> 어떤 작업 T에 대한 컴퓨터 프로그램의 성능을 P로 측정했을 때 경험 E로 인해 성능이 향상됐다면, 이 컴퓨터 프로그램은 작업 T와 성능 측정 P에 대해 경험 E로 학습한 것이다.

## 1.2. 왜 머신러닝을 사용하는가?
> 머신러닝 기술을 적용하게 되면 대용량의 데이터를 분석하면 겉으로는 보이지 않던 패턴을 발견할 수 있기 때문이다.
>> * Ex1. 스팸 필터를 예로 들었을 때, 자주 등장하는 단어를 직접 업데이트 해주지 않아도, 자동으로 스팸을 분류해줌.<br>
>> * Ex2. Speech recognition의 경우, 'one'과 'two'의 사운드를 구분하는 프로그램을 작성할 때, 전통적인 방식으로는 하드코딩이 답이지만 머신러닝을 사용하면 간단해짐.

## 1.3. 머신러닝 시스템의 종류
총 3가지의 분류 기준으로 분류 가능
1. 사람의 감독 하에 훈련하는 것인지 그렇지 않은 것인지
2. 실시간으로 점진적인 학습을 하는지 아닌지
3. 단순하게 알고 있는 데이터 포인트와 새 데이터 포인트를 비교하는 것인지 아니면 훈련 데이터셋에서 과학자들처럼 패턴을 발견하여 예측 모델을 만드는지

> ### 1.3.1 지도학습과 비지도학습<br>
>> **학습하는 동안의 감독 형태나 정보량** 이 기준<br>
>>> #### 지도학습<br>
>>> * 알고리즘에 주입하는 훈련 데이터에 레이블이라는 원하는 답이 포함됨<br>
>>> * **분류**가 가장 대표적인 지도학습 작업(스팸필터)<br>
>>> * **회귀**가 그다음으로 대표적인 지도학습 작업 (예측변수라 부르는 특성(feature)을 사용해 타겟 수치를 예측하는 것)<br>
>>> * k-NN / Linear Regression / Logistic Regression / SVM / Decision Tree / Random Forest / Neural Network<br>
>>> #### 비지도학습<br>
>>> 훈련데이터에 레이블이 없어서 시스템이 아무런 도움 없이 학습해야하는 방법<br>
>>> * Clutsering (K-Means, HCA, Expectation Maximization)<br>
>>> * Visualizaition & dimensionality reduction(PCA, Kernel PCA, LLA, t=SNE)<br>
>>> * Association rule learning(Apriori, Eclat)<br>
##### 이어해라



> ### 1.3.2 배치학습과 온라인 학습


>>

> ### 1.3.3 사례 기반 학습과 모델 기반 학습
>> 머신러닝 시스템은 어떻게 **일반화** 되는가를 기준<br>
>> 주어진 훈련 데이터로 학습하지만 훈련 데이터는 본적 없는 새로운 데이터로 일반화 되어야 한다는 뜻

>>> #### 사례 기반 학습
>>> 시스템이 사례를 기억함으로써 학습하는 것
>>> Ex1. 스팸 필터에서 메일 사이의 유사도를 측정해서 스팸을 분류하는 것 <br>
>>> #### 모델 기반 학습
>>> 샘플로부터 일반화 시키는 다른 방법은 이 샘플들의 모델을 만들어 예측에 사용하는 것
>>> 데이터를 분석-> 모델 선택 -> 훈련 데이터로 모델을 훈련 -> 새로운 데이터에 모델을 적용해 예측하고 일반화되길 기대


## 1.4. 머신러닝의 주요 도전 과제

> #### 충분하지 않은 양의 훈련 데이터
> - 간단한 문제도 수천 개의 데이터가 필요 

> #### 대표성이 없는 훈련 데이터
> - sampling bias를 조심해야 함

> #### 낮은 품질의 데이터
> - outlier 최소화

> #### 관련 없는 특성
> - garbage in, garbage out<br>
> - Feature Engineering이 중요
>> + Feature selection: 가지고 있는 특성중에 훈련에 가장 유용한 특성을 선택
>> + Feature Extraction: 특성을 결합하여 더 유용한 특성을 만듬

> #### 훈련 데이터 과대적합
> 훈련 데이터에 너무 잘 맞지만 일반성이 떨어지는 경우
> Ex1. 택시운전사가 내 물건을 훔쳤을 때, 모든 택시운전기사를 도둑이라고 생각하는 것
> 훈련 데이터에 있는 잡음의 양에 비해 모델이 너무 복잡할 때 일어남
>> 해결법1. 파라미ㅌ 수가 적은 모델을 선택하거나, 훈련 데이터에 있는 특성 수를 줄이거나 모델에 제약을 가해 단순화
>> 해결법2. 훈련 데이터를 더 많이 모은다
>> 해결법3. 훈련데이터의 bias를 줄인다.

> #### 훈련 데이터 과소적합
> 과대적합의 반대
> 모델이 너무 단순해서 데이터의 내재된 구조를 학습하지 못할 때 일어나는 현상

# Chapter2. 머신러닝 프로젝트 처음부터 끝까지

## 2.1. 실제 데이터로 작업하기
> 여기에서 사용하는 데이터는 캘리포니아 주택가격 데이터 셋을 사용

## 2.2. 큰 그림 보기
> ### 2.2.1. 문제 정의
>> 비즈니스의 목적이 정확히 무엇인가요?
>> 현재 솔루션은 어떻게 구성되어 있나요?
>> 지도학습/비지도학습/강화학습 중 무엇일까요?
>> 분류나 회귀인가요 아니면 다른 어떤 작업인가요?
>> 배치학습과 온라인학습 중 어떤 것을 사용해야 하나요?
>> * 현재 예시에서는 레이블된 training set이 있기에 지도 학습이며 값을 예측해야하므로 회귀문제입니다. 데이터에 연속적이 흐름이 없으므로 빠르게 변하는 데이터에 적응하지 않아도 되고, 데이터가 메모리에 들어갈 마ㄴ큼 충분히 작기 때문에 일반적인 배치 학습이 적절합니다.

> ### 2.2.2. 성능 측정 지표 선택
>> 회귀 문제의 전형적 성능지표: RMSE(평균 제곱근 오차)
>> 이상치로 보이는 구역이 많을 경우: MAE(평균 절대 오차)

> ### 2.2.3. 가정 검사
>> 지금까지 만든 가정을 검사!
>> 예를 들어, 가격이 입력으로 들어가게 되는데, 이를 카테고리화(저렴,보통,고가) 같은 거로 바꾸는게 더 좋을지 검사

## 2.3 데이터 가져오기
> ### 2.3.1. 작업환경 만들기
> ### 2.3.2. 데이터 다운로드
> ### 2.3.3. 데이터 구조 훑어보기
>> pandas 
>> * head(): 첫 다섯 행
>> * info(): 데이터에 대한 간략한 설명과 전체 행 수, 각 특성의 데이터 타입과 널이 아닌 값의 개수
>> * value_counts(): 타입이 텍스트 타입일 경우 사용. 카테고리의 종류, 카테고리의 양 확인
>> * describe(): 숫자형 특성의 요약 정보
>> hist(): 막대그래프

> ### 2.3.4. 테스트 세트 만들기
>> * sklearn.model_selection import train_test_split 사용하면 편해
>> * skleran.model_selection import StratifiedShuggleSplit : 계층 샘플링 (예) 남/여 비율 맞춰서 샘플링

## 2.4. 데이터 이해를 위한 탐색과 시각화
> ### 2.4.1. 지리적 데이터 시각화
>> * 기본 scatter plot

<pre><code>
housing.plot(kind="scatter", x="logitude", y="latitude", alpha=0.1)

</code></pre>

>> * 옵션 들어간 scatter plot
>> 1. s / 원의 반지름은 구역의 인구 수
>> 2. c / 색깔은 가격

<pre><code>
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', figsize=(10,7),
             c='median_house_value', cmap=plt.get_cmap('jet), colorbar=True, shareex=False)

</code></pre>

> ### 2.4.2. 상관관계 조사
> 데이터 셋이 너무 크지 않으므로 모든 특성 간의 표준 상관계수(피어슨의 r) 를 corr()로 계산

<pre><code>
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

</code></pre>
> 특성 간의 관계를 확인하기 위해 산점도를 그려줄 수 있음

<pre><code>
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

</code></pre>

> ### 2.4.3. 특성 조합으로 실험

## 2.5. 머신러닝 알고리즘을 위한 데이터 준비
> ### 2.5.1. 데이터 정제
>> dropna(), drop(), fillna()
>> * scikit learn의 Imputer: 누락된 값 손쉽게 다루기

<pre><code>
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')
imputer.fit(housing_num)

imputer.statisctics_ #각 특성의 중간값을 계산해서 저장해놓은 것
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))
</code></pre>

> ### 2.5.2. 텍스트와 범주형 특성 다루기
>> * pandas.factorize() : 카테고리를 텍스트에서 숫자로 바꿔줌
<pre><code>
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded = [1, 0, 2, ... ]
housing_categories = ['<1H OCEAN', 'NEAR OCEAN', ... ]

</code></pre>

>> * OneHotEncoder: One-hot encoding
<pre><code>
from sklearn.preprocessing import OneHotEncoder

# 얘는 텍스트->숫자->원핫 벡터 과정을 사용
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot.toarray()

# 변환 한번에 해버리기
import sklearn.preprocessing import CategoricalEncoder
cat_encoder = CategoricalEncoder() # 밀집 행렬을 원할 경우, CategoricalEncoder(encoding='onehot-dense')

housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
housing_cat_1hot = cat_encoder.fit(transform(housing_cat_reshaped)

cat_encoder.categories_ = ['1H OCEAN', 'INLAND', 'NEAR BAY', ...]
</pre></code>

> ### 2.5.3. 나만의 변환기
!! 이건 하고 싶을 떄 다시봐랏

> ### 2.5.4 특성 스케일링



 
 
 
 
 
 






<pre><code>
tf.Session()

</code></pre>



