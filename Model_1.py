
# ================== housing의 데이터 구조를 확인 ========================== #

from sklearn.datasets import fetch_california_housing
import numpy as np

# 사이킷런 선형회귀 테스트 데이터베이스 (정답 파일을 제공한다)
housing = fetch_california_housing()

# 데이터의 전체 구조
print(f"데이터의 전체 구조 => {', '.join(housing.keys())}")
# 속성의 개수
print(f"속성의 개수 => {', '.join(housing.feature_names)}")
# 각 속성의 데이터 값
print(f"각 속성의 데이터 값 => {housing.data.shape}")
# 각 속성의 정답의 값
print(f"각 속성의 정답의 값 => {housing.target.shape}")
# data의 0번째(첫번째 레코드) index의 값을 출력
print(housing.data[0])


# ================== housing의 data 부분의 데이터프레임을 생성해 값들을 확인 ========================== #

import pandas as pd

# data 자료를 df 변수에 저장
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# 데이터 프레임의 정보를 표시
print(df.info())
# 데이터 프레임의 각 속성별의 통계값을 표시
print(df.describe())


# ================== 결측값 생성 ========================== #

# 속성이 문자열이 아닌 index일때 사용
df.iloc[10, df.columns.get_loc('HouseAge')] = None
df.iloc[15, df.columns.get_loc('HouseAge')] = None

# 속성이 문자열일때 사용
df.loc[10, 'HouseAge'] = None
df.loc[15, 'HouseAge'] = None

# 결측값과 정상값을 출력
print("결측값 개수 : ",df["HouseAge"].isna().sum())
print("정상값 개수 : ",df["HouseAge"].notnull().sum())

# 결측값 보정
df["HouseAge"] = df['HouseAge'].fillna(df['HouseAge'].mean())
print("결측값 개수 : ",df["HouseAge"].isna().sum())


# 또 다른 훈련테스트를 위해 복제본 데이터 프레임 저장
cpdf = df.loc[:,:]
print(cpdf)


# ========== 정규화가 안된 data를 훈련하기 위해 분할 ============= #

from sklearn.model_selection import train_test_split

# 넘파이말고도 판다스 데이터도 넣을 수 있다!
x_train, x_test , y_train, y_test = train_test_split(df, housing.target, random_state=11)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# ========== 정규화가 안된 data를 (traget은 되어있음) 가지고 훈련을 함 ============= #

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(32, activation="relu", input_shape=(8,)))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

# metrics=["mae"] : 절대 오차의 평균
model.compile(loss="mse", optimizer="adam", metrics=['mae'])
# batch_size : 가중치 업데이트의 배치수를 지정
result = model.fit(x_train, y_train, batch_size=10, epochs=50)

print(result.history.keys())


# ========== 정규화가 안된 data의 훈련 결과를 출력 ============ #

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.plot(result.history['loss'])
plt.plot(result.history['mae'])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["train"], loc=("upper right"))
plt.show()


# ========== 정규화가 안된 data의 훈련값을 산점도로 출력 ============ #

y_pred = model.predict(x_test)
print(y_pred)

plt.scatter(y_test,y_pred) # 정답과 예측값을 xy좌표로 나타낸다
plt.ylabel("y_pred")
plt.xlabel("y_test")
plt.plot(y_test,y_test,'r') # 정답을 직선으로 나타낸다
plt.show()


# ========== data를 정규화시킴 ============ #

from sklearn.preprocessing import StandardScaler

# scaler 객체를 생성
scale = StandardScaler()
# 평균과 표준편차(또는 다른 통계적 값들)를 계산하여 scaler 객체에 저장 (원본변환)
cpdf = scale.fit_transform(cpdf)


# ========== data를 정규화시킨 값을 다른 데이터 프레임에 넣어 출력 ============= #

# 계산한 통계적 값을 사용하여 다른 데이터프레임을 표준편차 값으로 변환
dfpd = pd.DataFrame(cpdf,columns=housing.feature_names)

print(dfpd.head(5)) # 각속성의 앞부분 index 5개의 값들을 확인
print(dfpd.describe()) # 정규분포된 데이터 프레임의 각 속성별의 통계값을 표시


# ========== 정규화를 시킨 데이터를 다시 분할 ============= #

xx_train, xx_test, yy_train, yy_test = train_test_split(cpdf,housing.target,random_state=11)

print(xx_train.shape)
print(yy_train.shape)
print(xx_test.shape)
print(yy_test.shape)

# ========== 정규화가 된 data를 (traget은 되어있음) 가지고 훈련을 함 ============= #

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

# early_stopping 객체를 생성
early_stopping = EarlyStopping(
    monitor='accuracy',       # 검증 데이터의 정확도를 모니터링
    min_delta=0.001,              # 정확도 변화의 최소 개선치
    patience=10,                  # 개선이 없는 에포크 수
    verbose=1,                    # 훈련 과정 출력 여부
    mode='max'                    # 정확도는 최대화가 목표
)


model2 = Sequential()

model2.add(Dense(32, activation="relu", input_shape=(8,)))
model2.add(Dense(16, activation="relu"))
model2.add(Dense(8, activation="relu"))
model2.add(Dense(1))

# metrics=["mae"] : 절대 오차의 평균
model2.compile(loss="mse", optimizer="adam", metrics=['acc'])
# batch_size : 가중치 업데이트의 배치수를 지정
result2 = model2.fit(xx_train, yy_train, batch_size=10, epochs=50, callbacks=[early_stopping])

print(result2.history.keys())


# ========== 정규화가 된 data의 훈련 결과를 출력 ============ #

plt.figure(figsize=(8,5))
plt.plot(result2.history['loss'])
plt.plot(result2.history['acc']) # 선형회귀 데이터 특성상 정확도는 크게 의미가 없다
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["train","acc"], loc=("upper right"))
plt.show()


# ========== 정규화가 된 data의 훈련값을 산점도로 출력 ============ #

yy_pred = model2.predict(xx_test)
print(yy_pred)

plt.scatter(yy_test, yy_pred) # 정답과 예측값을 xy좌표로 나타낸다
plt.ylabel("y_pred")
plt.xlabel("y_test")
plt.plot(yy_test, yy_test,'r') # 정답을 직선으로 나타낸다
plt.show()


# ========== 정규화가 된 data의 훈련값을 평가 ============ #

eval = model2.evaluate(yy_test,yy_pred)
print(eavl)
