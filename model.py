import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils import DataUtils
import seaborn as sns

data = DataUtils.get_country_data()
features = ['TimeUnit']
target = ['cured', 'death', 'active_cases']
data['TimeUnit'] = np.arange(len(data.index))
data['Lag1'] = data['active_cases'].shift(1).fillna(0)
data['Lag2'] = data['active_cases'].shift(2).fillna(0)
train, test = train_test_split(data, test_size=0.25, random_state=42)
X_train, y_train = train[['TimeUnit']], train[['active_cases']]
X_test, y_test = test[['TimeUnit']], test[['active_cases']]

lr = LinearRegression()
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
print('LR with TimeUnit')
print('R2:', r2_score(y_test, predicted))

X_train, y_train = train[['Lag1']], train[['active_cases']]
X_test, y_test = test[['Lag1']], test[['active_cases']]

lr = LinearRegression()
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
print('LR with Lag1')
print('R2:', r2_score(y_test, predicted))

X_train, y_train = train[['TimeUnit','Lag1']], train[target]
X_test, y_test = test[['TimeUnit','Lag1']], test[target]

lr = MultiOutputRegressor(LinearRegression())
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
print('MultiOutputRegressor with TimeUnit, Lag1')
print('R2:', r2_score(y_test, predicted))


lr = RegressorChain(LinearRegression(), order=[0,1,2])
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
print('RegressorChain with TimeUnit, Lag1')
print('R2:', r2_score(y_test, predicted))


