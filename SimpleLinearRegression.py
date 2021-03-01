######################################################
# SIMPLE LINEAR REGRESSION
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.float_format', lambda x: '%.2f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

######################################################
# SIMPLE LINEAR REGRESSION WITH OLS USING SCIKIT-LEARN
######################################################

df = pd.read_csv("datasets/salary.csv")

X = df[["YearsExperience"]]
y = df[["Salary"]]

# Scatter Graph
plt.scatter(X, y)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

reg_model = LinearRegression().fit(X, y) # ŷ = β0 + β1x
print("β0: ",reg_model.intercept_[0]) # β0
print("β1: ",reg_model.coef_[0][0]) # β1

y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)

# ŷ = 25792.2 + 9449.96 * YearsExperience
reg_model.intercept_[0] + reg_model.coef_[0][0]*6


g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},ci=False, color="r")
g.set_title(f"Model Denklemi: Salery = {round(reg_model.intercept_[0], 2)} + {round(reg_model.coef_[0][0], 2)} * YearsExperience")
g.set_xlabel("YearsExperience")
g.set_ylabel("Salary")
plt.show()


######################################################
# SIMPLE LINEAR REGRESSION WITH OLS FROM SCRATCH
######################################################

b1 = ((np.array(X) - np.array(df["YearsExperience"]).mean()) * (np.array(y) - np.array(df["Salary"]).mean())).sum() \
     / (((np.array(X) - np.array(df["YearsExperience"]).mean()) ** 2).sum())

b0 = np.array(df["Salary"]).mean() - b1 * np.array(df["YearsExperience"]).mean()


b0 + b1 * 6


######################################################
# SIMPLE LINEAR REGRESSION WITH GRADIENT DESCENT FROM SCRATCH
######################################################

df = pd.read_csv("datasets/Advertising.csv")
X = df["radio"]
Y = df["sales"]

# Cost function
def cost_function(Y, b, w, X):
    m = len(Y)  # gözlem sayısı
    sse = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse / m
    return mse

# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return b, w


# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000


train(Y, initial_b, initial_w, X, learning_rate, num_iters)

######################################################
# Model Validation (Holdout, Cross Validation)
######################################################

df = pd.read_csv("datasets/Advertising.csv", index_col=0)
df.head()
df.info()
df.shape

X = df[["TV"]]
y = df[["sales"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

######################################################
# Model
######################################################
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_model.intercept_
reg_model.coef_
reg_model.score(X_train, y_train)

g = sns.regplot(x=X_train, y=y_train, scatter_kws={'color': 'b', 's': 9},
                ci=False, color="r")
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()


######################################################
# Tahmin & Tahmin Başarısını Değerlendirme
######################################################

# y_hat = b + wX
# y_hat = 6.8 + 0.05*TV


# Tahmin başarısını değerlendirmek için 4 yol var.
# 1. Tüm veri ile model kur tüm veri ile hataya bak.
# 2. train seti ile model kur test seti ile hataya bak. (holdout)
# 3. K Fold CV ile tüm veri üzerinden hataya bak.
# 4. Veriyi en baştan train test şeklinde ayır.
# Train setine CV uygula validasyon hatasına bak test seti için de test hatana bak.

# train hatası
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# test hatası
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))

######################################################
# MULTIPLE LINEAR REGRESSION
######################################################


df = pd.read_csv("datasets/Advertising.csv", index_col=0)

X = df.drop('sales', axis=1)
y = df[["sales"]]


fig,axs= plt.subplots(1,3,sharey=True) # sharey : share same y axis across the plot
df.plot(kind="scatter",x='TV',y='sales',ax=axs[0],figsize=(10,5))
df.plot(kind="scatter",x='radio',y='sales',ax=axs[1],figsize=(10,5))
df.plot(kind="scatter",x='newspaper',y='sales',ax=axs[2],figsize=(10,5))
plt.show()

######################################################
# MODEL
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=1)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_model.intercept_
reg_model.coef_
reg_model.score(X_train, y_train)


######################################################
# Tahmin & Tahmin Başarısını Değerlendirme
######################################################

# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

# train hatası
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# test hatası
y_pred = reg_model.predict(X_test)
print("mse: ", mean_squared_error(y_test, y_pred))
print("rmse: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("mae: ",  mean_absolute_error(y_test, y_pred))

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10, scoring="neg_mean_squared_error")))