import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from utils import RWN, get_rfweight

def func(x):
    
    return (1 + x[:, 0] - x[:, 1] - 2 * x[:, 2]) ** 3 * (x[:, 0] - x[:, 1] > 0)


train_size = 200
test_size = 1000
p = 10
snrdb = 10

fvar = np.var(func(np.random.rand(10000, p)))
sigma = np.sqrt(fvar / (10 ** (snrdb / 10) - 1))


np.random.seed(123)

x_train = np.random.rand(train_size, p)
y_train = func(x_train) + sigma * np.random.randn(train_size)
x_test = np.random.rand(test_size, p)
y_test = func(x_test)

params_mrf = {
    'min_samples_split': [2, 3, 4, 5, 6, 7]
}

model = RandomForestRegressor(n_estimators=100)
reg_mrf = GridSearchCV(model, params_mrf)
reg_mrf.fit(x_train, y_train)

hs = 256 
batch_size = 100
n_iter = 2000
lr = 1e-3
tol = 1e-5
device = 'cuda'
d = False
verbose = True

tau = 1e-3

mrf = reg_mrf.best_estimator_
mrf.fit(x_train, y_train)
mrfw, mrfwn = get_rfweight(mrf, x_train)

model_rwn = RWN(hs, device)
model_rwn.fit(x_train, y_train, mrfw, tau, d, batch_size, n_iter, lr, tol, verbose)
y_pred = model_rwn.predict(x_test)
np.mean((y_test - y_pred) ** 2) #testing error




