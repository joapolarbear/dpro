import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# load true labels
label_dict = {}
times = []
with open("running_time.txt", "r") as f:
    for line in f:
        splitted = line.split(":")
        sample_id = int(splitted[0].strip())
        time = float(splitted[1].strip())
        if time > 1000:
            continue
        label_dict[sample_id] = time
        times.append(time)

# load feature vecs
feature_dict = {}
feature_file_names = os.listdir("./dataset/feature_vecs")
for fn in feature_file_names:
    vec = []
    sample_id = int(fn.split(".txt")[0].strip())
    with open(os.path.join("./dataset/feature_vecs", fn), "r") as f:
        for line in f:
            value = float(line)
            if value == -1:
                value = 0
            vec.append(value)
    feature_dict[sample_id] = np.array(vec)

X_list = []
y_list = []

for sample_id in feature_dict.keys():
    if sample_id in label_dict:
        X_list.append(feature_dict[sample_id])
        y_list.append(label_dict[sample_id])

X = np.array(X_list)
X = normalize(X, axis=0, norm='max')
y = np.array(y_list)
y_max = y.max()
y_average = np.average(y)
# y = y / y_max

# remove all zero columns
X = X[:, ~np.all(X == 0, axis=0)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

params = {
        'learning_rate': [0.1],
        'colsample_bytree': [0.5, 0.6, 0.8, 1.0],
        'colsample_bynode': [0.2, 0.4, 0.6, 0.8, 1.0],
        'reg_alpha': [0.8, 1, 1.5, 2],
        'reg_lambda': [1,2,3,4,5],
        'max_depth': [50],
        'n_estimators': [200],
        'objective': ['reg:squarederror'],
}

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bynode=0.5, colsample_bytree=0.4, reg_alpha=1.5, reg_lambda=3, learning_rate=0.1, max_depth=50, n_estimators=200)
xg_reg.fit(X_train, y_train)

skf = KFold(n_splits=5, shuffle = True, random_state = 1001)
grid_search = GridSearchCV(xg_reg, params, scoring='neg_mean_squared_error', n_jobs=16, refit=True, cv=skf, verbose=2, return_train_score=True)

grid_search.fit(X_train, y_train)

print('\n Best score:')
print(grid_search.best_score_)
print('\n Best hyperparameters:')
print(grid_search.best_params_)

xg_reg_best = grid_search.best_estimator_

predicted = xg_reg_best.predict(X_test)
residual = []
for i in range(len(predicted)):
    residual.append(np.abs(predicted[i]-y_test[i]) / y_test[i])

print("Average: {}, Median: {}".format(np.average(residual), np.median(residual)))