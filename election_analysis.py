import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import classification_report
import random

data = pd.read_csv('/Users/apple/Documents/ElectionData/GE2014/ElectionData.csv')

data1 = data.drop(['State'], axis = 1)
data2 = data1.iloc[:, 0:5]
data3 = data2[data2['Y'] < 2]
data4 = data3.sample(frac = 1)

train = data4[:230]
test = data4[230:]

train_x = train.drop(['Y'], axis = 1)
train_y = train['Y']

test_x = test.drop(['Y'], axis = 1)
test_y = test['Y']

train_pool = Pool(train_x, train_y, cat_features = [0, 3])
test_pool = Pool(test_x, test_y, cat_features = [0, 3])

model = CatBoostClassifier(verbose = False, iterations = 250)
model.fit(train_pool, eval_set = test_pool, use_best_model = True)

prob = model.predict_proba(test_x)
clas = model.predict(test_x)

print('Accuracy: %.2f' %(100*sum(test_y == clas)/len(clas)) + '%')