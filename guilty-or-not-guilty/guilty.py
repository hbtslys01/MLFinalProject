from sklearn.utils import shuffle
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from datetime import datetime as dt


def get_time(t):
    dt.strptime(t, "%m/%d/%Y")


a = get_time("06/20/2015")
b = get_time("06/08/2020")
c = get_time("06/30/2010")


def process_data(data):

    temp = data[['State', 'License Type', 'Violation',
                 'County', 'Issuing Agency', 'Violation Status']]
    print(temp.head(10))
    # enc = OrdinalEncoder()
    # # enc.fix(temp)
    # after_proce = enc.fit_transform(temp)
    # min_max_scaler = MinMaxScaler()
    # X_train_minmax = min_max_scaler.fit_transform(data['Summons Number'])
    # issue_date = data['Issue Date']
    # issue_date = issue_date.apply(get_time)
    # issue_date.loc[issue_date >= a and issue_date <= b] = 2
    # issue_date.loc[issue_date >= c and issue_date < a] = 1
    # issue_date.loc[issue_date < c or issue_date > b] = 0
    # print(issue_date.head(20))


#import matplotlib.pyplot as plt
data_train = pd.read_csv('train.csv', header=None)  # read dataset
# data_train_y = data_train.iloc[:, -1]
# data_train_x = data_train.iloc[:, 0:-1]
# data_test = pd.read_csv('test.csv', header=None)  # read dataset
# data_test_y = data_test.iloc[:, -1]
# data_test_x = data_test.iloc[:, 0:-1]
process_data(data_train)
# D = [2, 3, 4]
# n = np.array(range(20))
# C = 0.001*2**n
# max_acc = np.zeros(3)
# for i in range(3):
#     accuracy_tst = []
#     accuracy_trn = []
#     d = D[i]
#     # ========Your Code Here============
#     for j in C:
#         clf = SVC(C=j, kernel='poly',
#                   degree=D[i], random_state=0, gamma='auto')
#         clf.fit(data_train_x, data_train_y)
#         s = clf.score(data_test_x, data_test_y)
#         accuracy_tst.append(s)
#         accuracy_trn.append(clf.score(data_train_x, data_train_y))
#         max_acc[i] = max(max_acc[i], s)
#     print('The maximum testing accuracy achieved with Linear SVM is: ' + str(max_acc))
