# Kaggle Bike Sharing Demand
# Joey L. Maalouf
# Approach: Support Vector Regression

################################################################################
# import any necessary modules

import csv
from sklearn.svm import SVR
import matplotlib.pyplot as plt

################################################################################
# define our functions


def to_int(input):
    try:
        return int(input)
    except TypeError:
        return [int(input[0]), int(input[1])]


def read_data(filename, xy):
    datalist = []
    # read in the file
    data = open(filename)
    reader = csv.reader(data, delimiter=',')
    for row in reader:
        if (xy == 'x'):
            # store just the hour and weather
            datalist.append([row[0][11:13], row[4]])
        elif (xy == 'y'):
            # store just the count
            datalist.append(row[11])
    return [to_int(i) for i in datalist[1:]]

################################################################################
# read in the data

print("Let's start reading in the data...")
x_train = read_data('train.csv', 'x')
x_test = read_data('test.csv', 'x')
y_train = read_data('train.csv', 'y')
print("Finished reading in the data!\n")

# shorten datasets for the sake of seeing if the model works:
# (take out later)
# x_train = [x1 for [x1, x2] in x_train[0:10]]
# x_test = [x1 for [x1, x2] in x_test[0:10]]
# x_train = x_train[0:10]
# x_test = x_test[0:10]
# y_train = y_train[0:10]
# maybe keep these?
# x_train = (x_train, 2)
# x_test = (x_test, 2)
# print(x_train)
# print(x_test)

################################################################################
# fit regression model

print("Let's start creating our trainers...")
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)  # increase degree (better fit)?
print("Finished creating our trainers!\n")
print("Let's start training our models...")
model_rbf = svr_rbf.fit(x_train, y_train)
# model_lin = svr_lin.fit(x_train, y_train)
model_poly = svr_poly.fit(x_train, y_train)
print("Finished training our models!\n")
print("Let's start predicting our new data...")
y_test_rbf = model_rbf.predict(x_test)
# y_test_lin = model_lin.predict(x_test)
y_test_poly = model_poly.predict(x_test)
print("Finished predicting our new data!\n")

################################################################################
# look at the results

# TODO: since we have 2 features, do a 3d plot! duh
plt.scatter(x_train, y_train, c='k', label='data')
plt.hold('on')
plt.plot(x_test, y_test_rbf, c='g', label='RBF model')
# plt.plot(x_test, y_test_lin, c='r', label='Linear model')
plt.plot(x_test, y_test_poly, c='b', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
