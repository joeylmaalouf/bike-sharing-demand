import csv
# from matplotlib import pyplot
from sklearn.linear_model import SGDClassifier


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
            # store nothing but the hour
            datalist.append([row[0][11:13], row[4]])
        elif (xy == 'y'):
            # store nothing but the count
            datalist.append(row[11])
    datalist = [to_int(i) for i in datalist[1:]]
    return datalist

x_train = read_data('train.csv', 'x')
x_test = read_data('test.csv', 'x')
y_train = read_data('train.csv', 'y')

clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(x_train, y_train)
y_test = clf.predict(x_test)
for item in y_test:
    print(item)
# the problem: y_test is nothing but 5s and 88s
# probably because the y we pass in should be just two classes to decide between
# rather than many different numbers
# the solution: make many different classifiers, train on different sets of data
# make a classifier for count being between 0 and 50 or not, 51-100 or not, etc.
# for i, val in enumerate(x_train, y_train):
#     if (val >= 0 and val <= 50):
#         x_train_0.append(x_train(i))
#         y_train_0.append(val)
#     else if (val > 50 and val <= 100):
#         x_train_51.append(x_train(i))
#         y_train_51.append(val)
#     etc.
# clf0.fit(x_train_0, y_train_0)
# clf51.fit(x_train_51, y_train_51)
