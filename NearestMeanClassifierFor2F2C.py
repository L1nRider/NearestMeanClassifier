# nearest mean classifier for data with 2 features and 2 classes
# data include train.csv and test.csv
# each .csv file has 100 rows and 3 columns
# the last column is class label which has 2 classes
# pick the first 2 columns to print on xy plain
# use train.csv to calculate mean
# then use this mean to calculate the error rate of test.csv
import csv
import numpy as np
from plotDecBoundaries import plotDecBoundaries

data_train = np.zeros((100, 2))
label_train = np.zeros(100, dtype=np.int)
mean_train = np.zeros((2, 2))

data_test = np.zeros((100, 2))
label_test = np.zeros(100, dtype=np.int)

train_row_cnt = 0
test_row_cnt = 0

sum1_x = 0
sum1_y = 0
sum2_x = 0
sum2_y = 0
class1_cnt = 0
class2_cnt = 0

error_train = 0
error_test = 0


# load train data
with open('dataset/synthetic1_train.csv') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        data_train[train_row_cnt, 0] = float(row[0])
        data_train[train_row_cnt, 1] = float(row[1])
        label_train[train_row_cnt] = int(row[2])
        train_row_cnt += 1

        # calculate sum for each class
        if int(row[2]) == 1:
            sum1_x += float(row[0])
            sum1_y += float(row[1])
            class1_cnt += 1
        else:
            sum2_x += float(row[0])
            sum2_y += float(row[1])
            class2_cnt += 1

# calculate mean
mean_train[0, 0] = sum1_x / class1_cnt
mean_train[0, 1] = sum1_y / class1_cnt
mean_train[1, 0] = sum2_x / class2_cnt
mean_train[1, 1] = sum2_y / class2_cnt

# count the error points of train data
for i in range(0, train_row_cnt):
    d1_train = np.sqrt(np.sum(np.square(data_train[i] - mean_train[0])))
    d2_train = np.sqrt(np.sum(np.square(data_train[i] - mean_train[1])))

    if d1_train < d2_train and label_train[i] == 2:
        error_train += 1
    elif d2_train < d1_train and label_train[i] == 1:
        error_train += 1


# load in test data
with open('dataset/synthetic1_test.csv') as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        data_test[test_row_cnt, 0] = float(row[0])
        data_test[test_row_cnt, 1] = float(row[1])
        label_test[test_row_cnt] = int(row[2])
        test_row_cnt += 1


# count the error points of test data
for j in range(0, test_row_cnt):
    d1_test = np.sqrt(np.sum(np.square(data_test[j] - mean_train[0])))
    d2_test = np.sqrt(np.sum(np.square(data_test[j] - mean_train[1])))

    if d1_test < d2_test and label_test[j] == 2:
        error_test += 1
    elif d2_test < d1_test and label_test[j] == 1:
        error_test += 1

print("train set error rate:", error_train / train_row_cnt)
print("test set error rate:", error_test / test_row_cnt)
plotDecBoundaries(data_train, label_train, mean_train)
plotDecBoundaries(data_test, label_test, mean_train)

