import numpy as np
import numpy.linalg as lin

from efficient_cancer_data import read_training_data

# (a) Use the QR algorithm to find the least-squares linear model for the data.

# (b)Apply the linear model from (a) to the data set validate.data and predict the
# malignancy of the tissues. You have to define a classifier function.
# Classifier will check to see if the is current matrix value is a non-negative or not.
# If the current matrix value is non-negative, set it to 1, else set it to -1.
# 1 indicates that the specimen is malignant, -1 means the specimen is benign.

# (c) What is the percentage of samples that are incorrectly classified?
# Is it greater or smaller than the success rate of the training data?


# Solving for Least Squres:
# We want to solve for x in: Ax = b
# Recall that A = QR
# So:
# transpose(A) * A * x = tranpose(A) * b
# transpose(A) = transpose(Q * R) = transpose(R) * transpose(Q)
# Since we know what transpose(A) and A are equal to, we can substitute then out
# transpose(A) * A = transpose(R) * transpose(Q) * Q * R
# transpose(Q) * Q = I  -> I is the identity matrix
# transpose(R) * transpose(Q) * Q * R = trnapose(R) * R
# So now: transpose(R) * R * x = transpose(R) * transpose(Q) * b
# inverse(transpose(R)) * transpose(R) * R * x = inverse(transpose(R)) * transpose(R) * transpose(Q) * b
# R * x = transpose(Q) * b
# x = inverse(R) * tranpose(Q) * b
def least_squares(A, b):
    Q, R = lin.qr(A)
    x_hat = np.matmul(np.matmul(lin.inv(R), np.transpose(Q)), b)
    return x_hat

# classifier checks each variable in the passed in array to see if it is
# non-negative. If it is non-negative, it sets a new array and the same current
# position to be 1, otherwise it sets it to -1. It then returns the new array
def classifier(b):
    X, Y = np.shape(b)
    c = b
    
    for i in range(X):
        for j in range(Y):
            if b[i][j] >= 0:
                c[i][j] = 1
            else:
                c[i][j] = -1

    return c

# compare compares the values in the arrays b and c, adding 1 to Percentage
# each time their values are the same at the same position. 
# Compare returns Percentage divided by the total positions in b and c
# multiplied by 100
def compare(b, c):
    X, Y = np.shape(b)
    Percentage = 0
    
    for i in range(X):
        for j in range(Y):
            if b[i][j] == c[i][j]:
                Percentage += 1
    
    return float((Percentage / (X*Y)) * 100)


A_train, b_train = read_training_data('./train.data')
A_valid, b_valid = read_training_data('./validate.data')

A_train = np.array(A_train).astype(np.float64)
b_train = np.array(b_train).astype(np.float64)

A_valid = np.array(A_valid).astype(np.float64)
b_valid = np.array(b_valid).astype(np.float64)

x_train_hat = least_squares(A_train, b_train)
x_valid_hat = least_squares(A_valid, b_valid)

b_train_hat = np.matmul(A_train, x_train_hat)
b_valid_hat = np.matmul(A_valid, x_valid_hat)

c_train = classifier(b_train_hat)
c_valid = classifier(b_valid_hat)

percentage_train = compare(b_train, c_train)
percentage_valid = compare(b_valid, c_valid)

print("train.data incorrect classification percentage: ", 100 - percentage_train, "\n")
print("validate.data incorrect classification percentage: ", 100 - percentage_valid, "\n")

if percentage_train > percentage_valid:
    print("The success rate of guesses in regards to train.data were greater than that of validate.data")
elif percentage_train < percentage_valid:
    print("The success rate of guesses in regards to validate.data were greater than that of train.data")
else:
    print("validate.data and train.data had the same success rate")