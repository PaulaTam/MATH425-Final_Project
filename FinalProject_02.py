import numpy as np
import numpy.linalg as lin

from FinalProjectFiles.efficient_cancer_data import read_training_data

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
    x = np.matmul(np.matmul(lin.inv(R), np.transpose(Q)), b)

