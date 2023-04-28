import csv
import numpy as np
import matplotlib.pyplot as plt

training_set_labels = "./FinalProjectFiles/handwriting_training_set_labels.txt"
training_set = "./FinalProjectFiles/handwriting_training_set.txt"  # 4000x400
test_set_labels = "./FinalProjectFiles/handwriting_test_set_labels.txt"
test_set = "./FinalProjectFiles/handwriting_test_set.txt"  # 1000x400

# Each example is 20pixel by 20pixel
# Currently need to replace the value 10 in labels as 0

# Every 400 rows in training set => one digits e.g. 0-399 => 0

# Make classifier for handwriting
# Use training set and compute SVD fo each class/digit matrix
# we want to look at U.T if using A, V if we are looking at A.T (A transpose)
# Use SVD to do classification on 5, 10, 15, 20 vectors as a basis

# this function is to read both the set and label files and create arrays for them


def read_set_and_labels(file_set, file_label):
    set_list = []
    with open(file_set) as f:
        read_file = csv.reader(f, delimiter=',')
        for line in read_file:
            set_list.append(line)
    data_set = np.array(set_list, dtype=float)  # set datatype to float

    label_list = []
    with open(file_label) as f:
        read_file = csv.reader(f, delimiter='\n')
        for line in read_file:
            label_list.append(line)
    data_labels = np.array(label_list, dtype=int)  # set datatype to int
    # replace all 10s with 0s
    data_labels = np.where(data_labels == 10, 0, data_labels)

    return data_set, data_labels

# this function is to use the labels to see which rows contains which labels


def create_index(file_label):
    index = []
    for i in range(10):  # range from 0-9 --> our values for the labels
        # without [0], returns tuple of numpy arrays
        index.append(np.where(file_label == i)[0])
        # [0] is the row, [1] is all 0s since the 0 column since label array is just a vector
    return index  # returns a list of arrays that contain the row index for each number from 0-9


def svd(data_set, data_index):
    SVD = []
    for i in range(10):  # to iterate from 0-9
        row = data_index[i]
        A = data_set[row].T  # need to transpose the data
        # full_matrices=False since we are working with 2D arrays    U, S, V_T = np.linalg.svd(A, full_matrices=False)
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        SVD.append([U, S, Vt])  # returning svd
        # to check
        # A_x = U @ np.diag(S) @ Vt #can use @ as a shorcut for np.matmul
    return SVD

# this function is to check if the dataset correctly outputs image


def show_data(data_set):
    # each row is (400,), therefore we need to shape it to a 20x20 matrix to show image
    # since each data point is a 20x20 matrix #.reshape breaks it
    data_set.shape = (20, 20)
    show = data_set.T  # transpose the image since the original is rotated sideways, can transpose bc its a numpy array
    # show the image in black and white instead of a bunch of colors
    plt.imshow(show, cmap='gray')


train_data_set, train_data_labels = read_set_and_labels(
    training_set, training_set_labels)
train_index = create_index(train_data_labels)


svd_train = svd(train_data_set, train_index)

test_data_set, test_data_labels = read_set_and_labels(
    test_set, test_set_labels)
test_index = create_index(test_data_labels)


#global variables
accuracies = []  # holds the accuracy for the 5, 10, 15, 20 basis result
classification_function_call_counter = 0
digits_to_skip = []  # holds the values of test_data_labels to skip

def classification(test_data_labels):
    # classify the test set using the first k basis vectors for k in [5, 10, 15, 20]
    
    for k in [5, 10, 15, 20]:  # 5-20 singular vector basis
        correct = 0  # keep track on what is classified correctly?
        correctness_counter_per_digit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(test_data_labels)):
            if test_data_labels[i] in digits_to_skip:   #if label value matches the digits that we don't need to test end current iteration
                continue 
            # transpose into a column vector
            test_set_vector = test_data_set[i].T
            max_similarity = 0  # keep track of similarity
            prediction = None  # keep track of the predicted digit label
            for j in range(10):  # looping over digits 0-9
                # tuple containing SVD of the training set for digit j
                svd_matrices = svd_train[j]
                # creating U matrix so it can be multiplied by test_set_vector
                u_matrix = svd_matrices[0]
                # creating the projection
                projection = u_matrix[:, :k].T @ test_set_vector
                # using euclidean distance as similarity measure
                similarity = np.linalg.norm(projection)
                if similarity > max_similarity:  # check to see the highest similarity score
                    max_similarity = similarity
                    prediction = j
            if prediction == test_data_labels[i]:
                correct += 1
                if prediction == 0:  # increment index 0 if prediction is a zero digit
                    correctness_counter_per_digit[0] += 1
                else:
                    # the prediction value matches the index of the array
                    correctness_counter_per_digit[prediction] += 1
                    
        # computed accuracy score for classification
        accuracy = correct / len(test_data_labels)
        global accuracies
        accuracies.append(accuracy)  # update the accuracy list
        
        # printing value of k and accuracy formatting with 2f
        print(f"Accuracy using first {k} basis vectors: {accuracy:.2f}")
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        correct_percentage_per_digit = []
        for value in correctness_counter_per_digit:
            correct_percentage_per_digit.append(value/100)
        plt.title(f"Accuracy for each digit using {k} singular basis vector")
        plt.bar(digits, correct_percentage_per_digit)
        plt.xlabel("Digits")
        plt.ylabel("Accuracy")
        plt.show()

    # plot the accuracy as a function of the number of basis vectors used
    plt.plot([5,10,15,20], accuracies)
    plt.title("Classification accuracy as a function of the number of basis vectors used")
    plt.xlabel("Number of basis vectors")
    plt.ylabel("Classification accuracy")
    plt.show()        

"""
two_stage_classification function will attempt to predict the digit from the 
test data using the first singular basis vector in U from SVD.
A bar graph will be generated once the result has compiled.
classification function is then called if there are if the number of
digits to skip is less then 10 (if there are 10 digits to skip this means
all 10 digit has sucessful prediction rate above 50%).
"""
def two_stage_classification(test_data_labels):
    correct_counter = 0
    # value at each indices holds the number of times that digit was predicted correctly
    correctness_counter_per_digit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    global digits_to_skip #holds which digit to skip in test_set_labels

    # iterate through the test_labels
    for i in range(len(test_data_labels)):
        # digit_counter[int(test_data_labels[i])]+=1 #the index of digit_counter matches the value at test_data_labels[i]
        # take the a single row in the handwristing_test_set and transpose to a column vector
        test_set_vector = test_data_set[i].T
        # only checking against test against the matrices that is a renders a zero picture
        svd_matrices = svd_train[0]
        u_matrix = svd_matrices[0]

        # calculuate the z for digit 0
        projection_of_test_vector_onto_u_column_vector = u_matrix[0] * \
            u_matrix[0].T * test_set_vector
        z = np.linalg.norm(test_set_vector -
                           projection_of_test_vector_onto_u_column_vector)
        smallest_z = z
        prediction = 0

        # test against 1 on and onwards (with same test_set_vector)
        for j in range(1, 10):
            # checking against matrices that render digits 1 and onwards
            svd_matrices = svd_train[j]
            u_matrix = svd_matrices[0]
            # proj A onto B = B * B^t * A
            projection_of_test_vector_onto_u_column_vector = u_matrix[0] * \
                u_matrix[0].T * test_set_vector
            # calculates the magnitude of z
            z = np.linalg.norm(test_set_vector -
                               projection_of_test_vector_onto_u_column_vector)
            if (z < smallest_z):
                smallest_z = z
                prediction = j
        if prediction == test_data_labels[i]:
            correct_counter += 1
            if prediction == 0:  # increment index 0 if prediction is a zero digit
                correctness_counter_per_digit[0] += 1
            else:
                # the prediction value matches the index of the array
                correctness_counter_per_digit[prediction] += 1


    # plot result of prediction with 1 basis vector
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    correct_percentage_per_digit = []
    for i in range(len(correctness_counter_per_digit)):
        correct_percentage_per_digit.append(correctness_counter_per_digit[i]/100)
        if correctness_counter_per_digit[i]/100 >= .50:    #determine which digit to skip
            digits_to_skip.append(i)      
    plt.title("Accuracy for each digit using one singular basis vector")
    plt.bar(digits, correct_percentage_per_digit)
    plt.xlabel("Digits")
    plt.ylabel("Accuracy")
    plt.show()
    
    print("Accuracy using the first basis vector is: ", correct_counter/1000)
    
    """
    classification function is called and will skip any test data that has 
    labels that has a high prediction success rate since the result has been
    calculated using the first basis vector
    """
    if len(digits_to_skip) != 10:        
        classification(test_data_labels)

two_stage_classification(test_data_labels)
# show_data(test_data_set[547])
