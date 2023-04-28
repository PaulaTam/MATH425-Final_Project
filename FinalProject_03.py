import csv
import numpy as np
import matplotlib.pyplot as plt

training_set_labels="./FinalProjectFiles/handwriting_training_set_labels.txt"
training_set="./FinalProjectFiles/handwriting_training_set.txt" #4000x400
test_set_labels="./FinalProjectFiles/handwriting_test_set_labels.txt" 
test_set="./FinalProjectFiles/handwriting_test_set.txt" #1000x400

#Each example is 20pixel by 20pixel
#Currently need to replace the value 10 in labels as 0

#Every 400 rows in training set => one digits e.g. 0-399 => 0

#Make classifier for handwriting
#Use training set and compute SVD fo each class/digit matrix
#we want to look at U.T if using A, V if we are looking at A.T (A transpose)
#Use SVD to do classification on 5, 10, 15, 20 vectors as a basis

#this function is to read both the set and label files and create arrays for them
def read_set_and_labels(file_set, file_label):
    set_list = []
    with open(file_set) as f:
        read_file = csv.reader(f, delimiter=',')
        for line in read_file:
            set_list.append(line)
    data_set = np.array(set_list, dtype= float) #set datatype to float
    
    label_list = []
    with open(file_label) as f:
        read_file = csv.reader(f, delimiter='\n')
        for line in read_file:
            label_list.append(line)
    data_labels = np.array(label_list, dtype=int) #set datatype to int
    data_labels = np.where(data_labels==10, 0, data_labels) #replace all 10s with 0s
    
    return data_set, data_labels
   
#this function is to use the labels to see which rows contains which labels
def create_index(file_label):
    index = []
    for i in range(10): #range from 0-9 --> our values for the labels
        index.append(np.where(file_label==i)[0]) #without [0], returns tuple of numpy arrays
        #[0] is the row, [1] is all 0s since the 0 column since label array is just a vector
    return index #returns a list of arrays that contain the row index for each number from 0-9

def svd(data_set, data_index):
    SVD = []
    for i in range(10): #to iterate from 0-9
        row = data_index[i] 
        A = data_set[row].T #need to transpose the data
        U, S, Vt = np.linalg.svd(A, full_matrices=False) #full_matrices=False since we are working with 2D arrays    U, S, V_T = np.linalg.svd(A, full_matrices=False)
        SVD.append([U, S, Vt]) #returning svd
        #to check
        #A_x = U @ np.diag(S) @ Vt #can use @ as a shorcut for np.matmul
    return SVD
  
#this function is to check if the dataset correctly outputs image
def show_data(data_set):
    #each row is (400,), therefore we need to shape it to a 20x20 matrix to show image
    data_set.shape = (20, 20) #since each data point is a 20x20 matrix #.reshape breaks it
    show = data_set.T #transpose the image since the original is rotated sideways, can transpose bc its a numpy array
    plt.imshow(show, cmap='gray') #show the image in black and white instead of a bunch of colors
 

train_data_set, train_data_labels = read_set_and_labels(training_set, training_set_labels)
train_index = create_index(train_data_labels)

test_data_set, test_data_labels = read_set_and_labels(test_set, test_set_labels)
test_index = create_index(test_data_labels)

svd_train = svd(train_data_set, train_index)

def classification(accuracies, test_data_set, test_data_labels):
    # classify the test set using the first k basis vectors for k in [5, 10, 15, 20]
    for k in [5, 10, 15, 20]: #5-20 singular vector basis 
        correct = 0 #keep track on what is classified correctly?
        for i in range(len(test_data_labels)):
            test_digit = test_data_set[i].T #transpose into a column vector 
            max_similarity = 0 #keep track of similarity
            prediction = None #keep track of the predicted digit label 
            for j in range(10): #looping over digits 1-9
                svd_matrices = svd_train[j] #tuple containing SVD of the training set for digit j 
                u_matrix = svd_matrices[0] #creating U matrix so it can be multiplied by test_digit
                projection = u_matrix[:, :k].T @ test_digit #creating the projection
                similarity = np.linalg.norm(projection) # using euclidean distance as similarity measure
                if similarity > max_similarity: #check to see the highest similarity score 
                    max_similarity = similarity
                    prediction = j
            if prediction == test_data_labels[i]:
                #print(prediction, test_data_labels[i])
                #print("true ", i, ",", prediction, ",", test_data_labels[i])
                correct += 1
           #elif prediction != test_data_labels[i]:
                #print("false ", i, ",", prediction, ",", test_data_labels[i])
                
        accuracy = correct / len(test_data_labels) #computed accuracy score for classification
        accuracies.append(accuracy) #update the accuracy list 
        print(f"Accuracy using first {k} basis vectors: {accuracy:.2f}") #printing value of k and accuracy formatting with 2f

accuracies = [] # initialize the accuracy list
classification(accuracies, test_data_set, test_data_labels) # call the classification function with the accuracy list as an argument

def two_stage_classification(accuracies, test_data_labels):
    correct_counter = 0
    
    #iterate through the test_labels
    for i in range(len(test_data_labels)):
        test_set_vector = test_data_set[i].T #take the a single row in the tandwristing_test_set and transpose to a column vector
        svd_matrices = svd_train[0] #only checking against test against the matrices that is a renders a zero picture
        u_matrix = svd_matrices[0]
        
        #calculuate the z for digit 0
        projection_of_test_vector_onto_u_column_vector = u_matrix[0] * u_matrix[0].T * test_set_vector
        z = np.linalg.norm(test_set_vector - projection_of_test_vector_onto_u_column_vector)
        smallest_z = z
        prediction = 0

        #test against 1 on and onwards (with same test_set_vector)
        for j in range(1,10):
            svd_matrices = svd_train[j] #checking against matrices that render digits 1 and onwards
            u_matrix = svd_matrices[0]
            projection_of_test_vector_onto_u_column_vector = u_matrix[0] * u_matrix[0].T * test_set_vector #proj A onto B = B * B^t * A
            z = np.linalg.norm(test_set_vector - projection_of_test_vector_onto_u_column_vector) #calculates the magnitude of z
            if (z < smallest_z):
                smallest_z = z
                prediction = j
        if prediction == test_data_labels[i]:
                correct_counter+=1
        print("the prediction was: ", prediction, "the actual digit: ", test_data_labels[i])
        
    print("the correct percentage of using 1 singular basis is: ", correct_counter/len(test_data_labels)*100)
    
# plot the accuracy as a function of the number of basis vectors used
plt.plot([5, 10, 15, 20], accuracies, '-o')
plt.title("Classification accuracy as a function of the number of basis vectors used")
plt.xlabel("Number of basis vectors")
plt.ylabel("Classification accuracy")
plt.show()

def check_singular_values(svd):
    sv_lengths = []
    for usvt in svd: #for every svd (each digit has their own svd)
        s = list(usvt[1]) #get the singular values. it is represented as a vector
        cut = set(s) #since sigma values are unique, we can remove duplicate values using set()
        sv_lengths.append(len(cut)) #length of the set
    return sv_lengths
        
sv = check_singular_values(svd_train)
