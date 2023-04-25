import csv
import numpy as np
import sympy as sym
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
        index.append(np.where(file_label==i)[0])
    return index #returns a list of arrays that contain the row index for each number from 0-9
  
    
#this function is to check if the dataset correctly outputs image
def show_data(data_set):
    #each row is (400,), therefore we need to shape it to a 20x20 matrix to show image
    data_set.shape = (20, 20) #since each data point it a 20x20 matrix
    show = data_set.T #transpose the image since the original is rotated sideways, can transpose bc its a numpy array
    plt.imshow(show, cmap='gray') #show the image in black and white instead of a bunch of colors
    
train_data_set, train_data_labels = read_set_and_labels(training_set, training_set_labels)
train_index = create_index(train_data_labels)
#show_data(train_data_set[3333]) #shows image of 8
#print(train_data_labels[3333]) #prints 8
#print(train_index)

test_data_set, test_data_labels = read_set_and_labels(test_set, test_set_labels)
test_index = create_index(test_data_labels)
#print(test_index)