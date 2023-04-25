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

#Make classifier for handwriting
#Use training set and compute SVD fo each class/digit matrix
#Use SVD to do classification on 5, 10, 15, 20 vectors as a basis

def read_set_and_labels(file_set, file_label):
    set_list = []
    with open(file_set) as f:
        read_file = csv.reader(f, delimiter=',')
        for line in read_file:
            set_list.append(line)
    data_set = np.array(set_list, dtype= float)
    
    label_list = []
    with open(file_label) as f:
        read_file = csv.reader(f, delimiter='\n')
        for line in read_file:
            label_list.append(line)
    data_labels = np.array(label_list, dtype=int)
    data_labels = np.where(data_labels==10, 0, data_labels) #replace all 10s with 0s
    
    return data_set, data_labels

#this function is to check if the dataset correctly outputs image
def show_data(data_set):
    data_set.shape = (20, 20) #since each data point it a 20x20 matrix
    show = data_set.T #transpose the image since the original is rotated sideways
    plt.imshow(show, cmap='gray') #show the image in black and white

data_set, data_labels = read_set_and_labels(training_set, training_set_labels)
show_data(data_set[3333]) #shows image of 8
print(data_labels[3333]) #prints 8