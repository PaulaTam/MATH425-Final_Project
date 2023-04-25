import numpy as np
import sympy as sym

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
        for line in f:
            set_list.append(line)