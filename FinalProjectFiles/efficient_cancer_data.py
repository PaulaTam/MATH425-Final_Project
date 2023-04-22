# Copyright 2013 Philip N. Klein
#from vec import Vec
#from vecutil import vec2list #vecutil package does not exist, was unable to download
#from vec import Vector2 #From the current vec 0.5 documentation, the only class available is Vector2
#import pandas as pd
import numpy as np
from sympy import Matrix

filename = "./train.data"

def read_training_data(fname, D=None):
    """Given a file in appropriate format, and given a set D of features,
    returns the pair (A, b) consisting of
    a P-by-D matrix A and a P-vector b,
    where P is a set of patient identification integers (IDs).

    For each patient ID p,
      - row p of A is the D-vector describing patient p's tissue sample,
      - entry p of b is +1 if patient p's tissue is malignant, and -1 if it is benign.

    The set D of features must be a subset of the features in the data (see text).
    """
    file = open(fname)
    params = ["radius", "texture", "perimeter","area","smoothness","compactness","concavity","concave points","symmetry","fractal dimension"];
    stats = ["(mean)", "(stderr)", "(worst)"]
    feature_labels = set([y+x for x in stats for y in params])
    feature_map = {params[i]+stats[j]:j*len(params)+i for i in range(len(params)) for j in range(len(stats))}
    if D is None: D = feature_labels
    feature_vectors = {}
    #patient_diagnoses = {}
    A = []
    b = []
    for line in file:
        #newline = line.strip()
        row = line.split(",")
        #print({f:float(row[feature_map[f]+2]) for f in D})
        patient_ID = int(row[0])
        b.append(-1) if row[1] == 'B' else b.append(1)
        feature_vectors[patient_ID] = np.array(D, {f:float(row[feature_map[f]+2]) for f in D})
        #changed Vec to numpy.array
        A.append(feature_vectors[patient_ID].tolist())
        #removed vec2list and used .tolist() function
    return Matrix(A), Matrix(b)

read_training_data(filename)