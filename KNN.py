# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:11:22 2016

@author: ceciliaLee
"""
from numpy import *
import operator
from os import listdir

              
# create a dataset which contains 4 samples with 2 classes
def createDataSet(): 
    # create a matrix: each row as a sample  
    group=array([[1.0, 0.8], [1.0, 1.0], [0.2, 0.3], [0.0, 0.1]]) 
    labels=['A', 'A', 'B', 'B'] # four samples and two classes 
    return group,labels

######################## Codes for KNN Clasifier  
# method: KNNClasifier(unLabel,dataSet,labels,k)
# Input:      unLabel:  feature vector of the sample you need to clasify
#             dataSet:  dataset (N*M samples) with known label (class)
#             labels:   data set labels (1xM vector)  
#             k:        number of neighbors to use for comparison   
              
# Output:     the label(class) of the unknown sample
              
def KNNClasifier(unLabel,dataSet,labels,k):
    numSamples=dataSet.shape[0]  # shape[0] stands for the number of rows , axis=1 get number of columns 

    ## Step1: calculate Euclidean distance  
    ## tile(A, reps): Construct an array by repeating A for reps times 
    diff=tile(unLabel, (numSamples, 1)) - dataSet 
    sqDiff=diff**2  # square of difference
    sqDist = sum(sqDiff, axis = 1) 
    distance=sqDist**0.5  ## sqrt
    
    ## Step 2: Sort the distances (ascending)  
    sortedDistIndex=argsort(distance) #returns the indices that would sort an array in a ascending order
    
    classCount={} ## Define a dictionary
    
    ## Step 3 : get class label; Choose the min k distance
    for i in range(k):
        voteILabel=labels[sortedDistIndex[i]]
    ## Step 4: Count the tiems labels occur
        classCount[voteILabel]=classCount.get(voteILabel,0)+1 # when the key voteLabel is not in dictionary classCount, get() will return 0 
        
    ## Step 5: The most voted class will return
    maxCount=0
    for key, value in classCount.items():
        if value>maxCount:
            maxCount=value
            maxIndex=key
    return maxIndex