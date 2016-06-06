# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:11:22 2016

Applying KNN algorithm in dating dataset 
@author: ceciliaLee
"""
from numpy import *
import operator
from os import listdir
 
              
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


#################  Example: dating data  ################# 
## transform the records in .txt 
## transform the input format, read data from files  
## using datingdataset2.txt to test the algorithm 

## Read records from files and tranform them into matrix data
def file2matrix(filename,dim2):
    fr = open(filename)
    arrayOfLines=fr.readlines()
    numberOfLines = len(arrayOfLines)         #get the number of lines in the file
    returnMat = zeros((numberOfLines,dim2))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in arrayOfLines:
        line = line.strip()  ## delete Enter
        listFromLine = line.split('\t') ## separate with tab
        returnMat[index,:] = listFromLine[0:dim2]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
    
## using matplotlib to draw scatter graph 
## test order: KNNClasify.drawScatter(datingDataMat)
import matplotlib
import matplotlib.pyplot as plt
def drawScatter(datingDataMat,datingLabels):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

## Normalization 
def autoNorm(dataSet):
    minValue = dataSet.min(0)  
    maxValue = dataSet.max(0)  
    ranges = maxValue - minValue  # the difference between the maximum and minimum
    
    m = dataSet.shape[0]  # get the number of row
    minValTiled=tile(minValue,(m,1))
    maxValTiled=tile(maxValue,(m,1))
    
    
    normDataSet = zeros(shape(dataSet))  # the return matrix
    normDataSet=(dataSet-minValTiled)/(maxValTiled-minValTiled)
    return normDataSet, ranges, minValue

## test codes for dating class   
def datingClassTest():
    rate=0.1
    datingDataMat,datingLabels=file2matrix('/Users/ceciliaLee/Desktop/datingTestSet2.txt',3) ## read data
    normMat, ranges,minValue = autoNorm(datingDataMat) ## normalize data
    testVector_count=int((normMat.shape[0])*rate) ## count the number of test vectors;determine which part of data are used as training samples and test samples respectively 
    error_count=0.0 ## number of samples that classified incorrectly
    for i in range(testVector_count):
        classifiedResult = KNNClasifier(normMat[i,:],normMat[testVector_count:normMat.shape[0],:],datingLabels[testVector_count:normMat.shape[0]],3)
        print 'result from KNN Classifier is:  %d, the actual class is : %d' % (classifiedResult,datingLabels[i])
        if (classifiedResult != datingLabels[i]):
            error_count+=1 ## if predict wrongly, number of error sample increase 1
    print 'the total error rate is: %f' % (error_count/(testVector_count*1.0)) ## calculate the error rate
    
## Applying in real systems. unlabeled sample input by users 
def classifyPeople():
    results=['not at all','in small does','in large does']
    percentTimeGame=float(raw_input('percentage of time spent playing video games?'))
    freFlyMilesEarn=float(raw_input('frequent flier miles earned per year?'))
    iceConsumed=float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat,datingLabels=file2matrix('/Users/ceciliaLee/Desktop/datingTestSet2.txt',3)
    normMat, ranges,minValue=autoNorm(datingDataMat)
    inputArray=array([percentTimeGame,freFlyMilesEarn,iceConsumed])
    classifiedResult=KNNClasifier((inputArray-minValue)/ranges,normMat,datingLabels,3)
    print 'You will probably like this person: ', results[classifiedResult-1]
            
    

    
    
    
    
               
              