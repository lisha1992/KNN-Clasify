# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 00:29:46 2016

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

##### Applying KNN in hand-writting recognition system
## transform images (32*32) into vectors (1*1024)
def image2vector(fileName):
    vector=zeros((1,1024)) ## define the returned vector of image
    imgFile=open(fileName)
    ## loop for recording data intovector
    for i in range(32): ## read numbers from 1st line to 32th line
        lineString=imgFile.readline()
        for j in range(32):  ## read numbers from 1st column to 32th column in each line
            vector[0,32*i+j] = int(lineString[j])
    return vector    

## recognizing hand-writting numbers by KNN 
def handWritingReconTest():
    handWriLabels=[]  ## define the label list of hand-writing numbers
    ## Training set
    trainSampleList=listdir('/Users/ceciliaLee/Desktop/digits/trainingDigits') ## get content 获取目录内容
    trainSample_count=len(trainSampleList)
    trainMat=zeros((trainSample_count,1024))
    for i in range(trainSample_count):
        ## 由文件名解析数字，如 1_23.txt 的分类是1，是数字1的25个实例
        fileNameStr=trainSampleList[i]
        fileStr=fileNameStr.split('.')[0]
        classStr=int(fileStr.split('_')[0])
        handWriLabels.append(classStr)
        trainMat[i,:]=image2vector('/Users/ceciliaLee/Desktop/digits/trainingDigits/%s' % fileNameStr)
    ## Testing set    
    testSampleList=listdir('/Users/ceciliaLee/Desktop/digits/testDigits')
    lenTest=len(testSampleList)
    error_count=0.0
    for i in range(lenTest):
        fileNameStr=testSampleList[i]
        fileStr=fileNameStr.split('.')[0]
        classStr=int(fileStr.split('_')[0])
        vectorForTest=image2vector('/Users/ceciliaLee/Desktop/digits/testDigits/%s' % fileNameStr)
        classifiedResult=KNNClasifier(vectorForTest,trainMat,handWriLabels,3)
        print 'The classified result by KNN is: %d, the actual class is: %d' % (classifiedResult,classStr)
        if (classifiedResult != classStr):
            error_count+=1
    print '\nThe total number of incorrectly classified samples is: %d' % error_count
    print '\nThe error rate is: %f' % (error_count/(lenTest*0.1))