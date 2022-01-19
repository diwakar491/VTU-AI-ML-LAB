# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:19:04 2022

@author: Diwakar
"""

import numpy as np
import math
import csv

def read_data(filename):

    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        metadata = next(datareader)
        traindata=[]
        for row in datareader:
            traindata.append(row)
    
    return (metadata, traindata)

def splitDataset(dataset, splitRatio):         #splits dataset to training set and test set based on split ratio
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	testset = list(dataset)
	i=0
	while len(trainSet) < trainSize:
	    trainSet.append(testset.pop(i))
	return [trainSet, testset]
       
def classify(data,test):
    
    total_size = data.shape[0]
    print("training data size=",total_size)
    print("test data sixe=",test.shape[0])
    target=np.unique(data[:,-1])
    count = np.zeros((target.shape[0]), dtype=np.int32)    
    prob = np.zeros((target.shape[0]), dtype=np.float32)
    print("target   count  probability")
    for y in range(target.shape[0]):
        for x in range(data.shape[0]):
            if data[x,data.shape[1]-1] == target[y]:
                count[y] += 1
        prob[y]=count[y]/total_size    # comptes the probability of target
        print(target[y],"\t",count[y],"\t",prob[y])
    
    prob0 = np.zeros((test.shape[1]-1), dtype=np.float32)
    prob1 = np.zeros((test.shape[1]-1), dtype=np.float32)
    accuracy=0
    print("Instance prediction taget")    
    for t in range(test.shape[0]):
        for k in range(test.shape[1]-1):  # for each attribute in column
            count1=count0=0
            for j in range(data.shape[0]):
                if test[t,k]== data[j,k] and data[j,data.shape[1]-1]== target[0]:
                    count0+=1
                elif test[t,k]== data[j,k] and data[j,data.shape[1]-1]== target[1]:
                    count1+=1
            prob0[k]= count0/count[0]        #Find no probability of each attribute
            prob1[k]= count1/count[1]       #Find yes probability of each attribute
            

        probno=prob[0]
        probyes=prob[1]
        for i in range(test.shape[1]-1):
             probno=probno*prob0[i]
             probyes=probyes*prob1[i]
        
        
        if probno>probyes:     # prediction
            predict='no'
        else:
            predict='yes'
        print(t+1,"\t",predict,"\t    ",test[t,test.shape[1]-1])

        
        if predict== test[t,test.shape[1]-1]:     # computing accuracy
           accuracy+=1
        
    final_accuracy=(accuracy/test.shape[0])*100
    print("accuracy",final_accuracy,"%")

    return  

metadata, traindata = read_data("tennis.csv")
splitRatio = 0.6
trainingset, testset = splitDataset(traindata, splitRatio)
training=np.array(trainingset)
testing=np.array(testset)
print("------------------Training Data-------------------")
print(trainingset)
print("-------------------Test Data-------------------")
print(testset)
classify(training,testing)

    