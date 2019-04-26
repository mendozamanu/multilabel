#Script para clasificar los datasets grandes, particionados mediante iterative train-test split 70-30

import numpy as np
import scipy.sparse as sp
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import datetime

def readDataFromFile (fileName):
    "This functions reads data from a file and store it in two matrices"
    #Open the file
    file = open(fileName, 'r')
 
    #Now we have to read the first line and check if it's sparse or dense
    firstLine = file.readline()
    words = firstLine.split()
    word = words[1]
    if word[:-1] == 'SPARSE':
        sparse = True #The file is in sparse mode
    else:
        sparse = False #The file is in dense mode
 
 
    secondLine = file.readline()
    words = secondLine.split()
    instances = int(words[1])
    thirdLine = file.readline()
    words = thirdLine.split()
    attributes = int(words[1])
    fourthLine = file.readline()
    words = fourthLine.split()
    labels = int(words[1])
    #Now we do a loop reading all the other lines
    #Then we read the file, different way depending if sparse or dense
 
    #The loop starts in the first line of data
    #We have to store that data in two matrices
    X = np.zeros((instances, attributes), dtype=float)
    y = np.zeros((instances, labels), dtype=int)
    numberLine = 0
    for line in file.readlines():
        putToX = True
        firstIndex = 1
        numberData = 0
        numberY = 0
        for data in line.split():
            if sparse:#Sparse format, we have to split each data
                if data == '[':
                    putToX = False
 
                if putToX == True and (data != '[' and data != ']'):
                    sparseArray = data.split(':')
                    lastIndex = int(sparseArray[0])
                    for i in range(firstIndex, lastIndex - 1):
                        X[numberLine, i-1] = float(0)
                    X[numberLine, lastIndex-1] = float(sparseArray[1])
                    firstIndex = lastIndex-1
                else:
                    if (data != '[') and (data != ']'):
                        aux = float(data)
                        y[numberLine, numberY] = int(aux)
                        numberY += 1
               
            else:#Dense format
                if data == '[':
                    putToX = False
 
                if putToX == True and (data != '[' and data != ']'):
                    X[numberLine, numberData] = float(data)
                else:
                    if (data != '[') and (data != ']'):
                        #This is good for the dense format
                        aux = float(data)
                        y[numberLine, numberY] = int(aux)
                        numberY += 1
            numberData += 1
       
        numberLine += 1
    X = sp.csr_matrix(X)
    file.close()
    return X, y

#Listado de datasets a ejecutar
dataset = {
   'delicious',
   'bookmarks',
   'mediamill',
   'tmc2007'
}

def timeStamp(fn, fmt='{fname}_%Y-%m-%d-%H-%M-%S.report'):
    return datetime.datetime.now().strftime(fmt).format(fname=fn)

for s in dataset:
    fp = open(timeStamp('./datasets/'+s+'/'+s), 'w')
    
    X_train,y_train=readDataFromFile('./datasets/'+s+'/'+s+'.strain')
    print('Reading: ./datasets/'+s+'/'+s+'.strain')
    X_test,y_test=readDataFromFile('./datasets/'+s+'/'+s+'.stest')
    print('Reading: ./datasets/'+s+'/'+s+'.stest')
    classif = BinaryRelevance(classifier=RandomForestClassifier(n_estimators=10),require_dense=[False,True])
    classif.fit(X_train,y_train)
    y_score = classif.predict(X_test)

    #y_prob = classif.predict_proba(X_test)

    #-----------------------------------------#
    #Medidas: sklearn.metrics...(true,predict,..)
    acc= sklearn.metrics.accuracy_score(y_test, y_score)
    fp.write("Accuracy: %0.5f\n"%acc)
    #-----------------------------------------#
    hl=sklearn.metrics.hamming_loss(y_test, y_score)
    fp.write("Hamming loss: %0.5f\n"%hl)
    #-----------------------------------------#
    #Coverage
    #c=sklearn.metrics.coverage_error(y_test, y_prob.toarray(), sample_weight=None)
    #print ("Coverage: %0.5f - 1"%c)
    #-----------------------------------------#
    #Ranking loss
    #rl=sklearn.metrics.label_ranking_loss(y_test, y_prob.toarray(), sample_weight=None)
    #print("Ranking loss: %0.5f"%rl)
    #-----------------------------------------#
    #F1 score
    #f1= sklearn.metrics.f1_score(y_test, y_score)
    #print ("Accuracy: %0.5f"%f1)
    #-----------------------------------------#
    #Mean average precision
    m=sklearn.metrics.average_precision_score(y_test, y_score.toarray(), average='macro', pos_label=1, sample_weight=None)
    fp.write("Mean average precision: %0.5f\n"%m)
    #-----------------------------------------#
    #Micro-average AUC
    rmi=sklearn.metrics.roc_auc_score(y_test, y_score.toarray(), average='micro', sample_weight=None, max_fpr=None)
    fp.write("ROC AUC micro: %0.5f\n"%rmi)
    fp.close()
    
#print("Problems while trying to calculate coverage & ranking loss due to probab measure")
