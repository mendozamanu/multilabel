#Script para clasificar los datasets grandes, particionados mediante random 10-fold

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

def timeStamp(fn, fmt='{fname}_%Y-%m-%d-%H-%M-%S.rclassif'):
    return datetime.datetime.now().strftime(fmt).format(fname=fn)


nfolds=10
fold_accuracy = []
fold_hamming = []
fold_prec = []
fold_auc = []
for s in dataset:
    fp = open(timeStamp('./datasets/'+s+'/'+s), 'w')
    for i in range(0, nfolds):
        
        X_train,y_train=readDataFromFile('./datasets/'+s+'/'+s+str(i)+'.rtrain')
        print('Reading: ./datasets/'+s+'/'+s+str(i)+'.rtrain')
        X_test,y_test=readDataFromFile('./datasets/'+s+'/'+s+str(i)+'.rtest')
        print('Reading: ./datasets/'+s+'/'+s+str(i)+'.rtest')
        classif = BinaryRelevance(classifier=RandomForestClassifier(n_estimators=10),require_dense=[False,True])
        classif.fit(X_train,y_train)
        y_score = classif.predict(X_test)

        #y_prob = classif.predict_proba(X_test)

        #-----------------------------------------#
        #Medidas: sklearn.metrics...(true,predict,..)
        acc= sklearn.metrics.accuracy_score(y_test, y_score)
        fold_accuracy.append(acc)
        #-----------------------------------------#
        hl=sklearn.metrics.hamming_loss(y_test, y_score)
        fold_hamming.append(hl)
        #Mean average precision
        m=sklearn.metrics.average_precision_score(y_test, y_score.toarray(), average='macro', pos_label=1, sample_weight=None)
        fold_prec.append(m)
        #-----------------------------------------#
        #Micro-average AUC
        rmi=sklearn.metrics.roc_auc_score(y_test, y_score.toarray(), average='micro', sample_weight=None, max_fpr=None)
        fold_auc.append(rmi)

fp.write("Avg accuracy: ")
fp.write(str(sum(fold_accuracy)/len(fold_accuracy))+'\n')
fp.write("Hamming loss: ")
fp.write(str(sum(fold_hamming)/len(fold_hamming))+'\n')
fp.write("Mean average precision: ")
fp.write(str(sum(fold_prec)/len(fold_prec))+'\n')
fp.write("Micro-average AUC: ")
fp.write(str(sum(fold_auc)/len(fold_auc))+'\n')


fp.close()
    
#print("Problems while trying to calculate coverage & ranking loss due to probab measure")
