#Script para clasificar los datasets grandes, particionados mediante random 10-fold

import numpy as np
import scipy.sparse as sp
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.adapt import MLkNN

from sklearn.neighbors import KNeighborsClassifier

import sklearn.metrics
import datetime
from sklearn.metrics import classification_report,confusion_matrix


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
   'Delicious',
   'bookmarks',
   'mediamill',
   'tmc2007',
   'bibtex',
   'Corel5k',
   'emotions',
   'Enron',
   'genbase',
   'medical',
   'scene',
   'yeast'
}
classifier = {
    BinaryRelevance(classifier=KNeighborsClassifier(n_neighbors=5),require_dense=[False,True]),
    LabelPowerset(classifier=KNeighborsClassifier(n_neighbors=5),require_dense=[False,True]),
    ClassifierChain(classifier=KNeighborsClassifier(n_neighbors=5),require_dense=[False,True]),
    MLkNN(k=5)
}

def timeStamp(fn, fmt='{fname}_%Y-%m-%d-%H-%M-%S.lclassif'):
    return datetime.datetime.now().strftime(fmt).format(fname=fn)


nfolds=10
fold_accuracy = []
fold_hamming = []
fold_prec = []
fold_auc = []
fold_aucm = []
#fold_cover = []
#fold_rank = []
for s in dataset:
    print('Reading: ./datasets/'+s+'/'+s+'.ltrain')
    print('Reading: ./datasets/'+s+'/'+s+'.ltest')     
    for cl in classifier:
        fp = open(timeStamp('./datasets/'+s+'/'+s+str(cl).split('(')[0]), 'w')
        for i in range(0, nfolds):
            
            X_train,y_train=readDataFromFile('./datasets/'+s+'/'+s+str(i)+'.ltrain')
            X_test,y_test=readDataFromFile('./datasets/'+s+'/'+s+str(i)+'.ltest')
            classif = cl
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
            
            #-----------------------------------------#
            #Coverage\n",
            #c=sklearn.metrics.coverage_error(y_test, y_prob.toarray(), sample_weight=None)
            #fold_cover.append(c)
            #-----------------------------------------#
            #Ranking loss\n",
            #rl=sklearn.metrics.label_ranking_loss(y_test, y_prob.toarray(), sample_weight=None)
            #fold_rank.append(rl)
            
            #Mean average precision
            m=sklearn.metrics.average_precision_score(y_test, y_score.toarray(), average='macro', pos_label=1, sample_weight=None)
            fold_prec.append(m)
            #-----------------------------------------#
            #Micro-average AUC
            rmi=sklearn.metrics.roc_auc_score(y_test, y_score.toarray(), average='micro', sample_weight=None, max_fpr=None)
            fold_auc.append(rmi)
            #rma=sklearn.metrics.roc_auc_score(y_test, y_score.toarray(), average='macro', sample_weight=None, max_fpr=None)
            #fold_aucm.append(rma)

        fp.write("Avg accuracy: ")
        fp.write(str(sum(fold_accuracy)/len(fold_accuracy))+'\n')
        fp.write("Hamming loss: ")
        fp.write(str(sum(fold_hamming)/len(fold_hamming))+'\n')
        fp.write("Mean average precision: ")
        fp.write(str(sum(fold_prec)/len(fold_prec))+'\n')
        fp.write("Micro-average AUC: ")
        fp.write(str(sum(fold_auc)/len(fold_auc))+'\n')
        #fp.write("Macro-average AUC: ")
        #fp.write(str(sum(fold_aucm)/len(fold_aucm))+'\n')
        
        fold_accuracy = []
        fold_hamming = []
        fold_prec = []
        fold_auc = []
        fold_aucm = []
        
        fp.write(classification_report(y_test,y_score))
        fp.close()

    
#print("Problems while trying to calculate coverage & ranking loss due to probab measure")
