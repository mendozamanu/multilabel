#Script para clasificar los datasets grandes, particionados mediante random 10-fold

import numpy as np
import sys
import scipy.sparse as sp
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.adapt import MLkNN
import functools
import sklearn.metrics.base

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

def average_precision_score(y_true, y_score, average="macro", pos_label=1,
                            sample_weight=None):
    def _binary_uninterpolated_average_precision(
            y_true, y_score, pos_label=1, sample_weight=None):
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)

        recall[np.isnan(recall)] = 0

        return -np.sum(np.diff(recall) * np.array(precision)[:-1])

    
    average_precision = functools.partial(_binary_uninterpolated_average_precision,
                                pos_label=pos_label)
    
    return sklearn.metrics.base._average_binary_score(average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)


if len(sys.argv) <= 1:
    print "Correct use: multilabelKfold.py input-file "
    sys.exit()

s = str(sys.argv[1])

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
fold_precm = []

fold_auc = []
fold_cover = []
fold_rank = []
skip=0

print('Reading: ./datasets/'+s+'/'+s+'.ltrain')
print('Reading: ./datasets/'+s+'/'+s+'.ltest')     
for cl in classifier:
    print ('Classif: ' + str(cl).split('(')[0])
    fp = open(timeStamp('./datasets/'+s+'/'+s+str(cl).split('(')[0]), 'w')
    for i in range(0, nfolds):
        
        skip=0
        
        X_train,y_train=readDataFromFile('./datasets/'+s+'/'+s+str(i)+'.ltrain')
        X_test,y_test=readDataFromFile('./datasets/'+s+'/'+s+str(i)+'.ltest')
        classif = cl
        
        for j in range(0, y_train.shape[1]):
            if len(np.unique(y_train[:,j]))==1 : #Saltar fold si hay [0], luego div medidas por num real (nuevo contador)
                skip=1
        
        #print (y_train.min(), y_train.max())
        #Cuando en sklearn una entrada es y_score -> y_prob y si es y_test -> prediccion (y_score)
        sys.stdout.flush()

        classif.fit(X_train,y_train)
        y_score = classif.predict(X_test)

        if skip==0 :
            y_prob = classif.predict_proba(X_test.todense())
            #-----------------------------------------#
            #Coverage\n",
            c=sklearn.metrics.coverage_error(y_test, y_prob.toarray(), sample_weight=None)
            fold_cover.append(c)
            
            #-----------------------------------------#
            #Ranking loss\n",
            rl=sklearn.metrics.label_ranking_loss(y_test, y_prob.toarray(), sample_weight=None)
            fold_rank.append(rl)
            #-----------------------------------------#
            #Mean average precision
            m=average_precision_score(y_test, y_prob.toarray(), average='macro', pos_label=1, sample_weight=None)
            fold_prec.append(m)
            
            m2=average_precision_score(y_test, y_prob.toarray(), average='micro', pos_label=1, sample_weight=None)
            fold_precm.append(m2)
            
            #-----------------------------------------#
            #Micro-average AUC
            rmi=sklearn.metrics.roc_auc_score(y_test, y_prob.toarray(), average='micro', sample_weight=None, max_fpr=None)
            fold_auc.append(rmi)
            

        #-----------------------------------------#
        #Medidas: sklearn.metrics...(true,predict,..)
        acc= sklearn.metrics.accuracy_score(y_test, y_score)
        fold_accuracy.append(acc)
        #-----------------------------------------#
        hl=sklearn.metrics.hamming_loss(y_test, y_score)
        fold_hamming.append(hl)


    fp.write("Avg accuracy: ")
    fp.write(str(sum(fold_accuracy)/len(fold_accuracy))+'\n')
    fp.write("Hamming loss: ")
    fp.write(str(sum(fold_hamming)/len(fold_hamming))+'\n')
    
    fp.write("Coverage: ")
    if len(fold_cover)>0:
        fp.write(str(sum(fold_cover)/len(fold_cover))+'\n')
    
    fp.write("Ranking loss: ")
    if len(fold_rank)>0:
        fp.write(str(sum(fold_rank)/len(fold_rank))+'\n')
    
    fp.write("Mean average precision (macro, micro): ")
    if len(fold_prec)>0:
        fp.write(str(sum(fold_prec)/len(fold_prec))+', ')
        fp.write(str(sum(fold_precm)/len(fold_precm))+'\n')

    fp.write("Micro-average AUC: ")
    if len(fold_auc)>0:
        fp.write(str(sum(fold_auc)/len(fold_auc))+'\n')
    fold_accuracy = []
    fold_hamming = []
    fold_prec = []
    fold_auc = []
    fp.write(classification_report(y_test, y_score, digits=20))
    fp.close()