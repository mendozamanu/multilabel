#!/usr/bin/python2.7

import sys
import os
import scipy.sparse as sp

import numpy as np
import arff
#from sklearn.covariance import EmpiricalCovariance

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


def label_correlation(y, s):
    """Correlation between labels in a label matrix
    Parameters
    ----------
    y : array-like (n_labels, n_samples)
        Label matrix
    s : float
        Smoothness parameter
    Returns
    -------
    L : array-like (n_labels, n_labels)
        Label correlation matrix
    """
    L = np.zeros(shape=[y.shape[0], y.shape[0]])

    for i in range(0, y.shape[0]):
        yi = sum(y[i,:])
        for j in range(0, y.shape[0]):
            coincidence = 0
            for k in range(0, y.shape[1]):
                if (int(y[i,k]) == int(1)) and (int(y[j,k]) == int(1)):
                    coincidence += 1
            L[i,j] = (coincidence + s)/(yi + 2*s)
    
    return L

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

endfile = {
    'ltrain',
    'train',
    'rtrain',
    'ltest',
    'test',
    'rtest'
}

nfolds=10

for s in dataset: #Para los difs datasets
    print './datasets/'+s
    if not os.path.exists("./plots/"+s):
        os.makedirs("./plots/"+s)
    for endfm in endfile: #Para recorrer todos los difs trains
        for fld in range(0, nfolds): #Para recorrer los difs folds
            name='./datasets/'+s+'/'+s+str(fld)+'.'+endfm
            # Read arff file
            
            X, y=readDataFromFile(name)

            L=label_correlation(y.transpose(), 0.19)
            #print L

            fp=open(name+'.coov', 'w')
            L.tofile(fp, sep=" ", format='%s')
            fp.close()
            #tri_lower_no_diag = np.tril(L, k=-1)
            #print tri_lower_no_diag
            tri_indx=L[np.tril_indices(L.shape[0], -1)]
            #print tri_indx

            #listas para almac los indices de los intervalos
            l0=[]
            l1=[] 
            l2=[] 
            l3=[] 
            l4=[] 
            l5=[] 
            l6=[] 
            l7=[] 
            l8=[] 
            l9=[] 
            #Contador para el vector de correlacs
            a=0

            n=L.shape[0]
            cor = np.zeros((n*n) - n)

            #print cor.shape

            for i in range(0, L.shape[0]):
                for j in range(L.shape[1]):
                    if j!=i: #Cogemos todos los valores menos la diagonal ya que importa el orden (etq 1-2 no tiene misma correlac q etiq 2-1)
                        if(L[i][j]<=0.1): #intervalo 0-0.1
                            l0.append([i, j])
                        if(L[i][j]>0.1 and L[i][j]<=0.2): #intervalo 0.1-0.2
                            l1.append([i, j])
                        if(L[i][j]>0.2 and L[i][j]<=0.3): #intervalo 0.2-0.3
                            l2.append([i, j])
                        if(L[i][j]>0.3 and L[i][j]<=0.4): #intervalo 0.3-0.4
                            l3.append([i, j])
                        if(L[i][j]>0.4 and L[i][j]<=0.5): #intervalo 0.4-0.5
                            l4.append([i, j])
                        if(L[i][j]>0.5 and L[i][j]<=0.6): #intervalo 0.5-0.6
                            l5.append([i, j])
                        if(L[i][j]>0.6 and L[i][j]<=0.7): #intervalo 0.6-0.7
                            l6.append([i, j])
                        if(L[i][j]>0.7 and L[i][j]<=0.8): #intervalo 0.7-0.8
                            l7.append([i, j])
                        if(L[i][j]>0.8 and L[i][j]<=0.9): #intervalo 0.8-0.9
                            l8.append([i, j])
                        if(L[i][j]>0.9 and L[i][j]<=1): #intervalo 0.9-1.0
                            l9.append([i, j])
                        
                        cor[a]=L[i][j]
                        a+=1

            cor= -np.sort(-cor)
            #print l0
            
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            labelscorrel=[len(l0), len(l1), len(l2), len(l3), len(l4), len(l5), len(l6), len(l7), len(l8), len(l9)] 
            objects=('0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1')
            y_pos = np.arange(len(objects))
            plt.bar(y_pos, labelscorrel, align='center', alpha=0.5)
            plt.xticks(y_pos, objects)
            plt.xlabel("Correlation interval")
            plt.ylabel('Number of label pairs')
            plt.title(s+': '+'Correlation between labels')
            for i,j in zip(labelscorrel, y_pos):
                plt.annotate(str(labelscorrel[j]), xy=(j, i+(np.max(labelscorrel)*0.01)), horizontalalignment='center')

            plt.savefig('./plots/'+s+'/'+s+str(fld)+endfm+'corrlabls.png')
            plt.close()

            plt.plot(cor)
            plt.axis([0, cor.shape[0], 0, 1.1])
            plt.xlabel('Distinct label pairs')
            #plt.set_xlim(0, cor.shape[0])
            plt.ylabel('Correlation')
            plt.title(s+': '+'Correlation distribution')

            plt.annotate(str("{0:.3f}".format(cor[0])), xy=(0, cor[0]+0.02))
            plt.annotate(str("{0:.3f}".format(cor[-1])), xy=((n*n)-n-2, cor[-1]+0.02))

            plt.savefig('./plots/'+s+'/'+s+str(fld)+endfm+'corrordered.png')
            plt.close()
