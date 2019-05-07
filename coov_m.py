#!/usr/bin/python2.7

import sys
from sklearn.model_selection import KFold
import numpy as np
import arff
from skmultilearn.model_selection.measures import folds_label_combination_pairs_without_evidence
from skmultilearn.model_selection.measures import example_distribution
from skmultilearn.model_selection.measures import label_combination_distribution
from skmultilearn.model_selection.measures import folds_without_evidence_for_at_least_one_label_combination
from sklearn.covariance import EmpiricalCovariance

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

# Call
if len(sys.argv) <= 1:
    print "Correct use: coov.py input-file [output-file-prefix]"
    sys.exit()

# Read arff file
if sys.argv[1].lower().endswith('.arff') == False :
    sys.exit("Dataset format unknown, please use .arff datasets")    
dataset = arff.load(open(sys.argv[1], 'rb'))
data = np.array(dataset['data'])

#We have to get the number of clases from the raw file
file = open(sys.argv[1], "r")
line = file.readline()

flag = False
for i in line.split():
    if flag is True:
        number = i
        break
    if (i == "-C") or (i == "-c"):
        flag = True

if (flag==False):
    file.close()
    sys.exit("Wrong format for the dataset header")

if number[-1:] == "'":
    number = number[:-1]
file.close()
#Now we have the number stored, knowing that positive means the first attributes and negative the last ones


nominalIndexArray = []
nominals = []
aux = 0
#from attributes we can get if its nominal
if int(number) > 0:
    for x in dataset['attributes'][int(number):]:
        if (len(x[1]) > 2) and (x[1] != ("NUMERIC" or "REAL" or "INTEGER" or "STRING")):
            nominalIndexArray.append(aux)
            nominals.append(x[1])
        aux +=1
else:
    for x in dataset['attributes'][:int(number)]:
        if (len(x[1]) > 2) and (x[1] != ("NUMERIC" or "REAL" or "INTEGER" or "STRING")):
            nominalIndexArray.append(aux)
            nominals.append(x[1])
        aux +=1

#Split the data in X and Y
if(int(number)>0):
    y = data[:,0:int(number)].astype(int)
    x = data[:,int(number):]
else:
    y = data[:,int(number):].astype(int)
    x = data[:,:int(number)]

if len(nominalIndexArray) > 0:
    #Change the nominal attributes to numeric ones
    index = 0
    X = []
    for k in x:
        numericVector = []
        for i in range(0, len(nominalIndexArray)):
            #Ahora tenemos que crear el vector que le vamos a poner al final de cada 
            checkIfMissing = False
            for aux in nominals[i]:
                if aux == k[nominalIndexArray[i]]:
                    #Add 1 to the array
                    checkIfMissing = True
                    numericVector.append(1)
                else:
                    #Add 0 to the array
                    checkIfMissing = True
                    numericVector.append(0)
            if checkIfMissing is False:
                #Add another 1 to the array
                numericVector.append(1)
            else:
                numericVector.append(0)
        auxVector = np.append(k, [numericVector])
        #Substract that nominals values
        auxVector = np.delete(auxVector, nominalIndexArray)
        X.append(auxVector)
                   
    X = np.array(X)
else:
    X = np.array(x)

# Sparse or dense?
sizeofdouble = 8
sizeofint = 4
sizeofptr = 8
dense_size = len(X)*len(X[0])*sizeofdouble+len(X)*sizeofptr
#nz = np.count

L=label_correlation(y.transpose(), 0.19)
#print L
fp=open(sys.argv[1]+'.coov', 'w')
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

import matplotlib.pyplot as plt
labelscorrel=[len(l0), len(l1), len(l2), len(l3), len(l4), len(l5), len(l6), len(l7), len(l8), len(l9)] 
objects=('0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1')
y_pos = np.arange(len(objects))
plt.bar(y_pos, labelscorrel, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Num of labels')
plt.title('Correlation labels')

plt.show()

plt.plot(cor)
plt.xlabel('Labels')
plt.ylabel('Correlation')
plt.title('Correlation ordered')
plt.show()