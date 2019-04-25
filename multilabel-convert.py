#!/usr/bin/python2.7

import sys
from skmultilearn.model_selection import IterativeStratification
import numpy as np
import arff
from skmultilearn.model_selection.measures import folds_label_combination_pairs_without_evidence
from skmultilearn.model_selection.measures import example_distribution
from skmultilearn.model_selection.measures import label_combination_distribution
from skmultilearn.model_selection.measures import folds_without_evidence_for_at_least_one_label_combination
from skmultilearn.model_selection import iterative_train_test_split


# Call
if len(sys.argv) <= 1:
    print "multilabelKfold.py input-file [output-file-prefix]"
    sys.exit()

# Read arff file
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
#nz = np.count_nonzero(X)
#Count_nonzero is not working so i'll count the non zeroes by myself (This will take a little longer but better than not counting them)
nz = 0
for i in range(0, len(X)):
    for j in range(0, len(X[0])):
        if X[i][j] != '0.0':
            nz += 1
sparse_size = nz*(sizeofdouble+sizeofint)+2*len(X)*sizeofptr+len(X)*sizeofint

sparse = False if sparse_size >= dense_size else True

# Use input file as output suffix if no other given
suffix = sys.argv[3] if len(sys.argv) == 4 else sys.argv[1][:sys.argv[1].rfind('.')]

#Complete file

fp = open(suffix+'.complete', 'w')

#Save header
if sparse:
    fp.write('[MULTILABEL, SPARSE]\n')
else:
    fp.write('[MULTILABEL, DENSE]\n')
fp.write('$ %d\n' % len(X)) #Number of objects
fp.write('$ %d\n' % len(X[0])) #Number of attributes
fp.write('$ %d\n' % abs(int(number))) #Number of labels

#Data
for i in range(0, len(X)):
	
	#fp.write('[ ')
	for j in range(0, len(y[i])):
		if y[i][j] == '0.0':
			aux = str(y[i][j]).split('.')[0]
			fp.write(str(int(aux))+' ')
		else:
			fp.write(str(int(y[i][j]))+' ')
	fp.write('\n')

#Save header
fp.close()
