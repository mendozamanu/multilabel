#!/usr/bin/python2.7

import sys
from sklearn.model_selection import StratifiedKFold
import numpy as np
import arff
from skmultilearn.model_selection.measures import folds_label_combination_pairs_without_evidence
from skmultilearn.model_selection.measures import example_distribution
from skmultilearn.model_selection.measures import label_combination_distribution
from skmultilearn.model_selection.measures import folds_without_evidence_for_at_least_one_label_combination

class Transfomer:
    def transform_to_multiclass(self, y):
        self.label_count = y.shape[1]
        self.unique_combinations = {}
        self.reverse_combinations=[]
        self.last_id = 0
        train_vector = []
        for labels_applied in y:
            label_string = ",".join(map(str, labels_applied))

            if label_string not in self.unique_combinations:
                self.unique_combinations[label_string] = self.last_id
                self.reverse_combinations.append(labels_applied)
                self.last_id += 1

            train_vector.append(self.unique_combinations[label_string])
        return train_vector
    
def stratified_folds(n_splits, y):
    t=Transfomer()
    kf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=False)
    folds = [x[1] for x in list(kf.split(np.zeros(y.shape[0]),t.transform_to_multiclass(y)))]
    return folds


# Call
if len(sys.argv) <= 2:
    print "Correct use: multilabel-labelset.py input-file f [output-file-prefix]"
    sys.exit()

f = int(sys.argv[2])

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

kfold = 0
folds = []
desired_number = []

# Labelset k-fold partition
kf = stratified_folds(f, y)


print ("Generating kfolds...")
for test_index in kf:
    train_index = [x for x in range(X.shape[0]) if x not in test_index]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #Classes do not match
    folds.append(train_index)
    desired_number.append((X.shape[0]*(f-1))/f)
    #Training file

    fp = open(suffix+str(kfold)+'.ltrain', 'w')
    
    #Save header
    if sparse:
        fp.write('[MULTILABEL, SPARSE]\n')
    else:
        fp.write('[MULTILABEL, DENSE]\n')
    fp.write('$ %d\n' % len(X_train)) #Number of objects
    fp.write('$ %d\n' % len(X_train[0])) #Number of attributes
    fp.write('$ %d\n' % abs(int(number))) #Number of labels
    
    #Data
    for i in range(0, len(X_train)):
        if sparse:
            for j in range(0, len(X_train[i])):
                if(X_train[i][j] != '0.0'):
            		if(X_train[i][j] == 'YES'):
                		fp.write('1'+' ')
            else:
                fp.write(str(j+1)+':'+str(X_train[i][j])+' ')
        else:
            for j in range(0, len(X_train[i])):
		if(X_train[i][j] == 'YES'):
		    fp.write('1'+' ')
		elif (X_train[i][j] == 'NO'):
		    fp.write('0'+' ')
		else:
		    fp.write(str(X_train[i][j])+' ')
        
        fp.write('[ ')
        for j in range(0, len(y_train[i])):
            if y_train[i][j] == '0.0':
                aux = str(y_train[i][j]).split('.')[0]
                fp.write(str(int(aux))+' ')
            else:
                fp.write(str(int(y_train[i][j]))+' ')
        fp.write(']\n')
    fp.close()
    
    #Testing file
    fp = open(suffix+str(kfold)+'.ltest', 'w')

    #Save header
    if sparse:
        fp.write('[MULTILABEL, SPARSE]\n')
    else:
        fp.write('[MULTILABEL, DENSE]\n')
    fp.write('$ %d\n' % len(X_test))
    fp.write('$ %d\n' % len(X_test[0]))
    fp.write('$ %d\n' % abs(int(number)))
    
    #Data
    for i in range(0, len(X_test)):
        if sparse:
            for j in range(0, len(X_test[i])):
                if(X_test[i][j] != '0.0'):
            		if(X_test[i][j] == 'YES'):
            			fp.write('1'+' ')
            	else:
                        fp.write(str(j+1)+':'+str(X_test[i][j])+' ')
        else:
            for j in range(0, len(X_test[i])):
                if(X_test[i][j] == 'YES'):
            		fp.write('1'+' ')
        	elif (X_test[i][j] == 'NO'):
            		fp.write('0'+' ')
        	else:
                    fp.write(str(X_test[i][j])+' ')
        
        fp.write('[ ')
        for j in range(0, len(y_test[i])):
            if y_test[i][j] == '0.0':
                aux = str(y_test[i][j]).split('.')[0]
                fp.write(str(int(aux))+' ')
            else:
                fp.write(str(int(y_test[i][j]))+' ')
        fp.write(']\n')
    fp.close()

    kfold += 1

fp = open(suffix+'.lmeasures', 'w')
FLZ = folds_label_combination_pairs_without_evidence(y, folds, 1) 
FZ = folds_without_evidence_for_at_least_one_label_combination(y, folds, 1)
LD = label_combination_distribution(y, folds, 1) 
ED = example_distribution(folds, desired_number)
fp.write("Label distribution: ")
fp.write(str(LD)+'\n')
fp.write("Example distribution: ")
fp.write(str(ED)+'\n')
fp.write("Number of fold-label pairs with 0 positive examples, FLZ: ")
fp.write(str(FLZ)+'\n')
fp.write("Number of folds that contain at least 1 label with 0 positive examples, FZ: ")
fp.write(str(FZ)+'\n')
fp.close()
