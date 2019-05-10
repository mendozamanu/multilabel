from collections import Counter
import numpy as np
import os
import sys

endfile = {
    "ltrain",
    "train",
    "rtrain",
    "ltest",
    "test",
    "rtest"
}

if len(sys.argv) <1:
    print "card-kfolds.py dataset"
    sys.exit()
df = str(sys.argv[1])

print ("Computing measures for the chosen dataset...")
print "--> "+ df
nfolds=10
if not os.path.exists("./plots/"+df):
    os.makedirs("./plots/"+df)

for endfm in endfile:
    for fld in range(0, nfolds):
        datafilename="./datasets/"+df+"/"+df+str(fld)+"."+endfm
        
        datafile=open(datafilename)
        l0=datafile.readline()
        l0=l0.split()
        sparse = l0[1]
        if sparse[:-1] == 'SPARSE':
            sparse = True #The file is in sparse mode
        else:
            sparse = False
        
        l1=datafile.readline()
        l2=datafile.readline()
        l3=datafile.readline()
        instances=int(l1.split()[1])
        #print instances
        features=int(l2.split()[1])
        #print features
        labels=int(l3.split()[1])
        #print labels

        l4=datafile.readline()

        avg=0
        tmp=0
        dist=[]
        insts = np.zeros(labels,dtype=int)

        nwdfname="./plots/"+df+"/"+df+str(fld)+endfm+".frecs"
        fp=open(nwdfname, 'w')
        while l4 != "":
            if(l4 == ' '):
                pass
            else:
                if sparse == False:  
                    label = map(int, l4.strip().split()[features+1:features+1+labels])
                    #To remove the '[' ']' from the labels extraction
                    dist.append(''.join(map(str, l4.strip().split()[features+1:features+1+labels])))
                    #print dist en dist tenemos todas las combinacs, luego hacemos el set
                    tmp = sum(label)
                    insts[tmp] += 1
                    avg += sum(label)
                    #print avg
                else:
                    #Sparse . find '[' and start reading until ']'
                    label = map(int, l4.strip().split()[l4.strip().split().index('[')+1:l4.strip().split().index(']')])
                    dist.append(''.join(map(str,l4.strip().split()[l4.strip().split().index('[')+1:l4.strip().split().index(']')])))
                    tmp = sum(label)
                    insts[tmp] += 1
                    avg += sum(label)

            l4=datafile.readline()
        
        fp.write("Num of instances per label-count (0, 1, 2, ... nlabel)\n")
        for i in range(0, insts.shape[0]):
            fp.write(str(i) + ' ' + str(insts[i])+'\n')
        
        fp.write("Labels frequency: \n")
        
        aux=np.zeros(shape=(labels, 2))
        
        for i in range(0, labels):
            aux [i] = (sum(int(row[i]) for row in dist), i+1)
            
            
        aux = aux[(-aux[:,0]).argsort()]
        
        for s in aux:
            fp.write(str(int(s[1]))+' '+str(int(s[0]))+'\n')

        countr=Counter(dist)
        fp.write ("Label combinations frequency: \n")
        for value, count in countr.most_common():
            fp.write(str(int(value, 2))+' '+ str(count)+'\n')

       
        datafile.close()
        fp.close()
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        #insts[] is the vector to plot
        flbs = np.trim_zeros(insts, 'b')
        objects=range(0, flbs.shape[0])
        y_pos = np.arange(len(objects))
        plt.figure(figsize=(15,9))
        plt.bar(y_pos, flbs, align='center', alpha=0.5)
        plt.xticks(y_pos, objects)
        plt.ylabel('Instances')
        plt.xlabel('Num of active labels')
        plt.title(df+': '+'Label frecuency')

        for i,j in zip(flbs, y_pos):
            plt.annotate(str(flbs[j]), xy=(j,i+(np.max(flbs)*0.01)), horizontalalignment='center')

        plt.savefig('./plots/'+df+'/'+df+str(fld)+endfm+'freclbs.png')
        plt.close()

    #Division on python 2.7 returns int by default, 
    #in python3 it returns float so we have to "force" float div on python2.7
