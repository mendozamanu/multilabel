from collections import Counter
def key(item):
    return item[1]
def cardinality(df):
    print ("Computing measures for the chosen dataset...")
    datafilename="./datasets/"+df+"/"+df+".complete"
    
    if datafilename.lower().endswith('.complete') == False :
        sys.exit("Dataset format unknown, please use .arff datasets")    
    
    datafile=open(datafilename)
    l0=datafile.readline()
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
    dist=[]

    nwdfname="./datasets/"+df+"/"+df+".dsetm"
    fp=open(nwdfname, 'w')
    while l4 != "":
        if(l4 == ' '):
            pass
        else:  
            label = map(int, l4.strip().split()[features+1:features+1+labels])
            #To remove the '[' ']' from the labels extraction
            dist.append(''.join(map(str, l4.strip().split()[features+1:features+1+labels])))
            #print dist en dist tenemos todas las combinacs, luego hacemos el set
            avg += sum(label)
            #print avg

        l4=datafile.readline()
    
    fp.write("Labels frequency: \n")
    lf=[]
    for i in range(0, labels):
        aux= (sum(int(row[i]) for row in dist), i+1)
        lf.append(list(aux))
    #np.sum de la columna
    lf.sort(reverse=True)
    #print lf
    for s in lf:
        fp.write(str(s[1])+' '+str(s[0])+'\n')

    countr=Counter(dist)
    fp.write ("Label combinations frequency: \n")
    for value, count in countr.most_common():
    	fp.write(str(value)+' '+ str(count)+'\n')
    #print countr
    un_combs=set(dist)
    #print sorted(un_combs)
    #print ("----------------")
    fp.write ("Cardinality: ")
    card = avg/(instances*1.0)
    fp.write(str(card)+'\n')
    
    fp.write("Density: ")
    fp.write (str(card/(labels*1.0))+'\n')

    fp.write("Distinct: ")
    fp.write(str(len(un_combs))+'\n')
   
    datafile.close()
    fp.close()
    #Division on python 2.7 returns int by default, 
    #in python3 it returns float so we have to "force" float div on python2.7

def main():
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
    for ds in dataset:
        cardinality(ds)
  
if __name__== "__main__":
  main()