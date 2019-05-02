#Listado de datasets a ejecutar
import os

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

for s in dataset:
   name='./datasets/'+s+'/'+s+'.arff'
   #run python2.7 kfold-labelset.py './datasets/'+s+'/'+s'.arff'
   os.system('nohup python2.7 /home/i32petoj/pruebasManuel/multilabel/multilabel-convert.py ' + str(name) + ' ' + ' &')
