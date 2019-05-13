import os
import sys

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

for s in dataset:
    os.system('nohup python2.7 ./iterativekfoldbase.py '+ str(s) + ' &')
