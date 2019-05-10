#!/usr/bin/python2.7

import sys
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
    os.system('nohup python2.7 ./coov-kfolds.py '+ str(s) + ' &')