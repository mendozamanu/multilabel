#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math

methods = {
    '_iterative',
    '_random',
    '_labelset'
}
for method in methods:
    br = pd.read_csv('csv/BinaryRelevance'+method+'.csv', sep=';', decimal='.', index_col=False)

    cc = pd.read_csv('csv/ClassifierChain'+method+'.csv', sep=';', decimal='.', index_col=False)

    lp = pd.read_csv('csv/LabelPowerset'+method+'.csv', sep=';', decimal='.', index_col=False)

    ml = pd.read_csv('csv/MLkNN'+method+'.csv', sep=';', decimal='.', index_col=False)

    tables = [
    br,
    cc,
    lp,
    ml
    ]

    tmp=np.zeros(len(tables))
    tmp2=np.zeros(len(tables))
    tmp3=np.zeros(len(tables))
    tmp4=np.zeros(len(tables))
    tmp5=np.zeros(len(tables))
    tmp6=np.zeros(len(tables))
    tmp7=np.zeros(len(tables))
    tmp8=np.zeros(len(tables))
    tmp9=np.zeros(len(tables))
    tmp10=np.zeros(len(tables))
    tmp11=np.zeros(len(tables))


    
    for tb in tables:
        tm=np.zeros(len(tb['Dataset']))
        tb['Acc Rank'] = tm
        tb['HL Rank'] = tm
        tb['Coverage Rank'] = tm
        tb['Ranking loss Rank'] = tm
        tb['Avg prec macro Rank'] = tm
        tb['Avg prec micro Rank'] = tm
        tb['ROC Rank'] = tm
        tb['F1 micro Rank'] = tm
        tb['Recall micro Rank'] = tm
        tb['F1 macro Rank'] = tm
        tb['Recall macro Rank'] = tm

    j=0
    p=1
    for i in range(0, len(br['Dataset'])):
        j=0
        for tb in tables:
            #print tb['Accuracy'][i]
            #tenemos q hacer ranks de todos los tb['Accuracy'][i]
            #print tb
            tmp[j]=tb['Accuracy'][i]
            tmp2[j]=tb['Hamming Loss'][i]
            tmp3[j]=tb['Coverage'][i]
            tmp4[j]=tb['Ranking loss'][i]
            tmp5[j]=tb['Avg precision macro'][i]
            tmp6[j]=tb['Avg precision micro'][i]
            tmp7[j]=tb['ROC AUC'][i]
            tmp8[j]=tb['f1 score (micro)'][i]
            tmp9[j]=tb['Recall (micro)'][i]
            tmp10[j]=tb['f1 score (macro)'][i]
            tmp11[j]=tb['Recall (macro)'][i]
            j+=1
            
        #print tmp7
        p=1
        while True:
            j=0
            idx=np.argmax(tmp)
            idx2=np.argmin(tmp2)
            idx3=np.argmax(tmp3)
            idx4=np.argmax(tmp4)
            idx5=np.argmax(tmp5)
            idx6=np.argmax(tmp6)
            idx7=np.argmax(tmp7)
            idx8=np.argmax(tmp8)
            idx9=np.argmax(tmp9)
            idx10=np.argmax(tmp10)
            idx11=np.argmax(tmp11)
            
            if math.isnan(tmp[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['Acc Rank'][i] = float('nan')
            else:
                tmp[idx] = 0
                tables[idx]['Acc Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp2[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['HL Rank'][i] = float('nan')
            else:
                tmp2[idx2] = np.inf
                tables[idx2]['HL Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp3[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['Coverage Rank'][i] = float('nan')
            else:
                tmp3[idx3] = 0
                tables[idx3]['Coverage Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp4[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['Ranking loss Rank'][i] = float('nan')
            else:
                tmp4[idx4] = 0
                tables[idx4]['Ranking loss Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp5[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['Avg prec macro Rank'][i] = float('nan')
            else:
                tmp5[idx5] = 0
                tables[idx5]['Avg prec macro Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp6[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['Avg prec micro Rank'][i] = float('nan')
            else:
                tmp6[idx6] = 0
                tables[idx6]['Avg prec micro Rank'][i] = p
            #----------------------------------------------------
            #print idx7
            if math.isnan(tmp7[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['ROC Rank'][i] = float('nan')
            else:
                tmp7[idx7] = 0
                tables[idx7]['ROC Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp8[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['F1 micro Rank'][i] = float('nan')
            else:
                tmp8[idx8] = 0
                tables[idx8]['F1 micro Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp9[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['Recall micro Rank'][i] = float('nan')
            else:
                tmp9[idx9] = 0
                tables[idx9]['Recall micro Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp10[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['F1 macro Rank'][i] = float('nan')
            else:
                tmp10[idx10] = 0
                tables[idx10]['F1 macro Rank'][i] = p
            #----------------------------------------------------
            if math.isnan(tmp11[j]):
                for rw in range(0, len(tables)):
                    tables[rw]['Recall macro Rank'][i] = float('nan')
            else:
                tmp11[idx11] = 0
                tables[idx11]['Recall macro Rank'][i] = p
            #----------------------------------------------------
           
            p+=1
            j+=1
            if p>4:
                break

    br['Mean'] = br[['Acc Rank', 'HL Rank', 'Coverage Rank', 'Ranking loss Rank', 'Avg prec macro Rank', 'Avg prec micro Rank', 
                    'ROC Rank', 'F1 micro Rank', 'Recall micro Rank', 'F1 macro Rank', 'Recall macro Rank']].mean(axis=1)

    cc['Mean'] = cc[['Acc Rank', 'HL Rank', 'Coverage Rank', 'Ranking loss Rank', 'Avg prec macro Rank', 'Avg prec micro Rank', 
                    'ROC Rank', 'F1 micro Rank', 'Recall micro Rank', 'F1 macro Rank', 'Recall macro Rank']].mean(axis=1)

    lp['Mean'] = lp[['Acc Rank', 'HL Rank', 'Coverage Rank', 'Ranking loss Rank', 'Avg prec macro Rank', 'Avg prec micro Rank', 
                    'ROC Rank', 'F1 micro Rank', 'Recall micro Rank', 'F1 macro Rank', 'Recall macro Rank']].mean(axis=1)

    ml['Mean'] = ml[['Acc Rank', 'HL Rank', 'Coverage Rank', 'Ranking loss Rank', 'Avg prec macro Rank', 'Avg prec micro Rank', 
                    'ROC Rank', 'F1 micro Rank', 'Recall micro Rank', 'F1 macro Rank', 'Recall macro Rank']].mean(axis=1)

    br.to_csv('csv/ranked'+method+'-br.csv', sep=';', decimal='.', index=False)

    cc.to_csv('csv/ranked'+method+'-cc.csv', sep=';', decimal='.', index=False)

    lp.to_csv('csv/ranked'+method+'-lp.csv', sep=';', decimal='.', index=False)

    ml.to_csv('csv/ranked'+method+'-mlknn.csv', sep=';', decimal='.', index=False)

