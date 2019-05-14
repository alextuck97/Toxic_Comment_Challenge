# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:18:37 2019

Preprocess and clean the data.

Use Porter Stemmer to stem comments and make new column 'stem_text'.
Use regular expression tokenizer to prune words with repeated characters.
Remove all digits and punctuation.
Make a new column 'clean' to mark which comments are not innappropriate.
Join test.csv with test_labels.csv.

Save to new files test_stemmed.csv and train_stemmed.csv.

@author: Alex
"""

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer


import numpy as np
import pandas as pd
import string
import re

pattern = re.compile(r"(.)\1{2,}")
porter=PorterStemmer()
tokenizer=RegexpTokenizer('\w[a-z]{1,20}\w')
#Limit to 20 character words to reduce spammy results
tab=str.maketrans(dict.fromkeys(string.punctuation+string.digits))

def stemmer(row):
    sentence=row['comment_text']
    
    sentence=sentence.translate(tab)
    
    tokens = tokenizer.tokenize(sentence.lower())
    
    stemmed=[]
    
    for t in tokens:
        w=pattern.sub(r"\1\1",t)
        s=porter.stem(w)
        stemmed.append(s)
        stemmed.append(" ")
        
    return "".join(stemmed)

def makeCleanCol(row):
    '''
    If all comment characteristics are 0, set
    'clean' = 1
    '''
    if sum(row[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate']])>0:#3 to 9 b/c I forgot saving stem_text saved index as new column
        val=0
    else:
        val=1
    return val

if __name__=='__main__':
    
    
    test_data=pd.read_csv('toxic_comments/test.csv')
    train_data=pd.read_csv('toxic_comments/train.csv')
    
    test_labels=pd.read_csv('toxic_comments/test_labels.csv')
    
    test_data=test_data.merge(test_labels, on='id')
    
    test_data=test_data[test_data['toxic']!=-1].copy()
    
    test_data['stem_text']=test_data.apply(stemmer,axis=1)
    train_data['stem_text']=train_data.apply(stemmer,axis=1)
    
    test_data['clean']=test_data.apply(makeCleanCol,axis=1)
    train_data['clean']=train_data.apply(makeCleanCol,axis=1)
    
    test_data.replace('',np.nan,inplace=True)
    train_data.replace('',np.nan,inplace=True)
    
    test_data.dropna(inplace=True,axis=0)
    train_data.dropna(inplace=True,axis=0)
    
    test_data.to_csv('toxic_comments/test_stemmed.csv')
    train_data.to_csv('toxic_comments/train_stemmed.csv')
    
    
    
    
    
    
    
    
    
    
    
    
    
        