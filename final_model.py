# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:24:38 2019

Build a dictionary of models to make predictions on unseen
comments. 

@author: Alex
"""

import sklearn.metrics as met
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

from model_cross_validation import formatData


import pandas as pd

from sklearn.feature_extraction import text
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import ComplementNB



from sklearn.svm import LinearSVC
import re
import string

pattern = re.compile(r"(.)\1{2,}")
porter=PorterStemmer()
tokenizer=RegexpTokenizer('\w[a-z]{1,20}\w')
#Limit to 20 character words to reduce spammy results
tab=str.maketrans(dict.fromkeys(string.punctuation+string.digits))



def trainModel():
    '''
    Train a dictionary of models based on the best models grid search
    returned. I chose models with high precision over recall.
    '''

    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    #models = {'toxic':LinearSVC(C=1,dual=False,penalty='l2',fit_intercept=True,tol=.001), 
    #          'severe_toxic':LinearSVC(C=9,dual=False,penalty='l1',fit_intercept=True,tol=1e-5), 
    #          'obscene':LinearSVC(C=3,dual=False,penalty='l2',fit_intercept=True,tol=.1), 
    #          'threat':RandomForestClassifier(class_weight='balanced',n_estimators=10,criterion='gini',min_samples_leaf=13,min_samples_split=10), 
    #          'insult':LinearSVC(C=4,dual=False,penalty='l2',fit_intercept=True,tol=.001), 
    #          'identity_hate':RandomForestClassifier(class_weight=None,n_estimators=10,criterion='entropy',min_samples_leaf=1,min_samples_split=26)}
    
    models=dict.fromkeys(categories, [ComplementNB(),None])
    
    for cat in categories:
        train_data = formatData('toxic_comments/train_stemmed.csv',cat)
        vocab=pd.read_csv(f'vocab/vocab_{cat}.csv',header=None)[0] 
        
        v=text.CountVectorizer(vocabulary=vocab,binary=True)
        #v=text.CountVectorizer(binary=False)
        
        train_fvs=v.transform(train_data['stem_text'])
        train_labels=train_data[cat]
        
        models[cat][0].fit(train_fvs,train_labels)
        models[cat][1]=v
    
    return models
    

def preprocessComment(comment):
    '''
    Stem and clean a comment
    '''
    sentence=comment.translate(tab)
    
    tokens = tokenizer.tokenize(sentence.lower())
    
    stemmed=[]
    
    for t in tokens:
        w=pattern.sub(r"\1\1",t)
        s=porter.stem(w)
        stemmed.append(s)
        stemmed.append(" ")
        
    return "".join(stemmed)


def makePrediction(comment, model):
    '''
    Make a prediction on a new comment
    '''
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    prediction = dict.fromkeys(categories,0)
    
    s=preprocessComment(comment)
    
    for cat in categories:
        vocab=pd.read_csv(f'vocab/vocab_{cat}.csv',header=None)[0] 
        v=text.CountVectorizer(vocabulary=vocab,binary=True)
        
        fv=v.transform([s])
        
        prediction[cat]=model[cat].predict(fv)[0]
    
    return prediction
    


if __name__=='__main__':
    
    m=trainModel()
    
    for key, value in m.items():
        train_data = formatData('toxic_comments/train_stemmed.csv',key)
        test_data = formatData('toxic_comments/test_stemmed.csv',key)
        
        train_fvs = value[1].transform(train_data['stem_text'])
        train_true = train_data[key]
        train_pred = value[0].predict(train_fvs)

        test_fvs = value[1].transform(test_data['stem_text'])
        test_true = test_data[key]
        test_pred = value[0].predict(test_fvs)
        
        print(f"{key} Train\n")
        print(met.confusion_matrix(train_true,train_pred))
        
        print(f"{key} Test\n")
        print(met.confusion_matrix(test_true,test_pred))
                
        
    
    #met.confusion_matrix()
    
    #print(makePrediction('very mean comment go die',m))
    #print(makePrediction('very nice happy comment yay',m))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    