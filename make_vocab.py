# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 23:10:54 2019

Generate the features to be used for each category
of comments. Words are stemmed, then each occurence of 
a word is counted. Each word is only counted once per
comment. Out of the 300 most frequent words in the toxic
category and the 300 most frequent words in the clean
category, the vocabulary is selected as the words in the
toxic category but not in the clean category. 

@author: Alex
"""


from nltk.tokenize import SpaceTokenizer
from sklearn.feature_extraction import text
import pandas as pd
from nltk.stem.porter import PorterStemmer

s_tok=SpaceTokenizer()
porter=PorterStemmer()

additional_stopwords=['wiki','utc','wikipedia']
stop_words=text.ENGLISH_STOP_WORDS.union(additional_stopwords)
stemmed_stop_words=set([porter.stem(w) for w in stop_words])


def makeFrequencyDict(comments):
    '''
    Make a dictionary of the frequency a word appears in the corpus.
    Frequencies are normalized by number of example comments.
    '''
    n_examples=len(comments)
    freqDict={}
    
    for c in comments:
        
        
        stemmed_tokens=s_tok.tokenize(c)
        stemmed_tokens=[token for token in stemmed_tokens if token != '']#Remove blank space
            
        stemmed_tokens=set(stemmed_tokens)
        for s in stemmed_tokens:
            if s in freqDict and s not in stemmed_stop_words:
                freqDict[s]+=1/n_examples
                
            elif s not in stemmed_stop_words:
                freqDict[s]=1/n_examples
                
    
    return freqDict

        

    

if __name__=='__main__':
    train_data=pd.read_csv('toxic_comments/train_stemmed.csv')
    
    catagories = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
    
    for cat in catagories:
        
        print(cat)
        train_cat=train_data[train_data[cat]==1]['stem_text']
        train_not_cat=train_data[train_data['clean']==1]['stem_text']
    
    
        d_cat=makeFrequencyDict(train_cat)
        d_not_cat=makeFrequencyDict(train_not_cat)
        
        freq_cat=sorted(d_cat.items(), key = lambda kv:(kv[1], kv[0]))[-300:]
        freq_not=sorted(d_not_cat.items(), key = lambda kv:(kv[1], kv[0]))[-300:]
    
        cat_set=set([pair[0] for pair in freq_cat])
        not_set=set([pair[0] for pair in freq_not])
    
        unique_to_cat=cat_set-not_set
    
        s=pd.Series(list(unique_to_cat))
        s.to_csv(f"vocab/vocab_{cat}.csv",index=False,header=None)
    
    #nned to standardize how the data is formatted.
    #Pretty sure stem_text is the same as how makeFrequencyDict
    #does it, but nned to standardize for future.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
