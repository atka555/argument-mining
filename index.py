# -*- coding: utf-8 -*-
import json
import string
from pprint import pprint
from nltk.tokenize import word_tokenize
import nltk
import glob
from nltk.corpus import wordnet as wn
import spacy
import itertools
import random
import numpy as np
from sklearn.utils import shuffle

# from spacy.lang.pl import STOP_WORDS
# sources: https://github.com/bieli/stopwords/blob/master/polish.stopwords.txt and https://github.com/stopwords-iso/stopwords-pl


maps = glob.glob("maps/*.json")

'''
    conclusionPremiseDict Create dictionary of pairs with an identifier with the following form: 
    {id: {"conclusion": <SINGLE_CONCLUSION>, "premises":[<LIST_OF_PREMISES>]}}
'''
def conclusionPremiseDict(premises, conclusions):
    pairs = {}
    for i, x in enumerate(conclusions):
        # print (x)
        pairs[i] = {'conclusion':x, 'premises':[]}
        id_to = x['fromID']
        for p in premises:
            if p['toID'] == id_to:
                pairs[i]['premises'].append(p)
                
    return pairs

'''
    aduPairs create list of ADU pairs containing connected conclusion and premise [[conclusion, premise]] with text not id
'''
def aduPairs(edgePairs, nodesById):
    aduPair = []
    for pair in edgePairs.values():
        for p in pair['premises']:
          aduPair.append([nodesById[pair['conclusion']['toID']]['text'], nodesById[p['fromID']]['text']])
    return(aduPair)

'''
    pairs creates conclusion - premise pairs for one map
'''
def pairs(map):
    with open(map) as f:
        data = json.loads(f.read())
    #Creating nodesById dictionary which has nodeID as key and whole node as value for more efficient data extraction.
    nodesById = {}
    for i, node in enumerate(data['nodes']):
        nodesById[node['nodeID']] = node
    # pprint(nodesById)
    #Premises are nodes that have ingoing edges that are type 'RA' and outgoing edges that are type 'I'.
    premises = [x for x in data['edges'] if nodesById[x['fromID']]['type'] == 'I' and nodesById[x['toID']]['type'] == 'RA' ]

    #Conclusions are nodes that have ingoing edges that are type 'I' and outgoing edges that are type 'RA'.
    conclusions = [x for x in data['edges'] if nodesById[x['toID']]['type'] == 'I' and nodesById[x['fromID']]['type'] == 'RA' ]
    edgePairs = conclusionPremiseDict(premises, conclusions)
    adus = aduPairs(edgePairs, nodesById)
    return adus, conclusions, premises, nodesById

'''
    comb makes combination of conclusions and premises lists and returns list of pairs that are not conclusion-premise pairs 
'''

def comb(conclusions, premises, l, nodesById):
    combList = [(x,y) for x in conclusions for y in premises] 
    smallCombList = []
    for _ in range(l):
        p = random.choice(combList)
        smallCombList.append([nodesById[p[0]['toID']]['text'], nodesById[p[1]['fromID']]['text']])
    return smallCombList


'''
    truePairs is list of all conclusion-premise pairs; falsePairs is list od conclusion-premise non pairs
'''
truePairs = []
conclusions = []
premises = []
nodesById = {}

for m in maps:
    adus, c, p, n = pairs(m)
    truePairs.extend(adus)
    conclusions.extend(c)
    premises.extend(p)
    nodesById = {**nodesById, **n}

falsePairs = comb(conclusions, premises, len(truePairs), nodesById)


'''
     http://zil.ipipan.waw.pl/SpacyPL
'''
nlp = spacy.load('pl_spacy_model')

'''
    semSimilarity takes two sentences, tokenize them, lemmatize words, then for every word in first sentence finds most 
    similar word in second sentence; return list of max similarity for words
'''
def sentenceSimilarity (s1, s2):
    lem1 = nlp(s1)
    lem2 = nlp(s2)
    maxSimilarity = []
    for token1 in lem1:
        if wn.synsets(token1.lemma_) != []:
            wordSimilarity = []
            word1 = wn.synsets(token1.lemma_)[0]
            for token2 in lem2:
                if wn.synsets(token2.lemma_) != []:
                    word2 = wn.synsets(token2.lemma_)[0]
                    wordSimilarity.append(word1.wup_similarity(word2))
            wordSimilarityWNone = [x for x in wordSimilarity if x]
            if len(wordSimilarityWNone) != 0:
                maxSimilarity.append(max(wordSimilarityWNone))       
    return maxSimilarity
    
'''
    similarity takes two sentenes, returns sum of average value of two lists and divide this by two
'''
def similarity(s1, s2):
    max1 = sentenceSimilarity(s1, s2)
    max2 = sentenceSimilarity(s2, s1)

    if float(len(max1)) == 0 or float(len(max2)) == 0:
        return 0
    else:
        return (sum(max1) / float(len(max1)) + sum(max2) / float(len(max2))) / 2

simTruePairs = []
trueLabels = []
simFalsePairs = []
falseLabels = []

for p in truePairs:
    # print(similarity(p[0], p[1]))
    simTruePairs.append(similarity(p[0], p[1]))
    trueLabels.append(1)

for p in falsePairs:
    # print(similarity(p[0], p[1]))
    simFalsePairs.append(similarity(p[0], p[1]))
    falseLabels.append(0)

print(simTruePairs)
print(simFalsePairs)

simTruePairs.extend(simFalsePairs)
trueLabels.extend(falseLabels)

trainSamples = np.array(simTruePairs)
trainLabels = np.array(trueLabels)
trainLabels, trainSamples = shuffle(trainLabels, trainSamples)

print(trainSamples)
print(trainLabels)



# removeStopwords remove words that are the most common words in any natural language, this function also removes punctuation
# def removeStopwords(adus):
#     all_filtered_words = []
#     for i in adus:
#         filtered_words = [word for word in word_tokenize(i[0]+i[1]) if word not in STOP_WORDS and word not in string.punctuation]
#         all_filtered_words += filtered_words
#     return(all_filtered_words)

# words = removeStopwords(adus)