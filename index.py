# -*- coding: utf-8 -*-
import json
import string
from pprint import pprint
from nltk.tokenize import word_tokenize
import nltk
# from spacy.lang.pl import STOP_WORDS
# sources: https://github.com/bieli/stopwords/blob/master/polish.stopwords.txt and https://github.com/stopwords-iso/stopwords-pl


with open('debateTVP/nodeset16955.json') as f:
    data = json.loads(f.read())

#Creating nodesById dictionary which has nodeID as key and whole node as value for more efficient data extraction.
nodesById = {}
for i, node in enumerate(data['nodes']):
  nodesById[node['nodeID']] = node

#Premises are nodes that have ingoing edges that are type 'RA' and outgoing edges that are type 'I'.
premises = [x for x in data['edges'] if nodesById[x['fromID']]['type'] == 'I' and nodesById[x['toID']]['type'] == 'RA' ]

#Conclusions are nodes that have ingoing edges that are type 'I' and outgoing edges that are type 'RA'.
conclusions = [x for x in data['edges'] if nodesById[x['toID']]['type'] == 'I' and nodesById[x['fromID']]['type'] == 'RA' ]

# conclusionPremiseDict Create dictionary of pairs with an identifier with the following form: 
# {id: {"conclusion": <SINGLE_CONCLUSION>, "premises":[<LIST_OF_PREMISES>]}}
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

edgePairs = conclusionPremiseDict(premises, conclusions)
    
# aduPairs create list of ADU pairs containing connected conclusion and premise [[conclusion, premise]]
def aduPairs(edgePairs):
    aduPair = []
    for pair in edgePairs.values():
        for p in pair['premises']:
          aduPair.append([nodesById[pair['conclusion']['toID']]['text'], nodesById[p['fromID']]['text']])
    return(aduPair)

# adus = aduPairs(edgePairs)



# removeStopwords remove words that are the most common words in any natural language, this function also removes punctuation
# def removeStopwords(adus):
#     all_filtered_words = []
#     for i in adus:
#         filtered_words = [word for word in word_tokenize(i[0]+i[1]) if word not in STOP_WORDS and word not in string.punctuation]
#         all_filtered_words += filtered_words
#     return(all_filtered_words)

# words = removeStopwords(adus)