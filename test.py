# -*- coding: utf-8 -*-

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
import spacy

nlp = spacy.load('pl_spacy_model')
# http://zil.ipipan.waw.pl/SpacyPL


# doc3 = nlp(u"psy")
# print(doc3)
# for token in doc3:
#     print (token, token.lemma, token.lemma_)




# semSimilarity takes two sentences, tokenize them, lemmatize words, then for every word in first sentence finds most 
# similar word in second sentence; return list of max similarity for words
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
    
# similarity takes two sentenes, returns sum of average value of two lists and divide this by two
def similarity(s1, s2):
    max1 = sentenceSimilarity(s1, s2)
    max2 = sentenceSimilarity(s2, s1)

    return (sum(max1) / float(len(max1)) + sum(max2) / float(len(max2))) / 2

sentences = "słoń są super."
focus_sentence = "psy śmieją pięknymi samochód."

print(sentenceSimilarity(focus_sentence, sentences))
print(sentenceSimilarity(sentences, focus_sentence))
print(similarity(sentences, focus_sentence))
print(similarity(focus_sentence, sentences))