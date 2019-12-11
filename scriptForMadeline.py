import numpy as np
from rake_nltk import Rake
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
import pandas as pd


MAX_PHRASE_LEN = 2
MODEL_FILE = '../glove.twitter.27B/glove.twitter.27B.50d.txt'

# glove_input_file = MODEL_FILE
word2vec_output_file = '../word2vec.txt'
# glove2word2vec(glove_input_file, word2vec_output_file)

# model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
model = gensim.models.KeyedVectors.load('model.model')
print('Loaded!')
# model = gensim.models.
# model.save('model.model')

# model = Word2Vec.load('model.model')
def categorize(tweet):
    r = Rake(max_length=MAX_PHRASE_LEN)
    r.extract_keywords_from_text(tweet)
    return r.get_ranked_phrases()
'''
Go through tweets, rake keywords from each
For each keyword, generate a set of related words (using word embeddings? glove
'''

def similar_words(model, tweet):
    phrases = categorize(tweet)
    ret = set()
    for p in phrases:
        words = p.split()
        words = filter(lambda x: x in model.vocab, words)
        for w in words:
            sims = model.most_similar(w)
            for s in sims:
                ret.add(s[0])
    return ret

trump = pd.read_csv('Trump.csv') #These may need to be changed
warren = pd.read_csv('Elizabeth Warren.csv') # ^
trump['History'] = ''
warren['History'] = ''

for i,x in trump.iterrows():
    tweet = x['Tweet']
    # print('tweet: ', tweet)
    # print(type(tweet))
    if type(tweet) == str:
        words = list(similar_words(model, tweet))
        spacedOut = ' '.join(words)
        trump.at[i, 'History'] = spacedOut

for i,x in warren.iterrows():
    tweet = x['Tweet']
    words = list(similar_words(model, tweet))
    spacedOut = ' '.join(words)
    warren.at[i, 'History'] = spacedOut

bigDataFrame = pd.concat(trump, warren)
trump.to_csv('newTrumpWithHistory.csv')
warren.to_csv('newWarrenWithHistory.csv')
bigDataFrame.to_csv('concatenatedDFWithHistory.csv')
# g = categorize('I love Europe and foreign policy')

# print(g)

# for phrase in g:
#     words = phrase.split()
#     for word in words:
#         print(model.most_similar(word))

# print(model.most_similar('president'))
# print(model.most_similar('china'))