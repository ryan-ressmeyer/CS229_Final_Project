import csv
import numpy as np
import gensim

from rake_nltk import Rake
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec

MAX_PHRASE_LEN = 2
MODEL_FILE = '../glove.twitter.27B/glove.twitter.27B.50d.txt'

# glove_input_file = MODEL_FILE


def categorize(tweet):
    r = Rake(max_length=MAX_PHRASE_LEN)
    r.extract_keywords_from_text(tweet)
    return r.get_ranked_phrases()
'''
Go through tweets, rake keywords from each
For each keyword, generate a set of related words (using word embeddings? glove
'''
count = 0
def get_history(model, tweet):
    # print('row: ', count)
    # count += 1
    phrases = categorize(tweet)
    ret = ''
    for p in phrases:
        words = p.split()
        words = filter(lambda x: x in model.vocab, words)
        for w in words:
            sims = model.most_similar(w)
            for s in sims:
                ret += s[0]
                ret += ' '
    # space-separated string of words in history
    return ret



# g = categorize('I love Europe and foreign policy')

# print(g)

# for phrase in g:
#     words = phrase.split()
#     for word in words:
#         print(model.most_similar(word))
        
# print(model.most_similar('president'))
# print(model.most_similar('china'))

accounts = ['Adam Schiff.csv', 'Alexandria Cortez.csv', 'Biden.csv', 'Elizabeth Warren.csv', 'Hillary.csv', 'Kamala Harris.csv', 'Obama.csv', 'Sanders.csv', 'Trump.csv']
word2vec_output_file = '../word2vec.txt'
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
print('Loaded model!')
# for acc in accounts:
acc = 'Elizabeth Warren copy.csv'
with open(acc, 'r+') as csvfile:
    print('here')
    reader = csv.reader(csvfile)
    writer = csv.writer(csvfile)
    histories = [[get_history(model, row[2])] for row in reader]
    writer.writerows(histories)
    
    # for row in reader:
    #     hist = get_history(model, row[2])
    #     writer.writerow([hist])

    # wtr = csv.writer(open(csv, 'w'), delimiter=',', lineterminator='\n')