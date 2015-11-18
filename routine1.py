__author__ = 'eduardo'

from step1_lib import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer


with open("bigram_pp.pickle", "rb") as vocab_file:
    vocab = pickle.load(vocab_file)

bigram_pp = BiGramPreProcessor(vocab=vocab)

with open("svm_trained.classifier", "rb") as cl_file:
    classifier = pickle.load(cl_file)

news_cl = NewsClassifier(classifier)

corpus = batch_get_corpus(bigram_pp, url_lists=["test_urls.txt"], url_lists_tags=["ND"])

labels = news_cl.get_labels(corpus)

for i, label in enumerate(labels):
    if label == u"D":
        print "news number " + str(i+1) + " :" + " about Natural Disaster"
    else:
        print "news number " + str(i+1) + " :" + " not about Natural Disaster"