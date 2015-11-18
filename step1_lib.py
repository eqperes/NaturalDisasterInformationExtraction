"""
Step1 - Classify News as 'D' if they are about natural disasters.
Portuguese news only.
"""

__author__ = 'eduardo'


from sklearn.linear_model import SGDClassifier
import codecs
from nltk.stem import RSLPStemmer
from gensim.corpora import Dictionary
from newspaper import Article
from nltk.tokenize import word_tokenize
import math
from scipy.sparse import csr_matrix
import numpy as np
import pickle
import feedparser
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import CountVectorizer

IDF_PATH = "../data/idf.txt"

class PreProcessor(object):

    def __init__(self, idf_path=IDF_PATH, use_idf=True):
        self.stemmer = RSLPStemmer()
        self.term_dict, self.freq = self.dict_from_idf(idf_path)
        self.max_freq = float(max(self.freq.values()))
        self.vocab_size = len(self.term_dict)
        self.use_idf = use_idf

    def dict_from_idf(self, idf_path):
        my_dict = Dictionary()
        freq_dict = {}
        with codecs.open(idf_path, mode="rb", encoding="utf8") as in_file:
            for line in in_file:
                splitted_line = line.split(" ")
                stemmed_word = self.stemmer.stem(splitted_line[0])
                frequency = int(splitted_line[1])
                if frequency < 5:
                    break
                else:
                    id_tuple = my_dict.doc2bow([stemmed_word], allow_update=True)
                    word_id = id_tuple[0][0]
                    freq_dict[word_id] = frequency + freq_dict.setdefault(word_id, 0)
        return my_dict, freq_dict

    def idf(self, term_id):
        return 1#math.log(self.max_freq/self.freq[term_id])

    def url_to_bow(self, url):
        print url
        tokenized_doc = http2tokenized_stemmed(url)
        bow_doc = self.term_dict.doc2bow(tokenized_doc)
        new_bow_doc = []
        for i in range(0, len(bow_doc)):
            new_bow_doc.append((bow_doc[i][0], bow_doc[i][1]*self.idf(bow_doc[i][0])))
        if self.use_idf:
            return new_bow_doc
        else:
            return bow_doc

    def corpus_from_urllist(self, url_list, label):
        urls = url_list
        docs_bow = [self.url_to_bow(url) for url in urls]
        labels = [label] * len(urls)
        return NewsCorpus(urls, labels, docs_bow, self.vocab_size)

def http2tokenized_stemmed(url):
    article = Article(url, language="pt")
    article.download()
    article.parse()
    full_text = 3 * (article.title + " ") + article.text
    return word_tokenize(full_text)


def urltxt2url_generator(in_file):
    with open(in_file, "rb") as read_file:
        for line in read_file:
            yield line


def feed2url_generator(feed):
    d = feedparser.parse(feed)
    for entry in d.entries:
        yield entry.link.split("url=")[1]


def rss_list2url_generator(in_file):
    with open(in_file, "rb") as read_file:
        for line in read_file:
            for url in feed2url_generator(line):
                yield url


def generator2txt(gen, out_file):
    with open(out_file, "wb") as write_file:
        for line in gen:
            write_file.write(line + "\n")


def batch_get_corpus(preprocessor, url_lists=None, url_lists_tags=None, rss_lists=None, rss_lists_tags=None):
    first = True
    if url_lists is not None:
        for i, url_list in enumerate(url_lists):
            effective_list = list(urltxt2url_generator(url_list))
            print "i" + str(i)
            if first:
                corpus = preprocessor.corpus_from_urllist(effective_list, url_lists_tags[i])
                first = False
            else:
                corpus.concatenate(preprocessor.corpus_from_urllist(effective_list, url_lists_tags[i]))
    if rss_lists is not None:
        for j, rss_list in enumerate(rss_lists):
            effective_list = list(rss_list2url_generator(rss_list))
            print "j" + str(j)
            if first:
                corpus = preprocessor.corpus_from_urllist(effective_list, rss_lists_tags[j])
                first = False
            else:
                corpus.concatenate(preprocessor.corpus_from_urllist(effective_list, rss_lists_tags[j]))
    return corpus


class BiGramPreProcessor(PreProcessor):
    def __init__(self, url_list=None, vocab=None):
        self.stemmer = RSLPStemmer()
        self.vectorizer = CountVectorizer(preprocessor=self.stemmer.stem, tokenizer=tokenizer_with_numeric,
                                          ngram_range=(1,2))
        if url_list is not None:
            self.fit_vocab(url_list)
        else:
            self.vectorizer.vocabulary_ = vocab
        self.vocab_size = len(self.vectorizer.vocabulary_)

    def fit_vocab(self, url_list):
        text_generator = url2text_generator(url_list)
        self.vectorizer.fit(text_generator)

    def url_to_bow(self, url):
        print url
        text_generator = url2text_generator([url])
        sparse_matrix = self.vectorizer.transform(text_generator)
        return [(sparse_matrix.indices[i], value) for i, value in enumerate(sparse_matrix.data)]

    def idf(self, term_id):
        return None

    def dict_from_idf(self, idf_path):
        return None


def tokenizer_with_numeric(text):
    return [replace_if_numeric(token) for token in word_tokenize(text)]

def replace_if_numeric(token):
    if token.isdigit():
        return "NNNnumericNNN"
    else:
        return token

def url2text_generator(url_list):
    for url in url_list:
        article = Article(url, language="pt")
        article.download()
        article.parse()
        full_text = (article.title + " ") + article.text
        yield full_text

def rss_list_labelize(classifier, preprocessor, rss_list):
    urls = list(rss_list2url_generator(rss_list))
    corpus = preprocessor.corpus_from_urllist(urls, "ND")
    tags = classifier.get_labels(corpus)
    corpus.tags = tags
    return corpus


def load_classifier(file_path):
    """Load Classifier from pickle"""
    with open(file_path, "rb") as in_file:
        classifier = pickle.load(in_file)
    return classifier


def f_measure(precision, recall):
    return 2*precision*recall/(precision+recall)


def load_classifier(file_path):
    with open(file_path, "rb") as in_file:
        classifier = pickle.load(in_file)
    return classifier


class NewsClassifier(object):

    def __init__(self, classifier):
        self.classifier = classifier

    def train(self, newscorpus):
        self.classifier.fit(newscorpus.sparse_matrix(), newscorpus.labels)

    def get_labels(self, newscorpus):
        return self.classifier.predict(newscorpus.sparse_matrix())

    def save(self, file_path):
        with open(file_path, "wb") as out_file:
            pickle.dump(self, out_file)

    def cross_validation(self, corpus, proportion, n_fold):
        accuracy = []
        precision = []
        recall = []
        for i in range(0, n_fold):
            train, test = corpus.random_split_train_test(proportion)
            self.train(train)
            accuracy.append(test.accuracy(self.get_labels(test)))
            precision_recall = test.precision_recall(self.get_labels(test))
            precision.append(precision_recall[0])
            recall.append(precision_recall[1])
        accuracy = np.array(accuracy)
        return np.mean(accuracy), np.mean(precision), np.mean(recall)


class NewsCorpus(object):
    """Corpus of news documents"""
    def __init__(self, urls, labels, docs_bow, vocab_size):
        if len(urls) != len(labels) != len(docs_bow):
            raise TypeError("Lists must have the same size")
        else:
            self.urls = np.array(urls)
            self.labels = np.array(labels)
            self.docs_bow = np.array(docs_bow)
        self.vocab_size = vocab_size

    @classmethod
    def copy(cls, corpus):
        return NewsCorpus(corpus.urls, corpus.labels, corpus.docs_bow, corpus.vocab_size)

    def sparse_matrix(self):
        indptr = [0]
        indices = []
        data = []
        for i, doc in enumerate(self.docs_bow):
            for term_tuple in doc:
                index = term_tuple[0]
                indices.append(index)
                data.append(term_tuple[1])
            indptr.append(len(indices))
        rows = len(self.docs_bow)
        columns = self.vocab_size
        sparse_matrix = csr_matrix((data, indices, indptr), shape=(rows, columns), dtype=float)
        return sparse_matrix

    def concatenate(self, newscorpus):
        if not self.vocab_size == newscorpus.vocab_size:
            raise TypeError("Vocabulary size is not the same!")
        else:
            self.urls = np.concatenate((self.urls, newscorpus.urls))
            self.docs_bow = np.concatenate((self.docs_bow, newscorpus.docs_bow))
            self.labels = np.concatenate((self.labels, newscorpus.labels))

    def _data_frame(self):
        df = pd.DataFrame()
        df["urls"] = self.urls
        df["labels"] = self.labels
        df["docs_bow"] = self.docs_bow
        df["vocab_size"] = [self.vocab_size] * len(self.labels)
        return df

    def save_csv(self, path):
        self._data_frame().to_csv(path, encoding="utf8")

    def save_pickle(self, path):
        self._data_frame().to_pickle(path)

    def random_split_train_test(self, proportion=0.5):
        train_markers = (np.random.rand(len(self.urls),) < proportion)
        train_corpus = NewsCorpus(self.urls[train_markers], self.labels[train_markers], self.docs_bow[train_markers],
                                  self.vocab_size)
        test_corpus = NewsCorpus(self.urls[train_markers == False], self.labels[train_markers == False],
                                 self.docs_bow[train_markers == False], self.vocab_size)
        return train_corpus, test_corpus

    def accuracy(self, labels):
        return float(np.sum(labels == self.labels))/len(self.labels)

    def precision_recall(self, labels, positive_label="D"):
        labels = np.array(labels)
        true_positives = np.sum((labels == self.labels) * (labels == positive_label))
        model_positives = np.sum(labels == positive_label)
        real_positives = np.sum(self.labels == positive_label)
        return float(true_positives)/model_positives , float(true_positives)/real_positives


class PreProcessor(object):

    def __init__(self, idf_path=IDF_PATH, use_idf=True):
        self.stemmer = RSLPStemmer()
        self.term_dict, self.freq = self.dict_from_idf(idf_path)
        self.max_freq = float(max(self.freq.values()))
        self.vocab_size = len(self.term_dict)
        self.use_idf = use_idf

    def dict_from_idf(self, idf_path):
        my_dict = Dictionary()
        freq_dict = {}
        with codecs.open(idf_path, mode="rb", encoding="utf8") as in_file:
            for line in in_file:
                splitted_line = line.split(" ")
                stemmed_word = self.stemmer.stem(splitted_line[0])
                frequency = int(splitted_line[1])
                if frequency < 5:
                    break
                else:
                    id_tuple = my_dict.doc2bow([stemmed_word], allow_update=True)
                    word_id = id_tuple[0][0]
                    freq_dict[word_id] = frequency + freq_dict.setdefault(word_id, 0)
        return my_dict, freq_dict

    def idf(self, term_id):
        return math.log(self.max_freq/self.freq[term_id])

    def url_to_bow(self, url):
        print url
        tokenized_doc = http2tokenized_stemmed(url)
        bow_doc = self.term_dict.doc2bow(tokenized_doc)
        new_bow_doc = []
        for i in range(0, len(bow_doc)):
            new_bow_doc.append((bow_doc[i][0], bow_doc[i][1]*self.idf(bow_doc[i][0])))
        if self.use_idf:
            return new_bow_doc
        else:
            return bow_doc

    def corpus_from_urllist(self, url_list, label):
        urls = url_list
        docs_bow = [self.url_to_bow(url) for url in urls]
        labels = [label] * len(urls)
        return NewsCorpus(urls, labels, docs_bow, self.vocab_size)

