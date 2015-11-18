__author__ = 'eduardo'
from nltk.tokenize import word_tokenize, sent_tokenize
import pycrfsuite
from newspaper import Article
import codecs
import re


class CRFCorpus(object):
    def __init__(self, documents):
        self.documents = documents

    @classmethod
    def from_urllist(cls, urls):
        documents = []
        for url in urls:
            article = Article(url)
            article.download()
            article.parse()
            if article.text == "" or article.text is None:
                continue
            else:
                documents.append(NewsDocument(url, article.title, article.text))
        return CRFCorpus(documents)

    def tag_from_nedict(self, nedict):
        for document in self.documents:
            for sentence in document.sentences:
                sentence.tag_from_nedict(nedict.properdict)

    def tag_from_tags(self, tags):
        i = 0
        for document in self.documents:
            for sentence in document.sentences:
                sentence.tags = tags[i]
                i += 1

    def tag_from_file(self, path):
        tags = []
        with open(path, "rb") as infile:
            tag = []
            for line in infile:
                if line == "\n":
                    tags.append(tag)
                    tag = []
                else:
                    tag.append(line.split("\n")[0])
        self.tag_from_tags(tags)


    def to_txt(self, path, only_d=False):
        first = True
        with codecs.open(path, "wb", encoding="utf8") as infile:
            for document in self.documents:
                for sentence in document.sentences:
                    if (only_d is False) or ("B-D" in sentence.tags or "I-D" in sentence.tags):
                        infile.write(sentence.crfsuite_format())
                        infile.write("\n\n")
            infile.write("\n\n")



class NewsDocument(object):
    def __init__(self, url="", title="", text="", description="", sentences=None):
        self.url = url
        self.title = title
        self.text = text
        self.description = description
        if sentences is None:
            self.sentences = self._generate_sentences()
        else:
            self.sentences = sentences

    def _generate_sentences(self):
        text_sentences = sent_tokenize(self.title + ". " + self.text)
        sentences = [Sentence(text_sentence) for text_sentence in text_sentences]
        return sentences

    def gen_ne(self):
        for sentence in self.sentences:
            ne = sentence.get_ne()
            if len(ne) > 0:
                yield (ne, sentence.text)

    def get_ne(self):
        return list(self.gen_ne())

    def tag_from_crfsuite(self, tagger):
        for sentence in self.sentences:
            sentence.tags = sentence.get_crfsuite_tags(tagger)

    @classmethod
    def reconstruct(cls, file_path):
        sentences = []
        with codecs.open(file_path, "rb", encoding="utf8") as infile:
            great_string = infile.read()
            for block in great_string.split("\n\n"):
                tokens = []
                tags = []
                for line in block.split("\n"):
                    if line not in ["", "\n"]:
                        tokens.append(line.split("word=")[1].split("\t")[0])
                        tags.append(line.split("\t")[0])
                sentences.append(Sentence(tokenized_sentence=tokens, tags=tags))
        return NewsDocument(sentences=sentences)


class Sentence(object):
    def __init__(self, sentence_text="", tokenized_sentence=None, tags=None):
        if tokenized_sentence is not None:
            self.tokenized_sentence = tokenized_sentence
            self.text = " ".join(tokenized_sentence)
        else:
            self.text = sentence_text
            self.tokenized_sentence = word_tokenize(self.text)
        self.features = [feature_detector(self.tokenized_sentence, i) for i in range(0, len(self.tokenized_sentence))]
        self.size = len(self.tokenized_sentence)
        if tags is None:
            self.tags = ["O"] * self.size
        else:
            if len(tags) != self.size:
                raise AttributeError("Tags list must have the same size as the list of tokens")
            else:
                self.tags = tags

    def crfsuite_line(self, index):
        feature_strings = [key+"="+unicode(value) for key, value in self.features[index].iteritems()]
        return self.tags[index] + "\t" + "\t".join(feature_strings)

    def crfsuite_format(self):
        return "\n".join([self.crfsuite_line(index) for index in range(0, self.size)])

    def itemsequence(self):
        return pycrfsuite.ItemSequence(self.features)

    def get_crfsuite_tags(self, crfsuite_tagger):
        tags = crfsuite_tagger.tag(self.features)
        return tags

    def tag_from_nedict(self, nedict):
        tags = get_tag(self.tokenized_sentence, nedict)
        self.tags = tags

    def get_ne(self):
        nes = []
        indexes_B = [i for i, tag in enumerate(self.tags) if tag == "B-D"]
        if len(indexes_B) > 0:
            for index in indexes_B:
                ne = self.tokenized_sentence[index]
                for j in range(index+1, self.size):
                    if self.tags[j] == "I-D":
                        ne += " " + self.tokenized_sentence[j]
                    else:
                        break
                nes.append(ne)
        return nes



def feature_detector(tokens, index):
    """
    Function used to extract features of a token in a sentence.
    If should be modified if any change in the feature set is required.

    :return: a dictionary with the token's features
    :rtype: dictionary
    :type tokens: a list of tuples
    :param tokens: list of (token, POStag) corresponding to a sentence
    :type index: int
    :param index: index of token in sentence whose features will be returned
    :param history: deprecated
    """

    word = tokens[index]
    sentence_len = len(tokens)

    if index == 0:
        prevword = prevprevword = "_*not*_*here*_"
        prevshape = prevtag = prevprevtag = "_*not*_*here*_"
    elif index == 1:
        prevword = tokens[index-1].lower()
        prevprevword = "_*not*_*here*_"
        prevshape = prevprevtag = "_*not*_*here*_"
    else:
        prevword = tokens[index-1].lower()
        prevprevword = tokens[index-2].lower()
        prevshape = _shape(prevword)
    if index >= len(tokens)-1:
        nextword = nextnextword = "_*not*_*here*_"
    elif index == len(tokens)-2:
        nextword = tokens[index+1].lower()
        nextnextword = "_*not*_*here*_"
    else:
        nextword = tokens[index+1].lower()
        nextnextword = tokens[index+2].lower()

    features_strings = {
        ### Tokenizing related features:
        'word': "word",
        'wordlower': "word.lower()",
        'sentencelen' : "str(sentence_len > 10)",
        'prevword': "prevword",
        'nextword': "nextword",
        'prevprevword': "prevprevword",
        'nextnextword': "nextnextword",
        'isnextword': "str(nextword != '_*not*_*here*_')",

        ### Affixes related features:
        'prefix2': "word[:2].lower()",
        'lastprefix3': "prevword[:3].lower()",
        'nextprefix3': "nextword[:3].lower()",
        'suffix2': "word[-2:].lower()",
        'lastsuffix3': "prevword[-3:].lower()",
        'nextsuffix3': "nextword[-3:].lower()",
        'prefix3': "word[:3].lower()",
        'suffix3': "word[-3:].lower()",

        ### Morfologically related features:
        'shape': "_shape(word)",
        'prevshape': "_shape(prevword)",
        'nextshape': "_shape(nextword)",
        'prevprevshape': "_shape(prevprevword)",
        'nextnextshape': "_shape(nextnextword)",
        'wordlen': "len(word)",
        'qwordlen': "_q_len(len(word))",

    }

    feature_set = {"word", "wordlower", "prevword", "nextword",
                   "prefix3", "suffix3", "shape", "prevshape", "nextshape"}

    features = {}
    for key in features_strings:
        if key in feature_set:
            features[key] = eval(features_strings[key])
    return features


def _q_len(size):
    """ This is qualitative mesure of length, used to strongly reduce the
    possible values the length related feature may take. The convention used is
    starting from 1 for the smaller ones, then slowly increasing to bigger ones
    """
    qlen = 0
    if size <= 3:
        qlen = 1
    elif size > 3 and size <= 5:
        qlen = 2
    elif size > 5 and size <= 7:
        qlen = 3
    elif size > 7 and size <= 10:
        qlen = 4
    elif size > 10 and size <= 13:
        qlen = 5
    elif size > 13:
        qlen = 6
    else:
        print "Strange value found on _q_len method: returning zero instead!"
    return qlen


def _shape(word):
    """ This function returns a characteristic of the word, in a sense it tells
    if it is made solely of numbers, if it's upper or lower case, punctuations...
    We identifiy:
    0-None   1-number   2-punct   3-upcase   4-downcase   5-mixedcase   6-other
    """
    if word == None:
        shape_case = "0"   # ''
    elif re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word):
        shape_case = "1"   # 'number'
    elif re.match('\W+$', word):
        shape_case = "2"   # 'punct'
    elif re.match('[A-Z][a-z]+$', word):
        shape_case = "3"   # 'upcase'
    elif re.match('[a-z]+$', word):
        shape_case = "4"   # 'downcase'
    elif re.match('\w+$', word):
        shape_case = "5"   # 'mixedcase'
    else:
        shape_case = "6"   # 'other'
    return shape_case


def get_tag(sentence, my_dict):
    nb_tokens = len(sentence)
    tag = ["O" for token in sentence]
    __, __ = tag_when_possible(sentence, my_dict, tag, nb_tokens)
    return tag


def tag_when_possible(phrase, ne_tuples_dict, tags, original_nb_tokens_in_phrase):
    max_n_gram_size = 20
    tagged_elements = []
    tagged_elements_index = []
    for i in range(0, max_n_gram_size):
        n_gram_size = max_n_gram_size - i
        if n_gram_size <= original_nb_tokens_in_phrase:
            for j in range(0, original_nb_tokens_in_phrase + 1 - n_gram_size):
                if available_n_gram(tags, j, j + n_gram_size):
                    searchable_tuple = tuple(phrase[j:j+n_gram_size])
                    if searchable_tuple in ne_tuples_dict:
                        tag_type = ne_tuples_dict[searchable_tuple]
                        create_tags(tags, j, j+n_gram_size, tag_type)
                        tagged_elements.append(searchable_tuple)
                        tagged_elements_index.append(j)
    return tagged_elements, tagged_elements_index


def available_n_gram(tags, start_index, end_index):
    test_list = ["O"]*(end_index - start_index)
    if tags[start_index:end_index] == test_list:
        return True
    else:
        return False


def create_tags(tags, start_index, end_index, tag_type):
    tags[start_index] = "B-" + tag_type
    for i in range(start_index+1, end_index):
        tags[i] = "I-" + tag_type


def csv2nedict(csv_path):
    examples = pd.read_table(csv_path, sep=",", encoding="utf8")
    nedict = NEDict()
    for __, row in examples.iterrows():
        nedict.add_ne(word_tokenize(row["sentence"]), row["type"])
    return nedict


def txt2nedict(path):
    nedict = NEDict()
    with codecs.open(path, "rb", encoding="utf8") as infile:
        for sentence in infile:
            real_sentence = sentence.split("\n")[0]
            nedict.add_ne(word_tokenize(real_sentence), "D")
    return nedict


class NEDict(object):
    """
    Named Entity dictionary
    """
    def __init__(self, description=None):
        self.properdict = {}
        if description == None:
            self.description = ""

    def add_ne(self, tokens, type):
        """
        Add named entities (list of tokens) to the dictionary.
        """
        if tokens in [[], None]:
            raise TypeError("Tokens must be a list of strings")
        else:
            self.properdict[tuple(tokens)] = type