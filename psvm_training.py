#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:01:30 2018

@author: xabuka
"""

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


from gensim import corpora, models, similarities
from pprint import pprint  # pretty-printer
# from six import iteritems #not load dictionary into memory
import os.path


class MyCorpus(object):
    def __init__(self, corpus_file, dictionary):
        """
        Checks if a dictionary has been given as a parameter.
        If no dictionary has been given, it creates one and saves it in the disk.
        """
        self.file_name = corpus_file
        if not dictionary is None:
            self.dictionary = dictionary
            # else:
            #     self.dictionary = dictionary

    def __iter__(self):
        for line in open(self.file_name, encoding='utf-8'):
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.split())


def read_text(corpus_file):
    with open(corpus_file, encoding='utf-8') as f:  # we can define file_name
        documents = f.read().splitlines()
    texts = [[word for word in document.split()] for document in documents]

    return texts


def read_full_text(corpus_file):
    """
    read the file as one sentence
    :param corpus_file:
    :return:
    """
    with open(corpus_file, encoding='utf-8') as f:  # we can define file_name
        documents = f.read()
    texts = [documents.split(), []]
    # print (texts)
    return texts


def training_phase(corpus_files, dialect):
    counter = 0

    for corpus_file in corpus_files:
        texts = read_text(corpus_file)

        if counter == 0:
            # print(counter)
            temp = texts
            counter = 1
        else:
            second_text = texts
    texts = temp + texts
    # print(len(texts))


    #  Bag of words
    # dictionary = corpora.Dictionary(line.split() for line in open(corpus_files[0], encoding='utf-8'))
    dictionary = corpora.Dictionary(texts)
    # print(dictionary.token2id)
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    dictionary.save('parameters/' + dialect[0] + '_' + dialect[1] + '.dict')
    # pprint(dictionary.token2id)

    # - collect statistics about all tokens (trainig_data)
    corpus_memory_friendly = [dictionary.doc2bow(text) for text in read_full_text(corpus_files[0])]  # second_text]
    # corpus_memory_friendly = MyCorpus(corpus_files[0],dictionary)  # doesn't load the corpus into memory!
    corpora.MmCorpus.serialize('parameters/' + dialect[0] + '_' + dialect[1] + '.mm',
                               corpus_memory_friendly)  # store to disk, for later use
    # for vector in corpus_memory_friendly:  # load one vector into memory at a time
    #   pprint(vector)



    return dictionary, corpus_memory_friendly


def upload_data(dialect):
    # 2 Transformation
    if (os.path.exists('parameters/PSVM_' + dialect[0] + '_' + dialect[1] + '.dict')):
        dictionary = corpora.Dictionary.load('parameters/PSVM_' + dialect[0] + '_' + dialect[1] + '.dict')
        corpus = corpora.MmCorpus('parameters/PSVM_' + dialect[0] + '_' + dialect[1] + '.mm')
        print("True files generated from first tutorial")
    else:
        print("Please run first tutorial to generate data set")

    return dictionary, corpus


def build_tfidf_model(corpus):
    """
    This process serves two goals:

    1. To bring out hidden structure in the corpus, discover relationships between words and use them to describe
    the documents in a new and (hopefully) more semantic way.
    2. To make the document representation more compact. This both improves efficiency (new representation consumes
    less resources) and efficacy (marginal data trends are ignored, noise-reduction).
    """

    tfidf = models.TfidfModel(corpus)  # ,normalize = True)  # step 1 -- initialize a model
    # It can also optionally normalize the resulting vectors to (Euclidean) unit length.
    # print(tfidf)
    corpus_tfidf = tfidf[corpus]  # step 2 -- use the model to transform vectors
    # for doc in corpus_tfidf:
    # print(doc)
    return corpus_tfidf, tfidf


def build_dictionary_phase(dict, dialect):#(corpus_files,dict, dialect):

    """
    build a dictionary for the two dialects
    :param corpus_file: list contains the name of the dialect files
    :param dialect: list of dialects name
    :return: dictionary (id,token)
    """

    #counter = 0
    #for corpus_file in corpus_files:
        # texts = dict
        # s = read_text(corpus_file)
        # print(texts)
        # print(s)
        # if counter == 0:
        #     dictionary = corpora.Dictionary(texts)
        #
        # else:
        #     dictionary.add_documents(texts)
        # counter = counter+1


    dictionary = corpora.Dictionary(dict)
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    dictionary.save('parameters/PSVM_' + dialect[0] + '_' + dialect[1] + '.dict')
    return dictionary


def build_corpus_phase (dictionary, document,dialect):
    """
    build a corpus for a document (sentence) cuz of parallel copurs
    :param document: the corpus sentence
    :param dictionary : dicitonary for all words in the files
    :param dialect: list of 2 dialects name
    :return: corpus (word, occurrence)
    """

    # - collect statistics about all tokens (trainig_data)
    corpus = [dictionary.doc2bow(text) for text in document]  # second_text]
    # corpus_memory_friendly = MyCorpus(corpus_files[0],dictionary)  # doesn't load the corpus into memory!
    corpora.MmCorpus.serialize('parameters/PSVM_' + dialect[0] + '_' + dialect[1] + '.mm',
                               corpus)  # store to disk, for later use

    return corpus



def compute_similarity(vec_tfidf, corpus_model, dialect):
    # compute similarity
    # TO_DO: if the index avaliable then just load, else enter it for the first time
    index = similarities.MatrixSimilarity(corpus_model)
    index.save('parameters/PSVM_' + dialect[0] + '_' + dialect[1] + '.index')
    index = similarities.MatrixSimilarity.load('parameters/PSVM_' + dialect[0] + '_' + dialect[1] + '.index')
    sims = index[vec_tfidf]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sorted
    # pprint(list(enumerate(sims)))  print (document_number, document_similarity) 2-tuples

    # pprint(sims)
    return sims


def test_corpus(document, Model, dictionary):
    # convert the test document / sentence to BOW model then compute the space mdoel
    vec_bow = dictionary.doc2bow(document.split())
    vec_tfidf = Model[vec_bow]  # convert the query to model space

    return vec_tfidf