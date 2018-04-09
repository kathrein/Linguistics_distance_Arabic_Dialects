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
    dictionary = corpora.Dictionary(texts)
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    dictionary.save('parameters/' + dialect[0] + '_' + dialect[1] + '.dict')

    # - collect statistics about all tokens (trainig_data)
    corpus_memory_friendly = [dictionary.doc2bow(text) for text in read_text(corpus_files[0])]#second_text]
    corpora.MmCorpus.serialize('parameters/' + dialect[0] + '_' + dialect[1] + '.mm',
                               corpus_memory_friendly)  # store to disk, for later use



    return dictionary, corpus_memory_friendly


def upload_data(dialect):
    # 2 Transformation
    if (os.path.exists('parameters/' + dialect[0] + '_' + dialect[1] + '.dict')):
        dictionary = corpora.Dictionary.load('parameters/' + dialect[0] + '_' + dialect[1] + '.dict')
        corpus = corpora.MmCorpus('parameters/' + dialect[0] + '_' + dialect[1] + '.mm')
        print("True files generated from first tutorial")
    else:
        print("Please run first tutorial to generate data set")

    return dictionary, corpus


def build_lsi_model(corpus,dictionary):
    """
    This process serves two goals:

    1. To bring out hidden structure in the corpus, discover relationships between words and use them to describe
    the documents in a new and (hopefully) more semantic way.
    2. To make the document representation more compact. This both improves efficiency (new representation consumes
    less resources) and efficacy (marginal data trends are ignored, noise-reduction).
    """

    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)
    corpus_lsi = lsi[corpus]  # step 2 -- use the model to transform vectors
    # for doc in corpus_tfidf:
    # print(doc)




    return corpus_lsi, lsi


def compute_similarity(vec_model, corpus_model, dialect):
    # compute similarity
    # TO_DO: if the index avaliable then just load, else enter it for the first time
    index = similarities.MatrixSimilarity(corpus_model)
    index.save('parameters/' + dialect[0] + '_' + dialect[1] + '.index')
    index = similarities.MatrixSimilarity.load('parameters/' + dialect[0] + '_' + dialect[1] + '.index')
    sims = index[vec_model]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sorted
    #pprint(list(enumerate(sims)))  #print (document_number, document_similarity) 2-tuples

    #pprint(sims)
    #print(len(sims))
    return sims


def test_corpus(document, Model, dictionary):
    # convert the test document / sentence to BOW model then compute the space mdoel
    vec_bow = dictionary.doc2bow(document.split())
    vec_tfidf = Model[vec_bow]  # convert the query to model space

    return vec_tfidf