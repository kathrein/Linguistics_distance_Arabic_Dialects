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
import os.path
import premodel
import numpy



def training_comp_lsi(comprarble_folder):
    counter = 0
    folders = [item for item in os.listdir(comprarble_folder) if os.path.isdir(os.path.join(comprarble_folder, item))]

    for folder in folders:
        folderpath = os.path.join(comprarble_folder, folder)
        texts = [[word for word in document.split()] for document in premodel.read_folder(folderpath)]
        if counter == 0 :
            dictionary = corpora.Dictionary(texts)

        else:
            dictionary.add_documents(texts)

    paths = [os.path.join(comprarble_folder,folders[0]+'/'),os.path.join(comprarble_folder,folders[1]+'/')]
    #print(list(premodel.read_set_of_file(paths[0] )))
    corpus = [dictionary.doc2bow(text) for text in [premodel.read_folder(paths[0])[0].split()]]
    corpus = corpus +  [dictionary.doc2bow(text) for text in [premodel.read_folder(paths[1])[0].split()]]
    #corpus = [dictionary.doc2bow(text) for text in list(premodel.read_set_of_file(paths[0] ))]
    #corpus = corpus + [dictionary.doc2bow(text) for text in list(premodel.read_set_of_file(paths[1]))]

    return dictionary,corpus

def training_phase(folder, dialect):
    counter = 0

    for file in os.listdir(folder):
        extension = os.path.splitext(file)[1]
        if extension == '.txt':
            filepath = os.path.join(folder, file)
            texts = premodel.read_text(filepath)
            if counter == 0:
                dictionary = corpora.Dictionary(texts)

            else:
                dictionary.add_documents(texts)

            #ct.add_documents
    #print(len(dictionary))

    #  Bag of words
    #dictionary = corpora.Dictionary(texts)
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    dictionary.save('parameters/' + dialect[0] + '_'+dialect[1]+'.dict')

    # - collect statistics about all tokens (trainig_data)
    corpus = [dictionary.doc2bow(text) for text in list(premodel.read_set_of_file(folder))]
    corpora.MmCorpus.serialize('parameters/' + dialect[0] + '_'+dialect[1] +'.mm',corpus)  # store to disk, for later use


    return dictionary, corpus

def build_ldamodel_training(folder, dialect):
    counter = 0

    for file in os.listdir(folder):
        extension = os.path.splitext(file)[1]
        if extension == '.txt':
            filepath = os.path.join(folder, file)
            texts = premodel.read_text(filepath)
            if counter == 0:
                dictionary = corpora.Dictionary(texts)

            else:
                dictionary.add_documents(texts)
            counter = counter +1

    # texts = [['bank', 'river', 'shore', 'water'],
    #          ['river', 'water', 'flow', 'fast', 'tree'],
    #          ['bank', 'water', 'fall', 'flow'],
    #          ['bank', 'bank', 'water', 'rain', 'river'],
    #          ['river', 'water', 'mud', 'tree'],
    #          ['money', 'transaction', 'bank', 'finance'],
    #          ['bank', 'borrow', 'money'],
    #          ['bank', 'finance'],
    #          ['finance', 'money', 'sell', 'bank'],
    #          ['borrow', 'sell'],
    #          ['bank', 'loan', 'sell']]

   # dictionary = Dictionary(texts)
    dictionary.save('parameters/LDAmodel_'+ dialect[1] + '.dict')
    corpus = [dictionary.doc2bow(text) for text in list(premodel.read_set_of_file(folder))]
    corpora.MmCorpus.serialize('parameters/LDAmodel_'+ dialect[1] + '.mm',corpus)  # store to disk, for later use

    return dictionary, corpus

def build_comparable_ldamodel_training(comp_folder, dialect):
        counter = 0
        folders = [comp_folder + dialect[0]+'/', comp_folder + dialect[1]+'/']
        for folder in folders:
            for file in os.listdir(folder):

                extension = os.path.splitext(file)[1]
                if extension == '.txt':

                    filepath = os.path.join(folder, file)
                    texts = premodel.read_text(filepath)
                    if counter == 0:
                        dictionary = corpora.Dictionary(texts)
                    else:
                        dictionary.add_documents(texts)
                    counter = counter +1

        dictionary.save('parameters/comp_LDAmodel_' + dialect[1] + '.dict')
        corpus = [dictionary.doc2bow(text) for text in list(premodel.read_set_of_file(folders[0]))]
        corpus = corpus + [dictionary.doc2bow(text) for text in list(premodel.read_set_of_file(folders[1]))]
        corpora.MmCorpus.serialize('parameters/comp_LDAmodel_' + dialect[1] + '.mm', corpus)  # store to disk, for later use

        return dictionary, corpus



def build_lsi_model(corpus,dictionary):
    """
    This process serves two goals:
    1. To bring out hidden structure in the corpus, discover relationships between words and use them to describe
    the documents in a new and (hopefully) more semantic way.
    2. To make the document representation more compact. This both improves efficiency (new representation consumes
    less resources) and efficacy (marginal data trends are ignored, noise-reduction).
    """
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=300)#100
    corpus_lsi = lsi[corpus]  # step 2 -- use the model to transform vectors
    # for doc in corpus_tfidf:
    # print(doc)
    return corpus_lsi, lsi




def build_ldamodel(corpus,dictionary):
    """
    This process serves two goals:

    1. To bring out hidden structure in the corpus, discover relationships between words and use them to describe
    the documents in a new and (hopefully) more semantic way.
    2. To make the document representation more compact. This both improves efficiency (new representation consumes
    less resources) and efficacy (marginal data trends are ignored, noise-reduction).
    """
    numpy.random.seed(1)  # setting random seed to get the same results each time.
    lda_model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=300)
    return lda_model



def compute_similarity(vec_model, corpus_model, dialect):
    # compute similarity
    # TO_DO: if the index avaliable then just load, else enter it for the first time
    index = similarities.MatrixSimilarity(corpus_model)
    index.save('parameters/' + dialect[0] + '_' + dialect[1] + '.index')
    index = similarities.MatrixSimilarity.load('parameters/' + dialect[0] + '_' + dialect[1] + '.index')
    sims = index[vec_model]  # perform a similarity query against the corpus
    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sorted
    #pprint(list(enumerate(sims)))  #print (document_number, document_similarity) 2-tuples

    pprint(sims)
    #print(len(sims))
    return sims


def test_corpus(document, Model, dictionary):
    # convert the test document / sentence to BOW model then compute the space mdoel
    vec_bow = dictionary.doc2bow(document.split())
    vec_tfidf = Model[vec_bow]  # convert the query to model space

    return vec_tfidf