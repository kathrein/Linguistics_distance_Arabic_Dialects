#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:01:30 2018

@author: xabuka
"""

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import svm_trainig as svm


if __name__ == '__main__':
    dialect = ['PAl','Syr']

    dictionary_memory_friendly, corpus_memory_friendly = svm.training_phase('/Users/xabuka/Desktop/mycorpus.txt', dialect)

    dictionary, corpus = svm.upload_data(dialect)

    corpus_tfidf,tfidf = svm.build_tfidf_model(corpus)

    # lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
    doc = "تكتفي من وفد بدها وجبات يارب"
    vec_bow = dictionary.doc2bow(doc.split())
    vec_tfidf = tfidf[vec_bow]  # convert the query to LSI space
    # pprint(vec_tfidf)

    similarity = svm.compute_similarity(vec_tfidf, corpus_tfidf,dialect)

    # compute the avg similarities cross the corpus
    """summation = 0
    for x,y in sims:
        if y != 0: 
            summation = summation+y
    print(len(sims))
    print(summation/len(sims))"""