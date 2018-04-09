#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:01:30 2018

@author: xabuka
"""

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import lsi_training as lsi_model
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dialect_one", "-f", type=str, help='the first dialect text file.', required=True)

parser.add_argument("--dialect_two", "-s", type=str, help='the second dialect text file.', required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    # dialect = ['pa','sy']
    dialect = [args.dialect_one, args.dialect_two]
    corpus_files = ['clean_data/' + dialect[0] + '.txt', 'clean_data/' + dialect[1] + '.txt']
    dictionary_memory_friendly, corpus_memory_friendly = lsi_model.training_phase(corpus_files, dialect)

    dictionary, corpus = lsi_model.upload_data(dialect)

    # print('here', len(corpus))
    corpus_lsi, lsi = lsi_model.build_lsi_model(corpus,dictionary)

    summation = 0
    with open(corpus_files[1], encoding='utf-8') as f:  # we can define file_name
        documents = f.read().splitlines()
    print(type(documents))

    for document in lsi_model.read_full_text(corpus_files[1]):

        vec_lsi = lsi_model.test_corpus(' '.join(document), lsi, dictionary)
        # pprint(vec_tfidf)
        similarity = lsi_model.compute_similarity(vec_lsi, corpus_lsi, dialect)

        # compute the avg similarities cross the corpus
        summation = summation + sum(y for _, y in similarity if y > 0)
    # print(summation)
    print('Number of document in {0} = {1}'.format(dialect[0], len(corpus)))
    print('Number of document in {0} = {1}'.format(dialect[1], len(lsi_model.read_text(corpus_files[1]))))
    print('The avg similarity between {0} and {1} is {2} '.format(dialect[0], dialect[1], summation/((len(corpus))*(len(lsi_model.read_text(corpus_files[1]))))))