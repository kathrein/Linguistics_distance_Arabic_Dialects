#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:01:30 2018

@author: xabuka
"""

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#command :  python3 LSI.py -c clean_data/padic -s pa

import lsi_training as lsi_model
import argparse
import premodel
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_folder", "-c", type=str, help='the folder contains the corpus files.', required=True)

parser.add_argument("--dialect_one", "-f", type=str, help='the train dialect.', required=False)

parser.add_argument("--dialect_two", "-s", type=str, help='the test dialect text file.', required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    # dialect = ['pa','sy']
    dialect = ['LSI', args.dialect_two]
    folder = args.corpus_folder + '/' # clean_data/Nizar + /
    corpus_files = [folder+ dialect[0] + '.txt', folder + dialect[1] + '.txt']

    dictionary_memory_friendly, corpus_memory_friendly = lsi_model.training_phase(folder, dialect)

    dictionary, corpus = premodel.upload_data(dialect)
    corpus_lsi, lsi = lsi_model.build_lsi_model(corpus,dictionary)

    summation = 0
    with open(corpus_files[1], encoding='utf-8') as f:  # we can define file_name
        documents = f.read().splitlines()
    count = 0
    for document in premodel.read_full_text(corpus_files[1]):

        vec_lsi = lsi_model.test_corpus(' '.join(document), lsi, dictionary)
        similarity = lsi_model.compute_similarity(vec_lsi, corpus_lsi, dialect)

        # compute the avg similarities cross the corpus
        #summation = summation + sum(y for _, y in similarity if y > 0)
        # for x,y in similarity :
        #     if not (y == 0) :
        #         count = count+1

    # print(count)
    # if count == 0 : # if there is only one file
    #     count = 1
    print('Number of document in {0} = {1}'.format(dialect[0], len(corpus)))
    print('Number of document in {0} = {1}'.format(dialect[1], len(premodel.read_text(corpus_files[1]))))
    #print('The avg similarity between {0} and {1} is {2} '.format(dialect[0], dialect[1], summation/count))