#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:01:30 2018

@author: xabuka
"""

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# command: python3 psvm.py -f msa -s pa -t pc

import psvm_training as svm
from pprint import pprint
import argparse
import premodel
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--corpus_folder", "-c", type=str, help='the folder contains the corpus files.', required=True)

parser.add_argument("--dialect_one", "-f", type=str, help='the first dialect text file.', required=True)

parser.add_argument("--dialect_two", "-s", type=str, help='the second dialect text file.', required=True)

parser.add_argument("--corpus_type", "-t", type=str, help='pc: parallel corpus, cc: comparable corpus', required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    # dialect = ['pa','sy']
    dialect = [args.dialect_one, args.dialect_two]
    corpus_files = ['clean_data/nizar/' + dialect[0] + '.txt', 'clean_data/nizar/' + dialect[1] + '.txt']


    #dictionary = svm.build_dictionary_phase(corpus_files, dialect)
    if args.corpus_type.lower() == 'pc':

        with open(corpus_files[0], encoding='utf-8') as f_train:
            train_document = f_train.read().splitlines()

        with open(corpus_files[1], encoding='utf-8') as f_test:
            test_document = f_test.read().splitlines()

    elif args.corpus_type.lower() == 'cc':
        folders = ['clean_data/comparable/'+args.dialect_one+'/','clean_data/comparable/'+args.dialect_two+ '/']
        train_document = list(premodel.read_folder(folders[0]))
        test_document = list(premodel.read_folder(folders[1]))
        print(len(train_document))
        #pprint((test_document))
        #sys.exit()

    summation = 0
    for i, (a, b) in enumerate(zip(train_document, test_document)):
        dict = [a.split(),b.split()]
        dictionary = svm.build_dictionary_phase(corpus_files,dict, dialect)
        #xprint(dictionary.token2id)
        #print(a)
        if a == '':
            continue
        corpus = svm.build_corpus_phase(dictionary, [a.split(), []],dialect)
        #for vector in corpus:  # load one vector into memory at a time
           # pprint(vector)
        corpus_tfidf, tfidf = svm.build_tfidf_model(corpus)
        #print(''.join(b))
        vec_tfidf = svm.test_corpus(''.join(b), tfidf, dictionary)
        #print(vec_tfidf)
        similarity = svm.compute_similarity(vec_tfidf, corpus_tfidf, dialect)
        #print(similarity)
        summation = summation + sum(y for _, y in similarity if y > 0)
        #break
        # for x, y in similarity:
        #     if y < 0.3:
        #         print(a)
        #         print(b)

    print('Number of document in {0} = {1}'.format(dialect[0], len(train_document)))
    print('Number of document in {0} = {1}'.format(dialect[1], len(test_document)))
    print('The avg similarity between {0} and {1} is {2} '.format(dialect[0], dialect[1], summation/len(train_document)))  # /(len(corpus))))