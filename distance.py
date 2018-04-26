#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:01:30 2018

@author: xabuka
"""

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import lsi_training as models
import argparse
import premodel
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full


parser = argparse.ArgumentParser()
parser.add_argument("--corpus_folder", "-c", type=str, help='the folder contains the corpus files.', required=True)

parser.add_argument("--dialect_one", "-f", type=str, help='the train dialect.', required=True)

parser.add_argument("--dialect_two", "-s", type=str, help='the test dialect text file.', required=True)


if __name__ == '__main__':
    args = parser.parse_args()
    # dialect = ['pa','sy']
    dialect = [args.dialect_one, args.dialect_two]
    folder = args.corpus_folder + '/'
    corpus_files = [folder + dialect[0] + '.txt', folder + dialect[1] + '.txt']

    dictionary, corpus = models.build_ldamodel_training(folder, dialect)

    #dictionary, corpus = premodel.upload_data(dialect)

    # print('here', len(corpus))
    lda_model = models.build_ldamodel(corpus,dictionary)

    #now we add the two dialects to test the distance betwen them
    with open(corpus_files[0], encoding='utf-8') as f:  # we can define file_name
        first_documents = f.read()
    first_dialect = [word for word in first_documents.split()]

    with open(corpus_files[1], encoding='utf-8') as f:  # we can define file_name
        second_documents = f.read()
    second_dialect = [word for word in second_documents.split()]
    # now let's make these into a bag of words format

    bow_first_dialect = lda_model.id2word.doc2bow(first_dialect)
    bow_second_dialect = lda_model.id2word.doc2bow(second_dialect)

    # we can now get the LDA topic distributions for these
    lda_bow_first_dialect = lda_model[bow_first_dialect]
    lda_bow_second_dialect = lda_model[bow_second_dialect]


    print('Hellinger distance between 1 and 2 ')
    print(hellinger(lda_bow_first_dialect, lda_bow_second_dialect))

    print('Jcard Distance')
    print(jaccard(bow_first_dialect,bow_second_dialect))


    print('kullback_leibler between 1 to 2')
   # print(kullback_leibler(lda_bow_first_dialect, lda_bow_second_dialect))

    print('kullback_leibler between 2 to 1')
   # print(kullback_leibler(lda_bow_second_dialect,lda_bow_first_dialect))

    