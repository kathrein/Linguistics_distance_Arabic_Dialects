#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:01:30 2018

@author: xabuka
"""

# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#command
#python3 distance.py -c clean_data/padic  -m vsm -t pc -f msa  -s msa

import lsi_training as models
import psvm_training as svm
import argparse
import premodel
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
import sys, os

parser = argparse.ArgumentParser()
parser.add_argument("--method_name", "-m", type=str, help='vsm: vector space model, lsi: LSI model, d: distance, com: common words', required=True)

parser.add_argument("--corpus_folder", "-c", type=str, help='the folder contains the corpus files.', required=True)
parser.add_argument("--dialect_one", "-f", type=str, help='the train dialect.', required=False)
parser.add_argument("--dialect_two", "-s", type=str, help='the test dialect text file.', required=True)
parser.add_argument("--corpus_type", "-t", type=str, help='pc: parallel corpus, cc: comparable corpus, up: un-parallel corpus', required=True)

def comparable_corpus_distance(folder,dialect):
    dictionary, corpus = models.build_comparable_ldamodel_training(folder, dialect)
    lda_model = models.build_ldamodel(corpus, dictionary)
    folders = [folder + dialect[0] + '/', folder + dialect[1] + '/']

    Hellinger_summation = 0
    Jaaccard_summation = 0
    for file in os.listdir(folders[0]):
        try:

            extension = os.path.splitext(file)[1]
            if extension == '.txt':
                first_filepath = os.path.join(folders[0], file)
                second_filepath = os.path.join(folders[1], file)

                with open(first_filepath, encoding='utf-8') as f:  # we can define file_name
                    first_documents = f.read()
                first_dialect = [word for word in first_documents.split()]

                # print(first_dialect)
                with open(second_filepath, encoding='utf-8') as f:  # we can define file_name
                    second_documents = f.read()
                second_dialect = [word for word in second_documents.split()]

                # print(second_dialect)
                bow_first_dialect = lda_model.id2word.doc2bow(first_dialect)
                bow_second_dialect = lda_model.id2word.doc2bow(second_dialect)
                # print(bow_first_dialect)
                # we can now get the LDA topic distributions for these
                lda_bow_first_dialect = lda_model[bow_first_dialect]
                lda_bow_second_dialect = lda_model[bow_second_dialect]

                # print(lda_bow_first_dialect)

                print('Hellinger distance between 1 and 2 ')
                print(hellinger(lda_bow_first_dialect, lda_bow_second_dialect))
                Hellinger_summation = Hellinger_summation + hellinger(lda_bow_first_dialect, lda_bow_second_dialect)
                print('Jcard Distance')
                print(jaccard(bow_first_dialect, bow_second_dialect))
                Jaaccard_summation = Jaaccard_summation + jaccard(bow_first_dialect, bow_second_dialect)
                # sys.exit()

        except:
            pass

    print('total hellinger = ', Hellinger_summation / 10197)
    print('Total JC = ', Jaaccard_summation / 10197)

def corpus_distance(folder,dialect,corpus_files):
    dictionary, corpus = models.build_ldamodel_training(folder, dialect)

    # dictionary, corpus = premodel.upload_data(dialect)

    # print('here', len(corpus))
    lda_model = models.build_ldamodel(corpus, dictionary)

    # now we add the two dialects to test the distance betwen them
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
    print(jaccard(bow_first_dialect, bow_second_dialect))

    print('kullback_leibler between 1 to 2')
    # print(kullback_leibler(lda_bow_first_dialect, lda_bow_second_dialect))

    print('kullback_leibler between 2 to 1')
    # print(kullback_leibler(lda_bow_second_dialect,lda_bow_first_dialect))


def corpus_distance(folder, dialect, corpus_files):
    dictionary, corpus = models.build_ldamodel_training(folder, dialect)

    # dictionary, corpus = premodel.upload_data(dialect)

    # print('here', len(corpus))
    lda_model = models.build_ldamodel(corpus, dictionary)

    # now we add the two dialects to test the distance betwen them
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
    print(jaccard(bow_first_dialect, bow_second_dialect))

    print('kullback_leibler between 1 to 2')
    # print(kullback_leibler(lda_bow_first_dialect, lda_bow_second_dialect))

    print('kullback_leibler between 2 to 1')
    # print(kullback_leibler(lda_bow_second_dialect,lda_bow_first_dialect))

# for parallel and comparable corpus
def compute_vsm(train_document, test_document,dialect):
    summation = 0
    for i, (a, b) in enumerate(zip(train_document, test_document)):
        dict = [a.split(), b.split()]
        dictionary = svm.build_dictionary_phase( dict, dialect)
        if a == '':
            continue
        corpus = svm.build_corpus_phase(dictionary, [a.split(), []], dialect)
        corpus_tfidf, tfidf = svm.build_tfidf_model(corpus)
        vec_tfidf = svm.test_corpus(''.join(b), tfidf, dictionary)
        similarity = svm.compute_similarity(vec_tfidf, corpus_tfidf, dialect)
        summation = summation + sum(y for _, y in similarity if y > 0)

    print('Number of document in {0} = {1}'.format(dialect[0], len(train_document)))
    print('Number of document in {0} = {1}'.format(dialect[1], len(test_document)))
    print('The avg similarity between {0} and {1} is {2} '.format(dialect[0], dialect[1], summation / len(train_document)))

def compute_common_words(corpus_files):
    #jaccard similarity
    first_dialect = set(open(corpus_files[0], encoding='utf-8').read().split())
    second_dialect = set(open(corpus_files[1], encoding='utf-8').read().split())

    # if len(first_dialect) - len(second_dialect) > 200:
    #     return -1

    intersection = first_dialect.intersection(second_dialect)
    if (len(first_dialect) + len(second_dialect)) == 0:
        return -1
    else :
        cw = len(intersection) / (len(first_dialect) + len(second_dialect))
        print('overlapping between {0} and {1} = {2}'.format(corpus_files[0],corpus_files[1],cw))
        #print(1 - cw)

        return cw


def comparable_comon_words(folder, dialect):
    overlap = 0
    number_of_files = 0
    compute_common = 0
    folders = [folder + dialect[0] + '/', folder + dialect[1] + '/']
    for file in os.listdir(folders[0]):
        extension = os.path.splitext(file)[1]
        if extension == '.txt':
            first_filepath = os.path.join(folders[0], file)
            second_filepath = os.path.join(folders[1], file)
            corpus_files = [first_filepath,second_filepath]
            compute_common = compute_common_words(corpus_files)
            if compute_common > 0 :
                overlap = overlap + compute_common
                number_of_files = number_of_files +1
    print('overall similarity for comparable corpus = {0}, and length = {1}'.format(overlap/number_of_files, number_of_files))





def compute_unparallel_vsm(corpus_files,dialect):
    dictionary, corpus = svm.training_phase(corpus_files, dialect)
    #dictionary, corpus = svm.upload_data(dialect)
    # print('here', len(corpus))
    corpus_tfidf, tfidf = svm.build_tfidf_model(corpus)

    summation = 0
    with open(corpus_files[1], encoding='utf-8') as f:  # we can define file_name
        documents = f.read().splitlines()
        # print(type(documents))
        # for x in svm.read_full_text(corpus_files[1]):
        #   print(type(' '.join(x)))
    # print(len(documents))
    # for document in documents:
    for document in svm.read_full_text(corpus_files[1]):
        # print(document)

        vec_tfidf = svm.test_corpus(' '.join(document), tfidf, dictionary)
        # pprint(vec_tfidf)
        similarity = svm.compute_similarity(vec_tfidf, corpus_tfidf, dialect)

        # compute the avg similarities cross the corpus
        summation = summation + sum(y for _, y in similarity if y > 0)
    # print(summation)
    print('Number of document in {0} = {1}'.format(dialect[0], len(corpus)))
    print('Number of document in {0} = {1}'.format(dialect[1], len(svm.read_full_text(corpus_files[1]))))
    print('The avg similarity between {0} and {1} is {2} '.format(dialect[0], dialect[1], summation))

def compute_lsi(folder,dialect,corpus_files):
    dictionary, corpus = models.training_phase(folder, dialect)
    corpus_lsi, lsi = models.build_lsi_model(corpus, dictionary)

    summation = 0
    with open(corpus_files[1], encoding='utf-8') as f:  # we can define file_name
        documents = f.read().splitlines()
    count = 0
    for document in premodel.read_full_text(corpus_files[1]):
        vec_lsi = models.test_corpus(' '.join(document), lsi, dictionary)
        models.compute_similarity(vec_lsi, corpus_lsi, dialect)


    print('Number of document in {0} = {1}'.format(dialect[0], len(corpus)))
    print('Number of document in {0} = {1}'.format(dialect[1], len(premodel.read_text(corpus_files[1]))))


def main():
    args = parser.parse_args()
    # dialect = ['pa','sy']
    if args.dialect_one is None:
        args.dialect_one = 'LSI'

    dialect = [args.dialect_one, args.dialect_two]
    folder = args.corpus_folder + '/'
    corpus_files = [folder + dialect[0] + '.txt', folder + dialect[1] + '.txt']

    comparable_folders = [folder + args.dialect_one + '/', folder + args.dialect_two + '/']#used for comparable corpus

    if args.method_name.lower() == 'd': # compute Hellinger distance
        if args.corpus_type.lower() == 'cc':
            comparable_corpus_distance(folder,dialect)
        else:
            corpus_distance(folder,dialect,corpus_files)

    elif args.method_name.lower() =='vsm': # compute vector space model
        if args.corpus_type.lower() == 'pc':
            with open(corpus_files[0], encoding='utf-8') as f_train:
                train_document = f_train.read().splitlines()

            with open(corpus_files[1], encoding='utf-8') as f_test:
                test_document = f_test.read().splitlines()
            compute_vsm(train_document, test_document, dialect)
        elif args.corpus_type.lower() == 'cc':
            train_document = list(premodel.read_folder(comparable_folders[0]))
            test_document = list(premodel.read_folder(comparable_folders[1]))
            compute_vsm(train_document, test_document, dialect)
        elif args.corpus_type.lower() == 'up':
            compute_unparallel_vsm(corpus_files,dialect)
    elif args.method_name.lower() == 'lsi':
        if args.corpus_type.lower() == 'cc':
            print()
        else:
            compute_lsi(folder,dialect,corpus_files)
    elif args.method_name.lower() == 'com':
        if args.corpus_type.lower() =='cc':
            comparable_comon_words(folder,dialect)
        else :
            compute_common_words(corpus_files)






if __name__ == '__main__':
    main()
