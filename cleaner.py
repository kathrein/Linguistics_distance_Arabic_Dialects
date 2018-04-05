#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:05:30 2018

@author: xabuka
"""

import sys
import string
from alphabet_detector import AlphabetDetector


def remove_punctuations(text):
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،'
    return ''.join(ch for ch in text if ch not in punctuations)


def remove_punctuation(s):
    translator = str.maketrans('', '', string.punctuation + "،")
    return s.translate(translator)


def keep_only_arabic(words):
    ad = AlphabetDetector()
    tokens = [token for token in words if ad.is_arabic(token)]
    tweet = ' '.join(tokens)
    return tweet


def remove_digits(text):
    remove_digits = str.maketrans('', '', string.digits)
    res = text.translate(remove_digits)
    return res


def clean(corpus_file):
    
    print("reading corpus ...")
    corpus = open(corpus_file).read()
    
    print("removing punctuations and digits")
    clean_text = remove_punctuation(corpus)
    alphapet_text = remove_digits(clean_text)
    del corpus
    del clean_text

    print("remove non Arabic")
    pure_arabic_text = keep_only_arabic(alphapet_text.split())
    del alphapet_text

    #print(pure_arabic_text.split())
    print("get the words list")
    words = pure_arabic_text.split()
    del pure_arabic_text

    return ' '.join(words)

#print as one line
def print_to_file(clean_corpus,output_file):
    fout = open(output_file, 'w', encoding = 'utf-8')
    fout.write(clean_corpus) 

def usage():
    return "please provide a corpus file"


if __name__ == '__main__':
    if len(sys.argv) == 2:
        corpus = sys.argv[1]  # file_name
        #print(clean(corpus))
        print_to_file(clean(corpus),'clean_data/'+sys.argv[1])
    else:
        print(usage())
        sys.exit(-1)
