#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 18:05:30 2018

@author: xabuka
"""

import sys
import string
from alphabet_detector import AlphabetDetector
from pprint import pprint
import delete_repeated_char as del_char



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
    corpus = open(corpus_file).readlines()
    new_corpus = []
    #pprint(corpus)
    
    for line in corpus:
        
    
        #print("removing punctuations and digits")
        clean_text = remove_punctuation(line)
        alphapet_text = remove_digits(clean_text)
        #del corpus
        del clean_text
    
        #print("remove non Arabic")
        pure_arabic_text = keep_only_arabic(alphapet_text.split())
        del alphapet_text

        final_update_text = del_char.delete_repeat_char(pure_arabic_text)
        new_corpus.append(final_update_text)
    
    #print(new_corpus)
    #return ' '.join(words), new_corpus
    return new_corpus


#print as one line
def print_to_file(new_corpus,output_file):
    fout = open(output_file, 'w', encoding = 'utf-8')
    for line in new_corpus:
        if not line.strip(): continue # remove empty lines
        fout.write(line+'\n') 

def usage():
    return "please provide a corpus file"


if __name__ == '__main__':
    if len(sys.argv) == 2:
        corpus = 'clean_data/'+sys.argv[1]  # file_name
        #print(clean(corpus))
        print_to_file(clean(corpus),'clean_data/clean_'+sys.argv[1])
    else:
        print(usage())
        sys.exit(-1)
