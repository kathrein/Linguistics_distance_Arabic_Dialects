import nltk
from pprint import pprint
import argparse
import premodel

#command
#  python3 most_common.py -c clean_data/sdc -n 100 -f jo


parser = argparse.ArgumentParser()
parser.add_argument("--corpus_folder", "-c", type=str, help='the folder contains the corpus files.', required=True)
parser.add_argument("--dialect", "-f", type=str, help='the dialect  file', required=True)
parser.add_argument("--count_common", "-n", type=int, help='number of most common words', required=True)
parser.add_argument("--corpus_type", "-t", type=str, help='pc: parallel corpus, cc: comparable corpus, up: un-parallel corpus', required=False)



if __name__ == '__main__':
    args = parser.parse_args()
    file_path = args.corpus_folder+'/' +args.dialect+'.txt'
    folder  = args.corpus_folder+'/'+args.dialect+'/'

    if args.corpus_type == 'cc':
        text = ''.join(list(premodel.read_folder(folder)))
        #print(nltk.tokenize.word_tokenize(text2))
    else :
        with open(file_path, encoding = 'utf-8') as f:
            text = f.read()

    allWords = nltk.tokenize.word_tokenize(text)
    #print(text)
    #print(allWords)
    #print(allWords)
    allWordDist = nltk.FreqDist(allWords)
    pprint(allWordDist.most_common(args.count_common))




    # for word, frequency in allWordDist.most_common(10):
    #     print('%s;%d' % (word, frequency))
