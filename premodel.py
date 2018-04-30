import os.path
from gensim import corpora, models, similarities


class MyCorpus(object):
    def __init__(self, corpus_file, dictionary):
        """
        Checks if a dictionary has been given as a parameter.
        If no dictionary has been given, it creates one and saves it in the disk.
        """
        self.file_name = corpus_file
        if not dictionary is None:
            self.dictionary = dictionary
            # else:
            #     self.dictionary = dictionary

    def __iter__(self):
        for line in open(self.file_name, encoding='utf-8'):
            # assume there's one document per line, tokens separated by whitespace
            yield self.dictionary.doc2bow(line.split())



def read_text(corpus_file):
    with open(corpus_file, encoding='utf-8') as f:  # we can define file_name
        documents = f.read().splitlines()
    texts = [[word for word in document.split()] for document in documents]

    return texts


def read_full_text(corpus_file):
    with open(corpus_file, encoding='utf-8') as f:  # we can define file_name
        documents = f.read()
    texts = [documents.split(), []]
    # print (texts)
    return texts


def read_set_of_file(folder):
    lists = []
    counter = 0
    for file in os.listdir(folder):
        #print(len(os.listdir(folder)))
        extension = os.path.splitext(file)[1]
        if extension == '.txt':
            print (counter, file)
            filepath = os.path.join(folder, file)
            f = open(filepath, mode='r', encoding='utf-8')
            lists = lists + read_full_text(filepath)
            f.close()
            counter = counter +1
    return filter(None, lists)

def read_folder(folder):
    list = []
    for file in os.listdir(folder):
        extension = os.path.splitext(file)[1]
        if extension == '.txt':
            filepath = os.path.join(folder, file)
            f = open(filepath,mode = 'r', encoding = 'utf-8')
            list.append(f.read())
            f.close
    return list


def upload_data(dialect):
    # 2 Transformation
    if (os.path.exists('parameters/' + dialect[0] + '_' + dialect[1] + '.dict')):
        dictionary = corpora.Dictionary.load('parameters/' + dialect[0] + '_' + dialect[1] + '.dict')
        corpus = corpora.MmCorpus('parameters/' + dialect[0] + '_' + dialect[1] + '.mm')
        print("True files generated from first tutorial")
    else:
        print("Please run first tutorial to generate data set")

    return dictionary, corpus