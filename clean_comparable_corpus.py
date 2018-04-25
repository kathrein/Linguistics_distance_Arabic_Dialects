import  cleaner
import os
import sys



if __name__ == '__main__':
    if len(sys.argv) == 2:
        corpus = 'clean_data/comparable/'+sys.argv[1]+'/'  # file_name

        for file in os.listdir(corpus):
            extension = os.path.splitext(file)[1]
            if extension == '.txt':
                filepath = os.path.join(corpus, file)
                cleaner.print_to_file(cleaner.clean(filepath), 'clean_data/new_comparable/' + sys.argv[1]+'/'+os.path.basename(file))
        #print(clean(corpus))



    else:
        print(cleaner.usage())
        sys.exit(-1)
