# This Python file uses the following encoding: utf-8
import jieba
import codecs
import numpy as np
from KMeans_shorttext import KmeansST

texts = {}
textnum = 0

with open('stopwords_chinese.txt', 'r') as f:
    stopwords = []
    for line in f:
        word = str(line).rstrip()
        stopwords.append(word)
stopwords_set = set(stopwords)

with open("text.txt", "r") as f:
    for line in f:
        text = str(line).rstrip()
        word_list = jieba.cut(text)
        text_words = []
        for word in word_list:
            if word not in stopwords_set:
                text_words.append(word)
        texts[textnum] = text_words
        textnum += 1


with codecs.open('text_processed.txt', 'w', 'utf-8') as f:
    for i in range(textnum):
        for word in texts[i]:
            f.write(word+' ')
        f.write('\n')


allwords_set = set()
for i in range(textnum):
    allwords_set.update(texts[i])

allwords = list(allwords_set)
texts_vectors = np.zeros((textnum, len(allwords)), dtype=np.int)
for i in range(textnum):
    for word in texts[i]:
        idx = allwords.index(word)
        texts_vectors[i][idx] += 1

# print(texts_vectors)

cluster = KmeansST(4, texts_vectors)
cluster.iterate()
cluster.print_result()


