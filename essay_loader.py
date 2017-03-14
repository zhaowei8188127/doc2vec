import json
import os
from collections import namedtuple
from gensim import utils

from sklearn import metrics
from sklearn import linear_model

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

def load_data():
    my_docs = []
    decoder = json.JSONDecoder()

    essay_folder = '/Users/weizhao/Data/qidTomark/'
    file_names = os.listdir(essay_folder)
    for file_name in file_names:
        with open(essay_folder + file_name, encoding='utf-8') as data:
            for line_no, line in enumerate(data):
                arr = line.strip().split("\t")
                essay_title = arr[0].strip()
                essay = decoder.decode(arr[3])
                essay = normalize_text(essay)
                tokens = utils.to_unicode(essay).split()
                words = tokens
                tags = [essay_title + "_" + str(line_no)]  # `tags = [tokens[0]]` would also work at extra memory cost

                if essay_title == '1547129':
                    sentiment = [1]
                    if line_no // 350 == 0 :
                        split = 'train'
                    elif line_no // 500 == 0:
                        split = "test"
                    else:
                        split = "extra"
                else:
                    sentiment = [0]
                    if line_no // 13 == 0 :
                        split = 'train'
                    elif line_no // 18 == 0:
                        split = "test"
                    else:
                        split = "extra"

                my_docs.append(SentimentDocument(words, tags, split, sentiment))
                # // floor div

    # train_docs = [doc for doc in my_docs if doc.split == 'train']
    # test_docs = [doc for doc in my_docs if doc.split == 'test']
    # doc_list = my_docs[:]
    # print('%d docs: %d train-sentiment, %d test-sentiment' % (len(doc_list), len(train_docs), len(test_docs)))

    return my_docs

def train_lr_model(X_train, y_train, X_test, y_test, train_docs, test_docs):
    lr_model = linear_model.Ridge(alpha=.5)
    lr_model.fit(X_train, y_train)
    y_test_score = lr_model.predict(X_test)
    y_test_predict = [1 if x > 0.5 else 0 for x in y_test_score]
    accuracy_score = metrics.accuracy_score(y_test, y_test_predict)

    print("accuracy score : %.2f" % accuracy_score)
    print(type(y_test_score))
    print(type(y_test_predict))

    return y_test_predict, y_test_score


def print_test_cases(X_test, y_test, y_test_predict, y_test_score, test_docs):
    for i in range(len(y_test)):
        yt = y_test[i]
        ytp = y_test_predict[i]
        if yt == 0 and ytp == 1:
            print("yt: %d, ytp: %d, score: %f" % (yt, ytp, y_test_score[i]))
            print(test_docs[i])
            print("\n")

    print("*" * 80)

    for i in range(len(y_test)):
        yt = y_test[i]
        ytp = y_test_predict[i]
        if yt == 1 and ytp == 0:
            print("yt: %d, ytp: %d, score: %f" % (yt, ytp, y_test_score[i]))
            print(test_docs[i])
            print("\n")