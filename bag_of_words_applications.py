import essay_loader
import nltk


def document_features(document, word_features):
    document_words = set(document.words)
    features = []
    for word in word_features:
        # features["contains({})".format(word)] = (word in document_words)
        features.append(word in document_words)
    return features

def features_words(my_docs):
    all_words = []
    for doc in my_docs:
        for word in doc.words:
            all_words.append(word)
    freq_word_dist = nltk.FreqDist(all_words)
    feature_words = [word for (word, freq) in freq_word_dist.most_common(250)][:-2000]
    print(feature_words)

    return feature_words

def classify_bag_of_words(my_docs):
    train_docs = [doc for doc in my_docs if doc.split == 'train']
    test_docs = [doc for doc in my_docs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(my_docs), len(train_docs), len(test_docs)))

    feature_words = features_words(my_docs)

    X_train = [document_features(doc, feature_words) for doc in train_docs]
    y_train = [doc.sentiment[0] for doc in train_docs]

    X_test = [document_features(doc, feature_words) for doc in test_docs]
    y_test = [doc.sentiment[0] for doc in test_docs]

    return X_train, y_train, X_test, y_test, train_docs, test_docs

if __name__ == '__main__':
    my_docs = essay_loader.load_data()
    X_train, y_train, X_test, y_test, train_docs, test_docs = classify_bag_of_words(my_docs)
    y_test_predict, y_test_score = essay_loader.train_lr_model(X_train, y_train, X_test, y_test, train_docs, test_docs)
    essay_loader.print_test_cases(X_test, y_test, y_test_predict, y_test_score, test_docs)