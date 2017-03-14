from gensim.models import Doc2Vec
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
from sklearn import linear_model
import essay_loader


def load_models():
    model_name1 = "./models1/dmc"
    model_name2 = "./models1/dbow"
    model_name3 = "./models1/dmm"

    simple_models = []
    simple_models.append(Doc2Vec.load(model_name1))
    simple_models.append(Doc2Vec.load(model_name2))
    simple_models.append(Doc2Vec.load(model_name3))

    return simple_models


def test_model(simple_models):
    for model in simple_models:
        inferred_docvec = model.infer_vector(my_docs[0].words)
        inferred_docvec2 = model.infer_vector(my_docs[1].words)
        print('%s:\n %s' % (model, model.docvecs.most_similar([inferred_docvec], topn=3)))

        sim = model.docvecs.similarity_unseen_docs(model, my_docs[0].words, my_docs[1].words)
        print('for doc %.4f...' % sim)

def cluster(model, my_docs):
    model = simple_models[0]

    X = []
    labels = []
    for doc in my_docs:
        inferred_docvec = model.infer_vector(doc.words, alpha=0.1, min_alpha=0.0001, steps=5)
        X.append(inferred_docvec)
        labels.append(int(doc.tags[0].split("_")[0]))

    km = KMeans(n_clusters=33, init='k-means++', max_iter=100, n_init=1, verbose=True)
    km.fit(X)

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))

    mm = {}
    for i in range(len(labels)):
        label = labels[i]
        km_label = km.labels_[i]
        if not mm.__contains__(km_label):
            mm[km_label] = {}
        tag = my_docs[i].tags[0].split("_")[0]
        if not mm[km_label].__contains__(tag):
            mm[km_label][tag] = 0
        mm[km_label][tag] += 1
    return mm


def classify(model, my_docs):
    train_docs = [doc for doc in my_docs if doc.split == 'train']
    test_docs = [doc for doc in my_docs if doc.split == 'test']
    print('%d docs: %d train-sentiment, %d test-sentiment' % (len(my_docs), len(train_docs), len(test_docs)))

    X_train = []
    y_train = []
    for doc in train_docs:
        inferred_docvec = model.infer_vector(doc.words, alpha=0.1, min_alpha=0.0001, steps=5)
        X_train.append(inferred_docvec)
        y_train.append(doc.sentiment[0])

    X_test = []
    y_test = []
    for doc in test_docs:
        inferred_docvec = model.infer_vector(doc.words, alpha=0.1, min_alpha=0.0001, steps=5)
        X_test.append(inferred_docvec)
        y_test.append(doc.sentiment[0])
    return X_train, y_train, X_test, y_test, train_docs, test_docs


if __name__ == '__main__':
    my_docs = essay_loader.load_data()
    simple_models = load_models()
    # test_model(simple_models)
    # cluster(simple_models[0], my_docs)
    X_train, y_train, X_test, y_test, train_docs, test_docs = classify(simple_models[0], my_docs)
    essay_loader.train_lr_model(X_train, y_train, X_test, y_test, train_docs, test_docs)

    #  plot_original(X[:2000], labels[:2000])