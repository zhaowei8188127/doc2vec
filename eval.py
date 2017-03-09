import numpy as np
import statsmodels.api as sm
from random import sample
from contextlib import contextmanager
from timeit import default_timer


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def logistic_predictor_from_data(train_targets, train_regressors):
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    # print(predictor.summary())
    return predictor


def error_rate_for_model(test_model, train_set, test_set, infer=False, infer_steps=3, infer_alpha=0.1,
                         infer_subsample=0.1):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets, train_regressors = zip(*[(doc.sentiment, test_model.docvecs[doc.tags[0]]) for doc in train_set])
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if infer:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in
                           test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_data]
    test_regressors = sm.add_constant(test_regressors)

    # predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return error_rate, errors, len(test_predictions), predictor
