import pandas as pd

from sklearn.model_selection import *


def cross_validation(model, x, y, cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv = StratifiedKFold(n_splits=cv, random_state=33, shuffle=True)
    results = cross_validate(estimator=model,
                             X=x,
                             y=y,
                             cv=cv,
                             scoring=_scoring,
                             return_train_score=True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
                         "Training Precision scores": results['train_precision'],
                         "Training Recall scores": results['train_recall'],
                         "Training F1 scores": results['train_f1'],
                         "Test Accuracy scores": results['test_accuracy'],
                         "Test Precision scores": results['test_precision'],
                         "Test Recall scores": results['test_recall'],
                         "Test F1 scores": results['test_f1']
                         }).T
