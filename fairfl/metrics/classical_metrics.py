import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ClassicalMetrics:
    @staticmethod
    def calculate_classical_metrics(y_pred, y_true):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    def return_as_pd(self, y_pred, y_true):
        accuracy, precision, recall, f1 = self.calculate_classical_metrics(y_pred=y_pred, y_true=y_true)
        return pd.DataFrame({'f1_score': [f1]})
