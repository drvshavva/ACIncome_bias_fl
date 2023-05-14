import pandas as pd
from os.path import dirname

from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression

from model.preprocess import ACSIncomePreprocess
from utils.read_data import DatasetUtils
from metrics.bias_metrics import ACSIncomeBiasMetrics


class ACSIncomeModel:
    def __init__(self):
        self.utils = DatasetUtils()
        self.preprocess = ACSIncomePreprocess()
        self.bias_metrics = ACSIncomeBiasMetrics()
        self.model_path = dirname(dirname(__file__)) + "/models"

    def __preprocess_train_test(self, train, test):
        return self.preprocess.get_x_y_preprocessed(train, test)

    def print_model_metrics_for_state(self, train, test, state_name):
        train_state, test_state = self.utils.get_state_data(train, test, state_name)
        print("-----------------------------------")
        print(f"  RESULTS for state: {state_name} ")
        print("-----------------------------------")
        print("Race Dağılımı Train:")
        print(train_state.groupby(['RAC1P_others', 'PINCP']).size())
        self.print_model_metrics(train_state, test_state)

    def print_model_metrics(self, train, test, pipeline=LogisticRegression()):
        x_train, y_train, x_test, y_test = self.__preprocess_train_test(train, test)
        print(f"Eğitim örnek sayısı: {len(y_train)}")
        print(f"Test örnek sayısı: {len(y_test)}")

        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)

        print(classification_report(y_test, predicted))

        test_df = pd.concat(
            [x_test.reset_index(drop=True), y_test.to_frame(), pd.DataFrame(predicted, columns=["PINCP_predicted"])],
            axis=1)
        self.bias_metrics.calculate_bias_metrics(test_df)
