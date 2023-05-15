import pandas as pd
from os.path import dirname

from sklearn.linear_model import LogisticRegression

from fairfl.model.preprocess import ACSIncomePreprocess
from fairfl.utils.read_data import DatasetUtils
from fairfl.metrics.bias_metrics import ACSIncomeBiasMetrics, BiasType
from fairfl.metrics.classical_metrics import ClassicalMetrics


class ACSIncomeModel:
    def __init__(self, bias_type=BiasType.race):
        self.bias_type = bias_type
        self.utils = DatasetUtils()
        self.preprocess = ACSIncomePreprocess()
        self.bias_metrics = ACSIncomeBiasMetrics(bias_type=bias_type)
        self.cls_metrics = ClassicalMetrics()
        self.model_path = dirname(dirname(dirname(__file__))) + "/models"

    def __preprocess_train_test(self, train, test):
        return self.preprocess.get_x_y_preprocessed(train, test)

    def print_model_metrics_for_state(self, train, test, state_name):
        train_state, test_state = self.utils.get_state_data(train, test, state_name)
        res_df = pd.DataFrame({"state": [state_name]})
        group = "RAC1P_others" if self.bias_type is BiasType.race else "SEX_Male"
        df_group = train_state.groupby([group, 'PINCP']).size().to_dict()
        df_group_keys = list(df_group.keys())
        res_df_group = pd.DataFrame({x: [df_group[x]] for x in df_group_keys})
        test = test.drop('ST', axis=1)
        res_df_metrics = self.print_model_metrics(train_state, test)
        return pd.concat([res_df, res_df_group, res_df_metrics], axis=1)

    def print_model_metrics(self, train, test, pipeline=LogisticRegression()):
        x_train, y_train, x_test, y_test = self.__preprocess_train_test(train, test)
        res_df = pd.DataFrame({"egitim_ornek_sayisi": [len(y_train)],
                               "test_ornek_sayisi": [len(y_test)]})

        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)

        res_df_metrics = self.cls_metrics.return_as_pd(y_pred=predicted, y_true=y_test)
        test_df = pd.concat(
            [x_test.reset_index(drop=True), y_test.to_frame(), pd.DataFrame(predicted, columns=["PINCP_predicted"])],
            axis=1)

        res_df_bias_metrics = self.bias_metrics.calculate_bias_metrics(test_df)
        return pd.concat([res_df, res_df_metrics, res_df_bias_metrics], axis=1)
