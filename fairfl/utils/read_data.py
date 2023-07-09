import numpy as np
import pandas as pd
from os.path import dirname

from fairfl.utils.constants import ACSIncome_categories, _STATE_CODES, states

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from fairfl.model.preprocess import ACSIncomePreprocess


class DatasetUtils:
    def __init__(self):
        self.dataset_path = dirname(dirname(dirname(__file__))) + "/dataset"
        self.states = states
        self._state_codes = _STATE_CODES
        self.categories = ACSIncome_categories
        self.preprocess = ACSIncomePreprocess()

    @staticmethod
    def get_acs_income_data():
        data_dict = fetch_openml(
            data_id=43141,
            data_home=".",
            cache=True,
            as_frame=True,
            return_X_y=False,
        )
        df_all = data_dict['data'].copy(deep=True)
        df_all['PINCP'] = data_dict['target']
        df_all["PINCP"] = np.where(df_all["PINCP"] > 50_000, 1, 0)
        return df_all

    def get_acs_income_data_processed(self):
        df_all = self.get_acs_income_data()

        for key, value in self.categories["MAR"].items():
            df_all.MAR.replace(key, value, inplace=True)

        for key, value in self.categories["SEX"].items():
            df_all.SEX.replace(key, value, inplace=True)

        # 24 farklı kategori var ama 15'den düşük olanlar lise diplomasına sahip
        # 15 Den küçükleri below lise diploması şeklinde tek bir sınıfa indirebiliriz
        self.categories["SCHL"].update({i: f'below high-school' for i in range(1, 16)})
        for key, value in self.categories["SCHL"].items():
            df_all.SCHL.replace(key, value, inplace=True)

        # RAC1P race durumunu inceleyelim
        self.categories["RAC1P"].update({i: f'others' for i in range(1, 10) if i != 2})
        for key, value in self.categories["RAC1P"].items():
            df_all.RAC1P.replace(key, value, inplace=True)

        return df_all

    def __filter_state(self, df, state):
        _df = df.query(f"ST == {int(self._state_codes[state])}")
        return _df.drop('ST', axis=1)


    def get_selected_states(self, df, selected_states):
        _states = [int(self._state_codes[state]) for state in selected_states]
        _df = df[df['ST'].isin(_states)]
        _df = _df.drop('ST', axis=1)
        return _df

    def get_state_data(self, train: pd.DataFrame, test: pd.DataFrame, state_name):
        """
        Tüm veri setinden belirtilen state için veriyi filtreleyip gönderir

        Parameters
        ----------
        train: pd.DataFrame
               train veri seti
        test: pd.DataFrame
                test veri seti
        state_name: string
                State'in ismi State.<state> şeklinde gönderilir

        Returns
        -------
        train, test: pd.DataFrame
                    Belirtilen state filtrelenmiş veri seti
        """
        df_train = self.__filter_state(train, state_name)
        df_test = self.__filter_state(test, state_name)
        return df_train, df_test

    def read_train_test_data(self):
        train = pd.read_csv(f"{self.dataset_path}/train.csv")
        test = pd.read_csv(f"{self.dataset_path}/test.csv")
        return train, test

    def split_data_and_save(self, preprocess=True):
        df = self.get_acs_income_data_processed()
        name = ""
        if preprocess:
            df = self.preprocess.preprocess(df)
            name = "_preprocessed"
        train, test = train_test_split(df, test_size=0.3, random_state=42)
        train.to_csv(f"{self.dataset_path}/train{name}.csv", index=False)
        test.to_csv(f"{self.dataset_path}/test{name}.csv", index=False)
