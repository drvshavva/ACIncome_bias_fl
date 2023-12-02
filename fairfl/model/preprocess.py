from os.path import dirname

import pandas as pd
from sklearn.preprocessing import *
from sklearn.preprocessing import LabelEncoder


class ACSIncomePreprocess:
    def __init__(self):
        self.dataset_path = dirname(dirname(dirname(__file__))) + "/dataset"

    @staticmethod
    def __school(df):
        schl_order = ["below high-school", "Regular high school diploma", "GED or alternative credential",
                      "Some college, but less than 1 year", "1 or more years of college credit, no degree",
                      "Associate's degree", "Bachelor's degree", "Master's degree",
                      "Professional degree beyond a bachelor's degree", "Doctorate degree"]
        enc = OrdinalEncoder(categories=[schl_order])
        df["SCHL"] = enc.fit_transform(df[["SCHL"]])
        return df

    @staticmethod
    def __age(df):
        df["AGEP"] = pd.cut(df["AGEP"], bins=[0, 25, 45, 65, 100], labels=["Young", "Middle", "Senior", "Old"],
                            ordered=True)
        enc_age = OrdinalEncoder(categories=[["Young", "Middle", "Senior", "Old"]])
        df["AGEP"] = enc_age.fit_transform(df[["AGEP"]])
        return df

    @staticmethod
    def __working_hours(df):
        df["WKHP"] = pd.cut(df["WKHP"], bins=[0, 25, 40, 60, 100],
                            labels=["Part-time", "Full-time", "Over-time", "Workaholic"], ordered=True)
        enc_hpw = OrdinalEncoder(categories=[["Part-time", "Full-time", "Over-time", "Workaholic"]])
        df["WKHP"] = enc_hpw.fit_transform(df[["WKHP"]])
        return df

    @staticmethod
    def __one_hot_to_others(df):
        df = pd.get_dummies(df, columns=['COW', 'MAR', 'OCCP', 'RELP', 'SEX'], dtype=float)
        return df

    @staticmethod
    def __categorical_to_others(df):
        # Identify categorical columns
        df_2 = df.copy()
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df_2[col] = le.fit_transform(df_2[col])
        return df_2

    @staticmethod
    def split_x_y(df):
        x = df.drop('PINCP', axis=1)
        y = df['PINCP']
        return x, y

    @staticmethod
    def race(df):
        df_x = df.copy()
        df_x['RAC1P'].replace(df_x['RAC1P'].unique()[df_x['RAC1P'].unique() != 'Black or African American alone'],
                              'others', inplace=True)
        df_x['RAC1P'].replace(df_x['RAC1P'].unique()[df_x['RAC1P'].unique() == 'Black or African American alone'],
                              'Black', inplace=True)
        df_x['RAC1P'] = df_x['RAC1P'].replace({'Black': 0, 'others': 1})
        return df_x

    def preprocess(self, df, one_hot=True):
        df = self.race(df)
        df = self.__school(df)
        df = self.__age(df)
        df = self.__working_hours(df)
        df = self.__one_hot_to_others(df) if one_hot else self.__categorical_to_others(df)
        return df

    def get_preprocessed_data(self):
        train = pd.read_csv(f"{self.dataset_path}/train_preprocessed.csv")
        test = pd.read_csv(f"{self.dataset_path}/test_preprocessed.csv")

        return train, test

    def get_x_y_preprocessed(self, train, test):
        x_train, y_train = self.split_x_y(train)
        x_test, y_test = self.split_x_y(test)
        return x_train, y_train, x_test, y_test
