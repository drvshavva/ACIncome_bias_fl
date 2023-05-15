import pandas as pd


class BiasType:
    race = "race"
    gender = "gender"
    age = "age"


class ACSIncomeBiasMetrics:
    def __init__(self, bias_type=BiasType.race):
        self.bias_type = bias_type

    def __calculate_ratios(self, df):
        priviliged, unpriviliged = self.__get_priviliged_unpriviliged(df)

        num_of_priviliged = priviliged.shape[0]
        num_of_unpriviliged = unpriviliged.shape[0]

        unpriviliged_labels = unpriviliged[unpriviliged["PINCP_predicted"] == 1].shape[0]
        unpriviliged_ratio = unpriviliged_labels / num_of_unpriviliged

        priviliged_labels = priviliged[priviliged["PINCP_predicted"] == 1].shape[0]
        priviliged_ratio = priviliged_labels / num_of_priviliged
        return unpriviliged_ratio, priviliged_ratio

    def calculate_disparate_impact(self, df):
        """
        ayrıcalıklı grup --> male
        iki grubun 1 olması olasılıkları arasındaki oran 1 'e yakın olmalı"""
        unpriviliged_ratio, priviliged_ratio = self.__calculate_ratios(df)
        try:
            di = priviliged_ratio / unpriviliged_ratio
        except:
            di = 0
        return di

    def calculate_statistical_parity(self, df):
        """ iki grubun 1 olması olasılıkları arasındaki fark --> 0'a yakın olması beklenir --> demografig parity olarakta bilinir"""
        unpriviliged_ratio, priviliged_ratio = self.__calculate_ratios(df)
        return priviliged_ratio - unpriviliged_ratio

    @staticmethod
    def perf_measure(y_actual, y_hat):
        tp, fp, tn, fn = 0, 0, 0, 0

        for i in range(len(y_hat)):
            if y_actual[i] == y_hat[i] == 1:
                tp += 1
            if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
                fp += 1
            if y_actual[i] == y_hat[i] == 0:
                tn += 1
            if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
                fn += 1

        return tp, fp, tn, fn

    def __get_priviliged_unpriviliged(self, df):
        if self.bias_type is BiasType.race:
            priviliged = df[df["RAC1P_White alone"] == 1]
            unpriviliged = df[df["RAC1P_others"] == 1]
        elif  self.bias_type is BiasType.gender:
            priviliged = df[df["SEX_Male"] == 1]
            unpriviliged = df[df["SEX_Female"] == 1]
        else:
            # todo: age için devam edilecek
            pass
        return priviliged, unpriviliged

    def __calculate_cm(self, df):
        priviliged, unpriviliged = self.__get_priviliged_unpriviliged(df)

        TPU, FPU, TNU, FNU = self.perf_measure(unpriviliged["PINCP"].values, unpriviliged["PINCP_predicted"].values)
        TPP, FPP, TNP, FNP = self.perf_measure(priviliged["PINCP"].values, priviliged["PINCP_predicted"].values)
        return TPU, FPU, TNU, FNU, TPP, FPP, TNP, FNP

    def calculate_equal_opportunity_diff(self, df):
        """ iki grubun TPR ler arasındaki fark --> 0'a yakın olması beklenir"""
        TPU, FPU, TNU, FNU, TPP, FPP, TNP, FNP = self.__calculate_cm(df)

        un_tpr = TPU / (TPU + FNU)
        pri_tpr = TPP / (TPP + FNP)

        return pri_tpr - un_tpr

    def calculate_eo(self, df):
        TPU, FPU, TNU, FNU, TPP, FPP, TNP, FNP = self.__calculate_cm(df)
        try:
            res = 0.5 * (
                    abs((FPP / (FPP + TNP)) - (FPU / (FPU + TNU))) + abs((TPP / (TPP + FNP)) - (TPU / (TPU + FNU))))
        except:
            res = 0
        return res

    def calculate_bias_metrics(self, df):
        di = self.calculate_disparate_impact(df)
        eod = self.calculate_equal_opportunity_diff(df)
        sp = self.calculate_statistical_parity(df)
        eo = self.calculate_eo(df)

        return pd.DataFrame(
            {'disparate_impact': [di],
             'equal_opportunity_diff': [eod],
             'statistical_parity': [sp],
             'equal_opportunity': [eo]})

    def return_bias_metrics(self, df):
        di = self.calculate_disparate_impact(df)
        sp = self.calculate_statistical_parity(df)
        eo = self.calculate_eo(df)
        eod = self.calculate_equal_opportunity_diff(df)
        return di, sp, eo, eod
