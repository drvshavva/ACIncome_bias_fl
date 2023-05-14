class ACSIncomeBiasMetrics:
    @staticmethod
    def __calculate_ratios(df):
        # priviliged = df[df["SEX_Male"] == 1]
        priviliged = df[df["RAC1P_White alone"] == 1]
        num_of_priviliged = priviliged.shape[0]

        # unpriviliged = df[df["SEX_Female"] == 1]
        unpriviliged = df[df["RAC1P_others"] == 1]
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

    def __calculate_cm(self, df):
        white_df = df[df["RAC1P_White alone"] == 1]
        others_df = df[df["RAC1P_others"] == 1]

        TPU, FPU, TNU, FNU = self.perf_measure(others_df["PINCP"].values, others_df["PINCP_predicted"].values)
        TPP, FPP, TNP, FNP = self.perf_measure(white_df["PINCP"].values, white_df["PINCP_predicted"].values)
        # print(f"True Positive Unpriviliged:{TPU}")
        # print(f"False Positive Unpriviliged:{FPU}")
        # print(f"True Positive Priviliged:{TPP}")
        # print(f"False Positive Priviliged:{FPP}")
        # print(f"True Negative Unpriviliged:{TNU}")
        # print(f"False Negative Unpriviliged:{FNU}")
        # print(f"True Negative Priviliged:{TNP}")
        # print(f"False Negative Priviliged:{FNP}")
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

        print(f"Disparate impact: {di}")
        print(f"Equal opportunity diff: {eod}")
        print(f"Statistical parity: {sp}")
        print(f"EO: {eo}")

    def return_bias_metrics(self, df):
        di = self.calculate_disparate_impact(df)
        sp = self.calculate_statistical_parity(df)
        eo = self.calculate_eo(df)
        eod = self.calculate_equal_opportunity_diff(df)
        return di, sp, eo, eod
