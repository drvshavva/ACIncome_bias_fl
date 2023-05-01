class ACSIncomeBiasMetrics:
    @staticmethod
    def __calculate_ratios(df):
        male_df = df[df["SEX_Male"] == 1]
        num_of_priviliged = male_df.shape[0]

        female_df = df[df["SEX_Female"] == 1]
        num_of_unpriviliged = female_df.shape[0]

        unpriviliged_labels = female_df[female_df["PINCP_predicted"] == 1].shape[0]
        unpriviliged_ratio = unpriviliged_labels / num_of_unpriviliged

        priviliged_labels = male_df[male_df["PINCP_predicted"] == 1].shape[0]
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
        male_df = df[df["SEX_Male"] == 1]
        female_df = df[df["SEX_Female"] == 1]

        TPU, FPU, TNU, FNU = self.perf_measure(female_df["PINCP"].values, female_df["PINCP_predicted"].values)
        TPP, FPP, TNP, FNP = self.perf_measure(male_df["PINCP"].values, male_df["PINCP_predicted"].values)
        return TPU, FPU, TNU, FNU, TPP, FPP, TNP, FNP

    def calculate_equal_opportunity_diff(self, df):
        """ iki grubun TPR ler arasındaki fark --> 0'a yakın olması beklenir"""
        TPU, FPU, TNU, FNU, TPP, FPP, TNP, FNP = self.__calculate_cm(df)

        female_tpr = TPU / (TPU + FNU)
        male_tpr = TPP / (TPP + FNP)

        return male_tpr - female_tpr

    def calculate_eo(self, df):
        TPU, FPU, TNU, FNU, TPP, FPP, TNP, FNP = self.__calculate_cm(df)
        return 0.5 * (abs((FPP / (FPP + TNP)) - (FPU / (FPU + TNU))) + abs((TPP / (TPP + FNP)) - (TPU / (TPU + FNU))))

    def calculate_bias_metrics(self, df):
        di = self.calculate_disparate_impact(df)
        eod = self.calculate_equal_opportunity_diff(df)
        sp = self.calculate_statistical_parity(df)
        eo = self.calculate_eo(df)

        print(f"Disparate impact: {di}")
        # print(f"Equal opportunity diff: {eod}")
        print(f"Statistical parity: {sp}")
        # print(f"EO: {eo}")

    def return_bias_metrics(self, df):
        di = self.calculate_disparate_impact(df)
        sp = self.calculate_statistical_parity(df)
        return di, sp
