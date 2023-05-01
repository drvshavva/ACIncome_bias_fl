import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class Plots:
    @staticmethod
    def plot_bar_chart(labels: list, values: list, title: str):
        """
        This method plot bar chart

        :param labels: list of labels, :type list
        :param values: count of each label values, :type list
        :param title: title of plot
        :return: plot
        """
        y_pos = np.arange(len(labels))
        plt.figure(figsize=(24.0, 16.0))
        plt.bar(y_pos, values, align='center')
        plt.xticks(y_pos, labels)
        plt.ylabel('Count')
        plt.title(title)
        # plt.savefig(f"{RESULTS_PATH}bar_chart_{title}")
        return plt

    @staticmethod
    def plot_pie_chart(labels: list, values: list, title: str):
        """
        This method plot pie chart

        :param labels: list of labels, :type list
        :param values: count of each label values, :type list
        :param title: title of plot
        :return: plot
        """
        plt.figure(figsize=(24.0, 16.0))
        plt.pie(values, labels=labels, startangle=90, autopct='%.1f%%')
        plt.title(title)
        # plt.savefig(f"{RESULTS_PATH}pie_chart_{title}")
        return plt

    @staticmethod
    def plot_count_plot(label_name: str, data: pd.DataFrame, title: str):
        """
        This method returns count plot of the dataset

        :param label_name: name of the class, :type str
        :param data: input dataFrame, :type DataFrame
        :param title: title of plot
        :return plt
        """
        plt.figure(figsize=(24.0, 16.0))
        sns.countplot(x=label_name, data=data)
        plt.title(title)
        # plt.savefig(f"{RESULTS_PATH}plot_count_{title}")
        return plt

    @staticmethod
    def plot_group_count_plot(df, x="SEX", hue="PINCP"):
        aq_palette = sns.diverging_palette(225, 35, n=2)
        by_sex = sns.countplot(x=x, hue=hue, data=df, palette=aq_palette)
