import random
from operator import itemgetter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

TIMESTAMP = 'Timestamp'
NUMHEALTHY = 'Number of Healthy Agents'

mpl.rcParams["axes.labelsize"] = 'small'

class SimulationData(object):
    """
    Holds all the data relevant to our simulation. Maps for each timestamp the distribution of
    healthy and contaminated agents.
    This data will be represented as a simulation result.
    """

    def __init__(self):
        self.timestamp_to_healthy = {}
        self.timestamp_to_contaminated = {}

    def add_data(self, timestamp, healthy, contaminated):
        """
        Add data about distribution of healthy and contaminated agents in the game.
        :param timestamp Current timestamp
        :param healthy: Proportion of healthy agents.
        :param contaminated:  Proportion of contaminated agents.
        :return:
        """
        if timestamp in self.timestamp_to_contaminated or timestamp in self.timestamp_to_healthy:
            raise KeyError("The timestamp " + str(timestamp) + " already exists in our map")

        self.timestamp_to_healthy[timestamp] = healthy
        self.timestamp_to_contaminated[timestamp] = contaminated

    def get_length(self):
        return max(self.timestamp_to_healthy.items(),key=itemgetter(0))[0]

    def plot(self):
        """
        Represent the data in a graph.
        Two histograms of the proportion of agents.
        :return:
        """
        healthy_distribution = np.array(list(self.timestamp_to_healthy.values()))
        contaminated_distribution = np.array(list(self.timestamp_to_contaminated.values()))

        timestamps = np.array(list(self.timestamp_to_healthy.keys()))
        plt.plot(timestamps, healthy_distribution, '-', label='Healthy')
        plt.plot(timestamps, contaminated_distribution, '-', label='Contaminated')
        plt.legend(loc='upper right')
        plt.show()

    def healthy_won(self):
        """
        True if the healthy team won, false otherwise.
        :return:
        """
        end_time = self.get_length()
        return self.timestamp_to_healthy[end_time] != 0 and self.timestamp_to_contaminated[end_time] == 0

    def output_to_csv(self):
        """
        Save data in a csv.
        :return:
        """
        pass

def to_dataframe(simulations, save=False, name="Sim.csv"):
    max_length = 0
    win_count = 0
    for simulation in simulations:
        win_count += simulation.healthy_won() * 1
        curr_length = simulation.get_length()
        if curr_length > max_length:
            max_length = curr_length

    jumps = int(max_length / 200) + 1

    print("Jumps: " + str(jumps))
    print("Winning ratio: " + str(win_count / len(simulations)))
    data = []
    for simulation in simulations:
        for timestamp, num_healthy in simulation.timestamp_to_healthy.items():
            if timestamp % jumps == 0:
                data.append([timestamp, num_healthy])

    df = pd.DataFrame(data, columns=[TIMESTAMP, NUMHEALTHY])
    if save:
        df.to_csv(name)
    return df


def represent_as_box_plot(df):
    """
    Given several simulation data, represent the data at each timestamp as a box plot.
    :param df: Pandas dataframe of the data
    :return:
    """
    max_time = df.loc[df[TIMESTAMP].idxmax()][TIMESTAMP]
    max_time = 700
    boxplot_ranges = np.arange(0, max_time, 20)
    print(boxplot_ranges)
    print("Number of boxplots: " + str(boxplot_ranges.shape))
    print(len(df[TIMESTAMP]))
    limited_df = df.loc[df[TIMESTAMP].isin(boxplot_ranges)]
    # limited_df = limited_df.loc[::2]

    # print(df)
    # sns.swarmplot(x="Timestamp", y="NumHealthy", data=df)
    boxplot = sns.boxplot(x=TIMESTAMP, y=NUMHEALTHY, data=limited_df, color='aqua')
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel(NUMHEALTHY, fontsize=12)
    plt.xticks(fontsize=9)
    # plt.xlabel.
    # plt.ylabel(fontsize=12)
    plt.yticks(fontsize=9)

    # boxplot.set_xticks(np.arange(0, max_time, int(max_time / 30)))
    # boxplot.set_xticklabels(np.arange(0, max_time, int(max_time / 30)))

    plt.show()





if __name__ == "__main__":
    df = pd.read_csv("no_flats_gc_potential_100_sim.csv")
    represent_as_box_plot(df)
