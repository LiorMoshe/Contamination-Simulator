import random
from operator import itemgetter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

def represent_as_box_plot(simulations):
    """
    Given several simulation data, represent the data at each timestamp as a box plot.
    :param simulations:
    :return:
    """
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


    df = pd.DataFrame(data, columns=['Timestamp', 'NumHealthy'])
    max_time = df.loc[df['Timestamp'].idxmax()]['Timestamp']

    boxplot_ranges = np.arange(0, max_time, int(max_time / 40))
    print("Number of boxplots: " + str(boxplot_ranges.shape))
    limited_df = df.loc[df['Timestamp'].isin(boxplot_ranges)]

    # print(df)
    # sns.swarmplot(x="Timestamp", y="NumHealthy", data=df)
    boxplot = sns.boxplot(x='Timestamp', y='NumHealthy', data=limited_df)
    plt.xticks(fontsize=8)

    # boxplot.set_xticks(np.arange(0, max_time, int(max_time / 30)))
    # boxplot.set_xticklabels(np.arange(0, max_time, int(max_time / 30)))

    plt.show()





if __name__ == "__main__":
    sim_data = SimulationData()
    for i in range(100):
        x = random.randint(0, 50)
        sim_data.add_data(i, x, 50 - x)

    sim_data.plot()
