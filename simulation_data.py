from agent import InternalState
import matplotlib.pyplot as plt
import random
import numpy as np

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

    def plot(self):
        """
        Represent the data in a graph.
        Two histograms of the proportion of agents.
        :return:
        """
        healthy_distribution = np.array(list(self.timestamp_to_healthy.values()))
        contaminated_distribution = np.array(list(self.timestamp_to_contaminated.values()))

        timestamps = np.array(list(self.timestamp_to_healthy.keys()))
        # bins = np.linspace(0, len(self.timestamp_to_contaminated), len(self.timestamp_to_contaminated) + 1)

        # print(healthy_distribution)
        # print(bins)
        # plt.hist(healthy_distribution, bins, alpha=0.5, label='Healthy')
        # plt.hist(contaminated_distribution, bins, alpha=0.5, label='Contaminated')
        plt.plot(timestamps, healthy_distribution, '-', label='Healthy')
        plt.plot(timestamps, contaminated_distribution, '-', label='Contaminated')
        plt.legend(loc='upper right')
        plt.show()

    def output_to_csv(self):
        """
        Save data in a csv.
        :return:
        """
        pass




if __name__ == "__main__":
    sim_data = SimulationData()
    for i in range(50):
        x = random.randint(0, 50)
        sim_data.add_data(i, x, 50 - x)

    sim_data.plot()
