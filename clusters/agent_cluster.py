from utils import get_slope, euclidean_dist
import math
import numpy as np

class AgentCluster(object):

    def __init__(self, id, agents, enemy=False):
        self._id = id
        self.agents = agents
        self.enemy = enemy
        self.allocated_action = False
        self.target_loc = None

        for agent in self.agents.values():
            agent.allocate(self)

    def size(self):
        return len(self.agents)

    def has_agent(self, agent_index):
        return any(agent.index == agent_index for agent in self.agents.values())

    def add_agent(self, agent):
        self.agents[agent.index] = agent

    def remove_agents(self, agent):
        del self.agents[agent.index]

    def release(self, agent_idx):
        del self.agents[agent_idx]

        # if len(self.agents) == 0:
            # print("Removed agent " + str(agent_idx) + " from cluster " + str(self._id))

    def get_center(self):
        """
        Get the center of the cluster.
        :return:
        """
        # print("Cluster id: " + str(self._id) + " Keys: " + str(self.agents.keys()))
        return np.sum(np.vstack([agent.get_position() for agent in self.agents.values()]), axis=0) / len(self.agents)

    def rotate_to_center(self):
        """
        Move all members of the cluster toward the center of the cluster.
        :return:
        """
        if len(self.agents) > 1:
            center = self.get_center()
            self.rotate_to_target(center)

    def rotate_to_target(self, target):
        """
        Rotate members of the cluster to a target location.
        Compute the vector for each member of the cluster in which it needs to move to get to a specific target.
        :param target:
        :return: A mapping of action for each agent in the cluster.
        """
        for idx, agent in self.agents.items():
            agent.set_orientation(math.atan(get_slope(target[0], target[1],
                                                      agent.get_position()[0], agent.get_position()[1])))

    def move_to_target(self, target):
        for idx, agent in self.agents.items():
            # print("Agent position: " + str(agent.get_position()))
            # print("Target: " + str(target))
            if abs(agent.get_position()[0] - target[0]) < 1e-10:
                theta = 0
            else:
                slope = get_slope(target[0], target[1], agent.get_position()[0], agent.get_position()[1])
                theta = math.atan(slope)

            factor = -1 if agent.get_position()[0] > target[0] else 1
            agent.action = np.array([math.cos(theta), math.sin(theta)]) * factor

        self.allocated_action = False

    def is_on_target(self):
        return euclidean_dist(self.get_center(), self.target_loc) < 1


    def move_randomly(self):
        """
        Move each agent in the direction of it's orientation.
        :return:
        """
        for agent in self.agents.values():
            # agent.action = np.array([1, 0])
            agent.action = np.random.rand(1,2)

    def merge(self, cluster):
        """
        Merge this cluster with another one.
        :param cluster:
        :return:
        """

        for idx, agent in cluster.agents.items():
            agent.free()
            agent.allocate(self)
            self.agents[idx] = agent

    def free_all(self):
        for agent in self.agents.values():
            agent.free()

    def get_index(self):
        return self._id

    def split(self):
        """
        Checks if the cluster is too further apart that we need to split it to multiple clusters
        :return: None
        """
        pass
