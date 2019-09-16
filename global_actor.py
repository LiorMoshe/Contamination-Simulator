import numpy as np
from utils import get_slope, euclidean_dist
import math

"""
Different policies that can be used by a global actor. It receives the global state, meaning we know the states
of all the agents in the game and decide on actions for each one of the healthy agents.
"""

id_counter = 1

class AgentCluster(object):

    def __init__(self, id, agents, enemy=False):
        self._id = id
        self.agents = agents
        self.enemy = enemy
        self.allocated_action = False
        self.target_loc = None

        for agent in self.agents.values():
            agent.allocate(self)

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

    def split(self):
        """
        Checks if the cluster is too further apart that we need to split it to multiple clusters
        :return: None
        """


class GlobalActor(object):

    def __init__(self, min_obs_rad, max_obs_rad):

        # Clusters of agents controlled by the global actor.
        self._clusters = {}
        self.min_obs_rad = min_obs_rad
        self.max_obs_rad = max_obs_rad
        self._enemy_clusters = {}


    def analyze(self, global_state):
        """
        Analyze the game based on the global state.
        Which cluster I hold, which ones the enemy holds. Which clusters of mine are in the greatest danger.
        Which cluster should pursuit what? Which one should evade?
        :param global_state
        :return:
        """
        pass

    def clean_all_enemy(self):
        for cluster in self._enemy_clusters.values():
            cluster.free_all()

        self._enemy_clusters = {}

    def clean_clusters(self, clusters, log=False):
        to_be_removed = []
        for cluster_id, cluster in clusters.items():
            cluster.allocated_action = False
            if len(cluster.agents) == 0:
                to_be_removed.append(cluster_id)

        for cluster_id in to_be_removed:
            if log:
                print("Cleaning cluster " + str(cluster_id))
            del clusters[cluster_id]

    @staticmethod
    def find_clusters(agents, clusters, distance_matrix, radius, enemy=False):
        """
        Based the given list of agents return a mapping of the clusters.
        :param agents:
        :param clusters: Reference to where we will save the clusters that we will find.
        :param distance_matrix
        :param enemy
        :return:
        """
        for idx, agent in agents.items():

            distances = []
            for other_idx, other_agent in agents.items():
                distances.append((other_idx, distance_matrix[idx, other_idx],
                                  other_agent.get_cluster_id()))

            distances.sort(key=lambda tup: tup[1])

            # Go over the closest healthy agents and put them together in a cluster.
            cluster_agents = {idx: agent} if not agent.is_allocated() \
                else clusters[agent.get_cluster_id()].agents

            create_cluster = False
            for agent_idx, dist, cluster_id in distances:
                if dist < radius:

                    if agent.is_allocated() and clusters[agent.get_cluster_id()].has_agent(agent_idx):
                        continue

                    # Merge two clusters together if possible.
                    if cluster_id is not None and agent.is_allocated():
                        clusters[agent.get_cluster_id()].merge(clusters[cluster_id])
                        del clusters[cluster_id]

                    elif cluster_id is not None:
                        clusters[cluster_id].add_agent(agent)
                    elif agent.is_allocated():
                        clusters[agent.get_cluster_id()].add_agent(agents[agent_idx])
                    else:
                        cluster_agents[agent_idx] = agents[agent_idx]
                        create_cluster = True

            global  id_counter
            if create_cluster:
                clusters[id_counter] = AgentCluster(id_counter, cluster_agents, enemy=enemy)
                id_counter += 1


    def closest_enemy_cluster_to_target(self, target):
        min_dist = float('inf')
        min_idx = None
        for idx, enemy_cluster in self._enemy_clusters.items():
            curr_dist = euclidean_dist(enemy_cluster.get_center(), target)
            if curr_dist < min_dist:
                min_dist = curr_dist
                min_idx = idx

        return min_idx


    def act(self, global_state):
        """
        Given the global state return actions for all the healthy agents which we control.
        Update the clusters in the process
        :param global_state:
        :return:
        """
        self.clean_clusters(self._clusters)
        self.clean_all_enemy()

        # Update the clusters, both ours and the enemies.
        self.find_clusters(global_state.healthy_agents, self._clusters, global_state.distance_matrix, 2 * self.max_obs_rad)
        self.find_clusters(global_state.contaminated_agents, self._enemy_clusters, global_state.distance_matrix,
                           2 * self.max_obs_rad, enemy=True)

        # Sort the enemy clusters by size.
        cluster_idx_to_size = []
        for enemy_cluster_idx, enemy_cluster in self._enemy_clusters.items():
            cluster_idx_to_size.append((enemy_cluster_idx, len(enemy_cluster.agents)))

        cluster_idx_to_size.sort(key= lambda tup: tup[1], reverse=True)

        # Allocate target clusters for each one of my clusters.
        for cluster_idx, cluster_size in cluster_idx_to_size:
            for my_cluster in self._clusters.values():
                if my_cluster.target_loc is None or my_cluster.is_on_target():
                    my_cluster.target_loc = self._enemy_clusters[cluster_idx].get_center()
                else:
                    my_cluster.target_loc = self._enemy_clusters[
                        self.closest_enemy_cluster_to_target(my_cluster.target_loc)].get_center()

                my_cluster.move_to_target(my_cluster.target_loc)
                my_cluster.allocated_action = True

        # Handle leftovers.
        for cluster in self._clusters.values():
            if not cluster.allocated_action and len(cluster_idx_to_size) > 0:
                cluster.move_to_target(self._enemy_clusters[cluster_idx_to_size[0]])
                cluster.allocated_action = True



        # self.log_clusters()

    def log_clusters(self):
        print("-------- CLUSTER LOG START----------")
        for cluster_id, cluster in self._clusters.items():
            print("Cluster id " + str(cluster_id) + " number of agents in cluster  " + str(len(cluster.agents)))
            print("Keys: " + str(cluster.agents.keys()))

        print("------- CLUSTER LOG END -----")



