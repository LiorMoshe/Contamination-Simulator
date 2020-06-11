from clusters.agent_cluster import AgentCluster
import numpy as np
from utils import euclidean_dist

id_counter = 1

class ClusterManager(object):
    """
    A singleton which manages all the clusters that we have during the simulation.
    """

    instance = None

    def __init__(self, smin, smax, robot_radius):
        if not ClusterManager.instance:
            ClusterManager.instance = ClusterManager.__ClusterManager(smin, smax, robot_radius)

    class __ClusterManager(object):

        def __init__(self, smin, smax, robot_radius):
            self.smin = smin
            self.smax = smax
            self.robot_radius = robot_radius
            self.reset()

        def reset(self):
            global  id_counter
            id_counter = 1
            self._healthy_clusters = {}
            self._contaminated_clusters = {}

        def update_clusters(self, global_state):
            self.clean_clusters(self._healthy_clusters)
            self.clean_all_enemy()

            self.find_clusters(global_state.healthy_agents, self._healthy_clusters, global_state.distance_matrix,
                               2 * self.smax)
            self.find_clusters(global_state.contaminated_agents, self._contaminated_clusters, global_state.distance_matrix,
                               2 * self.smax, enemy=True)

        def get_num_agents(self, cluster_id):
            if cluster_id in self._healthy_clusters:
                return self._healthy_clusters[cluster_id].get_num_agents()
            elif cluster_id in self._contaminated_clusters:
                return self._contaminated_clusters[cluster_id].get_num_agents()
            else:
                raise Exception("Cluster {0} is not saved in the ClusterManager!".format(cluster_id))

        def clean_all_enemy(self):
            for cluster in self._contaminated_clusters.values():
                cluster.free_all()

            self._contaminated_clusters = {}

        def clean_clusters(self, clusters):
            to_be_removed = []
            for cluster_id, cluster in clusters.items():
                cluster.allocated_action = False
                if len(cluster.agents) == 0:
                    to_be_removed.append(cluster_id)

            for cluster_id in to_be_removed:
                del clusters[cluster_id]

        def get_healthy_clusters(self):
            return self._healthy_clusters

        def get_contaminated_clusters(self):
            return self._contaminated_clusters

        def add_healthy_cluster(self, cluster):
            self._healthy_clusters[cluster.get_index()] = cluster

        def add_contaminated_cluster(self, cluster):
            self._contaminated_clusters[cluster.get_index()] = cluster

        def remove_healthy_cluster(self, cluster_id):
            del self._healthy_clusters[cluster_id]

        def remove_contaminated_cluster(self, cluster_id):
            del self._contaminated_clusters[cluster_id]

        def allocate_cluster(self, agents):
            global id_counter
            id_counter+=1
            return AgentCluster(id_counter, agents, self.smin, self.smax, self.robot_radius, enemy=False)

        def find_clusters(self, agents, clusters, distance_matrix, radius, enemy=False):
            """
            Based the given list of agents return a mapping of the clusters.
            :param agents: Mapping of idx to agent
            :param clusters: Reference to where we will save the clusters that we will find.
            :param distance_matrix
            :param enemy Whether we need to find our cluster or enemy's clusters.
            :return:
            """
            for idx, agent in agents.items():

                distances = []
                for other_idx, other_agent in agents.items():
                    distances.append((other_idx, distance_matrix[idx, other_idx],
                                      other_agent.get_cluster_id()))

                distances.sort(key=lambda tup: tup[1])

                # Go over the closest healthy agents and put them together in a cluster.
                # TODO - There is a bug in here
                cluster_agents = {idx: agent}
                if agent.is_allocated():
                    if agent.get_cluster_id() in clusters:
                        cluster_agents = clusters[agent.get_cluster_id()].agents
                    else:
                        agent.free()
                # cluster_agents = {idx: agent} if not agent.is_allocated() \
                #     else clusters[agent.get_cluster_id()].agents
                create_cluster = False
                for agent_idx, dist, cluster_id in distances:
                    if dist < radius:

                        if agent.is_allocated() and clusters[agent.get_cluster_id()].has_agent(agent_idx):
                            continue

                        # Merge two clusters together if possible.
                        if cluster_id is not None and agent.is_allocated() and cluster_id in clusters:
                            if clusters[agent.get_cluster_id()].mergeable(clusters[cluster_id]):
                                res = clusters[agent.get_cluster_id()].merge(clusters[cluster_id])
                                if res:
                                    del clusters[cluster_id]

                        elif cluster_id is not None and cluster_id in clusters:
                            clusters[cluster_id].add_agent(agent)
                        elif agent.is_allocated():
                            clusters[agent.get_cluster_id()].add_agent(agents[agent_idx])
                        else:
                            cluster_agents[agent_idx] = agents[agent_idx]
                            create_cluster = True

                global id_counter
                if create_cluster:
                    clusters[id_counter] = AgentCluster(id_counter, cluster_agents, self.smin, self.smax,
                                                        self.robot_radius, enemy=enemy)
                    id_counter += 1


                # Make sure there aren't nearly identical clusters.
                to_be_removed = []
                for first_cluster in clusters.values():
                    for second_cluster in clusters.values():
                        if first_cluster.get_index() != second_cluster.get_index() and \
                            euclidean_dist(first_cluster.get_center(), second_cluster.get_center()) < 1:
                            res = first_cluster.merge(second_cluster)
                            if res:
                                to_be_removed.append(second_cluster.get_index())

                # logging.info("Clusters to be removed: " + str(to_be_removed))
                for index in set(to_be_removed):
                    del clusters[index]





def move_to_clusters_center(clusters):
    clusters_center = np.sum(np.vstack([cluster.get_center() for cluster in clusters.values()]),
                             axis=0) / len(clusters)

    for cluster in clusters.values():
        cluster.move_to_target(clusters_center)