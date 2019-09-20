from clusters.cluster_manager import ClusterManager
from heuristics.gather_conquer import *

"""
Different policies that can be used by a global actor. It receives the global state, meaning we know the states
of all the agents in the game and decide on actions for each one of the healthy agents.
"""


class GlobalActor(object):

    def __init__(self, min_obs_rad, max_obs_rad):

        # Clusters of agents controlled by the global actor.
        self._clusters = {}
        self.min_obs_rad = min_obs_rad
        self.max_obs_rad = max_obs_rad
        self._enemy_clusters = {}

    def reset(self):
        self._clusters = {}
        self._enemy_clusters = {}

    def gather_conquer_act(self):
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()
        contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()

        target_map = gather_and_conquer(healthy_clusters, contaminated_clusters)

        for healthy_cluster_id, target in target_map.items():
            target_cluster = target.group.name
            logging.debug("The cluster " + str(healthy_cluster_id) + " is moving to cluster " + str(target.cluster_id)
                          + " which is " + str(target_cluster) + " in location: " + str(target.to_loc))
            curr_cluster = healthy_clusters[healthy_cluster_id]
            curr_cluster.move_to_target(curr_cluster.target_loc)



