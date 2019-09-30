from clusters.cluster_manager import ClusterManager, move_to_clusters_center
from heuristics.gather_conquer import onion_gathering
from heuristics.gather_conquer import gather_and_conquer
import logging

class Adversary(object):

    def gather_all(self):
        """
        Gather all agents into the same cluster and then direct them to the center of all the clusters. Once we have
        one cluster direct it to the other clusters.
        :return:
        """
        clusters = ClusterManager.instance.get_contaminated_clusters()

        if len(clusters) > 0:
            move_to_clusters_center(clusters)

    def onion_gathering(self, global_state):
        onion_gathering(ClusterManager.instance.get_contaminated_clusters(), global_state)

    def gather_and_conquer(self):
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()
        contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()

        target_map = gather_and_conquer(contaminated_clusters, healthy_clusters)

        for contaminated_cluster_id, target in target_map.items():
            target_cluster = target.group.name
            logging.debug("The cluster " + str(contaminated_cluster_id) + " is moving to cluster " + str(target.cluster_id)
                          + " which is " + str(target_cluster) + " in location: " + str(target.to_loc))
            curr_cluster = contaminated_clusters[contaminated_cluster_id]
            curr_cluster.move_to_target(curr_cluster.target_loc)

