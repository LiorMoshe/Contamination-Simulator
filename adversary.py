from clusters.cluster_manager import ClusterManager, move_to_clusters_center
from heuristics.gather_conquer import onion_gathering

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

