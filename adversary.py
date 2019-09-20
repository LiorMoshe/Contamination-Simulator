from clusters.cluster_manager import ClusterManager
import numpy as np

class Adversary(object):

    def gather_all(self):
        """
        Gather all agents into the same cluster and then direct them to the center of all the clusters. Once we have
        one cluster direct it to the other clusters.
        :return:
        """
        clusters = ClusterManager.instance.get_contaminated_clusters()

        clusters_center = np.sum(np.vstack([cluster.get_center() for cluster in clusters.values()]), axis=0) / len(clusters)

        for cluster in clusters.values():
            cluster.move_to_target(clusters_center)
