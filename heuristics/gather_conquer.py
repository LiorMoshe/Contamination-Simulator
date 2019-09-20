from utils import euclidean_dist
from collections import namedtuple
from agent import InternalState
import logging

"""
Implementation of the gather and conquer heuristic.
"""

# Output of the algorithm.
Target = namedtuple("Target", "group cluster_id from_loc to_loc")


def closest_cluster_to(target, clusters, except_indices=[]):
    min_dist = float('inf')
    min_idx = None
    # clusters = self._enemy_clusters if enemy else self._clusters
    for idx, cluster in clusters.items():
        curr_dist = euclidean_dist(cluster.get_center(), target)
        if curr_dist < min_dist and idx not in except_indices:
            min_dist = curr_dist
            min_idx = idx

    return min_idx

def gather_and_conquer(healthy_clusters, contaminated_clusters):
    """
    Given the groups of healthy and contaminated clusters run the gather and conquer heuristic and return
    a mapping such that each healthy cluster has a target cluster which is either healthy (in order to gather)
    or contaminated (in order to conquer).
    :param healthy_clusters:
    :param contaminated_clusters:
    :return: Mapping from each healthy cluster to it's target cluster.
    """
    targets = {}

    # Sort the enemy clusters by size.
    cluster_idx_to_size = []
    for enemy_cluster_idx, enemy_cluster in contaminated_clusters.items():
        cluster_idx_to_size.append((enemy_cluster_idx, len(enemy_cluster.agents)))

    cluster_idx_to_size.sort(key=lambda tup: tup[1], reverse=True)

    for cluster_idx, cluster_size in cluster_idx_to_size:
        for my_cluster in healthy_clusters.values():

            if (my_cluster.target_loc is None or my_cluster.is_on_target()):

                if my_cluster.size() > cluster_size:
                    # Assign a new target.
                    my_cluster.target_loc = contaminated_clusters[cluster_idx].get_center()
                    targets[my_cluster.get_index()] = Target(InternalState.CONTAMINATED, cluster_idx,
                                                                 from_loc=my_cluster.get_center(), to_loc=my_cluster.target_loc)
                    logging.debug("Cluster " + str(my_cluster.get_index()) + " got assigned to a new contaminated target"
                                + " with id " + str(cluster_idx) +  " in location" + str(my_cluster.target_loc))
                else: continue
            else:
                # If there is currently a target continue until the cluster gets there.
                closest_cluster_to_target = closest_cluster_to(my_cluster.target_loc, contaminated_clusters)

                if closest_cluster_to_target is None:
                    my_cluster.target_loc = None
                    continue

                my_cluster.target_loc = contaminated_clusters[closest_cluster_to_target].get_center()
                targets[my_cluster.get_index()] = Target(InternalState.CONTAMINATED, closest_cluster_to_target,
                                                         from_loc=my_cluster.get_center(), to_loc=my_cluster.target_loc)

            my_cluster.allocated_action = True


    # Handle leftovers, direct them to the closest cluster which is ours in order to make them merge.
    for cluster in healthy_clusters.values():
        if not cluster.allocated_action:
            closest_cluster = closest_cluster_to(cluster.get_center(), healthy_clusters,
                                                 except_indices=[cluster.get_index()])

            if closest_cluster is not None:
                cluster.target_loc = healthy_clusters[closest_cluster].get_center()
                targets[cluster.get_index()] = Target(InternalState.HEALTHY, closest_cluster,
                                                      from_loc=cluster.get_center(), to_loc=cluster.target_loc)
                cluster.allocated_action = False
    return targets