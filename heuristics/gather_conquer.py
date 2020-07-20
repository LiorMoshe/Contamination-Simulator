from utils import euclidean_dist, closest_cluster_to
from collections import namedtuple
from clusters.cluster_manager import move_to_clusters_center
from clusters.cluster_manager import ClusterManager
from agent import InternalState
import logging
import numpy as np

"""
Implementation of the gather and conquer heuristic.
"""

# Output of the algorithm.
Target = namedtuple("Target", "group cluster_id from_loc to_loc")

def onion_gathering(healthy_clusters, global_state):
    if len(healthy_clusters) > 1:
        for cluster in healthy_clusters.values():
            cluster.move_to_target(np.array([50,50]))
        return
        # move_to_clusters_center(healthy_clusters)
        # return

    contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()
    for my_cluster in healthy_clusters.values():
        # TODO -Addon code to test onion structure.
        print("Cluster: " + str(my_cluster._id))
        # my_cluster.log_observations(global_state)
        # if my_cluster.onion:
        my_cluster.log_observations(global_state)
        if not my_cluster.converged_onion():
            my_cluster.create_onion_structure()
        else:
            # print("STOPPED")
            # my_cluster.stop()
            print("MOVING TO 10,10")
            my_cluster.move_in_formation(np.array([10, 10]))

            # closest = closest_cluster_to(my_cluster.get_center(), contaminated_clusters)
            # my_cluster.move_in_formation(contaminated_clusters[closest])

        continue

        # if (my_cluster.has_weak_point(global_state) and my_cluster.size() > 3) or my_cluster.size() > 10:
        #     print("Creating onion struct.")
        #     my_cluster.create_onion_structure()
        #     continue

def maximal_stable_cycles(healthy_clusters,  contaminated_clusters, global_state):
    """
    Gather everyone together and then split into maximal stable cycles. Variant of the idea of gather and conquer.
    But always move in the form of a stable cycle.
    :param healthy_clusters:
    :param global_state:
    :return:
    """
    # for idx, cluster in healthy_clusters.items():
    #     if cluster.size() > 6 and cluster.size() < 10:
    #         if not cluster.converged_to_msc():
    #             print("Creating MSC")
    #             cluster.create_msc()
    #         else:
    #             print("STOPPED")
    #             cluster.stop()
    #     elif cluster.size() > 10:
    #
    if len(contaminated_clusters) == 0:
        if len(healthy_clusters) > 1:
            move_to_clusters_center(healthy_clusters)
            return

        gathered_cluster = list(healthy_clusters.values())[0]
        if not gathered_cluster.converged_to_msc():
            print("Creating msc")
            gathered_cluster.create_msc()
        else:
            print("STOPPED")
            gathered_cluster.log_observations(global_state)
            # gathered_cluster.stop()
            gathered_cluster.move_in_formation(np.array([90, 90]))
        return

    targets = {}

    # Sort the enemy clusters by size.
    cluster_idx_to_size = []
    for enemy_cluster_idx, enemy_cluster in contaminated_clusters.items():
        cluster_idx_to_size.append((enemy_cluster_idx, len(enemy_cluster.agents)))

    cluster_idx_to_size.sort(key=lambda tup: tup[1], reverse=True)

    for cluster_idx, cluster_size in cluster_idx_to_size:
        for my_cluster in healthy_clusters.values():

            if not my_cluster.converged_to_msc():
                print("Cluster " + str(my_cluster._id) + " didnt converge to msc.")
                my_cluster.create_msc()
                continue

            if (my_cluster.target_loc is None or my_cluster.is_on_target()):

                if my_cluster.size() > cluster_size:
                    # Assign a new target.
                    my_cluster.target_loc = contaminated_clusters[cluster_idx].get_center()
                    targets[my_cluster.get_index()] = Target(InternalState.CONTAMINATED, cluster_idx,
                                                             from_loc=my_cluster.get_center(),
                                                             to_loc=my_cluster.target_loc)
                    logging.debug(
                        "Cluster " + str(my_cluster.get_index()) + " got assigned to a new contaminated target"
                        + " with id " + str(cluster_idx) + " in location" + str(my_cluster.target_loc))
                else:
                    continue
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

    # Iterate over the enemy clusters by size.
    for cluster_idx, cluster_size in cluster_idx_to_size:

        # Find matching healthy cluster that we can send there.
        for my_cluster in healthy_clusters.values():

            # If the cluster already reached its target, check if its feasible to send it to this one.
            if (my_cluster.target_loc is None or my_cluster.is_on_target()):
                if my_cluster.size() > cluster_size:
                    # Assign a new target.
                    my_cluster.target_loc = contaminated_clusters[cluster_idx].get_center()
                    targets[my_cluster.get_index()] = Target(InternalState.CONTAMINATED, cluster_idx,
                                                                 from_loc=my_cluster.get_center(), to_loc=my_cluster.target_loc)
                    logging.debug("Cluster " + str(my_cluster.get_index()) + " got assigned to a new contaminated target"
                                + " with id " + str(cluster_idx) +  " in location" + str(my_cluster.target_loc))
                    my_cluster.allocated_action = True
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

            # my_cluster.allocated_action = True


    # Handle leftovers, direct them to the closest cluster which is ours in order to make them merge.
    for cluster in healthy_clusters.values():
        if not cluster.allocated_action:
            # print("NO ACTION")
            closest_cluster = closest_cluster_to(cluster.get_center(), healthy_clusters,
                                                 except_indices=[cluster.get_index()])

            if closest_cluster is not None:
                cluster.target_loc = healthy_clusters[closest_cluster].get_center()
                targets[cluster.get_index()] = Target(InternalState.HEALTHY, closest_cluster,
                                                      from_loc=cluster.get_center(), to_loc=cluster.target_loc)
                cluster.allocated_action = False
    return targets