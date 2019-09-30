from clusters.cluster_manager import ClusterManager, move_to_clusters_center
from heuristics.gather_conquer import *
from operator import itemgetter
import logging
import numpy as np
from path_planning.rrt_graph import RRT
from utils import get_slope
from math import sqrt

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

    def msc(self, global_state):
        maximal_stable_cycles(ClusterManager.instance.get_healthy_clusters(),
                              ClusterManager.instance.get_contaminated_clusters(), global_state)

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

    def onion_gathering(self, global_state):
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()
        onion_gathering(healthy_clusters, global_state)

    def strategic_movement(self, global_state):
        """
        Move in stable structures (msc and onion structures) toward the center of all the clusters.
        Once we have a strong stabilized structure conquer the clusters of the adversary.
        :param global_state:
        :return:
        """
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()
        contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()

        if len(healthy_clusters) > 1:
            center_of_clusters = np.sum(np.vstack([cluster.get_center() for cluster in healthy_clusters.values()]),
                             axis=0) / len(healthy_clusters)

            for cluster in healthy_clusters.values():
                if cluster.size() > 3:
                    if not cluster.did_converge():
                        cluster.stabilize_structure()
                    else:
                        cluster.move_in_formation(center_of_clusters)
                else:
                    cluster.move_to_target(center_of_clusters)
        elif len(healthy_clusters) == 1:
            gathered_cluster = list(healthy_clusters.values())[0]
            gathered_cluster.log_observations(global_state)

            if not gathered_cluster.did_converge():
                gathered_cluster.stabilize_structure()
            else:
                if len(contaminated_clusters) > 0:
                    closest = closest_cluster_to(gathered_cluster.get_center(), contaminated_clusters)
                    gathered_cluster.move_in_formation(contaminated_clusters[closest].get_center())
                else:
                    gathered_cluster.stop()


    def compute_rand_area(self):
        """
        Compute bounding area on our clusters.
        :return:
        """
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for cluster in healthy_clusters.values():
            center = cluster.get_center()
            if center[0] < min_x:
                min_x = center[0]

            if center[0] > max_x:
                max_x = center[0]

            if center[1]  < min_y:
                min_y = center[1]

            if center[1] > max_y:
                max_y = center[1]

        return [min_x, max_x, min_y, max_y]

    def check_intrusion(self, healthy_point, contaminated_start, contaminated_target):
        """
        Check if healthy cluster can intrude the path of the contaminated one.
        :param healthy_point:
        :param contaminated_start:
        :param contaminated_target:
        :return: True if intrusion is possible, false otherwise.
        """
        # Compute the defining line of the contaminated agent's path.
        slope = get_slope(contaminated_start[0], contaminated_start[1], contaminated_target[0], contaminated_target[1])
        b = -1
        c = -slope * contaminated_start[0] +contaminated_start[1]

        dist_from_line = abs(slope * healthy_point[0] + b * healthy_point[1] + c) / sqrt(slope ** 2 + b ** 2)
        dist_from_start = euclidean_dist(healthy_point, contaminated_start)

        # Based on pythagoras check if we can intrude the path of the contaminated cluster.
        return dist_from_start ** 2 - dist_from_line ** 2 > dist_from_line ** 2

    def find_intrusion_target(self, healthy_point, contaminated_start, contaminated_target):
        slope = get_slope(contaminated_start[0], contaminated_start[1], contaminated_target[0], contaminated_target[1])

        x = slope * contaminated_start[0] - contaminated_start[1] + (healthy_point[0]) / slope + healthy_point[1]
        y = slope * (x - contaminated_start[0]) + contaminated_start[1]
        return np.array([x, y])

    def lead_to_center_of_contaminated(self, global_state):
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()
        contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()

        if len(contaminated_clusters) > 0:
            center_of_contaminated = np.sum(
                np.vstack([cluster.get_center() for cluster in contaminated_clusters.values()]),
                axis=0) / len(contaminated_clusters)

            for healthy_cluster in healthy_clusters.values():
                healthy_cluster.move_to_target(center_of_contaminated)

    def line_of_defense_heuristic(self, global_state):
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()
        contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()

        if len(healthy_clusters) == 1:
            gathered_cluster = list(healthy_clusters.values())[0]

            # if gathered_cluster.has_weak_point(global_state):
            #     logging.info("Densified")
            #     gathered_cluster.densify()

            # Once we have one big cluster start conquering other enemy clusters which are closest to you.
            if len(contaminated_clusters) > 0:
                closest_cluster = closest_cluster_to(gathered_cluster.get_center(), contaminated_clusters)
                gathered_cluster.move_to_target(contaminated_clusters[closest_cluster].get_center(),
                                                cluster_id=closest_cluster)
            return


        if len(contaminated_clusters) == 0:
            return

        center_of_contaminated = np.sum(np.vstack([cluster.get_center() for cluster in contaminated_clusters.values()]),
                             axis=0) / len(contaminated_clusters)



        ordered_healthy_clusters = sorted([(cluster.get_index(), euclidean_dist(cluster.get_center(), center_of_contaminated))
                                         for cluster in healthy_clusters.values()], key=itemgetter(1), reverse=True)

        # handled_list = []
        leftovers = []
        for healthy_idx, dist_from_center in ordered_healthy_clusters:
            healthy_cluster = healthy_clusters[healthy_idx]

            priorities = {}

            for cont_idx, contaminated_cluster in contaminated_clusters.items():
                if euclidean_dist(contaminated_cluster.get_center(), center_of_contaminated) > dist_from_center \
                    and healthy_cluster.size() > contaminated_cluster.size():
                    # Check if it's feasible to capture this cluster.
                    if self.check_intrusion(healthy_cluster.get_center(), contaminated_cluster.get_center(),
                                            center_of_contaminated):
                        priorities[cont_idx] = contaminated_cluster.size()

            if len(priorities) > 0:
                # Get max priority.
                chosen_cluster = max(priorities.items(), key=itemgetter(1))[0]
                target = self.find_intrusion_target(healthy_cluster.get_center(),
                                                    contaminated_clusters[chosen_cluster].get_center(),
                                                    center_of_contaminated)
                healthy_cluster.move_to_target(target)
                # handled_list.append(healthy_idx)
            else:
                # If there is no defense that can be done, look at clusters which are closer to the center and merge with them.
                leftovers.append(healthy_idx)

        logging.info("Amount of leftovers: " + str(len(leftovers)) + " compared to total clusters: " + str(len(healthy_clusters)))
        # matrix = np.zeros((len(leftovers), len(leftovers)))
        handled = []
        for cluster_idx in leftovers:

            # if cluster_idx not in handled:
            min_dist = float('inf')
            min_idx = None
            for other_cluster_idx in leftovers:
                if cluster_idx != other_cluster_idx and other_cluster_idx not in handled:
                    curr_dist = euclidean_dist(healthy_clusters[cluster_idx].get_center(),
                                                healthy_clusters[other_cluster_idx].get_center())
                    if curr_dist < min_dist:
                        min_dist = curr_dist
                        min_idx = other_cluster_idx

            if min_idx is not None:
                handled.append(min_idx)
                healthy_clusters[cluster_idx].move_to_target(healthy_clusters[min_idx].get_center())
            else:
                healthy_clusters[cluster_idx].move_to_target(healthy_clusters[cluster_idx].get_center())
            handled.append(cluster_idx)

    def rrt_heuristic(self, global_state):
        """
        Build a graph based on RRT from each healthy cluster to their center, while treating centers of contaminated
        clusters as obstacles.
        :param global_state:
        :return:
        """
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()


        if len(healthy_clusters) > 1:
            contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()
            # obstacles = [np.concatenate([cluster.get_center(), [6]]) for cluster in contaminated_clusters.values()]
            center_of_clusters = np.sum(np.vstack([cluster.get_center() for cluster in healthy_clusters.values()]),
                                         axis=0) / len(contaminated_clusters)

            bounding_area =  self.compute_rand_area()
            logging.info("Calculated bounding area of healthy clusters: " + str(bounding_area))

            cluster_to_planner = {}
            no_path_cnt = 0
            for healthy_cluster in healthy_clusters.values():

                # Compute the obstacles of this cluster.
                obstacles = []
                for contaminated_cluster in  contaminated_clusters.values():
                    if healthy_cluster.size() < contaminated_cluster.size():
                        obstacles.append(np.concatenate([contaminated_cluster.get_center(),
                                                         [contaminated_cluster.get_radius()]]))

                rrt_planner = RRT(healthy_cluster.get_center(), center_of_clusters, obstacle_list=obstacles,
                                                                      rand_area=bounding_area, max_iter=100)

                path = rrt_planner.planning()
                if path is None:
                    no_path_cnt += 1
                cluster_to_planner[healthy_cluster.get_index()] = rrt_planner

            for cluster_idx, planner in cluster_to_planner.items():
                curr_cluster = healthy_clusters[cluster_idx]
                if len(planner.node_list) > 1:
                    curr_cluster.move_to_target(planner.node_list[1].location())
                else:
                    curr_cluster.move_to_target(planner.node_list[0].location())
        else:

            contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()
            gathered_cluster = list(healthy_clusters.values())[0]

            if gathered_cluster.has_weak_point(global_state):
                logging.info("Densified")
                gathered_cluster.densify()

            # Once we have one big cluster start conquering other enemy clusters which are closest to you.
            if len(contaminated_clusters) > 0:
                closest_cluster = closest_cluster_to(gathered_cluster.get_center(), contaminated_clusters)
                gathered_cluster.move_to_target(contaminated_clusters[closest_cluster].get_center(),
                                                cluster_id=closest_cluster)
    #
    # def onion_heuristic(self, global_state):
    #     healthy_cluster = ClusterManager.instance.get_healthy_clusters()
    #     contaminated_clusters




    def second_heuristic(self, global_state):
        """
        TODO- Find a good name for the heuristic.
        This heuristic is based on the fact that in order to win we need to have a majority of the players in a cluster.
        We will try to have a little bit than a majority and than gather into a single cluster as fast as possible.
        :return:
        """
        EXTRAS = 0
        healthy_clusters = ClusterManager.instance.get_healthy_clusters()

        for cluster in healthy_clusters.values():
            logging.info("Healthy cluster " + str(cluster.get_index()) + " in location: " + str(cluster.get_center())
                         + " number of agents: " + str(cluster.size()))

        contaminated_clusters = ClusterManager.instance.get_contaminated_clusters()

        logging.info("Healthy agents: " + str(len(global_state.healthy_agents)) + " Contaminated agents: " + str(len(global_state
                                                                                                     .contaminated_agents)))
        logging.info("Healthy clusters: " + str(len(healthy_clusters)) + " Contaminated clusters " + str(len(contaminated_clusters)))

        clusters_center = None

        if len(contaminated_clusters) > 0:
            clusters_center = np.sum(np.vstack([cluster.get_center() for cluster in contaminated_clusters.values()]),
                                     axis=0) / len(contaminated_clusters)
            avg_dist_from_center = 0.0
            avg_contaminated_cluster_size = 0.0

            for contaminated_cluster in contaminated_clusters.values():
                avg_dist_from_center += euclidean_dist(contaminated_cluster.get_center(), clusters_center)
                avg_contaminated_cluster_size += contaminated_cluster.size()

            avg_dist_from_center /= len(contaminated_clusters)
            avg_contaminated_cluster_size /= len(contaminated_clusters)


        if len(global_state.healthy_agents) >= len(global_state.contaminated_agents) + EXTRAS:
            # If we have majority gather all agents.
            if len(healthy_clusters) != 1:
                move_to_clusters_center(healthy_clusters)
            else:
                # Once we have one big cluster start conquering other enemy clusters which are closest to you.
                if len(contaminated_clusters) > 0:
                    gathered_cluster = list(healthy_clusters.values())[0]
                    closest_cluster = closest_cluster_to(gathered_cluster.get_center(), contaminated_clusters)
                    gathered_cluster.move_to_target(contaminated_clusters[closest_cluster].get_center(),
                                                    cluster_id=closest_cluster)
        else:
            # Order healthy clusters by size.
            ordered_healthy_clusters = sorted([(healthy_cluster.get_index(), healthy_cluster.size())
                                               for healthy_cluster in healthy_clusters.values()], key=itemgetter(1),
                                              reverse=True)

            # Allocate actions for healthy clusters based on given size.
            for healthy_idx, cluster_size in ordered_healthy_clusters:
                healthy_cluster = healthy_clusters[healthy_idx]
                priority_map = {}

                for contaminated_idx, contaminated_cluster in contaminated_clusters.items():
                    if cluster_size > contaminated_cluster.size():
                        distance = euclidean_dist(healthy_cluster.get_center(), contaminated_cluster.get_center())
                        priority_map[contaminated_idx] = contaminated_cluster.size() / (distance ** 2)

                # for other_healthy_cluster in

                if len(priority_map) == 0:
                    # If there is no enemy cluster smaller look to merge with closest cluster of our group.
                    closest_cluster = closest_cluster_to(healthy_cluster.get_center(), healthy_clusters,
                                                         except_indices=[healthy_cluster.get_index()])

                    logging.info("Closest cluster: " + str(closest_cluster))
                    healthy_cluster.move_to_target(healthy_clusters[closest_cluster].get_center(),
                                                   cluster_id=closest_cluster)
                else:
                    # Move to cluster with highest priority.
                    chosen_cluster = max(priority_map.items(), key=itemgetter(1))[0]
                    healthy_cluster.move_to_target(contaminated_clusters[chosen_cluster].get_center(),
                                                   cluster_id=chosen_cluster)