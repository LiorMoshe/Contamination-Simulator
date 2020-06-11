import math
import numpy as np
from defense.defense_utils import get_distance_from_sides, filter_locations_based_on_observers
from defense.monotonic import greedy_placement, get_possible_locations_in_enclosed_area, get_internal_locations


class MonotonicIterator(object):
    """
    Iterates over possible fence sizes for each iteration of a given cluster until there is convergence.
    """


    def __init__(self, num_agents, smin, smax, robot_radius, center=(0,0)):
        self.num_agents = num_agents
        self.smin = smin
        self.smax = smax
        self.robot_radius = robot_radius
        self.fence_size = 3
        self.center = center
        self.converged = False
        self.best_value = 0

    def update_num_agents(self, num_agents):
        self.num_agents = num_agents

    def update_center(self, center):
        self.center = center

    def get_locations(self):
        return self.best_locations

    def iterate(self):
        """
        Perform one iteration of the monotonic heuristic,  for a given fence size compute the optimal
        component. Update if there is any convergence.
        :return:
        """
        print("Start Iter")
        current_fence_size = self.fence_size
        passed = False
        changed_locations = False
        while not passed and current_fence_size < self.num_agents:
            kept_locations = None
            for edge_length in np.arange(2*self.smax-1, self.smin, -1):
                print("FSize: {0} ELength: {1} BestValue: {2} NAgents: {3}".format(current_fence_size, edge_length, self.best_value,
                                                                                   self.num_agents))
                if (int((self.num_agents - current_fence_size) / current_fence_size) < self.best_value - 3):
                    # If the number of agents left is worse than the best value, there is nothing to check.
                    continue

                print("PASSED, Center: {0}".format(self.center))
                passed = True
                # print("Fence Size: {0}, Edge Length: {1}".format(fence_size, edge_length))
                dist_from_center = get_distance_from_sides(current_fence_size, edge_length)
                # First place the members of the fence.
                curr_fence = []
                angle = 2 * math.pi / current_fence_size
                for i in range(current_fence_size):
                    curr_fence.append((self.center[0] + dist_from_center * math.cos(angle * i),
                                       self.center[1] + dist_from_center * math.sin(angle * i)))

                # print("Computed fence: {0}".format(curr_fence))
                # After initializing the fence get all the possible areas in the enclosed area.
                # locations = get_possible_locations_in_enclosed_area(curr_fence, smin, smax, dist_from_center, robot_radius)
                if kept_locations is None:
                    # fence, smin, smax, center, dist_from_center, robot_radius, taken = []
                    # print("FSize: {0}, ELength: {1}".format(c, edge_length))
                    # locations = get_possible_locations_in_enclosed_area\
                    #     (curr_fence, self.smin, self.smax, self.center, dist_from_center, self.robot_radius)
                    locations = get_internal_locations\
                        (curr_fence, self.smin, self.smax, self.center, dist_from_center, self.robot_radius)
                    print("Got kept locations, Number of locations: {0}".format(len(locations)))
                    kept_locations = locations
                else:
                    locations = filter_locations_based_on_observers(kept_locations, curr_fence, self.smin, self.smax)
                    kept_locations = locations

                if (len(locations) < self.num_agents - current_fence_size):
                    print("Skipped")
                    continue
                # Place agents in given locations according
                placed_locations, value = greedy_placement(self.num_agents- current_fence_size,
                                                           curr_fence, locations, self.smin, self.smax,self.robot_radius)
                print("Required locations: {0}, Got: {1}".format(self.num_agents- current_fence_size, len(placed_locations)))
                print("Placed greedily, value: {0}, Best Value: {1}".format(value, self.best_value))
                if value > self.best_value:
                    print("Updated best value.")
                    self.best_value = value
                    self.best_locations = placed_locations
                    print("Num agents: {0}, Locations: {1}".format(self.num_agents, len(self.best_locations)))
                    self.fence_size = current_fence_size
                    changed_locations = True
            current_fence_size += 1

        if (current_fence_size >= self.num_agents):
            self.converged = True

        print("End Iter")
        return changed_locations

    def did_converge(self):
        return self.converged

    def update_num_agents(self, num_agents):
        """
        If the number of agents is updated we need to converge from the beginning.
        :param num_agents:
        :return:
        """
        if (num_agents != self.num_agents):
            self.converged = False
            self.fence_size = 3
            self.best_value = 0
            self.best_locations = []
            self.best_value = 0