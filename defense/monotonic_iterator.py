import math
import numpy as np
from defense.defense_utils import get_distance_from_sides, filter_locations_based_on_observers
from defense.monotonic import greedy_placement, get_possible_locations_in_enclosed_area


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
        # msc_size = math.floor(
        #     (math.pi * self.smax) / ((self.smax / 2) * math.acos(1 - (2 * (self.smin ** 2)) / (self.smax ** 2))))
        #
        # self.best_locations = []
        # # if (self.num_agents <= msc_size):
        #     self.converged = True
        #     self.best_value = num_agents
        # else:
        self.converged = False
        self.best_value = 0

    def update_center(self, center):
        self.center = center

    def iterate(self):

        current_fence_size = self.fence_size
        passed = False
        changed_locations = False
        while not passed and current_fence_size < self.num_agents:
            kept_locations = None
            for edge_length in np.arange(2*self.smax-1, self.smin, -1):
                if (int((self.num_agents - current_fence_size) / current_fence_size) < self.best_value - 3):
                    # print("FSize: {0} E: {1} Remaining: {2}, Best: {3}, Skipping".format(fence_size, edge_length, n-fence_size,best_value))
                    # If the number of agents left is worse than the best value, there is nothing to check.
                    continue


                passed = True
                # print("Fence Size: {0}, Edge Length: {1}".format(fence_size, edge_length))
                dist_from_center = get_distance_from_sides(self.fence_size, edge_length)
                # First place the members of the fence.
                curr_fence = []
                angle = 2 * math.pi / self.fence_size
                for i in range(current_fence_size):
                    curr_fence.append((dist_from_center * math.cos(angle * i), dist_from_center * math.sin(angle * i)))

                # print("Computed fence: {0}".format(curr_fence))
                # After initializing the fence get all the possible areas in the enclosed area.
                # locations = get_possible_locations_in_enclosed_area(curr_fence, smin, smax, dist_from_center, robot_radius)
                if kept_locations is None:
                    # fence, smin, smax, center, dist_from_center, robot_radius, taken = []
                    # print("FSize: {0}, ELength: {1}".format(c, edge_length))
                    locations = get_possible_locations_in_enclosed_area\
                        (curr_fence, self.smin, self.smax, self.center, dist_from_center, self.robot_radius)
                    kept_locations = locations
                else:
                    locations = filter_locations_based_on_observers(kept_locations, curr_fence, self.smin, self.smax)
                    kept_locations = locations

                if (len(locations) < self.num_agents - current_fence_size):
                    continue
                # Place agents in given locations according
                placed_locations, value = greedy_placement(self.num_agents- current_fence_size,
                                                           curr_fence, locations, self.smin, self.smax,self.robot_radius)
                # print("Placed greedily, value: {0}, Best Value: {1}".format(value, best_value))
                if value > self.best_value:
                    self.best_value = value
                    self.best_locations = placed_locations + curr_fence
                    changed_locations = True
            current_fence_size += 1

        if (current_fence_size >= self.num_agents):
            self.converged = True

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