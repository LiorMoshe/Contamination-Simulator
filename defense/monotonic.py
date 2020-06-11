

"""
Contains a simple implementation of the monotonic heuristic which will be used by agents to defend themselves.
There are two main placing functions which are used to find possible placement areas:
1. get_possible_locations_in_enclosed_area: Compute accurately all the locations inside the enclosed
area - more costly.
2. get_internal_locations: Get all the locations inside of the component of the given fence based on the values
of smin and smax.
"""
from operator import itemgetter

from defense.position_circles import positionCircles
from defense.defense_utils import get_intersections, can_source_see_target, get_distance_from_sides, \
    filter_locations_based_on_observers
from utils import euclidean_dist
import math
import numpy as np
import time
import copy


def get_internal_locations(observers, smin, smax, center, limit_dist, robot_radii, taken=[]):
    """
    Given a list of observers and their observation radii return all the possible locations in which agents can be
    placed (this only includes location that has one or more observers).
    :return: All possible positions of placing an agent (may contain duplicates).
    """
    total_positions = []
    for observer in observers:
        positions = positionCircles(observer, smax, robot_radii)
        for pos in positions:
            if (euclidean_dist(pos, observer) >= smin and euclidean_dist(pos, center) < limit_dist - 1):
                is_valid = True
                for other in taken:
                    dist = euclidean_dist(pos, other)
                    if (dist < robot_radii * 2):
                        is_valid = False

                if is_valid:
                    total_positions.append((pos[0], pos[1]))
    return total_positions

# loc_times = 0
# total_loc = 0.
def get_possible_locations_in_enclosed_area(fence, smin, smax, center, dist_from_center, robot_radius, taken=[]):
    """
    Get all the locations in thee enclosed area.
    """
    # global loc_times
    # global total_loc
    # start = time.time()
    # loc_times += 1
    total_positions = []

    # Compute the intersection points and draw circles from them.
    bare_intersections = []
    point_to_intersections = {}
    for idx, point in enumerate(fence):

        # Get the neighboring points.
        prev_point = fence[(idx - 1)  % len(fence)]
        next_point = fence[(idx + 1) % len(fence)]

        # Get intersection points of the observation areas.
        prev_first, prev_second = get_intersections(prev_point[0], prev_point[1], smax, point[0], point[1],smax)
        next_first, next_second = get_intersections(point[0], point[1],smax, next_point[0], next_point[1],smax)

        # Choose the distance intersection point for the previous neighbor.
        if (prev_first[0] ** 2 + prev_first[1] ** 2 < prev_second[0] ** 2 + prev_second[0] ** 2):
            distant_prev = prev_second
        else:
            distant_prev = prev_first

        # Choose the distance intersection point for the next neighbor.
        if (next_first[0] ** 2 + next_first[1] ** 2 < next_second[0] ** 2 + next_second[1] ** 2):
            distant_next = next_second
        else:
            distant_next = next_first


        # Add both intersection points to the list of distance intersection points.
        if distant_prev not in bare_intersections:
            bare_intersections.append(distant_prev)

        if distant_next not in bare_intersections:
            bare_intersections.append(distant_next)

        point_to_intersections[point] = [distant_prev, distant_next]

    for idx in range(0, len(fence)):
        point = fence[idx]

        prev_intersection, next_intersection = point_to_intersections[point]
        positions = positionCircles(point, smax, robot_radius)

        for pos in positions:
            first_int_dist = euclidean_dist(pos, prev_intersection)
            second_int_dist = euclidean_dist(pos, next_intersection)
            center_dist = euclidean_dist(pos, center)
            dist_to_agent = euclidean_dist(pos, point)


            # Add all the points that are  closer to the center and out of the observation area of the intersection point.
            if first_int_dist > smax and second_int_dist > smax and center_dist < dist_from_center \
                    and dist_to_agent >= smin:
                is_valid = True
                for other in taken:
                    current_dist = euclidean_dist(pos, other)
                    if (current_dist <= robot_radius * 2 + 0.5):
                        is_valid = False

                if is_valid:
                    total_positions.append(pos)
    # total_loc += time.time() - start
    return total_positions


greedy_times = 0
total_greedy = 0.
def greedy_placement(num_agents, fence, locations, smin, smax, robot_radius):
    """
    Place agents greedily assuming the given fence is one of a monotonic component.
    :param fence:
    :param locations:
    :param smin:
    :param smax:
    :param robot_radius:
    :return:
    """
    taken = copy.deepcopy(fence)
    total_val = 0
    while num_agents > 0:
        # print("Greedy num agents: {0}".format(num_agents))
        best_value = float('-inf')
        best_locations = []
        # Go over all possibilities.
        location_to_observers = {location: 0 for location in locations}
        for location in locations:
            if location not in taken:
                # if edge_length > smax or edge_length < smin:
                agent_to_observed = {agent: 0 for agent in fence}
                # else:
                #     agent_to_observed = {agent: 2 for agent in fence}
                taken.append(location)
                # Count number of agents observing now.

                for agent in fence:
                    for pos in taken:
                        if agent != pos and can_source_see_target(source=agent, target=pos, other_agents=taken,
                                                                  smin=smin, smax=smax, robot_radius=robot_radius):
                            agent_to_observed[agent] += 1
                            if pos in location_to_observers:
                                location_to_observers[pos] += 1


                min_choice = min(agent_to_observed.items(), key=itemgetter(1))

                min_agent = min_choice[0]
                min_value = min_choice[1]

                # print("Min value: {0}, Best Value: {1}".format(min_value, best_value))
                if min_value > best_value:
                    best_value = min_value
                    best_locations = [location]

                elif min_value == best_value:
                    best_locations.append(location)

                taken.remove(location)

        if len(best_locations) != 0:

            if len(best_locations) == 1:
                best_loc = best_locations[0]
            else:
                # Break ties according to number of observers and distance from the center.
                max_observed = []
                max_obs_factor = float('-inf')
                for loc in best_locations:
                    curr_val = location_to_observers[loc]
                    if curr_val > max_obs_factor:
                        max_obs_factor = curr_val
                        max_observed = [loc]
                    elif curr_val == max_obs_factor:
                        max_observed.append(loc)

                if len(max_observed) == 1:
                    best_loc = max_observed[0]
                else:
                    # Break remaining ties based on center dist.
                    closest_dist = float('inf')
                    for loc in max_observed:
                        center_dist = math.sqrt(loc[0] ** 2 + loc[1] ** 2)
                        if center_dist < closest_dist:
                            best_loc = loc



            num_agents -= 1
            taken.append(best_loc)
            total_val = best_value
        else:
            break

    return taken, total_val


def monotonic_heuristic(n, smin, smax, robot_radius, placing_func=get_internal_locations):
    msc_size = math.floor((math.pi * smax) / ((smax / 2) * math.acos(1 - (2 * (smin ** 2)) / (smax ** 2))))
    if (n <= msc_size):
        # This can be a clique, irrelevant to use.
        return [], n + 1

    best_value = float('-inf')
    best_locations = None
    for fence_size in range(3, n):
        kept_locations = None
        for edge_length in np.arange(2*smax-1, smin, -1):
            if (int((n - fence_size) / fence_size) < best_value - 3):
                # print("FSize: {0} E: {1} Remaining: {2}, Best: {3}, Skipping".format(fence_size, edge_length, n-fence_size,best_value))
                # If the number of agents left is worse than the best value, there is nothing to check.
                continue
            # print("Fence Size: {0}, Edge Length: {1}".format(fence_size, edge_length))
            dist_from_center = get_distance_from_sides(fence_size, edge_length)
            # First place the members of the fence.
            curr_fence = []
            angle = 2 * math.pi / fence_size
            for i in range(fence_size):
                curr_fence.append((dist_from_center * math.cos(angle * i), dist_from_center * math.sin(angle * i)))

            # print("Computed fence: {0}".format(curr_fence))
            # After initializing the fence get all the possible areas in the enclosed area.
            # locations = get_possible_locations_in_enclosed_area(curr_fence, smin, smax, dist_from_center, robot_radius)
            if kept_locations is None:
                # fence, smin, smax, center, dist_from_center, robot_radius, taken = []
                print("FSize: {0}, ELength: {1}".format(fence_size, edge_length))
                locations = placing_func(curr_fence, smin, smax, (0, 0), dist_from_center, robot_radius)
                kept_locations = locations
            else:
                locations = filter_locations_based_on_observers(kept_locations, curr_fence, smin, smax)
                kept_locations = locations

            if (len(locations) < n - fence_size):
                continue
            # Place agents in given locations according
            placed_locations, value = greedy_placement(n - fence_size, curr_fence, locations, smin, smax,robot_radius)
            # print("Placed greedily, value: {0}, Best Value: {1}".format(value, best_value))
            if value > best_value:
                best_value = value
                best_locations = placed_locations + curr_fence

    # Return the locations and the wpc value which is lower bounded by the lowest connectivity factor + 1.
    return best_locations, best_value + 1

if __name__=="__main__":
    import time
    start = time.time()
    # locations = general_monotonic_heuristic(20, 2, 6, 0.25, get_internal_locations)
    locations, value = monotonic_heuristic(20, 2, 6, 0.25, get_possible_locations_in_enclosed_area)
    diff = time.time() - start
    print("Final Locations: {0}".format(locations))
    print("Diff: {0}, Final Value: {1}".format(diff, value))


    print("Loc Times: {0}, Greedy Times: {1}".format(loc_times, greedy_times))
    print("Average Loc: {0}, Average Greedy: {1}".format(total_loc / loc_times, total_greedy / greedy_times))