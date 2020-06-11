# form utils import euclidean_dist
import math

from utils import euclidean_dist


def is_concealed_by_agents(source, target, agent_set, robot_radius):
    """
    Check if the given target is concealed from the source by the one of the agents in the agent set.
    :param source:
    :param target:
    :param agent_set:
    :return:
    """
    for agent in agent_set:
        if agent != source and agent != target:
            if fixed_check_collision(source, target, agent, robot_radius):
                return True

    return False

def get_distance_from_sides(sides, edge_length):
    return edge_length / (2 * math.sin(math.pi / sides))

def check_line_collision(a,b,c,center,radius):
    """
    Checks if the given line intersects the circle at the given center and radius.
    :param a:
    :param b:
    :param c:
    :param center:
    :param radius:
    :return:
    """
    dist = ((abs(a*center[0] + b*center[1] + c)) / math.sqrt(a**2 + b**2))
    return radius >= dist


def fixed_check_collision(source, target, obstacle, radius):
    if (target[0] == source[0]):
        if (abs(obstacle[0] - target[0]) > radius):
            return False
        else:
            return obstacle[1] >= source[1] and obstacle[1] <= target[1] or \
                    obstacle[1] >= target[1] and obstacle[1] <= source[1]

    m = (target[1] - source[1]) / (target[0] - source[0])
    a = m
    b = -1
    c = -m * target[0] + target[1]

    left_a = right_a = a
    left_b = right_b = b
    left_c = c - radius
    right_c = c + radius

    if check_line_collision(left_a, left_b, left_c, obstacle, radius) or \
        check_line_collision(right_a, right_b, right_c, obstacle, radius):
            if (obstacle[0] >= source[0] and obstacle[0] <= target[0]) or (obstacle[0] >= target[0] and obstacle[0] <= source[0]) \
                or (obstacle[1] >= source[1] and obstacle[1] <= target[1]) or (obstacle[1] >=target[1] and obstacle[1] <= source[1]):
                return True
            else:
                return False

    x_val = obstacle[0]
    left_y_val = m * x_val + left_c
    right_y_val = m * x_val + right_c

    is_between = (obstacle[1] >= left_y_val and obstacle[1] <= right_y_val) or \
                 (obstacle[1] >= right_y_val and obstacle[1] <= left_y_val)
    return is_between

def can_source_see_target(source, target, other_agents, smin, smax, robot_radius):
    curr_dist = euclidean_dist(source, target)
    return curr_dist >= smin and curr_dist <= smax and not \
        is_concealed_by_agents(source, target, other_agents, robot_radius)


def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)

    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d
        y2=y0+a*(y1-y0)/d
        x3=x2+h*(y1-y0)/d
        y3=y2-h*(x1-x0)/d

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d

        return (x3, y3),(x4, y4)

def filter_locations_based_on_observers(locations, observers, smin, smax):
    loc_map = {loc: False for loc in locations}
    for loc in locations:
        for obs in observers:
            if euclidean_dist(loc, obs) >=smin and euclidean_dist(loc, obs) <= smax:
                loc_map[loc] = True
                break

    locations[:] = [loc for loc in locations if loc_map[loc]]
    return locations