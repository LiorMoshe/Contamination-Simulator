import math
import random
import matplotlib.pyplot as plt

# pretty tuple indices
X = 0
Y = 1
RADIUS = 2

# I've never seen good documentation for the following type of parameter and I'm afraid that
# this is no exception. Briefly, after sorting the radii in descending order by size, the list
# is split along SORT_PARAM_1 and the second piece # is randomized. The pieces are then added
#  back together and the list is split along SORT_PARAM_2 and the first piece is
# shuffled. The lists are then added together and returned.

SORT_PARAM_1 = .80
SORT_PARAM_2 = .10
# (1, 0) = totally sorted   - appealing border, very dense center, sparse midradius
# (0, 1), (1, 1) = totally randomized  - well packed center, ragged border

# these constants control how close our points are placed to each other
RADIAL_RESOLUTION = .25
ANGULAR_RESOLUTION = math.pi / 4

# this keeps the boundaries from touching
PADDING = 0


def assert_no_intersections(f):
    def asserter(*args, **kwargs):
        circles = f(*args, **kwargs)
        intersections = 0
        for c1 in circles:
            for c2 in circles:
                if c1 is not c2 and distance(c1, c2) < c1[RADIUS] + c2[RADIUS]:
                    intersections += 1
                    break
        print
        "{0} intersections".format(intersections)
        if intersections:
            raise AssertionError('Doh!')
        return circles

    return asserter


@assert_no_intersections
def positionCircles(center, circle_radii, radius, angular_res=ANGULAR_RESOLUTION, radial_res=RADIAL_RESOLUTION):
    points = base_points(center, angular_res, radial_res)
    # print("Finished base points")
    free_points = []
    initial = center + (radius,)
    circles = [initial]
    point_count = 0
    while True:
        i, L = 0, len(free_points)
        while i < L:
            if available(circles, free_points[i], radius):
                make_circle(free_points.pop(i), radius, circles, free_points)
                break
            else:
                i += 1
        else:
            exit = False
            for current_radii, point in points:
                if current_radii > circle_radii:
                    exit = True
                    break
                # print("Point: ", point)
                point_count += 1
                if available(circles, point, radius):
                    make_circle(point, radius, circles, free_points)
                    break
                else:
                    if not contained(circles, point):
                        free_points.append(point)
            if exit:
                break

    circles.remove(initial)
    return circles

def make_circle(point, radius, circles, free_points):
    new_circle = point + (radius,)
    circles.append(new_circle)
    i = len(free_points) - 1
    while i >= 0:
        if contains(new_circle, free_points[i]):
            free_points.pop(i)
        i -= 1


def available(circles, point, radius):
    for circle in circles:
        if distance(point, circle) < radius + circle[RADIUS] + PADDING:
            return False
    return True


def base_points(center, radial_res, angular_res):
    """
    Get all possible points, might be some places where they override one another.
    Based on the resolution iterates based on different angles and radiis.
    :param center:
    :param radial_res:
    :param angular_res:
    :return:
    """
    circle_angle = 2 * math.pi
    r = 0
    while 1:
        theta = 0
        while theta <= circle_angle:
            yield r, (center[0] + r * math.cos(theta), center[1] + r * math.sin(theta))
            r_ = math.sqrt(r) if r > 1 else 1
            theta += angular_res / r_
        r += radial_res


def distance(p0, p1):
    return math.sqrt((p0[X] - p1[X]) ** 2 + (p0[Y] - p1[Y]) ** 2)


def contains(circle, point):
    return distance(circle, point) < circle[RADIUS] + PADDING


def contained(circles, point):
    return any(contains(c, point) for c in circles)

if __name__=="__main__":
    center = (0,0)
    smax = 6
    robot_radius = 0.25

    tikz_txt = "\\draw[] {(0, 0) circle (6) node {}};\n"

    positions = positionCircles(center, smax, robot_radius)
    print("Finished positioning all circles")
    # print(positions)

    for pos in positions:
        tikz_txt += "\\draw[dashed] {(" + str(pos[0])+","+str(pos[1])+")" + " circle (" + str(robot_radius) + ") node {}};\n"
        # print(pos)

    print(tikz_txt)
    fig, ax = plt.subplots(figsize=(50, 50))
    ax.add_artist(plt.Circle(center, smax,color='g', fill=False))

    for pos in positions:
        ax.add_artist(plt.Circle(pos, robot_radius, color='r', fill=False))

    plt.ylim((-6,6))
    plt.xlim((-6,6))
    plt.show()