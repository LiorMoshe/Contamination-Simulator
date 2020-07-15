"""
Compute the upper bound for the wpc value of a component
under given settings.
"""
import math

SMIN = 2
SMAX = 1
robot_radius = 0.25

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


def get_angle_relative_zero(point):
    """
    Returns angle in degrees relative to (0,0)
    """
    angle = math.atan2(point[1], point[0]) * 180 / math.pi
    return angle if angle > 0 else angle + 360

center = (0,0)
def find_bare_agent_upper_bound():
    other_point = (center[0] + SMAX, center[1])

    intersection, _ = get_intersections(center[0], center[1], SMAX, other_point[0], other_point[1], SMAX)

    other_point_angle = get_angle_relative_zero(other_point)
    intersection_angle = get_angle_relative_zero(intersection)

    if intersection_angle > other_point_angle:
        diff = intersection_angle - other_point_angle
    else:
        diff = other_point_angle - intersection_angle


    large_diff = (360 - 2 * diff) * math.pi / 180
    print("Large Diff: {0}".format(large_diff))
    bound = (large_diff * SMAX) / (2 * robot_radius) + 1
    return bound

if __name__=="__main__":
    print(find_bare_agent_upper_bound())