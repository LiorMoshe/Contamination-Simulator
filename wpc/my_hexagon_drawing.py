import matplotlib.pyplot as plt
import math
import threading
from utils import to_rad, euclidean_dist
from shapely.geometry import LineString, Point, MultiPoint
from matplotlib.patches import Polygon
import numpy as np
import copy
# from wpc.concave_hull import alpha_shape



DEBOUNCE_DUR = 0.25
t = None


hexagon = [(0,0),(0,1),(1,2),(2,1),(2,0),(1,-1),]

SMIN = 2.0
SMAX = 6.0
ROBOT_RADIUS = max(SMIN / 4, 0.25)
ROBOT_RADIUS = 0.2

# Counter for agents indices.   
agent_cnt = 1

# Mapping of all the added agents, maps agent index to location
agent_mapping = {}
edge_list = []
border = None
text_patch = None
center = None

class AgentObject(object):

    def __init__(self, idx, loc):
        self.idx = idx
        self.loc = loc
        self.edges = []

    def add_edge(self, edge, plot):
        self.edges.append(edge)

    def set_edges(self, edges):
        self.edges = edges
        return self

    def remove_edge(self, edge):
        self.edges.remove(edge)

    def get_point(self):
        """
        Get the agents location as Shapely's point.
        :return:
        """
        return Point(self.loc[0], self.loc[1])

    def get_loc(self):
        return self.loc

    def get_num_edges(self):
        return len(self.edges)

class AgentDrawing(AgentObject):

    def __init__(self, idx, loc, outer_color='g', inner_color='g'):
        super(AgentDrawing, self).__init__(idx, loc)
        self.update_loc(loc, inner_color=inner_color, outer_color=outer_color)
        self.edges = {}

    def update_loc(self, loc, inner_color='g', outer_color='g'):
        x, y = loc
        self.loc = loc
        self.robot_circle = plt.Circle((x, y), ROBOT_RADIUS, color=inner_color, fill=True)
        self.circles = [plt.Circle((x, y), SMIN, color='r', fill=False),
                        plt.Circle((x, y), SMAX, color=outer_color, fill=False)]


    def draw(self, ax):
        for circle in self.circles:
            ax.add_artist(circle)
        ax.add_artist(self.robot_circle)

    # def mark_as_hull(self, ax):
    #     self.robot_circle.remove()
    #     self.robot_circle = plt.Circle((self.loc[0], self.loc[1]), ROBOT_RADIUS, color='k', fill=True)
    #     ax.add_artist(self.robot_circle)

    def remove(self):
        for circle in self.circles:
            circle.remove()
        self.robot_circle.remove()

    def add_edge(self, edge, plot):
        self.edges[edge] = plot

    def get_plot(self, edge):
        return  self.edges[edge]

    def remove_edge(self, edge):
        del self.edges[edge]

    def get_agent_object(self):
        return AgentObject(self.idx, self.loc).set_edges(list(self.edges.keys()))

fig, ax = plt.subplots(figsize=(50, 50))

def apply_agent_concealment(idx):
    """
    Check if a new agent conceals the edges between the agents, this is used when a new agent is added to the
    observation graph and some edges between agents might be removed.
    :param idx:
    :return:
    """
    global agent_mapping
    global edge_list
    agent_loc = agent_mapping[idx].get_loc()
    agent_point = Point(agent_loc[0], agent_loc[1])
    agent_circle = agent_point.buffer(ROBOT_RADIUS).boundary

    to_be_removed = []
    for edge in edge_list:
        edge_as_list = list(edge)
        first_loc = agent_mapping[edge_as_list[0]].get_loc()
        second_loc = agent_mapping[edge_as_list[1]].get_loc()
        edge_line = LineString([(first_loc[0], first_loc[1]), (second_loc[0], second_loc[1])])

        intersection = agent_circle.intersection(edge_line)
        if not intersection.is_empty:
            to_be_removed.append(edge)


    for edge in to_be_removed:
        remove_edge(edge, agent_mapping, edge_list)

def test_concealment(agent_map, first_idx, second_idx):
    """
    Check if the edge between two agents is even possible, if there is an agent that conceals one of the other
    agents than this edge is not possible.
    Check if the circle of the robot (based on its diameter) intersects with the line between the agents.
    :param first_idx:
    :param second_idx:
    :return:
    """

    # Create an object which represents the edge.
    first_loc = agent_map[first_idx].get_loc()
    second_loc = agent_map[second_idx].get_loc()
    edge_line = LineString([(first_loc[0], first_loc[1]), (second_loc[0], second_loc[1])])

    for idx, agent in agent_map.items():
        if idx != first_idx and idx != second_idx:
            agent_loc = agent.get_loc()
            agent_point = Point(agent_loc[0], agent_loc[1])
            agent_circle = agent_point.buffer(ROBOT_RADIUS).boundary


            intersection = agent_circle.intersection(edge_line)
            if not intersection.is_empty:
                return True
    return False

def add_edge(agent_map, edges, first_idx, second_idx):
    """
    Add a new edge to our edge list.
    Currently concealment test is naive - go over all the other agents and check that it does not conceal the current
    agent.
    :param first_idx:
    :param second_idx:
    :return:
    """

    edge = frozenset((first_idx, second_idx))
    if edge not in edges:
        # \
            # and not test_concealment(agent_map, first_idx, second_idx):
        edge_list.append(edge)
        return edge

    return None

def plot_edges(agent_map, edges):
    """
    Plot a given list of edges, assumes that each edge is represented by python's frozenset.
    :param edges:
    :return:
    """

    for edge in edges:
        edge_as_list = list(edge)
        first_loc = agent_map[edge_as_list[0]].get_loc()
        second_loc = agent_map[edge_as_list[1]].get_loc()

        line, = plt.plot([first_loc[0], second_loc[0]], [first_loc[1], second_loc[1]], 'b-')
        agent_map[edge_as_list[0]].add_edge(edge, line)
        agent_map[edge_as_list[1]].add_edge(edge, line)

def draw_edges(agent_map, edges):
    """
    Given the current agent_mapping, draw the edges of the observation graph.
    :return:
    """
    new_edges = []
    for idx, agent in agent_map.items():
        for second_idx, second_agent in agent_map.items():
            dist = euclidean_dist(agent.get_loc(), second_agent.get_loc())
            if dist > SMIN and dist <= SMAX:
                curr_edge = add_edge(agent_map, edges, idx, second_idx)
                if curr_edge is not None:
                    new_edges.append(curr_edge)

    plot_edges(agent_map, new_edges)

def update_agent_edges(agent_map, edges, agent_idx):
    """
    Updates the edge list following the addition of an agent. Needs to create edges for this agent with all
    the other agents which it can observe.
    :param agent_idx:
    :return:
    """


    agent_loc = agent_map[agent_idx].get_loc()
    new_edges = []

    for idx, agent in agent_map.items():
        if idx != agent_idx:
            dist = euclidean_dist(agent.get_loc(), agent_loc)
            if dist > SMIN and dist <= SMAX:
                edge = add_edge(agent_map, edges, agent_idx, idx)
                if edge is not None:
                    new_edges.append(edge)

    plot_edges(agent_map, new_edges)

def mark_convex_hull():
    global agent_mapping
    hull = get_convex_hull_indices(agent_mapping)
    for idx, agent in agent_mapping.items():
        agent.robot_circle.remove()
        loc = agent.get_loc()

        if idx in hull:
            agent.robot_circle = plt.Circle((loc[0], loc[1]), ROBOT_RADIUS, color='k', fill=True)
        else:
            agent.robot_circle = plt.Circle((loc[0], loc[1]), ROBOT_RADIUS, color='b', fill=True)
        ax.add_artist(agent.robot_circle)


def generate_hexagon(max_rad):
    return [(0, 0), (max_rad, 0), (max_rad, -math.sqrt(3) * max_rad),
    (0, -math.sqrt(3)* max_rad), (max_rad + math.cos(to_rad(60)) * max_rad, -math.sqrt(3) * max_rad + math.sin(to_rad(60)) * max_rad),
            (0 - math.cos(to_rad(60)) * max_rad, -math.sqrt(3) * max_rad + math.sin(to_rad(60)) * max_rad)]

def on_press(event):
    """
    Activated on press of any key when pyplot's figure is open.
    :param event:
    :return:
    """
    global t
    global agent_mapping
    global edge_list
    if t is None:
        t = threading.Timer(DEBOUNCE_DUR, on_singleclick, [event])
        t.start()
    if event.dblclick:
        nearest = find_nearest_agent(event.xdata, event.ydata)
        if nearest is not None:
            t.cancel()
            remove_agent(nearest, agent_mapping, edge_list)
            plot_wpc()
            mark_convex_hull()


def remove_agent(idx, agent_map, edges, plot=True):
    global t
    global border


    if plot:
        agent_map[idx].remove()

    # Remove all the edges of the agent.
    to_be_removed = []
    for edge in edges:
        if idx in edge:
            to_be_removed.append(edge)

    for edge in to_be_removed:
        remove_edge(edge, agent_map, edges, plot)

    if plot:
        print("Removing idx: ", idx)
    del agent_map[idx]
    draw_edges(agent_map, edges)

    if plot:
        if border is not None:
            border.remove()
            border = plot_agent_boundary()

        # update_center()
        plt.draw()


    t = None

def remove_edge(edge, agent_map, edges, plot=True):

    edge_as_list = list(edge)

    # Remove the actual plot of the edge.
    if plot:
        agent_map[edge_as_list[0]].get_plot(edge).remove()

    # Remove the edge from the agents mapping.
    agent_map[edge_as_list[0]].remove_edge(edge)
    agent_map[edge_as_list[1]].remove_edge(edge)
    edges.remove(edge)

def on_singleclick(event):
    global t
    global border
    global agent_mapping
    global edge_list
    print("You single-clicked", event.button, event.xdata, event.ydata)
    t = None
    idx = add_agent(event.xdata, event.ydata)
    update_agent_edges(agent_mapping, edge_list, idx)

    if border is not None:
        border.remove()
        border = plot_agent_boundary()

    mark_convex_hull()
    plot_wpc()
    plt.draw()


def update_center():
    global center
    avg_loc = np.array([0.0,0.0])
    for agent in agent_mapping.values():
        avg_loc += np.array(agent.get_loc())
    avg_loc /= len(agent_mapping)
    if center is None:
        center = AgentDrawing(-1, avg_loc.tolist(),inner_color='r')
    else:
        center.remove()
        center.update_loc(avg_loc.tolist(), inner_color='r')
    print("Drawing center")
    center.draw(ax)


def add_agent(x, y, color='g'):
    global agent_cnt
    global agent_mapping

    agent_cnt += 1
    agent_mapping[agent_cnt] = AgentDrawing(agent_cnt, (x, y), outer_color=color)
    agent_mapping[agent_cnt].draw(ax)

    # update_center()
    apply_agent_concealment(agent_cnt)
    return agent_cnt

def find_nearest_agent(x, y):
    for idx, agent in agent_mapping.items():
        if euclidean_dist((x,y), agent.get_loc()) < 0.5:
            return idx
    return None

def find_weak_point(agent_map, indices):
    """
    Given a list of indices and a map of agents, find the weak point which is the agent
    that observes the lowest number of agents. Breaks ties arbitrarily.
    :param agent_map:
    :param indices:
    :return:
    """
    min_idx = None
    min_obs = float('inf')
    for idx in indices:
        num_obs = agent_map[idx].get_num_edges()
        if num_obs < min_obs:
            min_obs = num_obs
            min_idx = idx
    return min_idx, min_obs

def compute_wpc(agent_map, edges):
    copied_agent_map = copy.deepcopy({idx: agent.get_agent_object() for idx, agent in agent_map.items()})
    copied_edges = copy.deepcopy(edges)

    total_agents = 0
    required_agents = 0

    while (len(copied_agent_map) > 0):
        hull = get_convex_hull_indices(copied_agent_map)
        weak_point, bareness_factor = find_weak_point(copied_agent_map, hull)
        delta = bareness_factor - total_agents
        if delta > 0:
            required_agents += delta + 1
            total_agents += delta + 1

        total_agents += 1
        remove_agent(weak_point, copied_agent_map, copied_edges,plot=False)
    return required_agents



def plot_polygon(polygon):
    """
    Given shapely's polygon draw it using pyplot
    :param polygon:
    :return:
    """
    patch = Polygon(np.array(polygon.exterior.xy).T, fc='#999999',
                         ec='#000000', fill=True)
    return ax.add_patch(patch)

def compute_convex_hull(agent_map):
    """
    Given a mapping of agents, compute the convex hull.
    :param agent_map:
    :return:
    """
    points = []
    for agent in agent_map.values():
        points.append(agent.get_point())

    point_collection = MultiPoint(points)
    return point_collection.convex_hull

def get_convex_hull_indices(agent_map):
    """
    Get the indices of the agents in the convex hull.
    :param agent_map:
    :return:
    """
    if len(agent_map) <= 3:
        return set(agent_map.keys())

    hull_indices = set()
    hull = compute_convex_hull(agent_map)
    coordinates = np.array(hull.boundary.xy).T
    for i in range(coordinates.shape[0]):
        for idx, agent in agent_map.items():
            dist = euclidean_dist(agent.get_loc(), coordinates[i, :])
            if dist < ROBOT_RADIUS:
                hull_indices.add(idx)
                break
    return hull_indices

def plot_agent_boundary():
    """
    Draws boundary around the agents, currently draws the convex hull of the whole structure.
    :return:
    """
    global agent_mapping
    if len(agent_mapping) > 0:
        return plot_polygon(compute_convex_hull(agent_mapping))
    else:
        return None

def plot_wpc():
    global agent_mapping
    global edge_list
    global text_patch

    if text_patch is not None:
        try:
            text_patch.remove()
        except Exception:
            text_patch = None

    wpc = compute_wpc(agent_mapping, edge_list)
    # Present number of healthy and contaminated agents in a textbox.
    textstr = "WPC: " + str(wpc)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    text_patch = ax.text(-0.1, 1.1,textstr, transform=ax.transAxes, fontsize=14,
                   verticalalignment='top', bbox=props)

# Annotation that will be used later.
annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(idx):
    global agent_mapping
    pos = agent_mapping[idx].get_loc()
    annot.xy = pos
    text = str(agent_mapping[idx].get_num_edges())
    # \
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)

def on_hover(event):
    """
    Callback for onhover event.
    :param event
    :return:
    """
    vis = annot.get_visible()
    if event.inaxes == ax:
        nearest = find_nearest_agent(event.xdata, event.ydata)
        if nearest:
            update_annot(nearest)
            annot.set_visible(True)
            plt.draw()
        else:
            if vis:
                annot.set_visible(False)
                plt.draw()

def draw_msc():
    global agent_mapping
    global edge_list
    msc_size = math.floor((math.pi * SMAX) / ((SMAX / 2) * math.acos(1 - (2 * (SMIN ** 2)) / (SMAX ** 2))))
    angle_diff = math.acos(1 - (2 * (SMIN ** 2)) / (SMAX ** 2))
    print(msc_size)

    center = (5,4)
    rad = SMAX / 2

    for i in range(msc_size):
        idx = add_agent(center[0] + rad * math.cos(angle_diff * (i - 1)), center[1] + rad * math.sin(angle_diff * (i-1)))
        update_agent_edges(agent_mapping, edge_list, idx)



# hexagon = generate_hexagon(SMAX / 2)
#
# print(hexagon)
# xs, ys = zip(*hexagon)
#
# for i in range(len(hexagon)):
#     add_agent(xs[i], ys[i])

# draw_msc()

draw_edges(agent_mapping, edge_list)
border = plot_agent_boundary()


fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect("motion_notify_event", on_hover)
# print(compute_wpc(agent_mapping, edge_list))
automin, automax = plt.xlim()
plt.xlim(automin-6, automax+8)
automin, automax = plt.ylim()
plt.ylim(automin-10, automax+7)
plot_wpc()
mark_convex_hull()
plt.show()

