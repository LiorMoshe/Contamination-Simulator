from utils import get_slope, euclidean_dist, get_max_stable_cycle_size
import math
import numpy as np
import logging

class AgentCluster(object):

    def __init__(self, id, agents, enemy=False):
        logging.info("Creating new cluster with id: " + str(id) + "  enemys: " + str(enemy))
        self._id = id
        self.agents = agents
        self.enemy = enemy
        self.allocated_action = False
        self.target_loc = None
        self.onion = False
        self.msc = False

        self.moving = False
        self.prev_dist = float('inf')

        for agent in self.agents.values():
            agent.allocate(self)

    def size(self):
        return len(self.agents)

    def has_agent(self, agent_index):
        return any(agent.index == agent_index for agent in self.agents.values())

    def add_agent(self, agent):
        self.target_loc = None
        self.agents[agent.index] = agent

    def remove_agents(self, agent):
        del self.agents[agent.index]

    def release(self, agent_idx):
        del self.agents[agent_idx]

    def get_center(self):
        """
        Get the center of the cluster.
        :return:
        """
        return np.sum(np.vstack([agent.get_position() for agent in self.agents.values()]), axis=0) / len(self.agents)

    def rotate_to_center(self):
        """
        Move all members of the cluster toward the center of the cluster.
        :return:
        """
        if len(self.agents) > 1:
            center = self.get_center()
            self.rotate_to_target(center)

    def rotate_to_target(self, target):
        """
        Rotate members of the cluster to a target location.
        Compute the vector for each member of the cluster in which it needs to move to get to a specific target.
        :param target:
        :return: A mapping of action for each agent in the cluster.
        """
        for idx, agent in self.agents.items():
            agent.set_orientation(math.atan(get_slope(target[0], target[1],
                                                      agent.get_position()[0], agent.get_position()[1])))

    def move_to_target(self, target, cluster_id = None):
        if self.msc:
            self.move_in_formation(target)
            return

        for idx, agent in self.agents.items():
            self.move_agent_to_target(agent, target)

            # If there is a target cluster given save it, it will be used for debugging purposes.
            if cluster_id:
                agent.target_cluster = cluster_id

        self.allocated_action = False

    def move_agent_to_target(self, agent, target, pace=1.0):
        if abs(agent.get_position()[0] - target[0]) < 1e-10:
            theta = 0
        else:
            slope = get_slope(target[0], target[1], agent.get_position()[0], agent.get_position()[1])
            theta = math.atan(slope)

        factor = -1 if agent.get_position()[0] > target[0] else 1
        agent.action = np.array([math.cos(theta), math.sin(theta)]) * factor * pace

    def is_on_target(self):
        return euclidean_dist(self.get_center(), self.target_loc) < 1


    def move_randomly(self):
        """
        Move each agent in the direction of it's orientation.
        :return:
        """
        for agent in self.agents.values():
            # agent.action = np.array([1, 0])
            agent.action = np.random.rand(1,2)

    def merge(self, cluster):
        """
        Merge this cluster with another one.
        :param cluster:
        :return:
        """

        for idx, agent in cluster.agents.items():
            agent.free()
            agent.allocate(self)
            self.agents[idx] = agent

        self.target_loc = None

    def free_all(self):
        for agent in self.agents.values():
            agent.free()

    def get_index(self):
        return self._id

    def get_radius(self):
        """
        Get radius of  cluster in order
        :return:
        """
        center = self.get_center()
        max_dist = 0
        for agent in self.agents.values():
            curr_dist = euclidean_dist(center, agent.get_position())
            if curr_dist > max_dist:
                max_dist = curr_dist

        # logging.info("Radius of cluster: " + str(self._id) + " is " + str(max_dist + 6))
        return (max_dist + 6) / 2

    def has_weak_point(self, global_state):
        """
        Checks whether the agents can observe most of the cluster.
        :return:
        """
        num_agents = len(self.agents)
        for idx, agent in self.agents.items():
            agent_observation = agent.get_observation(global_state.distance_matrix[idx, :],
                                                        global_state.angle_matrix[idx, :],
                                                      {**global_state.healthy_agents, **global_state.contaminated_agents})

            if len(agent_observation) < 5:
                return True

        return False

    def log_observations(self, global_state):
        """
        Log the agent with maximal obs in the cluster and the agent with minimal obs in the cluster.
        :return:
        """
        min_obs = float('inf')
        max_obs = 0
        for idx, agent in self.agents.items():
            agent_observation = agent.get_observation(global_state.distance_matrix[idx, :],
                                                        global_state.angle_matrix[idx, :],
                                                      {**global_state.healthy_agents, **global_state.contaminated_agents})
            obs_size = len(agent_observation)
            if obs_size < min_obs:
                min_obs = obs_size

            if obs_size > max_obs:
                max_obs = obs_size

        logging.info("Cluster " + str(self._id) + " min obs: " + str(min_obs) + " max obs: " + str(max_obs))

    def create_msc(self):
        """
        Create from our cluster a maximal stable cycle.
        :return:
        """
        if len(self.agents) < 3:
            return

        center = self.get_center()
        num_agents = len(self.agents)
        idx = list(self.agents.keys())[0]
        min_rad = self.agents[idx].min_obs_rad + 0.3
        jump_size =  360 / (num_agents - 1)

        for num, agent in enumerate(self.agents.values()):
            target = np.array([center[0] + min_rad * math.cos(jump_size * num),
                               center[1] + min_rad * math.sin(jump_size * num)])
            dist = euclidean_dist(agent.get_position(), target)
            # Pace was d / 4
            self.move_agent_to_target(agent, target, pace=min(dist, 1.0))

    def move_in_formation(self, target):
        """
        Observe the current formation of robots, give each one an assigned vector of movements such that the formation
        will be stable. Meaning if have an MSC formation it will be kept as it is even after the movement of the agents.
        Look at the relative position of each agent from the center and move in the same relation to the target point.
        :return:
        """
        center = self.get_center()
        self.target_loc = target
        for agent in self.agents.values():
            slope = get_slope(center[0], center[1], agent.get_position()[0], agent.get_position()[1])
            theta = math.atan(slope)
            dist = euclidean_dist(center, agent.get_position())

            agents_target = target + np.array([dist * math.cos(theta), dist * math.sin(theta)])
            self.move_agent_to_target(agent, agents_target, pace=1.0)


    def converged_to_msc(self):
        """
        Check if the cluster converged to msc by calculating the average distance from the center of the structure.
        :return:
        """
        if len(self.agents) < 3:
            return True

        idx = list(self.agents.keys())[0]
        min_obs = self.agents[idx].min_obs_rad
        center = self.get_center()
        avg_dist = 0.0
        for agent in self.agents.values():
            avg_dist += euclidean_dist(agent.get_position(), center)

        avg_dist /= len(self.agents)

        diff = abs(avg_dist - min_obs)
        if abs(diff - self.prev_dist) < 1e-4 or (self.target_loc is not None and diff < 0.5):
            return True

        self.prev_dist = diff
        return diff < 1e-2

    def mergeable(self, cluster):
        """
        Check if another cluster can join this one without ruining it's stable structure.
        :param cluster:
        :return:
        """
        if self.msc:
            first_idx = list(self.agents.keys())[0]
            min_rad = self.agents[first_idx].min_obs_rad
            max_rad = self.agents[first_idx].max_obs_rad
            max_stable_cycle = get_max_stable_cycle_size(min_rad, max_rad)

            return self.size() + cluster.size() < max_stable_cycle
        return True


    def create_onion_structure(self):
        """
        Create a structure of several layers which is somewhat similar to an onion.
        :return:
        """
        if len(self.agents) == 0:
            return

        num_agents = len(self.agents)
        idx = list(self.agents.keys())[0]
        center = self.get_center()
        locations = [center]
        min_rad = self.agents[idx].min_obs_rad
        max_rad = self.agents[idx].max_obs_rad
        agents_per_minrad_ring = min(get_max_stable_cycle_size(min_rad, max_rad), len(self.agents) - 1)

        # Calculate the amount of rings required and how many agents should be sent to the center.
        num_rings = 0
        total = 1 + (agents_per_minrad_ring * (2 + num_rings)) * (num_rings + 1) / 2
        while total <= num_agents:
            num_rings += 1
            total = 1 + (agents_per_minrad_ring * (2 + num_rings) * (num_rings + 1) / 2)

        final_total = int(1 + (agents_per_minrad_ring * (1 + num_rings) * num_rings / 2))
        leftovers = num_agents - final_total
        for i in range(leftovers):
            locations.append(center)

        # Each ring multiplies by size based on it's radius from the center.
        rings_sizes = []
        for i in range(num_rings):
            rings_sizes.append(agents_per_minrad_ring * (i + 1))



        for ring in range(num_rings):
            jump = 360 / (rings_sizes[ring] - 1)
            for i in range(rings_sizes[ring]):
                locations.append(np.array([center[0] + (ring + 1) * min_rad * math.cos(jump * i),
                                           center[1] + (ring + 1) * min_rad * math.sin(jump * i)]))

        for idx, agent in enumerate(self.agents.values()):
            d = euclidean_dist(agent.get_position(), locations[idx])
            self.move_agent_to_target(agent, locations[idx], pace= min(1.0, d))

        self.onion = True

    def stop(self):
        """
        Stop the cluster completely.
        :return:
        """
        for agent in self.agents.values():
            agent.action = np.array([0, 0])

    def converged_onion(self):
        """
        Check if the cluster converged to the onion structure, if it is in the structure
        there is an agent which is right at the center (or more than one agent).
        In case there is a target which we move toward we will perform several relaxations on the convergence
        test in order to make the formation move.
        :return:
        """
        center = self.get_center()
        closest = float('inf')
        for agent in self.agents.values():
            dist = euclidean_dist(center, agent.get_position())
            if dist < 1e-2:
                self.onion = False
                return True

            if dist < closest:
                closest = dist

        if abs(closest - self.prev_dist) < 1e-4 or (self.target_loc is not None and closest < 0.8):
            return True

        self.prev_dist = closest
        return False

    def stabilize_structure(self):
        """
        Adapt to a certain formation based on the amount of agents in the formation.
        :return:
        """
        msc_size = 9
        if self.target_loc is not None and self.is_on_target():
            # print("Removing target")
            self.target_loc = None
        if len(self.agents) <= msc_size:
            self.msc = True
            self.onion = False
            self.create_msc()
        else:
            self.msc = False
            self.onion = True
            self.create_onion_structure()


    def did_converge(self):
        """
        Check whether our structure converged, if it's an msc we try to build we will
        run the msc convergence test and if its an onion structure we will perform  the onion structure
        convergence test.
        :return:
        """
        if self.msc:
            return self.converged_to_msc()
        elif self.onion:
            return self.converged_onion()
        else:
            return False

    def densify(self):
        """
        Make sure the cluster is as close as possible to a full observable component.
        :return:
        """
        center = self.get_center()
        self.move_to_target(center)

