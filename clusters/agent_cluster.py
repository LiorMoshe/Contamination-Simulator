
from utils import get_slope, euclidean_dist, get_max_stable_cycle_size, to_rad, get_max_dense_circle_size, \
    get_agent_arc_size
import math
import numpy as np
import logging
from collections import namedtuple
import random
from defense.monotonic_iterator import MonotonicIterator

DefenseState = namedtuple("DefenseState", "fence_size, edge_length,")

class AgentCluster(object):

    def __init__(self, id, agents, smin, smax, robot_radius, enemy=False):
        logging.info("Creating new cluster with id: " + str(id) + "  enemys: " + str(enemy))
        self._id = id
        self.agents = agents
        self.enemy = enemy
        self.allocated_action = False
        self.target_loc = None
        self.onion = False
        self.msc = False
        self.odc = False
        self.smin = smin
        self.smax = smax
        self.robot_radius = robot_radius
        self.mode = "D"
        self.target_cluster = None


        self.msc_size = math.floor((math.pi * self.smax) / ((self.smax / 2) *
                                                            math.acos(1 - (2 * (self.smin ** 2)) / (self.smax ** 2))))
        self.moving = False
        self.prev_dist = float('inf')
        self.defense_state = None
        self.defender = MonotonicIterator(len(agents), smin, smax, robot_radius)

        for agent in self.agents.values():
            agent.allocate(self)

    def switch_to_attack(self, target_cluster):
        from clusters.cluster_manager import ClusterManager

        adversary = ClusterManager.instance.get_contaminated_cluster(target_cluster)
        total_agents = adversary.get_num_agents()
        while total_agents > 0:
            deployed = min(self.msc_size, total_agents)
            self.deploy_agents(deployed, target_cluster)
            total_agents -= deployed

        return len(self.agents) > 0
        # self.mode = "A"
        # self.target_cluster = target_cluster

    def switch_to_defense(self):
        self.mode = "D"

    def get_num_agents(self):
        return len(self.agents)

    def deploy_agents(self, num_agents, target_cluster_id):
        """
        Deploy a subset of agents toward the target location. This will split our cluster.
        :param num_agents:
        :param target_loc:
        :return:
        """
        from clusters.cluster_manager import ClusterManager
        if num_agents > len(self.agents):
            print("Exception: Asked to deploy {0} agents when there are only {1} agents".format(num_agents,
                                                                                                len(self.agents)))
            raise("Exception: Asked to deploy {0} agents when there are only {1} agents".format(num_agents,
                                                                                                len(self.agents)))

        print("Deploying agents.")
        deployed = list(self.agents.keys())[:num_agents]
        deployed_agents = {}
        for idx in deployed:
            self.agents[idx].free()
            deployed_agents[idx] = self.agents[idx]
            del self.agents[idx]

        new_cluster = ClusterManager.instance.allocate_cluster(deployed_agents)
        # todo- This is only implemented for healthy agents.
        ClusterManager.instance.add_healthy_cluster(new_cluster)
        new_cluster.mode = "A"
        new_cluster.create_msc()
        new_cluster.target_loc = ClusterManager.instance.get_contaminated_cluster(target_cluster_id).get_center()
        # new_cluster.target_cluster = target_cluster_id
        # new_cluster.move_in_formation(target=target_loc)

    def act_attack(self):
        """
        Act according to the attack strategy.
        Move to the given target.
        :return:
        """
        if self.is_on_target():
            self.mode = "D"
        else:
            if not self.did_converge():
                self.stabilize_structure()
            else:
                self.move_in_formation(self.target_loc)


    def act_defense(self, target):
        """
        This method is activated when we are in defense mode and there are no agents that we can merge
        with that are close to us.
        In that case we wish to become as safe as possible and if we are safe we move randomly through
        the game's space.
        :return:
        """
        if (len(self.agents) <= 3):
            self.move_in_formation(target)

        if len(self.agents) <= self.msc_size:
            if not self.did_converge():
                self.stabilize_structure()
            else:
                # Explore if we converged.
                # print("Sent clique to explore")
                self.move_in_formation(target=target)
                # self.explore()
        elif not self.defender.did_converge():
            # self.defender.update_num_agents(len(self.agents))
            self.defender.num_agents = len(self.agents)
            self.defender.update_center(self.get_center())
            res = self.defender.iterate()
            if res:
                self.move_agents_to_targets(self.defender.get_locations())
        else:
            self.move_in_formation(target=target)
            # self.stop()
            # Move randomly in a formation.
            # self.deploy_agents(num_agents=5, target_loc=np.array([90,90]))
            # self.stop()
            # self.move_in_formation(target=np.array([0,0]), pace=0.1)
            # self.explore()


    def get_diameter(self):
        """
        The diameter of the cluster is the longest distance between two agents in it.
        :return:
        """
        if (len(self.agents) <= 1):
            return  0
        else:
            max_dist = float('-inf')
            for first_agent_id in self.agents:
                for second_agent_id in self.agents:
                    if first_agent_id != second_agent_id:
                        dist = euclidean_dist(self.agents[first_agent_id].get_position(),
                                              self.agents[second_agent_id].get_position())
                        if dist > max_dist:
                            max_dist = dist
            return max_dist


    def get_mode(self):
        return self.mode

    def size(self):
        return len(self.agents)

    def has_agent(self, agent_index):
        return any(agent.index == agent_index for agent in self.agents.values())

    def add_agent(self, agent):
        if not self.mode == "A":
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
        # logging.info("Agent " + str(agent.index) + " location: " + str(agent.get_position()) + " and target: " + str(target))
        if abs(agent.get_position()[0] - target[0]) < 1e-10:
            theta = 0
        else:
            slope = get_slope(target[0], target[1], agent.get_position()[0], agent.get_position()[1])
            # logging.info("Got slope: " + str(slope))
            theta = math.atan(slope)

        factor = -1 if agent.get_position()[0] > target[0] else 1
        # logging.info("Setting action: " + str(np.array([math.cos(theta), math.sin(theta)]) * factor * pace))
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
        # print("Cluster Agents InMerge: {0}".format(cluster.agents))
        for idx, agent in cluster.agents.items():
            # print("MergingIdx: {0}".format(idx))
            agent.free()
            agent.allocate(self)
            self.agents[idx] = agent

        if self.get_mode() != "A":
            self.target_loc = None
        return True

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

    def get_odc_targets(self):
        center = self.get_center()
        num_agents = len(self.agents)
        radius = self.smax / 2
        arc_size = get_agent_arc_size(self.robot_radius, radius)
        jump = (2 * math.pi - num_agents * arc_size) / num_agents + arc_size
        targets = []
        for i in range(num_agents):
            targets.append((center[0] + radius * math.cos(jump * i), center[1] + radius * math.sin(jump * i)))
        return targets

    def create_odc(self):

        # If this is simply a clique create an msc.
        clique_size = get_max_stable_cycle_size(self.smin, self.smax)
        if len(self.agents) <= clique_size:
            self.create_msc()
            return

        # Check if there aren't more agents in the component than in an msc.
        odc_size = get_max_dense_circle_size(self.smax, self.robot_radius)
        if len(self.agents) > odc_size:
            return

        self.move_agents_to_targets(self.get_odc_targets())


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
        max_rad = self.agents[idx].max_obs_rad
        jump_size =  360 / (num_agents)
        targets = []

        for num, agent in enumerate(self.agents.values()):
            angle = to_rad(jump_size * num)
            target = np.array([center[0] + (max_rad / 2) * math.cos(angle),
                               center[1] + (max_rad / 2) * math.sin(angle)])

            targets.append(target)
            # dist = euclidean_dist(agent.get_position(), target)
            # # Pace was d / 4
            # self.move_agent_to_target(agent, target, pace=min(dist, 1.0))

        self.move_agents_to_targets(targets)


    def move_agents_to_targets(self, targets):
        for num, agent in enumerate(self.agents.values()):
            dist = euclidean_dist(agent.get_position(), targets[num])
            # Pace was d / 4
            self.move_agent_to_target(agent, targets[num], pace=min(dist, 1.0))


    def explore(self):
        """
        Move the whole formation toward a random location.
        :return:
        """
        # todo- Decide how should we choose the random target, currently its toward a random location.
        curr_center = self.get_center()
        max_dist = float('-inf')
        for agent in self.agents.values():
            curr_dist = euclidean_dist(agent.get_position(), curr_center)
            if curr_dist > max_dist:
                max_dist = curr_dist

        angle = random.random() * math.pi * 2
        target_loc = np.array([curr_center[0] + 1.5*max_dist * math.cos(angle),
                                         curr_center[1] + 1.5 * max_dist * math.sin(angle)])
        self.move_in_formation(target=target_loc)

    def move_in_formation(self, target, pace=1.0):
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
            self.move_agent_to_target(agent, agents_target, pace=pace)


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

        # todo - Currently we won't merge attacking clusters, may change in the future.
        if cluster.get_mode() == "A" or self.get_mode() == "A":
            return False

        if cluster._id == self._id:
            return False

        if self.msc:
            first_idx = list(self.agents.keys())[0]
            min_rad = self.agents[first_idx].min_obs_rad
            max_rad = self.agents[first_idx].max_obs_rad
            max_stable_cycle = get_max_stable_cycle_size(min_rad, max_rad)

            return self.size() + cluster.size() < max_stable_cycle
        return True


    @staticmethod
    def  compute_cycle_size(min_rad, max_rad, cycle_size):
        """
        :param self:
        :return:
        """
        theta = math.acos(1 - (2 * min_rad ** 2) / (max_rad ** 2))
        return math.floor( 2  * math.pi / theta)

    def create_onion_structure(self):
        """
        Create a structure of several layers which is somewhat similar to an onion.
        :return:
        """
        if len(self.agents) == 0:
            return

        EPSILON = 10
        num_agents = len(self.agents)
        idx = list(self.agents.keys())[0]
        center = self.get_center()
        locations = []
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

        # Each ring multiplies by size based on it's radius from the center.
        rings_sizes = []
        for i in range(num_rings):
            rings_sizes.append(agents_per_minrad_ring * (i + 1))
        # rings_sizes = [agents_per_minrad_ring]
        # for i in range(1, num_rings):
        #     rings_sizes.append(self.compute_cycle_size(min_rad, min_rad, i))

        # print("Ring sizes: " + str(rings_sizes))

        leftovers = num_agents - final_total + 1
        # print("Leftovers: ", leftovers)
        if leftovers > 0:
            jump = 360 / (rings_sizes[num_rings - 1] + rings_sizes[0])
        else:
            jump = 0
        # jump = 360 / (leftovers) if leftovers > 0 else 0
        start =  num_rings * EPSILON
        for i in range(leftovers):
            angle = to_rad(start + jump * i)
            locations.append(np.array([center[0] + (max_rad / 2 + (num_rings) * (min_rad)) * math.cos(angle),
                                       center[1] + (max_rad / 2  + (num_rings) * (min_rad)) * math.sin(angle)]))
            # locations.append(center)



        for ring in range(num_rings):
            jump = 360 / (rings_sizes[ring])
            start = ring * EPSILON
            for i in range(rings_sizes[ring]):
                angle = to_rad(start + jump * i)
                # locations.append(np.array([center[0] + (ring + 1) * (max_rad / 2) * math.cos(angle),
                #                            center[1] + (ring + 1) * (max_rad / 2) * math.sin(angle)]))
                locations.append(np.array([center[0] + (max_rad / 2 + (ring) * (min_rad)) * math.cos(angle),
                                           center[1] + (max_rad / 2 + (ring) * (min_rad)) * math.sin(angle)]))

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

    def physical_onion_formation(self):
        """
        Create an onion formation which takes note of the dimension of the robots. In this formation
        the leftover robots are not directed toward the center of the formation.
        :return:
        """

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

    def stabilize_structure(self, use_odc=False):
        """
        Adapt to a certain formation based on the amount of agents in the formation.
        :return:
        """
        if len(self.agents) == 0:
            return

        agent = self.agents[list(self.agents.keys())[0]]
        msc_size = get_max_stable_cycle_size(agent.min_obs_rad, agent.max_obs_rad)
        if self.target_loc is not None and self.is_on_target():
            self.target_loc = None
        if len(self.agents) <= msc_size:
            self.msc = True
            self.onion = False
            self.create_msc()
        elif not use_odc:
            self.msc = False
            self.onion = True
            self.create_onion_structure()
        else:
            self.create_odc()


    def converged_to_targets(self, targets):
        allocated = []
        for target in targets:
            for agent in self.agents.values():
                if agent.index not in allocated and euclidean_dist(target, agent.get_position()) < 1e-2:
                    allocated.append(agent.index)

        return len(allocated) == len(self.agents)

    def odc_converged(self):
        return self.converged_to_targets(self.get_odc_targets())

    def msc_converged(self):
        """
        Check if the msc converged by running the algorithm that creates the msc and checking that there is
        an agent allocated in each target location.
        :return:
        """
        if len(self.agents) < 3:
            return

        center = self.get_center()
        num_agents = len(self.agents)
        idx = list(self.agents.keys())[0]
        max_rad = self.agents[idx].max_obs_rad
        jump_size =  360 / (num_agents)

        targets = []
        for num in range(len(self.agents.values())):
            angle = to_rad(jump_size * num)
            targets.append(np.array([center[0] + (max_rad / 2) * math.cos(angle),
                               center[1] + (max_rad / 2) * math.sin(angle)]))

        return self.converged_to_targets(targets)


    def did_converge(self,use_odc=False):
        """
        Check whether our structure converged, if it's an msc we try to build we will
        run the msc convergence test and if its an onion structure we will perform  the onion structure
        convergence test.
        :return:
        """
        if self.msc:
            return self.msc_converged()
        elif use_odc:
            return self.odc_converged()
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

if __name__=="__main__":
    size = AgentCluster.compute_cycle_size(3, 9,0.5)
    print(size)
