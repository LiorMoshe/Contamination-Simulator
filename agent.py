from collections import namedtuple
from enum import Enum

import numpy as np
from gym import spaces


class InternalState(Enum):
    HEALTHY=1
    CONTAMINATED=2

# An observation will contain the distance bearing and state of the agent that we observed.
Observation = namedtuple("Observation", "dist bearing state")

class ContaminationAgent(object):
    """
    Description of the agent in the contamination problem. The agent will hold it's current position
    in the simulated 2D space and it's velocity in each axis.
    The agent can't explicitly communicate with other agents and it can observe agents which are at a distance
    between it's minimal and maximal observation radii.
    """

    def __init__(self, environment, state, agent_idx):
        self.min_obs_rad = environment.min_obs_rad
        self.max_obs_rad = environment.max_obs_rad
        self.torus = environment.torus
        self.world_size = environment.world_size
        self.internal_state = state
        self.index = agent_idx

        # Position of our agent.
        self.pos = None

        # Velocity of the agent.
        self.vel = None

        # Orientation of the agent.
        self.orientation = None

        # Angular velocity.
        self.angular_velocity = None
        self.action = None

        # Rotation matrix which will be used to direct the actions based on the orientation of the agent.
        self.rotation_matrix = None

        # Current cluster the agent is in.
        self.cluster = None


    @property
    def action_space(self):
        return spaces.Box(low=-1, high=+1, shape=(2,), dtype=np.float32)

    @property
    def observation_space(self):
        return spaces.Box(low=np.array([self.min_obs_rad, self.min_obs_rad, 0]),
                          high=np.array([self.max_obs_rad, self.max_obs_rad, 2 * np.pi]), shape=(3,), dtype=np.float32)

    def reset(self, state):
        self.set_state(state)
        self.vel = np.zeros_like(state)
        self.angular_velocity = 0

    def set_orientation(self, orientation):
        self.orientation = orientation

        # Rotation of moving vector counterclockwise in orientation degrees.
        r_matrix_2 = np.squeeze([[np.cos(orientation), -np.sin(orientation)],
                                 [np.sin(orientation), np.cos(orientation)]])

        # Computes the rotation matrix based on the given angle
        self.rotation_matrix = r_matrix_2

    def get_orientation(self):
        return np.array([self.orientation])

    def get_state(self):
        return np.concatenate((self.pos, np.array([self.orientation])))

    def set_state(self, state):
        self.set_position(state[:2])
        self.set_orientation(state[2])


    def set_position(self, pos):
        self.pos = np.squeeze(pos)

    def get_position(self):
        return self.pos

    def get_observation(self, distance_matrix, angle_matrix, agents):
        """
        Get the current observation of the agent based on the given distance matrix.
        The observation of the agent will be it's distance from each agent and bearing.
        Note: We don't keep index of agents that we see because we wish to have anonymity.
        :param distance_matrix: Distance from other agents.
        :param angle_matrix Angles relative to agents.
        :param List of all the agents in the environment.
        :return: A list of observations.
        """
        observation = []
        for idx, dist in enumerate(distance_matrix):
            if dist >= self.min_obs_rad and dist <= self.max_obs_rad:
                observation.append(Observation(dist, angle_matrix[idx], agents[idx].internal_state))

        return observation

    def state_transition(self, observation):
        """
        Update the agent's state based on the given observation.
        If there is a majority of some internal state which isn't our agent's state than we
        will switch states.
        :param observation:
        :return:
        """
        healthy_cnt = contaminated_cnt = 0
        prev_state = self.internal_state

        if prev_state.value == 1:
            healthy_cnt += 1
        else:
            contaminated_cnt += 1

        for agent_obs in observation:
            if agent_obs.state.value == 1:
                healthy_cnt += 1
            else:
                contaminated_cnt += 1

        if healthy_cnt == contaminated_cnt:
            return

        self.internal_state = InternalState.HEALTHY if healthy_cnt > contaminated_cnt else InternalState.CONTAMINATED

        if prev_state.value != self.internal_state.value and self.is_allocated():
            self.free(notify=True)

    def allocate(self, cluster):
        """
        Allocate the agent to a given cluster.
        :param cluster:
        :return:
        """
        # if self.internal_state.value == 1:
            # print("Allocated new cluster: " + str(cluster._id))
        # if not self.is_allocated():
        self.cluster = cluster

    def free(self, notify=False):
        """
        Remove the agent from it's allocated cluster.
        :return:
        """
        if notify:
            self.cluster.release(self.index)
        self.cluster = None

    def is_allocated(self):
        """
        Check whether the agent is allocated to a given cluster.
        :return:
        """
        return self.cluster is not None

    def get_cluster_id(self):
        if self.cluster is None:
            return None
        return self.cluster._id





